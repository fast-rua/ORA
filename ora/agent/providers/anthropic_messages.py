"""The Anthropic Messages provider: the canonical message IR goes out through
``anthropic_request`` and the streamed reply comes back through ``anthropic_events``
and ``anthropic_reply``.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ora.agent.providers.anthropic_history import _has_complete_anthropic_thinking_signature
from ora.agent.contracts import PolicySessionError
from .anthropic_events import accumulate_anthropic_events
from .stream_failures import http_status_failure
from ora.agent.evidence.canonical import canonical_bytes
from ora.agent.providers.attempts import (
    _StreamInterrupted,
    partial_credential_echo,
    record_attempt_failure,
)
from ora.agent.providers.recovery import ProviderRecoveryDeadline, RecoveryWindow, bounded_request
from .transport_failures import (
    ProviderClosed,
    _KIMI_CODING_BASE_URLS,
    _KIMI_TRANSIENT_SIGNATURE_REJECTION,
    _echo_failure,
    _record_provider_failure,
    _retryable_transport_error,
)
from .anthropic_reply import (
    _anthropic_blocks_with_tool_inputs,
    _anthropic_reply,
    _rejected_tool_input_reply,
)
from .anthropic_request import (
    _anthropic_tool_names,
    _declared_property_types,
    translate_anthropic_request,
)


# visible_only_v1 replay composes only at the commits that composed it.
_ANTHROPIC_CONTINUATION_REPLAY_MODES = {"exact_v1"}


@dataclass
class AnthropicMessagesProvider:
    """Direct Anthropic Messages adapter with canonical response normalization."""

    base_url: str
    api_key: str
    timeout_seconds: float = 900.0
    max_retries: int = 2
    preflight_probe: bool = field(default=False, repr=False, kw_only=True)
    recovery_policy: str = field(default="legacy_v1", kw_only=True)
    recovery_window_s: float | None = field(default=None, kw_only=True)
    request_deadline_monotonic: float | None = field(default=None, repr=False, kw_only=True)
    attempt_observer: Callable[[Mapping[str, Any]], None] | None = field(
        default=None, repr=False, kw_only=True,
    )
    anthropic_version: str = "2023-06-01"
    auth_scheme: str = "x_api_key"
    continuation_replay: str = "exact_v1"
    session_affinity_id: str | None = None
    # Host-local credential rotation: one session stays on one key for cache
    # locality and moves to the next when the current key is exhausted,
    # rejected, or an incomplete stream can safely retry. Keys are operator
    # environment values; only their ordinal
    # index is ever reported.
    standby_api_keys: tuple[str, ...] = ()
    credential_start_index: int = 0
    client: Any = None
    _owns_client: bool = field(default=False, init=False)
    _credential_index: int = field(default=0, init=False)
    _credential_ring: tuple[str, ...] = field(default=(), init=False)
    _rotations: list[dict[str, Any]] = field(default_factory=list, init=False)
    closed: bool = field(default=False, init=False)
    protocol: str = field(default="anthropic_messages", init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url or not self.api_key:
            raise PolicySessionError("native provider base URL and API key are required")
        if any(not key for key in self.standby_api_keys):
            raise PolicySessionError("standby provider credentials must be non-empty")
        # The rotation ring starts at this session's assigned key and returns
        # to it after one lap; provider usage limits reset on their own clock.
        self._credential_ring = (self.api_key, *self.standby_api_keys)
        if self.auth_scheme not in {"bearer", "x_api_key"}:
            raise PolicySessionError("unsupported Anthropic provider auth scheme")
        if self.continuation_replay not in _ANTHROPIC_CONTINUATION_REPLAY_MODES:
            raise PolicySessionError("unsupported Anthropic continuation replay mode")
        if self.session_affinity_id is not None and not re.fullmatch(
            r"[A-Za-z0-9:._-]{1,200}", self.session_affinity_id,
        ):
            raise PolicySessionError("invalid provider session-affinity identity")
        if self.client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - dependency gate
                raise PolicySessionError("httpx is required for native_api policies") from exc
            self.client = httpx.Client(timeout=httpx.Timeout(
                self.timeout_seconds, connect=min(10.0, self.timeout_seconds),
            ))
            self._owns_client = True

    def _rotate_credential(self, *, status_code: int, detail: str) -> bool:
        """Move to the next host-local credential after an exhausted key.

        The ring is cyclic: a provider usage limit resets on the provider's
        own clock, so a key that is exhausted now is a key to come back to,
        not a key to discard. One full lap without success means every
        credential is currently exhausted, which the caller turns into a
        patient wait instead of a fatal fault.
        """

        ring = self._credential_ring
        if len(ring) < 2:
            return False
        self._credential_index += 1
        self.api_key = ring[self._credential_index % len(ring)]
        self._rotations.append({
            "schema": "ora.provider-credential-rotation.v1",
            "credential_index": self._credential_index,
            "credential_start_index": self.credential_start_index,
            "completed_laps": self._credential_index // len(ring),
            "status_code": status_code,
            "unix_s": time.time(),
        })
        return True

    def credential_rotations(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._rotations]

    def _redact(self, text: str) -> str:
        for key in sorted(self._credential_ring, key=len, reverse=True):
            if key:
                text = text.replace(key, "[REDACTED]")
        return text

    def prepare_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return translate_anthropic_request(request)

    def _messages_url(self) -> str:
        if self.base_url.endswith("/v1/messages"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    def _record_attempt_failure(
        self, exc: Exception, *, body: Mapping[str, Any], model_call_id: str,
        attempt: int, credential_index: int, next_index: int | None,
        started: float, started_monotonic: float, retry_delay: float,
        events: Sequence[Mapping[str, Any]], phase: str,
    ) -> None:
        record_attempt_failure(self, exc, body=body, model_call_id=model_call_id,
            attempt=attempt, credential_index=credential_index, next_index=next_index,
            started=started, started_monotonic=started_monotonic,
            retry_delay=retry_delay, events=events, phase=phase)

    def _request_once(self, body: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
        raw_events: list[dict[str, Any]] = []
        first_event_at: float | None = None
        headers = {
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        if self.auth_scheme == "bearer":
            headers["authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-api-key"] = self.api_key
        if self.session_affinity_id is not None:
            headers["x-session-id"] = self.session_affinity_id
        response = self.client.stream(
            "POST",
            self._messages_url(),
            headers=headers,
            content=canonical_bytes(body),
        )
        with response as stream:
            if stream.status_code >= 400:
                detail = stream.read().decode("utf-8", errors="replace")[-4000:]
                detail = self._redact(detail)
                rotated = (
                    self._rotate_credential(
                        status_code=stream.status_code, detail=detail,
                    )
                    if stream.status_code in {401, 402, 403, 429}
                    and not self.preflight_probe and self.max_retries > 0 else False
                )
                error = http_status_failure(stream.status_code, detail, base_url=self.base_url)
                kimi_signature_flake = (
                    stream.status_code == 400
                    and self.base_url in _KIMI_CODING_BASE_URLS
                    and _KIMI_TRANSIENT_SIGNATURE_REJECTION in detail
                    and _has_complete_anthropic_thinking_signature(body)
                )
                setattr(error, "retryable", error.retryable or kimi_signature_flake or rotated)
                if rotated:
                    setattr(error, "credential_rotated", True)
                if kimi_signature_flake:
                    setattr(error, "retry_delay_s", 30.0)
                if stream.status_code == 429:
                    # A provider-side throttle asks for patience, not three
                    # sub-second attempts; a fatal fault here kills the whole
                    # persistent session and its sunk rounds.
                    setattr(error, "retry_delay_s", 20.0)
                    setattr(error, "retry_budget", max(self.max_retries, 6))
                if rotated and self._credential_index >= len(
                    self._credential_ring
                ):
                    # One full lap: every configured credential is currently
                    # rejecting. Provider usage limits reset on their own
                    # clock, so wait for that rather than killing a persistent
                    # session and its completed rounds. This is the most
                    # patient case and therefore wins over the throttle wait.
                    setattr(error, "retry_delay_s", 90.0)
                    setattr(error, "retry_budget", max(self.max_retries, 8))
                raise error
            if body.get("stream") is False:
                value = json.loads(stream.read())
                if not isinstance(value, Mapping):
                    raise PolicySessionError("native provider returned non-object JSON")
                return [dict(value)], 0.0
            try:
                announced_event: str | None = None
                for line in stream.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)
                    if text.startswith(":"):
                        continue
                    if text.startswith("event:"):
                        if announced_event is not None:
                            raise PolicySessionError(
                                "Anthropic SSE event lacks its data payload"
                            )
                        announced_event = text[6:].strip()
                        if not announced_event:
                            raise PolicySessionError(
                                "Anthropic SSE event name is empty"
                            )
                        continue
                    if not text.startswith("data:"):
                        raise PolicySessionError(
                            "Anthropic provider returned malformed SSE framing"
                        )
                    data = text[5:].strip()
                    if data == "[DONE]":
                        announced_event = None
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise PolicySessionError("native provider returned malformed SSE JSON") from exc
                    if not isinstance(parsed, Mapping):
                        raise PolicySessionError("native provider SSE item must be an object")
                    if (
                        announced_event is not None
                        and parsed.get("type") != announced_event
                    ):
                        raise PolicySessionError(
                            "Anthropic SSE event name does not match its payload"
                        )
                    announced_event = None
                    if first_event_at is None:
                        first_event_at = time.monotonic()
                    raw_event = dict(parsed)
                    raw_events.append(raw_event)
                    # Anthropic defines message_stop as the protocol terminal
                    # event. Do not require the HTTP/SSE transport to close
                    # cleanly after a complete response: some loopback routers
                    # reset the connection once the terminal event has been
                    # forwarded. A failure before message_stop remains a
                    # partial stream, which may be discarded and retried before
                    # returning any tool calls to the Native dispatcher.
                    if raw_event.get("type") == "message_stop":
                        break
                if announced_event is not None:
                    raise PolicySessionError(
                        "Anthropic SSE ended before an announced event payload"
                    )
                if not any(event.get("type") in {"message_stop", "error"} for event in raw_events):
                    raise EOFError("Anthropic stream ended before message_stop")
            except Exception as exc:
                # Framing/JSON/SSL certificate failures remain terminal. Only
                # transport interruption (including H's raw TLS EOF case) or
                # clean premature EOF can retry before any tool is dispatched.
                partial = _StreamInterrupted(
                    raw_events, transient=_retryable_transport_error(exc) or isinstance(exc, EOFError),
                )
                raise partial from exc
        return raw_events, first_event_at

    @staticmethod
    def _parse_events(
        raw_events: Sequence[Mapping[str, Any]], model_call_id: str, model: str,
        tool_names_by_alias: Mapping[str, str] | None = None,
        declared_types_by_tool: Mapping[str, Mapping[str, str]] | None = None,
    ) -> tuple[
        dict[str, Any], dict[str, Any] | None, str | None,
        dict[str, Any] | None,
    ]:
        """Accumulate the events, decode each tool input, then build the reply."""

        response = accumulate_anthropic_events(raw_events)
        blocks, rejected_tool_inputs = _anthropic_blocks_with_tool_inputs(
            response, declared_types_by_tool,
        )
        if rejected_tool_inputs:
            return _rejected_tool_input_reply(
                response, blocks, rejected_tool_inputs,
                model_call_id=model_call_id,
                tool_names_by_alias=tool_names_by_alias,
            )
        return _anthropic_reply(
            response, blocks, model_call_id=model_call_id, model=model,
            tool_names_by_alias=tool_names_by_alias,
        )

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str
    ) -> dict[str, Any]:
        started_unix = time.time()
        started_mono = time.monotonic()
        window = RecoveryWindow.for_provider(self)
        tool_names_by_alias = _anthropic_tool_names(request)
        body = self.prepare_request(request)
        raw_events: list[dict[str, Any]] = []
        first_event_at: float | None = None
        attempts = 0
        incomplete_block_retries = 0
        while True:
            if self.closed:
                closed = ProviderClosed()
                _record_provider_failure(closed, attempts)
                raise closed
            attempts += 1
            attempt_started = time.time()
            attempt_monotonic = time.monotonic()
            credential_index = (
                self.credential_start_index + self._credential_index
            ) % len(self._credential_ring)
            try:
                raw_events, first_event_at = (bounded_request(self, body, window)
                    if window is not None else self._request_once(body))
            except Exception as exc:
                _record_provider_failure(exc, attempts)
                retryable = bool(getattr(exc, "retryable", False))
                if not retryable:
                    retryable = _retryable_transport_error(exc)
                retry_budget = getattr(exc, "retry_budget", None)
                if not isinstance(retry_budget, int):
                    retry_budget = self.max_retries
                will_retry = (
                    not self.preflight_probe and self.max_retries > 0
                    and retryable and attempts < retry_budget + 1
                )
                partial_events = getattr(exc, "partial_events", [])
                echoed = partial_credential_echo(exc, self._credential_ring)
                will_retry = will_retry and not echoed
                retry_delay = getattr(exc, "retry_delay_s", None)
                if isinstance(retry_delay, (int, float)) and retry_delay > 0:
                    delay = min(retry_delay * 2 ** (attempts - 1), 60.0)
                else:
                    delay = min(2 ** (attempts - 1), 4)
                if window is not None:
                    bounded_delay = window.transport_delay(exc, attempts, started=attempt_monotonic)
                    will_retry = bounded_delay is not None and not echoed
                    delay = bounded_delay or 0.0
                if self.closed:
                    will_retry = False
                switch = window is None and will_retry and isinstance(exc, _StreamInterrupted) and len(self._credential_ring) > 1
                next_index = (self.credential_start_index + self._credential_index + int(switch)) % len(self._credential_ring)
                self._record_attempt_failure(
                    exc, body=body, model_call_id=model_call_id, attempt=attempts,
                    credential_index=credential_index,
                    next_index=next_index if will_retry else None,
                    started=attempt_started, started_monotonic=attempt_monotonic,
                    retry_delay=delay if will_retry else 0.0,
                    events=partial_events, phase="transport",
                )
                if echoed:
                    raise _echo_failure(exc, attempts) from exc
                if not will_retry:
                    raise
                if switch:
                    self._credential_index += 1
                    self.api_key = self._credential_ring[self._credential_index % len(self._credential_ring)]
                if window is not None:
                    try:
                        window.pause(delay)
                    except ProviderRecoveryDeadline as deadline:
                        _record_provider_failure(deadline, attempts)
                        raise deadline from exc
                else:
                    time.sleep(delay)
                continue
            try:
                serialized_events = json.dumps(raw_events, ensure_ascii=False)
                if any(key in serialized_events for key in self._credential_ring):
                    raise PolicySessionError(
                        "native provider echoed credential material; response was discarded"
                    )
                message, usage, finish_reason, output_rejection = (
                    self._parse_events(
                        raw_events, model_call_id,
                        str(request.get("model", "")), tool_names_by_alias,
                        _declared_property_types(request.get("tools")),
                    )
                )
                if finish_reason is None:
                    raise PolicySessionError(
                        "native provider response has no completion status; request was not retried"
                    )
            except PolicySessionError as exc:
                setattr(exc, "response_started", bool(raw_events))
                _record_provider_failure(exc, attempts)
                # The complete response is parsed before any policy tool can run.
                # One exact replay is therefore side-effect safe when a gateway
                # emits message_stop while omitting only content_block_stop.  Do
                # not generalize this to arbitrary partial streams or malformed
                # tool arguments.
                safe_incomplete_blocks = (
                    str(exc)
                    == "Anthropic message_stop precedes completed content blocks"
                )
                will_retry = (
                    not self.preflight_probe and safe_incomplete_blocks
                    and incomplete_block_retries < 1 and attempts < self.max_retries + 1
                )
                delay = min(2 ** (attempts - 1), 4) if will_retry else 0.0
                if window is not None:
                    if safe_incomplete_blocks and incomplete_block_retries < 1:
                        setattr(exc, "retryable", True)
                    bounded_delay = window.delay(exc, attempts)
                    will_retry = bounded_delay is not None and not any(
                        key in json.dumps(raw_events, ensure_ascii=False) for key in self._credential_ring)
                    delay = bounded_delay or 0.0
                if self.closed:
                    will_retry = False
                self._record_attempt_failure(
                    exc, body=body, model_call_id=model_call_id, attempt=attempts,
                    credential_index=credential_index,
                    next_index=credential_index if will_retry else None,
                    started=attempt_started, started_monotonic=attempt_monotonic,
                    retry_delay=delay, events=raw_events, phase="parse",
                )
                if not will_retry:
                    raise
                if safe_incomplete_blocks:
                    incomplete_block_retries += 1
                if window is not None:
                    try:
                        window.pause(delay)
                    except ProviderRecoveryDeadline as deadline:
                        _record_provider_failure(deadline, attempts)
                        raise deadline from exc
                else:
                    time.sleep(delay)
                continue
            break
        if window is not None and window.remaining() <= 0:
            error = ProviderRecoveryDeadline(raw_events)
            _record_provider_failure(error, attempts)
            raise error
        response_id = next((
            str(message["id"])
            for event in raw_events
            for message in (
                [event.get("message")]
                if event.get("type") == "message_start"
                else [event]
            )
            if isinstance(message, Mapping)
            and isinstance(message.get("id"), str)
            and message["id"]
        ), None)
        finished = time.time()
        return {
            "message": message,
            "usage": usage,
            "finish_reason": finish_reason,
            **({
                "output_rejection": output_rejection,
            } if output_rejection is not None else {}),
            "response_id": response_id,
            "raw_events": raw_events,
            "wire_request": body,
            "model_call_id": model_call_id,
            "started_unix_s": started_unix,
            "finished_unix_s": finished,
            "ttft_s": None if first_event_at is None else round(first_event_at - started_mono, 6),
            "attempts": attempts,
        }

    def close(self) -> None:
        self.closed = True
        if self._owns_client:
            self.client.close()
