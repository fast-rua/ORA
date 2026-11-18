"""The OpenAI-compatible Chat Completions provider.

The agent owns one OpenAI/ChatML-shaped canonical message IR; this adapter
sends it as is, streams the reply, and keeps the provider's continuation state.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from ora.agent.contracts import PolicySessionError
from .openai_wire_options import (
    REASONING_FIELDS,
    TOOL_STRICTNESS_MODES,
    checkpoint_request_options,
    normalize_usage,
    profile_bound,
    project_wire_body,
    validate_request_options,
)
from .stream_failures import http_status_failure, stream_failure_after_content
from ora.agent.evidence.canonical import canonical_bytes
from ora.agent.providers.continuation import (
    PROVIDER_CONTINUATION_MESSAGE_KEY,
    ProviderContinuationState,
)
from ora.agent.providers.attempts import partial_credential_echo, record_attempt_failure
from ora.agent.providers.tool_names import (
    openai_tool_names,
    project_openai_tool_names,
    response_tool_name,
)
from ora.agent.providers.recovery import ProviderRecoveryDeadline, RecoveryWindow, bounded_request
from .transport_failures import (
    ProviderClosed,
    _echo_failure,
    _record_provider_failure,
    _reject_truncated_tools,
    _reject_invalid_tool_arguments,
    _retryable_transport_error,
)


_OPENAI_CONTINUATION_KIND = "openai_chat_reasoning_content_v1"


@dataclass
class OpenAICompatibleProvider:
    """Small direct Chat Completions adapter with bounded safe retries."""

    base_url: str
    api_key: str
    timeout_seconds: float = 900.0
    max_retries: int = 2
    preflight_probe: bool = field(default=False, repr=False, kw_only=True)
    recovery_policy: str = field(default="legacy_v1", kw_only=True)
    recovery_window_s: float | None = field(default=None, kw_only=True)
    request_deadline_monotonic: float | None = field(default=None, repr=False, kw_only=True)
    attempt_observer: Callable[[Mapping[str, Any]], None] | None = field(default=None, repr=False, kw_only=True)
    client: Any = None
    _owns_client: bool = field(default=False, init=False)
    closed: bool = field(default=False, init=False)
    protocol: str = field(default="openai_chat_completions", init=False)
    output_token_parameter: str = field(default="max_tokens", kw_only=True)
    # The closed x_session_id_run_v1 mode: a non-secret run identity sent as
    # X-Session-ID so a gateway or prefix-caching router keeps one upstream
    # session (and its prompt cache) per run. Admitted for HTTPS endpoints and
    # HTTP loopback gateways; refused for plain HTTP to any other host.
    session_affinity_id: str | None = field(default=None, kw_only=True)
    # Profile-owned "strict" statement on function tools (openai_wire_options).
    tool_strictness: str | None = field(default=None, kw_only=True)
    reasoning_field: str = field(default="reasoning_content", kw_only=True)  # vLLM: "reasoning"
    # Profile-owned sampling / template / routing fields (openai_wire_options).
    request_options: Mapping[str, Any] | None = field(default=None, kw_only=True)

    checkpoint_request_options = staticmethod(checkpoint_request_options)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url or not self.api_key:
            raise PolicySessionError("native provider base URL and API key are required")
        if self.output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise PolicySessionError("unsupported OpenAI output token parameter")
        if self.tool_strictness not in {None, *TOOL_STRICTNESS_MODES} or self.reasoning_field not in REASONING_FIELDS:
            raise PolicySessionError("unsupported OpenAI tool_strictness or reasoning_field")
        try:
            self.request_options = validate_request_options(self.request_options)
        except ValueError as exc:
            raise PolicySessionError(f"unsupported OpenAI request_options: {exc}") from exc
        if self.session_affinity_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9:._-]{1,200}", self.session_affinity_id):
                raise PolicySessionError("invalid provider session-affinity identity")
            endpoint = urlsplit(self.base_url)
            if endpoint.scheme != "https" and (
                endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            ):
                raise PolicySessionError(
                    "provider session affinity is admitted only for a loopback gateway or an HTTPS endpoint"
                )
        if self.client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - dependency gate
                raise PolicySessionError("httpx is required for native_api policies") from exc
            self.client = httpx.Client(timeout=httpx.Timeout(
                self.timeout_seconds, connect=min(10.0, self.timeout_seconds),
            ))
            self._owns_client = True

    def _request_once(self, body: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
        raw_events: list[dict[str, Any]] = []
        first_event_at: float | None = None
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        if self.session_affinity_id is not None:
            headers["x-session-id"] = self.session_affinity_id
        response = self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
            content=canonical_bytes(body),
        )
        with response as stream:
            if stream.status_code >= 400:
                detail = stream.read().decode("utf-8", errors="replace")[-4000:]
                detail = detail.replace(self.api_key, "[REDACTED]")
                raise http_status_failure(stream.status_code, detail, base_url=self.base_url)
            if body.get("stream") is False:
                value = json.loads(stream.read())
                if not isinstance(value, Mapping):
                    raise PolicySessionError("native provider returned non-object JSON")
                return [dict(value)], 0.0
            try:
                announced_event: str | None = None
                finish_received = False
                for line in stream.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)
                    if text.startswith(":"):
                        continue
                    if text.startswith("event:"):
                        if announced_event is not None:
                            raise PolicySessionError(
                                "native provider SSE event lacks its data payload"
                            )
                        announced_event = text[6:].strip()
                        if not announced_event:
                            raise PolicySessionError(
                                "native provider SSE event name is empty"
                            )
                        continue
                    if not text.startswith("data:"):
                        raise PolicySessionError(
                            "native provider returned malformed SSE framing"
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
                            "native provider SSE event name does not match its payload"
                        )
                    announced_event = None
                    if first_event_at is None:
                        first_event_at = time.monotonic()
                    raw_events.append(dict(parsed))
                    choices = parsed.get("choices")
                    if isinstance(choices, list):
                        finish_received |= any(
                            isinstance(choice, Mapping)
                            and isinstance(choice.get("finish_reason"), str)
                            for choice in choices
                        )
                if announced_event is not None:
                    raise PolicySessionError(
                        "native provider SSE ended before an announced event payload"
                    )
                if not finish_received:
                    raise EOFError("native provider stream ended before completion status")
            except Exception as exc:
                if raw_events:
                    failure = stream_failure_after_content(exc)
                    failure.partial_events = raw_events
                    if self.api_key in json.dumps(raw_events, ensure_ascii=False):
                        failure.retryable = False
                    raise failure from exc
                raise
        return raw_events, first_event_at

    @profile_bound
    def prepare_request(
        request: Mapping[str, Any], *, output_token_parameter: str = "max_tokens",
        tool_strictness: str | None = None, reasoning_field: str = "reasoning_content",
        request_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the exact Chat Completions wire body (a provider's own options when bound)."""

        forbidden = sorted(set(request) & {
            "input", "instructions", "system", "thinking", "output_config",
            "input_schema",
        })
        if forbidden:
            raise PolicySessionError(
                "OpenAI canonical request contains Anthropic/Responses fields: "
                + ", ".join(forbidden)
            )
        messages = request.get("messages")
        model = request.get("model")
        if not isinstance(messages, list) or not isinstance(model, str) or not model:
            raise PolicySessionError("OpenAI request lacks messages or model")
        body = copy.deepcopy(dict(request))
        replayed: list[dict[str, Any]] = []
        for raw_message in messages:
            if not isinstance(raw_message, Mapping):
                raise PolicySessionError("OpenAI request message must be an object")
            message = copy.deepcopy(dict(raw_message))
            raw_state = message.pop(PROVIDER_CONTINUATION_MESSAGE_KEY, None)
            if raw_state is None:
                if {"reasoning_content", "reasoning"} & set(message):
                    raise PolicySessionError(
                        "unsigned OpenAI reasoning_content cannot enter replay history"
                    )
                replayed.append(message)
                continue
            if message.get("role") != "assistant" or not isinstance(raw_state, Mapping):
                raise PolicySessionError("OpenAI continuation is not bound to an assistant")
            state = ProviderContinuationState.from_mapping(raw_state)
            state.validate_binding(
                message=raw_message,
                provider_protocol="openai_chat_completions",
                model=model,
            )
            if state.native_kind != _OPENAI_CONTINUATION_KIND:
                raise PolicySessionError("unsupported OpenAI continuation native kind")
            if set(state.native_payload) != {"reasoning_content"}:
                raise PolicySessionError("OpenAI continuation payload is not canonical")
            reasoning = state.native_payload.get("reasoning_content")
            # An explicitly returned empty string is a valid signed payload:
            # some compatible providers use it for zero-reasoning tool turns.
            # Preserve it exactly; missing/non-string payloads remain invalid.
            if not isinstance(reasoning, str):
                raise PolicySessionError("OpenAI continuation lacks reasoning_content")
            message["reasoning_content"] = reasoning
            replayed.append(message)
        body["messages"] = replayed
        project_openai_tool_names(body, openai_tool_names(request))
        try:
            project_wire_body(body, tool_strictness=tool_strictness, reasoning_field=reasoning_field,
                              request_options=request_options)
        except ValueError as exc:
            raise PolicySessionError(str(exc)) from exc
        if output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise PolicySessionError("unsupported OpenAI output token parameter")
        if output_token_parameter == "max_completion_tokens" and "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        return body

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str
    ) -> dict[str, Any]:
        started_unix = time.time()
        started_mono = time.monotonic()
        window = RecoveryWindow.for_provider(self)
        body = self.prepare_request(request)  # the recorded wire request, exactly
        tool_names_by_alias = openai_tool_names(request)
        saw_content = False
        raw_events: list[dict[str, Any]] = []
        first_event_at: float | None = None
        attempts = 0
        while True:
            if self.closed:
                closed = ProviderClosed()
                _record_provider_failure(closed, attempts)
                raise closed
            attempts += 1
            attempt_started, attempt_mono = time.time(), time.monotonic()
            try:
                raw_events, first_event_at = (bounded_request(self, body, window)
                    if window is not None else self._request_once(body))
                break
            except Exception as exc:
                _record_provider_failure(exc, attempts)
                echoed = partial_credential_echo(exc, (self.api_key,))
                if window is not None:
                    delay = window.transport_delay(exc, attempts, started=attempt_mono)
                    partial = getattr(exc, "partial_events", [])
                    if self.closed or echoed:
                        delay = None
                    record_attempt_failure(self, exc, body=body, model_call_id=model_call_id,
                        attempt=attempts, credential_index=0, next_index=0 if delay is not None else None,
                        started=attempt_started, started_monotonic=attempt_mono,
                        retry_delay=delay or 0, events=partial, phase="transport")
                    if echoed:
                        raise _echo_failure(exc, attempts) from exc
                    if delay is None:
                        raise
                    try:
                        window.pause(delay)
                    except ProviderRecoveryDeadline as deadline:
                        _record_provider_failure(deadline, attempts)
                        raise deadline from exc
                    continue
                retryable = bool(getattr(exc, "retryable", False))
                if not retryable:
                    retryable = _retryable_transport_error(exc)
                partial = getattr(exc, "partial_events", [])
                will_retry = (
                    not self.closed
                    and not self.preflight_probe and not saw_content and retryable
                    and attempts < self.max_retries + 1 and not echoed
                )
                delay = min(2 ** (attempts - 1), 4) if will_retry else 0
                record_attempt_failure(self, exc, body=body, model_call_id=model_call_id,
                    attempt=attempts, credential_index=0,
                    next_index=0 if will_retry else None,
                    started=attempt_started, started_monotonic=attempt_mono,
                    retry_delay=delay, events=partial, phase="transport")
                if echoed:
                    raise _echo_failure(exc, attempts) from exc
                if not will_retry:
                    raise
                time.sleep(delay)

        try:
            content: list[str] = []
            reasoning: list[str] = []
            calls: dict[int, dict[str, Any]] = {}
            usage: dict[str, Any] | None = None
            finish_reason: str | None = None
            finish_seen = False
            for event in raw_events:
                normalized = normalize_usage(event.get("usage"))
                if normalized is not None:
                    usage = normalized
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                if len(choices) != 1 or not isinstance(choices[0], Mapping):
                    raise PolicySessionError(
                        "native provider must return exactly one response choice"
                    )
                if finish_seen:
                    raise PolicySessionError(
                        "native provider returned response content after completion"
                    )
                choice = choices[0]
                finish = choice.get("finish_reason")
                if isinstance(finish, str):
                    finish_reason = finish
                    finish_seen = True
                delta = choice.get("delta") if body.get("stream") is not False else choice.get("message")
                if not isinstance(delta, Mapping):
                    continue
                if isinstance(delta.get("content"), str):
                    content.append(delta["content"])
                    saw_content = True
                if isinstance(delta.get(self.reasoning_field), str):
                    reasoning.append(delta[self.reasoning_field])
                    saw_content = True
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for raw in tool_calls:
                        if not isinstance(raw, Mapping):
                            continue
                        index = raw.get("index", len(calls))
                        index = index if isinstance(index, int) else len(calls)
                        current = calls.setdefault(index, {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if isinstance(raw.get("id"), str):
                            current["id"] = raw["id"]
                        function = raw.get("function")
                        if isinstance(function, Mapping):
                            if isinstance(function.get("name"), str):
                                current["function"]["name"] += function["name"]
                            if isinstance(function.get("arguments"), str):
                                current["function"]["arguments"] += function["arguments"]
                        saw_content = True
            if self.api_key and self.api_key in json.dumps(raw_events, ensure_ascii=False):
                raise PolicySessionError("native provider echoed credential material; response was discarded")
            normalized_calls = []
            for index, call in sorted(calls.items()):
                if not call.get("id"):
                    call["id"] = f"{model_call_id}-tool-{index:02d}"
                call["function"]["name"] = response_tool_name(call["function"]["name"], tool_names_by_alias)
                normalized_calls.append(call)
            if finish_reason is None:
                incomplete = PolicySessionError(
                    "native provider response has no completion status; request was not retried"
                )
                setattr(incomplete, "retryable", False)
                raise incomplete
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(content),
            }
            if reasoning:
                message["reasoning_content"] = "".join(reasoning)
            if normalized_calls:
                message["tool_calls"] = normalized_calls
            output_rejection = None
            invalid_arguments = None if finish_reason == "length" else _reject_invalid_tool_arguments(
                message, provider_stop_reason=finish_reason)
            if finish_reason == "length" and normalized_calls:
                message, output_rejection = _reject_truncated_tools(
                    message, provider_stop_reason="length")
                finish_reason = "invalid_tool_input"
            elif invalid_arguments is not None:
                message, output_rejection = invalid_arguments
                finish_reason = "invalid_tool_input"
            elif reasoning:
                continuation = ProviderContinuationState.create(
                    provider_protocol="openai_chat_completions",
                    model=str(body.get("model", "")),
                    native_kind=_OPENAI_CONTINUATION_KIND,
                    native_payload={"reasoning_content": "".join(reasoning)},
                    canonical_message=message,
                )
                message[PROVIDER_CONTINUATION_MESSAGE_KEY] = continuation.to_dict()
        except PolicySessionError as exc:
            setattr(exc, "response_started", bool(raw_events))
            _record_provider_failure(exc, attempts)
            if window is not None:
                record_attempt_failure(self, exc, body=body, model_call_id=model_call_id,
                    attempt=attempts, credential_index=0, next_index=None,
                    started=attempt_started, started_monotonic=attempt_mono,
                    retry_delay=0, events=raw_events, phase="parse")
            raise
        if window is not None and window.remaining() <= 0:
            error = ProviderRecoveryDeadline(raw_events)
            _record_provider_failure(error, attempts)
            raise error
        finished = time.time()
        return {
            "message": message,
            "usage": usage,
            "finish_reason": finish_reason,
            **({"output_rejection": output_rejection} if output_rejection is not None else {}),
            "raw_events": raw_events,
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
