"""Same-credential recovery of one logical model call: attempt limits, retries, backoff.

Two recovery policies retry a failed attempt with the same credential:

``same_credential_deadline_v1``
    One fixed incident deadline (the provider's ``recovery_window_s``, else
    ``RECOVERY_WINDOW_S``) bounds the whole logical call: every attempt, its
    full streamed body and every backoff. Kept so that plans which name it
    still replay and verify.

``first_token_deadline_v1`` (the maintained default)
    No limit on a call's total duration. Each attempt must deliver its first
    stream event (for ``stream: false`` its whole body) within the provider's
    ``recovery_window_s`` -- read as the first-token limit per attempt -- and
    after that no SSE line may be more than ``STALL_S`` apart. Either timeout
    is a retryable transport failure (``ProviderAttemptTimeout``): a stall
    after content is a stream break and spends ``STREAM_BREAK_RETRIES``; a
    first-token timeout spends a budget of its own of the same size
    (``first_token_timeouts``). An attempt that fails before its first
    event, for any reason (HTTP 429/5xx, a transient connection or TLS fault,
    a first-token timeout), is retried with the same backoff when it started
    before the no-content window (``no_content_window_s``, the same
    ``recovery_window_s``) had passed since the first such failure; an
    attempt that delivers content closes the window. So a dead endpoint is
    given up on about one window after its first failure, not after the run's
    whole wall-time budget, and an outage shorter than the window always gets
    an attempt after it ends. Only the run's own deadline
    (``request_deadline_monotonic``), when one is set, bounds the call as a
    whole and still ends it with ``ProviderRecoveryDeadline``.
"""
from __future__ import annotations

import asyncio
import json
import math
import queue
import ssl
import subprocess
import sys
import threading
import time
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.canonical import canonical_bytes
from .attempts import _StreamInterrupted
from .stream_failures import STREAM_BREAK_RETRIES, http_status_failure, is_stream_break


#: One fixed deadline for the whole logical call (kept for plans that name it).
DEADLINE_POLICY = "same_credential_deadline_v1"
#: A first-token limit per attempt and a stall limit, with no total limit.
FIRST_TOKEN_POLICY = "first_token_deadline_v1"
#: The maintained recovery: what a profile's recovery window and the baselines select.
RECOVERY_POLICY = FIRST_TOKEN_POLICY
#: Every policy that retries with the same credential (``legacy_v1`` aside).
RECOVERY_POLICIES = frozenset({DEADLINE_POLICY, FIRST_TOKEN_POLICY})
#: The incident window, or under ``first_token_deadline_v1`` the first-token
#: limit per attempt, when the provider names no ``recovery_window_s``.
RECOVERY_WINDOW_S = 180.0
#: ``first_token_deadline_v1``: the longest gap between SSE lines (or events)
#: once an attempt's first event has arrived.
STALL_S = 180.0
#: How much later than its worker the parent process enforces the same
#: first-token and stall limits, so the worker's own verdict normally wins.
WATCHDOG_SLACK_S = 10.0
#: The single-attempt transport process ``bounded_request`` supervises.
WORKER_COMMAND = (sys.executable, "-m", "ora.agent.providers.recovery_worker")
_RETRYABLE_TLS_REASONS = frozenset({"SSLV3_ALERT_BAD_RECORD_MAC"})
#: TLS states that report socket flow, never trust.
#:
#: ``SSLWantReadError`` and ``SSLWantWriteError`` mean the operation would
#: block and should be repeated; ``SSLZeroReturnError`` means the peer closed
#: the connection cleanly. None of the three can carry a certificate verdict,
#: so none of them is the trust failure the fail-closed branch below exists to
#: catch. They are skipped rather than made retryable on their own: a benign
#: flow state is not itself a reason to retry, it is only a reason not to
#: disqualify whatever transport error wraps it.
_NON_TRUST_TLS_STATES = (
    ssl.SSLWantReadError, ssl.SSLWantWriteError, ssl.SSLZeroReturnError,
)


def retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


class ProviderRecoveryDeadline(PolicySessionError):
    def __init__(self, events=()):
        super().__init__("native provider recovery deadline exhausted; no reply was accepted")
        self.retryable = False
        self.partial_events = list(events)
        self.response_started = bool(events)


#: ``ProviderAttemptTimeout`` messages, by the limit that fired.
ATTEMPT_TIMEOUT_MESSAGES = {
    "first_token": "native provider attempt timed out before its first event",
    "stall": "native provider stream stalled after response content",
}


class ProviderAttemptTimeout(PolicySessionError):
    """One ``first_token_deadline_v1`` attempt exceeded its first-token or stall limit.

    A retryable transport failure: nothing reached the policy. ``attempt_timeout``
    names the limit (``first_token`` or ``stall``) and ``attempt_timeout_s`` its
    length; the attempt record carries both.
    """

    def __init__(self, limit: str, limit_s: float, events=()):
        super().__init__(ATTEMPT_TIMEOUT_MESSAGES[limit])
        self.retryable = True
        self.attempt_timeout = limit
        self.attempt_timeout_s = float(limit_s)
        self.partial_events = list(events)
        self.response_started = bool(events)


def transient_transport(exc: BaseException) -> bool:
    import httpx

    # HTTP transports can wrap certificate errors as ConnectError. Bad trust
    # configuration must fail closed, while a record-integrity alert before an
    # accepted response is safe to retry inside the existing incident window.
    #
    # Both judgements read the whole cause chain, because the provider raises
    # its faults wrapped. In a live failure the transport error that decides
    # retryability is never the outermost frame: `recovery_worker` classifies
    # the `PolicySessionError` it caught, so testing only that frame answered
    # False for every wrapped `httpx.TransportError` there has ever been.
    retryable_ssl = False
    transport = False
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError) and not isinstance(
            current, _NON_TRUST_TLS_STATES
        ):
            reason = str(getattr(current, "reason", "") or "").upper()
            message = str(current).upper()
            record_integrity_alert = any(
                reason == candidate or f"[SSL: {candidate}]" in message
                for candidate in _RETRYABLE_TLS_REASONS
            )
            if isinstance(current, ssl.SSLEOFError) or record_integrity_alert:
                retryable_ssl = True
            else:
                return False
        if isinstance(current, (httpx.TransportError, EOFError)):
            transport = True
        current = current.__cause__ or current.__context__
    return retryable_ssl or transport


@dataclass
class RecoveryWindow:
    """One logical request's incident: its deadline, attempt limits and counted retries.

    Under ``same_credential_deadline_v1`` ``deadline`` is the fixed incident
    deadline. Under ``first_token_deadline_v1`` it is the run's deadline, or
    infinite without one, ``first_token_s`` / ``stall_s`` bound each attempt,
    and ``no_content_window_s`` (``first_token_s`` unless given) bounds how
    long failures before content are retried, from ``no_content_since``.
    """

    deadline: float
    retries_enabled: bool
    stream_breaks: int = 0
    policy: str = DEADLINE_POLICY
    first_token_s: float | None = None
    stall_s: float | None = None
    first_token_timeouts: int = 0
    no_content_failures: int = 0
    no_content_window_s: float | None = None
    #: When the open no-content window opened (``time.monotonic()``); None while none is open.
    no_content_since: float | None = None

    def __post_init__(self):
        if self.policy == FIRST_TOKEN_POLICY and self.no_content_window_s is None:
            self.no_content_window_s = self.first_token_s

    @classmethod
    def for_provider(cls, provider):
        if provider.recovery_policy == "legacy_v1":
            return None
        if provider.recovery_policy not in RECOVERY_POLICIES:
            raise PolicySessionError("unsupported native provider recovery policy")
        # A slow provider (Kimi K3 at high effort streams single replies past
        # 180 s) carries its own window from the RunPlan.
        window = getattr(provider, "recovery_window_s", None) or RECOVERY_WINDOW_S
        deadline = time.monotonic() + window
        remaining_run = provider.request_deadline_monotonic
        if remaining_run is not None:
            if type(remaining_run) not in (float, int) or not math.isfinite(remaining_run):
                raise PolicySessionError("provider run deadline is invalid")
            deadline = min(deadline, remaining_run)
        enabled = not provider.preflight_probe and provider.max_retries > 0
        if provider.recovery_policy == FIRST_TOKEN_POLICY:
            # No incident deadline: only the run's own deadline bounds the call.
            run = math.inf if remaining_run is None else float(remaining_run)
            return cls(run, enabled, policy=FIRST_TOKEN_POLICY, first_token_s=float(window),
                       stall_s=float(STALL_S), no_content_window_s=float(window))
        return cls(deadline, enabled)

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def delay(self, exc: Exception, attempt: int) -> float | None:
        if (not self.retries_enabled or isinstance(exc, ProviderRecoveryDeadline)
                or getattr(exc, "http_status", None) in {401, 402, 403}
                or not (getattr(exc, "retryable", False) or transient_transport(exc))):
            return None
        hint = getattr(exc, "retry_delay_s", None)
        delay = float(hint) if type(hint) in (float, int) and math.isfinite(hint) and hint > 0 else min(2 ** min(attempt - 1, 5), 20.0)
        # Do not start a known-too-early request when Retry-After exceeds the
        # remaining incident budget. Spend the remaining wait then terminate.
        return min(delay, self.remaining()) if self.remaining() > 0 else None

    def transport_delay(self, exc: Exception, attempt: int, *, started: float) -> float | None:
        """``delay`` for a transport-phase failure, bounding stream breaks.

        ``started`` is the failed attempt's start (``time.monotonic()``).

        A break after response content is retried as a fresh request at most
        ``STREAM_BREAK_RETRIES`` times per logical call, inside the same
        deadline and backoff. The running count is stamped on the exception
        so the attempt record states how much of the budget was spent. Under
        ``first_token_deadline_v1`` a stall is such a break (it follows
        content), and first-token timeouts are counted apart, against a limit
        of the same size, as ``first_token_timeouts``. A failure without
        content, a first-token timeout included, is counted as
        ``no_content_failures`` and opens the no-content window when none is
        open. It is retried only when its attempt started less than
        ``no_content_window_s`` after the window opened: the window's age at
        ``started``, 0.0 for an attempt that started before it opened, is
        stamped on the exception as ``no_content_elapsed_s``. A break after
        content closes the window.

        The window opens when its first failure is seen, and each failure is
        judged by when its attempt started, not when it was seen. An attempt's
        first-token limit equals the window by default, so judged when seen, a
        first-token timeout inside an open window would always end the call:
        the request a restarting router or replica lost, which one retry
        recovers. Judged by its start, every attempt that starts inside the
        window is retried and the first one that starts after it is the last.
        An endpoint that refuses or answers 5xx is given up on about one
        window after its first failure, one that accepts and never answers
        after two retries, and an outage shorter than the window always gets
        an attempt after it ends.
        """
        delay = self.delay(exc, attempt)
        if is_stream_break(exc):
            self.stream_breaks += 1
            if self.stream_breaks > STREAM_BREAK_RETRIES:
                delay = None
        if self.policy == FIRST_TOKEN_POLICY:
            if getattr(exc, "attempt_timeout", None) == "first_token":
                self.first_token_timeouts += 1
                if self.first_token_timeouts > STREAM_BREAK_RETRIES:
                    delay = None
            if is_stream_break(exc):
                self.no_content_since = None  # content arrived: a later failure opens a new window
            else:
                if self.no_content_since is None:
                    self.no_content_since = time.monotonic()
                # The window's age when the failed attempt started; 0.0 if it started before the window opened.
                elapsed = max(0.0, float(started - self.no_content_since))
                self.no_content_failures += 1
                if self.no_content_window_s is not None and elapsed >= self.no_content_window_s:
                    delay = None
                setattr(exc, "no_content_elapsed_s", elapsed)
            setattr(exc, "first_token_timeouts", self.first_token_timeouts)
            setattr(exc, "no_content_failures", self.no_content_failures)
            setattr(exc, "no_content_window_s", self.no_content_window_s)
        setattr(exc, "stream_breaks", self.stream_breaks)
        return delay

    def pause(self, delay: float) -> None:
        time.sleep(min(delay, self.remaining()))
        if self.remaining() <= 0:
            raise ProviderRecoveryDeadline()


def _bounded_request_in_process(provider: Any, body: Mapping[str, Any], window: RecoveryWindow,
                                *, event_sink=None, progress_sink=None):
    """Use cancellable async I/O beneath the synchronous Provider contract.

    A read timeout alone restarts on each byte and cannot bound a trickling
    stream. wait_for encloses connection, headers, body and client cleanup;
    cancellation closes the attempt before another request may start. No
    detached thread can later publish a partial reply or execute tools.

    Under ``first_token_deadline_v1`` a watchdog cancels the attempt the same
    way when its first event takes longer than ``window.first_token_s`` from
    the attempt's start (connection, headers and lines included), or when,
    after that event, no SSE line arrives for ``window.stall_s``; the run's
    deadline, if any, still bounds it too. ``progress_sink`` hears of each
    line after the first event that is not itself an event.
    """
    import httpx

    events, first = [], None
    started = last_line = time.monotonic()
    anthropic = provider.protocol == "anthropic_messages"
    url = provider._messages_url() if anthropic else provider.base_url + "/chat/completions"
    headers = {"content-type": "application/json"}
    if anthropic:
        headers["anthropic-version"] = provider.anthropic_version
    # Both protocols carry the run's affinity identity; the provider refused it
    # at construction unless its gateway is admitted for it.
    if getattr(provider, "session_affinity_id", None) is not None:
        headers["x-session-id"] = provider.session_affinity_id
    if anthropic and provider.auth_scheme == "x_api_key":
        headers["x-api-key"] = provider.api_key
    else:
        headers["authorization"] = "Bearer " + provider.api_key
    keys = getattr(provider, "_credential_ring", (provider.api_key,))

    async def read():
        nonlocal first, last_line
        timeout = min(provider.timeout_seconds, window.remaining())
        if timeout <= 0:
            raise ProviderRecoveryDeadline()
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=min(10.0, timeout))) as client:
            async with client.stream("POST", url, headers=headers, content=canonical_bytes(body)) as stream:
                if stream.status_code >= 400:
                    detail = ""
                    # Authentication rejection is terminal at the headers;
                    # an unbounded error body must not turn it into an outage.
                    if stream.status_code not in {401, 402, 403}:
                        try:
                            raw_detail = await asyncio.wait_for(stream.aread(), min(2.0, window.remaining()))
                            detail = raw_detail.decode("utf-8", errors="replace")[-4000:]
                        except (asyncio.TimeoutError, httpx.TransportError):
                            detail = "error body unavailable"
                    for key in sorted(keys, key=len, reverse=True):
                        detail = detail.replace(key, "[REDACTED]")
                    exc = http_status_failure(stream.status_code, detail, base_url=provider.base_url)
                    hint = retry_after_seconds(stream.headers.get("retry-after"))
                    exc.retry_delay_s = hint if hint is not None else 20.0 if stream.status_code == 429 else None
                    raise exc
                if body.get("stream") is False:
                    value = json.loads(await stream.aread())
                    if not isinstance(value, Mapping):
                        raise PolicySessionError("native provider returned non-object JSON")
                    return [dict(value)], time.monotonic()
                announced, terminal = None, False
                try:
                    async for line in stream.aiter_lines():
                        last_line = time.monotonic()
                        if not line or line.startswith(":") or line.startswith("event:"):
                            if first is not None and progress_sink is not None:
                                progress_sink()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("event:"):
                            if announced is not None or not line[6:].strip():
                                raise PolicySessionError("native provider SSE event lacks its data payload")
                            announced = line[6:].strip()
                            continue
                        if not line.startswith("data:"):
                            raise PolicySessionError("native provider returned malformed SSE framing")
                        data = line[5:].strip()
                        if data == "[DONE]":
                            terminal = not anthropic
                            break
                        try:
                            parsed = json.loads(data)
                        except ValueError as exc:
                            raise PolicySessionError("native provider returned malformed SSE JSON") from exc
                        if not isinstance(parsed, Mapping) or announced is not None and parsed.get("type") != announced:
                            raise PolicySessionError("native provider SSE event does not match its payload")
                        announced = None
                        first = time.monotonic() if first is None else first
                        events.append(dict(parsed))
                        if event_sink is not None:
                            event_sink(dict(parsed))
                        if anthropic and parsed.get("type") in {"message_stop", "error"}:
                            terminal = True
                            break
                    if announced is not None or not terminal:
                        raise EOFError("native provider stream ended before protocol completion")
                except Exception as exc:
                    if events:
                        raise _StreamInterrupted(events, transient=transient_transport(exc)) from exc
                    raise
        return events, first

    async def run():
        if window.remaining() <= 0:
            raise ProviderRecoveryDeadline()
        try:
            return await asyncio.wait_for(read(), window.remaining())
        except asyncio.TimeoutError as exc:
            raise ProviderRecoveryDeadline(events) from exc

    async def watched():
        if window.remaining() <= 0:
            raise ProviderRecoveryDeadline()
        task = asyncio.ensure_future(read())
        fired = None
        try:
            while True:
                if task.done():
                    return task.result()
                if first is None:
                    fired, limit_s, until = "first_token", window.first_token_s, started + window.first_token_s
                else:
                    fired, limit_s, until = "stall", window.stall_s, max(first, last_line) + window.stall_s
                wait = min(until, window.deadline) - time.monotonic()
                if wait > 0:
                    done, _ = await asyncio.wait({task}, timeout=wait)
                    if done:
                        return task.result()
                    continue
                if window.remaining() <= 0:
                    fired = "deadline"
                break
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001 - the cancelled attempt's own end
                    pass
        # Raised outside any handler: the record's error chain is this timeout alone.
        if fired == "deadline":
            raise ProviderRecoveryDeadline(events)
        raise ProviderAttemptTimeout(fired, limit_s, events)
    return asyncio.run(watched() if window.policy == FIRST_TOKEN_POLICY else run())


def _frames(lines) -> list[dict]:
    frames = []
    for line in lines:
        try:
            frame = json.loads(line)
        except (ValueError, UnicodeError):
            continue  # A killed process may leave an incomplete final frame.
        if isinstance(frame, dict):
            frames.append(frame)
    return frames


def _supervise_worker(command, payload: bytes, cwd: Path, window: RecoveryWindow):
    """Run one ``first_token_deadline_v1`` worker under the attempt's own limits.

    There is no total deadline to hand ``subprocess.run``, yet a worker stuck
    where cancellation cannot reach (a DNS resolver thread) must not hang the
    run. The parent reads the worker's frames as they come and kills and reaps
    it when its first event takes longer than ``first_token_s``, or a later
    frame longer than ``stall_s``, each plus ``WATCHDOG_SLACK_S``, or when the
    run's deadline passes. Returns the frames, the limit that fired (or None), the
    exit status and whether a result frame arrived.
    """
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, cwd=cwd)
    arrivals: queue.Queue = queue.Queue()

    def feed():
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass  # The worker died or was killed; its exit decides the outcome.

    def drain():
        for line in iter(process.stdout.readline, b""):
            arrivals.put((time.monotonic(), line))
        arrivals.put((time.monotonic(), None))

    threads = [threading.Thread(target=target, daemon=True) for target in (feed, drain)]
    for thread in threads:
        thread.start()
    started = last = time.monotonic()
    first, frames, fired, finished = None, [], None, False
    try:
        while True:
            if first is None:
                kind, until = "first_token", started + window.first_token_s + WATCHDOG_SLACK_S
            else:
                kind, until = "stall", last + window.stall_s + WATCHDOG_SLACK_S
            wait = min(until, window.deadline) - time.monotonic()
            if wait <= 0:
                fired = "deadline" if window.remaining() <= 0 else kind
                break
            try:
                at, line = arrivals.get(timeout=wait)
            except queue.Empty:
                continue
            if line is None:
                break
            frame = next(iter(_frames([line])), None)
            if frame is None:
                continue
            frames.append(frame)
            last = at
            if frame.get("kind") == "event" and first is None:
                first = at
            if frame.get("kind") == "result":
                finished = True
                break
        if finished:
            # The attempt is over; only interpreter teardown is left to wait for.
            try:
                process.wait(timeout=WATCHDOG_SLACK_S)
            except subprocess.TimeoutExpired:
                pass
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=1.0)
        process.stdout.close()
    return frames, fired, process.returncode, finished


def bounded_request(provider: Any, body: Mapping[str, Any], window: RecoveryWindow):
    """Supervise one transport process, including DNS and client teardown.

    Async cancellation cannot forcibly cancel a platform DNS resolver thread;
    process ownership supplies that last wall-time bound. This worker has no
    tools or environment connection. Parent reaps it before retrying, and
    partial event frames remain failure evidence only.
    """
    if window.remaining() <= 0:
        raise ProviderRecoveryDeadline()
    first_token = window.policy == FIRST_TOKEN_POLICY
    binding = {"protocol": provider.protocol, "base_url": provider.base_url, "api_key": provider.api_key,
               "timeout_seconds": provider.timeout_seconds,
               "deadline": None if first_token and math.isinf(window.deadline) else window.deadline,
               "session_affinity_id": getattr(provider, "session_affinity_id", None), "body": dict(body)}
    if first_token:
        binding.update(recovery_policy=FIRST_TOKEN_POLICY, first_token_s=window.first_token_s,
                       stall_s=window.stall_s)
    if provider.protocol == "anthropic_messages":
        binding.update(url=provider._messages_url(), auth_scheme=provider.auth_scheme,
            anthropic_version=provider.anthropic_version)
    command = list(WORKER_COMMAND)
    cwd = Path(__file__).resolve().parents[3]
    fired = None
    if first_token:
        frames, fired, returncode, finished = _supervise_worker(command, canonical_bytes(binding), cwd, window)
        timed_out = fired == "deadline"
    else:
        try:
            outcome = subprocess.run(command, input=canonical_bytes(binding), stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=cwd, timeout=window.remaining(), check=False)
            output, timed_out = outcome.stdout, False
        except subprocess.TimeoutExpired as exc:
            # subprocess.run kills and reaps its exact child before propagating.
            output, timed_out = exc.stdout or b"", True
        frames, returncode = _frames(output.splitlines()), (None if timed_out else outcome.returncode)
        finished = False
    partial = [frame["event"] for frame in frames if frame.get("kind") == "event"]
    if timed_out or window.remaining() <= 0:
        raise ProviderRecoveryDeadline(partial)
    if fired in ATTEMPT_TIMEOUT_MESSAGES:
        limit_s = window.first_token_s if fired == "first_token" else window.stall_s
        raise ProviderAttemptTimeout(fired, limit_s, partial)
    final = next((frame for frame in reversed(frames) if frame.get("kind") == "result"), None)
    # A worker reaped only during its teardown already delivered its result.
    if (returncode != 0 and not finished) or final is None:
        exc = PolicySessionError("native provider transport worker failed before completion")
        exc.retryable = True
        exc.partial_events, exc.response_started = partial, bool(partial)
        raise exc
    if final.get("ok") is True:
        window.no_content_since = None  # content arrived: a later failure opens a new window
        return final["events"], final["first_event_at"]
    exc = PolicySessionError(str(final.get("message", "native provider transport failed")))
    for key in ("retryable", "http_status", "retry_delay_s", "response_started", "upstream_stream_break"):
        setattr(exc, key, final.get(key))
    if final.get("attempt_timeout") in ATTEMPT_TIMEOUT_MESSAGES:
        exc.attempt_timeout = final["attempt_timeout"]
        exc.attempt_timeout_s = final.get("attempt_timeout_s")
    exc.partial_events = final.get("partial_events", partial)
    exc.credential_echo = final.get("credential_echo") is True
    exc.remote_exception_type = final.get("exception_type")
    exc.remote_error_chain = final.get("error_chain", [])
    raise exc
