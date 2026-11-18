"""Private single-attempt transport process; stdin/stdout are parent-owned IPC."""
from __future__ import annotations

import json
import math
import sys
import time
from types import SimpleNamespace

from .recovery import FIRST_TOKEN_POLICY, RecoveryWindow, _bounded_request_in_process, transient_transport
from .attempts import _failure_evidence, echoes_credential


def main() -> None:
    value = json.loads(sys.stdin.buffer.read())
    provider = SimpleNamespace(**{key: value[key] for key in (
        "protocol", "base_url", "api_key", "timeout_seconds", "session_affinity_id")})
    provider._credential_ring = (provider.api_key,)
    if provider.protocol == "anthropic_messages":
        for key in ("auth_scheme", "anthropic_version"):
            setattr(provider, key, value[key])
        provider._messages_url = lambda: value["url"]

    sent = [time.monotonic()]

    def send(frame):
        sys.stdout.write(json.dumps(frame, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        sent[0] = time.monotonic()

    def heartbeat():
        # Lines that are not events still show the parent's stall watchdog a live stream.
        if time.monotonic() - sent[0] >= 1.0:
            send({"kind": "progress"})

    if value.get("recovery_policy") == FIRST_TOKEN_POLICY:
        deadline = math.inf if value["deadline"] is None else value["deadline"]
        window = RecoveryWindow(deadline, False, policy=FIRST_TOKEN_POLICY,
                                first_token_s=value["first_token_s"], stall_s=value["stall_s"])
    else:
        window = RecoveryWindow(value["deadline"], False)
    try:
        events, first = _bounded_request_in_process(provider, value["body"], window,
            event_sink=lambda event: send({"kind": "event", "event": event}),
            progress_sink=heartbeat if window.policy == FIRST_TOKEN_POLICY else None)
        send({"kind": "result", "ok": True, "events": events, "first_event_at": first})
    except Exception as exc:
        raw_partial = getattr(exc, "partial_events", [])
        # Decided on the raw events: the frame below carries only redacted ones.
        echo = echoes_credential(raw_partial, (provider.api_key,))
        evidence = _failure_evidence(exc, keys=(provider.api_key,), events=raw_partial)
        send({"kind": "result", "ok": False, "exception_type": type(exc).__name__,
            "message": str(exc).replace(provider.api_key, "[REDACTED]"),
            "credential_echo": echo,
            "retryable": not echo and bool(getattr(exc, "retryable", False) or transient_transport(exc)),
            "http_status": getattr(exc, "http_status", None),
            "retry_delay_s": getattr(exc, "retry_delay_s", None),
            "response_started": bool(getattr(exc, "response_started", False)),
            "upstream_stream_break": getattr(exc, "upstream_stream_break", False) is True,
            **({"attempt_timeout": exc.attempt_timeout, "attempt_timeout_s": exc.attempt_timeout_s}
               if getattr(exc, "attempt_timeout", None) is not None else {}),
            "partial_events": evidence["partial_events"], "error_chain": evidence["error_chain"]})


if __name__ == "__main__":
    main()
