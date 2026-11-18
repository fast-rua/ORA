"""Python's same-credential recovery decisions over a grid of failures, for both deadline policies.

Usage: python tests/support/agent_core/provider/recovery_decisions.py OUT.json

Calls RecoveryWindow.for_provider / delay / transport_delay,
transient_transport and retry_after_seconds directly with a frozen clock (no
sleeping, no network), for exception shapes described as JSON specs that the
TypeScript test rebuilds with its own classes.
"""
from __future__ import annotations

import itertools
import json
import ssl
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

from ora.agent.contracts import PolicySessionError  # noqa: E402
from ora.agent.providers import recovery  # noqa: E402
from ora.agent.providers.recovery import (  # noqa: E402
    FIRST_TOKEN_POLICY, ProviderAttemptTimeout, ProviderRecoveryDeadline, RecoveryWindow, retry_after_seconds,
    transient_transport,
)
from ora.agent.providers.stream_failures import http_status_failure  # noqa: E402

NOW_MONO = 1000.0
NOW_UNIX = 1_800_000_000.0
#: ``no_content_since`` by the grid's ``prior``: none open, open 12.5 s, open exactly the 30 s window.
NO_CONTENT_SINCE = {0: None, 2: NOW_MONO - 12.5, 3: NOW_MONO - 30.0}
#: How long before its failure is seen (NOW_MONO) the grid's attempt started, by ``attempt``: a whole
#: 30 s first-token limit, at once, 1 s. Against a window open 30 s: started when it opened, at its
#: end, inside it.
STARTED_BEFORE = {1: 30.0, 3: 0.0, 6: 1.0}


class FrozenTime:
    def __init__(self) -> None:
        self.mono = NOW_MONO

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return NOW_UNIX

    def sleep(self, seconds) -> None:  # pragma: no cover - never called here
        raise AssertionError("no sleeping in decision tests")


FRAMES = {
    "connect_error": lambda: httpx.ConnectError("All connection attempts failed"),
    "remote_protocol": lambda: httpx.RemoteProtocolError("peer closed connection"),
    "read_timeout": lambda: httpx.ReadTimeout(""),
    "eof": lambda: EOFError("eof"),
    "ssl_cert": lambda: ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"),
    "ssl_eof": lambda: ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
    "ssl_bad_mac": lambda: ssl.SSLError(1, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac"),
    "ssl_want_read": lambda: ssl.SSLWantReadError(2, "The operation did not complete (read)"),
    "ssl_generic": lambda: ssl.SSLError(1, "[SSL: WRONG_VERSION_NUMBER] wrong version number"),
    "value_error": lambda: ValueError("v"),
    "pse": lambda: PolicySessionError("p"),
}


def build(spec):
    kind = spec["kind"]
    if kind == "deadline":
        return ProviderRecoveryDeadline(spec.get("events", []))
    if kind == "attempt_timeout":
        return ProviderAttemptTimeout(spec["limit"], spec["limit_s"], spec.get("events", []))
    if kind == "http":
        return http_status_failure(spec["status"], spec.get("detail", ""), base_url=spec.get("base_url", "https://x/v1"))
    if kind == "chain":
        exc = None
        for name in reversed(spec["frames"]):
            outer = FRAMES[name]()
            outer.__cause__ = exc
            exc = outer
        return exc
    exc = PolicySessionError("failure")
    for key, value in spec.get("attrs", {}).items():
        setattr(exc, key, value)
    return exc


def number(value):
    if value is None:
        return None
    return {"value": value, "float": isinstance(value, float)}


def exception_specs():
    specs = [{"kind": "deadline"}, {"kind": "deadline", "events": [{"a": 1}]},
             {"kind": "attempt_timeout", "limit": "first_token", "limit_s": 900.0},
             {"kind": "attempt_timeout", "limit": "stall", "limit_s": 180.0, "events": [{"a": 1}]}]
    # A worker's timeout as the parent rebuilds it from the result frame.
    specs.append({"kind": "pse", "attrs": {"retryable": True, "attempt_timeout": "first_token",
                                           "attempt_timeout_s": 30.0}})
    specs.append({"kind": "pse", "attrs": {"retryable": True, "attempt_timeout": "stall", "attempt_timeout_s": 180.0,
                                           "response_started": True, "partial_events": [{"x": 1}]}})
    for status in (400, 401, 402, 403, 404, 408, 409, 429, 500, 502, 503, 599, 600):
        specs.append({"kind": "http", "status": status})
    specs.append({"kind": "http", "status": 408, "base_url": "http://127.0.0.1:8317/v1",
                  "detail": json.dumps({"error": {"message": "stream disconnected before completion: x"}})})
    specs.append({"kind": "http", "status": 408, "base_url": "https://api.example/v1",
                  "detail": json.dumps({"error": {"message": "stream disconnected before completion"}})})
    specs.append({"kind": "http", "status": 408, "base_url": "http://[::1]:8317/v1",
                  "detail": "{\"error\": {\"message\": \"stream disconnected before completion\""})
    for frames in (["connect_error"], ["pse", "remote_protocol"], ["read_timeout"], ["eof"], ["pse", "eof"],
                   ["connect_error", "ssl_cert"], ["connect_error", "ssl_eof"], ["pse", "ssl_bad_mac"],
                   ["connect_error", "ssl_want_read"], ["ssl_generic"], ["value_error"], ["pse"],
                   ["pse", "value_error", "connect_error"], ["ssl_want_read", "value_error"],
                   ["ssl_eof", "ssl_cert"], ["ssl_cert", "ssl_eof"]):
        specs.append({"kind": "chain", "frames": frames})
    hints = [None, 3.0, 0.0, -1.0, 20.0, 250.0, 1.5]
    for retryable, hint, started, partial, upstream in itertools.product(
            [True, False, None], hints, [True, False, None], [None, [], [{"x": 1}]], [None, True, False]):
        attrs = {}
        if retryable is not None:
            attrs["retryable"] = retryable
        if hint is not None:
            attrs["retry_delay_s"] = hint
        if started is not None:
            attrs["response_started"] = started
        if partial is not None:
            attrs["partial_events"] = partial
        if upstream is not None:
            attrs["upstream_stream_break"] = upstream
        specs.append({"kind": "pse", "attrs": attrs})
    for status in (401, 403, 429, 503, None):
        specs.append({"kind": "pse", "attrs": {"retryable": True, "http_status": status}})
    return specs


def json_error_cases():
    """CPython json.loads messages (and implicit StopIteration context) on malformed text."""
    import random

    rng = random.Random(7)
    seeds = ['{"a": [1, 2.5, -3e2, true, false, null], "b": {"c": "d\\n\\u00e9\\ud83d\\ude00"}}',
             '[{"choices": [{"delta": {"content": "x"}, "finish_reason": null}]}]', '"plain"', "123", "-0.5e-3",
             '{"k": "\u4e2d\u6587", "n": [[], {}]}']
    fixed = ["", " ", "{bad", "[1,]", "[", "{", '{"a"', '{"a":', '{"a":1', '"abc', '"a\\x"', '"\\u12"',
             '"\\u1234', "1 2", "-", "-a", "01", "1.", "1e", "\ufeff{}", "tru", "NaN", "-Infinity", "[1 2]",
             '{"a" 1}', '"a\x01"', "\n\n  x", '["\\ud800\\u12"]', '["\\ud800\\uzzzz", 1]', '{"a":1}x',
             "not json", "[DONE]", '{"a":"\u2028', '{"é": "😀", x}', '{"a": 1,\n "b": ]}']
    texts = list(fixed)
    alphabet = list('{}[],:"\\ \n-.eE0123456789abtrufnlsxu\x01é😀')
    for _ in range(400):
        text = list(rng.choice(seeds))
        for _ in range(rng.randint(1, 3)):
            op = rng.random()
            pos = rng.randrange(len(text) + 1)
            if op < 0.4 and text:
                del text[min(pos, len(text) - 1)]
            elif op < 0.7:
                text.insert(pos, rng.choice(alphabet))
            elif text:
                text[min(pos, len(text) - 1)] = rng.choice(alphabet)
        texts.append("".join(text))
    out = []
    for text in texts:
        try:
            json.loads(text)
            out.append({"text": text, "ok": True})
        except json.JSONDecodeError as exc:
            context = exc.__context__
            out.append({"text": text, "message": str(exc),
                        "context": None if context is None else [type(context).__name__, str(context)]})
    return out


def main() -> None:
    frozen = FrozenTime()
    recovery.time = frozen
    for_provider = []
    for policy, window_s, run, preflight, retries in itertools.product(
            ["legacy_v1", "same_credential_deadline_v1", FIRST_TOKEN_POLICY, "other_v1"], [None, 0, 30, 900.0],
            [None, 1050.0, 1100, 2000.0, "nan", "inf", True, "x"], [False, True], [0, 2]):
        run_value = {"nan": float("nan"), "inf": float("inf")}.get(run, run) if isinstance(run, str) else run
        if run == "x":
            run_value = "x"
        provider = SimpleNamespace(recovery_policy=policy, recovery_window_s=window_s,
                                   request_deadline_monotonic=run_value, preflight_probe=preflight, max_retries=retries)
        item = {"policy": policy, "window_s": window_s, "run": run, "preflight": preflight, "max_retries": retries}
        try:
            window = RecoveryWindow.for_provider(provider)
            item["result"] = None if window is None else {
                # Infinite without a run deadline (first_token_deadline_v1): JSON has no infinity.
                "deadline": None if window.deadline == float("inf") else window.deadline,
                "retries_enabled": window.retries_enabled, "policy": window.policy,
                "first_token_s": window.first_token_s, "stall_s": window.stall_s,
                "no_content_window_s": window.no_content_window_s}
        except PolicySessionError as exc:
            item["error"] = str(exc)
        for_provider.append(item)
    delays = []
    for spec in exception_specs():
        for policy, attempt, remaining, enabled, prior in itertools.product(
                ["same_credential_deadline_v1", FIRST_TOKEN_POLICY], [1, 2, 3, 5, 6, 9], [100.0, 2.5, 0.0, None],
                [True, False], [0, 2, 3]):
            if policy != FIRST_TOKEN_POLICY and remaining is None:
                continue  # only first_token_deadline_v1 runs without a deadline
            if policy == FIRST_TOKEN_POLICY and (attempt not in (1, 3, 6) or remaining == 2.5):
                continue  # a smaller grid: its backoff is the deadline policy's
            exc = build(spec)
            deadline = float("inf") if remaining is None else NOW_MONO + remaining
            window = (RecoveryWindow(deadline, enabled, policy=FIRST_TOKEN_POLICY, first_token_s=30.0, stall_s=180.0)
                      if policy == FIRST_TOKEN_POLICY else RecoveryWindow(deadline, enabled))
            window.stream_breaks = prior
            window.first_token_timeouts = prior
            window.no_content_failures = prior + 5
            # No window open yet, one open 12.5 s (inside the 30 s window), one open exactly 30 s.
            window.no_content_since = NO_CONTENT_SINCE[prior]
            started = NOW_MONO - STARTED_BEFORE.get(attempt, 0.0)
            plain = window.delay(build(spec), attempt)
            transport = window.transport_delay(exc, attempt, started=started)
            delays.append({"spec": spec, "policy": policy, "attempt": attempt, "started": started,
                           "remaining": remaining,
                           "enabled": enabled, "prior": prior, "delay": number(plain),
                           "transport_delay": number(transport), "stream_breaks": window.stream_breaks,
                           "exc_stream_breaks": exc.stream_breaks,
                           "first_token_timeouts": window.first_token_timeouts,
                           "exc_first_token_timeouts": getattr(exc, "first_token_timeouts", None),
                           "no_content_failures": window.no_content_failures,
                           "exc_no_content_failures": getattr(exc, "no_content_failures", None),
                           "no_content_since": window.no_content_since,
                           "exc_no_content_window_s": getattr(exc, "no_content_window_s", None),
                           "exc_no_content_elapsed_s": getattr(exc, "no_content_elapsed_s", None)})
    transient = [{"spec": spec, "transient": transient_transport(build(spec))}
                 for spec in exception_specs() if spec["kind"] in ("chain", "http", "deadline")]
    retry_after = []
    for value in [None, "3", " 3 ", "1.5", "0", "-5", "1e3", "1_000", "+2", ".5", "5.", "inf", "-inf", "nan",
                  "Infinity", "abc", "", "1__0", "_1", "0x10", "Wed, 21 Oct 2015 07:28:00 GMT",
                  "Fri, 15 Jan 2027 08:00:30 GMT", "15 Jan 2027 08:01:00 +0000", "Fri, 15 Jan 2027 09:00:00 +0100",
                  "Fri, 15 Jan 2027 03:00:10 EST", "15 Jan 27 08:00:20 GMT", "Friday, 15 Jan 2027 08:00:40 UT",
                  "garbage date", "Fri, 32 Jan 2027 08:00:00 GMT"]:
        retry_after.append({"value": value, "seconds": number(retry_after_seconds(value))})
    out = {"now_mono": NOW_MONO, "now_unix": NOW_UNIX, "for_provider": for_provider, "delays": delays,
           "transient": transient, "retry_after": retry_after, "json_errors": json_error_cases()}
    Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps({"for_provider": len(for_provider), "delays": len(delays), "transient": len(transient),
                      "retry_after": len(retry_after)}))


if __name__ == "__main__":
    main()
