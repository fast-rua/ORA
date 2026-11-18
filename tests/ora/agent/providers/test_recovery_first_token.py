"""first_token_deadline_v1: a first-token and a stall limit per attempt, no total limit.

Against local fakes only. The deadline policy it sits beside is
``test_recovery_deadline.py``'s.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ora.agent.contracts import PolicySessionError
from ora.agent.providers import factory, openai_compatible, recovery
from ora.agent.providers.anthropic_messages import AnthropicMessagesProvider
from ora.agent.providers.openai_compatible import OpenAICompatibleProvider
from ora.agent.providers.stream_failures import STREAM_BREAK_RETRIES

FT = recovery.FIRST_TOKEN_POLICY
KEY = "offline-primary-key"


def provider(protocol, url="https://offline.invalid", **kwargs):
    cls = AnthropicMessagesProvider if protocol == "anthropic" else OpenAICompatibleProvider
    return cls(url, KEY, recovery_policy=FT, **kwargs)


def request(stream):
    return {"model": "offline", "max_tokens": 64, "stream": stream,
            "messages": [{"role": "user", "content": "Act from current public evidence."}]}


def unary(protocol):
    if protocol == "anthropic":
        return {"type": "message", "id": "accepted", "role": "assistant", "model": "offline",
                "content": [{"type": "text", "text": "accepted"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1}}
    return {"id": "accepted", "choices": [{"message": {"role": "assistant", "content": "accepted"},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


def chunks(count, finish=True):
    """A Chat Completions stream of ``count`` content deltas."""
    lines = [{"choices": [{"index": 0, "delta": {"content": f"part {n} "}, "finish_reason": None}]}
             for n in range(count)]
    if finish:
        lines.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    return lines


class NoSleep:
    """``recovery.time`` without backoff sleeps, which are recorded instead."""

    def __init__(self):
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)

    def __getattr__(self, name):
        return getattr(time, name)


class VirtualTime(NoSleep):
    """``NoSleep`` whose monotonic clock moves only by the backoff it records.

    Fast attempts (an HTTP 503, a reply) take no time on it, so when each
    attempt starts and its failure is seen does not depend on how long a
    worker took to start. It stands in for ``time`` in both ``recovery`` and
    ``openai_compatible``, which reads each attempt's start.
    """

    def __init__(self):
        super().__init__()
        self.now = 0.0

    def sleep(self, seconds):
        super().sleep(seconds)
        self.now += seconds

    def monotonic(self):
        return self.now


class Frozen:
    """``recovery.time`` reading a clock the test sets."""

    def __init__(self, now=0.0):
        self.now = now

    def monotonic(self):
        return self.now


def ft_window(**kwargs):
    return recovery.RecoveryWindow(math.inf, True, policy=FT, first_token_s=30.0, stall_s=180.0, **kwargs)


def unavailable():
    exc = PolicySessionError("injected 503")
    exc.http_status, exc.retryable = 503, True
    return exc


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        server = self.server
        server.requests.append(self.path)
        step = server.script[min(len(server.requests), len(server.script)) - 1]
        if step == "silent":  # never answer the headers
            server.stopped.wait(30)
            return
        if step == "unavailable":
            data = b'{"error": {"message": "temporarily unavailable"}}'
            self.send_response(503)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if step == "unary":
            data = json.dumps(unary(server.protocol)).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            if step == "stall":  # two events, then nothing at all
                for event in chunks(2, finish=False):
                    self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
                server.stopped.wait(30)
                return
            # "steady": one event every gap_s, then [DONE]
            for event in chunks(server.steady_events):
                if server.stopped.wait(server.gap_s):
                    return
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def server():
    host = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host.requests, host.stopped, host.protocol = [], threading.Event(), "openai"
    host.steady_events, host.gap_s = 10, 0.3
    thread = threading.Thread(target=host.serve_forever, daemon=True)
    thread.start()
    yield host
    host.stopped.set()
    host.shutdown()
    host.server_close()
    thread.join()


def no_worker_left():
    children = subprocess.run(["ps", "--ppid", str(os.getpid()), "-o", "args="], capture_output=True, text=True)
    return "ora.agent.providers.recovery_worker" not in children.stdout


# -- decisions ----------------------------------------------------------------------------------


def test_the_window_is_a_per_attempt_first_token_limit_and_only_the_run_deadline_bounds_the_call():
    p = provider("openai", recovery_window_s=900.0)
    window = recovery.RecoveryWindow.for_provider(p)
    assert (window.policy, window.first_token_s, window.stall_s) == (FT, 900.0, recovery.STALL_S)
    # The same window bounds how long failures before content are retried.
    assert (window.no_content_window_s, window.no_content_since) == (900.0, None)
    assert ft_window().no_content_window_s == 30.0 and ft_window(no_content_window_s=60.0).no_content_window_s == 60.0
    assert recovery.RecoveryWindow(time.monotonic() + 100, True).no_content_window_s is None
    assert math.isinf(window.deadline) and math.isinf(window.remaining())
    p.request_deadline_monotonic = time.monotonic() + 50
    assert 0 < recovery.RecoveryWindow.for_provider(p).remaining() <= 50
    assert recovery.RecoveryWindow.for_provider(provider("openai")).first_token_s == recovery.RECOVERY_WINDOW_S
    assert recovery.STALL_S == 180.0 and recovery.RECOVERY_POLICY == FT


def test_a_first_token_timeout_and_a_stall_are_retryable_and_spend_separate_budgets():
    window = recovery.RecoveryWindow(math.inf, True, policy=FT, first_token_s=30.0, stall_s=180.0)
    delays = []
    for attempt in range(1, STREAM_BREAK_RETRIES + 2):
        exc = recovery.ProviderAttemptTimeout("first_token", 30.0)
        delays.append(window.transport_delay(exc, attempt, started=time.monotonic()))
        assert exc.first_token_timeouts == attempt and exc.stream_breaks == 0
    # Three retries, the fourth first-token timeout is terminal; backoff as for any transport failure.
    assert delays == [1, 2, 4, None]
    stall = recovery.ProviderAttemptTimeout("stall", 180.0, [{"choices": []}])
    assert window.transport_delay(stall, 1, started=time.monotonic()) == 1
    assert (stall.stream_breaks, stall.first_token_timeouts, stall.response_started) == (1, 4, True)
    # The deadline policy never stamps the first-token count.
    old = recovery.RecoveryWindow(time.monotonic() + 100, True)
    exc = recovery.ProviderAttemptTimeout("first_token", 30.0)
    assert old.transport_delay(exc, 1, started=time.monotonic()) == 1 and not hasattr(exc, "first_token_timeouts")


def test_both_recoveries_take_a_window_and_legacy_refuses_one():
    for policy in (FT, recovery.DEADLINE_POLICY):
        built = factory.build_provider("openai_compatible", base_url="https://offline.invalid", api_key=KEY,
                                       recovery_policy=policy, recovery_window_s=600.0)
        assert (built.recovery_policy, built.recovery_window_s) == (policy, 600.0)
    with pytest.raises(PolicySessionError, match="recovery_window_s requires"):
        factory.build_provider("openai_compatible", base_url="https://offline.invalid", api_key=KEY,
                               recovery_window_s=600.0)
    with pytest.raises(PolicySessionError, match="unsupported native provider recovery policy"):
        factory.build_provider("openai_compatible", base_url="https://offline.invalid", api_key=KEY,
                               recovery_policy="first_token_deadline_v2")


def test_failures_before_content_are_retried_until_the_no_content_window_has_passed(monkeypatch):
    clock = Frozen(1000.0)
    monkeypatch.setattr(recovery, "time", clock)
    window = ft_window()
    seen = []
    # Fast failures of attempts started 0, 1, 3, 7, 15, 29.9 and 30 s after the first, which opens
    # the window (30 s).
    for attempt, at in enumerate([0.0, 1.0, 3.0, 7.0, 15.0, 29.9, 30.0], start=1):
        clock.now = 1000.0 + at
        exc = unavailable()
        seen.append((window.transport_delay(exc, attempt, started=clock.now), exc.no_content_elapsed_s))
        assert (exc.no_content_failures, exc.no_content_window_s) == (attempt, 30.0)
    assert seen == [(1, 0.0), (2, 1.0), (4, 3.0), (8, 7.0), (16, 15.0), (20.0, pytest.approx(29.9)), (None, 30.0)]
    assert window.no_content_since == 1000.0
    # The count no longer decides: a longer window keeps retrying past the old eight retries.
    window = ft_window(no_content_window_s=900.0)
    for attempt in range(1, 41):
        clock.now = 2000.0 + 20 * attempt
        assert window.transport_delay(unavailable(), attempt, started=clock.now) is not None
    clock.now = 2000.0 + 20 + 900.0
    exc = unavailable()
    assert window.transport_delay(exc, 41, started=clock.now) is None and exc.no_content_failures == 41


def test_content_closes_the_no_content_window_and_breaks_keep_their_own_budget(monkeypatch):
    clock = Frozen(0.0)
    monkeypatch.setattr(recovery, "time", clock)
    window = ft_window()
    window.transport_delay(unavailable(), 1, started=0.0)
    clock.now = 25.0
    window.transport_delay(unavailable(), 2, started=25.0)
    # A stream break after content is not a failure without content: it closes the window.
    clock.now = 26.0
    broken = recovery.ProviderAttemptTimeout("stall", 180.0, [{"choices": []}])
    assert window.transport_delay(broken, 3, started=25.5) is not None
    assert (broken.no_content_failures, broken.stream_breaks, window.no_content_since) == (2, 1, None)
    assert not hasattr(broken, "no_content_elapsed_s")
    # 40 s after the first failure, yet the window the next failure opens is new.
    clock.now = 40.0
    exc = unavailable()
    assert window.transport_delay(exc, 4, started=40.0) == 8
    assert (exc.no_content_failures, exc.no_content_elapsed_s, window.no_content_since) == (3, 0.0, 40.0)
    # So does a completed transport attempt, before a parse-phase retry.
    window.no_content_since = 40.0
    monkeypatch.setattr(recovery, "_supervise_worker", lambda *args: (
        [{"kind": "result", "ok": True, "events": [{"choices": []}], "first_event_at": 41.0}], None, 0, True))
    p = provider("openai", recovery_window_s=30.0)
    assert recovery.bounded_request(p, {"stream": True}, window) == ([{"choices": []}], 41.0)
    assert window.no_content_since is None


def test_a_failure_is_judged_by_when_its_attempt_started(monkeypatch):
    """The window opens when its first failure is seen; an attempt that starts inside it is retried.

    The first-token limit equals the window, so judged when it is seen, a
    first-token timeout inside an open window would always end the call.
    """
    clock = Frozen(30.0)
    monkeypatch.setattr(recovery, "time", clock)
    # An endpoint that never answers: each attempt times out 30 s after it starts.
    window = ft_window()
    first = recovery.ProviderAttemptTimeout("first_token", 30.0)
    assert window.transport_delay(first, 1, started=0.0) == 1
    assert (first.first_token_timeouts, first.no_content_elapsed_s, window.no_content_since) == (1, 0.0, 30.0)
    # The retry starts 1 s into the window and is retried, although its timeout is seen 31 s in.
    clock.now = 61.0
    second = recovery.ProviderAttemptTimeout("first_token", 30.0)
    assert window.transport_delay(second, 2, started=31.0) == 2
    assert (second.first_token_timeouts, second.no_content_elapsed_s) == (2, 1.0)
    # The third starts 33 s in: two retries, and the call ends inside the first-token budget.
    clock.now = 93.0
    third = recovery.ProviderAttemptTimeout("first_token", 30.0)
    assert window.transport_delay(third, 3, started=63.0) is None
    assert (third.first_token_timeouts, third.no_content_failures, third.no_content_elapsed_s) == (3, 3, 33.0)
    # A restart: a 503 opens the window, and the request the restarting server lost
    # (started 1 s in, its timeout seen 31 s in) is retried.
    window = ft_window()
    clock.now = 100.0
    assert window.transport_delay(unavailable(), 1, started=100.0) == 1
    clock.now = 131.0
    lost = recovery.ProviderAttemptTimeout("first_token", 30.0)
    assert window.transport_delay(lost, 2, started=101.0) == 2 and lost.no_content_elapsed_s == 1.0
    # The first attempt that starts at or after the window's end is the last, whatever it fails with.
    clock.now = 133.0
    exc = unavailable()
    assert window.transport_delay(exc, 3, started=133.0) is None and exc.no_content_elapsed_s == 33.0
    # The first-token budget still ends a call whose timeouts content keeps separating.
    window = ft_window()
    delays = []
    for attempt in range(1, 8):
        clock.now = 1000.0 + 100 * attempt
        exc = (recovery.ProviderAttemptTimeout("first_token", 30.0) if attempt % 2
               else recovery.ProviderAttemptTimeout("stall", 180.0, [{"choices": []}]))
        delays.append(window.transport_delay(exc, attempt, started=clock.now - 30.0))
    assert delays[:-1] == [1, 2, 4, 8, 16, 20.0] and delays[-1] is None
    assert (window.first_token_timeouts, window.stream_breaks, exc.no_content_elapsed_s) == (4, 3, 0.0)


# -- the real worker process --------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_a_first_token_timeout_is_retried_and_the_retry_is_accepted(server, monkeypatch, protocol):
    clock = NoSleep()
    monkeypatch.setattr(recovery, "time", clock)
    server.protocol, server.script = protocol, ["silent", "unary"]
    failures = []
    p = provider(protocol, f"http://127.0.0.1:{server.server_port}", recovery_window_s=1.0,
                 attempt_observer=failures.append)
    started = time.monotonic()
    reply = p.complete(request=request(stream=False), model_call_id="one-call")
    assert reply["attempts"] == 2 and len(server.requests) == 2
    assert time.monotonic() - started >= 1.0
    [record] = failures
    assert (record["recovery_policy"], record["attempt_timeout"], record["attempt_timeout_s"]) == (FT, "first_token", 1.0)
    assert (record["first_token_timeouts"], record["stream_breaks"], record["partial_event_count"]) == (1, 0, 0)
    assert record["retry_scheduled"] is True and clock.sleeps == [1]
    assert record["error_chain"][-1] == {"type": "ProviderAttemptTimeout",
                                         "message": recovery.ATTEMPT_TIMEOUT_MESSAGES["first_token"]}
    assert no_worker_left()
    p.close()


def test_an_endpoint_that_never_answers_is_given_up_after_two_retries(server, monkeypatch):
    """The window opens at the first timeout; the third attempt starts a whole limit later, and is the last."""
    monkeypatch.setattr(recovery, "time", NoSleep())
    server.script = ["silent"]
    failures = []
    p = provider("openai", f"http://127.0.0.1:{server.server_port}", recovery_window_s=1.0,
                 attempt_observer=failures.append)
    with pytest.raises(PolicySessionError, match="timed out before its first event") as caught:
        p.complete(request=request(stream=True), model_call_id="one-call")
    assert caught.value.retryable is True  # retryable, yet it started after the window
    # Three attempts (a slow worker's attempt may time out before its request reaches the server).
    assert [row["attempt"] for row in failures] == [1, 2, 3] and 1 <= len(server.requests) <= 3
    assert [row["first_token_timeouts"] for row in failures] == [1, 2, 3]
    assert [row["retry_scheduled"] for row in failures] == [True, True, False]
    elapsed = [row["no_content_elapsed_s"] for row in failures]
    assert elapsed[0] == 0.0 and elapsed[1] < 1.0 <= elapsed[2]
    p.close()


def test_a_dead_endpoint_is_retried_until_the_window_and_given_up_with_its_last_error(server, monkeypatch):
    clock = VirtualTime()
    monkeypatch.setattr(recovery, "time", clock)
    monkeypatch.setattr(openai_compatible, "time", clock)
    server.script = ["unavailable"]
    failures = []
    p = provider("openai", f"http://127.0.0.1:{server.server_port}", recovery_window_s=30.0,
                 attempt_observer=failures.append)
    with pytest.raises(PolicySessionError, match="HTTP 503") as caught:
        p.complete(request=request(stream=False), model_call_id="one-call")
    assert caught.value.http_status == 503
    # Attempts start 0, 1, 3, 7, 15 and 31 s after the first failure: the sixth starts past the 30 s window.
    assert len(server.requests) == 6 and clock.sleeps == [1, 2, 4, 8, 16]
    assert [row["no_content_failures"] for row in failures] == [1, 2, 3, 4, 5, 6]
    assert [row["no_content_elapsed_s"] for row in failures] == [0.0, 1.0, 3.0, 7.0, 15.0, 31.0]
    assert [row["retry_scheduled"] for row in failures] == [True] * 5 + [False]
    assert {row["no_content_window_s"] for row in failures} == {30.0}
    assert all("no_content_retry_limit" not in row for row in failures)
    p.close()


@pytest.mark.parametrize("outage", [1, 9])
def test_an_outage_shorter_than_the_window_recovers(server, monkeypatch, outage):
    """Nine failures were the old limit; inside a 120 s window the tenth attempt is still made."""
    clock = VirtualTime()
    monkeypatch.setattr(recovery, "time", clock)
    monkeypatch.setattr(openai_compatible, "time", clock)
    server.script = ["unavailable"] * outage + ["unary"]
    failures = []
    p = provider("openai", f"http://127.0.0.1:{server.server_port}", recovery_window_s=120.0,
                 attempt_observer=failures.append)
    assert p.complete(request=request(stream=False), model_call_id="one-call")["attempts"] == outage + 1
    assert [row["no_content_failures"] for row in failures] == list(range(1, outage + 1))
    assert all(row["retry_scheduled"] for row in failures) and failures[-1]["no_content_elapsed_s"] < 120.0
    assert clock.sleeps == [1, 2, 4, 8, 16, 20.0, 20.0, 20.0, 20.0][:outage]
    p.close()


def test_a_stalled_stream_is_a_stream_break_retried_as_a_fresh_request(server, monkeypatch):
    monkeypatch.setattr(recovery, "time", NoSleep())
    monkeypatch.setattr(recovery, "STALL_S", 1.0)
    server.script, server.steady_events, server.gap_s = ["stall", "steady"], 2, 0.0
    failures = []
    p = provider("openai", f"http://127.0.0.1:{server.server_port}", recovery_window_s=5.0,
                 attempt_observer=failures.append)
    reply = p.complete(request=request(stream=True), model_call_id="one-call")
    assert reply["attempts"] == 2 and reply["message"]["content"] == "part 0 part 1 "
    [record] = failures
    assert (record["attempt_timeout"], record["attempt_timeout_s"], record["stream_breaks"]) == ("stall", 1.0, 1)
    assert record["partial_event_count"] == 2 and record["first_token_timeouts"] == 0
    p.close()


def test_a_slow_but_steady_stream_longer_than_the_window_succeeds_in_one_attempt(server, monkeypatch):
    """The regression: a healthy 22 tok/s reply ran past the 900 s incident deadline and was lost."""
    monkeypatch.setattr(recovery, "STALL_S", 1.0)
    server.script, server.steady_events, server.gap_s = ["steady"], 8, 0.3
    failures = []
    p = provider("openai", f"http://127.0.0.1:{server.server_port}", recovery_window_s=1.0,
                 attempt_observer=failures.append)
    started = time.monotonic()
    reply = p.complete(request=request(stream=True), model_call_id="one-call")
    assert time.monotonic() - started > 2.0  # well past the 1 s window
    assert reply["attempts"] == 1 and failures == [] and len(server.requests) == 1
    p.close()


def test_the_run_deadline_still_ends_the_call(server):
    server.script = ["silent"]
    failures = []
    p = provider("openai", f"http://127.0.0.1:{server.server_port}", recovery_window_s=30.0,
                 attempt_observer=failures.append)
    p.request_deadline_monotonic = time.monotonic() + 1.5
    started = time.monotonic()
    with pytest.raises(recovery.ProviderRecoveryDeadline):
        p.complete(request=request(stream=True), model_call_id="one-call")
    assert time.monotonic() - started < 3.0
    assert len(failures) == 1 and failures[0]["retry_scheduled"] is False
    assert no_worker_left()
    p.close()


# -- the parent's own watchdog ------------------------------------------------------------------

_STUCK_BEFORE = "import sys, time; sys.stdin.buffer.read(); time.sleep(60)"
_STUCK_AFTER = ("import json, sys, time; sys.stdin.buffer.read(); "
                "print(json.dumps({'kind': 'event', 'event': {'choices': []}}), flush=True); time.sleep(60)")


@pytest.mark.parametrize(("script", "limit"), [(_STUCK_BEFORE, "first_token"), (_STUCK_AFTER, "stall")])
def test_the_parent_kills_and_reaps_a_worker_stuck_past_its_limits(monkeypatch, script, limit):
    """A worker that cannot cancel itself (a DNS resolver thread) is still bounded."""
    monkeypatch.setattr(recovery, "WORKER_COMMAND", (sys.executable, "-c", script))
    monkeypatch.setattr(recovery, "WATCHDOG_SLACK_S", 0.5)
    monkeypatch.setattr(recovery, "STALL_S", 1.0)
    failures = []
    p = provider("openai", recovery_window_s=1.0, max_retries=0, attempt_observer=failures.append)
    started = time.monotonic()
    with pytest.raises(recovery.ProviderAttemptTimeout) as caught:
        p.complete(request=request(stream=True), model_call_id="one-call")
    assert time.monotonic() - started < 10
    assert caught.value.attempt_timeout == limit
    assert failures[0]["attempt_timeout"] == limit and failures[0]["retry_scheduled"] is False
    assert failures[0]["partial_event_count"] == (1 if limit == "stall" else 0)
    children = subprocess.run(["ps", "--ppid", str(os.getpid()), "-o", "args="], capture_output=True, text=True)
    assert "time.sleep(60)" not in children.stdout
    p.close()
