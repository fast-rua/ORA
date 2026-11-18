"""Operator stop versus in-flight policy work (RoboDojo D3R-006).

An explicit ``ora stop`` delivers SIGTERM to the Supervisor. Its
handler interrupts the PolicySession from another thread, which closes the
provider under any in-flight request. Every error or reply that surfaces after
that stop is the stop itself: the run ends with operator authority, no retry
or new request is started, no returned tool call runs, and the prefix is not a
valid behavioral sample. An identical failure without a stop stays an
infrastructure fault.

The episodes run on the Pi agent core; a stop while the core's own request is
in flight is ``tests/agent_core/test_operator_stop.py``. The Python providers
that preflight's probes use are checked here directly.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import httpx
import pytest

from ora.agent.evidence.canonical import canonical_bytes, sha256_bytes
from ora.agent.evidence.trace_verify import verify_agent_trace
from ora.agent.providers.anthropic_messages import AnthropicMessagesProvider
from ora.agent.providers.openai_compatible import OpenAICompatibleProvider
from ora.agent.providers.transport_failures import ProviderClosed
from ora.orchestrator.episode.policy_factory import build_policy_session
from ora.orchestrator.spec.run_spec import resolve_run_spec
from tests.support.orchestrator import multi_round
from tests.support.orchestrator.multi_round import live_exp01b_spec


ROOT = Path(__file__).resolve().parents[4]
BASE_CONFIG = "synthetic-native-atomic-causal-cache-v1"


def _operator_stop(until: Callable[[], bool]) -> None:
    """Deliver the stop signal and wait for the interrupt thread to land."""

    signal.raise_signal(signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while not until():
        if time.monotonic() > deadline:
            raise AssertionError("operator interrupt did not reach the session")
        time.sleep(0.005)


class _ClosedClientPolicy:
    """A policy runtime whose turn fails with the closed-client error once the stop landed."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.capabilities = delegate.capabilities

    def act(self, turn: Any) -> Any:
        _operator_stop(lambda: self.delegate.delegate.interrupted)
        raise RuntimeError(
            "Cannot send a request, as the client has been closed."
        )

    def interrupt(self, reason: str = "operator_interrupt") -> None:
        self.delegate.interrupt(reason)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def _plan() -> dict[str, Any]:
    spec = live_exp01b_spec(BASE_CONFIG)
    spec["experiment"] = "operator-stop-cancellation-test"
    spec["output_workspace"] = "outputs/operator-stop-cancellation-test"
    spec["milestone"] = "OPERATOR-STOP-QUALIFICATION"
    spec["runtime"]["terminal_actions"] = 1
    spec["budgets"] = {
        "max_robot_decisions": 2,
        "max_model_calls": 4,
        "max_sim_steps": 2,
        "wall_time_s": 30,
    }
    return resolve_run_spec(spec, root=ROOT)


def _run(
    tmp_path: Path, plan: Mapping[str, Any], responses: list[dict[str, Any]] | None, **kwargs: Any,
) -> tuple[dict[str, Any], Path]:
    run_root = tmp_path / "run"
    previous = signal.getsignal(signal.SIGTERM)
    try:
        result = multi_round.run(run_root, plan, responses, run_id="operator-stop", **kwargs)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return result, run_root


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def _assert_operator_stop(result: Mapping[str, Any], run_root: Path) -> None:
    assert result["state"] == "incomplete"
    assert result["termination"] == "operator_interrupt"
    assert result["termination_authority"] == "operator"
    assert result["behavioral_sample_valid"] is False
    assert result["official_success"] is None
    assert result["decisions"] == 0
    assert "error" not in result
    assert not (run_root / "native" / "runtime-fault.json").exists()
    events = _jsonl(run_root / "events.jsonl")
    kinds = {event.get("type") for event in events}
    assert "runtime_fault" not in kinds
    assert "trusted_fault" not in kinds
    assert "action_submitted" not in kinds
    decisions = [
        event["decision"] for event in events
        if event.get("type") == "termination_authority_decision"
    ]
    assert decisions and decisions[-1]["authority"] == "operator"
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    assert status["termination"] == "operator_interrupt"
    assert status["termination_authority"] == "operator"


def test_closed_client_error_after_stop_is_not_a_trusted_fault(
    tmp_path: Path,
) -> None:
    def policy_factory(**kwargs: Any) -> _ClosedClientPolicy:
        return _ClosedClientPolicy(build_policy_session(**kwargs))

    result, run_root = _run(
        tmp_path, _plan(), multi_round.close_every_request(4, "close_gripper-after-stop"),
        policy_session_factory=policy_factory,
    )

    _assert_operator_stop(result, run_root)
    cancelled = [
        event for event in _jsonl(run_root / "events.jsonl")
        if event.get("type") == "policy_turn_cancelled"
    ]
    assert len(cancelled) == 1
    assert cancelled[0]["error_type"] == "RuntimeError"
    assert "client has been closed" in cancelled[0]["error"]
    assert cancelled[0]["discarded_action"] is False


@pytest.fixture
def dropping_server() -> Iterator[int]:
    """A loopback model endpoint that reads each request and drops the connection unanswered."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("content-length", "0")))
            self.close_connection = True
            self.connection.shutdown(2)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_identical_transport_failure_without_stop_stays_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dropping_server: int,
) -> None:
    plan = _plan()
    policy = plan["policy"]
    endpoint = f"http://127.0.0.1:{dropping_server}/v1"
    policy.update(endpoint=endpoint, require_live_marker=False)
    monkeypatch.setenv(policy["base_url_env"], endpoint)
    monkeypatch.setenv(policy["api_key_env"], "sk-ant-local-transport-test-abcdefghij")
    monkeypatch.delenv(f"{policy['api_key_env']}S", raising=False)
    result, run_root = _run(tmp_path, plan, None)

    assert result["state"] == "incomplete"
    assert result["termination"] == "trusted_fault"
    assert result["termination_authority"] == "infrastructure"
    assert (run_root / "native" / "runtime-fault.json").is_file()
    assert not any(
        event.get("type") == "policy_turn_cancelled"
        for event in _jsonl(run_root / "events.jsonl")
    )
    # Unmarked, the failed request stays unanswered: the fault is evidence.
    agent_events = _jsonl(run_root / "agent-events.jsonl")
    (error,) = [e for e in agent_events if e.get("type") == "provider_error"]
    assert "interrupted" not in error["payload"]
    assert "request_without_response:model-call-000000" in verify_agent_trace(run_root)["errors"]
    # A marker without an operator stop before it is refused, not trusted.
    error["payload"]["interrupted"] = True
    _rewrite_chain(run_root / "agent-events.jsonl", agent_events)
    errors = verify_agent_trace(run_root)["errors"]
    assert any(item.endswith(":interrupted_provider_error") for item in errors)
    assert "request_without_response:model-call-000000" in errors


def _rewrite_chain(path: Path, events: list[dict[str, Any]]) -> None:
    previous = None
    for event in events:
        event["previous_event_sha256"] = previous
        event["event_sha256"] = sha256_bytes(canonical_bytes(
            {key: value for key, value in event.items() if key != "event_sha256"}))
        previous = event["event_sha256"]
    path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


_REQUEST = {
    "model": "m",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 64,
    "stream": True,
}


@pytest.mark.parametrize(
    "provider_type", [AnthropicMessagesProvider, OpenAICompatibleProvider],
)
def test_provider_closed_mid_request_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, provider_type: Any,
) -> None:
    monkeypatch.setattr(
        "ora.agent.providers.openai_compatible.time.sleep", lambda _s: None,
    )

    class Client:
        def __init__(self) -> None:
            self.calls: list[Any] = []
            self.provider: Any = None

        def stream(self, *_args: Any, **kwargs: Any) -> Any:
            self.calls.append(kwargs["content"])
            self.provider.close()
            raise httpx.ReadError("[Errno 9] Bad file descriptor")

    client = Client()
    provider = provider_type(
        "https://provider.invalid", "secret", client=client, max_retries=2,
    )
    client.provider = provider
    with pytest.raises(httpx.ReadError) as caught:
        provider.complete(request=_REQUEST, model_call_id="closed-mid-request")
    assert len(client.calls) == 1
    assert caught.value.provider_failure["attempts"] == 1

    with pytest.raises(ProviderClosed):
        provider.complete(request=_REQUEST, model_call_id="after-close")
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "provider_type", [AnthropicMessagesProvider, OpenAICompatibleProvider],
)
def test_open_provider_still_retries_the_same_transport_error(
    monkeypatch: pytest.MonkeyPatch, provider_type: Any,
) -> None:
    monkeypatch.setattr(
        "ora.agent.providers.openai_compatible.time.sleep", lambda _s: None,
    )

    class Client:
        calls: list[Any] = []

        def stream(self, *_args: Any, **kwargs: Any) -> Any:
            self.calls.append(kwargs["content"])
            raise httpx.ReadError("[Errno 9] Bad file descriptor")

    client = Client()
    client.calls = []
    provider = provider_type(
        "https://provider.invalid", "secret", client=client, max_retries=2,
    )
    with pytest.raises(httpx.ReadError):
        provider.complete(request=_REQUEST, model_call_id="open-retry")
    assert len(client.calls) == 3


def test_proposal_returned_after_stop_is_discarded(tmp_path: Path) -> None:
    from ora.agent import ActionProposal, NoAction
    from ora.orchestrator.episode.operator_stop import act_unless_stopped

    class Planner:
        def act(self, _turn: Any) -> ActionProposal:
            return ActionProposal(action={"action_id": "late"}, model_calls=2)

    events = tmp_path / "events.jsonl"
    stopped = act_unless_stopped(Planner(), None, lambda: True, events)
    assert isinstance(stopped, NoAction)
    assert stopped.reason == "operator_interrupt"
    assert stopped.model_calls == 2
    (record,) = _jsonl(events)
    assert record["type"] == "policy_turn_cancelled"
    assert record["discarded_action"] is True
    assert record["policy_outcome"] == "action"

    running = act_unless_stopped(Planner(), None, lambda: False, events)
    assert isinstance(running, ActionProposal)

    class Failing:
        def act(self, _turn: Any) -> Any:
            raise RuntimeError("transport failed")

    with pytest.raises(RuntimeError, match="transport failed"):
        act_unless_stopped(Failing(), None, lambda: False, events)
    assert len(_jsonl(events)) == 1
