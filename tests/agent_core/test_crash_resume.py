"""The Pi agent core dies mid-episode; the restarted core finishes it exactly once.

One Synthetic episode with multi-call batches (two observes; an aggregated
``mark_point`` batch; a read plus a mutation the batch admission refuses;
single mutations; ``finish_episode``) runs on the Pi core once without a crash,
the reference, then again with the core killed once:

* at one of ten points of the turn (``ORA_PI_CRASH_AT=<point>:<n>``, the core
  SIGKILLs itself there): in flight, after the reply, after a recorded result,
  after Python ran a call TypeScript never recorded, after an aggregated batch
  ran, after the action, after the act finished; under lossless retention
  (native_slots_v4) and under rewrite_compact_v1 (native_slots_v3);
* while Python runs a tool that records its own events (the artifact reader's
  ``context_artifact_retrieval``), or after the core appended them;
* after the core appended one of the Supervisor's own records (receipt, fresh
  observation, verdict, termination), before answering it: the re-sent record
  reuses the recorded event;
* from outside, by SIGKILL to the node process while a provider request is in
  flight.

The bridge must restart the core once and resume the act. Every canonical
request is byte-identical to the reference's and none is recorded twice; the
event types equal the reference's once the resume evidence is dropped; each tool call has
one proposal and one result, and each tool execution ran once (counted at the
tool surface the bridge drives, not self-reported); the run is valid.
"""

from __future__ import annotations

import collections
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes
from tests.support.agent_core.episodes import tool_reply

pytestmark = pytest.mark.usefixtures("pi_core")

#: ``<point>:<n>``, see ``ora-pi/src/core/crash_points.ts``.
CRASH_POINTS = [
    "after_request:1", "after_response:1", "after_tool_result:1", "after_invoke:2", "after_batch_run:1",
    "after_action:1", "after_finish:1", "after_tool_result:7", "after_response:6", "after_invoke:6",
]


#: Retentions the crash points run under, each with the composition it requires.
RETENTIONS = {"action_step_lossless_v4": "native_slots_v4", "rewrite_compact_v1": "native_slots_v3"}
SESSION = "native-policy-00000000000000000000000000000001"  # episodes pins the policy-session id
# The first observe's tool image, a recoverable Context Artifact from action step 0.
IMAGE_REF = f"artifact://{SESSION}/1b56afbb3050f2f0d7b42a8666cbe27a382bb355e641973e60614adcee9c1f6b"
#: Crash points while a tool that records its own events runs: in flight in Python
#: (the core died after sending tools.invoke), after its answer, after the core
#: appended its events, after its result.
ARTIFACT_CRASH_POINTS = [
    "in_invoke:1", "in_invoke:4", "in_invoke:5", "after_invoke:4", "after_runtime_events:1",
    "after_runtime_events:2", "after_tool_result:6",
]
#: Crash points after the core appended a Supervisor record, before answering it.
RECORD_CRASH_POINTS = [
    "after_record.environment_receipt:1", "after_record.fresh_observation_published:2",
    "after_record.official_verdict:1", "after_record.run_terminated:1",
]


def batch_script() -> list[dict[str, Any]]:
    return [
        tool_reply(("r0a", "observe", {"views": ["agentview"]}), ("r0b", "observe", {"views": ["wrist"]})),
        tool_reply(("ma", "mark_point", {"point_id": "P0", "view": "wrist", "x": 2, "y": 1}),
                   ("mb", "mark_point", {"point_id": "P1", "view": "wrist", "x": 1, "y": 1})),
        tool_reply(("m0", "close_gripper", {})),
        tool_reply(("r1a", "observe", {"views": ["agentview"]}), ("m1", "open_gripper", {})),
        tool_reply(("m1b", "open_gripper", {})),
        tool_reply(("f0", "finish_episode", {"outcome": "failure", "reason": "crash probe", "attempts": ["scripted"]})),
    ]


def artifact_script() -> list[dict[str, Any]]:
    """``batch_script`` plus a reply whose tools record their own events: ``observe(ref)``
    reopens a historical image, ``artifact_read`` refuses an unknown ref."""

    replies = batch_script()
    replies.insert(3, tool_reply(("a0", "observe", {"ref": IMAGE_REF}),
                                 ("a1", "artifact_read", {"ref": f"artifact://{SESSION}/{'0' * 64}"})))
    return replies


@pytest.fixture(scope="module")
def references(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The uncrashed Pi run of a script under a retention, once per module."""

    runs: dict[tuple[str, str], dict[str, Any]] = {}

    def reference(script: str = "batch", retention: str = "action_step_lossless_v4") -> dict[str, Any]:
        if (script, retention) not in runs:
            with episodes.environment(ORA_PI_CRASH_AT=None, ORA_PI_FAKE_DELAY_MS=None):
                run = episodes.run_core("pi", tmp_path_factory.mktemp(f"crash-reference-{script}"),
                                        responses=SCRIPTS[script](), actions=2,
                                        composition=RETENTIONS[retention], retention=retention)
            assert run["core"] == "pi" and run["evaluation"]["status"] == "valid"
            assert not (run["root"] / "native" / "pi-restarts.jsonl").exists()
            runs[script, retention] = run
        return runs[script, retention]
    return reference


@pytest.fixture(scope="module")
def reference(references: Any) -> dict[str, Any]:
    return references()


SCRIPTS = {"batch": batch_script, "artifacts": artifact_script}


class Executions:
    """Each real tool execution, counted where the Pi bridge drives the tool surface."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ora.agent.pi import policy_session

        self.calls: list[tuple[str, ...]] = []
        self.planners: list[Any] = []
        self.in_batch = False
        original = policy_session.PiPlanner.__init__
        executions = self

        def init(planner: Any, *args: Any, **kwargs: Any) -> None:
            original(planner, *args, **kwargs)
            executions.planners.append(planner)
            tools, gateway = planner.tools, planner.gateway
            invoke = getattr(tools, "invoke_with_context", None)
            if callable(invoke):
                def counted(name: str, arguments: Any, *, call_context: Any) -> Any:
                    executions.calls.append(("invoke", str(call_context["model_call_id"]),
                                             str(call_context["tool_call_id"])))
                    return invoke(name, arguments, call_context=call_context)
                tools.invoke_with_context = counted
            else:  # rewrite_compact_v1: no artifact reader, tools run without a call context
                plain = tools.invoke

                def counted_plain(name: str, arguments: Any) -> Any:
                    if not executions.in_batch:  # an aggregated batch's calls count as "execute"
                        executions.calls.append(("invoke-plain", name, json.dumps(arguments, sort_keys=True)))
                    return plain(name, arguments)
                tools.invoke = counted_plain
            execute = gateway.execute

            def counted_execute(name: str, arguments: Any) -> Any:
                executions.calls.append(("execute", name, json.dumps(arguments, sort_keys=True)))
                executions.in_batch = True
                try:
                    return execute(name, arguments)
                finally:
                    executions.in_batch = False
            gateway.execute = counted_execute

        monkeypatch.setattr(policy_session.PiPlanner, "__init__", init)


def assert_resumed_once(reference: dict[str, Any], pi: dict[str, Any], executions: Executions,
                        invoked: int = 6) -> dict[str, Any]:
    root: Path = pi["root"]
    restarts_path = root / "native" / "pi-restarts.jsonl"
    restarts = [json.loads(line) for line in restarts_path.read_text().splitlines()] if restarts_path.exists() else []
    resumed = [event for event in pi["events"] if event["type"] == "agent_core_resumed"]
    assert sum(1 for record in restarts if "restart" in record) == 1, restarts
    assert len(resumed) == 1, resumed
    lost = resumed[0]["payload"]["mode"] == "request_lost"
    dropped = {"agent_core_resumed", "provider_attempt_failed"} if lost else {"agent_core_resumed"}
    assert pi["core"] == "pi"
    assert [reference["evaluation"]["status"], pi["evaluation"]["status"]] == ["valid", "valid"], \
        pi["evaluation"].get("reasons")
    assert pi["result"]["state"] == reference["result"]["state"]
    # No request is recorded twice: a lost in-flight request is re-sent under its
    # recorded provider_request, a replayed reply is not sent at all.
    assert len(pi["requests"]) == len(reference["requests"])
    for index, (a, b) in enumerate(zip(reference["requests"], pi["requests"])):
        assert a == b, (index, episodes.first_difference(json.loads(a), json.loads(b)))
    assert [e["type"] for e in pi["events"] if e["type"] not in dropped] == [e["type"] for e in reference["events"]]
    per_call: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for event in pi["events"]:
        if event["type"] in {"tool_call_proposed", "tool_result"}:
            per_call[event["tool_call_id"]][event["type"]] += 1
    assert {call: dict(count) for call, count in per_call.items()
            if count["tool_call_proposed"] != 1 or count["tool_result"] != 1} == {}
    invokes = collections.Counter(key for key in executions.calls if key[0] == "invoke")
    assert [list(key) for key, count in invokes.items() if count != 1] == []
    plain = [key for key in executions.calls if key[0] == "invoke-plain"]
    # Six calls (eight with the artifact reply) go through tools.invoke, the two
    # aggregated mark_point calls through the gateway; a call run twice adds one.
    assert len(invokes) + len(plain) == invoked, executions.calls
    assert sum(1 for key in executions.calls if key[0] == "execute") == 2
    return resumed[0]["payload"]


@pytest.mark.parametrize("retention", list(RETENTIONS))
@pytest.mark.parametrize("crash", CRASH_POINTS)
def test_pi_core_resumes_after_crash_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           references: Any, crash: str, retention: str) -> None:
    reference = references("batch", retention)
    executions = Executions(monkeypatch)
    monkeypatch.setenv("ORA_PI_CRASH_AT", crash)
    pi = episodes.run_core("pi", tmp_path, responses=batch_script(), actions=2,
                           composition=RETENTIONS[retention], retention=retention)
    assert (pi["root"] / "native" / "pi-crash-fired").read_text().strip() == crash
    assert_resumed_once(reference, pi, executions)


def retrievals(run: dict[str, Any]) -> list[list[Any]]:
    return [[event["tool_call_id"], event["payload"]["status"]]
            for event in run["events"] if event["type"] == "context_artifact_retrieval"]


@pytest.mark.parametrize("crash", ARTIFACT_CRASH_POINTS)
def test_a_tool_that_records_its_own_events_runs_and_records_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, references: Any, crash: str) -> None:
    """Its events travel with its result; a restarted core reproduces them from the cached execution."""

    reference = references("artifacts")
    assert retrievals(reference) == [["a0", "ok"], ["a1", "rejected"]]
    executions = Executions(monkeypatch)
    monkeypatch.setenv("ORA_PI_CRASH_AT", crash)
    pi = episodes.run_core("pi", tmp_path, responses=artifact_script(), actions=2)
    assert (pi["root"] / "native" / "pi-crash-fired").read_text().strip() == crash
    assert_resumed_once(reference, pi, executions, invoked=8)
    assert retrievals(pi) == retrievals(reference)


@pytest.mark.parametrize("crash", RECORD_CRASH_POINTS)
def test_a_record_resent_after_a_restart_is_recorded_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: dict[str, Any], crash: str) -> None:
    executions = Executions(monkeypatch)
    monkeypatch.setenv("ORA_PI_CRASH_AT", crash)
    pi = episodes.run_core("pi", tmp_path, responses=batch_script(), actions=2)
    assert (pi["root"] / "native" / "pi-crash-fired").read_text().strip() == crash
    restarts = [json.loads(line) for line in (pi["root"] / "native" / "pi-restarts.jsonl").read_text().splitlines()]
    assert [record["method"] for record in restarts if "restart" in record] == [
        "official_verdict" if crash.startswith("after_record.official_verdict") else "record"]
    # Between acts nothing is resumed: the events equal the reference's exactly, none twice.
    assert [e["type"] for e in pi["events"]] == [e["type"] for e in reference["events"]]
    assert [reference["evaluation"]["status"], pi["evaluation"]["status"]] == ["valid", "valid"]
    assert pi["requests"] == reference["requests"]


def test_pi_core_resumes_after_external_sigkill_in_flight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          reference: dict[str, Any]) -> None:
    """SIGKILL from outside the core while its fourth provider request is in flight."""

    executions = Executions(monkeypatch)
    # Every scripted reply takes 1.5 s, so the request is in flight when the kill lands.
    monkeypatch.setenv("ORA_PI_FAKE_DELAY_MS", "1500")
    events_path = tmp_path / "pi" / "agent-events.jsonl"
    killed: dict[str, Any] = {}
    done = threading.Event()

    def kill_in_flight() -> None:
        while not done.is_set():
            text = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
            if text.count('"type": "provider_request"') >= 4:
                time.sleep(0.4)
                process = executions.planners[0].process
                killed.update(pid=process.pid, events=text.count("\n"))
                os.kill(process.pid, signal.SIGKILL)
                return
            time.sleep(0.05)

    killer = threading.Thread(target=kill_in_flight, daemon=True)
    killer.start()
    try:
        pi = episodes.run_core("pi", tmp_path, responses=batch_script(), actions=2)
    finally:
        done.set()
        killer.join(timeout=5)
    assert killed, "the episode finished before the fourth provider request"
    assert not (pi["root"] / "native" / "pi-crash-fired").exists()
    resumed = assert_resumed_once(reference, pi, executions)
    assert resumed["mode"] == "request_lost", resumed
