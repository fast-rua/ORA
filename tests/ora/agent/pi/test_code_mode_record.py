"""What a script asked of the library, on the record (``ora/agent/pi/code_mode.py``, ``agent/code_mode/record.py``).

The robot socket's real script path records the calls here, into a hash-chained trace: each ``query`` and ``explain``
request is a nested ``robot.<op>`` call pair of the script (``_code_mode_record_op``) with what was asked and what was
returned, numbered among the script's calls; each extended call's result records the inputs its explanation renders
from, so ``explain`` answers from the record for a call past the in-memory window or after a restart, the trace still
verifies and every explanation a script read renders again from the record alone. A ``base`` contract records what it
recorded before.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ora.agent.code_mode import record
from ora.agent.code_mode.contract import resolve_contract, with_reads
from ora.agent.evidence.agent_events import library_op
from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.evidence.trace_verify import verify_agent_trace
from ora.agent.pi import code_mode as session_module
from ora.agent.pi.code_mode import PiCodeMode
from ora.agent.pi.robot_socket import PiRobotSocket
from ora.agent.toolkit import explain
from ora.agent.toolkit.interface import stages
from tests.support.agent.agent_events import TrajectoryRecorder

ROOT = Path(__file__).resolve().parents[4]
ANSWERS = json.loads((ROOT / "tests/data/code_mode/perception-answers.json").read_text(encoding="utf-8"))["answers"]
LOOK = next(item for item in ANSWERS if item["tool"] == "geometry_query" and "points" in item["result"])
REFUSED = next(item for item in ANSWERS if item["tool"] == "geometry_query" and "error" in item["result"])
MODEL, SCRIPT = "model-call-000001", "call-script"



class Tools:
    """geometry_query answered from recorded results, noting the stage event its own code notes."""

    def __init__(self) -> None:
        self.answers = [LOOK["result"]]

    def invoke_with_context(self, name, arguments, *, call_context):
        stages.stage("depth_window", window_px=2)
        return dict(self.answers.pop(0) if len(self.answers) > 1 else self.answers[0])


class Session(PiCodeMode, PiRobotSocket):
    """The policy session's script path over the robot socket's own recording, into ``run_root``'s trace."""

    def __init__(self, root: Path, library: str | None = "extended", recorder: TrajectoryRecorder | None = None):
        contract = resolve_contract(True)
        self.code_mode = with_reads({**contract, **({"library": library} if library else {})}, toolkit=True,
                                    moves=True)
        self.run_root, self.artifacts, self.tools = root, ArtifactStore(root), Tools()
        self.recorder = recorder or TrajectoryRecorder(root, run_id="run", policy_session_id="pi-session")
        self.robot = SimpleNamespace(script_calls=0, snapshot={"observation": {"observation_step": 0}})
        self._script_names = {"geometry_query": "geometry_query"}
        self.gateway = SimpleNamespace(arguments=lambda name, arguments: dict(arguments),
                                       classify_effect=lambda name, arguments: "read_only", definitions=lambda: [])
        self.submitted_action = self.last_action_step = self.last_action_event_id = None
        self.last_fresh_observation_event_id, self.policy_session_id = None, "pi-session"

    def _call(self, method, params):  # the agent core's record RPC
        assert method == "record"
        return self.recorder.append(params["type"], **params["fields"])

    def _current_observation_id(self):
        return None  # nothing is kept for reuse here

    def _project(self, name, result, *, call_id, step):
        return {"visible": dict(result)}


def _state() -> dict[str, Any]:
    return {"seq": 0, "actions": 0, "calls": 0, "cap": 4, "log": [], "frames": [], "uncounted_s": 0.0,
            "queries": 0, "query_s": 0.0, "model_call_id": MODEL, "tool_call_id": SCRIPT}


def _look(session: Session, state: dict[str, Any], item: dict[str, Any] = LOOK) -> dict[str, Any]:
    return session._code_mode_call("geometry_query", item["arguments"], model_call=MODEL, parent=SCRIPT, state=state)


def _events(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]


def _artifact(root: Path, ref: dict[str, Any]) -> Any:
    return json.loads((root / ref["path"]).read_text())


def test_a_call_records_its_explain_inputs_and_a_request_is_a_nested_call_numbered_among_the_calls(tmp_path) -> None:
    session, state = Session(tmp_path), _state()
    first = _look(session, state)
    asked = session._code_mode_op({"op": "explain", "call_id": first["call_id"]}, state)
    second = _look(session, state)
    assert (first["call_id"], second["call_id"]) == (f"{SCRIPT}#001", f"{SCRIPT}#003"), "the request took #002"
    assert [entry["call"] for entry in state["log"]] == [1, 3] and state["calls"] == 2, "a request is no call"
    events = _events(tmp_path)
    assert [(event["type"], event["tool_call_id"], library_op(event)) for event in events] == [
        ("tool_call_proposed", f"{SCRIPT}#001", None), ("tool_result", f"{SCRIPT}#001", None),
        ("tool_call_proposed", f"{SCRIPT}#002", "explain"), ("tool_result", f"{SCRIPT}#002", "explain"),
        ("tool_call_proposed", f"{SCRIPT}#003", None), ("tool_result", f"{SCRIPT}#003", None)]
    # The call's result records what its explanation renders from: the stage event its tool noted among them.
    inputs = _artifact(tmp_path, events[1]["payload"]["explain_inputs"])
    assert inputs == {"schema": "ora.code-mode-explain-inputs.v1", "tool": "geometry_query",
                      "template_sha256": record.template_sha256("geometry_query"),
                      "renderer_sha256": record.RENDERER_SHA256, "arguments": LOOK["arguments"],
                      "answer": {"result": LOOK["result"]}, "events": [{"stage": "depth_window", "fields": {"window_px": 2}}]}
    assert set(events[1]["payload"]) == {"name", "ok", "origin", "explain_inputs"}
    # The request: what was asked and what was returned, read-only, under the script's call.
    proposed, returned = events[2], events[3]
    origin = {"kind": "code_mode", "parent_tool_call_id": SCRIPT, "seq": 2, "op": "explain"}
    assert proposed["payload"] == {"name": "robot.explain", "arguments": {"name": None, "arguments": {
        "call_id": first["call_id"]}}, "effect": "read_only", "batch_valid": True, "origin": origin}
    assert returned["payload"] == {"name": "robot.explain", "ok": True, "origin": origin}
    assert _artifact(tmp_path, returned["artifact_ref"]) == asked and asked["ok"] is True
    assert asked["text"] == explain.render("geometry_query", arguments=LOOK["arguments"], result=LOOK["result"],
                                           events=inputs["events"]).text
    unknown = session._code_mode_op({"op": "explain", "call_id": f"{SCRIPT}#002"}, state)
    assert unknown["error"]["code"] == "unknown_call", "a request is not a call to explain"
    assert _artifact(tmp_path, _events(tmp_path)[-1]["artifact_ref"]) == unknown, "a refused request is recorded"


def test_explain_answers_from_the_record_past_the_window_and_after_a_restart(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(session_module, "EXPLAIN_CALLS", 1)
    session, state = Session(tmp_path), _state()
    session.tools.answers = [LOOK["result"], REFUSED["result"]]
    first = _look(session, state)
    failed = _look(session, state, REFUSED)
    assert list(session._code_mode_calls) == [failed["call_id"]], "only the latest call is kept in memory"
    expected = explain.render("geometry_query", arguments=LOOK["arguments"], result=LOOK["result"],
                              events=[{"stage": "depth_window", "fields": {"window_px": 2}}]).text
    assert session._code_mode_op({"op": "explain", "call_id": first["call_id"]})["text"] == expected
    restarted = Session(tmp_path, recorder=session.recorder)  # a new session over the same run: nothing in memory
    assert restarted._code_mode_calls is None
    assert restarted._code_mode_op({"op": "explain", "call_id": first["call_id"]})["text"] == expected
    whole = restarted._code_mode_op({"op": "explain", "call_id": failed["call_id"]})["text"]
    assert failed["explain"] == record.attached(whole) and "refusal" in whole, "the attached text, from the record"
    assert restarted._code_mode_op({"op": "explain", "call_id": f"{SCRIPT}#009"})["error"]["code"] == "unknown_call"


def _sealed(root: Path, *, wrong: bool = False) -> Path:
    """One reply whose script looked twice (the second refused), asked for an explanation and got a query refused;
    ``wrong`` records a request whose text the record does not reproduce."""

    root.mkdir()
    (root / "plan.json").write_text(json.dumps({
        "policy": {"code_mode": {"contract": "ora.code-mode-script.v1", "library": "extended"}},
        "policy_composition": {"tool_surface": {"mutation_tools": ["move_to"]}},
    }), encoding="utf-8")
    recorder = TrajectoryRecorder(root, run_id="run", policy_session_id="pi-session")
    recorder.append("provider_request", observation_step=0, model_call_id=MODEL)
    recorder.append("provider_response", observation_step=0, model_call_id=MODEL)
    recorder.append("tool_call_proposed", observation_step=0, model_call_id=MODEL, tool_call_id=SCRIPT, payload={
        "name": "run_python", "arguments": {"code": "..."}, "effect": "mutation", "batch_valid": True})
    session, state = Session(root, recorder=recorder), _state()
    session.tools.answers = [LOOK["result"], REFUSED["result"]]
    first = _look(session, state)
    _look(session, state, REFUSED)
    session._code_mode_op({"op": "explain", "call_id": first["call_id"]}, state)
    session._code_mode_op({"op": "query", "name": "nowhere", "arguments": {}}, state)
    if wrong:
        session._code_mode_record_op("explain", None, {"call_id": first["call_id"]},
                                     {"ok": True, "call_id": first["call_id"], "text": "something else\n"},
                                     model_call=MODEL, parent=SCRIPT, state=state)
    recorder.append("tool_result", observation_step=0, model_call_id=MODEL, tool_call_id=SCRIPT,
                    payload={"name": "run_python", "ok": True},
                    artifact_ref=session.artifacts.put_json({"ok": True, "calls": state["log"]}, kind="native_tool_result"))
    return root


def test_the_trace_verifies_and_every_explanation_a_script_read_renders_again_from_it(tmp_path) -> None:
    root = _sealed(tmp_path / "run")
    verification = verify_agent_trace(root)
    assert verification["status"] == "valid", verification["errors"]
    script = _artifact(root, _events(root)[-1]["artifact_ref"])
    assert [("explain" in entry) for entry in script["calls"]] == [False, True], "the refused look carries its own"
    assert [library_op(event) for event in _events(root) if event["type"] == "tool_result"] == [
        None, None, "explain", "query", None]
    assert record.check_explanations(root) == []
    assert record.check_explanations(_sealed(tmp_path / "wrong", wrong=True)) == [
        f"code_mode_explain_not_reproduced:{SCRIPT}#005"]


def test_a_text_rendered_from_another_template_is_not_held_to_this_one(tmp_path, monkeypatch) -> None:
    root = _sealed(tmp_path / "run", wrong=True)
    monkeypatch.setattr(record, "template_sha256", lambda tool: "another template")
    assert record.check_explanations(root) == []


def test_a_text_rendered_by_other_renderer_code_is_not_held_to_this_one(tmp_path, monkeypatch) -> None:
    """A later change to ``toolkit/explain.py`` itself does not turn an old run's faithful record into a failed check."""
    root = _sealed(tmp_path / "run", wrong=True)
    assert record.check_explanations(root), "this code holds the text it rendered"
    monkeypatch.setattr(record, "RENDERER_SHA256", "the renderer as a later commit has it")
    assert record.check_explanations(root) == []


def test_a_base_contract_records_no_inputs_and_serves_no_request(tmp_path) -> None:
    session, state = Session(tmp_path, library=None), _state()
    answer = _look(session, state)
    assert "call_id" not in answer and session._code_mode_ops(state) == {}
    proposed, returned = _events(tmp_path)
    assert proposed["payload"] == {"name": "geometry_query", "arguments": LOOK["arguments"], "effect": "read_only",
                                   "batch_valid": True,
                                   "origin": {"kind": "code_mode", "parent_tool_call_id": SCRIPT, "seq": 1}}
    assert returned["payload"] == {"name": "geometry_query", "ok": True, "origin": proposed["payload"]["origin"]}
    assert session._code_mode_calls is None
