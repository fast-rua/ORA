"""``robot.query`` and the query allowance in the policy session (``ora/agent/pi/code_mode.py``, ``library: extended``).

A query is a read-only op of the script's socket: answered over the tools' data interface, charged to the script's
query allowance (``max_queries_per_script`` and ``query_time_s``), recorded as a nested ``robot.query`` request of the
script (a refused one too), never as a tool call, and counted against neither call limit. ``_code_mode_charge_query`` is the allowance's one entry, which ``dry_run`` calls too. A ``base``
contract serves no such op and states no allowance.
"""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from ora.agent.code_mode import query_spec
from ora.agent.code_mode.contract import QUERY_LIMITS, resolve_contract, with_reads
from ora.agent.code_mode.runtime import CodeModeToolRuntime
from ora.agent.code_mode.sandbox import read_op, sandbox_status
from ora.agent.code_mode.shell import CommandOutcome, run_command
from ora.agent.pi import code_mode as session_module
from ora.agent.pi.code_mode import PiCodeMode
from ora.agent.error_classes import ToolError
from tests.support.agent.kinematics_bridge import Bridge

POSE = [0.12, -0.2, 0.9, 0.0, 0.0, 0.0, 1.0]
SANDBOX = sandbox_status()


class Executor:
    def native_actions_per_action(self) -> int:
        return 1

    @contextlib.contextmanager
    def invocation(self, name, context, requested):
        yield SimpleNamespace(reentrant=False)


class Session(PiCodeMode):
    last_fresh_observation_event_id = "observation-event"

    def __init__(self, contract, host) -> None:
        self.code_mode, self.run_root, self.robot, self._serving = contract, host.control_root.parent, None, 0
        self.tool_action_executor = Executor()
        self.tools = CodeModeToolRuntime(base=host, contract=contract)
        self.gateway = SimpleNamespace(definitions=lambda: [])  # the model's own tools
        self.calls: list[str] = []
        self.records: list[tuple[str, dict[str, Any]]] = []  # what the robot socket's recorder was given
        self.artifacts = SimpleNamespace(put_json=lambda value, kind: {"kind": kind, "value": value})

    def _code_mode_call(self, name, arguments, *, model_call, parent, state):
        self.calls.append(name)
        return {"ok": True}

    def _robot_step(self) -> int:
        return 0

    def _script_record(self, event_type, fields):
        self.records.append((event_type, dict(fields)))
        return {"event_id": f"event-{len(self.records)}"}


def contract(**limits: Any) -> dict[str, Any]:
    return with_reads(resolve_contract({"library": "extended", **limits}), toolkit=True, moves=True)


@pytest.fixture
def host(published):
    published._toolkit_tools_enabled = True
    published.align_plan_callback = bridge = Bridge()
    published.motion_probe_callback = bridge.probe
    return published


def state() -> dict[str, Any]:
    return {"seq": 0, "actions": 0, "calls": 0, "cap": 12, "log": [], "frames": [], "uncounted_s": 0.0,
            "queries": 0, "query_s": 0.0}


def fake_script(monkeypatch, steps) -> None:
    """A script that does ``steps(handle_call, handle_op)``, in the real sandbox's place."""

    def run_command_(command, *, handle_call, handle_op=None, **options):
        steps(handle_call, handle_op)
        return CommandOutcome(status="completed", exit_code=0)

    monkeypatch.setattr(session_module, "run_command", run_command_)


# --- the allowance ---------------------------------------------------------------------------------------------

def test_the_charge_admits_before_the_work_and_adds_the_time_after(host) -> None:
    session = Session(contract(max_queries_per_script=2, query_time_s=10), host)
    spent = state()
    session._code_mode_charge_query(spent, 0.0)  # admission: counted, no time yet
    assert (spent["queries"], spent["query_s"]) == (1, 0.0)
    session._code_mode_charge_query(spent, 3.5, count=0)  # the time it took: never counted again, never refused
    assert (spent["queries"], spent["query_s"]) == (1, 3.5)
    session._code_mode_charge_query(spent, 2.0)  # a caller that charges once, after the work
    assert (spent["queries"], spent["query_s"]) == (2, 5.5)
    with pytest.raises(ToolError) as error:  # the count is spent
        session._code_mode_charge_query(spent, 0.0)
    assert error.value.code == "script_query_limit" and error.value.retryable is False
    assert error.value.constraints == {"max_queries": 2, "query_time_s": 10, "queries": 2, "query_s": 5.5}
    assert (spent["queries"], spent["query_s"]) == (2, 5.5), "a refusal charges nothing"
    session._code_mode_charge_query(spent, 99.0, count=0)  # a late charge is never refused
    assert spent["query_s"] == 104.5


def test_the_seconds_are_spent_before_the_count_when_a_query_overruns(host) -> None:
    session = Session(contract(query_time_s=3), host)
    spent = state()
    session._code_mode_charge_query(spent, 0.0)
    session._code_mode_charge_query(spent, 4.0, count=0)  # the last query may overrun: it was admitted
    with pytest.raises(ToolError) as error:
        session._code_mode_charge_query(spent, 0.0)
    assert error.value.code == "script_query_limit" and "4.0 s" in str(error.value) and "3 s" in str(error.value)
    assert error.value.constraints["queries"] == 1


def test_a_state_that_names_no_queries_starts_from_none_and_a_contract_without_limits_has_the_defaults(host) -> None:
    older = {key: value for key, value in contract().items() if key not in QUERY_LIMITS}
    session = Session(older, host)
    bare: dict[str, Any] = {}
    session._code_mode_charge_query(bare, 0.5)
    assert bare == {"queries": 1, "query_s": 0.5}
    bare.update(queries=QUERY_LIMITS["max_queries_per_script"])
    with pytest.raises(ToolError, match="script_query_limit|already made"):
        session._code_mode_charge_query(bare, 0.0)


# --- the op ----------------------------------------------------------------------------------------------------

def test_a_query_is_answered_charged_and_recorded_but_is_no_tool_call(host, monkeypatch) -> None:
    session = Session(contract(), host)
    answers = []
    asked = {"arm": "left", "joints_deg": [0] * 6}

    def steps(handle_call, handle_op):
        answers.append(handle_op({"op": "query", "name": "fk", "arguments": asked}))
        answers.append(handle_op({"op": "query", "name": "robot_state", "arguments": {}}))

    fake_script(monkeypatch, steps)
    result = session._code_mode_bash({"command": "true"}, {"model_call_id": "m", "tool_call_id": "c"})
    assert [item["ok"] for item in answers] == [True, True]
    assert answers[0] == {"ok": True, "name": "fk", "result": {"poses": [[0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]]}}
    assert set(answers[1]["result"]) == {"arms", "public"}
    assert result["calls"] == [] and session.calls == [], "no tool call: no entry in the result"
    # Each request is on the record as a nested call of the script, with what was asked and what was returned.
    assert [(kind, fields["tool_call_id"]) for kind, fields in session.records] == [
        ("tool_call_proposed", "c#001"), ("tool_result", "c#001"), ("tool_call_proposed", "c#002"),
        ("tool_result", "c#002")]
    proposed, returned = session.records[0][1], session.records[1][1]
    origin = {"kind": "code_mode", "parent_tool_call_id": "c", "seq": 1, "op": "query"}
    assert proposed["payload"] == {"name": "robot.query", "arguments": {"name": "fk", "arguments": asked},
                                   "effect": "read_only", "batch_valid": True, "origin": origin}
    assert proposed["model_call_id"] == "m" and returned["causal_parent_id"] == "event-1"
    assert returned["payload"] == {"name": "robot.query", "ok": True, "origin": origin}
    assert returned["artifact_ref"]["value"] == answers[0]
    assert result["actions_executed"] == 0
    left = result["allowance_left"]
    assert left["calls"] == 24 and left["actions"] == 12, "no call limit counts a query"
    assert left["queries"]["count"] == 30 and 179.0 < left["queries"]["seconds"] <= 180.0


def test_the_allowance_refuses_past_its_count_and_says_what_was_spent(host, monkeypatch) -> None:
    session = Session(contract(max_queries_per_script=2), host)
    answers = []

    def steps(handle_call, handle_op):
        for _ in range(3):
            answers.append(handle_op({"op": "query", "name": "robot_state", "arguments": {}}))

    fake_script(monkeypatch, steps)
    result = session._code_mode_bash({"command": "true"}, {"model_call_id": "m", "tool_call_id": "c"})
    assert [item["ok"] for item in answers] == [True, True, False]
    assert [fields["payload"]["ok"] for kind, fields in session.records if kind == "tool_result"] == [True, True, False]
    assert session.records[-1][1]["artifact_ref"]["value"] == answers[2], "a refused request is recorded too"
    assert answers[2]["error"]["code"] == "script_query_limit" and answers[2]["error"]["retryable"] is False
    assert answers[2]["error"]["constraints"]["max_queries"] == 2 and answers[2]["error"]["constraints"]["queries"] == 2
    assert result["allowance_left"]["queries"]["count"] == 0 and result["calls"] == []


def test_the_allowance_refuses_past_its_seconds(host, monkeypatch) -> None:
    clock = iter([0.0, 2.5, 100.0, 100.0, 100.0, 100.0])  # the first query takes 2.5 s of a 2 s allowance
    monkeypatch.setattr(session_module.time, "monotonic", lambda: next(clock))
    session = Session(contract(query_time_s=2), host)
    spent = state()
    first = session._code_mode_op({"op": "query", "name": "robot_state", "arguments": {}}, spent)
    second = session._code_mode_op({"op": "query", "name": "robot_state", "arguments": {}}, spent)
    assert first["ok"] is True, "the query that overran was admitted and is answered"
    assert second["ok"] is False and second["error"]["code"] == "script_query_limit"
    assert spent["queries"] == 1 and spent["query_s"] == pytest.approx(2.5)
    assert session._code_mode_allowance(spent)["allowance_left"]["queries"] == {"count": 31, "seconds": 0.0}


def test_what_cannot_be_asked_or_answered_costs_no_query(host) -> None:
    session = Session(contract(), host)
    spent = state()
    bad = session._code_mode_op({"op": "query", "name": "ik", "arguments": {"arm": "left"}}, spent)
    assert bad["ok"] is False and bad["error"]["code"] == "invalid_arguments" and bad["error"]["field"] == "poses"
    unknown = session._code_mode_op({"op": "query", "name": "dry_run", "arguments": {}}, spent)
    assert unknown["error"]["code"] == "unknown_query"
    assert session._code_mode_op({"op": "query", "name": "ik", "arguments": []}, spent)["error"]["field"] == "arguments"
    host.align_plan_callback = None
    nobridge = session._code_mode_op({"op": "query", "name": "ik", "arguments": {"arm": "left", "poses": POSE}}, spent)
    assert nobridge["error"]["code"] == "query_unavailable", "an unavailable query is charged for the ask, not refused free"
    assert spent["queries"] == 1
    host._toolkit_tools_enabled = False
    legacy = session._code_mode_op({"op": "query", "name": "robot_state", "arguments": {}}, spent)
    assert legacy["error"]["code"] == "query_unavailable" and "toolkit" in legacy["error"]["message"]
    assert spent["queries"] == 1, "no interface to bind: nothing to charge"


def test_an_unexpected_failure_still_answers_the_script(host, monkeypatch) -> None:
    session = Session(contract(), host)
    monkeypatch.setattr("ora.agent.code_mode.query.observation.proprioception",
                        lambda api: (_ for _ in ()).throw(RuntimeError("boom")))
    answer = session._code_mode_op({"op": "query", "name": "robot_state", "arguments": {}}, state())
    assert answer["ok"] is False and answer["error"]["code"] == "internal_tool_error"
    assert "RuntimeError" in answer["error"]["message"] and "boom" not in answer["error"]["message"]


def test_the_documentation_of_a_query_is_looked_up_like_a_functions(host) -> None:
    session = Session(contract(), host)
    index = session._code_mode_op({"op": "docs"})["text"]
    lines = [line for line in index.splitlines() if line.startswith(("- tools.", '- robot.query("'))]
    tools, queries = lines[:-7], lines[-7:]
    assert tools and all(line.startswith("- tools.") for line in tools), "the tools first, then one line per query"
    assert queries[0].startswith('- robot.query("ik", arm: str, poses: [[7]; 1-256], seed_joints_deg?: [6], ')
    assert [line.split(",")[0].split(")")[0] for line in queries] == [
        f'- robot.query("{name}"' for name in query_spec.NAMES]
    assert queries[-1].endswith('(robot.docs("query_robot_state"))')
    for spelling in ("query_ik", "query.ik"):
        entry = session._code_mode_op({"op": "docs", "name": spelling})
        assert entry["ok"] and entry["name"] == "query.ik" and "Inverse kinematics of one arm" in entry["text"]
    part = session._code_mode_op({"op": "docs", "name": "query_reach_check", "part": "arguments"})
    assert part["ok"] and "- interaction:" in part["text"] and "from_index" in part["text"]
    unknown = session._code_mode_op({"op": "docs", "name": "query_dry"})
    assert unknown["error"]["code"] == "unknown_function"
    allowed = unknown["error"]["constraints"]["allowed"]
    assert "query_ik" in allowed and "geometry_query" in allowed


def test_a_base_contract_serves_no_query_and_states_no_allowance(host, tmp_path) -> None:
    base = resolve_contract(True)
    session = Session(base, host)
    assert session._code_mode_ops(state()) == {} and session._code_mode_allowance(state()) == {}
    refusal = read_op({"op": "query", "name": "robot_state", "arguments": {}}, None)
    assert refusal["ok"] is False and refusal["error"]["code"] == "op_unavailable"


# --- from a script, through the sandbox -------------------------------------------------------------------------

needs_sandbox = pytest.mark.skipif(not SANDBOX["ok"], reason=f"code-mode sandbox unavailable: {SANDBOX.get('reason')}")
LIMITS = {"wall_s": 60, "cpu_s": 30, "memory_mib": 2048}


def program(tmp_path, session, spent, code: str):
    """``code`` as a ``bash`` command's program of ``session``, with the library's names imported."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "program.py").write_text("import json, robot\nfrom robot import ToolRefused\n" + code, encoding="utf-8")
    return run_command("python program.py", handle_call=lambda name, arguments: {"ok": False}, workspace=workspace,
                       extended=True, **session._code_mode_ops(spent), **LIMITS)


@needs_sandbox
def test_a_program_queries_with_typed_poses_and_degrees_and_reads_a_refusal(host, tmp_path) -> None:
    session = Session(contract(max_queries_per_script=3), host)
    spent = state()
    code = (
        "from ora_lib import Pose\n"
        "pose = Pose([0.12, -0.2, 0.9], [0, 0, 0, 1], frame='world')\n"
        "ik = robot.query('ik', arm='left', poses=[pose], seed_joints_deg=[0, 0, 0, 0, 0, 0])\n"
        "fk = robot.query('fk', arm='left', joints_deg=ik['rows'][0]['joints_deg'])\n"
        "try:\n    robot.query('ik', arm='left')\nexcept ToolRefused as error:\n    mistake = [error.code, error.answer['error']['field']]\n"
        "robot.query('robot_state')\n"
        "try:\n    robot.query('fk', arm='left', joints_deg=[0] * 6)\nexcept ToolRefused as error:\n    spent = error.code\n"
        "print(json.dumps([ik['rows'][0]['status'], [round(v, 6) for v in fk['poses'][0][:3]], mistake, spent]))")
    outcome = program(tmp_path, session, spent, code)
    assert outcome.status == "completed", outcome.output
    status, position, mistake, refusal = json.loads(outcome.output)
    assert status == "ok" and position == pytest.approx([0.12, -0.2, 0.9], abs=1e-5)
    assert mistake == ["invalid_arguments", "poses"] and refusal == "script_query_limit"
    assert outcome.calls == 0, "no tool call crossed the socket"
    assert spent["queries"] == 3


@needs_sandbox
def test_a_pose_in_another_frame_is_refused_in_the_sandbox_with_the_conversion_to_make(host, tmp_path) -> None:
    session = Session(contract(), host)
    code = ("from ora_lib import Pose\n"
            "try:\n    robot.query('ik', arm='left', poses=[Pose([0, 0, 0.1], [0, 0, 0, 1], frame='tcp:left')])\n"
            "except ValueError as error:\n    print(error)")
    outcome = program(tmp_path, session, state(), code)
    assert outcome.status == "completed", outcome.output
    assert "tcp:left" in outcome.output and ".to('world')" in outcome.output


@needs_sandbox
def test_a_bash_program_queries_through_the_robot_library(host, tmp_path) -> None:
    session = Session(contract(), host)
    spent = state()
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "look.py").write_text(
        "from robot import query\nprint(sorted(query('robot_state')['arms']))\n"
        "print(query('screen_path', knots_deg={'left': [[0] * 6, [10] * 6]})['status'])\n", encoding="utf-8")
    outcome = run_command("python look.py", handle_call=lambda name, arguments: {"ok": False}, workspace=workspace,
                          extended=True, **session._code_mode_ops(spent), **LIMITS)
    assert outcome.status == "completed", outcome.output
    assert outcome.output.split() == ["['left',", "'right']", "clear"]
    assert outcome.calls == 0 and spent["queries"] == 2


def test_a_base_script_can_not_query(host) -> None:
    from ora.agent.code_mode import robot

    with pytest.raises(RuntimeError, match="needs the extended library"):
        robot.query("robot_state")
