"""``dry_run=True`` on a motion tool in the policy session (``ora/agent/pi/code_mode.py``, ``library: extended``).

The flag reaches the tool out of band (``dry_run.dry``), never as an argument, and only where the surface plans on the
policy side; the call is recorded as a read-only nested call of the script (``origin.dry_run``) and charged to the
script's queries, not its calls or actions; its answer is the plan's record in degrees, through the causal filter, with
its ``explain``. A ``base`` contract, a tool without a dry run and ``dry_run=False`` are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ora.agent.code_mode import record
from ora.agent.code_mode.contract import resolve_contract, with_reads
from ora.agent.pi.code_mode import PiCodeMode
from ora.agent.session_core.tool_arguments import _tool_error
from ora.agent.toolkit import explain
from ora.agent.toolkit.interface import dry_run, stages
from ora.agent.error_classes import ToolError

RECORD = {
    "dry_run": True, "action": "move_to", "status": "planned", "feasible": True, "failure_code": None,
    "failure_stage": None, "route": "line",
    "arms": {"left": {"target_pose": [0.4, 0, 0.2, 0, 0, 0, 1], "predicted_pose": [0.4, 0, 0.2, 0, 0, 0, 1],
                      "end_joints_deg": [22.9, 0, 11.5, 0, 0, 0], "joint6": {"min_limit_margin_deg": 170.0}}},
    "screening": {"status": "clear", "segments_screened": 16, "min_inter_arm_clearance_m": None, "against": "x"},
    "motion": {"native_actions": 16, "whole_native_actions": 16, "settle_native_actions": 3, "duration_s": 0.64,
               "fraction_of_move": 1.0, "native_action_budget": 50},
    # what the tool's own constraints carry in radians and a key the causal filter drops
    "compiled_pose": {"targets": {"left": {"rotation_angle_rad": 3.141592653589793, "sha256": "x"}}},
    "claim_boundary": "the robot's own motion",
}
EVENTS = [("compiled", {"arm": "left", "interaction": "free_space", "translation_distance_m": 0.1,
                        "rotation_angle_deg": 0.0, "jaw_symmetry": "requested", "target_reachable": True})]


class Gateway:
    def classify_effect(self, name: str, arguments: Any) -> str:
        return "mutation"

    def arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in arguments.items() if value is not None}

    def definitions(self) -> list[dict[str, Any]]:
        return []


class Session(PiCodeMode):
    """Script calls answered with a dry-run record, noting how the tool would have been asked."""

    def __init__(self, contract: dict[str, Any], root: Path, *,
                 capabilities: tuple[str, ...] = ("policy_angles_degrees_v1",), record: dict[str, Any] | None = None,
                 images: list[dict[str, Any]] | None = None):
        self.code_mode, self.run_root, self.gateway = contract, root, Gateway()
        self.images = images or []
        self._script_names = {name: name for name in ("move_to", "move_pose", "vision_ground", "recover_pose")}
        self.tools = SimpleNamespace(surface=SimpleNamespace(capabilities=capabilities))
        self.record, self.seen = RECORD if record is None else record, []
        self.artifacts = SimpleNamespace(put_json=lambda value, kind: {"kind": kind})

    def _current_observation_id(self) -> str | None:
        return None

    def _script_call(self, call, *, ids, origin, preset, result_fields=None):
        self.seen.append({"arguments": dict(call.arguments), "dry": dry_run.requested(), "collecting": stages.active(),
                          "origin": dict(origin), "preset": preset})
        if preset is not None and preset[1] is not None:
            result = _tool_error(preset[1])
        else:
            for name, fields in EVENTS:
                stages.stage(name, **fields)
            result = dict(self.record if dry_run.requested() else {"status": "executed", "remaining_budget": 3})
        outcome = {"ok": "error" not in result, "call_id": ids[1], "tool": call.name, "result": result,
                   "observation_step": 0, "images": self.images}
        if result_fields is not None:
            self.seen[-1]["recorded"] = result_fields(outcome)
        return outcome

    def _project(self, name, result, *, call_id, step):
        return {"visible": result}


def _session(tmp_path: Path, library: str | None = "extended", **kwargs: Any) -> Session:
    contract = with_reads({**resolve_contract(True), **({"library": library} if library else {})},
                          toolkit=True, moves=True)
    return Session(contract, tmp_path, **kwargs)


def _state() -> dict[str, Any]:
    return {"seq": 0, "actions": 0, "calls": 0, "cap": 4, "log": [], "frames": [], "uncounted_s": 0.0}


def _call(session: Session, name: str, arguments: dict[str, Any], state: dict[str, Any] | None = None):
    state = _state() if state is None else state
    return session._code_mode_call(name, arguments, model_call="model-1", parent="call-1", state=state), state


def test_a_dry_run_asks_the_tool_out_of_band_and_is_recorded_as_a_read_only_call_of_the_script(tmp_path) -> None:
    session = _session(tmp_path)
    answer, state = _call(session, "move_to", {"targets": {"left": {"position_m": [0.4, 0, 0.2]}}, "dry_run": True})
    (asked,) = session.seen
    assert asked["arguments"] == {"targets": {"left": {"position_m": [0.4, 0, 0.2]}}}, "dry_run is not a tool argument"
    assert asked["dry"] is True and asked["collecting"] is True and not dry_run.requested() and not stages.active()
    assert asked["origin"] == {"kind": "code_mode", "parent_tool_call_id": "call-1", "seq": 1, "dry_run": True}
    assert asked["preset"] == ("read_only", None), "never a world-changing call: no admission, no action"
    assert answer["ok"] is True and answer["dry_run"] is True and answer["call_id"] == "call-1#001"
    assert answer["tool"] == "move_to" and answer["result"]["feasible"] is True
    assert (state["actions"], state["calls"]) == (0, 0), "not a call, not an action"
    assert state["log"] == [{"call": 1, "tool": "move_to", "ok": True, "dry_run": True}]


def test_a_dry_run_counts_against_the_queries_and_nothing_else(tmp_path) -> None:
    session = _session(tmp_path)
    _answer, state = _call(session, "move_to", {"targets": {}, "dry_run": True})
    assert state["queries"] == 1 and state["query_s"] >= 0.0
    assert (state["actions"], state["calls"]) == (0, 0)
    limited = _session(tmp_path)
    limited.code_mode = {**limited.code_mode, "max_queries_per_script": 1}
    state = _state()
    _call(limited, "move_to", {"targets": {}, "dry_run": True}, state)
    answer, state = _call(limited, "move_to", {"targets": {}, "dry_run": True}, state)
    assert answer["ok"] is False and answer["error"]["code"] == "script_query_limit" and state["queries"] == 1
    assert len(limited.seen) == 2 and limited.seen[1]["preset"][1].code == "script_query_limit"
    assert limited.seen[1]["arguments"] == {"targets": {}}, "the refusal is recorded; no plan was made for it"


def test_the_answer_is_the_record_in_degrees_through_the_causal_filter_with_its_explanation(tmp_path) -> None:
    session = _session(tmp_path)
    answer, state = _call(session, "move_to", {"targets": {}, "dry_run": True})
    compiled = answer["result"]["compiled_pose"]["targets"]["left"]
    assert compiled == {"rotation_angle_deg": pytest.approx(180.0)}, "degrees; the hash is not read by a policy"
    assert "claim_boundary" in answer["result"] and "_rad" not in json.dumps(answer)
    events = [{"stage": name, "fields": fields} for name, fields in EVENTS]
    expected = record.render(record.explain_inputs("move_to", {"targets": {}}, answer, events))
    assert expected.text.startswith(explain.render("move_to", arguments={"targets": {}}, result=answer["result"],
                                                   events=events).text)
    assert answer["explain"] == expected.text.rstrip() and answer["explain"].startswith("move_to(left)  dry run: feasible")
    assert "explain" not in state["log"][0], "a plan that works is not repeated in the script's calls"
    assert session._code_mode_op({"op": "explain", "call_id": "call-1#001"}) == {
        "ok": True, "call_id": "call-1#001", "text": expected.text}
    plain = _session(tmp_path, capabilities=())
    assert _call(plain, "move_to", {"targets": {}, "dry_run": True})[0]["result"]["compiled_pose"]["targets"]["left"] == {
        "rotation_angle_rad": pytest.approx(3.141592653589793)}


def test_an_infeasible_plan_is_an_answer_whose_explanation_the_script_result_carries(tmp_path) -> None:
    record = {"dry_run": True, "action": "move_to", "status": "infeasible", "feasible": False,
              "failure_code": "table_collision", "failure_stage": "screening",
              "refusal": {"code": "table_collision", "message": "the arm goes below the table", "constraints": {
                  "fraction_of_move": 0.5, "native_action": 3}}, "claim_boundary": "x"}
    session = _session(tmp_path, record=record)
    answer, state = _call(session, "move_to", {"targets": {}, "dry_run": True})
    assert answer["ok"] is True and answer["result"]["feasible"] is False
    assert answer["explain"].startswith("move_to(left)  dry run: would be refused, table_collision")
    assert state["log"][0]["explain"] == answer["explain"], "a failed plan is in the calls like a failed call"


def test_the_numbers_and_the_picture_reach_the_model_whether_the_script_prints_them_or_not(tmp_path) -> None:
    record = {**RECORD, "summary": "left: 16 native actions, least clearance 23 mm at 62% of the path"}
    session = _session(tmp_path, record=record, images=[{"path": "/run/preview.png", "sha256": "x"}])
    _answer, state = _call(session, "move_to", {"targets": {}, "dry_run": True})
    (entry,) = state["log"]
    assert entry["summary"] == record["summary"] and entry["images"] == 1
    assert state["frames"] == ["/run/preview.png"]


def test_a_refused_plan_s_own_words_reach_the_model_too(tmp_path) -> None:
    record = {key: value for key, value in RECORD.items() if key not in ("arms", "route", "screening", "motion")}
    record = {**record, "status": "infeasible", "feasible": False, "failure_code": "goal_unreachable",
              "failure_stage": "plan", "refusal": {"code": "goal_unreachable", "message": "no joint solution here"}}
    _answer, state = _call(_session(tmp_path, record=record), "move_to", {"targets": {}, "dry_run": True})
    assert state["log"][0]["summary"] == "no joint solution here"


def test_dry_run_false_is_a_real_call_without_the_key_and_a_non_boolean_is_refused(tmp_path) -> None:
    session = _session(tmp_path)
    answer, state = _call(session, "move_to", {"targets": {}, "dry_run": False})
    assert session.seen[0]["arguments"] == {"targets": {}} and session.seen[0]["dry"] is False
    assert "dry_run" not in answer and answer["result"]["status"] == "executed"
    answer, _state2 = _call(session, "move_to", {"targets": {}, "dry_run": "yes"})
    assert answer["ok"] is False and answer["error"]["code"] == "invalid_arguments"
    assert answer["error"]["field"] == "dry_run" and session.seen[1]["dry"] is False
    answer, _state3 = _call(session, "move_to", {"targets": {}, "dry_run": None})
    assert answer["result"]["status"] == "executed" and session.seen[2]["arguments"] == {"targets": {}}


def test_a_tool_without_a_dry_run_leaves_the_argument_to_the_tool_and_every_library_has_a_dry_run(tmp_path) -> None:
    session = _session(tmp_path)
    _call(session, "vision_ground", {"camera": "head", "dry_run": True})
    assert session.seen[0]["arguments"] == {"camera": "head", "dry_run": True} and session.seen[0]["dry"] is False
    for library in ("base", None):
        base = _session(tmp_path, library=library)
        answer, _state = _call(base, "move_to", {"targets": {}, "dry_run": True})
        assert base.seen[0]["arguments"] == {"targets": {}} and base.seen[0]["dry"] is True
        assert answer["dry_run"] is True


def test_a_plan_reports_itself_and_nothing_of_the_official_limit(tmp_path) -> None:
    """A plan's line is the plan's own; what is used of the official limit is not part of it (removed 2026-11-03:
    give-ups quoted it in place of the plan)."""
    record = {**RECORD, "summary": "left: 16 native actions, least clearance 23 mm at 62% of the path"}
    session = _session(tmp_path, record=record)
    loop = SimpleNamespace(native_allowance=SimpleNamespace(used=905), plan={"runtime": {"native_action_limit": 1050}})
    session.tools = SimpleNamespace(surface=session.tools.surface, toolkit_executor=SimpleNamespace(loop=loop))
    answer, state = _call(session, "move_to", {"targets": {}, "dry_run": True})
    assert answer["result"]["summary"] == record["summary"] and state["log"][0]["summary"] == record["summary"]
    assert "official" not in json.dumps(answer) and "1050" not in json.dumps(state["log"])


#: Three rotations: none, a quarter turn about world z (both exact at five decimals) and a general one.
ROTATIONS = [([0.0, 0.0, 0.0, 1.0], 1e-6), ([0.0, 0.0, 0.7071067811865476, 0.7071067811865476], 1e-6),
             ([0.1, -0.3, 0.2, 0.9], 5e-6)]


@pytest.mark.parametrize("quaternion, tolerance", ROTATIONS)
def test_a_plan_gives_each_arm_s_predicted_axes_in_world(tmp_path, quaternion, tolerance) -> None:
    """RO-27: a plan's predicted pose came as a quaternion only, and 38 calls converted it by hand. Each arm's record
    gives its approach and jaw-line axes in world beside it: the pose's rotation times the surface's local axes."""
    from scipy.spatial.transform import Rotation

    from ora.agent.composition.embodiment.geometry import public_local_axes

    record = {**RECORD, "arms": {"left": {**RECORD["arms"]["left"], "predicted_pose": [0.4, 0.0, 0.2, *quaternion]}}}
    session = _session(tmp_path, record=record)
    session.tools.surface.view_config = {"controlled_frame_convention": "arx_x5_link6"}
    answer, _state = _call(session, "move_to", {"targets": {}, "dry_run": True})
    arm = answer["result"]["arms"]["left"]
    keys = list(arm)
    assert keys[keys.index("predicted_pose") + 1:keys.index("predicted_pose") + 3] == [
        "predicted_approach_world", "predicted_jaw_line_world"]
    local, rotation = public_local_axes("arx_x5_link6"), Rotation.from_quat(quaternion)
    for name, axis in (("predicted_approach_world", "approach"), ("predicted_jaw_line_world", "closing")):
        assert len(arm[name]) == 3 and all(value == round(value, 5) for value in arm[name])
        assert max(abs(a - b) for a, b in zip(arm[name], rotation.apply(local[axis]))) <= tolerance, name
    assert arm["predicted_pose"] == record["arms"]["left"]["predicted_pose"], "the pose itself as it was"


def test_predicted_axes_need_a_controlled_frame_not_the_extended_library(tmp_path) -> None:
    """Every dry run's predicted pose carries its axes where the surface publishes its controlled frame (user decision
    2026-11-06: on every backend, in either library and in tool-call mode); none without that convention."""
    base = _session(tmp_path, library="base")
    base.tools.surface.view_config = {"controlled_frame_convention": "arx_x5_link6"}
    arm = _call(base, "move_to", {"targets": {}, "dry_run": True})[0]["result"]["arms"]["left"]
    assert "predicted_approach_world" in arm and "predicted_jaw_line_world" in arm
    arm = _call(_session(tmp_path), "move_to", {"targets": {}, "dry_run": True})[0]["result"]["arms"]["left"]
    assert "predicted_approach_world" not in arm and "predicted_jaw_line_world" not in arm
