"""What a refused or failed script call says in the policy session (``ora/agent/pi/code_mode.py``, ``library: extended``).

Read off RoboCasa365's code-mode runs (audit, 2026-11-06): a ``targets`` wrapper on a single-arm surface was refused as missing
mode fields, a ``mark_point`` whose marks all failed raised ``mark_point: refused`` with no code, and an unknown name's
``allowed`` listed tools a script cannot call and never the ``ora_lib`` function meant. Each now says what to do, with
the extended library only: a ``base`` contract's answers are those it always had.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ora.agent.code_mode import robot
from ora.agent.code_mode.contract import resolve_contract, with_reads
from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.error_classes import ToolError
from ora.agent.pi.code_mode import PiCodeMode
from ora.agent.session_core.tool_arguments import _tool_error
from tests.support.agent.surface_catalog import catalog_grants

#: A ``mark_point`` whose eight marks all failed, each with its own code (``status: failed``, no top-level error).
MESSAGES = {"inconsistent_mark": "independent RGB-D measurements disagree by 0.19 m, more than the 0.025 m maximum",
            "depth_discontinuity": "the depth window straddles an edge"}
MARKS = {"status": "failed", "success": True, "solved_count": 0, "failed_count": 8, "mark_count": 8, "marks": [
    {"point_id": f"p{index}", "point_uv": [100 + index, 120], "status": "failed", "error_code": code,
     "error_message": MESSAGES[code], "xyz_m": None}
    for index, code in enumerate(["inconsistent_mark"] * 6 + ["depth_discontinuity"] * 2)]}
#: The model's own tools a script cannot call, with the one ``WITHHELD``.
OWN = ("check_task", "finish_episode", "report_issue", "artifact_read", "compute")


class Gateway:
    def classify_effect(self, name: str, arguments: Any) -> str:
        return "mutation" if name.startswith("move") else "read_only"

    def arguments(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in arguments.items() if value is not None}

    def definitions(self) -> list[dict[str, Any]]:
        return [{"function": {"name": name}} for name in OWN if name != "compute"]


class Session(PiCodeMode):
    """Script calls on a surface's definitions, answered from ``answers`` (a free-play refusal for a name of no tool)."""

    def __init__(self, root: Path, profile: str, *, library: str | None = "extended",
                 answers: dict[str, dict[str, Any]] | None = None) -> None:
        contract = with_reads({**resolve_contract(True), **({"library": library} if library else {})},
                              toolkit=False, moves=True)
        definitions = list(resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile)).definitions)
        self.code_mode, self.run_root, self.gateway, self.answers = contract, root, Gateway(), answers or {}
        self._script_names = {name: name for name in [item["name"] for item in definitions] + list(OWN)}
        self.tools = SimpleNamespace(surface=SimpleNamespace(capabilities=(), lifecycle_tools=("finish_episode",)),
                                     script_definitions=lambda: definitions)
        self.artifacts, self.seen = SimpleNamespace(put_json=lambda value, kind: {"kind": kind}), []

    def _current_observation_id(self) -> str | None:
        return None

    def _script_call(self, call, *, ids, origin, preset, result_fields=None):
        self.seen.append({"name": call.name, "arguments": dict(call.arguments), "origin": dict(origin),
                          "preset": preset})
        if preset is not None and preset[1] is not None:
            result = _tool_error(preset[1])
        elif call.name not in self._script_names.values():  # free play's refusal (robot_socket._script_call)
            result = _tool_error(ToolError(f"unknown tool: {call.name}", code="unknown_tool", retryable=False,
                                           constraints={"allowed": sorted(set(self._script_names.values()))}))
        else:
            result = self.answers.get(call.name, {"status": "submitted"})
        ok = "error" not in result and result.get("status") not in {"error", "failed", "invalid", "rejected"}
        outcome = {"ok": ok, "call_id": ids[1], "tool": call.name, "result": result, "observation_step": 0,
                   "images": []}
        if result_fields is not None:
            result_fields(outcome)
        return outcome

    def _project(self, name, result, *, call_id, step):
        return {"visible": result}


def _state() -> dict[str, Any]:
    return {"seq": 0, "actions": 0, "calls": 0, "cap": 4, "log": [], "frames": [], "uncounted_s": 0.0}


def _call(session: Session, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _state()
    return session._code_mode_call(name, arguments, model_call="model-1", parent="call-1", state=state), state


# --- a targets wrapper where the tool has none --------------------------------------------------------------------

WRAPPED = {"position_mode": "absolute", "position_m": [1.2, 0.4, 0.9], "orientation_mode": "hold_current"}


@pytest.mark.parametrize("dry", [{}, {"dry_run": True}], ids=["call", "dry_run"])
def test_a_targets_wrapper_on_a_single_arm_surface_is_refused_naming_its_form(tmp_path, dry) -> None:
    """The schema's own refusal said the mode fields were missing; 6 of the 7 were dry runs."""
    session = Session(tmp_path, "robocasa_mobile_direct_v1")
    answer, state = _call(session, "move_pose", {"targets": {"right": WRAPPED}, **dry})
    error = answer["error"]
    assert answer["ok"] is False and error["code"] == "invalid_arguments" and error["field"] == "targets"
    assert error["message"] == ("this surface's move_pose takes one arm's target fields directly, with no targets "
                                "wrapper: tools.move_pose(**target(p_world))")
    assert error["received"] == ["right"] and state["log"][0]["error"] == "invalid_arguments"
    (seen,) = session.seen
    assert seen["preset"][1].code == "invalid_arguments" and seen["origin"].get("dry_run") is (True if dry else None)
    assert answer.get("dry_run") is (True if dry else None)
    assert "this surface's move_pose takes one arm's target fields" in str(robot.ToolRefused("move_pose", answer))
    moved = _call(session, "move_to", {"targets": {"right": {"position_m": [1, 0, 1]}}})[0]["error"]["message"]
    assert moved.endswith("takes its fields directly, with no targets wrapper: tools.move_to(position_m=p_world)")
    turned = _call(session, "rotate_wrist", {"targets": {"right": 30.0}})[0]["error"]["message"]
    assert turned.endswith("no targets wrapper: tools.rotate_wrist(angle_deg, speed?, speed_m_s?, slow_approaching?, "
                           "dry_run?)")


def test_the_flat_form_and_a_surface_that_takes_targets_are_left_to_the_tool(tmp_path) -> None:
    flat = Session(tmp_path, "robocasa_mobile_direct_v1")
    assert _call(flat, "move_pose", WRAPPED)[0]["ok"] is True and flat.seen[0]["preset"] is None
    dual = Session(tmp_path, "robodojo_placement_toolkit_v6")
    assert _call(dual, "move_pose", {"targets": {"left": WRAPPED}})[0]["ok"] is True and dual.seen[0]["preset"] is None
    base = Session(tmp_path, "robocasa_mobile_direct_v1", library="base")
    _call(base, "move_pose", {"targets": {"right": WRAPPED}})
    assert base.seen[0]["preset"] is None, "a base contract's call goes to the tool as it always did"


# --- a call whose items failed, each with its own code ----------------------------------------------------------

def test_a_mark_point_whose_marks_all_failed_says_each_code(tmp_path) -> None:
    """11 scripts ended on ``mark_point: refused``: the codes and reasons were in the marks, not in an error."""
    session = Session(tmp_path, "robocasa_mobile_direct_v1", answers={"mark_point": MARKS})
    answer, state = _call(session, "mark_point", {"view": "agentview", "point_ids": ["a"], "points_uv": [[1, 2]]})
    error = answer["error"]
    assert answer["ok"] is False and error["code"] == "inconsistent_mark"
    assert error["message"] == (f"8 of 8 marks failed: inconsistent_mark x6 ({MESSAGES['inconsistent_mark']}), "
                                f"depth_discontinuity x2 ({MESSAGES['depth_discontinuity']})")
    assert error["constraints"] == {"failed": {"inconsistent_mark": 6, "depth_discontinuity": 2}, "of": 8}
    refused = str(robot.ToolRefused("mark_point", answer))
    (entry,) = state["log"]
    for code in MESSAGES:
        assert code in refused and code in entry["summary"]
    assert entry["error"] == "inconsistent_mark" and answer["result"] == MARKS, "the marks as the tool answered them"
    assert robot.typed("mark_point", answer).failure_code == "inconsistent_mark"


def test_items_are_summarised_only_for_a_failed_answer_without_an_error_with_the_extended_library(tmp_path) -> None:
    partial = {**MARKS, "status": "partial"}
    assert "error" not in _call(Session(tmp_path, "robocasa_mobile_direct_v1", answers={"mark_point": partial}),
                                "mark_point", {})[0]
    base = Session(tmp_path, "robocasa_mobile_direct_v1", library="base", answers={"mark_point": MARKS})
    answer, state = _call(base, "mark_point", {})
    assert "error" not in answer and "summary" not in state["log"][0]
    assert str(robot.ToolRefused("mark_point", answer)) == "mark_point: refused", "base reads as it always did"


# --- a name of no tool -------------------------------------------------------------------------------------------

def test_an_unknown_name_lists_what_a_script_can_call_and_the_import_of_a_library_name(tmp_path) -> None:
    session = Session(tmp_path, "robocasa_mobile_direct_v1")
    error = _call(session, "object_geometry", {"target": "t"})[0]["error"]
    assert error["code"] == "unknown_tool" and error["field"] == "name" and error["received"] == "object_geometry"
    assert error["message"] == ("unknown tool: object_geometry; object_geometry is ora_lib's: from ora_lib import "
                                "object_geometry")
    allowed = error["constraints"]["allowed"]
    assert {"move_pose", "move_base", "mark_point", "vision_segment"} <= set(allowed)
    assert not set(OWN) & set(allowed), "the model's own tools and compute are no script function"
    target = _call(session, "target", {})[0]["error"]["message"]
    assert target.endswith("; target is robot's: from robot import target")
    near = _call(session, "pose_move", {})[0]["error"]
    assert near["message"].startswith("unknown tool: pose_move; the nearest: move_pose")
    assert near["constraints"]["nearest"][0] == "move_pose"
    assert session.seen[0]["preset"][1].code == "unknown_tool", "refused before the tool, and recorded"
    base = Session(tmp_path, "robocasa_mobile_direct_v1", library="base")
    plain = _call(base, "object_geometry", {})[0]["error"]
    assert plain["message"] == "unknown tool: object_geometry" and "check_task" in plain["constraints"]["allowed"], (
        "base keeps free play's refusal")
