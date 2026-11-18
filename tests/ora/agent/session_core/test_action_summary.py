"""D1: one plain-language line heads the last-action result (``last_action_summary_v1``).

The line restates facts the effect already carries -- distance from target,
the outcome and its code, whether the arm moved. A move that stopped under way
also says where and which way, and any contact it infers is worded as inference.
The full detail follows it unchanged.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re

import pytest

from tests.support.agent.delivery_lessons_fixtures import (
    DEGREES, DEDUP, SUMMARY, dojo_gripper_feedback, libero_hybrid_receipt, libero_receipt, project,
    robodojo_receipt, robotwin_receipt, skill_receipt,
)
from ora.agent.session_core.action_summary import CAPABILITY, action_summary

FACTS = ("public_motion_effect_v1", "current_public_robot_facts_v1", "contact_motion_feedback_v1")
#: Moves cut short by their budget, as sealed runs received them.
CUT_MOVES = json.loads((Path(__file__).parents[3] / "data/code_mode/cut-moves.json").read_text())["receipts"]
#: Planner refusals as sealed runs received them, trimmed to the fields the line reads.
REFUSALS = json.loads((Path(__file__).parents[3] / "data/code_mode/planner-refusals.json").read_text())["receipts"]


def _stalled_gripper():
    feedback = {"left": dojo_gripper_feedback(measured=0.41, requested=0.0),
                "right": dojo_gripper_feedback()}
    return robodojo_receipt(action="set_gripper", status="not_reached", stage="execution",
                            code="gripper_stalled_before_target", pose_error=None,
                            native_actions=20, feedback=feedback, gripper_check=0.41)


CASES = [
    ("robotwin refusal", robotwin_receipt(), "move_pose",
     "left arm 104 mm / 120° from target, plan rejected: inter arm collision, not moved"),
    ("robodojo arrival", robodojo_receipt(), None,
     "left arm 8.7 mm / 1.6° from target, reached, ran 9 of 20 native actions"),
    ("robodojo planner failure",
     robodojo_receipt(status="planner_failed", stage="planner", code="planner_failed",
                      pose_error=(0.3558, 0.5), native_actions=0), None,
     "left arm 356 mm / 29° from target, planner failed, not moved"),
    ("robodojo stalled gripper", _stalled_gripper(), None,
     "left gripper opening 0.41 / target 0.00, not reached: gripper stalled before target, "
     "ran 20 of 20 native actions"),
    ("libero arrival", libero_receipt(), None,
     "right arm 3.3 mm / 0.4° from target, reached, TCP moved 50 mm"),
    ("libero hybrid gripper", libero_hybrid_receipt(), None,
     "right gripper 0.03 from target opening, reached, TCP moved 0.2 mm"),
    ("closed grasp skill", skill_receipt(), None,
     "left arm 188 mm / 150° from target, aborted at pregrasp: IK FAIL, 2 of 6 stages attempted"),
]


def test_capability_name_is_the_one_the_fixtures_declare():
    assert CAPABILITY == SUMMARY


@pytest.mark.parametrize(("label", "receipt", "public_name", "expected"), CASES,
                         ids=[case[0] for case in CASES])
def test_summary_restates_distance_outcome_and_motion(label, receipt, public_name, expected):
    effect = project(receipt, (SUMMARY, *FACTS), public_name=public_name)
    assert effect["action_summary"] == expected


@pytest.mark.parametrize("item,expected", list(zip(CUT_MOVES, (
    "left arm 116 mm from target, not reached: native action budget exhausted, TCP moved 222 mm",
    "right arm 262 mm from target, not reached: native action budget exhausted, TCP moved 270 mm",
    "right arm 2.6 mm from target, not reached: native action budget exhausted, TCP moved 332 mm",
))), ids=("move_pose budget", "20-action budget", "nearly there"))
def test_a_cut_move_states_the_distance_left_and_that_the_budget_ended_it(item, expected):
    assert "0.1 mm" in item["recorded_summary"] and "controller timeout" in item["recorded_summary"]
    assert action_summary(item["receipt"]) == expected


def _with_progress(outcome, **fields):
    receipt = robodojo_receipt(status="not_reached", stage="execution", code="native_action_budget_exhausted",
                               pose_error=(0.042, 0.2), native_actions=20)
    receipt["arms"]["left"]["motion_progress"] = {"outcome": outcome, **fields}
    return project(receipt, (SUMMARY, *FACTS))


def test_the_projected_effect_carries_what_a_cut_move_reports():
    effect = _with_progress("cut_short", remaining_to_goal_m=0.1157)
    assert effect["failure_code"] == "controller_timeout", "the public code stays"
    assert effect["action_summary"].startswith("left arm 116 mm from target, not reached: native action budget exhausted")


def test_a_budget_stop_short_of_the_last_knot_keeps_its_residual_and_names_the_budget():
    summary = _with_progress("budget")["action_summary"]
    assert summary.startswith("left arm 42 mm / 11° from target, not reached: native action budget exhausted")


def test_another_outcome_keeps_the_public_code():
    assert "controller timeout" in _with_progress("still_moving")["action_summary"]


def test_a_left_arm_move_never_reports_the_right_arm_translation():
    """The shared contact detail reads the right arm; a left move must not cite it."""
    receipt = robodojo_receipt()
    receipt["after"]["right"]["pose_xyz_xyzw"] = [0.5, -0.35, 0.92, 0.0, 0.0, 0.7071, 0.7071]
    summary = project(receipt, (SUMMARY, *FACTS))["action_summary"]
    assert "TCP moved" not in summary


def test_a_terminated_round_is_named_last():
    receipt = robodojo_receipt()
    receipt["terminal_after_action"] = True
    assert project(receipt, (SUMMARY, *FACTS))["action_summary"].endswith(", round ended")


def _numbers(value, found):
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        for rendered in (f"{value * 1000:.0f}", f"{value * 1000:.1f}", f"{value:.2f}",
                         f"{math.degrees(value):.0f}", f"{math.degrees(value):.1f}"):
            found.add(rendered)
        if isinstance(value, int):
            found.add(str(value))
    elif isinstance(value, dict):
        for child in value.values():
            _numbers(child, found)
    elif isinstance(value, list):
        # A count of listed items, and the length of a listed 3-vector, restate it.
        found.add(str(len(value)))
        if len(value) == 3 and all(isinstance(item, (int, float)) for item in value):
            _numbers(math.dist(value, (0.0, 0.0, 0.0)), found)
        for child in value:
            _numbers(child, found)
    return found


@pytest.mark.parametrize(("label", "receipt", "public_name", "expected"), CASES,
                         ids=[case[0] for case in CASES])
def test_summary_adds_no_number_and_no_advice(label, receipt, public_name, expected):
    """Every figure in the line is a rounding of a value the effect already carries."""
    legacy = project(receipt, FACTS, public_name=public_name)
    summary = action_summary(legacy)
    assert summary == expected
    derivable = _numbers(legacy, set())
    for number in re.findall(r"\d+(?:\.\d+)?", summary):
        assert number in derivable, (number, summary)
    for word in ("try", "should", "retry", "consider", "instead", "recommend", "please",
                 "must", "next", "advice"):
        assert not re.search(rf"\b{word}\b", summary, re.IGNORECASE), word


def _stopped(code, *, cause="unknown", vector=(0.0, -0.001, -0.037), evidence=None, targets=("left",)):
    state = {"pose_error": {"translation_m": 0.0077}, "realized_translation_world_m": list(vector),
             **({"stop_evidence": evidence} if evidence else {})}
    return {"status": "not_reached", "failure_code": code, "physical_cause": cause,
            "detail": {"target_arms": list(targets), "arms": {"left": state}}}


def test_a_tracking_stop_says_where_which_way_which_joint_and_what_that_suggests():
    line = action_summary(_stopped("controller_tracking_error", evidence={
        "path_fraction": 0.882, "lagging_joint": 2, "lag_rad": 0.0155}))
    assert line == ("left arm 7.7 mm from target, not reached: controller tracking error, TCP moved 37 mm, "
                    "stopped 88% along its path moving down, joint 2 fell 0.9° behind, "
                    "probably touched something (inferred, no contact sensor)")


def test_a_backend_without_knots_says_the_same_from_what_it_measured():
    line = action_summary(_stopped("blocked", vector=(0.02, 0.003, 0.0)))
    assert line.endswith("stopped 72% along its path moving +x, probably touched something "
                         "(inferred, no contact sensor)")


@pytest.mark.parametrize(("code", "cause"), [("contact_stop", "unknown"), ("tracking_diverged", "unknown"),
                                             ("controller_tracking_error", "blocked_by_contact")])
def test_a_stop_with_a_measured_cause_infers_nothing(code, cause):
    line = action_summary(_stopped(code, cause=cause))
    assert "stopped 83% along its path moving down" in line and "inferred" not in line


def test_a_stop_without_travel_names_no_direction():
    assert "moving" not in action_summary(_stopped("controller_stalled", vector=(0.0, 0.0, 0.001)))
    assert "along its path" not in action_summary(_stopped("contact_stop", vector=(0.0, 0.0, 0.0)))


def test_a_move_that_ended_another_way_adds_no_clause():
    assert "along its path" not in action_summary(_stopped("controller_timeout"))
    assert "along its path" not in action_summary(_stopped("controller_tracking_error", targets=("left", "right")))


def test_a_recovery_cut_by_its_budget_states_the_joint_travel_left_not_the_cut_point():
    effect = {"status": "not_reached", "failure_code": "controller_timeout", "detail": {
        "target_arms": ["left"], "arms": {"left": {
            "targeted": True, "pose_error": {"translation_m": 0.0, "orientation_rad": 0.0},
            "realized_translation_world_m": [0.0, 0.0, 0.322],
            "recovery_progress": {"outcome": "cut_short", "joint_distance_end_rad": 0.279547}}}}}
    assert action_summary(effect) == ("left arm 16° of joint travel left, not reached: native action budget "
                                      "exhausted, TCP moved 322 mm")


def test_summary_heads_the_effect_and_leaves_the_detail_alone():
    legacy = project(robodojo_receipt(), FACTS)
    delivered = project(robodojo_receipt(), (SUMMARY, *FACTS))
    text = json.dumps(delivered, ensure_ascii=False, sort_keys=True)
    assert text.startswith('{"action": "move_pose", "action_summary": "left arm 8.7 mm')
    assert {key: value for key, value in delivered.items() if key != "action_summary"} == legacy


def test_summary_is_born_in_degrees_whatever_else_is_declared():
    alone = project(robodojo_receipt(), (SUMMARY, *FACTS))["action_summary"]
    combined = project(robodojo_receipt(), (SUMMARY, DEGREES, DEDUP, *FACTS))["action_summary"]
    assert alone == combined


def test_no_summary_without_the_capability():
    assert "action_summary" not in project(robodojo_receipt(), FACTS)


def _as_projected(receipt):
    """A recorded receipt in the projection's units: delivery had restated its angles in degrees."""
    effect = copy.deepcopy(receipt)
    effect.pop("recorded_summary")
    for state in (effect["detail"].get("arms") or {}).values():
        error = state.get("pose_error") if isinstance(state, dict) else None
        if isinstance(error, dict) and "orientation_deg" in error:
            error["orientation_rad"] = math.radians(error.pop("orientation_deg"))
    return effect


def _first_recoveries(detail):
    arms = [state for state in (detail.get("arms") or {}).values() if state]
    sources = [detail.get("planning_diagnostics"), *(state.get("planning_diagnostics") for state in arms),
               *((state.get("planner") or {}).get("planning_diagnostics") for state in arms)]
    return {source["recovery"][0] for source in sources if source and source.get("recovery")}


@pytest.mark.parametrize("label", sorted(REFUSALS))
def test_a_planner_refusal_ends_with_the_planner_s_own_words(label):
    """LIBERO's one diagnostic and RoboTwin's per-arm ones reach the line the model reads, quoted as recorded."""
    receipt = REFUSALS[label]
    head, said = action_summary(_as_projected(receipt)).split(". ", 1)
    assert head == receipt["recorded_summary"]
    quoted = said.split(": ", 1)[1] if label.endswith("two-arm") else said
    assert quoted in _first_recoveries(receipt["detail"]) and "an approach tilted" in quoted
    assert said.startswith("right arm: ") == label.endswith("two-arm")


def test_a_robodojo_refusal_quotes_the_arm_s_diagnostic_and_nothing_else_does():
    said = "No joint configuration reaches this pose even with nothing in the way."
    state = {"targeted": True, "pose_error": {"translation_m": 0.31, "orientation_rad": 0.0},
             "planning_diagnostics": {"recovery": [said, "Try a different target position."]}}
    effect = {"status": "planner_failed", "failure_code": "planner_failed",
              "detail": {"target_arms": ["left"], "arms": {"left": state}}}
    assert action_summary(effect) == f"left arm 310 mm / 0.0° from target, planner failed, motion unknown. {said}"
    rejected = {**effect, "status": "plan_rejected", "failure_code": "paced_path_wrist_unwind"}
    assert said not in action_summary(rejected)
    empty = {**effect, "detail": {"target_arms": ["left"], "arms": {"left": {**state, "planning_diagnostics": {
        "recovery": []}}}}}
    assert action_summary(empty) == "left arm 310 mm / 0.0° from target, planner failed, motion unknown"


def _no_cause_diagnostics():
    """Each planner's diagnostic when it found no cause, from the code that writes it."""
    from ora.environment.backends.robodojo.ik import SELECTION_RECOVERY
    from ora.environment.backends.robotwin.planning import ik_planning_diagnostics
    from ora.environment.controllers.motion_planner import planning_diagnostics

    yield planning_diagnostics({"status": "refused", "reason": "goal_unreachable", "request": {}, "detail": {}})
    for ik in ({"returned_solution_count": 4, "valid_solution_count": 0},
               {"returned_solution_count": 4, "valid_solution_count": 2}, None):
        yield ik_planning_diagnostics({"planner_status": "IK_FAIL", **({"ik": ik} if ik else {})})
    yield {"recovery": [SELECTION_RECOVERY["no_valid_endpoint"]]}


def test_a_refusal_with_no_cause_adds_only_what_it_had():
    """A planner's fixed hint when it found no cause says nothing the line should repeat."""
    for diagnostic in _no_cause_diagnostics():
        effect = {"status": "planner_failed", "failure_code": "planner_failed",
                  "detail": {"target_arms": ["right"], "planning_diagnostics": diagnostic}}
        assert action_summary(effect) == "planner failed, motion unknown", diagnostic["recovery"]


def test_unknown_effect_shapes_get_no_line():
    assert action_summary({"action": "unknown", "status": "world_state_unknown"}) == (
        "world state unknown, motion unknown")
    assert action_summary({"kind": "servo"}) is None
    assert action_summary(None) is None


def _robocasa_refusal(reach_check=None, *, status="plan_rejected", torso=None, diagnostics=None):
    """A RoboCasa admission refusal, projected as the policy receives it."""
    from ora.agent.session_core.receipt_projection import (
        ROBOCASA_SPARSE_MOTION_RECEIPT_SCHEMA, ReceiptProjectionContext, resolve_receipt_effect_projector,
    )

    record = {"schema": ROBOCASA_SPARSE_MOTION_RECEIPT_SCHEMA, "action_id": "action-1", "action": "move_pose",
              "execution_status": status, "failure_stage": "plan_validation",
              "failure_code": "target_unreachable_at_base_pose", "executor": None, "motion_strategy": "none",
              "target_arms": ["right"], "synchronization": {"mode": "single_arm", "requested": False,
                                                            "executed": torso is not None},
              "arms": {"right": {"targeted": True, "interaction": "precision",
                                 "pose_error": {"translation_m": 0.64, "orientation_rad": 0.19}}},
              "terminal_after_action": False}
    if reach_check is not None:
        record["reach_check"] = reach_check
    if torso is not None:
        record["torso"] = torso
    if diagnostics is not None:
        record["planning_diagnostics"] = diagnostics
    context = ReceiptProjectionContext(last_action_id="action-1", last_action_public_name="move_pose",
                                       surface_supports=lambda _name: False)
    return resolve_receipt_effect_projector(record["schema"])(record, context)


def test_a_robocasa_line_refusal_says_where_its_ik_stopped_why_and_whether_the_target_is_reachable():
    effect = _robocasa_refusal({"path_kind": "line", "path_fraction": 0.424, "cause": "ik_unsolved",
                                "at_pose": [2.41, -1.1, 1.05, 0.0, 0.0, 0.0, 1.0], "endpoint": "reachable"},
                               status="not_reached", torso={"from_m": 0.3126, "to_m": 0.2877, "target_m": 0.28,
                                                            "stop_reason": "reached", "arm_path_replanned": False})
    summary = action_summary(effect)
    assert summary == (
        "right arm 640 mm / 11° from target, not reached: target unreachable at base pose, executed, "
        "torso moved from 0.313 to 0.288 m first. The straight line has no IK solution 42% of the way, "
        "at (2.410, -1.100, 1.050). The target itself is reachable from here: the line is what fails.")
    for word in ("try", "should", "retry", "consider", "instead", "recommend", "please", "must", "next"):
        assert not re.search(rf"\b{word}\b", summary, re.IGNORECASE), word


def _shared_planner_diagnostic(said):
    """The shared planner's diagnostic of a goal it could not reach, with its own first sentence."""
    from ora.environment.controllers.motion_planner import planning_diagnostics

    diagnostic = planning_diagnostics({"status": "refused", "reason": "goal_unreachable", "request": {}, "detail": {}})
    return {**diagnostic, "recovery": [said, *diagnostic["recovery"]]}


def test_a_robocasa_refusal_quotes_the_planner_and_names_the_part_and_where_to_stand():
    said = ("The robot reaches this pose with nothing in the way, but there its arm would be 2.3 cm inside what the "
            "depth shows; with the fingertips at the same point, an approach tilted 10 degrees plans.")
    suggestion = {"x_m": 2.305, "y_m": -1.174, "yaw_deg": 0.0, "torso_height_m": 0.1, "distance_m": 0.206,
                  "turn_deg": 0.0, "arm_path": "planned"}
    effect = _robocasa_refusal({"path_kind": "joint", "base_suggestions": [suggestion, {**suggestion, "x_m": 2.0}],
                                "base_search": {"asked": 39, "ik_reached": 18}},
                               diagnostics=_shared_planner_diagnostic(said))
    summary = action_summary(effect)
    assert summary.startswith("right arm 640 mm / 11° from target, plan rejected: target unreachable at base pose, "
                              f"not moved. {said} Standing at x 2.305, y -1.174, yaw 0.0 deg")
    assert summary.endswith("(0.206 m and 0.0 deg from here), torso 0.100 m, the arm reaches it clear of the depth "
                            "seen now; a path from the arm's current joints plans there. 1 more in base_suggestions.")
    line = action_summary(_robocasa_refusal({"path_kind": "line", "path_fraction": 0.6, "penetration_m": 0.014,
                                             "robot_part": "arm", "base_suggestions": [],
                                             "base_search": {"asked": 12, "ik_reached": 0, "max_travel_m": 1.5}}))
    assert line.endswith("The arm enters what the depth shows by 14 mm 60% of the way along the line. No base pose "
                         "within 1.5 m reaches it clear of the depth seen now (12 asked of the IK, 0 reached).")


def test_a_robocasa_out_of_reach_and_a_blocked_base_move_say_their_numbers():
    reach = action_summary(_robocasa_refusal({"path_kind": "reach", "distance_from_mount_m": 1.1891, "reach_m": 1.05,
                                              "nearest_mount_height_m": 1.04}))
    assert reach.endswith("The target is 1.189 m from the arm mount at the nearest height the lift allows, beyond its "
                          "1.05 m reach.")
    base = action_summary(_robocasa_refusal({"path_kind": "base", "robot_part": "base", "path_fraction": 0.75,
                                             "penetration_m": 0.013, "obstacle_height_m": 0.19,
                                             "obstacle_xy_m": [2.6, -1.235]}))
    assert base.endswith("The base's footprint meets what the depth shows 75% of the way, 0.190 m above the floor at "
                         "(2.600, -1.235).")
