"""A refusal's words (``_refusal_words.py``) and the infeasible dry-run record that carries them (``_dry_run.py``)."""
from __future__ import annotations

import math

from ora.agent.toolkit.tools import _dry_run
from ora.agent.toolkit.tools._refusal_words import NO_CAUSE, admission_words, first_recovery, refusal_words


def test_a_branch_jump_names_the_joint_and_both_angles_in_degrees():
    words = admission_words({"path_kind": "line", "path_fraction": 0.2, "cause": "joint_branch_jump", "joint": 4,
                             "jump_rad": 0.41, "bound_rad": 0.25, "endpoint": "unreachable"})
    assert words == [f"The straight line's IK changes branch 20% of the way: joint 4 jumps {math.degrees(0.41):.1f} deg, "
                     f"more than {math.degrees(0.25):.1f} deg.", "The target itself is not reachable from here either."]


def test_a_fixed_hint_is_no_cause_and_an_unknown_shape_has_no_words():
    assert first_recovery({"recovery": [next(iter(NO_CAUSE))]}) is None
    assert first_recovery({"recovery": ["Its arm would be 2 cm inside what the depth shows."]})
    assert admission_words(None) == [] and admission_words({"path_kind": "joint"}) == []
    assert refusal_words({}, None) == ""


def test_an_environment_s_refusal_without_a_message_is_answered_with_the_planner_s_and_the_detail_s_words():
    said = "Its arm would be 2.3 cm inside what the depth shows; an approach tilted 10 degrees plans."
    plan = {"status": "refused", "failure_code": "target_in_collision_at_base_pose", "failure_stage": "plan_validation",
            "message": None, "constraints": {
                "path_kind": "joint", "planning_diagnostics": {"recovery": [said]},
                "base_suggestions": [{"x_m": 1.0, "y_m": 2.0, "yaw_deg": 90.0, "torso_height_m": 0.1,
                                      "distance_m": 0.3, "turn_deg": 10.0, "arm_path": "not_planned"}]}}
    record = _dry_run.from_plan("move_pose", plan, None)
    assert record["feasible"] is False and record["refusal"]["message"].startswith(said)
    assert "Standing at x 1.000, y 2.000, yaw 90.0 deg (0.300 m and 10.0 deg from here), torso 0.100 m" in (
        record["refusal"]["message"])
    assert record["summary"] == f"refused: target_in_collision_at_base_pose; nothing moves. {record['refusal']['message']}"
    own = _dry_run.from_plan("move_pose", {**plan, "message": "the backend's own words"}, None)
    assert own["summary"].endswith("nothing moves. the backend's own words")


def test_a_refusal_s_summary_is_never_shortened_though_its_legend_is():
    long = "x" * (_dry_run.HEADLINE_REASON_CHARS + 50)
    record = {"failure_code": "path_blocked", "refusal": {"message": long}}
    assert _dry_run._headline(record).endswith("...") and _dry_run._headline(record, None).endswith(long)


def test_a_stand_whose_move_is_admitted_says_so_and_names_no_torso():
    words = admission_words({"path_kind": "line", "base_suggestions": [
        {"x_m": 2.008, "y_m": -0.866, "yaw_deg": 84.1, "distance_m": 0.011, "turn_deg": -6.4, "arm_path": "admitted"},
        {"x_m": 2.365, "y_m": -0.809, "yaw_deg": 90.5, "distance_m": 0.38, "turn_deg": 0.0, "arm_path": "admitted"}]})
    assert words == ["Standing at x 2.008, y -0.866, yaw 84.1 deg (0.011 m and -6.4 deg from here), this move is "
                     "admitted as this call checks it, clear of the depth seen now.", "1 more in base_suggestions."]


def test_a_base_move_that_carries_the_arm_into_the_depth_says_retract_it_first():
    detail = {"path_kind": "base", "robot_part": "arm", "path_fraction": 0.375, "obstacle_height_m": 1.309,
              "obstacle_xy_m": [0.68, -1.543]}
    assert admission_words(detail) == ["The arm, carried at the arm's joints, meets what the depth shows 38% of the "
                                       "way, 1.309 m above the floor at (0.680, -1.543): retract the arm before the "
                                       "base moves."]
    assert admission_words({**detail, "robot_part": "held_object"})[0].startswith(
        "The held object, carried at the arm's joints, meets")
    assert admission_words({**detail, "robot_part": "base"})[0].startswith("The base's footprint meets")
    none = admission_words({"path_kind": "joint", "base_suggestions": [],
                            "base_search": {"asked": 20, "ik_reached": 9, "arm_blocked": 4, "max_travel_m": 1.5}})
    assert none == ["No base pose within 1.5 m reaches it clear of the depth seen now (20 asked of the IK, 9 reached); "
                    "the arm, carried at its joints, would meet the depth on the way to 4 of them: retract it first."]
