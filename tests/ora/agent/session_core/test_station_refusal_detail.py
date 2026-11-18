"""A YAM station refusal reaches the policy with its check, its plan and the operator's words."""

from __future__ import annotations

from ora.agent.session_core.receipt_projection import ReceiptProjectionContext, resolve_receipt_effect_projector
from ora.agent.session_core.refusal_detail import STATION_RECEIPT_SCHEMA, public_refusal_detail

CONTEXT = ReceiptProjectionContext(last_action_id="a1", last_action_public_name="move_pose")


def _receipt(**changes):
    return {"schema": STATION_RECEIPT_SCHEMA, "action_id": "a1", "action": "move_pose",
            "execution_status": "plan_rejected", "executor": None, "motion_strategy": "none",
            "target_arms": ["left"], "synchronization": {"mode": "single_arm"}, "arms": {},
            "terminal_after_action": False, **changes}


def test_a_planner_refusal_names_its_check_beside_the_folded_code():
    record = _receipt(failure_stage="planner", failure_code="target_ik_unreachable",
                      refusal={"arm": "left", "closest_position_error_m": 0.0072390393})
    effect = resolve_receipt_effect_projector(STATION_RECEIPT_SCHEMA)(record, CONTEXT)
    assert effect["failure_code"] == "planner_failed"
    assert effect["detail"]["admission_refusal"] == {"check": "target_ik_unreachable", "arm": "left",
                                                     "closest_position_error_m": 0.0072}


def test_a_motion_answered_without_moving_carries_its_plan_and_the_operators_words():
    plan = {"schema": "ora.yam-motion-plan.v1", "arms": ["right"], "ticks": 120, "duration_s": 2.4,
            "waypoints": {"right": [[0.4, -0.1, 0.06123456, 1.0, 0.0, 0.0, 0.0]]}, "min_inter_arm_distance_m": 0.31}
    declined = public_refusal_detail(_receipt(failure_stage="operator", failure_code="operator_declined",
                                              plan=plan, operator_says="  too close to the cup "), "other.schema")
    assert declined["operator"] == {"decision": "declined", "reason": "too close to the cup"}
    assert "schema" not in declined["plan"] and declined["plan"]["waypoints"]["right"][0][2] == 0.0612
    read_only = public_refusal_detail(_receipt(failure_stage="operator", failure_code="dry_run_read_only",
                                               plan=plan), "other.schema")
    assert set(read_only) == {"plan"} and read_only["plan"]["ticks"] == 120
    # A motion the station executed carries neither.
    assert public_refusal_detail(_receipt(execution_status="reached", failure_code=None, plan=plan), "x") == {}


def test_a_refusal_says_how_high_the_arm_reaches_and_which_parts_meet():
    unreachable = public_refusal_detail(_receipt(failure_stage="planner", failure_code="target_ik_unreachable",
                                                 refusal={"arm": "left", "closest_position_error_m": 0.0323,
                                                          "highest_reachable_z_m": 0.0801}), "x")
    assert unreachable["admission_refusal"]["highest_reachable_z_m"] == 0.08
    nowhere = public_refusal_detail(_receipt(failure_stage="planner", failure_code="target_ik_unreachable",
                                             refusal={"arm": "left", "highest_reachable_z_m": None}), "x")
    assert nowhere["admission_refusal"]["highest_reachable_z_m"] is None
    nearest = public_refusal_detail(_receipt(failure_stage="planner", failure_code="target_ik_unreachable",
                                             refusal={"arm": "right", "highest_reachable_z_m": None,
                                                      "nearest_reachable_position_m": [0.4512, -0.0598, 0.03],
                                                      "nearest_reachable_offset_m": 0.0601}), "x")
    assert nearest["admission_refusal"]["nearest_reachable_position_m"] == [0.451, -0.06, 0.03]
    assert nearest["admission_refusal"]["nearest_reachable_offset_m"] == 0.06
    none = public_refusal_detail(_receipt(failure_stage="planner", failure_code="target_ik_unreachable",
                                          refusal={"arm": "right", "nearest_reachable_position_m": None,
                                                   "nearest_reachable_offset_m": None}), "x")
    assert none["admission_refusal"]["nearest_reachable_position_m"] is None
    odd = public_refusal_detail(_receipt(failure_stage="planner", failure_code="target_ik_unreachable",
                                         refusal={"arm": "right", "nearest_reachable_position_m": [0.4, "x", 0.03]}), "x")
    assert "nearest_reachable_position_m" not in odd["admission_refusal"]
    near = public_refusal_detail(_receipt(failure_stage="plan_validation", failure_code="inter_arm_clearance",
                                          refusal={"tick": 32, "distance_m": 0.0586, "moving_arms": ["left"],
                                                   "closest_parts": {"left": "camera", "right": "tip_right"}}), "x")
    assert near["admission_refusal"] == {"check": "inter_arm_clearance", "tick": 32, "distance_m": 0.0586,
                                         "moving_arms": ["left"],
                                         "closest_parts": {"left": "camera", "right": "tip_right"}}
    # Anything but two plain part names stays behind.
    odd = public_refusal_detail(_receipt(failure_stage="plan_validation", failure_code="inter_arm_clearance",
                                         refusal={"tick": 1, "closest_parts": {"left": "Red Cube!", "right": "x"},
                                                  "moving_arms": ["torso"]}), "x")
    assert "closest_parts" not in odd["admission_refusal"] and "moving_arms" not in odd["admission_refusal"]
