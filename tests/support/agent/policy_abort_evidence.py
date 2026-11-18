"""Test helpers for policy abort evidence, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/finish/test_policy_abort_evidence.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations


def _tool_error(code, tool="grasp_plan"):
    return {"type": "tool_error", "tool": tool, "arguments": {}, "time": 1.0,
            "error": {"code": code, "message": "x", "retryable": True}}


def _receipt(**fields):
    return {"type": "motion_receipt", "time": 1.0, "consumed_decision": True,
            "record": {"path": "receipts/action-0001.json", "sha256": "0" * 64,
                       "receipt": {"action": "move_pose", "action_id": "action-1", **fields}}}


def _skill(stop_reason, stages):
    return {"type": "motion_receipt", "time": 1.0, "consumed_decision": True,
            "record": {"skill": "grasp_execute", "action_id": "action-2", "status": "aborted",
                       "stop_reason": stop_reason, "stage_receipts": stages}}


def _stage(stage, status, reason, receipt=None):
    return {"stage": stage, "status": status, "reason": reason, "action_id": f"action-2:{stage}",
            "receipt": None if receipt is None else {"path": "receipts/a.json", "sha256": "0" * 64,
                                                     "receipt": receipt}}


#: Every distinct failure shape the seven sealed g012 runs recorded.
G012_RECORD = (
    _tool_error("stale_observation", "grasp_plan"), _tool_error("stale_observation", "align_plan"),
    _tool_error("invalid_arguments", "compute"), _tool_error("invalid_arguments", "move_pose"),
    _tool_error("stale_candidate_observation", "grasp_execute"),
    _receipt(failure_stage="planner", failure_code="ik_failed:move", execution_status="planner_failed"),
    _receipt(failure_stage="planner", failure_code="ik_failed:local", execution_status="planner_failed"),
    _receipt(failure_stage="planner", failure_code="ik_failed:precontact", execution_status="planner_failed"),
    _receipt(failure_stage="plan_validation", failure_code="ik_branch_jump:move", execution_status="plan_rejected"),
    _receipt(failure_stage="execution", failure_code="controller_stalled", execution_status="not_reached"),
    _receipt(action="set_gripper", failure_stage="execution", failure_code="gripper_stalled_before_target",
             execution_status="not_reached"),
    _receipt(failure_stage="execution", failure_code="native_action_budget_exhausted",
             execution_status="not_reached"),
    _receipt(action="wait", failure_stage="execution", failure_code="episode_terminal_during_execution",
             execution_status="not_reached"),
    _receipt(failure_stage=None, failure_code=None, execution_status="reached"),
    _skill("pregrasp_not_reached", [
        _stage("open", "executed", "executed", {"execution_status": "reached"}),
        _stage("pregrasp", "not_reached", "controller_stalled",
               {"failure_stage": "execution", "failure_code": "controller_stalled",
                "execution_status": "not_reached"})]),
    _skill("sequence_completed", [
        _stage("close", "executed", "gripper_stalled_before_target",
               {"failure_stage": "execution", "failure_code": "gripper_stalled_before_target",
                "execution_status": "not_reached"}),
        _stage("verify", "executed", "verified_against_observation")]),
)
#: Receipts the Harness itself marks as an infrastructure fault.
INFRASTRUCTURE_RECEIPTS = {
    "execution_fault": _receipt(execution_status="execution_fault"),
    "execution_fault_record": _receipt(execution_status="not_reached", execution_fault={"code": "x"}),
    "execution_uncertain": _receipt(execution_status="not_reached", execution_uncertain=True),
    "planner_error": _receipt(failure_stage="planner", failure_code="planner_internal_error",
                              execution_status="planner_failed"),
    "invalid_measured_state": _receipt(failure_stage="execution", failure_code="invalid_measured_state",
                                       execution_status="not_reached"),
    "skill_fault": _skill("trusted_execution_fault", [_stage("approach", "unavailable", "trusted_execution_fault")]),
    "skill_stage_fault": _skill("approach_not_reached", [
        _stage("approach", "not_reached", "execution_fault", {"execution_status": "execution_fault"})]),
}
_INTER_ARM_DIAGNOSTICS = {"outcome": "inter_arm_collision_path", "phase": "joint_space",
                          "rejection_counts": {"inter_arm_collision_path": 1}}
#: The g013 kernel unit's other-arm refusals, in the shapes
#: sparse_motion, inter_arm and recovery write them: ordinary planner and
#: recovery refusals of a target or route, not an infrastructure failure.
INTER_ARM_REFUSALS = {
    "single_arm_move": _receipt(failure_stage="planner", failure_code="inter_arm_collision:move",
                                execution_status="planner_failed", planning_diagnostics=_INTER_ARM_DIAGNOSTICS),
    "lockstep_pair_local": _receipt(failure_stage="planner", failure_code="inter_arm_collision:local",
                                    execution_status="planner_failed", planning_diagnostics=_INTER_ARM_DIAGNOSTICS),
    "after_motion_started": _receipt(failure_stage="planner", failure_code="inter_arm_collision:lift",
                                     execution_status="not_reached", planning_diagnostics=_INTER_ARM_DIAGNOSTICS),
    "recovery_before_motion": _receipt(action="recover_pose", failure_stage="plan_validation",
                                       failure_code="recovery_inter_arm_collision_path",
                                       execution_status="plan_rejected"),
    "recovery_mid_route": _receipt(action="recover_pose", failure_stage="execution",
                                   failure_code="recovery_inter_arm_collision_path", execution_status="not_reached"),
    "grasp_execute_stage": _skill("pregrasp_not_reached", [
        _stage("open", "executed", "executed", {"execution_status": "reached"}),
        _stage("pregrasp", "planner_failed", "inter_arm_collision:move",
               {"failure_stage": "planner", "failure_code": "inter_arm_collision:move",
                "execution_status": "planner_failed", "planning_diagnostics": _INTER_ARM_DIAGNOSTICS})]),
}
