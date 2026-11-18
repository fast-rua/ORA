"""The pose tools publish RoboDojo's speed options (``end_effector_speed_v1``) where the surface paces.

On every default surface that paces (LIBERO, RoboTwin, RoboCasa365, RoboLab)
move_pose, move_to, rotate_wrist and rotate_pitch publish ``speed``,
``speed_m_s`` and ``slow_approaching``, and the receipt projection shows how the
move was paced only where the capability is declared. What a call does with
them is the toolkit's (``tests/ora/agent/toolkit/tools/move_pose/test_speed.py``).
"""

from __future__ import annotations


import pytest

from tests.support.agent.surface_catalog import surface_of
from ora.agent.session_core.receipt_projection import (
    ReceiptProjectionContext, resolve_receipt_effect_projector,
)
from ora.agent.session_core import pacing_detail as motion_speed

LIBERO = "libero_placement_direct_overhead_v1"
ROBOTWIN = "robotwin_toolchain_b2_closed_noservo_v2"
ROBOCASA = "robocasa_mobile_direct_v1"
ROBOLAB = "robolab_direct_v1"
SPEED_FIELDS = {"speed", "speed_m_s", "slow_approaching"}
POSE_TOOLS = ("move_pose", "move_to", "rotate_wrist", "rotate_pitch")
#: Each default that publishes the speed, in its own tool files.
PUBLISHED = (LIBERO, ROBOTWIN, ROBOCASA, ROBOLAB)


@pytest.mark.parametrize("profile", PUBLISHED)
def test_a_pacing_default_publishes_the_speed_options_on_every_pose_tool(profile):
    surface = surface_of(profile)
    assert surface.supports(motion_speed.CAPABILITY)
    for item in surface.definitions:
        if item["name"] in POSE_TOOLS:
            properties = item["inputSchema"]["properties"]
            assert SPEED_FIELDS <= set(properties), item["name"]
            assert properties["speed"]["enum"] == ["fast", "normal", "careful", None]
            assert not SPEED_FIELDS & set(item["inputSchema"].get("required", ()))
        elif "properties" in item["inputSchema"]:
            assert not SPEED_FIELDS & set(item["inputSchema"]["properties"]), item["name"]


def _sparse_record(**extra):
    return {"schema": "ora.libero-sparse-motion-receipt.v1", "action_id": "action-1", "action": "move_pose",
            "execution_status": "reached", "executor": ["libero_curobo_joint_position"],
            "motion_strategy": "global_then_local_settle", "failure_stage": None, "failure_code": None,
            "target_arms": ["right"], "synchronization": {"mode": "single_arm"}, "arms": {},
            "terminal_after_action": False, **extra}


def _project(record, *, declared):
    context = ReceiptProjectionContext(last_action_id="action-1", last_action_public_name="move_pose",
                                       surface_supports=lambda name: declared and name == motion_speed.CAPABILITY)
    return resolve_receipt_effect_projector(record["schema"])(record, context)


def test_the_effect_shows_the_pacing_where_the_speed_is_published():
    motion = {"speed": {"speed": "normal", "slow_approaching": False, "linear_m_s": 0.3,
                        "planned_peak_linear_m_s": 0.3}, "wrist": {"travel_deg": 12.0}, "planned_ticks": 20,
              "duration_s": 1.0}
    effect = _project(_sparse_record(motion=motion), declared=True)
    assert effect["detail"]["motion"] == motion
    assert "motion" not in _project(_sparse_record(motion=motion), declared=False)["detail"]


def test_a_pacing_refusal_keeps_its_code_and_numbers():
    refused = _sparse_record(execution_status="plan_rejected", failure_stage="pacing",
                             failure_code="paced_path_wrist_unwind",
                             motion={"speed": {"speed": "fast"}, "refused": {
                                 "code": "paced_path_wrist_unwind", "message": "long text",
                                 "path_turn_deg": 215.1, "requested_turn_deg": 146.5}})
    effect = _project(refused, declared=True)
    assert effect["failure_code"] == "paced_path_wrist_unwind" and effect["failure_stage"] == "pacing"
    assert effect["detail"]["motion"]["refused"] == {"path_turn_deg": 215.1, "requested_turn_deg": 146.5}
    recover = {"schema": "ora.libero-pro-hybrid-motion-receipt.v2", "action_id": "action-1",
               "action": "recover_motion", "motion_status": "refused",
               "feedback": {"planning_failure_reason": "paced_motion_budget"},
               "pacing": {"refused": {"code": "paced_motion_budget", "tick_budget": 2000, "needed_ticks": 2400}}}
    effect = _project(recover, declared=True)
    assert (effect["status"], effect["failure_stage"], effect["failure_code"]) == (
        "plan_rejected", "pacing", "paced_motion_budget")
    assert effect["detail"]["motion"]["refused"] == {"tick_budget": 2000, "needed_ticks": 2400}
