"""Host-safe contracts of the RoboLab executor (no simulator import).

The RoboCasa365 executor's twin contracts are in
``tests/ora/environment/backends/robocasa/test_robocasa_host_safe.py``.
"""

from __future__ import annotations

import math

import pytest

from ora.agent.evidence.policy_state import _FORBIDDEN_POLICY_KEY_PATTERNS
from ora.agent.session_core.receipt_projection import public_motion_failure_code
from ora.environment.benchmarks import robolab as robolab_benchmark
from ora.environment.backends.robolab import motion as robolab_motion
from ora.environment.controllers import robolab_droid as D


@pytest.mark.parametrize("stop_reason", ["episode_ended", "no_progress", "budget_exhausted"])
def test_arm_stops_use_public_codes_the_abort_gate_does_not_count(stop_reason):
    status, stage, code = robolab_motion._arm_status(stop_reason)
    assert (status, stage) == ("not_reached", "execution")
    assert public_motion_failure_code("execution", code) != "execution_fault", code


def test_arrival_and_contact_stops_are_reached():
    assert robolab_motion._arm_status("reached")[0] == "reached"
    assert robolab_motion._arm_status("contact_detected")[0] == "reached"


def test_jaw_outcomes_map_to_the_shared_codes():
    motion = robolab_motion
    assert motion.gripper_failure_code("reached_target", terminal=False) is None
    assert motion.gripper_failure_code("stalled_before_target", terminal=False) == "gripper_stalled_before_target"
    assert motion.gripper_failure_code("moving_at_budget", terminal=False) == "gripper_not_reached"
    assert motion.gripper_failure_code("interrupted", terminal=True) == "episode_terminal_during_execution"


def test_robolab_opening_is_normalized_on_the_finger_joint():
    assert robolab_motion.normalized_opening(0.0) == 1.0
    assert robolab_motion.normalized_opening(D.GRIPPER_CLOSED_QPOS_RAD) == 0.0
    assert robolab_motion.normalized_opening(-0.1) == 1.0


def test_free_space_lifts_only_beyond_its_radius():
    radius, lift = D.FREE_SPACE_DIRECT_RADIUS_M, D.FREE_SPACE_LIFT_M
    start = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    near = [radius * 0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]
    far = [radius * 2.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]
    assert robolab_motion.free_space_waypoints(start, near) == [near]
    lifted = robolab_motion.free_space_waypoints(start, far)
    assert len(lifted) == 3 and lifted[-1] == far
    assert math.isclose(lifted[0][2], 0.2 + lift) and math.isclose(lifted[1][2], 0.2 + lift)


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


class _FakeSession:
    native_actions = 0
    physics_steps_per_tick = 8
    round_index = 0
    is_terminal = False

    def arm_state(self):
        return {"right": {"pose_xyz_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}}


def test_receipt_shapes_carry_no_policy_forbidden_key():
    """The receipt reaches the policy as previous_execution_receipt; contact forces never may."""

    session = _FakeSession()
    session.bridge = None
    run = robolab_motion.MotionRun(session, {"action_id": "a", "action": "move_pose", "arguments": {}})
    receipt = run.receipt(status="reached", strategy="cartesian_servo_direct", executor=["x"],
                          target_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], interaction="contact_approach",
                          extra={"endpoint": {"stop_reason": "contact_detected", "waypoints_total": 1,
                                              "waypoints_reached": 1}})
    base_result = {"stop_reason": "base_contact", "travel_m": 0.1, "moved": {"x_m": 0.1, "y_m": 0.0, "yaw_rad": 0.0},
                   "base_error": {"position_m": 0.2, "yaw_rad": 0.0}}
    for key in [*_keys(receipt), *_keys(base_result)]:
        if key == "contact_approach":
            continue
        assert not any(pattern.search(key) for pattern in _FORBIDDEN_POLICY_KEY_PATTERNS), key


def test_only_the_admitted_roster_validates():
    task = sorted(robolab_benchmark.TASKS)[0]
    assert robolab_benchmark.validate_runtime_extension({"task": task, "seed": 0})["task"] == task
    with pytest.raises(ValueError):
        robolab_benchmark.validate_runtime_extension({"task": "NotATask", "seed": 0})
    with pytest.raises(ValueError):
        robolab_benchmark.validate_runtime_extension({"task": task, "seed": -1})
    assert robolab_benchmark.public_context()["environment_id"] == "robolab"


def test_the_policy_reads_how_an_arm_move_ended():
    """``endpoint.stop_reason`` crosses through the shared projection: a contact stop reads as one, not as an arrival."""

    import json

    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from ora.agent.session_core.receipt_projection import ReceiptProjectionContext, resolve_receipt_effect_projector

    session = _FakeSession()
    session.bridge = None
    run = robolab_motion.MotionRun(session, {"action_id": "a", "action": "move_pose", "arguments": {}})
    receipt = run.receipt(status="reached", strategy="cartesian_servo_direct", executor=["x"],
                          target_pose=[0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], interaction="contact_approach",
                          extra={"endpoint": {"stop_reason": "contact_detected", "waypoints_total": 1,
                                              "waypoints_reached": 0}})
    context = ReceiptProjectionContext(last_action_id="a", last_action_public_name="move_pose",
                                       surface_supports=resolve_tool_surface("robolab_direct_v1").supports)
    effect = resolve_receipt_effect_projector(receipt["schema"])(json.loads(json.dumps(receipt)), context)
    assert effect["detail"]["endpoint"] == {"stop_reason": "contact_detected"}
    assert effect["action_summary"].startswith("right arm 30 mm / 0.0° from target, reached, "
                                               "ended by the gripper contact stop")
