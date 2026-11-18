"""RoboLab keeps a requested end-effector speed at its Cartesian servo (``end_effector_speed_v1``).

The servo follows no planned joint path, so there is nothing to time or wrist-check before the move: a
request lowers the carrot step per tick (``motion_pace.step_caps``), without ramps, and the default caps
nothing. The fake bridge's TCP reaches each carrot in one native action.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from ora.agent.session_core.receipt_projection import ReceiptProjectionContext, resolve_receipt_effect_projector
from ora.environment.backends.robolab import motion
from ora.environment.controllers import robolab_droid as D

TICK = D.CONTROL_PERIOD_S
IDENTITY = [0.0, 0.0, 0.0, 1.0]


class Bridge:
    def __init__(self, pose: list[float]) -> None:
        self.pose = list(pose)
        self.targets: list[list[float]] = []

    def tcp_pose(self) -> list[float]:
        return list(self.pose)

    def step(self, target: list[float], _finger: float) -> dict[str, Any]:
        self.targets.append(list(target))
        self.pose = list(target)
        return {}

    def gripper_contact_force(self) -> float:
        return 0.0


class Session:
    finger_action = 0.0
    native_actions = 0
    physics_steps_per_tick = 8
    round_index = 0
    is_terminal = False
    home_pose = [0.0, 0.0, 0.5, *IDENTITY]
    last_safe_pose = None

    def __init__(self, bridge: Bridge) -> None:
        self.bridge = bridge

    def arm_state(self) -> dict[str, Any]:
        return {"right": {"pose_xyz_xyzw": self.bridge.tcp_pose()}}

    def after_native_action(self, _result: Any) -> None:
        self.native_actions += 1

    def must_stop(self) -> bool:
        return False

    def workspace_refusal(self, _pose: Any) -> None:
        return None


def _move(end_x: float, *, speed: Any = None, interaction: str = "precision") -> tuple[dict[str, Any], Bridge]:
    bridge = Bridge([0.0, 0.0, 0.5, *IDENTITY])
    arguments: dict[str, Any] = {"targets": {"right": {"position_m": [end_x, 0.0, 0.5], "orientation_xyzw": IDENTITY}},
                                 "interactions": {"right": interaction}}
    if speed is not None:
        arguments["speed"] = speed
    receipt = motion.execute_motion_v3(Session(bridge), {
        "action_id": "action-1", "action": "move_pose", "arguments": arguments})
    return receipt, bridge


def _speeds(bridge: Bridge) -> np.ndarray:
    points = np.asarray([[0.0, 0.0, 0.5], *[target[:3] for target in bridge.targets]])
    return np.linalg.norm(np.diff(points, axis=0), axis=1) / TICK


def test_the_default_keeps_the_interactions_step():
    receipt, bridge = _move(0.2)
    assert receipt["execution_status"] == "reached"
    assert _speeds(bridge).max() == pytest.approx(D.STEP_TRANSLATION_M["precision"] / TICK)
    assert receipt["motion"]["speed"]["speed"] == "fast" and receipt["motion"]["speed"]["linear_m_s"] is None


@pytest.mark.parametrize("speed,cap", [
    ({"speed": "careful", "slow_approaching": False}, 0.1),
    ({"speed": "custom", "speed_m_s": 0.05, "slow_approaching": False}, 0.05),
])
def test_a_requested_speed_lowers_the_step_and_the_budget_grows_with_it(speed, cap):
    # 0.9 m at 0.05 m/s needs 270 ticks, more than the servo's own 150.
    receipt, bridge = _move(0.2 if cap == 0.1 else 0.9, speed=speed)
    assert receipt["execution_status"] == "reached"
    assert _speeds(bridge).max() <= cap * 1.001
    assert receipt["motion"]["speed"]["linear_m_s"] == cap


def test_the_interactions_bound_stays_the_ceiling():
    receipt, bridge = _move(0.04, speed={"speed": "normal", "slow_approaching": False}, interaction="contact_approach")
    assert receipt["execution_status"] == "reached"
    assert _speeds(bridge).max() <= D.STEP_TRANSLATION_M["contact_approach"] / TICK + 1e-9


def test_slow_approaching_holds_the_last_two_centimetres():
    receipt, bridge = _move(0.2, speed={"speed": "normal", "slow_approaching": True})
    speeds = _speeds(bridge)
    x = np.asarray([target[0] for target in bridge.targets])
    assert speeds[x >= 0.2 - 0.02 + 1e-6].max() <= 0.1 * 1.001 < speeds.max()
    assert receipt["motion"]["speed"]["slow"]["linear_m_s"] == 0.1


def test_the_slow_zone_counts_the_path_left_not_the_next_waypoint():
    caps = motion.step_caps({"speed": "fast", "slow_approaching": True}, tick_s=TICK)
    start = [0.0, 0.0, 0.5, *IDENTITY]
    near_a_corner = [[0.01, 0.0, 0.5, *IDENTITY], [0.3, 0.0, 0.5, *IDENTITY]]
    assert motion.step_limits("free_space", caps, start, near_a_corner)[0] == D.STEP_TRANSLATION_M["free_space"]
    assert motion.step_limits("free_space", caps, start, near_a_corner[:1])[0] == pytest.approx(0.1 * TICK)


def test_no_receipt_claims_a_joint_cap_or_a_wrist_check():
    report = motion.motion_report({"speed": "normal", "slow_approaching": True})
    assert set(report) == {"speed"}
    assert "joint_speed_deg_s" not in report["speed"] and "joint_speed_deg_s" not in report["speed"]["slow"]


def test_recover_pose_runs_at_the_default_speed():
    bridge = Bridge([0.02, 0.0, 0.5, *IDENTITY])
    receipt = motion.execute_motion_v3(Session(bridge), {
        "action_id": "action-1", "action": "recover_motion", "arguments": {"targets": {"right": "home"}}})
    assert receipt["execution_status"] == "reached" and receipt["motion"]["speed"]["speed"] == "fast"


def test_the_policy_sees_the_request_where_the_speed_is_published():
    receipt, _bridge = _move(0.2, speed={"speed": "careful", "slow_approaching": False})
    context = ReceiptProjectionContext(last_action_id="action-1", last_action_public_name="move_pose",
                                       surface_supports=lambda name: name == "end_effector_speed_v1")
    effect = resolve_receipt_effect_projector(receipt["schema"])(json.loads(json.dumps(receipt)), context)
    assert effect["detail"]["motion"]["speed"]["linear_m_s"] == 0.1


def test_the_observation_embeds_the_receipt_without_implementation_evidence(tmp_path):
    """RoboLab's receipts hold no timing (no path is timed); its observation still embeds the policy's copy."""

    from ora.environment.backends.robolab.session import RobolabSession

    receipt, bridge = _move(0.2, speed={"speed": "careful", "slow_approaching": False})
    sealed = {**receipt, "motion": {**receipt["motion"], "timing": {"implementation": "motion_pace"}},
              "planner": {"schema": "an-implementation-record", "status": "planned"}}
    session = RobolabSession(config={"task": "BananaInBowlTask", "seed": 0}, output_root=tmp_path / "native")
    bridge.clock = lambda: 0
    bridge.cameras = lambda: {"agentview": {"rgb": np.zeros((4, 4, 3)), "depth": np.ones((4, 4)),
                                            "source_camera": "agentview", "intrinsic_cv": np.eye(3),
                                            "cam2world_cv": np.eye(4)}}
    bridge.jaw_separation_m, bridge.finger_qpos = (lambda: 0.08), (lambda: 0.0)
    session.bridge = bridge
    session.last_receipt = {"path": "receipts/action-0000.json", "sha256": "0" * 64, "receipt": sealed}
    observation, _images = session.observe(tmp_path / "policy")
    copy_ = observation["previous_execution_receipt"]["receipt"]
    assert copy_["motion"] == receipt["motion"] and copy_["planner"] == {"status": "planned"}
    files = "".join(path.read_text(errors="replace") for path in (tmp_path / "policy").rglob("*.json"))
    assert files and "motion_pace" not in files and '"timing":' not in files


def test_robolab_names_no_robot_to_the_timing():
    """No joint path is timed here, so no jerk limit applies: the servo's caps are per tick (step_caps)."""

    import inspect

    assert "pace_path" not in inspect.getsource(motion)
