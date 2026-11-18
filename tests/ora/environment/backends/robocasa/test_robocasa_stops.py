"""How a RoboCasa365 arm or base motion ends, and how the policy reads the stop (RoboCasa365 run audit, 2026-11-06).

No simulator: a fake arm moves each joint toward its target at its own rate, never past its own stop, and its grip
site is ``0.4 *`` the first three joints in metres plus the last three as a rotation vector (as in
``test_robocasa_motion_pace.py``); a fake base moves along its command at a fixed rate. The paced path is handed in
directly, so no timing worker runs.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.evidence.policy_state import _FORBIDDEN_POLICY_KEY_PATTERNS
from ora.agent.session_core.receipt_projection import ReceiptProjectionContext, resolve_receipt_effect_projector
from ora.environment.backends.robocasa import motion
from ora.environment.controllers import robocasa_omron as C

LEVER_M = 0.4
#: A 25-knot path that turns joint 1 by 0.5 rad, one knot per tick.
KNOTS = np.linspace(np.zeros(7), [0.5, 0, 0, 0, 0, 0, 0], 26)[1:].tolist()


def pose_of(joints: Any) -> list[float]:
    joints = np.asarray(joints, dtype=float)
    return [*(LEVER_M * joints[:3]), *Rotation.from_rotvec(joints[4:7]).as_quat()]


class Arm:
    """OmronIO stand-in: each joint moves toward its target by at most its ``rate`` per tick, never past ``stop``."""

    model = data = None
    gripper_directive = C.GRIPPER_OPEN_COMMAND

    def __init__(self, *, rate: Any = 1.0, stop: dict[int, float] | None = None,
                 contact_at: float | None = None) -> None:
        self.q = np.zeros(7)
        self.rate = np.broadcast_to(np.asarray(rate, dtype=float), (7,))
        self.stop = stop or {}
        self.contact_at = contact_at
        self.control_steps = 0
        self.holds = 0

    def arm_qpos(self) -> list[float]:
        return self.q.tolist()

    def grip_pose(self) -> list[float]:
        return pose_of(self.q)

    def base_pose(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def contact_forces(self) -> dict[str, float]:
        touching = self.contact_at is not None and self.q[0] >= self.contact_at
        return {"gripper": 2 * C.CONTACT_FORCE_STOP_N if touching else 0.0, "base": 0.0}

    def step(self, arm_joints: Any = None, **_command: Any) -> dict[str, Any]:
        self.control_steps += 1
        if arm_joints is not None:
            self.q = self.q + np.clip(np.asarray(arm_joints, dtype=float) - self.q, -self.rate, self.rate)
            for joint, limit in self.stop.items():
                self.q[joint] = min(self.q[joint], limit)
        return {}

    def hold_measured_arm(self) -> None:
        self.holds += 1


class Session:
    round_index = 0
    is_terminal = False
    physics_steps_per_tick = 25

    def __init__(self, io: Any) -> None:
        self.io = io

    def arm_state(self) -> dict[str, Any]:
        return {"right": {"pose_xyz_xyzw": self.io.grip_pose()}}

    def must_stop(self) -> bool:
        return False

    def after_control_step(self, _result: Any) -> None:
        pass

    def base_path_refusal(self, _target: Any) -> None:
        """move_base's footprint check before anything moves (PR A): the path here is clear."""
        return None


class Paced:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def rows(self) -> list[list[float]]:
        return [list(row) for row in self._rows]

    def report(self) -> dict[str, Any]:
        return {"planned_ticks": len(self._rows)}


@pytest.fixture(autouse=True)
def _paced_as_given(monkeypatch):
    monkeypatch.setattr(motion, "_paced_plan", lambda _session, plan, *_args: Paced(plan["path"]))


def _move(arm: Arm, knots: list[list[float]] = KNOTS, interaction: str = "precision") -> dict[str, Any]:
    """Run one admitted pose move along ``knots`` through the executor's own path and endpoint check."""

    session = Session(arm)
    pose = pose_of(knots[-1])
    action = {"action_id": "action-1", "action": "move_pose", "arguments": {
        "targets": {"right": {"position_m": pose[:3], "orientation_xyzw": pose[3:]}},
        "interactions": {"right": interaction}}}
    plan = {"refusal": None, "code": None, "path": knots, "torso_height_m": None, "kind": "joint"}
    return motion._run_plan(motion.MotionRun(session, action), session, plan, pose, interaction, "joint_path")


# ------------------------------------------------------------------ RO-21: the settle after the last knot

def test_joints_at_rest_short_of_the_last_knot_end_the_settle_at_once_as_blocked():
    arm = Arm(stop={0: 0.48})  # 0.02 rad short: the grip site is 8 mm off
    receipt = _move(arm)
    endpoint = receipt["endpoint"]
    assert endpoint["stop_reason"] == "settle_stalled"
    assert (receipt["execution_status"], receipt["failure_code"]) == ("not_reached", "blocked")
    # At rest from the tick the last knot went out: three still ticks, not the 20-tick wait.
    assert endpoint["ticks"] == len(KNOTS) + C.ARM_STILL_STEPS - 1
    assert endpoint["ticks"] < len(KNOTS) + C.ARM_SETTLE_MAX_STEPS - 1
    assert arm.holds == 1 and receipt["native_actions"] == endpoint["ticks"] + 1
    assert endpoint["lagging_joint"] == 1 and endpoint["lag_rad"] == pytest.approx(0.02)
    assert "path_fraction" not in endpoint


def test_joints_still_creeping_are_not_cut_and_settle():
    knots = np.linspace(np.zeros(7), [0.05, 0, 0, 0, 0, 0, 0], 11)[1:].tolist()
    arm = Arm(rate=[0.002, 1, 1, 1, 1, 1, 1])  # 2 mrad per tick, twice the still bound
    receipt = _move(arm, knots)
    endpoint = receipt["endpoint"]
    assert endpoint["stop_reason"] == "settled" and receipt["execution_status"] == "reached"
    assert endpoint["ticks"] > len(knots) + C.ARM_STILL_STEPS
    assert "lagging_joint" not in endpoint and arm.holds == 0


def test_joints_at_rest_with_the_grip_site_on_target_are_reached():
    arm = Arm(stop={0: 0.49})  # 0.01 rad short: the grip site is 4 mm off, inside the 5 mm tolerance
    receipt = _move(arm)
    assert receipt["endpoint"]["stop_reason"] == "settle_stalled"
    assert (receipt["execution_status"], receipt["failure_code"]) == ("reached", None)
    # It still moved 0.01 rad on the tick the last knot went out, then rested three ticks.
    assert receipt["endpoint"]["ticks"] == len(KNOTS) + C.ARM_STILL_STEPS


def test_a_joint_trailing_its_moving_target_is_named_with_how_far_along_the_path_it_stopped():
    knots = np.linspace(np.zeros(7), [0, 0.5, 0, 0, 0, 0, 0], 26)[1:].tolist()
    receipt = _move(Arm(stop={1: 0.0}), knots)
    endpoint = receipt["endpoint"]
    assert (endpoint["stop_reason"], receipt["failure_code"]) == ("joint_lag", "blocked")
    # The lag passes 0.2 rad at knot 11 and holds for three ticks.
    assert endpoint["ticks"] == 13 and endpoint["path_fraction"] == round(13 / 25, 3)
    assert endpoint["lagging_joint"] == 2 and endpoint["lag_rad"] == pytest.approx(0.26)


# ------------------------------------------------------------------ RO-22: a stalled base

class Base:
    """OmronIO stand-in for the base: it moves along its commanded heading at most ``rate`` metres per tick."""

    def __init__(self, rate: float, *, contact_after: int | None = None) -> None:
        self.pose = [0.0, 0.0, 0.0]
        self.rate = rate
        self.contact_after = contact_after
        self.control_steps = 0

    def base_pose(self) -> list[float]:
        return list(self.pose)

    def base_anchor(self) -> list[float]:
        return self.pose[:2]

    def anchor_offset(self) -> list[float]:
        return [0.0, 0.0]

    def slide_axes(self) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, 1.0]]

    def contact_forces(self) -> dict[str, float]:
        pressed = self.contact_after is not None and self.control_steps >= self.contact_after
        return {"gripper": 0.0, "base": 2 * C.BASE_CONTACT_STOP_N if pressed else 0.0}

    def grip_pose(self) -> list[float]:
        return [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def step(self, *, base_velocity_world: Any = None, **_command: Any) -> dict[str, Any]:
        self.control_steps += 1
        vx, vy, _yaw_rate = base_velocity_world or (0.0, 0.0, 0.0)
        speed = math.hypot(vx, vy)
        if speed > 1e-9:
            travel = min(self.rate, speed * C.CONTROL_PERIOD_S)
            self.pose = [self.pose[0] + vx / speed * travel, self.pose[1] + vy / speed * travel, self.pose[2]]
        return {}


def _drive(base: Base, x_m: float = 0.2) -> dict[str, Any]:
    session = Session(base)
    action = {"action_id": "action-1", "action": "move_base",
              "arguments": {"target": {"x_m": x_m, "y_m": 0.0, "yaw_rad": 0.0}}}
    return motion.execute_motion_v3(session, action)


def test_a_pinned_base_is_judged_stalled_within_fifteen_ticks():
    receipt = _drive(Base(rate=0.0))
    assert receipt["base"]["stop_reason"] == "no_progress" and receipt["failure_code"] == "controller_stalled"
    # One window, then the three braking ticks.
    assert receipt["native_actions"] == C.BASE_STALL_WINDOW_STEPS + 3 <= 15


def test_a_slowly_converging_base_is_not_cut():
    base = Base(rate=0.001)  # 1 mm per tick: 10 mm per window, twice the stall bound
    receipt = _drive(base)
    assert receipt["base"]["stop_reason"] == "reached" and receipt["execution_status"] == "reached"
    assert receipt["native_actions"] > 150


# ------------------------------------------------------------------ RO-24: what the policy reads

SURFACE = resolve_tool_surface("robocasa_mobile_direct_v1")


def _projected(receipt: dict[str, Any], public_name: str = "move_pose") -> dict[str, Any]:
    context = ReceiptProjectionContext(last_action_id=receipt["action_id"], last_action_public_name=public_name,
                                       surface_supports=SURFACE.supports)
    return resolve_receipt_effect_projector(receipt["schema"])(json.loads(json.dumps(receipt)), context)


def _keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [key for item_key, item in value.items() for key in (item_key, *_keys(item))]
    if isinstance(value, list):
        return [key for item in value for key in _keys(item)]
    return []


def _private(effect: dict[str, Any]) -> list[str]:
    return [key for key in _keys(effect) if any(pattern.search(key) for pattern in _FORBIDDEN_POLICY_KEY_PATTERNS)]


def test_a_stop_at_rest_names_its_mechanism_and_its_joint():
    effect = _projected(_move(Arm(stop={0: 0.48})))
    detail = effect["detail"]
    assert detail["endpoint"] == {"stop_reason": "settle_stalled"}
    assert detail["arms"]["right"]["stop_evidence"] == {"lagging_joint": 1, "lag_deg": pytest.approx(1.15, abs=0.01)}
    assert effect["action_summary"] == (
        "right arm 8.0 mm / 0.0° from target, not reached: blocked, ended with the joints at rest short of the "
        "path's end, TCP moved 192 mm, stopped 96% along its path moving +x, joint 1 fell 1.1° behind, "
        "probably touched something (inferred, no contact sensor)")
    assert not _private(effect)


def test_a_joint_lag_stop_names_its_mechanism():
    knots = np.linspace(np.zeros(7), [0, 0.5, 0, 0, 0, 0, 0], 26)[1:].tolist()
    effect = _projected(_move(Arm(stop={1: 0.0}), knots))
    assert effect["detail"]["endpoint"] == {"stop_reason": "joint_lag"}
    assert effect["detail"]["arms"]["right"]["stop_evidence"]["path_fraction"] == 0.52
    assert "not reached: blocked, ended by joint lag, " in effect["action_summary"]
    assert "stopped 52% along its path" in effect["action_summary"]


def test_a_contact_stop_short_of_the_target_reads_as_one_and_states_no_force():
    effect = _projected(_move(Arm(contact_at=0.2), interaction="contact_approach"))
    assert effect["status"] == "reached" and effect["detail"]["endpoint"] == {"stop_reason": "contact_detected"}
    assert effect["action_summary"] == (
        "right arm 120 mm / 0.0° from target, reached, ended by the gripper contact stop, TCP moved 80 mm")
    assert not _private(effect) and "_n" not in json.dumps(effect["detail"]["endpoint"])


def test_an_arrival_keeps_its_line():
    effect = _projected(_move(Arm()))
    assert effect["detail"]["endpoint"] == {"stop_reason": "settled"}
    assert effect["action_summary"] == "right arm 0.0 mm / 0.0° from target, reached, TCP moved 200 mm"
    # At rest with the grip site on target is an arrival too: the line names no stop.
    effect = _projected(_move(Arm(stop={0: 0.49})))
    assert effect["detail"]["endpoint"] == {"stop_reason": "settle_stalled"}
    assert effect["action_summary"] == "right arm 4.0 mm / 0.0° from target, reached, TCP moved 196 mm"


def test_base_contact_reads_as_base_contact_not_as_an_inferred_touch():
    effect = _projected(_drive(Base(rate=0.01, contact_after=5)), "move_base")
    assert effect["failure_code"] == "blocked" and effect["detail"]["base"]["stop_reason"] == "base_contact"
    assert effect["action_summary"] == "not reached: blocked, ended by base contact, executed"
    assert "sensor" not in effect["action_summary"] and not _private(effect)


def test_a_stalled_base_names_the_stall():
    effect = _projected(_drive(Base(rate=0.0)), "move_base")
    assert effect["detail"]["base"]["stop_reason"] == "no_progress"
    assert effect["action_summary"].startswith("not reached: controller stalled, executed")
