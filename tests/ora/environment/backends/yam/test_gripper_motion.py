"""A YAM gripper is ramped, never stepped, and holds an object where it stopped (motion.MotionRun.gripper).

On 2026-11-14 a close commanded as one step crossed the whole aperture in 0.26 s and snapped a fingertip.
"""

from __future__ import annotations

import types

import numpy as np

from ora.environment.backends.yam import motion
from ora.environment.controllers import yam_station as Y


class _Rig:
    """Fingers that follow their command up to a fast servo's speed and stop on an object ``width`` wide."""

    def __init__(self, finger_m: float, width: float | None = None) -> None:
        self.finger = {arm: finger_m for arm in Y.ARMS}
        self.width = width
        self.commands: list[float] = []

    def finger_m(self, arm: str) -> float:
        return self.finger[arm]

    def arm_q(self, arm: str) -> np.ndarray:
        return np.zeros(6)

    def step(self, arms, fingers):
        command = float(fingers["right"])
        self.commands.append(command)
        current = self.finger["right"]
        moved = current + float(np.clip(command - current, -0.01, 0.01))     # 0.5 m/s: the servo is not the limit
        if self.width is not None:
            moved = max(moved, self.width / 2)
        self.finger["right"] = moved
        return types.SimpleNamespace(events=[])


def _run(rig: _Rig) -> motion.MotionRun:
    session = types.SimpleNamespace(
        rig=rig, control_ticks=0, commanded_q={arm: np.zeros(6) for arm in Y.ARMS}, finger_targets={},
        gripper_goals={}, on_control_tick=lambda: None, on_guard_events=lambda events: None, must_stop=lambda: False,
        opening=lambda arm: min(1.0, max(0.0, rig.finger[arm] / Y.FINGER_OPEN_M)),
        aperture_m=lambda arm: 2.0 * rig.finger[arm], arm_state=lambda: {})
    return motion.MotionRun(session, {"action_id": "a", "action": "set_gripper", "arguments": {}})


def test_a_close_is_ramped_at_the_closing_speed_and_reaches_closed():
    rig = _Rig(Y.FINGER_OPEN_M)
    outcome = _run(rig).gripper({"right": "close"})["right"]
    steps = np.abs(np.diff([Y.FINGER_OPEN_M, *rig.commands]))
    assert steps.max() <= Y.GRIPPER_CLOSE_SPEED_M_S * Y.CONTROL_PERIOD_S + 1e-12
    assert outcome["state"] == "reached_target"
    seconds = len(rig.commands) * Y.CONTROL_PERIOD_S
    assert seconds >= Y.FINGER_OPEN_M / Y.GRIPPER_CLOSE_SPEED_M_S * 0.9, "the whole travel took its time"


def test_jaws_that_stall_on_an_object_are_held_there_not_driven_on_to_closed():
    rig = _Rig(Y.FINGER_OPEN_M, width=0.03)
    run = _run(rig)
    outcome = run.gripper({"right": "close"})["right"]
    assert outcome["state"] == "stalled_before_target" and abs(outcome["measured_aperture_mm"] - 30.0) < 0.5
    held = run.session.finger_targets["right"]
    assert abs(held - (0.015 - Y.GRIPPER_HOLD_SQUEEZE_M)) < 1e-9, "held just past where it stopped"
    assert rig.commands[-1] == held
    # The command never ran more than the lead ahead of the stopped jaw.
    assert min(rig.commands) >= 0.015 - Y.GRIPPER_COMMAND_LEAD_M - 1e-9
    assert run.session.gripper_goals["right"] == "close"


def test_an_open_is_ramped_at_the_opening_speed():
    rig = _Rig(0.0)
    outcome = _run(rig).gripper({"right": "open"})["right"]
    steps = np.abs(np.diff([0.0, *rig.commands]))
    assert steps.max() <= Y.GRIPPER_OPEN_SPEED_M_S * Y.CONTROL_PERIOD_S + 1e-12
    assert outcome["state"] == "reached_target"


def test_the_tick_budget_covers_a_whole_close():
    assert Y.GRIPPER_TICKS * Y.CONTROL_PERIOD_S > Y.FINGER_OPEN_M / Y.GRIPPER_CLOSE_SPEED_M_S + Y.GRIPPER_STALL_WINDOW \
        * Y.CONTROL_PERIOD_S
