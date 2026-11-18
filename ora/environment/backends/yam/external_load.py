"""What a YAM arm is pushing against: measured joint torque minus what the arm itself needs.

i2rt feeds each joint its gravity torque from its own model (KDL inverse dynamics times
``gravity_comp_factor``) plus a Coulomb friction term every control cycle, and adds the position
loop's torque on top. A free arm moving slowly therefore measures about that model torque plus a
small tracking term; an arm pressing on the table, an object or the other arm measures more. The
residual, measured minus model, is the load from outside. A residual above a joint's limit
(``yam_station.EXTERNAL_LOAD_STOP_NM``) for ``EXTERNAL_LOAD_STOP_TICKS`` ticks stops the motion
where the arm is (``external_load_stop``), well before the motor-limit guard (80% of the motor's
range, about 22 N·m on a DM4340), which a collision never reached.

This is the robot's own torque sensing, not a contact truth: it says which joint is loaded and by
how much, not what touches what. The limits are a first, conservative setting to be tuned on the
cell; each motion's peak residuals are kept for that (``rig-loads/``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ora.environment.controllers import yam_station as Y


def model_torque(robot: Any, q: Sequence[float], velocity: Sequence[float] | None) -> np.ndarray | None:
    """The torque i2rt's own model gives each arm joint (gravity and Coulomb friction); None without its model."""

    kdl = getattr(robot, "kdl", None)
    if kdl is None:
        return None
    q = np.asarray(q, dtype=float)
    joints = len(q)
    gravity = np.asarray(kdl.compute_inverse_dynamics(q, np.zeros(joints), np.zeros(joints)), float)[:joints]
    factor = np.asarray(getattr(robot, "gravity_comp_factor", np.ones(joints)), float)[:joints]
    torque = gravity * factor
    friction = getattr(robot, "_coulomb_friction", None)
    if getattr(robot, "use_coulomb_friction", False) and isinstance(friction, np.ndarray) and velocity is not None:
        torque = torque + friction[:joints] * np.sign(np.asarray(velocity, float))
    return torque


class LoadMonitor:
    """Per arm: how long each joint's residual has stayed over its limit, and the peaks since the last take."""

    def __init__(self, arms: Sequence[str], limits_nm: Sequence[float] = Y.EXTERNAL_LOAD_STOP_NM,
                 ticks: int = Y.EXTERNAL_LOAD_STOP_TICKS) -> None:
        self.limits = np.asarray(limits_nm, dtype=float)
        self.ticks = int(ticks)
        self._over = {arm: np.zeros(len(self.limits), int) for arm in arms}
        self._peaks = {arm: np.zeros(len(self.limits)) for arm in arms}

    def update(self, arm: str, residual: Sequence[float]) -> dict[str, Any] | None:
        """One tick's residual; the stop event once a joint has stayed over its limit for the ticks."""

        load = np.abs(np.asarray(residual, dtype=float))
        self._peaks[arm] = np.maximum(self._peaks[arm], load)
        over = load > self.limits
        self._over[arm] = np.where(over, self._over[arm] + 1, 0)
        stuck = np.flatnonzero(self._over[arm] >= self.ticks)
        if not len(stuck):
            return None
        joint = int(stuck[np.argmax(load[stuck] / self.limits[stuck])])
        return {"code": "external_load_stop", "arm": arm, "joint": f"joint{joint + 1}",
                "residual_torque_nm": round(float(residual[joint]), 2), "limit_nm": float(self.limits[joint]),
                "detail": f"the {arm} arm's joint {joint + 1} met {abs(float(residual[joint])):.1f} N·m from outside "
                          f"for {self.ticks} ticks (limit {self.limits[joint]:.1f}): the motion stopped where it was"}

    def reset(self, arm: str) -> None:
        self._over[arm][:] = 0

    def take_peaks(self) -> dict[str, list[float]]:
        """Each arm's peak residual per joint since the last take, and start again."""

        peaks = {arm: [round(float(value), 3) for value in values] for arm, values in self._peaks.items()}
        for values in self._peaks.values():
            values[:] = 0.0
        return peaks


def residual(robot: Any, observation: Mapping[str, Any]) -> np.ndarray | None:
    """Measured minus model torque for one arm's observation; None when the robot has no model."""

    expected = model_torque(robot, observation["joint_pos"], observation.get("joint_vel"))
    if expected is None:
        return None
    return np.asarray(observation["joint_eff"], dtype=float)[:len(expected)] - expected
