"""A YAM gripper gives torque only while it is told to work; otherwise it is relaxed.

A gripper held "where it is" by position control still pushes whenever its target is not
quite reachable: against a stop, against an object, or because its [closed, open] limits
are off. Held like that for minutes, its DM4310 overheats and the motor's protection stops
the whole arm's control loop. So every program that takes the arms over relaxes the
grippers at once, and a gripper works only for an explicit gripper command.

``relax`` zeroes the gripper motor's position and damping gains, its gravity term and its
friction feed-forward in i2rt's control loop: the motor stays enabled but gives no torque,
the jaws stay where they are and move by hand. ``engage`` restores what ``relax`` took.

A gripper command runs in force control (``take_force_control``, ``apply_force``): the motor's own
position gain is off, its damping stays, and every tick the host sends a torque capped so the
jaws can never push harder than ``GRIPPER_FORCE_LIMIT_N``, from the first contact on:

    torque = clip(GRIPPER_SOFT_KP · error, ±(cap − friction)) + sign(error) · friction  (friction past the deadband)

with ``cap`` the limit's torque. It replaces position control behind i2rt's limiter, which engaged
only after it had seen the jaws blocked: a position loop of 20 N·m/rad on a 0.4 rad error pushes
about 8 N·m, some 37 times the cap, until then. The control law is the acef teleoperation follower's
(``trash_can/yam_bimanual_lcm_follower.py``), with the friction term inside the cap rather than added
after it. ``limit_force`` still lowers i2rt's limiter from 50 N for any position-controlled grip.

The jaws' command moves by ``ramp`` (``GRIPPER_CLOSE_SPEED_M_S``, ``GRIPPER_OPEN_SPEED_M_S``, never more than
``GRIPPER_COMMAND_LEAD_M`` ahead of the measured jaw) and stops where ``GRIPPER_STALL`` sees the jaw arrive or
stall, in an episode's motion (``motion.MotionRun.gripper``) and in the operator's ``grip``
(``cell_adapter.YamHeld.grip``) alike.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from ora.environment.controllers import yam_station as Y
from ora.environment.controllers.gripper_stall import GripperStallRule

GRIPPER_STALL = GripperStallRule(
    still_step=Y.GRIPPER_STALL_STILL_STEP, window=Y.GRIPPER_STALL_WINDOW,
    minimum_samples=Y.GRIPPER_STALL_WINDOW + 1, arrival_tolerance=Y.GRIPPER_ARRIVAL_TOLERANCE,
    tick_period_s=Y.CONTROL_PERIOD_S, tick_unit="yam_control_tick",
)

_SAVED = "_ora_gripper_gains"
_FORCE = "_ora_gripper_force_control"


def has_gripper(robot: Any) -> bool:
    return getattr(robot, "_gripper_index", None) is not None


def is_relaxed(robot: Any) -> bool:
    return getattr(robot, _SAVED, None) is not None


def in_force_control(robot: Any) -> bool:
    return bool(getattr(robot, _FORCE, False))


def relax(robot: Any) -> None:
    if not has_gripper(robot):
        return
    if is_relaxed(robot):
        if in_force_control(robot):          # leave force control: no damping, no torque either
            index = robot._gripper_index
            kd = np.array(robot._kd, float)
            kd[index] = 0.0
            robot.update_kp_kd(np.array(robot._kp, float), kd)
            _set_gripper_torque(robot, 0.0)
            _current_command_gains(robot)
            setattr(robot, _FORCE, False)
        return
    index = robot._gripper_index
    kp, kd = np.array(robot._kp, float), np.array(robot._kd, float)
    saved: dict[str, Any] = {"kp": float(kp[index]), "kd": float(kd[index])}
    kp[index] = kd[index] = 0.0
    robot.update_kp_kd(kp, kd)
    factor = getattr(robot, "gravity_comp_factor", None)
    if factor is not None:
        saved["gravity"] = float(factor[index])
        factor[index] = 0.0
    friction = getattr(robot, "_coulomb_friction", None)
    if isinstance(friction, np.ndarray) and friction.ndim == 1:
        saved["friction"] = float(friction[index])
        friction[index] = 0.0
    _current_command_gains(robot)
    setattr(robot, _SAVED, saved)


def take_force_control(robot: Any) -> None:
    """The gripper for a command: position gain off, its own damping on, no gravity or friction feed-forward;
    ``apply_force`` sets its torque every tick."""

    if not has_gripper(robot) or in_force_control(robot):
        return
    relax(robot)                             # saves the gains to restore and zeroes the feed-forward
    index = robot._gripper_index
    kd = np.array(robot._kd, float)
    kd[index] = getattr(robot, _SAVED)["kd"]
    robot.update_kp_kd(np.array(robot._kp, float), kd)
    _current_command_gains(robot)
    setattr(robot, _FORCE, True)


def ramp(commanded_m: float, measured_m: float, target_m: float, *, closing: bool) -> float:
    """The next tick's jaw command toward ``target_m``: one tick at the closing (or opening) speed from the last
    command, and never more than ``GRIPPER_COMMAND_LEAD_M`` from the measured jaw."""

    step = (Y.GRIPPER_CLOSE_SPEED_M_S if closing else Y.GRIPPER_OPEN_SPEED_M_S) * Y.CONTROL_PERIOD_S
    stepped = float(commanded_m) + float(np.clip(float(target_m) - float(commanded_m), -step, step))
    return float(np.clip(stepped, float(measured_m) - Y.GRIPPER_COMMAND_LEAD_M,
                         float(measured_m) + Y.GRIPPER_COMMAND_LEAD_M))


def force_torque(target_rad: float, measured_rad: float, *, force_n: float = Y.GRIPPER_FORCE_LIMIT_N) -> float:
    """The motor torque toward ``target_rad`` from ``measured_rad`` (motor angles), never above ``force_n``'s torque."""

    cap = max(0.0, float(force_n)) * Y.GRIPPER_FORCE_TO_TORQUE
    friction = min(Y.GRIPPER_FRICTION_NM, cap)
    error = float(target_rad) - float(measured_rad)
    push = float(np.clip(Y.GRIPPER_SOFT_KP_NM_PER_RAD * error, -(cap - friction), cap - friction))
    return push + (float(np.sign(error)) * friction if abs(error) > Y.GRIPPER_DEADBAND_RAD else 0.0)


def apply_force(robot: Any, target_opening: float, measured_opening: float,
                force_n: float = Y.GRIPPER_FORCE_LIMIT_N) -> float | None:
    """Send the force-control torque for this tick (after the tick's position command); the torque, or None when the
    gripper is not in force control or its [closed, open] motor angles are unknown."""

    limits = getattr(robot, "_gripper_limits", None)
    if not in_force_control(robot) or limits is None or len(limits) != 2:
        return None
    closed, opened = float(limits[0]), float(limits[1])
    motor = lambda opening: closed + float(np.clip(opening, 0.0, 1.0)) * (opened - closed)
    torque = force_torque(motor(target_opening), motor(measured_opening), force_n=force_n)
    _set_gripper_torque(robot, torque)
    return torque


def _set_gripper_torque(robot: Any, torque: float) -> None:
    commands = getattr(robot, "_commands", None)
    lock = getattr(robot, "_command_lock", None)
    if commands is None or lock is None:
        return
    with lock:
        torques = getattr(commands, "torques", None)
        torques = np.zeros(len(robot._kp)) if torques is None else np.array(torques, float)
        torques[robot._gripper_index] = float(torque)
        commands.torques = torques


def engage(robot: Any) -> None:
    saved = getattr(robot, _SAVED, None)
    if not has_gripper(robot) or saved is None:
        return
    setattr(robot, _FORCE, False)
    index = robot._gripper_index
    kp, kd = np.array(robot._kp, float), np.array(robot._kd, float)
    kp[index], kd[index] = saved["kp"], saved["kd"]
    robot.update_kp_kd(kp, kd)
    if "gravity" in saved:
        robot.gravity_comp_factor[index] = saved["gravity"]
    if "friction" in saved:
        robot._coulomb_friction[index] = saved["friction"]
    _current_command_gains(robot)
    setattr(robot, _SAVED, None)


def limit_force(robot: Any, newtons: float) -> float | None:
    """Set the grip force i2rt's limiter holds once the jaws are blocked; the force it held before, or None.

    ``get_yam_robot`` builds the limiter with 50 N and offers no argument for it; the limiter keeps the force
    inside the map it was built with, so the map is rebuilt for the new force.
    """

    limiter = getattr(robot, "_gripper_force_limiter", None)
    if not has_gripper(robot) or limiter is None or not newtons > 0:
        return None
    before = float(limiter.max_force)
    mapping = limiter.gripper_force_torque_map
    limiter.max_force = float(newtons)
    limiter.gripper_force_torque_map = partial(mapping.func, *mapping.args,
                                               **{**mapping.keywords, "gripper_force": float(newtons)})
    robot._limit_gripper_force = float(newtons)
    return before


def _current_command_gains(robot: Any) -> None:
    """The command the control loop is sending already: it takes the new gains now, not at
    the next command."""

    commands = getattr(robot, "_commands", None)
    lock = getattr(robot, "_command_lock", None)
    if commands is None or lock is None:
        return
    with lock:
        commands.kp = robot._kp
        commands.kd = robot._kd
