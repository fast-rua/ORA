"""The end-effector speed a motion request asks for (``speed`` of an ``ora-motion-action.v3`` move_pose).

A policy names how fast the controlled point may move: a preset (``fast``,
the default; ``normal``; ``careful``), or ``speed_m_s``, a top linear speed of
the controlled point in m/s; and ``slow_approaching``, which also holds the
last ``SLOW_APPROACH_M`` of the path to ``careful``. Speed is limited at the
controlled point first; each joint's own speed and acceleration stay the
ceiling at every speed, so a fast request may not be reached and a slow one
practically always is.

The vocabulary and the numbers are Yulun's RoboDojo design (the toolkit's
``ora/agent/toolkit/tools/move_pose/_smooth.py`` presets, ``custom_pace`` and
``options``, and ``lib.motion_plan.timing.SLOW_APPROACH_M``), stated here once
so the agent, which validates the call, and every environment, which executes
it, read the same request (rule L1 keeps them from importing each other).
``ora/environment/controllers/motion_pace.py`` turns a request into caps and
times a joint path from any planner under them;
``tests/ora/environment/controllers/test_motion_pace_parity.py`` keeps these
numbers equal to the toolkit's.

The wire form is ``{"speed": <preset or "custom">, "slow_approaching": bool}``
plus ``"speed_m_s"`` exactly when ``speed`` is ``custom``. A move_pose that
names no speed option carries no ``speed`` at all and runs at the default.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

#: The presets, the default and the name a numeric ``speed_m_s`` request carries.
SPEED_PRESETS: tuple[str, ...] = ("fast", "normal", "careful")
DEFAULT_SPEED = "fast"
CUSTOM_SPEED = "custom"
#: The wire form of a tool's own caps (``lib.motion_plan.timing.Pace``), never a policy argument.
EXPLICIT_SPEED = "explicit"
CAP_FIELDS: tuple[str, ...] = ("linear_m_s", "angular_deg_s", "linear_accel_m_s2", "angular_accel_deg_s2",
                               "joint_speed_rad_s", "joint_accel_rad_s2")
#: Explicit caps may also name the joint jerk the timing certifies the move at (rad/s^3 per joint), which a
#: preset never does: a robot motion profile's own pace (RoboDojo's ARX X5 profile v2 ``fast``). Without it the
#: robot's jerk applies (``environment.controllers.motion_timing.JOINT_JERK_LIMITS``), and a robot's documented
#: jerk is never exceeded (``motion_timing.pace_jerk``).
OPTIONAL_CAP_FIELDS: tuple[str, ...] = ("joint_jerk_rad_s3",)
#: The policy arguments that name a speed (a call that gives none runs at the default).
SPEED_ARGUMENTS: tuple[str, ...] = ("speed", "speed_m_s", "slow_approaching")

#: Each preset's caps. ``linear_m_s`` / ``angular_deg_s`` cap the controlled point's speed and turn rate,
#: ``linear_accel_m_s2`` / ``angular_accel_deg_s2`` how fast it may reach them (``None``: no Cartesian cap,
#: the joints alone limit the move); ``joint_speed_rad_s`` / ``joint_accel_rad_s2`` cap every joint at every
#: speed, and a backend lowers them to its own joint ceiling, never raises them.
PRESET_CAPS: dict[str, dict[str, float | None]] = {
    # Full speed: no Cartesian cap; 80% of the ARX X5's 5 rad/s joint limit, where the design was measured.
    "fast": {"linear_m_s": None, "angular_deg_s": None, "linear_accel_m_s2": None, "angular_accel_deg_s2": None,
             "joint_speed_rad_s": 4.0, "joint_accel_rad_s2": 40.0},
    "normal": {"linear_m_s": 0.3, "angular_deg_s": 60.0, "linear_accel_m_s2": 1.0, "angular_accel_deg_s2": 200.0,
               "joint_speed_rad_s": 2.5, "joint_accel_rad_s2": 15.0},
    "careful": {"linear_m_s": 0.1, "angular_deg_s": 20.0, "linear_accel_m_s2": 0.5, "angular_accel_deg_s2": 100.0,
                "joint_speed_rad_s": 1.2, "joint_accel_rad_s2": 8.0},
}
#: ``speed_m_s``: the turn cap per m/s (normal's 60 deg/s at 0.3 m/s, careful's 20 at 0.1), never below
#: careful's, both caps reached in ``RAMP_S`` from rest, and fast's joint caps.
TURN_DEG_S_PER_M_S = 200.0
MIN_TURN_DEG_S = 20.0
RAMP_S = 0.25
#: ``slow_approaching``: the last this-many metres of the path also obey this preset.
SLOW_APPROACH_M = 0.02
SLOW_PRESET = "careful"


class SpeedArgumentError(ValueError):
    """A refused speed argument: the message, the field it names and what was received."""

    def __init__(self, message: str, *, field: str, received: Any = None,
                 constraints: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.field = field
        self.received = received
        self.constraints = dict(constraints or {})


def custom_caps(speed_m_s: float) -> dict[str, float]:
    """The caps of a numeric top speed: that linear cap, the turn cap per m/s and fast's joint caps.

    Example: custom_caps(0.05)["angular_deg_s"] == 20.0."""

    turn = max(MIN_TURN_DEG_S, TURN_DEG_S_PER_M_S * float(speed_m_s))
    fast = PRESET_CAPS[DEFAULT_SPEED]
    return {"linear_m_s": float(speed_m_s), "angular_deg_s": turn,
            "linear_accel_m_s2": float(speed_m_s) / RAMP_S, "angular_accel_deg_s2": turn / RAMP_S,
            "joint_speed_rad_s": float(fast["joint_speed_rad_s"]),
            "joint_accel_rad_s2": float(fast["joint_accel_rad_s2"])}


def speed_request(arguments: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The validated speed of a call, in wire form, and the arguments without the speed fields.

    ``None`` when the call names no speed option (each null counts as omitted). Raises
    ``SpeedArgumentError``. Example: speed_request({"speed": "normal"})[0] ==
    {"speed": "normal", "slow_approaching": False}."""

    rest = {key: value for key, value in arguments.items() if key not in SPEED_ARGUMENTS}
    if all(arguments.get(key) is None for key in SPEED_ARGUMENTS):
        return None, rest
    speed = arguments.get("speed")
    speed_m_s = arguments.get("speed_m_s")
    if speed_m_s is not None:
        if speed is not None:
            raise SpeedArgumentError("give speed or speed_m_s, not both", field="speed_m_s",
                                     received=speed_m_s, constraints={"speed": speed})
        if (isinstance(speed_m_s, bool) or not isinstance(speed_m_s, (int, float))
                or not math.isfinite(speed_m_s) or speed_m_s <= 0):
            raise SpeedArgumentError("speed_m_s must be a number above 0 (m/s)", field="speed_m_s",
                                     received=speed_m_s, constraints={"exclusive_minimum": 0})
        speed, speed_m_s = CUSTOM_SPEED, float(speed_m_s)
    else:
        speed = DEFAULT_SPEED if speed is None else speed
        if speed not in SPEED_PRESETS:
            raise SpeedArgumentError("speed must be fast, normal or careful", field="speed", received=speed,
                                     constraints={"allowed": list(SPEED_PRESETS), "default": DEFAULT_SPEED})
    slow = arguments.get("slow_approaching")
    slow = False if slow is None else slow
    if not isinstance(slow, bool):
        raise SpeedArgumentError("slow_approaching must be true or false", field="slow_approaching",
                                 received=slow)
    careful = float(PRESET_CAPS[SLOW_PRESET]["linear_m_s"] or 0.0)
    if slow and speed == SLOW_PRESET:
        raise SpeedArgumentError("slow_approaching has no effect with speed=careful", field="slow_approaching",
                                 constraints={"speed": speed, "allowed_speeds": ["fast", "normal"]})
    if slow and speed_m_s is not None and speed_m_s <= careful:
        raise SpeedArgumentError(
            f"slow_approaching has no effect with speed_m_s at or below {careful:g} m/s",
            field="slow_approaching", constraints={"speed_m_s": speed_m_s, "speed_m_s_above": careful})
    return validate_speed({"speed": speed, "slow_approaching": slow,
                           **({"speed_m_s": speed_m_s} if speed_m_s is not None else {})}), rest


def _explicit_caps(value: Any) -> dict[str, float | None]:
    if (not isinstance(value, Mapping) or not set(CAP_FIELDS) <= set(value)
            or set(value) - set(CAP_FIELDS) - set(OPTIONAL_CAP_FIELDS)):
        raise ValueError(f"explicit speed caps must contain {sorted(CAP_FIELDS)} "
                         f"and may add {sorted(OPTIONAL_CAP_FIELDS)}")
    result: dict[str, float | None] = {}
    for key in (*CAP_FIELDS, *(name for name in OPTIONAL_CAP_FIELDS if name in value)):
        cap = value[key]
        joint = key.startswith("joint_")
        if cap is None and not joint:
            result[key] = None
        elif isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
            raise ValueError(f"explicit speed cap {key} must be a finite number above 0")
        else:
            result[key] = float(cap)
    return result


def validate_speed(value: Any) -> dict[str, Any]:
    """The normalized wire form of a speed request; raises ``ValueError`` on anything else.

    Example: validate_speed({"speed": "custom", "speed_m_s": 0.2, "slow_approaching": True})."""

    if not isinstance(value, Mapping):
        raise ValueError("move_pose speed must be an object")
    speed = value.get("speed")
    if speed == EXPLICIT_SPEED:
        if set(value) != {"speed", "caps", "slow_caps", "slow_approaching"}:
            raise ValueError("an explicit speed names its caps, slow_caps and slow_approaching")
        slow = value["slow_approaching"]
        if not isinstance(slow, bool) or slow != (value["slow_caps"] is not None):
            raise ValueError("explicit slow_approaching must say whether slow_caps are given")
        return {"speed": EXPLICIT_SPEED, "caps": _explicit_caps(value["caps"]),
                "slow_caps": _explicit_caps(value["slow_caps"]) if slow else None, "slow_approaching": slow}
    expected = {"speed", "slow_approaching"} | ({"speed_m_s"} if speed == CUSTOM_SPEED else set())
    if set(value) != expected:
        raise ValueError(f"move_pose speed must contain {sorted(expected)}")
    if speed not in (*SPEED_PRESETS, CUSTOM_SPEED):
        raise ValueError("move_pose speed.speed must be fast, normal, careful or custom")
    slow = value["slow_approaching"]
    if not isinstance(slow, bool):
        raise ValueError("move_pose speed.slow_approaching must be a boolean")
    result: dict[str, Any] = {"speed": str(speed), "slow_approaching": slow}
    if speed == CUSTOM_SPEED:
        top = value["speed_m_s"]
        if isinstance(top, bool) or not isinstance(top, (int, float)) or not math.isfinite(top) or top <= 0:
            raise ValueError("move_pose speed.speed_m_s must be a finite number above 0")
        result["speed_m_s"] = float(top)
    if slow and speed == SLOW_PRESET:
        raise ValueError("move_pose speed.slow_approaching has no effect with speed careful")
    if slow and speed == CUSTOM_SPEED and result["speed_m_s"] <= float(PRESET_CAPS[SLOW_PRESET]["linear_m_s"] or 0):
        raise ValueError("move_pose speed.slow_approaching has no effect at or below careful's speed")
    return result


def speed_caps(value: Mapping[str, Any] | None) -> tuple[dict[str, float | None], dict[str, float | None] | None]:
    """The caps of a (validated) request and of its slow final segment, or None without one.

    ``None`` is the default speed. Example: speed_caps(None)[0] == PRESET_CAPS["fast"]."""

    request = validate_speed(value) if value is not None else {"speed": DEFAULT_SPEED, "slow_approaching": False}
    if request["speed"] == EXPLICIT_SPEED:
        return dict(request["caps"]), (dict(request["slow_caps"]) if request["slow_approaching"] else None)
    caps = (custom_caps(request["speed_m_s"]) if request["speed"] == CUSTOM_SPEED
            else dict(PRESET_CAPS[request["speed"]]))
    return caps, (dict(PRESET_CAPS[SLOW_PRESET]) if request["slow_approaching"] else None)


__all__ = [
    "CAP_FIELDS", "CUSTOM_SPEED", "DEFAULT_SPEED", "EXPLICIT_SPEED", "MIN_TURN_DEG_S", "OPTIONAL_CAP_FIELDS", "PRESET_CAPS",
    "RAMP_S", "SLOW_APPROACH_M", "SLOW_PRESET",
    "SPEED_ARGUMENTS", "SPEED_PRESETS", "SpeedArgumentError", "TURN_DEG_S_PER_M_S", "custom_caps", "speed_caps",
    "speed_request", "validate_speed",
]
