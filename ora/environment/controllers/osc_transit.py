"""A free-space move the planner found clear and nearly straight, run as one straight OSC line.

Host-safe: the route choice and the reference are plain functions of measured poses, a
planner's grip-site path and the bound profile's ``osc_transit`` limits, so the session,
its dry run and the tests compute them the same way.

The planner checked its own joint path for collisions, the whole arm on it. The straight
line runs on the operational-space controller instead, whose arm posture is its own: of
that line only the gripper's path is what the planner checked (it lies within
``max_path_deviation_m`` of the planner's), the arm's links on the way are not.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

#: The route a free-space move ran: the planner's joint path, or the straight OSC line.
ROUTE_JOINT = "joint_path"
ROUTE_OSC = "osc_straight_line"
OSC_TRANSIT_FIELDS = frozenset({
    "osc_kp", "follow_speed_m_s", "follow_angular_rad_s", "accel_m_s2", "angular_accel_rad_s2",
    "max_path_deviation_m", "max_rotation_rad", "settle_ticks",
})


class OscTransitError(ValueError):
    """An osc_transit section or reference request is malformed."""


def _positive(value: Any, field: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or not 0 < float(value) <= maximum:
        raise OscTransitError(f"osc_transit.{field} must be in (0, {maximum}]")
    return float(value)


def validate_osc_transit(section: Any, *, osc_output_max_m: float = 0.05) -> dict[str, Any]:
    """Validate one profile's ``osc_transit`` section.

    The follow speed is admitted only where its lead, ``(2 / sqrt(osc_kp))`` times the speed,
    stays inside the OSC controller's per-axis reach (LIBERO's ``osc_pose.json``: 0.05 m).
    """

    if not isinstance(section, Mapping) or set(section) != OSC_TRANSIT_FIELDS:
        raise OscTransitError("osc_transit fields drifted")
    kp = _positive(section["osc_kp"], "osc_kp", 300.0)
    speed = _positive(section["follow_speed_m_s"], "follow_speed_m_s", 1.0)
    if speed * 2.0 / math.sqrt(kp) > osc_output_max_m:
        raise OscTransitError("osc_transit.follow_speed_m_s leads the controller past its reach")
    _positive(section["follow_angular_rad_s"], "follow_angular_rad_s", 6.0)
    _positive(section["accel_m_s2"], "accel_m_s2", 10.0)
    _positive(section["angular_accel_rad_s2"], "angular_accel_rad_s2", 40.0)
    _positive(section["max_path_deviation_m"], "max_path_deviation_m", 0.1)
    _positive(section["max_rotation_rad"], "max_rotation_rad", math.pi / 2)
    ticks = section["settle_ticks"]
    if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 200:
        raise OscTransitError("osc_transit.settle_ticks must be an integer in [1, 200]")
    return dict(section)


def lead_s(section: Mapping[str, Any]) -> float:
    """The critically damped OSC controller's lag per unit speed: ``2 / sqrt(osc_kp)`` seconds."""

    return 2.0 / math.sqrt(float(section["osc_kp"]))


def rotation_angle(first_xyzw: Sequence[float], second_xyzw: Sequence[float]) -> float:
    """The angle between two unit quaternions, in radians (0 to pi)."""

    dot = abs(sum(float(a) * float(b) for a, b in zip(first_xyzw, second_xyzw, strict=True)))
    return 2.0 * math.acos(min(1.0, dot))


def path_deviation(positions: Sequence[Sequence[float]]) -> float:
    """The largest distance of a path's points from the straight segment between its first and last point."""

    if not positions:
        raise OscTransitError("a path needs at least one point")
    start, end = [float(v) for v in positions[0]], [float(v) for v in positions[-1]]
    axis = [e - s for s, e in zip(start, end)]
    length2 = sum(a * a for a in axis)
    worst = 0.0
    for point in positions:
        offset = [float(p) - s for p, s in zip(point, start)]
        t = 0.0 if length2 == 0 else max(0.0, min(1.0, sum(o * a for o, a in zip(offset, axis)) / length2))
        nearest = [s + t * a for s, a in zip(start, axis)]
        worst = max(worst, math.dist([float(p) for p in point], nearest))
    return worst


def route_choice(
    section: Mapping[str, Any] | None, *, interaction: str | None, slow_approaching: bool, plan_reference: bool,
    planner_path_positions: Sequence[Sequence[float]] | None, rotation_rad: float,
) -> dict[str, Any] | None:
    """Which route a planned free-space move runs, and why: None under a profile with no ``osc_transit``.

    The OSC line needs every one of: a ``free_space`` move (a pre-contact segment or a recovery keeps
    the planner's path), no ``slow_approaching`` (its slow final segment is the joint route's), not a
    previewed ``plan_ref`` (that run is the plan the dry run showed), a turn of at most
    ``max_rotation_rad``, and the planner's grip-site path within ``max_path_deviation_m`` of the
    straight line from its start to its goal.
    """

    if section is None:
        return None
    limits = {"max_path_deviation_m": float(section["max_path_deviation_m"]),
              "max_rotation_rad": float(section["max_rotation_rad"])}
    record: dict[str, Any] = {"route": ROUTE_JOINT, "reason": None, "rotation_rad": float(rotation_rad),
                              "path_deviation_m": None, "limits": limits,
                              "collision_checked": "planner_joint_path"}
    if interaction != "free_space":
        record["reason"] = "not_free_space"
    elif slow_approaching:
        record["reason"] = "slow_approaching_requested"
    elif plan_reference:
        record["reason"] = "previewed_plan_runs_as_planned"
    elif rotation_rad > limits["max_rotation_rad"]:
        record["reason"] = "turn_above_osc_limit"
    elif not planner_path_positions:
        record["reason"] = "planner_path_unavailable"
    else:
        record["path_deviation_m"] = deviation = path_deviation(planner_path_positions)
        if deviation > limits["max_path_deviation_m"]:
            record["reason"] = "planner_path_not_straight"
        else:
            record.update(route=ROUTE_OSC, reason="planner_path_clear_and_straight",
                          collision_checked="planner_gripper_path_only")
    return record


def _slerp(first: Sequence[float], second: Sequence[float], fraction: float) -> list[float]:
    a = [float(v) for v in first]
    b = [float(v) for v in second]
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = [-v for v in b], -dot
    if dot > 0.9995:
        mixed = [x + fraction * (y - x) for x, y in zip(a, b)]
    else:
        theta = math.acos(min(1.0, dot))
        s = math.sin(theta)
        wa, wb = math.sin((1 - fraction) * theta) / s, math.sin(fraction * theta) / s
        mixed = [wa * x + wb * y for x, y in zip(a, b)]
    norm = math.sqrt(sum(v * v for v in mixed))
    return [v / norm for v in mixed]


def straight_line_reference(
    start_xyzw: Sequence[float], goal_xyzw: Sequence[float], *, linear_m_s: float, angular_rad_s: float,
    linear_accel_m_s2: float, angular_accel_rad_s2: float, tick_s: float, lead_s: float,
) -> dict[str, Any]:
    """The straight line from ``start`` to ``goal`` (position lerp, orientation slerp), one aim per tick.

    The progress ``s`` along the line rises from rest to rest with the tightest of the four caps
    (each cap on the move's own length or turn), sampled at whole ticks so the last tick ends on the
    goal. Each tick aims at the line's pose at ``s + lead_s * ds/dt``, never past the goal, so the
    controller's lag lands the hand on the line; the last tick aims at the goal.
    """

    length = math.dist([float(v) for v in start_xyzw[:3]], [float(v) for v in goal_xyzw[:3]])
    turn = rotation_angle(start_xyzw[3:], goal_xyzw[3:])
    rate_caps = [cap / extent for cap, extent in ((linear_m_s, length), (angular_rad_s, turn)) if extent > 1e-9]
    accel_caps = [cap / extent for cap, extent in ((linear_accel_m_s2, length), (angular_accel_rad_s2, turn))
                  if extent > 1e-9]
    if not rate_caps:
        return {"ticks": 0, "aims": [], "length_m": length, "turn_rad": turn}
    rate, accel = min(rate_caps), min(accel_caps)
    ramp = rate / accel
    if rate * ramp >= 1.0:  # never reaches the cruise rate: a triangle
        ramp = math.sqrt(1.0 / accel)
        rate = accel * ramp
        duration = 2.0 * ramp
    else:
        duration = 2.0 * ramp + (1.0 - rate * ramp) / rate
    ticks = max(1, math.ceil(duration / tick_s - 1e-9))
    stretch = duration / (ticks * tick_s)

    def progress(time_s: float) -> float:
        t = min(duration, max(0.0, time_s * stretch))
        if t < ramp:
            return 0.5 * accel * t * t
        if t > duration - ramp:
            r = duration - t
            return 1.0 - 0.5 * accel * r * r
        return 0.5 * accel * ramp * ramp + rate * (t - ramp)

    def pose(fraction: float) -> list[float]:
        position = [float(a) + fraction * (float(b) - float(a)) for a, b in zip(start_xyzw[:3], goal_xyzw[:3])]
        return position + _slerp(start_xyzw[3:], goal_xyzw[3:], fraction)

    samples = [progress(k * tick_s) for k in range(ticks + 2)]
    aims = []
    for k in range(1, ticks + 1):
        speed = (samples[k + 1] - samples[k - 1]) / (2.0 * tick_s)
        aims.append(pose(1.0 if k == ticks else min(1.0, samples[k] + lead_s * speed)))
    return {"ticks": ticks, "aims": aims, "length_m": length, "turn_rad": turn,
            "duration_s": ticks * tick_s}


__all__ = [
    "OSC_TRANSIT_FIELDS", "OscTransitError", "ROUTE_JOINT", "ROUTE_OSC", "lead_s", "path_deviation",
    "rotation_angle", "route_choice", "straight_line_reference", "validate_osc_transit",
]
