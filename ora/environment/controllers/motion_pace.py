"""Planner-neutral end-effector pacing: time any joint path so the controlled point keeps a requested speed.

A planner (cuRobo V2, a straight-line IK chain, ...)
answers a joint path; this module decides how fast to run it. The speed is
limited at the controlled point first -- its linear and angular speed and
acceleration, from the request's preset or ``speed_m_s``
(``ora.contracts.motion_speed``) -- and each joint's speed and acceleration
stay the ceiling, lowered further to what the backend's executor can follow.
``slow_approaching`` also holds the last ``SLOW_APPROACH_M`` of the path to
``careful``, entered without a stop. The caller supplies forward kinematics of
its controlled point; nothing here knows a robot, a planner or a simulator.

Before anything moves it also checks the wrist: a path whose controlled point
turns far more than the turn from its start to its end orientation is a
wrist unwinding the long way round, typically at its last joint's limit, and
is refused (``paced_path_wrist_unwind``); a budget cut never stops a large
wrist turn part way (``Grid.atomic``); and the wrist joint's extremes along
the planned path are reported, because both ends alone can hide a swing.

The semantics are Yulun's RoboDojo toolkit timing, generalized in two ways: the
tick is the backend's control period instead of the ARX X5's 40 ms native
action, and the joint caps are bounded by the backend's ceiling instead of the
X5's 5 rad/s. The toolkit now asks this module for its timing, through
``kinematics.pace_path``; ``tools/move_pose/_smooth.py`` keeps its presets, and
``tests/ora/environment/controllers/test_motion_pace_parity.py`` keeps these
presets and the toolkit's equal.

numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.contracts.motion_speed import (
    CUSTOM_SPEED, DEFAULT_SPEED, SLOW_APPROACH_M, SLOW_PRESET, speed_caps, validate_speed,
)

from .pace_refusal import (
    BUDGET_CODE, WRIST_UNWIND_CODE, PaceRefusal, atomic_start, budget_refusal, cover_refusal, unsplit_refusal,
)

#: A grid segment that moves nothing is crossed in this long (one ARX X5 physics step, as the lib does).
IDLE_SEGMENT_S = 0.004
#: Grid spacing along a planner's joint path (largest joint), as the lib's joint grid.
JOINT_GRID_SPACING_RAD = 0.02
#: Path rows this close (largest joint) to the row before them repeat it and are dropped before timing.
DUPLICATE_ROW_RAD = 1e-6
#: A path row where the joint-space direction turns by more than this is a corner: the grid keeps it exactly
#: and the move stops there. A smooth planner path has none (at most 6 degrees on 219 recorded LIBERO plans).
CORNER_DEG = 20.0
#: A monotonic turn of the wrist joint at least this large runs whole or not at all under a budget cut.
LARGE_WRIST_TURN_RAD = math.pi / 2
#: The controlled point may turn at most this much more than from its start to its end orientation;
#: beyond it the wrist is unwinding the long way round and the move is refused before anything moves.
MAX_UNREQUESTED_TURN_DEG = 45.0
@dataclass(frozen=True)
class Pace:
    """Speed and acceleration caps of one move; ``None`` leaves a Cartesian cap off (joint caps still hold)."""

    linear_m_s: float | None
    angular_deg_s: float | None
    linear_accel_m_s2: float | None
    angular_accel_deg_s2: float | None
    joint_speed_rad_s: float
    joint_accel_rad_s2: float
    ramp_free_below_m: float = 0.02
    ramp_free_below_deg: float = 5.0
    ramp_free_below_rad: float = 0.05
    #: The joint jerk the timing worker certifies this pace at (rad/s^3, every joint); ``None``: the robot's
    #: (``motion_timing.pace_jerk``). Only a robot motion profile's own pace names one (explicit caps).
    joint_jerk_rad_s3: float | None = None

    def __post_init__(self) -> None:
        for value in (self.linear_m_s, self.angular_deg_s, self.linear_accel_m_s2, self.angular_accel_deg_s2):
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError("pace caps must be positive or None")
        for value in (self.joint_speed_rad_s, self.joint_accel_rad_s2):
            if not (math.isfinite(value) and value > 0):
                raise ValueError("joint caps must be positive")
        if self.joint_jerk_rad_s3 is not None and not (math.isfinite(self.joint_jerk_rad_s3)
                                                       and self.joint_jerk_rad_s3 > 0):
            raise ValueError("a pace's joint jerk must be positive or None")

    def within(self, joint_speed_rad_s: float | None = None, joint_accel_rad_s2: float | None = None) -> "Pace":
        """This pace with its joint caps no higher than a backend's ceiling. Example: pace.within(1.0)."""

        return replace(
            self,
            joint_speed_rad_s=min(self.joint_speed_rad_s, joint_speed_rad_s or math.inf),
            joint_accel_rad_s2=min(self.joint_accel_rad_s2, joint_accel_rad_s2 or math.inf))

    @classmethod
    def from_caps(cls, caps: Mapping[str, float | None]) -> "Pace":
        jerk = caps.get("joint_jerk_rad_s3")
        return cls(caps["linear_m_s"], caps["angular_deg_s"], caps["linear_accel_m_s2"],
                   caps["angular_accel_deg_s2"], float(caps["joint_speed_rad_s"] or 0.0),
                   float(caps["joint_accel_rad_s2"] or 0.0),
                   joint_jerk_rad_s3=None if jerk is None else float(jerk))


def paces(request: Mapping[str, Any] | None) -> tuple[Pace, Pace | None]:
    """The pace of a speed request (``ora.contracts.motion_speed`` wire form; None: the default) and its slow pace.

    Example: paces({"speed": "normal", "slow_approaching": True})[1].linear_m_s == 0.1."""

    caps, slow = speed_caps(request)
    return Pace.from_caps(caps), (Pace.from_caps(slow) if slow is not None else None)


@dataclass(frozen=True)
class Zone:
    """Grid segments ``first`` .. end also obey ``pace``, measured on one arm's own path (lib ``timing.Zone``)."""

    arm: str
    first: int
    pace: Pace
    linear_m: tuple[float, ...]
    angular_rad: tuple[float, ...]


@dataclass(frozen=True)
class Grid:
    """A path on grid nodes: fractions, the joints of each moving arm, segment lengths and forced stops.

    ``linear_m`` / ``angular_rad`` are the controlled point's length and turn over each grid segment (the
    largest over the arms); ``atomic`` are node spans a budget cut may not fall strictly inside.
    """

    fractions: tuple[float, ...]
    joints: Mapping[str, np.ndarray]
    linear_m: tuple[float, ...]
    angular_rad: tuple[float, ...]
    stops: tuple[int, ...] = ()
    zones: tuple[Zone, ...] = ()
    atomic: tuple[tuple[int, int], ...] = ()

    @property
    def segments(self) -> int:
        return len(self.fractions) - 1

    def joint_steps(self) -> np.ndarray:
        """Largest joint change of any arm over each grid segment."""

        steps = [np.max(np.abs(np.diff(values, axis=0)), axis=1) for values in self.joints.values()]
        return np.max(np.vstack(steps), axis=0) if steps else np.zeros(self.segments)


def _ratio(cap: float | None, length: float) -> float:
    return math.inf if cap is None or length <= 1e-12 else cap / length


def ramp_free(grid: Grid, pace: Pace) -> bool:
    """Whether the move is short enough to run at constant speed with no ramps (lib ``timing.ramp_free``)."""

    linear, angular, joint = (float(np.sum(grid.linear_m)), float(np.sum(grid.angular_rad)),
                              float(np.sum(grid.joint_steps())))
    return (linear < pace.ramp_free_below_m and math.degrees(angular) < pace.ramp_free_below_deg
            and (linear > 0 or joint < pace.ramp_free_below_rad) and not grid.stops)


def parameterize(grid: Grid, pace: Pace, *, idle_segment_s: float = IDLE_SEGMENT_S) -> tuple[np.ndarray, np.ndarray]:
    """Node speeds (grid segments per second) and segment durations (s) under every cap at once.

    The fastest progress along the grid that respects the controlled point's speed and acceleration and
    each joint's, by a forward-backward pass (a small TOPP); lib ``timing.parameterize``."""

    linear, angular, joint = np.asarray(grid.linear_m), np.asarray(grid.angular_rad), grid.joint_steps()
    omega = None if pace.angular_deg_s is None else math.radians(pace.angular_deg_s)
    alpha = None if pace.angular_accel_deg_s2 is None else math.radians(pace.angular_accel_deg_s2)
    caps = np.asarray([min(_ratio(pace.linear_m_s, linear[i]), _ratio(omega, angular[i]),
                           _ratio(pace.joint_speed_rad_s, joint[i])) for i in range(grid.segments)])
    accels = np.asarray([min(_ratio(pace.linear_accel_m_s2, linear[i]), _ratio(alpha, angular[i]),
                             _ratio(pace.joint_accel_rad_s2, joint[i])) for i in range(grid.segments)])
    for zone in grid.zones:
        steps = np.max(np.abs(np.diff(grid.joints[zone.arm], axis=0)), axis=1)
        z_omega = None if zone.pace.angular_deg_s is None else math.radians(zone.pace.angular_deg_s)
        z_alpha = None if zone.pace.angular_accel_deg_s2 is None else math.radians(zone.pace.angular_accel_deg_s2)
        for i in range(zone.first, grid.segments):
            caps[i] = min(caps[i], _ratio(zone.pace.linear_m_s, zone.linear_m[i]), _ratio(z_omega, zone.angular_rad[i]),
                          _ratio(zone.pace.joint_speed_rad_s, steps[i]))
            accels[i] = min(accels[i], _ratio(zone.pace.linear_accel_m_s2, zone.linear_m[i]),
                            _ratio(z_alpha, zone.angular_rad[i]), _ratio(zone.pace.joint_accel_rad_s2, steps[i]))
    if not np.all(np.isfinite(caps)):
        # A segment that moves nothing is crossed in one idle step.
        caps = np.where(np.isfinite(caps), caps, 1.0 / idle_segment_s)
        accels = np.where(np.isfinite(accels), accels, 1e9)
    if ramp_free(grid, pace):
        return np.asarray([caps[0], *[min(caps[i - 1], caps[i]) for i in range(1, grid.segments)], caps[-1]]), 1.0 / caps
    speeds = np.empty(grid.segments + 1)
    speeds[0], speeds[-1] = 0.0, 0.0
    for node in range(1, grid.segments):
        speeds[node] = 0.0 if node in grid.stops else min(caps[node - 1], caps[node])
    for node in range(grid.segments):
        speeds[node + 1] = min(speeds[node + 1], math.sqrt(speeds[node] ** 2 + 2 * accels[node]))
    for node in range(grid.segments - 1, -1, -1):
        speeds[node] = min(speeds[node], math.sqrt(speeds[node + 1] ** 2 + 2 * accels[node]))
    durations = np.empty(grid.segments)
    for segment in range(grid.segments):
        total = speeds[segment] + speeds[segment + 1]
        # A segment that starts and ends at rest takes the accelerate-then-brake time.
        durations[segment] = max(2.0 / total if total > 0 else 0.0, 1.0 / caps[segment],
                                 2.0 * math.sqrt(1.0 / accels[segment]) if total == 0 else 0.0)
    return speeds, durations


def _progress(speeds: np.ndarray, durations: np.ndarray, time: float, constant: bool = False) -> float:
    elapsed = 0.0
    for segment, duration in enumerate(durations):
        if time <= elapsed + duration or segment == len(durations) - 1:
            tau = min(max(time - elapsed, 0.0), duration)
            start, end = speeds[segment], speeds[segment + 1]
            if constant:
                advanced = tau / duration
            elif start + end > 0:
                acceleration = (end - start) / duration
                advanced = start * tau + 0.5 * acceleration * tau * tau
            else:
                advanced = tau / duration
            return segment + min(1.0, max(0.0, advanced))
        elapsed += duration
    return float(len(durations))


def knots(grid: Grid, pace: Pace, *, tick_s: float, idle_segment_s: float = IDLE_SEGMENT_S) -> dict[str, Any]:
    """One joint target per arm per control tick, ending exactly on the last grid node (lib ``timing.knots``).

    The duration is stretched to whole ticks, so no cap is exceeded. Returns ``{"joints": {arm: rows},
    "fractions": [...], "duration_s": T, "ticks": N}``."""

    if not (math.isfinite(tick_s) and tick_s > 0):
        raise ValueError("tick_s must be positive")
    speeds, durations = parameterize(grid, pace, idle_segment_s=idle_segment_s)
    constant = ramp_free(grid, pace)
    total = float(np.sum(durations))
    count = max(1, math.ceil(total / tick_s - 1e-9))
    fractions, joints = [], {arm: [] for arm in grid.joints}
    for knot in range(1, count + 1):
        position = (float(grid.segments) if knot == count
                    else _progress(speeds, durations, total * knot / count, constant))
        node = min(int(math.floor(position)), grid.segments - 1)
        weight = position - node
        fractions.append(float(grid.fractions[node] + weight * (grid.fractions[node + 1] - grid.fractions[node])))
        for arm, values in grid.joints.items():
            joints[arm].append([float(v) for v in values[node] + weight * (values[node + 1] - values[node])])
    return {"joints": joints, "fractions": fractions, "duration_s": round(count * tick_s, 6), "ticks": count}


def cut(grid: Grid, nodes: int) -> Grid:
    """The first ``nodes`` segments of a grid (a move that stops part of the way; lib ``timing.cut``)."""

    nodes = max(1, min(int(nodes), grid.segments))
    zones = tuple(Zone(zone.arm, zone.first, zone.pace, zone.linear_m[:nodes], zone.angular_rad[:nodes])
                  for zone in grid.zones if zone.first < nodes)
    return Grid(fractions=tuple(grid.fractions[:nodes + 1]),
                joints={arm: values[:nodes + 1] for arm, values in grid.joints.items()},
                linear_m=tuple(grid.linear_m[:nodes]), angular_rad=tuple(grid.angular_rad[:nodes]),
                stops=tuple(stop for stop in grid.stops if stop < nodes), zones=zones,
                atomic=tuple(span for span in grid.atomic if span[1] <= nodes))


def slow_final(grid: Grid, lengths: Mapping[str, tuple[Sequence[float], Sequence[float]]], pace: Pace,
               length_m: float = SLOW_APPROACH_M) -> Grid:
    """``grid`` whose last ``length_m`` of each arm's own path also obeys ``pace`` (lib ``timing.slow_final``).

    One trajectory: the profile slows continuously into the zone, with no stop; a path shorter than
    ``length_m`` is slow throughout. Unlike the lib, the zone starts at the node at or before the boundary
    (so it is between ``length_m`` and ``length_m`` plus one grid spacing long), never after it."""

    zones = []
    for arm in grid.joints:
        linear, angular = (tuple(float(v) for v in values) for values in lengths[arm])
        if len(linear) != grid.segments or len(angular) != grid.segments:
            raise ValueError("slow_final needs the arm's length of every grid segment")
        left = np.concatenate([np.cumsum(np.asarray(linear)[::-1])[::-1], [0.0]])
        first = int(np.argmax(left <= length_m + 1e-12))
        if first > 0 and left[first] < length_m - 1e-12:
            # Start at the node at or before the boundary, so the whole last length_m is slow: the caps hold
            # at grid nodes, and a zone that began after the boundary left the move fast inside it.
            first -= 1
        zones.append(Zone(arm, first, pace, linear, angular))
    return Grid(grid.fractions, grid.joints, grid.linear_m, grid.angular_rad, grid.stops, tuple(zones), grid.atomic)


def zone_starts(grid: Grid) -> dict[str, int]:
    """The grid node where each arm's zone starts."""

    return {zone.arm: zone.first for zone in grid.zones}


def angle_between(first: Sequence[float], second: Sequence[float]) -> float:
    """Rotation angle (rad) between two xyzw orientations (lib ``paths.angle_between``)."""

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    dot = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return 2.0 * math.acos(min(1.0, dot))


def segment_lengths(poses: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Cartesian length and turn of each segment between consecutive xyz+xyzw poses."""

    linear = tuple(math.dist(a[:3], b[:3]) for a, b in zip(poses, poses[1:]))
    angular = tuple(angle_between(a[3:7], b[3:7]) for a, b in zip(poses, poses[1:]))
    return linear, angular


def caps(pace: Pace) -> dict[str, Any]:
    """A pace's speed caps as a receipt reports them (lib ``move.caps``); its jerk only when it names one."""

    return {"linear_m_s": pace.linear_m_s, "angular_deg_s": pace.angular_deg_s,
            "joint_speed_deg_s": round(math.degrees(pace.joint_speed_rad_s), 3),
            **({} if pace.joint_jerk_rad_s3 is None else {"joint_jerk_rad_s3": pace.joint_jerk_rad_s3})}


def peaks(grid: Grid, pace: Pace) -> dict[str, float]:
    """The top speed and turn rate of the controlled point along a planned grid (lib ``move.peaks``).

    Each grid segment's length over its planned duration, under every cap of the pace, joints included."""

    _, durations = parameterize(grid, pace)
    moving = durations > 0
    linear = np.asarray(grid.linear_m)[moving] / durations[moving]
    angular = np.asarray(grid.angular_rad)[moving] / durations[moving]
    return {"planned_peak_linear_m_s": round(float(linear.max(initial=0.0)), 4),
            "planned_peak_angular_deg_s": round(math.degrees(float(angular.max(initial=0.0))), 2)}


def fit_budget(grid: Grid, pace: Pace, budget: int, *, tick_s: float) -> tuple[Grid, dict[str, Any], float]:
    """The longest first part of ``grid`` whose knots fit ``budget`` ticks (lib ``checks.fit_budget``).

    A cut never falls inside one of ``grid.atomic`` (a large wrist turn, a joint-line bridge): it moves back to
    the span's start. Returns (grid part, its knots, fraction of the move). Raises ``PaceRefusal``
    (``paced_motion_budget``) when not even one segment fits, or when the part that fits ends inside an atomic
    span that starts the move (``constraints.unsplit_span``)."""

    planned = knots(grid, pace, tick_s=tick_s)
    if planned["ticks"] <= budget:
        return grid, planned, 1.0
    low, high, best = 1, grid.segments, None
    while low <= high:
        middle = (low + high) // 2
        candidate = knots(cut(grid, middle), pace, tick_s=tick_s)
        if candidate["ticks"] <= budget:
            best, low = (middle, candidate), middle + 1
        else:
            high = middle - 1
    if best is None:
        raise cover_refusal(budget)
    nodes, span = atomic_start(grid.atomic, best[0])
    if nodes == 0:
        raise unsplit_refusal(budget, planned["ticks"], grid.fractions[span[0]], grid.fractions[span[1]])
    if nodes != best[0]:
        best = (nodes, knots(cut(grid, nodes), pace, tick_s=tick_s))
    part = cut(grid, best[0])
    return part, best[1], float(part.fractions[-1])


# --------------------------------------------------------------- a planner's joint path on a grid


@dataclass(frozen=True)
class PathGrid:
    """A planner's joint path resampled on a grid, with the controlled point's pose at every node."""

    grid: Grid
    poses: tuple[tuple[float, ...], ...]
    arm: str


def wrist_turn_spans(values: Sequence[float], *, min_turn_rad: float = LARGE_WRIST_TURN_RAD) -> tuple[tuple[int, int], ...]:
    """Node spans over which one joint turns monotonically by at least ``min_turn_rad``.

    Example: wrist_turn_spans([0, 1, 2, 2, 1]) == ((0, 2),)."""

    deltas = np.diff(np.asarray(values, dtype=float))
    spans, first, last, sign, turned = [], 0, 0, 0, 0.0
    for index, delta in enumerate(deltas):
        step = 0 if abs(delta) <= 1e-12 else (1 if delta > 0 else -1)
        if step == 0:
            continue
        if step != sign:
            if sign != 0 and turned >= min_turn_rad - 1e-12:
                spans.append((first, last + 1))
            first, sign, turned = index, step, 0.0
        turned += abs(float(delta))
        last = index
    if sign != 0 and turned >= min_turn_rad - 1e-12:
        spans.append((first, last + 1))
    return tuple(spans)


def distinct_rows(points: np.ndarray, *, tolerance_rad: float = DUPLICATE_ROW_RAD) -> np.ndarray:
    """The path without rows that repeat the row before them, its first and last rows kept exactly.

    A planner's timed output holds the start and the goal for a few rows (cuRobo repeats both). Kept, each
    repeat is a grid segment of zero length, which the timing crosses at an unbounded acceleration, so the
    move would skip its ramps. Example: distinct_rows(np.array([a, a, b, b])) has the rows a and b."""

    keep = [0]
    for index in range(1, len(points)):
        if float(np.max(np.abs(points[index] - points[keep[-1]]))) > tolerance_rad:
            keep.append(index)
    result = points[keep].copy()
    if len(result) > 1:
        # The goal is the planner's last row exactly, never a repeat within the tolerance of it.
        result[-1] = points[-1]
    return result


def _spread(points: np.ndarray, cost: np.ndarray, count: int) -> list[np.ndarray]:
    """``count`` + 1 nodes along the polyline through ``points``, evenly spaced in ``cost`` per piece."""

    total = float(np.sum(cost))
    reach = np.concatenate([[0.0], np.cumsum(cost)])
    nodes = []
    for index in range(count + 1):
        at = total * index / count
        piece = min(int(np.searchsorted(reach, at, side="right")) - 1, len(cost) - 1)
        while cost[piece] <= 0.0 and piece < len(cost) - 1:
            piece += 1
        weight = 0.0 if cost[piece] <= 0.0 else (at - reach[piece]) / cost[piece]
        nodes.append(points[piece] + min(max(weight, 0.0), 1.0) * (points[piece + 1] - points[piece]))
    nodes[0], nodes[-1] = points[0].copy(), points[-1].copy()
    return nodes


def corners(points: np.ndarray) -> list[int]:
    """The row indices that split the path: its ends and every row where its direction turns past CORNER_DEG."""

    split = [0]
    for index in range(1, len(points) - 1):
        before, after = points[index] - points[index - 1], points[index + 1] - points[index]
        norms = float(np.linalg.norm(before)) * float(np.linalg.norm(after))
        if norms <= 0.0:
            continue
        cosine = float(before @ after) / norms
        if cosine < math.cos(math.radians(CORNER_DEG)):
            split.append(index)
    return [*split, len(points) - 1]


def _even_nodes(points: np.ndarray, spacing_rad: float) -> np.ndarray:
    """Nodes evenly spaced along the polyline through ``points`` (largest-joint length), at most
    ``spacing_rad`` apart, from its first row to its last, both exact, and through every corner.

    Even spacing is what the timing assumes: it keeps one speed per node in grid segments per second, so
    two adjacent segments of very different lengths (a planner's short steps where its own profile is
    slow, next to long ones) would let the controlled point jump in speed across the node they share and
    skip its ramp. Between corners (``CORNER_DEG``; a smooth planner path has none) the rows themselves
    are not kept; the move stops at a corner (``Grid.stops``), where its joints change direction. Grids the
    toolkit builds (lines and joint lines) are already even."""

    steps = np.max(np.abs(np.diff(points, axis=0)), axis=1) if len(points) > 1 else np.zeros(0)
    if float(np.sum(steps)) <= 0.0:
        return np.asarray([points[0], points[0]])
    nodes = [points[0].copy()]
    splits = corners(points)
    for first, last in zip(splits, splits[1:]):
        piece = steps[first:last]
        count = max(1, math.ceil(float(np.sum(piece)) / spacing_rad - 1e-9))
        nodes.extend(_spread(points[first:last + 1], piece, count)[1:])
    return np.asarray(nodes)


def joint_path_grid(path: Sequence[Sequence[float]], fk: Callable[[list[list[float]]], Sequence[Sequence[float]]], *,
                    arm: str = "arm", wrist_joint_index: int | None = None,
                    spacing_rad: float = JOINT_GRID_SPACING_RAD) -> PathGrid:
    """A planner's joint path (start first) as a grid at most ``spacing_rad`` apart, measured with ``fk``.

    Rows that repeat the row before them are dropped first (``distinct_rows``); consecutive rows are joined
    by straight joint lines, as an executor tracking them would, and the nodes are spread evenly along them
    (``_even_nodes``). ``fk`` maps joint rows to the controlled point's poses (x, y, z, qx, qy, qz, qw) in one
    call. Fractions are the nodes' share of the path's largest-joint length. A monotonic turn of
    ``wrist_joint_index`` of at least ``LARGE_WRIST_TURN_RAD`` is an atomic span."""

    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or len(points) < 1 or not np.isfinite(points).all():
        raise ValueError("a joint path needs finite rows")
    return _measured(_even_nodes(distinct_rows(points), spacing_rad), fk, arm, wrist_joint_index)


def _measured(joints: np.ndarray, fk: Callable[[list[list[float]]], Sequence[Sequence[float]]], arm: str,
              wrist_joint_index: int | None) -> PathGrid:
    poses = [tuple(float(v) for v in pose) for pose in fk(joints.tolist())]
    if len(poses) != len(joints) or any(len(pose) != 7 or not all(map(math.isfinite, pose)) for pose in poses):
        raise ValueError("forward kinematics must answer one finite xyz+xyzw pose per joint row")
    linear, angular = segment_lengths(poses)
    steps = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
    total = float(np.sum(steps))
    fractions = (tuple(float(v) for v in np.concatenate([[0.0], np.cumsum(steps) / total]))
                 if total > 0 else tuple(index / (len(joints) - 1) for index in range(len(joints))))
    fractions = (*fractions[:-1], 1.0)
    atomic = (wrist_turn_spans(joints[:, wrist_joint_index]) if wrist_joint_index is not None else ())
    # The joints change direction at a corner, so the move stops there (as the lib's concatenate does).
    stops = tuple(corners(joints)[1:-1]) if total > 0 else ()
    return PathGrid(Grid(fractions, {arm: joints}, linear, angular, stops=stops, atomic=atomic), tuple(poses), arm)


def paced_grid(measured: PathGrid, pace: Pace, fk: Callable[[list[list[float]]], Sequence[Sequence[float]]], *,
               wrist_joint_index: int | None = None, spacing_rad: float = JOINT_GRID_SPACING_RAD) -> PathGrid:
    """The same path with its nodes spread evenly in ``pace``'s acceleration caps, measured again with ``fk``.

    The timing keeps one speed per node in grid segments per second. Every speed cap holds inside each
    segment, but the acceleration passes carry that one node speed from a segment into the next, so where
    the two differ in the length whose acceleration binds (the controlled point's, under a Cartesian ramp),
    the point's speed steps at the node and its ramp is exceeded. Even joint spacing leaves such differences
    wherever the arm's Jacobian changes along the path; spacing the nodes by each piece's largest length
    over its acceleration cap (linear length over the linear ramp, turn over the turn ramp, joint step over
    the joint acceleration) makes the binding length the same in every segment. Nodes stay on the path, at
    most ``spacing_rad`` apart, first, last and corners exact; a pace without a Cartesian ramp keeps the
    even joint grid.
    Example: paced_grid(joint_path_grid(path, fk), paces({"speed": "normal", "slow_approaching": False})[0], fk)."""

    if pace.linear_accel_m_s2 is None and pace.angular_accel_deg_s2 is None:
        return measured
    grid = measured.grid
    joints = grid.joints[measured.arm]
    alpha = math.inf if pace.angular_accel_deg_s2 is None else math.radians(pace.angular_accel_deg_s2)
    steps = grid.joint_steps()
    cost = np.maximum.reduce([np.asarray(grid.linear_m) / (pace.linear_accel_m_s2 or math.inf),
                              np.asarray(grid.angular_rad) / alpha, steps / pace.joint_accel_rad_s2])
    total = float(np.sum(cost))
    if total <= 0.0:
        return measured
    # One cost per segment along the whole path, no coarser than the even grid and small enough that no
    # segment's largest joint step exceeds spacing_rad.
    density = float(np.max(np.where(cost > 0.0, steps / np.where(cost > 0.0, cost, 1.0), 0.0)))
    each = min(total / grid.segments, spacing_rad / density if density > 0.0 else math.inf)
    nodes = [joints[0].copy()]
    splits = corners(joints)
    for first, last in zip(splits, splits[1:]):
        piece = cost[first:last]
        if float(np.sum(piece)) <= 0.0:
            nodes.extend(joints[first + 1:last + 1])
            continue
        count = max(1, math.ceil(float(np.sum(piece)) / each - 1e-9))
        nodes.extend(_spread(joints[first:last + 1], piece, count)[1:])
    return _measured(np.asarray(nodes), fk, measured.arm, wrist_joint_index)


def wrist_report(start: Sequence[float], knot_rows: Sequence[Sequence[float]], *, joint_index: int,
                 limits_rad: tuple[float, float]) -> dict[str, float]:
    """One wrist joint from the start joints through every planned knot, in degrees (lib ``joint6_extremes``).

    start, end, min and max, ``travel_deg`` (the sum of its turns, which both ends alone can hide) and
    ``min_limit_margin_deg``, the least distance to either joint limit."""

    values = np.asarray([float(start[joint_index]), *(float(row[joint_index]) for row in knot_rows)])
    low, high = float(limits_rad[0]), float(limits_rad[1])
    margin = float(min(np.min(values - low), np.min(high - values)))
    return {"start_deg": round(math.degrees(values[0]), 2), "end_deg": round(math.degrees(values[-1]), 2),
            "min_deg": round(math.degrees(values.min()), 2), "max_deg": round(math.degrees(values.max()), 2),
            "travel_deg": round(math.degrees(float(np.sum(np.abs(np.diff(values))))), 2),
            "min_limit_margin_deg": round(math.degrees(margin), 2)}


def path_turns(poses: Sequence[Sequence[float]]) -> dict[str, float]:
    """How far the controlled point turns along a path and from its start to its end orientation, in degrees."""

    _, angular = segment_lengths(poses)
    path = math.degrees(float(sum(angular)))
    requested = math.degrees(angle_between(poses[0][3:7], poses[-1][3:7]))
    return {"path_turn_deg": round(path, 2), "requested_turn_deg": round(requested, 2)}


def check_unwind(poses: Sequence[Sequence[float]], *, max_unrequested_deg: float = MAX_UNREQUESTED_TURN_DEG,
                 evidence: Mapping[str, Any] | None = None) -> dict[str, float]:
    """The path's turns; raises ``PaceRefusal`` (``paced_path_wrist_unwind``) for a large unrequested turn.

    A path may turn the controlled point a little more than the turn from its start to its end orientation (a
    planner's detour); turning more than ``max_unrequested_deg`` beyond it means the wrist goes the long way
    round, typically at its last joint's limit, which the move did not ask for."""

    turns = path_turns(poses)
    extra = turns["path_turn_deg"] - turns["requested_turn_deg"]
    if extra > max_unrequested_deg:
        raise PaceRefusal(
            "the planned path turns the hand far more than the move asks for (the wrist would unwind the long "
            "way round, typically at its joint limit); nothing was executed. Ask for a target orientation nearer "
            "the current one; jaw_flip_retry, when present, says the half-turn jaw twin was tried too",
            code=WRIST_UNWIND_CODE,
            constraints={**turns, "unrequested_turn_deg": round(extra, 2),
                         "max_unrequested_turn_deg": max_unrequested_deg, **dict(evidence or {})})
    return turns


# --------------------------------------------------------------- one paced move


@dataclass(frozen=True)
class PacedPath:
    """A joint path timed for its speed request: the knots to execute and what the receipt reports."""

    grid: Grid
    knots: Mapping[str, Any]
    fraction: float
    pace: Pace
    slow_pace: Pace | None
    speed: Mapping[str, Any]
    wrist: Mapping[str, Any]
    slow_from_knot: int | None = None
    whole_ticks: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def rows(self) -> list[list[float]]:
        """The knots of the one arm, one joint row per tick."""

        (rows,) = self.knots["joints"].values()
        return [list(row) for row in rows]

    def report(self) -> dict[str, Any]:
        """``motion.speed`` and ``motion.wrist`` of a receipt, plus the planned ticks and duration, and the
        ``timing`` evidence: which implementation timed the move (``motion_timing.timing_evidence``)."""

        return {"speed": dict(self.speed), "wrist": dict(self.wrist), "planned_ticks": int(self.knots["ticks"]),
                "duration_s": float(self.knots["duration_s"]),
                **({"budget_limited": {"fraction_of_move": round(self.fraction, 3),
                                       "needed_ticks": self.whole_ticks}} if self.fraction < 1.0 else {}),
                **({"timing": dict(self.extra["timing"])} if "timing" in self.extra else {})}


def speed_report(request: Mapping[str, Any] | None, pace: Pace, slow_pace: Pace | None) -> dict[str, Any]:
    """The requested speed and its caps, as ``motion.speed`` reports them (RoboDojo's ``motion.speed`` keys)."""

    request = validate_speed(request) if request is not None else {"speed": DEFAULT_SPEED, "slow_approaching": False}
    return {"speed": request["speed"],
            **({"speed_m_s": request["speed_m_s"]} if request["speed"] == CUSTOM_SPEED else {}),
            "slow_approaching": request["slow_approaching"], **caps(pace),
            **({"slow_approach_m": SLOW_APPROACH_M, "slow": caps(slow_pace)} if slow_pace is not None else {})}


def pace_path(path: Sequence[Sequence[float]], fk: Callable[[list[list[float]]], Sequence[Sequence[float]]], *,
              speed: Mapping[str, Any] | None, tick_s: float, joint_speed_ceiling_rad_s: float | None,
              wrist_joint_index: int, wrist_limits_rad: tuple[float, float], budget_ticks: int,
              slow_approach: bool = True, cut_to_budget: bool = False,
              joint_accel_ceiling_rad_s2: float | None = None, arm: str = "arm",
              robot: str | None = None, unwind_check: bool = True) -> PacedPath:
    """Time one planner path for a speed request, check its wrist and fit its budget, before anything moves.

    ``speed`` is the request's wire form (``None``: the default). ``joint_speed_ceiling_rad_s`` is what the
    backend's executor can follow; every pace's joint caps are lowered to it. ``slow_approach`` false leaves
    out the request's slow final segment (a path that ends short of the move's goal, such as a pre-contact
    segment). A path that needs more than ``budget_ticks`` is refused (``paced_motion_budget``), or with
    ``cut_to_budget`` cut to the longest first part that fits, never inside a large wrist turn. Raises
    ``PaceRefusal`` (``paced_path_wrist_unwind``, ``paced_motion_budget``).

    With a timing worker in this process (``motion_timing.client``) the path is timed by TOPP-RA and
    certified by Ruckig at ``robot``'s hardware jerk limits (``motion_timing.JOINT_JERK_LIMITS``; ``None``:
    the unverified default); otherwise by this module's own timing below. The receipt's ``timing`` says
    which one ran. A path the library cannot time within the limits (``timing_infeasible``) falls back to
    this module's timing and says so. ``unwind_check`` false reports a wrist's path turn without
    refusing it: a recovery's own path back, which unwinds on purpose."""

    pace, slow_pace = paces(speed)
    pace = pace.within(joint_speed_ceiling_rad_s, joint_accel_ceiling_rad_s2)
    slow_pace = slow_pace.within(joint_speed_ceiling_rad_s, joint_accel_ceiling_rad_s2) if slow_pace else None
    from . import motion_timing

    timing = motion_timing.client()
    fallback = None
    if timing is not None:
        paced = motion_timing.library_pace(
            timing, path, fk, speed=speed, pace=pace, slow_pace=slow_pace, tick_s=tick_s,
            joint_speed_ceiling_rad_s=joint_speed_ceiling_rad_s, joint_accel_ceiling_rad_s2=joint_accel_ceiling_rad_s2,
            wrist_joint_index=wrist_joint_index, wrist_limits_rad=wrist_limits_rad, budget_ticks=budget_ticks,
            slow_approach=slow_approach, cut_to_budget=cut_to_budget, arm=arm, robot=robot, unwind_check=unwind_check)
        if isinstance(paced, PacedPath):
            return paced
        fallback = paced
    measured = paced_grid(joint_path_grid(path, fk, arm=arm, wrist_joint_index=wrist_joint_index), pace, fk,
                          wrist_joint_index=wrist_joint_index)
    grid = measured.grid
    zoned = slow_pace if slow_approach else None
    if zoned is not None:
        grid = slow_final(grid, {arm: (grid.linear_m, grid.angular_rad)}, zoned)
    start = [float(v) for v in measured.grid.joints[arm][0]]
    whole = knots(grid, pace, tick_s=tick_s)
    turns = path_turns(measured.poses)
    if unwind_check:
        turns = check_unwind(measured.poses, evidence={"wrist": wrist_report(
            start, whole["joints"][arm], joint_index=wrist_joint_index, limits_rad=wrist_limits_rad)})
    if cut_to_budget:
        part, planned, fraction = fit_budget(grid, pace, int(budget_ticks), tick_s=tick_s)
    elif whole["ticks"] > budget_ticks:
        raise budget_refusal(int(budget_ticks), whole["ticks"])
    else:
        part, planned, fraction = grid, whole, 1.0
    slow_from = None
    if zoned is not None:
        node = zone_starts(part).get(arm)
        if node is not None:
            first = float(part.fractions[node])
            slow_from = next((index + 1 for index, value in enumerate(planned["fractions"])
                              if value > first + 1e-12), None)
    report = speed_report(speed, pace, zoned)
    report.update(peaks(part, pace))
    wrist = {**wrist_report(start, planned["joints"][arm], joint_index=wrist_joint_index,
                            limits_rad=wrist_limits_rad), **turns}
    return PacedPath(part, planned, fraction, pace, zoned, report, wrist, slow_from, int(whole["ticks"]),
                     {"timing": motion_timing.timing_evidence(fallback=fallback)})


# --------------------------------------------------------------- a local Cartesian executor


@dataclass(frozen=True)
class StepCaps:
    """Per-tick caps of a local Cartesian executor that steers toward a pose target each tick.

    Such an executor has no planned path to time; it can still keep the requested speed by commanding no
    larger a step per tick than the cap allows (no ramps). Within ``slow_within_m`` of the target the slow
    caps also hold. ``None`` caps are off.
    """

    linear_m: float | None
    angular_rad: float | None
    slow_linear_m: float | None = None
    slow_angular_rad: float | None = None
    slow_within_m: float = 0.0

    def limits(self, remaining_m: float) -> tuple[float | None, float | None]:
        """The linear and angular step allowed this tick, ``remaining_m`` from the target."""

        linear, angular = self.linear_m, self.angular_rad
        if self.slow_linear_m is not None and remaining_m <= self.slow_within_m + 1e-12:
            linear = self.slow_linear_m if linear is None else min(linear, self.slow_linear_m)
            angular = (self.slow_angular_rad if angular is None
                       else min(angular, self.slow_angular_rad or angular))
        return linear, angular

    def apply(self, translation: Sequence[float], rotation: Sequence[float],
              remaining_m: float) -> tuple[np.ndarray, np.ndarray]:
        """A commanded translation and rotation vector scaled down to this tick's caps."""

        linear, angular = self.limits(remaining_m)
        result = []
        for vector, cap in ((np.asarray(translation, dtype=float), linear),
                            (np.asarray(rotation, dtype=float), angular)):
            norm = float(np.linalg.norm(vector))
            result.append(vector * (cap / norm) if cap is not None and norm > cap else vector)
        return result[0], result[1]

    def min_ticks(self, distance_m: float, angle_rad: float) -> int:
        """The fewest ticks these caps allow for a move of this size (a lower bound for its loop budget)."""

        ticks = 0.0
        if self.linear_m is not None:
            ticks = max(distance_m / self.linear_m, angle_rad / (self.angular_rad or math.inf))
        if self.slow_linear_m is not None:
            ticks += min(distance_m, self.slow_within_m) / self.slow_linear_m
        return int(math.ceil(ticks - 1e-9))


def step_caps(request: Mapping[str, Any] | None, *, tick_s: float) -> StepCaps | None:
    """The per-tick caps of a speed request (wire form) at a tick of ``tick_s``; None when nothing is capped.

    The default (fast, no slow zone) caps nothing, so a local executor runs exactly as before.
    Example: step_caps({"speed": "careful", "slow_approaching": False}, tick_s=0.05).linear_m == 0.005."""

    move, slow = paces(request)
    if move.linear_m_s is None and slow is None:
        return None

    def per_tick(pace: Pace | None) -> tuple[float | None, float | None]:
        if pace is None or pace.linear_m_s is None:
            return None, None
        return pace.linear_m_s * tick_s, math.radians(pace.angular_deg_s or math.inf) * tick_s

    linear, angular = per_tick(move)
    slow_linear, slow_angular = per_tick(slow)
    return StepCaps(linear, angular, slow_linear, slow_angular, SLOW_APPROACH_M if slow is not None else 0.0)


__all__ = [
    "BUDGET_CODE", "CORNER_DEG", "DUPLICATE_ROW_RAD", "Grid", "IDLE_SEGMENT_S", "JOINT_GRID_SPACING_RAD", "LARGE_WRIST_TURN_RAD",
    "MAX_UNREQUESTED_TURN_DEG", "Pace", "PacedPath", "PaceRefusal", "PathGrid", "SLOW_PRESET", "StepCaps",
    "WRIST_UNWIND_CODE", "Zone", "angle_between", "atomic_start", "budget_refusal", "caps", "check_unwind",
    "corners", "cover_refusal", "cut", "distinct_rows", "fit_budget", "joint_path_grid",
    "knots", "pace_path", "paced_grid", "paces", "parameterize", "path_turns", "peaks", "ramp_free", "segment_lengths",
    "slow_final", "speed_report", "step_caps", "unsplit_refusal", "wrist_report", "wrist_turn_spans",
    "zone_starts",
]
