"""Time parameterization: what a move's path and speed are, and the call that times them.

A move is a joint path on a grid and a ``Pace``. The environment times it with
the shared motion base (``kinematics.pace_path``): every speed and acceleration
cap at once at the controlled point and each joint, the jerk it can follow,
the wrist check, and one joint target per native action (40 ms). ``timed`` asks
once per arm and per stretch between forced stops, puts the arms on one
timeline (a shorter move is slowed uniformly, which keeps every cap) and joins
the stretches. Only a cut is ``budget``'s: the longest first part that fits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: One native action: 10 physics steps of 4 ms.
NATIVE_ACTION_S = 0.04
#: The X5 joint velocity limit and the step the environment admits per native action.
JOINT_SPEED_LIMIT_RAD_S = 5.0
MAX_KNOT_STEP_RAD = JOINT_SPEED_LIMIT_RAD_S * NATIVE_ACTION_S
#: A move's slow final segment, when it asks for one.
SLOW_APPROACH_M = 0.02
#: No budget: a whole path's timing.
UNBOUNDED_NATIVE_ACTIONS = 1_000_000
CAPS = ("linear_m_s", "angular_deg_s", "linear_accel_m_s2", "angular_accel_deg_s2", "joint_speed_rad_s",
        "joint_accel_rad_s2")


@dataclass(frozen=True)
class Pace:
    """Speed and acceleration caps of one move; ``None`` leaves a Cartesian cap off (joint caps still hold).

    ``joint_jerk_rad_s3`` is the jerk the environment's timing certifies the move at, every joint; ``None`` (every
    preset) leaves it the robot's. Only a robot motion profile's own pace names one (``tools/move_pose/_profile``).
    """
    linear_m_s: float | None
    angular_deg_s: float | None
    linear_accel_m_s2: float | None
    angular_accel_deg_s2: float | None
    joint_speed_rad_s: float
    joint_accel_rad_s2: float
    joint_jerk_rad_s3: float | None = None

    def __post_init__(self):
        for value in (self.linear_m_s, self.angular_deg_s, self.linear_accel_m_s2, self.angular_accel_deg_s2):
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError("pace caps must be positive or None")
        if not 0 < self.joint_speed_rad_s < JOINT_SPEED_LIMIT_RAD_S or not self.joint_accel_rad_s2 > 0:
            raise ValueError("joint speed must lie in (0, 5) rad/s and joint acceleration be positive")
        if self.joint_jerk_rad_s3 is not None and not (math.isfinite(self.joint_jerk_rad_s3)
                                                       and self.joint_jerk_rad_s3 > 0):
            raise ValueError("joint jerk must be positive or None")

    def caps(self) -> dict:
        """Pure computation: the caps as the timing request names them (the jerk only when the pace names one).

        Example: pace.caps()['linear_m_s']."""
        caps = {name: getattr(self, name) for name in CAPS}
        if self.joint_jerk_rad_s3 is not None:
            caps["joint_jerk_rad_s3"] = self.joint_jerk_rad_s3
        return caps

    def record(self) -> dict:
        """Pure computation: the pace as finite JSON, the jerk only when it names one (a pace without one keeps the
        record, and so the kept-plan key, every earlier plan had). Example: Pace(**pace.record()) == pace."""
        value = {name: getattr(self, name) for name in CAPS}
        if self.joint_jerk_rad_s3 is not None:
            value["joint_jerk_rad_s3"] = self.joint_jerk_rad_s3
        return value


@dataclass(frozen=True)
class Grid:
    """A path on grid nodes: fractions, the joints of each moving arm and forced stops."""
    fractions: tuple[float, ...]
    joints: Mapping[str, np.ndarray]
    stops: tuple[int, ...] = ()

    @property
    def segments(self) -> int:
        return len(self.fractions) - 1


@dataclass(frozen=True)
class Timed:
    """A timed path (or its first part): one joint row per arm per native action."""
    joints: Mapping[str, list]
    fractions: tuple[float, ...]
    duration_s: float
    native_actions: int
    #: The native actions the whole path needs, and the part of it these rows cover.
    needed_native_actions: int
    fraction: float
    #: Per arm: the knot (1-based) its slow final segment starts on, and that knot's fraction of the move.
    slow_from: Mapping[str, dict]
    #: The largest ``planned_peak_*`` over the arms.
    peaks: Mapping[str, float]


def concatenate(first: Grid, second: Grid) -> Grid:
    """Pure computation: two grids joined with a stop where they meet (``second`` starts where ``first`` ends).

    Example: concatenate(joint_part, line_part)."""
    total = first.segments + second.segments
    fractions = tuple(index / total for index in range(total + 1))
    joints = {arm: np.vstack([first.joints[arm], second.joints[arm][1:]]) for arm in first.joints}
    return Grid(fractions, joints, tuple(first.stops) + (first.segments,)
                + tuple(first.segments + stop for stop in second.stops))


def _resampled(start: Sequence[float], rows: Sequence[Sequence[float]], count: int) -> list[list[float]]:
    """``rows`` (the positions at ticks 1..n after ``start``) as the same motion over ``count`` ticks."""
    path = np.vstack([np.asarray(start, dtype=float), np.asarray(rows, dtype=float).reshape(-1, len(start))])
    if len(path) - 1 == count:
        return path[1:].tolist()
    ticks = np.arange(len(path), dtype=float)
    at = np.linspace(0.0, len(path) - 1, count + 1)[1:]
    return np.column_stack([np.interp(at, ticks, path[:, joint]) for joint in range(path.shape[1])]).tolist()


def timed(api, grid: Grid, pace: Pace, *, slow_pace: Pace | None = None,
          budget: int = UNBOUNDED_NATIVE_ACTIONS, unwind_check: bool = True) -> Timed:
    """Replay: recorded kinematics answers. ``grid`` timed under ``pace``, its first part when over ``budget``.

    ``slow_pace`` also caps the last ``SLOW_APPROACH_M`` of each arm's path, entered without a stop.
    ``unwind_check`` false leaves a wrist that turns the long way round alone (a recovery's own unwind).
    Raises ``checks.PlanRefusal`` (``paced_path_wrist_unwind``, ``paced_motion_budget``) before anything moves.
    Example: timed(api, plan.grid, pace, budget=50)."""
    from . import checks, kinematics

    nodes = [0, *grid.stops, grid.segments]
    spans = list(zip(nodes, nodes[1:]))
    joints: dict[str, list] = {arm: [] for arm in grid.joints}
    fractions: list[float] = []
    slow_from: dict[str, dict] = {}
    peaks: dict[str, float] = {}
    left, duration, needed, ticks_so_far = int(budget), 0.0, 0, 0
    reached = 0.0
    for index, (first, last) in enumerate(spans):
        if left < 1:
            break
        answers = {}
        final = index == len(spans) - 1
        for arm, values in grid.joints.items():
            part = np.asarray(values[first:last + 1], dtype=float)
            if float(np.max(np.abs(part - part[0]))) <= 1e-9:
                continue
            answer = kinematics.pace_path(api, arm=arm, joints=part.tolist(), pace=pace,
                                          slow_pace=slow_pace if final else None, budget_ticks=left,
                                          cut_to_budget=True, unwind_check=unwind_check)
            if answer["status"] != "timed":
                raise checks.PlanRefusal(str(answer.get("message") or "the path could not be timed"),
                                         code=str(answer["reason"]), constraints=dict(answer.get("constraints") or {}))
            answers[arm] = answer
        count = max((int(answer["ticks"]) for answer in answers.values()), default=1)
        low, high = float(grid.fractions[first]), float(grid.fractions[last])
        for arm, values in grid.joints.items():
            answer = answers.get(arm)
            start = np.asarray(values[first], dtype=float).tolist()
            joints[arm].extend(_resampled(start, answer["path_joints"], count) if answer is not None
                               else [start] * count)
            if answer is not None and answer.get("slow_from_tick") is not None:
                tick = int(answer["slow_from_tick"])
                knot = max(1, math.ceil(tick * count / int(answer["ticks"])))
                at = float(answer["fractions"][min(tick, len(answer["fractions"])) - 1])
                slow_from[arm] = {"knot": ticks_so_far + knot,
                                  "fraction_of_move": round(low + at * (high - low), 4),
                                  "path_left_m": SLOW_APPROACH_M}
            for key, value in ((answer or {}).get("speed") or {}).items():
                if key.startswith("planned_peak_"):
                    peaks[key] = max(peaks.get(key, 0.0), float(value))
        cover = min((float(answer["fraction_of_move"]) for answer in answers.values()), default=1.0)
        slowest = max(answers.values(), key=lambda answer: int(answer["ticks"]), default=None)
        fractions.extend(low + (high - low) * float(value)
                         for value in ([1.0] if slowest is None else slowest["fractions"]))
        needed += max((int(answer["needed_ticks"]) for answer in answers.values()), default=1)
        duration += count * NATIVE_ACTION_S
        ticks_so_far += count
        left -= count
        reached = low + cover * (high - low)
        if cover < 1.0:
            break
    else:
        reached = 1.0
    needed = max(needed, ticks_so_far + (1 if reached < 1.0 else 0))
    return Timed({arm: rows for arm, rows in joints.items()}, tuple(fractions), round(ticks_so_far * NATIVE_ACTION_S, 6),
                 ticks_so_far, needed, reached if reached < 1.0 else 1.0, slow_from, peaks)


__all__ = ["CAPS", "Grid", "JOINT_SPEED_LIMIT_RAD_S", "MAX_KNOT_STEP_RAD", "NATIVE_ACTION_S", "Pace", "SLOW_APPROACH_M",
           "Timed", "UNBOUNDED_NATIVE_ACTIONS", "concatenate", "timed"]
