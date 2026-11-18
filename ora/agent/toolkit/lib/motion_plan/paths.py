"""Paths a move can take: a straight Cartesian line, a joint-space line, and their grids.

Pure computation. Poses are [x, y, z, qx, qy, qz, qw] of the controlled
point (link6) in the episode-local world frame, metres and xyzw; joints are
radians. A line moves the position linearly and turns the orientation evenly
(slerp). Grids are fixed by the path alone, never by the speed, so changing a
speed never changes the inverse kinematics a move needs. ``arc`` and corner
blending are later extensions (they only have to produce a pose per grid
fraction, as ``line_poses`` does).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: Grid spacing along a Cartesian line: IK is solved at most this far apart.
GRID_SPACING_M = 0.003
GRID_SPACING_RAD = math.radians(1.5)
#: Grid spacing along a joint line (largest joint).
JOINT_GRID_SPACING_RAD = 0.02


def _unit(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=float)
    return value / np.linalg.norm(value)


def angle_between(first: Sequence[float], second: Sequence[float]) -> float:
    """Pure computation: rotation angle (rad) between two xyzw orientations. Example: angle_between(q, q) == 0."""
    dot = abs(float(np.dot(_unit(first), _unit(second))))
    return 2.0 * math.acos(min(1.0, dot))


def slerp(start: Sequence[float], end: Sequence[float], fraction: float) -> list[float]:
    """Pure computation: the xyzw orientation ``fraction`` of the way from ``start`` to ``end``.

    Example: slerp([0, 0, 0, 1], [0, 0, 1, 0], 0.5)."""
    a, b = _unit(start), _unit(end)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        value = a + fraction * (b - a)
    else:
        theta = math.acos(min(1.0, dot))
        value = (math.sin((1 - fraction) * theta) * a + math.sin(fraction * theta) * b) / math.sin(theta)
    return [float(v) for v in value / np.linalg.norm(value)]


@dataclass(frozen=True)
class Line:
    """The straight line from ``start`` to ``goal`` (two 7-vector poses)."""
    start: tuple[float, ...]
    goal: tuple[float, ...]

    @property
    def length_m(self) -> float:
        return math.dist(self.start[:3], self.goal[:3])

    @property
    def angle_rad(self) -> float:
        return angle_between(self.start[3:], self.goal[3:])

    def pose(self, fraction: float) -> list[float]:
        """Pure computation: the pose at ``fraction`` in [0, 1]. Example: Line(a, b).pose(0.5)."""
        position = [self.start[i] + fraction * (self.goal[i] - self.start[i]) for i in range(3)]
        return [float(v) for v in position] + slerp(self.start[3:], self.goal[3:], fraction)


def line(start: Sequence[float], goal: Sequence[float]) -> Line:
    """Pure computation: the straight line between two poses. Example: line(measured, target)."""
    return Line(tuple(float(v) for v in start), tuple(float(v) for v in goal))


def line_grid(path: Line, *, spacing_m: float = GRID_SPACING_M, spacing_rad: float = GRID_SPACING_RAD) -> list[float]:
    """Pure computation: path fractions 0..1, at most ``spacing_m`` / ``spacing_rad`` apart.

    Example: line_grid(line(a, b))."""
    count = max(1, math.ceil(path.length_m / spacing_m - 1e-9), math.ceil(path.angle_rad / spacing_rad - 1e-9))
    return [index / count for index in range(count + 1)]


def joint_grid(start: Sequence[float], goal: Sequence[float], *, spacing_rad: float = JOINT_GRID_SPACING_RAD) -> list[float]:
    """Pure computation: fractions of a joint-space line, its largest joint at most ``spacing_rad`` apart.

    Example: joint_grid(q0, q1)."""
    largest = float(np.max(np.abs(np.asarray(goal, dtype=float) - np.asarray(start, dtype=float))))
    count = max(1, math.ceil(largest / spacing_rad - 1e-9))
    return [index / count for index in range(count + 1)]


def joint_points(start: Sequence[float], goal: Sequence[float], fractions: Sequence[float]) -> np.ndarray:
    """Pure computation: joints at each fraction of the straight joint line. Example: joint_points(q0, q1, [0, 1])."""
    first, last = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    return np.asarray([first + fraction * (last - first) for fraction in fractions])


def part_of(starts: Mapping[str, Sequence[float]], goals: Mapping[str, Sequence[float]],
            fraction: float) -> dict[str, list[float]]:
    """Pure computation: each arm's pose ``fraction`` of the way along its straight line.

    Example: part_of({'left': a}, {'left': b}, 0.5)."""
    return {arm: line(starts[arm], goal).pose(fraction) for arm, goal in goals.items()}


def precontact_fraction(path: Line, *, translation_m: float, rotation_rad: float) -> float:
    """Pure computation: where a staged contact approach hands over to its final straight segment.

    The final segment is at most ``translation_m`` and ``rotation_rad`` long;
    0.0 means the whole line already fits. Example: precontact_fraction(line(a, b), translation_m=0.04,
    rotation_rad=0.16)."""
    remaining = 1.0
    if path.length_m > translation_m:
        remaining = min(remaining, translation_m / path.length_m)
    if path.angle_rad > rotation_rad:
        remaining = min(remaining, rotation_rad / path.angle_rad)
    return max(0.0, 1.0 - remaining)


__all__ = ["GRID_SPACING_M", "GRID_SPACING_RAD", "JOINT_GRID_SPACING_RAD", "Line", "angle_between", "joint_grid",
           "joint_points", "line", "line_grid", "part_of", "precontact_fraction", "slerp"]
