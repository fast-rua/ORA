"""Checks of a planned joint path: IK holes, continuity, Cartesian deviation and the budget.

Pure computation over kinematics rows the caller obtained. Every refusal
names where along the path it happened (``PlanRefusal.code`` and
``constraints``) and is raised before anything moves; a tool may relax any
bound by passing its own. The budget is the timing's (``timing.timed``).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class PlanRefusal(ValueError):
    """The move was refused before any native action (``code``, ``constraints``)."""

    def __init__(self, message: str, *, code: str, constraints: dict | None = None):
        super().__init__(message)
        self.code, self.constraints = code, dict(constraints or {})


def holes(rows: Sequence[dict]) -> list[tuple[int, int]]:
    """Pure computation: maximal runs (first, last row index) of rows that did not solve.

    Example: holes([{'status': 'ok'}, {'status': 'ik_failed'}]) == [(1, 1)]."""
    runs, first = [], None
    for index, row in enumerate(rows):
        if row["status"] != "ok" and first is None:
            first = index
        elif row["status"] == "ok" and first is not None:
            runs.append((first, index - 1))
            first = None
    if first is not None:
        runs.append((first, len(rows) - 1))
    return runs


def refusal_for(row: dict, *, arm: str, fraction: float, pose: Sequence[float]) -> PlanRefusal:
    """Pure computation: the refusal for a path point the IK chain could not continue through.

    Example: refusal_for(row, arm='left', fraction=0.5, pose=pose)."""
    jump = row["status"] == "discontinuous"
    return PlanRefusal(
        "the IK solution jumps along the path (a branch change); nothing was executed" if jump
        else "a point of the path has no admissible IK solution; nothing was executed",
        code="paced_path_joint_jump" if jump else "paced_path_unreachable",
        constraints={"arm": arm, "fraction_of_move": round(float(fraction), 4),
                     "pose_xyz_xyzw": [round(float(v), 5) for v in pose], "reason": row["status"],
                     **({"joint_step_rad": round(float(row["joint_step_rad"]), 4)}
                        if row.get("joint_step_rad") is not None else {})})


def bridge(nodes: np.ndarray, first: int, last: int) -> np.ndarray:
    """Pure computation: replace nodes ``first..last`` by the straight joint line between their neighbours.

    Example: bridge(nodes, 3, 5)."""
    result = np.array(nodes, dtype=float)
    left, right = result[first - 1], result[last + 1]
    span = last - first + 2
    for offset, node in enumerate(range(first, last + 1), start=1):
        result[node] = left + offset / span * (right - left)
    return result


def deviation(expected: Sequence[Sequence[float]], actual: Sequence[Sequence[float]]) -> float:
    """Pure computation: largest position distance between two pose lists. Example: deviation([a], [a]) == 0."""
    return max((math.dist(a[:3], b[:3]) for a, b in zip(expected, actual)), default=0.0)


def turn_deviation_deg(expected: Sequence[Sequence[float]], actual: Sequence[Sequence[float]]) -> float:
    """Pure computation: largest rotation (degrees) between two pose lists' xyzw orientations.

    Example: turn_deviation_deg([a], [a]) == 0."""
    from .orientation import turn_deg
    return max((turn_deg(a[3:7], b[3:7]) for a, b in zip(expected, actual)), default=0.0)


def joint_steps_ok(values: np.ndarray, limit: float) -> bool:
    """Pure computation: every consecutive joint step within ``limit``. Example: joint_steps_ok(nodes, 0.2)."""
    return bool(np.all(np.max(np.abs(np.diff(values, axis=0)), axis=1) <= limit + 1e-12)) if len(values) > 1 else True


#: A blocked ``screen_path`` status and the refusal message for it.
_SCREEN_REFUSALS = {
    "self_collision": "the arm's SDK collision spheres meet each other along the planned path",
    "table_collision": "the arm's SDK collision spheres go below the table top along the planned path",
    "inter_arm_collision": "the two arms' SDK collision spheres come within the inter-arm margin along the "
                           "planned path; move one arm at a time or choose targets that keep the arms apart",
}


#: A planned path's refusal sentence by the planner's neutral reason.
PLANNER_REFUSALS = {
    "goal_unreachable": "no joint configuration reaches the goal pose in the collision world",
    "goal_in_collision": "the goal joints collide with the world, the other arm or the arm itself",
    "goal_below_support": "the goal puts a fingertip below the supporting surface",
    "start_in_collision": "the measured joints already touch the world or the arm itself",
    "start_outside_limits": "the measured joints are outside the joint limits",
    "path_blocked": "no collision-free path to the goal was found",
}


def planner_refusal(answer: dict, *, arm: str) -> PlanRefusal:
    """Pure computation: the refusal of a ``kinematics.plan_path`` answer; its code is the planner's neutral reason.

    A goal the world refused below its support adds that refusal's heights in the world frame of the policy's poses
    (``frame: world``, ``support_z_m`` and the fingertip's or load's) with how far below and the tolerance.
    Example: planner_refusal({'reason': 'path_blocked'}, arm='left').code == 'path_blocked'."""
    reason = str(answer.get("reason"))
    goal = answer.get("goal_refusal") or {}
    message = str(goal.get("message") or PLANNER_REFUSALS.get(reason, "the planner refused the goal"))
    world = goal.get("in_world") if isinstance(goal.get("in_world"), dict) else {}
    heights = {**world, **{key: round(float(goal[key]), 4) for key in ("depth_m", "tolerance_m") if key in goal}} if world else {}
    return PlanRefusal(f"{message}; nothing was executed", code=reason, constraints={"arm": arm, **heights})


def screened(result: dict, fractions: Sequence[float]) -> None:
    """Pure computation: refuse a planned path whose ``kinematics.screen_path`` answer is not clear.

    ``fractions`` are the knots' fractions of the move (segment ``k`` ends on knot ``k``). The refusal code is
    the screen status (``self_collision``, ``table_collision``, ``inter_arm_collision``; ``path_blocked``
    otherwise). Example: screened({'status': 'clear'}, [1.0]) is None."""
    status = result.get("status")
    if status == "clear":
        return None
    segment = int(result.get("first_blocked_segment") or 1)
    fraction = float(fractions[min(max(segment, 1), len(fractions)) - 1]) if fractions else 1.0
    evidence = {key: result[key] for key in ("arm", "min_inter_arm_clearance_m", "limiting_links",
                                             "min_table_clearance_m", "limiting_link")
                if result.get(key) is not None}
    raise PlanRefusal(f"{_SCREEN_REFUSALS.get(status, 'the planned path is blocked')}; nothing was executed",
                      code=status if status in _SCREEN_REFUSALS else "path_blocked",
                      constraints={"fraction_of_move": round(fraction, 4), "native_action": segment, **evidence})


#: The ARX X5 wrist roll (joint 6) limit (RoboDojo x5_v2/X5A.urdf); joints 1-5 allow +/-10 rad.
JOINT6_LIMIT_RAD = 3.14


def joint6_extremes(start: Sequence[float], knot_joints: Sequence[Sequence[float]]) -> dict:
    """Pure computation: joint 6 (wrist roll) along a planned path, in degrees.

    From the start joints through every knot: start, end, min and max, ``travel_deg`` (the sum of its
    turns, which both ends alone can hide) and ``min_limit_margin_deg`` to ``JOINT6_LIMIT_RAD``.
    Example: joint6_extremes(q0, knots['joints']['left'])['max_deg']."""
    values = np.asarray([float(start[5]), *(float(row[5]) for row in knot_joints)])
    return {"start_deg": round(math.degrees(values[0]), 2), "end_deg": round(math.degrees(values[-1]), 2),
            "min_deg": round(math.degrees(values.min()), 2), "max_deg": round(math.degrees(values.max()), 2),
            "travel_deg": round(math.degrees(float(np.sum(np.abs(np.diff(values))))), 2),
            "min_limit_margin_deg": round(math.degrees(JOINT6_LIMIT_RAD - float(np.max(np.abs(values)))), 2)}


__all__ = ["JOINT6_LIMIT_RAD", "PLANNER_REFUSALS", "PlanRefusal", "bridge", "deviation", "holes", "joint6_extremes",
           "joint_steps_ok", "planner_refusal", "refusal_for", "screened", "turn_deviation_deg"]
