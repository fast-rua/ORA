"""A pose plan that turns the wrist the long way, and the straight joint lines to the same pose that may replace it.

The pacing refuses a path that turns the hand more than ``MAX_UNREQUESTED_TURN_DEG`` beyond the turn from its start
to its end orientation (``motion_pace.check_unwind``). A planner's answer can go the long way to a pose a straight
joint line reaches directly:

* a detour (``same_goal``): trajectory optimisation swings the hand on the way to goal joints the straight line
  reaches with little turn (on RoboTwin pure translations the line turned 0 to 7 degrees where the plan turned 48 to
  156); tried for a plan the pacing would refuse;
* a goal a whole turn away (``alias``): a goal joint more than half a turn from its start whose 2 pi equivalent lies
  inside the joint's limits (``nearest_alias``; only the ALOHA's +/-4.2 rad leave such room, and its wrist is the
  joint that winds); every link stands where it stood, so the pose is the same and that joint turns the short way;
  tried for every plan;
* another IK solution of the same pose (``ik_pool``): the planner's IK keeps every converged, feasible seed of its
  attempts (``detail.ik_attempts``) but plans to the one it ranks best, whatever that does to the wrist; at the
  Panda's last-joint limit another arm branch can reach the same orientation with the wrist inside its range
  (``pool_goals``); tried for a plan the pacing would refuse, the ``POOL_LINES`` nearest the start.

``choose`` ranks the candidates as the pacing reads them, within the rule first and then the smallest path turn; the
plan itself takes part when the rule accepts it and wins a tie, so a line replaces a plan the pacing would accept only
when it turns the hand less, and a line the rule refuses is never answered. ``Worker._unwound`` measures the turns and
runs the planner's own checks; this module is pure numpy.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.environment.controllers.motion_pace import JOINT_GRID_SPACING_RAD, MAX_UNREQUESTED_TURN_DEG

TWO_PI = 2.0 * math.pi
#: How many other IK solutions of the goal pose a refused plan is compared with: those nearest the start.
POOL_LINES = 8
#: Goals closer than this on every joint (rad) are one goal.
SAME_GOAL_RAD = 1e-3


def unrequested(turns: Mapping[str, float]) -> float:
    """How far a path turns the hand beyond its start-to-goal turn, in degrees (``motion_pace.path_turns``)."""
    return float(turns["path_turn_deg"]) - float(turns["requested_turn_deg"])


def refused(turns: Mapping[str, float]) -> bool:
    """Whether the pacing's wrist check would refuse a path of these turns."""
    return unrequested(turns) > MAX_UNREQUESTED_TURN_DEG


def nearest_alias(start: Sequence[float], goal: Sequence[float],
                  limits: Sequence[Sequence[float]]) -> list[float] | None:
    """``goal`` with each joint at its 2 pi equivalent nearest ``start`` inside ``limits`` (low row, high row).

    None when no joint has a nearer equivalent in range (a Panda's or an X5's limits leave none).
    Example: nearest_alias([-2.0], [3.5], [[-4.2], [4.2]]) == [3.5 - 2 * math.pi]."""
    nearest = []
    for begin, end, lower, upper in zip(start, goal, *limits):
        value = float(end) - TWO_PI * round((float(end) - float(begin)) / TWO_PI)
        nearest.append(value if abs(value - float(end)) > 1e-9 and float(lower) <= value <= float(upper)
                       else float(end))
    return nearest if nearest != [float(value) for value in goal] else None


def pool_goals(attempts: Any, start: Sequence[float], limits: Sequence[Sequence[float]],
               known: Sequence[Sequence[float]]) -> list[list[float]]:
    """The other IK solutions of the goal pose in the planner's attempts, as goals a straight line may reach.

    ``attempts`` is ``detail.ik_attempts``: each captured attempt's seeds, ``accepted`` where converged and feasible.
    Each accepted seed inside ``limits`` is taken at its 2 pi equivalent nearest ``start``; goals within
    ``SAME_GOAL_RAD`` of ``known`` (the plan's own goal), of its equivalent or of each other count once; the
    ``POOL_LINES`` nearest ``start`` (largest joint) are answered, nearest first."""
    seen = {tuple(np.round(np.asarray(value, dtype=np.float64) / SAME_GOAL_RAD))
            for goal in known for value in (goal, nearest_alias(start, goal, limits) or goal)}
    low, high = (np.asarray(row, dtype=np.float64) for row in limits)
    found: list[list[float]] = []
    for attempt in attempts if isinstance(attempts, list) else ():
        if not isinstance(attempt, Mapping) or attempt.get("status") != "captured":
            continue
        for joints, accepted in zip(attempt.get("joint_positions") or (), attempt.get("accepted") or ()):
            values = np.asarray(joints, dtype=np.float64)
            if accepted is not True or values.shape != low.shape or not np.all((low <= values) & (values <= high)):
                continue
            goal = nearest_alias(start, values.tolist(), limits) or values.tolist()
            key = tuple(np.round(np.asarray(goal) / SAME_GOAL_RAD))
            if key not in seen:
                seen.add(key)
                found.append(goal)
    found.sort(key=lambda goal: float(np.max(np.abs(np.asarray(goal) - np.asarray(start, dtype=np.float64)))))
    return found[:POOL_LINES]


def straight_line(start: Sequence[float], goal: Sequence[float]) -> Any:
    """The straight joint line from ``start`` to ``goal``, ``JOINT_GRID_SPACING_RAD`` apart on the largest joint."""
    first, last = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    count = max(1, int(np.ceil(float(np.max(np.abs(last - first))) / JOINT_GRID_SPACING_RAD - 1e-9)))
    line = np.linspace(first, last, count + 1)
    line[-1] = last
    return line


def as_trajectory(line: Any, joint_names: Sequence[str], dt_s: float) -> dict[str, Any]:
    """A line as a planner trajectory: constant speed at the plan's own interval, at rest at both ends.

    The pacing retimes it anyway; the velocities only keep the trajectory's shape."""
    velocities = np.zeros_like(line)
    if len(line) >= 3:
        velocities[1:-1] = (line[2:] - line[:-2]) / (2.0 * float(dt_s))
    return {"joint_names": list(joint_names), "positions": line.tolist(), "velocities": velocities.tolist(),
            "dt_s": float(dt_s)}


def choose(candidates: list[dict[str, Any]], lines: Mapping[int, Any],
           blocked: Callable[[Any], dict[str, Any] | None]) -> int | None:
    """The index of the candidate to answer (None: none), each candidate's ``outcome`` written.

    ``candidates`` carry ``source`` and their turns; ``lines`` holds the joint line of each one but the plan's.
    A candidate the rule refuses ``unwinds``; in rank order a line ``blocked`` refuses is ``blocked`` (with why),
    the first left is ``chosen`` and the rest are ``ranked_lower``, never checked."""
    order = sorted(range(len(candidates)), key=lambda index: (
        refused(candidates[index]), candidates[index]["path_turn_deg"], index))
    chosen = None
    for index in order:
        candidate = candidates[index]
        if refused(candidate):
            candidate["outcome"] = "unwinds"
        elif chosen is not None:
            candidate["outcome"] = "ranked_lower"
        else:
            why = blocked(lines[index]) if index in lines else None
            if why is None:
                candidate["outcome"], chosen = "chosen", index
            else:
                candidate.update({"outcome": "blocked", "blocked": why})
    return chosen


__all__ = ["POOL_LINES", "SAME_GOAL_RAD", "TWO_PI", "as_trajectory", "choose", "nearest_alias", "pool_goals", "refused",
           "straight_line", "unrequested"]
