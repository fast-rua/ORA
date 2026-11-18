"""One arm move planned from the measured state to its targets: a joint path on a grid.

Replay: the inverse kinematics comes from recorded ``kinematics`` answers
(``kinematics.py``); everything else is pure, float-deterministic computation,
so a replayed tool rebuilds exactly the recorded action. Three path kinds:

* ``plan_line``: the straight Cartesian line of each arm (position lerp,
  orientation slerp) on a grid at most 3 mm / 1.5 degrees apart, each grid
  pose solved from the previous solution. An IK hole or branch change inside
  the line is bridged by a joint-space line between its good neighbours when
  the bridge stays within ``max_deviation_m`` of the line and within
  ``MAX_BRIDGE_TURN_DEG`` of its orientation (checked with FK); otherwise the
  move is refused at that point before anything moves. The orientation bound
  matters: past joint 6's half-turn limit the IK answer wraps by a full turn,
  and the joint line between the two sides spins the wrist about the approach
  axis, which leaves the tool point (on that axis) where it is.
* ``plan_joint``: a joint-space line to one IK solution of each target (the
  free-space route where no planner runs).
* ``plan_global``: the shared planner's collision-free path of each arm in the
  world of the current public depth (``kinematics.plan_path``). Two arms are
  planned one by one, each with the other held at its measured joints; the
  lockstep ``screen_path`` of the timed move re-checks them against each other.

Two arms share one grid. ``timing.timed`` then turns it into one joint target
per native action on one timeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ...interface.api import DataInterface
from . import checks, kinematics, paths
from .timing import Grid

#: Largest joint change admitted between two grid poses of a straight line (a branch flip is far larger).
LINE_STEP_RAD = 0.25
#: A joint-line bridge over an IK hole may turn the wrist at most this far from the line's orientation
#: (fold_clothes L-0007: 17 admitted bridges spun it 172-180 degrees within 3.4 mm of the line).
MAX_BRIDGE_TURN_DEG = 10.0
#: A free-space endpoint may be up to a half-turn away per joint, as the released global route allowed.
ENDPOINT_STEP_RAD = math.pi


@dataclass(frozen=True)
class PathPlan:
    """A joint grid plus what the receipt and the expected poses need."""
    grid: Grid
    route: str
    goals: Mapping[str, Sequence[float]]
    lines: Mapping[str, paths.Line] | None
    bridged: tuple[dict, ...] = ()
    #: Answers of the kinematics query that carried a warning (``kinematics.warnings``).
    warnings: tuple[dict, ...] = ()


def _screen(arms: Sequence[str]) -> str:
    return "self_and_measured_other_arm" if len(arms) == 1 else "self"


def plan_line(api: DataInterface, *, starts: Mapping[str, Sequence[float]], goals: Mapping[str, Sequence[float]],
              start_joints: Mapping[str, Sequence[float]], max_deviation_m: float | None = 0.005) -> PathPlan:
    """Replay: recorded kinematics answers. The straight line of every arm as a continuous joint grid.

    Example: plan_line(api, starts={'left': p0}, goals={'left': p1}, start_joints={'left': q0})."""
    arms = tuple(sorted(goals))
    lines = {arm: paths.line(starts[arm], goals[arm]) for arm in arms}
    count = max(len(paths.line_grid(path)) - 1 for path in lines.values())
    fractions = [index / count for index in range(count + 1)]
    joints, bridged, warned = {}, [], []
    for arm in arms:
        poses = [lines[arm].pose(fraction) for fraction in fractions]
        chain = kinematics.ik_chain(api, arm=arm, poses=poses[1:], seed=start_joints[arm],
                                    max_joint_step_rad=LINE_STEP_RAD, ranking="nearest", screen=_screen(arms))
        warned.extend(kinematics.warnings(chain))
        rows = chain["rows"]
        nodes = np.asarray([list(start_joints[arm])] + [row["joints_rad"] or [math.nan] * 6 for row in rows],
                           dtype=float)
        for first, last in checks.holes(rows):
            if last == len(rows) - 1:
                raise checks.refusal_for(rows[first], arm=arm, fraction=fractions[first + 1], pose=poses[first + 1])
            nodes = checks.bridge(nodes, first + 1, last + 1)
            span = list(range(first + 1, last + 2))
            actual = kinematics.fk(api, arm=arm, joints=nodes[span].tolist())
            off = checks.deviation([poses[node] for node in span], actual)
            turn = checks.turn_deviation_deg([poses[node] for node in span], actual)
            if max_deviation_m is not None and off > max_deviation_m:
                refusal = checks.refusal_for(rows[first], arm=arm, fraction=fractions[first + 1],
                                             pose=poses[first + 1])
                refusal.constraints.update(joint_line_bridge_deviation_m=round(off, 4),
                                           max_deviation_m=max_deviation_m)
                raise refusal
            if max_deviation_m is not None and turn > MAX_BRIDGE_TURN_DEG:
                raise checks.PlanRefusal(
                    "the straight line needs the wrist to unwind (it would spin about the approach axis between "
                    "two IK branches, typically at joint 6's half-turn limit); nothing was executed. Reach the "
                    "goal by another route, e.g. a free-space move first, or with empty jaws the other jaw "
                    "orientation", code="paced_path_wrist_unwind",
                    constraints={"arm": arm, "fraction_of_move": round(float(fractions[first + 1]), 4),
                                 "pose_xyz_xyzw": [round(float(v), 5) for v in poses[first + 1]],
                                 "reason": rows[first]["status"], "joint_line_bridge_turn_deg": round(turn, 1),
                                 "max_bridge_turn_deg": MAX_BRIDGE_TURN_DEG})
            bridged.append({"arm": arm, "from_fraction": round(fractions[first], 4),
                            "to_fraction": round(fractions[last + 2], 4), "deviation_m": round(off, 5),
                            "turn_deg": round(turn, 2), "reason": rows[first]["status"]})
        joints[arm] = nodes
    return PathPlan(Grid(tuple(fractions), joints), "line", dict(goals), lines, tuple(bridged), tuple(warned))


def _endpoints(api: DataInterface, goals: Mapping[str, Sequence[float]],
               start_joints: Mapping[str, Sequence[float]]) -> tuple[dict[str, list[float]], list[dict]]:
    arms = tuple(sorted(goals))
    result, warned = {}, []
    for arm in arms:
        chain = kinematics.ik_chain(api, arm=arm, poses=[goals[arm]], seed=start_joints[arm],
                                    max_joint_step_rad=ENDPOINT_STEP_RAD, ranking="canonical_first",
                                    screen=_screen(arms))
        warned.extend(kinematics.warnings(chain))
        row = chain["rows"][0]
        if row["status"] != "ok":
            raise checks.refusal_for(row, arm=arm, fraction=1.0, pose=goals[arm])
        result[arm] = row["joints_rad"]
    return result, warned


def endpoint_joints(api: DataInterface, *, goals: Mapping[str, Sequence[float]],
                    start_joints: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Replay: recorded kinematics answers. One canonical-first IK solution per target (the free-space route).

    Example: endpoint_joints(api, goals={'left': p1}, start_joints={'left': q0})."""
    return _endpoints(api, goals, start_joints)[0]


def plan_joint(api: DataInterface, *, goals: Mapping[str, Sequence[float]],
               start_joints: Mapping[str, Sequence[float]],
               goal_joints: Mapping[str, Sequence[float]] | None = None) -> PathPlan:
    """Replay: recorded kinematics answers. A joint-space line per arm to its target's IK solution.

    Example: plan_joint(api, goals={'left': p1}, start_joints={'left': q0})."""
    arms = tuple(sorted(goals))
    targets, warned = (dict(goal_joints), []) if goal_joints is not None else _endpoints(api, goals, start_joints)
    count = max(len(paths.joint_grid(start_joints[arm], targets[arm])) - 1 for arm in arms)
    fractions = [index / count for index in range(count + 1)]
    joints = {arm: paths.joint_points(start_joints[arm], targets[arm], fractions) for arm in arms}
    return PathPlan(Grid(tuple(fractions), joints), "joint", dict(goals), None, (), tuple(warned))


class PlannerUnavailable(RuntimeError):
    """No planner answered for this move (none here, a stale observation, an arm past the limits it plans within):
    the caller keeps its joint route."""


def plan_global(api: DataInterface, *, goals: Mapping[str, Sequence[float]]) -> PathPlan:
    """Replay: recorded kinematics answers. Each arm's planned path from its measured joints to its goal.

    Raises ``checks.PlanRefusal`` with the planner's neutral reason, or ``PlannerUnavailable``.
    Example: plan_global(api, goals={'left': p1})."""
    arms = tuple(sorted(goals))
    found = {}
    for arm in arms:
        answer = kinematics.plan_path(api, arm=arm, goal_pose=goals[arm])
        # An arm wound past the planner's limits says nothing about the goal, and the joint route brings it back.
        if answer["status"] == "unavailable" or answer.get("reason") == "start_outside_limits":
            raise PlannerUnavailable(str(answer.get("reason")))
        if answer["status"] != "planned":
            raise checks.planner_refusal(answer, arm=arm)
        found[arm] = np.vstack([np.asarray(answer["start_joints"], dtype=float),
                                np.asarray(answer["path_joints"], dtype=float)[1:]])
    count = max(len(rows) - 1 for rows in found.values())
    fractions = [index / count for index in range(count + 1)]
    joints = {arm: np.column_stack([np.interp(fractions, np.linspace(0.0, 1.0, len(rows)), rows[:, joint])
                                    for joint in range(rows.shape[1])]) for arm, rows in found.items()}
    return PathPlan(Grid(tuple(fractions), joints), "global", dict(goals), None)


def expected_poses(api: DataInterface, plan: PathPlan, knot_joints: Mapping[str, Sequence[Sequence[float]]],
                   fractions: Sequence[float]) -> dict[str, list[list[float]]]:
    """Replay: recorded kinematics answers. The link6 pose each knot should reach, by FK of the timed rows.

    Example: expected_poses(api, plan, joints, f)."""
    return {arm: kinematics.fk(api, arm=arm, joints=list(rows)) for arm, rows in knot_joints.items()}


__all__ = ["ENDPOINT_STEP_RAD", "LINE_STEP_RAD", "PathPlan", "PlannerUnavailable", "endpoint_joints", "expected_poses",
           "plan_global", "plan_joint", "plan_line"]
