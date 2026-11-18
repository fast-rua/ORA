"""Recovery moves planned in lib: back along recorded moves' measured paths, or home by a screened unwind.

The plans of recover_pose and return_home (the RoboDojo recovery executor
stays for the released surfaces' recover_motion), each executed as one
``follow_joint_trajectory`` marked ``recovery`` (so it keeps the previous
anchor):

* **measured retrace** (recover_pose): the environment keeps an undo stack per
  arm (``exec.recovery.<arm>.undo``: the last five arm-moving tool calls, each
  with its start joints and pose and ``measured_path_rad``, the joints
  measured at its start and at the end of every native action of its
  trajectories). While the arm stands on the top entry's path (within
  ``RETRACE_TOLERANCE_RAD`` of it: after a stall, a gripper command, or a
  retrace cut to its budget), the plan is that path from where the arm is back
  to the call's start (``retrace_nodes``): where the arm actually was, not
  where it was commanded. A retrace that runs every knot pops the entry, so
  the next one undoes the call before it.
* **home** (return_home): where a shared planner runs, its collision-free path to
  the episode reset joints (``global_option``). Otherwise, or when it finds
  none, link6 lifts straight up by up to ``LIFT_M``,
  orientation held (``lift_goal``; a Cartesian line, skipped at the workspace
  ceiling or without a continuous IK solution), then the joints unwind to the
  episode reset joints in the first order the screen keeps clear
  (``unwind_orders``: the straight joint line, fold-then-yaw, yaw-then-fold);
  without the lift when only that is clear. Joint targets are the stored
  joints' whole-turn equivalents nearest the measured joints
  (``unwind_target``).

Every candidate is timed (``timing``) and screened as it would run (self,
table top, both arms in lockstep; ``kinematics.screen_path``) before anything
moves; the first clear one runs as one trajectory that stops only where its
path turns a corner (lift to unwind, fold to yaw). None clear refuses with the
code of the screen that blocked (``REFUSAL_CODES``). Over the budget the
first part runs and is reported as cut short (``timing.timed``), never
as an arrival.

Replay: the IK, FK and screens are recorded ``kinematics`` answers and the
one trajectory is the recorded primitive; the rest is float-deterministic
computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ...interface.api import DataInterface
from . import checks, execute, kinematics, paths, planner, timing

RETRACE = "measured_retrace"
JOINT = "joint_interpolation"
PLANNED = "planned_path"
METHODS = (RETRACE, JOINT, PLANNED)
#: The arm may stand this far (largest joint) from the recorded path and still retrace it.
RETRACE_TOLERANCE_RAD = 0.05
#: Recorded nodes closer than this to the last kept one are dropped (a stall records many).
NODE_STEP_RAD = 0.005
#: The home lift: link6 straight up by up to this much, never above the workspace ceiling.
LIFT_M = 0.10
MINIMUM_LIFT_M = 0.01
#: A recovery has arrived when every joint is this close to its target.
JOINT_ARRIVAL_RAD = 0.02
#: ARX X5 joint limits (RoboDojo third_party/curobo/robot/x5_v2/X5A.urdf) and its pitch joints (joint2-5).
X5_JOINT_LIMITS_RAD = ((-10.0, 10.0),) * 5 + ((-3.14, 3.14),)
X5_PITCH_JOINTS = (1, 2, 3, 4)
#: A pitch joint already outside [-pi, pi] may stay out by this much more (the IK rule's hold tolerance).
PITCH_HOLD_TOLERANCE_RAD = 1e-3
ORDERS = ("straight_joint_line", "fold_then_yaw", "yaw_then_fold")
#: A blocked screen status and the recovery refusal code it gives (the first two are the released codes).
REFUSAL_CODES = {"self_collision": "recovery_self_collision_path",
                 "inter_arm_collision": "recovery_inter_arm_collision_path",
                 "table_collision": "recovery_table_collision_path"}
PROGRESS_SCHEMA = "ora.toolkit-recovery-progress.v1"


@dataclass(frozen=True)
class Option:
    """One arm's candidate path: phases of joint nodes, each starting where the one before ended.

    ``method`` is ``measured_retrace`` or ``joint_interpolation``; ``detail`` is what the result reports
    about it (retrace, unwind order, lift)."""
    method: str
    phases: tuple[np.ndarray, ...]
    detail: Mapping = field(default_factory=dict)


def global_option(api: DataInterface, *, arm: str, target: Sequence[float], detail: Mapping | None = None) -> Option | None:
    """Replay: recorded kinematics answers. The shared planner's path to joints ``target``, or None without one.

    Example: global_option(api, arm='left', target=q_home)."""
    answer = kinematics.plan_path(api, arm=arm, goal_joints=target)
    if answer["status"] != "planned":
        return None
    rows = np.vstack([np.asarray(answer["start_joints"], dtype=float),
                      np.asarray(answer["path_joints"], dtype=float)[1:]])
    return Option(PLANNED, (rows,), {**dict(detail or {}), "planned_path": True})


@dataclass(frozen=True)
class Planned:
    """The chosen candidate, timed, fitted to the budget and screened, ready for ``run``."""
    options: Mapping[str, Option]
    grid: timing.Grid
    knots: Mapping
    fraction: float
    whole_native_actions: int
    expected: Mapping[str, list]
    settle: Mapping
    goals: Mapping[str, list]
    targets: Mapping[str, list]
    native_action_budget: int
    pace: timing.Pace
    screens: tuple[dict, ...] = ()


def joint_distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Pure computation: the largest joint difference. Example: joint_distance(q, q) == 0."""
    return float(np.max(np.abs(np.asarray(first, dtype=float) - np.asarray(second, dtype=float))))


def _turns(value: float, reference: float, lower: float, upper: float) -> int:
    if not all(math.isfinite(x) for x in (value, reference, lower, upper)) or not lower <= value <= upper:
        return 0
    first, last = math.ceil((lower - value) / math.tau), math.floor((upper - value) / math.tau)
    turns = min(max(round((reference - value) / math.tau), first), last)
    proposed = value + turns * math.tau
    nearer = abs(proposed - reference) < abs(value - reference)
    return turns if turns and nearer and lower <= proposed <= upper else 0


def _pitch_excess(value: float) -> float:
    return max(0.0, abs(float(value)) - math.pi)


def unwind_target(stored: Sequence[float], measured: Sequence[float], *,
                  limits: Sequence[Sequence[float]] = X5_JOINT_LIMITS_RAD,
                  pitch: Sequence[int] = X5_PITCH_JOINTS) -> dict:
    """Pure computation: the stored joints' whole-turn equivalent nearest the measured joints.

    Each joint turns by whole turns only when that stays inside its limits and is strictly nearer the
    measured joint; a turn that would carry a pitch joint farther outside [-pi, pi] than it is measured
    is dropped (``pitch_vetoed_turns``). A whole turn of a revolute joint leaves link6 where it was.
    Example: unwind_target([3.0, 0, 0, 0, 0, 0], [-3.0, 0, 0, 0, 0, 0])['equivalent_turns'][0] == -1."""
    stored = [float(v) for v in stored]
    turns = [_turns(value, float(near), float(bound[0]), float(bound[1]))
             for value, near, bound in zip(stored, measured, limits)]
    vetoed = [index for index in pitch if turns[index] and _pitch_excess(stored[index] + turns[index] * math.tau)
              > (_pitch_excess(measured[index]) + PITCH_HOLD_TOLERANCE_RAD if _pitch_excess(measured[index]) > 0
                 else 0.0)]
    kept = [0 if index in vetoed else count for index, count in enumerate(turns)]
    target = [value + count * math.tau for value, count in zip(stored, kept)]
    return {"status": "unwound" if any(kept) else "not_needed", "stored_joints": stored, "joint_target": target,
            "equivalent_turns": kept, "correction_rad": [count * math.tau for count in kept],
            **({"pitch_vetoed_turns": [turns[i] if i in vetoed else 0 for i in range(len(turns))]}
               if vetoed else {})}


def unwind_orders(start: Sequence[float], target: Sequence[float]) -> list[tuple[str, list[list[float]]]]:
    """Pure computation: candidate joint-space orders to ``target`` (waypoints after ``start``), straight first.

    fold_then_yaw moves joints 2-6 first, then the base yaw; yaw_then_fold the other way round.
    Example: unwind_orders(q0, q1)[0] == ('straight_joint_line', [q1])."""
    goal = [float(v) for v in target]
    fold = [float(start[0]), *goal[1:]]
    yaw = [goal[0], *(float(v) for v in start[1:])]
    return [("straight_joint_line", [goal]), ("fold_then_yaw", [fold, goal]), ("yaw_then_fold", [yaw, goal])]


def joint_options(start: Sequence[float], target: Sequence[float], *, before: Sequence[np.ndarray] = (),
                  detail: Mapping | None = None) -> list[Option]:
    """Pure computation: one ``joint_interpolation`` option per distinct unwind order.

    ``before`` are phases the option runs first (the home lift; the orders then start where it ends).
    Degenerate segments are dropped, and an order identical to an earlier one is not repeated.
    Example: joint_options(q0, q1)[0].detail['unwind_order'] == 'straight_joint_line'."""
    first = np.asarray(before[-1][-1] if before else start, dtype=float)
    options, seen = [], []
    for name, waypoints in unwind_orders(first, target):
        phases, cursor = [], first
        for point in waypoints:
            point = np.asarray(point, dtype=float)
            if joint_distance(cursor, point) > 1e-9:
                phases.append(np.vstack([cursor, point]))
                cursor = point
        if not phases:
            phases = [np.vstack([first, first])]
        key = [phase.tolist() for phase in phases]
        if key in seen:
            continue
        seen.append(key)
        options.append(Option(JOINT, (*(np.asarray(p, dtype=float) for p in before), *phases),
                              {**dict(detail or {}), "unwind_order": name}))
    return options


def retrace_nodes(anchor: Sequence[float], path: Sequence[Sequence[float]] | None, current: Sequence[float], *,
                  tolerance: float = RETRACE_TOLERANCE_RAD) -> tuple[np.ndarray | None, dict]:
    """Pure computation: the recorded path from where the arm stands back to the anchor, or why there is none.

    ``path`` is the anchor's ``measured_path_rad``. The arm's place on it is the nearest point of the
    polyline (a later segment wins a tie); the nodes are the current joints, then the recorded joints from
    that segment's start back to the first row, thinned to ``NODE_STEP_RAD``. ``status``: ``retraced``,
    ``not_recorded`` (no path: an old run or a move that records none), ``record_of_another_move`` (its
    first row is not the anchor) or ``arm_moved_since`` (the arm is farther than ``tolerance`` from it).
    Example: retrace_nodes(q0, [q0, q1], q1)[1]['status'] == 'retraced'."""
    if not path:
        return None, {"status": "not_recorded"}
    rows = np.asarray(path, dtype=float)
    here = np.asarray(current, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != len(here) or not np.all(np.isfinite(rows)):
        return None, {"status": "not_recorded"}
    if joint_distance(rows[0], anchor) > 1e-6:
        return None, {"status": "record_of_another_move"}
    best = (joint_distance(rows[0], here), 0, 0.0)
    for index in range(len(rows) - 1):
        start, step = rows[index], rows[index + 1] - rows[index]
        length = float(np.dot(step, step))
        fraction = 0.0 if length <= 0 else min(1.0, max(0.0, float(np.dot(here - start, step)) / length))
        distance = joint_distance(start + fraction * step, here)
        if distance <= best[0] + 1e-12:
            best = (distance, index, fraction)
    distance, index, fraction = best
    recorded = len(rows) - 1
    if distance > tolerance:
        return None, {"status": "arm_moved_since", "recorded_native_actions": recorded,
                      "joint_distance_before_rad": round(distance, 6)}
    kept = [here]
    for node in rows[index::-1]:
        if joint_distance(node, kept[-1]) >= NODE_STEP_RAD:
            kept.append(node)
    if joint_distance(kept[-1], rows[0]) > 0:
        if len(kept) > 1 and joint_distance(kept[-1], rows[0]) < NODE_STEP_RAD:
            kept[-1] = rows[0]
        else:
            kept.append(rows[0])
    return np.asarray(kept, dtype=float), {
        "status": "retraced", "recorded_native_actions": recorded,
        "from_native_action": round(index + fraction, 3), "joint_distance_before_rad": round(distance, 6)}


def path_position(path: Sequence[Sequence[float]], joints: Sequence[float]) -> float:
    """Pure computation: how far along ``path`` (0 at its start, 1 at its end, by joint-path length) the
    nearest point to ``joints`` lies. Example: path_position([q0, q1], q1) == 1.0."""
    rows, here = np.asarray(path, dtype=float), np.asarray(joints, dtype=float)
    lengths = np.linalg.norm(np.diff(rows, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 0:
        return 0.0
    best = (joint_distance(rows[0], here), 0.0)
    for index, length in enumerate(lengths):
        step = rows[index + 1] - rows[index]
        fraction = 0.0 if length <= 0 else min(1.0, max(0.0, float(np.dot(here - rows[index], step)) / length ** 2))
        distance = joint_distance(rows[index] + fraction * step, here)
        if distance <= best[0] + 1e-12:
            best = (distance, float(np.sum(lengths[:index])) + fraction * float(length))
    return best[1] / total


def lift_goal(pose: Sequence[float], bounds: Mapping[str, Sequence[float]]) -> tuple[list[float] | None, dict]:
    """Pure computation: link6 lifted straight up by up to ``LIFT_M`` (orientation held), or why not.

    Example: lift_goal(pose, {'x': [-1, 1], 'y': [-1, 1], 'z': [0, 2]})[0][2] == pose[2] + 0.1."""
    pose = [float(v) for v in pose]
    lifted = [pose[0], pose[1], min(pose[2] + LIFT_M, float(bounds["z"][1])), *pose[3:]]
    lift = lifted[2] - pose[2]
    inside = all(float(bounds[axis][0]) <= value <= float(bounds[axis][1]) for axis, value in zip("xyz", lifted))
    if lift < MINIMUM_LIFT_M or not inside:
        return None, {"status": "skipped", "reason": "workspace_ceiling", "lift_m": round(lift, 6)}
    return lifted, {"status": "planned", "lift_m": round(lift, 6)}


def plan_lift(api: DataInterface, *, arm: str, pose: Sequence[float], joints: Sequence[float],
              bounds: Mapping[str, Sequence[float]]) -> tuple[np.ndarray | None, dict, tuple]:
    """Replay: recorded kinematics answers. The home lift as joint nodes along its straight line.

    Returns (nodes or None, the lift's report, kinematics warnings). A lift without a continuous IK
    solution is skipped with the planner's reason, as the released executor skipped it.
    Example: plan_lift(api, arm='left', pose=p, joints=q, bounds=b)."""
    goal, report = lift_goal(pose, bounds)
    if goal is None:
        return None, report, ()
    try:
        found = planner.plan_line(api, starts={arm: pose}, goals={arm: goal}, start_joints={arm: joints},
                                  max_deviation_m=0.005)
    except checks.PlanRefusal as error:
        return None, {"status": "skipped", "reason": error.code, "lift_m": report["lift_m"]}, ()
    return np.asarray(found.grid.joints[arm], dtype=float), report, found.warnings


def _breakpoints(nodes: np.ndarray) -> list[float]:
    steps = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    total = float(np.sum(steps))
    if total <= 0:
        return [0.0, 1.0]
    return [0.0, *(float(v) / total for v in np.cumsum(steps)[:-1]), 1.0]


def _at(nodes: np.ndarray, fractions: Sequence[float]) -> np.ndarray:
    breaks = np.asarray(_breakpoints(nodes))
    if len(nodes) == 2 or breaks[-1] == breaks[0]:
        return np.asarray([nodes[0] + f * (nodes[-1] - nodes[0]) for f in fractions])
    return np.asarray([[float(np.interp(f, breaks, nodes[:, joint])) for joint in range(nodes.shape[1])]
                       for f in fractions])


def combine(options: Mapping[str, Option]) -> timing.Grid:
    """Pure computation: the arms' options on one grid, phase by phase, with a stop between phases.

    In each phase every arm's nodes are kept exactly (the union of the arms' breakpoints along their own
    joint-path length), and segments are split to at most ``paths.JOINT_GRID_SPACING_RAD`` (largest joint);
    an arm with fewer phases holds its last joints. A phase in which no arm moves is
    dropped; when nothing moves at all, the grid is one segment holding where the arms are.
    Example: combine({'left': o})."""
    arms = sorted(options)
    count = max(len(option.phases) for option in options.values())
    joints = {arm: [] for arm in arms}
    stops, nodes = [], 0
    for phase in range(count):
        parts = {}
        for arm in arms:
            phases = options[arm].phases
            if phase < len(phases):
                parts[arm] = np.asarray(phases[phase], dtype=float)
            else:
                last = np.asarray(phases[-1][-1], dtype=float)
                parts[arm] = np.vstack([last, last])
        if all(joint_distance(part[0], part[-1]) <= 1e-12 and len(part) == 2 for part in parts.values()):
            continue
        fractions = sorted({round(value, 12) for part in parts.values() for value in _breakpoints(part)})
        sampled = {arm: _at(part, fractions) for arm, part in parts.items()}
        # Grid nodes at most paths.JOINT_GRID_SPACING_RAD apart (largest joint), as every joint path is gridded.
        steps = np.max(np.vstack([np.max(np.abs(np.diff(values, axis=0)), axis=1) for values in sampled.values()]),
                       axis=0)
        pieces = [max(1, math.ceil(float(step) / paths.JOINT_GRID_SPACING_RAD - 1e-9)) for step in steps]
        if nodes:
            stops.append(nodes)
        for arm, values in sampled.items():
            dense = [values[0]] + [values[i] + (k / count) * (values[i + 1] - values[i])
                                   for i, count in enumerate(pieces) for k in range(1, count + 1)]
            joints[arm].append(np.asarray(dense if not joints[arm] else dense[1:]))
        nodes += sum(pieces)
    if not nodes:
        # Nothing moves: one native action holding where the arms are.
        joints = {arm: [np.vstack([options[arm].phases[-1][-1]] * 2)] for arm in arms}
        nodes = 1
    stacked = {arm: np.vstack(parts) for arm, parts in joints.items()}
    return timing.Grid(tuple(index / nodes for index in range(nodes + 1)), stacked, tuple(stops))


def _candidates(options: Mapping[str, Sequence[Option]]) -> list[dict[str, Option]]:
    arms = sorted(options)
    combinations = [()]
    for arm in arms:
        combinations = [ranks + (rank,) for ranks in combinations for rank in range(len(options[arm]))]
    ranked = sorted(combinations, key=lambda ranks: (sum(ranks), ranks))
    return [{arm: options[arm][rank] for arm, rank in zip(arms, ranks)} for ranks in ranked]


def _refusal(screens: Sequence[dict], native_action_budget: int) -> checks.PlanRefusal:
    statuses = [screen["status"] for screen in screens]
    status = next((name for name in ("inter_arm_collision", "table_collision", "self_collision")
                   if name in statuses), "self_collision")
    return checks.PlanRefusal(
        "no recovery path is clear: every candidate meets the arm itself, the table top or the other arm in "
        "the SDK collision spheres; nothing was executed",
        code=REFUSAL_CODES[status], constraints={"native_actions": 0, "screens": list(screens),
                                                  "native_action_budget": native_action_budget})


def plan(api: DataInterface, *, options: Mapping[str, Sequence[Option]], start_joints: Mapping[str, Sequence[float]],
         goals: Mapping[str, Sequence[float]], paces: Mapping[str, timing.Pace], settle: Mapping,
         native_action_budget: int) -> Planned:
    """Replay: recorded kinematics answers. The first clear candidate, timed and fitted to the budget.

    ``options`` lists each arm's options in preference order; candidates combine them, fewest
    fallbacks first. A candidate with a retracing arm runs at ``paces['measured_retrace']``, else at
    ``paces['joint_interpolation']``. Each candidate's whole timed path is screened from the measured
    joints (``kinematics.screen_path``); the first clear one is fitted to ``native_action_budget`` less
    the settle. A recovery may unwind the wrist the long way: that is what it is for. ``goals`` are the anchors' link6 poses; each arm's target is the last
    node of its option. Raises ``checks.PlanRefusal`` (``REFUSAL_CODES``, ``paced_motion_budget``) or
    ``kinematics.KinematicsUnavailable``.
    Example: plan(api, options={'left': joint_options(q0, q1)}, start_joints={'left': q0},
    goals={'left': p1}, paces={'measured_retrace': normal, 'joint_interpolation': fast},
    settle={'native_actions': 2, 'joint_tolerance_rad': 0.003, 'joint_speed_rad_s': 0.05},
    native_action_budget=20)."""
    settle = dict(settle, native_actions=min(int(settle["native_actions"]), max(int(native_action_budget) - 1, 0)))
    screens = []
    for candidate in _candidates(options):
        pace = paces[RETRACE] if any(option.method == RETRACE for option in candidate.values()) else paces[JOINT]
        grid = combine(candidate)
        whole = timing.timed(api, grid, pace, unwind_check=False)
        screen = kinematics.screen_path(api, {arm: [list(start_joints[arm]), *rows]
                                              for arm, rows in whole.joints.items()})
        label = {arm: {"method": option.method, **({"unwind_order": option.detail["unwind_order"]}
                                                   if "unwind_order" in option.detail else {}),
                       **({"lift": option.detail["retract"]["status"]} if "retract" in option.detail else {})}
                 for arm, option in candidate.items()}
        screens.append({"candidate": label, "status": screen.get("status"),
                        **{key: screen[key] for key in ("arm", "first_blocked_segment") if screen.get(key) is not None}})
        if screen.get("status") != "clear":
            continue
        budget = int(native_action_budget) - settle["native_actions"]
        part = whole if whole.native_actions <= budget else timing.timed(api, grid, pace, budget=budget,
                                                                         unwind_check=False)
        knots = {"joints": part.joints, "fractions": list(part.fractions), "duration_s": part.duration_s,
                 "native_actions": part.native_actions}
        expected = {arm: kinematics.fk(api, arm=arm, joints=rows) for arm, rows in knots["joints"].items()}
        return Planned(dict(candidate), grid, knots, part.fraction, whole.native_actions, expected, settle,
                       {arm: [float(v) for v in goal] for arm, goal in goals.items()},
                       {arm: [float(v) for v in option.phases[-1][-1]] for arm, option in candidate.items()},
                       int(native_action_budget), pace, tuple(screens))
    raise _refusal(screens, int(native_action_budget))


def run(api: DataInterface, planned: Planned, *, action_id: str, observation: Mapping,
        recovery: Mapping) -> dict:
    """Replay: the one recorded trajectory primitive. Run ``planned`` as one recovery trajectory.

    ``recovery`` is the trajectory's ``recovery`` argument (``ora.contracts.joint_trajectory``): either
    ``{"targets": {arm: "home" | "backtrack_last_safe"}, "recovery_targets": {arm: anchor}}`` (the anchors
    exactly as the state holds them) or ``{"targets": {arm: "retrace"}, "retrace": {arm: {"moves": n,
    "arm_joint_qpos": start}}}`` (the arm's retrace stack depth and its top move's start joints). The host
    admits the action only while they are current; the trajectory keeps the previous anchor, and a retrace
    that runs every knot pops the move it undid. Returns ``receipt``, ``remaining_budget`` and ``summary``
    (``execute.summary``). Example: run(api, planned, action_id='action-4', observation=obs.identity.json(),
    recovery={'targets': {'left': 'home'}, 'recovery_targets': {'left': anchor}})."""
    part = ({"fraction": planned.fraction, "goals_xyz_xyzw": dict(planned.goals)} if planned.fraction < 1.0
            else None)
    action = execute.trajectory_action(action_id=action_id, observation=observation, knots=planned.knots["joints"],
                                       expected=planned.expected, settle=planned.settle, part_of_move=part)
    action["arguments"]["recovery"] = _plain(recovery)
    result = execute.follow(api, action, settle_native_actions=planned.settle["native_actions"], route="recovery")
    return {"receipt": result["receipt"], "remaining_budget": result["remaining_budget"],
            "summary": execute.summary(result["receipt"])}


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, (bool, str, int)) or value is None:
        return value
    return float(value)


def measured_end(planned: Planned, arm: str, summary: Mapping) -> list[float] | None:
    """Pure computation: the arm's joints measured after the trajectory, from its receipt evidence, or None.

    The environment reports, per joint, the planned last knot minus the measured joints
    (``arms.<arm>.final_joint_error_per_joint_rad``); without it the joints are unknown.
    Example: measured_end(planned, 'left', summary)."""
    residual = ((summary.get("arms") or {}).get(arm) or {}).get("final_joint_error_per_joint_rad")
    last = planned.knots["joints"][arm][-1]
    if not isinstance(residual, (list, tuple)) or len(residual) != len(last):
        return None
    return [float(knot) - float(error) for knot, error in zip(last, residual)]


def _state(planned: Planned, summary: Mapping, distance: float | None) -> str:
    status, stop, settle = summary.get("status"), summary.get("stop") or {}, summary.get("settle") or {}
    if status == "completed":
        if planned.fraction < 1.0:
            return "cut_short"
        if settle.get("settled") is True:
            return "arrived"
        if settle.get("settled") is False and distance is not None:
            return "arrived_not_settled" if distance <= JOINT_ARRIVAL_RAD else "still_moving_at_window_end"
        return "unknown"
    if status == "stopped":
        return "stalled" if stop.get("reason") in ("tracking", "progress") else "unknown"
    if status == "budget":
        return "still_moving_at_window_end"
    if status == "terminal":
        return "episode_terminal"
    return "unknown"


def outcome(planned: Planned, arm: str, start: Sequence[float], end: Sequence[float] | None,
            summary: Mapping) -> dict:
    """Pure computation: how one arm's recovery ended, from the trajectory's measured evidence.

    ``summary`` is ``execute.summary`` of the receipt. ``arrived``: every knot ran and the settle settled;
    ``arrived_not_settled``: every knot ran, the settle did not converge, but the joints ended within
    ``JOINT_ARRIVAL_RAD`` of the target (else ``still_moving_at_window_end``); ``cut_short``: every knot of a
    plan cut to its budget ran (calling again continues); ``stalled``: a tracking or progress guard stopped
    the arm; ``still_moving_at_window_end``: the budget ended before the last knot; ``episode_terminal``;
    ``unknown`` when the evidence needed is missing (never a guess). ``end`` is the measured joints after the
    action (``measured_end``; None: unknown, and ``joint_distance_end_rad`` is null).
    Example: outcome(planned, 'left', q0, q1, summary)['outcome']."""
    target = planned.targets[arm]
    distance = None if end is None else joint_distance(end, target)
    option = planned.options[arm]
    return {"schema": PROGRESS_SCHEMA, "method": option.method, "outcome": _state(planned, summary, distance),
            "joint_distance_start_rad": round(joint_distance(start, target), 6),
            "joint_distance_end_rad": None if distance is None else round(distance, 6),
            **{key: value for key, value in option.detail.items()},
            "planned_native_actions": planned.knots["native_actions"],
            "joint6": checks.joint6_extremes(start, planned.knots["joints"][arm]),
            **({"cut_short": {"fraction_of_path": round(planned.fraction, 3),
                              "needed_native_actions": planned.whole_native_actions,
                              "native_action_budget": planned.native_action_budget}}
               if planned.fraction < 1.0 else {})}


__all__ = ["JOINT", "JOINT_ARRIVAL_RAD", "LIFT_M", "METHODS", "MINIMUM_LIFT_M", "NODE_STEP_RAD", "ORDERS",
           "Option", "PLANNED", "PROGRESS_SCHEMA", "Planned", "REFUSAL_CODES", "RETRACE", "RETRACE_TOLERANCE_RAD",
           "X5_JOINT_LIMITS_RAD", "X5_PITCH_JOINTS", "combine", "global_option", "joint_distance", "joint_options",
           "lift_goal", "measured_end", "outcome", "path_position", "plan", "plan_lift", "retrace_nodes", "run",
           "unwind_orders", "unwind_target"]
