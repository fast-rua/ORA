"""``motion_action_v3`` on the bimanual YAM station: plan, check, approve, execute.

A motion is planned in full on the station model before anything moves. Each targeted
arm's Cartesian path (free-space moves lift, traverse and descend) is sampled finely and
solved by warm-started IK, then timed by the strictest of the TCP's linear and angular
speed and every joint's speed; with ``runtime.motion_timing`` the same samples are timed
by ORA's shared timing worker instead, under the same caps plus an acceleration cap and a
jerk-limited start and stop (``shared_motion``). The two arms' joint schedules are merged
tick by tick and every tick, as timed, is checked against the joint limits, the table
clearance of the interaction and the inter-arm distance; a motion that starts inside a
clearance may only leave it (``Departure``). Only a plan that passes is offered to the operator (real arms
in ``on`` mode) and executed; in ``read`` mode the plan is the answer. Arrival is judged
on the measured joints. With ``runtime.motion_planner`` the paths and the admission's reach
question come from ORA's shared planner instead (``shared_planner``): a free-space target is a
collision-free plan in the public-depth world, a precision or contact target the same straight
line solved by the shared IK; the timing, the checks and everything after them stay as here.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.wait import wait_receipt_block
from ora.environment.controllers import yam_station as Y
from ora.environment.controllers.gripper_stall import (
    GRIPPER_STALL_REASON, gripper_outcome, gripper_outcome_state, gripper_stalled,
)
from ora.environment.controllers.pace_refusal import PaceRefusal

from . import shared_motion, shared_planner
from .grippers import GRIPPER_STALL, ramp
from .common import PLAN_SCHEMA, SPARSE_RECEIPT_SCHEMA, YamError, pose_error, write_json
from .kinematics import pose_matrix, pose_xyz_xyzw

ARM_EXECUTOR = "yam_ik_joint_schedule"
GRIPPER_EXECUTOR = "yam_linear_4310_gripper"
HOLD_EXECUTOR = "yam_hold"
SAMPLE_TRANSLATION_M = 0.002
SAMPLE_ROTATION_RAD = 0.01


class Refusal(Exception):
    """A plan the checks refuse: ``stage`` and a stable ``code``, with detail."""

    def __init__(self, stage: str, code: str, **detail: Any) -> None:
        super().__init__(code)
        self.stage, self.code, self.detail = stage, code, detail


@dataclass
class Schedule:
    """Per-tick joint targets of every arm the motion moves."""

    ticks: list[dict[str, np.ndarray]]
    finals: dict[str, np.ndarray]
    summary: dict[str, Any]
    #: A receipt's ``motion`` (speed, wrist, timing per arm) with ``runtime.motion_timing``; None without it.
    motion: dict[str, Any] | None = None

    def receipt_extra(self) -> dict[str, Any]:
        return {"plan": self.summary, **({} if self.motion is None else {"motion": self.motion})}


class Departure:
    """A clearance gate that lets a motion starting inside it only move out of it.

    ``minimum`` is the gate. A motion that starts closer than it (the arm left resting 12 mm
    above the table by a precision descent, the arms parked near each other) is not refused
    for where it already is: each tick must keep the largest distance reached so far, up to
    ``minimum``, so the motion may leave the near zone and never approach further.
    """

    def __init__(self, minimum: float, start: float | None, tolerance: float = 0.0) -> None:
        self.minimum, self.tolerance = minimum, tolerance
        self.reached = math.inf if start is None else start

    def admits(self, distance: float) -> bool:
        floor = min(self.minimum, self.reached)
        self.reached = max(self.reached, distance)
        return distance >= floor - self.tolerance


def _inter_arm_refusal(kin: Any, tick: int, distance: float, moving: Sequence[str]) -> Refusal:
    """``inter_arm_clearance`` with the two nearest parts and the arms the motion moved."""

    parts = kin.inter_arm_closest_parts() if hasattr(kin, "inter_arm_closest_parts") else None
    detail: dict[str, Any] = {"tick": tick, "distance_m": distance, "moving_arms": sorted(moving)}
    if parts:
        detail["closest_parts"] = parts
    return Refusal("plan_validation", "inter_arm_clearance", **detail)


def highest_reachable_z(session: Any, arm: str, target: np.ndarray, floor: float, seeds: Sequence[Any]) -> float | None:
    """The highest TCP height, at ``target``'s xy and orientation and down to ``floor``, that IK solves (1 cm steps)."""

    for z in np.arange(target[2, 3] - 0.01, floor - 1e-9, -0.01):
        lower = target.copy()
        lower[2, 3] = z
        for seed in seeds:
            if session.kinematics.ik(arm, lower, seed, max_iterations=200).ok:
                return round(float(z), 3)
    return None


#: The nearest-reachable search: rings out to this radius in 1 cm steps, in these directions relative to the
#: horizontal direction from the target to the arm's base (reach usually improves toward the base).
NEAREST_SEARCH_RADIUS_M = 0.20
NEAREST_SEARCH_TURNS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)


def nearest_reachable_position(session: Any, arm: str, target: np.ndarray, base: np.ndarray,
                               seeds: Sequence[Any]) -> list[float] | None:
    """The closest position to ``target``, at its height and orientation, that IK solves: rings 1 cm apart out to
    ``NEAREST_SEARCH_RADIUS_M``, the direction toward the arm's base first. One read-only answer in place of a grid
    of dry runs (2026-11-14: a model searched the table centre with 30-120 dry runs per script)."""

    for moved in nearest_candidates(target, base):
        if any(session.kinematics.ik(arm, moved, seed, max_iterations=200).ok for seed in seeds):
            return [round(float(value), 3) for value in moved[:3, 3]]
    return None


def nearest_candidates(target: np.ndarray, base: np.ndarray) -> Any:
    """The poses ``nearest_reachable_position`` tries, in its order: inside the workspace and the reach."""

    toward = base[:2] - target[:2, 3]
    if float(np.linalg.norm(toward)) < 1e-6:
        toward = np.array([1.0, 0.0])
    toward = toward / np.linalg.norm(toward)
    for radius in np.arange(0.01, NEAREST_SEARCH_RADIUS_M + 1e-9, 0.01):
        for turn in NEAREST_SEARCH_TURNS_DEG:
            angle = math.radians(turn)
            direction = np.array([toward[0] * math.cos(angle) - toward[1] * math.sin(angle),
                                  toward[0] * math.sin(angle) + toward[1] * math.cos(angle)])
            moved = target.copy()
            moved[:2, 3] = target[:2, 3] + radius * direction
            x, y = moved[:2, 3]
            if not (Y.WORKSPACE_X_M[0] <= x <= Y.WORKSPACE_X_M[1] and Y.WORKSPACE_Y_M[0] <= y <= Y.WORKSPACE_Y_M[1]):
                continue
            if float(np.linalg.norm(moved[:3, 3] - base)) > Y.MAX_REACH_FROM_BASE_M:
                continue
            yield moved


# ---------------------------------------------------------------- admission
def admit_target(session: Any, arm: str, target: np.ndarray, interaction: str) -> None:
    x, y, z = target[:3, 3]
    if not (Y.WORKSPACE_X_M[0] <= x <= Y.WORKSPACE_X_M[1] and Y.WORKSPACE_Y_M[0] <= y <= Y.WORKSPACE_Y_M[1]
            and z <= Y.WORKSPACE_Z_MAX_M):
        raise Refusal("admission", "target_outside_workspace", arm=arm)
    # Where this cell's arm base is: the robot model as the cell's calibration corrected it.
    base = np.asarray(session.kinematics.chain(arm)[0]) if session.kinematics is not None \
        else np.asarray(Y.BASE_IN_WORLD_XYZ_M[arm])
    if float(np.linalg.norm(target[:3, 3] - base)) > Y.MAX_REACH_FROM_BASE_M:
        raise Refusal("admission", "target_out_of_reach", arm=arm)
    table = session.rig.table_top_z_m
    if table is not None and z < table + Y.TABLE_CLEARANCE_M[interaction]:
        raise Refusal("admission", "target_below_table_clearance", arm=arm,
                      tcp_height_above_table_m=float(z - table))
    if shared_planner.active(session) is not None:
        try:
            return _admit_shared(session, arm, target, interaction, base)
        except shared_planner.Refused as refused:     # a start further past a limit than a stop explains
            raise Refusal("plan_validation", refused.code, arm=arm, **refused.detail) from None
        except shared_planner.Unplanned:
            pass    # the planner failed: the station's own IK answers the question
    # The pose itself, from the current, home and last-safe joint angles: a target no
    # seed solves is reported as such, not as a failure somewhere along the path.
    seeds = [session.rig.arm_q(arm), session.home_q.get(arm), (session.last_safe_q or {}).get(arm)]
    best = None
    for seed in (s for s in seeds if s is not None):
        result = session.kinematics.ik(arm, target, seed, max_iterations=400, start=seeds[0])
        if result.ok:
            return
        if best is None or result.position_error_m < best.position_error_m:
            best = result
    # Say how high this arm reaches here with this approach, so a retry need not step down blindly.
    floor = (table if table is not None else 0.0) + Y.TABLE_CLEARANCE_M[interaction]
    solved = [s for s in seeds if s is not None]
    reachable = highest_reachable_z(session, arm, target, floor, solved)
    nearest: dict[str, Any] = {}
    # The nearest-position search (up to ~0.8 s of IK) runs only on a session that asked for it
    # (runtime.reach_query, ora play --reach-query).
    if (getattr(session, "config", None) or {}).get("reach_query") is True:
        position = nearest_reachable_position(session, arm, target, base, solved)
        nearest = {"nearest_reachable_position_m": position,
                   "nearest_reachable_offset_m": None if position is None else
                   round(float(np.linalg.norm(np.asarray(position) - target[:3, 3])), 3)}
    raise Refusal("planner", "target_ik_unreachable", arm=arm,
                  closest_position_error_m=None if best is None else best.position_error_m,
                  highest_reachable_z_m=reachable, **nearest)


def _admit_shared(session: Any, arm: str, target: np.ndarray, interaction: str, base: np.ndarray) -> None:
    """``admit_target``'s reach question asked of the shared IK, in the station's words."""

    if shared_planner.solves(session, arm, [target])[0]:
        return
    table = session.rig.table_top_z_m
    floor = (table if table is not None else 0.0) + Y.TABLE_CLEARANCE_M[interaction]
    reachable = shared_planner.highest_reachable_z(session, arm, target, floor)
    nearest: dict[str, Any] = {}
    if (getattr(session, "config", None) or {}).get("reach_query") is True:
        position = shared_planner.nearest_reachable_position(session, arm, list(nearest_candidates(target, base)))
        nearest = {"nearest_reachable_position_m": position,
                   "nearest_reachable_offset_m": None if position is None else
                   round(float(np.linalg.norm(np.asarray(position) - target[:3, 3])), 3)}
    raise Refusal("planner", "target_ik_unreachable", arm=arm, closest_position_error_m=None,
                  highest_reachable_z_m=reachable, **nearest)


def free_space_waypoints(session: Any, arm: str, current: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    """Lift, traverse and descend at the highest reachable transit height, else go direct.

    A YAM approaching straight down reaches only about 0.2 m above its base plane near the
    base and less further out (measured on the station model, 2026-10-28), so the transit
    height is searched downward from the preferred lift until both corners solve by IK.
    A direct path is still checked tick by tick for table clearance.
    """

    horizontal = math.hypot(target[0, 3] - current[0, 3], target[1, 3] - current[1, 3])
    if horizontal <= Y.FREE_SPACE_DIRECT_RADIUS_M:
        return [target]
    floor = max(current[2, 3], target[2, 3])
    preferred = floor + Y.FREE_SPACE_LIFT_M
    table = session.rig.table_top_z_m
    if table is not None:
        preferred = max(preferred, table + Y.FREE_SPACE_LIFT_M)
    seed = session.rig.arm_q(arm)
    for height in np.arange(preferred, floor + 0.005, -0.01):
        lift, over = current.copy(), target.copy()
        lift[:3, :3] = target[:3, :3]
        lift[2, 3] = over[2, 3] = height
        first = session.kinematics.ik(arm, lift, seed, max_iterations=200)
        if first.ok and session.kinematics.ik(arm, over, first.q, max_iterations=200).ok:
            return [lift, over, target]
    return [target]


# ---------------------------------------------------------------- planning
def _interpolate(a: np.ndarray, b: np.ndarray, steps: int) -> list[np.ndarray]:
    from scipy.spatial.transform import Rotation, Slerp

    slerp = Slerp([0.0, 1.0], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))
    out = []
    for i in range(1, steps + 1):
        s = i / steps
        pose = np.eye(4)
        pose[:3, :3] = slerp([s]).as_matrix()[0]
        pose[:3, 3] = (1 - s) * a[:3, 3] + s * b[:3, 3]
        out.append(pose)
    return out


def plan_arm_path(session: Any, arm: str, q_start: np.ndarray, waypoints: Sequence[np.ndarray],
                  interaction: str) -> tuple[list[float], list[np.ndarray], list[int]]:
    """Sampled IK along the Cartesian path, timed by TCP and joint speed: (times, joint samples, the index
    of the sample that reaches each waypoint)."""

    kin = session.kinematics
    times, samples, reached = [0.0], [np.asarray(q_start, float)], []
    current = kin.fk(arm, q_start)
    linear, angular = Y.LINEAR_SPEED_M_S[interaction], Y.ANGULAR_SPEED_RAD_S[interaction]
    for index, waypoint in enumerate(waypoints):
        translation, rotation = pose_error(pose_xyz_xyzw(waypoint), pose_xyz_xyzw(current))
        steps = max(1, math.ceil(max(translation / SAMPLE_TRANSLATION_M, rotation / SAMPLE_ROTATION_RAD)))
        previous_pose = current
        for pose in _interpolate(current, waypoint, steps):
            result = kin.ik(arm, pose, samples[-1], max_iterations=60, start=q_start)
            if not result.ok:
                result = kin.ik(arm, pose, samples[-1], max_iterations=400, start=q_start)
            if not result.ok:
                raise Refusal("planner", "ik_unreachable", arm=arm, waypoint_index=index,
                              position_error_m=result.position_error_m)
            step_t, step_r = pose_error(pose_xyz_xyzw(pose), pose_xyz_xyzw(previous_pose))
            joint_step = float(np.max(np.abs(result.q - samples[-1])))
            if joint_step > 0.35:
                raise Refusal("planner", "ik_branch_jump", arm=arm, waypoint_index=index, joint_step_rad=joint_step)
            times.append(times[-1] + max(step_t / linear, step_r / angular, joint_step / Y.MAX_JOINT_SPEED_RAD_S))
            samples.append(result.q)
            previous_pose = pose
        current = waypoint
        reached.append(len(samples) - 1)
    return times, samples, reached


def station_times(session: Any, arm: str, samples: Sequence[np.ndarray], interaction: str) -> list[float]:
    """``plan_arm_path``'s timing of a joint path the station did not sample itself: each step at the strictest of
    the interaction's TCP speed and turn rate and the joint speed cap."""

    kin = session.kinematics
    linear, angular = Y.LINEAR_SPEED_M_S[interaction], Y.ANGULAR_SPEED_RAD_S[interaction]
    poses = [pose_xyz_xyzw(kin.fk(arm, np.asarray(q, float))) for q in samples]
    times = [0.0]
    for index in range(1, len(samples)):
        step_t, step_r = pose_error(poses[index], poses[index - 1])
        joint_step = float(np.max(np.abs(np.asarray(samples[index]) - np.asarray(samples[index - 1]))))
        times.append(times[-1] + max(step_t / linear, step_r / angular, joint_step / Y.MAX_JOINT_SPEED_RAD_S, 1e-9))
    return times


def _arm_path(session: Any, arm: str, q_start: np.ndarray, waypoints: Sequence[np.ndarray],
              interaction: str) -> tuple[list[float], list[np.ndarray], list[int], dict[str, Any] | None]:
    """One arm's path: (times, joint samples, the sample index reaching each waypoint, the receipt's
    ``motion.planned_by`` entry or None without ``runtime.motion_planner``).

    With the shared planner a free-space target is its collision-free plan and any other its straight line; a
    planner that fails leaves the arm to the station's own planning, recorded so."""

    record = None
    if shared_planner.active(session) is not None:
        target = waypoints[-1]
        route = shared_planner.plan_pose if interaction == "free_space" else shared_planner.plan_line
        try:
            samples, record = route(session, arm, q_start, target)
        except shared_planner.Refused as refused:
            stage = "plan_validation" if refused.code == "joint_limit" else "planner"
            raise Refusal(stage, refused.code, arm=arm, **refused.detail) from None
        except shared_planner.Unplanned as error:
            record = shared_planner.station_record(f"the shared planner failed: {error}")
            if interaction == "free_space":
                waypoints = free_space_waypoints(session, arm, session.kinematics.fk(arm, q_start), target)
                if len(waypoints) > 1:
                    record["route"] = "lift_traverse_descend"
        else:
            return station_times(session, arm, samples, interaction), samples, [len(samples) - 1], record
    elif shared_planner.enabled(session):
        record = shared_planner.station_record("no planner was bound")
    times, samples, reached = plan_arm_path(session, arm, q_start, waypoints, interaction)
    return times, samples, reached, record


def _resample(times: list[float], samples: list[np.ndarray], period: float) -> list[np.ndarray]:
    total = times[-1]
    count = max(1, math.ceil(total / period))
    array = np.asarray(samples)
    out = []
    for k in range(1, count + 1):
        t = min(k * period, total)
        out.append(np.array([np.interp(t, times, array[:, j]) for j in range(array.shape[1])]))
    return out


def _retimed(session: Any, arm: str, samples: Sequence[np.ndarray], speed: Mapping[str, Any],
             stops: Sequence[int] = ()) -> tuple[list[np.ndarray] | None, Any]:
    """``samples`` as the shared worker timed them, a row per tick, and the arm's entry for the receipt's ``motion``.

    (None, None) without ``runtime.motion_timing``; (None, entry) where the station times the path itself
    although the switch is on. A timing refusal is the plan's refusal."""

    try:
        timed = shared_motion.retime(session, arm, samples, speed, stops)
    except PaceRefusal as refusal:
        raise Refusal("planner", refusal.code, arm=arm, **refusal.constraints) from None
    return (timed.rows if isinstance(timed, shared_motion.Timed) else None), timed


def plan_motion(session: Any, plans: Mapping[str, tuple[list[np.ndarray], str]]) -> Schedule:
    """Plan every targeted arm, merge their schedules and check every tick."""

    kin, rig = session.kinematics, session.rig
    start = {arm: rig.arm_q(arm) for arm in Y.ARMS}
    per_arm, interaction_of, timing, planned = {}, {}, {}, {}
    for arm, (waypoints, interaction) in plans.items():
        times, samples, reached, record = _arm_path(session, arm, start[arm], waypoints, interaction)
        if record is not None:
            planned[arm] = record
        rows, entry = _retimed(session, arm, samples, shared_motion.speed_request(interaction), reached)
        # The same path either way; the per-tick checks below read the rows as they are timed.
        per_arm[arm] = _resample(times, samples, Y.CONTROL_PERIOD_S) if rows is None else rows
        if entry is not None:
            timing[arm] = entry
        interaction_of[arm] = interaction
    count = max(len(schedule) for schedule in per_arm.values())
    ticks = []
    min_inter_arm, min_table = math.inf, {arm: math.inf for arm in per_arm}
    kin.set_state(start, session.finger_state())
    table_gate = {arm: Departure(Y.TABLE_CLEARANCE_M[interaction_of[arm]], kin.table_distance(arm), 1e-4)
                  for arm in per_arm}
    arm_gate = Departure(Y.INTER_ARM_MIN_DISTANCE_M, kin.inter_arm_distance())
    for k in range(count):
        targets = {arm: schedule[min(k, len(schedule) - 1)] for arm, schedule in per_arm.items()}
        state = {**start, **targets}
        kin.set_state(state, session.finger_state())
        for arm, q in targets.items():
            if not kin.within_limits(arm, q, start=start[arm]):
                raise Refusal("plan_validation", "joint_limit", arm=arm, tick=k)
            table = kin.table_distance(arm)
            if table is not None:
                min_table[arm] = min(min_table[arm], table)
                if not table_gate[arm].admits(table):
                    raise Refusal("plan_validation", "table_clearance", arm=arm, tick=k, distance_m=table)
        distance = kin.inter_arm_distance()
        min_inter_arm = min(min_inter_arm, distance)
        if not arm_gate.admits(distance):
            raise _inter_arm_refusal(kin, k, distance, per_arm)
        ticks.append(targets)
    kin.set_state(start, session.finger_state())
    finals = {arm: schedule[-1] for arm, schedule in per_arm.items()}
    summary = {
        "schema": PLAN_SCHEMA, "arms": sorted(per_arm), "ticks": count,
        "duration_s": count * Y.CONTROL_PERIOD_S,
        "waypoints": {arm: [pose_xyz_xyzw(w) for w in plans[arm][0]] for arm in per_arm},
        "start_tcp": {arm: pose_xyz_xyzw(kin.fk(arm, start[arm])) for arm in per_arm},
        "min_inter_arm_distance_m": min_inter_arm,
        "min_table_distance_m": {arm: (None if math.isinf(v) else v) for arm, v in min_table.items()},
        "max_joint_travel_rad": {arm: float(np.max(np.abs(finals[arm] - start[arm]))) for arm in per_arm},
    }
    kin.set_state(start, session.finger_state())
    motion = shared_motion.motion_report(timing, count)
    if planned:
        # Evidence only: the policy's copy of a receipt leaves ``motion.planned_by`` out (``policy_receipt``).
        motion = {**(motion or {"planned_ticks": count, "duration_s": count * Y.CONTROL_PERIOD_S}),
                  "planned_by": planned}
    return Schedule(ticks, finals, summary, motion)


# ---------------------------------------------------------------- execution
class MotionRun:
    def __init__(self, session: Any, action: Mapping[str, Any]) -> None:
        self.session, self.rig, self.action = session, session.rig, action
        self.before = session.arm_state()
        self.ticks_before = session.control_ticks
        self.started = time.time()
        self.guard_events: list[dict[str, Any]] = []
        self.approval: dict[str, Any] | None = None

    def _tick(self, arm_targets: Mapping[str, np.ndarray], fingers: Mapping[str, float]) -> bool:
        hold = {arm: self.session.commanded_q[arm] for arm in Y.ARMS}
        hold.update(arm_targets)
        result = self.rig.step(hold, {**self.session.finger_targets, **fingers})
        self.session.commanded_q.update({arm: np.asarray(q, float) for arm, q in arm_targets.items()})
        self.session.control_ticks += 1
        self.session.on_control_tick()
        if result.events:
            self.guard_events.extend(result.events)
            self.session.on_guard_events(result.events)
            return False
        return not self.session.must_stop()

    def follow(self, schedule: Schedule) -> str:
        for targets in schedule.ticks:
            if not self._tick(targets, {}):
                return "guard_stop" if self.guard_events else "episode_ended"
        within = 0
        for _ in range(Y.MAX_SETTLE_TICKS):
            errors = [self._tcp_error(arm, q) for arm, q in schedule.finals.items()]
            if all(t <= Y.POSITION_TOLERANCE_M and r <= Y.ORIENTATION_TOLERANCE_RAD for t, r in errors):
                within += 1
                if within >= Y.SETTLE_TICKS:
                    return "reached"
            else:
                within = 0
            if not self._tick(schedule.finals, {}):
                return "guard_stop" if self.guard_events else "episode_ended"
        return "not_settled"

    def _tcp_error(self, arm: str, q_goal: np.ndarray) -> tuple[float, float]:
        kin = self.session.kinematics
        goal = pose_xyz_xyzw(kin.fk(arm, q_goal))
        measured = pose_xyz_xyzw(kin.fk(arm, self.rig.arm_q(arm)))
        return pose_error(goal, measured)

    def gripper(self, goals: Mapping[str, str]) -> dict[str, dict[str, Any]]:
        """Ramp each commanded jaw to its goal and stop where it arrives or stalls.

        The command moves by ``grippers.ramp`` (the closing or opening speed, a few millimetres ahead of the
        measured jaw at most), never as one step to the end. A jaw that stalls (on an object, or on a stop) is then held
        ``GRIPPER_HOLD_SQUEEZE_M`` past where it stopped instead of being driven on to the end of its travel.
        """

        targets = {arm: (Y.FINGER_CLOSED_M if goal == "close" else Y.FINGER_OPEN_M) for arm, goal in goals.items()}
        normalized = {arm: (0.0 if goal == "close" else 1.0) for arm, goal in goals.items()}
        commanded = {arm: float(self.rig.finger_m(arm)) for arm in goals}
        self.session.gripper_goals.update(goals)
        traces = {arm: [self.session.opening(arm)] for arm in goals}
        done: dict[str, int] = {}
        used = 0
        for used in range(1, Y.GRIPPER_TICKS + 1):
            for arm in goals:
                if arm not in done:
                    commanded[arm] = ramp(commanded[arm], float(self.rig.finger_m(arm)), targets[arm],
                                          closing=goals[arm] == "close")
            self.session.finger_targets.update(commanded)
            if not self._tick({}, commanded):
                break
            for arm in goals:
                if arm in done:
                    continue
                traces[arm].append(self.session.opening(arm))
                arrived = GRIPPER_STALL.arrived(traces[arm][-1], normalized[arm])
                if arrived or gripper_stalled(traces[arm], target=normalized[arm], rule=GRIPPER_STALL):
                    done[arm] = used
                    if not arrived:
                        # Hold the object with a small squeeze; the force limit caps what that asks of the motor.
                        squeeze = Y.GRIPPER_HOLD_SQUEEZE_M if goals[arm] == "close" else -Y.GRIPPER_HOLD_SQUEEZE_M
                        commanded[arm] = float(np.clip(self.rig.finger_m(arm) - squeeze,
                                                       Y.FINGER_CLOSED_M, Y.FINGER_OPEN_M))
            if len(done) == len(goals):
                break
        self.session.finger_targets.update(commanded)
        if done and used < Y.GRIPPER_TICKS and not self.guard_events:
            self._tick({}, commanded)    # the hold reaches the jaws now, not at the next motion's first tick
        out = {}
        for arm in goals:
            measured, ticks = traces[arm][-1], done.get(arm, used)
            state = gripper_outcome_state(
                reached=GRIPPER_STALL.arrived(measured, normalized[arm]),
                stalled=gripper_stalled(traces[arm], target=normalized[arm], rule=GRIPPER_STALL),
                ticks_used=ticks, ticks_budget=Y.GRIPPER_TICKS)
            out[arm] = gripper_outcome(
                rule=GRIPPER_STALL, state=state, target=normalized[arm], measured=measured,
                ticks_used=ticks, ticks_budget=Y.GRIPPER_TICKS,
                extra={"measured_aperture_mm": self.session.aperture_m(arm) * 1000.0,
                       "goal": goals[arm]})
        return out

    def hold(self, ticks: int) -> int:
        executed = 0
        for _ in range(ticks):
            if not self._tick({}, {}):
                break
            executed += 1
        return executed

    def _record_loads(self) -> None:
        """The motion's peak external load per joint (a real rig's ``external_load.LoadMonitor``), in the worker's
        evidence only (``rig-loads/<action_id>.json``): the data the stop limits are tuned on."""

        loads = getattr(self.rig, "loads", None)
        root = getattr(self.session, "output_root", None)
        if loads is None or root is None:
            return
        write_json(root / "rig-loads" / f"{self.action['action_id']}.json", {
            "schema": "ora.yam-motion-loads.v1", "action": self.action["action"],
            "peak_residual_torque_nm": loads.take_peaks(), "limits_nm": [float(v) for v in loads.limits],
            "stop_ticks": loads.ticks,
            "stops": [event for event in self.guard_events if event.get("code") == "external_load_stop"]})

    def receipt(self, *, status: str, strategy: str, executor: Sequence[str] | None, target_arms: Sequence[str],
                stage: str | None = None, code: str | None = None,
                targets: Mapping[str, np.ndarray] | None = None, interactions: Mapping[str, str] | None = None,
                grippers: Mapping[str, Mapping[str, Any]] | None = None,
                extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._record_loads()
        after = self.session.arm_state()
        arms: dict[str, Any] = {}
        for arm in Y.ARMS:
            record: dict[str, Any] = {"targeted": arm in target_arms,
                                      "interaction": (interactions or {}).get(arm),
                                      "pose_error": None, "gripper_error": None, "plan_validation": None}
            if targets and arm in targets:
                translation, rotation = pose_error(pose_xyz_xyzw(targets[arm]), after[arm]["pose_xyz_xyzw"])
                record["pose_error"] = {"translation_m": translation, "orientation_rad": rotation}
            if grippers and arm in grippers:
                record["gripper_outcome"] = dict(grippers[arm])
            arms[arm] = record
        ticks = self.session.control_ticks - self.ticks_before
        mode = "hold" if self.action["action"] == "wait" else (
            "simultaneous" if len(target_arms) > 1 else "single_arm")
        return {
            "schema": SPARSE_RECEIPT_SCHEMA, "action_id": self.action["action_id"], "action": self.action["action"],
            "arguments": deepcopy(dict(self.action["arguments"])), "consumed": True, "state_changed": ticks > 0,
            "execution_status": status, "motion_strategy": strategy, "failure_stage": stage, "failure_code": code,
            "executor": list(executor) if executor else None, "target_arms": sorted(target_arms),
            "synchronization": {"mode": mode, "requested": len(target_arms) > 1, "executed": ticks > 0},
            "arms": arms, "before": self.before, "after": after,
            "steps_used": ticks, "native_actions": ticks, "native_actions_total": self.session.control_ticks,
            "rig_mode": self.rig.mode, "guard_events": list(self.guard_events),
            "round_index": self.session.round_index, "official_evaluator_queried": False,
            "terminal_after_action": bool(self.session.is_terminal),
            "started_unix_s": self.started, "finished_unix_s": time.time(), **dict(extra or {}),
            # Evidence and front ends only: the context projection drops "timestamps" (causal_value.py).
            **({"timestamps": {"approval": dict(self.approval)}} if self.approval else {}),
        }


def _arm_status(stop: str) -> tuple[str, str | None, str | None]:
    if stop == "reached":
        return "reached", None, None
    if stop == "guard_stop":
        return "not_reached", "execution", "guard_stop"
    if stop == "episode_ended":
        return "not_reached", "execution", "episode_terminal_during_execution"
    return "not_reached", "execution", "not_settled"


def _gate(run: MotionRun, summary: Mapping[str, Any], **receipt: Any) -> dict[str, Any] | None:
    """The operator gate: a read-only rig answers with the plan; a live rig asks the operator."""

    if run.rig.mode == "read":
        return run.receipt(status="plan_rejected", strategy="dry_run", executor=None, stage="operator",
                           code="dry_run_read_only", extra={"plan": dict(summary)}, **receipt)
    approved, words = run.rig.operator_approval(summary)
    run.approval = getattr(run.rig, "last_approval", None)
    if not approved:
        return run.receipt(status="plan_rejected", strategy="none", executor=None, stage="operator",
                           code="operator_declined", extra={"plan": dict(summary), "operator_says": words},
                           **receipt)
    return None


def move_plans(session: Any, arguments: Mapping[str, Any]) -> tuple[dict[str, tuple], dict[str, np.ndarray]]:
    """Each targeted arm's waypoints and interaction, admitted; the targets as matrices. Raises ``Refusal``."""

    targets = {arm: pose_matrix(t["position_m"], t["orientation_xyzw"]) for arm, t in arguments["targets"].items()}
    plans = {}
    # The shared planner routes a free-space move itself (``shared_planner.plan_pose``): no lift to search for.
    lifts = shared_planner.active(session) is None
    for arm, target in targets.items():
        interaction = arguments["interactions"][arm]
        admit_target(session, arm, target, interaction)
        current = session.kinematics.fk(arm, session.rig.arm_q(arm))
        plans[arm] = ((free_space_waypoints(session, arm, current, target) if interaction == "free_space" and lifts
                       else [target]), interaction)
    return plans, targets


def recovery_goals(session: Any, arguments: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    """The joints a recovery returns each arm to; None where one has no such state."""

    goals = {}
    for arm, strategy in arguments["targets"].items():
        source = session.home_q if strategy == "home" else (session.last_safe_q or {})
        if arm not in source:
            return None
        goals[arm] = source[arm]
    return goals


def execute_motion_v3(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    from .action_plan import scheduled

    run = MotionRun(session, action)
    name, arguments = action["action"], action["arguments"]
    if name == "move_base":
        return run.receipt(status="plan_rejected", strategy="none", executor=None, target_arms=[],
                           stage="admission", code="mobile_base_unavailable")
    if name in {"move_pose", "move_free_space"}:
        if "plan_ref" in arguments:
            return run.receipt(status="plan_rejected", strategy="none", executor=None, target_arms=[],
                               stage="planner", code="plan_reference_unsupported")
        targets = {arm: pose_matrix(t["position_m"], t["orientation_xyzw"]) for arm, t in arguments["targets"].items()}
        interactions = dict(arguments["interactions"])
        common = {"target_arms": sorted(targets), "targets": targets, "interactions": interactions}
        try:
            plans, _ = move_plans(session, arguments)
            schedule = scheduled(session, action, lambda: plan_motion(session, plans))
        except Refusal as refusal:
            return run.receipt(status="plan_rejected", strategy="none", executor=None, stage=refusal.stage,
                               code=refusal.code, extra={"refusal": refusal.detail}, **common)
        gated = _gate(run, schedule.summary, **common)
        if gated is not None:
            return gated
        session.last_safe_q = {arm: session.rig.arm_q(arm) for arm in Y.ARMS}
        stop = run.follow(schedule)
        status, stage, code = _arm_status(stop)
        strategy = "lift_traverse_descend" if any(len(p[0]) > 1 for p in plans.values()) else "cartesian_direct"
        routes = {entry.get("route") for entry in ((schedule.motion or {}).get("planned_by") or {}).values()}
        if "plan" in routes:
            strategy = "planned_joint_path"
        elif "lift_traverse_descend" in routes:
            strategy = "lift_traverse_descend"
        return run.receipt(status=status, strategy=strategy, executor=[ARM_EXECUTOR], stage=stage, code=code,
                           extra=schedule.receipt_extra(), **common)
    if name == "set_gripper":
        goals = dict(arguments["targets"])
        common = {"target_arms": sorted(goals)}
        summary = {"schema": PLAN_SCHEMA, "gripper": goals}
        gated = _gate(run, summary, **common)
        if gated is not None:
            return gated
        outcomes = run.gripper(goals)
        failures = {arm: _gripper_failure(o["state"], run) for arm, o in outcomes.items()}
        code = next((c for c in failures.values() if c is not None), None)
        return run.receipt(status="reached" if code is None else "not_reached", strategy="gripper",
                           executor=[GRIPPER_EXECUTOR], stage=None if code is None else "execution", code=code,
                           grippers=outcomes, **common)
    if name == "recover_motion":
        strategies = dict(arguments["targets"])
        common = {"target_arms": sorted(strategies)}
        goals = recovery_goals(session, arguments)
        if goals is None:
            return run.receipt(status="plan_rejected", strategy="none", executor=None, stage="plan_validation",
                               code="recovery_state_unavailable", **common)
        try:
            schedule = scheduled(session, action, lambda: plan_joint_motion(session, goals))
        except Refusal as refusal:
            return run.receipt(status="plan_rejected", strategy="none", executor=None, stage=refusal.stage,
                               code=refusal.code, extra={"refusal": refusal.detail}, **common)
        gated = _gate(run, schedule.summary, **common)
        if gated is not None:
            return gated
        stop = run.follow(schedule)
        status, stage, code = _arm_status(stop)
        tcp_targets = {arm: session.kinematics.fk(arm, q) for arm, q in goals.items()}
        return run.receipt(status=status, strategy="recover_" + "_".join(sorted(set(strategies.values()))),
                           executor=[ARM_EXECUTOR], stage=stage, code=code, targets=tcp_targets,
                           interactions={arm: "free_space" for arm in goals},
                           extra=schedule.receipt_extra(), **common)
    if name == "wait":
        ticks = int(arguments["ticks"])
        executed = run.hold(ticks)
        return run.receipt(
            status="reached" if executed == ticks else "not_reached", strategy="hold", executor=[HOLD_EXECUTOR],
            target_arms=[], stage=None if executed == ticks else "execution",
            code=None if executed == ticks else "episode_terminal_during_execution",
            extra={"wait": wait_receipt_block(
                ticks_requested=ticks, physics_steps=executed, tick_unit="yam_control_tick",
                tick_period_s=Y.CONTROL_PERIOD_S, terminated_early=executed < ticks, physics_steps_per_tick=1)},
        )
    raise YamError(f"unsupported motion v3 action on the YAM station: {name!r}")


def _gripper_failure(state: str, run: MotionRun) -> str | None:
    if state == "reached_target":
        return None
    if run.guard_events:
        return "guard_stop"
    if state == "stalled_before_target":
        return GRIPPER_STALL_REASON
    return "gripper_not_reached"


def plan_joint_motion(session: Any, goals: Mapping[str, np.ndarray]) -> Schedule:
    """A straight joint-space move to known-safe configurations, checked tick by tick.

    Every arm moves along its joint line at one shared progress, at half the joint speed cap: a constant
    rate, or with ``runtime.motion_timing`` the progress the shared worker times for the arm that travels
    furthest, which the others follow (their joints travel less, so every cap holds for them too)."""

    kin, rig = session.kinematics, session.rig
    start = {arm: rig.arm_q(arm) for arm in Y.ARMS}
    travels = {arm: float(np.max(np.abs(np.asarray(goals[arm]) - start[arm]))) for arm in goals}
    travel = max(travels.values())
    count = max(1, math.ceil(travel / (Y.MAX_JOINT_SPEED_RAD_S * 0.5 * Y.CONTROL_PERIOD_S)))
    progress = [k / count for k in range(1, count + 1)]
    lead = max(travels, key=travels.get)
    rows, entry = _retimed(session, lead, [start[lead], np.asarray(goals[lead], float)],
                           shared_motion.joint_speed_request(Y.MAX_JOINT_SPEED_RAD_S * 0.5))
    if rows is not None:
        # The timed rows lie on the lead arm's joint line: its furthest joint gives each tick's progress.
        joint = int(np.argmax(np.abs(np.asarray(goals[lead], float) - start[lead])))
        span = float(goals[lead][joint] - start[lead][joint])
        progress = [*(float((row[joint] - start[lead][joint]) / span) for row in rows[:-1]), 1.0]
        count = len(progress)
    ticks, min_inter_arm = [], math.inf
    kin.set_state(start, session.finger_state())
    table_gate = {arm: Departure(Y.TABLE_CLEARANCE_M["precision"], kin.table_distance(arm)) for arm in goals}
    arm_gate = Departure(Y.INTER_ARM_MIN_DISTANCE_M, kin.inter_arm_distance())
    for k, s in enumerate(progress, start=1):
        targets = {arm: (1 - s) * start[arm] + s * np.asarray(goals[arm], float) for arm in goals}
        kin.set_state({**start, **targets}, session.finger_state())
        for arm in goals:
            table = kin.table_distance(arm)
            if table is not None and not table_gate[arm].admits(table):
                raise Refusal("plan_validation", "table_clearance", arm=arm, tick=k, distance_m=table)
        distance = kin.inter_arm_distance()
        min_inter_arm = min(min_inter_arm, distance)
        if not arm_gate.admits(distance):
            raise _inter_arm_refusal(kin, k, distance, goals)
        ticks.append(targets)
    kin.set_state(start, session.finger_state())
    summary = {"schema": PLAN_SCHEMA, "arms": sorted(goals), "ticks": count, "duration_s": count * Y.CONTROL_PERIOD_S,
               "joint_space": True, "min_inter_arm_distance_m": min_inter_arm,
               "max_joint_travel_rad": {arm: float(np.max(np.abs(np.asarray(goals[arm]) - start[arm])))
                                        for arm in goals}}
    return Schedule(ticks, {arm: np.asarray(q, float) for arm, q in goals.items()}, summary,
                    shared_motion.motion_report({} if entry is None else {lead: entry}, count))
