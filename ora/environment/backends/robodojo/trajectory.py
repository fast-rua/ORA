"""``follow_joint_trajectory``: one action, smooth inside, guarded per physics step.

The caller (the toolkit lib) owns every planning choice and hands over, per
arm, the joint target at the end of each native action (*knots*). This
executor only admits, executes, guards and reports:

* **Admission** (nothing is written before it passes): the knots lie inside
  the live articulation limits, and the first knot from the measured joints
  and every later knot from its predecessor stay within the admission step
  (at most 0.2 rad, the arm's 5 rad/s over one native action). A refusal is a
  ``plan_rejected`` receipt with zero native actions.
* **Execution** (``knots_linear_ff``): physics step ``j`` (1..10) of knot ``k``
  commands ``q[k-1] + j/10 * (q[k] - q[k-1])`` with the velocity feed-forward
  ``(q[k] - q[k-1]) / 0.04 s``; knot 0 is the measured joints. Step 10 lands
  exactly on the knot, so the arm never restarts from its measurement and
  never holds still inside a tick. The last executed step carries velocity 0,
  so a held arm is never given a feed-forward.
* **Guards**, evaluated after every physics step on the measured joints:
  *tracking* trips when the commanded-minus-measured joint error of an arm
  exceeds the knot's bound for ``tracking_window_steps`` consecutive steps
  (default bound: a quarter of the knot step, at least 0.012 rad, so it
  scales with the speed; ``stop.rule`` ``bound``), or when it keeps growing
  (``guards.tracking_growth``: above 0.004 rad and up by more than
  0.0015 rad over the last 10 physics steps, for 10 consecutive steps;
  ``stop.rule`` ``growing_error``), which catches an arm pushed back by an
  obstacle within about two native actions. ``expect_contact`` (a
  deliberate push or press) turns the growing-error rule off; the bound
  still stops a real block (``stop.note`` says so); *progress* trips when, over ``window_steps`` physics steps, the
  measured joints advance along the commanded direction by less than
  ``min_ratio`` of the commanded advance (arm blocked or stalled). On a trip
  at step ``j`` the remaining steps of that tick hold the joints measured at
  the trip with velocity 0 and the action ends at the tick boundary: the arm
  never chases a target it fell behind (no lag-then-snap).
* **Settle** (optional, explicit): after the last knot, up to
  ``settle.native_actions`` hold ticks until every joint is within
  ``joint_tolerance_rad`` of the last knot and moves slower than
  ``joint_speed_rad_s``; the default is none. An arm that does not settle
  but whose link6 pose is within the endpoint acceptance (10 mm, 0.1 rad) is
  reached with ``failure_code`` ``arm_not_settled`` (stage ``settle``), the
  joint error in ``trajectory.settle`` and ``motion_progress``; only a pose
  outside it is ``endpoint_residual_exceeded``.
* **Gripper knots** (optional): one normalized opening per knot, interpolated
  per physics step on the primary finger from the drive target the plant
  still applies, and never leading its measurement by more than
  ``gripper.max_lead_m``.

``speed_segments`` (optional, per arm) names the knots each speed of the
caller's plan covers; the receipt (``trajectory.speed_segments`` and each
arm's ``motion_progress``) adds the native actions each one spent.

Recovery anchor: a trajectory moves the previous anchor
(``exec.recovery.<arm>.previous``) to the joints measured at its start and
records with it the joints measured at the end of every native action it ran
(``measured_path_rad``), which a later recovery can retrace. A trajectory
marked ``recovery`` (a recovery the toolkit planned) keeps the anchor and its
path, as the released recovery executor does; its receipt says so
(``trajectory.recovery``). Undo stacks (``undo_stacks``, the last
``UNDO_DEPTH`` arm-moving tool calls per arm): each non-recovery trajectory
pushes an entry for its tool call, or extends the entry when the same call
already moved the arm; a recovery marked ``retrace`` that runs every knot
pops the entry it named, one that stops short records its progress there.
The receipt names the operation (``trajectory.undo``) and each arm's
``motion_progress.undo_depth``.

A trajectory marked ``part_of_move`` (the caller cut its move to fit a
budget) that runs every knot ends ``not_reached`` with
``native_action_budget_exhausted``; ``trajectory.part_of_move`` and each
arm's ``motion_progress`` (outcome ``cut_short``) carry the measured
distance left to the goal, and the residuals and the held pose target are the
cut point's.

The receipt reports raw measured residuals (final joint error per joint,
link6 error against the expected final pose when the caller supplied it, the
largest tracking error) and how the action ended; it never reports arrival
against a threshold the caller did not choose. The rate-limited
``set_gripper`` executor (``gripper_ramp``) shares the tick runner.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.joint_trajectory import (
    ARMS, CONTROLLER_INTERFACE, MAX_TRAJECTORY_NATIVE_ACTIONS, NATIVE_ACTION_S, PHYSICS_STEP_S,
    requires_joint_trajectory_backend, speed_segment_spend,
)

from .common import physical_gripper, pose_error, pose_reached
from .native_sequence import run_tick

TRAJECTORY_EXECUTOR = "robodojo_joint_trajectory_knots_linear_ff"
TRAJECTORY_STRATEGY = "joint_trajectory"
TRAJECTORY_SCHEMA = "ora.robodojo-trajectory-receipt.v1"
PROGRESS_SCHEMA = "ora.robodojo-trajectory-progress.v1"
COMMAND_INVARIANT_SCHEMA = "ora.command-invariant.v3"
APPLIED_TARGETS = "per_physics_step_sdk_joint_position_and_velocity_buffers"
#: Default tracking bound: this fraction of the knot step, never below the floor. Calibrated on the
#: 2026-10-28 real-card runs (67 trajectories, build_tower and push_T): clean moves never lagged more than
#: 0.0055 rad (fast 0.0046 at knot steps up to 0.16 rad, normal 0.0051, careful 0.0003), so the bound keeps
#: at least 2.2x margin over every clean move; a careful deliberate push reached 0.0094 rad and passes.
#: The previous default (three knot steps, at least 0.02 rad) let a normal move drive 10 native actions into a
#: block tower before it tripped.
TRACKING_KNOTS = 0.25
TRACKING_FLOOR_RAD = 0.012
#: Admission tolerance on the live joint limits (float32 articulation buffers).
LIMIT_EPSILON_RAD = 1e-6
STOP_CODES = {"tracking": ("execution", "sustained_joint_tracking_error"),
              "progress": ("execution", "controller_stalled"),
              "budget": ("execution", "native_action_budget_exhausted"),
              "terminal": ("execution", "episode_terminal_during_execution"),
              "not_settled": ("endpoint_verification", "endpoint_residual_exceeded"),
              "reached_not_settled": ("settle", "arm_not_settled"),
              "part_of_move": ("execution", "native_action_budget_exhausted")}
#: ``motion_progress.outcome`` of a completed trajectory that was only part of the caller's move.
CUT_SHORT = "cut_short"


def per_step_action(action: Mapping[str, Any]) -> bool:
    """Actions this module executes instead of ``sparse_motion``."""
    return requires_joint_trajectory_backend(action)


def execute_per_step_action(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    if action["action"] == "set_gripper":
        from .gripper_ramp import execute_rate_limited_gripper
        return execute_rate_limited_gripper(session, action)
    return execute_trajectory(session, action)


def execution_budget(session: Any) -> int:
    return min(MAX_TRAJECTORY_NATIVE_ACTIONS, int(getattr(session, "request_action_budget", 20)))


def arm_key(session: Any, arm: str) -> str:
    manager = session.environment.robot_manager
    return manager.process_name(session.robots[arm].arm_name)


def gripper_key(session: Any, arm: str) -> str:
    manager = session.environment.robot_manager
    return manager.process_name(session.robots[arm].gripper_name)


def held_gripper_targets(session: Any, arms: Any, fingers: Mapping[str, float]) -> dict[str, float]:
    """The primary-finger drive target the plant still applies to each named jaw (the measured finger if none)."""
    retained = session.environment.robot_manager.control_manager.prev_control[0]
    result = {}
    for arm in arms:
        entry = retained.get(gripper_key(session, arm))
        position = entry.get("position") if isinstance(entry, Mapping) else None
        result[arm] = (float(position[0]) if position is not None and len(position) and np.isfinite(position[0])
                       else float(fingers[arm]))
    return result


def hold_retained_velocity(session: Any, keys: Sequence[str]) -> list[str]:
    """Zero the velocity feed-forward the plant would keep applying to a held arm; returns the keys rewritten.

    The last executed step already carries velocity 0 except when an official
    terminal cut the action; this keeps the retained command a pure hold then.
    """
    retained = session.environment.robot_manager.control_manager.prev_control[0]
    rewritten = []
    for key in keys:
        entry = retained.get(key)
        if isinstance(entry, Mapping) and np.any(np.asarray(entry.get("velocity", 0), dtype=float) != 0):
            retained[key] = {**entry, "velocity": [0.0] * len(entry["position"])}
            rewritten.append(key)
    return rewritten


class _Measurements:
    """Per-arm measured joints of the last two physics steps (joint speed by finite difference)."""

    def __init__(self, start: Mapping[str, np.ndarray], fingers: Mapping[str, float]) -> None:
        self.current = {arm: np.asarray(value, dtype=float) for arm, value in start.items()}
        self.previous = dict(self.current)
        self.fingers = {arm: float(value) for arm, value in fingers.items()}

    def update(self, measured: Mapping[str, Any]) -> None:
        self.previous = dict(self.current)
        for arm in self.current:
            self.current[arm] = np.asarray(measured[arm]["arm_joint_qpos"], dtype=float)
            self.fingers[arm] = float(measured[arm]["gripper_finger_qpos"][0])

    def speed(self) -> float:
        return max(float(np.max(np.abs(self.current[arm] - self.previous[arm]))) / PHYSICS_STEP_S
                   for arm in self.current)


class TrajectorySource:
    """Per-physics-step targets, guards and evidence of one admitted trajectory."""

    def __init__(self, session: Any, arms: Sequence[str], knots: Mapping[str, np.ndarray],
                 start: Mapping[str, np.ndarray], fingers: Mapping[str, float],
                 gripper_knots: Mapping[str, Sequence[float]], guards: Mapping[str, Any],
                 max_lead_m: float | None, last_knot: int) -> None:
        self.session, self.arms = session, tuple(arms)
        self.knots = {arm: np.vstack([start[arm], knots[arm]]) for arm in self.arms}
        self.robots = {arm: session.robots[arm] for arm in self.arms}
        self.keys = {arm: arm_key(session, arm) for arm in self.arms}
        self.gripper_keys = {arm: gripper_key(session, arm) for arm in gripper_knots}
        self.gripper = {arm: [float(physical_gripper(self.robots[arm], value)[0]) for value in values]
                        for arm, values in gripper_knots.items()}
        #: Gripper knots start from the drive target the plant still applies, not the measured finger.
        self.start_fingers = held_gripper_targets(session, self.arms, fingers)
        self.lead = max_lead_m
        self.measured = _Measurements(start, fingers)
        self.last_knot, self.knot, self.mode = int(last_knot), 1, "trajectory"
        steps = {arm: np.max(np.abs(np.diff(self.knots[arm], axis=0)), axis=1) for arm in self.arms}
        bound = guards["tracking_rad"]
        self.tracking = {arm: (np.full(len(steps[arm]), float(bound)) if isinstance(bound, (int, float))
                               else np.asarray(bound, dtype=float) if bound is not None
                               else np.maximum(TRACKING_FLOOR_RAD, TRACKING_KNOTS * steps[arm]))
                         for arm in self.arms}
        self.window = int(guards["tracking_window_steps"])
        self.progress = guards["progress"]
        history = (self.progress or {}).get("window_steps", 1) + 1
        self.history = {arm: deque(maxlen=history) for arm in self.arms}
        self.over = {arm: 0 for arm in self.arms}
        self.growth = guards.get("tracking_growth")
        span = (self.growth or {}).get("over_steps", 1) + 1
        self.error_history = {arm: deque(maxlen=span) for arm in self.arms}
        self.rising = {arm: 0 for arm in self.arms}
        self.commanded = {arm: np.asarray(start[arm], dtype=float) for arm in self.arms}
        self.last_gripper: dict[str, float] = {}
        self.errors = {arm: 0.0 for arm in self.arms}
        self.max_error = {arm: 0.0 for arm in self.arms}
        self.tick_error = {arm: 0.0 for arm in self.arms}
        self.stop: dict[str, Any] | None = None
        self.hold: dict[str, np.ndarray] | None = None
        self._new_stop = False

    # -- tick control -------------------------------------------------------
    def begin_tick(self, knot: int | None, mode: str = "trajectory") -> None:
        """Run knot ``knot`` next, or (``knot`` None) hold the last commanded joints with velocity 0."""
        self.mode = mode
        if knot is not None:
            self.knot = knot
        self.tick_error = {arm: 0.0 for arm in self.arms}

    # -- StepSource ----------------------------------------------------------
    def request(self, index: int) -> dict[str, Any]:
        moving = self.mode == "trajectory" and self.stop is None
        result: dict[str, Any] = {}
        for arm in self.arms:
            if moving:
                previous, target = self.knots[arm][self.knot - 1], self.knots[arm][self.knot]
                fraction = (index + 1) / 10.0
                position = previous + fraction * (target - previous)
                velocity = (target - previous) / NATIVE_ACTION_S
                if self.knot == self.last_knot and index == 9:
                    velocity = np.zeros_like(velocity)
            else:
                position = self.hold[arm] if self.hold is not None else self.commanded[arm]
                velocity = np.zeros(len(position))
            self.commanded[arm] = np.asarray(position, dtype=float)
            result[self.keys[arm]] = {"position": [float(v) for v in position],
                                      "velocity": [float(v) for v in velocity]}
        for arm, values in self.gripper.items():
            result[self.gripper_keys[arm]] = {"position": self._gripper_step(arm, values, index, moving)}
        return result

    def _gripper_step(self, arm: str, values: Sequence[float], index: int, moving: bool) -> list[float]:
        robot = self.robots[arm]
        if moving:
            previous = values[self.knot - 2] if self.knot >= 2 else self.start_fingers[arm]
            value = previous + (index + 1) / 10.0 * (values[self.knot - 1] - previous)
        else:
            value = self.last_gripper.get(arm, values[min(self.knot, len(values)) - 1])
        measured = self.measured.fingers.get(arm)
        if self.lead is not None and measured is not None:
            value = float(np.clip(value, measured - self.lead, measured + self.lead))
        value = float(np.clip(value, *robot.gripper_scale))
        self.last_gripper[arm] = value
        return [value, value * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]]

    def observe(self, index: int, measured: Mapping[str, Any]) -> None:
        self.measured.update(measured)
        self._new_stop = False
        for arm in self.arms:
            current = self.measured.current[arm]
            error = float(np.max(np.abs(self.commanded[arm] - current)))
            self.errors[arm] = error
            self.max_error[arm] = max(self.max_error[arm], error)
            self.tick_error[arm] = max(self.tick_error[arm], error)
            if self.mode != "trajectory" or self.stop is not None:
                continue
            bound = float(self.tracking[arm][self.knot - 1])
            self.over[arm] = self.over[arm] + 1 if error > bound else 0
            if self.over[arm] >= self.window:
                self._trip("tracking", arm, index, error=error, bound=bound)
                continue
            if self.growth is not None:
                history = self.error_history[arm]
                history.append(error)
                rise = error - history[0] if len(history) == history.maxlen else 0.0
                self.rising[arm] = (self.rising[arm] + 1 if error > self.growth["min_error_rad"]
                                    and rise > self.growth["min_rise_rad"] else 0)
                if self.rising[arm] >= self.growth["sustained_steps"]:
                    # Pushed back by something: the lag keeps growing, well before it reaches the bound.
                    self._trip("tracking", arm, index, error=error, bound=bound, error_rise_rad=rise)
                    self.stop["rule"] = "growing_error"
                    continue
            self.history[arm].append((self.commanded[arm].copy(), current.copy()))
            if self.progress is not None and len(self.history[arm]) == self.history[arm].maxlen:
                (commanded_then, measured_then), (commanded_now, measured_now) = self.history[arm][0], self.history[arm][-1]
                direction = commanded_now - commanded_then
                commanded = float(np.linalg.norm(direction))
                if commanded >= self.progress["min_commanded_rad"]:
                    advanced = float(np.dot(measured_now - measured_then, direction) / commanded)
                    if advanced < self.progress["min_ratio"] * commanded:
                        self._trip("progress", arm, index, error=error, commanded_rad=commanded, advanced_rad=advanced)

    def _trip(self, reason: str, arm: str, index: int, **evidence: float) -> None:
        self.stop = {"reason": reason, "arm": arm, "knot": self.knot, "physics_step": index + 1,
                     **({"rule": "bound"} if reason == "tracking" else {}),
                     **({"note": "stopped by the tracking bound despite expect_contact"}
                        if reason == "tracking" and getattr(self, "expect_contact", False) else {}),
                     "native_action": int(self.session.environment.take_action_cnt[0]),
                     **{key: round(float(value), 6) for key, value in evidence.items()}}
        self.hold = {name: self.measured.current[name].copy() for name in self.arms}
        self._new_stop = True

    def annotation(self) -> dict[str, Any]:
        value: dict[str, Any] = {"trajectory": {
            "mode": self.mode, "knot_index": self.knot,
            "tracking_error_rad": {arm: round(self.errors[arm], 6) for arm in self.arms},
            "over_bound_steps": dict(self.over)}}
        if self._new_stop:
            value["stop"] = dict(self.stop)
        return value

    def tick_record(self) -> dict[str, Any]:
        return {"knot": {"mode": self.mode, "index": self.knot,
                         "targets_rad": {arm: [float(v) for v in self.knots[arm][self.knot]] for arm in self.arms},
                         "max_tracking_error_rad": {arm: round(self.tick_error[arm], 6) for arm in self.arms}},
                **({"stop": dict(self.stop)} if self.stop is not None else {})}


def _admission(session: Any, arms: Sequence[str], knots: Mapping[str, np.ndarray],
               start: Mapping[str, np.ndarray], step_limit: float) -> dict[str, Any] | None:
    """Why the knots cannot be executed as given, or ``None``; no write happens before this passes."""
    from .ik import articulation_limits

    for arm in arms:
        limits = np.asarray(articulation_limits(session, arm), dtype=float)
        outside = np.argwhere((knots[arm] < limits[:, 0] - LIMIT_EPSILON_RAD)
                              | (knots[arm] > limits[:, 1] + LIMIT_EPSILON_RAD))
        if len(outside):
            knot, joint = (int(value) for value in outside[0])
            return {"code": "trajectory_joint_limit", "arm": arm, "knot": knot + 1, "joint": joint,
                    "value_rad": float(knots[arm][knot, joint]), "limits_rad": limits[joint].tolist()}
        first = float(np.max(np.abs(knots[arm][0] - start[arm])))
        if first > step_limit + 1e-9:
            return {"code": "trajectory_start_discontinuous", "arm": arm, "knot": 1, "step_rad": first,
                    "max_joint_step_rad": step_limit}
        if len(knots[arm]) > 1:
            steps = np.max(np.abs(np.diff(knots[arm], axis=0)), axis=1)
            worst = int(np.argmax(steps))
            if float(steps[worst]) > step_limit + 1e-9:
                return {"code": "trajectory_step_exceeds_limit", "arm": arm, "knot": worst + 2,
                        "step_rad": float(steps[worst]), "max_joint_step_rad": step_limit}
    return None


def _settled(source: TrajectorySource, final: Mapping[str, np.ndarray], settle: Mapping[str, Any]) -> dict[str, Any]:
    error = max(float(np.max(np.abs(final[arm] - source.measured.current[arm]))) for arm in source.arms)
    speed = source.measured.speed()
    return {"joint_error_rad": round(error, 6), "joint_speed_rad_s": round(speed, 6),
            "settled": error <= settle["joint_tolerance_rad"] and speed <= settle["joint_speed_rad_s"]}


def execute_trajectory(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    from .sparse_motion import _SparseAction, _remember_backtrack

    arguments = action["arguments"]
    plan = arguments["arms"]
    arms = tuple(arm for arm in ARMS if arm in plan)
    execution = _SparseAction(session, action, arms)
    execution.budget = execution_budget(session)
    knots = {arm: np.asarray(plan[arm]["knots_rad"], dtype=float) for arm in arms}
    count = len(knots[arms[0]])
    start = {arm: np.asarray(execution.before[arm]["arm_joint_qpos"], dtype=float) for arm in arms}
    expected = {arm: plan[arm]["expected_link6_xyz_xyzw"] for arm in arms if "expected_link6_xyz_xyzw" in plan[arm]}
    final_poses = {arm: list(poses[-1]) for arm, poses in expected.items()} or None
    gripper_knots = {arm: plan[arm]["gripper_knots"] for arm in arms if "gripper_knots" in plan[arm]}
    refusal = _admission(session, arms, knots, start, arguments["admission"]["max_joint_step_rad"])
    if refusal is not None:
        return _finish(execution, status="plan_rejected", stage="plan_validation", code=refusal["code"],
                       executor=None, strategy=None, summary={"status": "refused", "admission": refusal, "knots_planned": count,
                                               "knots_executed": 0}, final_poses=final_poses, gripper=bool(gripper_knots))
    recovery = arguments.get("recovery")
    if recovery is None:
        _remember_backtrack(session, execution.before, arms)
    execution.begin([arm_key(session, arm) for arm in arms] + [gripper_key(session, arm) for arm in gripper_knots])
    last = min(count, execution.remaining())
    fingers = {arm: float(execution.before[arm]["gripper_finger_qpos"][0]) for arm in arms}
    expect_contact = bool(arguments.get("expect_contact"))
    guards = dict(arguments["guards"], tracking_growth=None) if expect_contact else arguments["guards"]
    source = TrajectorySource(session, arms, knots, start, fingers, gripper_knots, guards,
                              arguments["gripper"]["max_lead_m"], last)
    source.expect_contact = expect_contact
    action_id = str(action["action_id"])

    def tick(knot: int | None, mode: str) -> None:
        source.begin_tick(knot, mode)
        run_tick(session, action_id=action_id, segment=mode, source=source, held_targets=execution.held_targets,
                 journal=execution.journal, tracking=execution.tracking)
        execution.used += 1

    executed, trace = 0, []
    for knot in range(1, last + 1):
        if session.is_terminal:
            break
        tick(knot, "trajectory")
        executed = knot if source.stop is None or source.stop["physics_step"] == 10 else knot - 1
        trace.append({"knot": knot, **{f"{arm}_max_error_rad": round(source.tick_error[arm], 6) for arm in arms}})
        if source.stop is not None:
            break
    if source.stop is not None and source.stop["physics_step"] == 10:
        # Tripped on the tick's last step: hold where the arm stopped for one
        # journaled tick, or keep that hold as the retained target.
        if execution.remaining() > 0 and not session.is_terminal:
            tick(None, "stop_hold")
        else:
            retained = session.environment.robot_manager.control_manager.prev_control[0]
            for arm in arms:
                retained[source.keys[arm]] = {"position": source.hold[arm].tolist(), "velocity": [0.0] * 6}
    final = {arm: source.knots[arm][count] for arm in arms}
    settle_request = arguments["settle"]
    completed = source.stop is None and executed == count
    settle = None
    if completed:
        settle = {"requested_native_actions": settle_request["native_actions"], "native_actions": 0,
                  **_settled(source, final, settle_request)}
        while (settle_request["native_actions"] and not settle["settled"] and execution.remaining() > 0
               and settle["native_actions"] < settle_request["native_actions"] and not session.is_terminal):
            tick(None, "settle")
            settle.update(_settled(source, final, settle_request), native_actions=settle["native_actions"] + 1)
    rewritten = hold_retained_velocity(session, list(source.keys.values()))
    after = session._proprioception()["end_effectors"]
    part = arguments.get("part_of_move")
    reason = ("terminal" if session.is_terminal and not completed else source.stop["reason"] if source.stop
              else "budget" if executed < count else "part_of_move" if part is not None
              else "not_settled" if settle_request["native_actions"] and not settle["settled"] else None)
    if reason == "not_settled" and final_poses and all(
            pose_reached(final_poses[arm], after[arm]["pose_xyz_xyzw"]) for arm in final_poses):
        # The link6 pose is inside the endpoint acceptance; only the joints did not come to rest (an arm in
        # contact, still converging): reached, with the settle failure named, never an endpoint residual.
        reason = "reached_not_settled"
    for arm in arms:
        session.pose_targets[arm] = (list(final_poses[arm]) if completed and final_poses and arm in final_poses
                                     else list(after[arm]["pose_xyz_xyzw"]))
    for arm, values in gripper_knots.items():
        session.gripper_directives[arm] = float(values[max(executed, 1) - 1])
    arms_summary = {}
    for arm in arms:
        residual = final[arm] - np.asarray(after[arm]["arm_joint_qpos"], dtype=float)
        link6 = None
        if final_poses and arm in final_poses:
            distance, angle = pose_error(final_poses[arm], after[arm]["pose_xyz_xyzw"])
            link6 = {"translation_m": round(distance, 6), "orientation_rad": round(angle, 6)}
        arms_summary[arm] = {"final_joint_error_rad": round(float(np.max(np.abs(residual))), 6),
                             "final_joint_error_per_joint_rad": [round(float(v), 6) for v in residual],
                             "max_tracking_error_rad": round(source.max_error[arm], 6), "final_link6_error": link6}
    summary = {"status": ("completed" if completed else "terminal" if reason == "terminal" else
                          "stopped" if source.stop else "budget"),
               "knots_planned": count, "knots_executed": executed, "stop": source.stop, "settle": settle,
               "arms": arms_summary, "tracking_trace": trace, "guards": _public_guards(source),
               "retained_velocity_zeroed": rewritten, "expect_contact": expect_contact,
               **({"jaw_symmetry": dict(arguments["jaw_symmetry"])} if arguments.get("jaw_symmetry") else {})}
    segments = arguments.get("speed_segments")
    if segments is not None:
        summary["speed_segments"] = {arm: speed_segment_spend(rows, executed) for arm, rows in segments.items()}
    if part is not None:
        # The knots end short of the caller's goal: residuals and the held pose target are the cut point's.
        summary["part_of_move"] = {
            "fraction_of_move": round(float(part["fraction"]), 4), "residuals_against": "cut_point",
            "remaining_to_goal_m": {arm: round(pose_error(goal, after[arm]["pose_xyz_xyzw"])[0], 6)
                                    for arm, goal in part["goals_xyz_xyzw"].items() if arm in after}}
    stacks = getattr(session, "undo_stacks", None)
    undo = None
    if recovery is None:
        remember_measured_path(session, execution.before, arms, execution.tracking)
        undo = record_undo(stacks, getattr(session, "tool_call", None), execution.before, arms, execution.tracking)
    else:
        summary["recovery"] = {"targets": dict(recovery["targets"]), "previous_anchor": "kept"}
        if "retrace" in recovery:
            # Undone when every knot ran (the call's start reached); a cut or stopped retrace stays on top.
            undo = retrace_undo(stacks, recovery["retrace"], completed=completed and part is None, after=after)
    if undo is not None:
        summary["undo"] = undo
    if stacks is not None:
        summary["undo_depth"] = {arm: len(stacks.get(arm, [])) for arm in arms}
    stage, code = STOP_CODES[reason] if reason else (None, None)
    reached = reason in (None, "reached_not_settled")
    return _finish(execution, status="reached" if reached else "not_reached", stage=stage, code=code,
                   executor=[TRAJECTORY_EXECUTOR], summary=summary, final_poses=final_poses,
                   gripper=bool(gripper_knots))


#: One undo entry per arm-moving tool call: the call's trajectories are concatenated (False: one per trajectory).
UNDO_PER_TOOL_CALL = True


def _measured(before: Mapping[str, Any], arm: str, samples: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    return [[float(v) for v in before[arm]["arm_joint_qpos"]]] + [
        [float(v) for v in sample["measured"][arm]["arm_joint_qpos"]] for sample in samples]


def apply_undo(stacks: dict[str, list] | None, undo: Mapping[str, Any], before: Mapping[str, Any],
               samples: Sequence[Mapping[str, Any]]) -> None:
    """Apply one recorded undo operation (``trajectory.undo``) to the undo stacks; live and restore share it.

    ``push``: a new entry (the tool call, the start joints and pose, the measured path: the start, then the
    joints measured at the end of every native action), the oldest beyond ``UNDO_DEPTH`` dropped;
    ``extend``: the same tool call's next trajectory appended to its entry's path; ``pop``: a completed
    retrace removes the top entry; ``cut``: a retrace that stopped short records its progress on the top.
    """
    from ora.contracts.execution_state import UNDO_DEPTH

    if stacks is None:
        return
    for arm, op in undo["arms"].items():
        moves = list(stacks.get(arm, []))
        if op == "push":
            call = undo.get("tool_call") or {}
            moves.append({"tool_call_id": call.get("tool_call_id"), "tool": call.get("tool"),
                          "ordinal": call.get("ordinal"),
                          "arm_joint_qpos": [float(v) for v in before[arm]["arm_joint_qpos"]],
                          "pose_xyz_xyzw": [float(v) for v in before[arm]["pose_xyz_xyzw"]],
                          "measured_path_rad": _measured(before, arm, samples)})
        elif op == "extend":
            moves[-1] = {**moves[-1], "measured_path_rad": moves[-1]["measured_path_rad"] + _measured(
                before, arm, samples)}
        elif op == "pop":
            moves = moves[:-1]
        elif op == "cut":
            moves[-1] = {**moves[-1], "undo_progress": dict(undo["progress"][arm])}
        stacks[arm] = moves[-UNDO_DEPTH:]


def record_undo(stacks: dict[str, list] | None, tool_call: Mapping[str, Any] | None,
                before: Mapping[str, Any], arms: Sequence[str],
                samples: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """A non-recovery trajectory's undo operation (push, or extend the same tool call's entry), applied."""
    if stacks is None:
        return None
    call_id = (tool_call or {}).get("tool_call_id")
    undo = {"tool_call": dict(tool_call) if tool_call else None, "arms": {
        arm: ("extend" if UNDO_PER_TOOL_CALL and call_id is not None and stacks.get(arm)
              and stacks[arm][-1].get("tool_call_id") == call_id else "push") for arm in arms}}
    apply_undo(stacks, undo, before, samples)
    return undo


def _fraction_undone(path: Sequence[Sequence[float]], joints: Sequence[float]) -> float:
    """1 - (path length from the start to the joints' nearest point on the path) / (path length)."""
    rows, here = np.asarray(path, dtype=float), np.asarray(joints, dtype=float)
    lengths = np.linalg.norm(np.diff(rows, axis=0), axis=1)
    total = float(np.sum(lengths))
    if total <= 0:
        return 1.0
    best = (float(np.max(np.abs(rows[0] - here))), 0.0)
    for index, length in enumerate(lengths):
        step = rows[index + 1] - rows[index]
        fraction = 0.0 if length <= 0 else min(1.0, max(0.0, float(np.dot(here - rows[index], step)) / length ** 2))
        distance = float(np.max(np.abs(rows[index] + fraction * step - here)))
        if distance <= best[0] + 1e-12:
            best = (distance, float(np.sum(lengths[:index])) + fraction * float(length))
    return round(1.0 - best[1] / total, 4)


def retrace_undo(stacks: dict[str, list] | None, stated: Mapping[str, Mapping[str, Any]], *, completed: bool,
                 after: Mapping[str, Any]) -> dict[str, Any] | None:
    """A retrace's undo operation, applied: pop the entry it named when it ran every knot, else its progress."""
    if stacks is None:
        return None
    named = [arm for arm, entry in stated.items()
             if stacks.get(arm) and stacks[arm][-1]["arm_joint_qpos"] == list(entry["arm_joint_qpos"])]
    if completed:
        undo = {"arms": {arm: "pop" for arm in named}}
    else:
        undo = {"arms": {arm: "cut" for arm in named}, "progress": {arm: {
            "retrace_calls": int((stacks[arm][-1].get("undo_progress") or {}).get("retrace_calls", 0)) + 1,
            "fraction_undone": _fraction_undone(stacks[arm][-1]["measured_path_rad"], after[arm]["arm_joint_qpos"])}
            for arm in named}}
    apply_undo(stacks, undo, {}, ())
    return undo


def remember_measured_path(session: Any, before: Mapping[str, Any], arms: Sequence[str],
                           samples: Sequence[Mapping[str, Any]]) -> None:
    """The previous anchor's path: its joints, then the joints measured at the end of every native action.

    ``execution_request`` drops a path whose first row is no longer the anchor (a later move without one).
    """
    paths = getattr(session, "last_safe_paths", None)
    if paths is None:
        return
    for arm in arms:
        paths[arm] = [[float(v) for v in before[arm]["arm_joint_qpos"]]] + [
            [float(v) for v in sample["measured"][arm]["arm_joint_qpos"]] for sample in samples]


def _public_guards(source: TrajectorySource) -> dict[str, Any]:
    return {"tracking_window_steps": source.window, "progress": source.progress, "tracking_growth": source.growth,
            "tracking_bound_rad": {arm: {"min": round(float(np.min(source.tracking[arm])), 6),
                                         "max": round(float(np.max(source.tracking[arm])), 6)}
                                   for arm in source.arms}}


def _finish(execution: Any, *, status: str, stage: str | None, code: str | None, executor: list[str] | None,
            summary: Mapping[str, Any], final_poses: Mapping[str, Sequence[float]] | None,
            gripper: bool, strategy: str | None = TRAJECTORY_STRATEGY) -> dict[str, Any]:
    session = execution.session
    summary = {"schema": TRAJECTORY_SCHEMA, "controller_interface": CONTROLLER_INTERFACE,
               "interpolation": execution.action["arguments"]["interpolation"],
               "native_actions": execution.used, **dict(summary)}
    extras = {}
    for arm in execution.arms:
        arm_summary = (summary.get("arms") or {}).get(arm, {})
        part = summary.get("part_of_move")
        outcome = ("refused" if summary["status"] == "refused"
                   else "arrived_not_settled" if status == "reached" and code == "arm_not_settled"
                   else "arrived" if status == "reached"
                   else summary["status"] if summary["status"] != "completed"
                   else CUT_SHORT if part and code == STOP_CODES["part_of_move"][1] else "not_settled")
        execution.segments[arm].append({
            "segment": TRAJECTORY_STRATEGY, "knots_planned": summary["knots_planned"],
            "knots_executed": summary["knots_executed"], "native_actions": execution.used,
            "reached": status == "reached",
            **{key: arm_summary[key] for key in ("final_joint_error_rad", "max_tracking_error_rad") if key in arm_summary}})
        extras[arm] = {"motion_progress": {"schema": PROGRESS_SCHEMA, "segment": TRAJECTORY_STRATEGY,
                                           "outcome": outcome, "native_actions": execution.used,
                                           **({"stop_reason": summary["stop"]["reason"]}
                                              if summary.get("stop") else {}),
                                           **({"expect_contact": True} if summary.get("expect_contact") else {}),
                                           **({"jaw_symmetry": summary["jaw_symmetry"][arm]}
                                              if arm in (summary.get("jaw_symmetry") or {}) else {}),
                                           **({"stop_note": summary["stop"]["note"]}
                                              if (summary.get("stop") or {}).get("note") else {}),
                                           **({"settle_joint_error_rad": (summary.get("settle") or {}).get(
                                               "joint_error_rad")} if code == "arm_not_settled" else {}),
                                           **({"speed_segments": summary["speed_segments"][arm]}
                                              if arm in (summary.get("speed_segments") or {}) else {}),
                                           **({"remaining_to_goal_m": part["remaining_to_goal_m"].get(arm),
                                               "pose_error_against": "cut_point"}
                                              if part and arm in part["remaining_to_goal_m"] else {}),
                                           **({"undo_depth": summary["undo_depth"][arm]}
                                              if arm in (summary.get("undo_depth") or {}) else {})}}
    verified = session._held_targets_unchanged(execution.held_targets)
    return execution.finish(
        status=status, strategy=strategy, stage=stage, code=code, executor=executor,
        target_poses=final_poses, arm_extras=extras,
        extra={"trajectory": summary, "controller_interface": CONTROLLER_INTERFACE,
               "command_invariant": {
                   "schema": COMMAND_INVARIANT_SCHEMA,
                   "controlled_dimension": "joint_trajectory_and_gripper" if gripper else "joint_trajectory",
                   "held_dimension": "other_arms_and_grippers" if gripper else "gripper_directive",
                   "verified": verified, "applied_targets": APPLIED_TARGETS}})


__all__ = [
    "COMMAND_INVARIANT_SCHEMA", "CUT_SHORT", "PROGRESS_SCHEMA", "TRACKING_FLOOR_RAD", "TRACKING_KNOTS", "TRAJECTORY_EXECUTOR",
    "TRAJECTORY_SCHEMA", "TRAJECTORY_STRATEGY", "TrajectorySource", "arm_key", "execute_per_step_action",
    "execute_trajectory", "execution_budget", "gripper_key", "held_gripper_targets", "hold_retained_velocity",
    "UNDO_PER_TOOL_CALL", "apply_undo", "per_step_action", "record_undo", "remember_measured_path", "retrace_undo", "run_tick",
]
