"""``motion_action_v3`` execution for the RoboLab Droid Franka + Robotiq 2F-85.

Two routes. A free move plans (``shared_motion.global_route``): the shared planner's
collision-free joint path, timed and wrist-checked by the shared pacing before anything
moves, then streamed one row per 15 Hz native action as joint targets. A short move
and a contact approach stay on the servo: each native action sends RoboLab's absolute
differential-IK target, the TCP carrot one bounded step toward the next waypoint,
converted to the flange, and the move's end-effector speed (``end_effector_speed_v1``)
lowers that step to the request's caps (``motion_pace.step_caps``), without ramps. The
servo also ends every planned path, and takes a free move when the planner cannot
answer; a planner refusal stops the move. Arrival is judged on the measured TCP.
Receipts use the born-unified sparse layout the Native receipt projector reads.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.wait import wait_receipt_block
from ora.environment.controllers import robolab_droid as D
from ora.environment.controllers.gripper_stall import (
    GRIPPER_STALL_REASON, GripperStallRule, gripper_outcome, gripper_outcome_state, gripper_stalled,
)
from ora.environment.controllers.motion_pace import PaceRefusal, StepCaps, paces, speed_report, step_caps

from . import shared_motion
from .common import SPARSE_RECEIPT_SCHEMA, RobolabError, pose_error, rotvec_error

ARM_EXECUTOR = "robolab_abs_ik_cartesian_servo"
PATH_EXECUTOR = "robolab_planned_joint_path"
GRIPPER_EXECUTOR = "robolab_robotiq_binary_joint"
HOLD_EXECUTOR = "robolab_hold"
GRIPPER_STALL = GripperStallRule(
    still_step=D.GRIPPER_STALL_STILL_STEP, window=D.GRIPPER_STALL_WINDOW,
    minimum_samples=D.GRIPPER_STALL_WINDOW + 1, arrival_tolerance=D.GRIPPER_ARRIVAL_TOLERANCE,
    tick_period_s=1.0 / D.CONTROL_FREQ_HZ, tick_unit="robolab_native_action",
)


def normalized_opening(finger_joint_rad: float) -> float:
    """0 closed, 1 open, on the Robotiq finger joint (0 open, pi/4 closed)."""

    return float(min(1.0, max(0.0, 1.0 - finger_joint_rad / D.GRIPPER_CLOSED_QPOS_RAD)))


def gripper_failure_code(state: str, *, terminal: bool) -> str | None:
    """One public code for a jaw command; ``None`` only when the jaws arrived."""

    if state == "reached_target":
        return None
    if terminal:
        return "episode_terminal_during_execution"
    if state == "stalled_before_target":
        return GRIPPER_STALL_REASON
    return "gripper_not_reached"


def _carrot(current: Sequence[float], goal: Sequence[float], step_m: float, step_rad: float) -> list[float]:
    from scipy.spatial.transform import Rotation

    delta = np.asarray(goal[:3]) - np.asarray(current[:3])
    distance = float(np.linalg.norm(delta))
    if distance > step_m:
        delta *= step_m / distance
    rotvec = rotvec_error(goal[3:7], current[3:7])
    angle = float(np.linalg.norm(rotvec))
    if angle > step_rad:
        rotvec = rotvec * (step_rad / angle)
    orientation = (Rotation.from_rotvec(rotvec) * Rotation.from_quat(list(current[3:7]))).as_quat()
    return [*(np.asarray(current[:3]) + delta).tolist(), *orientation.tolist()]


def free_space_waypoints(current: Sequence[float], target: Sequence[float]) -> list[list[float]]:
    horizontal = math.hypot(target[0] - current[0], target[1] - current[1])
    if horizontal <= D.FREE_SPACE_DIRECT_RADIUS_M:
        return [list(target)]
    height = max(current[2], target[2]) + D.FREE_SPACE_LIFT_M
    return [[current[0], current[1], height, *target[3:7]], [target[0], target[1], height, *target[3:7]], list(target)]


def path_length(start: Sequence[float], waypoints: Sequence[Sequence[float]]) -> tuple[float, float]:
    """The metres and radians from ``start`` through every waypoint in turn."""

    metres, radians, previous = 0.0, 0.0, start
    for waypoint in waypoints:
        translation, rotation = pose_error(waypoint, previous)
        metres, radians, previous = metres + translation, radians + rotation, waypoint
    return metres, radians


def step_limits(interaction: str, caps: StepCaps | None, current: Sequence[float],
                remaining: Sequence[Sequence[float]]) -> tuple[float, float]:
    """One tick's carrot step: the interaction's bound, lowered to the request's caps.

    ``remaining`` are the waypoints still ahead; the last 2 cm along them obey the slow caps."""

    step_m, step_rad = D.STEP_TRANSLATION_M[interaction], D.STEP_ROTATION_RAD[interaction]
    if caps is None:
        return step_m, step_rad
    linear, angular = caps.limits(path_length(current, remaining)[0])
    return min(step_m, linear or math.inf), min(step_rad, angular or math.inf)


def motion_report(speed: Mapping[str, Any] | None) -> dict[str, Any]:
    """``motion`` of a servo receipt: the request and its Cartesian caps. No joint cap applies to the servo."""

    move, slow = paces(speed)
    report = speed_report(speed, move, slow)
    report.pop("joint_speed_deg_s")
    if "slow" in report:
        report["slow"] = {key: value for key, value in report["slow"].items() if key != "joint_speed_deg_s"}
    return {"speed": report}


class MotionRun:
    def __init__(self, session: Any, action: Mapping[str, Any]) -> None:
        self.session, self.bridge, self.action = session, session.bridge, action
        self.before = session.arm_state()
        self.steps_before = session.native_actions
        self.started = time.time()

    def _step(self, target: Sequence[float]) -> bool:
        result = self.bridge.step(list(target), self.session.finger_action)
        self.session.after_native_action(result)
        return not self.session.must_stop()

    def move_arm(self, waypoints: list[list[float]], interaction: str,
                 speed: Mapping[str, Any] | None = None) -> dict[str, Any]:
        # A requested speed caps the carrot step per tick, and the servo gets the ticks those caps need
        # on top of its own budget; the default caps nothing.
        caps = step_caps(speed, tick_s=D.CONTROL_PERIOD_S)
        budget = D.MAX_STEPS
        if caps is not None:
            budget += caps.min_ticks(*path_length(self.bridge.tcp_pose(), waypoints))
        stop_reason, within, index = "budget_exhausted", 0, 0
        anchor: tuple[int, int, float] | None = None
        for used in range(budget):
            current = self.bridge.tcp_pose()
            goal = waypoints[index]
            final = index == len(waypoints) - 1
            translation, rotation = pose_error(goal, current)
            tolerance_m = D.POSITION_TOLERANCE_M if final else 0.02
            tolerance_rad = D.ORIENTATION_TOLERANCE_RAD if final else 0.2
            if translation <= tolerance_m and rotation <= tolerance_rad:
                if not final:
                    index += 1
                    continue
                within += 1
                if within >= D.SETTLE_STEPS:
                    stop_reason = "reached"
                    break
            else:
                within = 0
            if interaction == "contact_approach" and self.bridge.gripper_contact_force() >= D.CONTACT_FORCE_STOP_N:
                stop_reason = "contact_detected"
                break
            if anchor is None or anchor[0] != index:
                anchor = (index, used, translation)
            elif used - anchor[1] >= D.STALL_WINDOW_STEPS:
                if anchor[2] - translation < D.STALL_PROGRESS_M and translation > tolerance_m:
                    stop_reason = "no_progress"
                    break
                anchor = (index, used, translation)
            step_m, step_rad = step_limits(interaction, caps, current, waypoints[index:])
            if not self._step(_carrot(current, goal, step_m, step_rad)):
                stop_reason = "episode_ended"
                break
        return {"stop_reason": stop_reason, "waypoints_total": len(waypoints),
                "waypoints_reached": index + (1 if stop_reason == "reached" else 0)}

    def follow_path(self, rows: Sequence[Sequence[float]]) -> dict[str, Any]:
        """Stream a timed joint path, one row per native action; name how it ended.

        Measured joints lagging a row by ``D.PATH_BLOCKED_LAG_RAD`` for
        ``D.PATH_BLOCKED_TICKS`` ticks mean the arm is held back by something."""

        lagging = 0
        for index, row in enumerate(rows):
            result = self.bridge.step_joints([float(value) for value in row], self.session.finger_action)
            self.session.after_native_action(result)
            if self.session.must_stop():
                return {"stop_reason": "episode_ended", "rows_total": len(rows), "rows_sent": index + 1}
            lag = float(np.max(np.abs(np.asarray(self.bridge.arm_qpos()) - np.asarray(row, dtype=np.float64))))
            lagging = lagging + 1 if lag > D.PATH_BLOCKED_LAG_RAD else 0
            if lagging >= D.PATH_BLOCKED_TICKS:
                return {"stop_reason": "no_progress", "rows_total": len(rows), "rows_sent": index + 1}
        return {"stop_reason": "reached", "rows_total": len(rows), "rows_sent": len(rows)}

    def gripper(self, target: str) -> dict[str, Any]:
        """Drive the jaws until they arrive or stall; name how they ended."""

        self.session.finger_action = D.GRIPPER_CLOSE_ACTION if target == "close" else D.GRIPPER_OPEN_ACTION
        self.session.gripper_target = target
        goal = 0.0 if target == "close" else 1.0
        hold = self.bridge.tcp_pose()
        trace = [normalized_opening(self.bridge.finger_qpos())]
        used = 0
        for used in range(1, D.GRIPPER_STEPS + 1):
            if not self._step(hold):
                break
            trace.append(normalized_opening(self.bridge.finger_qpos()))
            if GRIPPER_STALL.arrived(trace[-1], goal) or gripper_stalled(trace, target=goal, rule=GRIPPER_STALL):
                break
        measured = trace[-1]
        state = gripper_outcome_state(
            reached=GRIPPER_STALL.arrived(measured, goal),
            stalled=gripper_stalled(trace, target=goal, rule=GRIPPER_STALL),
            ticks_used=used, ticks_budget=D.GRIPPER_STEPS)
        return gripper_outcome(
            rule=GRIPPER_STALL, state=state, target=goal, measured=measured,
            ticks_used=used, ticks_budget=D.GRIPPER_STEPS,
            extra={"measured_aperture_mm": self.bridge.jaw_separation_m() * 1000.0,
                   "finger_joint_rad": self.bridge.finger_qpos()})

    def hold(self, ticks: int) -> int:
        target = self.bridge.tcp_pose()
        executed = 0
        for _ in range(ticks):
            if not self._step(target):
                break
            executed += 1
        return executed

    def receipt(self, *, status: str, strategy: str, executor: Sequence[str] | None,
                stage: str | None = None, code: str | None = None,
                target_pose: Sequence[float] | None = None, interaction: str | None = None,
                gripper: Mapping[str, Any] | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        after = self.session.arm_state()
        arm: dict[str, Any] = {"targeted": self.action["action"] != "wait", "interaction": interaction,
                               "pose_error": None, "gripper_error": None, "plan_validation": None}
        if target_pose is not None:
            translation, rotation = pose_error(target_pose, after["right"]["pose_xyz_xyzw"])
            arm["pose_error"] = {"translation_m": translation, "orientation_rad": rotation}
        if gripper is not None:
            arm["gripper_outcome"] = dict(gripper)
        steps = self.session.native_actions - self.steps_before
        return {
            "schema": SPARSE_RECEIPT_SCHEMA, "action_id": self.action["action_id"], "action": self.action["action"],
            "arguments": deepcopy(dict(self.action["arguments"])), "consumed": True, "state_changed": steps > 0,
            "execution_status": status, "motion_strategy": strategy, "failure_stage": stage, "failure_code": code,
            "executor": list(executor) if executor else None,
            "target_arms": [] if self.action["action"] == "wait" else ["right"],
            "synchronization": {"mode": "hold" if self.action["action"] == "wait" else "single_arm",
                                "requested": False, "executed": steps > 0},
            "arms": {"right": arm}, "before": self.before, "after": after,
            "steps_used": steps, "native_actions": steps, "native_actions_total": self.session.native_actions,
            "sim_steps_total": self.session.native_actions * self.session.physics_steps_per_tick,
            "round_index": self.session.round_index, "official_evaluator_queried": True,
            "terminal_after_action": bool(self.session.is_terminal),
            "started_unix_s": self.started, "finished_unix_s": time.time(), **dict(extra or {}),
        }


def _arm_status(stop_reason: str) -> tuple[str, str | None, str | None]:
    if stop_reason in {"reached", "contact_detected"}:
        return "reached", None, None
    if stop_reason == "episode_ended":
        return "not_reached", "execution", "episode_terminal_during_execution"
    if stop_reason == "no_progress":
        return "not_reached", "execution", "controller_stalled"
    return "not_reached", "execution", "execution_step_budget_exhausted"


def _move(run: MotionRun, session: Any, pose: list[float], interaction: str, speed: Mapping[str, Any] | None,
          *, servo_strategy: str | None = None, refuse_reach: bool = False, twin: bool = False) -> dict[str, Any]:
    """Move to ``pose`` by the route it takes; a move the planner or the pacing refuses never starts.

    ``twin``: a pose move, which may run its jaw twin when the wrist check refuses its path (``pace_or_twin``)."""

    current = session.bridge.tcp_pose()
    planner: Mapping[str, Any] | None = None
    if shared_motion.global_route(interaction, current, pose):
        planned = shared_motion.plan(session, pose)
        if planned["status"] == "refused":
            return run.receipt(status="plan_rejected", strategy="none", executor=None, stage="planner",
                               code=str(planned["reason"]), target_pose=pose, interaction=interaction,
                               extra={"planning_diagnostics": planned["planning_diagnostics"]})
        if planned["status"] == "planned":
            return _run_planned(run, session, planned, pose, interaction, speed, twin=twin)
        planner = {"status": "unavailable", "reason": planned["reason"]}
    refusal = session.workspace_refusal(pose) if refuse_reach else None
    if refusal is not None:
        return run.receipt(status="plan_rejected", strategy="none", executor=None, stage="admission",
                           code=refusal, target_pose=pose, interaction=interaction)
    waypoints = free_space_waypoints(current, pose) if interaction == "free_space" else [pose]
    result = run.move_arm(waypoints, interaction, speed)
    status, stage, code = _arm_status(result["stop_reason"])
    strategy = servo_strategy or ("cartesian_servo_lift_traverse" if len(waypoints) > 1 else "cartesian_servo_direct")
    motion = motion_report(speed)
    return run.receipt(status=status, strategy=strategy, executor=[ARM_EXECUTOR], stage=stage, code=code,
                       target_pose=pose, interaction=interaction,
                       extra={"endpoint": result, "motion": motion if planner is None else {**motion, "planner": planner}})


def _run_planned(run: MotionRun, session: Any, planned: Mapping[str, Any], pose: list[float], interaction: str,
                 speed: Mapping[str, Any] | None, *, twin: bool = False) -> dict[str, Any]:
    """Time a planned path, stream it, and let the servo settle the last millimetres.

    A jaw twin the pacing ran instead (``shared_motion.pace_or_twin``) is the pose the servo settles on and the
    receipt judges; its ``motion.wrist.jaw_flip_retry`` says so."""

    try:
        paced, pose, _retried = shared_motion.pace_or_twin(session, planned, pose, speed, twin=twin)
    except PaceRefusal as refusal:
        move, slow = paces(speed)
        return run.receipt(status="plan_rejected", strategy="none", executor=None, stage="pacing", code=refusal.code,
                           target_pose=pose, interaction=interaction,
                           extra={"motion": {"speed": speed_report(speed, move, slow),
                                             "refused": {"code": refusal.code, "message": str(refusal),
                                                         **refusal.constraints}}})
    path = run.follow_path(paced.rows())
    result = path if path["stop_reason"] != "reached" else {**run.move_arm([pose], "precision", speed), "path": path}
    status, stage, code = _arm_status(result["stop_reason"])
    return run.receipt(status=status, strategy="planned_joint_path", executor=[PATH_EXECUTOR, ARM_EXECUTOR],
                       stage=stage, code=code, target_pose=pose, interaction=interaction,
                       extra={"endpoint": result, "motion": paced.report()})


def execute_motion_v3(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    run = MotionRun(session, action)
    name, arguments = action["action"], action["arguments"]
    if name in {"move_pose", "move_free_space"}:
        if "plan_ref" in arguments:
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="planner", code="plan_reference_unsupported")
        if set(arguments["targets"]) != {"right"}:
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="admission", code="arm_not_available")
        target = arguments["targets"]["right"]
        pose = list(target["position_m"]) + list(target["orientation_xyzw"])
        interaction = arguments["interactions"]["right"]
        session.last_safe_pose = session.bridge.tcp_pose()
        return _move(run, session, pose, interaction, arguments.get("speed"), refuse_reach=True, twin=True)
    if name == "set_gripper":
        if set(arguments["targets"]) != {"right"}:
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="admission", code="arm_not_available")
        outcome = run.gripper(arguments["targets"]["right"])
        code = gripper_failure_code(outcome["state"], terminal=bool(session.is_terminal))
        return run.receipt(status="reached" if code is None else "not_reached", strategy="gripper",
                           executor=[GRIPPER_EXECUTOR], stage=None if code is None else "execution",
                           code=code, gripper=outcome)
    if name == "recover_motion":
        strategy = arguments["targets"].get("right")
        pose = session.home_pose if strategy == "home" else session.last_safe_pose
        if pose is None:
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="plan_validation", code="recovery_state_unavailable")
        return _move(run, session, pose, "free_space", None, servo_strategy=f"recover_{strategy}")
    if name == "wait":
        ticks = int(arguments["ticks"])
        executed = run.hold(ticks)
        return run.receipt(
            status="reached" if executed == ticks else "not_reached", strategy="hold", executor=[HOLD_EXECUTOR],
            stage=None if executed == ticks else "execution",
            code=None if executed == ticks else "episode_terminal_during_execution",
            extra={"wait": wait_receipt_block(
                ticks_requested=ticks, physics_steps=executed * session.physics_steps_per_tick,
                tick_unit="robolab_native_action", tick_period_s=1.0 / D.CONTROL_FREQ_HZ,
                terminated_early=executed < ticks, physics_steps_per_tick=session.physics_steps_per_tick)},
        )
    raise RobolabError(f"unsupported motion v3 action on RoboLab: {name!r}")
