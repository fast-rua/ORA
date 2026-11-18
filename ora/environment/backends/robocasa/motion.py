"""``motion_action_v3`` execution for the RoboCasa PandaOmron.

Admission (``session.arm_path_plan``) turns an arm target into a joint path on
the shared motion base. The path is then timed at the grip site for
the move's end-effector speed (``motion_pace``, ``end_effector_speed_v1``,
Yulun's RoboDojo design): one joint target per 20 Hz step, each joint and the
grip site within the interaction's per-tick bounds, and a path that would turn
the hand far past its goal or outlast the action's ticks is refused before
anything moves; a pose move whose path the wrist check refuses runs its jaw
twin instead when the twin is admitted and the jaws are open
(``jaw_twin_path``, ``controllers/jaw_flip_retry.py``). The arm streams those
targets to its absolute joint-position controller, and arrival is judged on
the measured joints and grip site, never on the command. ``move_base`` drives
the holonomic base to a world ``(x, y, yaw)`` (after ASPIRE's
``behavior.navigate_to_pose``) with the arm joints held; the speed options do
not reach it. Receipts use the born-unified sparse
layout that the Native receipt projector already understands.
"""

from __future__ import annotations

import math
import time
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from ora.contracts.wait import wait_receipt_block
from ora.environment.controllers import jaw_flip_retry, robocasa_omron as C
from ora.environment.controllers.base_placement import BASE_PATH_CODE
from ora.environment.controllers.gripper_stall import (
    GRIPPER_STALL_REASON, GripperStallRule, gripper_outcome, gripper_outcome_state, gripper_stalled,
)
from ora.environment.controllers.motion_pace import PacedPath, PaceRefusal, pace_path, paces, speed_report
from ora.environment.controllers.motion_progress import JointBlockedMonitor

from .common import SPARSE_RECEIPT_SCHEMA, RoboCasaError, _quaternion_error_rotvec, pose_error, wrap_angle
from . import kinematics
from .robot import OmronIO

ARM_EXECUTOR = "robocasa_joint_position_path"
BASE_EXECUTOR = "robocasa_base_velocity_servo"
GRIPPER_EXECUTOR = "robocasa_grip_controller"
HOLD_EXECUTOR = "robocasa_hold"
GRIPPER_STALL = GripperStallRule(
    still_step=C.GRIPPER_STALL_STILL_STEP, window=C.GRIPPER_STALL_WINDOW_STEPS,
    minimum_samples=C.GRIPPER_STALL_WINDOW_STEPS + 1, arrival_tolerance=C.GRIPPER_ARRIVAL_TOLERANCE,
    tick_period_s=1.0 / C.CONTROL_FREQ_HZ, tick_unit="robocasa_control_step",
)
#: ``robot0_joint7``, the Panda's last wrist joint, whose turns the pacing reports and checks.
WRIST_JOINT_INDEX = 6
#: The PandaOmron's arm is a Panda: the timing worker certifies its paths at the Panda's jerk limits.
ROBOT = "franka_panda"
#: The ticks a paced arm path may plan: the action's, less the settle allowance after its last target.
ARM_PACE_BUDGET_TICKS = C.ARM_MAX_STEPS - C.ARM_SETTLE_MAX_STEPS
#: How many times a path is timed again, at a lower joint ceiling, to fit the grip-site bound of its interaction.
GRIP_STEP_ATTEMPTS = 8


def gripper_failure_code(state: str, *, terminal: bool) -> str | None:
    """One public code for a jaw command; ``None`` only when the jaws arrived."""

    if state == "reached_target":
        return None
    if terminal:
        return "episode_terminal_during_execution"
    if state == "stalled_before_target":
        return GRIPPER_STALL_REASON
    return "gripper_not_reached"


def grip_step_ratio(start_pose: Sequence[float], poses: Sequence[Sequence[float]], interaction: str) -> float:
    """How far the worst tick of a timed path exceeds the interaction's grip-site bound (at most 1: within it)."""

    import numpy as np

    worst, previous = 0.0, start_pose
    for pose in poses:
        worst = max(worst, math.dist(pose[:3], previous[:3]) / C.ARM_STEP_TRANSLATION_M[interaction],
                    float(np.linalg.norm(_quaternion_error_rotvec(pose[3:7], previous[3:7])))
                    / C.ARM_STEP_ROTATION_RAD[interaction])
        previous = pose
    return worst


def pace_arm_path(start: Sequence[float], path: Sequence[Sequence[float]], fk: Callable[[list[list[float]]], Any], *,
                  speed: Mapping[str, Any] | None, interaction: str, wrist_limits_rad: tuple[float, float]) -> PacedPath:
    """An admitted joint path, ``start`` first, timed for the request; one knot per control tick.

    Every joint keeps the interaction's per-tick bound (``C.ARM_JOINT_STEP_RAD``), the backend's ceiling. A
    timing that breaks the interaction's grip-site bound (``C.ARM_STEP_TRANSLATION_M``, ``C.ARM_STEP_ROTATION_RAD``)
    is timed again at a joint ceiling lowered by the excess, so the bounds of a contact approach hold at every
    speed. Raises ``PaceRefusal`` (a wrist unwinding the long way round, or more than
    ``ARM_PACE_BUDGET_TICKS``); nothing has moved then."""

    rows = [[float(value) for value in row] for row in (start, *path)]
    ceiling = C.ARM_JOINT_STEP_RAD[interaction] / C.CONTROL_PERIOD_S
    start_pose = fk(rows[:1])[0]
    for _attempt in range(GRIP_STEP_ATTEMPTS):
        paced = pace_path(rows, fk, speed=speed, tick_s=C.CONTROL_PERIOD_S, joint_speed_ceiling_rad_s=ceiling,
                          wrist_joint_index=WRIST_JOINT_INDEX, wrist_limits_rad=wrist_limits_rad,
                          budget_ticks=ARM_PACE_BUDGET_TICKS, robot=ROBOT)
        ratio = grip_step_ratio(start_pose, fk(paced.rows()), interaction)
        if ratio <= 1.0 + 1e-9:
            break
        ceiling /= ratio * 1.01
    return paced


class MotionRun:
    """One executed motion action and its evidence."""

    def __init__(self, session: Any, action: Mapping[str, Any]) -> None:
        self.session = session
        self.io: OmronIO = session.io
        self.action = action
        self.before = session.arm_state()
        self.base_before = self.io.base_pose()
        self.steps_before = self.io.control_steps
        self.started = time.time()
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------- stepping
    def _step(self, **command: Any) -> bool:
        """Advance one control step; return False when the episode must stop.

        Once the episode must stop (success latched, horizon reached or
        aborted) no further step is taken: the worker builds robosuite with
        ``ignore_done=True``, so nothing else keeps a motion from stepping
        past the horizon.
        """

        if self.session.must_stop():
            return False
        result = self.io.step(**command)
        self.session.after_control_step(result)
        return not self.session.must_stop()

    # -------------------------------------------------------------- torso
    def move_torso(self, target_m: float) -> dict[str, Any]:
        """Drive the torso lift to ``target_m`` with the arm goal held; name how it ended."""

        start = previous = self.io.torso_height()
        stop_reason, within = "budget_exhausted", 0
        for _ in range(C.TORSO_MAX_STEPS):
            height = self.io.torso_height()
            error = target_m - height
            still = abs(height - previous) <= C.TORSO_STILL_STEP_M
            previous = height
            if abs(error) <= C.TORSO_TOLERANCE_M and still:
                within += 1
                if within >= C.TORSO_SETTLE_STEPS:
                    stop_reason = "reached"
                    break
            else:
                within = 0
            if not self._step(torso_delta_m=error):
                stop_reason = "episode_ended"
                break
        return {"stop_reason": stop_reason, "from_m": start, "to_m": self.io.torso_height(), "target_m": target_m}

    # ---------------------------------------------------------------- arm
    def follow_path(self, knots: list[list[float]], interaction: str) -> dict[str, Any]:
        """Stream a paced joint path as joint targets, one per tick; name how it ended.

        ``knots`` come from ``pace_arm_path``, which already kept the
        interaction's per-tick bounds and the requested speed. The arm has
        arrived once the measured joints stay near the last knot for
        ``C.ARM_SETTLE_STEPS`` ticks (``settled``). It is held back by something
        when the measured joints trail the target by ``C.ARM_BLOCKED_LAG_RAD``
        for ``C.ARM_BLOCKED_STEPS`` ticks (``joint_lag``), or, once the last
        knot is sent, when they come to rest short of it (``settle_stalled``:
        ``C.ARM_STILL_STEPS`` ticks each moving less than
        ``C.ARM_STILL_STEP_RAD``, ``JointBlockedMonitor``'s rule) or are still
        unsettled ``C.ARM_SETTLE_MAX_STEPS`` ticks after it (``settle_timeout``).
        Those three stops name the joint then farthest from its target
        (``lagging_joint``, 1 is the base joint, and ``lag_rad``), and a lag stop
        how far along its knots the target was (``path_fraction``).
        """

        import numpy as np

        samples = [np.asarray(sample, dtype=np.float64) for sample in knots]
        final = commanded = samples[-1]
        index, used, within, lagging, waited = 0, 0, 0, 0, 0
        stop_reason = "budget_exhausted"
        measured = np.asarray(self.io.arm_qpos(), dtype=np.float64)
        # Fed every tick against the target it held; a new target resets its count,
        # so it counts only ticks against the last knot once that has been sent.
        at_rest = JointBlockedMonitor({"right": measured}, consecutive_physics_steps=C.ARM_STILL_STEPS,
                                      maximum_joint_step_rad=C.ARM_STILL_STEP_RAD)
        while used < C.ARM_MAX_STEPS:
            if index < len(samples):
                commanded = samples[index]
                index += 1
            used += 1
            if not self._step(arm_joints=commanded):
                stop_reason = "episode_ended"
                break
            measured = np.asarray(self.io.arm_qpos(), dtype=np.float64)
            if interaction == "contact_approach" and self.io.contact_forces()["gripper"] >= C.CONTACT_FORCE_STOP_N:
                stop_reason = "contact_detected"
                break
            lagging = lagging + 1 if float(np.max(np.abs(measured - commanded))) > C.ARM_BLOCKED_LAG_RAD else 0
            if lagging >= C.ARM_BLOCKED_STEPS:
                stop_reason = "joint_lag"
                break
            resting = at_rest.sample({"right": measured}, {"right": commanded}, tolerance_rad=C.ARM_JOINT_SETTLE_RAD,
                                     stage_index=index, planned_stages=len(samples), waypoint_wait_steps=())
            if index >= len(samples):
                waited += 1
                if float(np.max(np.abs(measured - final))) <= C.ARM_JOINT_SETTLE_RAD:
                    within += 1
                    if within >= C.ARM_SETTLE_STEPS:
                        stop_reason = "settled"
                        break
                else:
                    within = 0
                # Held short of the last target, at rest or past the wait: arrived
                # if the grip site is on target anyway, otherwise blocked.
                if resting is not None:
                    stop_reason = "settle_stalled"
                    break
                if waited >= C.ARM_SETTLE_MAX_STEPS:
                    stop_reason = "settle_timeout"
                    break
        result: dict[str, Any] = {"stop_reason": stop_reason, "path_samples": len(samples),
                                  "samples_commanded": index, "ticks": used}
        if stop_reason in {"joint_lag", "settle_stalled", "settle_timeout"}:
            behind = np.abs(measured - commanded)
            joint = int(np.argmax(behind))
            if stop_reason == "joint_lag":
                result["path_fraction"] = round(index / len(samples), 3)
            result.update(lagging_joint=joint + 1, lag_rad=round(float(behind[joint]), 6))
        if stop_reason in {"joint_lag", "contact_detected", "settle_stalled", "settle_timeout"}:
            # Stop pushing: hold the arm where it is.
            self.io.hold_measured_arm()
            self._step()
        return result

    def gripper(self, target: str) -> dict[str, Any]:
        """Drive the jaws until they arrive or stall; name how they ended."""

        command = C.GRIPPER_CLOSE_COMMAND if target == "close" else C.GRIPPER_OPEN_COMMAND
        goal = 0.0 if target == "close" else 1.0
        self.session.gripper_target = target
        trace = [self.io.gripper_opening()]
        used = 0
        for used in range(1, C.GRIPPER_STEPS + 1):
            if not self._step(gripper=command):
                break
            trace.append(self.io.gripper_opening())
            if GRIPPER_STALL.arrived(trace[-1], goal) or gripper_stalled(trace, target=goal, rule=GRIPPER_STALL):
                break
        measured = trace[-1]
        state = gripper_outcome_state(
            reached=GRIPPER_STALL.arrived(measured, goal),
            stalled=gripper_stalled(trace, target=goal, rule=GRIPPER_STALL),
            ticks_used=used, ticks_budget=C.GRIPPER_STEPS)
        return gripper_outcome(
            rule=GRIPPER_STALL, state=state, target=goal, measured=measured,
            ticks_used=used, ticks_budget=C.GRIPPER_STEPS,
            extra={"measured_aperture_mm": measured * C.MAXIMUM_APERTURE_M * 1000.0})

    # ---------------------------------------------------------------- base
    def move_base(self, target: Mapping[str, float]) -> dict[str, Any]:
        """Drive the base body origin to a world ``(x, y, yaw)``.

        The servo steers the yaw-hinge anchor (slides move it, the hinge
        does not), so translation and rotation decouple; each slide command
        carries the measured friction offset. It stops on arrival, on base
        contact (``base_contact``) or, every tick outside the tolerance, on less
        than ``C.BASE_STALL_PROGRESS_M`` of progress over the last
        ``C.BASE_STALL_WINDOW_STEPS`` ticks (``no_progress``).
        """

        goal = (float(target["x_m"]), float(target["y_m"]), float(target["yaw_rad"]))
        start = self.io.base_pose()
        travel = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if travel > C.BASE_MAX_TRAVEL_M:
            return {"stop_reason": "target_too_far", "travel_m": travel, "refused": True}
        offset = self.io.anchor_offset()
        goal_anchor = (
            goal[0] + math.cos(goal[2]) * offset[0] - math.sin(goal[2]) * offset[1],
            goal[1] + math.sin(goal[2]) * offset[0] + math.cos(goal[2]) * offset[1],
        )
        axes = self.io.slide_axes()
        stop_reason = "budget_exhausted"
        within = 0
        # The progress measure of this tick and the BASE_STALL_WINDOW_STEPS before it, oldest first.
        progress: deque[float] = deque(maxlen=C.BASE_STALL_WINDOW_STEPS + 1)
        for used in range(C.BASE_MAX_STEPS):
            x, y, yaw = self.io.base_pose()
            ax, ay = self.io.base_anchor()
            dx, dy = goal_anchor[0] - ax, goal_anchor[1] - ay
            anchor_distance = math.hypot(dx, dy)
            origin_distance = math.hypot(goal[0] - x, goal[1] - y)
            dyaw = wrap_angle(goal[2] - yaw)
            if origin_distance <= C.BASE_POSITION_TOLERANCE_M and abs(dyaw) <= C.BASE_YAW_TOLERANCE_RAD:
                within += 1
                if within >= C.BASE_SETTLE_STEPS:
                    stop_reason = "reached"
                    break
            else:
                within = 0
            if used > 2 and self.io.contact_forces()["base"] > C.BASE_CONTACT_STOP_N:
                stop_reason = "base_contact"
                break
            progress.append(anchor_distance + 0.3 * abs(dyaw))
            if (within == 0 and len(progress) == progress.maxlen
                    and progress[0] - progress[-1] < C.BASE_STALL_PROGRESS_M):
                stop_reason = "no_progress"
                break
            speed = min(C.BASE_MAX_LINEAR_SPEED_M_S, C.BASE_LINEAR_GAIN * anchor_distance)
            vx, vy = ((dx / anchor_distance) * speed, (dy / anchor_distance) * speed) if anchor_distance > 1e-4 else (0.0, 0.0)
            command = [0.0, 0.0]
            for index, axis in enumerate(axes):
                along = vx * axis[0] + vy * axis[1]
                if abs(along) > 1e-3:
                    along += math.copysign(C.BASE_SLIDE_DEADBAND_COMMAND, along)
                command[index] = along
            wx = command[0] * axes[0][0] + command[1] * axes[1][0]
            wy = command[0] * axes[0][1] + command[1] * axes[1][1]
            yaw_rate = max(-C.BASE_MAX_YAW_RATE_RAD_S, min(C.BASE_MAX_YAW_RATE_RAD_S, C.BASE_YAW_GAIN * dyaw))
            if not self._step(base_velocity_world=(wx, wy, yaw_rate)):
                stop_reason = "episode_ended"
                break
        for _ in range(3):  # brake to zero velocity; no tick once the episode has ended
            if not self._step(base_velocity_world=(0.0, 0.0, 0.0)):
                break
        end = self.io.base_pose()
        return {
            "stop_reason": stop_reason,
            "travel_m": math.hypot(end[0] - start[0], end[1] - start[1]),
            "moved": {"x_m": end[0] - start[0], "y_m": end[1] - start[1],
                      "yaw_rad": wrap_angle(end[2] - start[2])},
            "base_error": {"position_m": math.hypot(goal[0] - end[0], goal[1] - end[1]),
                           "yaw_rad": abs(wrap_angle(goal[2] - end[2]))},
        }

    def hold(self, ticks: int) -> int:
        executed = 0
        for _ in range(ticks):
            if not self._step():
                break
            executed += 1
        return executed

    # ------------------------------------------------------------- receipt
    def receipt(
        self, *, status: str, strategy: str, executor: Sequence[str] | None,
        stage: str | None = None, code: str | None = None,
        target_pose: Sequence[float] | None = None, interaction: str | None = None,
        gripper: Mapping[str, Any] | None = None, base: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        after = self.session.arm_state()
        arm: dict[str, Any] = {
            "targeted": self.action["action"] not in {"wait", "move_base"},
            "interaction": interaction,
            "pose_error": None,
            "gripper_error": None,
            "plan_validation": None,
        }
        if target_pose is not None:
            translation, rotation = pose_error(target_pose, after["right"]["pose_xyz_xyzw"])
            arm["pose_error"] = {"translation_m": translation, "orientation_rad": rotation}
        if gripper is not None:
            arm["gripper_outcome"] = dict(gripper)
        steps = self.io.control_steps - self.steps_before
        receipt = {
            "schema": SPARSE_RECEIPT_SCHEMA,
            "action_id": self.action["action_id"],
            "action": self.action["action"],
            "arguments": deepcopy(dict(self.action["arguments"])),
            "consumed": True,
            "state_changed": steps > 0,
            "execution_status": status,
            "motion_strategy": strategy,
            "failure_stage": stage,
            "failure_code": code,
            "executor": list(executor) if executor else None,
            "target_arms": [] if self.action["action"] in {"wait", "move_base"} else ["right"],
            "synchronization": {"mode": "hold" if self.action["action"] == "wait" else "single_arm",
                                "requested": False, "executed": steps > 0},
            "arms": {"right": arm},
            "before": self.before,
            "after": after,
            "base_before": {"pose_xy_yaw": self.base_before},
            "base_after": {"pose_xy_yaw": self.io.base_pose()},
            **({"base": dict(base)} if base is not None else {}),
            "steps_used": steps,
            # One native action per control tick: the count the policy's
            # native-action allowance spends (orchestrator/episode/native_allowance.py).
            "native_actions": steps,
            "control_steps_total": self.io.control_steps,
            "sim_steps_total": self.io.control_steps,
            "round_index": self.session.round_index,
            "official_evaluator_queried": True,
            "terminal_after_action": bool(self.session.is_terminal),
            "started_unix_s": self.started,
            "finished_unix_s": time.time(),
            **dict(extra or {}),
        }
        return receipt


#: The stops of the wait after the last knot that judge the grip site: reached on target, else blocked
#: (``settled`` judges it too, as ``endpoint_residual_exceeded``).
_HELD_SHORT = frozenset({"settle_stalled", "settle_timeout"})


def _arm_status(stop_reason: str, endpoint_ok: bool = True) -> tuple[str, str | None, str | None]:
    """Status, stage and code of an arm stop. ``failure_code`` ``blocked`` has three stops, which
    ``endpoint.stop_reason`` names: ``joint_lag``, ``settle_stalled`` and ``settle_timeout``."""

    if stop_reason == "contact_detected" or (stop_reason in {"settled", *_HELD_SHORT} and endpoint_ok):
        return "reached", None, None
    if stop_reason in _HELD_SHORT:
        return "not_reached", "execution", "blocked"
    if stop_reason == "settled":
        return "not_reached", "endpoint_verification", "endpoint_residual_exceeded"
    if stop_reason == "episode_ended":
        return "not_reached", "execution", "episode_terminal_during_execution"
    if stop_reason in {"joint_lag", "no_progress"}:
        return "not_reached", "execution", "blocked" if stop_reason == "joint_lag" else "controller_stalled"
    return "not_reached", "execution", "execution_step_budget_exhausted"


def _paced_plan(session: Any, plan: Mapping[str, Any], interaction: str, speed: Mapping[str, Any] | None) -> PacedPath:
    """The plan's joint path timed for the move's speed, at the measured torso height."""

    io = session.io
    return pace_arm_path(
        io.arm_qpos(), plan["path"], lambda rows: kinematics.grip_poses(io.model, io.data, rows),
        speed=speed, interaction=interaction, wrist_limits_rad=kinematics.joint_range(io.model, C.ARM_JOINTS[WRIST_JOINT_INDEX]),
    )


def jaw_twin_path(session: Any, refusal: PaceRefusal, pose: list[float], interaction: str,
                  speed: Mapping[str, Any] | None) -> tuple[PacedPath, list[float], dict[str, Any]] | None:
    """The move to ``pose`` whose path ``refusal`` stopped, admitted and paced once more to its jaw twin.

    ``controllers/jaw_flip_retry.py`` says when: a wrist refusal, with the jaws commanded open
    (``io.gripper_directive``). The twin (the grip site turned a half-turn about its approach, local +z) must be
    admitted on the arm alone at the measured torso height and paced like the request. Returns the paced path, the
    twin pose and the retry's record, which the path's wrist report carries, or None when the refusal stands. A
    real call and its dry run (``action_plan.py``) both ask this, for a pose move only."""

    def admit(goal: list[float]) -> Any:
        again = session.arm_path_plan(goal, interaction, allow_torso=False)
        return str(again["code"]) if again["refusal"] is not None else _paced_plan(session, again, interaction, speed)

    retried = jaw_flip_retry.retry(pose, refusal, admit,
                                   jaws_open=lambda: session.io.gripper_directive == C.GRIPPER_OPEN_COMMAND)
    if retried is None:
        return None
    goal, paced, record = retried
    return jaw_flip_retry.recorded(paced, record), goal, record


def _pacing_refusal(speed: Mapping[str, Any] | None, interaction: str, refusal: PaceRefusal) -> dict[str, Any]:
    """``motion`` of a receipt whose path the pacing refused: the request, its caps and the numbers."""

    ceiling = C.ARM_JOINT_STEP_RAD[interaction] / C.CONTROL_PERIOD_S
    move, slow = paces(speed)
    return {"speed": speed_report(speed, move.within(ceiling), None if slow is None else slow.within(ceiling)),
            "refused": {"code": refusal.code, "message": str(refusal), **refusal.constraints}}


def _run_plan(run: MotionRun, session: Any, plan: Mapping[str, Any], pose: list[float], interaction: str,
              strategy: str, *, speed: Mapping[str, Any] | None = None, checkpoint: bool = False,
              twin: bool = False) -> dict[str, Any]:
    """Pace and execute an admitted plan: the torso first when it needs to move, then the arm path.

    The pacing runs before anything moves. ``checkpoint`` remembers the joints a move starts from once it is
    admitted, the ``previous`` of ``recover_pose``. ``twin``: a pose move, which runs its jaw twin when the wrist
    check refuses its path (``jaw_twin_path``); the twin is then the pose arrival and the receipt are judged by."""

    if plan["refusal"] is not None:
        return run.receipt(status="plan_rejected", strategy="none", executor=None, stage="plan_validation",
                           code=plan["code"], target_pose=pose, interaction=interaction,
                           extra={"reach_check": plan["refusal"],
                                  **({"planning_diagnostics": plan["refusal"]["planning_diagnostics"]}
                                     if "planning_diagnostics" in plan["refusal"] else {})})
    torso: dict[str, Any] = {}
    if plan["torso_height_m"] is None:
        try:
            paced = _paced_plan(session, plan, interaction, speed)
        except PaceRefusal as refusal:
            retried = jaw_twin_path(session, refusal, pose, interaction, speed) if twin else None
            if retried is None:
                return run.receipt(status="plan_rejected", strategy="none", executor=None, stage="pacing",
                                   code=refusal.code, target_pose=pose, interaction=interaction,
                                   extra={"motion": _pacing_refusal(speed, interaction, refusal)})
            paced, pose, _record = retried
        if checkpoint:
            session.remember_safe_joints()
    else:
        # The arm alone cannot reach at this torso height: the lift moves first
        # with the arm joints held, and the path is planned from up there, at
        # the measured height (it settles a few millimetres from its target)
        # and in the world seen from it.
        if checkpoint:
            session.remember_safe_joints()
        torso = {"torso": run.move_torso(plan["torso_height_m"])}
        if torso["torso"]["stop_reason"] != "reached":
            status, stage, code = _arm_status(
                "episode_ended" if torso["torso"]["stop_reason"] == "episode_ended" else "no_progress")
            return run.receipt(status=status, strategy=strategy, executor=[ARM_EXECUTOR],
                               stage=stage, code=code, target_pose=pose, interaction=interaction, extra=torso)
        replanned = session.arm_path_plan(pose, interaction, allow_torso=False)
        torso["torso"]["arm_path_replanned"] = replanned["refusal"] is None
        if replanned["refusal"] is not None:
            return run.receipt(status="not_reached", strategy=strategy, executor=[ARM_EXECUTOR],
                               stage="plan_validation", code=replanned["code"], target_pose=pose,
                               interaction=interaction, extra={**torso, "reach_check": replanned["refusal"]})
        try:
            paced = _paced_plan(session, replanned, interaction, speed)
        except PaceRefusal as refusal:
            retried = jaw_twin_path(session, refusal, pose, interaction, speed) if twin else None
            if retried is None:
                return run.receipt(status="not_reached", strategy=strategy, executor=[ARM_EXECUTOR], stage="pacing",
                                   code=refusal.code, target_pose=pose, interaction=interaction,
                                   extra={**torso, "motion": _pacing_refusal(speed, interaction, refusal)})
            paced, pose, _record = retried
    result = run.follow_path(paced.rows(), interaction)
    translation, rotation = pose_error(pose, session.io.grip_pose())
    endpoint_ok = translation <= C.ARM_POSITION_TOLERANCE_M and rotation <= C.ARM_ORIENTATION_TOLERANCE_RAD
    status, stage, code = _arm_status(result["stop_reason"], endpoint_ok)
    return run.receipt(status=status, strategy=strategy, executor=[ARM_EXECUTOR], stage=stage, code=code,
                       target_pose=pose, interaction=interaction,
                       extra={"endpoint": result, **torso, "motion": paced.report()})


def execute_motion_v3(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    """Run one validated ``motion_action_v3`` proposal; return its receipt."""

    run = MotionRun(session, action)
    name = action["action"]
    arguments = action["arguments"]
    if name in {"move_pose", "move_free_space"}:
        if "plan_ref" in arguments:
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="planner", code="plan_reference_unsupported")
        targets = arguments["targets"]
        if set(targets) != {"right"}:
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="admission", code="arm_not_available")
        target = targets["right"]
        pose = list(target["position_m"]) + list(target["orientation_xyzw"])
        interaction = arguments["interactions"]["right"]
        bounds = session.workspace_refusal(pose)
        if bounds is not None:
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="admission", code=bounds, target_pose=pose, interaction=interaction,
                               extra={"reach_check": session.reach_detail(pose, bounds, interaction,
                                                                          arguments.get("speed"))})
        plan = session.arm_path_plan(pose, interaction, speed=arguments.get("speed"))
        strategy = "ik_line" if plan["kind"] == "line" else "joint_path"
        return _run_plan(run, session, plan, pose, interaction, strategy, speed=arguments.get("speed"),
                         checkpoint=True, twin=True)
    if name == "set_gripper":
        targets = arguments["targets"]
        if set(targets) != {"right"}:
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="admission", code="arm_not_available")
        outcome = run.gripper(targets["right"])
        code = gripper_failure_code(outcome["state"], terminal=bool(session.is_terminal))
        return run.receipt(status="reached" if code is None else "not_reached", strategy="gripper",
                           executor=[GRIPPER_EXECUTOR], stage=None if code is None else "execution",
                           code=code, gripper=outcome)
    if name == "recover_motion":
        strategy = arguments["targets"].get("right")
        joints = session.home_arm_qpos if strategy == "home" else session.last_safe_arm_qpos
        if joints is None:
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="plan_validation", code="recovery_state_unavailable")
        plan = session.joint_target_plan(list(joints))
        pose = kinematics.grip_poses(session.io.model, session.io.data, [joints])[0]
        return _run_plan(run, session, plan, pose, "free_space", f"recover_{strategy}")
    if name == "wait":
        ticks = int(arguments["ticks"])
        executed = run.hold(ticks)
        return run.receipt(
            status="reached" if executed == ticks else "not_reached", strategy="hold",
            executor=[HOLD_EXECUTOR], stage=None if executed == ticks else "execution",
            code=None if executed == ticks else "episode_terminal_during_execution",
            extra={"wait": wait_receipt_block(
                ticks_requested=ticks, physics_steps=executed * session.physics_steps_per_tick,
                tick_unit="robocasa_control_step", tick_period_s=1.0 / C.CONTROL_FREQ_HZ,
                terminated_early=executed < ticks,
                physics_steps_per_tick=session.physics_steps_per_tick,
            )},
        )
    if name == "move_base":
        blocked = session.base_path_refusal(arguments["target"])
        if blocked is not None:  # the footprint would meet the depth on the way: nothing moves
            return run.receipt(status="plan_rejected", strategy="none", executor=None, stage="admission",
                               code=BASE_PATH_CODE, extra={"reach_check": blocked})
        result = run.move_base(arguments["target"])
        if result.get("refused"):
            return run.receipt(status="plan_rejected", strategy="none", executor=None,
                               stage="admission", code=result["stop_reason"], base=result)
        reached = result["stop_reason"] == "reached"
        code = None if reached else {
            "base_contact": "blocked", "no_progress": "controller_stalled",
            "episode_ended": "episode_terminal_during_execution",
        }.get(result["stop_reason"], "execution_step_budget_exhausted")
        return run.receipt(status="reached" if reached else "not_reached", strategy="base_servo",
                           executor=[BASE_EXECUTOR], stage=None if reached else "execution",
                           code=code, base=result)
    raise RoboCasaError(f"unsupported motion v3 action: {name!r}")
