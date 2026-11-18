"""RoboCasa's read-only answer to a dry run (``contracts/action_plan.py``): what ``motion.execute_motion_v3`` would run.

The same admission and the same pacing the real call uses, stopped before the first step, the jaw twin of a path the
wrist check refuses included (``motion.jaw_twin_path``; its segment then names the retry, ``jaw_flip_retry``). Its
plans are kept, so the real call runs the path shown. A move that needs the lift first shows the path its admission
planned from the lifted height (``session._lift_plan``); the real call plans it again from where the lift stops. The
base is a closed-loop servo with no path: its answer is the footprint check the real call makes
(``session.base_path_refusal``). A refusal carries what the real call's receipt does, base suggestions included.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ora.environment.controllers import base_placement, robocasa_omron as C
from ora.contracts.action_plan import body_points, planned, refused, segment, unavailable
from ora.environment.controllers.kept_plans import previewing
from ora.environment.controllers.motion_pace import PaceRefusal

from . import kinematics
from .motion import _paced_plan, jaw_twin_path

#: What a base move's answer says of its footprint when the check passed.
FOOTPRINT_CLEAR = ("clear: along the servo's path neither the base's footprint nor the arm carried at its joints "
                   "meets the depth seen now")


def _arm(session: Any, plan: Mapping[str, Any], pose: list[float], interaction: str,
         speed: Mapping[str, Any] | None, *, twin: bool = False) -> dict[str, Any]:
    if plan["refusal"] is not None:  # the lift's record stays in the receipt, as the real call's projection keeps it
        return refused(plan["code"], "plan_validation",
                       **{key: value for key, value in plan["refusal"].items() if key != "torso_admission"})
    io = session.io
    if plan["torso_height_m"] is not None:
        return _lifted(session, plan, pose)
    retried = None
    try:
        paced = _paced_plan(session, plan, interaction, speed)
    except PaceRefusal as refusal:
        twin_path = jaw_twin_path(session, refusal, pose, interaction, speed) if twin else None
        if twin_path is None:
            return refused(refusal.code, "pacing", str(refusal), **refusal.constraints)
        paced, _goal, retried = twin_path
    report = paced.report()
    poses = kinematics.grip_poses(io.model, io.data, paced.rows())
    body = body_points(lambda rows: kinematics.grip_poses(io.model, io.data, rows), io.arm_qpos())
    return planned("line" if plan["kind"] == "line" else "global", [
        segment("arm", "right", "planned path", planned=True, poses=[io.grip_pose(), *poses],
                native_actions=report["planned_ticks"], duration_s=report["duration_s"],
                **({"jaw_flip_retry": retried} if retried is not None else {}))], body={"right": body})


def _raised(pose: list[float], lift_m: float) -> list[float]:
    return [pose[0], pose[1], pose[2] + lift_m, *pose[3:]]


def _lifted(session: Any, plan: Mapping[str, Any], pose: list[float]) -> dict[str, Any]:
    """The lift, a servo, then the arm path its admission planned and paced from the lifted height."""
    io = session.io
    lift = float(plan.get("lift_m") or 0.0)
    here, up = io.grip_pose(), _raised(io.grip_pose(), lift)
    torso = segment("torso", "right", "the lift moves first", planned=False, poses=[here, up],
                    height_m=float(plan["torso_height_m"]))
    admitted = plan.get("lift_paced")
    if admitted is None:
        return planned("lift_then_arm", [torso, segment("arm", "right", "planned from where the lift stops, once it has",
                                                        planned=False, poses=[up, pose])])
    paced, retried = admitted
    report = paced.report()
    poses = [_raised(item, lift) for item in kinematics.grip_poses(io.model, io.data, paced.rows())]
    return planned("lift_then_arm", [torso, segment(
        "arm", "right", "planned from the lifted height; the real call plans it again from where the lift stops",
        planned=True, poses=[up, *poses], native_actions=report["planned_ticks"], duration_s=report["duration_s"],
        **({"jaw_flip_retry": retried} if retried is not None else {}))])


def plan_action(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    name, arguments = action["action"], action["arguments"]
    if name in {"move_pose", "move_free_space"}:
        if "plan_ref" in arguments:
            return refused("plan_reference_unsupported", "planner")
        if set(arguments["targets"]) != {"right"}:
            return refused("arm_not_available", "admission")
        target = arguments["targets"]["right"]
        pose = [*target["position_m"], *target["orientation_xyzw"]]
        interaction = arguments["interactions"]["right"]
        bounds = session.workspace_refusal(pose)
        if bounds is not None:
            return refused(bounds, "admission",
                           **session.reach_detail(pose, bounds, interaction, arguments.get("speed")))
        with previewing(session.motion_planner):  # the twin's plan is kept too
            plan = session.arm_path_plan(pose, interaction, speed=arguments.get("speed"))
            return _arm(session, plan, pose, interaction, arguments.get("speed"), twin=True)
    if name == "recover_motion":
        strategy = arguments["targets"].get("right")
        joints = session.home_arm_qpos if strategy == "home" else session.last_safe_arm_qpos
        if joints is None:
            return refused("recovery_state_unavailable", "plan_validation")
        with previewing(session.motion_planner):
            plan = session.joint_target_plan(list(joints))
        pose = kinematics.grip_poses(session.io.model, session.io.data, [joints])[0]
        return _arm(session, plan, pose, "free_space", None)
    if name == "move_base":
        return _base(session, arguments["target"])
    return unavailable(f"{name} moves nothing a plan can answer for")


def _base(session: Any, target: Mapping[str, float]) -> dict[str, Any]:
    start = session.io.base_pose()
    travel = math.hypot(float(target["x_m"]) - start[0], float(target["y_m"]) - start[1])
    if travel > C.BASE_MAX_TRAVEL_M:
        return refused("target_too_far", "admission", travel_m=travel, maximum_m=C.BASE_MAX_TRAVEL_M)
    blocked = session.base_path_refusal(target)
    if blocked is not None:
        return refused(base_placement.BASE_PATH_CODE, "admission", **blocked)
    here, there = [*start[:2], 0.0, 0.0, 0.0, 0.0, 1.0], [float(target["x_m"]), float(target["y_m"]), 0.0, 0.0, 0.0, 0.0, 1.0]
    return planned("base_servo", [segment("base", None, "the base servo drives toward this point; it has no path",
                                          planned=False, poses=[here, there], yaw_rad=float(target["yaw_rad"]),
                                          footprint=FOOTPRINT_CLEAR)])


__all__ = ["plan_action"]
