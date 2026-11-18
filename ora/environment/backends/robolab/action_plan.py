"""RoboLab's read-only answer to a dry run (``contracts/action_plan.py``): what ``motion.execute_motion_v3`` would run.

A free move's planned path is timed as the real call times it and kept, so the real call runs the path shown; a path the
wrist check refuses is retried to its jaw twin as the real call retries it (``shared_motion.pace_or_twin``), its plan
kept too, and the answer then shows the twin, its path segment naming the retry (``jaw_flip_retry``). A short move and a
contact approach run on the servo, which has no path ahead of time: the answer draws the line it is commanded along,
marked unplanned, as it does for the servo's last millimetres after a planned path.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.contracts.action_plan import body_points, planned, refused, segment, unavailable
from ora.environment.controllers.kept_plans import previewing
from ora.environment.controllers.motion_pace import PaceRefusal

from . import shared_motion
from .motion import free_space_waypoints

SETTLE = "the servo settles the last millimetres"
SERVO = "the servo is commanded along this line; it has no path"


def _move(session: Any, pose: list[float], interaction: str, speed: Mapping[str, Any] | None, *,
          refuse_reach: bool, twin: bool = False) -> dict[str, Any]:
    current = session.bridge.tcp_pose()
    if shared_motion.global_route(interaction, current, pose):
        with previewing(session.motion_planner):
            answer = shared_motion.plan(session, pose)
            if answer["status"] == "refused":
                return refused(str(answer["reason"]), "planner", planning_diagnostics=answer["planning_diagnostics"])
            if answer["status"] == "planned":
                try:
                    paced, goal, retried = shared_motion.pace_or_twin(session, answer, pose, speed, twin=twin)
                except PaceRefusal as refusal:
                    return refused(refusal.code, "pacing", str(refusal), **refusal.constraints)
        if answer["status"] == "planned":
            report = paced.report()
            poses = shared_motion.forward_tcp(session, paced.rows())
            return planned("global", [
                segment("arm", "right", "planned path", planned=True, poses=[current, *poses],
                        native_actions=report["planned_ticks"], duration_s=report["duration_s"],
                        **({"jaw_flip_retry": retried} if retried is not None else {})),
                segment("arm", "right", SETTLE, planned=False, poses=[poses[-1], goal])],
                body={"right": body_points(lambda rows: shared_motion.forward_tcp(session, rows),
                                           session.bridge.arm_qpos())})
    refusal = session.workspace_refusal(pose) if refuse_reach else None
    if refusal is not None:
        return refused(refusal, "admission")
    waypoints = free_space_waypoints(current, pose) if interaction == "free_space" else [pose]
    body = (body_points(lambda rows: shared_motion.forward_tcp(session, rows), session.bridge.arm_qpos())
            if session.motion_planner is not None else ())
    return planned("servo", [segment("arm", "right", SERVO, planned=False, poses=[current, *waypoints])],
                   body={"right": body})


def plan_action(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    name, arguments = action["action"], action["arguments"]
    if name in {"move_pose", "move_free_space"}:
        if "plan_ref" in arguments:
            return refused("plan_reference_unsupported", "planner")
        if set(arguments["targets"]) != {"right"}:
            return refused("arm_not_available", "admission")
        target = arguments["targets"]["right"]
        return _move(session, [*target["position_m"], *target["orientation_xyzw"]], arguments["interactions"]["right"],
                     arguments.get("speed"), refuse_reach=True, twin=True)
    if name == "recover_motion":
        strategy = arguments["targets"].get("right")
        pose = session.home_pose if strategy == "home" else session.last_safe_pose
        if pose is None:
            return refused("recovery_state_unavailable", "plan_validation")
        return _move(session, list(pose), "free_space", None, refuse_reach=False)
    return unavailable(f"{name} moves nothing a plan can answer for")


__all__ = ["plan_action"]
