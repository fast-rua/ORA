"""The YAM station's read-only answer to a dry run (``contracts/action_plan.py``): the schedule the move would run.

The station plans by kinematics, on the robot model, whether the rig is the cell or the real arms; nothing reaches them
here. The schedule is kept, so the real call follows the ticks shown; its key holds the measured joints, so it cannot
outlive a motion.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from ora.environment.controllers import yam_station as Y
from ora.contracts.action_plan import body_points, decimated, planned, refused, segment, unavailable
from ora.environment.controllers.kept_plans import plan_key

from .kinematics import pose_xyz_xyzw
from .motion import Refusal, Schedule, move_plans, plan_joint_motion, plan_motion, recovery_goals

SCHEDULE = "scheduled path"


def _key(session: Any, action: Mapping[str, Any]) -> str:
    return plan_key("schedule", {"action": action["action"], "arguments": action["arguments"],
                                 "joints": {arm: session.rig.arm_q(arm).tolist() for arm in Y.ARMS}})


def scheduled(session: Any, action: Mapping[str, Any], make: Callable[[], Schedule]) -> Schedule:
    """The schedule a dry run of ``action`` kept on this state, else ``make()``."""
    kept = session.kept_schedules.pop(_key(session, action), None)
    if kept is None:
        return make()
    session.ran_previewed_plan = True
    return kept


def _answer(session: Any, action: Mapping[str, Any], schedule: Schedule, route: str) -> dict[str, Any]:
    session.kept_schedules = {_key(session, action): schedule}
    segments = []
    for arm in schedule.finals:
        rows = decimated([tick[arm] for tick in schedule.ticks if arm in tick])
        start = pose_xyz_xyzw(session.kinematics.fk(arm, session.rig.arm_q(arm)))
        segments.append(segment("arm", arm, SCHEDULE, planned=True,
                                poses=[start, *(pose_xyz_xyzw(session.kinematics.fk(arm, np.asarray(row)))
                                                for row in rows)],
                                native_actions=len(schedule.ticks), duration_s=schedule.summary["duration_s"]))
    body = {arm: body_points(lambda rows, selected=arm: [
        pose_xyz_xyzw(session.kinematics.fk(selected, np.asarray(row))) for row in rows], session.rig.arm_q(arm).tolist())
        for arm in schedule.finals}
    return planned(route, segments, body=body, min_inter_arm_distance_m=schedule.summary.get("min_inter_arm_distance_m"))


def plan_action(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    name, arguments = action["action"], action["arguments"]
    try:
        if name in {"move_pose", "move_free_space"}:
            if "plan_ref" in arguments:
                return refused("plan_reference_unsupported", "planner")
            plans, _targets = move_plans(session, arguments)
            return _answer(session, action, plan_motion(session, plans), "scheduled")
        if name == "recover_motion":
            goals = recovery_goals(session, arguments)
            if goals is None:
                return refused("recovery_state_unavailable", "plan_validation")
            return _answer(session, action, plan_joint_motion(session, goals), "joint_line")
    except Refusal as refusal:
        return refused(refusal.code, refusal.stage, **refusal.detail)
    if name == "move_base":
        return refused("mobile_base_unavailable", "admission")
    return unavailable(f"{name} moves nothing a plan can answer for")


__all__ = ["plan_action", "scheduled"]
