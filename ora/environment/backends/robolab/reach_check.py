"""Read-only reach check for RoboLab: the shared IK, per exact TCP pose.

A mixin of ``session.RobolabSession`` plus the backend's ``reach_check_batch``
(the controller contract in ``controllers/reach_check.py``).

* A query is a pose of the public Robotiq TCP in the robot root frame, metres and
  ``xyzw``. It goes to the planner's IK as the ``panda_hand`` pose that puts the
  TCP there (``shared_motion.hand_from_tcp_pose``), seeded from the measured joints
  or, chained, from an earlier query's solution, in the world a plan would see:
  the last observation's depth with the robot removed and the goal hand's region
  cleared. A pose whose fingertips lie below the support is unsolved.
* The answer is judged at the TCP against the arrival bound the moves end on.

It plans no trajectory, and a contact approach or a short move runs on the servo, so
this is not ``move_pose``'s admission (``matches_move_pose_admission: false``), as
on LIBERO. It needs a planner and a current observation; without them the check
refuses and the annotation says unavailable.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.contracts.reach_check import START_UNREACHABLE
from ora.environment.controllers import robolab_droid as D
from ora.contracts.action_plan import is_request

from . import shared_motion
from .action_plan import plan_action
from .common import RobolabError, pose_error

IK_MODEL = "robolab_reach_ik_depth_world_v1"
ARMS = ("right",)
CLAIM_BOUNDARY = (
    "One IK solve per pose by the IK solver of the global planner move_pose uses, seeded from the measured "
    "joints or a chained query's solution, with joint limits, self-collision and a collision world built from "
    "the current depth images of both cameras: the robot removed, the space around the goal hand cleared, a "
    "held object moving with the hand, and space no camera sees counted as free. A pose with a fingertip more "
    "than 1 cm below the supporting surface under it is unsolved. The solution's TCP is checked against the "
    f"{D.POSITION_TOLERANCE_M:g} m / {D.ORIENTATION_TOLERANCE_RAD:g} rad arrival bound. It plans no trajectory "
    "and does not model the servo route, so move_pose can still refuse or fail to reach a pose reported "
    "reachable, and a pose reported unreachable may still be reachable: the seed search is finite."
)


def _blank(reason: str) -> dict[str, Any]:
    return {"reachable": False, "reason": reason, "route": None, "position_error_m": None,
            "orientation_error_rad": None, "joint_limit_margin_rad": None, "joint_delta_rad": None}


def reach_check_batch(session: Any, arm: str, queries: Sequence[Mapping[str, Any]], *,
                      position_tolerance_m: float, orientation_tolerance_rad: float) -> dict[str, Any]:
    """One row per query, in order; nothing is commanded and no physics advances."""
    from ora.environment.controllers.reach_check import joint_delta_rad, joint_limit_margin_rad

    if arm not in ARMS:
        raise RobolabError("RoboLab's reach check exposes only the right arm")
    clock = session.bridge.clock()
    answer = shared_motion.solve_ik(session, [query["pose_xyzw"] for query in queries],
                                    [query["from_index"] for query in queries])
    if answer is None:
        raise RobolabError("the shared IK cannot answer: no planner or no current observation")
    if session.bridge.clock() != clock:
        raise RobolabError("read-only reach check observed the simulation advance")
    measured = session.bridge.arm_qpos()
    limits = list(zip(answer["joint_limits"][0], answer["joint_limits"][1]))
    rows: list[dict[str, Any]] = []
    for query, solved in zip(queries, answer["results"]):
        start = query["from_index"]
        if start is not None and not rows[start]["reachable"]:
            rows.append(_blank(START_UNREACHABLE))
            continue
        if not solved["solved"]:
            rows.append(_blank("ik_failed"))
            continue
        joints = solved["joint_positions"]
        seed = measured if start is None else answer["results"][start]["joint_positions"]
        tcp = shared_motion.tcp_from_hand_pose(solved["tool_pose_base_xyzw"])
        position, rotation = pose_error(query["pose_xyzw"], tcp)
        within = position <= position_tolerance_m and rotation <= orientation_tolerance_rad
        rows.append({"reachable": within, "reason": None if within else "fk_residual_exceeded", "route": None,
                     "position_error_m": position, "orientation_error_rad": rotation,
                     "joint_limit_margin_rad": joint_limit_margin_rad(joints, limits),
                     "joint_delta_rad": joint_delta_rad(joints, seed)})
    return {"ik_model": IK_MODEL, "matches_move_pose_admission": False, "claim_boundary": CLAIM_BOUNDARY,
            "results": rows}


class _Adapter:
    """The one controller-contract method, bound to this session."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def reach_check_batch(self, arm: str, queries: Sequence[Mapping[str, Any]], **tolerances: float):
        return reach_check_batch(self._session, arm, queries, **tolerances)


class RobolabReachCheckMixin:
    """Serve the internal reach check on the read-only IK bridge, and nothing else."""

    def align_plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        from ora.contracts.reach_check import REQUEST_SCHEMA, ReachCheckRequestDrift
        from ora.environment.controllers.reach_check import check_reach

        if is_request(request):  # a dry run's plan of one action
            return plan_action(self, request["action"])
        if not isinstance(request, Mapping) or request.get("schema") != REQUEST_SCHEMA:
            raise RobolabError("RoboLab serves only the internal reach check on the read-only IK bridge")
        try:
            return check_reach(
                request, adapter=_Adapter(self),
                control_profile={"profile": {"arrival_check": {
                    "position_tolerance_m": D.POSITION_TOLERANCE_M,
                    "orientation_tolerance_rad": D.ORIENTATION_TOLERANCE_RAD}}},
                available_arms=ARMS)
        except ReachCheckRequestDrift as error:
            raise RobolabError(str(error)) from error


__all__ = ["CLAIM_BOUNDARY", "IK_MODEL", "RobolabReachCheckMixin", "reach_check_batch"]
