"""Read-only reach check on the shared planner's IK, per query.

Each query is one IK solve by the planner ``move_pose`` plans with, in the world a
plan to that pose would see, seeded from the measured joints or, for a chained
query, from the solution an earlier query in the batch reached. The planner picks
the path whatever the interaction, so the interaction changes no answer here.
Nothing is written to the plant and no physics advances.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.reach_check import CLAIM_BOUNDARY, IK_MODEL, START_UNREACHABLE
from ora.environment.controllers.reach_check import joint_delta_rad, joint_limit_margin_rad
from ora.environment.controllers.reachability import cartesian_pose_error_twist

from . import reachability, shared_motion
from .reachability import fk_calibrated


def _row(reachable: bool, reason: str | None, *, position: float | None = None, rotation: float | None = None,
         margin: float | None = None, delta: float | None = None) -> dict[str, Any]:
    return {"reachable": reachable, "reason": reason, "route": None, "position_error_m": position,
            "orientation_error_rad": rotation, "joint_limit_margin_rad": margin, "joint_delta_rad": delta}


def _solved(session: Any, arm: str, query: Mapping[str, Any], solution: Sequence[float], seed: Sequence[float],
            limits: Sequence[Sequence[float]], *, tolerances: tuple[float, float]) -> dict[str, Any]:
    error = cartesian_pose_error_twist([float(item) for item in query["pose_xyzw"]],
                                       reachability.fk_link6(session, arm, solution))
    position, rotation = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
    within = position <= tolerances[0] and rotation <= tolerances[1]
    return _row(within, None if within else "fk_residual_exceeded", position=position, rotation=rotation,
                margin=joint_limit_margin_rad(solution, limits), delta=joint_delta_rad(solution, seed))


def _rows(session: Any, arm: str, queries: Sequence[Mapping[str, Any]], *,
          tolerances: tuple[float, float]) -> list[dict[str, Any]]:
    measured = session._proprioception()["end_effectors"][arm]
    if not fk_calibrated(session, arm, measured):
        # The measured link6 frame disagrees with FK of the measured joints: the arm is
        # still settling or the model is off. Asking after the next observation is valid.
        return [_row(False, "fk_calibration_mismatch") for _ in queries]
    answer, unavailable = shared_motion.solve_reach(session, arm, queries)
    if answer is None:
        return [_row(False, unavailable) for _ in queries]
    limits = list(zip(*answer["joint_limits"]))
    rows: list[dict[str, Any]] = []
    for query, solved in zip(queries, answer["results"]):
        start = query["from_index"]
        if solved["skipped"]:
            rows.append(_row(False, START_UNREACHABLE))
        elif not solved["solved"]:
            rows.append(_row(False, solved["refusal"] or "ik_failed"))
        else:
            seed = answer["seed_joints"] if start is None else answer["results"][start]["joint_positions"]
            rows.append(_solved(session, arm, query, solved["joint_positions"], seed, limits, tolerances=tolerances))
    return rows


def reach_check_batch(session: Any, arm: str, queries: Sequence[Mapping[str, Any]], *,
                      position_tolerance_m: float, orientation_tolerance_rad: float) -> dict[str, Any]:
    """One row per query in order; nothing is written and no physics advances."""
    clock = session._clock()
    rows = _rows(session, arm, queries, tolerances=(position_tolerance_m, orientation_tolerance_rad))
    if session._clock() != clock:
        session._fault("reach_check_advanced_physics")
        raise RuntimeError("read-only reach check advanced physics")
    return {"ik_model": IK_MODEL, "matches_move_pose_admission": False, "claim_boundary": CLAIM_BOUNDARY,
            "results": rows}
