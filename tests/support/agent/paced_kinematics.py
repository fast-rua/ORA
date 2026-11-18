"""The ``pace_path`` answer of a toolkit test's stub kinematics: the shared pacing without its timing worker.

A stub's toy kinematics (joints 0-2 are the link6 position) unless it passes its own ``fk``.
"""
import json

from ora.agent.toolkit.interface.types import Response
from ora.environment.controllers import motion_pace, motion_timing


def toy_fk(rows):
    return [[*row[:3], 0.0, 0.0, 0.0, 1.0] for row in rows]


def paced(request, fk=toy_fk):
    """The ``ora.kinematics-result.v1`` data of one ``pace_path`` request."""
    motion_timing.configure(None)
    common = {"schema": "ora.kinematics-result.v1", "operation": "pace_path", "arm": request["arm"]}
    try:
        path = motion_pace.pace_path(
            request["joints_rad"], fk, speed=request["speed"], tick_s=0.04, joint_speed_ceiling_rad_s=None,
            wrist_joint_index=5, wrist_limits_rad=(-3.14, 3.14), budget_ticks=request["budget_ticks"],
            slow_approach=request["slow_approach"], cut_to_budget=request["cut_to_budget"], arm=request["arm"],
            unwind_check=request["unwind_check"])
    except motion_pace.PaceRefusal as refusal:
        return {**common, "status": "refused", "reason": refusal.code, "message": str(refusal),
                "constraints": dict(refusal.constraints)}
    report = path.report()
    return {**common, "status": "timed", "reason": None, "path_joints": path.rows(),
            "fractions": list(path.knots["fractions"]), "duration_s": report["duration_s"],
            "ticks": report["planned_ticks"], "needed_ticks": int(path.whole_ticks or report["planned_ticks"]),
            "fraction_of_move": float(path.fraction), "slow_from_tick": path.slow_from_knot,
            "speed": dict(report["speed"]), "wrist": dict(report["wrist"])}


def paced_response(request, fk=toy_fk):
    """A stub's whole ``Response`` to one ``pace_path`` request."""
    return Response(json.dumps({"status": "ok", "method": "kinematics", "data": paced(request, fk)}).encode())


__all__ = ["paced", "paced_response", "toy_fk"]
