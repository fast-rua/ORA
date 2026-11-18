"""Read-only kinematics query shared by Agent and Environment (``ora.kinematics-request.v1``).

The toolkit lib plans arm motion itself (path, timing, inverse kinematics
along the path, checks) and needs the backend's own kinematic model for it.
This query answers, without advancing physics or writing a controller:

* ``ik_chain``: one arm's joint solution for each pose of a sequence, each
  seeded from the previous solution (continuity), with the per-pose step,
  joint-limit margin, FK residual and, on request, a straight-joint-path
  screen of each step against the arm itself and the other arm at its
  measured joints;
* ``fk``: link6 poses of joint vectors;
* ``screen_path``: the arms' joint polylines screened in lockstep for
  self-collision, the table top and inter-arm clearance (status ``clear``,
  ``self_collision``, ``table_collision`` or ``inter_arm_collision`` at
  ``first_blocked_segment``), the re-check of a planned path;
* ``plan_path``: one arm's collision-free joint path from its measured joints
  to a link6 pose or to joints, by the shared motion planner in the collision
  world of the current public depth, with the other arm held where it is
  (status ``planned``, ``refused`` with one neutral ``reason``, or
  ``unavailable``);
* ``pace_path``: one arm's joint path timed for a speed request, wrist-checked and
  fitted to a budget, one joint row per native action (status ``timed``,
  ``refused`` with ``reason``, its message and constraints);
* ``contact_path``: whether a deliberate contact's joint path keeps every link
  but the fingers out of that world (status ``clear`` or ``blocked``).

It travels over the read-only IK bridge exactly as the reach check does
(``align_plan`` routes on the request ``schema``), so the worker, RPC and
coordinator carry it unchanged; the agent calls it as the interface method
``kinematics``. A backend without the operation refuses with
``unsupported_operation``. Units: radians, metres, xyzw quaternions in the
episode-local world frame.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .motion_speed import validate_speed
from .refusal import EnvironmentRefusal

REQUEST_SCHEMA = "ora.kinematics-request.v1"
RESULT_SCHEMA = "ora.kinematics-result.v1"
OPERATIONS = ("ik_chain", "fk", "screen_path", "plan_path", "pace_path", "contact_path")
#: One request answers at most this many poses or joint vectors.
MAXIMUM_POSES = 256
#: A joint path to time or check holds at most this many rows.
MAXIMUM_PATH_ROWS = 20000
RANKINGS = ("nearest", "canonical_first")
SCREENS = ("none", "self", "self_and_measured_other_arm")
ROW_STATUSES = ("ok", "ik_failed", "discontinuous", "path_blocked")
ARMS = ("left", "right")
UNSUPPORTED = "unsupported_operation"

_FIELDS = {
    "ik_chain": {"schema", "operation", "arm", "seed_joints_rad", "poses_xyz_xyzw", "ranking",
                 "max_joint_step_rad", "screen"},
    "fk": {"schema", "operation", "arm", "joints_rad"},
    "screen_path": {"schema", "operation", "knots_rad"},
    "plan_path": {"schema", "operation", "arm", "goal_pose_xyz_xyzw", "goal_joints"},
    "pace_path": {"schema", "operation", "arm", "joints_rad", "speed", "slow_approach", "budget_ticks",
                  "cut_to_budget", "unwind_check"},
    "contact_path": {"schema", "operation", "arm", "joints_rad"},
}


class KinematicsRequestDrift(ValueError):
    """The request is not a kinematics request this version knows."""


def _refuse(message: str, field: str, received: Any, allowed: Any) -> EnvironmentRefusal:
    return EnvironmentRefusal(message, field=field, received=received, allowed=allowed)


def _rows(value: Any, field: str, width: int, maximum: int = MAXIMUM_POSES) -> list[list[float]]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise _refuse(f"{field} must hold 1..{maximum} rows", field,
                      len(value) if isinstance(value, list) else value, [f"1..{maximum} rows"])
    result = []
    for index, row in enumerate(value):
        if (not isinstance(row, list) or len(row) != width
                or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
                       for item in row)):
            raise _refuse(f"{field}[{index}] must be {width} finite numbers", f"{field}[{index}]", row,
                          [f"{width} finite numbers"])
        result.append([float(item) for item in row])
    return result


def validate_request(request: Any) -> dict[str, Any]:
    """The normalized request, ``KinematicsRequestDrift`` or a public ``EnvironmentRefusal``."""
    if not isinstance(request, Mapping) or request.get("schema") != REQUEST_SCHEMA:
        raise KinematicsRequestDrift("not a kinematics request")
    operation = request.get("operation")
    if operation not in OPERATIONS:
        raise _refuse("operation is not a kinematics operation", "operation", operation, list(OPERATIONS))
    if set(request) != _FIELDS[operation]:
        raise KinematicsRequestDrift(f"kinematics {operation} fields drifted")
    if operation == "screen_path":
        knots = request["knots_rad"]
        if not isinstance(knots, Mapping) or not knots or set(knots) - set(ARMS):
            raise _refuse("knots_rad must name left, right or both", "knots_rad", knots, list(ARMS))
        polylines = {arm: _rows(knots[arm], f"knots_rad.{arm}", 6) for arm in ARMS if arm in knots}
        if len({len(rows) for rows in polylines.values()}) != 1:
            raise _refuse("both polylines need the same knot count", "knots_rad", None, ["equal lengths"])
        return {"schema": REQUEST_SCHEMA, "operation": operation, "knots_rad": polylines}
    arm = request["arm"]
    if arm not in ARMS:
        raise _refuse("arm must be left or right", "arm", arm, list(ARMS))
    if operation == "fk":
        return {"schema": REQUEST_SCHEMA, "operation": operation, "arm": arm,
                "joints_rad": _rows(request["joints_rad"], "joints_rad", 6)}
    if operation == "plan_path":
        pose, joints = request["goal_pose_xyz_xyzw"], request["goal_joints"]
        if (pose is None) == (joints is None):
            raise _refuse("plan_path takes a goal pose or goal joints", "goal_pose_xyz_xyzw", None,
                          ["goal_pose_xyz_xyzw or goal_joints"])
        return {"schema": REQUEST_SCHEMA, "operation": operation, "arm": arm,
                "goal_pose_xyz_xyzw": None if pose is None else _rows([pose], "goal_pose_xyz_xyzw", 7)[0],
                "goal_joints": None if joints is None else _rows([joints], "goal_joints", 6)[0]}
    if operation == "contact_path":
        return {"schema": REQUEST_SCHEMA, "operation": operation, "arm": arm,
                "joints_rad": _rows(request["joints_rad"], "joints_rad", 6, MAXIMUM_PATH_ROWS)}
    if operation == "pace_path":
        budget = request["budget_ticks"]
        if type(budget) is not int or budget < 1:
            raise _refuse("budget_ticks must be a positive integer", "budget_ticks", budget, ["1.."])
        if any(type(request[key]) is not bool for key in ("slow_approach", "cut_to_budget", "unwind_check")):
            raise KinematicsRequestDrift("kinematics pace_path flags drifted")
        try:
            speed = validate_speed(request["speed"])
        except ValueError as error:
            raise _refuse(str(error), "speed", request["speed"], ["a speed request"]) from error
        return {"schema": REQUEST_SCHEMA, "operation": operation, "arm": arm,
                "joints_rad": _rows(request["joints_rad"], "joints_rad", 6, MAXIMUM_PATH_ROWS), "speed": speed,
                "slow_approach": request["slow_approach"], "budget_ticks": budget,
                "cut_to_budget": request["cut_to_budget"], "unwind_check": request["unwind_check"]}
    seed = request["seed_joints_rad"]
    if seed is not None:
        seed = _rows([seed], "seed_joints_rad", 6)[0]
    if request["ranking"] not in RANKINGS:
        raise _refuse("ranking is not known", "ranking", request["ranking"], list(RANKINGS))
    if request["screen"] not in SCREENS:
        raise _refuse("screen is not known", "screen", request["screen"], list(SCREENS))
    step = request["max_joint_step_rad"]
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or not 0 < step <= 2 * math.pi:
        raise _refuse("max_joint_step_rad must be in (0, 2*pi]", "max_joint_step_rad", step, ["(0, 6.283185]"])
    return {"schema": REQUEST_SCHEMA, "operation": operation, "arm": arm, "seed_joints_rad": seed,
            "poses_xyz_xyzw": _rows(request["poses_xyz_xyzw"], "poses_xyz_xyzw", 7),
            "ranking": request["ranking"], "max_joint_step_rad": float(step), "screen": request["screen"]}


def unsupported(backend: str) -> EnvironmentRefusal:
    """The refusal of a backend that serves no kinematics query."""
    return _refuse(f"{backend} answers no kinematics query ({UNSUPPORTED})", "schema", REQUEST_SCHEMA, [])


__all__ = [
    "ARMS", "KinematicsRequestDrift", "MAXIMUM_PATH_ROWS", "MAXIMUM_POSES", "OPERATIONS", "RANKINGS", "REQUEST_SCHEMA",
    "RESULT_SCHEMA", "ROW_STATUSES", "SCREENS", "UNSUPPORTED", "unsupported", "validate_request",
]
