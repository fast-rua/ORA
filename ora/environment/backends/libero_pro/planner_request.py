"""The LIBERO session's planner requests, and their planner-neutral form.

The session asks in its own small shape: ``panda_joint1..7`` and one goal, or a
batch of IK queries, each with a ``world_input`` when the planner has a world.
``neutral_plan_request`` and ``neutral_ik_request`` turn them into the
contract of ``controllers/motion_planner.py``; ``planner_port.py`` turns the
answers back into the session's records. A query may name an earlier one
(``from_index``) whose solution seeds it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.reach_check import MAXIMUM_POSES
from ora.environment.controllers.depth_world import WORLD_MODEL
from ora.environment.controllers.motion_planner import (
    IK_REQUEST_SCHEMA, PLAN_REQUEST_SCHEMA, validate_ik_request, validate_plan_request,
)

PANDA_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
_HEX = frozenset("0123456789abcdef")
_IK_REQUEST_FIELDS = frozenset({"joint_names", "joint_positions", "queries"})
_IK_QUERY_FIELDS = frozenset({"target_pose_base_xyzw", "from_index"})


class PlannerRequestError(RuntimeError):
    """A LIBERO planner request drifted from its small closed shape."""


def _finite_vector(value: Any, length: int) -> bool:
    return (isinstance(value, list) and len(value) == length
            and all(not isinstance(item, bool) and isinstance(item, (int, float))
                    and math.isfinite(float(item)) for item in value))


def _unit_pose(value: Any) -> bool:
    return _finite_vector(value, 7) and math.isclose(
        sum(float(item) ** 2 for item in value[3:]), 1.0, abs_tol=1e-5)


def validate_world_input(value: Any) -> None:
    """A world-mode request names its world-input file by absolute path and sha256."""
    if (not isinstance(value, Mapping) or set(value) != {"path", "sha256"}
            or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or not (isinstance(value["sha256"], str) and len(value["sha256"]) == 64
                    and set(value["sha256"]) <= _HEX)):
        raise PlannerRequestError("planner world_input must be an absolute path and its sha256")


def validate_libero_plan_request(value: Any) -> dict[str, Any]:
    """The closed plan request: measured Panda joints, one goal, and ``world_input`` in world mode."""
    if not isinstance(value, Mapping):
        raise PlannerRequestError("planner request must be an object")
    common = {"joint_names", "joint_positions"}
    goal_fields = {"target_pose_base_xyzw", "goal_joint_positions"}
    if set(value) - common - goal_fields - {"world_input"} or not common.issubset(value):
        raise PlannerRequestError("planner request fields drifted")
    if "world_input" in value:
        validate_world_input(value["world_input"])
    present_goals = goal_fields.intersection(value)
    if len(present_goals) != 1:
        raise PlannerRequestError("planner request requires exactly one goal")
    if value["joint_names"] != PANDA_JOINT_NAMES or not _finite_vector(value["joint_positions"], 7):
        raise PlannerRequestError("planner request has invalid current joints")
    goal_name = next(iter(present_goals))
    if not _finite_vector(value[goal_name], 7):
        raise PlannerRequestError("planner request has an invalid goal")
    if goal_name == "target_pose_base_xyzw" and not _unit_pose(value[goal_name]):
        raise PlannerRequestError("planner target quaternion must be normalized")
    return json.loads(json.dumps(value))


def validate_libero_ik_request(value: Any) -> dict[str, Any]:
    """The closed IK request: measured Panda joints and 1..MAXIMUM_POSES chained poses.

    A world-mode request adds the ``world_input`` its plans would use.
    """
    if not isinstance(value, Mapping) or set(value) - {"world_input"} != _IK_REQUEST_FIELDS:
        raise PlannerRequestError("IK request fields drifted")
    if "world_input" in value:
        validate_world_input(value["world_input"])
    if value["joint_names"] != PANDA_JOINT_NAMES or not _finite_vector(value["joint_positions"], 7):
        raise PlannerRequestError("IK request has invalid current joints")
    queries = value["queries"]
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAXIMUM_POSES:
        raise PlannerRequestError(f"IK request must hold 1 to {MAXIMUM_POSES} queries")
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping) or set(query) != _IK_QUERY_FIELDS:
            raise PlannerRequestError("IK query fields drifted")
        if not _unit_pose(query["target_pose_base_xyzw"]):
            raise PlannerRequestError("IK target must be a finite pose with a unit quaternion")
        start = query["from_index"]
        if start is not None and (isinstance(start, bool) or not isinstance(start, int)
                                  or not 0 <= start < index):
            raise PlannerRequestError("IK from_index must name an earlier query")
    return json.loads(json.dumps(value))


def _neutral_world(world_input: Mapping[str, Any] | None) -> dict[str, Any]:
    return {"model": "none"} if world_input is None else {"model": WORLD_MODEL, "input": dict(world_input)}


def neutral_plan_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """A session plan request as ``ora.motion-plan-request.v1`` (the held load comes from the world)."""
    value = validate_libero_plan_request(request)
    world = _neutral_world(value.get("world_input"))
    goal = ({"tool_frame": "panda_hand", "pose_base_xyzw": value["target_pose_base_xyzw"]}
            if "target_pose_base_xyzw" in value else {"joint_positions": value["goal_joint_positions"]})
    return validate_plan_request({
        "schema": PLAN_REQUEST_SCHEMA, "robot": "franka_panda",
        "start": {"joint_names": value["joint_names"], "joint_positions": value["joint_positions"]},
        "goal": goal, "world": world, "constraints": [],
        "held_object": {"source": "none" if world["model"] == "none" else "world"},
    })


def neutral_ik_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """A session IK request as ``ora.motion-ik-request.v1``."""
    value = validate_libero_ik_request(request)
    world = _neutral_world(value.get("world_input"))
    return validate_ik_request({
        "schema": IK_REQUEST_SCHEMA, "robot": "franka_panda",
        "start": {"joint_names": value["joint_names"], "joint_positions": value["joint_positions"]},
        "queries": [{"tool_frame": "panda_hand", "pose_base_xyzw": query["target_pose_base_xyzw"],
                     "from_index": query["from_index"]} for query in value["queries"]],
        "world": world, "held_object": {"source": "none" if world["model"] == "none" else "world"},
    })


def failure_reason(record: Mapping[str, Any]) -> str:
    """A plan record's reason; an implementation error is never relabelled unreachable."""
    if record.get("status") == "planner_error":
        return "planner_error"
    return str(record.get("planner_status") or record.get("status") or "planner_error")


def valid_query(record: Mapping[str, Any]) -> bool | None:
    """A plan record's query validity, ``None`` where the record does not know it."""
    if isinstance(record.get("valid_query"), bool):
        return record["valid_query"]
    if record.get("status") == "planner_error":
        return None
    return True if record.get("status") == "planned" else None


__all__ = [
    "PANDA_JOINT_NAMES", "PlannerRequestError", "failure_reason", "neutral_ik_request", "neutral_plan_request",
    "valid_query", "validate_libero_ik_request", "validate_libero_plan_request", "validate_world_input",
]
