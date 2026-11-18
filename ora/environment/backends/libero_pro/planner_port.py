"""The LIBERO session's planner records from a planner-neutral implementation.

The session asks in its own request shape (``planner_request.py``) and reads
every answer in one record shape, which is also its plan evidence. The planner
answers in ``ora.motion-plan-answer.v1`` (``controllers/motion_planner.py``,
selected by ``runtime.motion_planner.implementation``). ``NeutralPlannerPort``
turns the session's requests into neutral ones and each answer back into a
record (``ora.libero-plan-record.v1``, ``ora.libero-ik-record.v1``): the neutral
reason is its planner status, the IK seed populations an implementation keeps
(``detail.ik_attempts``) are the record's ``planning_diagnostic_evidence``, the
neutral answer is kept whole inside it, and receipts, probes, reach checks and
evidence read that record whichever planner runs.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.environment.controllers.motion_planner import planning_diagnostics

from .planner_request import (
    neutral_ik_request, neutral_plan_request, validate_libero_ik_request, validate_libero_plan_request,
)

PLAN_RECORD_SCHEMA = "ora.libero-plan-record.v1"
IK_RECORD_SCHEMA = "ora.libero-ik-record.v1"
#: The IK evidence schema ``planning_diagnostics.ik_candidate_tally`` reads.
IK_DIAGNOSTICS_SCHEMA = "ora.libero-ik-diagnostics.v1"
_STATUS = {"planned": "planned", "refused": "not_reachable", "error": "planner_error"}


def plan_record(answer: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """The session's record of one neutral plan answer to its ``request``."""
    status, reason = answer["status"], answer["reason"]
    world = answer.get("world")
    return {
        "schema": PLAN_RECORD_SCHEMA, "status": _STATUS[status], "reachable": status == "planned",
        "planner_status": "success" if status == "planned" else reason,
        "valid_query": None if status == "error" else not str(reason).startswith("start_"),
        "world_model": (answer.get("request") or {}).get("world", {}).get("model", "none"),
        "planning_time_s": float(answer["timing_s"]["total"]),
        "start_ee_pose_base_xyzw": answer.get("start_tool_pose_base_xyzw"),
        "requested_target_pose_base_xyzw": request.get("target_pose_base_xyzw"),
        "requested_goal_joint_positions": request.get("goal_joint_positions"),
        "trajectory": answer.get("trajectory"),
        "error": answer["detail"]["error"] if status == "error" else None,
        "request": dict(request),
        **({"world": dict(world["summary"])} if world else {}),
        "planning_diagnostics": planning_diagnostics(answer),
        **_ik_evidence(answer),
        "answer": dict(answer),
    }


def _ik_evidence(answer: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's IK seed populations, which the motion probe tallies."""
    attempts = (answer.get("detail") or {}).get("ik_attempts")
    if not isinstance(attempts, list):
        return {}
    model = (answer.get("request") or {}).get("world", {}).get("model", "none")
    return {"planning_diagnostic_evidence": {"schema": IK_DIAGNOSTICS_SCHEMA, "attempts": list(attempts),
                                             "attempts_dropped": 0, "world_model": model,
                                             "extra_solver_calls": 0}}


def ik_record(answer: Mapping[str, Any]) -> dict[str, Any]:
    """The session's record of one neutral IK answer (what ``reach_check`` reads)."""
    model = (answer.get("request") or {}).get("world", {}).get("model", "none")
    record: dict[str, Any] = {"schema": IK_RECORD_SCHEMA, "world_model": model,
                              "solve_time_s": float(answer["timing_s"]["total"]), "answer": dict(answer)}
    if answer["status"] == "error":
        return {**record, "status": "planner_error", "error": dict(answer["detail"]["error"])}
    world = answer.get("world")
    return {
        **record, "status": "solved", "joint_names": answer["joint_names"], "joint_limits": answer["joint_limits"],
        "start_ee_pose_base_xyzw": answer["start_tool_pose_base_xyzw"],
        "results": [{"solved": row["solved"], "skipped": row["skipped"], "joint_positions": row["joint_positions"],
                     "ee_pose_base_xyzw": row["tool_pose_base_xyzw"], "position_error_m": row["position_error_m"],
                     "rotation_error": row["rotation_error"]} for row in answer["results"]],
        **({"world": {"inputs_sha256": world["inputs_sha256"], "queries": world["queries"]}} if world else {}),
    }


class NeutralPlannerPort:
    """A ``MotionPlannerClient`` behind the session's planner interface."""

    def __init__(self, client: Any) -> None:
        self.client = client

    @property
    def world_model(self) -> str:
        return self.client.world_model

    @property
    def implementation(self) -> str:
        return self.client.implementation

    @property
    def binding(self) -> dict[str, Any]:
        return self.client.binding

    def start(self) -> dict[str, Any]:
        return self.client.start()

    def close(self) -> None:
        self.client.close()

    def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_libero_plan_request(request)
        return plan_record(self.client.plan(neutral_plan_request(validated)), validated)

    def solve_ik(self, request: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_libero_ik_request(request)
        return ik_record(self.client.solve_ik(neutral_ik_request(validated)))


__all__ = ["IK_RECORD_SCHEMA", "NeutralPlannerPort", "PLAN_RECORD_SCHEMA", "ik_record", "plan_record"]
