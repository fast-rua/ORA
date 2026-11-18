"""LIBERO behind a planner-neutral implementation: requests, records, reasons and the reach check."""
from __future__ import annotations

import copy
from typing import Any

import pytest

from ora.agent.session_core.receipt_projection import public_motion_failure_code
from ora.contracts.reach_check import REQUEST_SCHEMA
from ora.environment.backends.libero_pro import reach_check as libero_reach
from ora.environment.backends.libero_pro.common import _sparse_planning_failure
from ora.environment.backends.libero_pro.planner_port import NeutralPlannerPort, ik_record
from ora.environment.backends.libero_pro.planner_request import failure_reason, neutral_ik_request, valid_query
from ora.environment.controllers.depth_world import WORLD_MODEL
from ora.environment.controllers.motion_planner import (
    IK_ANSWER_SCHEMA, PLAN_ANSWER_SCHEMA, REFUSAL_REASONS, validate_ik_answer, validate_plan_answer,
    validate_plan_request,
)
from tests.support.libero.reach_check import DOWN, NAMES, FakeIk, _Session

JOINTS = [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]
POSE = [0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]
WORLD_INPUT = {"path": "/inputs/world.npz", "sha256": "a" * 64}
SESSION_REQUEST = {"joint_names": NAMES, "joint_positions": JOINTS, "target_pose_base_xyzw": POSE,
                  "world_input": WORLD_INPUT}


class Client:
    """A planner-neutral client that answers from canned outcomes and checks every request it gets."""

    implementation, world_model = "curobo_v2", WORLD_MODEL

    def __init__(self, status: str = "planned", reason: str | None = None) -> None:
        self.status, self.reason = status, reason
        self.requests: list[dict[str, Any]] = []
        self.binding = {"implementation": "curobo_v2"}

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(validate_plan_request(request))
        planned = self.status == "planned"
        return validate_plan_answer({
            "schema": PLAN_ANSWER_SCHEMA, "status": self.status, "reason": self.reason,
            "trajectory": {"joint_names": NAMES, "positions": [JOINTS, JOINTS], "velocities": None, "dt_s": 0.02}
            if planned else None,
            "start_tool_pose_base_xyzw": POSE, "planner": {"implementation": "curobo_v2"},
            "world": {"model": WORLD_MODEL, "sha256": "b" * 64, "inputs_sha256": "c" * 64,
                      "summary": {"world_model": WORLD_MODEL, "sha256": "b" * 64, "inputs_sha256": "c" * 64,
                                  "goal_refusal": {"message": "raise it"} if self.reason == "goal_below_support"
                                  else None}},
            "timing_s": {"total": 1.25}, "request": request, "detail": {"solver": {}},
        }, request)


def test_a_plan_s_ik_seed_populations_are_the_record_s_ik_evidence_the_probe_tallies():
    from ora.environment.backends.libero_pro.planning_diagnostics import ik_candidate_tally
    from ora.environment.backends.libero_pro.planner_port import IK_DIAGNOSTICS_SCHEMA

    population = {"status": "captured", "accepted": [True, False, False, False],
                  "feasible": [True, False, True, True], "converged": [True, True, False, True],
                  "outside_position_limits": [False, False, False, True],
                  "candidate_counts": {"optimized": 4, "accepted": 1}, "rejection_counts": {"solver_infeasible": 1}}
    client = Client("refused", "path_blocked")
    plan = client.plan
    client.plan = lambda request: {**plan(request), "detail": {"solver": {}, "ik_attempts": [population]}}
    record = NeutralPlannerPort(client).plan(SESSION_REQUEST)
    evidence = record["planning_diagnostic_evidence"]
    assert evidence["schema"] == IK_DIAGNOSTICS_SCHEMA and evidence["attempts"] == [population]
    assert evidence["world_model"] == WORLD_MODEL
    assert record["planning_diagnostics"]["candidate_counts"] == {"optimized": 4, "accepted": 1}
    tally = ik_candidate_tally(evidence["attempts"][-1])
    assert (tally["returned"], tally["valid_endpoint"]) == (4, 1)
    assert {key: tally["rejections"][key] for key in ("joint_limit", "solver_unsuccessful",
                                                      "fk_residual_exceeded")} == {
        "joint_limit": 1, "solver_unsuccessful": 1, "fk_residual_exceeded": 1}
    # An answer without seed populations (an error, a joint goal) carries no IK evidence.
    assert "planning_diagnostic_evidence" not in NeutralPlannerPort(Client()).plan(SESSION_REQUEST)


def test_the_port_sends_neutral_requests_and_returns_the_session_s_record():
    client = Client()
    record = NeutralPlannerPort(client).plan(SESSION_REQUEST)
    [sent] = client.requests
    assert sent["goal"] == {"tool_frame": "panda_hand", "pose_base_xyzw": POSE}
    assert sent["world"] == {"model": WORLD_MODEL, "input": WORLD_INPUT} and sent["held_object"] == {"source": "world"}
    assert record["status"] == "planned" and record["reachable"] is True and record["planner_status"] == "success"
    assert record["trajectory"]["positions"] == [JOINTS, JOINTS] and record["start_ee_pose_base_xyzw"] == POSE
    assert record["request"] == SESSION_REQUEST and record["world"]["sha256"] == "b" * 64
    assert record["world_model"] == WORLD_MODEL and record["answer"]["planner"]["implementation"] == "curobo_v2"
    assert record["planning_diagnostics"]["outcome"] == "planned"


@pytest.mark.parametrize("reason", REFUSAL_REASONS)
def test_a_refusal_carries_its_neutral_reason_into_the_session_and_its_projections(reason):
    record = NeutralPlannerPort(Client("refused", reason)).plan(SESSION_REQUEST)
    assert record["status"] == "not_reachable" and record["reachable"] is False
    assert failure_reason(record) == reason
    start = reason.startswith("start_")
    assert valid_query(record) is (not start)
    status, stage, code = _sparse_planning_failure(reason)
    assert (status, stage, code) == ("plan_rejected" if start else "planner_failed", "planner", reason)
    assert public_motion_failure_code("planner", reason) == ("invalid_start_state" if start else "planner_failed")
    assert "MotionGen" not in str(record) and "CuRobo" not in str(record["planning_diagnostics"])


def test_an_error_is_a_planner_fault_for_the_session():
    client = Client()
    error = {"schema": PLAN_ANSWER_SCHEMA, "status": "error", "reason": "planner_error", "trajectory": None,
             "request": None, "planner": {"implementation": "curobo_v2"}, "timing_s": {"total": 0.1},
             "detail": {"error": {"type": "RuntimeError", "message": "device lost"}}}
    client.plan = lambda request: validate_plan_answer(error)
    record = NeutralPlannerPort(client).plan(SESSION_REQUEST)
    assert record["status"] == "planner_error" and failure_reason(record) == "planner_error"
    assert record["error"]["type"] == "RuntimeError" and record["valid_query"] is None
    assert _sparse_planning_failure("planner_error")[0] == "execution_fault"
    assert public_motion_failure_code("planner", "timeout") == "planner_error"


class NeutralIk:
    """``FakeIk``'s answers as a planner-neutral implementation gives them."""

    implementation, world_model = "curobo_v2", "none"

    def __init__(self) -> None:
        self.fake = FakeIk()

    def solve_ik(self, request: dict[str, Any]) -> dict[str, Any]:
        session = {"joint_names": request["start"]["joint_names"],
                   "joint_positions": request["start"]["joint_positions"],
                   "queries": [{"target_pose_base_xyzw": query["pose_base_xyzw"], "from_index": query["from_index"]}
                               for query in request["queries"]]}
        record = self.fake.solve_ik(session)
        return validate_ik_answer({
            "schema": IK_ANSWER_SCHEMA, "request": neutral_ik_request(session), "status": "solved", "reason": None,
            "planner": {"implementation": "curobo_v2"}, "timing_s": {"total": record["solve_time_s"]}, "detail": {},
            "joint_names": record["joint_names"], "joint_limits": record["joint_limits"],
            "start_tool_pose_base_xyzw": record["start_ee_pose_base_xyzw"], "world": None,
            "results": [{"solved": row["solved"], "skipped": row["skipped"], "refusal": None,
                         "joint_positions": row["joint_positions"], "tool_pose_base_xyzw": row["ee_pose_base_xyzw"],
                         "position_error_m": row["position_error_m"], "rotation_error": row["rotation_error"]}
                        for row in record["results"]]}, request)


def test_the_reach_check_reads_the_same_rows_through_the_port_and_names_no_library():
    queries = [{"pose_xyzw": [-0.1, 0.1, 1.05, *DOWN], "interaction": "free_space", "from_index": None},
               {"pose_xyzw": [-0.1, 0.1, 0.98, *DOWN], "interaction": "contact_approach", "from_index": 0},
               {"pose_xyzw": [0.6, 0.6, 1.05, *DOWN], "interaction": "free_space", "from_index": None}]
    request = {"schema": REQUEST_SCHEMA, "arm": "right", "queries": copy.deepcopy(queries)}
    direct = _Session(FakeIk()).align_plan(copy.deepcopy(request))
    neutral = _Session(NeutralPlannerPort(NeutralIk())).align_plan(copy.deepcopy(request))
    assert neutral["results"] == direct["results"]
    assert neutral["ik_model"] == libero_reach.IK_MODEL_ROBOT_ONLY and "CuRobo" not in neutral["claim_boundary"]


def test_an_ik_error_reaches_the_reach_check_as_a_fault():
    error = {"schema": "ora.motion-ik-answer.v1", "status": "error", "reason": "planner_error", "request": None,
             "planner": {"implementation": "curobo_v2"}, "timing_s": {"total": 0.1},
             "detail": {"error": {"type": "RuntimeError", "message": "device lost"}}}
    record = ik_record(validate_ik_answer(error))
    assert record["status"] == "planner_error" and record["error"]["type"] == "RuntimeError"
