"""Original IK sample evidence survives direct motion without changing admission."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np

from ora.environment.backends.libero_pro.planner_port import NeutralPlannerPort
from ora.environment.backends.libero_pro.session import LiberoProSession
from ora.environment.controllers.motion_planner import PLAN_ANSWER_SCHEMA, validate_plan_answer
from tests.support.agent.motion_runtime import _action, _pose

#: One attempt's four optimised seeds, none feasible, as ``detail.ik_attempts`` keeps them.
POPULATION = {
    "status": "captured", "scope": "all_optimized_seeds_for_one_original_ik_attempt",
    "joint_names": [f"panda_joint{i}" for i in range(1, 8)], "joint_positions": [[0.1] * 7] * 4,
    "feasible": [False] * 4, "converged": [False, True, True, True], "accepted": [False] * 4,
    "candidate_counts": {"optimized": 4, "returned": 2, "returned_successful": 0,
                         "converged": 3, "feasible": 0, "accepted": 0},
    "rejection_counts": {"position_tolerance": 1, "rotation_tolerance": 0, "solver_infeasible": 4,
                         "outside_position_limits": 1, "self_collision": None},
    "unknown_reasons": ["per_term_constraint_costs_not_retained_by_sdk"],
}


class Client:
    """A planner-neutral client that refuses every goal with its IK seed populations."""

    implementation, world_model = "curobo_v2", "none"

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def plan(self, request):
        self.requests.append(deepcopy(request))
        return validate_plan_answer({
            "schema": PLAN_ANSWER_SCHEMA, "status": "refused", "reason": "goal_unreachable", "trajectory": None,
            "start_tool_pose_base_xyzw": [.4, 0., .5, 0., 0., 0., 1.], "planner": {"implementation": "curobo_v2"},
            "world": None, "timing_s": {"total": 0.1}, "request": request,
            "detail": {"solver": {}, "ik_attempts": [POPULATION]},
        }, request)


def test_direct_move_failure_keeps_sample_diagnostics_and_zero_execution(tmp_path):
    client = Client()
    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}
    session.motion_planner = NeutralPlannerPort(client)
    session.output_root = tmp_path
    session.action_sequence = session.sim_steps = session.world_mutation_sequence = 0
    session.last_receipt = None
    session.workspace_bounds_m = {"x": [-.6, .6], "y": [-.5, .5], "z": [-.1, 1.3]}
    session._require_environment = lambda: SimpleNamespace()
    session._current_state = lambda: (np.array([.4, 0., .5]), np.array([0., 0., 0., 1.]), .2)
    session._motion_state = lambda: {
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)], "joint_positions": [0.] * 7,
        "base_pose_world_xyzw": [0., 0., 0., 0., 0., 0., 1.],
        "hand_pose_world_xyzw": [.4, 0., .5, 0., 0., 0., 1.],
        "grip_site_pose_world_xyzw": [.4, 0., .5, 0., 0., 0., 1.],
    }
    receipt = session._execute_planned_motion(_action(
        "move_free_space", {"arm": "right", "target_pose": _pose()}))
    assert len(client.requests) == 1
    assert receipt["steps_used"] == 0 and receipt["state_changed"] is False
    assert receipt["feedback"]["planning_failure_reason"] == "goal_unreachable"
    public = receipt["planner"]["planning_diagnostics"]
    assert public["phase"] == "ik" and public["candidate_counts"]["accepted"] == 0
    assert public["rejection_counts"]["self_collision"] is None
    assert "joint_positions" not in json.dumps(public)
    path = tmp_path / receipt["planner"]["canonical_artifact"]["path"]
    saved = json.loads(path.read_text())
    assert saved["planner_implementation"] == "curobo_v2"
    assert saved["planning_diagnostic_evidence"]["attempts"][0]["joint_positions"] == POPULATION["joint_positions"]
    assert session.sim_steps == session.world_mutation_sequence == 0
