"""A world-mode LIBERO session hands the planner a fresh public snapshot for every call."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from ora.environment.backends.libero_pro.reach_check import reach_check_batch
from ora.environment.backends.libero_pro.session import LiberoProError, LiberoProSession
from ora.environment.controllers.depth_world import WORLD_MODEL, read_world_inputs
from tests.support.agent.motion_runtime import _action, _pose

JOINTS = [0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.8]
NEAR, FAR = 0.01, 10.0
META = {name: {"intrinsic_K": np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]),
               "extrinsic_cam2world": np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, height], [0, 0, 0, 1]]),
               "depth_near": NEAR, "depth_far": FAR, "height": None, "width": None}
        for height, name in ((1.5, "agentview"), (1.1, "robot0_eye_in_hand"))}


class Planner:
    implementation = "curobo_v2"

    def __init__(self, world_model: str) -> None:
        self.world_model = world_model
        self.requests: list[dict[str, Any]] = []

    def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.requests.append(dict(request))
        return {
            "schema": "ora.libero-plan-record.v1", "status": "not_reachable", "reachable": False,
            "world_model": self.world_model, "planner_status": "goal_unreachable",
            "valid_query": True, "request": dict(request),
            **({"world": {"world_model": WORLD_MODEL, "sha256": "b" * 64, "inputs_sha256": "c" * 64}}
               if self.world_model != "none" else {}),
        }

    def solve_ik(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.requests.append(dict(request))
        return {"status": "planner_error", "error": {"type": "Injected", "message": "stop here"}}


def _session(tmp_path: Path, world_model: str) -> Any:
    raw = {
        "agentview_depth": np.linspace(0.2, 0.8, 64 * 64, dtype=np.float32).reshape(64, 64, 1),
        "robot0_eye_in_hand_depth": np.full((64, 64, 1), 0.5, dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
    }
    environment = SimpleNamespace(
        current_raw_obs=[raw], get_gripper_hold_directive=lambda: 1.0,
        get_camera_meta=lambda camera_name, height, width: META[camera_name],
    )
    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.motion_planner = Planner(world_model)
    session.output_root = tmp_path
    session.action_sequence = 3
    session.sim_steps = 0
    session.last_receipt = None
    session.world_mutation_sequence = 0
    session.motion_probe_sequence = 0
    session.plan_references = {}
    session.workspace_bounds_m = {"x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3]}
    session._require_environment = lambda: environment
    session._current_state = lambda: (np.array([0.4, 0.0, 0.5]), np.array([0.0, 0.0, 0.0, 1.0]), 0.5)
    session._motion_state = lambda: {
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_positions": list(JOINTS),
        "base_pose_world_xyzw": [-0.66, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0],
        "hand_pose_world_xyzw": [0.4, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0],
        "grip_site_pose_world_xyzw": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
    }
    return session, raw


def test_a_world_plan_writes_the_published_depth_and_records_its_hash(tmp_path):
    session, raw = _session(tmp_path, WORLD_MODEL)
    receipt = session._execute_planned_motion(_action("move_free_space", {"arm": "right", "target_pose": _pose()}))
    assert receipt["motion_status"] == "refused" and receipt["state_changed"] is False
    (request,) = session.motion_planner.requests
    world_input = request["world_input"]
    path = tmp_path / "curobo/world-inputs/action-0003.npz"
    assert world_input["path"] == str(path.resolve())
    inputs = read_world_inputs(path, world_input["sha256"])
    # Exactly the depth an observation would publish: metric, flipped to the OpenCV top-left order.
    agentview = inputs.cameras[0]
    expected = (NEAR / (1.0 - raw["agentview_depth"][..., 0] * (1.0 - NEAR / FAR)))[::-1]
    np.testing.assert_array_equal(agentview.depth_m, expected.astype(np.float32))
    np.testing.assert_array_equal(agentview.cam2world_cv, META["agentview"]["extrinsic_cam2world"])
    assert agentview.depth_range_m == (NEAR, FAR)
    assert inputs.joint_positions.tolist() == request["joint_positions"] == JOINTS
    assert inputs.base_pose_world_xyzw.tolist() == [-0.66, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0]
    assert inputs.gripper_directive == 1.0 and inputs.finger_gap_m == pytest.approx(0.04)
    planner = receipt["planner"]
    assert planner["world_model"] == WORLD_MODEL and planner["world"]["sha256"] == "b" * 64
    assert planner["world_input"] == {"path": "curobo/world-inputs/action-0003.npz",
                                      "sha256": world_input["sha256"]}


def test_one_probe_snapshot_serves_every_variant_and_reach_checks_get_their_own(tmp_path):
    session, _ = _session(tmp_path, WORLD_MODEL)
    session.probe_motion({
        "schema": "ora.motion-probe-request.v1", "observation_id": "observation-0",
        "observation_sha256": "0" * 64, "variant_policy": "parallel_jaw_equivalents",
        "queries": [{"query_id": "query-0", "arm": "right", "pose": _pose()}],
    })
    requests = session.motion_planner.requests
    assert len(requests) == 2 and requests[0]["world_input"] == requests[1]["world_input"]
    assert requests[0]["world_input"]["path"].endswith("curobo/world-inputs/probe-000000.npz")
    with pytest.raises(LiberoProError, match="CuRobo IK failed"):
        reach_check_batch(session, "right", [{"pose_xyzw": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                                              "from_index": None}],
                          position_tolerance_m=0.01, orientation_tolerance_rad=0.1)
    assert requests[2]["world_input"]["path"].endswith("curobo/world-inputs/reach-000001.npz")
    assert sorted(item.name for item in (tmp_path / "curobo/world-inputs").iterdir()) == [
        "probe-000000.npz", "reach-000001.npz"]


def test_a_robot_only_planner_gets_no_world_inputs(tmp_path):
    session, _ = _session(tmp_path, "none")
    receipt = session._execute_planned_motion(_action("move_free_space", {"arm": "right", "target_pose": _pose()}))
    (request,) = session.motion_planner.requests
    assert "world_input" not in request and "world_input" not in receipt["planner"]
    assert not (tmp_path / "curobo/world-inputs").exists()
