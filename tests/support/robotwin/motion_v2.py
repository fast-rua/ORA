"""Test helpers for robotwin motion v2, shared by several test modules.

They were defined in
``tests/ora/environment/backends/robotwin/test_robotwin_motion_v2.py``; test modules import
them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ora.agent.toolkit.host.runtime.core import ManipulationToolRuntime
from ora.environment.backends.robotwin.session import RoboTwinLiveSession
from ora.environment.controllers.profile_capabilities import (
    FROZEN_CONTROL_PROFILE_SHA256, OLDEST_COMPOSED_CONTROL_PROFILE,
)
from tests.support.agent.surface_catalog import surface_of


ROOT = Path(__file__).resolve().parents[3]
#: The oldest control profile this tree composes; a sparse RoboTwin session
#: refuses every older one.
CONTROL_PROFILE = ROOT / (
    "ora/policy/embodiments/aloha-agilex/"
    f"control-profile-v{OLDEST_COMPOSED_CONTROL_PROFILE}.json"
)
CONTROL_PROFILE_SHA256 = FROZEN_CONTROL_PROFILE_SHA256[OLDEST_COMPOSED_CONTROL_PROFILE]


def pose(x: float = 0.0) -> dict[str, list[float]]:
    return {
        "position_m": [x, 0.0, 1.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def primitive_runtime(
    tmp_path: Path, *, profile: str = "robotwin_toolchain_b2_closed_noservo_v2",
) -> tuple[ManipulationToolRuntime, dict[str, Any], list[dict[str, Any]]]:
    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = surface_of(profile)
    runtime._submitted_mutations = []
    runtime.control_root = tmp_path
    runtime.workspace_bounds_m = {
        "x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0],
    }
    control = {
        "observation_id": "observation-0", "observation_sha256": "0" * 64,
        "generation": 0, "open": True,
    }
    observation = {
        "proprioception": {"end_effectors": {
            "left": {
                "pose_xyz_xyzw": [-0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            "right": {
                "pose_xyz_xyzw": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
        }},
    }
    runtime._current_grip_state = lambda: (  # type: ignore[method-assign]
        control, tmp_path, observation,
        observation["proprioception"]["end_effectors"]["right"][
            "pose_xyz_xyzw"
        ],
        0.0,
    )
    persisted: list[dict[str, Any]] = []
    runtime._persist_action = (  # type: ignore[method-assign]
        lambda _path, _control, submitted: persisted.append(dict(submitted))
    )
    return runtime, observation, persisted


def _probe_public_result(*, observation_id: str, plan_ref: str) -> dict[str, Any]:
    planner = {
        "called": True, "status": "Success", "stage": "SUCCESS",
        "valid_query": True, "attempts": 1, "trajopt_attempts": 1,
        "used_graph": False, "goalset_index": 0, "path_steps": 4,
        "interpolation_dt_s": 0.004,
        "errors": {"position": 0.0, "rotation": 0.0, "cspace": None},
        "timings_s": {
            "solve": 0.01, "ik": 0.001, "graph": 0.0,
            "trajopt": 0.009, "finetune": 0.0, "total": 0.01,
        },
        "error": None,
    }
    pose_value = [-0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    variant = {
        "variant_id": "variant-00-tool-z-+0deg", "reachable": True,
        "target_pose_xyzw": pose_value, "planner": planner,
        "endpoint_error": {"translation_m": 0.001, "orientation_rad": 0.01},
        "trajectory_steps": 4, "trajectory_duration_s": 0.016,
        "plan_ref": plan_ref,
    }
    return {
        "schema": "ora.motion-probe-result.v1",
        "observation_id": observation_id,
        "world_mutation_sequence": 0,
        "variant_policy": "exact",
        "robot_decisions_consumed": 0,
        "world_state_changed": False,
        "results": [{
            "query_id": "q0", "arm": "left", "reachable": True,
            "selected_variant_id": variant["variant_id"],
            "plan_ref": plan_ref,
            "selected_target_pose_xyzw": pose_value,
            "variants": [variant],
        }],
    }


class Cameras:
    def __init__(self) -> None:
        self.picture_updates = 0

    def update_picture(self) -> None:
        self.picture_updates += 1

    @staticmethod
    def _rgb(value: int) -> np.ndarray:
        return np.full((24, 32, 3), value, dtype=np.uint8)

    def get_rgb(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            "head_camera": {"rgb": self._rgb(50)},
            "front_camera": {"rgb": self._rgb(80)},
            "left_camera": {"rgb": self._rgb(110)},
            "right_camera": {"rgb": self._rgb(140)},
        }

    def get_observer_rgb(self) -> np.ndarray:
        return self._rgb(20)


def transaction_plant(**kwargs: Any) -> Any:
    """The correctness plant, with joint and gripper drive targets to digest."""

    from types import SimpleNamespace
    from tests.support.robotwin.correctness_repair import Plant

    plant = Plant(**kwargs)
    gripper = {"left": 0.0, "right": 0.0}
    plant.gripper_writes = []

    def write_gripper(value: float, arm: str, per_step: float = 0.0) -> None:
        plant.write_gripper(value, arm, per_step)
        plant.gripper_writes.append(arm)
        gripper[arm] = float(value)

    plant.robot.set_gripper = write_gripper
    for arm in ("left", "right"):
        setattr(plant.robot, f"{arm}_arm_joints", [
            SimpleNamespace(get_drive_target=lambda a=arm, i=index: np.asarray([plant.targets[a][i]]))
            for index in range(6)
        ])
        setattr(plant.robot, f"{arm}_gripper", [
            (SimpleNamespace(get_drive_target=lambda a=arm: np.asarray([gripper[a]])),),
        ])
    return plant


def transaction_session(
    tmp_path: Path, plant: Any, *, failing: tuple[str, ...] = (),
) -> tuple[RoboTwinLiveSession, list[str]]:
    """A v10 sparse session on ``plant``; also returns the arms each planner call named."""

    import copy
    from tests.support.robotwin.correctness_repair import branch_candidates, live_session

    live = live_session(tmp_path, plant)
    live._public_to_planner_wxyz = lambda arm, pose: pose
    calls: list[str] = []

    def planner(arm: str) -> Any:
        def plan(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            calls.append(arm)
            if arm in failing:
                return {"status": "Failure", "planner_status": "IK_FAIL"}
            raw = branch_candidates(arm)
            for candidate in raw["candidates"]:
                candidate["position"][:, 2] = 0.0
            return copy.deepcopy(raw)
        return plan

    for arm in ("left", "right"):
        setattr(plant.robot, f"{arm}_plan_path", planner(arm))
        setattr(plant.robot, f"{arm}_plan_joint_path", planner(arm))
    return live, calls


def _transaction_target(arm: str) -> dict[str, list[float]]:
    return {
        "position_m": [-0.18 if arm == "left" else 0.22, 0.0, 0.8],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def transaction_move(*arms: str) -> dict[str, Any]:
    from tests.support.robotwin.correctness_repair import action as correctness_action

    return correctness_action("move_pose", {
        "targets": {arm: _transaction_target(arm) for arm in arms},
        "interactions": {arm: "free_space" for arm in arms},
    })


def _live_robotwin_native_spec() -> dict[str, Any]:
    """The default RoboTwin Native RunSpec, expanded to its explicit fields.

    The baseline marker (and its context-scheme record) is dropped so the RunPlan camera-topology and
    control-profile checks see each mutated field directly, instead of the
    baseline layer refusing the conflicting override first. An earlier study's own
    composition (pressure retention, motion_action_v2, visible_only_v1
    replay, control-profile-v1) composes only at its launch commits.
    """

    from ora.orchestrator.spec.defaults import expand_native_defaults
    from tests.support.orchestrator.run_specs import native_spec

    spec = expand_native_defaults(native_spec("robotwin"), root=ROOT)
    spec["policy"].pop("baseline")
    spec["policy"].pop("context_scheme")
    return spec
