"""Test helpers for libero correctness repair, shared by several test modules.

They were defined in
``tests/ora/environment/backends/libero_pro/test_libero_correctness_repair.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ora.environment.backends.libero_pro.session import LiberoProSession, _CameraMetadataEnvironment
from ora.environment.controller_replay import ControllerReplayRecorder


ROOT = Path(__file__).resolve().parents[3]
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
START = [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]
TARGET = [0.25, 0.0, 0.85, 0.0, 0.0, 0.0, 1.0]


class _Plant:
    """Only the simulator API is replaced; controller dispatch stays real."""

    def __init__(self) -> None:
        self.commands: list[list[float]] = []
        self.endpoint = list(TARGET)
        self.pose = list(START)
        self.termination_tick: int | None = None
        self.truncation_tick: int | None = None
        self.fail_restore = False
        self.malformed_observation = False
        self._action_dim = 7
        # The jaws start open: robosuite's PandaGripper holds (+1, -1) after an open command.
        self.robot = SimpleNamespace(
            _joint_positions=np.zeros(7), controller_config={"type": "OSC_POSE"},
            action_dim=7, gripper=SimpleNamespace(current_action=[1.0, -1.0]),
        )
        self.robot._load_controller = self.load_controller
        self.robots = [self.robot]

    def load_controller(self) -> None:
        joint = self.robot.controller_config["type"] == "JOINT_POSITION"
        if not joint and self.commands and self.fail_restore:
            raise RuntimeError("injected controller restoration failure")
        self.robot.action_dim = 8 if joint else 7

    def observation(self) -> dict[str, Any]:
        return {"pose": list(self.pose), "robot0_gripper_qpos": [0.024, -0.024],
                "actual_tick": len(self.commands)}

    def step(self, command: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        assert len(command) == self._action_dim == 8
        self.commands.append(np.asarray(command).tolist())
        # A command is normalized on the loaded JOINT_POSITION reach: RoboSuite's 0.05 rad
        # unless a motion profile bound its own (v3), as the real controller scales it.
        reach = float(self.robot.controller_config.get("output_max", 0.05))
        self.robot._joint_positions += np.asarray(command[:7]) * reach
        self.pose = list(self.endpoint)
        tick = len(self.commands)
        return (
            None if self.malformed_observation else self.observation(), 0.0,
            tick == self.termination_tick, tick == self.truncation_tick, {},
        )


class _Planner:
    implementation = "curobo_v2"

    def __init__(self) -> None:
        self.binding = {"revision": "qualified-model", "worker": {"pid": 11}}
        self.requests: list[dict[str, Any]] = []
        self.fk_delta = 0.0
        self.no_op = False
        self.refuse: str | None = None

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(deepcopy(request))
        if self.refuse:
            return {"reachable": False, "status": "not_reachable",
                    "planner_status": self.refuse}
        start = request["joint_positions"]
        goal = request.get("goal_joint_positions", [0.05 + x for x in start])
        return {
            "reachable": True, "status": "planned", "planner_status": "success",
            "planning_time_s": 0.01,
            "start_ee_pose_base_xyzw": [0.2 + self.fk_delta, 0, 0.9, 0, 0, 0, 1],
            "trajectory": {"positions": [start] if self.no_op else [start, goal],
                           "joint_names": JOINT_NAMES, "dt_s": 0.004},
        }


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
             profile_name: str = "motion-profile-v2.json") -> tuple[Any, _Plant]:
    controllers = ModuleType("robosuite.controllers")
    controllers.load_controller_config = lambda *, default_controller: {  # type: ignore[attr-defined]
        "type": default_controller,
    }
    monkeypatch.setitem(sys.modules, "robosuite.controllers", controllers)
    plant = _Plant()
    worker = _CameraMetadataEnvironment(plant)
    profile = json.loads((ROOT / "ora/policy/embodiments/franka-panda" / profile_name).read_text())
    worker.harness_motion_profile = profile

    class _Environment:
        def __init__(self) -> None:
            self.current_raw_obs = [plant.observation()]
            self._elapsed_steps = 0
            self.refresh_steps = 0
            self.fail_attribute: str | None = None
            self.env = self

        def set_env_attr(self, name: str, value: Any) -> None:
            setattr(worker, name, value)

        def get_env_attr(self, name: str) -> list[Any]:
            if name == self.fail_attribute:
                raise RuntimeError("injected observation transport failure")
            return [getattr(worker, name)]

        def step(self, *_args: Any, **_kwargs: Any) -> Any:
            self.refresh_steps += 1
            raise AssertionError("joint execution must not add an OSC refresh step")

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.output_root = tmp_path
    session.environment = _Environment()
    session.motion_planner = _Planner()
    session.motion_profile = profile
    session.action_sequence = 0
    session.sim_steps = 0
    session.max_sim_steps = 10_000
    session.round_index = 0
    session.terminated = False
    session.truncated = False
    session.first_termination = None
    session.world_mutation_sequence = 0
    session.world_mutation_actions = []
    session.motion_probe_sequence = 0
    session.plan_references = {}
    session.home_joint_state = [0.0] * 7
    session.last_backtrack_joint_state = [-0.1] * 7
    session.robustness_episode = None
    session.workspace_bounds_m = {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}
    session.source_manifest = {"revision": "source-a"}
    session.qualified_controller_binding = {"controller": "qualified-source"}
    session.abort_reason = None
    session.control_fault_reason = None
    session.suite = "libero_10"
    session.task_id = session.seed = 0
    session.robustness_case = "core"
    session.replay_recorder = ControllerReplayRecorder(
        tmp_path, "libero_pro", session.source_manifest, ["agentview"],
    )
    session._current_state = lambda: (  # type: ignore[method-assign]
        np.asarray(session.environment.current_raw_obs[0]["pose"][:3]),
        np.asarray(session.environment.current_raw_obs[0]["pose"][3:]), 0.6,
    )
    session._motion_state = lambda: {  # type: ignore[method-assign]
        "joint_names": JOINT_NAMES,
        "joint_positions": plant.robot._joint_positions.tolist(),
        "base_pose_world_xyzw": [0, 0, 0, 0, 0, 0, 1],
        "hand_pose_world_xyzw": [0.2, 0, 0.9, 0, 0, 0, 1],
        "grip_site_pose_world_xyzw": plant.pose,
        "external_wrench_clipped": None,
    }
    session.home_tcp_pose = list(START)
    session.last_backtrack_tcp_pose = list(START)
    monkeypatch.setattr(_CameraMetadataEnvironment, "harness_grip_site_pose", property(
        lambda self: {"position_m": np.asarray(self._delegate.pose[:3]),
                      "orientation_xyzw": np.asarray(self._delegate.pose[3:])},
    ))

    def grip_site_fk(self: Any, value: Any) -> None:
        # A stand-in chain for the pacing: 0.1 m of grip site per radian of the first three joints,
        # and the last joint turning it about world z.
        self._grip_site_fk_result = {
            "poses": [[*(np.asarray(START[:3]) + 0.1 * np.asarray(row[:3])).tolist(),
                       0.0, 0.0, float(np.sin(row[6] / 2)), float(np.cos(row[6] / 2))]
                      for row in value["joints"]],
            "joint_limits_rad": [[-2.8973, 2.8973]] * 7,
        }

    monkeypatch.setattr(_CameraMetadataEnvironment, "harness_grip_site_fk", property(
        lambda self: dict(self._grip_site_fk_result), grip_site_fk,
    ))
    return session, plant
