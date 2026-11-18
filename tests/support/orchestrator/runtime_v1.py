"""Test helpers for runtime v1, shared by several test modules.

They were defined in ``tests/ora/orchestrator/episode/test_runtime_v1.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ora.orchestrator.spec.run_spec import file_sha256
from ora.agent.context import ContextProjector
from ora.environment import SyntheticSession
from ora.agent.toolkit.host.runtime.factory import build_tool_runtime
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.environment.backends.libero_pro.session import libero_osc_action_contract


ROOT = Path(__file__).resolve().parents[3]
PROMPT = ROOT / "ora/policy/core/thin-react-v1.md"
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
AGENT_RULES = ROOT / "ora/policy/planners/native/project-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}


def publish_synthetic(tmp_path: Path):
    # The run layout the toolkit reads a published observation from.
    source = tmp_path / "policy-observations"
    session = SyntheticSession(tmp_path / "native", "synthetic_tool_boundary", 0)
    session.start()
    observation, _ = session.observe(source)
    workspace = tmp_path / "sandbox/workspace"
    control = tmp_path / "control"
    state_store = PolicyStateStore(tmp_path, "test-run", "test-plan")
    snapshot = state_store.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        cause="reset",
        task_instruction="Move safely once.",
        previous_public_action=None,
        remaining_budget={"robot_decisions": 1, "model_calls": 2},
    )
    state_record = state_store.steps[-1]
    projector = ContextProjector(workspace, control, PROMPT, MANUAL, AGENT_RULES)
    pointer = projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="Move safely once.",
        remaining_budget={"robot_decisions": 1, "model_calls": 2},
        policy_state={
            "step": snapshot["step"], "path": state_record["path"],
            "sha256": state_record["sha256"],
            "state_sha256": snapshot["state_sha256"],
        },
    )
    tools = robot_tools(control, tmp_path / "events.jsonl", state_root=state_store.root)
    return observation, pointer, workspace, control, tools, state_store


def robot_tools(control: Path, events: Path, bounds=BOUNDS, *, allowed_tools=None, state_root=None):
    """The ``robot_runtime_v2`` tools on ``control``, built as a run builds them (the toolkit serves them)."""

    return build_tool_runtime(
        control_root=control, manual_path=MANUAL, event_path=events, workspace_bounds_m=bounds,
        allowed_tools=allowed_tools, state_root=state_root, profile_id="robot_runtime_v2",
        component_overrides=None,
    )


def scripted_pose(z: float) -> dict:
    """One scripted ``robot.submit_action`` proposal: the left arm to height ``z``."""
    return {"arm": "left", "pose": {"position_m": [0, 0, z], "orientation_xyzw": [0, 0, 0, 1]}}


def synthetic_plan() -> dict:
    return {
        "schema": "ora.run-plan.v1",
        "plan_sha256": "test-plan", "plan_id": "runtime-v1-m0-test",
        "experiment": "runtime-v1", "config": "minimal", "milestone": "M0",
        "output_workspace": "outputs/runtime-v1",
        "task": "synthetic_tool_boundary", "task_instruction": "Move safely once.",
        "seed": 0, "evidence_tier": "diagnostic",
        "policy": {
            "kind": "scripted", "model": "scripted-policy-v1",
            "provider": "trusted_scripted", "reasoning_effort": "none",
            "actions": [scripted_pose(1.02)],
        },
        "runtime": {"kind": "synthetic", "workspace_bounds_m": BOUNDS},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 2, "max_sim_steps": 5, "wall_time_s": 30},
        "policy_context": {
            "codex_project": [{"path": "ora/policy/planners/native/project-v1.md", "sha256": file_sha256(AGENT_RULES)}],
            "episode_prefix": [{"path": "ora/policy/core/thin-react-v1.md", "sha256": file_sha256(PROMPT)}],
            "tool_manual": [{"path": "ora/policy/embodiments/aloha-agilex/manual-v1.md", "sha256": file_sha256(MANUAL)}],
            "composition_sha256": "test",
        },
        "source": {"commit": "test", "dirty": False}, "qualification": {"required": False},
    }


class FakeLiberoEnvironment:
    def __init__(self):
        self.task_descriptions = ["put the bowl on the plate"]
        self.last_action = None
        self.closed = False
        self.step_count = 0
        self.gripper_hold_directive = -1.0
        self.current_raw_obs = {
            "agentview_image": np.zeros((1, 3, 4, 3), dtype=np.uint8),
            "agentview_depth": np.ones((1, 3, 4, 1), dtype=np.float32),
            "robot0_eye_in_hand_image": np.zeros((1, 3, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand_depth": np.ones((1, 3, 4, 1), dtype=np.float32),
            "robot0_eef_pos": np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32),
            "robot0_eef_quat": np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            "robot0_gripper_qpos": np.asarray([[0.04, -0.04]], dtype=np.float32),
        }
        calibration = {
            "intrinsic_K": np.eye(3), "extrinsic_cam2world": np.eye(4),
            "depth_near": 0.1, "depth_far": 10.0, "height": 3, "width": 4,
        }

        class VectorEnvironment:
            def get_env_attr(self, name):
                if name == "harness_camera_meta":
                    return [{
                        "agentview": calibration,
                        "robot0_eye_in_hand": calibration,
                    }]
                if name == "harness_motion_state":
                    return [{
                        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
                        "joint_positions": [0.0] * 7,
                        "joint_velocities": [0.0] * 7,
                        "external_wrench_clipped": None,
                        "external_force_magnitude_n": None,
                    }]
                if name == "harness_gripper_hold_directive":
                    return [-1.0]
                raise AssertionError(name)

        self.env = VectorEnvironment()

    def get_grip_site_pose(self):
        raw = self.current_raw_obs
        selected = raw[0] if isinstance(raw, np.ndarray) else {
            key: value[0] for key, value in raw.items()
        }
        return {
            "frame": "robosuite_grip_site",
            "site_name": "gripper0_grip_site",
            "position_m": selected["robot0_eef_pos"],
            "orientation_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0]),
        }

    def get_gripper_hold_directive(self):
        return self.gripper_hold_directive

    def reset(self, reset_state_ids=None):
        self.last_reset_state_ids = (
            None if reset_state_ids is None
            else np.asarray(reset_state_ids).tolist()
        )
        return {}, {}

    def step(self, action):
        self.step_count += 1
        self.last_action = np.asarray(action)
        delta = self.last_action[0, :3] * 0.05
        if isinstance(self.current_raw_obs, np.ndarray):
            selected = dict(self.current_raw_obs[0])
        else:
            selected = {
                key: value[0].copy() for key, value in self.current_raw_obs.items()
            }
        selected["robot0_eef_pos"] += delta
        if self.last_action[0, 6] > 0:
            self.gripper_hold_directive = 1.0
            selected["robot0_gripper_qpos"] = np.asarray(
                [0.0, 0.0], dtype=np.float32,
            )
        elif self.last_action[0, 6] < 0:
            self.gripper_hold_directive = -1.0
            selected["robot0_gripper_qpos"] = np.asarray(
                [0.04, -0.04], dtype=np.float32,
            )
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return {}, np.asarray([1.0]), np.asarray([self.step_count >= 2]), np.asarray([False]), {}

    def close(self):
        self.closed = True


class DivergingLiberoEnvironment(FakeLiberoEnvironment):
    def step(self, action):
        observation, reward, _term, trunc, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        selected["robot0_eef_pos"] = np.asarray([-0.9, 0.0, 0.5], dtype=np.float32)
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, np.asarray([False]), trunc, info


class GraduallyDivergingLiberoEnvironment(FakeLiberoEnvironment):
    """Move away in bounded increments without a one-tick corruption jump."""

    def step(self, action):
        self.step_count += 1
        self.last_action = np.asarray(action)
        selected = {
            key: value[0].copy() for key, value in self.current_raw_obs.items()
        } if not isinstance(self.current_raw_obs, np.ndarray) else dict(
            self.current_raw_obs[0]
        )
        selected["robot0_eef_pos"] = selected["robot0_eef_pos"].copy()
        selected["robot0_eef_pos"][0] -= 0.03
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return {}, np.asarray([0.0]), np.asarray([False]), np.asarray([False]), {}


class GraduallyRotatingAwayLiberoEnvironment(FakeLiberoEnvironment):
    """Accumulate bounded orientation error without corrupting one tick."""

    def __init__(self):
        super().__init__()
        self.orientation_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0])

    def get_grip_site_pose(self):
        value = super().get_grip_site_pose()
        value["orientation_xyzw"] = self.orientation_xyzw.copy()
        return value

    def step(self, action):
        self.step_count += 1
        self.last_action = np.asarray(action)
        self.orientation_xyzw = (
            Rotation.from_rotvec([0.0, 0.0, 0.15])
            * Rotation.from_quat(self.orientation_xyzw)
        ).as_quat()
        return {}, np.asarray([0.0]), np.asarray([False]), np.asarray([False]), {}


class GripperContactMotionEnvironment(FakeLiberoEnvironment):
    """Block closure while contact moves the EEF in bounded increments."""

    def step(self, action):
        self.step_count += 1
        self.last_action = np.asarray(action)
        selected = {
            key: value[0].copy() for key, value in self.current_raw_obs.items()
        } if not isinstance(self.current_raw_obs, np.ndarray) else dict(
            self.current_raw_obs[0]
        )
        selected["robot0_eef_pos"] = selected["robot0_eef_pos"].copy()
        selected["robot0_eef_pos"][0] += 0.03
        selected["robot0_gripper_qpos"] = np.asarray(
            [0.024, -0.024], dtype=np.float32,
        )
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return {}, np.asarray([0.0]), np.asarray([False]), np.asarray([False]), {}


class CurvedProgressLiberoEnvironment(FakeLiberoEnvironment):
    """Follow a bounded curved path that still reaches the requested target."""

    def step(self, action):
        observation, reward, _term, trunc, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        selected["robot0_eef_pos"] = selected["robot0_eef_pos"].copy()
        lateral_delta = {1: 0.025, 2: 0.0, 3: -0.025}.get(
            self.step_count, 0.0,
        )
        selected["robot0_eef_pos"][1] += lateral_delta
        self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, np.asarray([False]), trunc, info


class ContactPerturbedLiberoEnvironment(FakeLiberoEnvironment):
    """Deflect one finger without changing the pose-only command channel."""

    def step(self, action):
        observation, reward, term, trunc, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        if float(np.asarray(action)[0, 6]) == 0.0:
            selected["robot0_gripper_qpos"] = np.asarray(
                [0.032, -0.04], dtype=np.float32,
            )
            self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, term, trunc, info


class ContactBlockedGripperEnvironment(FakeLiberoEnvironment):
    """Keep a close directive while contact prevents full finger closure."""

    def step(self, action):
        observation, reward, _term, trunc, info = super().step(action)
        selected = dict(self.current_raw_obs[0])
        if float(np.asarray(action)[0, 6]) > 0.0:
            selected["robot0_gripper_qpos"] = np.asarray(
                [0.024, -0.024], dtype=np.float32,
            )
            self.current_raw_obs = np.asarray([selected], dtype=object)
        return observation, reward, np.asarray([False]), trunc, info


class RotatingLiberoEnvironment(FakeLiberoEnvironment):
    """Apply RoboSuite's configured physical rotation range end to end."""

    def __init__(self):
        super().__init__()
        self.orientation_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0])
        self.actions: list[np.ndarray] = []

    def get_grip_site_pose(self):
        value = super().get_grip_site_pose()
        value["orientation_xyzw"] = self.orientation_xyzw.copy()
        return value

    def step(self, action):
        normalized = np.asarray(action, dtype=np.float64)
        self.actions.append(normalized.copy())
        physical_rotvec = normalized[0, 3:6] * 0.5
        self.orientation_xyzw = (
            Rotation.from_rotvec(physical_rotvec)
            * Rotation.from_quat(self.orientation_xyzw)
        ).as_quat()
        return super().step(action)


def _fake_libero_metadata(tmp_path: Path, public_environment_context: dict) -> dict:
    return {
        "suite": "libero_goal_task", "task_id": 0,
        "task_language": "put the bowl on the plate", "task_name": "task",
        "bddl_file": str(tmp_path / "hidden.bddl"),
        "init_states_file": "hidden.pruned_init", "reset_id": 0,
        "trial_count": 50,
        "controller_action_contract": libero_osc_action_contract({
            "control_delta": True,
            "input_min": -1, "input_max": 1,
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        }),
        "public_environment_context": public_environment_context,
    }
