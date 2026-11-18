"""Test helpers for manipulation skill driver, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/skills/test_manipulation_skill_driver.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from ora.agent.context import ContextProjector
from ora.agent.toolkit.host.skills import driver as skill_driver
from tests.support.agent.surface_catalog import handlers_only
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.environment import SyntheticSession


ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
PROMPT = ROOT / "tests/data/prompts/startup-prompt-v1.md"
AGENTS = ROOT / "ora/policy/planners/native/project-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}

CONTACT = {"position_m": [0.2, 0.1, 0.9], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


def _runtime(tmp_path: Path):
    source = tmp_path / "source"
    session = SyntheticSession(
        tmp_path / "native", "skill_driver_synthetic", 0,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    session.start()
    observation, _images = session.observe(source)
    control = tmp_path / "control"
    projector = ContextProjector(
        tmp_path / "workspace", control, PROMPT, MANUAL, AGENTS,
        expose_manual=False, main_camera="agentview",
    )
    projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="test",
        remaining_budget={"robot_decisions": 8, "model_calls": 8},
        policy_state={
            "step": 0, "path": "unused", "sha256": "unused",
            "state_sha256": "unused",
        },
    )
    # The handlers serve this fixture, as they did before the surface declared the toolkit.
    runtime = handlers_only(ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS,
        state_root=None, tool_surface_profile="native_atomic_causal_v1",
        controlled_end_effector="right",
    ))
    # A single-arm target-state surface that does not declare the skill. The
    # fixture grants exactly what a declaring profile would: the capability and
    # the name in `allowed_tools`.
    runtime.surface = replace(
        runtime.surface,
        capabilities=(*runtime.surface.capabilities, skill_driver.CAPABILITY),
    )
    runtime.allowed_tools = (*runtime.allowed_tools, "grasp_execute")
    return runtime


def _observation_id(runtime) -> str:
    control = json.loads(
        (runtime.control_root / "current-observation.json").read_text(encoding="utf-8")
    )
    return str(control["observation_id"])


def _publish_grasp_candidate(runtime, *, observation_id: str, candidate_id="grasp-a1"):
    root = runtime.artifact_root / "grasp-queries" / "grasp-query-fixture"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "query_ref": "grasp-query-fixture",
        "identity": {"observation_id": observation_id, "target_ref": "target-1"},
        "compiled_candidates": [{
            "candidate_id": candidate_id, "rank": 0,
            "pregrasp_pose": {
                "position_m": [0.2, 0.1, 0.96],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "contact_pose": dict(CONTACT),
            "gripper_opening_m": 0.04,
        }],
    }), encoding="utf-8")


def _submitted(runtime) -> dict:
    return json.loads(
        (runtime.control_root / "submitted-action.json").read_text(encoding="utf-8")
    )


def _driven(runtime, action, stage, index, observation):
    return runtime.drive_skill_stage(
        action, stage=stage, stage_index=index, observation=observation,
        observation_root=runtime.control_root, completed=[],
        capture_observation=lambda: (observation, runtime.control_root),
    )
