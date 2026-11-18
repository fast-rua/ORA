"""Test helpers for atomic causal runtime, shared by several test modules.

They were defined in
``tests/ora/agent/session_core/test_atomic_causal_runtime.py``; test modules import
them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from pathlib import Path

from ora.agent.context import ContextProjector
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from tests.support.agent.toolkit_run import served
from ora.environment import SyntheticSession


ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
PROMPT = ROOT / "tests/data/prompts/episode-task-v1.md"
PROJECT = ROOT / "ora/policy/environments/synthetic/boundary-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}


def _runtime(
    tmp_path: Path, *, tool_surface_profile: str = "native_atomic_causal_v1", serve: bool = True,
) -> ManipulationToolRuntime:
    source = tmp_path / "source"
    session = SyntheticSession(
        tmp_path / "environment", "atomic", 0, max_actions=8,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    session.start()
    observation, _images = session.observe(source)
    control = tmp_path / "control"
    workspace = tmp_path / "workspace"
    projector = ContextProjector(
        workspace, control, PROMPT, MANUAL, PROJECT,
        expose_manual=False, main_camera="agentview",
    )
    projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="atomic qualification",
        remaining_budget={"robot_decisions": 8, "model_calls": 16},
        policy_state={
            "step": 0, "path": "unused", "sha256": "unused",
            "state_sha256": "unused",
        },
    )
    # Served as a run serves it (``toolkit_run.served``); ``serve=False`` leaves that to a caller that changes it first.
    runtime = ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS,
        state_root=None, tool_surface_profile=tool_surface_profile,
        controlled_end_effector="right",
    )
    return served(runtime) if serve else runtime
