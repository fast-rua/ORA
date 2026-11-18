"""Test helpers for robodojo observe result, shared by several test modules.

They were defined in
``tests/ora/environment/backends/robodojo/test_robodojo_observe_result.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.support.agent.surface_catalog import catalog_grants, handlers_only, surface_of
from tests.support.agent.toolkit_run import served
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.context.observation.contact_points import project_policy_public_facts
from tests.support.robodojo.perception import (
    BOUNDS, FakeHeadGrounding, FakeX5Proposer, _synthetic_robodojo_observation,
)


ROOT = Path(__file__).resolve().parents[3]
SURFACE = "robodojo_placement_toolkit_v5"

# Recorded verbatim from run b50ff4c37763 observation-0002 (imitate_sorting_sequence).
RECORDED_CONTROLLER_FEEDBACK = {
    "arrival_comparison": "<=",
    "gripper_drive_limit": {
        "applies_to": "explicit_gripper_commands",
        "coupling": "both_drive_targets_from_limited_primary_target",
        "kind": "measured_primary_relative", "maximum_normalized_offset": 0.2,
        "pose_hold": "preserve_last_applied_drive_target",
        "reference": "current_primary_finger_encoder",
    },
    "gripper_tolerance_normalized": 0.03, "native_actions_per_command": 20,
    "orientation_tolerance_rad": 0.1, "position_tolerance_m": 0.01,
    "schema": "ora.controller-feedback.v1",
}


def _gripper_feedback(measured: float, requested: float, drive: float) -> dict:
    command = {0.0: "close", 1.0: "open"}[requested]
    effective = 0.1851851851851852 if requested == 0.0 else 1.0
    residual = abs(measured - effective)
    return {
        "schema": "ora.gripper-feedback.v1",
        "requested_opening_normalized": requested, "drive_target_opening_normalized": drive,
        "measured_opening_normalized": measured, "effective_target_opening_normalized": effective,
        "requested_command": command,
        "arrival_check": {"comparison": "<=", "residual_normalized": residual,
                          "tolerance_normalized": 0.03, "within_tolerance": residual <= 0.03},
        "drive_target_source": "resolved_sdk_joint_position_target",
        "measurement_source": "primary_finger_encoder",
        "pose_hold_semantics": "preserve_last_applied_drive_target",
    }


def _runtime(tmp_path: Path, *, handlers: bool = False, serve: bool = True) -> ManipulationToolRuntime:
    source, observation = _synthetic_robodojo_observation(tmp_path / "observations")
    observation["controller_feedback"] = json.loads(json.dumps(RECORDED_CONTROLLER_FEEDBACK))
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
    observation["robot_geometry_verification"] = {
        "schema": GEOMETRY_VERIFICATION_SCHEMA, "scope": "loaded_robot_geometry_only", "passed": True,
        "descriptor_sha256": surface_of(SURFACE).view_config["public_geometry_descriptor"]["sha256"],
    }
    effectors = observation["proprioception"]["end_effectors"]
    # One open arm and one arm holding a stalled close, as in the recorded scene.
    effectors["left"]["gripper_feedback"] = _gripper_feedback(1.0, 1.0, 1.0)
    effectors["right"].update(gripper=0.5, gripper_command=0.0, gripper_command_target="close",
                              gripper_finger_qpos=[0.02, 0.02])
    effectors["right"]["gripper_feedback"] = _gripper_feedback(0.5, 0.0, 0.3)
    (source / "observation.json").write_text(json.dumps(observation, sort_keys=True, indent=2) + "\n")
    control = tmp_path / "control"
    control.mkdir()
    runtime = ManipulationToolRuntime(
        control_root=control,
        manual_path=ROOT / "ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md",
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS, state_root=None,
        tool_surface_profile=SURFACE, toolkit_grants=catalog_grants(SURFACE), controlled_end_effector="right",
        segmentation_backend=FakeHeadGrounding(), grounding_backend=FakeHeadGrounding(),
        grasp_backend=FakeX5Proposer(), public_facts_projector=project_policy_public_facts)
    pointer = {"open": True, "observation_id": observation["observation_id"],
               "observation_path": str(source / "observation.json")}
    runtime._current = lambda: (pointer, source, observation)
    # Served as a run serves it (``toolkit_run.served``); ``handlers=True``: by the legacy tool handlers;
    # ``serve=False``: by neither yet, for a caller that changes the observation first.
    if handlers:
        return handlers_only(runtime)
    return served(runtime) if serve else runtime
