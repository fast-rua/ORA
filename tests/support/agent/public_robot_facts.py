"""Test helpers for public robot facts, shared by several test modules.

They were defined in ``tests/ora/agent/context/observation/test_public_robot_facts.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from tests.support.agent.surface_catalog import surface_of
from tests.support.agent.observation_fixtures import observation_fixture as _observation


ROOT = Path(__file__).resolve().parents[3]

#: A public-facts composition without complete current visual delivery. Its
#: registered example (``*_placement_local_vla_v3``) left the tree with VLA
#: support, and so did the explicit-motion v3 base it was derived from; the
#: maintained default of each backend now carries the capability set, minus
#: complete current visual delivery, beside the former example's view
#: configuration, so both producers stay covered.
_PUBLIC_FACTS_ONLY = {
    "libero": dict(
        base="libero_placement_direct_overhead_v1",
        dropped={"full_current_visual_v1"},
        added=set(),
        view_config={"automatic_actual_views": ["agentview", "wrist"], "coordinate_convention": "opencv_top_left_xy",
                     "default_observe_views": ["agentview", "wrist"], "derived_resolution": [256, 256],
                     "rgb_resolution": "native", "schema": "ora.policy-view-config.v1",
                     "current_observation_delivery": "all_public_v1", "controlled_frame_convention": "panda_grip_site",
                     "public_geometry_descriptor": "ora/policy/embodiments/franka-panda/descriptor-public-geometry-direct-v1-controller-feedback-v1-session-split-v1-flat-surface-v1.json"}),
    "robotwin": dict(
        base="robotwin_toolchain_b2_closed_noservo_v2",
        dropped={"full_current_visual_v1"},
        added={"policy_observer_camera_v1"},
        view_config={"automatic_actual_views": ["observer_camera", "auxiliary_contact_sheet"],
                     "coordinate_convention": "opencv_top_left_xy", "default_observe_views": ["observer_camera"],
                     "derived_resolution": [256, 256], "rgb_resolution": "native",
                     "schema": "ora.policy-view-config.v1", "current_observation_delivery": "all_public_v1",
                     "controlled_frame_convention": "aloha_grip_center_tcp",
                     "auxiliary_contact_sheet_cameras": ["head_camera", "front_camera", "left_camera", "right_camera"],
                     "public_geometry_descriptor": "ora/policy/embodiments/aloha-agilex/descriptor-public-geometry-v5-session-split-v1.json"}),
}


def _public_facts_surface(backend):
    import dataclasses

    spec = _PUBLIC_FACTS_ONLY[backend]
    base = surface_of(spec["base"])
    view = dict(spec["view_config"])
    descriptor = view["public_geometry_descriptor"]
    view["public_geometry_descriptor"] = {
        "path": descriptor, "sha256": hashlib.sha256((ROOT / descriptor).read_bytes()).hexdigest()}
    capabilities = tuple(sorted((set(base.capabilities) - spec["dropped"]) | spec["added"]))
    results = copy.deepcopy(dict(base.result_schemas))
    if "observer_camera" in view["automatic_actual_views"]:
        cameras = results["observe"]["properties"]["public_robot_facts"]["properties"]["cameras"]
        cameras["properties"]["observer_camera"] = {"$ref": "#/$defs/camera"}
    return dataclasses.replace(base, capabilities=capabilities, view_config=view, result_schemas=results)


def fixture(root, backend="libero", sequence=1, *, published=False):
    single_arm_defaults = {"robocasa": "robocasa_mobile_direct_v1", "robolab": "robolab_direct_v1"}
    surface = (surface_of("robodojo_placement_toolkit_v5") if backend == "robodojo"
               else surface_of("yam_perception_v1") if backend == "yam"
               else surface_of(single_arm_defaults[backend]) if backend in single_arm_defaults
               else _public_facts_surface(backend))
    cameras = (["head", "left_wrist", "right_wrist"] if backend == "robodojo"
               else ["overhead", "front", "wrist_left", "wrist_right"] if backend == "yam"
               else ["agentview", "wrist"] if backend in {"libero", "robocasa", "robolab"} else
               ["observer_camera", "head_camera", "front_camera", "left_camera", "right_camera"])
    observation, source = _observation(root, cameras, sequence=sequence, published=published)
    if backend == "robolab":
        effectors = {"right": {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robotiq_2f85_tcp", "gripper": .5, "jaw_separation_m": .0425,
            "gripper_finger_qpos": [.39], "gripper_command": 1., "gripper_command_target": "open"}}
    elif backend == "yam":
        # The YAM session's shape (backends/yam/session.py): one finger slide, its jaw separation twice it.
        effectors = {arm: {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1], "pose_frame": "yam_linear_4310_tcp",
            "gripper": .5, "jaw_separation_m": .04695, "gripper_opening_m": .04695,
            "gripper_finger_qpos": [.023475], "gripper_command_target": "open", "arm_joint_qpos": [0.0] * 6}
            for arm in ("left", "right")}
    elif backend in {"libero", "robocasa"}:
        effectors = {"right": {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robosuite_grip_site", "gripper": .5,
            "gripper_finger_qpos": [.012, -.028]}}
    else:
        effectors = {arm: {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "controlled_point_semantics": "parallel_jaw_grip_center_tcp", "gripper_command_target": "close",
            "gripper_state": {"commanded_opening_normalized": 0., "drive_target_opening_normalized": 0.,
                "measured_opening_normalized": .18181818, "measured_primary_joint_position": 0.}}
            for arm in ("left", "right")}
    if backend == "robodojo":
        from ora.contracts.gripper_feedback import gripper_feedback
        from ora.environment.controllers.robodojo_arx import public_feedback_contract
        effectors = {arm: {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robodojo_episode_local_ee_link", "gripper": 1.,
            "gripper_finger_qpos": [.044, .0345], "gripper_command": .8,
            "gripper_command_target": "close", "gripper_feedback": gripper_feedback(
                requested_opening_normalized=0., drive_target_opening_normalized=.8,
                measured_opening_normalized=1., effective_target_opening_normalized=.01/.054,
                tolerance_normalized=.03, drive_target_source="resolved_sdk_joint_position_target",
                measurement_source="primary_finger_encoder", pose_hold_semantics="preserve_last_applied_drive_target")}
            for arm in ("left", "right")}
        observation["controller_feedback"] = public_feedback_contract()
    observation["proprioception"] = {"end_effectors": effectors}
    if backend in {"robocasa", "robolab"}:
        observation["proprioception"]["single_arm"] = True
    if backend == "robocasa":
        # The RoboCasa session publishes its jaw command with every observation.
        effectors["right"].update(gripper_command=1.0, gripper_command_target="unknown")
        observation["proprioception"]["base"] = {"pose_xy_yaw": [1.5, -2.0, 0.5], "frame": "robocasa_world",
                                                 "torso_height_m": 0.1}
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
    observation["robot_geometry_verification"] = {
        "schema": GEOMETRY_VERIFICATION_SCHEMA,
        "descriptor_sha256": surface.view_config["public_geometry_descriptor"]["sha256"],
        "scope": "loaded_robot_geometry_only", "passed": True,
    }
    from ora.agent.evidence.policy_state import REQUIRED_ABSENT
    observation["forbidden_modalities_absent"] = sorted(REQUIRED_ABSENT)
    observation["main_camera"] = cameras[0]
    for name in cameras:
        calibration = {"intrinsic_cv": [[10, 0, 6], [0, 10, 5], [0, 0, 1]],
            "extrinsic_cv": [[1, 0, 0, sequence], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "cam2world_cv": [[1, 0, 0, -sequence], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "source_camera": "private_adapter_name", "object_gt": {"must_not_escape": True}}
        path = source / name / "calibration.json"
        path.write_text(json.dumps(calibration))
        observation["cameras"][name]["files"]["calibration"] = {
            "path": name + "/calibration.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}
        depth = source / name / "depth-m.npy"
        np.save(depth, np.ones((10, 12), dtype=np.float32))
        observation["cameras"][name]["files"]["depth_numpy_m_f32"] = {
            "path": name + "/depth-m.npy", "sha256": hashlib.sha256(depth.read_bytes()).hexdigest(),
            "bytes": depth.stat().st_size}
    (source / "observation.json").write_text(json.dumps(observation))
    return surface, observation, source


def published_fixture(tmp_path, backend):
    from ora.agent.evidence.policy_state import PolicyStateStore
    from ora.agent.context.observation.projector import ContextProjector
    from tests.support.agent.atomic_causal_runtime import PROMPT, MANUAL, PROJECT

    surface, observation, source = fixture(tmp_path, backend, published=True)
    store = PolicyStateStore(tmp_path, "public-data", "plan")
    budgets = {"model_calls": 11, "robot_decisions": 7, "wall_time_s": 179.5}
    store.publish(observation=observation, observation_root=source, cause="reset",
        task_instruction="Inspect public robot evidence.", remaining_budget=budgets, previous_public_action=None)
    projector = ContextProjector(tmp_path / "workspace", tmp_path / "control", PROMPT, MANUAL, PROJECT,
        main_camera=observation["main_camera"], run_root=tmp_path, view_config=surface.view_config)
    pointer = projector.publish(observation=observation, observation_root=source,
        task_instruction="Inspect public robot evidence.", remaining_budget=budgets, policy_state=store.steps[-1])
    return surface, pointer, projector, store
