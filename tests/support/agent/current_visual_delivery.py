"""Test helpers for current visual delivery, shared by several test modules.

They were defined in ``tests/ora/agent/context/test_current_visual_delivery.py``;
test modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from PIL import Image

from ora.agent.composition.baseline import load_native_baseline
from ora.agent.context.observation.projector import ContextProjector
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.environment.synthetic import SyntheticSession
from tests.support.agent.observation_delivery import Delivery, host
from tests.support.agent.surface_catalog import surface_of


ROOT = Path(__file__).resolve().parents[3]


def published(root, backend, *, after_action=False):
    from tests.support.agent.atomic_causal_runtime import PROMPT, MANUAL, PROJECT
    if backend == "robodojo":  # its default has its own baseline file
        instance = json.loads((ROOT / "ora/policy/baselines/robodojo-toolkit-v6.json").read_text())["instances"][backend]
    else:
        instance = load_native_baseline(ROOT)["instances"][backend]
    surface = surface_of(instance["tool_surface"])
    cameras = instance["runtime"]["policy_cameras"]
    environment = SyntheticSession(root / "environment", "visual", 0,
        main_camera=cameras[0], camera_names=tuple(cameras))
    environment.start()
    observation, _ = environment.observe(root / "source")
    source = root / "source" / observation["observation_id"]
    # Synthetic sensors stand in for the two maintained robot workers; supply
    # their explicit measured state and descriptor verification, never infer it
    # from a commanded gripper value.
    assert surface.supports("current_public_robot_facts_v1")
    if backend in {"libero_pro", "robocasa"}:
        effectors = {"right": {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robosuite_grip_site", "gripper": .5,
            "gripper_command_target": "open", "gripper_finger_qpos": [.012, -.028]}}
    elif backend == "robolab":
        effectors = {"right": {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robotiq_2f85_tcp", "gripper": .5, "jaw_separation_m": .0425,
            "gripper_finger_qpos": [.39], "gripper_command": 1., "gripper_command_target": "open"}}
    else:
        effectors = {arm: {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "controlled_point_semantics": "parallel_jaw_grip_center_tcp",
            "gripper_command_target": "open", "gripper": .18181818,
            "gripper_state": {"commanded_opening_normalized": 1.,
                "drive_target_opening_normalized": 1.,
                "measured_opening_normalized": .18181818,
                "measured_primary_joint_position": 0.}}
            for arm in ("left", "right")}
    if backend == "robodojo":
        # The RoboDojo default instance: the ARX link6 frame and the
        # typed gripper feedback the worker publishes (test_public_robot_facts).
        from ora.contracts.gripper_feedback import gripper_feedback
        from ora.environment.controllers.robodojo_arx import public_feedback_contract
        effectors = {arm: {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robodojo_episode_local_ee_link", "gripper": 1.,
            "gripper_finger_qpos": [.044, .0345], "gripper_command": .8,
            "gripper_command_target": "open", "gripper_feedback": gripper_feedback(
                requested_opening_normalized=1., drive_target_opening_normalized=1.,
                measured_opening_normalized=1., effective_target_opening_normalized=1.,
                tolerance_normalized=.03, drive_target_source="resolved_sdk_joint_position_target",
                measurement_source="primary_finger_encoder", pose_hold_semantics="preserve_last_applied_drive_target")}
            for arm in ("left", "right")}
        observation["controller_feedback"] = public_feedback_contract()
    observation["proprioception"]["end_effectors"] = effectors
    if backend in {"robocasa", "robolab"}:
        observation["proprioception"]["single_arm"] = True
    if backend == "robocasa":
        observation["proprioception"]["base"] = {"pose_xy_yaw": [1.5, -2.0, 0.5], "frame": "robocasa_world",
                                                 "torso_height_m": 0.1}
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
    observation["robot_geometry_verification"] = {
        "schema": GEOMETRY_VERIFICATION_SCHEMA,
        "descriptor_sha256": surface.view_config["public_geometry_descriptor"]["sha256"],
        "scope": "loaded_robot_geometry_only", "passed": True,
    }
    # Nontrivial public calibration, including a moving optical origin, catches
    # the earlier mistake of silently substituting EEF position for the camera.
    for index, camera in enumerate(cameras):
        rgb = source / camera / "rgb.png"
        with Image.open(rgb) as image:
            pixels = image.copy()
        pixels.putpixel((1, 1), (index * 31, 117, 29))
        pixels.save(rgb)
        observation["cameras"][camera]["files"]["rgb_lossless_png"].update(
            sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(), bytes=rgb.stat().st_size)
        calibration = source / camera / "calibration.json"
        value = json.loads(calibration.read_text())
        value["intrinsic_cv"][0][0] = 50.25 + index
        value["extrinsic_cv"][0][3] = -0.123 - index
        value["extrinsic_cv"][1][3] = -0.234
        value["extrinsic_cv"][2][3] = -1.345
        if backend in {"robotwin", "robodojo"}:
            value["extrinsic_cv"] = value["extrinsic_cv"][:3]
        value["object_gt_pose"] = [999, 999, 999]  # Never propagate arbitrary fields.
        calibration.write_text(json.dumps(value))
        observation["cameras"][camera]["files"]["calibration"].update(
            sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(), bytes=calibration.stat().st_size)
        depth = source / camera / "depth-m.npy"
        np.save(depth, np.array([[0, np.nan, 1, 2], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float32))
        observation["cameras"][camera]["files"]["depth_numpy_m_f32"].update(
            sha256=hashlib.sha256(depth.read_bytes()).hexdigest(), bytes=depth.stat().st_size)
    (source / "observation.json").write_text(json.dumps(observation))
    store = PolicyStateStore(root, "visual-test", "plan-test")
    store.publish(observation=observation, observation_root=source, cause="mutation" if after_action else "initial",
        task_instruction="Inspect", previous_public_action=None, remaining_budget={})
    projector = ContextProjector(root / "workspace", root / "control", PROMPT, MANUAL, PROJECT,
        main_camera=cameras[0], run_root=root, view_config=surface.view_config,
        camera_roles={name: role for name, role in {"wrist": "right_wrist", "left_camera": "left_wrist", "right_camera": "right_wrist",
                                                    "left_wrist": "left_wrist", "right_wrist": "right_wrist"}.items() if name in cameras})
    context = projector.publish(observation=observation, observation_root=source,
        task_instruction="Inspect", remaining_budget={}, policy_state=store.steps[-1],
        previous_action={"schema": "ora-action.v3", "action": "move_pose", "action_id": "action-1",
            "observation_id": "earlier", "arguments": {}} if after_action else None)
    return context, projector, store, surface, source


def delivery_for(root, context, surface, **options):
    """The observations the Pi core delivers for ``context`` (its Python draft, hints applied)."""

    planner = host(root, tools=SimpleNamespace(surface=surface, controlled_end_effector="right"),
                   observation_id=context["current_visual_observation"]["observation_id"],
                   actual_sequence=0, **options)
    return Delivery(planner)
