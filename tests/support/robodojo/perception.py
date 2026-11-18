"""Test helpers for robodojo perception, shared by several test modules.

They were defined in
``tests/ora/environment/backends/robodojo/test_robodojo_perception.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


BOUNDS = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _renderer_calibration(source: str, intrinsic: np.ndarray) -> dict[str, object]:
    """Mirror the nested numpy read-backs the session serialises per camera."""

    return {
        "source": "original_render_product_CameraParams",
        "render_product": f"/Render/RenderProduct_{source}",
        "original_robodojo_intrinsic": intrinsic.tolist(),
        "sdk_raw_aperture_intrinsic": intrinsic.tolist(),
        "parameters": {
            "cameraProjection": np.eye(4).ravel().tolist(),
            "cameraViewTransform": np.eye(4).ravel().tolist(),
            "renderProductResolution": [64, 48],
            "cameraModel": "pinhole",
            "cameraAperture": [20.955, 15.716],
            "cameraApertureOffset": [0.0, 0.0],
        },
        "cam2world_cv": np.eye(4).tolist(),
        "simulation_time_s": 2.0,
        "reference_time": {"referenceTimeNumerator": 120, "referenceTimeDenominator": 60},
        "current_physical_pose_verified": True,
    }


def _wrist_cam2world(sign: float) -> np.ndarray:
    """A wrist camera displaced sideways so its points do not dedupe with head."""

    pose = np.eye(4)
    pose[:3, 3] = [sign * 0.30, 0.05, 0.02]
    return pose


def _synthetic_robodojo_observation(
    root: Path, *, head_depth_patch: tuple[slice, slice, float] | None = None,
) -> tuple[Path, dict[str, object]]:
    """Write the RoboDojo per-camera layout with the session's exact keys.

    ``head_depth_patch`` (rows, columns, depth) overwrites part of the head
    background, e.g. to place points below the action workspace.
    """

    observation_id = "robodojo:hang_mugs:layout-0:observation-0003"
    target = root / observation_id
    target.mkdir(parents=True)
    intrinsic = np.array([[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]])
    cameras: dict[str, object] = {}
    for public, sign in (("head", 0.0), ("left_wrist", 1.0), ("right_wrist", -1.0)):
        source = f"cam_{public}"
        rgb = np.full((48, 64, 3), [52, 58, 66], dtype=np.uint8)
        depth = np.full((48, 64), 1.2, dtype=np.float32)
        if public == "head":
            rgb[14:34, 20:40] = [200, 60, 40]
            depth[14:34, 20:40] = 0.8
            if head_depth_patch is not None:
                rows, columns, value = head_depth_patch
                depth[rows, columns] = value
        cam2world = np.eye(4) if public == "head" else _wrist_cam2world(sign)
        camera_root = target / public
        camera_root.mkdir()
        rgb_path, depth_path = camera_root / "rgb.png", camera_root / "depth-m.npy"
        calibration_path = camera_root / "calibration.json"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth, allow_pickle=False)
        calibration_path.write_text(json.dumps({
            "source_camera": source, "intrinsic_cv": intrinsic.tolist(),
            "extrinsic_cv": np.linalg.inv(cam2world).tolist(),
            "cam2world_cv": cam2world.tolist(),
            "depth_unit": "metres", "depth_array_order": "opencv_top_left_row_major",
            "policy_quaternion_order": "xyzw", "world_frame": "robodojo_episode_local",
            "depth_kind": "distance_to_image_plane", "finite_depth_fraction": 1.0,
            "renderer_calibration": _renderer_calibration(source, intrinsic),
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        cameras[public] = {
            "source_camera": source, "width": 64, "height": 48,
            "mount": "fixed_world" if public == "head" else public,
            "calibration_dynamic": public != "head",
            "modalities": ["rgb", "depth", "calibration"],
            "files": {
                role: {
                    "path": path.relative_to(target).as_posix(),
                    "sha256": _sha256(path), "bytes": path.stat().st_size,
                }
                for role, path in (
                    ("rgb_lossless_png", rgb_path),
                    ("depth_numpy_m_f32", depth_path),
                    ("calibration", calibration_path),
                )
            },
        }
    payload = {
        "schema": "ora-observation.v1", "observation_id": observation_id,
        "task": "hang_mugs", "sequence": 3, "encoding": "metric_rgbd_v1",
        "captured_unix_s": 0.0, "main_camera": "head", "cameras": cameras,
        "proprioception": {
            "end_effectors": {
                arm: {
                    "pose_xyz_xyzw": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                    "pose_frame": "robodojo_episode_local_ee_link",
                    "gripper": 1.0, "gripper_finger_qpos": [0.044, 0.044],
                    "arm_joint_qpos": [0.0] * 6, "gripper_command": 1.0,
                    "gripper_command_target": "open",
                }
                for arm in ("left", "right")
            },
            "single_arm": False,
        },
        "previous_execution_receipt": None,
        "forbidden_modalities_absent": True,
    }
    (target / "observation.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return target, payload


class FakeHeadGrounding:
    backend_id = "sam3_text_test_double"
    backend_version = "v1"

    def configuration(self) -> dict[str, object]:
        return {"fixture": "robodojo_head"}

    def ground(
        self, *, rgb_u8: object, text: str, max_candidates: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert np.asarray(rgb_u8).shape == (48, 64, 3)
        assert text == "red mug"
        mask = np.zeros((48, 64), dtype=bool)
        mask[14:34, 20:40] = True
        return [{
            "source_index": 0, "rank": 0, "mask": mask,
            "backend_score": 0.9, "bbox_xyxy": [20, 14, 40, 34],
        }][:max_candidates], {"request_id": "ground-x5", "returned_candidate_count": 1}


class FakeX5Proposer:
    backend_id = "contact_graspnet_test_double"
    backend_version = "v1"

    def configuration(self) -> dict[str, object]:
        return {"pregrasp_offset_m": 0.04}

    def propose(self, **kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        assert kwargs["compiler_identity"]["id"] == (
            "contact_graspnet_to_robodojo_x5_left_link6_v1"
        )
        scene = np.asarray(kwargs["scene_points_camera_m"])
        target = np.asarray(kwargs["target_points_camera_m"])
        assert scene.ndim == 2 and scene.shape[1] == 3 and len(target) == 20 * 20
        # Contact-GraspNet's Panda origin sits 103.4 mm behind the visible
        # 0.8 m surface; approach runs along camera +z into the scene.
        return [{
            "candidate_id": "grasp-0123456789abcdef01234567",
            "source_index": 0, "rank": 0,
            "source_gripper_origin_xyz_m": [0.0, 0.0, 0.6966],
            "source_pregrasp_origin_xyz_m": [0.0, 0.0, 0.6566],
            "approach_world": [0.0, 0.0, 1.0], "jaw_world": [1.0, 0.0, 0.0],
            "gripper_opening_m": 0.05, "backend_score": 0.7,
            "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
            "checks": {"workspace": "passed", "aperture": "passed",
                       "ik": "not_checked", "collision": "not_checked"},
            "execution_ready": False,
        }], {"request_id": "grasp-x5", "rejected": []}
