"""One public snapshot of a depth world: the arrays a planner worker builds its collision world from.

A session writes a snapshot (``write_world_inputs``, ``.npz``, no pickle) and a
worker reads it by path and sha256 (``read_world_inputs``), so a host tool reading
the same file gets the same world and hash. ``depth_world.py`` holds what a
snapshot becomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


WORLD_INPUT_SCHEMA = "ora.libero-world-input.v1"
WORLD_INPUT_SCHEMA_V2 = "ora.depth-world-input.v2"
CAMERAS = ("agentview", "wrist")
_CAMERA_FIELDS = ("depth_m", "intrinsic_cv", "cam2world_cv", "depth_range_m")
_STATE_FIELDS = (
    "base_pose_world_xyzw", "joint_positions", "gripper_finger_qpos", "gripper_directive",
)
INPUT_FIELDS = frozenset(
    {"schema"} | {f"{camera}_{name}" for camera in CAMERAS for name in _CAMERA_FIELDS}
    | set(_STATE_FIELDS)
)
_OTHER_ARM_FIELDS = ("other_arm_joint_positions", "other_arm_base_pose_world_xyzw")


class DepthWorldError(ValueError):
    """World inputs are missing, malformed or do not match their recorded hash."""


@dataclass(frozen=True)
class DepthCamera:
    name: str
    depth_m: Any
    intrinsic_cv: Any
    cam2world_cv: Any
    depth_range_m: tuple[float, float]


@dataclass(frozen=True)
class WorldInputs:
    """One public snapshot; ``content_sha256`` hashes the arrays, not the file bytes."""

    content_sha256: str
    cameras: tuple[DepthCamera, ...]
    base_pose_world_xyzw: Any
    joint_positions: Any
    gripper_finger_qpos: Any
    gripper_directive: float
    other_arm_joint_positions: Any = None
    other_arm_base_pose_world_xyzw: Any = None

    @property
    def finger_gap_m(self) -> float:
        return float(np.abs(self.gripper_finger_qpos).sum())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"ora.libero-world-input-content.v1\0")
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps([name, value.dtype.str, list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def pose_matrix(pose_xyzw: Any) -> Any:
    """A 4x4 transform from ``[x, y, z, qx, qy, qz, qw]``."""
    pose = np.asarray(pose_xyzw, dtype=np.float64)
    x, y, z, w = pose[3:] / np.linalg.norm(pose[3:])
    transform = np.eye(4)
    transform[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    transform[:3, 3] = pose[:3]
    return transform


def write_world_inputs(
    path: Path, *, cameras: Mapping[str, Mapping[str, Any]], base_pose_world_xyzw: Sequence[float],
    joint_positions: Sequence[float], gripper_finger_qpos: Sequence[float], gripper_directive: float,
    other_arm: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, str]:
    """Write one snapshot as an ``.npz`` (no pickle) and return its path and file sha256.

    ``cameras`` maps each camera to its published ``depth_m`` (metres, OpenCV
    top-left row-major), ``intrinsic_cv``, ``cam2world_cv`` (world frame) and
    ``depth_range_m`` (near, far), exactly as the policy observation holds them:
    exactly ``CAMERAS`` for a one-arm snapshot. ``other_arm`` (``joint_positions``,
    ``base_pose_world_xyzw``) makes it a two-arm snapshot with any named cameras.
    """
    if other_arm is None:
        arrays: dict[str, Any] = {"schema": np.asarray(WORLD_INPUT_SCHEMA)}
        names = list(CAMERAS)
        if set(cameras) != set(CAMERAS):
            raise DepthWorldError("world inputs need exactly the agentview and wrist cameras")
    else:
        names = sorted(cameras)
        arrays = {"schema": np.asarray(WORLD_INPUT_SCHEMA_V2), "camera_names": np.asarray(names),
                  "other_arm_joint_positions": np.asarray(other_arm["joint_positions"], dtype=np.float64),
                  "other_arm_base_pose_world_xyzw": np.asarray(other_arm["base_pose_world_xyzw"], dtype=np.float64)}
    for camera in names:
        record = cameras[camera]
        arrays[f"{camera}_depth_m"] = np.asarray(record["depth_m"], dtype=np.float32)
        arrays[f"{camera}_intrinsic_cv"] = np.asarray(record["intrinsic_cv"], dtype=np.float64)
        arrays[f"{camera}_cam2world_cv"] = np.asarray(record["cam2world_cv"], dtype=np.float64)
        arrays[f"{camera}_depth_range_m"] = np.asarray(record["depth_range_m"], dtype=np.float64)
    arrays["base_pose_world_xyzw"] = np.asarray(base_pose_world_xyzw, dtype=np.float64)
    arrays["joint_positions"] = np.asarray(joint_positions, dtype=np.float64)
    arrays["gripper_finger_qpos"] = np.asarray(gripper_finger_qpos, dtype=np.float64)
    arrays["gripper_directive"] = np.asarray(float(gripper_directive), dtype=np.float64)
    _validate_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DepthWorldError(f"refusing to overwrite world inputs: {path}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)
    return {"path": str(path), "sha256": _sha256_file(path)}


def _camera_names(arrays: Mapping[str, Any]) -> list[str]:
    """The snapshot's cameras, after checking its fields and schema."""
    schema = str(arrays.get("schema"))
    if schema == WORLD_INPUT_SCHEMA:
        names, fields = list(CAMERAS), INPUT_FIELDS
    elif schema == WORLD_INPUT_SCHEMA_V2 and "camera_names" in arrays:
        names = [str(name) for name in arrays["camera_names"].tolist()]
        fields = frozenset({"schema", "camera_names", *_OTHER_ARM_FIELDS, *_STATE_FIELDS}
                           | {f"{camera}_{name}" for camera in names for name in _CAMERA_FIELDS})
    else:
        raise DepthWorldError("world input fields or schema drifted")
    if set(arrays) != fields or not names:
        raise DepthWorldError("world input fields or schema drifted")
    return names


def _validate_arrays(arrays: Mapping[str, Any]) -> None:
    names = _camera_names(arrays)
    for camera in names:
        depth = arrays[f"{camera}_depth_m"]
        near_far = arrays[f"{camera}_depth_range_m"]
        if (depth.ndim != 2 or not depth.size or arrays[f"{camera}_intrinsic_cv"].shape != (3, 3)
                or arrays[f"{camera}_cam2world_cv"].shape != (4, 4) or near_far.shape != (2,)
                or not all(np.isfinite(arrays[f"{camera}_{name}"]).all() for name in _CAMERA_FIELDS[1:])
                or not 0 <= near_far[0] < near_far[1]):
            raise DepthWorldError(f"{camera} world input is malformed")
    joints = (7,) if len(names) == 2 and "camera_names" not in arrays else arrays["joint_positions"].shape
    shapes = {"base_pose_world_xyzw": (7,), "joint_positions": joints,
              "gripper_finger_qpos": (2,), "gripper_directive": ()}
    if "other_arm_joint_positions" in arrays:
        shapes.update(other_arm_joint_positions=joints, other_arm_base_pose_world_xyzw=(7,))
    for name, shape in shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise DepthWorldError(f"world input {name} is malformed")
    if not math.isclose(float(np.linalg.norm(arrays["base_pose_world_xyzw"][3:])), 1.0, abs_tol=1e-5):
        raise DepthWorldError("world input base quaternion is not normalized")
    if "other_arm_base_pose_world_xyzw" in arrays and not math.isclose(
            float(np.linalg.norm(arrays["other_arm_base_pose_world_xyzw"][3:])), 1.0, abs_tol=1e-5):
        raise DepthWorldError("world input other-arm base quaternion is not normalized")


def read_world_inputs(path: Path, sha256: str) -> WorldInputs:
    """Read a snapshot whose file must still hash to ``sha256``."""
    path = Path(path)
    if not path.is_file() or _sha256_file(path) != sha256:
        raise DepthWorldError("world input file is missing or its sha256 differs")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    _validate_arrays(arrays)
    cameras = tuple(DepthCamera(
        name=camera, depth_m=arrays[f"{camera}_depth_m"], intrinsic_cv=arrays[f"{camera}_intrinsic_cv"],
        cam2world_cv=arrays[f"{camera}_cam2world_cv"],
        depth_range_m=tuple(float(item) for item in arrays[f"{camera}_depth_range_m"]),
    ) for camera in _camera_names(arrays))
    return WorldInputs(
        content_sha256=_content_sha256(arrays), cameras=cameras, base_pose_world_xyzw=arrays["base_pose_world_xyzw"],
        joint_positions=arrays["joint_positions"], gripper_finger_qpos=arrays["gripper_finger_qpos"],
        gripper_directive=float(arrays["gripper_directive"]),
        other_arm_joint_positions=arrays.get("other_arm_joint_positions"),
        other_arm_base_pose_world_xyzw=arrays.get("other_arm_base_pose_world_xyzw"),
    )


def valid_spheres(value: Any) -> Any:
    """``[x, y, z, r]`` rows without the planner's unused slots."""
    spheres = np.asarray(value, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(spheres).all():
        raise DepthWorldError("robot spheres are non-finite")
    # CuRobo marks unused or detached sphere slots with a non-positive radius.
    return spheres[spheres[:, 3] > 0]


def other_arm_spheres_base(inputs: WorldInputs, spheres_own_base: Any) -> Any:
    """The other arm's spheres, given in its own base frame, in this arm's base frame."""
    spheres = valid_spheres(spheres_own_base)
    transform = np.linalg.inv(pose_matrix(inputs.base_pose_world_xyzw)) @ pose_matrix(
        inputs.other_arm_base_pose_world_xyzw)
    return np.hstack([spheres[:, :3] @ transform[:3, :3].T + transform[:3, 3], spheres[:, 3:]])


__all__ = [
    "CAMERAS", "DepthCamera", "DepthWorldError", "INPUT_FIELDS", "WORLD_INPUT_SCHEMA", "WORLD_INPUT_SCHEMA_V2",
    "WorldInputs", "other_arm_spheres_base", "pose_matrix", "read_world_inputs", "valid_spheres", "write_world_inputs",
]
