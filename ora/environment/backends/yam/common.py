"""Schemas and artifact helpers for the YAM worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OBSERVATION_SCHEMA = "ora-observation.v1"
SPARSE_RECEIPT_SCHEMA = "ora.yam-sparse-motion-receipt.v1"
VERDICT_SCHEMA = "ora.yam-official-verdict.v1"
RESET_SCHEMA = "ora.yam-reset.v1"
PLAN_SCHEMA = "ora.yam-motion-plan.v1"
FORBIDDEN_MODALITIES = [
    "actor_state", "object_gt_pose", "segmentation", "contact",
    "success_predicate", "expert_program", "task_source",
]
# Rig modes. ``sim``: the station model, kinematic. ``read``: the real cameras and joint
# encoders, motors never enabled, every motion answered with its plan. ``on``: the real
# arms move, each motion after the operator approves it (or ``auto`` when configured).
RIG_MODES = ("sim", "read", "on")


class YamError(RuntimeError):
    """Invalid configuration, sensor, control or lifecycle evidence."""


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                    default=lambda item: np.asarray(item).tolist()) + "\n")
    temporary.replace(path)


def rotvec_error(target_xyzw: Sequence[float], current_xyzw: Sequence[float]) -> np.ndarray:
    """Rotation vector of target * inverse(current), world frame."""

    from scipy.spatial.transform import Rotation

    error = Rotation.from_quat(list(target_xyzw)) * Rotation.from_quat(list(current_xyzw)).inv()
    return error.as_rotvec()


def pose_error(target: Sequence[float], current: Sequence[float]) -> tuple[float, float]:
    translation = float(np.linalg.norm(np.asarray(target[:3]) - np.asarray(current[:3])))
    rotation = float(np.linalg.norm(rotvec_error(target[3:7], current[3:7])))
    return translation, rotation
