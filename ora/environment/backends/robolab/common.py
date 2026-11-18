"""Numerical and artifact helpers for the RoboLab worker."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

OBSERVATION_SCHEMA = "ora-observation.v1"
SPARSE_RECEIPT_SCHEMA = "ora.robolab-sparse-motion-receipt.v1"
VERDICT_SCHEMA = "ora.robolab-official-verdict.v1"
#: RoboLab's graded Score inside the official verdict (``RobolabSession.robolab_score``).
SCORE_SCHEMA = "ora.robolab-subtask-score.v1"
RESET_SCHEMA = "ora.robolab-reset.v1"
FORBIDDEN_MODALITIES = [
    "actor_state", "object_gt_pose", "segmentation", "contact",
    "success_predicate", "expert_program", "task_source",
]


class RobolabError(RuntimeError):
    """Invalid source, control, sensor or lifecycle evidence."""


#: Seconds a worker waits for Kit's non-daemon threads after close before it ends itself.
CLOSE_EXIT_GRACE_S = 10.0


def hard_exit(delay_s: float = 0.0) -> None:
    """End a worker process without Kit's in-process teardown (``RobolabSession.close``).

    Kit's ``app.close()`` can segfault on teardown and leave the card's Vulkan device lost for every later
    start (10-18); a worker process that owns one session ends with ``os._exit`` instead, as RoboDojo's does.
    """
    if delay_s:
        time.sleep(delay_s)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                                    default=lambda item: array(item).tolist()) + "\n")
    temporary.replace(path)


def rotation_wxyz(quat_wxyz: Any) -> np.ndarray:
    q = array(quat_wxyz).astype(float)
    if q.shape != (4,) or not np.isfinite(q).all() or not 0.98 <= np.linalg.norm(q) <= 1.02:
        raise RobolabError("invalid measured quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_xyzw_from_matrix(matrix: Any) -> list[float]:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=float).reshape(3, 3)).as_quat()
    if xyzw[3] < 0:
        xyzw = -xyzw
    return [float(value) for value in xyzw]


def matrix_from_xyzw(xyzw: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat([float(value) for value in xyzw]).as_matrix()


def rotvec_error(target_xyzw: Sequence[float], current_xyzw: Sequence[float]) -> np.ndarray:
    """Rotation vector of target * inverse(current), world frame."""

    from scipy.spatial.transform import Rotation

    error = Rotation.from_quat(list(target_xyzw)) * Rotation.from_quat(list(current_xyzw)).inv()
    return error.as_rotvec()


def pose_error(target: Sequence[float], current: Sequence[float]) -> tuple[float, float]:
    translation = float(np.linalg.norm(np.asarray(target[:3]) - np.asarray(current[:3])))
    rotation = float(np.linalg.norm(rotvec_error(target[3:7], current[3:7])))
    return translation, rotation
