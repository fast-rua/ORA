"""Schemas, the backend error and shared helpers of the RoboCasa session."""

from __future__ import annotations

import math
from typing import Any, Sequence

# Generic helpers shared with the LIBERO backend (pure Python, no simulator imports).
from ora.environment.backends.libero_pro.common import (  # noqa: F401  re-exported
    _atomic_json,
    _file_hash,
    _jsonable,
    _quaternion_error_rotvec,
)

OBSERVATION_SCHEMA = "ora-observation.v1"
SPARSE_RECEIPT_SCHEMA = "ora.robocasa-sparse-motion-receipt.v1"
VERDICT_SCHEMA = "ora.robocasa-official-verdict.v1"
RESET_SCHEMA = "ora.robocasa-reset.v1"
SOURCE_SCHEMA = "ora.robocasa-source.v1"
REGION_ORDER_SCHEMA = "ora.robocasa-region-order.v1"
FORBIDDEN_MODALITIES = [
    "actor_state", "object_gt_pose", "segmentation", "contact",
    "success_predicate", "expert_program", "task_source",
]


class RoboCasaError(RuntimeError):
    """A RoboCasa worker contract was violated."""


def wrap_angle(value: float) -> float:
    """Map an angle to (-pi, pi]."""

    return math.atan2(math.sin(value), math.cos(value))


def quaternion_xyzw_from_matrix(matrix: Any) -> list[float]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64).reshape(3, 3)).as_quat()
    if xyzw[3] < 0:
        xyzw = -xyzw
    return [float(value) for value in xyzw]


def matrix_from_quaternion_xyzw(xyzw: Sequence[float]) -> Any:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat([float(value) for value in xyzw]).as_matrix()


def pose_error(target: Sequence[float], current: Sequence[float]) -> tuple[float, float]:
    """Translation (m) and rotation (rad) between two xyz+xyzw poses."""

    import numpy as np

    translation = float(np.linalg.norm(np.asarray(target[:3]) - np.asarray(current[:3])))
    rotation = float(np.linalg.norm(_quaternion_error_rotvec(target[3:7], current[3:7])))
    return translation, rotation
