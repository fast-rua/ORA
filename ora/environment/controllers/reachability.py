"""Backend-neutral Cartesian error and differential-IK primitives.

The pose error twist, a numerical Jacobian and one bounded damped-least-squares
step, from a forward-kinematics callable alone: what a local Cartesian servo and a
residual check need. Answering "can this arm hold this pose" belongs to the
shared planner's IK (``motion_planner.py``), and a single-seed descent no longer
stands in for it.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np


class ReachabilityInputRejected(ValueError):
    """One reachability input is structurally invalid."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details)


def _reject(code: str, message: str, **details: Any) -> None:
    raise ReachabilityInputRejected(code, message, **details)


def _quaternion_matrix_xyzw(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        _reject("quaternion", "Cartesian pose quaternion is invalid")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        _reject("quaternion", "Cartesian pose quaternion has zero norm")
    x, y, z, w = quaternion / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.asarray([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ], dtype=np.float64)
    if angle < 1e-8:
        return 0.5 * skew
    sine = math.sin(angle)
    if abs(sine) > 1e-6:
        return angle * skew / (2.0 * sine)
    # Stable near-pi axis extraction for large diagnostic rotations.
    diagonal = np.maximum(0.0, (np.diag(matrix) + 1.0) * 0.5)
    axis = np.sqrt(diagonal)
    if np.linalg.norm(axis) <= 1e-8:
        _reject("orientation_error", "Cartesian orientation error is singular")
    largest = int(np.argmax(axis))
    if largest == 0:
        axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
        axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
    elif largest == 1:
        axis[0] = math.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
        axis[2] = math.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
    else:
        axis[0] = math.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
        axis[1] = math.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
    axis /= np.linalg.norm(axis)
    return angle * axis


def cartesian_pose_error_twist(
    target_pose_xyzw: Sequence[float], current_pose_xyzw: Sequence[float],
) -> np.ndarray:
    """Return world-frame translation and rotation-vector error."""

    target = np.asarray(target_pose_xyzw, dtype=np.float64)
    current = np.asarray(current_pose_xyzw, dtype=np.float64)
    if (
        target.shape != (7,) or current.shape != (7,)
        or not np.isfinite(target).all() or not np.isfinite(current).all()
    ):
        _reject("pose", "Cartesian target/current pose is invalid")
    rotation = _quaternion_matrix_xyzw(target[3:]) @ (
        _quaternion_matrix_xyzw(current[3:]).T
    )
    return np.concatenate([target[:3] - current[:3], _rotation_vector(rotation)])


def numerical_cartesian_jacobian(
    joints: Sequence[float], *, fk_public_pose: Callable[[Sequence[float]], Sequence[float]],
    epsilon_rad: float,
) -> np.ndarray:
    """Numerically differentiate the public TCP pose in the world frame."""

    q = np.asarray(joints, dtype=np.float64)
    if q.ndim != 1 or not np.isfinite(q).all() or epsilon_rad <= 0.0:
        _reject("jacobian_input", "Cartesian Jacobian input is invalid")
    current = np.asarray(fk_public_pose(q), dtype=np.float64)
    columns: list[np.ndarray] = []
    for index in range(q.shape[0]):
        perturbed = q.copy()
        perturbed[index] += epsilon_rad
        pose = np.asarray(fk_public_pose(perturbed), dtype=np.float64)
        columns.append(cartesian_pose_error_twist(pose, current) / epsilon_rad)
    jacobian = np.stack(columns, axis=1)
    if jacobian.shape != (6, q.shape[0]) or not np.isfinite(jacobian).all():
        _reject("jacobian", "Cartesian numerical Jacobian is invalid")
    return jacobian


def bounded_dls_joint_delta(
    error_twist: Sequence[float], jacobian: Any, *, damping: float,
    maximum_translation_step_m: float, maximum_rotation_step_rad: float,
    maximum_joint_step_rad: float,
) -> np.ndarray:
    """Solve one bounded damped-least-squares Cartesian update."""

    error = np.asarray(error_twist, dtype=np.float64).copy()
    matrix = np.asarray(jacobian, dtype=np.float64)
    if error.shape != (6,) or matrix.ndim != 2 or matrix.shape[0] != 6:
        _reject("dls_shape", "Cartesian DLS input shape is invalid")
    if not np.isfinite(error).all() or not np.isfinite(matrix).all():
        _reject("dls_finite", "Cartesian DLS input is non-finite")
    for values, limit in ((error[:3], maximum_translation_step_m), (error[3:], maximum_rotation_step_rad)):
        norm = float(np.linalg.norm(values))
        if norm > limit:
            values *= float(limit) / norm
    regularized = matrix @ matrix.T + (float(damping) ** 2) * np.eye(6)
    try:
        delta = matrix.T @ np.linalg.solve(regularized, error)
    except np.linalg.LinAlgError as exc:
        _reject("dls_singular", "Cartesian DLS solve failed", error=str(exc))
    maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
    if maximum > maximum_joint_step_rad:
        delta *= float(maximum_joint_step_rad) / maximum
    if not np.isfinite(delta).all():
        _reject("dls_non_finite", "Cartesian DLS produced non-finite joints")
    return delta
