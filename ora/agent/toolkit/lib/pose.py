"""Measured-frame pose compilation, independent of tool preferences."""

from . import _motion_math
from .values import JsonValue, PoseM


def compile_target(request: JsonValue, current: PoseM) -> JsonValue:
    """Pure computation: compile a world/end-effector request in metres/degrees.

    Replay: recomputed locally; no simulator or IK. Quaternion order is xyzw.
    Example: compile_target(JsonValue.of(request), measured_pose).
    """
    return JsonValue.of(
        _motion_math.compile_enhanced_move_pose(
            request.json(),
            current_pose_xyz_xyzw=[*current.position_m, *current.orientation_xyzw],
            controlled_point="end_effector",
        )
    )


def approach_orientation(direction: tuple[float, float, float], roll_rad: float,
                         *, local_axes: JsonValue) -> tuple[float, float, float, float]:
    """Pure computation: align declared approach/closing axes to a world direction.

    Example: approach_orientation((0, 0, -1), 0., local_axes=axes). Inputs are
    unitless world direction and local axes, roll in radians; output xyzw. No
    candidate preference, translation, IK or physical state access.
    """
    import math
    local_axes = local_axes.json()
    import numpy as np

    forward = np.asarray(direction, dtype=np.float64)
    axes = np.asarray([local_axes["approach"], local_axes["closing"]], dtype=np.float64)
    if (forward.shape != (3,) or axes.shape != (2, 3)
            or not np.isfinite(forward).all() or not np.isfinite(axes).all()
            or not math.isfinite(roll_rad) or np.linalg.norm(forward) == 0
            or any(np.linalg.norm(axis) == 0 for axis in axes)
            or abs(float(np.dot(axes[0], axes[1]))) > 1e-8):
        raise ValueError("approach requires finite nonzero direction and orthogonal local axes")
    forward /= np.linalg.norm(forward)
    reference = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(reference, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    cos_roll, sin_roll = math.cos(roll_rad), math.sin(roll_rad)
    rolled_right = right * cos_roll + up * sin_roll
    rolled_up = up * cos_roll - right * sin_roll
    # World columns in (approach, closing, third) order, and the local frame in
    # the same order. R = world @ local.T then maps each declared local axis to
    # its world counterpart, whichever columns the descriptor put them in.
    approach_local = np.asarray(local_axes["approach"], dtype=np.float64)
    closing_local = np.asarray(local_axes["closing"], dtype=np.float64)
    approach_local /= np.linalg.norm(approach_local)
    closing_local /= np.linalg.norm(closing_local)
    world = np.stack([forward, rolled_right, rolled_up], axis=1)
    local = np.stack(
        [approach_local, closing_local, np.cross(approach_local, closing_local)], axis=1
    )
    matrix = world @ local.T
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            float((matrix[2, 1] - matrix[1, 2]) / scale),
            float((matrix[0, 2] - matrix[2, 0]) / scale),
            float((matrix[1, 0] - matrix[0, 1]) / scale),
            float(0.25 * scale),
        )
    index = int(np.argmax(np.diag(matrix)))
    i, j, k = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][index]
    scale = math.sqrt(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k]) * 2.0
    parts = [0.0, 0.0, 0.0]
    parts[i] = 0.25 * scale
    parts[j] = float((matrix[j, i] + matrix[i, j]) / scale)
    parts[k] = float((matrix[k, i] + matrix[i, k]) / scale)
    return (*parts, float((matrix[k, j] - matrix[j, k]) / scale))
