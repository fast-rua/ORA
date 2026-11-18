"""Forward kinematics of the PandaOmron arm at the current base pose, and the geometry of its paths.

Solving and planning belong to the shared motion base (``shared_motion.py``);
this is what the executor and the pacing need of the robot's own model. Every
function works on scratch copies of the robot's joints and reads nothing of the
scene, and none writes the live simulation.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from ora.environment.controllers import robocasa_omron as C

from .common import RoboCasaError, _quaternion_error_rotvec, matrix_from_quaternion_xyzw, quaternion_xyzw_from_matrix


def joint_path_samples(start: Sequence[float], targets: Sequence[Sequence[float]]) -> list[Any]:
    """Arm configurations every ``C.JOINT_PATH_CHECK_STRIDE_RAD`` along the joint-space interpolation."""

    import math

    import numpy as np

    samples = []
    current = np.asarray(start, dtype=np.float64)
    for target in targets:
        goal = np.asarray(target, dtype=np.float64)
        count = max(1, int(math.ceil(float(np.max(np.abs(goal - current))) / C.JOINT_PATH_CHECK_STRIDE_RAD)))
        samples.extend(current + (goal - current) * (index / count) for index in range(1, count + 1))
        current = goal
    return samples


def joint_range(model: Any, name: str) -> tuple[float, float]:
    """The lower and upper limit of one named joint in the loaded model."""

    import mujoco

    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0:
        raise RoboCasaError(f"joint is missing: {name}")
    low, high = model.jnt_range[joint]
    return float(low), float(high)


def grip_poses(model: Any, data: Any, samples: Sequence[Sequence[float]]) -> list[list[float]]:
    """Grip-site poses (xyz + xyzw) of arm joint samples, by forward kinematics on a scratch copy."""

    import mujoco
    import numpy as np

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, C.GRIP_SITE)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in C.ARM_JOINTS]
    qadr = np.asarray([model.jnt_qposadr[joint] for joint in joints])
    poses = []
    for sample in samples:
        scratch.qpos[qadr] = np.asarray(sample, dtype=np.float64)
        mujoco.mj_kinematics(model, scratch)
        poses.append([float(v) for v in scratch.site_xpos[site]] + quaternion_xyzw_from_matrix(scratch.site_xmat[site]))
    return poses


def line_cuts(start_pose: Sequence[float], target_pose: Sequence[float]) -> list[list[float]]:
    """Grip-site poses along the straight line, every ``C.LINE_STEP_M`` and ``C.LINE_STEP_RAD`` of turn; the last is the target."""

    import math

    import numpy as np

    start = np.asarray(start_pose[:3], dtype=np.float64)
    goal = np.asarray(target_pose[:3], dtype=np.float64)
    rotvec = np.asarray(_quaternion_error_rotvec(target_pose[3:7], start_pose[3:7]), dtype=np.float64)
    count = max(1, int(math.ceil(max(float(np.linalg.norm(goal - start)) / C.LINE_STEP_M,
                                     float(np.linalg.norm(rotvec)) / C.LINE_STEP_RAD))))
    cuts = []
    for index in range(1, count + 1):
        fraction = index / count
        orientation = list(target_pose[3:7]) if index == count else _rotate_quaternion(start_pose[3:7], rotvec * fraction)
        cuts.append([*(float(value) for value in start + (goal - start) * fraction), *orientation])
    return cuts


def hand_in_base(model: Any, data: Any, samples: Sequence[Sequence[float]]) -> list[Any]:
    """The hand body's 4x4 transform in the arm base frame at each arm joint sample, on a scratch copy."""

    import mujoco
    import numpy as np

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in C.ARM_JOINTS]
    qadr = np.asarray([model.jnt_qposadr[joint] for joint in joints])
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.ARM_BASE_BODY)
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.HAND_BODY)
    if base < 0 or hand < 0:
        raise RoboCasaError("arm base or hand body is missing")

    def transform(body: int) -> Any:
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(scratch.xmat[body], dtype=np.float64).reshape(3, 3)
        matrix[:3, 3] = scratch.xpos[body]
        return matrix

    answers = []
    for sample in samples:
        scratch.qpos[qadr] = np.asarray(sample, dtype=np.float64)
        mujoco.mj_kinematics(model, scratch)
        answers.append(np.linalg.inv(transform(base)) @ transform(hand))
    return answers


def _rotate_quaternion(quaternion_xyzw: Sequence[float], rotvec_world: Any) -> list[float]:
    """``quaternion_xyzw`` turned by the world-frame rotation vector ``rotvec_world``."""

    import numpy as np

    rotation = np.asarray(matrix_from_quaternion_xyzw(quaternion_xyzw))
    angle = float(np.linalg.norm(rotvec_world))
    if angle < 1e-12:
        return list(quaternion_xyzw)
    axis = np.asarray(rotvec_world, dtype=np.float64) / angle
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    turn = np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)
    return quaternion_xyzw_from_matrix(turn @ rotation)


def jaw_equivalent(pose: Sequence[float]) -> list[float]:
    """The same parallel-jaw grasp rotated 180 degrees about the approach axis."""

    import numpy as np

    rotation = np.asarray(matrix_from_quaternion_xyzw(pose[3:7]))
    flip = np.diag([-1.0, -1.0, 1.0])
    return list(pose[:3]) + quaternion_xyzw_from_matrix(rotation @ flip)
