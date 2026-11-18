"""The depth world for a two-arm robot: named cameras, the other arm in the world, the tool's own frame."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ora.environment.controllers.depth_world import (
    PUBLIC_DEPTH_VOXEL_X5_V1, ROBOT_WORLDS, WORLD_MODEL, DepthWorldError, Scene, curobo_v2_grid, other_arm_spheres_base,
    pose_matrix, read_world_inputs, tool_from_hand, world_parameters, write_world_inputs,
)

BASE = [0.3, -0.45, 0.765, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
OTHER_BASE = [-0.3, -0.45, 0.765, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
JOINTS = [0.0] * 6


def _camera(height=0.8):
    """A head camera looking straight down at nothing but a table 0.2 m below the base."""
    intrinsic = np.array([[320.0, 0.0, 160.0], [0.0, 320.0, 120.0], [0.0, 0.0, 1.0]])
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.diag([1.0, -1.0, -1.0])
    cam2world[:3, 3] = [0.0, 0.0, 0.565 + height]
    return {"depth_m": np.full((240, 320), height, dtype=np.float32), "intrinsic_cv": intrinsic,
            "cam2world_cv": cam2world, "depth_range_m": (0.0, 1000.0)}


def _inputs(tmp_path):
    record = write_world_inputs(
        tmp_path / "x5.npz", cameras={"head": _camera(), "right_wrist": _camera(0.9)}, base_pose_world_xyzw=BASE,
        joint_positions=JOINTS, gripper_finger_qpos=[0.044, 0.044], gripper_directive=-1.0,
        other_arm={"joint_positions": JOINTS, "base_pose_world_xyzw": OTHER_BASE})
    return read_world_inputs(tmp_path / "x5.npz", record["sha256"])


def _hand(position):
    frame = np.eye(4)
    frame[:3, 3] = position
    return frame


def test_a_two_arm_snapshot_keeps_its_cameras_and_the_other_arm(tmp_path):
    inputs = _inputs(tmp_path)
    assert [camera.name for camera in inputs.cameras] == ["head", "right_wrist"]
    assert inputs.joint_positions.shape == (6,) and inputs.other_arm_joint_positions.tolist() == JOINTS
    # The arms stand 0.6 m apart along the world's x, which is this base's -y.
    moved = other_arm_spheres_base(inputs, np.array([[0.0, 0.0, 0.0, 0.05]]))
    assert moved[0, :3] == pytest.approx([0.0, 0.6, 0.0], abs=1e-9) and moved[0, 3] == 0.05


def test_the_other_arm_stands_in_the_world_and_no_goal_carve_clears_it(tmp_path):
    inputs = _inputs(tmp_path)
    parameters = PUBLIC_DEPTH_VOXEL_X5_V1
    other = np.array([[0.3, 0.3, 0.3, 0.05]])
    scene = Scene(inputs, robot_spheres_base=np.zeros((0, 4)), start_hand_base=_hand([0.3, -0.4, 0.4]),
                  parameters=parameters, other_arm_spheres=other)
    # A goal hand box right at that sphere would carve it.
    world = scene.world(_hand([0.3, 0.3, 0.2]))
    index = np.floor((other[0, :3] - parameters.grid_low_m) / parameters.voxel_size_m).astype(int)
    assert world.occupied[tuple(index)] and world.summary["other_arm_voxels"] > 100
    clear = Scene(inputs, robot_spheres_base=np.zeros((0, 4)), start_hand_base=_hand([0.3, -0.4, 0.4]),
                  parameters=parameters).world(_hand([0.3, 0.3, 0.2]))
    assert not clear.occupied[tuple(index)] and clear.sha256 != world.sha256


def test_every_robot_names_its_parameters_and_the_turn_to_the_hand_convention():
    assert world_parameters("none", "arx_x5") is None and world_parameters(WORLD_MODEL, "arx_x5") is ROBOT_WORLDS["arx_x5"][0]
    assert tool_from_hand("franka_panda") == pytest.approx(np.eye(4))
    turn = tool_from_hand("arx_x5")[:3, :3]
    assert turn @ [0.0, 0.0, 1.0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)  # the hand's approach is link6's +x
    assert turn @ [0.0, 1.0, 0.0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)  # the jaw line stays y
    assert np.linalg.det(turn) == pytest.approx(1.0)


def test_the_x5_grid_exports_whole_voxels():
    grid = curobo_v2_grid(np.zeros(PUBLIC_DEPTH_VOXEL_X5_V1.grid_shape, dtype=np.float32), PUBLIC_DEPTH_VOXEL_X5_V1)
    assert grid["features"].shape == (141, 181, 81)


def test_a_one_arm_snapshot_still_needs_its_two_cameras(tmp_path):
    with pytest.raises(DepthWorldError):
        write_world_inputs(tmp_path / "a.npz", cameras={"head": _camera()}, base_pose_world_xyzw=BASE,
                           joint_positions=[0.0] * 7, gripper_finger_qpos=[0.0, 0.0], gripper_directive=-1.0)
