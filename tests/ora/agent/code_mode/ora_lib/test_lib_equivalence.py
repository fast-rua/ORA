"""``ora_lib.geometry`` (``ora/agent/code_mode/ora_lib/geometry.py``) computes what ``ora/agent/toolkit/lib`` computes.

The sandbox has no ORA package, and the lib's ``values``, ``geometry``, ``pose`` and ``numeric`` import its
data interface, so ``ora_lib`` carries small equivalents. Each one is held here to the lib function it
replaces, on the same inputs: rigid transforms of points and poses, a camera's inverse extrinsic, mask
back-projection, principal geometry and a pose offset in its own frame.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ora.agent.code_mode.ora_lib import Point, Pose, geometry, offset
from ora.agent.toolkit.lib import geometry as lib_geometry
from ora.agent.toolkit.lib.observation import CameraRGBD
from ora.agent.toolkit.lib.values import Array, JsonValue, PointCloudM, PoseM, TransformM

RNG = np.random.default_rng(11)


def rigid() -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(RNG.normal(size=3)).as_matrix()
    matrix[:3, 3] = RNG.normal(size=3)
    return matrix


def test_rigid_transforms_of_points_and_poses_equal_the_lib() -> None:
    matrix, points = rigid(), RNG.normal(size=(50, 3))
    expected = lib_geometry.transform_points(PointCloudM("a", Array.of(points)), TransformM("a", "b", Array.of(matrix)))
    np.testing.assert_array_equal(geometry.transform_points(points, matrix), expected.xyz_m.numpy())
    xyzw = Rotation.from_rotvec(RNG.normal(size=3)).as_quat()
    lib_pose = lib_geometry.transform_pose(PoseM("a", tuple(points[0]), tuple(xyzw)), TransformM("a", "b", Array.of(matrix)))
    xyz, q = geometry.transform_pose(points[0], xyzw, matrix)
    np.testing.assert_allclose(xyz, lib_pose.position_m, rtol=0, atol=1e-15)
    np.testing.assert_allclose(q, lib_pose.orientation_xyzw, rtol=0, atol=1e-15)
    assert q[3] >= 0
    for bad in (np.diag([1.0, 1.0, -1.0, 1.0]), np.diag([2.0, 1.0, 1.0, 1.0])):
        with pytest.raises(ValueError):
            TransformM("a", "b", Array.of(bad))
        with pytest.raises(ValueError):
            geometry.rigid(bad)


def test_a_camera_inverse_and_mask_back_projection_equal_the_lib() -> None:
    height, width = 24, 32
    depth = RNG.uniform(0.3, 2.0, size=(height, width)).astype(np.float32)
    depth[3, 4], depth[5, 6], depth[7, 8] = np.nan, 0.0, 3.0
    intrinsic = np.array([[40.0, 0.0, 15.5], [0.0, 41.0, 11.5], [0.0, 0.0, 1.0]])
    extrinsic = rigid()
    camera = CameraRGBD("head", Array.of(np.zeros((height, width, 3), np.uint8)), Array.of(depth), Array.of(intrinsic),
                        TransformM("camera/head", "world", Array.of(np.linalg.inv(extrinsic))), JsonValue.of({}))
    np.testing.assert_array_equal(geometry.invert(extrinsic), camera.camera_to_world.matrix.numpy())
    mask = RNG.random((height, width)) > 0.4
    mask[3, 4] = mask[5, 6] = mask[7, 8] = True
    expected = lib_geometry.mask_to_cloud(camera, Array.of(mask)).xyz_m.numpy()
    np.testing.assert_array_equal(geometry.mask_to_cloud(depth, intrinsic, mask), expected)
    pixels = np.argwhere(mask & np.isfinite(depth) & (depth > 1e-6) & (depth <= 2.5))[:, ::-1]
    np.testing.assert_allclose(geometry.back_project(pixels, depth[pixels[:, 1], pixels[:, 0]], intrinsic), expected,
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(geometry.project(expected, intrinsic), pixels, rtol=0, atol=1e-9)


def test_principal_geometry_equals_the_lib() -> None:
    cloud = RNG.normal(size=(400, 3)) * [0.05, 0.02, 0.01] + [0.1, -0.2, 0.8]
    expected = lib_geometry.principal_geometry(PointCloudM("world", Array.of(cloud)))
    found = geometry.principal_geometry(cloud)
    for name in ("center_m", "lower_m", "upper_m", "axes", "extents_m", "variance_m2"):
        np.testing.assert_array_equal(found[name], getattr(expected, name).numpy(), err_msg=name)
    assert found["ambiguous_axes"] == expected.ambiguous_axes
    with pytest.raises(ValueError):
        geometry.principal_geometry(np.zeros((0, 3)))


def test_an_offset_in_the_pose_frame_equals_the_lib() -> None:
    xyzw = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_quat()
    expected = lib_geometry.offset_pose(PoseM("world", (0.1, 0.2, 0.9), tuple(xyzw)), delta_m=(0.0, 0.0, 0.1),
                                        frame="world")
    found = offset(Pose([0.1, 0.2, 0.9], xyzw, "world"), dz=0.1, frame="world")
    assert tuple(found.xyz) == expected.position_m and found.frame == expected.frame
    np.testing.assert_allclose(found.xyzw, expected.orientation_xyzw, rtol=0, atol=1e-15)
    assert offset(Point([0.1, 0.2, 0.9], "world"), dz=0.1, frame="world").xyz.tolist() == list(expected.position_m)
