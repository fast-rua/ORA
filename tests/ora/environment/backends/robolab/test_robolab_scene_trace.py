"""RoboLab's use of the Isaac scene reader: the shapes its tables are built of, the TCP as a body and an offset, and the
clock of one row per native action. What only a live stage shows (the mesh extents, the poses) is in the design note."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robodojo.isaac_scene import fan_corners, part_mesh
from ora.environment.backends.robolab.scene_trace import (
    CLOCK_UNIT, STRIDE, RobolabSceneTrace, cube_faces, cylinder_faces, tcp_link,
)
from ora.environment.controllers import robolab_droid as D
from ora.environment.scene_trace import Body, Mesh, merge

TRIANGLE = Mesh(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], np.float32), np.full((3, 4), 200, np.uint8),
                np.array([[0, 1, 2]]))


def _volume(points, counts, indices) -> float:
    """The divergence theorem over the triangles: the shape's volume, positive when its faces wind outward."""
    triangles = np.asarray(points, dtype=float)[np.asarray(indices)[fan_corners(counts)]]
    return float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def test_a_cube_is_closed_centred_and_the_size_usd_names() -> None:
    points, counts, indices = cube_faces(0.5)

    assert points.min() == -0.25 and points.max() == 0.25 and counts.sum() == len(indices) == 24
    assert _volume(points, counts, indices) == pytest.approx(0.125)


@pytest.mark.parametrize("axis, column", [("X", 0), ("Y", 1), ("Z", 2)])
def test_a_cylinder_runs_along_its_axis_and_is_closed(axis: str, column: int) -> None:
    points, counts, indices = cylinder_faces(0.1, 0.6, axis)

    extents = np.ptp(points, axis=0)
    assert extents[column] == pytest.approx(0.6) and np.delete(extents, column) == pytest.approx([0.2, 0.2])
    assert _volume(points, counts, indices) == pytest.approx(np.pi * 0.01 * 0.6, rel=0.01)


def test_a_shape_lands_in_the_body_s_frame_through_the_mesh_reader() -> None:
    to_body = np.eye(4)
    to_body[3, :3] = (1.0, 0.0, 0.0)  # USD's row vectors

    mesh = part_mesh(*cube_faces(0.2), to_body, rgb=(1.0, 0.0, 0.0))

    assert mesh.positions[:, 0].min() == pytest.approx(0.9) and len(mesh.faces) == 12
    assert (mesh.colors[:, 0] == 255).all() and (mesh.colors[:, 1] == 0).all()


def test_the_tcp_is_the_receipts_on_the_flange_body() -> None:
    bodies = [SimpleNamespace(key="robot/panda_link7"), SimpleNamespace(key="robot/Gripper/Robotiq_2F_85/base_link")]

    link = tcp_link(bodies)["right"]

    assert link["body"] == "robot/Gripper/Robotiq_2F_85/base_link"
    assert link["offset_xyz_xyzw"][:3] == list(D.TCP_OFFSET_FROM_FLANGE_M)
    # the bridge's TCP frame: x along the flange's y, y along its z, z along its x
    assert link["offset_xyz_xyzw"][3:] == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_a_row_is_one_native_action_and_the_poses_are_read_after_its_last_step(tmp_path: Path, monkeypatch) -> None:
    reads = {"count": 0}

    def pose() -> list[float]:
        reads["count"] += 1
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    trace = RobolabSceneTrace(tmp_path / "trace", run_id="r")
    poses = SimpleNamespace(refresh=lambda: None)
    body = Body("robot/Gripper/Robotiq_2F_85/base_link", "flange", "robot", pose, TRIANGLE)
    monkeypatch.setattr("ora.environment.backends.robolab.scene_trace.bodies_of", lambda _session: ([body], poses))

    trace.begin_round(None, 0)
    trace.begin_action({"action_sequence": 0, "action_id": "a0"})
    for _ in range(3):
        trace.after_tick()
    trace.end_action()
    manifest = trace.finish()

    assert manifest["shape"] == [4, 1, 7] and reads["count"] == 4
    assert manifest["clock"]["unit"] == CLOCK_UNIT and manifest["clock"]["stride"] == STRIDE == 8
    assert manifest["clock"]["hz"] == 120.0
    assert manifest["spans"] == [{"round": 0, "action_sequence": 0, "action_id": "a0", "first_step": 0, "last_step": 24}]
    assert manifest["tcp"]["right"]["body"] == body.key


def test_merging_a_prim_s_mesh_with_its_shapes_keeps_both() -> None:
    assert len(merge([TRIANGLE, part_mesh(*cube_faces(0.2), np.eye(4))]).faces) == 13
