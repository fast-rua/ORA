"""``ora_lib``'s frames layer (``ora/agent/code_mode/ora_lib/frames.py``) over a published observation.

Conversions round-trip across ``world``, ``tcp:<arm>``, ``camera:<name>`` and ``pixel:<name>``.
``pixels_to_world`` and ``project`` give what ``geometry_query`` gives on the same recorded observation, for
any number of points and with the same window median. Combining values of different frames raises a
``FrameError`` that names the conversion, and a bare list is refused because its frame would be a guess. The
helpers (``offset``, ``direction``) follow ``docs/robots/frames-and-units.md``.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ora.agent.code_mode.library import ObservationMount
from ora.agent.code_mode.ora_lib import (Direction, FrameError, ObservationUnavailable, Pixel, Point, Pose,
                                         direction, offset, pixels_to_world, project)
from ora.agent.code_mode.ora_lib import observation as library_observation


@pytest.fixture
def observation(published, tmp_path):
    mount = ObservationMount(published.control_root, tmp_path / "mount")
    mount.root.mkdir()
    mount.refresh()
    return library_observation.load(str(mount.root))


def same_rotation(a, b) -> None:
    assert abs(abs(float(np.dot(a, b))) - 1.0) < 1e-12


def test_points_poses_and_directions_round_trip_through_every_frame(observation) -> None:
    rng = np.random.default_rng(3)
    world = Point(rng.uniform([-0.4, -0.5, 0.75], [0.4, 0.3, 1.1], size=(32, 3)), frame="world")
    for path in (["camera:head", "world"], ["tcp:left", "camera:right_wrist", "tcp:right", "world"]):
        value = world
        for frame in path:
            value = value.to(frame, observation=observation)
            assert value.frame == frame
        np.testing.assert_allclose(value.xyz, world.xyz, atol=1e-12)
    # A frame's own origin is the measured pose (tcp) or the calibration (camera).
    np.testing.assert_allclose(Point([0, 0, 0], "tcp:left").to("world", observation=observation).xyz,
                               observation.pose("left").xyz, atol=1e-12)
    np.testing.assert_allclose(Point([0, 0, 0], "camera:head").to("world", observation=observation).xyz,
                               np.asarray(observation.camera("head").calibration["cam2world_cv"])[:3, 3], atol=1e-6)
    pose = Pose([0.1, -0.2, 0.9], Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_quat(), "world")
    back = pose.to("tcp:right", observation=observation).to("camera:head", observation=observation) \
        .to("world", observation=observation)
    np.testing.assert_allclose(back.xyz, pose.xyz, atol=1e-12)
    same_rotation(back.xyzw, pose.xyzw)
    assert observation.pose("right").to("tcp:right", observation=observation).xyz.tolist() == pytest.approx(
        [0, 0, 0], abs=1e-12)
    axis = Direction([0.3, -0.4, 0.5], "world")
    turned = axis.to("camera:left_wrist", observation=observation)
    assert turned.norm() == pytest.approx(axis.norm())
    np.testing.assert_allclose(turned.to("world", observation=observation).v, axis.v, atol=1e-12)
    # The image: a pixel lifted with its exact depth projects back onto itself; into another camera and back too.
    pixels = Pixel([[300, 190], [12, 470], [639, 0]], camera="head")
    lifted = pixels.to("world", window_px=0, observation=observation)
    np.testing.assert_allclose(project(lifted, "head", observation=observation).uv, pixels.uv, atol=1e-9)
    elsewhere = pixels.to("pixel:right_wrist", window_px=0, observation=observation)
    assert elsewhere.camera == "right_wrist" and elsewhere.frame == "pixel:right_wrist"


def test_pixels_to_world_and_project_agree_with_geometry_query(published, observation) -> None:
    from tests.support.agent.toolkit_run import on_surface, served

    published = served(on_surface(published, "robodojo_placement_toolkit_v5"))  # its geometry_query as a run serves it
    rng = np.random.default_rng(7)
    pixels = np.concatenate([rng.integers([0, 0], [640, 480], size=(44, 2)),
                             [[0, 0], [639, 479], [639, 0], [0, 479]]])
    for window in (0, 3, 15):
        lifted = pixels_to_world("head", pixels, window_px=window, observation=observation)
        assert lifted.frame == "world" and lifted.xyz.shape == (48, 3)  # no 16-point limit
        for start in range(0, len(pixels), 16):
            expected = published.invoke("geometry_query", {
                "camera": "head", "points": pixels[start:start + 16].tolist(), "window_px": window})["points"]
            for row, point, valid in zip(expected, lifted.xyz[start:start + 16], lifted.valid[start:start + 16]):
                assert valid == row["valid"]
                if valid:
                    np.testing.assert_allclose(point, row["world_xyz_m"], rtol=0, atol=1e-12)
    depths = Pixel(pixels[:16], camera="head").depth(observation=observation)
    expected = published.invoke("geometry_query", {"camera": "head", "points": pixels[:16].tolist()})["points"]
    np.testing.assert_allclose(depths, [row.get("depth_median_m", np.nan) for row in expected], rtol=0, atol=0)

    world = [[0, 0, 0.8], [0.1, -0.2, 0.9], [-0.3, 0.2, 0.77], [5, 0, 0.8], [0, 0, -2]]
    expected = published.invoke("geometry_query", {"camera": "head", "world_points_m": world})["world_points"]
    projected = project(Point(world, "world"), "head", observation=observation)
    inside = projected.in_frame(observation=observation)
    for row, uv, in_frame in zip(expected, projected.uv, inside):
        assert in_frame == row["in_frame"]
        if row["in_front"]:
            np.testing.assert_allclose(uv, row["pixel_uv"], rtol=0, atol=1e-9)
        else:
            assert np.isnan(uv).all()
    assert projected[projected.valid & inside].rounded().tolist() == [
        row["pixel_uv_rounded"] for row in expected if row["in_frame"]]


def test_combining_frames_raises_and_names_the_conversion(observation) -> None:
    point, step = Point([0.1, 0.0, 0.9], "world"), Direction([0, 0, 0.1], "camera:head")
    with pytest.raises(FrameError, match=r"Point in world with a Direction in camera:head.*"
                                         r"direction\.to\('world'\) or point\.to\('camera:head'\)"):
        point + step
    with pytest.raises(FrameError, match=r"other\.to\('world'\)"):
        point - Point([0, 0, 0], "tcp:left")
    with pytest.raises(FrameError, match=r"different images.*other\.to\('pixel:head'\)"):
        Pixel([1, 2], "head") - Pixel([1, 2], "left_wrist")
    with pytest.raises(TypeError, match=r"no frame.*Direction\(v, frame='world'\)"):
        point + [0, 0, 0.1]
    with pytest.raises(TypeError, match="no frame"):
        np.array([0, 0, 0.1]) + point
    with pytest.raises(FrameError, match="Pixel"):
        Point([1, 2, 3], "pixel:head")
    with pytest.raises(FrameError, match="unknown frame"):
        Point([1, 2, 3], "base")
    moved = point + step.to("world", observation=observation)
    assert moved.frame == "world" and (moved - point).norm() == pytest.approx(0.1)
    assert (point - Point([0.1, 0.0, 0.8], "world")).v.tolist() == pytest.approx([0, 0, 0.1])


def test_offset_and_direction(observation) -> None:
    pose = Pose([0.1, -0.2, 0.9], Rotation.from_euler("z", 90, degrees=True).as_quat(), "world")
    assert offset(pose, dz=0.1, frame="world").xyz.tolist() == pytest.approx([0.1, -0.2, 1.0])
    assert offset(pose, dx=0.05, frame="local").xyz.tolist() == pytest.approx([0.1, -0.15, 0.9])
    measured = observation.pose("left")
    along_tool = offset(measured, dx=0.05, frame="tcp:left", observation=observation)
    assert along_tool.xyz.tolist() == pytest.approx(offset(measured, dx=0.05, frame="local").xyz.tolist())
    assert along_tool.xyzw.tolist() == pytest.approx(measured.xyzw.tolist())
    np.testing.assert_allclose(
        offset(pose, dx=0.05, frame="tcp:left", observation=observation).xyz,
        pose.xyz + Rotation.from_quat(measured.xyzw).apply([0.05, 0, 0]), atol=1e-12)
    with pytest.raises(FrameError, match="no axes of its own"):
        offset(pose.position, dz=0.1, frame="local")

    # Image directions: the head camera looks down at the table from the robot's side.
    for name, axis in (("left", [-1, 0, 0]), ("right", [1, 0, 0]), ("up", [0, -1, 0]), ("down", [0, 1, 0])):
        found = direction(name, view="head", observation=observation)
        assert found.frame == "world" and found.norm() == pytest.approx(1.0)
        np.testing.assert_allclose(found.to("camera:head", observation=observation).v, axis, atol=1e-6)
    away = direction("up", view="head", horizontal=True, observation=observation)
    assert away.v.tolist() == pytest.approx([0, 1, 0], abs=1e-6)  # up the image = +Y, toward the far edge
    centre = Point([0.0, 0.0, 0.8], "world")
    for name, du, dv in (("right", 1, 0), ("left", -1, 0), ("down", 0, 1), ("up", 0, -1)):
        moved = project(centre + direction(name, view="head", observation=observation) * 0.02, "head",
                        observation=observation).uv - project(centre, "head", observation=observation).uv
        assert np.sign(np.round(moved, 6)).tolist() == [du, dv]


def test_units_are_si_and_others_say_so(observation) -> None:
    pose = Pose([0, 0, 1], [0, 0, 0, 1], "world")
    turned = pose.rotated("z", angle_deg=90)
    assert turned.rotated("z", angle_rad=-np.pi / 2).xyzw.tolist() == pytest.approx([0, 0, 0, 1], abs=1e-12)
    assert pose.axis("x").angle_deg(turned.axis("x")) == pytest.approx(90)
    assert pose.axis("x").angle_rad(turned.axis("x")) == pytest.approx(np.pi / 2)
    with pytest.raises(ValueError, match="exactly one of angle_deg and angle_rad"):
        pose.rotated("z", angle_deg=90, angle_rad=1.0)
    with pytest.raises(ValueError, match="unit quaternion"):
        Pose([0, 0, 1], [0, 0, 0, 2], "world")
    assert Pose([0, 0, 1], [0, 0, 0, 1.01], "world").xyzw.tolist() == [0, 0, 0, 1]  # renormalized, as inputs are


def test_without_a_mounted_observation_only_world_converts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(library_observation, "MOUNT", str(tmp_path / "absent"))
    point = Point([0.1, 0.2, 0.3], "world")
    assert point.to("world").xyz.tolist() == [0.1, 0.2, 0.3]
    with pytest.raises(ObservationUnavailable, match="no published observation is mounted"):
        point.to("camera:head")
    assert point.to_json() == point.tolist() == {"frame": "world", "xyz": [0.1, 0.2, 0.3]}


def test_a_direction_has_xyz_as_a_point_does() -> None:
    one = Direction([0.1, 0.2, 0.3], "world")
    assert one.xyz is one.v and one.xyz.tolist() == [0.1, 0.2, 0.3]
    assert Direction([[1, 0, 0], [0, 1, 0]], "world").xyz.shape == (2, 3)
    assert one.to_json() == {"frame": "world", "v": [0.1, 0.2, 0.3]}, "written as it was"


def test_numpy_reads_a_point_s_and_a_direction_s_values() -> None:
    point, turn = Point([0.123456, 2.0, 3.0], "world"), Direction([0.123456, 0.0, 1.0], "tcp:right")
    for got, expected in ((np.round(turn, 2), [0.12, 0.0, 1.0]), (np.round(point, 2), [0.12, 2.0, 3.0]),
                          (np.round(turn), [0.0, 0.0, 1.0]), (np.array(point), point.xyz), (np.asarray(turn), turn.v),
                          (np.asarray(Pose([0.1, 0.2, 0.3], [0, 0, 0, 1], "world").position), [0.1, 0.2, 0.3])):
        assert isinstance(got, np.ndarray) and got.dtype == np.float64
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    batch = Point([[1, 2, 3], [4, 5, 6]], "world")
    assert np.asarray(batch).shape == (2, 3) and np.array(batch, dtype=np.float32).dtype == np.float32
    assert np.linalg.norm(Direction([3, 4, 0], "world")) == pytest.approx(5.0)
    values = np.asarray(point)
    values[0] = 9.0
    assert point.xyz[0] == 0.123456, "a new array: the value keeps its numbers"


@pytest.mark.parametrize("ufunc", [np.add, np.subtract, np.multiply, np.rint, np.negative])
def test_a_ufunc_over_a_value_keeps_refusing(ufunc) -> None:
    world, tcp = Point([1, 2, 3], "world"), Point([1, 2, 3], "tcp:right")
    operands = (world,) if ufunc.nin == 1 else (world, tcp)
    with pytest.raises(TypeError, match="does not support ufuncs"):
        ufunc(*operands)


def test_combining_frames_raises_with_numpy_values_about() -> None:
    world, tcp = Point([1, 2, 3], "world"), Point([1, 2, 3], "tcp:right")
    with pytest.raises(FrameError, match=r"tcp:right.*world"):
        world - tcp
    with pytest.raises(TypeError, match="a bare ndarray has no frame"):
        np.array([0, 0, 1]) + world
    # A constructor reads numbers, so another frame's value would quietly change frame.
    for build in (lambda: Point(tcp, "world"), lambda: Direction(tcp, "world"), lambda: Point([world, tcp], "world"),
                  lambda: Pose(tcp, [0, 0, 0, 1], "world")):
        with pytest.raises(FrameError, match=r"is in tcp:right, not world: convert it first, e\.g\. value\.to\('world'\)"):
            build()
    assert Point(world, "world").xyz.tolist() == [1, 2, 3] and Pose(world, [0, 0, 0, 1], "world").frame == "world"


@pytest.fixture
def values(observation) -> dict:
    return {"Point": Point([1, 2, 3], "world"), "Direction": Direction([0, 0, 1], "world"),
            "Pose": Pose([0, 0, 1], [0, 0, 0, 1], "world"), "Pixel": Pixel([1, 2], "head"),
            "Observation": observation, "Camera": observation.camera("head")}


@pytest.mark.parametrize("kind,guess,listed", [
    ("Pose", "quaternion", ("xyzw", "position", "rotation", "to")),
    ("Pixel", "pixel_uv", ("uv", "u", "v", "camera", "depth")),
    ("Point", "v", ("xyz", "x", "frame", "valid", "mean")),
    ("Direction", "unit_vector", ("v", "xyz", "unit", "norm", "dot")),
    ("Observation", "mobile_base", ("robot_facts", "end_effectors", "cameras", "pose", "arms")),
    ("Camera", "intrinsics", ("K", "extrinsic", "depth_m", "rgb", "width", "name")),
])
def test_an_attribute_error_names_what_is_there(values, kind, guess, listed) -> None:
    value = values[kind]
    with pytest.raises(AttributeError) as missing:
        getattr(value, guess)
    message = str(missing.value)
    assert message.startswith(f"'{kind}' object has no attribute '{guess}'; it has: ")
    names = message.split("it has: ")[1].split(", ")
    assert set(listed) <= set(names) and not [name for name in names if name.startswith("_")]
    assert names == sorted(names) and (missing.value.name, missing.value.obj) == (guess, value)
    assert hasattr(value, guess) is False and getattr(value, guess, "absent") == "absent"


def test_the_values_stay_slotted_and_survive_pickling_and_copies() -> None:
    for value in (Point([1, 2, 3], "world"), Direction([0, 0, 1], "world"), Pose([0, 0, 1], [0, 0, 0, 1], "world"),
                  Pixel([1, 2], "head")):
        assert not hasattr(value, "__dict__")
        for copied in (pickle.loads(pickle.dumps(value)), copy.copy(value), copy.deepcopy(value)):
            assert type(copied) is type(value) and copied.to_json() == value.to_json()
    with pytest.raises(AttributeError) as private:
        Point([1, 2, 3], "world")._nothing
    assert str(private.value) == "'Point' object has no attribute '_nothing'", "no list for a private name"
    camera = object.__new__(library_observation.Camera)
    with pytest.raises(AttributeError, match="'Camera' object has no attribute '_files'"):
        camera._files  # a half-built object: nothing here reads the attributes it lacks
