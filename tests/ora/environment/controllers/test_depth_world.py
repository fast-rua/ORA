"""The public-depth collision world: determinism, robot filtering, carves and the held object."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ora.environment.controllers.depth_world import (
    CUROBO_UNOBSERVED_BORDER_VOXELS, NEAR_CELL_M, PUBLIC_DEPTH_VOXEL_V1, DepthWorldError, Scene, _within_spheres,
    build_world, read_world_inputs, signed_distance, write_world_inputs,
)
from tests.support.libero.world_inputs import (
    DOWN, TABLE_Z, blind as _blind, frame as _frame, overhead, read, write,
)

#: The synthetic robot base in the world; the scene is described in the base frame.
BASE_WORLD = np.array([-0.66, 0.0, 1.0])
#: A small object on the table (base frame): x, y footprint and top height.
OBJECT = ((0.35, 0.55), (-0.08, 0.08), 0.0)
NO_SPHERES = np.zeros((0, 4))


def _overhead_depth(camera_base, footprints):
    return overhead(camera_base, footprints, base_world=BASE_WORLD)


def _inputs(tmp_path, name="inputs.npz", *, footprints=(OBJECT,), wrist=None, gap=0.08, directive=-1.0):
    return read(write(tmp_path / name, agentview=_overhead_depth((0.45, 0.0, 0.8), footprints), wrist=wrist,
                      base_world=BASE_WORLD, gap=gap, directive=directive))


def _value(world, point):
    index = np.floor((np.asarray(point) - PUBLIC_DEPTH_VOXEL_V1.grid_low_m)
                     / PUBLIC_DEPTH_VOXEL_V1.voxel_size_m).astype(int)
    return float(world.esdf[tuple(index)]), bool(world.occupied[tuple(index)])


AWAY = _frame(DOWN, (0.2, 0.45, 0.5))
OBJECT_TOP = (0.45, 0.0, -0.005)


def test_world_is_deterministic_and_hash_binds_content_and_parameters(tmp_path):
    first = build_world(_inputs(tmp_path, "a.npz"), robot_spheres_base=NO_SPHERES,
                        start_hand_base=AWAY, goal_hand_base=AWAY)
    second = build_world(_inputs(tmp_path, "b.npz"), robot_spheres_base=NO_SPHERES,
                         start_hand_base=AWAY, goal_hand_base=AWAY)
    assert first.sha256 == second.sha256 == first.summary["sha256"]
    np.testing.assert_array_equal(first.esdf, second.esdf)
    assert first.esdf.shape == PUBLIC_DEPTH_VOXEL_V1.grid_shape == (127, 127, 107)
    assert first.esdf.dtype == np.float32
    assert _value(first, OBJECT_TOP)[1] is True
    assert _value(first, (0.45, 0.0, 0.10))[1] is False
    assert first.summary["points"]["world"] > 0 and first.summary["occupied_voxels"] > 0
    coarse = build_world(_inputs(tmp_path, "c.npz"), robot_spheres_base=NO_SPHERES,
                         start_hand_base=AWAY, goal_hand_base=AWAY,
                         parameters=replace(PUBLIC_DEPTH_VOXEL_V1, surface_thickness_m=0.03))
    assert coarse.sha256 != first.sha256
    moved = build_world(_inputs(tmp_path, "d.npz", footprints=(((0.40, 0.50), (0.10, 0.20), 0.0),)),
                        robot_spheres_base=NO_SPHERES, start_hand_base=AWAY, goal_hand_base=AWAY)
    assert moved.sha256 != first.sha256


def test_esdf_is_exact_within_the_clamp_and_signed_at_half_a_voxel():
    from scipy.ndimage import distance_transform_edt

    rng = np.random.default_rng(7)
    occupied = np.zeros((40, 30, 20), dtype=bool)
    occupied[5:12, 4:9, 3:15] = True
    occupied[tuple(rng.integers(0, (40, 30, 20), size=(30, 3)).T)] = True
    esdf = signed_distance(occupied, PUBLIC_DEPTH_VOXEL_V1)
    size, clamp = PUBLIC_DEPTH_VOXEL_V1.voxel_size_m, PUBLIC_DEPTH_VOXEL_V1.esdf_clamp_m
    outside = -np.minimum(distance_transform_edt(~occupied) - 0.5, clamp / size) * size
    inside = np.minimum(distance_transform_edt(occupied) - 0.5, clamp / size) * size
    np.testing.assert_allclose(esdf, np.where(occupied, inside, outside), atol=1e-6)
    assert esdf.min() >= -clamp - 1e-7
    block = np.zeros((20, 20, 20), dtype=bool)
    block[5:10, 5:10, 5:10] = True
    boundary = signed_distance(block, PUBLIC_DEPTH_VOXEL_V1)
    assert boundary[5, 7, 7] == pytest.approx(0.5 * size) and boundary[4, 7, 7] == pytest.approx(-0.5 * size)
    assert boundary[7, 7, 7] == pytest.approx(2.5 * size) and boundary[0, 7, 7] == pytest.approx(-4.5 * size)


def _within_spheres_all_pairs(points, spheres, margin):
    """The sphere test before the per-sphere search (kept verbatim as the oracle)."""
    inside = np.zeros(len(points), dtype=bool)
    if not len(points) or not len(spheres):
        return inside
    low = points.min(axis=0)
    shape = np.floor((points.max(axis=0) - low) / NEAR_CELL_M).astype(np.int64) + 1
    near = np.zeros(tuple(shape), dtype=bool)
    reach = spheres[:, 3:4] + margin
    first = np.clip(np.floor((spheres[:, :3] - reach - low) / NEAR_CELL_M).astype(np.int64), 0, shape)
    last = np.clip(np.floor((spheres[:, :3] + reach - low) / NEAR_CELL_M).astype(np.int64) + 1, 0, shape)
    for (a, b, c), (d, e, f) in zip(first.tolist(), last.tolist()):
        near[a:d, b:e, c:f] = True
    index = np.nonzero(near[tuple(np.floor((points - low) / NEAR_CELL_M).astype(np.int64).T)])[0]
    for start in range(0, len(index), 16384):
        chosen = index[start:start + 16384]
        squared = ((points[chosen, None, :] - spheres[None, :, :3]) ** 2).sum(axis=2)
        inside[chosen] = (squared <= (spheres[None, :, 3] + margin) ** 2).any(axis=1)
    return inside


def _on_and_near_surfaces(rng, spheres, margin, count):
    """Points exactly at ``radius + margin`` from a centre, and a few ulps either side of it."""
    rows = []
    for x, y, z, radius in spheres:
        direction = rng.normal(size=(count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        surface = np.array([x, y, z]) + direction * (radius + margin)
        rows += [surface, np.nextafter(surface, np.inf), np.nextafter(surface, -np.inf),
                 np.array([x, y, z]) + direction * (radius + margin) * (1 + 1e-15),
                 np.array([x, y, z]) + direction * (radius + margin) * (1 - 1e-15)]
        # On the axes, where the search box is tightest.
        for axis in range(3):
            for sign in (-1.0, 1.0):
                point = np.array([x, y, z], dtype=np.float64)
                point[axis] += sign * (radius + margin)
                rows.append(np.array([point, np.nextafter(point, np.inf), np.nextafter(point, -np.inf)]))
    return np.concatenate(rows)


@pytest.mark.parametrize("margin", [0.0, 0.02, 0.03])
def test_the_per_sphere_search_takes_exactly_the_points_the_all_pairs_test_takes(margin):
    rng = np.random.default_rng(11)
    spheres = np.column_stack([rng.uniform(-0.3, 0.6, size=(60, 3)), rng.uniform(0.005, 0.08, size=60)])
    spheres[1, :3] = spheres[0, :3] + 0.01  # overlapping spheres
    spheres[2, :3] = [0.6, 0.6, 0.6]        # past every point: no near cell
    cloud = rng.uniform(-0.5, 0.8, size=(200_000, 3))
    points = np.concatenate([cloud, _on_and_near_surfaces(rng, spheres, margin, 50), spheres[:, :3]])
    rng.shuffle(points)
    expected = _within_spheres_all_pairs(points, spheres, margin)
    assert expected.any() and not expected.all()
    np.testing.assert_array_equal(_within_spheres(points, spheres, margin), expected)
    # Voxel centres, as the carves test them.
    grid = (np.argwhere(np.ones((40, 40, 40), dtype=bool)) + 0.5) * 0.01
    centred = spheres[:5] + [0.2, 0.2, 0.2, 0.0]
    np.testing.assert_array_equal(_within_spheres(grid, centred, margin),
                                  _within_spheres_all_pairs(grid, centred, margin))


def test_the_sphere_search_on_empty_and_degenerate_inputs():
    spheres = np.array([[0.1, 0.0, 0.2, 0.05], [0.1, 0.0, 0.2, 0.05]])
    for points, balls in ((np.zeros((0, 3)), spheres), (np.ones((5, 3)), np.zeros((0, 4))),
                          (np.zeros((0, 3)), np.zeros((0, 4)))):
        found = _within_spheres(points, balls, 0.03)
        assert found.dtype == bool and found.shape == (len(points),) and not found.any()
    single = np.array([[0.1, 0.0, 0.2]])
    for points in (single, np.repeat(single, 3, axis=0), np.array([[0.1, 0.0, 0.28], [0.1, 0.0, 0.2801]])):
        np.testing.assert_array_equal(_within_spheres(points, spheres, 0.03),
                                      _within_spheres_all_pairs(points, spheres, 0.03))
    assert _within_spheres(single, spheres, 0.03).all()


def test_robot_points_are_filtered_by_spheres_and_hand_box(tmp_path):
    inputs = _inputs(tmp_path)
    covering = np.array([[0.45, 0.0, -0.03, 0.06]])
    world = build_world(inputs, robot_spheres_base=covering, start_hand_base=AWAY, goal_hand_base=AWAY)
    assert world.summary["points"]["robot"] > 0
    assert _value(world, OBJECT_TOP)[1] is False
    # A detached (non-positive radius) CuRobo slot filters nothing.
    detached = build_world(inputs, robot_spheres_base=[[0.45, 0.0, -0.03, -100.0]],
                           start_hand_base=AWAY, goal_hand_base=AWAY)
    assert _value(detached, OBJECT_TOP)[1] is True
    # The measured hand covers the object top: its points are the robot's, not the world's.
    over = _frame(DOWN, (0.45, 0.0, 0.10))
    held_by_box = build_world(inputs, robot_spheres_base=NO_SPHERES, start_hand_base=over,
                              goal_hand_base=AWAY)
    assert held_by_box.summary["points"]["held"] == 0
    assert _value(held_by_box, OBJECT_TOP)[1] is False


def test_the_arm_s_own_depth_past_its_spheres_does_not_stay_in_the_world(tmp_path):
    # A link surface the spheres under-cover: seen 2.5 cm off a sphere, past the start carve's
    # 2 cm. Left in the world it blocks the arm's next path. 3.5 cm off is the scene.
    sphere = [[0.30, 0.30, 0.20, 0.05]]
    near = ((0.3700, 0.3750), (0.29, 0.31), 0.20)
    far = ((0.3850, 0.3900), (0.29, 0.31), 0.20)
    scene = Scene(_inputs(tmp_path, "arm.npz", footprints=(near,)), robot_spheres_base=sphere, start_hand_base=AWAY)
    assert _value(scene.world(AWAY), (0.372, 0.30, 0.195))[1] is False
    other = Scene(_inputs(tmp_path, "scene.npz", footprints=(far,)), robot_spheres_base=sphere, start_hand_base=AWAY)
    assert _value(other.world(AWAY), (0.387, 0.30, 0.195))[1] is True


def test_goal_carve_clears_hand_and_standoff_and_start_carve_clears_the_start(tmp_path):
    inputs = _inputs(tmp_path)
    scene = Scene(inputs, robot_spheres_base=NO_SPHERES, start_hand_base=AWAY)
    away = scene.world(AWAY)
    assert away.summary["carved_voxels"] == {"start": 0, "goal": 0}
    assert _value(away, OBJECT_TOP)[1] is True
    # A grasp whose fingers reach 2 cm into the object top.
    grasp = _frame(DOWN, (0.45, 0.0, 0.085))
    carved = scene.world(grasp)
    assert carved.summary["carved_voxels"]["goal"] > 0
    esdf, occupied = _value(carved, OBJECT_TOP)
    assert occupied is False and esdf < 0
    assert carved.sha256 != away.sha256
    # Outside the oriented box the object stays, and the scene itself is unchanged.
    assert _value(carved, (0.37, 0.0, -0.005))[1] is True
    assert scene.world(AWAY).sha256 == away.sha256
    # A joint-space goal also clears its robot spheres.
    spheres = scene.world(AWAY, goal_spheres_base=[[0.42, 0.0, -0.01, 0.01]])
    assert spheres.summary["carved_voxels"]["goal"] > 0
    assert _value(spheres, (0.42, 0.0, -0.005))[1] is False
    # A start sphere below the seen surface filters no point but clears the thickened voxels.
    started = Scene(inputs, robot_spheres_base=[[0.45, 0.0, -0.06, 0.02]], start_hand_base=AWAY)
    assert started.point_counts["robot"] == 0 and started.carved_start > 0
    assert started.world(AWAY).summary["carved_voxels"]["start"] == started.carved_start


def test_closed_gripper_turns_points_between_the_pads_into_held_spheres(tmp_path):
    # A 3 cm wide object top held 10 cm along the approach axis, under a 4 cm gap.
    held_object = ((0.44, 0.46), (-0.015, 0.015), 0.0)
    hand = _frame(DOWN, (0.45, 0.0, 0.10))
    held = build_world(_inputs(tmp_path, "held.npz", footprints=(held_object,), gap=0.04, directive=1.0),
                       robot_spheres_base=NO_SPHERES, start_hand_base=hand, goal_hand_base=AWAY)
    assert held.summary["holding"] is True
    assert held.summary["points"]["held"] > 0
    assert 0 < held.summary["held_spheres"] == len(held.held_spheres_hand) <= 32
    centres = held.held_spheres_hand[:, :3]
    assert (np.abs(centres[:, 1]) < 0.02).all() and (centres[:, 2] > 0.09).all()
    assert _value(held, (0.45, 0.0, -0.005))[1] is False
    for name, gap, directive in (("open.npz", 0.04, -1.0), ("empty.npz", 0.002, 1.0)):
        free = build_world(_inputs(tmp_path, name, footprints=(held_object,), gap=gap, directive=directive),
                           robot_spheres_base=NO_SPHERES, start_hand_base=hand, goal_hand_base=AWAY)
        assert free.summary["holding"] is False and free.summary["held_spheres"] == 0
        assert len(free.held_spheres_hand) == 0
    assert free.sha256 != held.sha256


def test_world_inputs_are_hash_bound_and_closed(tmp_path):
    record = write_world_inputs(
        tmp_path / "inputs.npz", cameras={"agentview": _blind(), "wrist": _blind()},
        base_pose_world_xyzw=[0, 0, 0, 0, 0, 0, 1], joint_positions=[0.0] * 7,
        gripper_finger_qpos=[0.04, -0.04], gripper_directive=-1.0,
    )
    inputs = read_world_inputs(Path(record["path"]), record["sha256"])
    assert inputs.finger_gap_m == pytest.approx(0.08)
    with pytest.raises(DepthWorldError, match="sha256"):
        read_world_inputs(Path(record["path"]), "0" * 64)
    with pytest.raises(DepthWorldError, match="overwrite"):
        write_world_inputs(Path(record["path"]), cameras={"agentview": _blind(), "wrist": _blind()},
                           base_pose_world_xyzw=[0, 0, 0, 0, 0, 0, 1], joint_positions=[0.0] * 7,
                           gripper_finger_qpos=[0.04, -0.04], gripper_directive=-1.0)
    with pytest.raises(DepthWorldError, match="agentview and wrist"):
        write_world_inputs(tmp_path / "one.npz", cameras={"agentview": _blind()},
                           base_pose_world_xyzw=[0, 0, 0, 0, 0, 0, 1], joint_positions=[0.0] * 7,
                           gripper_finger_qpos=[0.04, -0.04], gripper_directive=-1.0)
    drifted = dict(np.load(record["path"]))
    drifted["extra"] = np.zeros(1)
    np.savez(tmp_path / "drifted.npz", **drifted)
    import hashlib
    digest = hashlib.sha256((tmp_path / "drifted.npz").read_bytes()).hexdigest()
    with pytest.raises(DepthWorldError, match="fields"):
        read_world_inputs(tmp_path / "drifted.npz", digest)
    # Both cameras contribute: the same table seen only by the wrist builds the same object.
    wrist_only = write_world_inputs(
        tmp_path / "wrist.npz",
        cameras={"agentview": _blind(), "wrist": _overhead_depth((0.45, 0.0, 0.8), (OBJECT,))},
        base_pose_world_xyzw=[*BASE_WORLD, 0, 0, 0, 1], joint_positions=[0.0] * 7,
        gripper_finger_qpos=[0.04, -0.04], gripper_directive=-1.0,
    )
    world = build_world(read_world_inputs(Path(wrist_only["path"]), wrist_only["sha256"]),
                        robot_spheres_base=NO_SPHERES, start_hand_base=AWAY, goal_hand_base=AWAY)
    assert _value(world, OBJECT_TOP)[1] is True


def test_grid_keeps_the_intended_volume_inside_the_kernels_unobserved_border():
    parameters = PUBLIC_DEPTH_VOXEL_V1
    border = CUROBO_UNOBSERVED_BORDER_VOXELS * parameters.voxel_size_m
    np.testing.assert_allclose(parameters.effective_low_m, [-0.20, -0.60, -0.25], atol=1e-9)
    np.testing.assert_allclose(parameters.effective_high_m, [1.01, 0.61, 0.76], atol=1e-9)
    np.testing.assert_allclose(parameters.effective_low_m - parameters.grid_low_m, [border] * 3, atol=1e-9)
    # The kernel's grid spans round(dims / voxel) + 1 voxels around the same centre.
    high = parameters.grid_low_m + np.asarray(parameters.grid_shape) * parameters.voxel_size_m
    np.testing.assert_allclose(high - parameters.effective_high_m, [border] * 3, atol=1e-9)
    np.testing.assert_allclose(0.5 * (parameters.effective_low_m + parameters.effective_high_m - 0.01),
                               parameters.grid_center_base_m, atol=1e-9)


def _hand_with_fingertips_at(x, y, tip_z):
    """A downward hand whose fingertips are at ``tip_z`` (base frame)."""
    return _frame(DOWN, (x, y, tip_z + PUBLIC_DEPTH_VOXEL_V1.fingertip_m))


def test_support_is_the_table_under_the_hand_not_the_object_top(tmp_path):
    scene = Scene(_inputs(tmp_path), robot_spheres_base=NO_SPHERES, start_hand_base=AWAY)
    # Over the 20 x 16 cm object, the dominant surface is still the table it stands on.
    support = scene.support(_hand_with_fingertips_at(0.45, 0.0, 0.05))
    assert support is not None and support["z_m"] == pytest.approx(TABLE_Z, abs=1e-6)
    assert support["columns"] >= PUBLIC_DEPTH_VOXEL_V1.support_min_columns
    # Every qualifying level, lowest first: the table, then the object top.
    levels = scene.supports(_hand_with_fingertips_at(0.45, 0.0, 0.05))
    assert levels[0]["z_m"] == pytest.approx(TABLE_Z, abs=1e-6) and levels[-1]["z_m"] > -0.01
    below = scene.supports(_hand_with_fingertips_at(0.45, 0.0, 0.05), below_index=levels[-1]["level_index"])
    assert below and all(level["z_m"] < -0.1 for level in below)


def test_goal_with_fingertips_inside_the_table_is_refused_and_the_table_is_never_carved(tmp_path):
    scene = Scene(_inputs(tmp_path), robot_spheres_base=NO_SPHERES, start_hand_base=AWAY)
    inside = scene.world(_hand_with_fingertips_at(0.75, 0.3, TABLE_Z - 0.03))
    refusal = inside.summary["goal_refusal"]
    assert refusal["reason"] == "fingertips_below_support"
    assert refusal["depth_m"] == pytest.approx(0.03, abs=1e-6)
    assert refusal["support_z_m"] == pytest.approx(TABLE_Z, abs=1e-6)
    assert "below the supporting surface" in refusal["message"]
    assert refusal["tolerance_m"] == PUBLIC_DEPTH_VOXEL_V1.support_tolerance_m == 0.01
    assert "more than the 1 cm allowed" in refusal["message"]
    # The carve box left the table under the goal in place.
    assert _value(inside, (0.75, 0.3, TABLE_Z - 0.005))[1] is True
    # Within the 1 cm tolerance the goal goes to the solvers; the carve still spares the table.
    for depth in (0.005, 0.0095):
        shallow = scene.world(_hand_with_fingertips_at(0.75, 0.3, TABLE_Z - depth))
        assert shallow.summary["goal_refusal"] is None
        assert _value(shallow, (0.75, 0.3, TABLE_Z - 0.005))[1] is True
    deeper = scene.world(_hand_with_fingertips_at(0.75, 0.3, TABLE_Z - 0.0105))
    assert deeper.summary["goal_refusal"]["depth_m"] == pytest.approx(0.0105, abs=1e-6)
    # The tolerance is a parameter of the set and enters the world hash.
    strict = Scene(_inputs(tmp_path, "strict.npz"), robot_spheres_base=NO_SPHERES, start_hand_base=AWAY,
                   parameters=replace(PUBLIC_DEPTH_VOXEL_V1, support_tolerance_m=0.0))
    tight = strict.world(_hand_with_fingertips_at(0.75, 0.3, TABLE_Z - 0.005))
    assert tight.summary["goal_refusal"]["depth_m"] == pytest.approx(0.005, abs=1e-6)
    assert tight.sha256 != scene.world(_hand_with_fingertips_at(0.75, 0.3, TABLE_Z - 0.005)).sha256
    # One centimetre above the table: accepted, and the table under the fingers stays.
    above = scene.world(_hand_with_fingertips_at(0.75, 0.3, TABLE_Z + 0.01))
    assert above.summary["goal_refusal"] is None
    assert above.summary["support"]["goal"]["z_m"] == pytest.approx(TABLE_Z, abs=1e-6)
    assert _value(above, (0.75, 0.3, TABLE_Z - 0.005))[1] is True
    # The carve box ends about a centimetre past the fingertips: an object top 1.5 cm
    # under them survives, one the fingertips reach into is cleared.
    boxed = Scene(_inputs(tmp_path, "boxed.npz", footprints=(((0.70, 0.80), (0.25, 0.35), 0.0),)),
                  robot_spheres_base=NO_SPHERES, start_hand_base=AWAY)
    over = boxed.world(_hand_with_fingertips_at(0.75, 0.3, 0.015))
    assert over.summary["goal_refusal"] is None and _value(over, (0.75, 0.3, -0.005))[1] is True
    into = boxed.world(_hand_with_fingertips_at(0.75, 0.3, -0.005))
    assert into.summary["goal_refusal"] is None and into.summary["carved_voxels"]["goal"] > 0
    assert _value(into, (0.75, 0.3, -0.005))[1] is False


def test_start_carve_spares_the_support_under_the_start_hand(tmp_path):
    inputs = _inputs(tmp_path)
    resting = _hand_with_fingertips_at(0.75, 0.3, TABLE_Z + 0.005)
    scene = Scene(inputs, robot_spheres_base=NO_SPHERES, start_hand_base=resting)
    assert scene.start_support["z_m"] == pytest.approx(TABLE_Z, abs=1e-6)
    world = scene.world(AWAY)
    assert _value(world, (0.75, 0.3, TABLE_Z - 0.005))[1] is True
    # The robot's own spheres still clear what they overlap, so the measured start stays valid.
    fingers = Scene(inputs, robot_spheres_base=[[0.75, 0.3, TABLE_Z - 0.005, 0.01]], start_hand_base=resting)
    assert _value(fingers.world(AWAY), (0.75, 0.3, TABLE_Z - 0.005))[1] is False


#: A held load whose far end steps down outside the jaws: tops 1.5 cm apart, so the
#: 2 cm ray thickening connects each step to the next.
STAIRS = tuple(((0.43 + 0.04 * step, 0.47 + 0.04 * step), (-0.02, 0.02), -0.015 * step) for step in range(4))


def _held_world(tmp_path, name, footprints, hand, gap=0.04):
    inputs = _inputs(tmp_path, name, footprints=footprints, gap=gap, directive=1.0)
    return Scene(inputs, robot_spheres_base=NO_SPHERES, start_hand_base=hand)


def test_the_whole_connected_load_is_attached_and_leaves_no_ghost(tmp_path):
    hand = _hand_with_fingertips_at(0.45, 0.0, -0.03)
    scene = _held_world(tmp_path, "stairs.npz", STAIRS, hand)
    world = scene.world(AWAY)
    assert world.summary["holding"] is True and world.summary["held_voxels"] > 0
    assert world.summary["held_component"] == "whole"
    assert 0 < world.summary["held_spheres"] <= PUBLIC_DEPTH_VOXEL_V1.held_sphere_slots
    # The far steps, outside the jaws and the hand box, moved to the hand.
    for x, top in ((0.53, -0.03), (0.57, -0.045)):
        assert _value(world, (x, 0.0, top - 0.005))[1] is False
    # Their spheres cover them, in the hand frame.
    held = world.held_spheres_hand
    far = (np.array([0.57, 0.0, -0.05]) - hand[:3, 3]) @ hand[:3, :3]
    assert (np.linalg.norm(held[:, :3] - far, axis=1) <= held[:, 3]).any()
    # The table and the object beside the load stay world.
    assert _value(world, (0.75, 0.3, TABLE_Z - 0.005))[1] is True
    # Without the closed jaws nothing is held and the steps are obstacles.
    open_scene = Scene(_inputs(tmp_path, "open-stairs.npz", footprints=STAIRS, gap=0.04, directive=-1.0),
                       robot_spheres_base=NO_SPHERES, start_hand_base=hand)
    assert _value(open_scene.world(AWAY), (0.57, 0.0, -0.05))[1] is True


def test_the_load_stops_at_its_support_and_at_the_reach_bound(tmp_path):
    # A load standing on the table: its thickened bottom meets the table top.
    resting = _held_world(tmp_path, "resting.npz", (((0.43, 0.47), (-0.02, 0.02), TABLE_Z + 0.02),),
                          _hand_with_fingertips_at(0.45, 0.0, TABLE_Z + 0.01))
    world = resting.world(AWAY)
    assert world.summary["held_voxels"] > 0
    assert _value(world, (0.45, 0.0, TABLE_Z + 0.015))[1] is False
    for point in ((0.55, 0.0, TABLE_Z - 0.005), (0.45, 0.1, TABLE_Z - 0.005)):
        assert _value(world, point)[1] is True
    # Only the load's own voxels: a 4 x 4 cm top thickened 2 cm, nothing of the table.
    assert len(resting.held_spheres_hand) and resting.held_voxels <= 4 * 4 * 3
    assert world.summary["held_component"] == "whole"
    # A 37 cm bar held near one end reaches the bound: it is joined to the scene, so
    # only the depth between the jaws is held and the rest of the bar stays world.
    bar = _held_world(tmp_path, "bar.npz", (((0.43, 0.80), (-0.02, 0.02), 0.0),),
                      _hand_with_fingertips_at(0.45, 0.0, -0.03)).world(AWAY)
    assert bar.summary["held_component"] == "seed_only" and bar.summary["held_voxels"] > 0
    assert _value(bar, (0.45, 0.0, -0.005))[1] is False
    for x in (0.60, 0.75):
        assert _value(bar, (x, 0.0, -0.005))[1] is True


def test_a_goal_never_clears_the_support_around_a_held_load_or_a_joint_goal(tmp_path):
    # A load held by its top, hanging below the fingertips, carried to a low goal
    # elsewhere: its swept spheres (plus their margin) reach the table there.
    load = ((0.43, 0.47), (-0.02, 0.02), TABLE_Z + 0.06)
    scene = _held_world(tmp_path, "carried.npz", (load,), _hand_with_fingertips_at(0.45, 0.0, TABLE_Z + 0.055))
    assert scene.held_voxels > 0
    goal = _hand_with_fingertips_at(0.75, 0.3, TABLE_Z + 0.015)
    world = scene.world(goal)
    assert world.summary["support"]["goal"]["z_m"] == pytest.approx(TABLE_Z, abs=1e-6)
    held = scene.held_spheres_hand
    assert ((held[:, :3] @ goal[2, :3] + goal[2, 3] - held[:, 3]) - PUBLIC_DEPTH_VOXEL_V1.start_carve_margin_m).min() < TABLE_Z
    assert _value(world, (0.75, 0.3, TABLE_Z - 0.005))[1] is True
    # The same for a joint-space goal's robot spheres, which still clear what stands on the table.
    plain = Scene(_inputs(tmp_path, "joint.npz"), robot_spheres_base=NO_SPHERES, start_hand_base=AWAY)
    joint = plain.world(AWAY, goal_spheres_base=[[0.75, 0.3, TABLE_Z, 0.02], [0.45, 0.0, -0.01, 0.01]])
    assert _value(joint, (0.75, 0.3, TABLE_Z - 0.005))[1] is True
    assert _value(joint, (0.45, 0.0, -0.005))[1] is False
    # A measured start must be free, so a start sphere still clears the support under it.
    start = Scene(_inputs(tmp_path, "start.npz"), robot_spheres_base=[[0.75, 0.3, TABLE_Z, 0.02]],
                  start_hand_base=_hand_with_fingertips_at(0.75, 0.3, TABLE_Z + 0.01))
    assert _value(start.world(AWAY), (0.75, 0.3, TABLE_Z - 0.005))[1] is False


def _sphere_bottoms(world, goal):
    held = world.held_spheres_hand
    return held[:, :3] @ goal[2, :3] + goal[2, 3] - held[:, 3]


def test_the_held_spheres_stop_above_the_support_unless_the_load_itself_goes_into_it(tmp_path):
    load = ((0.43, 0.47), (-0.02, 0.02), TABLE_Z + 0.06)
    scene = _held_world(tmp_path, "carried.npz", (load,), _hand_with_fingertips_at(0.45, 0.0, TABLE_Z + 0.055))
    # Set down with the seen load 2 cm above the table: over-cover must not refuse it.
    low = _hand_with_fingertips_at(0.75, 0.3, TABLE_Z + 0.015)
    world = scene.world(low)
    fitted = scene.held_spheres_hand
    assert (fitted[:, :3] @ low[2, :3] + low[2, 3] - fitted[:, 3]).min() < TABLE_Z
    assert world.summary["held_spheres"] == len(world.held_spheres_hand) > 0
    assert _sphere_bottoms(world, low).min() >= TABLE_Z + PUBLIC_DEPTH_VOXEL_V1.voxel_size_m - 1e-6
    # Away from the support the spheres are the fitted ones, and the world hash follows them.
    np.testing.assert_array_equal(scene.world(AWAY).held_spheres_hand, fitted)
    assert world.sha256 != scene.world(AWAY).sha256
    # A step of the load hangs 1.5 cm below the fingertips. Resting 5 mm into the table
    # (within the tolerance) it is still set down; 1.3 cm in, its spheres reach that deep,
    # into the support the carve left, and the goal is refused for it before any solver runs.
    stairs = _held_world(tmp_path, "stairs.npz", STAIRS, _hand_with_fingertips_at(0.45, 0.0, -0.03))
    resting = _hand_with_fingertips_at(0.75, 0.3, TABLE_Z + 0.01)
    assert _sphere_bottoms(stairs.world(resting), resting).min() >= TABLE_Z + 0.01 - 1e-6
    pressed = _hand_with_fingertips_at(0.75, 0.3, TABLE_Z + 0.002)
    deep = stairs.world(pressed)
    refusal = deep.summary["goal_refusal"]
    assert refusal["reason"] == "load_below_support" and refusal["depth_m"] == pytest.approx(0.013, abs=0.003)
    assert "the held object 1." in refusal["message"] and "below the supporting surface" in refusal["message"]
    assert TABLE_Z - 0.02 < _sphere_bottoms(deep, pressed).min() < TABLE_Z - 0.01
    assert _value(deep, (0.75, 0.3, TABLE_Z - 0.005))[1] is True


#: A box carried in the jaws, its seen top 1.5 cm below the fingertips.
CARRIED = (((0.41, 0.49), (-0.04, 0.04), TABLE_Z + 0.06),)


def test_a_load_is_refused_only_over_the_support_it_would_enter(tmp_path):
    scene = _held_world(tmp_path, "edge.npz", CARRIED, _hand_with_fingertips_at(0.45, 0.0, TABLE_Z + 0.075))
    table = scene.support(_hand_with_fingertips_at(0.55, 0.3, TABLE_Z))
    goal = _hand_with_fingertips_at(0.55, 0.3, TABLE_Z)
    assert scene.world(goal).summary["goal_refusal"]["reason"] == "load_below_support"
    # Past the table's edge the depth shows a lower surface under the load: nothing to enter there.
    edge = np.floor((0.70 - scene.parameters.grid_low_m[0]) / scene.parameters.voxel_size_m).astype(int)
    scene._tops[edge:] = table["level_index"] - 5
    assert scene._goal_refusal(_hand_with_fingertips_at(0.85, 0.3, TABLE_Z), table) is None
    assert scene._goal_refusal(goal, table)["depth_m"] == pytest.approx(0.015, abs=1e-6)


def test_a_stray_depth_point_is_no_load_below_the_support(tmp_path):
    scene = _held_world(tmp_path, "stray.npz", CARRIED, _hand_with_fingertips_at(0.45, 0.0, TABLE_Z + 0.075))
    goal = _hand_with_fingertips_at(0.75, 0.3, TABLE_Z + 0.05)
    stray = np.array([[0.0, 0.0, scene.parameters.fingertip_m + 0.08]])
    scene._load_points_hand = np.vstack([scene._load_points_hand, stray])
    assert scene.world(goal).summary["goal_refusal"] is None
    scene._load_points_hand = np.vstack([scene._load_points_hand, np.repeat(stray, 12, axis=0)])
    assert scene.world(goal).summary["goal_refusal"]["reason"] == "load_below_support"


def _across(tmp_path, name, rise, *, reach=0.12):
    """A step in the closed jaws at (0.45, 0) and a lower one running on to ``reach`` along the closing axis."""
    tip = TABLE_Z + 0.065 + rise
    steps = (((0.43, 0.47), (-0.02, 0.02), tip - 0.01), ((0.43, 0.47), (0.02, reach), tip - 0.03))
    return _held_world(tmp_path, name, steps, _hand_with_fingertips_at(0.45, 0.0, tip)), tip


def test_a_component_wide_past_the_jaws_and_standing_on_the_support_stays_scene(tmp_path):
    scene, tip = _across(tmp_path, "standing.npz", 0.0)
    world = scene.world(AWAY)
    assert world.summary["held_component"] == "seed_only"
    # What the jaws do not hold stays where it stands, an obstacle, not a load that follows the hand.
    assert _value(world, (0.45, 0.08, tip - 0.04))[1] is True


def test_the_same_width_in_the_air_is_the_load_as_a_bowl_held_by_its_rim_is(tmp_path):
    scene, tip = _across(tmp_path, "lifted.npz", 0.10)
    world = scene.world(AWAY)
    assert world.summary["held_component"] == "whole"
    assert _value(world, (0.45, 0.08, tip - 0.04))[1] is False


def test_a_load_within_the_overhang_stays_whole_on_the_table(tmp_path):
    scene, _tip = _across(tmp_path, "narrow.npz", 0.0, reach=0.05)
    assert scene.world(AWAY).summary["held_component"] == "whole"


def test_a_load_lowered_into_a_container_keeps_the_table_as_its_support(tmp_path):
    # A basket on the table: a floor and four walls, neither of them a support.
    floor = ((0.60, 0.80), (0.20, 0.40), TABLE_Z + 0.01)
    walls = (((0.60, 0.62), (0.20, 0.40), TABLE_Z + 0.08), ((0.78, 0.80), (0.20, 0.40), TABLE_Z + 0.08),
             ((0.60, 0.80), (0.20, 0.22), TABLE_Z + 0.08), ((0.60, 0.80), (0.38, 0.40), TABLE_Z + 0.08))
    load = ((0.43, 0.47), (-0.02, 0.02), TABLE_Z + 0.06)
    scene = _held_world(tmp_path, "basket.npz", (floor, *walls, load),
                        _hand_with_fingertips_at(0.45, 0.0, TABLE_Z + 0.055))
    goal = _hand_with_fingertips_at(0.70, 0.30, TABLE_Z + 0.05)
    world = scene.world(goal)
    assert world.summary["goal_refusal"] is None
    assert world.summary["support"]["goal"]["z_m"] == pytest.approx(TABLE_Z, abs=1e-6)
    assert _value(world, (0.61, 0.30, TABLE_Z + 0.07))[1] is True
    assert _value(world, (0.70, 0.30, TABLE_Z - 0.005))[1] is True


def test_jaws_that_never_closed_hold_nothing_and_a_hidden_window_falls_back_to_the_grid(tmp_path):
    hand = _hand_with_fingertips_at(0.45, 0.0, -0.03)
    wide = _held_world(tmp_path, "wide.npz", STAIRS, hand, gap=0.078)
    assert wide.holding is False and wide.held_voxels == 0
    assert _value(wide.world(AWAY), (0.57, 0.0, -0.05))[1] is True
    # The arm hides the table around the hand: the support comes from the whole grid.
    arm = [[0.45, 0.0, TABLE_Z, 0.2]]
    hidden = Scene(_inputs(tmp_path, "hidden.npz"), robot_spheres_base=arm, start_hand_base=AWAY)
    support = hidden.support(_hand_with_fingertips_at(0.45, 0.0, 0.05))
    assert support["window"] == "grid" and support["z_m"] == pytest.approx(TABLE_Z, abs=1e-6)
    assert Scene(_inputs(tmp_path, "seen.npz"), robot_spheres_base=NO_SPHERES, start_hand_base=AWAY).support(
        _hand_with_fingertips_at(0.75, 0.3, 0.05))["window"] == "local"


def test_the_table_between_low_jaws_is_support_not_load(tmp_path):
    # Jaws closed around a 2 cm load with the fingertips 1 cm into the table's
    # thickness: the pad slab reaches the table top, which stays world.
    hand = _hand_with_fingertips_at(0.45, 0.0, TABLE_Z - 0.01)
    scene = _held_world(tmp_path, "low.npz", (((0.43, 0.47), (-0.02, 0.02), TABLE_Z + 0.02),), hand)
    world = scene.world(AWAY)
    assert world.summary["held_component"] == "whole" and world.summary["held_voxels"] <= 4 * 4 * 3
    assert _value(world, (0.45, 0.06, TABLE_Z - 0.005))[1] is True
