"""The world's cuRobo V2 export holds the same occupancy the v0.7.8 kernel collides with."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ora.environment.controllers.depth_world import (
    CUROBO_UNOBSERVED_BORDER_VOXELS, PUBLIC_DEPTH_VOXEL_V1, DepthWorldError, build_world, curobo_v2_grid,
)
from tests.support.libero.world_inputs import DOWN, frame, overhead, read, write

BASE_WORLD = np.array([-0.66, 0.0, 1.0])
#: Two objects on the table and one past its far edge, in the unobserved border of the grid's +x face.
FOOTPRINTS = (((0.35, 0.55), (-0.08, 0.08), 0.0), ((0.60, 0.70), (0.20, 0.30), 0.10),
              ((0.98, 1.06), (-0.05, 0.05), 0.02))
P = PUBLIC_DEPTH_VOXEL_V1


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    path = tmp_path_factory.mktemp("v2") / "inputs.npz"
    inputs = read(write(path, agentview=overhead((0.6, 0.0, 0.8), FOOTPRINTS, base_world=BASE_WORLD),
                        base_world=BASE_WORLD))
    away = frame(DOWN, (0.2, 0.45, 0.5))
    return build_world(inputs, robot_spheres_base=np.zeros((0, 4)), start_hand_base=away, goal_hand_base=away)


def _v2_centres(grid):
    """V2's voxel centres: ``pose - dims / 2 + (i + 1/2) voxel`` with ``round(dims / voxel)`` voxels per axis."""
    size = grid["voxel_size_m"]
    shape = [round(dim / size) for dim in grid["dims_m"]]
    axes = [np.asarray(c) - d / 2 + (np.arange(n) + 0.5) * size
            for c, d, n in zip(grid["center_base_m"], grid["dims_m"], shape)]
    return shape, axes


def _v2_sample(grid, point):
    """V2's trilinear lookup (voxel centres at integers, only in-grid corners weigh, all outside reads free)."""
    size, features = grid["voxel_size_m"], grid["features"].astype(np.float32)
    shape = np.asarray(features.shape)
    local = (np.asarray(point) - np.asarray(grid["center_base_m"])) / size + shape * 0.5 - 0.5
    base = np.floor(local).astype(int)
    fraction = local - base
    total = weight = 0.0
    for corner in np.ndindex(2, 2, 2):
        index = base + corner
        if ((index < 0) | (index >= shape)).any():
            continue
        w = np.prod(np.where(corner, fraction, 1 - fraction))
        total, weight = total + w * features[tuple(index)], weight + w
    return total / weight if weight > 0 else np.inf


def test_v2_grid_is_the_effective_volume_with_the_same_occupancy(world):
    grid = curobo_v2_grid(world.esdf, P)
    border = CUROBO_UNOBSERVED_BORDER_VOXELS
    shape, axes = _v2_centres(grid)
    assert grid["features"].dtype == np.float16
    assert list(grid["features"].shape) == shape == [n - 2 * border for n in P.grid_shape] == [121, 121, 101]
    # The kernel truncates float32 dims / voxel; it agrees with the rounded shape.
    assert [int(np.float32(d) / np.float32(grid["voxel_size_m"])) for d in grid["dims_m"]] == shape
    # Every V2 voxel centre is the centre of v0.7.8 voxel i + border.
    for axis, centres in enumerate(axes):
        expected = P.grid_low_m[axis] + (np.arange(shape[axis]) + border + 0.5) * P.voxel_size_m
        np.testing.assert_allclose(centres, expected, atol=1e-9)
    np.testing.assert_allclose(np.asarray(grid["center_base_m"]) - np.asarray(grid["dims_m"]) / 2,
                               P.effective_low_m, atol=1e-9)
    inner = (slice(border, -border),) * 3
    occupied = world.occupied[inner]
    assert occupied.any() and (~occupied).any()
    # Negative inside: sign alone gives the occupancy, through float16, at every voxel.
    np.testing.assert_array_equal(grid["features"] < 0, occupied)
    np.testing.assert_allclose(grid["features"].astype(np.float32), -world.esdf[inner], atol=1e-4)
    assert float(np.abs(grid["features"]).min()) >= 0.5 * P.voxel_size_m - 1e-4


def test_v2_lookup_matches_v078_occupancy_at_voxel_centres_and_leaves_the_border_out(world):
    grid = curobo_v2_grid(world.esdf, P)
    border = CUROBO_UNOBSERVED_BORDER_VOXELS
    rng = np.random.default_rng(0)
    occupied = np.argwhere(world.occupied)
    effective = occupied[((occupied >= border) & (occupied < np.asarray(P.grid_shape) - border)).all(axis=1)]
    in_border = occupied[((occupied < border) | (occupied >= np.asarray(P.grid_shape) - border)).any(axis=1)]
    assert len(effective) and len(in_border), "the scene must reach into the border"
    free = np.argwhere(~world.occupied[(slice(border, -border),) * 3]) + border
    for index in [*effective[rng.choice(len(effective), 50)], *free[rng.choice(len(free), 50)]]:
        centre = P.grid_low_m + (index + 0.5) * P.voxel_size_m
        assert (_v2_sample(grid, centre) < 0) == bool(world.occupied[tuple(index)])
    # What v0.7.8 ignores in its border reads free in V2: outside the grid.
    for index in in_border[:20]:
        centre = P.grid_low_m + (index + 0.5) * P.voxel_size_m
        assert (centre < P.effective_low_m - 0.5 * P.voxel_size_m).any() or (
            centre > P.effective_high_m + 0.5 * P.voxel_size_m).any() or _v2_sample(grid, centre) > 0


def test_v2_grid_hash_binds_layout_and_values_and_odd_sizes_are_refused(world):
    grid = curobo_v2_grid(world.esdf, P)
    assert curobo_v2_grid(world.esdf.copy(), P)["sha256"] == grid["sha256"]
    changed = world.esdf.copy()
    changed[60, 60, 50] = -changed[60, 60, 50]
    assert curobo_v2_grid(changed, P)["sha256"] != grid["sha256"]
    odd = replace(P, voxel_size_m=0.013)
    with pytest.raises(DepthWorldError, match="whole voxels"):
        curobo_v2_grid(np.zeros((16, 16, 16), dtype=np.float32), odd)  # 10 voxels of 13 mm
