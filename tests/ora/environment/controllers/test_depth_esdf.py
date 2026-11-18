"""The depth world's ESDF (``depth_esdf``): the capped, boxed transforms are the whole-grid transform, bit for bit."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from ora.environment.controllers.depth_esdf import signed_distance
from ora.environment.controllers.depth_world import PUBLIC_DEPTH_VOXEL_V1, PUBLIC_DEPTH_VOXEL_X5_V1


def _signed_distance_whole_grid(occupied, parameters):
    """The ESDF before the capped, boxed transforms (kept verbatim as the oracle)."""
    def squared_distance(features, radius):
        distance = np.where(features, 0, 1 << 20).astype(np.int32)
        for axis in range(3):
            result = distance.copy()
            for step in range(1, radius + 1):
                low = tuple(slice(0, -step) if index == axis else slice(None) for index in range(3))
                high = tuple(slice(step, None) if index == axis else slice(None) for index in range(3))
                np.minimum(result[low], distance[high] + step * step, out=result[low])
                np.minimum(result[high], distance[low] + step * step, out=result[high])
            distance = result
        return distance

    size = parameters.voxel_size_m
    clamp = parameters.esdf_clamp_m / size
    radius = int(math.ceil(clamp)) + 1
    outside = np.minimum(np.sqrt(squared_distance(occupied, radius)) - 0.5, clamp)
    inside = np.minimum(np.sqrt(squared_distance(~occupied, radius)) - 0.5, clamp)
    return (np.where(occupied, inside, -outside) * size).astype(np.float32)


def _blobs(rng, shape, count):
    grid = np.zeros(shape, dtype=bool)
    for _ in range(count):
        low = rng.integers(0, shape)
        high = np.minimum(low + rng.integers(1, 12, size=3), shape)
        grid[tuple(slice(a, b) for a, b in zip(low, high))] = True
    return grid


@pytest.mark.parametrize("parameters", [
    PUBLIC_DEPTH_VOXEL_V1, PUBLIC_DEPTH_VOXEL_X5_V1,
    replace(PUBLIC_DEPTH_VOXEL_V1, esdf_clamp_m=0.05), replace(PUBLIC_DEPTH_VOXEL_V1, esdf_clamp_m=0.006),
    replace(PUBLIC_DEPTH_VOXEL_V1, voxel_size_m=0.02, esdf_clamp_m=0.07),
], ids=["v1", "x5", "clamp-5-voxels", "clamp-under-a-voxel", "coarse"])
def test_the_capped_boxed_esdf_is_the_whole_grid_transform_bit_for_bit(parameters):
    rng = np.random.default_rng(5)
    shape = (60, 50, 40)
    surface = np.zeros(shape, dtype=bool)
    surface[:, :, 8:10] = True  # a table across the grid
    corner = np.zeros(shape, dtype=bool)
    corner[0, 0, 0] = corner[-1, -1, -1] = True
    grids = [np.zeros(shape, dtype=bool), np.ones(shape, dtype=bool), corner, surface | _blobs(rng, shape, 12),
             _blobs(rng, shape, 3), rng.random(shape) < 0.02, ~(rng.random(shape) < 0.02)]
    single = np.zeros(shape, dtype=bool)
    single[30, 25, 20] = True
    grids.append(single)
    for occupied in grids:
        esdf = signed_distance(occupied, parameters)
        assert esdf.dtype == np.float32 and esdf.shape == shape
        assert esdf.tobytes() == _signed_distance_whole_grid(occupied, parameters).tobytes()
