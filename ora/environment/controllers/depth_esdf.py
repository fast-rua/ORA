"""The depth world's signed distance field (``depth_world.py``, step 7; numpy only).

Euclidean distance between voxel centres, positive inside occupied space and negative outside, half a voxel at
the boundary, clamped to the parameters' ``esdf_clamp_m``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _squared_distance(features: Any, steps: int, cap: int) -> Any:
    """Squared voxel distance to the nearest feature at most ``steps`` voxels away on each axis, at most ``cap``.

    Exact below ``cap``, ``cap`` everywhere else. No value exceeds ``cap``, so a step whose square reaches it would
    change nothing; ``signed_distance`` stops ``steps`` below that.
    """
    dtype = np.uint16 if cap + steps * steps <= np.iinfo(np.uint16).max else np.int64
    distance = np.where(features, 0, cap).astype(dtype)
    for axis in range(3):
        result = distance.copy()
        for step in range(1, steps + 1):
            low = tuple(slice(0, -step) if index == axis else slice(None) for index in range(3))
            high = tuple(slice(step, None) if index == axis else slice(None) for index in range(3))
            square = dtype(step * step)
            np.minimum(result[low], distance[high] + square, out=result[low])
            np.minimum(result[high], distance[low] + square, out=result[high])
        distance = result
    return distance


def _box(mask: Any, pad: int) -> tuple[slice, ...]:
    """The bounding box of ``mask``'s true voxels grown by ``pad`` voxels on every side, within the grid."""
    box = []
    for axis in range(3):
        hits = np.flatnonzero(mask.any(axis=tuple(other for other in range(3) if other != axis)))
        box.append(slice(max(int(hits[0]) - pad, 0), min(int(hits[-1]) + pad + 1, mask.shape[axis])))
    return tuple(box)


def signed_distance(occupied: Any, parameters: Any) -> Any:
    """The clamped ESDF of an occupancy grid, float32, positive inside.

    The Euclidean distance between voxel centres, exact within ``radius`` voxels on each axis (beyond that it lies
    past the clamp), less half a voxel and clamped. Every squared distance from ``cap`` on reads the clamp, so each
    transform keeps its values at most ``cap`` and runs only where one below it can occur: the outside transform
    within ``steps`` voxels of the occupied box, the inside one in that box grown by one voxel (a free voxel past it
    is never nearer an occupied one than the free border is). The result is the whole grid's, bit for bit.
    """
    size = parameters.voxel_size_m
    clamp = parameters.esdf_clamp_m / size
    radius = int(math.ceil(clamp)) + 1
    cap = math.ceil((clamp + 0.5) ** 2)
    while math.sqrt(cap) - 0.5 < clamp:
        cap += 1
    steps = min(radius, math.isqrt(cap - 1))
    outside = np.full(occupied.shape, cap, dtype=np.int64 if cap > np.iinfo(np.uint16).max else np.uint16)
    inside = np.zeros_like(outside)
    if occupied.any():
        box = _box(occupied, steps)
        outside[box] = _squared_distance(occupied[box], steps, cap)
        box = _box(occupied, 1)
        inside[box] = _squared_distance(~occupied[box], steps, cap)
    # One distance per integer squared distance, as the voxels' own arithmetic gives it.
    values = np.minimum(np.sqrt(np.arange(cap + 1, dtype=np.float64)) - 0.5, clamp)
    return np.where(occupied, (values * size).astype(np.float32)[inside],
                    (-values * size).astype(np.float32)[outside])
