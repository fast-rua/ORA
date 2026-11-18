"""What the depth world knows of a held load (``depth_world.py``, numpy only): the spheres it is
attached to the planner with, and whether a goal puts the hand or the load under the support."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

#: A component's column stands on a support when its lowest voxel lies this close above the support's top voxel.
STANDING_VOXELS = 2
#: Fewer depth points of a load than this below the support are noise, not a load.
LOAD_BELOW_MIN_POINTS = 8


def stands_past_jaws(voxels: Any, centres: Any, hand: Any, finger_gap_m: float, support: Mapping[str, Any],
                     parameters: Any) -> bool:
    """Whether a component is what the load touches: wide past the jaws and standing on the support under it.

    Width alone does not say: a bowl held by its rim reaches 13 cm past the jaws. It leaves
    the support as it lifts; a cup and its plate stay on it.
    """
    if not len(voxels):
        return False
    across = np.abs((centres - hand[:3, 3]) @ hand[:3, 1]).max() + 0.5 * parameters.voxel_size_m
    if across - 0.5 * finger_gap_m <= parameters.held_overhang_m:
        return False
    standing = voxels[voxels[:, 2] <= support["top_index"] + STANDING_VOXELS]
    return len(np.unique(standing[:, :2], axis=0)) >= parameters.held_rest_columns


def _over_support(point: Any, support: Mapping[str, Any], tops: Any, parameters: Any) -> bool:
    """Whether ``support`` is what lies under ``point``: inside the grid, and no lower surface seen there."""
    column = np.floor((point[:2] - parameters.grid_low_m[:2]) / parameters.voxel_size_m).astype(np.int64)
    near = tops[max(0, column[0] - 2):max(0, column[0] + 3), max(0, column[1] - 2):max(0, column[1] + 3)]
    observed = near[near >= 0]
    return bool(near.size) and not (len(observed) and (observed < support["level_index"]).all())


def fingertip_refusal(tips: Any, support: Mapping[str, Any], tops: Any, parameters: Any) -> dict[str, Any] | None:
    """Why a goal is refused for its fingertips (base frame), or ``None``."""
    floor = support["z_m"] - parameters.support_tolerance_m
    below = [float(tip[2]) for tip in tips if tip[2] < floor and _over_support(tip, support, tops, parameters)]
    if not below:
        return None
    depth = support["z_m"] - min(below)
    return {
        "reason": "fingertips_below_support",
        "fingertip_z_m": min(below), "support_z_m": support["z_m"], "depth_m": depth,
        "tolerance_m": parameters.support_tolerance_m,
        "message": (f"the goal puts a fingertip {depth * 100:.1f} cm below the supporting surface "
                    f"at z {support['z_m']:.3f} m (base frame) seen in the depth under it, more than "
                    f"the {parameters.support_tolerance_m * 100:.0f} cm allowed; "
                    "raise the target so the fingertips stay above that surface"),
    }


def load_refusal(points: Any, support: Mapping[str, Any], tops: Any, parameters: Any) -> dict[str, Any] | None:
    """Why a goal is refused for what the hand holds, its seen ``points`` there (base frame), or ``None``."""
    below = points[points[:, 2] < support["z_m"] - parameters.support_tolerance_m]
    size, low = parameters.voxel_size_m, parameters.grid_low_m
    columns = np.floor((below[:, :2] - low[:2]) / size).astype(np.int64)
    over = np.zeros(len(below), dtype=bool)
    for column in np.unique(columns, axis=0):
        chosen = (columns == column).all(axis=1)
        over[chosen] = _over_support(below[chosen][0], support, tops, parameters)
    if over.sum() < LOAD_BELOW_MIN_POINTS:
        return None
    lowest = float(below[over, 2].min())
    depth = support["z_m"] - lowest
    return {
        "reason": "load_below_support",
        "load_z_m": lowest, "support_z_m": support["z_m"], "depth_m": depth,
        "tolerance_m": parameters.support_tolerance_m,
        "message": (f"the goal puts the held object {depth * 100:.1f} cm below the supporting surface "
                    f"at z {support['z_m']:.3f} m (base frame) seen in the depth under it, more than "
                    f"the {parameters.support_tolerance_m * 100:.0f} cm allowed; "
                    "raise the target so what the hand holds stays above that surface"),
    }


def fit_held_spheres(centres_hand: Any, parameters: Any) -> Any:
    """At most ``held_sphere_slots`` spheres covering the held voxels (centres in the hand frame).

    Deterministic: farthest-point seeds from the first voxel in grid order,
    then a fixed number of k-means steps; each sphere reaches its farthest
    voxel centre plus half a voxel diagonal, so it covers its voxels whole.
    """
    corner = 0.5 * math.sqrt(3.0) * parameters.voxel_size_m
    points = np.asarray(centres_hand, dtype=np.float64).reshape(-1, 3)
    slots = parameters.held_sphere_slots
    if len(points) <= slots:
        return np.hstack([points, np.full((len(points), 1), corner)])
    chosen = [0]
    nearest = np.linalg.norm(points - points[0], axis=1)
    for _ in range(slots - 1):
        chosen.append(int(np.argmax(nearest)))
        nearest = np.minimum(nearest, np.linalg.norm(points - points[chosen[-1]], axis=1))
    centres = points[chosen]
    for _ in range(10):
        label = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
        for slot in range(slots):
            members = points[label == slot]
            if len(members):
                centres[slot] = members.mean(axis=0)
    label = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
    radius = np.zeros(slots)
    np.maximum.at(radius, label, np.linalg.norm(points - centres[label], axis=1))
    used = np.bincount(label, minlength=slots) > 0
    return np.hstack([centres[used], (radius[used] + corner)[:, None]])


def clip_to_floor(spheres_hand: Any, goal: Any, floor_z: float) -> Any:
    """``spheres_hand`` with nothing below ``floor_z`` at the hand pose ``goal``.

    A sphere covers its voxels whole and more, and those voxels hold each
    surface thickened along its camera ray, so the spheres reach past the load.
    Each is cut to its height over the floor; one centred below it covers
    only that excess and goes.
    """
    heights = spheres_hand[:, :3] @ goal[2, :3] + goal[2, 3]
    above = heights > floor_z
    clipped = spheres_hand[above].copy()
    clipped[:, 3] = np.minimum(clipped[:, 3], heights[above] - floor_z)
    return clipped


__all__ = ["clip_to_floor", "fit_held_spheres"]
