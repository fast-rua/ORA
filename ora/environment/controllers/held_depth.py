"""Which public depth is the robot's own beyond its body: the depth a held load links to, and the mixed pixels at the
robot's edge. numpy only.

``base_placement.public_points`` takes both out of what a base move may meet, and carries the load's: the planner's
world bounds its held load 0.15 m past the jaws and leaves a load that stands on a support to the scene, and in the
RoboCasa smokes the rest of a bowl held by its rim, left in the depth, refused base moves as the load meeting itself.

* ``load_voxels``: the voxels of the depth a chain of voxels links to the load's spheres. A chain runs through voxels
  within two voxels of each other (``NEIGHBOURS``). Where it runs on past the load's reach, the load touches the scene
  (a wall it hangs beside, the door a handle belongs to): each voxel within the reach goes to whichever it is fewer
  links from, the load's seeds or the depth past the reach. A fragment within three voxels of the load's depth
  (``FRAGMENT_NEIGHBOURS``) whose own chain stays within the reach joins it.
* ``edge_pixels``: a pixel next to one of the robot's own, well behind it and well in front of another neighbour, is a
  mixed pixel at the robot's silhouette (its depth between the robot's and the scene's), which moves with the robot.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

#: The voxels whose centres lie within two voxels of a voxel's: the depth world links its voxels after thickening each
#: surface 2 cm along its camera ray, and the public points here are not thickened. At SteamInMicrowave [59] (smoke
#: ``rc-ab-6864e57de``) the voxels next to each other linked 55 % of the depth that moved with the hand to the next
#: moment, within two voxels all of it, and none that stayed.
NEIGHBOURS = np.array([(a, b, c) for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)
                       if 0 < a * a + b * b + c * c <= 4])
#: A fragment of depth this near the load's (within three voxels) joins it: in smoke ``rc-ab-6d8eb1acf`` a piece of
#: 43 voxels that moved with the bowl lay 2.4 cm from the rest of it (StoreLeftoversInBowl, step 2858).
FRAGMENT_NEIGHBOURS = np.array([(a, b, c) for a in range(-3, 4) for b in range(-3, 4) for c in range(-3, 4)
                                if 0 < a * a + b * b + c * c <= 9])


def hops(cells: Any, sources: Any, offsets: Any = NEIGHBOURS, limit: int | None = None) -> Any:
    """Per cell of ``cells`` (distinct integer voxels), the fewest steps, each to a voxel at one of ``offsets``, from
    one of ``sources`` (-1: none reaches it, or not within ``limit`` steps)."""
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    found = np.full(len(cells), -1, dtype=np.int64)
    if not len(cells):
        return found
    reach = int(np.abs(offsets).max())
    shifted = cells - cells.min(axis=0) + reach
    size = shifted.max(axis=0) + reach + 1

    def key(voxels: Any) -> Any:
        return (voxels[..., 0] * size[1] + voxels[..., 1]) * size[2] + voxels[..., 2]

    order = np.argsort(key(shifted))
    keys = key(shifted)[order]
    frontier = np.nonzero(np.asarray(sources, dtype=bool))[0]
    found[frontier] = 0
    step = 0
    while len(frontier) and (limit is None or step < limit):
        step += 1
        near = key(shifted[frontier][:, None, :] + offsets[None]).reshape(-1)
        at = np.minimum(np.searchsorted(keys, near), len(keys) - 1)
        reached = order[at[keys[at] == near]]
        frontier = np.unique(reached[found[reached] < 0])
        found[frontier] = step
    return found


def _within(points: Any, spheres: Any, margin_m: float) -> Any:
    distance = np.sqrt(((points[:, None, :] - spheres[None, :, :3]) ** 2).sum(axis=2))
    return (distance <= spheres[None, :, 3] + margin_m).any(axis=1)


def load_voxels(points: Any, held: Any, margin_m: float, reach_m: float, shell_m: float, cell_m: float) -> Any:
    """The centres of the voxels (of ``cell_m``) of the depth ``points`` that is part of the load held as ``held``
    (world ``[x, y, z, radius]`` spheres), as the module docstring says: within ``reach_m`` of the spheres' centre,
    linked to the voxels next to them (within their radius, ``margin_m`` and a voxel's diagonal), the depth past the
    reach followed ``shell_m`` beyond it. ``(0, 3)`` when nothing is held or nothing links to it."""
    none = np.zeros((0, 3))
    held = np.asarray(held, dtype=np.float64).reshape(-1, 4)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(held) or not len(points):
        return none
    centre = held[:, :3].mean(axis=0)
    near = points[((points - centre) ** 2).sum(axis=1) <= (reach_m + shell_m) ** 2]
    if not len(near):
        return none
    cells = np.unique(np.floor(near / cell_m).astype(np.int64), axis=0)
    centres = (cells + 0.5) * cell_m
    inside = ((centres - centre) ** 2).sum(axis=1) <= reach_m ** 2
    seeds = inside & _within(centres, held, margin_m + math.sqrt(3.0) * cell_m)
    if not seeds.any():
        return none
    from_seeds = hops(cells, seeds)
    linked = from_seeds >= 0
    load, past = linked & inside, linked & ~inside
    if past.any():
        from_past = hops(cells, past)
        load &= (from_past < 0) | (from_seeds < from_past)
    apart = ~linked
    joined = True
    while joined:
        joined = False
        touching = apart & (hops(cells, load, FRAGMENT_NEIGHBOURS, limit=1) == 1)
        while touching.any():
            start = np.zeros(len(cells), dtype=bool)
            start[np.nonzero(touching)[0][0]] = True
            piece = np.zeros(len(cells), dtype=bool)
            piece[np.nonzero(apart)[0]] = hops(cells[apart], start[apart]) >= 0
            touching &= ~piece
            apart &= ~piece
            if not (piece & ~inside).any():
                load |= piece
                joined = True
    return centres[load]


def edge_pixels(camera: Mapping[str, Any], pixels: Any, mine: Any, step_m: float) -> Any:
    """Which of a camera's back-projected ``pixels`` (flat indices into its ``depth_m``) show the edge of the robot's
    own (``mine``, per pixel): next to an own pixel more than ``step_m`` nearer, and to a pixel more than ``step_m``
    farther, so between the two."""
    depth = np.asarray(camera["depth_m"], dtype=np.float64)
    near_m, far_m = (float(value) for value in camera["depth_range_m"])
    seen = np.where(np.isfinite(depth) & (depth > near_m) & (depth < far_m), depth, np.nan)
    pixels = np.asarray(pixels, dtype=np.int64)
    own = np.zeros(depth.shape, dtype=bool)
    own.reshape(-1)[pixels[np.asarray(mine, dtype=bool)]] = True
    padded, padded_own = np.pad(seen, 1, constant_values=np.nan), np.pad(own, 1)
    rows, columns = depth.shape
    nearer_own, farther = np.zeros(depth.shape, dtype=bool), np.zeros(depth.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    other = padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + columns]
                    nearer_own |= padded_own[1 + dr:1 + dr + rows, 1 + dc:1 + dc + columns] & (other < seen - step_m)
                    farther |= other > seen + step_m
    return (nearer_own & farther & ~own).reshape(-1)[pixels]


__all__ = ["FRAGMENT_NEIGHBOURS", "NEIGHBOURS", "edge_pixels", "hops", "load_voxels"]
