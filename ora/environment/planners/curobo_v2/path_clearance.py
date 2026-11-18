"""What the goal carve hides from a plan, and the check that finds it.

A plan runs in one grid: the scene with the goal carved out (``depth_world.py``, step 6), so a hand
can reach what it came for. That carve is the hand box at the goal, from the fingertips back past
the wrist, and the planner sees it from the path's first row. So a wrist, a wrist camera or an arm
link may pass through a box wall the world no longer holds, and at the goal may stand inside it.
Light objects give the joints no lag, so nothing downstream notices either.

``path_clearance`` reads a path against any world. ``Worker`` reads each plan against the scene with
no goal carve; a path that lies deeper than ``TOLERANCE_M`` is read again in ``FingerScene``, the
same scene whose goal carve clears what only the fingers and the load may touch. A path that fails
that too is planned again there, and refused when no plan stays clear. A path that passes first
time is the plan the planner gave, so a move that hides nothing changes in no way.

Fingers near what they came for pass, because the finger world still clears them; rows near the
start pass, because the start carve clears the arm's own surroundings in every world here. The
check knows only what the depth shows: an object out of sight stays out of the world.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Where the fingers begin along the approach, in the hand convention: 0.052 m on the Panda and the
#: Robotiq, 0.053 m on the X5, 0.061 m on the ALOHA, each measured from its own collision spheres.
FINGERS_FROM_M = 0.05
#: How far a link or the load may lie inside the world: the world's own voxel, as for a contact path.
TOLERANCE_M = 0.01
#: The sphere name of a held load (``engine.HELD_LINK``).
HELD_LINK = "attached_object"


class FingerScene:
    """``scene`` with a goal carve that stops where the fingers begin, for a joint goal the fingers' spheres only.

    A view: it shares the scene's grids, and its own parameters enter the world's hash."""

    def __init__(self, scene: Any, fingers: Any) -> None:
        parameters = scene.parameters
        box = parameters.hand_box_m
        self._scene = copy.copy(scene)
        # Scene.world reaches back ``goal_standoff_m`` past the box, so the box starts that far on.
        self._scene.parameters = dataclasses.replace(parameters, hand_box_m=(
            box[0], box[1], (FINGERS_FROM_M + parameters.goal_standoff_m, box[2][1])))
        self._fingers = fingers
        self.inputs_sha256 = scene.inputs_sha256

    def world(self, goal_hand_base: Any, goal_spheres_base: Any = None) -> Any:
        spheres = None if goal_spheres_base is None else np.asarray(goal_spheres_base)[self._fingers]
        return self._scene.world(goal_hand_base, spheres)


def path_clearance(rows: Iterable[Any], names: Sequence[str], esdf: Any, parameters: Any) -> dict[str, Any]:
    """A path's smallest clearance from the world ``esdf``: how far its nearest sphere stands off, negative inside.

    ``rows`` holds each path row's spheres ``[x, y, z, radius]`` in the base frame, ``names`` each sphere's link.
    Only the grid's own reach counts, so ``esdf_clamp_m`` bounds a large clearance, and a sphere with no radius
    (an empty load slot) or off the grid has none. Answers ``clearance_m``, ``row``, ``link``, ``radius_m`` and
    ``point_base_m``, with ``row`` ``None`` when no sphere was on the grid.
    Example: path_clearance([spheres], links, world.esdf, parameters)['clearance_m'] < -0.01 lies inside."""
    shape = np.asarray(parameters.grid_shape)
    low, size = parameters.grid_low_m, parameters.voxel_size_m
    worst: dict[str, Any] = {"clearance_m": float(parameters.esdf_clamp_m), "row": None, "link": None,
                             "radius_m": None, "point_base_m": None}
    for row, spheres in enumerate(rows):
        spheres = np.asarray(spheres, dtype=np.float64)
        index = np.floor((spheres[:, :3] - low) / size).astype(np.int64)
        inside = ((index >= 0) & (index < shape)).all(axis=1) & (spheres[:, 3] > 0)
        if not inside.any():
            continue
        clearance = -(spheres[inside, 3] + esdf[tuple(index[inside].T)])
        nearest = int(np.argmin(clearance))
        if float(clearance[nearest]) < worst["clearance_m"]:
            sphere = int(np.nonzero(inside)[0][nearest])
            worst = {"clearance_m": float(clearance[nearest]), "row": row, "link": names[sphere],
                     "radius_m": float(spheres[sphere, 3]), "point_base_m": spheres[sphere, :3].tolist()}
    return worst


def is_clear(found: Mapping[str, Any]) -> bool:
    return found["clearance_m"] >= -TOLERANCE_M


def what_stands(found: Mapping[str, Any], scene: Any, support: Mapping[str, Any] | None) -> str:
    """Words for what ``path_clearance`` found its sphere inside: the other arm, the support, or the depth."""
    parameters = scene.parameters
    point = np.asarray(found["point_base_m"], dtype=np.float64)
    radius = float(found["radius_m"])
    size, low = parameters.voxel_size_m, parameters.grid_low_m
    first = np.maximum(np.floor((point - radius - low) / size).astype(int), 0)
    last = np.ceil((point + radius - low) / size).astype(int) + 1
    if np.asarray(scene.other_arm)[first[0]:last[0], first[1]:last[1], first[2]:last[2]].any():
        return "the other arm"
    if support is not None and point[2] - radius <= float(support["z_m"]):
        return "the supporting surface"
    return "what the depth shows"


__all__ = ["FINGERS_FROM_M", "FingerScene", "HELD_LINK", "TOLERANCE_M", "is_clear", "path_clearance", "what_stands"]
