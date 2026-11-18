"""What a refused pose goal can still do: the tilted goals tried, and the words for what they showed.

``Worker._goal_diagnosis`` drives the solves; everything here is numpy and strings. A goal is tilted
about its grip point, the point between the fingertips, so the hand still closes on the same place:
the wrist swings back and the fingers point another way. Tilts run smallest first, so the first
one that plans is the smallest of those tried. Directions start outward from the arm's base,
where a reach goal usually gains most. Only a tilt that has a joint solution is planned, once, and
at most ``NEARBY_PLANS`` of them, so the search stays bounded. A finite search says what it found,
never that nothing exists.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping

import numpy as np

from ora.contracts.pose import matrix_quaternion
from ora.environment.controllers.depth_world import pose_matrix

#: The tilts tried, in degrees, each around ``TILT_DIRECTIONS`` directions: 32 IK solves at most.
TILTS_DEG = (10.0, 20.0, 30.0, 45.0)
TILT_DIRECTIONS = 8
#: Tilts with a joint solution that are planned, one attempt each: a failed plan is the search's dearest step.
NEARBY_PLANS = 2
#: Converged seeds checked for what stops them, of the last IK attempt's population.
BLOCKING_SEEDS = 8


def _turn(axis: Any, angle: float) -> Any:
    """Rodrigues' rotation about the unit ``axis``."""
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def _outward(approach: Any, grip: Any) -> Any:
    """A unit vector across ``approach``, leaning outward from the base at the origin of the frame."""
    for direction in (np.array([grip[0], grip[1], 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        across = direction - float(direction @ approach) * approach
        if np.linalg.norm(across) > 1e-6:
            return across / np.linalg.norm(across)
    raise ValueError("no direction lies across the approach")


def tilted_goals(pose: list[float], tool_from_hand: Any, grip_m: float) -> Iterator[dict[str, Any]]:
    """Each tilted goal of base-frame ``pose``, smallest tilt first: its degrees, pose and approach."""
    hand = pose_matrix(pose) @ tool_from_hand
    approach = hand[:3, 2]
    grip = hand[:3, 3] + approach * grip_m
    across = _outward(approach, grip)
    side = np.cross(approach, across)
    hand_from_tool = np.linalg.inv(tool_from_hand)
    for tilt in TILTS_DEG:
        for step in range(TILT_DIRECTIONS):
            angle = 2.0 * np.pi * step / TILT_DIRECTIONS
            lean = np.cos(angle) * across + np.sin(angle) * side
            turn = _turn(np.cross(approach, lean), np.radians(tilt))
            moved = np.eye(4)
            moved[:3, :3] = turn @ hand[:3, :3]
            moved[:3, 3] = grip - moved[:3, 2] * grip_m
            tool = moved @ hand_from_tool
            yield {"tilt_deg": tilt, "pose_base_xyzw": [*tool[:3, 3].tolist(), *matrix_quaternion(tool[:3, :3].tolist())],
                   "approach_base": moved[:3, 2].tolist()}


def what_blocks(checks: list[Mapping[str, Any]]) -> str | None:
    """Which constraint stops the seeds that reached the pose: ``self``, ``world`` or ``None`` unknown."""
    alone = {"self": 0, "world": 0}
    for check in checks:
        against_self, against_world = check["self_collision"] > 0, (check["world_collision"] or 0) > 0
        if against_self != against_world:
            alone["self" if against_self else "world"] += 1
    if alone["self"] == alone["world"]:
        return None
    return "self" if alone["self"] > alone["world"] else "world"


def _vector(values: Any) -> str:
    return "(" + ", ".join(f"{round(float(value), 2) + 0.0:.2f}" for value in values) + ")"


def nearby_sentence(found: Mapping[str, Any] | None, approach_world: Any, turn_world: Any) -> str:
    """What the tilts showed, as a clause: the smallest that planned, or that none of those tried did."""
    if found is None:
        return f"no approach tilted up to {TILTS_DEG[-1]:.0f} degrees was found to plan here either"
    tilted = turn_world @ np.asarray(found["approach_base"], dtype=float)
    return (f"with the fingertips at the same point, an approach tilted {found['tilt_deg']:.0f} degrees plans: "
            f"{_vector(tilted)} in world coordinates instead of {_vector(approach_world)}")


__all__ = ["BLOCKING_SEEDS", "NEARBY_PLANS", "TILTS_DEG", "TILT_DIRECTIONS", "nearby_sentence", "tilted_goals",
           "what_blocks"]
