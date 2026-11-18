"""Does a planner's robot model land where the simulator's does? One check for every robot.

A planner's collision spheres, tool frame and joint order only mean anything if
its forward kinematics agrees with what the simulator executes (cuRobo V2's
bundled Panda fingers sat 6.5 mm off robosuite's, found only by a table refusal).
A backend hands ``parity`` joint vectors with both answers as base-frame 4x4
transforms: live at session start, and from a recorded fixture in CI, where
neither simulator nor planner runs (``tests/data/fk-parity/``). numpy only.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

#: The position (m) and rotation (rad) a planner's tool frame may differ by: the arrival tolerance of each robot.
ROBOT_TOLERANCES: dict[str, tuple[float, float]] = {
    "aloha_agilex": (0.002, 0.02), "arx_x5": (0.002, 0.02), "franka_panda": (0.002, 0.02),
    "franka_robotiq": (0.002, 0.02), "yam_linear_4310": (0.002, 0.02),
}


def parity(samples: Sequence[Mapping[str, Any]], position_tolerance_m: float,
           rotation_tolerance_rad: float) -> dict[str, Any]:
    """The worst difference over ``samples`` (``simulator`` and ``planner`` 4x4 transforms) and whether it passes."""
    if not samples:
        raise ValueError("an FK parity check needs samples")
    worst_position, worst_rotation = 0.0, 0.0
    for sample in samples:
        first, second = (np.asarray(sample[key], dtype=np.float64) for key in ("simulator", "planner"))
        if first.shape != (4, 4) or second.shape != (4, 4) or not np.isfinite(first).all() \
                or not np.isfinite(second).all():
            raise ValueError("FK parity samples are 4x4 transforms")
        worst_position = max(worst_position, float(np.linalg.norm(first[:3, 3] - second[:3, 3])))
        cosine = (float(np.trace(first[:3, :3].T @ second[:3, :3])) - 1.0) / 2.0
        worst_rotation = max(worst_rotation, math.acos(max(-1.0, min(1.0, cosine))))
    return {"samples": len(samples), "position_error_m": worst_position, "rotation_error": worst_rotation,
            "position_tolerance_m": position_tolerance_m, "rotation_tolerance": rotation_tolerance_rad,
            "passed": worst_position <= position_tolerance_m and worst_rotation <= rotation_tolerance_rad}


__all__ = ["ROBOT_TOLERANCES", "parity"]
