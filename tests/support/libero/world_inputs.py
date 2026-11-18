"""Synthetic public depth for the LIBERO planner's collision world (``depth_world.py``).

A camera looks straight down on axis-aligned boxes standing on a table; the
scene is described in the robot base frame and the files carry world-frame
calibration, as a LIBERO observation does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ora.environment.controllers.depth_world import read_world_inputs, write_world_inputs

TABLE_Z = -0.12
#: Camera axes looking straight down: x right, y down the image (base -y), z = base -z.
DOWN = np.diag([1.0, -1.0, -1.0])


def frame(rotation: Any, position: Sequence[float]) -> Any:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def overhead(camera_base: Sequence[float], footprints: Sequence[Any], *, base_world: Sequence[float],
             size: int = 128, focal: float = 100.0) -> dict[str, Any]:
    """Depth of a downward camera over the table and ``(x_range, y_range, top_z)`` boxes."""
    centre = size / 2.0
    rows, columns = np.mgrid[0:size, 0:size].astype(np.float64)
    depth = np.full((size, size), 50.0)
    for x_range, y_range, top in [((0.0, 1.0), (-0.6, 0.6), TABLE_Z), *footprints]:
        candidate = camera_base[2] - top
        x = camera_base[0] + candidate * (columns - centre) / focal
        y = camera_base[1] - candidate * (rows - centre) / focal
        hit = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
        depth = np.where(hit & (candidate < depth), candidate, depth)
    return {"depth_m": depth.astype(np.float32),
            "intrinsic_cv": np.array([[focal, 0, centre], [0, focal, centre], [0, 0, 1.0]]),
            "cam2world_cv": frame(DOWN, np.asarray(camera_base) + np.asarray(base_world)),
            "depth_range_m": (0.01, 10.0)}


def blind() -> dict[str, Any]:
    """A camera with no valid pixel."""
    return {"depth_m": np.full((8, 8), 20.0, dtype=np.float32), "intrinsic_cv": np.eye(3),
            "cam2world_cv": np.eye(4), "depth_range_m": (0.01, 10.0)}


def write(path: Path, *, agentview: Any, wrist: Any = None, base_world: Sequence[float] = (0.0, 0.0, 0.0),
          joints: Sequence[float] = (0.0,) * 7, gap: float = 0.08, directive: float = -1.0) -> dict[str, str]:
    return write_world_inputs(
        path, cameras={"agentview": agentview, "wrist": wrist if wrist is not None else blind()},
        base_pose_world_xyzw=[*base_world, 0.0, 0.0, 0.0, 1.0], joint_positions=list(joints),
        gripper_finger_qpos=[gap / 2, -gap / 2], gripper_directive=directive,
    )


def read(record: dict[str, str]) -> Any:
    return read_world_inputs(Path(record["path"]), record["sha256"])
