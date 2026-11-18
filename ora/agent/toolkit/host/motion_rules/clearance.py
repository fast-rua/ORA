"""Observed clearance below a commanded jaw footprint, from public depth only.

``observed_clearance_reference_v2`` annotates each ``move_pose`` target (and
the ``move_to`` / ``rotate_wrist`` / ``rotate_pitch`` targets that compile to
it) with one measurement taken before submission: how far the lowest point of
the jaw footprint at the commanded pose lies above the highest surface the
current depth images show beneath it or within ``RADIUS_M`` of it.

Inputs are only what the policy already receives: the observation's public
depth images with their per-observation calibration, the measured robot state,
the public embodiment geometry (finger collision bounds in the controlled-point
frame, opened along their published directions by the measured finger
positions) and, for a held object, the policy's own earlier localization of it
(``observed_clearance_runtime``). No scene geometry, layout or contact data.

Method. The footprint is the two finger boxes (plus the held-object box) posed
at the target; its lowest corner is the reported lowest point. Depth pixels
whose rays can reach the footprint column are back-projected to world points.
Points of the robot are excluded with the public geometry at the measured
state, as vertical prisms over the XY hull of each arm's measured finger boxes,
held box, controlled point and wrist camera: dilated ``ROBOT_CORE_MARGIN_M``
from the lowest finger or held corner upward, and ``ROBOT_MARGIN_M`` from the
lowest of the finger roots, controlled point and wrist camera upward, because
the descriptor bounds only the fingers and the housing, wrist and links rise
above them (offline over a recorded batch: with the jaw prism alone, 138 of 236 negative
readings on arrived moves were the moving arm's own housing, wrist camera or
links, all within 9 cm of link6). Of the remaining points within ``RADIUS_M``
horizontally of any footprint solid and no higher than the footprint's top, the
surface height is the ``MIN_SUPPORT_POINTS``-th highest. The result is
``unknown`` when no depth is valid, or fewer than ``MIN_SUPPORT_POINTS`` points
remain or less than ``MIN_OBSERVED_FRACTION`` of the area's 1 cm cells holds
one. Nothing here blocks or alters a move.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from ora.agent.toolkit.lib._clearance import (
    BELOW_M, CELL_M, CLAIM_BOUNDARY, DepthView, FingerBounds, Footprint, MAX_DEPTH_M, MAX_WINDOW_PIXELS,
    MIN_OBSERVED_FRACTION, MIN_SUPPORT_POINTS, RADIUS_M, REASONS, ROBOT_CORE_MARGIN_M, ROBOT_FLOOR_MARGIN_M,
    ROBOT_MARGIN_M, ROOT_FACE, RobotVolume, SURFACE_BAND_M, box_corners, footprint_at, measure, robot_volume,
    root_face, rotation,
)


def view_from_files(camera: str, depth_path: Any, calibration: Mapping[str, Any]) -> DepthView:
    """A depth view from a public observation's depth array and calibration record."""
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
    if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError(f"{camera} calibration shape mismatch")
    return DepthView(camera=camera, depth_m=np.load(depth_path, allow_pickle=False), intrinsic=intrinsic,
                     camera_to_world=np.linalg.inv(extrinsic))


__all__ = [
    "BELOW_M", "CELL_M", "CLAIM_BOUNDARY", "DepthView", "FingerBounds", "Footprint", "MAX_DEPTH_M",
    "MAX_WINDOW_PIXELS", "MIN_OBSERVED_FRACTION", "MIN_SUPPORT_POINTS", "RADIUS_M", "REASONS", "ROBOT_CORE_MARGIN_M",
    "ROBOT_FLOOR_MARGIN_M", "ROBOT_MARGIN_M", "ROOT_FACE", "RobotVolume", "SURFACE_BAND_M", "box_corners",
    "footprint_at", "measure", "robot_volume", "root_face", "rotation", "view_from_files",
]
