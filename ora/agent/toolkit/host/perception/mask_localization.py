"""Mask localization: the depth-inlier selection, centre, footprint and principal
frame measured from one mask, and the per-surface projection of what a policy
result keeps of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import overhead_view, target_support
from .target_store import LOCALIZATION_METHOD
from ora.agent.toolkit.lib._vision_measurement import (
    TARGET_PRINCIPAL_FRAME_CAPABILITY, _EXTENDED_FRAME_KEYS, _mask_depth_inlier_selection, _mask_summary,
    _principal_frame,
)


def _mask_depth_localization(
    *, mask: Any, depth_m: Any, calibration_path: Path,
    extended_frame: bool = False,
) -> dict[str, Any]:
    """Estimate bounded object geometry from public RGB-D measurements."""

    import numpy as np
    from ora.agent.toolkit.lib import _rigid

    selected = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
    if selected.shape != depth.shape or intrinsic.shape != (3, 3):
        raise ValueError("segmentation localization RGB-D calibration shape mismatch")
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
    if extrinsic.shape != (4, 4) or not (
        np.isfinite(intrinsic).all() and np.isfinite(extrinsic).all()
    ):
        raise ValueError("segmentation localization calibration is invalid")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("segmentation localization focal lengths must be positive")

    inlier_mask, selection = _mask_depth_inlier_selection(mask=selected, depth_m=depth)
    valid_count = selection["valid_depth_point_count"]
    mask_count = selection["mask_pixel_count"]
    primary_count = selection["primary_component_point_count"]

    base = {
        "status": "insufficient_depth",
        "method": LOCALIZATION_METHOD,
        "frame": "world",
        "scope": "visible_mask_depth",
        "mask_pixel_count": mask_count,
        "valid_depth_point_count": valid_count,
        "valid_depth_component_count": selection["valid_depth_component_count"],
        "inlier_point_count": 0,
        "depth_valid_fraction": float(valid_count / mask_count) if mask_count else 0.0,
        "primary_component_fraction": (
            float(primary_count / valid_count) if valid_count else 0.0
        ),
        "depth_inlier_fraction": 0.0,
        "center_xyz_m": None,
        "aabb_min_xyz_m": None,
        "aabb_max_xyz_m": None,
        "extent_xyz_m": None,
        "principal_axis_world": None,
    }
    if extended_frame:
        base.update(dict.fromkeys(_EXTENDED_FRAME_KEYS), **target_support.empty(), **overhead_view.empty())
    if primary_count < 16:
        return base

    rows, columns = np.nonzero(inlier_mask)
    if len(rows) < 16:
        return base
    z = depth[rows, columns]
    camera_points = np.stack([
        (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
        (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
        z,
        np.ones_like(z),
    ], axis=1)
    try:
        camera_to_world = _rigid.invert(extrinsic)
    except np.linalg.LinAlgError as exc:
        raise ValueError("segmentation localization extrinsic is singular") from exc
    world_points = (camera_to_world @ camera_points.T).T[:, :3]
    if not np.isfinite(world_points).all():
        raise ValueError("segmentation localization produced non-finite points")
    center = np.median(world_points, axis=0)
    low, high = np.percentile(world_points, [2.5, 97.5], axis=0)
    centered = world_points - center
    base.update({
        "status": "localized",
        "inlier_point_count": int(len(world_points)),
        "depth_inlier_fraction": float(len(world_points) / primary_count),
        "center_xyz_m": [float(value) for value in center],
        "aabb_min_xyz_m": [float(value) for value in low],
        "aabb_max_xyz_m": [float(value) for value in high],
        "extent_xyz_m": [float(value) for value in high - low],
        **_principal_frame(centered, extended=extended_frame),
        **(target_support.estimate(mask=selected, depth_m=depth, intrinsic=intrinsic, object_points=world_points,
                                   camera_to_world=camera_to_world) if extended_frame else {}),
        **(_footprint(inlier_mask, depth, intrinsic, camera_to_world, world_points) if extended_frame else {}),
    })
    return base


def _footprint(inlier_mask: Any, depth: Any, intrinsic: Any, camera_to_world: Any, world_points: Any) -> dict[str, Any]:
    """``overhead_view.footprint`` of the inlier mask eroded by one pixel (its edge carries background depth)."""
    import cv2
    import numpy as np

    core = cv2.erode(np.asarray(inlier_mask, dtype=np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    rows, columns = np.nonzero(core)
    points = world_points
    if len(rows) >= 16:
        z = depth[rows, columns]
        camera_points = np.stack([
            (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
            np.ones_like(z),
        ], axis=1)
        points = (camera_to_world @ camera_points.T).T[:, :3]
    return overhead_view.footprint(points, camera_to_world[:3, 3])


def project_target_frame(result: Any, *, declared: bool, support_declared: bool = False,
                         footprint_declared: bool = False) -> Any:
    """Keep or drop the extended principal frame, the support height and the footprint on a policy result.

    The live producers measure all three into the artifact; each surface receives only what it
    declares (``target_principal_frame_v1``, ``target_support_height_v1``,
    ``overhead_footprint_localization_v1``), so one producer serves surfaces written before any
    existed. The frozen v1 batch producer measures none.
    """

    if not isinstance(result, Mapping):
        return result
    keys = ((() if declared else _EXTENDED_FRAME_KEYS) + (() if support_declared else target_support.KEYS)
            + (() if footprint_declared else (*overhead_view.KEYS, overhead_view.POSE_KEY)))
    candidates = result.get("candidates") or ()
    holders = [result, *(item for item in candidates if isinstance(item, Mapping))]
    for holder in holders:
        localization = holder.get("localization")
        if isinstance(localization, dict):
            for key in keys:
                localization.pop(key, None)
    return result
