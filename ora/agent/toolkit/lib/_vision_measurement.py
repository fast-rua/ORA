"""Mask localization: the depth-inlier selection, centre, footprint and principal
frame measured from one mask, and the per-surface projection of what a policy
result keeps of them.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import _vision_footprint as overhead_view, _vision_support as target_support
DEPTH_INLIER_SELECTION_METHOD = "valid_depth_primary_component_mad_v1"
LOCALIZATION_METHOD = "mask_depth_primary_component_robust_pca_v1"


def _mask_summary(mask: Any, depth_m: Any) -> dict[str, Any]:
    import numpy as np

    value = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    rows, columns = np.nonzero(value)
    if not len(rows):
        raise ValueError("segmentation mask is empty")
    selected = depth[value]
    valid = selected[
        np.isfinite(selected) & (selected > 1e-6) & (selected <= 2.5)
    ]
    summary: dict[str, Any] = {
        "pixel_count": int(len(rows)),
        "area_fraction": float(len(rows) / value.size),
        "bbox_xyxy": [
            int(columns.min()), int(rows.min()),
            int(columns.max()) + 1, int(rows.max()) + 1,
        ],
        "centroid_uv": [float(np.mean(columns)), float(np.mean(rows))],
        "depth_valid_fraction": float(len(valid) / len(selected)),
        "depth_m": None,
    }
    if len(valid):
        summary["depth_m"] = {
            "min": float(np.min(valid)),
            "median": float(np.median(valid)),
            "max": float(np.max(valid)),
        }
    return summary


def _mask_depth_inlier_selection(
    *, mask: Any, depth_m: Any,
) -> tuple[Any, dict[str, Any]]:
    """Share the existing localization input selection without trimming XYZ.

    A connected handle within the depth window remains part of the source.
    This is measurement selection, never object identity or completed shape.
    """

    import cv2
    import numpy as np

    selected = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if selected.ndim != 2 or not selected.size or selected.shape != depth.shape:
        raise ValueError("segmentation mask and depth must have matching image dimensions")
    valid = selected & np.isfinite(depth) & (depth > 1e-6) & (depth <= 2.5)
    valid_count = int(np.count_nonzero(valid))
    mask_count = int(np.count_nonzero(selected))
    component_count, labels = cv2.connectedComponents(
        valid.astype(np.uint8), connectivity=8,
    )
    component_sizes = np.bincount(labels.ravel(), minlength=component_count)
    if len(component_sizes) > 1:
        component_sizes[0] = 0
        primary_label = int(np.argmax(component_sizes))
        primary = labels == primary_label
        primary_count = int(component_sizes[primary_label])
    else:
        primary = np.zeros_like(valid)
        primary_count = 0
    median_depth = mad_depth = depth_half_width = None
    inlier_mask = np.zeros_like(valid)
    if primary_count >= 16:
        primary_depth = depth[primary]
        median_depth = float(np.median(primary_depth))
        mad_depth = float(np.median(np.abs(primary_depth - median_depth)))
        depth_half_width = max(0.03, 4.0 * 1.4826 * mad_depth)
        inlier_mask = primary & (np.abs(depth - median_depth) <= depth_half_width)
    retained_count = int(np.count_nonzero(inlier_mask))
    return inlier_mask, {
        "method": DEPTH_INLIER_SELECTION_METHOD,
        "mask_pixel_count": mask_count,
        "valid_depth_point_count": valid_count,
        "valid_depth_component_count": max(0, component_count - 1),
        "primary_component_point_count": primary_count,
        "inlier_point_count": retained_count,
        "depth_valid_fraction": float(valid_count / mask_count) if mask_count else 0.0,
        "primary_component_fraction": float(primary_count / valid_count) if valid_count else 0.0,
        "depth_inlier_fraction": float(retained_count / primary_count) if primary_count else 0.0,
        "median_depth_m": median_depth,
        "median_absolute_deviation_m": mad_depth,
        "depth_half_width_m": depth_half_width,
        "valid_depth_range_m": [1e-6, 2.5],
        "connectivity": 8,
        "minimum_component_points": 16,
        "minimum_depth_half_width_m": 0.03,
        "mad_scale": 4.0 * 1.4826,
        "touches_image_boundary": bool(selected[0].any() or selected[-1].any()
                                       or selected[:, 0].any() or selected[:, -1].any()),
    }


def _mask_depth_localization(
    *, mask: Any, depth_m: Any, calibration: Mapping[str, Any],
    extended_frame: bool = False,
) -> dict[str, Any]:
    """Estimate bounded object geometry from public RGB-D measurements."""

    import numpy as np

    from . import _rigid

    selected = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
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


def _principal_frame(centered_points: Any, *, extended: bool) -> dict[str, Any]:
    """The target's measured principal frame, with each axis beside its extent.

    The eigendecomposition was already being run and two of its three vectors
    thrown away, so the policy could read a target's position and size but
    never its orientation. Three tasks turn on exactly that and stand at zero
    across every cohort: a QR board faces the way its least-extent axis points,
    a fan's facing needs two axes to disambiguate, and an upright bottle is its
    greatest-extent axis.

    ``principal_extents_m`` keeps eigenvalue order, because that is what pairs
    each extent with ``principal_axes_world[i]``. Eigenvalue order is variance
    order and variance order is not size order -- a real target measured
    [0.1034, 0.0704, 0.0751] -- so the named largest/middle/smallest come from
    the sizes and each carries the axis that goes with it. Extents are the same
    2.5/97.5 percentile spread the centre and AABB already use, measured along
    each axis rather than along the world axes the AABB happens to use.

    Each axis is sign-normalised by its dominant component, so the same target
    measured twice reports the same frame rather than an arbitrary flip.
    """

    import numpy as np

    from ._principal_axes import principal_axes, scatter

    covariance = scatter(centered_points) / max(1, len(centered_points) - 1)
    _variances, eigenvectors = principal_axes(covariance)
    # Keep the released single-axis meaning: maximum variance, not maximum
    # percentile extent. Sparse point clouds can order these differently.
    principal = eigenvectors[:, 0]
    if principal[int(np.argmax(np.abs(principal)))] < 0:
        principal = -principal
    legacy_axis = [float(value) for value in principal]
    if not extended:
        return {"principal_axis_world": legacy_axis}
    axes, extents = [], []
    for index in range(3):
        axis = eigenvectors[:, index]
        sign_index = int(np.argmax(np.abs(axis)))
        if axis[sign_index] < 0:
            axis = -axis
        projection = centered_points @ axis
        low, high = np.percentile(projection, [2.5, 97.5])
        axes.append([float(value) for value in axis])
        extents.append(float(high - low))
    ranked = sorted(range(3), key=lambda i: extents[i], reverse=True)
    largest, middle, smallest = ranked
    return {
        "principal_axes_world": axes,
        "principal_extents_m": extents,
        "largest_extent_m": extents[largest],
        "middle_extent_m": extents[middle],
        "smallest_extent_m": extents[smallest],
        "largest_extent_axis_world": axes[largest],
        "middle_extent_axis_world": axes[middle],
        "smallest_extent_axis_world": axes[smallest],
        # The single axis older consumers read, unchanged in meaning.
        "principal_axis_world": legacy_axis,
    }


#: Surfaces declaring this receive the target's full measured principal frame.
#: Older and frozen surfaces keep the single axis they were written against;
#: their result schemas forbid unknown properties, and the producer is shared,
#: so the extra measurements are removed on the way out rather than never taken.
TARGET_PRINCIPAL_FRAME_CAPABILITY = "target_principal_frame_v1"

#: Measured by :func:`_principal_frame`, delivered only under the capability.
_EXTENDED_FRAME_KEYS = (
    "principal_axes_world", "principal_extents_m", "largest_extent_m", "middle_extent_m",
    "smallest_extent_m", "largest_extent_axis_world", "middle_extent_axis_world", "smallest_extent_axis_world",
)
