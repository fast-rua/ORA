"""Measurements of a segmented target: its mask on a depth image, to a size, an orientation and a height.

``object_geometry(target)`` measures a target the script got from ``vision_ground`` or ``vision_segment``:
the extent, principal axes, support height, height above support and mid height of the visible part of the
object, in the world frame, in metres. It is the arithmetic the tools' ``localization`` states
(``ora/agent/toolkit/lib/vision.py``: ``localize_mask``, ``mask_summary``; ``geometry.mask_to_cloud``,
``geometry.principal_geometry``), written for the sandbox, which has NumPy and SciPy but no OpenCV and no ORA
package; ``tests/ora/agent/code_mode/ora_lib`` holds it to the lib on recorded observations.

    candidate = tools.vision_ground(camera="head", text="red cup")["result"]["candidates"][0]
    cup = object_geometry(candidate)            # or object_geometry(candidate["target_ref"])
    cup.extent_xyz_m, cup.height_above_support_m, cup.mid_height_z_m
    grasp_height = cup.support_z_m + 0.5 * cup.height_above_support_m

A measurement is of what the camera saw: the visible surface, not the object's whole shape. The support height is
a single-view estimate from the depth just outside the mask (``support_height_basis``), null where no support is
visible. Nothing is chosen here: which candidate is the object is the caller's judgement.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from . import geometry
from . import observation as _observation
from .frames import Direction, Point

LOCALIZATION_METHOD = "mask_depth_primary_component_robust_pca_v1"
SUPPORT_BASIS = "single_view_depth_ring_estimate_not_ground_truth"
#: The least valid points a localization needs (``geometry.depth_inliers``'s ``minimum_points``).
MINIMUM_POINTS = 16
#: A support more than this far below the visible top is not this target's.
_MAXIMUM_HEIGHT_M = 0.35
#: A support plane tilted more than this is not a surface the target stands on.
_MAXIMUM_TILT_RAD = 0.35
_MINIMUM_SUPPORT_POINTS = 12
_MINIMUM_RESIDUAL_M = 0.004
_MAXIMUM_RING_PX = 60

_PRINCIPAL_KEYS = ("principal_axes_world", "principal_extents_m", "largest_extent_m", "middle_extent_m",
                   "smallest_extent_m", "largest_extent_axis_world", "middle_extent_axis_world",
                   "smallest_extent_axis_world")
_SUPPORT_KEYS = ("support_z_m", "height_above_support_m", "mid_height_z_m")


def mask_summary(mask, depth_m) -> dict[str, Any]:
    """The mask's pixel extent and valid-depth statistics, as ``vision.mask_summary`` states them.

    ``pixel_count``, ``area_fraction``, ``bbox_xyxy`` (pixels, ``x1``/``y1`` exclusive), ``centroid_uv``,
    ``depth_valid_fraction`` and ``depth_m`` (``min``, ``median``, ``max`` in metres over the pixels with depth in
    ``(1e-6, 2.5]``, else None). An empty mask raises ``ValueError``.
    """

    value = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    rows, columns = np.nonzero(value)
    if not len(rows):
        raise ValueError("segmentation mask is empty")
    selected = depth[value]
    valid = selected[np.isfinite(selected) & (selected > 1e-6) & (selected <= 2.5)]
    summary: dict[str, Any] = {
        "pixel_count": int(len(rows)),
        "area_fraction": float(len(rows) / value.size),
        "bbox_xyxy": [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1],
        "centroid_uv": [float(np.mean(columns)), float(np.mean(rows))],
        "depth_valid_fraction": float(len(valid) / len(selected)),
        "depth_m": None,
    }
    if len(valid):
        summary["depth_m"] = {"min": float(np.min(valid)), "median": float(np.median(valid)),
                              "max": float(np.max(valid))}
    return summary


def _ring(mask: np.ndarray) -> np.ndarray:
    """Pixels a short distance outside the mask, skipping its mixed-depth edge."""

    radius = math.sqrt(max(1, int(mask.sum())) / math.pi)
    inner = max(2, round(0.15 * radius))
    outer = min(_MAXIMUM_RING_PX, inner + max(4, round(0.6 * radius)))
    return geometry.dilate_disc(mask, outer) & ~geometry.dilate_disc(mask, inner)


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Robust ``z = a x + b y + c`` over the ring points, or None."""

    inliers = np.ones(len(points), dtype=bool)
    coefficients = None
    for _ in range(4):
        if int(inliers.sum()) < _MINIMUM_SUPPORT_POINTS:
            return None
        chosen = points[inliers]
        design = np.column_stack([chosen[:, 0], chosen[:, 1], np.ones(len(chosen))])
        coefficients, *_ = np.linalg.lstsq(design, chosen[:, 2], rcond=None)
        residual = points[:, 2] - (points[:, 0] * coefficients[0] + points[:, 1] * coefficients[1] + coefficients[2])
        spread = float(np.median(np.abs(residual[inliers] - np.median(residual[inliers]))))
        inliers = np.abs(residual) <= max(_MINIMUM_RESIDUAL_M, 3.0 * 1.4826 * spread)
    if coefficients is None or int(inliers.sum()) < _MINIMUM_SUPPORT_POINTS:
        return None
    return coefficients, inliers


def _support(mask: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, camera_to_world: np.ndarray,
             points: np.ndarray) -> dict[str, Any]:
    """The support plane under the target, fitted to the depth pixels in a ring outside its mask and evaluated under
    it (``target_support.estimate``): ``support_z_m``, ``height_above_support_m`` (the visible top, the 97.5th
    percentile height of ``points``, above it), ``mid_height_z_m``, ``support_point_count``."""

    result: dict[str, Any] = {"support_z_m": None, "height_above_support_m": None, "mid_height_z_m": None,
                              "support_point_count": 0, "support_height_basis": SUPPORT_BASIS}
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < _MINIMUM_SUPPORT_POINTS:
        return result
    ring = _ring(mask) & np.isfinite(depth) & (depth > 1e-6) & (depth < 2.5)
    rows, columns = np.nonzero(ring)
    if len(rows) < _MINIMUM_SUPPORT_POINTS:
        return result
    z = depth[rows, columns]
    camera = np.stack([(columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
                       (rows - intrinsic[1, 2]) * z / intrinsic[1, 1], z, np.ones_like(z)], axis=1)
    world = (camera_to_world @ camera.T).T[:, :3]
    top = float(np.percentile(points[:, 2], 97.5))
    middle = float(np.median(points[:, 2]))
    # The support lies below the target's visible surface and within a tabletop object's height of its top.
    beneath = world[(world[:, 2] < middle) & (world[:, 2] > top - _MAXIMUM_HEIGHT_M)]
    fitted = _fit_plane(beneath) if len(beneath) >= _MINIMUM_SUPPORT_POINTS else None
    if fitted is None:
        return result
    (a, b, c), inliers = fitted
    if math.atan(math.hypot(float(a), float(b))) > _MAXIMUM_TILT_RAD:
        result["support_point_count"] = int(inliers.sum())
        return result
    x, y = (float(value) for value in np.median(points[:, :2], axis=0))
    support = float(a * x + b * y + c)
    if not math.isfinite(support) or support >= top:
        return result
    result.update(support_z_m=support, height_above_support_m=top - support, mid_height_z_m=0.5 * (support + top),
                  support_point_count=int(inliers.sum()))
    return result


def localize_mask(mask, depth_m, intrinsic, camera_to_world) -> dict[str, Any]:
    """The geometry of a mask's visible surface in the world frame, as ``vision.localize_mask`` states it.

    ``mask`` and ``depth_m`` are HxW (metres along the optical axis), ``intrinsic`` the camera's 3x3 ``K`` and
    ``camera_to_world`` the 4x4 ``world <- camera`` transform. The mask's valid pixels are cut to the largest
    8-connected component and its pixels within the depth gate (``geometry.depth_inliers``), lifted to the world
    (``geometry.mask_to_cloud``, ``geometry.transform_points``) and measured (``geometry.principal_geometry``).
    Returns the lib's fields: ``status`` (``localized``, or ``insufficient_depth`` with the fields null), the pixel
    and point counts and fractions, ``center_xyz_m`` (the median), ``aabb_min_xyz_m`` and ``aabb_max_xyz_m`` (the
    2.5th and 97.5th percentiles per world axis), ``extent_xyz_m``, ``principal_axes_world`` (unit rows ordered by
    variance, each signed by its largest component), ``principal_extents_m`` (the percentile spread along each),
    ``largest_/middle_/smallest_extent_m`` with their ``..._extent_axis_world``, and ``support_z_m``,
    ``height_above_support_m``, ``mid_height_z_m`` (see ``_support``). The overhead footprint fields of the lib are
    not computed. A malformed calibration raises ``ValueError``.
    """

    selected = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(intrinsic, dtype=np.float64)
    if selected.shape != depth.shape or k.shape != (3, 3):
        raise ValueError("the mask, the depth and the intrinsic matrix do not match")
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("the intrinsic matrix must be finite with positive focal lengths")
    to_world = geometry.rigid(camera_to_world)
    selection = geometry.depth_inliers(selected, depth)
    mask_count, valid_count = selection["mask_pixels"], selection["valid_pixels"]
    primary_count = selection["primary_pixels"]
    result: dict[str, Any] = {
        "status": "insufficient_depth", "method": LOCALIZATION_METHOD, "frame": "world",
        "scope": "visible_mask_depth", "mask_pixel_count": mask_count, "valid_depth_point_count": valid_count,
        "valid_depth_component_count": selection["components"], "inlier_point_count": 0,
        "depth_valid_fraction": float(valid_count / mask_count) if mask_count else 0.0,
        "primary_component_fraction": float(primary_count / valid_count) if valid_count else 0.0,
        "depth_inlier_fraction": 0.0, "center_xyz_m": None, "aabb_min_xyz_m": None, "aabb_max_xyz_m": None,
        "extent_xyz_m": None, "principal_axis_world": None, **dict.fromkeys(_PRINCIPAL_KEYS),
        "support_z_m": None, "height_above_support_m": None, "mid_height_z_m": None, "support_point_count": 0,
        "support_height_basis": SUPPORT_BASIS,
    }
    if primary_count < MINIMUM_POINTS or selection["inlier_pixels"] < MINIMUM_POINTS:
        return result
    points = geometry.transform_points(geometry.mask_to_cloud(depth, k, selection["mask"]), to_world)
    if not np.isfinite(points).all():
        raise ValueError("the mask lifts to non-finite world points")
    shape = geometry.principal_geometry(points)
    axes, extents = shape["axes"], shape["extents_m"]
    largest, middle, smallest = sorted(range(3), key=lambda i: float(extents[i]), reverse=True)
    result.update({
        "status": "localized", "inlier_point_count": int(len(points)),
        "depth_inlier_fraction": float(len(points) / primary_count),
        "center_xyz_m": shape["center_m"].tolist(), "aabb_min_xyz_m": shape["lower_m"].tolist(),
        "aabb_max_xyz_m": shape["upper_m"].tolist(), "extent_xyz_m": (shape["upper_m"] - shape["lower_m"]).tolist(),
        "principal_axis_world": axes[0].tolist(), "principal_axes_world": axes.tolist(),
        "principal_extents_m": extents.tolist(), "largest_extent_m": float(extents[largest]),
        "middle_extent_m": float(extents[middle]), "smallest_extent_m": float(extents[smallest]),
        "largest_extent_axis_world": axes[largest].tolist(), "middle_extent_axis_world": axes[middle].tolist(),
        "smallest_extent_axis_world": axes[smallest].tolist(),
        **_support(selected, depth, k, to_world, points),
    })
    return result


def _field(name: str, kind: str) -> property:
    def read(self: "ObjectGeometry") -> Any:
        value = self.get(name)
        return None if value is None else (Point if kind == "point" else Direction)(value, "world")

    read.__doc__ = f"``{name}`` as a {kind} in ``world`` (None when the target was not localized)."
    return property(read)


class ObjectGeometry(dict):
    """A target's measured geometry (``object_geometry``): the fields of ``localize_mask`` as a dict, also readable
    as attributes (``g.extent_xyz_m``), plus ``target_ref``, ``camera`` and ``observation_id``.

    ``status`` is ``localized`` or ``insufficient_depth`` (every measurement None: read the counts). Typed values:
    ``center``, ``lower``, ``upper`` (a ``Point`` in ``world``), and ``axes`` (the three principal axes),
    ``largest_axis``, ``middle_axis`` and ``smallest_axis`` (a ``Direction`` in ``world``); ``largest_extent_m``
    pairs with ``largest_axis``. ``height_above_support_m`` is the visible top above the estimated support, so an
    object hidden behind another or seen from above reads low or None.
    """

    center = _field("center_xyz_m", "point")
    lower = _field("aabb_min_xyz_m", "point")
    upper = _field("aabb_max_xyz_m", "point")
    axes = _field("principal_axes_world", "direction")
    largest_axis = _field("largest_extent_axis_world", "direction")
    middle_axis = _field("middle_extent_axis_world", "direction")
    smallest_axis = _field("smallest_extent_axis_world", "direction")

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __repr__(self) -> str:
        name = f"ObjectGeometry({self.get('target_ref')}, camera={self.get('camera')}"
        if self.get("status") != "localized":
            return f"{name}, {self.get('status')}: {self['valid_depth_point_count']} valid depth points)"
        shown = [f"{key}={round(self[key], 4)}" for key in _SUPPORT_KEYS if self.get(key) is not None]
        return f"{name}, extent_xyz_m={[round(value, 4) for value in self['extent_xyz_m']]}" \
               + "".join(f", {item}" for item in shown) + ")"


def _target_ref(target: Any) -> str:
    if isinstance(target, str):
        return target
    if isinstance(target, Mapping) and isinstance(target.get("target_ref"), str):
        return target["target_ref"]
    if not isinstance(target, Mapping) and isinstance(getattr(target, "target_ref", None), str):
        return target.target_ref  # the observation's own Target (``Observation.target``)
    if isinstance(target, Mapping) and isinstance(target.get("result"), Mapping):
        raise TypeError("object_geometry takes one target: a candidate such as r['result']['candidates'][0], or its "
                        "target_ref")
    raise TypeError(f"object_geometry takes a target_ref or a candidate with one, not {type(target).__name__}")


def object_geometry(target, *, observation: _observation.Observation | None = None) -> ObjectGeometry:
    """The geometry of ``target``'s visible surface: its extent, principal axes, support height, height above
    support and mid height (see ``localize_mask``), in the world frame, in metres.

    ``target`` is a candidate of a ``vision_ground`` or ``vision_segment`` result, its ``target_ref``, or the
    observation's ``Target`` of it (``current().target(target_ref)``): the mask is
    the one that tool recorded, on the camera it segmented, measured on that observation's depth. A target belongs
    to the observation it was segmented on; after an action, segment again. Raises ``KeyError`` for a target this
    observation has no mask of (the message lists those it has), ``ObservationUnavailable`` without an observation.
    """

    reference = _target_ref(target)
    scene = observation if observation is not None else _observation.current()
    found = scene.target(reference)
    camera = scene.camera(found.camera)
    if found.mask.shape != camera.depth_m.shape:
        raise ValueError(f"the mask of {reference} is {found.mask.shape}, but the {camera.name} depth is "
                         f"{camera.depth_m.shape}")
    measured = localize_mask(found.mask, camera.depth_m, camera.K, camera.camera_to_world)
    return ObjectGeometry(measured, target_ref=reference, camera=camera.name, observation_id=scene.observation_id)


__all__ = ["LOCALIZATION_METHOD", "MINIMUM_POINTS", "SUPPORT_BASIS", "ObjectGeometry", "localize_mask",
           "mask_summary", "object_geometry"]
