"""Tool-local support-aware top-down height policy, derived only from public clouds."""
from __future__ import annotations
from typing import Any, Mapping
HEIGHT_RULE = 'support_aware_mid_height_v1'
PROFILE = dict.fromkeys(('height_rule', 'surface_percentile', 'top_clearance_m', 'table_clearance_m',
                         'fingertip_beyond_contact_centre_m', 'pregrasp_standoff_m', 'maximum_candidates'))
SUPPORT_RING_M = (.01, .06)
SUPPORT_MINIMUM_POINTS = 16
SUPPORT_PERCENTILE = 30.

def support_plane_z(target_world: Any, scene_world: Any) -> float | None:
    """Support height around the target footprint from public scene depth, or None."""
    import numpy as np

    target = np.asarray(target_world, dtype=np.float64)
    scene = np.asarray(scene_world, dtype=np.float64).reshape(-1, 3)
    scene = scene[np.isfinite(scene).all(axis=1)]
    if len(target) < 3 or len(scene) < SUPPORT_MINIMUM_POINTS:
        return None
    low = np.percentile(target[:, :2], 2, axis=0)
    high = np.percentile(target[:, :2], 98, axis=0)
    outside = np.maximum(np.maximum(low - scene[:, :2], 0.0), np.maximum(scene[:, :2] - high, 0.0))
    distance = np.linalg.norm(outside, axis=1)
    top = float(np.percentile(target[:, 2], 95))
    ring = (distance > SUPPORT_RING_M[0]) & (distance <= SUPPORT_RING_M[1]) & (scene[:, 2] <= top)
    if int(np.count_nonzero(ring)) < SUPPORT_MINIMUM_POINTS:
        return None
    # The support cannot lie above the lowest visible part of the target.
    return min(float(np.percentile(scene[ring, 2], SUPPORT_PERCENTILE)),
               float(np.percentile(target[:, 2], 5)))

def topdown_contact_height(target_world: Any, scene_points_camera_m: Any, camera_to_world: Any,
                           profile: Mapping[str, Any]) -> dict[str, Any]:
    """The jaw contact-centre height and the evidence it was derived from.

    ``target_world`` is the proposer's filtered target cloud in world metres;
    the scene cloud arrives in the target camera frame with its transform.
    """
    import numpy as np

    if (not isinstance(profile, Mapping) or set(profile) != set(PROFILE)
            or profile["height_rule"] != HEIGHT_RULE):
        raise ValueError("X5 public RGB-D top-down profile is invalid")
    scene = np.asarray(scene_points_camera_m, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    scene_world = (transform @ np.column_stack([scene, np.ones(len(scene))]).T).T[:, :3]
    target = np.asarray(target_world, dtype=np.float64)
    top = float(np.percentile(target[:, 2], float(profile["surface_percentile"])))
    support = support_plane_z(target, scene_world)
    if support is None:
        mid = float((np.percentile(target[:, 2], 5) + np.percentile(target[:, 2], 95)) / 2)
        floor = None
    else:
        mid = (support + top) / 2
        floor = support + float(profile["fingertip_beyond_contact_centre_m"]) + float(
            profile["table_clearance_m"])
    centre = max(mid, top - float(profile["top_clearance_m"]))
    if floor is not None:
        centre = max(centre, floor)
    fingertip = centre - float(profile["fingertip_beyond_contact_centre_m"])
    return {
        "height_rule": HEIGHT_RULE,
        "contact_centre_z_m": centre,
        "robust_top_z_m": top,
        "support_z_m": support,
        "support_source": "scene_depth_ring" if support is not None else "unavailable_target_extent_only",
        "mid_height_z_m": mid,
        "fingertip_z_m": fingertip,
        # Fingertips at or above the robust top close on air.
        "reaches_object": fingertip < top,
    }
