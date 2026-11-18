"""Footprint centre of a segmented target, and the wrist pose that views it from overhead.

``center_xyz_m`` is the median of the target's visible depth points. From an
oblique camera those points are the camera-facing side and top, so the median
sits horizontally toward the camera by up to half the object's depth: on the
LIBERO mugs it was 2.75 cm off the body origin (probe M, 2026-10-09), while a
stacking check such as LIBERO's ``On`` passes only within about 3 cm, so placing
by the median fails most of the time. The visible-point AABB midpoint is less
biased but is cut short by an occluding neighbour.

This module adds two things:

- ``footprint_center_xyz_m``: the centre of the occupied-cell box, in its own
  principal frame, of the target's points deprojected from its mask eroded by
  one pixel (boundary pixels carry background depth). From an overhead view the
  visible points are the whole top outline, and this centre was within 0.2 cm
  of the body origin on the LIBERO mugs (handles included) and bowls; from an
  oblique view it is still a near-side estimate. ``view_tilt_deg`` (the angle
  between the camera-to-target ray and straight down) and ``footprint_view``
  (``overhead`` at or below :data:`OVERHEAD_MAX_TILT_DEG`, else ``oblique``)
  say which case a measurement is.
- ``overhead_view_pose``: for an oblique measurement, the grip-site pose that
  puts the wrist camera :data:`VIEW_HEIGHT_M` above the footprint centre,
  looking straight down with the current jaw yaw. The camera-in-grip transform
  is read from the current observation's wrist calibration and grip-site pose,
  so no embodiment constant is assumed. It is a copyable ``move_pose`` target,
  not a reachability claim.

A scripted grasp-carry-place probe (probe R, 2026-10-09) placed 35/35 LIBERO
mugs and bowls within ``On`` with the overhead measurement and 9/35 with the
oblique median. Delivery is gated on :data:`CAPABILITY`: the producers measure
the fields, and :func:`ora.agent.toolkit.host.perception.mask_localization.project_target_frame`
removes them for surfaces that do not declare it.
"""

from __future__ import annotations

from ora.agent.toolkit.lib._vision_footprint import (
    CAPABILITY, KEYS, OVERHEAD_MAX_TILT_DEG, POSE_KEY, VIEW_HEIGHT_M, annotate, empty, footprint, view_pose,
)


__all__ = ["CAPABILITY", "KEYS", "OVERHEAD_MAX_TILT_DEG", "POSE_KEY", "VIEW_HEIGHT_M",
           "annotate", "empty", "footprint", "view_pose"]
