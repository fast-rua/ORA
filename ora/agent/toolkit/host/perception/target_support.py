"""Support-aware height of a segmented target, from one RGB-D view.

``center_xyz_m`` is the median of the target's visible depth points. From an
oblique or top-down camera those points are mostly the object's top and
camera-facing side, so the centre sits well above the object's middle, and a
parallel-jaw side grasp aimed at it closes near the top edge. Nothing the policy
receives says where the object stands.

This module estimates the local support plane beneath the target from the depth
pixels in a ring just outside its mask, evaluates that plane under the target,
and reports the support height, the height of the visible top above it and the
midpoint between the two. All three are single-view estimates: the ring can see
a neighbouring object, the far side of a container or nothing but background,
and the visible top is the highest *visible* surface, not the object's true top.
They are labelled as such and never presented as ground truth.

Delivery is gated on :data:`CAPABILITY`. The live producers measure the fields
beside the principal frame, and :func:`ora.agent.toolkit.host.perception.mask_localization.
project_target_frame` removes them for surfaces that do not declare it, whose
closed localization schemas would otherwise reject the extra properties.
"""

from __future__ import annotations

from typing import Any
from ora.agent.toolkit.lib._vision_support import BASIS, CAPABILITY, KEYS, empty, estimate


def projection(surface: Any, principal_frame_capability: str) -> dict[str, bool]:
    """Which measured target fields a surface receives, for ``project_target_frame``."""
    return {"declared": bool(surface.supports(principal_frame_capability)),
            "support_declared": bool(surface.supports(CAPABILITY))}


__all__ = ["BASIS", "CAPABILITY", "KEYS", "empty", "estimate", "projection"]
