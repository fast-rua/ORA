"""Current and historical public-camera projection for dual-arm backends."""
from __future__ import annotations
from typing import Any, Mapping

#: The surfaces whose observe draws each arm's TCP on every view (``toolkit/tools/observe``).
DUAL_ARM_OBSERVE_CAPABILITIES = ("yam_rgbd_v1", "robotwin_rgbd_v1", "robodojo_rgbd_v1")
#: What a dual-arm observe draws on every view it returns (each arm's TCP and history points).
DUAL_ARM_OBSERVE_MARKS = ("Marks: L = left arm TCP, R = right arm TCP, each drawn only where it falls inside "
                          "this view; names from history_point_ids are drawn the same way.")


def observe_image_labels(surface: Any, name: str, result: Mapping[str, Any], count: int) -> list[str] | None:
    """One label per image of a dual-arm observe result: its place, its camera and what is drawn on it.

    The images follow ``views`` in order. A current observe draws each arm's
    TCP (``L``, ``R``) and the requested mark_point names; a historical one
    returns the saved images unmarked. None when the result's images cannot be
    named one to one (another tool, another surface, or a count that differs).
    """

    supports = getattr(surface, "supports", None)
    if name != "observe" or not callable(supports) or not any(supports(item) for item in DUAL_ARM_OBSERVE_CAPABILITIES):
        return None
    views = result.get("views")
    if not isinstance(views, list) or len(views) != count or not all(isinstance(view, str) for view in views):
        return None
    if result.get("status") == "actual_historical_observation":
        return [f"observe image {index} of {count}: {view} camera, the saved image of observation step "
                f"{result.get('observation_step')}; no marks drawn." for index, view in enumerate(views, start=1)]
    return [f"observe image {index} of {count}: {view} camera. {DUAL_ARM_OBSERVE_MARKS}"
            for index, view in enumerate(views, start=1)]
