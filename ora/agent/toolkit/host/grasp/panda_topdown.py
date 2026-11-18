"""Public-RGB-D top-down grasp candidates for the Panda grip site, as a fallback.

LIBERO proposed with Contact-GraspNet alone, and 29 of 222 recorded ``grasp_plan``
calls got no candidate at all: the network returned none, nothing was rejected,
and the policy had to aim its own close, which held the object about half the
time against 0.82 for a followed candidate. This adds the family ALOHA and the
X5 already have, as a fallback only: top-down candidates are shown when the
network returned none, and never beside learned ones, so a call the network
answers is unchanged. Recorded RoboTwin runs give no reason to prefer top-down:
where both were tried on one task, top-down held less often.

The height rule is the X5's support-aware one (``x5_topdown.py``), from the
target mask and the scene depth around it. The grip site is the centre of the
fingertip pads (``compiler.py``), so the contact centre is the controlled point
and no baseline is subtracted; the fingertips lie 6.4 mm beyond it, the Panda
identity's ``controlled_point_to_contact_baseline_m``. The pre-grasp is the 40 mm
stand-off Contact-GraspNet candidates use on this surface, inside the bounded
local approach. Neither family sees simulator ground truth.
"""
from __future__ import annotations

from typing import Any

from ora.agent.toolkit.host.grasp.compiler import PANDA_GRIP_SITE_COMPILER_IDENTITY
from ora.agent.toolkit.host.grasp.x5_topdown import HEIGHT_RULE

CAPABILITY = "panda_public_rgbd_topdown_fallback_v1"
HYBRID_BACKEND_ID = "contact_graspnet_then_panda_rgbd_topdown"
PROFILE = {
    "height_rule": HEIGHT_RULE,
    "surface_percentile": 95.0,
    "top_clearance_m": 0.02,
    "table_clearance_m": 0.005,
    "fingertip_beyond_contact_centre_m": float(
        PANDA_GRIP_SITE_COMPILER_IDENTITY["controlled_point_to_contact_baseline_m"]),
    "pregrasp_standoff_m": 0.04,
    "maximum_candidates": 2,
}


def hybrid_backend_arguments(surface: Any) -> dict[str, Any] | None:
    """Constructor overrides for the fallback family a surface selects, else None."""
    if not surface.supports(CAPABILITY):
        return None
    from ora.agent.toolkit.host.grasp.proposers import CONTACT_FIRST, PcaTopDownGraspProposer

    return {"backend_id": HYBRID_BACKEND_ID, "selection": CONTACT_FIRST,
            "topdown_backend": PcaTopDownGraspProposer(topdown_profile=PROFILE)}
