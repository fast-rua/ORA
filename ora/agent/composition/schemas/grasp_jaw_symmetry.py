"""The grasp_plan result fields the parallel-jaw member choice adds, as the runtime fills them.

Beside the pooled IK annotation (``grasp_plan_ik_reachability_annotation_v1``): each candidate's ``ik_reachability`` gains
``jaw_symmetry`` (which member of the parallel-jaw class the published poses
carry, why, and both members' joint6 margins) and ``joint_headroom`` (the
chosen member's joint-limit margins at its pregrasp and contact solutions),
and the result-level ``ik_reachability`` gains the rule and its claim
boundary. The runtime half is ``agent/toolkit/tools/grasp_plan/_jaw.py``.

The joint6 margin is an ARX X5 fact (wrist roll about the link6 approach
axis), so only a RoboDojo surface whose controlled frame is ``arx_x5_link6``
declares the capability; its grasp_plan file holds the fields.
"""
from __future__ import annotations


CAPABILITY = "grasp_plan_jaw_symmetry_headroom_v1"
RULE = "reachable_then_larger_joint6_margin_then_smaller_joint_delta_then_compiled"
VARIANTS = ["exact", "jaw_flip"]
STAGES = ("pregrasp", "approach")
REASONS = [
    "larger_joint6_margin", "larger_joint_limit_margin", "margin_tie_smaller_joint_delta",
    "margin_and_joint_delta_tie", "only_reachable_variant", "no_reachable_variant",
    "ik_reference_unavailable",
]
#: Margins or joint moves this close are one IK answer, not a preference.
MARGIN_TIE_RAD = 0.001
DELTA_TIE_RAD = 0.001
CLAIM_BOUNDARY = (
    "Joint-limit margins of the backend's read-only IK solutions for the published pregrasp "
    "and contact poses at planning time; joint6 is the X5 wrist roll about the approach axis "
    "and is reported only when the measured joints prove it is the binding joint. An IK "
    "reference, not a carry guarantee: execution solves again, and lifting, carrying or "
    "rotating the wrist after the grasp moves every margin."
)
_MARGIN = {"type": ["number", "null"], "minimum": 0}
