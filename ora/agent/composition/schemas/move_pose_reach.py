"""The move_pose result fields the planning-time IK reference adds.

``compiled_pose.targets.<arm>`` gains ``jaw_symmetry`` (which turn of the
parallel jaw was submitted, why, and both turns' joint6 margins) under
``move_pose_jaw_symmetry_headroom_v1`` and ``reach_along_request`` (whether
the target solved and, if not, how far along the straight interpolation the
same route solves) under ``move_pose_reach_along_request_v1``. The runtime
half is ``agent/toolkit/tools/move_pose/_reach.py``. Both belong to a surface whose
move_pose compiles through ``enhanced_move_pose_v1``; the joint6 margin is an
ARX X5 fact, so the jaw turn only to the ``arx_x5_link6`` frame. The fields are
in each declaring surface's tool files.
"""
from __future__ import annotations

import copy


JAW_CAPABILITY = "move_pose_jaw_symmetry_headroom_v1"
REACH_CAPABILITY = "move_pose_reach_along_request_v1"
JAW_REASONS = (
    "requested_turn_margin_sufficient", "larger_joint6_margin", "requested_turn_kept",
    "only_reachable_variant", "no_reachable_variant", "ik_reference_unavailable",
)
#: Fractions of the requested motion sampled when its target does not solve.
LINE_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)

_NUMBER = {"type": ["number", "null"]}
_MARGIN = {"type": ["number", "null"], "minimum": 0}
_POSE = {
    "type": ["object", "null"], "additionalProperties": False, "required": ["position_m", "orientation_xyzw"],
    "properties": {"position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                   "orientation_xyzw": {"type": "array", "items": {"type": "number"}, "minItems": 4,
                                        "maxItems": 4}},
}
REACH_PROPERTY = {
    "type": "object",
    "description": "Read-only IK of the submitted target by the planner move_pose uses; when it does not solve, "
                   "the straight interpolation from the measured pose is sampled.",
    "oneOf": [
        {"additionalProperties": False, "required": ["target_reachable"],
         "properties": {"target_reachable": {"const": True}}},
        {"additionalProperties": False,
         "required": ["target_reachable", "reason", "requested_translation_m", "requested_rotation_rad",
                      "farthest_reachable_fraction", "farthest_reachable_translation_m",
                      "farthest_reachable_pose", "samples", "claim_boundary"],
         "properties": {
             "target_reachable": {"const": False},
             "reason": {"type": ["string", "null"]},
             "requested_translation_m": dict(_NUMBER), "requested_rotation_rad": dict(_NUMBER),
             "farthest_reachable_fraction": {"type": "number", "enum": [0.0, *LINE_FRACTIONS]},
             "farthest_reachable_translation_m": dict(_NUMBER),
             "farthest_reachable_pose": copy.deepcopy(_POSE),
             "samples": {"type": "array", "minItems": len(LINE_FRACTIONS), "maxItems": len(LINE_FRACTIONS),
                         "items": {"type": "object", "additionalProperties": False,
                                   "required": ["fraction", "reachable", "reason"],
                                   "properties": {"fraction": {"type": "number"}, "reachable": {"type": "boolean"},
                                                  "reason": {"type": ["string", "null"]}}}},
             "claim_boundary": {"type": "string", "minLength": 1},
         }},
        {"additionalProperties": False, "required": ["status", "failure"],
         "properties": {"status": {"const": "unavailable"}, "failure": {"type": "string"}}},
    ],
}


__all__ = [
    "JAW_CAPABILITY", "JAW_REASONS", "LINE_FRACTIONS", "REACH_CAPABILITY", "REACH_PROPERTY",
]
