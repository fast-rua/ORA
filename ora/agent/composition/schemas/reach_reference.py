"""The move-target reach pre-check that says which IK answered it.

``move_pose_reach_along_request_v1`` and ``reach_nearest_orientation_v1`` were
written for RoboDojo, whose reach check replayed the route ``move_pose`` took
(``matches_move_pose_admission: true`` while no shared planner is bound); their
schema and claim say so. The same field on another backend says less: RoboTwin's
check is the IK a global route plans through
(``environment/backends/robotwin/reach_ik.py``) and, like LIBERO's, plans no
trajectory, so both say false.
These generations carry the backend's own answer instead:

``move_pose_reach_along_request_v2``
    ``compiled_pose.targets.<arm>.reach_along_request`` of ``move_pose`` names
    the reach check's ``ik_model``, ``matches_move_pose_admission`` and the
    backend's claim boundary -- on a solved target too, where v1 said only
    ``true``.
``pose_primitive_reach_along_request_v2``
    the same field on ``move_to``, ``rotate_wrist`` and ``rotate_pitch``.
``reach_nearest_orientation_v2``
    the unsolved form also carries ``nearest_reachable_orientation`` (the v1
    shape), swept by the same reach check and holding the jaw centre the
    surface's controlled frame defines (``agent/toolkit/host/reach/jaw_center.py``: the
    controlled point itself on a grip-centre frame).

Each is its own capability, so every surface declaring a v1 name kept its
bytes; the fields are in each declaring surface's tool files. Runtime half: ``agent/toolkit/tools/move_pose/_reference.py`` through
``_reach``. The jaw-symmetry turn (``move_pose_jaw_symmetry_headroom_v1``)
stays an ARX X5 capability and is not part of this generation.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .move_pose_reach import REACH_PROPERTY
from ..surface.schema_validation import ToolSurfaceError

MOVE_POSE_CAPABILITY = "move_pose_reach_along_request_v2"
PRIMITIVE_CAPABILITY = "pose_primitive_reach_along_request_v2"
NEAREST_CAPABILITY = "reach_nearest_orientation_v2"
_REF = "ora/agent/composition/schemas/reach_reference.py"
NEAREST_CLAIM = (
    "Read-only IK at each candidate endpoint by the same reach check as reach_along_request (its ik_model "
    "and matches_move_pose_admission apply), from the measured joints: the requested orientation turned "
    "about axes perpendicular to its approach axis, holding the jaw centre; no path, collision, payload or "
    "arrival claim, and an orientation outside the sweep may solve."
)

BASIS_PROPERTIES = {
    "ik_model": {"type": ["string", "null"]},
    "matches_move_pose_admission": {"type": ["boolean", "null"]},
}


def _v1_variant(solved: bool) -> dict[str, Any]:
    return copy.deepcopy(next(item for item in REACH_PROPERTY["oneOf"]
                              if item.get("properties", {}).get("target_reachable") == {"const": solved}))


def _with_basis(variant: dict[str, Any]) -> dict[str, Any]:
    variant["properties"].update(copy.deepcopy(BASIS_PROPERTIES))
    variant["properties"].setdefault("claim_boundary", {"type": "string", "minLength": 1})
    variant["required"] = [*variant["required"],
                           *[key for key in (*BASIS_PROPERTIES, "claim_boundary") if key not in variant["required"]]]
    return variant


REACH_REFERENCE_PROPERTY = {
    "type": "object",
    "description": "Read-only reach pre-check of the submitted target by this backend's reach check (ik_model), "
                   "from the measured joints. matches_move_pose_admission says whether that check is the IK "
                   "move_pose admits through for this request; when it is not, the check is a reference only. "
                   "When the target does not solve, the straight interpolation from the measured pose is "
                   "sampled.",
    "oneOf": [
        _with_basis(_v1_variant(True)),
        _with_basis(_v1_variant(False)),
        copy.deepcopy(next(item for item in REACH_PROPERTY["oneOf"]
                           if item.get("properties", {}).get("status") == {"const": "unavailable"})),
    ],
}


def _error(message: str) -> Exception:
    return ToolSurfaceError(message)


def _descriptor(capability: str) -> dict[str, str]:
    return {"slot": f"derived:{capability}", "ref": _REF,
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _target_properties(results: Mapping[str, Any], name: str, capability: str) -> dict[str, Any]:
    result = results.get(name)
    compiled = result.get("properties", {}).get("compiled_pose") if isinstance(result, dict) else None
    if not isinstance(compiled, dict) or compiled.get("type") != "object":
        raise _error(f"{capability} requires the {name} compiled_pose result")
    targets = compiled.setdefault("properties", {}).setdefault("targets", {"type": "object"})
    return targets.setdefault("additionalProperties", {"type": "object"}).setdefault("properties", {})


__all__ = [
    "BASIS_PROPERTIES", "MOVE_POSE_CAPABILITY", "NEAREST_CAPABILITY", "NEAREST_CLAIM", "PRIMITIVE_CAPABILITY",
    "REACH_REFERENCE_PROPERTY",
]
