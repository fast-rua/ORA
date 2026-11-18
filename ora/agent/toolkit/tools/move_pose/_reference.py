"""The v2 reach pre-check: every annotation names the IK that answered it.

``_reach`` asks the read-only reach check about each submitted target, as under the v1 capabilities; on a surface
declaring a v2 generation this module shapes what the policy is told. Every annotation names the IK that answered
(``ik_model``), whether that IK is ``move_pose``'s own admission for this request (``matches_move_pose_admission``, from
the backend) and the backend's claim boundary. A solved target is still only that check's answer and an unsolved one is
not proof of anything. The submitted action never changes and a reference that cannot be taken never blocks the move.
The result schemas that carry these fields are ``composition/schemas/reach_reference.py``.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import _orientation as reach_orientation

MOVE_POSE_CAPABILITY = "move_pose_reach_along_request_v2"
PRIMITIVE_CAPABILITY = "pose_primitive_reach_along_request_v2"
NEAREST_CAPABILITY = "reach_nearest_orientation_v2"
NEAREST_CLAIM = (
    "Read-only IK at each candidate endpoint by the same reach check as reach_along_request (its ik_model "
    "and matches_move_pose_admission apply), from the measured joints: the requested orientation turned "
    "about axes perpendicular to its approach axis, holding the jaw centre; no path, collision, payload or "
    "arrival claim, and an orientation outside the sweep may solve."
)
#: The sampling half of an unsolved target's claim; the backend's own claim boundary follows it.
SAMPLING_CLAIM = (
    "Read-only IK at sampled points of the straight interpolation from the measured pose to the target; "
    "reachability is not proven monotone between samples, and no path, collision, payload or arrival claim."
)


def move_pose_declared(surface: Any) -> bool:
    return bool(surface.supports(MOVE_POSE_CAPABILITY))


def primitive_declared(surface: Any) -> bool:
    return bool(surface.supports(PRIMITIVE_CAPABILITY))


def basis(answer: Mapping[str, Any]) -> dict[str, Any]:
    """Which IK answered and whether it is move_pose's admission for this request."""
    return {"ik_model": answer.get("ik_model"),
            "matches_move_pose_admission": answer.get("matches_move_pose_admission")}


def solved(answer: Mapping[str, Any]) -> dict[str, Any]:
    """A target the reach check solved: still only that check's answer."""
    return {"target_reachable": True, **basis(answer), "claim_boundary": str(answer["claim_boundary"])}


def unsolved(host: Any, arm: str, reach: Mapping[str, Any], *, answer: Mapping[str, Any],
             target_pose: Sequence[float], interaction: str) -> dict[str, Any]:
    """The v1 reach-along fields of an unsolved target, re-claimed, plus the basis and nearest turn."""
    return {**reach, "claim_boundary": f"{SAMPLING_CLAIM} {answer['claim_boundary']}", **basis(answer),
            **nearest(host, arm, target_pose=target_pose, interaction=interaction)}


def nearest(host: Any, arm: str, *, target_pose: Sequence[float], interaction: str) -> dict[str, Any]:
    """``nearest_reachable_orientation`` on a surface declaring the v2 sweep, else nothing."""
    if not host.surface.supports(NEAREST_CAPABILITY):
        return {}
    return reach_orientation.sweep(host, arm, target_pose=target_pose, interaction=interaction, claim=NEAREST_CLAIM)


__all__ = ["MOVE_POSE_CAPABILITY", "NEAREST_CAPABILITY", "NEAREST_CLAIM", "PRIMITIVE_CAPABILITY", "SAMPLING_CLAIM",
           "basis", "move_pose_declared", "nearest", "primitive_declared", "solved", "unsolved"]
