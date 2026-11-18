"""Compact decision evidence for opt-in perception surfaces; no action authority."""
from __future__ import annotations

from ora.agent.composition.surface import tool_names

import base64
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping

CAPABILITY = "public_rgbd_perception_feedback_v2"


def enabled(surface: Any) -> bool:
    return surface is not None and surface.supports(CAPABILITY)


def image_label(index: int, role: str, *, camera: str = "", candidate_id: str = "",
                target_ref: str = "", step: int | None = None) -> dict[str, Any]:
    return dict(image_index=index, role=role, camera=camera, candidate_id=candidate_id,
                target_ref=target_ref, observation_step=step)


def grasp_review(result: dict[str, Any], manifest: Mapping[str, Any], overlay: Path) -> None:
    """Deliver the already source-bound public scene review, when available."""
    result["image_mapping"] = [image_label(0, "local_contact_candidates")]
    result["scene_review"] = {"status": "unavailable", "reason": "public_scene_review_unavailable",
        "scope": "observed_public_rgbd_only;unobserved_space_unknown;axes_and_ids_match_local_candidates"}
    review = manifest.get("review_visualization", {})
    if review.get("status") != "available":
        return
    artifact = review["artifact"]
    path = (overlay.parent / artifact["path"]).resolve()
    try:
        if not path.is_relative_to(overlay.parent.resolve()):
            raise ValueError("review path escaped query")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise ValueError("review image content drift")
    except (OSError, ValueError):
        result["scene_review"]["reason"] = "public_scene_review_integrity_unavailable"
        return
    result["images_base64"].append(base64.b64encode(data).decode("ascii"))
    result["image_mapping"].append(image_label(1, "global_public_scene_candidates"))
    result["scene_review"].update(status="available", reason="public_source_bound_review")


#: The tools that carry a measurement contract, named canonically.
#:
#: The membership gate below is this set and the lookup is this set, because
#: they were two literals for one fact and they drifted the moment the runtime
#: started resolving spellings: the gate was canonicalised, the table's
#: ``motion.probe`` key was not, and a `motion.probe` call passed the gate and
#: raised `KeyError` on the line after it -- reaching a policy as a
#: non-retryable `internal_tool_error`.
MEASUREMENT_CONTRACTS = frozenset({"observe", "mark_point"})


def enrich(host: Any, name: str, arguments: Mapping[str, Any], result: dict[str, Any]) -> None:
    # Every branch below names a tool. A surface may advertise either
    # spelling, and an unmatched branch skips enrichment silently.
    name = tool_names.canonical(name)
    if not enabled(host.surface):
        return
    if name == "vision_ground":
        for candidate in result["candidates"]:
            candidate["candidate_id"] = f"T{candidate['rank']}"
    if name in MEASUREMENT_CONTRACTS:
        contracts = {
            "observe": "RGB pixels: (u,v), top-left origin; world geometry: metres; each image is the named saved view, not a new capture. Missing/occluded pixels provide no evidence.",
            "mark_point": "xyz_m is visible world geometry in metres; residuals measure projection agreement, not semantic identity. Pending marks need complementary views; moved-camera pixels are observation-specific.",
        }
        assert contracts.keys() == MEASUREMENT_CONTRACTS
        result["measurement_contract"] = contracts[name]
    if "images_base64" in result and "image_mapping" not in result:
        views = result.get("views", [])
        candidates = result.get("candidates", [])
        mapping = []
        for index, _ in enumerate(result["images_base64"]):
            if name in {"vision_segment", "vision_ground"}:
                candidate = candidates[index - 1] if 0 < index <= len(candidates) else {}
                item = image_label(index, "source_overview" if index == 0 else "candidate_source_crop",
                    camera=arguments.get("camera", ""), candidate_id=candidate.get("candidate_id", ""),
                    target_ref=candidate.get("target_ref", ""), step=arguments.get("step"))
            else:
                item = image_label(index, "saved_public_view" if name == "observe" else "public_geometry_review",
                    camera=views[index] if index < len(views) else "", step=arguments.get("step"))
            mapping.append(item)
        result["image_mapping"] = mapping
