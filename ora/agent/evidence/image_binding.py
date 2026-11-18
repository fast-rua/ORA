"""Path and hash checks that bind a delivered image to its recorded evidence.

Every image a policy session is handed -- the fresh views of the current
observation and the historical post-close views of a skill -- is read from the
run workspace only after its path is shown to stay inside its evidence root
without crossing a symlink, and its bytes to match the recorded sha256. The
context projections that freeze these views and the policy-state writer that
records them verify through the same functions, which import nothing from the
projections.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

CURRENT_OBSERVATION_BUNDLE_SCHEMA = "ora.current-observation-bundle.v1"
FULL_VISUAL_SCHEMA = "ora.current-visual-observation.v1"


class PostActionObservationError(RuntimeError):
    """Mandatory fresh action-end evidence cannot be delivered safely."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    path = root / candidate
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PostActionObservationError("image path escapes its evidence root")
    if any((root / Path(*candidate.parts[:index])).is_symlink()
           for index in range(1, len(candidate.parts) + 1)):
        raise PostActionObservationError("image path is symlinked")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise PostActionObservationError("current image is unavailable")
    return path


def resolve_post_action_images(
    context: Mapping[str, Any], *, root: Path, path_key: str = "path",
) -> list[dict[str, Any]]:
    """Verify the set immediately before actual runtime delivery."""

    record = context.get("current_visual_observation") or context.get("post_action_observation")
    if record is None:
        return []
    if not isinstance(record, Mapping) or record.get("schema") not in {
        CURRENT_OBSERVATION_BUNDLE_SCHEMA, FULL_VISUAL_SCHEMA,
    }:
        raise PostActionObservationError("unsupported post-action observation schema")
    state = context.get("policy_state")
    if isinstance(state, Mapping) and (
        state.get("step") != record.get("observation_step")
        or state.get("state_sha256") != record.get("state_sha256")
    ):
        raise PostActionObservationError("post-action images differ from the current policy state")
    images = record.get("images")
    if not isinstance(images, (list, tuple)) or not images:
        raise PostActionObservationError("post-action observation has no fresh images")
    result = []
    seen = set()
    for image in images:
        if not isinstance(image, Mapping) or image.get("camera") in seen:
            raise PostActionObservationError("invalid or duplicate post-action image")
        seen.add(image.get("camera"))
        path = _safe_file(root, str(image.get(path_key, "")))
        if _hash(path) != image.get("sha256"):
            raise PostActionObservationError("post-action image hash mismatch")
        result.append({**dict(image), "path": path})
    return result


def resolve_skill_images(context: Mapping[str, Any], *, root: Path, path_key: str = "path") -> tuple[Path, ...]:
    """Verify materialized historical images before a managed backend reads them."""
    images = context.get("skill_observations", [])
    if not isinstance(images, list) or len(images) > 2:
        raise PostActionObservationError("invalid skill observation image count")
    paths = []
    for item in images:
        if item.get("role") != "post_close_before_lift":
            raise PostActionObservationError("invalid skill observation image role")
        path = _safe_file(root, item[path_key])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
            raise PostActionObservationError("materialized skill image hash mismatch")
        paths.append(path)
    return tuple(paths)


__all__ = [
    "CURRENT_OBSERVATION_BUNDLE_SCHEMA", "FULL_VISUAL_SCHEMA", "PostActionObservationError",
    "resolve_post_action_images", "resolve_skill_images",
]
