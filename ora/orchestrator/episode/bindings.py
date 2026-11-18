"""Preflight binding consumption, the backend's round scene and environment-owned round comparability."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping

from ..backends.registry import find


def _qualified_runtime_binding(
    run_root: Path, runtime_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load the exact external control binding accepted by preflight (the backend's), or None."""

    selected = find(runtime_config.get("kind"))
    hook = selected.supervisor_binding if selected is not None else None
    return hook(run_root, runtime_config) if hook is not None else None


def round_scene(plan: Mapping[str, Any]) -> Any | None:
    """The runtime backend's scheduled-round scene (RoboDojo's layout schedule), or None."""

    selected = find((plan.get("runtime") or {}).get("kind"))
    return selected.round_scene if selected is not None else None


ROUND_COMPARABILITY_FIELDS = (
    "round_end_basis", "protocol_comparable", "upstream_ending_condition",
)


def _round_comparability(verdict: Mapping[str, Any] | None) -> dict[str, Any]:
    """Mirror the environment's own round-ending classification, or nulls."""

    source = verdict if isinstance(verdict, Mapping) else {}
    return {field: source.get(field) for field in ROUND_COMPARABILITY_FIELDS}
