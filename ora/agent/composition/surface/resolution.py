"""The tool roles a surface manifest declares, checked against its definitions.

``toolkit_surface.resolve_toolkit_surface`` calls these two steps, in the order
it reports their errors: the disjoint mutation and lifecycle tool sets, then
the optional observation tools and call-effect profile.
"""

from __future__ import annotations

from typing import Any, Mapping

from .schema_validation import ToolSurfaceError


def resolve_tool_semantics(
    manifest: Mapping[str, Any], definitions: list[dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the disjoint mutation and lifecycle tool sets."""

    raw_mutations = manifest.get("mutation_tools", [])
    raw_lifecycle = manifest.get("lifecycle_tools", [])
    for label, values in (
        ("mutation_tools", raw_mutations), ("lifecycle_tools", raw_lifecycle),
    ):
        if (
            not isinstance(values, list)
            or not all(isinstance(name, str) and name for name in values)
            or len(values) != len(set(values))
        ):
            raise ToolSurfaceError(f"{label} must contain unique tool names")
    mutation_tools = tuple(raw_mutations)
    lifecycle_tools = tuple(raw_lifecycle)
    known = {item["name"] for item in definitions}
    if not set(mutation_tools + lifecycle_tools).issubset(known):
        raise ToolSurfaceError("tool semantics reference unknown tools")
    if set(mutation_tools) & set(lifecycle_tools):
        raise ToolSurfaceError("mutation and lifecycle tool semantics must be disjoint")
    return mutation_tools, lifecycle_tools


def resolve_effect_semantics(
    manifest: Mapping[str, Any],
    definitions: list[dict[str, Any]],
    mutation_tools: tuple[str, ...],
    lifecycle_tools: tuple[str, ...],
) -> tuple[tuple[str, ...], str | None]:
    """Return the optional observation tools and call-effect profile."""

    known = {item["name"] for item in definitions}
    has_effect_semantics = (
        "observation_tools" in manifest or "call_effect_profile" in manifest
    )
    observation_tools: tuple[str, ...] = ()
    call_effect_profile: str | None = None
    if has_effect_semantics:
        raw_observations = manifest.get("observation_tools")
        raw_effect_profile = manifest.get("call_effect_profile")
        if (
            not isinstance(raw_observations, list)
            or not raw_observations
            or not all(isinstance(name, str) and name for name in raw_observations)
            or len(raw_observations) != len(set(raw_observations))
        ):
            raise ToolSurfaceError("observation_tools must contain unique tool names")
        if not isinstance(raw_effect_profile, str) or not raw_effect_profile:
            raise ToolSurfaceError("call_effect_profile must be a stable id")
        observation_tools = tuple(raw_observations)
        call_effect_profile = raw_effect_profile
        if not set(observation_tools).issubset(known):
            raise ToolSurfaceError("observation_tools references unknown tools")
        if set(observation_tools) & set(mutation_tools + lifecycle_tools):
            raise ToolSurfaceError(
                "observation tools must be disjoint from mutation/lifecycle tools"
            )
    return observation_tools, call_effect_profile
