"""Lazy facade for policy composition: prompt sources, ToolSurfaces and inspection.

The work lives in five sub-packages -- ``schemas`` (capability binders),
``surface`` (ToolSurface resolution), ``embodiment`` (descriptors and cards),
``prompt`` (bundles, policy layers, context packs) and ``task_contract``
(reviewed completion contracts) -- with ``baseline``, ``ora_local``,
``locked_assets`` and ``inspection`` at the root. The names below are imported
from their module when first read, so importing one module of the package
does not load the others.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CONFIG_INSPECTION_SCHEMA": (".inspection", "SCHEMA"),
    "compiled_policy_paths": (".inspection", "compiled_policy_paths"),
    "inspect_policy_configuration": (".inspection", "inspect_policy_configuration"),
    "inspect_run_configuration": (".inspection", "inspect_run_configuration"),
    "LEGACY_CHANNELS": (".prompt.policy_layers", "LEGACY_CHANNELS"),
    "POLICY_CONTEXT_V3_SCHEMA": (".prompt.policy_layers", "POLICY_CONTEXT_V3_SCHEMA"),
    "V2_CHANNELS": (".prompt.policy_layers", "V2_CHANNELS"),
    "V3_CHANNELS": (".prompt.policy_layers", "V3_CHANNELS"),
    "PolicyCompositionError": (".prompt.policy_layers", "PolicyCompositionError"),
    "materialize_policy": (".prompt.policy_layers", "materialize_policy"),
    "EMBODIMENT_SECTION_HEADINGS": (".prompt.bundle", "EMBODIMENT_SECTION_HEADINGS"),
    "PROMPT_BUNDLE_AUDIT_SCHEMA": (".prompt.bundle", "PROMPT_BUNDLE_AUDIT_SCHEMA"),
    "PROMPT_BUNDLE_EMBODIMENT_FORMAT": (".prompt.bundle", "PROMPT_BUNDLE_EMBODIMENT_FORMAT"),
    "PROMPT_BUNDLE_LAYERS": (".prompt.bundle", "PROMPT_BUNDLE_LAYERS"),
    "PROMPT_BUNDLE_SCHEMA": (".prompt.bundle", "PROMPT_BUNDLE_SCHEMA"),
    "PromptBundleError": (".prompt.bundle", "PromptBundleError"),
    "audit_prompt_bundle": (".prompt.bundle", "audit_prompt_bundle"),
    "load_prompt_bundle": (".prompt.bundle", "load_prompt_bundle"),
    "prompt_bundle_template": (".prompt.bundle", "prompt_bundle_template"),
    "validate_prompt_bundle": (".prompt.bundle", "validate_prompt_bundle"),
    "PROFILE_CATALOG": (".surface.profile_catalog", "PROFILE_CATALOG"),
    "PROFILE_ROOT": (".surface.profile_catalog", "PROFILE_ROOT"),
    "RESOLVED_SCHEMA": (".surface.tool_surface", "RESOLVED_SCHEMA"),
    "TOOL_HELP_CONTRACT_CAPABILITY": (".surface.tool_surface", "TOOL_HELP_CONTRACT_CAPABILITY"),
    "TOOL_HELP_DESCRIPTION_MARKERS": (".surface.tool_surface", "TOOL_HELP_DESCRIPTION_MARKERS"),
    "ResolvedToolSurface": (".surface.tool_surface", "ResolvedToolSurface"),
    "ToolSurfaceError": (".surface.tool_surface", "ToolSurfaceError"),
    "canonical_sha256": (".surface.tool_surface", "canonical_sha256"),
    "file_sha256": (".surface.tool_surface", "file_sha256"),
    "resolve_tool_surface": (".surface.tool_surface", "resolve_tool_surface"),
    "validate_tool_help_contract": (".surface.tool_surface", "validate_tool_help_contract"),
    "validate_schema_instance": (".surface.tool_surface", "validate_schema_instance"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
