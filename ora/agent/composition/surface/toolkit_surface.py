"""File-backed toolkit surfaces: a surface directory plus each tool's own description file.

``ora/policy/tool_surfaces/<profile_id>/`` contains

- ``surface.json`` (``ora.toolkit-surface.v1``): the ordered tools, their
  roles, capabilities, view config, grants and receipt delivery;
- ``prompt.md``: the surface prompt, notes on the set as a whole, delivered
  as the last ``tool_manual`` layer when it has text (empty: no layer);
- optionally ``tool_descriptions``: a tool whose description this surface
  keeps beside ``surface.json`` instead of the tool's own, in the same format.
  Such a file may leave out ``resultSchema``: the tool's own then applies.
  ``"resultSchema": null`` says the tool has no public result schema (the
  Synthetic and ``robot_runtime_v2`` tools declare none);
- optionally ``result_overrides`` (``ora.toolkit-result-overrides.v1``): the
  result schemas this surface's delivery needs instead of the tools' own.
  The descriptions describe each tool's native tool_output receipt; a surface
  that still delivers receipts with the observation (v4) keeps its result
  schemas here and says so in its prompt.

Each tool is one directory, ``ora/agent/toolkit/tools/<name>/``: its entry
``tool.py`` and its ``description.json`` (``ora.toolkit-tool.v1``: the
description, input schema and result schema exactly as delivered).

Nothing is layered or rewritten at resolution time: the files are the
surface. Every file loads through ``policy.lock``, and the
ResolvedToolSurface records one component per file. A description belongs to
its tool, so editing it changes every surface that lists the tool; a sealed
run keeps what it was shown through its plan's commit and component hashes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..locked_assets import locked_asset
from .resolution import resolve_effect_semantics, resolve_tool_semantics
from .schema_validation import ToolSurfaceError, lint_input_schema
from .toolkit_grants import validate_grants

SURFACE_SCHEMA = "ora.toolkit-surface.v1"
TOOL_SCHEMA = "ora.toolkit-tool.v1"
RESULT_OVERRIDES_SCHEMA = "ora.toolkit-result-overrides.v1"
#: Each tool's directory: ``<name>/tool.py`` and ``<name>/description.json``.
TOOLS_DIR = Path("ora/agent/toolkit/tools")
#: A tool name: snake case, or dotted segments of it (``robot_runtime_v2``'s ``observation.dump_state``).
_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
_TOOL_KEYS = {"schema", "name", "description", "inputSchema", "resultSchema"}
#: A surface's own description file may share the tool's result schema instead of restating it.
_OWN_TOOL_KEYS = (_TOOL_KEYS, _TOOL_KEYS - {"resultSchema"})


def _asset(root: Path, relative: str, label: str):
    return locked_asset(root, relative, label=label, error=ToolSurfaceError)


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ToolSurfaceError(f"toolkit surface outside the source tree: {path}") from error


def _tool_file(root: Path, relative: str, name: str, keys: tuple[set[str], ...]):
    """One tool's description file and its asset, checked against the key sets it may have."""

    asset = _asset(root, relative, f"tool {name!r}")
    value = json.loads(asset.data.decode("utf-8"))
    if not isinstance(value, dict) or set(value) not in keys or value["schema"] != TOOL_SCHEMA:
        raise ToolSurfaceError(f"{relative} is not an {TOOL_SCHEMA} description")
    if value["name"] != name:
        raise ToolSurfaceError(f"{relative} describes {value['name']!r}")
    return value, asset


def _prompt_path(root: Path, manifest_path: Path, manifest: Mapping[str, Any]) -> str | None:
    name = manifest.get("prompt")
    if name is None:
        return None
    if not isinstance(name, str) or "/" in name or not name.endswith(".md"):
        raise ToolSurfaceError("toolkit surface prompt must be a Markdown file beside surface.json")
    return _relative(root, manifest_path.parent / name)


def surface_prompt(root: Path, manifest_path: Path, manifest: Mapping[str, Any]) -> str | None:
    """The surface prompt's tree-relative path, or None when the surface has no prompt text."""

    relative = _prompt_path(root, manifest_path, manifest)
    if relative is None or not _asset(root, relative, "surface prompt").data.strip():
        return None
    return relative


def resolve_toolkit_surface(
    profile_id: str, manifest_path: Path, manifest: Mapping[str, Any], revision: int, *,
    component_overrides: Mapping[str, Any] | None,
    toolkit_grants: Mapping[str, Any] | None,
    toolkit_receipt_delivery: str | None,
):
    """Read one toolkit surface and its tools' description files."""

    from .tool_surface import (
        TOOL_HELP_CONTRACT_CAPABILITY, _project_root, _resolve_runtime_identity,
        _resolve_view_config, _resolved_surface, validate_tool_help_contract,
    )

    if component_overrides:
        raise ToolSurfaceError("a toolkit surface takes no component overrides; edit its tool files")
    runtime_id, capabilities = _resolve_runtime_identity(manifest, profile_id)
    try:
        grants = validate_grants(toolkit_grants, capabilities)
    except ValueError as error:
        raise ToolSurfaceError(str(error)) from error
    required = manifest.get("required_grants", {"tools": [], "state_keys": []})
    if grants != required:
        raise ToolSurfaceError(f"{profile_id} requires toolkit_grants {required}")
    delivery = manifest.get("receipt_delivery")
    if toolkit_receipt_delivery is not None and toolkit_receipt_delivery != delivery:
        raise ToolSurfaceError(f"{profile_id} delivers receipts by {delivery} only")
    root = _project_root()
    tools = manifest.get("tools")
    if (not isinstance(tools, list) or not tools or len(tools) != len(set(tools))
            or not all(isinstance(name, str) and _NAME.match(name) for name in tools)):
        raise ToolSurfaceError("toolkit surface tools must be a non-empty list of unique tool names")
    definitions: list[dict[str, Any]] = []
    results: dict[str, Mapping[str, Any]] = {}
    descriptors: list[dict[str, str]] = []
    own_descriptions = manifest.get("tool_descriptions") or {}
    if not isinstance(own_descriptions, Mapping) or not all(
            isinstance(ref, str) and ref for ref in own_descriptions.values()):
        raise ToolSurfaceError("tool_descriptions must name each file by a path string: inline sha256 pins are not "
                               "accepted (policy.lock carries asset integrity)")
    for name in tools:
        shared = (TOOLS_DIR / name / "description.json").as_posix()
        own = name in own_descriptions
        relative = _relative(root, manifest_path.parent / own_descriptions[name]) if own else shared
        value, asset = _tool_file(root, relative, name, _OWN_TOOL_KEYS if own else (_TOOL_KEYS,))
        definition = {"name": name, "description": value["description"], "inputSchema": value["inputSchema"]}
        lint_input_schema(definition["inputSchema"], source=f"{profile_id}.{name}")
        definitions.append(definition)
        descriptors.append({"slot": f"tool:{name}", "ref": relative, "sha256": asset.sha256})
        if "resultSchema" not in value:  # the surface's own description, with the tool's own result
            value, asset = _tool_file(root, shared, name, (_TOOL_KEYS,))
            descriptors.append({"slot": f"result:{name}", "ref": shared, "sha256": asset.sha256})
        if value["resultSchema"] is not None:  # null: the tool declares no result schema
            results[name] = value["resultSchema"]
    overrides = manifest.get("result_overrides")
    if overrides is not None:
        if not isinstance(overrides, str) or "/" in overrides or not overrides.endswith(".json"):
            raise ToolSurfaceError("toolkit surface result_overrides must be a JSON file beside surface.json")
        relative = _relative(root, manifest_path.parent / overrides)
        asset = _asset(root, relative, "result overrides")
        value = json.loads(asset.data.decode("utf-8"))
        replaced = value.get("results") if isinstance(value, dict) else None
        if (value.get("schema") if isinstance(value, dict) else None) != RESULT_OVERRIDES_SCHEMA \
                or not isinstance(replaced, dict) or not replaced or set(replaced) - set(tools):
            raise ToolSurfaceError(f"{relative} is not an {RESULT_OVERRIDES_SCHEMA} file for this surface's tools")
        results.update(replaced)
        descriptors.append({"slot": "result_overrides", "ref": relative, "sha256": asset.sha256})
    # The prompt is bound here for identity; the RunSpec defaults deliver its
    # text as the last tool_manual layer (surface_prompt).
    prompt = _prompt_path(root, manifest_path, manifest)
    if prompt is not None:
        descriptors.append({"slot": "surface_prompt", "ref": prompt,
                            "sha256": _asset(root, prompt, "surface prompt").sha256})
    mutation_tools, lifecycle_tools = resolve_tool_semantics(manifest, definitions)
    observation_tools, call_effect_profile = resolve_effect_semantics(
        manifest, definitions, mutation_tools, lifecycle_tools,
    )
    view_config = _resolve_view_config(manifest)
    if TOOL_HELP_CONTRACT_CAPABILITY in capabilities:
        validate_tool_help_contract(definitions, results)
    return _resolved_surface(
        profile_id=profile_id, revision=revision, runtime_id=runtime_id,
        manifest=manifest, manifest_path=manifest_path, definitions=definitions,
        results=results, mutation_tools=mutation_tools,
        lifecycle_tools=lifecycle_tools, observation_tools=observation_tools,
        call_effect_profile=call_effect_profile, view_config=view_config,
        capabilities=capabilities, descriptors=descriptors,
    )


def declared_receipt_delivery(profile_id: str) -> str | None:
    """The receipt delivery a file-backed surface declares (its RunSpec default), else None."""

    from .profile_catalog import PROFILE_CATALOG

    manifest_path = PROFILE_CATALOG.get(profile_id)
    if manifest_path is None:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("receipt_delivery") if manifest.get("schema") == SURFACE_SCHEMA else None


def declared_required_grants(profile_id: str) -> dict[str, Any] | None:
    """The grants a file-backed surface requires (``surface.json`` ``required_grants``) when it names any, else None."""

    from .profile_catalog import PROFILE_CATALOG

    manifest_path = PROFILE_CATALOG.get(profile_id)
    if manifest_path is None:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grants = manifest.get("required_grants") if manifest.get("schema") == SURFACE_SCHEMA else None
    return dict(grants) if grants and (grants.get("tools") or grants.get("state_keys")) else None


def surface_prompt_sources(profile_id: str, root: Path) -> list[str]:
    """The tool_manual layers a surface adds: its prompt when it has text, else none."""

    from .profile_catalog import PROFILE_CATALOG

    manifest_path = PROFILE_CATALOG.get(profile_id)
    if manifest_path is None:
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SURFACE_SCHEMA:
        return []
    prompt = surface_prompt(Path(root), manifest_path, manifest)
    return [] if prompt is None else [prompt]
