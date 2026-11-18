"""Versioned, hash-bound composition of policy-visible tool surfaces."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


from .view_config import AUXILIARY_CONTACT_SHEET_LAYOUTS, PUBLIC_FACTS_DELIVERY
from ..embodiment.geometry import (
    PUBLIC_GEOMETRY_DESCRIPTORS, bind_public_geometry_descriptor, load_public_geometry_binding,
    PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS,
)
from ..embodiment import EmbodimentDescriptorError
from ora.contracts.action_plan import DRY_RUN_ARGUMENT, MOVING_TOOLS

#: Every registered surface is file-backed (``toolkit_surface.py``); its catalog entry is its manifest.
TOOLKIT_SURFACE_SCHEMA = "ora.toolkit-surface.v1"
RESOLVED_SCHEMA = "ora.resolved-tool-surface.v1"
VIEW_CONFIG_SCHEMA = "ora.policy-view-config.v1"
TOOL_HELP_CONTRACT_CAPABILITY = "policy_tool_help_v1"
TOOL_HELP_DESCRIPTION_MARKERS = (
    "Use when:",
    "Effect:",
    "Returns:",
    "Does not:",
    "On error:",
)
from . import profile_catalog

# These source-frozen profiles predate the model-visible help contract.  New
# catalog entries must opt in to TOOL_HELP_CONTRACT_CAPABILITY instead of
# silently inheriting this grandfather list.
LEGACY_TOOL_HELP_EXEMPT_PROFILES = frozenset({
    "robot_runtime_v2",
    "native_atomic_causal_v1",
})


from .schema_validation import (  # noqa: E402 - re-exported public validator API
    ANNOTATION_KEYWORDS, SchemaViolation, ToolSurfaceError, VALIDATED_KEYWORDS,
    lint_input_schema, validate_schema_instance,
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_tool_help_contract(
    definitions: Sequence[Mapping[str, Any]],
    result_schemas: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate the model-visible ``--help`` contract for a ToolSurface.

    Historical surfaces predate this contract and remain source-frozen.  A
    successor opts in through ``policy_tool_help_v1`` and must then document
    selection, effects, public results, non-guarantees and recovery for every
    tool.  Input properties are documented recursively, and every tool owns a
    Runtime-validated public result schema even when a provider protocol does
    not have a native output-schema field.
    """

    def check_input_schema(node: Mapping[str, Any], *, path: str) -> None:
        declared = node.get("type")
        declared_types = declared if isinstance(declared, list) else [declared]
        if "object" in declared_types:
            if node.get("additionalProperties") is not False:
                raise ToolSurfaceError(
                    f"{path} must set additionalProperties=false under "
                    f"{TOOL_HELP_CONTRACT_CAPABILITY}"
                )
            properties = node.get("properties", {})
            if not isinstance(properties, Mapping):
                raise ToolSurfaceError(f"{path}.properties must be an object")
            for name, child in properties.items():
                if not isinstance(child, Mapping):
                    raise ToolSurfaceError(f"{path}.{name} must be a schema object")
                description = child.get("description")
                if not isinstance(description, str) or not description.strip():
                    raise ToolSurfaceError(
                        f"{path}.{name} lacks a parameter description"
                    )
                check_input_schema(child, path=f"{path}.{name}")
        items = node.get("items")
        if isinstance(items, Mapping):
            check_input_schema(items, path=f"{path}[]")
        for keyword in ("oneOf", "anyOf", "allOf"):
            branches = node.get(keyword, [])
            if isinstance(branches, list):
                for index, branch in enumerate(branches):
                    if isinstance(branch, Mapping):
                        check_input_schema(
                            branch, path=f"{path}.{keyword}[{index}]",
                        )

    names: list[str] = []
    for definition in definitions:
        name = definition.get("name")
        if not isinstance(name, str) or not name:
            raise ToolSurfaceError("tool-help contract requires named tools")
        names.append(name)
        description = definition.get("description")
        if not isinstance(description, str):
            raise ToolSurfaceError(f"tool {name!r} lacks a description")
        missing = [
            marker for marker in TOOL_HELP_DESCRIPTION_MARKERS
            if marker not in description
        ]
        if missing:
            raise ToolSurfaceError(
                f"tool {name!r} help lacks sections: {', '.join(missing)}"
            )
        schema = definition.get("inputSchema")
        if not isinstance(schema, Mapping):
            raise ToolSurfaceError(f"tool {name!r} lacks an input schema")
        check_input_schema(schema, path=f"{name}.inputSchema")

    missing_results = set(names) - set(result_schemas)
    extra_results = set(result_schemas) - set(names)
    if missing_results or extra_results:
        raise ToolSurfaceError(
            "tool-help result schemas must cover exactly the advertised tools; "
            f"missing={sorted(missing_results)}, extra={sorted(extra_results)}"
        )


@dataclass(frozen=True)
class ResolvedToolSurface:
    profile_id: str
    revision: int
    runtime_id: str
    definitions: tuple[dict[str, Any], ...]
    result_schemas: Mapping[str, Mapping[str, Any]]
    mutation_tools: tuple[str, ...]
    lifecycle_tools: tuple[str, ...]
    observation_tools: tuple[str, ...]
    call_effect_profile: str | None
    view_config: Mapping[str, Any] | None
    capabilities: tuple[str, ...]
    provenance: Mapping[str, Any]
    components: tuple[dict[str, str], ...]
    manifest_sha256: str
    composition_sha256: str

    def supports(self, capability: str) -> bool:
        """Return one manifest-bound semantic capability.

        Runtime code must branch on these capabilities rather than duplicating
        profile-name allowlists.  A new profile therefore inherits only the
        behavior it explicitly declares in its hash-bound manifest.
        """

        return capability in self.capabilities

    def select(self, allowed: Sequence[str] | None = None) -> list[dict[str, Any]]:
        selected = {item["name"] for item in self.definitions} if allowed is None else set(allowed)
        known = {item["name"] for item in self.definitions}
        if not selected.issubset(known):
            raise ToolSurfaceError(
                f"tool allowlist contains unknown names: {sorted(selected - known)}"
            )
        return [copy.deepcopy(item) for item in self.definitions if item["name"] in selected]

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "schema": RESOLVED_SCHEMA,
            "profile_id": self.profile_id,
            "revision": self.revision,
            "runtime_id": self.runtime_id,
            "definitions": [copy.deepcopy(item) for item in self.definitions],
            "result_schemas": copy.deepcopy(dict(self.result_schemas)),
            "mutation_tools": list(self.mutation_tools),
            "lifecycle_tools": list(self.lifecycle_tools),
            "capabilities": list(self.capabilities),
            "provenance": copy.deepcopy(dict(self.provenance)),
            "components": [dict(item) for item in self.components],
            "manifest_sha256": self.manifest_sha256,
            "composition_sha256": self.composition_sha256,
        }
        if self.call_effect_profile is not None:
            snapshot["observation_tools"] = list(self.observation_tools)
            snapshot["call_effect_profile"] = self.call_effect_profile
        if self.view_config is not None:
            snapshot["view_config"] = copy.deepcopy(dict(self.view_config))
        return snapshot


def _resolve_view_config(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate one shared policy-view contract for tools and session delivery."""

    raw = manifest.get("view_config")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ToolSurfaceError("view_config must be an object")
    required = {
        "schema", "rgb_resolution", "derived_resolution",
        "automatic_actual_views", "default_observe_views",
        "coordinate_convention",
    }
    optional = {"auxiliary_contact_sheet_cameras", "current_visual_delivery", "current_observation_delivery", "controlled_frame_convention", "public_geometry_descriptor", "gripper_command_delivery", "controlled_point_projection", "jaw_contact_center_projection"}
    if not required.issubset(set(raw)) or set(raw) - required - optional:
        raise ToolSurfaceError(
            "view_config must contain exactly " + ", ".join(sorted(required))
            + " (optionally " + ", ".join(sorted(optional)) + ")"
        )
    if raw.get("schema") != VIEW_CONFIG_SCHEMA:
        raise ToolSurfaceError("view_config schema is unsupported")

    if "current_visual_delivery" in raw and raw["current_visual_delivery"] != "full_rgb_v1":
        raise ToolSurfaceError("unsupported current visual delivery")
    if "gripper_command_delivery" in raw and (
        raw["gripper_command_delivery"] != "commanded_beside_measured_v1"
        or raw.get("current_observation_delivery") != "all_public_v1"
    ):
        raise ToolSurfaceError("unsupported gripper command delivery")
    if "controlled_point_projection" in raw and (
        raw["controlled_point_projection"] != "per_camera_pixel_v1"
        or raw.get("current_observation_delivery") != "all_public_v1"
    ):
        raise ToolSurfaceError("unsupported controlled point projection")
    if "jaw_contact_center_projection" in raw and (
        raw["jaw_contact_center_projection"] != "descriptor_contact_center_v1"
        or "controlled_point_projection" not in raw
        or not isinstance(raw.get("public_geometry_descriptor"), str)
    ):
        raise ToolSurfaceError("unsupported jaw contact-centre projection")
    rgb_resolution = raw.get("rgb_resolution")
    if rgb_resolution != "native":
        if not (
            isinstance(rgb_resolution, list)
            and len(rgb_resolution) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in rgb_resolution
            )
        ):
            raise ToolSurfaceError("view_config.rgb_resolution must be native or [width, height]")
    derived_resolution = raw.get("derived_resolution")
    if not (
        isinstance(derived_resolution, list)
        and len(derived_resolution) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in derived_resolution
        )
    ):
        raise ToolSurfaceError("view_config.derived_resolution must be [width, height]")
    for field in ("automatic_actual_views", "default_observe_views"):
        values = raw.get(field)
        if not (
            isinstance(values, list)
            and values
            and len(values) == len(set(values))
            and all(isinstance(value, str) and value for value in values)
        ):
            raise ToolSurfaceError(f"view_config.{field} must contain unique view names")
    contact_sheet = "auxiliary_contact_sheet"
    capabilities = manifest.get("capabilities", [])
    if contact_sheet in raw["automatic_actual_views"] and (
        not isinstance(capabilities, list)
        or "automatic_current_auxiliary_contact_sheet_v1" not in capabilities
    ):
        raise ToolSurfaceError(
            "auxiliary_contact_sheet requires its automatic projection capability"
        )
    if contact_sheet in raw["default_observe_views"]:
        raise ToolSurfaceError(
            "auxiliary_contact_sheet is automatic derived evidence, not an observe camera"
        )
    sheet_cameras = raw.get("auxiliary_contact_sheet_cameras")
    if sheet_cameras is not None:
        if contact_sheet not in raw["automatic_actual_views"]:
            raise ToolSurfaceError(
                "auxiliary_contact_sheet_cameras requires the sheet in "
                "automatic_actual_views"
            )
        # The admitted counts are exactly the layouts the projection can
        # materialize, so a new sheet layout admits itself here.
        admitted_counts = sorted(AUXILIARY_CONTACT_SHEET_LAYOUTS)
        if not (
            isinstance(sheet_cameras, list)
            and len(sheet_cameras) in admitted_counts
            and len(sheet_cameras) == len(set(sheet_cameras))
            and all(
                isinstance(value, str) and value and value != contact_sheet
                for value in sheet_cameras
            )
        ):
            raise ToolSurfaceError(
                "auxiliary_contact_sheet_cameras must list three or four "
                "unique source camera names (admitted counts: "
                + ", ".join(str(count) for count in admitted_counts) + ")"
            )
    if raw.get("coordinate_convention") != "opencv_top_left_xy":
        raise ToolSurfaceError("view_config coordinate convention is unsupported")
    delivery = raw.get("current_observation_delivery")
    if delivery is not None:
        if delivery != PUBLIC_FACTS_DELIVERY or raw.get("controlled_frame_convention") not in PUBLIC_GEOMETRY_DESCRIPTORS:
            raise ToolSurfaceError("full public observation requires a supported controlled-frame convention")
        if raw.get("rgb_resolution") != "native":
            raise ToolSurfaceError("full public observation requires native RGB coordinates")
        if "current_public_robot_facts_v1" not in capabilities:
            raise ToolSurfaceError("full public observation requires its delivery capability")
        try:
            # The manifest names the descriptor by path; the resolved view
            # carries the {path, sha256} record of the bytes read here.
            geometry_record = bind_public_geometry_descriptor(raw.get("public_geometry_descriptor"), project_root=Path(__file__).resolve().parents[4])
            geometry_binding = load_public_geometry_binding(geometry_record, project_root=Path(__file__).resolve().parents[4])
        except EmbodimentDescriptorError as exc:
            raise ToolSurfaceError(str(exc)) from exc
        convention = raw["controlled_frame_convention"]
        expected_paths = {
            "ora/policy/embodiments/" + path
            for path in PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS[convention]
        }
        if geometry_record["path"] not in expected_paths:
            raise ToolSurfaceError("geometry descriptor does not match the controlled-frame convention")
        return {**copy.deepcopy(dict(raw)), "public_geometry_descriptor": geometry_record}
    elif "controlled_frame_convention" in raw or "public_geometry_descriptor" in raw:
        raise ToolSurfaceError("controlled-frame convention requires full public observation delivery")
    return copy.deepcopy(dict(raw))


def _project_root() -> Path:
    """The source tree whose ``policy.lock`` and retired registry bind the catalog.

    A rebound PROFILE_ROOT at <root>/ora/policy/tool_surfaces answers
    from that root; a test that widens the loading boundary to a filesystem
    anchor keeps this checkout.
    """

    profile_root = profile_catalog.PROFILE_ROOT.resolve()
    if profile_root.parts[-3:] == ("ora", "policy", "tool_surfaces"):
        return profile_root.parents[2]
    return Path(__file__).resolve().parents[4]


def _load_manifest(profile_id: str) -> tuple[Path, dict[str, Any], int]:
    """Find a registered surface manifest and its revision.

    Reads ``PROFILE_CATALOG`` from :mod:`profile_catalog` at call time, so a
    test that rebinds it there still steers resolution.
    """

    manifest_path = profile_catalog.PROFILE_CATALOG.get(profile_id)
    if manifest_path is None:
        raise ToolSurfaceError(f"unknown tool surface profile: {profile_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TOOLKIT_SURFACE_SCHEMA or manifest.get("profile_id") != profile_id:
        raise ToolSurfaceError(f"invalid tool surface manifest: {manifest_path}")
    revision = manifest.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ToolSurfaceError("tool surface revision must be positive")
    return manifest_path, manifest, revision


def _resolve_runtime_identity(
    manifest: Mapping[str, Any], profile_id: str,
) -> tuple[str, list[str]]:
    """Validate ``runtime_id`` and ``capabilities`` and the help-contract opt-in."""

    runtime_id = manifest.get("runtime_id")
    if not isinstance(runtime_id, str) or not runtime_id:
        raise ToolSurfaceError("tool surface runtime_id must be non-empty")
    capabilities = manifest.get("capabilities", [])
    if (
        not isinstance(capabilities, list)
        or not all(isinstance(name, str) and name for name in capabilities)
        or len(capabilities) != len(set(capabilities))
    ):
        raise ToolSurfaceError("tool surface capabilities must contain unique names")
    if (
        profile_id not in LEGACY_TOOL_HELP_EXEMPT_PROFILES
        and TOOL_HELP_CONTRACT_CAPABILITY not in capabilities
    ):
        raise ToolSurfaceError(
            f"new tool surface {profile_id!r} must declare "
            f"{TOOL_HELP_CONTRACT_CAPABILITY}"
        )
    return runtime_id, capabilities


#: Every action that moves the robot takes ``dry_run`` (``toolkit/interface/dry_run.py``); a surface declares it.
DRY_RUN_CAPABILITY = "action_dry_run_v1"
#: A surface that takes null for an omitted optional argument (``null_optional_arguments.py``).
NULL_AS_OMITTED_CAPABILITY = "optional_argument_null_as_omitted_v1"
DRY_RUN_HELP = (
    "true returns a preview of this call instead of acting: nothing moves and no native action is spent. It draws the "
    "planned path on the camera images that see it, with its start and end, and gives per arm the native actions, how "
    "far the hand travels and how much it turns, how high its path rises above its ends and where, and the least "
    "clearance between the jaws and what the depth images show, with where that is. It predicts no contact, no "
    "payload and nothing after the first contact; a dashed line is a servo's commanded line, not a planned path. The "
    "same call on the same observation then runs exactly the path shown. Worth a call near objects, while holding "
    "something, or for a long move.")
DRY_RUN_SHORT = "true returns a preview of this call instead of acting, as move_pose's dry_run says. Nothing moves."


def _with_dry_run(definitions: list[dict[str, Any]], mutation_tools: tuple[str, ...],
                  capabilities: Sequence[str]) -> list[dict[str, Any]]:
    """The definitions with ``dry_run`` on each tool that moves the robot."""

    kind: Any = ["boolean", "null"] if NULL_AS_OMITTED_CAPABILITY in capabilities else "boolean"
    result = copy.deepcopy(definitions)
    for item in result:
        if item["name"] in mutation_tools and item["name"] in MOVING_TOOLS:
            help_text = DRY_RUN_HELP if item["name"] == "move_pose" else DRY_RUN_SHORT
            item["inputSchema"].setdefault("properties", {})[DRY_RUN_ARGUMENT] = {"type": kind, "description": help_text}
    return result


def _resolved_surface(
    *, profile_id: str, revision: int, runtime_id: str, manifest: Mapping[str, Any],
    manifest_path: Path, definitions: list[dict[str, Any]],
    results: dict[str, Mapping[str, Any]], mutation_tools: tuple[str, ...],
    lifecycle_tools: tuple[str, ...], observation_tools: tuple[str, ...],
    call_effect_profile: str | None, view_config: dict[str, Any] | None,
    capabilities: list[str], descriptors: list[dict[str, str]],
) -> ResolvedToolSurface:
    """Hash the composition and freeze it into a ResolvedToolSurface."""

    manifest_hash = file_sha256(manifest_path)
    if DRY_RUN_CAPABILITY in capabilities:
        definitions = _with_dry_run(definitions, mutation_tools, capabilities)
    composition = {
        "profile_id": profile_id,
        "revision": revision,
        "runtime_id": runtime_id,
        "definitions": definitions,
        "result_schemas": results,
        "mutation_tools": list(mutation_tools),
        "lifecycle_tools": list(lifecycle_tools),
        "capabilities": capabilities,
        "provenance": manifest.get("provenance", {}),
        "components": descriptors,
        "manifest_sha256": manifest_hash,
    }
    if call_effect_profile is not None:
        composition["observation_tools"] = list(observation_tools)
        composition["call_effect_profile"] = call_effect_profile
    if view_config is not None:
        composition["view_config"] = copy.deepcopy(view_config)
    return ResolvedToolSurface(
        profile_id=profile_id,
        revision=revision,
        runtime_id=runtime_id,
        definitions=tuple(definitions),
        result_schemas=results,
        mutation_tools=mutation_tools,
        lifecycle_tools=lifecycle_tools,
        observation_tools=observation_tools,
        call_effect_profile=call_effect_profile,
        view_config=view_config,
        capabilities=tuple(capabilities),
        provenance=copy.deepcopy(dict(manifest.get("provenance", {}))),
        components=tuple(descriptors),
        manifest_sha256=manifest_hash,
        composition_sha256=canonical_sha256(composition),
    )


def resolve_tool_surface(
    profile_id: str = "robot_runtime_v2",
    *,
    component_overrides: Mapping[str, Any] | None = None,
    toolkit_grants: Mapping[str, Any] | None = None,
    toolkit_receipt_delivery: str | None = None,
) -> ResolvedToolSurface:
    """Resolve a registered surface: its ``surface.json``-format manifest and the tool files it names
    (``toolkit_surface.py``). Nothing is layered or rewritten; a surface takes no component overrides."""

    manifest_path, manifest, revision = _load_manifest(profile_id)
    from .toolkit_surface import resolve_toolkit_surface
    return resolve_toolkit_surface(
        profile_id, manifest_path, manifest, revision, component_overrides=component_overrides,
        toolkit_grants=toolkit_grants, toolkit_receipt_delivery=toolkit_receipt_delivery,
    )
