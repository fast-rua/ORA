"""``policy.context_pack``: bind an external context pack into a RunPlan.

A RunSpec names the pack by location and manifest hash::

    policy:
      context_pack:
        path: /absolute/path/to/pack        # or path_env: NAME_OF_A_VARIABLE
        manifest_sha256: <sha256 of manifest.json>
        profile: {...}                      # only for a manifest without ora_profile

The resolver loads the pack (every listed file checked against its hash),
resolves the delivery profile (``ora.agent.composition.prompt.context_pack``) and
writes the binding in the field's place (schema ``ora.context-pack-binding.v1``):
the manifest's path and hash, the file hashes, the profile and where it came
from. The profile's withheld tools leave ``tool_allowlist``. Once the prompt
layers are bound, ``bind_context_pack_delivery`` adds the composed system
text and episode-task template (from the plan's own compiled embodiment card
and completion-contract layers), and ``context_pack_tool_layer`` checks the
tool overlay against every advertised tool. A context pack runs on the Pi
agent core only. Without the field nothing is added to the plan.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.composition import ResolvedToolSurface
from ora.agent.composition.prompt.context_pack import (
    BINDING_SCHEMA, MANIFEST_NAME, ContextPackError, canonical_sha256, compact_system_text,
    context_management_off, dropped_tools, episode_task_template, load_bound_pack, load_context_pack, load_tool_overlay,
    overlay_tools, resolve_profile,
)
from ora.agent.context.lossless.management import APPEND_ONLY_SCHEMAS
from ora.agent.memory.artifacts import LOSSLESS_CONTEXT_RETENTION
from ora.agent.session_core.constants import NATIVE_CAUSAL_SLOTS

from .values import RunSpecError

#: What a RunSpec names; the resolver writes the binding in their place.
INPUT_FIELDS = frozenset({"path", "path_env", "manifest_sha256", "profile"})


def _pack_root(raw: Mapping[str, Any], environ: Mapping[str, str]) -> tuple[Path, str | None]:
    if ("path" in raw) == ("path_env" in raw):
        raise RunSpecError("policy.context_pack names its directory by exactly one of path or path_env")
    if "path" in raw:
        value, variable = raw["path"], None
    else:
        variable = raw["path_env"]
        if not isinstance(variable, str) or not variable:
            raise RunSpecError("policy.context_pack.path_env must name an environment variable")
        value = environ.get(variable)
        if not value:
            raise RunSpecError(f"policy.context_pack.path_env names {variable}, which is not set")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise RunSpecError("policy.context_pack's directory must be an absolute path")
    return Path(value), variable


def resolve_context_pack(
    policy: dict[str, Any], resolved_surface: ResolvedToolSurface, *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Replace ``policy["context_pack"]`` by its binding, or refuse it."""

    raw = policy["context_pack"]
    if not isinstance(raw, Mapping) or set(raw) - INPUT_FIELDS or "manifest_sha256" not in raw:
        raise RunSpecError(
            "policy.context_pack takes path or path_env, manifest_sha256 and an optional profile"
        )
    if policy.get("agent_core") != "pi":
        raise RunSpecError(
            "policy.context_pack runs on the Pi agent core only (policy.agent_core: pi)"
        )
    management = policy.get("context_management")
    if isinstance(management, Mapping) and management.get("schema") in APPEND_ONLY_SCHEMAS:
        # Both rewrite what each request delivers; their views were never composed.
        raise RunSpecError(
            f"policy.context_pack needs Context management v4-v7; {management['schema']} has its own delivery"
        )
    root, variable = _pack_root(raw, os.environ if environ is None else environ)
    try:
        pack = load_context_pack(root, raw["manifest_sha256"])
        profile, source = resolve_profile(pack, raw.get("profile"))
    except ContextPackError as exc:
        raise RunSpecError(str(exc)) from None
    if ("system" in profile or profile.get("episode_task") == "task_and_contract_v1") \
            and policy.get("context_composition") not in NATIVE_CAUSAL_SLOTS:
        raise RunSpecError(
            "a context pack's system or episode_task delivery needs a typed slot composition "
            "(policy.context_composition " + " or ".join(sorted(NATIVE_CAUSAL_SLOTS)) + ")"
        )
    switched_off = context_management_off(profile)
    if switched_off and policy.get("context_retention") != LOSSLESS_CONTEXT_RETENTION:
        raise RunSpecError(
            f"a context pack's context_management switches ({', '.join(switched_off)}) act on "
            f"{LOSSLESS_CONTEXT_RETENTION} only; this plan's retention is {policy.get('context_retention')}"
        )
    drop = dropped_tools(profile)
    protected = set(resolved_surface.mutation_tools) | set(resolved_surface.lifecycle_tools)
    if protected & set(drop):
        raise RunSpecError(
            "a context pack cannot withhold the surface's mutation or lifecycle tools: "
            + ", ".join(sorted(protected & set(drop)))
        )
    policy["tool_allowlist"] = [name for name in policy["tool_allowlist"] if name not in set(drop)]
    policy["context_pack"] = {
        "schema": BINDING_SCHEMA,
        "name": pack.name,
        "manifest": {"path": str(root / MANIFEST_NAME), "sha256": pack.manifest_sha256},
        "path_env": variable,
        "files": dict(sorted(pack.files.items())),
        "profile_source": source,
        "profile": profile,
    }


def _layer_text(root: Path, layer: Mapping[str, Any]) -> str:
    return (root / layer["path"]).read_text(encoding="utf-8")


def bind_context_pack_delivery(
    policy: dict[str, Any], policy_context: Mapping[str, Any], *, root: Path,
) -> None:
    """Add the texts the profile composes from this plan's bound prompt layers.

    ``system``: the pack's core text and the named sections of the compiled
    embodiment card, the last ``project_instructions`` layer of a typed slot
    composition. ``task_and_contract_v1``: ``{{TASK}}`` and every
    ``episode_task`` layer after the template (the public completion
    contract), as the channel joins them.
    """

    binding = policy["context_pack"]
    profile = binding["profile"]
    try:
        pack = load_bound_pack(binding)
        if "system" in profile:
            card = policy_context["project_instructions"][-1]
            text, kept = compact_system_text(
                pack.read_text(profile["system"]["core_file"]), _layer_text(root, card),
                profile["system"]["embodiment_sections"],
            )
            binding["system"] = {
                "text": text, "embodiment_card": copy.deepcopy(dict(card)), "embodiment_sections_kept": kept,
            }
        if profile.get("episode_task") == "task_and_contract_v1":
            layers = policy_context["episode_task"]
            if _layer_text(root, layers[0]).count("{{TASK}}") != 1:
                raise ContextPackError("the episode-task template layer must bind {{TASK}} once")
            contracts = [copy.deepcopy(dict(layer)) for layer in layers[1:]]
            binding["episode_task"] = {
                "template": episode_task_template([_layer_text(root, layer) for layer in contracts]),
                "contract_layers": contracts,
            }
    except ContextPackError as exc:
        raise RunSpecError(str(exc)) from None


def wire_tool_definitions(
    definitions: Sequence[Mapping[str, Any]], capabilities: Sequence[str],
) -> list[dict[str, Any]]:
    """The function tools the Provider receives for these advertised definitions (``NativeToolGateway``)."""

    from ora.agent.composition.surface.null_optional_arguments import deliver_tool_definitions
    from ora.agent.session_core.gateway import openai_tool_definitions

    delivered = deliver_tool_definitions(list(definitions), tuple(capabilities))
    return openai_tool_definitions([item["name"] for item in delivered], definitions=delivered)


def context_pack_tool_layer(
    policy: Mapping[str, Any], definitions: Sequence[Mapping[str, Any]], capabilities: Sequence[str],
) -> dict[str, Any] | None:
    """The derived layer of a pack's tool overlay; refuses a tool the overlay does not cover."""

    binding = policy.get("context_pack")
    tools = (binding or {}).get("profile", {}).get("tools") or {}
    if not binding or "overlay_file" not in tools:
        return None
    wire = wire_tool_definitions(definitions, capabilities)
    try:
        pack = load_bound_pack(binding)
        delivered = overlay_tools(wire, load_tool_overlay(pack, tools["overlay_file"]))
    except ContextPackError as exc:
        raise RunSpecError(str(exc)) from None
    return {
        "kind": "context_pack_tool_overlay_v1",
        "overlay_file": tools["overlay_file"],
        "withheld": list(tools.get("drop", [])),
        "tools": [item["function"]["name"] for item in delivered],
        "source_tools_sha256": canonical_sha256(wire),
        "delivered_tools_sha256": canonical_sha256(delivered),
    }


__all__ = [
    "INPUT_FIELDS", "bind_context_pack_delivery", "context_pack_tool_layer", "resolve_context_pack",
    "wire_tool_definitions",
]
