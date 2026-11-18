"""Prompt layers and policy composition of a RunSpec.

``bind_policy_context`` binds the ordered prompt-layer files the policy reads
(``policy_context``) to their content hashes: the typed v3 channels of a
Native slot composition, or the three legacy managed channels.
``compose_policy`` then records what the policy is offered -- the advertised
ToolSurface with its derived layers and the action codec -- as one hashed
``policy_composition``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Collection, Mapping

from ora.agent.code_mode.contract import current_generation, script_reads
from ora.agent.composition import (
    POLICY_CONTEXT_V3_SCHEMA, V2_CHANNELS,
    ResolvedToolSurface,
)
from ora.agent.session_core.constants import COMPOSITION_CONTEXT_CAPABILITIES, NATIVE_CAUSAL_SLOTS
from ora.agent.context.delivery.round_static_facts import CAPABILITY as ROUND_STATIC_FACTS
from ora.agent.composition.embodiment.environment_context import (
    EnvironmentContextError,
    resolve_environment_prompt_profile,
)
from ora.agent.memory.artifacts import (
    context_artifact_reader,
    recoverable_artifact_definitions,
    recoverable_artifact_result_schemas,
)
from ora.agent.toolkit.host.runtime.base import manual_sections

from .environment_context import resolve_typed_environment_prompt_profile
from .context_pack import bind_context_pack_delivery, context_pack_tool_layer
from .values import RunSpecError, _mapping, _resource_list, json_sha256


def bind_policy_context(
    spec: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: dict[str, Any], resolved_surface: ResolvedToolSurface, *,
    profile_id: str, root: Path,
) -> dict[str, Any]:
    """Return the hash-bound prompt layers."""

    policy_context = _mapping(spec.get("policy_context"), "policy_context")
    if resolved_surface.supports("current_public_robot_facts_v1"):
        _require_public_geometry_delivery(policy_context, runtime, root=root)
    if policy_context.get("schema") == POLICY_CONTEXT_V3_SCHEMA:
        bound_policy_context = _bind_typed_policy_context(
            policy_context, policy, runtime, resolved_surface, root=root,
        )
    else:
        bound_policy_context = _bind_legacy_policy_context(
            policy_context, policy, root=root,
        )
    if policy.get("code_mode"):
        # Code mode replaces the planner and the evidence-action core: each states the one-call-per-reply
        # loop that code mode changes.
        from ora.agent.toolkit.host.motion import joint_trajectory_surface

        # The round's static facts are stated once: by the context scheme's prompt where the plan delivers it (its
        # last paragraph), else by their own layer after the code core.
        bound_policy_context = _code_policy_layers(
            bound_policy_context, root=root, prompt=current_generation(policy["code_mode"]),
            round_static_facts=ROUND_STATIC_FACTS in COMPOSITION_CONTEXT_CAPABILITIES.get(
                policy.get("context_composition"), ()) and not _delivers_scheme_prompt(
                    bound_policy_context, policy, root=root),
            trajectories=joint_trajectory_surface(resolved_surface), reads=script_reads(policy["code_mode"]),
            play=spec.get("episode_mode") == "play")
    bound_policy_context["composition_sha256"] = json_sha256(bound_policy_context)
    return bound_policy_context


def _delivers_scheme_prompt(bound: Mapping[str, Any], policy: Mapping[str, Any], *, root: Path) -> bool:
    """Whether a channel delivers the prompt of the plan's own context scheme (``policy.context_scheme``)."""

    record = policy.get("context_scheme")
    if not isinstance(record, Mapping) or not isinstance(record.get("id"), str):
        return False
    from ora.agent.composition.context_scheme import load_context_scheme

    prompt = load_context_scheme(root, record["id"])["prompt"]
    return prompt is not None and any(
        isinstance(layer, Mapping) and layer.get("path") == prompt
        for layers in bound.values() if isinstance(layers, list) for layer in layers)


#: ``policy.code_mode``: the code policy's planner and core, replacing the native tool-call planner and core (the
#: RoboDojo toolkit v3 bundle's, which its v2 surface brought, retired with them on 2026-11-01).
#: Keyed by the contract's ``prompt`` generation (``agent/code_mode/contract.py``): ``v2`` (closed loop); its core
#: leaves the round's static facts to their own layer; ``v3`` (``library: extended``) rewrites the planner and keeps
#: ``v2``'s core and every layer after it. (``v1``, what code-mode runs sealed before ``v2`` read, was retired on
#: 2026-10-31.)
CODE_MODE_PLANNERS = {"v2": "ora/policy/planners/native/code-mode-coding-v2.md",
                      "v3": "ora/policy/planners/native/code-mode-coding-v3.md"}
CODE_MODE_CORE = {"v2": "ora/policy/core/code-policy-loop-v2.md", "v3": "ora/policy/core/code-policy-loop-v2.md"}
TOOL_CALL_CORES = ("ora/policy/core/evidence-action-loop-v8.md",)
#: From ``v2``: the ``round_static_facts`` message is described in its own layer, right after the code core, only
#: where the context composition delivers it (``round_static_facts_slot_v1``: ``native_slots_v4``).
CODE_MODE_ROUND_STATIC_FACTS = {"v2": "ora/policy/core/round-static-facts-v1.md",
                                "v3": "ora/policy/core/round-static-facts-v1.md"}
TOOL_CALL_PLANNERS = ("ora/policy/planners/native/",)
#: From ``v2``, on a surface whose moves run as joint trajectories (toolkit v5,
#: ``toolkit.host.motion.joint_trajectory_surface``): the budget rules of a program there (a trajectory's cap per action,
#: moves fitted to what is left, undo per call) follow the code planner, whichever tool-call planner it replaced. (The
#: released-primitive budget layer went with toolkit v2's planner on 2026-11-01.)
CODE_MODE_TRAJECTORY_BUDGET = {"v2": "ora/policy/planners/native/code-mode-trajectory-budget-v1.md",
                               "v3": "ora/policy/planners/native/code-mode-trajectory-budget-v1.md"}
#: From ``v3``, the trigger lines of what a program can do only where its surface serves it (the contract's
#: ``script_reads``, ``agent/code_mode/contract.py``) follow the code planner there, each with the reads it needs: the
#: planner itself names none of them, so a model is never told of a read its surface refuses.
CODE_MODE_READ_LAYERS = {"v3": {"query": "ora/policy/planners/native/code-mode-query-v1.md",
                                "dry_run": "ora/policy/planners/native/code-mode-dry-run-v1.md"}}
#: Free play in code mode (``play.code_mode``, ``ora play --code-mode``): what the code planner's episode words mean with
#: an operator (the instruction is the objective, a reply without a tool call ends it, asking is allowed, no closing
#: step, the allowances are the instruction's), right after the planner and its read layers.
CODE_MODE_PLAY = {"v2": "ora/policy/planners/native/code-mode-play-v1.md",
                  "v3": "ora/policy/planners/native/code-mode-play-v1.md"}
#: From ``v2``, the code forms of the tool-call text RoboDojo's default (baseline ``robodojo_toolkit_v7``, bundle
#: ``prompt_bundles/toolkit_v7``) delivers besides a planner and ``TOOL_CALL_CORES``: the ``v9`` core becomes
#: ``code-policy-loop-v3.md`` (``v9`` leaves the round's static facts to the context scheme's prompt, so no
#: static-facts layer follows it); the surface's ``prompt.md`` (turn protocol, numbers from ``compute``, what the calls
#: spend) is left out on that joint-trajectory surface, where the code planner, the trajectory budget layer and the
#: code core state each of its rules for a program (``None``: no layer in its place).
_V2_CODE_FORMS = {"ora/policy/core/evidence-action-loop-v9.md": "ora/policy/core/code-policy-loop-v3.md",
                  "ora/policy/tool_surfaces/robodojo_placement_toolkit_v5/prompt.md": None,
                  "ora/policy/tool_surfaces/robodojo_placement_toolkit_v6/prompt.md": None}
CODE_FORMS = {"v2": _V2_CODE_FORMS, "v3": _V2_CODE_FORMS}
#: Tool-call text code mode has no code form of, where a channel other than the manifest-only ``tool_manual``
#: delivers it: an evidence-action core other than ``TOOL_CALL_CORES`` and ``CODE_FORMS``, a prompt bundle's own
#: planner or core and a file-backed surface's ``prompt.md`` other than ``CODE_FORMS``'s. Code mode refuses a plan that
#: delivers any, instead of composing a prompt that states both the code loop and the tool-call loop.
UNPORTED_TOOL_CALL_TEXT = ("ora/policy/core/evidence-action-loop-", "ora/policy/tool_surfaces/")


def _unported_tool_call_text(path: str) -> bool:
    """Tool-call text without a code form: ``UNPORTED_TOOL_CALL_TEXT`` or a prompt bundle's planner or core."""
    return path.startswith(UNPORTED_TOOL_CALL_TEXT) or (
        path.startswith("ora/policy/prompt_bundles/") and path.endswith(("/planner.md", "/core.md")))


def _code_policy_layers(bound: Mapping[str, Any], *, root: Path, prompt: str = "v2",
                        round_static_facts: bool = False, trajectories: bool = False,
                        reads: Collection[str] = (), play: bool = False) -> dict[str, Any]:
    """Code mode's prompt layers on any surface: the tool-call planner becomes the code planner and the
    evidence-action core its code form (``CODE_FORMS``: from ``v2`` also the ``v9`` core and surface v5's prompt);
    from ``v2``, on a surface whose moves are joint trajectories (``trajectories``), the trajectory budget rules follow
    the code planner whichever planner it replaced, the round's static facts follow the core only where
    ``round_static_facts`` (the composition delivers them), and from ``v3`` the layers of ``reads`` follow the planner;
    in free play (``play``) the free-play layer (``CODE_MODE_PLAY``) follows them.
    Delivered tool-call text that has no code form (``_unported_tool_call_text``) is refused."""

    planner = CODE_MODE_PLANNERS[prompt]
    forms = CODE_FORMS.get(prompt, {})
    budget = CODE_MODE_TRAJECTORY_BUDGET.get(prompt) if trajectories else None
    static_facts = CODE_MODE_ROUND_STATIC_FACTS.get(prompt) if round_static_facts else None
    read_layers = [path for name, path in CODE_MODE_READ_LAYERS.get(prompt, {}).items() if name in reads]
    if play:
        read_layers.append(CODE_MODE_PLAY[prompt])
    result = copy.deepcopy(dict(bound))
    placed = False
    unported = []
    for channel, layers in result.items():
        if not isinstance(layers, list):
            continue
        kept = []
        for layer in layers:
            path = layer.get("path") if isinstance(layer, Mapping) else None
            if isinstance(path, str) and path.startswith(TOOL_CALL_PLANNERS) and path.endswith(".md") \
                    and channel in ("harness_system", "codex_project") and not placed:
                kept.append(_resource_list(root, [planner], "policy.code_mode planner")[0])
                if budget is not None:
                    kept.append(_resource_list(root, [budget], "policy.code_mode toolkit budget")[0])
                kept.extend(_resource_list(root, read_layers, "policy.code_mode reads") if read_layers else ())
                placed = True
                continue
            if isinstance(path, str) and path in forms and (forms[path] is not None or trajectories):
                if forms[path] is not None:
                    kept.append(_resource_list(root, [forms[path]], "policy.code_mode core")[0])
                continue
            if isinstance(path, str) and path in TOOL_CALL_CORES:
                kept.append(_resource_list(root, [CODE_MODE_CORE[prompt]], "policy.code_mode core")[0])
                if static_facts is not None:
                    kept.append(_resource_list(root, [static_facts], "policy.code_mode round static facts")[0])
                continue
            if isinstance(path, str) and _unported_tool_call_text(path) and channel != "tool_manual":
                unported.append(path)
            kept.append(layer)
        result[channel] = kept
    if unported:
        raise RunSpecError(f"policy.code_mode has no code form of the tool-call text {', '.join(unported)} "
                           "(RoboDojo: robodojo_placement_toolkit_v6 with baseline robodojo_toolkit_v7 at "
                           "policy.code_mode.prompt v2 or v3)")
    if not placed:
        channel = "harness_system" if "harness_system" in result else "codex_project"
        first = [planner, *([budget] if budget is not None else []), *read_layers]
        result[channel] = [*_resource_list(root, first, "policy.code_mode planner"), *result[channel]]
    return result


def _require_public_geometry_delivery(
    policy_context: Mapping[str, Any], runtime: Mapping[str, Any], *,
    root: Path,
) -> None:
    """The public geometry must arrive through its compiled embodiment card."""

    binding = runtime.get("environment_prompt_profile")
    public = runtime.get("public_environment_context")
    if not isinstance(binding, Mapping) or not isinstance(public, Mapping):
        raise RunSpecError("public geometry delivery requires its reviewed environment prompt profile")
    try:
        geometry_profile = resolve_environment_prompt_profile(public, project_root=root, profile_id=binding.get("profile_id"))
    except EnvironmentContextError as exc:
        raise RunSpecError(str(exc)) from exc
    compiled = geometry_profile.get("compiled_embodiment")
    if not isinstance(compiled, Mapping) or compiled.get("descriptor") != runtime["public_geometry_descriptor"]:
        raise RunSpecError("public geometry ToolSurface and compiled embodiment must be selected together")
    project_sources = policy_context.get("project_instructions", policy_context.get("codex_project"))
    if not isinstance(project_sources, list) or geometry_profile["embodiment_policy"] not in project_sources:
        raise RunSpecError("public geometry descriptor must be delivered through its exact compiled embodiment card")


def _bind_typed_policy_context(
    policy_context: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: Mapping[str, Any], resolved_surface: ResolvedToolSurface, *,
    root: Path,
) -> dict[str, Any]:
    """Typed v3 channels of a Native slot composition."""

    expected_channels = {"schema", *V2_CHANNELS}
    if set(policy_context) != expected_channels:
        raise RunSpecError(
            f"typed policy_context fields must be exactly {sorted(expected_channels)}"
        )
    if policy["kind"] != "native_api":
        raise RunSpecError("typed policy_context is currently supported only by native_api")
    # A v3 policy_context binds either causal slot layout: native_slots_v4 is
    # native_slots_v3 plus the round static-facts slot.
    if policy.get("context_composition") not in NATIVE_CAUSAL_SLOTS:
        raise RunSpecError(
            "policy_context requires policy.context_composition="
            + " or ".join(sorted(NATIVE_CAUSAL_SLOTS))
        )
    _require_native_slots_v3(policy, runtime, resolved_surface)
    if not resolved_surface.observation_tools or not resolved_surface.call_effect_profile:
        raise RunSpecError(
            "typed native slots require a ToolSurface observation/effect contract"
        )
    resolved_environment_profile = resolve_typed_environment_prompt_profile(runtime, root=root)
    project_sources = policy_context["project_instructions"]
    expected_project_sources = [
        resolved_environment_profile["environment_policy"],
        resolved_environment_profile["embodiment_policy"],
    ]
    # The environment-policy/embodiment pair stays the protected tail;
    # a unified prompt bundle may deliver its shared core layer ahead of
    # it in the same project channel (unified bundle contract).
    project_sources_match = (
        isinstance(project_sources, list)
        and len(project_sources) >= 2
        and project_sources[-2:] == expected_project_sources
    )
    if not project_sources_match:
        raise RunSpecError(
            "policy_context project sources do not match the environment profile"
        )
    bound_policy_context = {
        "schema": POLICY_CONTEXT_V3_SCHEMA,
        **{
            channel: _resource_list(
                root, policy_context[channel], f"policy_context.{channel}"
            )
            for channel in V2_CHANNELS
        },
    }
    return bound_policy_context


def _require_native_slots_v3(
    policy: Mapping[str, Any], runtime: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface,
) -> None:
    """native_slots_v3 binds the observation, codec and end-effector contract."""

    if policy.get("initial_observation") != "automatic":
        raise RunSpecError("native_slots_v3 requires initial_observation=automatic")
    if policy.get("context_profile") != "native_causal_context_v1":
        raise RunSpecError(
            "native_slots_v3 requires context_profile=native_causal_context_v1"
        )
    expected_codec = (
        "motion_action_v3"
        if resolved_surface.supports("motion_intent_router_v1")
        else "runtime_action_v3"
    )
    if policy.get("action_codec") != expected_codec:
        raise RunSpecError(
            f"native_slots_v3 requires action_codec={expected_codec}"
        )
    dual_arm_motion = (
        resolved_surface.supports("enhanced_move_pose_v1")
        and resolved_surface.supports("dual_arm_sparse_targets_v1")
    )
    if (
        resolved_surface.runtime_id != "native_atomic_causal"
        and not dual_arm_motion
    ):
        raise RunSpecError(
            "native_slots_v3 requires a native_atomic_causal or explicit "
            "dual-arm two-layer ToolSurface"
        )
    if tuple(resolved_surface.mutation_tools) != (
        "move_pose", "close_gripper", "open_gripper",
    ) and not resolved_surface.supports("enhanced_move_pose_v1"):
        raise RunSpecError(
            "native_slots_v3 requires an atomic or registered motion contract"
        )
    if dual_arm_motion and runtime.get("controlled_end_effector") is not None:
        raise RunSpecError(
            "dual-arm Native motion selects the end effector per tool call"
        )
    if not dual_arm_motion and runtime.get("controlled_end_effector") != "right":
        raise RunSpecError(
            "native_slots_v3 requires environment-bound controlled_end_effector=right"
        )


def _bind_legacy_policy_context(
    policy_context: Mapping[str, Any], policy: Mapping[str, Any], *,
    root: Path,
) -> dict[str, Any]:
    """The legacy managed channels: project, episode prefix and tool manual."""

    expected_channels = {"codex_project", "episode_prefix", "tool_manual"}
    if set(policy_context) != expected_channels:
        raise RunSpecError(
            f"policy_context channels must be exactly {sorted(expected_channels)}"
        )
    if policy.get("context_composition") is not None:
        raise RunSpecError("legacy policy_context must not set context_composition")
    bound_policy_context = {
        channel: _resource_list(
            root, policy_context[channel], f"policy_context.{channel}"
        )
        for channel in sorted(expected_channels)
    }
    return bound_policy_context


def compose_policy(
    policy: Mapping[str, Any], resolved_surface: ResolvedToolSurface,
    bound_policy_context: Mapping[str, Any], *, root: Path,
) -> dict[str, Any]:
    """Hash what the policy is offered: prompt layers, tools and codec."""

    if "context_pack" in policy:  # only a plan that names one carries the key
        if bound_policy_context.get("schema") != POLICY_CONTEXT_V3_SCHEMA:
            raise RunSpecError("policy.context_pack needs typed policy_context channels")
        bind_context_pack_delivery(policy, bound_policy_context, root=root)
    _, tool_surface_snapshot = _advertised_tool_surface(
        policy, resolved_surface, bound_policy_context, root=root,
    )
    if "context_pack" in policy:  # the Pi core checks its overlaid tools against this hash
        for layer in tool_surface_snapshot["derived_layers"]:
            if layer["kind"] == "context_pack_tool_overlay_v1":
                policy["context_pack"]["delivered_tools_sha256"] = layer["delivered_tools_sha256"]
    policy_composition = {
        "schema": "ora.policy-composition.v2",
        "prompt_layers": bound_policy_context,
        "tool_surface": tool_surface_snapshot,
        "action_codec": policy["action_codec"],
        "initial_observation": policy.get("initial_observation"),
    }
    policy_composition["composition_sha256"] = json_sha256(policy_composition)
    return policy_composition


def _advertised_tool_surface(
    policy: Mapping[str, Any], resolved_surface: ResolvedToolSurface,
    bound_policy_context: Mapping[str, Any], *, root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The allowlisted definitions plus the layers derived from the plan.

    Lossless Context Artifacts add their reader tools (unless the context
    scheme's ``context_tools`` leaves them out), and
    ``robot.read_manual`` gets the composed manual's section headings as an
    enum. Returns the advertised definitions and the ToolSurface snapshot.
    """

    advertised_definitions = resolved_surface.select(policy.get("tool_allowlist"))
    derived_layers: list[dict[str, Any]] = []
    reader = context_artifact_reader(policy)
    if reader:
        from ora.agent.memory.artifacts import artifact_read_name

        _artifact_reader = artifact_read_name(advertised_definitions)
        advertised_definitions = recoverable_artifact_definitions(
            advertised_definitions
        )
        derived_layers.append({
            "kind": "recoverable_context_artifacts_v1",
            "tools": ["observe", _artifact_reader],
            "effects": {
                "observe(ref=...)": "read_only",
                _artifact_reader: "read_only",
            },
            "logical_ref": "artifact://<policy_session_id>/<sha256>",
        })
    if any(item["name"] == "robot.read_manual" for item in advertised_definitions):
        derived_layers.append(_manual_section_enum(
            advertised_definitions, bound_policy_context, root=root,
        ))
    code_mode = policy.get("code_mode")
    if code_mode:  # the code tools (agent/code_mode): every other robot tool becomes a library function
        from ora.agent.code_mode.coding import advertised
        from ora.agent.code_mode.contract import script_functions

        lifecycle = tuple(resolved_surface.lifecycle_tools)
        functions = script_functions(advertised_definitions, lifecycle)
        # The schemas the runtime reads for its own description (``runtime.py``), which must equal this one.
        results = (recoverable_artifact_result_schemas(resolved_surface.result_schemas) if reader
                   else resolved_surface.result_schemas)
        advertised_definitions = advertised(advertised_definitions, code_mode, lifecycle, results)
        derived_layers.append({  # "tool" is this layer's recorded name, part of every code-mode plan's identity
            "kind": "code_mode_v1", "tool": "run_python", "effect": "script", "contract": dict(code_mode),
            **({"script_functions": [item["name"] for item in functions]} if functions else {}),
        })
    overlay_layer = context_pack_tool_layer(
        policy, advertised_definitions, resolved_surface.capabilities,
    )
    if overlay_layer is not None:  # a context pack with a tool overlay
        derived_layers.append(overlay_layer)
    tool_surface_snapshot = copy.deepcopy(resolved_surface.snapshot())
    tool_surface_snapshot["definitions"] = advertised_definitions
    if reader:
        tool_surface_snapshot["result_schemas"] = (
            recoverable_artifact_result_schemas(
                tool_surface_snapshot["result_schemas"]
            )
        )
    if code_mode:
        from ora.agent.code_mode.runtime import extra_result_schemas

        tool_surface_snapshot["result_schemas"] = {
            **tool_surface_snapshot["result_schemas"], **extra_result_schemas(code_mode),
        }
    tool_surface_snapshot["derived_layers"] = derived_layers
    tool_surface_snapshot["advertised_composition_sha256"] = json_sha256({
        "definitions": advertised_definitions,
        "result_schemas": tool_surface_snapshot["result_schemas"],
        "mutation_tools": tool_surface_snapshot["mutation_tools"],
        "lifecycle_tools": tool_surface_snapshot["lifecycle_tools"],
        **({
            "observation_tools": tool_surface_snapshot["observation_tools"],
            "call_effect_profile": tool_surface_snapshot["call_effect_profile"],
        } if "observation_tools" in tool_surface_snapshot else {}),
        "derived_layers": derived_layers,
    })
    return advertised_definitions, tool_surface_snapshot


def _manual_section_enum(
    advertised_definitions: list[dict[str, Any]],
    bound_policy_context: Mapping[str, Any], *, root: Path,
) -> dict[str, Any]:
    """Set the manual tool's section enum in place; return its derived layer."""

    sections: list[str] = []
    for resource in bound_policy_context["tool_manual"]:
        path = (root / resource["path"]).resolve()
        sections.extend(manual_sections(path))
    if len(sections) != len(set(sections)):
        raise RunSpecError("composed manual section headings must be unique")
    for definition in advertised_definitions:
        if definition["name"] == "robot.read_manual":
            definition["inputSchema"]["properties"]["section"]["enum"] = sections
    return {
        "kind": "manual_section_enum",
        "tool": "robot.read_manual",
        "sections": sections,
        "source_sha256": [
            item["sha256"] for item in bound_policy_context["tool_manual"]
        ],
    }
