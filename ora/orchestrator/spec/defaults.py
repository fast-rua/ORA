"""Atomic default composition for new Native robot RunSpecs.

Explicit ToolSurfaces retain their historical behavior. Baseline selection is
not host qualification, service startup or authority to launch an episode.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from ora.environment.benchmarks.libero_pro import PRACTICE_DRIVER as LIBERO_PRACTICE_DRIVER
from ora.agent.composition.baseline import (
    BASELINE_ID, BASELINE_CATALOGS, default_baseline_id, load_native_baseline,
)
from ora.agent.composition.context_scheme import CONTEXT_FIELDS, load_context_scheme

# Baselines whose RoboTwin contract admits either official scene configuration.
SCENE_CONFIGURATION_CHOICE_BASELINES = frozenset({BASELINE_ID})
# Native robot environments a baseline catalog may carry an instance for.
NATIVE_BASELINE_KINDS = frozenset({"libero_pro", "robotwin", "robodojo", "robocasa", "robolab"})
from ora.agent.composition.embodiment.environment_context import (
    resolve_environment_prompt_profile,
)
from ora.agent.composition.task_contract.delivery import (
    free_play_disclosure, resolve_task_contract_layer, task_file_layer,
)
from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.composition.surface.toolkit_surface import declared_required_grants, surface_prompt_sources
from ora.environment.definitions import resolve_runtime_descriptor

from ..backends.registry import find
from .native_provider import DEADLINE_RECOVERIES


#: The episode-task layer of a free-play session: operator instructions in turn, and messages the
#: operator sends while one runs, no task (v1, before messages joined a running instruction, stays
#: for the plans that bound it).
PLAY_EPISODE_TASK = "ora/policy/prompt_bundles/templates/episode-task-play-v2.md"
#: Tools a free-play session does not advertise: there is no task to check.
PLAY_WITHHELD_TOOLS = frozenset({"check_task"})


def play_tool_allowlist(names: list[str], surface: Any) -> list[str]:
    """The surface's ordered tools without the task checker and the lifecycle (finish) tools."""
    withheld = PLAY_WITHHELD_TOOLS | set(getattr(surface, "lifecycle_tools", ()) or ())
    return [name for name in names if name not in withheld]


def _fill(target: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(target[key], dict):
            _fill(target[key], value)


#: The official LIBERO-Pro driver a baseline instance names.
LIBERO_OFFICIAL_DRIVER = "direct_libero_v1"


def _check_runtime_contract(
    runtime: Mapping[str, Any], template: Mapping[str, Any],
    prefix: str = "runtime",
) -> None:
    for key, expected in template.items():
        # The timeouts and the execution host are launch bindings: a baseline
        # instance names a default, the RunSpec may bind another registered one.
        if key not in runtime or key in {"startup_timeout_s", "request_timeout_s", "required_host"}:
            continue
        actual = runtime[key]
        if (key == "environment_driver" and expected == LIBERO_OFFICIAL_DRIVER
                and actual == LIBERO_PRACTICE_DRIVER):
            # A LIBERO-Pro practice plan (partition practice) samples its initial
            # state from the seed; it is the one driver a baseline instance admits
            # in place of the official one. partition checks the seed side.
            continue
        if isinstance(expected, Mapping) and isinstance(actual, Mapping):
            _check_runtime_contract(actual, expected, f"{prefix}.{key}")
        elif actual != expected:
            raise ValueError(f"policy.baseline conflicts with {prefix}.{key}")


def expand_native_defaults(spec: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Expand one selected baseline without mutating caller-owned input."""
    result = copy.deepcopy(dict(spec))
    policy, runtime = result.get("policy"), result.get("runtime")
    if not isinstance(policy, dict) or not isinstance(runtime, dict):
        return result
    from ora.agent.composition.surface.toolkit_profiles import TOOLKIT_BASELINES

    explicit = policy.get("tool_surface")
    profile = explicit.get("profile") if isinstance(explicit, Mapping) else None
    if profile in TOOLKIT_BASELINES:
        policy.setdefault("baseline", TOOLKIT_BASELINES[profile])
    requested = policy.get("baseline")
    if "baseline" in policy and (not isinstance(requested, str) or requested not in BASELINE_CATALOGS):
        raise ValueError("policy.baseline must be a registered id string")
    # Native robot environments a baseline catalog may carry an instance for.
    selected = find(runtime.get("kind"))
    eligible = (policy.get("kind") == "native_api" and selected is not None
                and selected.capabilities.baseline_instance)
    if requested is not None and not eligible:
        raise ValueError("policy.baseline requires a supported Native robot environment")
    if not eligible or (requested is None and "tool_surface" in policy):
        if "context_scheme" in policy:
            raise ValueError("policy.context_scheme replaces a baseline's context; select a baseline")
        return result
    catalog = load_native_baseline(root, requested or default_baseline_id(runtime.get("kind")))
    # A free-play session (``ora play``) has no task to check or finish: its allowlist
    # is the complete ordered list without the task's checker and lifecycle tools.
    play = "play" in result
    # ``play.code_mode`` (``ora play --code-mode``): free play in the baseline's code mode, in place of Pi's own tools.
    play_code_mode = play and isinstance(result.get("play"), Mapping) and result["play"].get("code_mode") is True
    # A baseline may make code mode its default (``code_mode``: ``robodojo_toolkit_v7``, 2026-11-05). A RunSpec that
    # names policy.code_mode keeps its own, null included: null is tool calls, the opt-out. Free play has Pi's own
    # codemode unless it asks for code mode (``play.code_mode``), and a context pack composes with tool calls, so
    # neither gets it otherwise. The catalog alone decides, never what the host has installed: the sandbox is
    # preflight's to check (``code_mode_sandbox``).
    if (catalog.get("code_mode") is not None and "code_mode" not in policy and (not play or play_code_mode)
            and "context_pack" not in policy and runtime.get("kind") in catalog["instances"]):
        policy["code_mode"] = copy.deepcopy(catalog["code_mode"])
    scheme = None
    # A baseline names its context half as a scheme, and may name another for code mode
    # (``code_mode_context_scheme``: a code-mode policy's calls complete no ActionStep for long stretches, the case
    # exchange aging answers); a plan with a context pack keeps the scheme packs compose with.
    named = policy.get("context_scheme")
    if named is None:
        code_mode_scheme = catalog.get("code_mode_context_scheme")
        # Free play keeps the baseline's own scheme: its context is Pi's (ora-pi/src/play/runtime.ts).
        named = (code_mode_scheme if policy.get("code_mode") and code_mode_scheme is not None
                 and "context_pack" not in policy and not play else catalog.get("context_scheme"))
    if named is not None:
        # A named context scheme replaces the baseline's whole context half; the
        # rest of the baseline (surface, prompts, codec, runtime) stays as it is.
        scheme = load_context_scheme(root, named.get("id") if isinstance(named, Mapping) else named)
        if isinstance(named, Mapping) and dict(named) != scheme["record"]:
            raise ValueError("policy.context_scheme record does not match its scheme file")
        policy["context_scheme"] = scheme["record"]
        catalog = {**catalog, "policy": {
            **{k: v for k, v in catalog["policy"].items() if k not in CONTEXT_FIELDS},
            **copy.deepcopy(scheme["policy"]),
        }}
    if runtime["kind"] not in catalog["instances"]:
        # A catalog without an instance for this environment leaves an
        # omitted-baseline RunSpec as written.
        if requested is not None:
            raise ValueError(f"policy.baseline {catalog['id']} has no {runtime['kind']} instance")
        return result
    instance = catalog["instances"][runtime["kind"]]
    surface = {"profile": instance["tool_surface"], "component_overrides": {}}
    if requested is None and "toolkit_grants" not in result:
        # An environment's default surface may require grants (RoboDojo's toolkit v5: check_task); a RunSpec that
        # names no baseline gets them with that default, as a spec builder fills them for a selected surface.
        grants = declared_required_grants(instance["tool_surface"])
        if grants is not None:
            result["toolkit_grants"] = grants
    bundle_ref = instance["prompt_bundle"]
    raw_surface = policy.get("tool_surface", surface)
    if not isinstance(raw_surface, Mapping) or (
        raw_surface.get("profile") != surface["profile"]
        or raw_surface.get("component_overrides", {}) != surface["component_overrides"]
        or set(raw_surface) - {"profile", "component_overrides"}
    ):
        raise ValueError("policy.baseline tool_surface must match its unmodified instance")
    # Baseline semantics are one unit. Custom treatments choose an explicit
    # ToolSurface instead of silently inheriting a misleading baseline label.
    for field in (
        "action_codec", "context_profile", "context_composition", "context_retention",
        "summary_policy", "memory_policy", "provider_continuation_replay",
        "initial_observation", "context_management",
        "require_live_marker",
        "provider_recovery",
    ):
        if field in catalog["policy"] and field in policy and policy[field] != catalog["policy"][field]:
            # Either deadline recovery may stand in for the other: it decides how a failed
            # call is retried, not what the policy sees, and a RunSpec copied from a plan
            # of the earlier default (same_credential_deadline_v1) still resolves.
            if field == "provider_recovery" and {policy[field], catalog["policy"][field]} <= DEADLINE_RECOVERIES:
                continue
            raise ValueError(f"policy.baseline conflicts with policy.{field}")
    if "policy_context" in result:
        raise ValueError(
            "policy.baseline owns policy_context; choose an explicit custom surface"
        )
    if policy.get("supports_images") is not True:
        raise ValueError("policy.baseline requires explicit supports_images=true")
    if play and "context_pack" in policy:
        raise ValueError("free play does not take policy.context_pack")
    if play and policy.get("code_mode") and not play_code_mode:
        raise ValueError("free play has Pi's own codemode; it takes code mode only through play.code_mode "
                         "(ora play --code-mode)")
    # The zero_shot_no_check diagnostic tier is the baseline without the task checker.
    no_check = result.get("result_tier") == "zero_shot_no_check" and not play
    if "tool_allowlist" in policy or play or no_check:
        resolved_surface = resolve_tool_surface(
            surface["profile"], component_overrides=surface["component_overrides"],
            toolkit_grants=result.get("toolkit_grants"),
        )
        names = [item["name"] for item in resolved_surface.definitions]
        if play:
            names = play_tool_allowlist(names, resolved_surface)
        elif no_check:
            names = [name for name in names if name != "check_task"]
        if policy.setdefault("tool_allowlist", names) != names:
            raise ValueError("policy.baseline requires its complete ordered tool_allowlist"
                             + (" (free play: without check_task and lifecycle tools)" if play else "")
                             + (" (zero_shot_no_check: without check_task)" if no_check else ""))
    _fill(policy, catalog["policy"])
    policy["tool_surface"] = surface
    template = dict(instance["runtime"])
    if selected.expand_template is not None:
        # The backend's own adjustments: RoboTwin's choice of either official
        # scene configuration.
        template = selected.expand_template(runtime, template, requested)
    _check_runtime_contract(runtime, template)
    _fill(runtime, template)
    # Resolve metadata from the actual controller/cameras, never from a
    # caller-supplied public-context assertion.
    unasserted_runtime = {
        key: value for key, value in runtime.items()
        if key != "public_environment_context"
    }
    descriptor = resolve_runtime_descriptor(
        unasserted_runtime, action_codec=policy["action_codec"],
    )
    public = {
        "schema": "ora.public-environment-context.v1",
        "environment_id": descriptor.environment_id,
        "embodiment_id": descriptor.embodiment_id,
        "robot_models": list(descriptor.robot_models),
        "controller": descriptor.controller,
        "public_cameras": list(descriptor.public_cameras),
    }
    binding = resolve_environment_prompt_profile(
        public, project_root=root, profile_id=instance["environment_prompt_profile"],
    )
    prompt_binding = {k: binding[k] for k in (
        "schema", "profile_id", "profile_path", "profile_sha256", "metadata_sha256",
    )}
    for key, expected in (
        ("public_environment_context", public), ("environment_prompt_profile", prompt_binding),
    ):
        if key in runtime and runtime[key] != expected:
            raise ValueError(f"policy.baseline conflicts with runtime.{key}")
        runtime[key] = expected
    bundle_path = root / bundle_ref
    bundle_raw = bundle_path.read_bytes()
    layers = yaml.safe_load(bundle_raw)["layers"]
    # The episode-task template already tells the policy to treat a separately
    # advertised completion contract as the objective. Advertise it here, or
    # record why it was withheld: an undelivered contract that leaves no trace
    # is indistinguishable from a task that never had one.
    if play:
        # Free play withholds the benchmark task with its contract (episode-task-play-v2.md).
        contract_layer, contract_disclosure = None, free_play_disclosure(runtime)
    else:
        contract_layer, contract_disclosure = (task_file_layer(policy, runtime, root=root)
                                               or resolve_task_contract_layer(runtime, root=root))
    # A bundle with the optional prompt layers delivers the surface's prompt and
    # the context scheme's prompt after the planner; otherwise the surface prompt
    # stays a manifest-only tool_manual record, and the scheme's prompt is
    # delivered there only in code mode, which composes its own planner and core.
    surface_prompts = surface_prompt_sources(surface["profile"], root)
    system_prompts = []
    if "surface_prompt" in layers:
        if not surface_prompts:
            raise ValueError(f"{bundle_ref} delivers a surface prompt; {surface['profile']} has no prompt text")
        system_prompts.extend(surface_prompts)
    if "context_prompt" in layers:
        if scheme is None or scheme["prompt"] is None:
            raise ValueError(f"{bundle_ref} delivers a context-scheme prompt; "
                             "select a context scheme that has one")
        # Free play runs Pi's own context (its compaction, ora-pi/src/play/context_images.ts), never the scheme's
        # stages, so the scheme's prompt would describe a context the session does not have.
        if not play:
            system_prompts.append(scheme["prompt"])
    elif policy.get("code_mode") and not play and scheme is not None and scheme["prompt"] is not None:
        system_prompts.append(scheme["prompt"])
    result["policy_context"] = {
        "schema": "ora.policy-context.v3",
        "harness_system": [*layers["planner"]["sources"], *system_prompts],
        "project_instructions": [
            *layers["core"]["sources"],
            binding["environment_policy"], binding["embodiment_policy"],
        ],
        "episode_task": [
            PLAY_EPISODE_TASK if play else layers["episode_task"]["source"],
            *([contract_layer] if contract_layer is not None else []),
        ],
        "tool_manual": [*layers["tool_manual"]["sources"],
                        *([] if "surface_prompt" in layers else surface_prompts)],
    }
    policy["task_contract"] = contract_disclosure
    scope = catalog["required_service_capabilities"]
    if "prerequisite" not in result:
        result["prerequisite"] = {
            "schema": "ora.model-service-set-binding.v2",
            "capability_scope": list(scope), "status": "unbound_preflight_required",
        }
    prerequisite = result["prerequisite"]
    if not isinstance(prerequisite, Mapping) or (
        prerequisite.get("schema") != "ora.model-service-set-binding.v2"
        or prerequisite.get("capability_scope") != scope
    ):
        raise ValueError("policy.baseline requires its complete resident-service prerequisite")
    if "services" in prerequisite and (
        set(prerequisite) != {"schema", "capability_scope", "services"}
        or not isinstance(prerequisite["services"], Mapping)
        or set(prerequisite["services"]) != {"segmentation", "grasp_proposals"}
        or not all(
            isinstance(child, Mapping) for child in prerequisite["services"].values()
        )
    ):
        raise ValueError(
            "policy.baseline resident-service prerequisite has an invalid service set"
        )
    policy["baseline"] = {
        "id": catalog["id"], "instance": instance["id"],
        "selection": "explicit" if requested is not None else "omitted_tool_surface",
        "path": catalog["path"], "sha256": catalog["sha256"],
        "prompt_bundle": {
            "path": bundle_ref,
            "sha256": hashlib.sha256(bundle_raw).hexdigest(),
        },
    }
    return result
