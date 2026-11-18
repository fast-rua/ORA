"""Runtime, budget and session sections of a RunSpec.

``resolve_runtime`` checks the fields every environment shares (resource
slot, transport, execution capture, action workspace and public
geometry), then the selected environment's own fields through its backend's
``resolve_runtime`` (``backends/registry.py``); what differs between kinds
here is read from the backend's capabilities. Several ToolSurface bindings depend on the
runtime, so it returns the ToolSurface those bindings produced alongside the
runtime. ``resolve_budgets`` and ``resolve_session`` follow, in that order.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition import ResolvedToolSurface
from ora.agent.session_core.constants import NATIVE_CAUSAL_SLOTS
from ora.contracts.workspace import WorkspaceError, normalize_workspace_bounds
from ora.environment.definitions import (
    SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
    sparse_motion_physics_refusal,
)
from ora.environment.workspace_profiles import resolve_workspace_profile

from ..backends.registry import BACKENDS, BackendCapabilities, backend, find
from ..preflight.resources import ExecutionResourceError, resolve_execution_resource
from .values import RunSpecError, _mapping, _positive_int

RUNTIME_KINDS = frozenset(BACKENDS)
ENVIRONMENT_TRANSPORTS = {"in_process_direct", "local_rpc"}
#: ``play_scene_v1`` loads a play episode's RoboTwin scene with its official
#: predicate withheld; only ``play.binding.bind_play_runtime`` selects it.
ENVIRONMENT_SIGNAL_POLICIES = {"lifecycle_only", "task_recoverability_v1", "play_scene_v1"}
SESSION_CADENCES = {"persistent_agent_k_rounds_v1"}


def resolve_runtime(
    spec: Mapping[str, Any], policy: dict[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str, root: Path,
) -> tuple[dict[str, Any], ResolvedToolSurface]:
    """Return the runtime copy and the ToolSurface bound to it."""

    runtime = dict(_mapping(spec.get("runtime"), "runtime"))
    if runtime.get("kind") not in RUNTIME_KINDS:
        raise RunSpecError("runtime.kind must be synthetic, robotwin, libero_pro, robodojo, robocasa or robolab")
    selected = backend(runtime["kind"])
    _bind_runtime_profile_paths(runtime, root=root)
    _require_sparse_motion_qualification(policy, runtime)
    _bind_runtime_policy_contracts(spec, policy, runtime, root=root)
    _bind_execution_resource(runtime)
    _resolve_transport_and_signals(runtime, task=task)
    raw_workspace_profile = runtime.get("workspace_profile")
    _bind_execution_capture(runtime)
    _resolve_workspace_bounds(runtime, raw_workspace_profile)
    resolved_surface = _bind_public_geometry(runtime, policy, resolved_surface)
    selected.resolve_runtime(
        runtime, policy, resolved_surface,
        task=task, seed=seed, task_instruction=task_instruction,
    )
    _validate_runtime_extension(runtime)
    _require_surface_runtime(policy, runtime, resolved_surface)
    return runtime, resolved_surface


def _capabilities(runtime: Mapping[str, Any]) -> BackendCapabilities:
    """The capabilities of the runtime's backend; an unregistered kind has the defaults."""

    found = find(runtime.get("kind"))
    return found.capabilities if found is not None else BackendCapabilities()


def _bind_runtime_profile_paths(runtime: dict[str, Any], *, root: Path) -> None:
    """A motion or control profile given by path becomes its ``{path, sha256}`` record.

    The record hashes the bytes read now. A RunSpec may instead carry the
    record form a RunPlan holds; the factory and preflight verify it.
    """

    from ora.agent.composition.baseline import RUNTIME_PROFILE_FIELDS
    from ora.agent.composition.locked_assets import locked_asset

    for field in RUNTIME_PROFILE_FIELDS:
        if isinstance(runtime.get(field), str):
            runtime[field] = locked_asset(
                root, runtime[field], label=f"runtime.{field}", error=RunSpecError,
            ).record()


def _require_sparse_motion_qualification(
    policy: Mapping[str, Any], runtime: Mapping[str, Any],
) -> None:
    if policy["action_codec"] in SPARSE_MOTION_QUALIFIED_ENVIRONMENTS and (
        runtime.get("kind")
        not in SPARSE_MOTION_QUALIFIED_ENVIRONMENTS[policy["action_codec"]]
    ):
        raise RunSpecError(
            f"sparse motion codec {policy['action_codec']} has no "
            "source-bound motion qualification for "
            f"{runtime.get('kind')}; qualified: "
            + ", ".join(
                SPARSE_MOTION_QUALIFIED_ENVIRONMENTS[policy["action_codec"]]
            )
        )
    # A row's scripted record also binds the physics device it ran on
    # (definitions.SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES).
    physics_refusal = sparse_motion_physics_refusal(policy["action_codec"], runtime)
    if physics_refusal is not None:
        raise RunSpecError(physics_refusal)


def _bind_runtime_policy_contracts(
    spec: Mapping[str, Any], policy: dict[str, Any], runtime: dict[str, Any],
    *, root: Path,
) -> None:
    """Bind the policy fields that need the runtime: the task contract."""

    if "semantic_manifest" in policy:
        raise RunSpecError("policy.semantic_manifest is not supported")
    # A deliverable contract that is not delivered, and leaves no disclosure,
    # reads exactly like a task that never had one. Asserted on the one path
    # every RunSpec takes rather than in the package that happened to omit it.
    from ora.agent.composition.task_contract.delivery import (
        TaskContractError, bind_task_contract,
    )

    try:
        bind_task_contract(spec, policy, runtime, root=root)
    except TaskContractError as exc:
        raise RunSpecError(str(exc)) from exc


def _bind_execution_resource(runtime: dict[str, Any]) -> None:
    """Replace the selected resource profile with its resolved slot binding."""

    if "resource_binding" in runtime:
        raise RunSpecError(
            "runtime.resource_binding is resolver-owned; select resource_profile"
        )
    if "resource_concurrency_override" in runtime:
        raise RunSpecError(
            "runtime.resource_concurrency_override is not an operator authority; "
            "use distinct source-tracked resource profiles or wait for the "
            "exclusive lease"
        )
    resource_profile = runtime.pop("resource_profile", None)
    if resource_profile is not None:
        try:
            runtime["resource_binding"] = resolve_execution_resource(
                resource_profile,
                runtime_kind=str(runtime["kind"]),
                required_host=(
                    str(runtime["required_host"])
                    if isinstance(runtime.get("required_host"), str)
                    else None
                ),
            )
        except ExecutionResourceError as exc:
            raise RunSpecError(str(exc)) from exc


def _resolve_transport_and_signals(runtime: dict[str, Any], *, task: str) -> None:
    runtime.setdefault("transport", "in_process_direct")
    if runtime["transport"] not in ENVIRONMENT_TRANSPORTS:
        raise RunSpecError(
            f"runtime.transport must be one of {sorted(ENVIRONMENT_TRANSPORTS)}"
        )
    runtime.setdefault("environment_signal_policy", "lifecycle_only")
    if runtime["environment_signal_policy"] not in ENVIRONMENT_SIGNAL_POLICIES:
        raise RunSpecError(
            "runtime.environment_signal_policy must be one of "
            f"{sorted(ENVIRONMENT_SIGNAL_POLICIES)}"
        )
    if runtime["environment_signal_policy"] not in _capabilities(runtime).signal_policies:
        raise RunSpecError(
            f"{runtime['environment_signal_policy']} is currently supported only by robotwin"
        )
    if (
        runtime["environment_signal_policy"] == "task_recoverability_v1"
        and task != "adjust_bottle"
    ):
        raise RunSpecError(
            "task_recoverability_v1 is currently defined only for adjust_bottle"
        )


def _bind_execution_capture(runtime: dict[str, Any]) -> None:
    """Resolve execution capture; refuse the settings of the retired Servo tool."""

    from ora.environment.execution_capture import resolve_execution_capture
    retired = sorted(key for key in ("servo_profile", "episode_physics_budget", "servo_worker_abi")
                     if key in runtime)
    if retired:
        raise RunSpecError(f"bounded Servo is not part of this tree; remove runtime.{', runtime.'.join(retired)}")
    selected = find(runtime["kind"])
    try:
        if selected is not None and selected.capabilities.captures_execution:
            runtime["execution_capture"] = resolve_execution_capture(runtime.get("execution_capture"))
        elif selected is not None and "execution_capture" in runtime:
            # A backend without captures_execution (Synthetic) draws its world only when the
            # RunSpec asks (spec.quickstart.synthetic_spec does).
            runtime["execution_capture"] = resolve_execution_capture(runtime["execution_capture"])
    except ValueError as exc:
        raise RunSpecError(str(exc)) from exc


def _resolve_workspace_bounds(
    runtime: dict[str, Any], raw_workspace_profile: Any,
) -> None:
    """Bind a named workspace profile, or normalize explicit bounds."""

    if raw_workspace_profile is not None:
        if "workspace_bounds_m" in runtime:
            raise RunSpecError(
                "runtime must select workspace_profile or workspace_bounds_m, not both"
            )
        if not isinstance(raw_workspace_profile, str) or not raw_workspace_profile:
            raise RunSpecError("runtime.workspace_profile must be a stable profile id")
        try:
            workspace_profile = resolve_workspace_profile(raw_workspace_profile)
        except ValueError as exc:
            raise RunSpecError(str(exc)) from exc
        if workspace_profile["environment_id"] != runtime["kind"]:
            raise RunSpecError("action workspace profile does not match runtime.kind")
        runtime["workspace_profile"] = workspace_profile
        runtime["workspace_bounds_m"] = workspace_profile["bounds_m"]
    else:
        bounds = _mapping(
            runtime.get("workspace_bounds_m"), "runtime.workspace_bounds_m"
        )
        try:
            runtime["workspace_bounds_m"] = normalize_workspace_bounds(bounds)
        except WorkspaceError as exc:
            raise RunSpecError(str(exc)) from exc


def _bind_public_geometry(
    runtime: dict[str, Any], policy: dict[str, Any],
    resolved_surface: ResolvedToolSurface,
) -> ResolvedToolSurface:
    """Publish the workspace bounds and any public geometry descriptor."""

    from ora.agent.composition.surface.public_parameters import bind_public_workspace
    resolved_surface = bind_public_workspace(resolved_surface, runtime["workspace_bounds_m"])
    policy["tool_surface"]["composition_sha256"] = resolved_surface.composition_sha256
    if resolved_surface.supports("current_public_robot_facts_v1"):
        geometry_descriptor = resolved_surface.view_config["public_geometry_descriptor"]
        if "public_geometry_descriptor" in runtime and runtime["public_geometry_descriptor"] != geometry_descriptor:
            raise RunSpecError("runtime public geometry must match the selected ToolSurface descriptor")
        runtime["public_geometry_descriptor"] = copy.deepcopy(geometry_descriptor)
    elif "public_geometry_descriptor" in runtime:
        raise RunSpecError("public geometry descriptor requires its complete delivery surface")
    return resolved_surface


def bind_native_action_limit(runtime: dict[str, Any], official: int, name: str) -> None:
    """Bind the task's official native-action limit (RoboCasa and RoboLab runtimes).

    The episode's official step limit (one native action per control tick)
    is what the policy's native-action allowance counts down
    (orchestrator/episode/native_allowance.py); a spec may restate it, never change it.
    """
    if runtime.setdefault("native_action_limit", official) != official:
        raise RunSpecError(f"{name} native_action_limit must equal the task's official limit {official}")


def bind_official_limit_multiple(runtime: dict[str, Any], default: int | None = None) -> None:
    """Default how far past the official limit an episode runs (``contracts/official_limit.py``)."""
    from ora.contracts.official_limit import DEFAULT_OFFICIAL_LIMIT_MULTIPLE, official_limit_multiple

    runtime.setdefault("official_limit_multiple", DEFAULT_OFFICIAL_LIMIT_MULTIPLE if default is None else default)
    try:
        official_limit_multiple(runtime)
    except ValueError as exc:
        raise RunSpecError(f"runtime.{exc}") from exc


def _validate_runtime_extension(runtime: Mapping[str, Any]) -> None:
    from ora.environment.definitions import (
        EnvironmentDefinitionError,
        validate_runtime_extension,
    )

    try:
        validate_runtime_extension(runtime)
    except EnvironmentDefinitionError as exc:
        raise RunSpecError(str(exc)) from exc


def _require_surface_runtime(
    policy: Mapping[str, Any], runtime: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface,
) -> None:
    """Runtime requirements a ToolSurface or a Native benchmark imposes."""

    capabilities = _capabilities(runtime)
    if (
        policy["kind"] == "native_api"
        and capabilities.native_requires_images
        and policy["supports_images"] is not True
    ):
        raise RunSpecError("native robot benchmark policies require supports_images=true")


#: The robot decisions of a synchronous-toolkit plan whose RunSpec names no ``budgets.max_robot_decisions``; a
#: named budget is honoured. A free-play session on that surface keeps at most this many (``ora play`` names 400).
SYNCHRONOUS_TOOLKIT_DECISIONS = 100


def resolve_budgets(
    spec: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: Mapping[str, Any], *, synchronous_tools: bool,
) -> dict[str, Any]:
    budgets = dict(_mapping(spec.get("budgets"), "budgets"))
    if synchronous_tools:
        budgets.setdefault("max_robot_decisions", SYNCHRONOUS_TOOLKIT_DECISIONS)
    for field in ("max_robot_decisions", "max_model_calls", "max_sim_steps", "wall_time_s"):
        budgets[field] = _positive_int(budgets.get(field), f"budgets.{field}")
    if synchronous_tools and spec.get("episode_mode") == "play":
        budgets["max_robot_decisions"] = min(SYNCHRONOUS_TOOLKIT_DECISIONS, budgets["max_robot_decisions"])
    if runtime["kind"] == "synthetic" and runtime["terminal_actions"] > budgets["max_sim_steps"]:
        raise RunSpecError("runtime.terminal_actions must not exceed max_sim_steps")
    return budgets


def resolve_session(
    spec: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: Mapping[str, Any], resolved_surface: ResolvedToolSurface,
    *, seed: int,
) -> dict[str, Any] | None:
    """The optional persistent multi-round session; ``None`` for one episode."""

    session = None
    if "session" in spec:
        session = _persistent_session(
            spec, policy, runtime, resolved_surface, seed=seed,
        )
    # A backend refuses a session (or a single episode) its runtime cannot run:
    # RoboDojo's CPU-physics lane never restarts a round in place.
    validate = backend(runtime["kind"]).validate_session
    refusal = validate(runtime, session) if validate is not None else None
    if refusal is not None:
        raise RunSpecError(refusal)
    return session


def _persistent_session(
    spec: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: Mapping[str, Any], resolved_surface: ResolvedToolSurface,
    *, seed: int,
) -> dict[str, Any]:
    raw_session = dict(_mapping(spec.get("session"), "session"))
    if not {"cadence", "rounds"}.issubset(raw_session) or not set(
        raw_session
    ).issubset({"cadence", "rounds", "round_seeds"}):
        raise RunSpecError(
            "session fields must be cadence, rounds, and optional round_seeds"
        )
    if raw_session.get("cadence") not in SESSION_CADENCES:
        raise RunSpecError(
            f"session.cadence must be one of {sorted(SESSION_CADENCES)}"
        )
    rounds = _positive_int(raw_session.get("rounds"), "session.rounds")
    # A one-round session (K=1) keeps the session wrapper,
    # result schema and per-round evidence; the Supervisor never restarts
    # after the last round, so it simply ends after round 0.
    if rounds < 1 or rounds > 32:
        raise RunSpecError("session.rounds must be in [1,32]")
    if policy["kind"] != "native_api":
        raise RunSpecError(
            "persistent multi-round sessions currently require policy.kind=native_api"
        )
    # RoboDojo joined on its scripted sparse-motion qualification, whose
    # round-isolation case resets the same plant into a second round and
    # re-checks the layout identities, verdict rotation and command
    # budget.
    if not _capabilities(runtime).persistent_session:
        raise RunSpecError(
            "persistent multi-round sessions currently support synthetic, "
            "libero_pro, robotwin and robodojo"
        )
    if policy.get("context_composition") not in NATIVE_CAUSAL_SLOTS:
        raise RunSpecError(
            "persistent multi-round sessions require native_slots_v3 or native_slots_v4"
        )
    if "finish_episode" not in resolved_surface.lifecycle_tools:
        raise RunSpecError(
            "persistent multi-round sessions require a finish_episode lifecycle tool"
        )
    round_seeds = raw_session.get("round_seeds")
    if round_seeds is not None:
        if not isinstance(round_seeds, list) or len(round_seeds) != rounds:
            raise RunSpecError(
                "session.round_seeds must contain exactly session.rounds values"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in round_seeds
        ):
            raise RunSpecError(
                "session.round_seeds values must be non-negative integers"
            )
        if round_seeds[0] != seed:
            raise RunSpecError(
                "session.round_seeds[0] must equal the top-level seed"
            )
    return {
        "cadence": raw_session["cadence"],
        "rounds": rounds,
        "budget_scope": "per_round",
        "reset_owner": "supervisor",
        **(
            {"round_seeds": list(round_seeds)}
            if round_seeds is not None else {}
        ),
    }
