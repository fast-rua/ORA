"""native_api policy fields of a RunSpec: tools, context and observation.

The Harness itself runs the native_api loop, so this resolver owns the fields
that shape each Provider request: the tool allowlist, the context profile and
retention (with its ``context_management`` budget), token limits, and the
observation/composition contract. The Provider protocol check runs first and
the endpoint check right after the tool allowlist; both live in
``native_provider.py``. Last, ``agent_core`` names the implementation
that runs the loop (``agent_core.py``: ``pi``; a RunSpec naming the retired
``native`` core is refused). ``code_mode`` resolves to the code-mode script
contract (``agent/code_mode/contract.py``) when a RunSpec asks for it.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.agent.composition import ResolvedToolSurface
from ora.agent.context.lossless.management import (
    APPEND_ONLY_SCHEMAS,
    EXCHANGE_AGING_SCHEMAS,
    POINTER_FREE_SCHEMAS,
    ContextManagementConfigError,
    validate_context_management,
)
from ora.agent.context.causal_value import (
    ACTION_STEP_RETENTIONS, CONTEXT_PROFILES, CONTEXT_RETENTIONS, SUMMARY_POLICIES,
)
from ora.agent.code_mode.contract import SEQUENCE_TOOLS, SYNCHRONOUS_EXECUTION
from ora.agent.code_mode.contract import resolve_contract as resolve_code_mode_contract
from ora.agent.code_mode.contract import with_reads
from ora.agent.memory.artifacts import CONTEXT_TOOLS, LOSSLESS_CONTEXT_RETENTION
from ora.agent.session_core.constants import (
    NATIVE_CAUSAL_SLOTS, NATIVE_SLOTS_V4,
)

from ..episode import agent_core
from ..episode.execution_state import per_action_capped
from .context_pack import resolve_context_pack
from .native_provider import (
    resolve_native_provider_endpoint,
    resolve_native_provider_protocol,
)
from .policy import require_tool_allowlist
from .values import RunSpecError, _positive_int


def resolve_native_policy(
    policy: dict[str, Any], resolved_surface: ResolvedToolSurface,
    runtime_spec: Any,
) -> None:
    """native_api: Provider, tools, context retention, budgets, observation, agent core."""

    resolve_native_provider_protocol(policy, runtime_spec)
    tools = policy.setdefault(
        "tool_allowlist", [item["name"] for item in resolved_surface.definitions]
    )
    require_tool_allowlist(tools, resolved_surface)
    resolve_native_provider_endpoint(policy)
    context_retention = _resolve_native_context(policy)
    _resolve_native_token_budgets(policy)
    _resolve_native_observation(policy, context_retention)
    try:
        agent_core.resolve(policy)
    except agent_core.AgentCoreError as exc:
        raise RunSpecError(str(exc)) from exc
    _resolve_code_mode(policy, resolved_surface, runtime_spec)
    if "context_pack" in policy:  # only a plan that names one carries the key
        resolve_context_pack(policy, resolved_surface)


def _resolve_code_mode(policy: dict[str, Any], resolved_surface: ResolvedToolSurface,
                       runtime_spec: Any = None) -> None:
    """``policy.code_mode``: off unless a RunSpec asks; then the code-mode script contract.

    A tool that commits a skill sequence (``grasp_execute`` on a closed-skill surface) runs its stage loop inside
    the script's parent invocation as one action (``episode/code_mode_execution.py``). On a backend that caps each
    action at one primitive's native budget (RoboDojo; ``episode/execution_state.per_action_capped``, the predicate
    the coordinator binds that cap by) the sequence cannot fit in one action, so its closed skills cannot run in code
    mode; its synchronous toolkit surface can (the default ``robodojo_placement_toolkit_v6``). A RunSpec that names
    no backend is refused the same way.
    """

    try:
        contract = resolve_code_mode_contract(policy.get("code_mode", False))
    except ValueError as exc:
        raise RunSpecError(str(exc)) from exc
    if contract is None:
        policy.pop("code_mode", None)
        return
    if policy.get("agent_core") != "pi":
        raise RunSpecError("policy.code_mode runs on the Pi agent core (policy.agent_core: pi)")
    sequences = SEQUENCE_TOOLS & {item["name"] for item in resolved_surface.definitions}
    kind = runtime_spec.get("kind") if isinstance(runtime_spec, Mapping) else None
    if (sequences and not resolved_surface.supports(SYNCHRONOUS_EXECUTION)
            and (kind is None or per_action_capped(kind))):
        raise RunSpecError(f"policy.code_mode needs a synchronous toolkit surface: {', '.join(sorted(sequences))} "
                           f"on {resolved_surface.profile_id} commits a skill sequence that one script action "
                           "cannot hold on this backend (RoboDojo: the default surface "
                           "robodojo_placement_toolkit_v6)")
    # What the surface serves a script beyond its tools. A query needs a worker that answers reach and kinematics
    # requests: a synchronous surface's does, and one whose environment plans its moves does not.
    from ora.agent.composition.surface.tool_surface import DRY_RUN_CAPABILITY

    policy["code_mode"] = with_reads(contract, toolkit=resolved_surface.supports(SYNCHRONOUS_EXECUTION),
                                     moves=resolved_surface.supports(DRY_RUN_CAPABILITY))


def _resolve_native_context(policy: dict[str, Any]) -> Any:
    """Context profile, retention and summary policy; returns the retention."""

    policy.setdefault("context_profile", "native_minimal_v1")
    if policy["context_profile"] not in CONTEXT_PROFILES:
        raise RunSpecError("unsupported native context profile")
    context_retention = policy.get("context_retention")
    if context_retention is not None and context_retention not in CONTEXT_RETENTIONS:
        raise RunSpecError("unsupported native context retention")
    if "context_token_estimator" in policy and (
        policy["context_token_estimator"] != "image_aware_v1"
        or context_retention not in (None, "rewrite_compact_v1")
    ):
        raise RunSpecError("context_token_estimator=image_aware_v1 requires rewrite_compact_v1")
    policy.setdefault("summary_policy", "disabled")
    if policy["summary_policy"] not in SUMMARY_POLICIES:
        raise RunSpecError(
            "Runtime v1 requires summary_policy=disabled or the explicit "
            "rollout_checkpoint_v1 treatment"
        )
    _resolve_context_management(policy, context_retention)
    _resolve_context_tools(policy, context_retention)
    return context_retention


def _resolve_context_management(
    policy: dict[str, Any], context_retention: Any,
) -> None:
    """Only an ActionStep retention takes ``context_management``."""

    if context_retention in ACTION_STEP_RETENTIONS:
        try:
            management = validate_context_management(policy.get("context_management"))
        except ContextManagementConfigError as exc:
            raise RunSpecError(f"policy.{exc}") from exc
        if policy["summary_policy"] != "rollout_checkpoint_v1":
            raise RunSpecError(
                f"{context_retention} requires summary_policy=rollout_checkpoint_v1"
            )
        policy["context_management"] = dict(management)
        if "context_stages" in policy:
            # A v2 scheme's stage list (ora.agent.composition.schemas.context_stages): the core runs it; the
            # evidence reads the management object.
            from ora.agent.composition.schemas.context_stages import check_stage_list

            try:
                policy["context_stages"] = check_stage_list(policy["context_stages"], management)
            except ValueError as exc:
                raise RunSpecError(str(exc)) from exc
    else:
        if policy.get("context_management") is not None or policy.get("context_stages") is not None:
            raise RunSpecError(
                "policy.context_management requires an ActionStep retention"
            )
        if policy["summary_policy"] != "disabled":
            raise RunSpecError(
                "Runtime v1 requires summary_policy=disabled outside "
                f"{LOSSLESS_CONTEXT_RETENTION}"
            )


def _resolve_context_tools(policy: dict[str, Any], context_retention: Any) -> None:
    """``context_tools``: the Context tools the context scheme provides.

    Absent, lossless retention provides the artifact reader as it always did.
    Append-only management (v8, v9) must say which tools it provides: v8 shows
    no artifact Pointer and cannot provide the reader, v9 ages exchanges into
    Pointers and must provide it.
    """

    management = policy.get("context_management")
    schema = management.get("schema") if isinstance(management, dict) else None
    append_only = schema in APPEND_ONLY_SCHEMAS
    if "context_tools" not in policy:
        if append_only:
            raise RunSpecError(
                f"{management['schema']} requires policy.context_tools (its context scheme names them)"
            )
        return
    tools = policy["context_tools"]
    if (
        not isinstance(tools, list) or any(not isinstance(item, str) for item in tools)
        or len(set(tools)) != len(tools) or set(tools) - CONTEXT_TOOLS
    ):
        raise RunSpecError(
            f"policy.context_tools must be a list of distinct names from {sorted(CONTEXT_TOOLS)}"
        )
    if "artifact_read" in tools and context_retention != LOSSLESS_CONTEXT_RETENTION:
        raise RunSpecError(
            f"policy.context_tools artifact_read requires context_retention={LOSSLESS_CONTEXT_RETENTION}"
        )
    if "artifact_read" in tools and schema in POINTER_FREE_SCHEMAS:
        raise RunSpecError(
            f"{management['schema']} shows no recoverable-artifact Pointer; "
            "policy.context_tools cannot provide artifact_read"
        )
    if "artifact_read" not in tools and schema in EXCHANGE_AGING_SCHEMAS:
        raise RunSpecError(
            f"{management['schema']} ages exchanges into recoverable-artifact Pointers; "
            "policy.context_tools must provide artifact_read"
        )


def _resolve_native_token_budgets(policy: dict[str, Any]) -> None:
    """Context window, output and per-observation call limits."""

    for field, default in (
        ("context_window_tokens", None),
        ("max_output_tokens", 8192),
        ("max_model_calls_per_observation", 12),
    ):
        if default is not None:
            policy.setdefault(field, default)
        policy[field] = _positive_int(policy.get(field), f"policy.{field}")
    if policy.get("max_inline_images_per_request") is not None:
        # Only the action_step_pressure_* retentions capped inline images;
        # they compose only at the commits that composed them.
        raise RunSpecError(
            "policy.max_inline_images_per_request is not supported by any "
            "context retention of this tree"
        )
    if policy["context_window_tokens"] <= policy["max_output_tokens"] + 2048:
        raise RunSpecError("native context window leaves no safe input budget")


def _resolve_native_observation(
    policy: dict[str, Any], context_retention: Any,
) -> None:
    """Streaming, images, first observation and the context composition."""

    for field, default in (("stream", True), ("supports_images", True)):
        policy.setdefault(field, default)
        if not isinstance(policy[field], bool):
            raise RunSpecError(f"policy.{field} must be boolean")
    policy.setdefault("initial_observation", "automatic")
    if policy["initial_observation"] not in {"automatic", "tool_pull"}:
        raise RunSpecError(
            "native_api policy.initial_observation must be automatic or tool_pull"
        )
    context_composition = policy.get("context_composition")
    if context_composition is not None and context_composition not in NATIVE_CAUSAL_SLOTS:
        raise RunSpecError(
            "native_api policy.context_composition must be native_slots_v3 or native_slots_v4"
        )
    if (
        context_retention in ACTION_STEP_RETENTIONS
        and context_composition not in NATIVE_CAUSAL_SLOTS
    ):
        raise RunSpecError(
            f"{context_retention} requires context_composition=native_slots_v3"
        )
    # round_static_facts_slot_v1 protects its slot through the lossless layers.
    if (
        context_composition == NATIVE_SLOTS_V4
        and context_retention != LOSSLESS_CONTEXT_RETENTION
    ):
        raise RunSpecError(
            f"native_slots_v4 requires context_retention={LOSSLESS_CONTEXT_RETENTION}"
        )
    if context_retention in ACTION_STEP_RETENTIONS:
        if policy["context_profile"] != "native_causal_context_v1":
            raise RunSpecError("ActionStep retention requires native causal context")
        if policy.get("initial_observation", "automatic") != "automatic":
            raise RunSpecError("ActionStep retention requires automatic observations")
        if policy.get("supports_images", True) is not True:
            raise RunSpecError("ActionStep retention requires an image-capable provider")
    if policy.get("fresh_observation_gate") is not None:
        raise RunSpecError("native_api policy.fresh_observation_gate is not supported")
    policy.setdefault("require_live_marker", True)
    if not isinstance(policy["require_live_marker"], bool):
        raise RunSpecError("policy.require_live_marker must be boolean")
