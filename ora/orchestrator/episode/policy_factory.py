"""Construct one policy session without coupling the Coordinator to a provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.agent.contracts import (
    ModelProvider, PolicySessionError,
)
from ora.agent.session_core.policy_session import NativePolicySession
from ora.agent.sessions import ScriptedPolicySession
from ora.agent.scripted import ScriptedPlanner
from ora.agent.composition.prompt.context_pack import verify_context_pack_binding

from . import agent_core


def ensure_policy_session(
    value: Any, *, kind: str, supports_images: bool,
) -> Any:
    """Enforce one typed PolicySession at the orchestrator boundary."""

    if hasattr(value, "capabilities") and hasattr(value, "accept_runtime_event"):
        return value
    if kind == "native_api":
        return NativePolicySession(value, supports_images=supports_images)
    return ScriptedPolicySession(value)


def policy_session_hints(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return orchestration hints without exposing backend selection branches."""

    kind = str(policy.get("kind", ""))
    redactions: list[str] = []
    if kind == "native_api":
        redactions = [
            str(policy["base_url_env"]),
            str(policy["api_key_env"]),
            *(str(item) for item in policy.get("proxy_env", [])),
        ]
        compactor = policy.get("context_compactor")
        if isinstance(compactor, Mapping):  # the compactor's own Provider variables
            redactions += [str(compactor["base_url_env"]), str(compactor["api_key_env"]),
                           *(str(item) for item in compactor.get("proxy_env", []))]
    return {
        "session_kind": kind,
        "supports_images": bool(policy.get("supports_images", False)),
        "redacted_environment_names": tuple(redactions),
    }


def build_policy_session(
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    run_id: str,
    workspace: Path,
    control_root: Path,
    events_path: Path,
    compiled_policy: Mapping[str, Path],
    tools: Any,
    state_root: Path,
    record_context_bundle: Any,
    provider_override: ModelProvider | None = None,
    compactor_provider_override: ModelProvider | None = None,
) -> Any:
    policy = plan["policy"]
    if policy["kind"] == "scripted":
        return ScriptedPolicySession(ScriptedPlanner(
            tools=tools,
            actions=policy["actions"],
            state_probe=bool(policy.get("state_probe", False)),
        ))
    if policy["kind"] != "native_api":
        raise PolicySessionError(f"unsupported policy kind: {policy['kind']}")
    try:
        core = agent_core.selected(policy)
        node = agent_core.require_pi_runtime() if core == "pi" else None
    except agent_core.AgentCoreError as exc:
        raise PolicySessionError(str(exc)) from None
    if "context_pack" in policy:  # refuse before any process starts
        verify_context_pack_binding(policy["context_pack"], core=core)
    if core == "pi":
        # The default core: TypeScript on Pi, tools and authority in Python (ora.agent.pi).
        from ora.agent.pi.factory import build_pi_policy_session

        return build_pi_policy_session(
            plan=plan, run_root=run_root, run_id=run_id, tools=tools,
            compiled_policy=compiled_policy, provider_override=provider_override, node=node,
        )
    # Unreachable: agent_core.selected names only the Pi core (the Python Native core left ORA).
    raise PolicySessionError(f"unsupported agent core: {core}")


__all__ = ["build_policy_session", "policy_session_hints"]
