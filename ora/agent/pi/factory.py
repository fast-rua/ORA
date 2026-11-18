"""Build the Pi-core policy session (``policy.agent_core: pi``, the default) from a resolved plan.

Resolves the plan's provider settings and credentials (the qualified credential
ring of ``ora.agent.providers.credentials``) and hands them to the TypeScript
core as one provider spec; the core runs the provider, not Python.
The caller resolves the Node.js binary and passes it as ``node``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ora.agent.contracts import ModelProvider, PolicySessionError
from ora.agent.session_core.policy_session import NativePolicySession
from ora.agent.providers.compactor_binding import compactor_api_key, session_affinity_id
from ora.agent.providers.credentials import credential_ring, qualified_credential_index

from .core_process import user_credential_store
from .policy_session import PiPlanner


def _provider_spec(plan: Mapping[str, Any], *, run_root: Path, run_id: str,
                   provider_override: ModelProvider | None) -> dict[str, Any]:
    policy = plan["policy"]
    if provider_override is not None:
        explicit = getattr(provider_override, "pi_provider_spec", None)
        if isinstance(explicit, Mapping):  # replay harnesses hand the TS provider its script
            return dict(explicit)
        responses = getattr(provider_override, "responses", None)
        if responses is None:
            raise PolicySessionError(
                f"the Pi core runs its provider in TypeScript and takes only a FakeProvider's scripted "
                f"responses or a pi_provider_spec, not a {type(provider_override).__name__} object")
        return {"kind": "fake", "responses": list(responses)}
    if policy["provider"] not in {"openai_compatible", "anthropic_messages"}:
        raise PolicySessionError(f"unsupported native provider protocol: {policy['provider']}")
    base_url = os.environ.get(policy["base_url_env"])
    api_key = os.environ.get(policy["api_key_env"])
    if not base_url or not api_key:
        raise PolicySessionError(
            f"native provider environment is incomplete: {policy['base_url_env']}, {policy['api_key_env']}"
        )
    if base_url.rstrip("/") != policy["endpoint"]:
        raise PolicySessionError(f"native provider endpoint drift: expected {policy['endpoint']}")
    standby = [item.strip() for item in os.environ.get(f"{policy['api_key_env']}S", "").split(",")
               if item.strip() and item.strip() != api_key]
    selection = None
    qualification_path = run_root / "preflight.json"
    formal_plan = bool(plan.get("plan_id") and plan.get("plan_sha256"))
    if policy.get("require_live_marker", True) and (formal_plan or qualification_path.exists()):
        if not qualification_path.is_file():
            raise PolicySessionError("native credential qualification report is missing")
        pool, _ = credential_ring(api_key, standby, selected_index=0)
        selection = qualified_credential_index(
            plan, json.loads(qualification_path.read_text(encoding="utf-8")), len(pool),
        )
    ring, start = credential_ring(
        api_key, standby, (plan.get("runtime") or {}).get("resource_binding", {}).get("slot_id"),
        selected_index=selection,
    )
    if policy["provider"] == "anthropic_messages":
        # build_provider's anthropic_messages arguments; standby keys travel by environment.
        return {
            "kind": "anthropic_messages",
            "endpoint": policy["endpoint"],
            "api_key": ring[0],
            "standby_api_keys": list(ring[1:]),
            "credential_start_index": start,
            "recovery_policy": policy.get("provider_recovery", "legacy_v1"),
            "recovery_window_s": policy.get("provider_recovery_window_s"),
            "timeout_s": float(policy.get("timeout_s", 900)),
            "auth_scheme": policy.get("auth_scheme"),
            "output_token_parameter": policy.get("output_token_parameter"),
            "tool_strictness": policy.get("tool_strictness"),
            "reasoning_field": policy.get("reasoning_field"),
            "continuation_replay": policy.get("provider_continuation_replay", "exact_v1"),
            "session_affinity_id": (
                f"ora:{run_id}:native"
                if policy.get("provider_session_affinity") == "x_session_id_run_v1" else None
            ),
        }
    return {
        "kind": "openai_compatible",
        "endpoint": policy["endpoint"],
        "api_key": ring[0],
        "recovery_policy": policy.get("provider_recovery", "legacy_v1"),
        "recovery_window_s": policy.get("provider_recovery_window_s"),
        "timeout_s": float(policy.get("timeout_s", 900)),
        "auth_scheme": policy.get("auth_scheme"),
        "output_token_parameter": policy.get("output_token_parameter"),
        "tool_strictness": policy.get("tool_strictness"),
        "reasoning_field": policy.get("reasoning_field"),
        "request_options": policy.get("request_options"),
        "session_affinity_id": (
            f"ora:{run_id}:native"
            if policy.get("provider_session_affinity") == "x_session_id_run_v1" else None
        ),
    }


def _compactor_spec(plan: Mapping[str, Any], *, run_id: str,
                    provider_override: ModelProvider | None) -> dict[str, Any] | None:
    """The provider spec of a compactor bound to its own profile (``policy.context_compactor``).

    The same shape as ``_provider_spec`` for one key, no standby keys and the
    compactor's own session-affinity id. None without a binding, and for a
    scripted or replayed session, whose compactor stays scripted. The key
    travels by environment (``ORA_PI_COMPACTOR_API_KEY``), never in the spec.
    """

    binding = plan["policy"].get("context_compactor")
    if binding is None or provider_override is not None:
        return None
    spec = {
        "kind": binding["provider"],
        "endpoint": binding["endpoint"],
        "api_key": compactor_api_key(binding),
        "recovery_policy": binding.get("provider_recovery", "legacy_v1"),
        "recovery_window_s": binding.get("provider_recovery_window_s"),
        "timeout_s": float(plan["policy"].get("timeout_s", 900)),
        "auth_scheme": binding.get("auth_scheme"),
        "output_token_parameter": binding.get("output_token_parameter"),
        "tool_strictness": None,
        "reasoning_field": binding.get("reasoning_field"),
        "session_affinity_id": session_affinity_id(binding, run_id),
    }
    if binding["provider"] == "anthropic_messages":
        spec.update(standby_api_keys=[], credential_start_index=0, continuation_replay="exact_v1")
    else:
        spec["request_options"] = binding.get("request_options")
    return spec


def build_pi_policy_session(*, plan: Mapping[str, Any], run_root: Path, run_id: str, tools: Any,
                            compiled_policy: Mapping[str, Path],
                            provider_override: ModelProvider | None = None,
                            node: str | None = None) -> NativePolicySession:
    policy = plan["policy"]
    planner = PiPlanner(
        run_root=run_root, run_id=run_id, tools=tools, policy=policy,
        provider=_provider_spec(plan, run_root=run_root, run_id=run_id, provider_override=provider_override),
        compactor_provider=_compactor_spec(plan, run_id=run_id, provider_override=provider_override),
        project_instructions_path=compiled_policy["codex_project"],
        episode_prefix_path=compiled_policy["episode_prefix"],
        harness_system_path=compiled_policy.get("harness_system"),
        task_id=str(plan["task"]),
        node=node, interactive_socket=os.environ.get("ORA_PI_ATTACH_SOCKET") or None,
        episode_mode=plan.get("episode_mode"),
        play=_play_init(plan, run_root),
    )
    return NativePolicySession(planner, supports_images=bool(policy["supports_images"]))


def _play_init(plan: Mapping[str, Any], run_root: Path) -> dict[str, Any] | None:
    """A free-play run's Pi runtime: the run's workspace, the user's Pi, a user-level credential store."""

    if plan.get("episode_mode") != "play":
        return None
    workspace = (Path(run_root) / "sandbox" / "workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    store = user_credential_store()
    store.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return {"workspace": str(workspace), "auth_path": str(store),
            "agent_dir": os.environ.get("PI_CODING_AGENT_DIR") or None,
            # The latest observations and tool results whose images a request carries (context_images.ts).
            "context_images": dict((plan.get("play") or {}).get("context_images") or {}) or None,
            # A real cell: a stop at the cell (or a guard) ends a script's later motions in that turn (robot_socket.py).
            **({"stop_scripts_on": ["guard_stop", "operator_declined"]}
               if (plan.get("runtime") or {}).get("rig_mode") in ("read", "on") else {}),
            # The next step predicted while a motion runs (play/pipeline.ts), when the plan asks for it.
            **({"pipeline_predict": True} if (plan.get("play") or {}).get("pipeline_predict") is True else {}),
            # The priority each model request carries, when the plan names one (driver.ts requestPriority).
            **({"request_priority": dict(plan["play"]["request_priority"])}
               if (plan.get("play") or {}).get("request_priority") else {}),
            # Code mode (plan play.code_mode): ORA's coding tools replace Pi's own and its codemode (runtime.ts).
            **({"code_mode": True} if (plan.get("play") or {}).get("code_mode") is True else {})}
