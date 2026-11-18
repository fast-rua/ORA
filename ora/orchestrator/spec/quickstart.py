"""The Synthetic quickstart's RunSpec, and its provider from the ORA_* environment.

``synthetic_spec`` is the bounded one-action Synthetic episode ``ora run`` launches
(``ora.app.quickstart``); ``ora interactive`` and ``ora play`` lengthen it, and the
Synthetic backend's free play reuses its drawn world and policy layers.
``environment_policy`` is the quickstart's ``native_api`` policy from
``ORA_PROVIDER``, ``ORA_BASE_URL``, ``ORA_MODEL`` and the key variable, the
environment's counterpart of ``inputs.provider_policy``.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

#: Synthetic's ToolSurface. ``ora run --env synthetic``, ``ora interactive`` and ``ora play --env
#: synthetic`` name it here, so none of them reaches the RunSpec fallback (``spec.policy``) by accident.
TOOL_SURFACE = "robot_runtime_v2"


def environment_policy() -> dict[str, Any]:
    """The quickstart's native_api policy from ORA_PROVIDER, ORA_BASE_URL, ORA_MODEL and the key."""
    protocol = os.environ.get("ORA_PROVIDER", "openai_compatible")
    if protocol not in {"openai_compatible", "anthropic_messages"}:
        raise ValueError("ORA_PROVIDER must be openai_compatible or anthropic_messages")
    fallback_key = "ANTHROPIC_API_KEY" if protocol == "anthropic_messages" else "OPENAI_API_KEY"
    key_env = "ORA_API_KEY" if os.environ.get("ORA_API_KEY") else fallback_key
    if not os.environ.get(key_env):
        raise ValueError(f"Set ORA_API_KEY or {fallback_key} before running the quickstart")
    default_url = "https://api.anthropic.com" if protocol == "anthropic_messages" else "https://api.openai.com/v1"
    endpoint = os.environ.setdefault("ORA_BASE_URL", default_url).rstrip("/")
    model = os.environ.get("ORA_MODEL", "" if protocol == "anthropic_messages" else "gpt-4.1-mini")
    if not model:
        raise ValueError("Set ORA_MODEL to the model ID served by your endpoint")
    from urllib.parse import urlsplit
    parsed = urlsplit(endpoint)
    transport = "loopback_http_v1" if parsed.scheme == "http" and parsed.hostname in {
        "localhost", "127.0.0.1", "::1",
    } else "https_v1"
    return {
        "kind": "native_api", "provider": protocol, "model": model, "endpoint": endpoint,
        "endpoint_transport": transport, "api_key_env": key_env, "base_url_env": "ORA_BASE_URL",
        "context_window_tokens": 32768, "max_output_tokens": 2048,
        "supports_images": os.environ.get("ORA_SUPPORTS_IMAGES", "1") != "0",
        "stream": False, "require_live_marker": False,
        "reasoning_effort": os.environ.get("ORA_REASONING_EFFORT", "provider_default"),
    }


def synthetic_spec(policy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The bounded Synthetic RunSpec; `policy` defaults to the quickstart environment's."""
    return {
        "schema": "ora.run-spec.v1", "experiment": "quickstart", "config": "synthetic-native",
        "milestone": "M0", "task": "move-once", "seed": 0, "evidence_tier": "diagnostic",
        "output_workspace": "outputs/quickstart",
        "task_instruction": "Read the robot manual, then use robot.submit_action to move the left end "
            "effector to position_m [0, 0, 1.02] with orientation_xyzw [0, 0, 0, 1]. "
            "This Synthetic demonstration ends after one accepted action.",
        "policy": {"tool_surface": {"profile": TOOL_SURFACE, "component_overrides": {}},
                   **(dict(policy) if policy is not None else environment_policy())},
        "runtime": {"kind": "synthetic", "terminal_actions": 1,
                    "workspace_bounds_m": {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]},
                    # A drawn top view for the world view and process video (Control-only).
                    "execution_capture": {"enabled": True}},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 6,
                    "max_sim_steps": 1, "wall_time_s": 120},
        "policy_context": {
            "codex_project": ["ora/policy/planners/native/project-v1.md"],
            "episode_prefix": ["ora/policy/core/thin-react-v1.md"],
            "tool_manual": ["ora/policy/embodiments/aloha-agilex/manual-v1.md"],
        },
        "qualification": {"required": False},
    }


__all__ = ["TOOL_SURFACE", "environment_policy", "synthetic_spec"]
