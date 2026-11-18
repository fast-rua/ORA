"""The rollout compactor's own Provider, from a plan's ``policy.context_compactor`` binding.

The binding (``ora.orchestrator.spec_context_compactor``) records a profile's
wire fields and credential variable names. The Pi agent core builds the
compactor's provider from the spec ``ora.agent.pi.factory`` hands to TypeScript
(``build_compactor_provider`` built the retired Python Native core's). The key
comes from the binding's own ``api_key_env`` (the ``ora`` entry point fills it
from the user credential store when the environment leaves it unset); it is
never recorded. The provider has no standby keys and no retries, as the
policy-provider copy the compactor otherwise gets.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from ora.agent.contracts import ModelProvider, PolicySessionError
from ora.agent.providers import build_provider


def provider_binding(binding: Mapping[str, Any]) -> dict[str, str]:
    """What every ``context_compactor_*`` event of a bound compactor records."""

    return {key: str(binding[key]) for key in ("provider_profile", "model", "endpoint")}


def compactor_configuration(
    binding: Mapping[str, Any], environ: Mapping[str, str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Whether the environment can serve the binding; the detail names variables, never values.

    The key variable must be set. The base URL variable may be unset (the
    binding's endpoint is used, as ``ora run`` defaults a profile's); when set it
    must name that endpoint.
    """

    values = os.environ if environ is None else environ
    if not values.get(binding["api_key_env"]):
        return False, {"configured": False, "required_environment": [binding["api_key_env"]]}
    base_url = values.get(binding["base_url_env"])
    if base_url and base_url.rstrip("/") != binding["endpoint"]:
        return False, {"configured": True, "endpoint_matches_plan": False, "expected_endpoint": binding["endpoint"]}
    return True, {"configured": True, "endpoint_matches_plan": True}


def compactor_api_key(binding: Mapping[str, Any], environ: Mapping[str, str] | None = None) -> str:
    """The binding's key, or a refusal naming the variable or the endpoint drift."""

    values = os.environ if environ is None else environ
    configured, detail = compactor_configuration(binding, values)
    if not configured:
        if detail["configured"]:
            raise PolicySessionError(f"context compactor provider endpoint drift: expected {binding['endpoint']}")
        raise PolicySessionError(f"context compactor provider environment is incomplete: {binding['api_key_env']}")
    return str(values[binding["api_key_env"]])


def session_affinity_id(binding: Mapping[str, Any], run_id: str) -> str | None:
    """The compactor's own X-Session-ID: its prompt is not the policy's, so it shares no prefix."""

    return f"ora:{run_id}:compactor" if binding.get("provider_session_affinity") == "x_session_id_run_v1" else None


def build_compactor_provider(
    binding: Mapping[str, Any], *, run_id: str, api_key: str, timeout_s: float,
) -> ModelProvider:
    """The retired Native core's compactor provider: own client, no standby keys, no retries."""

    provider = build_provider(
        binding["provider"], base_url=binding["endpoint"], api_key=api_key,
        timeout_seconds=timeout_s,
        auth_scheme=binding.get("auth_scheme"),
        recovery_policy=binding.get("provider_recovery", "legacy_v1"),
        recovery_window_s=binding.get("provider_recovery_window_s"),
        output_token_parameter=binding.get("output_token_parameter"),
        reasoning_field=binding.get("reasoning_field"),
        request_options=binding.get("request_options"),
        session_affinity_id=session_affinity_id(binding, run_id),
    )
    provider.max_retries = 0
    return provider


__all__ = [
    "build_compactor_provider", "compactor_api_key", "compactor_configuration", "provider_binding",
    "session_affinity_id",
]
