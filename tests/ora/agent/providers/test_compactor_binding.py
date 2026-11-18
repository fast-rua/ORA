"""The rollout compactor's own Provider, from a plan's ``policy.context_compactor`` binding.

Its key comes from the binding's own variable (the ``ora`` entry point fills it
from the user credential store), a set base URL variable must name the binding's
endpoint, and the provider built from it has its own client, no retries and its
own session-affinity id. The environment worker is never handed the binding's
variables. Preflight probes the bound endpoint (marker and image, no tool
continuity) only for a bound plan. A Synthetic episode with a bound compactor on
the Pi core is ``tests/agent_core/test_context_compactor_binding.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ora.agent.contracts import PolicySessionError
from ora.agent.providers import AnthropicMessagesProvider, OpenAICompatibleProvider
from ora.agent.providers import user_store
from ora.agent.providers.compactor_binding import (
    build_compactor_provider, compactor_api_key, compactor_configuration, provider_binding,
)
from ora.orchestrator.episode.policy_factory import policy_session_hints

KEY = "sk-fake-compactor-binding-unit-0123456789"
BINDING = {
    "schema": "ora.context-compactor-binding.v1", "provider_profile": "compactor-b", "profile_source": "user",
    "provider": "openai_compatible", "model": "model-b", "endpoint": "http://127.0.0.1:9/b/v1",
    "endpoint_transport": "loopback_http_v1", "base_url_env": "COMPACTOR_B_BASE_URL",
    "api_key_env": "COMPACTOR_B_API_KEY", "proxy_env": ["COMPACTOR_B_PROXY"], "reasoning_effort": "low",
    "stream": True, "supports_images": True, "context_window_tokens": 262144, "max_output_tokens": 16384,
    "provider_recovery": "legacy_v1", "provider_session_affinity": "x_session_id_run_v1",
    "request_options": {"temperature": 0.2},
}


def test_the_key_comes_from_the_bindings_own_variable_and_a_set_base_url_must_match():
    assert compactor_api_key(BINDING, {"COMPACTOR_B_API_KEY": KEY}) == KEY
    assert compactor_api_key(BINDING, {"COMPACTOR_B_API_KEY": KEY, "COMPACTOR_B_BASE_URL": BINDING["endpoint"] + "/"}) == KEY
    assert compactor_configuration(BINDING, {}) == (
        False, {"configured": False, "required_environment": ["COMPACTOR_B_API_KEY"]})
    with pytest.raises(PolicySessionError, match="incomplete: COMPACTOR_B_API_KEY"):
        compactor_api_key(BINDING, {"NATIVE_MODEL_API_KEY": KEY})
    with pytest.raises(PolicySessionError, match="endpoint drift") as refused:
        compactor_api_key(BINDING, {"COMPACTOR_B_API_KEY": KEY, "COMPACTOR_B_BASE_URL": "http://127.0.0.1:9/a/v1"})
    assert KEY not in str(refused.value)


def test_a_stored_key_serves_the_binding_as_the_entry_point_applies_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ORA_CONFIG_HOME", str(tmp_path))
    user_store.save_key("COMPACTOR_B_API_KEY", KEY)
    environment: dict[str, str] = {}
    assert user_store.apply_stored_credentials(environment) == ["COMPACTOR_B_API_KEY"]
    assert compactor_api_key(BINDING, environment) == KEY


def test_the_provider_has_its_own_client_no_retries_and_its_own_affinity():
    provider = build_compactor_provider(BINDING, run_id="run-1", api_key=KEY, timeout_s=120.0)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert (provider.base_url, provider.api_key, provider.max_retries) == (BINDING["endpoint"], KEY, 0)
    assert provider.session_affinity_id == "ora:run-1:compactor"
    assert provider.request_options == {"temperature": 0.2} and provider.timeout_seconds == 120.0
    anthropic = build_compactor_provider(
        {**BINDING, "provider": "anthropic_messages", "request_options": None, "provider_session_affinity": None},
        run_id="run-1", api_key=KEY, timeout_s=120.0)
    assert isinstance(anthropic, AnthropicMessagesProvider) and anthropic.max_retries == 0
    assert provider_binding(BINDING) == {
        "provider_profile": "compactor-b", "model": "model-b", "endpoint": "http://127.0.0.1:9/b/v1"}


def test_the_environment_worker_is_not_handed_the_bindings_variables():
    policy = {"kind": "native_api", "base_url_env": "A_BASE_URL", "api_key_env": "A_API_KEY", "proxy_env": [],
              "supports_images": True}
    assert policy_session_hints(policy)["redacted_environment_names"] == ("A_BASE_URL", "A_API_KEY")
    bound = policy_session_hints({**policy, "context_compactor": BINDING})["redacted_environment_names"]
    assert bound == ("A_BASE_URL", "A_API_KEY", "COMPACTOR_B_BASE_URL", "COMPACTOR_B_API_KEY", "COMPACTOR_B_PROXY")


class _ProbeProvider:
    """A fake probe provider: echoes the marker, names the red image, records each stage."""

    calls: list[tuple[str, str, str]] = []
    wrong_marker_once = False

    def __init__(self, protocol: str, **kwargs: Any) -> None:
        self.base_url, self.api_key, self.preflight_probe = kwargs["base_url"], kwargs["api_key"], False

    def close(self) -> None:
        return None

    def complete(self, *, model_call_id: str, request: dict[str, Any]) -> dict[str, Any]:
        _ProbeProvider.calls.append((self.base_url, request["model"], model_call_id))
        if model_call_id == "preflight-marker":
            content = request["messages"][0]["content"].rsplit(": ", 1)[-1]
            if _ProbeProvider.wrong_marker_once:
                _ProbeProvider.wrong_marker_once, content = False, "no marker"
        else:
            content = "red"
        return {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}


def _plan(**policy: Any) -> dict[str, Any]:
    return {"plan_id": "p", "plan_sha256": "a" * 64, "runtime": {}, "policy": {
        "kind": "native_api", "require_live_marker": True, "context_compactor": BINDING, **policy}}


def test_preflight_probes_the_bound_endpoint_with_marker_and_image_stages_only(monkeypatch):
    from ora.orchestrator.preflight.gates import _compactor_provider_marker

    monkeypatch.setattr("ora.agent.providers.build_provider", _ProbeProvider)
    _ProbeProvider.calls = []
    monkeypatch.delenv("COMPACTOR_B_API_KEY", raising=False)
    missing = _compactor_provider_marker(_plan())
    assert (missing["check"], missing["passed"]) == ("compactor_provider_marker", False)
    assert missing["detail"] == {"configured": False, "required_environment": ["COMPACTOR_B_API_KEY"]}
    assert _compactor_provider_marker(_plan(require_live_marker=False))["passed"] is True
    monkeypatch.setenv("COMPACTOR_B_API_KEY", KEY)
    monkeypatch.delenv("COMPACTOR_B_BASE_URL", raising=False)
    check = _compactor_provider_marker(_plan())
    assert check["passed"] is True, check
    assert _ProbeProvider.calls == [(BINDING["endpoint"], "model-b", "preflight-marker"),
                                    (BINDING["endpoint"], "model-b", "preflight-image")]
    detail = check["detail"]
    assert (detail["provider_profile"], detail["model"], detail["continuity_required"]) == ("compactor-b", "model-b", False)
    assert detail["image_semantics_verified"] is True and detail["response_retry"] is False
    assert KEY not in json.dumps(check)
    _ProbeProvider.calls, _ProbeProvider.wrong_marker_once = [], True
    rerun = _compactor_provider_marker(_plan())
    assert rerun["passed"] is True and rerun["detail"]["response_retry"] is True
    assert [stage for _, _, stage in _ProbeProvider.calls] == ["preflight-marker", "preflight-marker", "preflight-image"]


def test_only_a_bound_plan_adds_the_compactor_check(monkeypatch):
    from ora.orchestrator.preflight import gates as preflight

    monkeypatch.setattr(preflight, "_runtime", lambda plan: [])
    monkeypatch.setattr(preflight, "_qualification", lambda plan: preflight._check("qualification", True, ""))
    monkeypatch.setattr(preflight, "_agent_core", lambda plan: preflight._check("agent_core", True, ""))
    names = []
    for bound in (False, True):
        plan = _plan(base_url_env="A_BASE_URL", api_key_env="A_API_KEY", require_live_marker=False)
        plan["policy"]["agent_core"] = "pi"
        if not bound:
            del plan["policy"]["context_compactor"]
        plan.update(source={"commit": "0" * 40}, runtime={"kind": "synthetic"})
        names.append([check["check"] for check in preflight.run_preflight(plan)["checks"]])
    unbound, bound_names = names
    assert "compactor_provider_marker" not in unbound
    position = bound_names.index("provider_marker") + 1
    assert bound_names[position] == "compactor_provider_marker"
    assert bound_names[:position] + bound_names[position + 1:] == unbound
