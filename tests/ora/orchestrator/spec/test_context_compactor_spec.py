"""``policy.context_compactor`` binds the rollout compactor to its own Provider profile.

The field resolves a built-in, user or catalog profile into the plan's binding
(wire fields and variable names, never a key), enters plan and model-visible
identity, and is refused wherever the compactor does not run or could not read
or answer its packet. A plan without the field carries no key at all.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from ora.orchestrator.evaluation.model_visible import model_visible_sha256
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from ora.orchestrator.spec.context_compactor import resolve_context_compactor
from tests.support.agent_core.episodes import management
from tests.support.orchestrator.native_synthetic_spec import synthetic_spec

ROOT = Path(__file__).resolve().parents[4]
WIRE_FIELDS = {
    "provider", "model", "endpoint", "endpoint_transport", "base_url_env", "api_key_env", "proxy_env",
    "reasoning_effort", "stream", "supports_images", "context_window_tokens", "max_output_tokens",
    "provider_recovery",
}


def _spec(compactor: Any = None, *, lossless: bool = True) -> dict[str, Any]:
    spec = synthetic_spec(seed=0)
    if lossless:
        spec["policy"].update(context_retention="action_step_lossless_v4", summary_policy="rollout_checkpoint_v1",
                              context_management=management())
    else:
        spec["policy"]["context_retention"] = "rewrite_compact_v1"
    if compactor is not None:
        spec["policy"]["context_compactor"] = compactor
    return spec


def _profile(**overrides: Any) -> dict[str, Any]:
    profile = {
        "protocol": "openai_compatible", "endpoint": "http://127.0.0.1:9/b/v1", "endpoint_transport": "loopback_http_v1",
        "model": "model-b", "model_env": "COMPACTOR_B_MODEL", "base_url_env": "COMPACTOR_B_BASE_URL",
        "api_key_env": "COMPACTOR_B_API_KEY", "proxy_env": [], "reasoning_effort": "low",
        "context_window_tokens": 262144, "max_output_tokens": 16384, "stream": True, "supports_images": True,
        "qualification_schema": "ora.native-provider-probe.v1", "qualification_version": 1,
    }
    profile.update(overrides)
    return profile


def _catalog(path: Path, **profiles: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump({"schema": "ora.native-provider-catalog.v1", "profiles": profiles}),
                    encoding="utf-8")
    return path


def test_a_plan_without_the_field_carries_no_key():
    plan = resolve_run_spec(_spec(), root=ROOT)
    assert "context_compactor" not in plan["policy"]


def test_a_builtin_profile_resolves_into_a_binding_that_enters_plan_identity():
    base = resolve_run_spec(_spec(), root=ROOT)
    plan = resolve_run_spec(_spec({"provider_profile": "deepseek_flash_openai_low_16k_v1"}), root=ROOT)
    binding = plan["policy"]["context_compactor"]
    assert binding["schema"] == "ora.context-compactor-binding.v1"
    assert (binding["provider_profile"], binding["profile_source"]) == ("deepseek_flash_openai_low_16k_v1", "builtin")
    assert (binding["provider"], binding["model"], binding["endpoint"], binding["api_key_env"]) == (
        "openai_compatible", "deepseek-flash", "https://api.deepseek.com", "DEEPSEEK_LOW_API_KEY")
    assert WIRE_FIELDS <= set(binding) and "provider_catalog" not in binding
    assert not {"api_key", "tool_strictness", "model_env", "provider_qualification_schema"} & set(binding)
    # The rest of the policy is the plan without the field.
    assert {key: value for key, value in plan["policy"].items() if key != "context_compactor"} == base["policy"]
    assert plan["plan_sha256"] != base["plan_sha256"]
    assert plan["policy_identity"]["model_visible_sha256"] != base["policy_identity"]["model_visible_sha256"]
    assert model_visible_sha256(plan, root=ROOT) == plan["policy_identity"]["model_visible_sha256"]


def test_a_user_profile_resolves_from_the_user_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ORA_CONFIG_HOME", str(tmp_path))
    _catalog(tmp_path / "providers.yaml", **{"compactor-b": _profile(recovery_window_s=600)})
    binding = resolve_run_spec(_spec({"provider_profile": "compactor-b"}), root=ROOT)["policy"]["context_compactor"]
    assert (binding["profile_source"], binding["model"], binding["endpoint"]) == (
        "user", "model-b", "http://127.0.0.1:9/b/v1")
    # A profile's window selects the first-token recovery, as provider_policy does for a policy.
    assert (binding["provider_recovery"], binding["provider_recovery_window_s"]) == (
        "first_token_deadline_v1", 600)


def test_a_catalog_binding_records_the_catalog_and_model_visible_identity_ignores_where_it_lives(tmp_path):
    first = _catalog(tmp_path / "one.yaml", b=_profile())
    (tmp_path / "elsewhere").mkdir()
    second = _catalog(tmp_path / "elsewhere/two.yaml", b=_profile())
    plans = [resolve_run_spec(_spec({"provider_profile": "b", "provider_catalog": str(path)}), root=ROOT)
             for path in (first, second)]
    binding = plans[0]["policy"]["context_compactor"]
    assert binding["profile_source"] == "catalog"
    assert binding["provider_catalog"]["path"] == str(first) and len(binding["provider_catalog"]["sha256"]) == 64
    assert plans[0]["plan_sha256"] != plans[1]["plan_sha256"]
    assert (plans[0]["policy_identity"]["model_visible_sha256"]
            == plans[1]["policy_identity"]["model_visible_sha256"])


@pytest.mark.parametrize(("compactor", "message"), [
    ("deepseek_flash_openai_low_16k_v1", "takes provider_profile and an optional provider_catalog"),
    ({"provider_profile": "deepseek_flash_openai_low_16k_v1", "model": "other"}, "takes provider_profile"),
    ({"provider_catalog": "x.yaml"}, "takes provider_profile"),
    ({"provider_profile": ""}, "must be a non-empty string"),
    ({"provider_profile": "no_such_profile"}, "policy.context_compactor: unknown provider profile"),
    ({"provider_profile": "b", "provider_catalog": "../outside.yaml"}, "inside the project"),
    ({"provider_profile": "b", "provider_catalog": "missing-catalog.yaml"}, "is unavailable"),
])
def test_a_malformed_field_is_refused(compactor, message):
    with pytest.raises(RunSpecError, match=message):
        resolve_run_spec(_spec(compactor), root=ROOT)


def test_the_compactor_must_exist_in_the_selected_context():
    with pytest.raises(RunSpecError, match="requires context_retention=action_step_lossless_v4"):
        resolve_run_spec(_spec({"provider_profile": "deepseek_flash_openai_low_16k_v1"}, lossless=False), root=ROOT)
    with pytest.raises(RunSpecError, match="requires a native_api policy"):
        resolve_context_compactor({"kind": "scripted", "context_compactor": {"provider_profile": "x"}}, {}, root=ROOT)
    spec = _spec({"provider_profile": "deepseek_flash_openai_low_16k_v1"})
    spec.update(episode_mode="play", play={})
    with pytest.raises(RunSpecError, match="does not apply to a free-play session"):
        resolve_run_spec(spec, root=ROOT)


@pytest.mark.parametrize(("profile", "message"), [
    (_profile(supports_images=False), "needs an image-capable profile"),
    (_profile(max_output_tokens=4096), "needs max_output_tokens >= 8192"),
    (_profile(context_window_tokens=131072), r"at least the policy's \(131072 < 262144\)"),
    (_profile(gateway_client_binding=True), "does not take a gateway-bound profile"),
    (_profile(protocol="anthropic_messages", request_options={"temperature": 0.2}), "names request_options"),
])
def test_a_profile_that_cannot_serve_the_compactor_is_refused(tmp_path, profile, message):
    catalog = _catalog(tmp_path / "catalog.yaml", b=profile)
    with pytest.raises(RunSpecError, match=message):
        resolve_run_spec(_spec({"provider_profile": "b", "provider_catalog": str(catalog)}), root=ROOT)


def test_resolution_leaves_the_caller_spec_unchanged():
    spec = _spec({"provider_profile": "deepseek_flash_openai_low_16k_v1"})
    before = copy.deepcopy(spec)
    resolve_run_spec(spec, root=ROOT)
    assert spec == before
