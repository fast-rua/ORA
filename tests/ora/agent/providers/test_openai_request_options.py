"""A profile can name closed request fields of the OpenAI-compatible wire.

Nemotron 3.5 Super VL switches thinking on through vLLM's
``chat_template_kwargs`` and its hosted endpoints take no ``reasoning_effort``;
model cards name their own sampling. ``request_options`` carries those fields
into the final wire body, the recorded wire is exactly what ``complete`` sends,
and profiles without the option keep their bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ora.agent.providers.openai_compatible import OpenAICompatibleProvider
from ora.agent.providers.factory import build_provider
from ora.agent.providers.profiles import (
    NativeProviderProfileError, load_builtin_provider_profiles, load_provider_profiles,
)
from ora.agent.contracts import PolicySessionError
from ora.orchestrator.spec.inputs import provider_policy

NEMOTRON = {"temperature": 0.6, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True}}


def _request():
    return {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 64, "stream": False}


def test_options_reach_the_wire_and_unset_profiles_keep_their_bytes():
    provider = build_provider("openai_compatible", base_url="https://example.invalid", api_key="test-secret-key",
                                     request_options=NEMOTRON)
    plain = build_provider("openai_compatible", base_url="https://example.invalid", api_key="test-secret-key")
    try:
        body = provider.prepare_request(_request())
        assert {key: body.pop(key) for key in NEMOTRON} == NEMOTRON
        assert body == plain.prepare_request(_request()) == OpenAICompatibleProvider.prepare_request(_request())
    finally:
        provider.close()
        plain.close()


def test_the_recorded_wire_is_the_body_complete_sends():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": "ok"}}]})

    provider = OpenAICompatibleProvider(base_url="https://example.invalid", api_key="test-secret-key", request_options=NEMOTRON,
                                        client=httpx.Client(transport=httpx.MockTransport(handler)))
    # The execution request as the agent core assembles it; prepare_request is the recorded wire.
    execution = {"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": [], "tool_choice": "auto",
                 "stream": False, "max_tokens": 64}
    wire = provider.prepare_request(execution)
    provider.complete(request=execution, model_call_id="call-0")
    assert sent == [dict(wire)]
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": True} and "reasoning_effort" not in sent[0]


def test_a_request_field_is_never_overwritten():
    provider = build_provider("openai_compatible", base_url="https://example.invalid", api_key="test-secret-key",
                                     request_options={"temperature": 1.0})
    try:
        with pytest.raises(PolicySessionError, match="collide"):
            provider.prepare_request({**_request(), "temperature": 0.2})
    finally:
        provider.close()


@pytest.mark.parametrize("options", [
    {}, {"seed": 1}, {"temperature": 3}, {"temperature": True}, {"top_p": 0}, {"top_k": 0},
    {"chat_template_kwargs": {}}, {"reasoning": "high"}, {"temperature": float("nan")}, {"return_token_ids": False},
])
def test_invalid_options_are_refused(options, tmp_path):
    with pytest.raises(PolicySessionError, match="request_options"):
        OpenAICompatibleProvider(base_url="https://example.invalid", api_key="test-secret-key", request_options=options)
    profile = {
        "protocol": "openai_compatible", "endpoint": "https://example.invalid", "model": "m", "model_env": "M",
        "base_url_env": "B", "api_key_env": "K", "proxy_env": [], "reasoning_effort": "low",
        "context_window_tokens": 200000, "max_output_tokens": 16384, "stream": True, "supports_images": True,
        "qualification_schema": "ora.native-provider-probe.v1", "qualification_version": 1,
        "request_options": options,
    }
    path = tmp_path / "profiles.yaml"
    path.write_text(json.dumps({"schema": "ora.native-provider-catalog.v1", "profiles": {"p": profile}}))
    with pytest.raises(NativeProviderProfileError, match="request_options"):
        load_provider_profiles(path)
    path.write_text(json.dumps({"schema": "ora.native-provider-catalog.v1", "profiles": {"p": {
        **profile, "protocol": "anthropic_messages", "request_options": {"temperature": 1.0}}}}))
    with pytest.raises(NativeProviderProfileError, match="request_options"):
        load_provider_profiles(path)


def test_anthropic_providers_refuse_the_option():
    with pytest.raises(PolicySessionError, match="request_options"):
        build_provider("anthropic_messages", base_url="https://example.invalid", api_key="test-secret-key",
                              request_options={"temperature": 1.0})


def test_new_model_profiles_carry_their_options_into_the_policy():
    profiles = load_builtin_provider_profiles()
    for profile_id, profile in profiles.items():
        if "request_options" in profile:
            assert provider_policy(profile_id)["request_options"] == profile["request_options"]


def test_the_pi_core_provider_spec_carries_the_profile_options(tmp_path, monkeypatch):
    # The Pi core runs the TypeScript port of this provider; its spec hands the port the same options
    # (ora-pi/test/provider_wire.test.ts compares the port's wire bytes with this module's).
    from ora.agent.pi.factory import _provider_spec

    profile_id = next(key for key, value in load_builtin_provider_profiles().items() if "request_options" in value)
    policy = provider_policy(profile_id)
    monkeypatch.setenv(policy["base_url_env"], policy["endpoint"])
    monkeypatch.setenv(policy["api_key_env"], "test-secret-key")
    spec = _provider_spec({"policy": policy}, run_root=tmp_path, run_id="run-1", provider_override=None)
    assert spec["kind"] == "openai_compatible"
    assert spec["request_options"] == policy["request_options"]
    plain = provider_policy("deepseek_flash_openai_low_16k_v1")
    monkeypatch.setenv(plain["base_url_env"], plain["endpoint"])
    monkeypatch.setenv(plain["api_key_env"], "test-secret-key")
    assert _provider_spec({"policy": plain}, run_root=tmp_path, run_id="run-1",
                          provider_override=None)["request_options"] is None
