"""Generate OpenAI-compatible wire-projection cases and Python's exact answers.

Usage: python tests/support/agent_core/provider/wire_cases.py OUT.json [--count N] [--seed S]

Each case is an execution request (as the agent core's request assembler builds it) plus
provider options. Python records `prepare_request` canonical bytes (or the
error), the request headers and URL both transports would send (captured by
fake HTTP clients, nothing leaves the process), so the TypeScript port can be
compared byte for byte. Also covers tool_alias, normalize_usage,
checkpoint_request_options, a profile's request_options (accepted, refused
or colliding with a request field) and the `<wire_tool_aliases>` disclosure
joining each shape of leading system message.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

from common import canonical_text  # noqa: E402
from ora.agent.contracts import PolicySessionError  # noqa: E402
from ora.agent.evidence.canonical import canonical_bytes  # noqa: E402
from ora.agent.providers import openai_compatible  # noqa: E402
from ora.agent.providers.continuation import ProviderContinuationState  # noqa: E402
from ora.agent.providers.openai_wire_options import (  # noqa: E402
    checkpoint_request_options, normalize_usage,
)
from ora.agent.providers.recovery import RecoveryWindow, _bounded_request_in_process  # noqa: E402
from ora.agent.providers.tool_names import tool_alias  # noqa: E402

API_KEY = "sk-wire-test-abcdefghijabcdef"
KIND = "openai_chat_reasoning_content_v1"
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

STRICT = ["observe", "move_pose", "set_gripper", "finish_episode", "geometry_query", "a" * 64]
NEEDS_ALIAS = [
    "robot.read_manual", "robot.submit_action", "observation.dump_state", "工具.观察",
    "3d_view", "_private", "x" * 70, "tool with space", "emoji😀tool", "a/b:c", "-dash",
]
COLLIDER = tool_alias("robot.read_manual")  # strict-looking name equal to an alias


class _Captured(Exception):
    pass


class _FakeAsyncStream:
    def __init__(self, sink, method, url, headers, content):
        sink.update(method=method, url=url, headers=dict(headers),
                    content_sha256=hashlib.sha256(content).hexdigest())

    async def __aenter__(self):
        raise _Captured()

    async def __aexit__(self, *args):
        return False


class _FakeAsyncClient:
    sink: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, headers, content):
        return _FakeAsyncStream(_FakeAsyncClient.sink, method, url, headers, content)


class _FakeSyncStream:
    def __enter__(self):
        raise _Captured()

    def __exit__(self, *args):
        return False


class _FakeSyncClient:
    def __init__(self):
        self.sink = {}

    def stream(self, method, url, headers, content):
        self.sink.update(method=method, url=url, headers=dict(headers),
                         content_sha256=hashlib.sha256(content).hexdigest())
        return _FakeSyncStream()

    def close(self):
        pass


def capture_deadline(provider, body):
    _FakeAsyncClient.sink = {}
    original = httpx.AsyncClient
    httpx.AsyncClient = _FakeAsyncClient
    try:
        _bounded_request_in_process(provider, body, RecoveryWindow(time.monotonic() + 100, False))
    except _Captured:
        pass
    finally:
        httpx.AsyncClient = original
    return dict(_FakeAsyncClient.sink)


def capture_legacy(provider, body):
    client = _FakeSyncClient()
    provider.client = client
    try:
        provider._request_once(body)
    except _Captured:
        pass
    return dict(client.sink)


# -- request generation ---------------------------------------------------------------


def tool_def(rng, name):
    params = {"type": "object", "properties": {
        "x": {"type": "number", "minimum": rng.choice([0, 0.0, -1.5, 1e-07]), "default": rng.choice([1.0, 2, 0.25])},
        "label": {"type": "string", "description": rng.choice(["plain", "ünïcödé", "line\nbreak", "😀"])},
    }, "required": ["x"]}
    fn = {"name": name, "description": f"tool {name}", "parameters": params}
    if rng.random() < 0.2:
        fn["strict"] = rng.choice([True, False])
    return {"type": "function", "function": fn}


def tool_call(rng, name, index):
    return {"id": f"call-{index}", "type": "function",
            "function": {"name": name, "arguments": json.dumps({"x": rng.choice([1, 1.0, 0.5])})}}


def make_request(rng, case_id):
    pool = STRICT + NEEDS_ALIAS
    if rng.random() < 0.08:
        pool = pool + [COLLIDER]
    tool_names = rng.sample(pool, rng.randint(0, 5))
    if COLLIDER in pool and rng.random() < 0.5:
        tool_names = tool_names + [COLLIDER, "robot.read_manual"]
    model = rng.choice(["deepseek-flash", "deepseek-ai/DeepSeek-V4.1-Flash", "smoke-model"])
    messages = [{"role": "system", "content": "You control one robot."}]
    if rng.random() < 0.3:
        messages.append({"role": "system", "content": "second system"})
    messages.append({"role": "user", "content": "Pick the alphabet soup and place it in the basket"})
    mutations = []
    for turn in range(rng.randint(0, 4)):
        if rng.random() < 0.4:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"observation {turn}"},
                {"type": "image_url", "image_url": {"url": PNG, "detail": rng.choice(["low", "high"])}},
            ]})
        names = [rng.choice(pool + [""])] if rng.random() < 0.7 else []
        calls = [tool_call(rng, name, f"{case_id}-{turn}-{i}") for i, name in enumerate(names)]
        assistant = {"role": "assistant", "content": rng.choice(["", "I will move.", "思考后行动"])}
        if calls:
            assistant["tool_calls"] = calls
        signed = rng.random() < 0.7
        if signed:
            reasoning = rng.choice(["", "Let me think.", "step 1\nstep 2", "x" * 50])
            state = ProviderContinuationState.create(
                provider_protocol="openai_chat_completions", model=model, native_kind=KIND,
                native_payload={"reasoning_content": reasoning}, canonical_message=assistant,
            ).to_dict()
            assistant["reasoning_content"] = reasoning
            assistant["_provider_continuation_state"] = state
        messages.append(assistant)
        for call in calls:
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                             "content": json.dumps({"ok": True, "value": 1.0})})
    request = {
        "model": model,
        "messages": messages,
        "tools": [tool_def(rng, name) for name in tool_names],
        "tool_choice": rng.choice(["auto", "auto", "none", "required"]),
        "max_tokens": rng.choice([16384, 2048, 32768]),
        "stream": rng.random() < 0.75,
    }
    if tool_names and rng.random() < 0.2:
        request["tool_choice"] = {"type": "function", "function": {"name": rng.choice(tool_names)}}
    effort = rng.choice(["low", "high", "provider_default"])
    if effort != "provider_default":
        request["reasoning_effort"] = effort
    if request["stream"]:
        request["stream_options"] = {"include_usage": True}
    if rng.random() < 0.05:
        del request["max_tokens"]
    # Mutations that Python must refuse (or accept) in a specific way.
    if rng.random() < 0.35:
        mutation = rng.choice([
            "forbidden", "no_model", "empty_model", "messages_dict", "message_str", "tamper_hash",
            "wrong_model", "wrong_protocol", "wrong_kind", "extra_payload", "nonstr_reasoning",
            "continuation_on_user", "unsigned_reasoning", "unsigned_reasoning_field", "none_state",
            "state_str", "rebound_content", "missing_state_key", "extra_state_key", "stale_reasoning_key",
            "tools_none", "tool_str", "tool_calls_none", "bad_schema", "bad_replay_policy", "short_sha",
            "response_id", "upper_sha",
        ])
        mutations.append(mutation)
        apply_mutation(rng, request, mutation, model)
    return request, mutations


def first_signed(request):
    for message in request["messages"]:
        if isinstance(message, dict) and "_provider_continuation_state" in message:
            return message
    return None


def resign(message, model, payload, protocol="openai_chat_completions", kind=KIND, response_id=None):
    canonical = {k: v for k, v in message.items() if k not in ("_provider_continuation_state",)}
    message["_provider_continuation_state"] = ProviderContinuationState.create(
        provider_protocol=protocol, model=model, native_kind=kind, native_payload=payload,
        canonical_message=canonical, response_id=response_id,
    ).to_dict()


def apply_mutation(rng, request, mutation, model):
    signed = first_signed(request)
    if mutation == "forbidden":
        request[rng.choice(["input", "system", "thinking", "output_config", "instructions"])] = "x"
        if rng.random() < 0.5:
            request["input_schema"] = {}
    elif mutation == "no_model":
        del request["model"]
    elif mutation == "empty_model":
        request["model"] = ""
    elif mutation == "messages_dict":
        request["messages"] = {"role": "user"}
    elif mutation == "message_str":
        request["messages"].append("hello")
    elif mutation == "tools_none":
        request["tools"] = None
    elif mutation == "tool_str":
        request["tools"].append("not-a-tool")
    elif mutation == "tool_calls_none":
        request["messages"].append({"role": "assistant", "content": "", "tool_calls": None})
    elif mutation == "unsigned_reasoning":
        request["messages"].append({"role": "assistant", "content": "", "reasoning_content": "unsigned"})
    elif mutation == "unsigned_reasoning_field":
        request["messages"].append({"role": "assistant", "content": "", "reasoning": "unsigned"})
    elif mutation == "continuation_on_user":
        user = {"role": "user", "content": "u"}
        resign(user, model, {"reasoning_content": "r"})
        request["messages"].append(user)
    elif signed is None:
        return
    elif mutation == "tamper_hash":
        signed["_provider_continuation_state"]["state_sha256"] = "0" * 64
    elif mutation == "wrong_model":
        resign(signed, model + "-other", {"reasoning_content": "r"})
    elif mutation == "wrong_protocol":
        resign(signed, model, {"reasoning_content": "r"}, protocol="anthropic_messages")
    elif mutation == "wrong_kind":
        resign(signed, model, {"reasoning_content": "r"}, kind="anthropic_thinking_blocks_v1")
    elif mutation == "extra_payload":
        resign(signed, model, {"reasoning_content": "r", "signature": "s"})
    elif mutation == "nonstr_reasoning":
        resign(signed, model, {"reasoning_content": rng.choice([5, None, ["a"], 1.0])})
    elif mutation == "none_state":
        signed["_provider_continuation_state"] = None
    elif mutation == "state_str":
        signed["_provider_continuation_state"] = "opaque"
    elif mutation == "rebound_content":
        signed["content"] = signed["content"] + " edited"
    elif mutation == "missing_state_key":
        del signed["_provider_continuation_state"]["response_id"]
    elif mutation == "extra_state_key":
        signed["_provider_continuation_state"]["extra"] = 1
    elif mutation == "stale_reasoning_key":
        signed["reasoning"] = "stale"
    elif mutation == "bad_schema":
        signed["_provider_continuation_state"]["schema"] = "ora.provider-continuation-state.v2"
    elif mutation == "bad_replay_policy":
        signed["_provider_continuation_state"]["replay_policy"] = "best_effort"
    elif mutation == "short_sha":
        signed["_provider_continuation_state"]["canonical_message_sha256"] = "abc"
    elif mutation == "upper_sha":
        state = signed["_provider_continuation_state"]
        state["canonical_message_sha256"] = state["canonical_message_sha256"].upper()
    elif mutation == "response_id":
        resign(signed, model, {"reasoning_content": "r"}, response_id="resp-1")


PROVIDER_VARIANTS = [
    {"base_url": "https://api.deepseek.com"},
    {"base_url": "https://inference-node:30743/v1", "reasoning_field": "reasoning",
     "session_affinity_id": "ora:run-1:native"},
    {"base_url": "http://127.0.0.1:8000/v1/", "session_affinity_id": "ora:quickstart-x:native"},
    {"base_url": "http://localhost:1/v1", "output_token_parameter": "max_completion_tokens"},
    {"base_url": "http://[::1]:9/v1", "session_affinity_id": "a.b-c_d:9"},
    {"base_url": "http://10.0.0.1:8000/v1", "session_affinity_id": "ora:run:native"},
    {"base_url": "https://gateway.example/v1", "tool_strictness": "explicit_false"},
    {"base_url": "https://gateway.example/v1", "tool_strictness": "explicit_false",
     "output_token_parameter": "max_completion_tokens", "reasoning_field": "reasoning"},
    {"base_url": "https://api.deepseek.com", "session_affinity_id": "bad id!"},
    {"base_url": "https://api.deepseek.com", "session_affinity_id": "x" * 201},
    {"base_url": "https://api.deepseek.com", "tool_strictness": "explicit_true"},
    {"base_url": "https://api.deepseek.com", "reasoning_field": "thinking"},
    {"base_url": "https://api.deepseek.com", "output_token_parameter": "max_output_tokens"},
    {"base_url": "////"},
    # A profile's request_options: accepted ones reach the wire, the rest are refused at construction.
    {"base_url": "https://openrouter.example/api/v1", "request_options": {"temperature": 1.0, "top_p": 1.0}},
    {"base_url": "https://openrouter.example/api/v1", "request_options": {
        "temperature": 0.6, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True},
        "provider": {"order": ["together"], "allow_fallbacks": False}}},
    {"base_url": "http://127.0.0.1:8000/v1", "tool_strictness": "explicit_false", "reasoning_field": "reasoning",
     "request_options": {"top_k": 20, "reasoning": {"effort": "high"}}},
    {"base_url": "https://api.deepseek.com", "request_options": {"top_k": -1, "temperature": 0}},
    {"base_url": "https://api.deepseek.com", "request_options": {}},
    {"base_url": "https://api.deepseek.com", "request_options": ["temperature"]},
    {"base_url": "https://api.deepseek.com", "request_options": {"seed": 1, "max_tokens": 9, "top_p": 0.5}},
    {"base_url": "https://api.deepseek.com", "request_options": {"temperature": 2.5}},
    {"base_url": "https://api.deepseek.com", "request_options": {"temperature": True}},
    {"base_url": "https://api.deepseek.com", "request_options": {"top_p": 0}},
    {"base_url": "https://api.deepseek.com", "request_options": {"top_k": 2.0}},
    {"base_url": "https://api.deepseek.com", "request_options": {"top_k": 0}},
    {"base_url": "https://api.deepseek.com", "request_options": {"reasoning": {}}},
    {"base_url": "https://api.deepseek.com", "request_options": {"provider": ["together"]}},
    {"base_url": "https://vllm.example/v1", "request_options": {"return_token_ids": True}},
    {"base_url": "https://vllm.example/v1", "request_options": {"return_token_ids": False}},
    {"base_url": "https://vllm.example/v1", "request_options": {"return_token_ids": 1}},
]


def run_case(request, variant):
    case = {"request": request, "provider": variant}
    try:
        provider = openai_compatible.OpenAICompatibleProvider(
            api_key=API_KEY, client=_FakeSyncClient(), recovery_policy="same_credential_deadline_v1", **variant)
    except PolicySessionError as exc:
        case["construct_error"] = str(exc)
        return case
    try:
        body = provider.prepare_request(copy.deepcopy(request))
    except PolicySessionError as exc:
        case["wire_error"] = {"type": "PolicySessionError", "message": str(exc)}
        return case
    except Exception as exc:  # noqa: BLE001 - Python raises non-policy errors on malformed input
        case["wire_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return case
    case["wire"] = canonical_text(body)
    case["deadline_request"] = capture_deadline(provider, body)
    case["legacy_request"] = capture_legacy(provider, body)
    return case


def request_option_cases(rng):
    """The static projection with request_options: merged, refused, or colliding with a request field."""
    options = [None, {"temperature": 1.0}, {"top_p": 0.5, "provider": {"sort": "throughput"}},
               {"temperature": 0.2, "top_k": 40}, {"seed": 3}, {"top_k": True}, {"return_token_ids": True},
               {"return_token_ids": "yes"}]
    out = []
    for index in range(8):
        request, mutations = make_request(rng, 100_000 + index)
        if index % 2:
            request["temperature"] = 0.7  # a field the request carries is refused, never overwritten
        for option in options:
            case = {"request": copy.deepcopy(request), "request_options": option, "mutations": mutations}
            try:
                body = openai_compatible.OpenAICompatibleProvider.prepare_request(copy.deepcopy(request), request_options=option)
            except Exception as exc:  # noqa: BLE001 - recorded with its type, as run_case does
                case["wire_error"] = {"type": type(exc).__name__, "message": str(exc)}
            else:
                case["wire"] = canonical_text(body)
            out.append(case)
    return out


#: Leading system messages the alias disclosure must join (or refuse), by shape.
ALIAS_SYSTEM_SHAPES = {
    "string": [{"role": "system", "content": "You control one robot."}],
    "parts": [{"role": "system", "content": [{"type": "text", "text": "You control one robot."}]}],
    "null": [{"role": "system", "content": None}],
    "empty": [{"role": "system", "content": ""}],
    "no_content": [{"role": "system"}],
    "absent": [],
    "two_leading": [{"role": "system", "content": "You control one robot."},
                    {"role": "system", "content": "second system"}],
    "object": [{"role": "system", "content": {"text": "not a part list"}}],
    "number": [{"role": "system", "content": 5}],
}
ALIAS_TOOL_SETS = {
    "dotted": ["robot.submit_action", "observation.dump_state", "observe"],
    "strict": ["observe", "move_pose", "finish_episode"],
}


def alias_disclosure_cases():
    """The `<wire_tool_aliases>` disclosure inside the leading system message, two turns per shape.

    Turn 0 is the opening request; turn 1 adds a signed assistant tool call, its
    result and a new user image, so the provider prefix can be compared across
    a continuation. Qwen3.x templates reject a system message after index 0.
    """
    rng = random.Random(20261018)
    variants = [PROVIDER_VARIANTS[0], PROVIDER_VARIANTS[7], PROVIDER_VARIANTS[16]]
    out = []
    for shape, leading in ALIAS_SYSTEM_SHAPES.items():
        for tools_label, names in ALIAS_TOOL_SETS.items():
            for variant_index, variant in enumerate(variants):
                model = "deepseek-ai/DeepSeek-V4.1-Flash"
                messages = [*copy.deepcopy(leading),
                            {"role": "user", "content": "Pick the alphabet soup and place it in the basket"}]
                request = {"model": model, "messages": messages,
                           "tools": [tool_def(rng, name) for name in names], "tool_choice": "auto",
                           "max_tokens": 16384, "stream": True, "stream_options": {"include_usage": True}}
                for turn in range(2):
                    if turn:
                        call = tool_call(rng, names[0], f"alias-{shape}-{tools_label}-{variant_index}")
                        assistant = {"role": "assistant", "content": "", "tool_calls": [call]}
                        resign(assistant, model, {"reasoning_content": "Let me think."})
                        assistant["reasoning_content"] = "Let me think."
                        request["messages"] += [
                            assistant,
                            {"role": "tool", "tool_call_id": call["id"], "name": names[0],
                             "content": json.dumps({"ok": True})},
                            {"role": "user", "content": [
                                {"type": "text", "text": "observation 1"},
                                {"type": "image_url", "image_url": {"url": PNG, "detail": "low"}}]},
                        ]
                    case = run_case(copy.deepcopy(request), variant)
                    case.update(id=f"alias-{shape}-{tools_label}-{variant_index}-{turn}", shape=shape,
                                tools=tools_label, variant=variant_index, turn=turn,
                                leading_systems=len(leading))
                    out.append(case)
    return out


def usage_cases(rng):
    values = [None, 5, "x", [], {}, {"prompt_tokens": 1}, {"prompt_tokens": True},
              {"prompt_tokens": -1, "completion_tokens": 2.7},
              {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}},
              {"prompt_tokens": 10, "prompt_cache_hit_tokens": 3, "prompt_cache_miss_tokens": 7},
              {"prompt_tokens": 10, "prompt_tokens_details": None, "prompt_cache_hit_tokens": None},
              {"completion_tokens_details": {"reasoning_tokens": 351}},
              {"prompt_tokens": 22721, "completion_tokens": 516, "total_tokens": 23237,
               "completion_tokens_details": {"reasoning_tokens": 351}},
              {"total_tokens": 1.0, "prompt_tokens": 9007199254740993},
              # sglang: reasoning at the top level, image tokens in the prompt details.
              {"prompt_tokens": 23495, "completion_tokens": 775, "total_tokens": 24270, "reasoning_tokens": 668,
               "prompt_tokens_details": {"cached_tokens": 0, "image_tokens": 128}},
              {"reasoning_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 7}}]
    choices = [None, True, False, 0, 1, -3, 2.5, 7.0, "4", [], {}]
    for _ in range(150):
        value = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens"):
            if rng.random() < 0.6:
                value[key] = rng.choice(choices)
        if rng.random() < 0.4:
            value["prompt_tokens_details"] = rng.choice([None, {}, {"cached_tokens": rng.choice(choices)}])
        if rng.random() < 0.4:
            value["completion_tokens_details"] = rng.choice(
                [None, {}, {"reasoning_tokens": rng.choice(choices)}])
        values.append(value)
    out = []
    for value in values:
        normalized = normalize_usage(value)
        out.append({"value": value, "normalized": None if normalized is None else canonical_text(normalized)})
    return out


COMPACTOR_MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "<closed_episode>completed episode evidence</closed_episode>"},
    {"type": "image_url", "image_url": {"url": PNG}},
]}]
COMPACTOR_REFS = [{"event_id": "agent-event-000001"}]


def compactor_request_cases() -> list[dict]:
    """The rollout compactor's canonical and wire request through a real provider instance.

    Python's compactor reads ``checkpoint_request_options`` from the provider it
    holds (in a run, the policy provider's copy) and sends the provider's wire
    projection. The provider's ``complete`` is replaced, so nothing is sent.
    """

    import tempfile

    from tests.support.agent.rollout_compactor import RolloutCheckpointCompactor
    from ora.agent.evidence.canonical import ArtifactStore

    cases = []
    for model in ["deepseek-flash", "deepseek-v4-pro", "deepseek-ai/DeepSeek-V4.1-Flash", "gpt-4.1-mini"]:
        for effort, stream in [("high", True), ("provider_default", False), ("low", True)]:
            provider = openai_compatible.OpenAICompatibleProvider(
                base_url="https://wire.invalid/v1", api_key=API_KEY, client=_FakeSyncClient())
            provider.complete = lambda **kwargs: {
                "message": {"role": "assistant", "content": json.dumps({"task_progress": []})},
                "usage": None, "finish_reason": "stop", "raw_events": [],
                "model_call_id": kwargs["model_call_id"], "started_unix_s": 0.0, "finished_unix_s": 0.0,
            }
            events: list[dict] = []

            def sink(event_type, **fields):
                events.append({"type": event_type, **fields})
                return {"event_id": f"agent-event-{len(events):06d}"}

            with tempfile.TemporaryDirectory() as root:
                artifacts = ArtifactStore(Path(root))
                compactor = RolloutCheckpointCompactor(
                    provider=provider, model=model, reasoning_effort=effort, stream=stream,
                    artifacts=artifacts, event_sink=sink,
                )
                try:
                    compactor.summarize(messages=copy.deepcopy(COMPACTOR_MESSAGES), source_refs=COMPACTOR_REFS)
                except PolicySessionError:
                    pass  # the scripted checkpoint is not a valid one; the requests are recorded first
                attempt = next(event for event in events if event["type"] == "context_compactor_attempt")
                canonical = (Path(root) / attempt["payload"]["canonical_request_ref"]["path"]).read_bytes()
                wire = (Path(root) / attempt["artifact_ref"]["path"]).read_bytes()
            cases.append({
                "model": model, "reasoning_effort": effort, "stream": stream,
                "messages": COMPACTOR_MESSAGES, "source_refs": COMPACTOR_REFS,
                "canonical": canonical.decode("utf-8"), "wire": wire.decode("utf-8"),
            })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cases = []
    for case_id in range(args.count):
        request, mutations = make_request(rng, case_id)
        variant = PROVIDER_VARIANTS[case_id % len(PROVIDER_VARIANTS)] if rng.random() < 0.2 else rng.choice(
            PROVIDER_VARIANTS[:8])
        case = run_case(request, variant)
        case["id"] = case_id
        case["mutations"] = mutations
        cases.append(case)
    aliases = [{"name": name, "alias": tool_alias(name)} for name in
               STRICT + NEEDS_ALIAS + [COLLIDER, "", "A", "a-b", "Ab_", "1", "_", "é", "a" * 65, "a.b" * 30]]
    checkpoint = [{"model": model, "options": checkpoint_request_options(model=model)} for model in
                  ["deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp",
                   "deepseek-ai/DeepSeek-V4.1-Flash", "gpt-4.1-mini", ""]]
    output = {"cases": cases, "aliases": aliases, "usage": usage_cases(rng), "checkpoint": checkpoint,
              "api_key": API_KEY}
    output["request_options"] = request_option_cases(rng)
    output["alias_disclosure"] = alias_disclosure_cases()
    output["compactor_requests"] = compactor_request_cases()
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    errors = sum(1 for case in cases if "wire_error" in case or "construct_error" in case)
    print(json.dumps({"cases": len(cases), "errors": errors}))


if __name__ == "__main__":
    main()
