"""Anthropic Messages cases and Python's exact answers, for the TypeScript port.

Usage: python tests/support/agent_core/provider/anthropic_cases.py OUT.json [--count N] [--seed S] [--sealed-cache DIR]

Covers translate_anthropic_request (randomized edge cases, plus the real
canonical requests of the LIBERO-Pro/RoboTwin/RoboDojo sealed runs that
``ORA_TEST_SEALED_*`` name, see ``tests/support/agent_core/sealed.py``), the
streamed-event parse (AnthropicMessagesProvider._parse_events) over valid and
corrupted event lists, the signed-thinking continuation round trip,
normalize_anthropic_usage and the Kimi signature check. Request answers are
the sha256 of canonical_bytes (and the text for small ones) or the error.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from common import canonical_text  # noqa: E402
from wire_cases import COLLIDER, NEEDS_ALIAS, PNG, STRICT, make_request  # noqa: E402
from ora.agent.contracts import PolicySessionError  # noqa: E402
from ora.agent.evidence.canonical import canonical_bytes  # noqa: E402
from ora.agent.providers import anthropic_messages, anthropic_reply, anthropic_request  # noqa: E402
from ora.agent.providers.anthropic_history import _has_complete_anthropic_thinking_signature  # noqa: E402
from tests.support.agent_core import sealed  # noqa: E402

SMALL = 60_000


def answer_request(request):
    try:
        body = anthropic_request.translate_anthropic_request(request)
    except PolicySessionError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # a Python crash is an answer too (type only)
        return {"crash": type(exc).__name__}
    data = canonical_bytes(body)
    out = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if len(data) <= SMALL:
        out["text"] = data.decode("utf-8")
    return out


def real_requests(per_env, cache):
    """A spread of each configured sealed run's canonical requests; the reason for each one skipped."""
    cases, sources = [], {}
    for env in sealed.ENVIRONMENTS:
        try:
            files = sealed.canonical_requests(env, cache)
        except sealed.Unavailable as exc:
            sources[env] = {"skip": str(exc)}
            continue
        picks = sorted(set([0, len(files) - 1] + [len(files) * k // per_env for k in range(per_env)]))
        for index in picks:
            if index >= len(files):
                continue
            request = json.loads(Path(files[index]).read_text(encoding="utf-8"))
            cases.append({"id": f"real:{env}:{Path(files[index]).name}", "request": request})
        sources[env] = {"requests": len(files), "cases": sum(1 for case in cases if case["id"].startswith(f"real:{env}:"))}
    return cases, sources


def anthropic_mutations(rng, request):
    """Anthropic-only edges on top of the shared generator's requests."""
    kind = rng.choice([
        "none", "named_choice", "none_choice", "stop_str", "stop_list", "temperature", "bad_effort",
        "max_tokens_bool", "max_tokens_zero", "max_tokens_float", "tool_content_obj", "tool_error_ok",
        "user_after_tool", "interleave_text", "bad_image", "http_image", "oneof_schema", "system_image",
        "unknown_block", "unknown_role", "args_obj", "args_bad_json", "args_list", "no_call_id",
        "tools_dict", "tool_no_function", "tool_bad_params",
    ])
    msgs = request["messages"]
    if kind == "named_choice" and request.get("tools"):
        request["tool_choice"] = {"type": "function", "function": {"name": request["tools"][0]["function"]["name"]}}
    elif kind == "none_choice":
        request["tool_choice"] = "none"
    elif kind == "stop_str":
        request["stop"] = "END"
    elif kind == "stop_list":
        request["stop"] = ["END", "STOP"]
    elif kind == "temperature":
        request["temperature"] = rng.choice([0, 0.0, 0.7])
        request["top_p"] = 0.9
    elif kind == "bad_effort":
        request["reasoning_effort"] = "minimal"
    elif kind == "max_tokens_bool":
        request["max_tokens"] = True
    elif kind == "max_tokens_zero":
        request["max_tokens"] = 0
    elif kind == "max_tokens_float":
        request["max_tokens"] = 2048.0
    elif kind == "tool_content_obj":
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c-obj", "type": "function", "function": {"name": "observe", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": "c-obj", "content": {"ok": False, "b": [1, 2.5], "a": "ü"}})
    elif kind == "tool_error_ok":
        msgs.append({"role": "assistant", "content": "x", "tool_calls": [
            {"id": "c-err", "type": "function", "function": {"name": "observe", "arguments": {"views": ["a"]}}}]})
        msgs.append({"role": "tool", "tool_call_id": "c-err", "content": json.dumps({"ok": False, "error": "no"})})
        msgs.append({"role": "tool", "tool_call_id": "c-err2", "content": "not json"})
    elif kind == "user_after_tool":
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c-u", "type": "function", "function": {"name": "observe", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": "c-u", "content": "{}"})
        msgs.append({"role": "user", "content": "next observation"})
    elif kind == "interleave_text":
        msgs.append({"role": "user", "content": "text first"})
        msgs.append({"role": "tool", "tool_call_id": "c-late", "content": "{}"})
    elif kind == "bad_image":
        msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,ab$c"}}]})
    elif kind == "http_image":
        msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": "https://example.com/a.png"}]})
    elif kind == "oneof_schema":
        request.setdefault("tools", []).append({"type": "function", "function": {
            "name": "move_pose", "description": "d", "parameters": {"oneOf": [
                {"type": "object", "properties": {"a": {"type": "number"}}},
                {"type": "object", "properties": {"b": {"type": "array"}}}]}}})
    elif kind == "system_image":
        msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": "sys"},
                                                      {"type": "image_url", "image_url": {"url": PNG}}]})
    elif kind == "unknown_block":
        msgs.append({"role": "user", "content": [{"type": "audio", "data": "x"}]})
    elif kind == "unknown_role":
        msgs.append({"role": "developer", "content": "x"})
    elif kind == "args_obj":
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c-o", "type": "function", "function": {"name": "robot.read_manual", "arguments": {"z": 1, "a": [1.0]}}}]})
    elif kind == "args_bad_json":
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c-b", "type": "function", "function": {"name": "observe", "arguments": "{bad"}}]})
    elif kind == "args_list":
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c-l", "type": "function", "function": {"name": "observe", "arguments": "[1]"}}]})
    elif kind == "no_call_id":
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "observe", "arguments": "{}"}}]})
    elif kind == "tools_dict":
        request["tools"] = {"a": 1}
    elif kind == "tool_no_function":
        request["tools"] = [{"type": "function"}]
    elif kind == "tool_bad_params":
        request["tools"] = [{"type": "function", "function": {"name": "x", "parameters": "nope"}}]
    return kind


def strip_openai_continuations(rng, request):
    if rng.random() < 0.8:
        for message in request.get("messages", []):
            if isinstance(message, dict):
                message.pop("_provider_continuation_state", None)
                message.pop("reasoning_content", None)


# -- event streams ---------------------------------------------------------------------------

TOOL_NAMES = STRICT[:4] + [anthropic_request._anthropic_tool_alias(name) for name in NEEDS_ALIAS[:3]]


def wire_tools(rng):
    tools = []
    for name in rng.sample(TOOL_NAMES, rng.randint(0, 4)):
        tools.append({"name": name, "description": "d", "input_schema": {"type": "object", "properties": {
            "targets": {"type": rng.choice(["object", "array", "string"])},
            "position_m": {"type": "array"}, "label": {"type": "string"}}}})
    return tools


def split_text(rng, text):
    parts, index = [], 0
    while index < len(text):
        step = rng.randint(1, max(1, len(text)))
        parts.append(text[index:index + step])
        index += step
    return parts or [""]


def valid_stream(rng):
    events = [{"type": "message_start", "message": {"id": rng.choice(["msg_01", "msg-x", ""]),
                                                   "type": "message", "role": "assistant",
                                                   "usage": {"input_tokens": rng.randint(1, 9000),
                                                             "cache_read_input_tokens": rng.choice([0, 4096, None])}}}]
    if events[0]["message"]["usage"]["cache_read_input_tokens"] is None:
        del events[0]["message"]["usage"]["cache_read_input_tokens"]
    index = 0
    kinds = rng.sample(["thinking", "text", "tool", "tool", "redacted"], rng.randint(1, 4))
    for kind in kinds:
        if rng.random() < 0.15:
            events.append({"type": "ping"})
        if kind == "text":
            events.append({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
            for part in split_text(rng, rng.choice(["I will observe.", "思考：先观察。", "a\nb", ""])):
                events.append({"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": part}})
        elif kind == "thinking":
            events.append({"type": "content_block_start", "index": index,
                           "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            for part in split_text(rng, rng.choice(["Let me think.", "step 1\nstep 2", "x" * 30])):
                events.append({"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": part}})
            for part in split_text(rng, rng.choice(["EqQBCkYIBxgCKkA", "ab-_cd", "sig=="])):
                events.append({"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": part}})
        elif kind == "redacted":
            events.append({"type": "content_block_start", "index": index,
                           "content_block": {"type": "redacted_thinking", "data": "opaque-blob"}})
        else:
            name = rng.choice(TOOL_NAMES)
            arguments = rng.choice([
                {"views": ["agentview"]}, {}, {"targets": json.dumps({"right": {"x": 1.0}})},
                {"position_m": "[-0.1, 0.0, 0.9]", "label": "{not json"}, {"targets": "[1,2]"},
                {"z": 1.5, "a": "ü"},
            ])
            events.append({"type": "content_block_start", "index": index,
                           "content_block": {"type": "tool_use", "id": rng.choice([f"toolu_{index}", ""]),
                                             "name": name, "input": {}}})
            for part in split_text(rng, json.dumps(arguments, ensure_ascii=rng.random() < 0.5)):
                events.append({"type": "content_block_delta", "index": index,
                               "delta": {"type": "input_json_delta", "partial_json": part}})
        events.append({"type": "content_block_stop", "index": index})
        index += 1
    stop = rng.choice(["end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal"])
    events.append({"type": "message_delta", "delta": {"stop_reason": stop},
                   "usage": {"output_tokens": rng.randint(0, 900), "cache_creation_input_tokens": rng.choice([0, 12])}})
    events.append({"type": "message_stop"})
    return events


def corrupt(rng, events):
    kind = rng.choice([
        "none", "none", "drop_stop", "drop_block_stop", "dup_start", "delta_wrong_block", "bad_delta",
        "error_overloaded", "error_other", "after_stop", "bad_json_tool", "list_json_tool", "non_stream",
        "non_stream_bad_block", "unknown_event", "unknown_block", "reuse_index", "no_signature",
        "missing_tool_id", "float_index", "bool_index", "empty_stream", "mixed",
    ])
    events = copy.deepcopy(events)
    if kind == "drop_stop":
        events = [event for event in events if event["type"] != "message_stop"]
    elif kind == "drop_block_stop":
        stops = [i for i, event in enumerate(events) if event["type"] == "content_block_stop"]
        if stops:
            del events[rng.choice(stops)]
    elif kind == "dup_start":
        events.insert(1, copy.deepcopy(events[0]))
    elif kind == "delta_wrong_block":
        events.insert(len(events) - 2, {"type": "content_block_delta", "index": 99, "delta": {"type": "text_delta", "text": "x"}})
    elif kind == "bad_delta":
        events.insert(2, {"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta"}})
    elif kind == "error_overloaded":
        events.insert(rng.randint(0, len(events)), {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
    elif kind == "error_other":
        events.insert(rng.randint(0, len(events)), {"type": "error", "error": "plain"})
    elif kind == "after_stop":
        events.append({"type": "ping"})
    elif kind in ("bad_json_tool", "list_json_tool"):
        for event in events:
            if event.get("delta", {}).get("type") == "input_json_delta":
                event["delta"]["partial_json"] = "{bad" if kind == "bad_json_tool" else "[1, 2]"
                break
    elif kind == "non_stream":
        events = [{"id": "msg_ns", "type": "message", "role": "assistant", "stop_reason": rng.choice(["end_turn", "tool_use"]),
                   "usage": {"input_tokens": 5, "output_tokens": 3},
                   "content": [{"type": "text", "text": "hello"},
                               {"type": "tool_use", "id": "toolu_ns", "name": "observe", "input": {"views": ["a"]}}]}]
    elif kind == "non_stream_bad_block":
        events = [{"id": "m", "content": ["text"], "stop_reason": "end_turn"}]
    elif kind == "unknown_event":
        events.insert(1, {"type": "message_progress"})
    elif kind == "unknown_block":
        events.insert(1, {"type": "content_block_start", "index": 7, "content_block": {"type": "server_tool_use"}})
    elif kind == "reuse_index":
        starts = [event for event in events if event["type"] == "content_block_start"]
        if starts:
            events.insert(len(events) - 2, copy.deepcopy(starts[0]))
    elif kind == "no_signature":
        events = [event for event in events if event.get("delta", {}).get("type") != "signature_delta"]
    elif kind == "missing_tool_id":
        for event in events:
            block = event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                block.pop("id", None)
    elif kind == "float_index":
        for event in events:
            if event["type"] == "content_block_stop":
                event["index"] = float(event["index"])
                break
    elif kind == "bool_index":
        for event in events:
            if event["type"] in ("content_block_start", "content_block_delta", "content_block_stop") and event["index"] in (0, 1):
                event["index"] = bool(event["index"])
    elif kind == "empty_stream":
        events = []
    elif kind == "mixed":
        events.append({"id": "x", "content": []})
    return kind, events


def answer_parse(events, model_call_id, model, aliases, declared):
    try:
        message, usage, finish, rejection = anthropic_messages.AnthropicMessagesProvider._parse_events(
            events, model_call_id, model, aliases, declared)
    except PolicySessionError as exc:
        return {"error": str(exc), "retryable": getattr(exc, "retryable", None)}
    except Exception as exc:
        return {"crash": type(exc).__name__, "message": str(exc)}
    return {"message": canonical_text(message), "usage": canonical_text(usage), "finish": finish,
            "rejection": canonical_text(rejection)}


def usage_cases(rng):
    cases = []
    for _ in range(60):
        value = {}
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens", "other"):
            if rng.random() < 0.6:
                value[key] = rng.choice([0, 1, 12345, -3, 2.0, 7.5, True, None, "9"])
        cases.append({"value": value, "answer": canonical_text(anthropic_reply.normalize_anthropic_usage(value))})
    cases.append({"value": None, "answer": canonical_text(anthropic_reply.normalize_anthropic_usage(None))})
    return cases


def signature_cases():
    cases = []
    for signature in ["", "abcd", "abc", "ab-_", "a+/b", "ab$c", "abcde", "AAAA====", "ü", "EqQBCkYIBxgCKkA",
                      "AAAA=", "AAA=A", "A===", "AA==AA==", "AAA=", "AA==", "A=AA", "AAAAA===", "AAAA=AAA",
                      "====", "=", "AB==", "ab-_cd==", "a-b_", "a+b-", "AA=", "AAA", "A"]:
        body = {"messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": signature}]}]}
        cases.append({"body": body, "answer": _has_complete_anthropic_thinking_signature(body)})
    cases.append({"body": {"messages": "x"}, "answer": False})
    return cases


def image_cases():
    """Data-URL base64 validity as `_anthropic_content` judges it (b64decode validate=True)."""
    cases = []
    for data in ["AAAA", "AAA", "AA==", "AAA=", "AAAA=", "AAAA====", "A===", "=AAA", "A=AA", "AA==AA==", "", "ab$c",
                 "AAAAA===", "AB==", "a-b_"]:
        request = {"model": "m", "max_tokens": 8, "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}]}]}
        cases.append({"id": f"image:{data}", "request": request, "answer": answer_request(request)})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--per-env", type=int, default=6)
    parser.add_argument("--sealed-cache", type=Path, default=None,
                        help="where sealed runs are extracted (default: a temporary directory)")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    translate = image_cases()
    with tempfile.TemporaryDirectory(prefix="ora-sealed-") as temporary:
        cases, real_sources = real_requests(args.per_env, args.sealed_cache or Path(temporary))
        for case in cases:
            translate.append({**case, "answer": answer_request(case["request"])})
    for index in range(args.count):
        request, _ = make_request(rng, f"r{index}")
        strip_openai_continuations(rng, request)
        usable = isinstance(request.get("messages"), list) and isinstance(request.get("tools", []), list)
        mutation = anthropic_mutations(rng, request) if usable and rng.random() < 0.6 else "none"
        translate.append({"id": f"rand:{index}:{mutation}", "request": request, "answer": answer_request(request)})

    parses, continuations = [], []
    for index in range(args.count):
        corruption, events = corrupt(rng, valid_stream(rng))
        tools = wire_tools(rng)
        declared = anthropic_request._declared_property_types(tools) if rng.random() < 0.5 else None
        aliases = {anthropic_request._anthropic_tool_alias(name): name for name in NEEDS_ALIAS[:3]} if rng.random() < 0.7 else None
        model = rng.choice(["claude-sonnet", "kimi-for-coding"])
        answer = answer_parse(events, f"model-call-{index:06d}", model, aliases, declared)
        parses.append({"id": f"parse:{index}:{corruption}", "events": events, "model": model,
                       "aliases": aliases, "declared_tools": tools if declared is not None else None,
                       "answer": answer})
        if "message" in answer:
            message, *_ = anthropic_messages.AnthropicMessagesProvider._parse_events(
                events, f"model-call-{index:06d}", model, aliases, declared)
            if "_provider_continuation_state" not in message:
                continue
            request = {"model": model, "max_tokens": 4096, "stream": True, "messages": [
                {"role": "user", "content": "go"}, message]}
            for call in message.get("tool_calls", []):
                request["messages"].append({"role": "tool", "tool_call_id": call["id"], "content": "{\"ok\": true}"})
            tweak = rng.choice(["none", "none", "tamper_payload", "other_model", "drop_tools_from_request", "edit_text"])
            if tweak == "tamper_payload":
                state = request["messages"][1]["_provider_continuation_state"]
                state["native_payload"]["content"][0]["extra"] = 1
            elif tweak == "other_model":
                request["model"] = "other-model"
            elif tweak == "edit_text":
                request["messages"][1]["content"] = request["messages"][1]["content"] + "!"
            if tweak != "drop_tools_from_request":
                request["tools"] = [{"type": "function", "function": {"name": name, "description": "", "parameters": {"type": "object"}}}
                                    for name in (aliases or {}).values()]
            continuations.append({"id": f"cont:{index}:{tweak}", "request": request, "answer": answer_request(request)})

    output = {"translate": translate, "parse": parses, "continuation": continuations,
              "usage": usage_cases(rng), "signature": signature_cases(),
              "collider": COLLIDER, "real_sources": real_sources}
    args.out.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    errors = sum(1 for case in translate if "error" in case["answer"])
    print(json.dumps({"translate": len(translate), "translate_errors": errors, "parse": len(parses),
                      "parse_errors": sum(1 for case in parses if "error" in case["answer"]),
                      "continuation": len(continuations)}))


if __name__ == "__main__":
    main()
