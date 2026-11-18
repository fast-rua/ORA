"""Translate the canonical message IR and its tool definitions into one Anthropic
Messages request.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from typing import Any, Mapping, Sequence

from ora.agent.contracts import PolicySessionError
from ora.agent.providers.continuation import (
    PROVIDER_CONTINUATION_MESSAGE_KEY,
    ProviderContinuationState,
    canonical_assistant_sha256,
)
from ora.agent.providers.tool_names import tool_alias


_DATA_URL = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)


def _anthropic_tool_alias(name: str) -> str:
    """Encode canonical dotted names for strict Anthropic-compatible gateways."""

    return tool_alias(name)


def _anthropic_tool_names(request: Mapping[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    raw_tools = request.get("tools", [])
    if not isinstance(raw_tools, list):
        return aliases
    for item in raw_tools:
        function = item.get("function") if isinstance(item, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str):
            continue
        alias = _anthropic_tool_alias(name)
        previous = aliases.setdefault(alias, name)
        if previous != name:
            raise PolicySessionError("Anthropic tool-name alias collision")
    return aliases


def _anthropic_content(content: Any) -> list[dict[str, Any]]:
    """Translate canonical text/image content into Anthropic blocks."""

    if isinstance(content, str):
        return [] if not content else [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise PolicySessionError("canonical message content must be text or a block list")
    blocks: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise PolicySessionError("canonical message content block must be an object")
        block_type = item.get("type")
        if block_type == "text" and isinstance(item.get("text"), str):
            blocks.append({"type": "text", "text": item["text"]})
            continue
        if block_type == "image_url":
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
            if not isinstance(url, str):
                raise PolicySessionError("canonical image_url block lacks a URL")
            match = _DATA_URL.fullmatch(url)
            if match is None:
                raise PolicySessionError(
                    "Anthropic adapter accepts only inline base64 image data URLs"
                )
            media_type, data = match.groups()
            try:
                base64.b64decode(data, validate=True)
            except Exception as exc:
                raise PolicySessionError("canonical image data URL is invalid base64") from exc
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            })
            continue
        raise PolicySessionError(f"unsupported canonical content block: {block_type!r}")
    return blocks


def _declared_property_types(
    tools: Any,
) -> dict[str, dict[str, str]]:
    """Map each tool's top-level property names to their declared JSON type.

    Used to decide which stringified arguments are safe to decode. Only the
    tool's own schema can answer that; guessing from the value's shape cannot.
    """

    declared: dict[str, dict[str, str]] = {}
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return declared
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not isinstance(schema, Mapping):
            continue
        types = _property_types(schema)
        if types:
            declared[name] = types
    return declared


def _property_types(schema: Mapping[str, Any]) -> dict[str, str]:
    """Collect property types, including through a top-level oneOf.

    move_pose declares its flat and enhanced forms as oneOf branches, so its
    properties live one level down and a reader that only looks at
    schema["properties"] sees an empty tool. Where branches disagree about a
    property's type the entry is dropped rather than guessed: an ambiguous
    declaration is not a basis for rewriting a value.
    """

    collected: dict[str, str] = {}
    ambiguous: set[str] = set()

    def absorb(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        properties = node.get("properties")
        if isinstance(properties, Mapping):
            for key, value in properties.items():
                if not (isinstance(key, str) and isinstance(value, Mapping)):
                    continue
                kind = value.get("type")
                if not isinstance(kind, str):
                    continue
                if key in collected and collected[key] != kind:
                    ambiguous.add(key)
                collected[key] = kind
        for branch_key in ("oneOf", "anyOf", "allOf"):
            branches = node.get(branch_key)
            if isinstance(branches, Sequence) and not isinstance(
                branches, (str, bytes)
            ):
                for branch in branches:
                    absorb(branch)

    absorb(schema)
    for key in ambiguous:
        collected.pop(key, None)
    return collected


def _decode_stringified_object_arguments(
    arguments: dict[str, Any],
    declared_types: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Recover object-valued tool arguments that arrived as JSON text.

    Qwen3's tool calls are XML, where every parameter value is text, and a
    server-side parser has to restore the declared types. vLLM's qwen3_xml
    parser does not always restore nested objects, so a call whose schema
    declares `targets: {...}` can arrive as the *string*
    '{"right": {"position": ...}}'. The ToolSurface then rejects it as
    invalid_motion_arguments, and the policy burns calls guessing at a format
    it had already produced correctly.

    What is safe to decode comes from the tool's own schema, not from the shape
    of the value. Passing declared_types decodes a string exactly when the
    property is declared "object" or "array", which leaves a genuine string
    parameter alone even when its text happens to look like JSON. Without it,
    only objects are decoded, since guessing is far more likely to corrupt a
    real string than to help.

    The array case is not hypothetical. Measured over three recorded pilot cells,
    the surviving invalid_motion_arguments rejections were dominated by
      {"position_m": "[-0.119, -0.003, 0.975]", ...}
    where the vector, not the enclosing object, is the value left as text.

    This is a transport repair, not schema coercion: it cannot fix a wrong key
    name, and a genuinely malformed argument still reaches the ToolSurface and
    is still reported.
    """

    decodable = {"object", "array"}
    repaired: dict[str, Any] = {}
    for name, value in arguments.items():
        if declared_types is not None:
            wanted = declared_types.get(name)
            opener = {"object": "{", "array": "["}.get(wanted or "")
            eligible = (
                isinstance(value, str)
                and wanted in decodable
                and value.lstrip()[:1] == opener
            )
        else:
            eligible = isinstance(value, str) and value.lstrip()[:1] == "{"
        if eligible:
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                repaired[name] = value
                continue
            # Accept the decoded value only if it matches what the schema
            # declared. A string that parses to the wrong kind is more likely a
            # genuine text argument than a transport artefact.
            expected = (
                dict if (declared_types or {}).get(name) == "object" else list
            ) if declared_types is not None else dict
            repaired[name] = decoded if isinstance(decoded, expected) else value
        else:
            repaired[name] = value
    return repaired


def _anthropic_tool_input(call: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    function = call.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise PolicySessionError("canonical tool call lacks a function name")
    raw = function.get("arguments", "{}")
    if isinstance(raw, Mapping):
        arguments = dict(raw)
    elif isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PolicySessionError("canonical tool arguments are invalid JSON") from exc
    else:
        raise PolicySessionError("canonical tool arguments must be JSON text or an object")
    if not isinstance(arguments, dict):
        raise PolicySessionError("canonical tool arguments must decode to an object")
    # History is already canonical; preserve string values exactly on replay.
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise PolicySessionError("canonical tool call lacks an ID")
    return call_id, function["name"], arguments


_ANTHROPIC_CONTINUATION_KIND = "anthropic_assistant_content_blocks.v1"


def _normalize_anthropic_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    model_call_id: str,
    tool_names_by_alias: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Normalize finalized Anthropic assistant blocks into canonical IR."""

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    normalized_calls: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise PolicySessionError("Anthropic text block lacks text")
            text_parts.append(text)
            continue
        if block_type == "thinking":
            thinking = block.get("thinking")
            signature = block.get("signature")
            if not isinstance(thinking, str):
                raise PolicySessionError("Anthropic thinking block lacks thinking text")
            if not isinstance(signature, str) or not signature:
                raise PolicySessionError(
                    "Anthropic thinking block lacks a replayable signature"
                )
            thinking_parts.append(thinking)
            continue
        if block_type == "redacted_thinking":
            data = block.get("data")
            if not isinstance(data, str) or not data:
                raise PolicySessionError(
                    "Anthropic redacted_thinking block lacks opaque data"
                )
            continue
        if block_type == "tool_use":
            arguments = block.get("input")
            if not isinstance(arguments, Mapping):
                raise PolicySessionError("Anthropic tool input must be an object")
            name = block.get("name")
            if not isinstance(name, str) or not name:
                raise PolicySessionError("Anthropic tool_use lacks a name")
            if tool_names_by_alias is not None:
                name = tool_names_by_alias.get(name, name)
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"{model_call_id}-tool-{index:02d}"
            normalized_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        dict(arguments), ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            })
            continue
        raise PolicySessionError(
            f"unsupported Anthropic assistant content block: {block_type!r}"
        )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if thinking_parts:
        message["reasoning_content"] = "".join(thinking_parts)
    if normalized_calls:
        message["tool_calls"] = normalized_calls
    return message


def _anthropic_continuation_blocks(
    message: Mapping[str, Any],
    *,
    model: str,
    tool_names_by_alias: Mapping[str, str],
) -> list[dict[str, Any]] | None:
    raw_state = message.get(PROVIDER_CONTINUATION_MESSAGE_KEY)
    if raw_state is None:
        return None
    if not isinstance(raw_state, Mapping):
        raise PolicySessionError("provider continuation state must be an object")
    state = ProviderContinuationState.from_mapping(raw_state)
    state.validate_binding(
        message=message,
        provider_protocol="anthropic_messages",
        model=model,
    )
    if state.native_kind != _ANTHROPIC_CONTINUATION_KIND:
        raise PolicySessionError("unsupported Anthropic continuation native kind")
    raw_blocks = state.native_payload.get("content")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise PolicySessionError("Anthropic continuation lacks assistant content blocks")
    if not all(isinstance(block, Mapping) for block in raw_blocks):
        raise PolicySessionError("Anthropic continuation content block is invalid")
    blocks = [copy.deepcopy(dict(block)) for block in raw_blocks]
    # A continuation is bound to the assistant message produced by the
    # request that created it.  The next request may intentionally expose a
    # different tool set.  Recover historical wire aliases from
    # the bound canonical tool calls themselves instead of requiring every
    # historical tool to remain in the next request's tool definitions.
    replay_tool_names = dict(tool_names_by_alias)
    canonical_calls = message.get("tool_calls")
    canonical_calls = canonical_calls if isinstance(canonical_calls, list) else []
    canonical_names_by_id: dict[str, str] = {}
    for raw_call in canonical_calls:
        if not isinstance(raw_call, Mapping):
            continue
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
            canonical_names_by_id[call_id] = name
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        call_id = block.get("id")
        wire_name = block.get("name")
        canonical_name = canonical_names_by_id.get(call_id)
        if not (
            isinstance(wire_name, str)
            and wire_name
            and isinstance(canonical_name, str)
            and canonical_name
        ):
            continue
        previous = replay_tool_names.setdefault(wire_name, canonical_name)
        if previous != canonical_name:
            raise PolicySessionError(
                "Anthropic continuation tool-name binding conflicts with the request"
            )
    normalized = _normalize_anthropic_blocks(
        blocks,
        model_call_id=state.response_id or "provider-continuation",
        tool_names_by_alias=replay_tool_names,
    )
    if canonical_assistant_sha256(normalized) != state.canonical_message_sha256:
        raise PolicySessionError(
            "Anthropic continuation payload does not match its canonical assistant"
        )
    return blocks


def translate_anthropic_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the canonical native request into an Anthropic Messages body."""

    forbidden = sorted(set(request) & {
        "input", "instructions", "system", "thinking", "output_config",
        "input_schema",
    })
    if forbidden:
        raise PolicySessionError(
            "Anthropic canonical request contains wire/Responses fields: "
            + ", ".join(forbidden)
        )
    model = request.get("model")
    max_tokens = request.get("max_tokens")
    messages = request.get("messages")
    if not isinstance(model, str) or not model:
        raise PolicySessionError("canonical provider request lacks a model")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise PolicySessionError("canonical provider request has invalid max_tokens")
    if not isinstance(messages, list):
        raise PolicySessionError("canonical provider request messages must be a list")

    system_blocks: list[dict[str, Any]] = []
    translated: list[dict[str, Any]] = []
    tool_names_by_alias = _anthropic_tool_names(request)

    def append_message(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if translated and translated[-1]["role"] == role:
            existing = translated[-1]["content"]
            if (
                blocks[0].get("type") == "tool_result"
                and any(item.get("type") != "tool_result" for item in existing)
            ):
                raise PolicySessionError(
                    "Anthropic tool results must immediately follow tool_use blocks"
                )
            existing.extend(blocks)
        else:
            translated.append({"role": role, "content": blocks})

    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            raise PolicySessionError("canonical provider message must be an object")
        role = raw_message.get("role")
        if role == "system":
            system_blocks.extend(_anthropic_content(raw_message.get("content", "")))
            continue
        if role == "user":
            append_message("user", _anthropic_content(raw_message.get("content", "")))
            continue
        if role == "assistant":
            blocks = _anthropic_continuation_blocks(
                raw_message,
                model=model,
                tool_names_by_alias=tool_names_by_alias,
            )
            if blocks is None:
                blocks = _anthropic_content(raw_message.get("content", ""))
                calls = raw_message.get("tool_calls")
                if isinstance(calls, list):
                    for raw_call in calls:
                        if not isinstance(raw_call, Mapping):
                            raise PolicySessionError("canonical tool call must be an object")
                        call_id, name, arguments = _anthropic_tool_input(raw_call)
                        blocks.append({
                            "type": "tool_use", "id": call_id,
                            "name": _anthropic_tool_alias(name), "input": arguments,
                        })
                # Normalized reasoning without a provider continuation is trace
                # evidence only.  It must never be forged into an unsigned
                # Anthropic thinking block.
            append_message("assistant", blocks)
            continue
        if role == "tool":
            call_id = raw_message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise PolicySessionError("canonical tool result lacks tool_call_id")
            content = raw_message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, sort_keys=True)
            is_error = False
            try:
                decoded = json.loads(content)
                is_error = isinstance(decoded, Mapping) and decoded.get("ok") is False
            except json.JSONDecodeError:
                pass
            block: dict[str, Any] = {
                "type": "tool_result", "tool_use_id": call_id, "content": content,
            }
            if is_error:
                block["is_error"] = True
            append_message("user", [block])
            continue
        raise PolicySessionError(f"unsupported canonical message role: {role!r}")

    renamed_tools = {
        canonical: alias for alias, canonical in tool_names_by_alias.items()
        if alias != canonical
    }
    if renamed_tools:
        # Anthropic-compatible gateways reject dots in tool names. Keep the
        # canonical request and task text untouched, but disclose the exact
        # provider-wire aliases so runtime facts that name capabilities cannot
        # disagree with the callable tools the model actually sees.
        system_blocks.append({
            "type": "text",
            "text": (
                "<wire_tool_aliases>\n"
                + json.dumps(renamed_tools, ensure_ascii=False, sort_keys=True)
                + "\n</wire_tool_aliases>"
            ),
        })

    tools: list[dict[str, Any]] = []
    raw_tools = request.get("tools", [])
    if not isinstance(raw_tools, list):
        raise PolicySessionError("canonical provider tools must be a list")
    for item in raw_tools:
        function = item.get("function") if isinstance(item, Mapping) else None
        if not isinstance(function, Mapping):
            raise PolicySessionError("canonical tool definition lacks function")
        name = function.get("name")
        schema = function.get("parameters")
        if not isinstance(name, str) or not isinstance(schema, Mapping):
            raise PolicySessionError("canonical tool definition is incomplete")
        wire_schema = copy.deepcopy(dict(schema))
        if "type" not in wire_schema and "oneOf" in wire_schema:
            # Anthropic-compatible gateways require
            # input_schema.type == "object"; a top-level oneOf whose
            # branches are all objects is semantically identical with the
            # explicit type, so the wire projection states it. The
            # composition-layer schema bytes stay untouched.
            wire_schema = {"type": "object", **wire_schema}
        tool = {
            "name": _anthropic_tool_alias(name),
            "description": function.get("description", ""),
            "input_schema": wire_schema,
        }
        tools.append(tool)

    body: dict[str, Any] = {
        "model": model,
        "messages": translated,
        "max_tokens": max_tokens,
        "stream": bool(request.get("stream", False)),
    }
    if system_blocks:
        body["system"] = system_blocks
    if tools:
        body["tools"] = tools
    tool_choice = request.get("tool_choice")
    if tool_choice == "auto":
        body["tool_choice"] = {"type": "auto"}
    elif tool_choice == "none":
        body["tool_choice"] = {"type": "none"}
    elif isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str):
            body["tool_choice"] = {
                "type": "tool", "name": _anthropic_tool_alias(name),
            }
    for field in ("temperature", "top_p"):
        if field in request:
            body[field] = request[field]
    reasoning_effort = request.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort != "provider_default":
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise PolicySessionError(
                "Anthropic reasoning_effort cannot be projected to output_config"
            )
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": reasoning_effort}
    stop = request.get("stop")
    if isinstance(stop, str):
        body["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        body["stop_sequences"] = stop
    return body
