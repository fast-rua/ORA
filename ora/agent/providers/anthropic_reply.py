"""Turn one accumulated Anthropic Messages response into the canonical assistant
reply, and normalize its usage.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError
from .openai_wire_options import nonnegative_count as _number
from .anthropic_events import AnthropicResponseEvents
from ora.agent.providers.continuation import (
    PROVIDER_CONTINUATION_MESSAGE_KEY,
    ProviderContinuationState,
)
from .anthropic_request import (
    _ANTHROPIC_CONTINUATION_KIND,
    _decode_stringified_object_arguments,
    _normalize_anthropic_blocks,
)
from .transport_failures import _reject_truncated_tools


def normalize_anthropic_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    uncached = _number(value.get("input_tokens"))
    cache_read = _number(value.get("cache_read_input_tokens"))
    cache_write = _number(value.get("cache_creation_input_tokens"))
    output = _number(value.get("output_tokens"))
    if all(item is None for item in (uncached, cache_read, cache_write, output)):
        return None
    total_input = sum(item or 0 for item in (uncached, cache_read, cache_write))
    output_tokens = output or 0
    return {
        "input_tokens": total_input,
        "cached_input_tokens": cache_read or 0,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "cache_read_status": (
            "reported" if "cache_read_input_tokens" in value else "missing"
        ),
        "cache_creation_status": (
            "reported" if "cache_creation_input_tokens" in value else "missing"
        ),
        "output_tokens": output_tokens,
        "reasoning_output_tokens": 0,
        "total_tokens": total_input + output_tokens,
    }


def _anthropic_blocks_with_tool_inputs(
    response: AnthropicResponseEvents,
    declared_types_by_tool: Mapping[str, Mapping[str, str]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Blocks in index order with decoded tool inputs, and the inputs rejected."""

    blocks: list[dict[str, Any]] = []
    rejected_tool_inputs: list[dict[str, Any]] = []
    for index, block in sorted(response.blocks_by_index.items()):
        fragments = "".join(response.tool_fragments.get(index, []))
        if fragments:
            try:
                # Same transport repair as the non-streaming path: a parser
                # that leaves an object-valued argument as JSON text would
                # otherwise reach the ToolSurface as a string. The Harness
                # streams, so this is the path that actually runs.
                arguments = json.loads(fragments)
                if isinstance(arguments, dict):
                    arguments = _decode_stringified_object_arguments(
                        arguments,
                        (declared_types_by_tool or {}).get(
                            str(block.get("name") or "")
                        ),
                    )
            except json.JSONDecodeError:
                rejected_tool_inputs.append({
                    "index": index,
                    "tool_call_id": block.get("id"),
                    "tool_name": block.get("name"),
                    "fragment_chars": len(fragments),
                    "fragment_sha256": hashlib.sha256(
                        fragments.encode("utf-8")
                    ).hexdigest(),
                    "reason": "invalid_json",
                })
            else:
                if not isinstance(arguments, Mapping):
                    rejected_tool_inputs.append({
                        "index": index,
                        "tool_call_id": block.get("id"),
                        "tool_name": block.get("name"),
                        "fragment_chars": len(fragments),
                        "fragment_sha256": hashlib.sha256(
                            fragments.encode("utf-8")
                        ).hexdigest(),
                        "reason": "not_an_object",
                    })
                else:
                    block["input"] = copy.deepcopy(dict(arguments))
        if block.get("type") == "tool_use":
            if (
                not isinstance(block.get("input"), Mapping)
                and not any(
                    item["index"] == index for item in rejected_tool_inputs
                )
            ):
                raise PolicySessionError("Anthropic tool use lacks complete input")
            if not isinstance(block.get("id"), str) or not block.get("id"):
                raise PolicySessionError("Anthropic tool use lacks a provider call id")
        blocks.append(block)
    return blocks, rejected_tool_inputs


def _rejected_tool_input_reply(
    response: AnthropicResponseEvents,
    blocks: list[dict[str, Any]],
    rejected_tool_inputs: list[dict[str, Any]],
    *,
    model_call_id: str,
    tool_names_by_alias: Mapping[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str, dict[str, Any]]:
    # Never guess or repair robot tool arguments.  A complete Provider
    # stream with malformed tool JSON is a recoverable no-world-change
    # response: retain its exact raw events, discard every tool block
    # from the canonical conversation, and let the policy retry on the
    # same observation through an explicitly counted model call.
    safe_blocks = [
        block for block in blocks if block.get("type") != "tool_use"
    ]
    message = _normalize_anthropic_blocks(
        safe_blocks,
        model_call_id=model_call_id,
        tool_names_by_alias=tool_names_by_alias,
    )
    rejection = {
        "code": "invalid_tool_input_json",
        "provider_stop_reason": response.stop_reason,
        "rejected_tool_blocks": rejected_tool_inputs,
        "discarded_tool_block_count": sum(
            block.get("type") == "tool_use" for block in blocks
        ),
        "policy_outcome": "continue_same_observation",
        "world_state_changed": False,
    }
    return (
        message,
        normalize_anthropic_usage(response.usage_values),
        "invalid_tool_input",
        rejection,
    )


def _anthropic_reply(
    response: AnthropicResponseEvents,
    blocks: list[dict[str, Any]],
    *,
    model_call_id: str,
    model: str,
    tool_names_by_alias: Mapping[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """The finish reason and canonical message of a reply whose tool inputs decoded.

    An output-limit stop that proposes tools is rejected whole; a reply with
    thinking blocks binds them as its provider continuation.
    """

    stop_reason = response.stop_reason
    finish_reason = {
        "tool_use": "tool_calls",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
    }.get(stop_reason, stop_reason)
    message = _normalize_anthropic_blocks(
        blocks,
        model_call_id=model_call_id,
        tool_names_by_alias=tool_names_by_alias,
    )
    if finish_reason == "length" and message.get("tool_calls"):
        message, rejection = _reject_truncated_tools(message, provider_stop_reason=stop_reason)
        return message, normalize_anthropic_usage(response.usage_values), "invalid_tool_input", rejection
    if any(
        block.get("type") in {"thinking", "redacted_thinking"}
        for block in blocks
    ):
        continuation = ProviderContinuationState.create(
            provider_protocol="anthropic_messages",
            model=model,
            native_kind=_ANTHROPIC_CONTINUATION_KIND,
            native_payload={"content": blocks},
            canonical_message=message,
            response_id=response.response_id,
        )
        message[PROVIDER_CONTINUATION_MESSAGE_KEY] = continuation.to_dict()
    return message, normalize_anthropic_usage(response.usage_values), finish_reason, None
