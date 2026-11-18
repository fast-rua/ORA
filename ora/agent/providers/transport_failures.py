"""Transport failures both wire adapters share: which errors a retry may cover, the
closed-provider error, bounded failure metadata on a raised exception, and the
refusal of a reply whose tool calls were truncated.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError
from ora.agent.providers.attempts import credential_echo_failure
from ora.agent.providers.recovery import transient_transport


_KIMI_CODING_BASE_URLS = {
    "https://api.kimi.com/coding",
    "https://api.kimi.com/coding/v1",
    "https://api.kimi.com/coding/v1/messages",
}
_KIMI_TRANSIENT_SIGNATURE_REJECTION = (
    "malformed encrypted reasoning content: invalid base64url encoding"
)


def _retryable_transport_error(exc: Exception) -> bool:
    return transient_transport(exc)


def _reject_truncated_tools(
    message: Mapping[str, Any], *, provider_stop_reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject the entire capped tool turn, preserving its wire as evidence.

    Even syntactically complete arguments cannot turn an output-limit stop into
    a completed proposal. Do not replay reasoning signed against discarded
    tools or synthesize a signature for the edited assistant message.
    """
    calls = message["tool_calls"]
    rejected = []
    for index, call in enumerate(calls):
        function = call.get("function", {})
        fragment = function.get("arguments", "")
        rejected.append({"index": index, "tool_call_id": call.get("id"),
            "tool_name": function.get("name"), "fragment_chars": len(fragment),
            "fragment_sha256": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
            "reason": "provider_output_limit"})
    return {"role": "assistant", "content": message.get("content", "")}, {
        "code": "truncated_tool_call", "provider_stop_reason": provider_stop_reason,
        "rejected_tool_blocks": rejected, "discarded_tool_block_count": len(calls),
        "policy_outcome": "continue_same_observation", "world_state_changed": False,
    }


def _invalid_arguments_reason(call: Mapping[str, Any]) -> str | None:
    """Why a complete call's arguments cannot be dispatched, or None (the agent core's decode rule)."""
    function = call.get("function", {})
    if not isinstance(function, Mapping) or "arguments" not in function:
        return None
    raw = function["arguments"]
    if not isinstance(raw, str):
        return None if isinstance(raw, Mapping) else "not_an_object"
    try:
        value = json.loads(raw)
    except ValueError:
        return "invalid_json"
    return None if isinstance(value, dict) else "not_an_object"


def _reject_invalid_tool_arguments(
    message: Mapping[str, Any], *, provider_stop_reason: str | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Reject a complete tool turn whose arguments do not decode to an object.

    As on the Anthropic path (``invalid_tool_input_json``): never guess or
    repair robot tool arguments. Keep the raw events, discard every tool call
    from the canonical conversation and let the policy retry on the same
    observation. Replayed, such a call also made strict servers (SGLang)
    refuse every later request ("function.arguments must be valid JSON").
    """
    calls = message.get("tool_calls") or []
    rejected = []
    for index, call in enumerate(calls):
        reason = _invalid_arguments_reason(call)
        if reason is None:
            continue
        function = call.get("function", {})
        raw = function.get("arguments")
        fragment = raw if isinstance(raw, str) else json.dumps(raw)
        rejected.append({"index": index, "tool_call_id": call.get("id"),
            "tool_name": function.get("name"), "fragment_chars": len(fragment),
            "fragment_sha256": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
            "reason": reason})
    if not rejected:
        return None
    return {"role": "assistant", "content": message.get("content", "")}, {
        "code": "invalid_tool_input_json", "provider_stop_reason": provider_stop_reason,
        "rejected_tool_blocks": rejected, "discarded_tool_block_count": len(calls),
        "policy_outcome": "continue_same_observation", "world_state_changed": False,
    }


def _record_provider_failure(exc: Exception, attempts: int) -> None:
    # Bounded metadata only: exception messages can contain endpoint/credential
    # details. The loop records this alongside the already sealed request ID.
    setattr(exc, "provider_failure", {
        "exception_type": type(exc).__name__,
        "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
        "attempts": attempts,
        "response_started": bool(getattr(exc, "response_started", False)),
    })


def _echo_failure(cause: Exception, attempts: int) -> PolicySessionError:
    failure = credential_echo_failure(cause)
    _record_provider_failure(failure, attempts)
    return failure


class ProviderClosed(PolicySessionError):
    """A closed adapter refuses to start or retry a request.

    ``close()`` is how an operator interrupt reaches an in-flight turn: the
    Supervisor's stop handler closes the PolicySession, which closes its
    provider from another thread. A transport error raised by that close is
    the stop itself, not a transient fault, so no retry may follow it and no
    new request may start after it.
    """

    retryable = False

    def __init__(self) -> None:
        super().__init__("native provider is closed; no request was sent")
