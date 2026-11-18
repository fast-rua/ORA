"""Tool-call argument decoding and structured tool-error results.

Shared by the Pi core's tool RPC (``ora.agent.pi.tool_rpc``, ``robot_socket``) and the session.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ora.agent.error_classes import ToolError


def _tool_error(exc: Exception) -> dict[str, Any]:
    """The failed result of a tool call: the error envelope, and any picture the refusal drew as the result's images."""
    structured = (
        exc.as_dict()
        if isinstance(exc, ToolError)
        else ToolError(str(exc), code="internal_tool_error", retryable=False).as_dict()
    )
    images = list(getattr(exc, "images_base64", None) or ()) if isinstance(exc, ToolError) else []
    return {"ok": False, "error": structured, **({"images_base64": images} if images else {})}



def _arguments(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str) or not function["name"]:
        raise ToolError("tool call lacks a function name", code="invalid_tool_call")
    raw = function.get("arguments", "{}")
    if isinstance(raw, Mapping):
        value = dict(raw)
    elif isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolError(
                "tool arguments are not valid JSON",
                field="arguments",
                received=raw,
            ) from exc
    else:
        raise ToolError(
            "tool arguments must be a JSON object",
            field="arguments",
            received=raw,
        )
    if not isinstance(value, dict):
        raise ToolError(
            "tool arguments must decode to an object",
            field="arguments",
            received=value,
        )
    return str(function["name"]), value
