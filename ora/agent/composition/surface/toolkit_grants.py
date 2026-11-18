"""Explicit, independently named robot tool and privileged-key grants."""

from collections.abc import Collection, Mapping

SYNCHRONOUS = "toolkit_synchronous_execution_v1"
LEGACY_VISIBLE = "toolkit_legacy_projection_v1"


def validate_grants(value, capabilities: Collection[str]) -> dict:
    """Normalize grants against the surface's declared capabilities."""
    if LEGACY_VISIBLE in capabilities and "faithful_check_task" in capabilities and value != {
        "tools": ["check_task"], "state_keys": ["privileged.task_check"]
    }:
        raise ValueError("legacy-visible toolkit requires both check_task and privileged.task_check grants")
    if value is None:
        return {"tools": [], "state_keys": []}
    if SYNCHRONOUS not in capabilities:
        raise ValueError("toolkit_grants requires the synchronous toolkit surface")
    if not isinstance(value, Mapping) or set(value) - {"tools", "state_keys"}:
        raise ValueError("toolkit_grants accepts only tools and state_keys")
    tools, keys = value.get("tools", []), value.get("state_keys", [])
    if tools not in ([], ["check_task"]) or keys not in ([], ["privileged.task_check"]):
        raise ValueError("only check_task and privileged.task_check may be granted")
    if bool(tools) != bool(keys):
        raise ValueError("check_task requires both the tool and privileged.task_check grants")
    return {"tools": list(tools), "state_keys": list(keys)}
