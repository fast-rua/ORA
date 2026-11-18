"""Explicit test-only grants for inspecting every registered contract.

Production resolution still refuses an ungranted legacy-visible surface.
Authority tests call the production resolver directly without this helper.
"""

import json
from dataclasses import replace


def catalog_grants(profile):
    from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG

    manifest = json.loads(PROFILE_CATALOG[profile].read_text(encoding="utf-8"))
    if {"toolkit_legacy_projection_v1", "faithful_check_task"} <= set(manifest["capabilities"]):
        return {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    return None


#: What makes a surface toolkit-only: synchronous, projected, and refused by the runtime handlers.
TOOLKIT_EXECUTION = ("toolkit_only_v1", "toolkit_synchronous_execution_v1", "toolkit_legacy_projection_v1")


def _for_handlers(surface):
    """``surface`` as the runtime handlers serve it: deferred submission, no receipt budget."""
    capabilities = tuple(name for name in surface.capabilities if name not in TOOLKIT_EXECUTION)
    results = {
        name: {**schema, "additionalProperties": True,
               "required": [key for key in schema["required"] if key != "remaining_budget"]}
        if "remaining_budget" in schema.get("required", ()) else schema
        for name, schema in surface.result_schemas.items()
    }
    return replace(surface, capabilities=capabilities, result_schemas=results)


def handlers_only(host):
    """Serve ``host``'s surface from the runtime handlers.

    Outside tests a toolkit-only surface refuses them. These tests keep driving the handlers on RoboDojo's surface: the
    handlers stay live for the other backends, and the equivalence tests compare both stacks on one surface.
    """
    host.surface = _for_handlers(host.surface)
    return host


def surface_of(profile, **options):
    """Resolve ``profile`` for the runtime handlers, with the grants a plan gives it."""
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface

    return _for_handlers(resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile), **options))
