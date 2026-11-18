"""Freeze the retained public observation selected for a point query."""
from ora.agent.toolkit.host.shared import saved_steps
from ora.agent.error_classes import ToolError
from .observe import adapter
from .failures import control_adapter


def selection(host, arguments):
    """The ``vision_segment`` tool-context section and the step it routes to.

    Reads only the host's current observation and policy-state chain, so snapshot
    replay builds the same section over a host that sees the run as it was at a boundary.
    """
    try:
        requested = saved_steps.route_current(host, arguments.get("step"))
        if requested is not None:
            control, _, _, _, _ = host._retention_observation(arguments["step"])
        else:
            control, _, _ = host._current()
        identity = {key: control[key] for key in ("observation_id", "observation_sha256")}
        return {"identity": identity}, requested
    except ToolError as error:
        return {"error": error.as_dict()}, None


def configure(host, arguments, api, store):
    selected, requested = selection(host, arguments)
    if "error" not in selected:
        identity = selected["identity"]
        api.adapters["observation"] = control_adapter(adapter(
            host, store, {identity["observation_id"]: (requested, identity)}))
    return selected
