"""Grant the exact public policy-state publication requested by geometry_query."""

from ora.agent.toolkit.host.shared import saved_steps

from ora.agent.error_classes import ToolError
from .failures import control_adapter
from .observe import adapter


def selection(host, arguments):
    """The ``geometry_query`` tool-context section: the publication ``step`` selects, or its refusal.

    Only the host's current observation and policy-state chain are read, so snapshot
    replay builds the same section over a host that sees the run as it was at a boundary.
    """
    try:
        step, state, _, _ = host._select_observation(
            saved_steps.route_current(host, arguments.get("step"))
        )
        identity = {
            "observation_id": state["observation"]["id"],
            "observation_sha256": state["observation"]["manifest"]["sha256"],
        }
    except ToolError as error:
        return {"error": error.as_dict()}
    return {
        "step": step,
        "source_state_sha256": state["state_sha256"],
        "identity": identity,
    }


def configure(host, arguments, api, store):
    selected = selection(host, arguments)
    if "error" not in selected:
        identity = selected["identity"]
        publications = {identity["observation_id"]: (selected["step"], identity)}
        api.adapters["observation"] = control_adapter(adapter(host, store, publications))
    return selected
