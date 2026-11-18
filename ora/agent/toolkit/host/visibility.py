"""Robot history contains parent-tool boundaries, not intermediate publications.

The Supervisor execution ledger is the authority. Internal observations remain
available to the executing tool and diagnostic replay, but cannot be selected
by a later model call. Explicit grasp baseline images are a separate released
projection and do not grant historical step access.
"""
import json

from ora.agent.error_classes import ToolError

CAPABILITY = "toolkit_legacy_projection_v1"


def visible_steps(host, states):
    """Return reset and parent start/completion steps whose identities match the state chain."""
    if not host.surface.supports(CAPABILITY):
        return {state["step"] for state in states}
    identities = {state["step"]: state["observation"]["id"] for state in states}
    visible = {state["step"] for state in states if state["cause"] == "reset"}
    try:
        for path in (host.control_root.parent / "native/tool-executions").glob("*.json"):
            value = json.loads(path.read_bytes())
            if value["schema"] != "ora.tool-execution.v1" or type(value["finished"]) is not bool:
                raise ValueError("invalid execution ledger visibility evidence")
            boundaries = [value["initial_observation"]]
            if value["finished"] and value["pending"] is None:
                boundaries.append(value["final_observation"])
            for boundary in boundaries:
                step = boundary["step"]
                if type(step) is not int:
                    raise ValueError("invalid execution boundary step")
                if step in identities:
                    if identities[step] != boundary["observation_id"]:
                        raise ValueError("execution boundary differs from state chain")
                    visible.add(step)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise ToolError("tool-boundary visibility evidence is unavailable",
                        code="artifact_integrity_fault", retryable=False) from error
    return visible


def refuse_step(step):
    return ToolError(
        f"step {step!r} is an internal publication, not a completed tool-call observation",
        code="invalid_arguments", field="step", received=step,
        constraints={"scope": "completed_tool_calls"},
    )


def check_selection(host, state):
    if host.surface.supports(CAPABILITY) and state["step"] not in visible_steps(host, [state]):
        raise refuse_step(state["step"])


def artifact_observations(host, control):
    """Observation allowlist for cached targets and query/preview artifacts."""
    if not host.surface.supports(CAPABILITY):
        return None
    current = control["observation_id"]
    if host.state_root is None:
        return {current}
    from ora.agent.toolkit.host.shared.saved_steps import round_timeline
    states, _ = round_timeline(host, control)
    allowed = visible_steps(host, states)
    return {current, *(state["observation"]["id"] for state in states if state["step"] in allowed)}
