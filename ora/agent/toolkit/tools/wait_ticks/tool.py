"""Hold the current commanded poses/jaws while time passes."""
from ...lib.observation import observe
from ...lib.values import PublicResult
from ..errors import ToolRefusal
from ..move_pose import _context
from ..move_pose.tool import propose
from .._receipts import model_receipt


def run(api, arguments):
    """Replay: recorded state/action only; mismatched wait is divergence.

    Live executes 1..20 native ticks through the trusted owner. One RoboDojo
    tick is 10 physical steps, 40 ms. No pose or jaw target is overwritten.
    """
    config = api.artifacts.tool_context()
    refusal = config["motion"]["mutation_refusals"].get("wait_ticks")
    if refusal is not None:
        raise ToolRefusal.from_public(refusal)
    capabilities = config["motion"]["capabilities"]
    if not {"shared_hold_wait_action_v1", "motion_intent_router_v1"} <= set(capabilities):
        raise ToolRefusal("wait_ticks is not selected on this surface", code="tool_unavailable", retryable=False)
    error = None
    if set(arguments) != {"ticks"}:
        error = "motion v3 wait arguments must contain exactly ['ticks']"
    elif type(arguments["ticks"]) is not int or not 1 <= arguments["ticks"] <= 20:
        error = "motion v3 wait ticks must be an integer in 1..20"
    if error is not None:
        raise ToolRefusal(error, field="ticks", received=arguments.get("ticks"),
                          constraints={"minimum": 1, "maximum": 20, "type": "integer"})
    identity = observe(api).identity.json()
    action = {"schema": "ora-motion-action.v3", "action": "wait",
              "action_id": f"action-{config['execution']['generation']}",
              "observation_id": identity["observation_id"],
              "bound_observation_sha256": identity["observation_sha256"],
              "arguments": {"ticks": arguments["ticks"]}}
    if _context.environment_plans(config):
        return propose(api, action, "wait_ticks")
    value = PublicResult.of(api.act.execute_action(action, 20)).data.json()
    return {"status": "executed", "action": "wait_ticks", **value,
            "receipt": model_receipt(value["receipt"])}
