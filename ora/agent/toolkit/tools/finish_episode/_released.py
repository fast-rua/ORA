"""``finish_episode`` on a surface that declares no outcome contract: success, failure or abort.

A failure needs a reason and its exhausted attempts, and a success check_task's confirmation where the surface has one. An
abort needs nothing: only the v2 contract gates it on infrastructure evidence.
"""

from ...lib.values import PublicResult
from ..errors import ToolRefusal
from ._verdict import refuse_unconfirmed


def run(api, args, config, checked):
    """Replay: compare a recorded exit request; never end an online episode."""
    if config["action_pending"]:
        raise ToolRefusal("an action proposal is pending; wait for the fresh observation before finishing",
                          code="action_pending_fresh_observation", retryable=False)
    if set(args) - set(config["released_arguments"]) or args.get("outcome") not in config["outcomes"]:
        raise ToolRefusal("finish_episode requires outcome success, failure or abort")
    outcome = str(args["outcome"])
    if outcome == "failure":
        attempts = args.get("attempts")
        if not str(args.get("reason", "")).strip():
            raise ToolRefusal("failure requires a reason")
        if (not isinstance(attempts, list) or not attempts
                or any(not isinstance(item, str) or not item.strip() for item in attempts)):
            raise ToolRefusal("failure requires non-empty exhausted attempts")
    if checked is not None:
        refuse_unconfirmed(checked, outcome)
    budget = config["remaining_budget"]
    remaining = budget.get("robot_decisions")
    record = dict(schema="ora.openeta-policy-finish.v1", **args, runtime_remaining_budget=budget,
                  early_exit=outcome != "success" and type(remaining) is int and remaining > 0,
                  early_exit_argument_required=False)
    return PublicResult.of(api.episode.finish_episode(request=record)).data.json()
