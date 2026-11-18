"""Settle an inline execution without dispatching its actions a second time."""
from ora.agent import ExecutedActions
from .termination import TerminationAuthority
from .actions import post_action_boundary
from .terminal import stop_at_environment_boundary


def settle_inline_execution(loop, outcome, call_increment):
    executor = getattr(loop, "toolkit_executor", None)
    if getattr(executor, "blocked", False):
        if executor.interruption["reason"] == "world_state_unknown":
            loop.interaction.abort_unknown_execution(model_calls=call_increment)
        else:
            loop.interaction.model_calls += call_increment
            loop.interaction.abort("tool_execution_interrupted")
        return True
    if executor is not None and loop.environment_status.get("terminal"):
        loop.interaction.record_policy_outcome(
            kind="no_action", model_calls=call_increment,
            reason="tool_execution_boundary", terminal=False)
        decision, reason = post_action_boundary({}, loop.environment_status)
        stop_at_environment_boundary(
            loop, decision, reason,
            terminal_turn_allowed=decision.authority != TerminationAuthority.INFRASTRUCTURE)
        return True
    if isinstance(outcome, ExecutedActions):
        loop.interaction.record_policy_outcome(
            kind="no_action", model_calls=call_increment,
            reason="actions_executed", terminal=False)
        return True
    return False
