"""Free play's side of an execution inside a policy turn (``ora play``, ``episode/play.py``).

A code-mode script (``play.code_mode``) and a synchronous toolkit tool execute their
actions inside the tool's call, through the Supervisor's toolkit executor
(``tool_execution.py``), and the agent core ends the act on them
(``ora-pi/src/play/driver.ts``, ``actions_executed``):

- ``settle_inline_execution``: the instruction goes on with the next act on the
  newest observation; an environment that ended the scene meanwhile ends it at the
  loop's top; an execution that left the world unknown, or was interrupted, ends
  the session as a control fault (the benchmark loop's
  ``tool_execution_boundary.settle_inline_execution``, for a session).
- ``script_stop_request``: while a code-mode script runs, the operator's pending
  stop (Esc, reset or quit), read from the inbox between the script's calls
  (``ScriptActionExecutor.stop_requested``, ``code_mode/shell.py``).
"""

from __future__ import annotations

from typing import Any

from ora.agent import ExecutedActions

from .actions import post_action_boundary
from .termination import TerminationAuthority, TerminationDecision


def settle_inline_execution(loop: Any, outcome: Any, call_increment: int) -> bool:
    """Settle a turn whose tools executed actions inside it; False for any other turn."""

    executor = getattr(loop, "toolkit_executor", None)
    if executor is None:
        return False
    if executor.blocked:
        reason = str((executor.interruption or {}).get("reason") or "tool_execution_interrupted")
        if loop.segment is not None:
            loop._finish_segment("control_fault")
        if reason == "world_state_unknown":
            loop.interaction.abort_unknown_execution(model_calls=call_increment)
        else:
            loop.interaction.model_calls += call_increment
            loop.interaction.abort("tool_execution_interrupted")
        decision = TerminationDecision(reason=reason, authority=TerminationAuthority.INFRASTRUCTURE, scope="session")
        loop.record_termination_decision(decision, source="tool_execution")
        loop.termination, loop.termination_authority = reason, decision.authority
        loop._announce("ended", termination=reason)
        return True
    executed = isinstance(outcome, ExecutedActions)
    if loop.environment_status.get("terminal"):
        loop.interaction.record_policy_outcome(
            kind="no_action", model_calls=call_increment,
            reason="actions_executed" if executed else "tool_execution_boundary", terminal=False)
        boundary = post_action_boundary({}, loop.environment_status)
        if boundary is not None and boundary[0].authority is TerminationAuthority.INFRASTRUCTURE:
            # A control fault, as after the policy's own action (PlayLoop._execute_proposal).
            decision, environment_reason = boundary
            loop.record_termination_decision(decision, source="environment_status")
            loop.interaction.mark_terminal(environment_reason)
            if loop.segment is not None:
                loop._finish_segment("control_fault")
            loop.termination, loop.termination_authority = environment_reason, decision.authority
        return True
    if executed:
        loop.interaction.record_policy_outcome(
            kind="no_action", model_calls=call_increment, reason="actions_executed", terminal=False)
        return True
    return False


def script_stop_request(loop: Any) -> str | None:
    """The operator's pending stop while a code-mode script runs; None while it may go on.

    The terminal's input reaches the inbox only while the Supervisor calls the agent
    core, so this calls it once (``serve_operator_input``) before reading the inbox.
    The stop itself applies at the loop's top, once the turn has ended.
    """

    serve = getattr(loop.planner, "serve_operator_input", None)
    if callable(serve):
        serve()
    loop._take_commands()
    return loop.play_stop
