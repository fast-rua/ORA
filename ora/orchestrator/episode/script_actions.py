"""The Supervisor's side of a free-play script's world-changing call (``ora-robot``).

In a free-play run (plan ``episode_mode: "play"``) the agent's scripts reach
the robot through the policy session's robot socket
(``ora/agent/pi/robot_socket.py``). Every script call runs on this, the
Supervisor's, thread while a policy turn is active -- the policy session waits
on its agent core's ``act`` -- so world changes stay one at a time and a
script's action is one decision of the current turn:

- ``stop_requested``, while a code-mode script runs (``code_mode/shell.py``
  asks between the program's messages): the loop's own
  ``script_stop_request()`` (free play: the operator's Esc, reset or quit);
  the command is then killed and reports ``stopped``.
- ``admit``, before the tool runs: the turn is active and not a terminal
  lifecycle turn, the run is neither ending nor interrupted, the environment
  has not reached a boundary and no round budget is exhausted
  (``EpisodeLoop._exhausted_budget``). A loop may refuse more through its own
  ``script_action_refusal()`` (free play: a pending operator control). The
  reasons a policy sees name nothing it may not know: an environment boundary
  is ``turn_must_end``, never its cause.
- ``execute``, after the tool persisted its proposal: the path of the policy's
  own proposal -- ``_accept_submitted_action`` (bound to the current
  observation), ``dispatch_action``, ``_record_receipt`` (decision and budget
  accounting, the receipt delivered to the policy session) and
  ``publish_observation`` -- with no model calls of its own (the turn's calls
  are counted when it ends). The turn then continues on the fresh observation
  (``interaction.begin_policy``).

A boundary the action reached is not settled inside the turn: the loop meets
``environment_status["terminal"]`` once the turn ends, as after any turn.
Each executed script action adds a ``script_action`` event (its action id and
observation steps) beside the ``action_submitted`` and ``motion_receipt``
records it shares with the policy's own actions.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping

from .actions import dispatch_action
from .observation import publish_observation
from .records import _append

if TYPE_CHECKING:
    from .loop import EpisodeLoop


class ScriptActionExecutor:
    """Admit and execute a script's persisted proposal inside the current policy turn."""

    def __init__(self, loop: EpisodeLoop) -> None:
        self.loop = loop

    def admit(self) -> str | None:
        loop = self.loop
        interaction = loop.interaction
        if interaction.phase != "policy_active":
            return "no_active_turn"
        if getattr(interaction, "_policy_return_phase", None) == "terminal":
            return "turn_must_end"
        if loop.termination is not None:
            return "run_ending"
        if loop.interrupted:
            return "operator_interrupt"
        if loop.environment_status is not None and loop.environment_status.get("terminal"):
            return "turn_must_end"
        exhausted = loop._exhausted_budget()
        if exhausted is not None:
            return exhausted
        refusal = getattr(loop, "script_action_refusal", None)
        return refusal() if callable(refusal) else None

    def stop_requested(self) -> str | None:
        """Why a running script must stop now, between its calls (free play: the operator's Esc, reset or quit,
        ``PlayLoop.script_stop_request``); None while it may go on."""

        request = getattr(self.loop, "script_stop_request", None)
        return request() if callable(request) else None

    def execute(self) -> Mapping[str, Any]:
        loop = self.loop
        step = loop.interaction.observation_step
        action, current = loop._accept_submitted_action(0)
        started_ns = time.monotonic_ns()
        receipt, captured_after_action = dispatch_action(loop, action, current, should_stop=loop._should_stop)
        duration_s = round(max(0, time.monotonic_ns() - started_ns) / 1e9, 6)
        loop._record_receipt(action, receipt, duration_s)
        loop.pointer, loop.environment_status = publish_observation(
            loop, "after_action", captured_observation=captured_after_action,
        )
        _append(loop.events, {
            "type": "script_action", "action_id": action.get("action_id"),
            "observation_step": step, "fresh_observation_step": loop.interaction.observation_step,
            "time": time.time(),
        })
        loop.interaction.begin_policy()
        receipt_id = receipt.get("receipt_id") or receipt.get("id")
        return {
            "receipt_id": str(receipt_id) if receipt_id is not None else None,
            "duration_s": duration_s,
            "observation_step": loop.interaction.observation_step,
            "decisions": loop.round_decisions,
        }


def bind_script_actions(loop: EpisodeLoop) -> None:
    """Bind the executor to a policy session that serves a robot socket (free play only)."""

    code_mode = bool((loop.plan.get("policy") or {}).get("code_mode"))
    host = loop.tools
    while "base" in vars(host):
        host = host.base
    if code_mode or loop.tools.surface.supports("toolkit_synchronous_execution_v1"):
        from .tool_execution import ToolActionExecutor
        if code_mode:  # scripts execute every action through one parent invocation (code_mode_execution.py)
            from .code_mode_execution import CodeModeActionExecutor as ToolActionExecutor
        executor = ToolActionExecutor(loop)
        loop.toolkit_executor = executor
        # Toolkit Dispatch reads the executor on the runtime build_tool_runtime made, under every wrapper
        # the coordinator adds (lossless retention's Artifact reader, then code mode's tools); a wrapper
        # forwards reads to its base, never writes.
        host.toolkit_executor = executor
        loop.planner.bind_tool_action_executor(
            executor, observation_receipts=loop.plan.get("toolkit_observation_receipts", False))
        return
    if loop.plan.get("episode_mode") != "play":
        return
    bind = getattr(loop.planner, "bind_script_action_executor", None)
    if callable(bind):
        bind(ScriptActionExecutor(loop))
