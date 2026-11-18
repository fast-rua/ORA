"""The Supervisor's executor for code mode (plan ``policy.code_mode``) on every backend.

Code mode runs a script as one parent invocation of ``ToolActionExecutor``
(``tool_execution.py``), so every action the script causes is admitted, executed,
published and recorded in that parent's ``ora.tool-execution.v1`` ledger, the
evidence the agent core ends the turn on. The toolkit executor counts a backend's
native actions (RoboDojo's official limit); a backend without that accounting
(LIBERO, RoboTwin) counts one per executed action, and robot decisions and the
other budgets still bound the run through ``admit``. A backend that counts
native actions but has no Supervisor-owned execution state (RoboCasa365) cannot
cap one action in the worker, so one action may use what the episode has left:
its measured spend is charged, never refused after the fact.

On RoboDojo the worker caps each action (``limited_action``), and a script's
parent lends each action it may run the most one action may use
(``joint_trajectory.native_action_cap``): a released primitive's 20, or a joint
trajectory's 250 where the surface's motion tools run one per move (toolkit
v4/v5). Each tool still asks for no more than its own budget, and there a motion
tool the script calls fits its plan to what the parent has left as well
(``nested_allowance``, which toolkit Dispatch delivers only on those surfaces), so
the parent's admission never stops a move mid-path.
Each action names the script's own call of the tool (``<call>#NNN``) to the
worker, which keeps one undo entry per arm-moving call (``recover_pose``).

A tool that commits a skill sequence (RoboTwin's closed-skill ``grasp_execute``)
persists one ``ora-skill-sequence-action.v1``; on a backend whose worker does not
cap one action's native budget, the script's parent runs its stage loop
(``dispatch_action``) as one action, exactly as the coordinator runs it after a
tool-call turn. It is one of the script's actions, and the script as a whole is
one robot decision (``record_receipt`` charges the parent's first spend). On
RoboDojo, whose worker caps each action at one primitive's budget, it is refused
before it starts; resolution refuses that surface first, by the same predicate
the coordinator binds the cap by (``execution_state.per_action_capped``).

A script's parent runs many calls, so when it is interrupted the record
(``native/tool-execution-interruption.json``) names the exception that stopped
it, its type, message and stage, rather than only the ledger completion that
fails after it (``execution ledger completion is not durable``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

from ora.contracts import SKILL_SEQUENCE_ACTION_SCHEMA
from ora.contracts.joint_trajectory import MAX_TRAJECTORY_NATIVE_ACTIONS

from .tool_execution import ToolActionExecutor

#: Remaining budget reported for a backend without a native-action limit: robot decisions bound it instead.
UNCOUNTED_REMAINING = 1_000_000
SKILL_SEQUENCE_RECEIPT = "ora.manipulation-skill-sequence-receipt.v1"


class _NoExecutionState:
    """A backend without Supervisor-owned execution state (RoboDojo's ``execution_state.py``): no recovery
    anchors to check and no per-action native budget to lend the worker."""

    def recovery_refusal(self, action: Any) -> None:
        return None

    @contextmanager
    def limited_action(self, budget: int, tool_call: Any = None):
        yield

    def finish_tool_call(self, receipt: Any) -> None:
        return None


class CodeModeActionExecutor(ToolActionExecutor):
    """``ToolActionExecutor`` for any backend: native counts where the backend reports them, else one per action."""

    def __init__(self, loop: Any) -> None:
        super().__init__(loop)
        self._no_state = _NoExecutionState()

    def _state_owner(self) -> Any:
        owner = getattr(self.loop, "execution_state_owner", None)
        return owner if owner is not None else self._no_state

    def _counted(self) -> bool:
        return type(super()._native_remaining()) is int

    def _worker_limited(self) -> bool:
        # The coordinator binds execution state for exactly the per-action capped backends (``per_action_capped``).
        return getattr(self.loop, "execution_state_owner", None) is not None

    def native_actions_per_action(self) -> int:
        if not self._counted():
            return 1
        if self._worker_limited():
            # The most one of its actions may use (``joint_trajectory.native_action_cap``): a released
            # primitive's 20, or a joint trajectory's where the surface's motion tools run them (toolkit v4/v5).
            return MAX_TRAJECTORY_NATIVE_ACTIONS if self._runs_trajectories() else super().native_actions_per_action()
        return max(1, self._native_remaining())

    def _runs_trajectories(self) -> bool:
        surface = getattr(getattr(self.loop, "tools", None), "surface", None)
        if surface is None:
            return False
        from ora.agent.toolkit.host.motion import joint_trajectory_surface
        return joint_trajectory_surface(surface)

    def _action_cap(self, action: Mapping[str, Any], remaining: int) -> int:
        return super()._action_cap(action, remaining) if self._worker_limited() else remaining

    def _native_remaining(self) -> int | None:
        remaining = super()._native_remaining()
        return remaining if type(remaining) is int else UNCOUNTED_REMAINING

    def _accepts(self, action: Mapping[str, Any]) -> bool:
        return super()._accepts(action) or (
            action["schema"] == SKILL_SEQUENCE_ACTION_SCHEMA and not self._worker_limited())

    def _native_used(self, raw: Any) -> int | None:
        used = super()._native_used(raw)
        if self._counted():
            return used
        # Uncounted: one per executed action. A skill sequence that dispatched no stage sums to 0 but still spent
        # its decision, as the tool-call submission does.
        sequence = isinstance(raw, Mapping) and raw.get("schema") == SKILL_SEQUENCE_RECEIPT
        return 1 if used is None or sequence else used


__all__ = ["CodeModeActionExecutor", "UNCOUNTED_REMAINING"]
