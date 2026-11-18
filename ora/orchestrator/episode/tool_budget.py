"""Supervisor-owned native-action allowance for one parent tool invocation.

Admission does not reserve a fixed twenty ticks. Each completed action charges
its measured consumption; refusal leaves the balance available to tool code.
"""

from dataclasses import dataclass, field
from typing import Any


def nonnegative(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass
class ToolBudget:
    tool_call_id: str
    limit: int
    used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        nonnegative(self.limit, "native_action_budget")
        if not self.tool_call_id:
            raise ValueError("tool budget requires a parent call identity")

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def admit(self, requested: int, episode_remaining: int, cap: int = 20) -> int:
        """One action may use at most the four independently supplied limits.

        ``cap`` is the action's own per-action limit (``joint_trajectory.native_action_cap``):
        20 for a legacy primitive, more for a joint trajectory or a rate-limited gripper.
        """
        return min(nonnegative(requested, "action budget"), self.remaining,
                   nonnegative(episode_remaining, "episode remaining budget"), nonnegative(cap, "action cap"))

    def charge(self, consumed: int, admitted: int) -> bool:
        """Return whether this is the first positive spend of the parent call."""
        nonnegative(consumed, "native_actions")
        nonnegative(admitted, "admitted budget")
        if consumed > admitted or consumed > self.remaining:
            raise ValueError("worker consumption exceeds the admitted action budget")
        first = self.used == 0 and consumed > 0
        self.used += consumed
        return first
