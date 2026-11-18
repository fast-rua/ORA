"""The native-action allowance a policy is spending, reported beside the others.

A backend with an official native-action limit ends the episode when that
limit is spent (or ``runtime.official_limit_multiple`` times it, under an
extended RunPlan), and a Harness physics-step cap below it ends the episode
sooner. Until this module the policy saw neither: every turn reported robot
decisions, model calls and wall time, while RoboDojo's binding limit (550 to
1600 native actions per task, 1000 under an earlier physics cap for a 1600-action
task) and the actions already used appeared nowhere, only the per-command
reservation and each receipt's own count.

The limit is read from the RunPlan: ``runtime.native_action_limit`` (the
official, task-registered limit) times ``runtime.official_limit_multiple``, and
``budgets.max_sim_steps`` divided by the backend's physics steps per native
action (the Harness cap). The effective limit is the smaller, so the policy
counts down what it really has and never sees where the official limit fell. Usage is summed from the executed receipts' own
``native_actions`` counts -- the same worker counters the RoboDojo evaluator
reconciles with the official verdict -- descending through staged composite
and committed skill-sequence wrappers. A receipt that carries no count makes
the remainder unknown (``None``) rather than guessed. A backend without an
official native-action limit reports nothing, so its remaining-budget shape is
unchanged. A plan at ``TOOL_BUDGET_LIMIT_MULTIPLE`` reports nothing either,
unless a Harness cap binds below it: no episode reaches that limit, so it is no
allowance the policy can spend. The count goes on (``remaining``), since a tool's
admission reads it.

A backend whose step cap is itself the evaluated limit (``step_cap_is_native_limit``:
LIBERO, whose ``budgets.max_sim_steps`` counts control ticks and carries the suite's
protocol limit) reports that cap as its limit, with no official one beside it.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.contracts.official_limit import TOOL_BUDGET_LIMIT_MULTIPLE, official_limit_multiple

from ..backends.registry import find

#: Wrappers whose children carry the worker counts.
_CHILD_RECORD_KEYS = ("phase_receipts", "stage_receipts")
_WRAPPER_SCHEMAS = {
    "ora.composite-motion-receipt.v1",
    "ora.manipulation-skill-sequence-receipt.v1",
}


def effective_native_action_limit(plan: Mapping[str, Any]) -> int | None:
    """min(extended official limit, Harness physics cap), or None without a limit."""

    runtime = plan.get("runtime") if isinstance(plan.get("runtime"), Mapping) else {}
    official = runtime.get("native_action_limit")
    if type(official) is not int or official <= 0:
        return _step_cap_limit(plan, runtime)
    extended = official * official_limit_multiple(runtime)
    # Physics steps one native action advances, for a backend that declares an
    # official native-action limit; a backend without it has no Harness cap to fold.
    selected = find(runtime.get("kind"))
    per_action = selected.capabilities.physics_steps_per_native_action if selected is not None else None
    budgets = plan.get("budgets") if isinstance(plan.get("budgets"), Mapping) else {}
    cap = budgets.get("max_sim_steps")
    if per_action and type(cap) is int and cap > 0:
        return min(extended, cap // per_action)
    return extended


def _step_cap_limit(plan: Mapping[str, Any], runtime: Mapping[str, Any]) -> int | None:
    """The plan's step cap as the native-action limit, for a backend that declares it so; else None."""

    selected = find(runtime.get("kind"))
    if selected is None or not selected.capabilities.step_cap_is_native_limit:
        return None
    budgets = plan.get("budgets") if isinstance(plan.get("budgets"), Mapping) else {}
    cap = budgets.get("max_sim_steps")
    return cap if type(cap) is int and cap > 0 else None


def receipt_native_actions(receipt: Any) -> int | None:
    """Native actions one executed receipt consumed, or None when uncounted."""

    if not isinstance(receipt, Mapping):
        return None
    nested = receipt.get("receipt")
    record = nested if isinstance(nested, Mapping) else receipt
    count = record.get("native_actions")
    if type(count) is int and count >= 0:
        return count
    if record.get("schema") not in _WRAPPER_SCHEMAS:
        return None
    total = 0
    for key in _CHILD_RECORD_KEYS:
        for item in record.get(key) or ():
            child = item.get("receipt") if isinstance(item, Mapping) else None
            if child is None:
                continue  # skipped or tool-driven stage: nothing dispatched
            counted = receipt_native_actions(child)
            if counted is None:
                return None
            total += counted
    return total


class NativeActionAllowance:
    """Per-round native-action usage against the effective limit."""

    def __init__(self, plan: Mapping[str, Any]) -> None:
        self.limit = effective_native_action_limit(plan)
        self.used: int | None = 0
        official = plan["runtime"].get("native_action_limit")
        # Without an official limit the limit is the backend's own evaluated step cap
        # (``_step_cap_limit``): always the episode's real allowance, so always reported.
        self.reported = (self.limit is not None
                         and (type(official) is not int
                              or self.limit < official * TOOL_BUDGET_LIMIT_MULTIPLE))

    def reset(self) -> None:
        self.used = 0

    def record(self, receipt: Any) -> None:
        if self.limit is None or self.used is None:
            return
        counted = receipt_native_actions(receipt)
        self.used = None if counted is None else self.used + counted

    def remaining(self) -> int | None:
        """Native actions left, or None without a limit or a count."""

        return None if self.limit is None or self.used is None else max(0, self.limit - self.used)

    def fields(self) -> dict[str, Any]:
        """Remaining-budget entries; empty for a backend without a limit or a plan the tool budgets end."""

        if not self.reported:
            return {}
        return {"native_actions_limit": self.limit, "native_actions_remaining": self.remaining()}


__all__ = [
    "NativeActionAllowance",
    "effective_native_action_limit", "receipt_native_actions",
]
