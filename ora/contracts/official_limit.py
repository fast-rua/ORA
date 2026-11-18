"""How far an episode may run past its benchmark's official limit, and how that is recorded.

A benchmark's official limit (RoboDojo ``step_lim``, the RoboCasa horizon, the
RoboLab episode length) ends its published protocol. A RunPlan may let the
episode run on to ``runtime.official_limit_multiple`` times that limit, so a
slow success is seen instead of cut off. The two results never share a field:

* ``official_success`` keeps its meaning: a success inside the official limit,
  comparable with every run recorded before this field existed.
* ``success_after_official_limit`` is a success only the extension admitted.
  No report adds it to an official success rate.

The official predicate is the same one on both sides of the limit; only the
field it lands in differs. The official result at the limit is kept as
``official_limit_snapshot``, a run that then continues does not change it.

A plan without the field runs at multiple 1, which is every sealed run and
every qualification: the official limit is also where the episode stops, and
the verdict carries none of these fields.

RoboDojo and RoboLab run at ``TOOL_BUDGET_LIMIT_MULTIPLE``: no episode reaches that limit,
so the tool budgets end a run, and the policy is told no native-action
allowance (``orchestrator/episode/native_allowance.py``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

#: What a new RunSpec runs at. Eighty decisions spend about 1,060 native actions,
#: so three times the official limit frees the tasks whose limit binds first
#: (RoboDojo ``align_blocks``: 200) and leaves ``max_robot_decisions`` the budget
#: that binds on the rest.
DEFAULT_OFFICIAL_LIMIT_MULTIPLE = 3
#: What a RoboDojo RunSpec runs at. The shortest task (200 native actions) gets 200,000, past six hours of wall time
#: at the 5 native actions a second a move runs at, so ``max_robot_decisions``, ``max_model_calls`` and ``wall_time_s``
#: end a run and no native-action count does.
TOOL_BUDGET_LIMIT_MULTIPLE = 1000
MAXIMUM_OFFICIAL_LIMIT_MULTIPLE = TOOL_BUDGET_LIMIT_MULTIPLE
#: The verdict fields a reader copies onto its own record.
OFFICIAL_LIMIT_FIELDS = (
    "official_native_action_limit", "official_limit_multiple", "extended_native_action_limit",
    "official_limit_snapshot", "success_after_official_limit",
    "success_after_official_limit_native_actions",
)


def official_limit_multiple(config: Mapping[str, Any]) -> int:
    """``official_limit_multiple`` of a runtime or worker config; absent is 1."""

    value = config.get("official_limit_multiple", 1)
    if type(value) is not int or not 1 <= value <= MAXIMUM_OFFICIAL_LIMIT_MULTIPLE:
        raise ValueError(
            f"official_limit_multiple must be an integer in [1, {MAXIMUM_OFFICIAL_LIMIT_MULTIPLE}]"
        )
    return value


def limit_snapshot(native_actions: int) -> dict[str, Any]:
    """The official result of an episode that reached its limit without a success."""

    return {"official_success": False, "native_actions": int(native_actions)}


def verdict_fields(
    *, official_limit: int, multiple: int, snapshot: Mapping[str, Any] | None,
    success_native_actions: int | None,
) -> dict[str, Any]:
    """What a verdict adds under an extended plan; nothing at multiple 1."""

    if multiple == 1:
        return {}
    return {
        "official_native_action_limit": int(official_limit),
        "official_limit_multiple": int(multiple),
        "extended_native_action_limit": int(official_limit) * int(multiple),
        "official_limit_snapshot": deepcopy(dict(snapshot)) if snapshot is not None else None,
        "success_after_official_limit": success_native_actions is not None,
        "success_after_official_limit_native_actions": success_native_actions,
    }


def record_fields(verdict: Mapping[str, Any] | None) -> dict[str, Any]:
    """The verdict's extended-limit fields for a round or run record; none when it has none."""

    source = verdict if isinstance(verdict, Mapping) else {}
    return {field: deepcopy(source[field]) for field in OFFICIAL_LIMIT_FIELDS if field in source}


__all__ = [
    "DEFAULT_OFFICIAL_LIMIT_MULTIPLE", "MAXIMUM_OFFICIAL_LIMIT_MULTIPLE", "OFFICIAL_LIMIT_FIELDS",
    "TOOL_BUDGET_LIMIT_MULTIPLE", "limit_snapshot", "official_limit_multiple", "record_fields", "verdict_fields",
]
