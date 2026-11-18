"""The task checker's verdict, withheld from the rollout Compactor where the policy saw it.

A test mirror of the rule ``ora/agent/context/lossless/compactor_privacy.py``
describes: the Pi core applies it (``ora-pi/src/context/projection.ts``,
``compactorHistoryMessage``). ``withhold_verdicts`` was in that module until
2026-11-04; it stays here for the tests that hold the rule against the Python
privacy scan (``causal_value._compactor_private_violations``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ora.agent.context.lossless.compactor_privacy import FINISH_VERDICT_REFUSAL_CODES

TASK_CHECK_TOOL = "check_task"
TASK_CHECK_VERDICT_FIELD = "task_success"
FINISH_TOOL = "finish_episode"
VERDICT_WITHHELD_REASON = "official_outcome_reserved"
#: The check_task result the Compactor receives instead of the checker's verdict.
TASK_CHECK_WITHHELD = {
    "status": "task_check_result_withheld_from_compactor",
    "reason": VERDICT_WITHHELD_REASON,
}
#: What replaces ``task_success`` in a finish refusal's constraints.
FINISH_VERDICT_WITHHELD = {"verdict_withheld_from_compactor": VERDICT_WITHHELD_REASON}


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def withhold_verdicts(visible: dict[str, Any]) -> dict[str, Any]:
    """Withhold the checker's verdict from one provider-visible message (in place)."""

    name = visible.get("name")
    if visible.get("role") != "tool" or name not in {TASK_CHECK_TOOL, FINISH_TOOL}:
        return visible
    content = visible.get("content")
    if not isinstance(content, str):
        return visible
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        return visible
    if not isinstance(decoded, Mapping):
        return visible
    if name == TASK_CHECK_TOOL:
        if TASK_CHECK_VERDICT_FIELD in decoded:
            visible["content"] = _compact(TASK_CHECK_WITHHELD)
        return visible
    error = decoded.get("error")
    constraints = error.get("constraints") if isinstance(error, Mapping) else None
    if (
        isinstance(constraints, Mapping)
        and isinstance(error.get("code"), str)
        and error["code"] in FINISH_VERDICT_REFUSAL_CODES
        and TASK_CHECK_VERDICT_FIELD in constraints
    ):
        withheld = {key: value for key, value in constraints.items() if key != TASK_CHECK_VERDICT_FIELD}
        withheld.update(FINISH_VERDICT_WITHHELD)
        visible["content"] = _compact({**decoded, "error": {**error, "constraints": withheld}})
    return visible


__all__ = [
    "FINISH_TOOL", "FINISH_VERDICT_WITHHELD", "TASK_CHECK_TOOL",
    "TASK_CHECK_VERDICT_FIELD", "TASK_CHECK_WITHHELD", "withhold_verdicts",
]
