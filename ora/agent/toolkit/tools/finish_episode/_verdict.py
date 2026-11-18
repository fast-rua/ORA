"""Finish policy over a verdict obtained exclusively through api.check_task."""

from typing import Any, Mapping
from ..errors import ToolRefusal as ToolError

CAPABILITY = "faithful_check_task"
#: ``context_pack.OFFICIAL_VERDICT_FINISH``: success is the official boundary's alone.
OFFICIAL_VERDICT = "finish_success_official_verdict_v1"
RECORD = "policy-task-check.json"
LATCHED_CODE = "task_success_latched"
UNCONFIRMED_CODE = "task_success_not_confirmed"


def refuse_latched(checked: Mapping[str, Any], arguments: Any) -> None:
    """Refuse any outcome but success once check_task has confirmed success."""

    outcome = arguments.get("outcome") if isinstance(arguments, Mapping) else None
    if outcome == "success" or checked.get("task_success") is not True:
        return
    raise ToolError(
        "task success is already confirmed by check_task in this round; finish_episode admits only "
        "outcome success now",
        code=LATCHED_CODE,
        retryable=True,
        retry_requires_changed_arguments=True,
        field="outcome",
        received=outcome,
        constraints={"allowed": ["success"], "task_success": True},
    )


def refuse_unconfirmed(checked: Mapping[str, Any], outcome: str) -> None:
    """Refuse success that check_task has not confirmed at the current boundary."""

    if outcome != "success":
        return
    if checked.get("task_success") is True:
        return
    terminal = checked.get("terminal") is True
    message = (
        "success requires check_task=true, and this round's environment has ended without official "
        "success: check_task cannot become true in this round"
        if terminal
        else "success requires check_task=true at the current state boundary, and check_task is not true"
    )
    raise ToolError(
        message,
        code=UNCONFIRMED_CODE,
        retryable=not terminal,
        field="outcome",
        received=outcome,
        constraints={
            "requires": "check_task=true",
            "task_success": checked.get("task_success"),
            "terminal": terminal,
        },
    )
