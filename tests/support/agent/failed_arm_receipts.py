"""Test helpers for failed arm receipts, shared by several test modules.

They were defined in
``tests/ora/agent/session_core/test_receipt_projection_failed_arms.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

from typing import Any


_SCHEMA = "ora.robotwin-sparse-motion-receipt.v7"


def _receipt(
    *, target_arms: list[str], statuses: dict[str, str],
) -> dict[str, Any]:
    """One planner-stage failure receipt with real per-arm planner records."""

    return {
        "schema": _SCHEMA,
        "action": {"action": "move_pose"},
        "action_id": "action-7",
        "execution_status": "planner_failed",
        "target_arms": list(target_arms),
        "synchronization": {
            "mode": "synchronized_group" if len(target_arms) > 1 else "single_arm",
        },
        "arms": {
            arm: {
                "targeted": arm in target_arms,
                "planner": (
                    {
                        "called": True,
                        "status": statuses[arm],
                        "stage": "IK_FAIL" if statuses[arm] != "Success" else "SUCCESS",
                        "path_steps": 0 if statuses[arm] != "Success" else 42,
                    }
                    if arm in target_arms
                    else {"called": False, "status": "not_called", "path_steps": 0}
                ),
            }
            for arm in ("left", "right")
        },
        "controller_tracking": {"completed": False, "arms": {}},
        "motion_strategy": "global",
        "failure_stage": "planner",
        "failure_code": "MotionGenStatus.IK_FAIL",
        "terminal_after_action": False,
    }
