"""A dual-arm motion failure must name the arm whose plan failed.

RoboTwin plans a multi-arm request all-or-nothing: if either arm's plan
fails, neither executes. Measured on one run, dual-arm ``move_pose`` calls
failed IK at 11.2% against 4.0% for single-arm, and only 3.2% of 3,886
receipts used the dual-arm form at all. The top-level failure code said only
that planning failed, so "never use both arms" was a cheaper lesson than
"retarget the left arm". The per-arm planner status is already recorded and
already projected; these tests pin the top-level fold of it.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from ora.agent.session_core.receipt_projection import (
    PUBLIC_MOTION_EFFECT_CAPABILITY,
    ReceiptProjectionContext,
    resolve_receipt_effect_projector,
)
from tests.support.agent.failed_arm_receipts import _SCHEMA, _receipt


def _project(record: dict[str, Any]) -> dict[str, Any]:
    projector = resolve_receipt_effect_projector(record["schema"])
    assert projector is not None
    return projector(record, ReceiptProjectionContext(
        last_action_id="action-7", last_action_public_name="move_pose",
        surface_supports=lambda capability: (
            capability == PUBLIC_MOTION_EFFECT_CAPABILITY
        ),
    ))


def test_dual_arm_planner_failure_names_the_single_failing_arm() -> None:
    """Only the left plan failed, so only the left arm needs retargeting."""

    unified = _project(_receipt(
        target_arms=["left", "right"],
        statuses={"left": "Fail", "right": "Success"},
    ))

    assert unified["failed_arms"] == ["left"]
    # The folded name sits beside the code it explains, not buried in detail.
    assert unified["failure_code"] == "planner_failed"


def test_both_arms_are_named_when_both_plans_fail() -> None:
    unified = _project(_receipt(
        target_arms=["left", "right"],
        statuses={"left": "Fail", "right": "Fail"},
    ))

    assert unified["failed_arms"] == ["left", "right"]


def test_single_arm_failure_adds_no_field_because_target_arms_already_says_it(
) -> None:
    """A field that merely repeats ``target_arms`` would be noise."""

    unified = _project(_receipt(
        target_arms=["left"], statuses={"left": "Fail"},
    ))

    assert "failed_arms" not in unified
    assert unified["detail"]["target_arms"] == ["left"]


def test_dual_arm_execution_failure_blames_no_arm() -> None:
    """Both plans succeeded; the fold has nothing to say and stays silent."""

    record = _receipt(
        target_arms=["left", "right"],
        statuses={"left": "Success", "right": "Success"},
    )
    record["execution_status"] = "not_reached"
    record["failure_stage"] = "execution"
    record["failure_code"] = "waypoint_tracking_timeout"

    unified = _project(record)

    assert "failed_arms" not in unified
    assert unified["failure_code"] == "controller_timeout"


def test_frozen_legacy_projection_is_unchanged() -> None:
    """Surfaces without the unified capability keep their exact projection."""

    record = _receipt(
        target_arms=["left", "right"],
        statuses={"left": "Fail", "right": "Success"},
    )
    legacy = resolve_receipt_effect_projector(_SCHEMA)(
        copy.deepcopy(record),
        ReceiptProjectionContext(
            last_action_id="action-7", last_action_public_name="move_pose",
        ),
    )

    assert "failed_arms" not in legacy
    # The per-arm evidence the fold reads is still projected on its own.
    assert legacy["arms"]["left"]["planner"]["status"] == "Fail"


@pytest.mark.parametrize("damage", ["arms", "planner", "status"])
def test_a_missing_per_arm_planner_record_invents_nothing(damage: str) -> None:
    record = _receipt(
        target_arms=["left", "right"],
        statuses={"left": "Fail", "right": "Success"},
    )
    if damage == "arms":
        record["arms"]["left"] = {"targeted": True}
    elif damage == "planner":
        record["arms"]["left"]["planner"] = {"called": False, "status": "not_called"}
    else:
        record["arms"]["left"]["planner"]["status"] = None

    unified = _project(record)

    assert "failed_arms" not in unified


@pytest.mark.parametrize(("raw", "expected"), [
    ("gripper_not_reached", "gripper_not_reached"),
    ("episode_terminal_during_execution", "episode_terminal_during_execution"),
    ("native_action_budget_exhausted", "controller_timeout"),
    ("private_unrecognized_exception", "execution_fault"),
])
def test_dojo_convergence_and_terminal_feedback_are_not_generic_execution_faults(raw, expected):
    record = {"schema": "ora.robodojo-sparse-motion-receipt.v1",
        "action_id": "action-7", "action": "set_gripper", "execution_status": "not_reached",
        "failure_stage": "execution", "failure_code": raw, "target_arms": ["right"],
        "terminal_after_action": raw == "episode_terminal_during_execution",
        "arms": {"right": {"targeted": True, "gripper_error": .3}}}
    result = _project(record)
    assert result["failure_code"] == expected
    assert result["status"] == "not_reached"
    assert result["physical_cause"] == "unknown"
    assert result["detail"]["arms"]["right"]["gripper_error"] == .3
    assert result["round_terminated"] == record["terminal_after_action"]
