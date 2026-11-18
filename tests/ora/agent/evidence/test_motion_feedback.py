"""A receipt's public contact detail passes the privileged-field audit (``ora/agent/evidence/motion_feedback.py``)."""

from __future__ import annotations

import pytest

from ora.agent.evidence.trace_privacy import _forbidden_path

PUBLIC = {"contact_estimated": True, "external_force_magnitude_n": 3.2}


def test_a_ledger_receipt_s_public_contact_detail_is_not_privileged() -> None:
    """A LIBERO or RoboTwin receipt carries ``detail.contact_motion``; a synchronous execution's ledger holds it per
    action (code mode executes every backend's actions through one)."""

    ledger = {"execution": {"actions": [{"receipt": {"detail": {"contact_motion": PUBLIC}}}]},
              "action_step_inventory": [{"actions": [{"receipt": {"detail": {"contact_motion": PUBLIC}}}]}]}
    assert _forbidden_path(ledger) is None
    leaked = {"execution": {"actions": [{"receipt": {"detail": {"contact_motion": {"contact_points": [1]}}}}]}}
    assert _forbidden_path(leaked) is not None, "only the public contact fields pass"


def test_the_last_action_effect_keeps_its_public_contact_detail() -> None:
    assert _forbidden_path({"last_action_effect": {"detail": {"contact_motion": PUBLIC}}}) is None
    assert _forbidden_path({"last_action_effect": {"detail": {"contact_motion": {"contact_points": [1]}}}}) is not None


# --- a RoboDojo stop: the receipt's own progress names it (``stall_evidenced_code``) ---------


def _project(receipt):
    from ora.agent.session_core.receipt_projection import (
        ReceiptProjectionContext, resolve_receipt_effect_projector,
    )

    projector = resolve_receipt_effect_projector(receipt["schema"])
    return projector(receipt, ReceiptProjectionContext(
        last_action_id=receipt["action_id"], last_action_public_name="move_pose",
        surface_supports=lambda capability: True))


def _budget_receipt(outcomes):
    """A RoboDojo receipt that ran out of native actions, as the SDK executor recorded them."""
    return {"schema": "ora.robodojo-sparse-motion-receipt.v1", "action_id": "action-3",
            "action": "move_pose", "execution_status": "not_reached", "failure_stage": "execution",
            "failure_code": "native_action_budget_exhausted", "target_arms": list(outcomes),
            "terminal_after_action": False,
            "arms": {arm: {"targeted": True, **({"motion_progress": {"outcome": outcome}} if outcome else {})}
                     for arm, outcome in outcomes.items()}}


@pytest.mark.parametrize("outcomes,expected", [
    ({"left": "stalled"}, "controller_stalled"),
    ({"left": "stalled", "right": "arrived"}, "controller_stalled"),
    ({"left": "stalled", "right": "still_moving_at_window_end"}, "controller_timeout"),
    ({"left": "still_moving_at_window_end"}, "controller_timeout"),
    ({"left": None}, "controller_timeout"),  # released receipts carry no progress block
])
def test_the_projection_reads_the_receipt_s_own_progress_before_calling_a_stop_a_timeout(outcomes, expected):
    assert _project(_budget_receipt(outcomes))["failure_code"] == expected


def test_a_stalled_recovery_is_never_projected_as_a_timeout():
    from ora.agent.session_core.receipt_projection import public_motion_failure_code

    record = _budget_receipt({"right": None})
    record["arms"]["right"]["recovery_progress"] = {"outcome": "stalled"}
    assert _project(record)["failure_code"] == "controller_stalled"
    assert public_motion_failure_code("execution", "native_action_budget_exhausted") == "controller_timeout"


def test_a_grasp_execute_stage_carries_the_stop_and_the_public_code():
    from ora.orchestrator.episode.motion_stage import dojo_motion_stage_evidence

    progress = {"outcome": "stalled", "still_native_actions": 3, "hold": "measured_joints_at_stall"}
    receipt = {**_budget_receipt({"left": "stalled"}), "failure_code": "controller_stalled"}
    receipt["arms"]["left"]["motion_progress"] = progress
    residual = {"reason": receipt["failure_code"], "errors_by_arm": {"left": {"translation_m": 0.08}}}
    public, extra = dojo_motion_stage_evidence(receipt, residual)
    assert public["reason"] == "controller_stalled" and public["errors_by_arm"] == residual["errors_by_arm"]
    assert extra == [{"quantity": "motion_progress", "value": {"left": progress}}]
    budget = {**receipt, "failure_code": "native_action_budget_exhausted",
              "arms": {"left": {**receipt["arms"]["left"],
                                "motion_progress": {"outcome": "still_moving_at_window_end"}}}}
    public, _ = dojo_motion_stage_evidence(budget, {"reason": "native_action_budget_exhausted"})
    assert public["reason"] == "controller_timeout"  # the private name never reaches the policy
    other = {"schema": "ora.robotwin-sparse-motion-receipt.v7", "failure_code": "x"}
    assert dojo_motion_stage_evidence(other, {"reason": "x"}) == ({"reason": "x"}, [])
