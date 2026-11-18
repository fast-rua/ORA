"""The closed grasp skill: one committed sequence the policy is not consulted in.

A grasp is one committed run with no stop: the candidate choice is the
decision, and there is nothing to decide halfway.
"""

from __future__ import annotations

import pytest

from ora.agent.error_classes import ToolError
from ora.agent.toolkit.host.skills.ledger import (
    GRASP_STAGES, SCHEMA, abandon, begin, pending_stage, record, stages_for,
    terminal_status, validate_state,
)


def _grasp(**overrides):
    kwargs = dict(
        skill="grasp_execute", candidate_id="G1",
        rationale="G1 is the only candidate clear of the neighbouring mug",
        observation_id="obs-1", observation_sha256="a" * 64,
    )
    kwargs.update(overrides)
    return begin(**kwargs)


def _run(state, stages, status="executed"):
    for index, stage in enumerate(stages):
        state, _transition = record(
            state, stage=stage, status=status, reason="ok", action_id=f"action-{index}",
        )
    return state


def test_a_grasp_runs_approach_close_and_verify_as_one_committed_sequence():
    state = _grasp()

    assert state["schema"] == SCHEMA
    assert state["sequence"] == list(GRASP_STAGES)
    assert pending_stage(state) == "approach"
    assert state["stop_before_commit"] is False

    state = _run(state, GRASP_STAGES)
    assert terminal_status(state) == "completed"


def test_an_unconfirmed_mutating_stage_fails_the_skill_closed():
    """Descending from an unconfirmed approach would verify something else."""

    state = _grasp()
    state, transition = record(
        state, stage="approach", status="not_reached", reason="planner_rejected_target",
    )

    assert transition == "approach_unconfirmed_skill_aborted"
    assert terminal_status(state) == "aborted"
    assert pending_stage(state) is None


def test_a_read_only_stage_that_did_not_execute_does_not_abort_the_skill():
    """Verification consumes evidence; failing to measure is not failing to act."""

    state = _run(_grasp(), ["approach", "close"])
    state, transition = record(
        state, stage="verify", status="unavailable",
        reason="visible_surface_evidence_is_insufficient",
    )

    assert transition == "sequence_completed"
    assert terminal_status(state) == "completed"


def test_stages_cannot_be_recorded_out_of_order_or_after_the_end():
    state = _grasp()
    with pytest.raises(ToolError) as wrong:
        record(state, stage="verify", status="executed", reason="ok")
    assert "approach is due" in str(wrong.value)

    finished = _run(state, GRASP_STAGES)
    with pytest.raises(ToolError):
        record(finished, stage="verify", status="executed", reason="ok")


def test_a_skill_can_be_abandoned_before_its_remaining_stages_run():
    state = _run(_grasp(), ["approach"])
    state, transition = abandon(state, "target_moved_under_the_gripper")

    assert transition == "abandoned:target_moved_under_the_gripper"
    assert terminal_status(state) == "aborted"
    with pytest.raises(ToolError):
        abandon(state, "again")


def test_the_rationale_is_committed_with_the_sequence():
    state = _grasp()
    assert state["rationale"].startswith("G1 is the only candidate")

    with pytest.raises(ToolError) as refused:
        _grasp(rationale="   ")
    assert refused.value.field == "rationale"


def test_a_candidate_id_is_required_because_the_plan_chose_it():
    with pytest.raises(ToolError) as refused:
        _grasp(candidate_id="")
    assert refused.value.field == "candidate_id"


@pytest.mark.parametrize("skill", ["grasp_and_place", "place_execute"])
def test_an_unknown_skill_is_refused(skill):
    with pytest.raises(ToolError):
        stages_for(skill)


def test_a_tampered_ledger_is_refused_rather_than_trusted():
    state = _run(_grasp(), ["approach"])
    state["completed"][0]["stage"] = "verify"
    with pytest.raises(ToolError) as refused:
        validate_state(state)
    assert refused.value.code == "skill_ledger_state"


def test_a_running_skill_has_no_terminal_status():
    with pytest.raises(ToolError):
        terminal_status(_grasp())


@pytest.mark.parametrize("stage", ["approach", "close", "verify"])
def test_terminated_is_final_and_cannot_be_abandoned(stage):
    state = _grasp()
    for preceding in GRASP_STAGES[:GRASP_STAGES.index(stage)]:
        state, _ = record(state, stage=preceding, status="executed", reason="executed")
    state, transition = record(state, stage=stage, status="terminated", reason="environment_terminal")
    assert transition == "environment_terminal"
    assert terminal_status(state) == "terminated"
    assert pending_stage(state) is None
    with pytest.raises(ToolError, match="already ended"):
        abandon(state, "retry")
