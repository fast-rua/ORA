"""abort is admitted only on an infrastructure failure the Harness recorded this round.

RoboDojo g012: with the policy failure exit withheld, six rounds
ended in finish_episode(abort) with task judgements after 0-48 of 80 decisions;
the runtime accepted any non-empty reason. The abort admission
(policy_abort_infrastructure_round_v2) reads the run's own event log since the
round's reset -- tool errors the runtime wrote and receipts the Supervisor
wrote -- and refuses abort when it holds no infrastructure-class failure. The
behavioural rows below are every distinct tool-error and receipt failure shape
of the seven sealed g012 runs.

Offline: the classification of the Harness record.
"""
from __future__ import annotations

import json

import pytest

from ora.agent.composition.schemas.abort_evidence import (
    EXECUTION_FAULT_CODES, PLANNER_FAULT_CODES, SKILL_FAULT_STOP_REASONS, TOOL_ERROR_CODES,
)
from ora.agent.toolkit.host.finish import abort_evidence
from tests.support.agent.policy_abort_evidence import (
    _tool_error, G012_RECORD, INFRASTRUCTURE_RECEIPTS, INTER_ARM_REFUSALS,
)


RESET = {"type": "reset", "time": 1.0, "record": {"round_index": 1}}


# --------------------------------------------------------------------------
# The classification of the Harness record
# --------------------------------------------------------------------------

def _signals(events):
    return abort_evidence.round_signals(json.dumps(event, sort_keys=True) for event in events)


def test_nothing_g012_recorded_is_an_infrastructure_failure():
    assert _signals(G012_RECORD) == []


@pytest.mark.parametrize("name", sorted(INFRASTRUCTURE_RECEIPTS))
def test_a_receipt_the_harness_marks_as_a_fault_qualifies(name):
    signals = _signals([*G012_RECORD, INFRASTRUCTURE_RECEIPTS[name]])
    assert len(signals) == 1 and signals[0]["source"] == "environment_receipt"


@pytest.mark.parametrize("code", sorted(TOOL_ERROR_CODES))
def test_an_infrastructure_tool_error_qualifies(code):
    assert _signals([_tool_error(code, "vision_ground")]) == [
        {"source": "tool_error", "tool": "vision_ground", "code": code}]


@pytest.mark.parametrize("code", ["invalid_arguments", "stale_observation", "target_ref_unavailable",
                                  "failure_exit_withheld", "abort_without_infrastructure_evidence",
                                  "task_success_not_confirmed",
                                  "grounding_failed", "segmentation_failed", "tool_unavailable"])
def test_a_refused_or_behavioural_tool_error_does_not(code):
    assert _signals([_tool_error(code)]) == []


@pytest.mark.parametrize("name", sorted(INTER_ARM_REFUSALS))
def test_an_inter_arm_refusal_is_not_an_infrastructure_failure(name):
    """E11G12-004a: the other arm in the way is a refused target or route, not a Harness failure."""
    assert _signals([RESET, *G012_RECORD, INTER_ARM_REFUSALS[name]]) == []
    for code in ("inter_arm_collision", "inter_arm_collision_path", "recovery_inter_arm_collision_path"):
        assert code not in TOOL_ERROR_CODES and code not in PLANNER_FAULT_CODES
        assert code not in EXECUTION_FAULT_CODES and code not in SKILL_FAULT_STOP_REASONS


def test_only_the_current_round_counts():
    fault = INFRASTRUCTURE_RECEIPTS["execution_fault"]
    assert _signals([_tool_error("internal_tool_error"), fault, RESET]) == []
    assert len(_signals([RESET, fault, _tool_error("internal_tool_error")])) == 2


def test_an_unreadable_line_is_not_evidence():
    lines = ["not json", json.dumps({"type": "tool_error"}), json.dumps([1, 2])]
    assert abort_evidence.round_signals(lines) == []
