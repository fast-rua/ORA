"""Executed outcomes carry evidence, never an action proposal to execute twice."""
import pytest
from ora.agent import ExecutedActions, policy_outcome_from_mapping, PolicySessionError


def test_executed_outcome_keeps_error_and_positive_spend():
    result = policy_outcome_from_mapping({"actions_executed": True, "attempts": 2,
        "execution": {"native_actions_used": 3}, "error": "failed after moving"})
    assert isinstance(result, ExecutedActions)
    assert result.model_calls == 2
    assert result.as_record()["error"] == "failed after moving"
    with pytest.raises(TypeError):
        result.execution["native_actions_used"] = 0


@pytest.mark.parametrize("execution", [None, {}, {"native_actions_used": 0}])
def test_no_spend_cannot_claim_an_executed_outcome(execution):
    with pytest.raises(PolicySessionError):
        policy_outcome_from_mapping({"actions_executed": True, "execution": execution})


def test_synchronous_supervisor_has_no_legacy_proposal_fallback():
    from types import SimpleNamespace
    from ora.orchestrator.episode.loop import EpisodeLoop
    from ora.orchestrator.faults import SupervisorError
    def forbidden(*args):
        raise AssertionError("legacy executor must not be entered")
    loop = SimpleNamespace(toolkit_executor=object(), _accept_submitted_action=forbidden)
    with pytest.raises(SupervisorError, match="legacy action proposal is disabled"):
        EpisodeLoop._execute_proposal(loop, 1)
