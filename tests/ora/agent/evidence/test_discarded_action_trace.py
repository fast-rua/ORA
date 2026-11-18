"""The agent-trace verifier accepts a proposal a budget discarded, and nothing looser.

A proposal is closed by its receipt and a fresh observation, or by the
Supervisor's ``action_discarded`` when a round budget ends the round first. The
discard must name the pending proposal and a round budget, and the run's
recorded termination must be that budget; an unclosed proposal or a discard
whose reason the run did not end on stays invalid.

The traces are frozen (``tests/data/agent_core/discard-trace``): one session per
case, recorded with the Python Native core before ORA removed it, so these are
also the check that the verifier still reads such sealed traces exactly as it
did. The Pi core's own discard, written live, is checked by
``tests/agent_core/test_model_call_budget.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.agent_events import read_agent_events
from ora.agent.evidence.trace_verify import verify_agent_trace

FIXTURES = Path(__file__).resolve().parents[4] / "tests/data/agent_core/discard-trace"


def _trace(tmp_path: Path, case: str) -> Path:
    """A run root holding the frozen trace of ``case`` and the artifacts it references."""

    root = tmp_path / case
    (root / "native").mkdir(parents=True)
    shutil.copytree(FIXTURES / "native/agent-artifacts", root / "native/agent-artifacts")
    shutil.copyfile(FIXTURES / f"{case}.jsonl", root / "agent-events.jsonl")
    return root


def _types(root: Path) -> list[str]:
    return [event["type"] for event in read_agent_events(root)]


def test_a_discard_closing_the_trailing_proposal_on_the_runs_termination_is_valid(tmp_path: Path) -> None:
    root = _trace(tmp_path, "discarded")
    events = read_agent_events(root)
    proposal = next(event for event in events if event["type"] == "action_proposed")
    discard = next(event for event in events if event["type"] == "action_discarded")
    terminated = next(event for event in events if event["type"] == "run_terminated")
    assert discard["causal_parent_id"] == proposal["event_id"]
    assert discard["payload"] == {"action_id": proposal["payload"]["action_id"], "reason": "model_call_budget"}
    assert discard["visibility"] == "derived"
    assert terminated["payload"]["termination"] == "model_call_budget"
    verification = verify_agent_trace(root)
    assert verification["status"] == "valid", verification["errors"]


def test_an_unclosed_proposal_stays_invalid(tmp_path: Path) -> None:
    root = _trace(tmp_path, "unclosed")
    assert "action_discarded" not in _types(root)
    assert verify_agent_trace(root)["errors"] == ["action_without_fresh_observation"]


def test_a_discard_the_run_did_not_end_on_is_invalid(tmp_path: Path) -> None:
    root = _trace(tmp_path, "termination-mismatch")
    index = _types(root).index("run_terminated")
    assert verify_agent_trace(root)["errors"] == [f"event:{index}:discarded_action_termination_mismatch"]


@pytest.mark.parametrize(("case", "reason"), [("reason-policy-finished", "policy_finished_episode"),
                                              ("reason-none", None)])
def test_a_discard_naming_no_round_budget_is_invalid(tmp_path: Path, case: str, reason: Any) -> None:
    root = _trace(tmp_path, case)
    discard = next(event for event in read_agent_events(root) if event["type"] == "action_discarded")
    assert discard["payload"]["reason"] == reason
    index = _types(root).index("action_discarded")
    assert verify_agent_trace(root)["errors"] == [
        f"event:{index}:action_discarded_contract", "action_without_fresh_observation",
    ]


def test_a_discard_must_name_the_pending_action() -> None:
    from ora.agent.pi.policy_session import PiPlanner

    planner = PiPlanner.__new__(PiPlanner)  # the check precedes any call into the agent core
    planner.last_action_event_id, planner.last_action_id = "agent-event-000006", "action-pending"
    with pytest.raises(PolicySessionError, match="not the pending submitted action"):
        planner.record_action_discarded({"action_id": "action-other", "reason": "model_call_budget"})


def test_policy_activity_after_a_discard_is_invalid(tmp_path: Path) -> None:
    root = _trace(tmp_path, "activity-after-discard")
    types = _types(root)
    discarded = types.index("action_discarded")
    after = [index for index, kind in enumerate(types)
             if index > discarded and kind in {"observation_received", "provider_request"}]
    assert after and verify_agent_trace(root)["errors"] == [
        f"event:{index}:activity_after_discarded_action" for index in after
    ]
