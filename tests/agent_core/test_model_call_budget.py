"""The run's model-call budget holds inside a turn, and a proposal the budget overtakes is discarded.

The Synthetic plan allows 5 model calls and 3 per observation. The scripted
model answers five times with text and no tool call (each answer earns a
``provider_no_action_continuation``), then calls ``close_gripper``. The first
turn ends at its per-observation cap (``model_call_limit``) and the Supervisor
asks again on the same observation. That second turn must be capped by what is
left of the run's budget (2), not by the budget published with the observation
(5): the episode ends at exactly 5 calls on ``model_call_budget``, no action is
proposed, and the run evaluates valid and seals.

Before, the agent cores read the remaining budget from the observation's context,
so the second turn ran a third call, proposed the move, overran the budget and
the post-turn guard ended the round with the proposal never executed: the
trace ended on ``action_proposed`` and the run was invalid
(``action_without_fresh_observation``). That guard is still the backstop for a
core that overruns; the second test reproduces such a core (the turn's budget
withheld) and checks that the Supervisor records ``action_discarded`` with the
budget's reason, so the run evaluates valid and seals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes

pytestmark = pytest.mark.usefixtures("pi_core")

MAX_MODEL_CALLS = 5
PER_OBSERVATION = 3


def _text_reply(index: int) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": f"Still studying the scene ({index})."},
            "usage": {"input_tokens": 4000, "output_tokens": 20}}


def _script() -> list[dict[str, Any]]:
    responses = [_text_reply(index) for index in range(MAX_MODEL_CALLS)]
    responses.append(episodes.tool_reply(("m0", "close_gripper", {})))
    return responses


def _budgets(plan: dict[str, Any]) -> dict[str, str]:
    plan["budgets"]["max_model_calls"] = MAX_MODEL_CALLS
    plan["policy"]["max_model_calls_per_observation"] = PER_OBSERVATION
    return {}


def _run(core: str, out: Path) -> dict[str, Any]:
    from ora.agent.providers.fake import FakeProvider
    from ora.orchestrator.evaluation.evaluate import seal

    run = episodes.run_core(core, out, provider=FakeProvider(responses=_script()), configure=_budgets)
    run["seal"] = seal(run["root"])
    return run


def test_a_turn_stops_at_the_runs_remaining_model_calls(tmp_path: Path) -> None:
    run = _run("pi", tmp_path)
    assert run["core"] == "pi"
    result = run["result"]
    assert result["termination"] == "model_call_budget"
    assert result["termination_authority"] == "experiment_budget"
    assert result["model_calls"] == MAX_MODEL_CALLS
    assert len(run["requests"]) == MAX_MODEL_CALLS
    types = [event["type"] for event in run["events"]]
    assert "action_proposed" not in types and "action_discarded" not in types
    assert run["evaluation"]["status"] == "valid", run["evaluation"]["reasons"]
    assert run["seal"]


def test_a_proposal_past_the_budget_is_discarded_and_the_run_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.agent.contracts import PolicyTurnInput

    original = PolicyTurnInput.runtime_kwargs

    def without_turn_budget(self: PolicyTurnInput) -> dict[str, Any]:
        # A core that caps a turn by the observation's stale budget only.
        kwargs = original(self)
        kwargs.pop("turn_budget", None)
        return kwargs

    monkeypatch.setattr(PolicyTurnInput, "runtime_kwargs", without_turn_budget)
    run = _run("pi", tmp_path)
    assert run["core"] == "pi"
    result = run["result"]
    assert result["termination"] == "model_call_budget"
    assert result["model_calls"] == MAX_MODEL_CALLS + 1
    assert result["decisions"] == 0
    events = run["events"]
    types = [event["type"] for event in events]
    proposal = next(event for event in events if event["type"] == "action_proposed")
    discard = next(event for event in events if event["type"] == "action_discarded")
    assert types.index("action_discarded") == types.index("action_proposed") + 1
    assert "environment_receipt" not in types
    assert discard["causal_parent_id"] == proposal["event_id"]
    assert discard["observation_step"] == proposal["observation_step"]
    assert discard["payload"] == {"action_id": proposal["payload"]["action_id"], "reason": "model_call_budget"}
    terminated = next(event for event in events if event["type"] == "run_terminated")
    assert terminated["payload"]["termination"] == "model_call_budget"
    assert run["evaluation"]["status"] == "valid", run["evaluation"]["reasons"]
    assert run["seal"]
    from viewer.process_timeline import compile_process_timeline

    timeline = compile_process_timeline(run["root"])
    titles = [event.get("title") for event in timeline["events"]]
    assert "Action not executed" in titles and "Unmatched action proposal" not in titles
