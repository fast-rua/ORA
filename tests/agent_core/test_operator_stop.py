"""``ora stop`` during a Pi-core episode: the stopped run is valid evidence and seals.

One Synthetic episode whose scripted model never answers request 4 runs on the
Pi core; SIGTERM reaches the Supervisor once that request is in the trace. The
aborted request is answered by one ``provider_error`` marked ``interrupted:
true`` after ``session_interrupted``, the run ends on the operator's authority,
the evaluation has no reason against it (status ``incomplete``, a stopped
run's) and the run seals. Its requests, the error payload and the event types
equal the golden ``operator-stop`` (``tests/support/agent_core/golden.py``),
recorded from the Python Native core before ORA removed it. Before, the request
stayed unanswered (``request_without_response``) and the run was invalid.

A stop outside a request ends the run the same way, with no further model
call: one that lands while the Supervisor runs a read-only tool call answers
that call; one that lands while the environment executes an action lets the
action finish with its receipt and fresh observation; one that lands while a
world-mutating call is dispatched, after the reply that carried it completed,
discards the proposal unexecuted (the Native core's "completed reply" stop).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes, golden, stops

pytestmark = pytest.mark.usefixtures("pi_core")

HANG_AT = 3  # the fourth request hangs until the stop


def stopped_run(core: str, out: Path) -> dict[str, Any]:
    from ora.orchestrator.evaluation.evaluate import seal

    provider = stops.hanging_fake(episodes.action_script(4), HANG_AT)
    with stops.stop_when_requested(out / core, HANG_AT + 1) as sent:
        run = episodes.run_core(core, out, provider=provider, actions=4)
    assert sent.is_set(), "the episode ended before the stop"
    run["seal"] = seal(run["root"])
    return run


def stopped_summary(core: str, out: Path) -> dict[str, Any]:
    """The stopped run on ``core``: the golden ``operator-stop``."""

    run = stopped_run(core, out)
    assert run["core"] == core
    assert run["result"]["termination"] == "operator_interrupt"
    assert run["result"]["termination_authority"] == "operator"
    assert run["evaluation"]["status"] == "incomplete" and run["evaluation"]["reasons"] == []
    assert run["seal"]
    types = [event["type"] for event in run["events"]]
    (error,) = [event for event in run["events"] if event["type"] == "provider_error"]
    assert error["payload"]["interrupted"] is True
    assert types.index("session_interrupted") < types.index("provider_error")
    assert len(run["requests"]) == HANG_AT + 1
    # The requests carry the remaining wall time with its reading normalized; the events, that reading.
    return golden.summarize(run["requests"], error=error["payload"], event_types=types)


def test_a_stop_during_a_request_answers_it_and_the_run_seals(tmp_path: Path) -> None:
    assert golden.check("operator-stop", stopped_summary("pi", tmp_path)) == []


def _stopped_cleanly(run: dict[str, Any]) -> list[str]:
    """The operator ended ``run``, no model call followed the stop, and it is valid, stopped evidence."""

    from ora.orchestrator.evaluation.evaluate import seal

    assert run["core"] == "pi"
    assert run["result"]["termination"] == "operator_interrupt"
    assert run["result"]["termination_authority"] == "operator"
    types = [event["type"] for event in run["events"]]
    assert "provider_request" not in types[types.index("session_interrupted"):]
    assert run["evaluation"]["status"] == "incomplete" and run["evaluation"]["reasons"] == []
    assert seal(run["root"])
    return types


def test_a_stop_during_a_read_only_tool_call_answers_the_call(tmp_path: Path, monkeypatch) -> None:
    seen = stops.stop_during_tool_call(monkeypatch, mutation=False, nth=2)
    run = episodes.run_core("pi", tmp_path, actions=4)
    assert seen["stopped_at"] == "r1"
    types = _stopped_cleanly(run)
    (answer,) = [event for event in run["events"] if event["type"] == "tool_result" and event["tool_call_id"] == "r1"]
    assert answer["payload"]["ok"] is True and answer["seq"] > types.index("session_interrupted")
    assert run["result"]["decisions"] == 1 and types.count("environment_receipt") == 1


def test_a_stop_during_execution_lets_the_action_finish_with_its_receipt(tmp_path: Path, monkeypatch) -> None:
    seen = stops.stop_during_execution(monkeypatch, nth=2)
    run = episodes.run_core("pi", tmp_path, actions=4)
    types = _stopped_cleanly(run)
    assert run["result"]["decisions"] == 2 and types.count("environment_receipt") == 2
    receipt = [event for event in run["events"] if event["type"] == "environment_receipt"][-1]
    assert receipt["payload"]["action_id"] == seen["stopped_at"]
    after_stop = types[types.index("session_interrupted"):]
    assert after_stop.index("environment_receipt") < after_stop.index("fresh_observation_published")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "act_unless_stopped turns the returned proposal into an operator NoAction and logs "
    "policy_turn_cancelled, but the episode loop never records it as action_discarded in the agent "
    "trace (as _discard_proposal does for a budget stop), so evaluation finds "
    "agent_trace:action_without_fresh_observation and the run cannot be sealed"))
def test_a_stop_during_a_mutation_dispatch_discards_the_proposal(tmp_path: Path, monkeypatch) -> None:
    seen = stops.stop_during_tool_call(monkeypatch, mutation=True, nth=2)
    run = episodes.run_core("pi", tmp_path, actions=4)
    assert seen["stopped_at"] == "m1"
    types = [event["type"] for event in run["events"]]
    assert run["result"]["decisions"] == 1 and types.count("environment_receipt") == 1
    cancelled = [json.loads(line) for line in (run["root"] / "events.jsonl").read_text().splitlines()
                 if '"policy_turn_cancelled"' in line]
    assert [record["discarded_action"] for record in cancelled] == [True]
    _stopped_cleanly(run)
