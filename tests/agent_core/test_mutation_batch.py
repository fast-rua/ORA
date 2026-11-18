"""A world mutation must be the only tool call of its reply (``mutation_batch_rejected``), on the Pi core.

When a reply carries a world-mutating call beside any other call, the Pi core
(``admitToolBatch`` and ``batchRejection`` in ``ora-pi/src/core/dispatch.ts``)
runs the other calls, executes none of the mutations, and answers each one
with ``mutation_batch_rejected``, retryable, naming how many other calls ran.
Nothing reaches the environment until a later reply sends the mutation alone;
the episode then goes on as usual and its trace is valid evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes

pytestmark = pytest.mark.usefixtures("pi_core")

OBSERVE = {"views": ["agentview"]}
SCRIPT = [
    episodes.tool_reply(("r0", "observe", OBSERVE), ("m0", "close_gripper", {})),  # read, then mutation
    episodes.tool_reply(("m1", "close_gripper", {}), ("m2", "open_gripper", {})),  # two mutations
    episodes.tool_reply(("m3", "open_gripper", {}), ("r1", "observe", OBSERVE)),  # mutation, then read
    episodes.tool_reply(("m4", "close_gripper", {})),  # alone: executed
    {"message": episodes.tool_reply(("f0", "finish_episode", {
        "outcome": "failure", "reason": "scripted end", "attempts": ["scripted"]}))["message"]},
]


def _tool_answers(request: bytes) -> dict[str, Any]:
    """The tool messages of one canonical request, by tool-call id."""

    answers = {}
    for message in json.loads(request)["messages"]:
        if message.get("role") == "tool":
            content = message["content"]
            answers[message["tool_call_id"]] = json.loads(content) if isinstance(content, str) else content
    return answers


def test_a_mutation_beside_other_calls_is_refused_and_the_other_calls_run(tmp_path: Path) -> None:
    run = episodes.run_core("pi", tmp_path, responses=SCRIPT, actions=1)
    assert run["core"] == "pi"
    assert run["result"]["state"] == "completed" and run["result"]["termination"] == "policy_finished_episode"
    assert run["result"]["decisions"] == 1 and run["result"]["model_calls"] == len(SCRIPT)
    assert run["evaluation"]["status"] == "valid" and run["evaluation"]["reasons"] == []

    answers = _tool_answers(run["requests"][-1])  # the finishing request carries every earlier answer
    ran = {"m0": 1, "m1": 0, "m2": 0, "m3": 1}
    for call_id, other_calls in ran.items():
        error = answers[call_id]["error"]
        assert error["code"] == "mutation_batch_rejected", call_id
        assert error["constraints"] == {"other_calls_executed": other_calls}, call_id
        assert error["retryable"] is True and error["infrastructure"] is False, call_id
    assert "error" not in answers["m4"]

    events = run["events"]
    proposed = {event["tool_call_id"]: event["payload"] for event in events if event["type"] == "tool_call_proposed"}
    assert {call_id: proposed[call_id]["batch_valid"] for call_id in ("r0", *ran, "r1", "m4")} == {
        "r0": True, "m0": False, "m1": False, "m2": False, "m3": False, "r1": True, "m4": True}
    results = {event["tool_call_id"]: event["payload"]["ok"] for event in events if event["type"] == "tool_result"}
    assert results["r0"] is results["r1"] is results["m4"] is True  # the reads ran, whatever their position
    assert not any(results[call_id] for call_id in ran)
    # Only the lone mutation became an action, and only it reached the environment.
    assert [event["tool_call_id"] for event in events if event["type"] == "action_proposed"] == ["m4"]
    assert [event["type"] for event in events].count("environment_receipt") == 1
    supervisor = [json.loads(line) for line in (run["root"] / "events.jsonl").read_text().splitlines()]
    assert [record["type"] for record in supervisor].count("action_submitted") == 1
