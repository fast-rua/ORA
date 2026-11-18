"""The rollout Compactor never receives the task checker's verdict.

The policy may call ``check_task`` and may be refused ``finish_episode(outcome=success)``
with ``task_success_not_confirmed`` or ``task_success_latched``; that refusal repeats
the checker's Boolean in ``error.constraints.task_success``. The Compactor packet
withholds the verdict in exactly those known places (the policy's own context
keeps the result as delivered). Before, the refusal's field made the privacy scan
reject every later compaction, so an episode that once tried to finish early ran
out of context. The same field anywhere else, or under another refusal code, is
still a privacy violation that rejects the packet. That an emergency compaction is
admitted after a refused finish is held by the recorded Context fixture
``tests/data/context_differential/captured_masking_00_*``, which the TypeScript
pipeline replays; that pipeline also builds the Compactor's history message and
packet (``compactorHistoryMessage``, ``packetMessageBlocks``), so the packet
here is the minimal one the privacy scan reads: each message's descriptor, then
its content.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from ora.agent.context.causal_value import _compactor_private_violations
from ora.agent.context.lossless.compactor_privacy import FINISH_VERDICT_REFUSAL_CODES
from tests.support.agent.compactor_verdicts import withhold_verdicts
from ora.agent.toolkit.host.finish import task_check as finish_task_check
from tests.support.agent.provider_messages import provider_message


def _refusal(code: str, constraints: dict[str, Any]) -> str:
    """A finish_episode refusal as the policy received it (its delivered spelling)."""

    return json.dumps({"error": {
        "code": code, "constraints": constraints, "field": "outcome", "infrastructure": False,
        "message": "success requires check_task=true at the current state boundary, and check_task is not true",
        "received": "success", "retryable": True,
    }, "ok": False}, sort_keys=True)


UNCONFIRMED = _refusal(finish_task_check.UNCONFIRMED_CODE,
                       {"requires": "check_task=true", "task_success": False, "terminal": False})
LATCHED = _refusal(finish_task_check.LATCHED_CODE, {"allowed": ["success"], "task_success": True})


def _tool(name: str, content: str, call_id: str = "finish-1") -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content,
            "_source_event_ids": ["agent-event-000007"]}


def _history(message: dict[str, Any]) -> dict[str, Any]:
    """The message as the Compactor's history shows it: the provider view, verdicts withheld."""

    return withhold_verdicts(provider_message(message))


def _packet(*messages: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for message in messages:
        visible = _history(message)
        content = visible.pop("content")
        blocks += [{"type": "text", "text": json.dumps(visible, sort_keys=True)},
                   {"type": "text", "text": content}]
    return [{"role": "user", "content": blocks}]


def test_the_masked_codes_are_the_finish_task_check_refusals() -> None:
    assert FINISH_VERDICT_REFUSAL_CODES == {finish_task_check.LATCHED_CODE, finish_task_check.UNCONFIRMED_CODE}


@pytest.mark.parametrize("content", [UNCONFIRMED, LATCHED], ids=["unconfirmed", "latched"])
def test_a_refused_finish_reaches_the_compactor_without_the_verdict(content: str) -> None:
    message = _tool("finish_episode", content)
    original = copy.deepcopy(message)
    unmasked = [{"role": "user", "content": [{"type": "text", "text": content}]}]
    assert _compactor_private_violations(unmasked) == [
        "packet[0].content[0].text<json>.error.constraints.task_success",
    ]
    visible = _history(message)
    assert message == original, "the policy's history is not changed"
    delivered, withheld = json.loads(content), json.loads(visible["content"])
    constraints = dict(delivered["error"]["constraints"])
    del constraints["task_success"]
    assert withheld == {**delivered, "error": {**delivered["error"], "constraints": {
        **constraints, "verdict_withheld_from_compactor": "official_outcome_reserved"}}}
    assert {key: value for key, value in visible.items() if key != "content"} == {
        "role": "tool", "tool_call_id": "finish-1", "name": "finish_episode"}
    assert _compactor_private_violations(_packet(message)) == []


def test_the_task_check_result_stays_withheld() -> None:
    visible = _history(_tool("check_task", '{"success": true, "task_success": false}'))
    assert json.loads(visible["content"]) == {
        "status": "task_check_result_withheld_from_compactor", "reason": "official_outcome_reserved"}


@pytest.mark.parametrize(("name", "content"), [
    # Another refusal code carries no verdict by contract: the field there is unexpected.
    ("finish_episode", _refusal("finish_reason_insufficient", {"task_success": False})),
    # The verdict outside the refusal's constraints.
    ("finish_episode", json.dumps({"error": {"code": finish_task_check.UNCONFIRMED_CODE,
                                             "constraints": {}, "task_success": False}, "ok": False})),
    # Any other tool's result.
    ("observe", json.dumps({"status": "actual", "task_success": True})),
    ("move_pose", UNCONFIRMED),
], ids=["other-code", "outside-constraints", "other-tool", "refusal-under-another-tool"])
def test_the_verdict_elsewhere_still_rejects_the_packet(name: str, content: str) -> None:
    assert _compactor_private_violations(_packet(_tool(name, content)))
