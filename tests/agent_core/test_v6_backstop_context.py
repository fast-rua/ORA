"""Context management v6 with the hard backstop on the Pi core (``backstop_exact_tail_exchanges``, 2026-11-14).

The episodes reproduce the RoboLab tool-call runs (2026-11-14) that ended ``context_exhausted``
under v6: many read-only calls in one open ActionStep, whose tool-returned
images and results no v6 layer ages, and a reply cut off at the output limit.
Without the backstop field a request is estimated over the hard limit and sent
anyway; with it every request is estimated first, and one over the limit has all
but the latest two exchanges aged (reasoning, read-only results, tool-returned
images), the open ActionStep's included, once its own estimate reaches L6's
watermark (0.8 of the hard limit). Every completed run evaluates valid.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes
from tests.support.agent_core.replay.runtime import compactor_responses

pytestmark = pytest.mark.usefixtures("pi_core")

#: 48,000 - 2,048 - 2,048, and L6's watermark, 0.8 of it. Synthetic's observation and every observe call bring two
#: images, which v6 estimates at 4,096 tokens each, so the latest two exchanges and the opening observation fit
#: under the watermark and the history does not.
WINDOW = 48000
HARD = 43904
WATERMARK = 35123


def management(**changes: Any) -> dict[str, Any]:
    return {
        "schema": "ora.native-context-management.v6",
        "causal_exact_tail_action_steps": 3,
        "inline_media_tail_action_steps": 2,
        "eager_large_result_bytes": 50000,
        "eager_preview_bytes": 2048,
        "eager_exact_deduplication": True,
        "trigger_inline_images": 24,
        "trigger_estimated_tokens": 48000,
        "trigger_provider_input_ratio": 0.7,
        "episode_checkpoint_trigger_tokens": 150000,
        "emergency_fallback_ratio": 0.8,
        "emergency_target_ratio": 0.65,
        "checkpoint_target_estimated_tokens": 1200,
        "checkpoint_max_estimated_tokens": 1600,
        "tail_aging_stride_action_steps": 4,
        **changes,
    }


BACKSTOP = {"backstop_exact_tail_exchanges": 2, "truncated_reply_reasoning": "withheld"}


def looking_script(calls: int, *, truncated_at: int | None = None) -> list[dict[str, Any]]:
    """``calls`` read-only ``observe`` calls of both cameras (their images come back as tool-returned images), none
    completing an ActionStep, one reply cut off at the output limit at ``truncated_at``; finish at the end."""

    responses: list[dict[str, Any]] = []
    for index in range(calls):
        if index == truncated_at:
            responses.append({"message": {"role": "assistant", "content": ""}, "finish_reason": "length",
                              "usage": None})
        reply = episodes.tool_reply((f"p{index}", "observe", {"views": ["agentview", "wrist"]}))
        reply["usage"] = None
        responses.append(reply)
    responses.append({"message": episodes.tool_reply(("f0", "finish_episode", {
        "outcome": "failure", "reason": "scripted end", "attempts": ["scripted"]}))["message"]})
    return responses


def _run(tmp_path: Path, script: list[dict[str, Any]], fields: dict[str, Any]) -> dict[str, Any]:
    provider = types.SimpleNamespace(pi_provider_spec={
        "kind": "fake", "responses": script, "compactor_responses": compactor_responses(0)})
    return episodes.run_core("pi", tmp_path / "out", actions=len(script), provider=provider, policy_overrides={
        "context_management": management(**fields), "context_window_tokens": WINDOW, "max_output_tokens": 2048,
        "max_model_calls_per_observation": len(script) + 2})


def _projections(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [event["payload"] for event in run["events"] if event["type"] == "context_projected"]


def _inline_images(message: dict[str, Any]) -> int:
    content = message.get("content")
    return sum(block.get("type") == "image_url" for block in content) if isinstance(content, list) else 0


def test_without_the_backstop_field_v6_sends_requests_estimated_over_the_hard_limit(tmp_path: Path) -> None:
    run = _run(tmp_path, looking_script(10), {})
    projections = _projections(run)
    assert all("request_estimate" not in payload and "hard_backstop" not in payload for payload in projections)
    assert max(payload["estimated_input_tokens"] for payload in projections) > HARD


def test_the_v6_backstop_keeps_every_request_of_an_open_action_step_under_the_limit(tmp_path: Path) -> None:
    run = _run(tmp_path, looking_script(10, truncated_at=4), BACKSTOP)
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    projections = _projections(run)
    assert all(payload["action_step_inventory"][-1]["action_step_index"] == 0 for payload in projections)
    assert all(payload["request_estimate"]["tokens"] <= HARD for payload in projections)
    backstops = [payload["hard_backstop"] for payload in projections if payload["hard_backstop"]]
    assert backstops and all(item["post_estimated_input_tokens"] <= HARD
                             and WATERMARK <= item["pre_estimated_input_tokens"]
                             and item["watermark_tokens"] == WATERMARK for item in backstops)
    # It ages at L6's watermark, before the hard limit, inside the open ActionStep where L6 has nothing to absorb.
    assert any(item["pre_estimated_input_tokens"] <= HARD for item in backstops)
    transitions = [item for payload in projections for item in payload["projection_transitions"]]
    assert {item["kind"] for item in transitions} == {"hard_backstop"}
    assert all(item["semantic_loss"] is False and item["aging"] == "own" for item in transitions)
    # The request of the latest backstop (each projection is one request, in order).
    index = max(position for position, payload in enumerate(projections) if payload["hard_backstop"])
    last = projections[index]
    record = last["hard_backstop_aging"]
    assert record["aged_exchanges"] > 0 and record["delivered_tool_images_externalized"] > 0
    assert all(payload["active_inline_images"] == sum(_inline_images(message) for message in json.loads(data)["messages"])
               for payload, data in zip(projections, run["requests"]))
    request = json.loads(run["requests"][index])
    # Only the latest two exchanges keep their tool-returned images; every aged one is a reference observe reopens.
    images = [message for message in request["messages"] if message["role"] == "user"
              and isinstance(message["content"], list) and any(
                  block.get("type") == "text" and "image(s) returned by" in block.get("text", "")
                  for block in message["content"])]
    assert [_inline_images(message) > 0 for message in images][-3:] == [False, True, True]
    assert run["evaluation"]["context_artifact_provenance"]["status"] == "valid"
    # The truncated reply was answered by the core's continuation and stays in the history.
    assert any(event["type"] == "provider_output_truncated" for event in run["events"])
