"""Context management v9 on the Pi core: exchange aging and the hard backstop.

The episodes reproduce what ended long code-mode runs on 262k-window models:
many model calls (perception, rewritten scripts) and no completed ActionStep,
so the rollout compaction (L6) has nothing to absorb.

* Provider input at the aging watermark ages, in one rewrite, every exchange
  older than the exact exchange tail once a batch of exchanges has left it:
  long tool-call arguments become a reference readable with
  ``artifact_read``, tool-returned images a reference ``observe(ref=...)``
  reopens. The history then appends again, and while the request stays above
  the watermark the next batch ages the same way. Under
  ``reasoning_replay: unaged_exchanges`` that rewrite also withholds the
  reasoning of exactly the exchanges it ages.
* Under ``camera_images: exchange_window_v1`` (the default scheme's) an
  observation's camera images stay inline where it arrived until its exchange
  ages, and then become references ``observe(ref=...)`` reopens.
* A request estimated over the hard limit is aged down to the latest two
  exchanges (the hard backstop) instead of being sent; one that still cannot
  fit ends the turn with ``context_exhausted``.

Every completed run evaluates valid.
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


def management(**changes: Any) -> dict[str, Any]:
    return {
        "schema": "ora.native-context-management.v9",
        "causal_exact_tail_action_steps": 3,
        "camera_images": "rolling_tail_v1",
        "camera_tail_observations": 2,
        "episode_checkpoint_trigger_tokens": 150000,
        "compaction_trigger_tokens": 150000,
        "emergency_fallback_ratio": 0.8,
        "emergency_target_ratio": 0.65,
        "checkpoint_target_estimated_tokens": 1200,
        "checkpoint_max_estimated_tokens": 1600,
        "image_estimate_tokens": 530,
        "exchange_aging_trigger_tokens": 20000,
        "exchange_aging_trigger_ratio": 0.5,
        "exact_tail_exchanges": 6,
        "aged_argument_bytes": 1024,
        "aged_result_bytes": 2048,
        "aged_preview_bytes": 256,
        "backstop_exact_tail_exchanges": 2,
        "reasoning_replay": "all",
        "reasoning_tail_exchanges": None,
        **changes,
    }


def perception_script(calls: int, usage: list[int] | None, *, message_bytes: int = 3000,
                      look: bool = False) -> list[dict[str, Any]]:
    """``calls`` read-only calls with a long argument, none completing an ActionStep; finish at the end.

    ``usage`` is each reply's Provider-reported input; ``None`` reports none. ``look``: every call is an
    ``observe`` of both cameras instead, whose images come back as tool-returned images.
    """

    responses = []
    for index in range(calls):
        call = (f"p{index}", "observe", {"views": ["agentview", "wrist"]}) if look else (
            f"p{index}", "report_issue", {"message": f"attempt {index}: " + "x" * message_bytes})
        reply = episodes.tool_reply(call)
        reply["usage"] = None if usage is None else {"input_tokens": usage[index], "output_tokens": 40}
        responses.append(reply)
    responses.append({"message": episodes.tool_reply(("f0", "finish_episode", {
        "outcome": "failure", "reason": "scripted end", "attempts": ["scripted"]}))["message"]})
    return responses


def _run(tmp_path: Path, script: list[dict[str, Any]], **policy: Any) -> dict[str, Any]:
    provider = types.SimpleNamespace(pi_provider_spec={
        "kind": "fake", "responses": script, "compactor_responses": compactor_responses(0)})
    return episodes.run_core("pi", tmp_path / "out", actions=len(script), provider=provider, policy_overrides={
        "context_management": management(**policy.pop("changes", {})), "context_tools": ["artifact_read"],
        "max_model_calls_per_observation": len(script) + 2, **policy})


def _projections(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [event["payload"] for event in run["events"] if event["type"] == "context_projected"]


def _aged_arguments(request: dict[str, Any]) -> list[bool]:
    return [json.loads(call["function"]["arguments"]).get("arguments_externalized") is True
            for message in request["messages"] if message["role"] == "assistant"
            for call in message.get("tool_calls") or []]


def test_many_calls_without_an_action_age_at_the_watermark_and_stay_under_the_limit(tmp_path: Path) -> None:
    usage = [8000] * 11 + [24000] + [12000] * 10  # the twelfth request crosses the 20,000 watermark
    run = _run(tmp_path, perception_script(16, usage))
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    projections = _projections(run)
    assert all(payload["action_step_inventory"][-1]["action_step_index"] == 0 for payload in projections)
    aging = [item for payload in projections for item in payload["projection_transitions"]
             if item["kind"] == "exchange_aging"]
    assert len(aging) == 1 and aging[0]["semantic_loss"] is False
    assert aging[0]["admission_provider_input_tokens"] == 24000
    assert aging[0]["reasons"] == ["provider_input_watermark", "new_exchanges_since_last_aging"]
    requests = [json.loads(data) for data in run["requests"]]
    plan = json.loads((run["root"] / "plan.json").read_text())
    assert plan["policy"]["context_tools"] == ["artifact_read"]
    # The reader, spelled as the surface spells its tools (artifact.read on the Synthetic surface).
    assert {"artifact_read", "artifact.read"} & {tool["function"]["name"] for tool in requests[-1]["tools"]}
    # Twelve exchanges when it crossed, six kept exact: the first six are references.
    last = _aged_arguments(requests[-1])
    assert last[:6] == [True] * 6 and not any(last[6:])
    relations = [payload["request_relation"] for payload in projections]
    assert relations.count("projection_epoch") == 1 and relations[-1] in {"append_only", "camera_tail_append"}
    assert all(payload["request_estimate"]["tokens"] <= payload["hard_input_tokens"] for payload in projections)
    provenance = run["evaluation"]["context_artifact_provenance"]
    assert provenance["status"] == "valid" and provenance["issued_refs"] > 0


@pytest.mark.parametrize("replay", ["all", "unaged_exchanges"])
def test_a_request_that_stays_above_the_watermark_ages_again_in_batches(tmp_path: Path, replay: str) -> None:
    # What aging cannot touch keeps every report above the watermark; each batch of six still ages.
    run = _run(tmp_path, perception_script(20, [30000] * 20), changes={"reasoning_replay": replay})
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    projections = _projections(run)
    aging = [(item["aged_exchanges_before"], item["aged_exchanges_after"])
             for payload in projections for item in payload["projection_transitions"] if item["kind"] == "exchange_aging"]
    assert aging == [(0, 6), (6, 12)]
    reasons = [payload["exchange_aging"]["admission"]["reason"] for payload in projections]
    assert reasons.count("new_exchanges_since_last_aging") == 2 and "awaiting_exchange_batch" in reasons
    assert [payload["request_relation"] for payload in projections].count("projection_epoch") == 2
    # unaged_exchanges moves the reasoning frontier with the aged one, in the same rewrites; all never moves it.
    records = [payload["exchange_aging"] for payload in projections]
    assert [item["reasoning_withheld_exchanges"] for item in records] == [
        item["aged_exchanges"] if replay == "unaged_exchanges" else 0 for item in records]


def test_tool_returned_images_of_aged_exchanges_leave_the_request(tmp_path: Path) -> None:
    run = _run(tmp_path, perception_script(16, [30000] * 16, look=True))
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    last = _projections(run)[-1]
    externalized = last["exchange_aging"]["delivered_tool_images_externalized"]
    assert last["exchange_aging"]["aged_exchanges"] == 6 and externalized > 0
    tool_images = sum(item["inline_images"] for item in last["action_step_inventory"])
    request = json.loads(run["requests"][-1])
    inline = sum(block.get("type") == "image_url" for message in request["messages"]
                 if isinstance(message["content"], list) for block in message["content"])
    assert last["active_inline_images"] == inline < tool_images
    markers = [block["text"] for message in request["messages"] if isinstance(message["content"], list)
               for block in message["content"] if block.get("type") == "text"
               and block["text"].startswith("<recoverable_artifact>")]
    assert len(markers) == externalized and all('"open":"observe"' in text for text in markers)
    assert run["evaluation"]["context_artifact_provenance"]["status"] == "valid"


def test_camera_images_stay_inline_until_their_exchange_ages(tmp_path: Path) -> None:
    # Every action brings an observation with its camera images; every report is above the watermark.
    script = episodes.action_script(8)
    for response in script:
        if "usage" in response:
            response["usage"] = {"input_tokens": 30000, "output_tokens": 30}
    run = _run(tmp_path, script, changes={"camera_images": "exchange_window_v1", "camera_tail_observations": None})
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    projections = _projections(run)
    assert all(payload["camera_delivery"]["camera_images"] == "exchange_window_v1"
               and "tail" not in payload["camera_delivery"] for payload in projections)
    relations = [payload["request_relation"] for payload in projections]
    assert set(relations) <= {"initial", "append_only", "projection_epoch"} and "projection_epoch" in relations
    requests = [json.loads(data) for data in run["requests"]]
    blocks = [[block for message in request["messages"] if isinstance(message["content"], list)
               for block in message["content"]] for request in requests]
    images = [sum(block.get("type") == "image_url" for block in request) for request in blocks]
    # Before the first aging every camera image is inline and the count only grows.
    first = relations.index("projection_epoch")
    assert images[:first] == sorted(images[:first]) and images[first] < images[first - 1]
    assert all(payload["exchange_aging"]["delivered_camera_images_externalized"] == 0 for payload in projections[:first])
    last = projections[-1]
    record = last["exchange_aging"]
    assert record["delivered_camera_images_externalized"] > 0 and record["delivered"]["camera_images"] > 0
    assert last["active_inline_images"] == images[-1]
    markers = [block["text"] for block in blocks[-1]
               if block.get("type") == "text" and block["text"].startswith("<recoverable_artifact>")]
    assert len(markers) == record["delivered_camera_images_externalized"] + record["delivered_tool_images_externalized"]
    assert all('"open":"observe"' in text for text in markers)
    # The latest observation's camera images are inline, after their labels.
    observations = [message for message in requests[-1]["messages"] if isinstance(message["content"], list)
                    and any(block.get("type") == "text" and "latest at this message" in block["text"]
                            for block in message["content"])]
    assert any(block.get("type") == "image_url" for block in observations[-1]["content"])
    assert not any(block.get("type") == "image_url" for block in observations[0]["content"])
    assert run["evaluation"]["context_artifact_provenance"]["status"] == "valid"


def test_the_hard_backstop_keeps_every_request_under_the_limit(tmp_path: Path) -> None:
    # Hard limit 32,000 - 2,048 - 2,048 = 27,904. No Provider usage is reported, so no watermark is
    # crossed and the backstop alone, on the Harness estimate, keeps each request under the limit.
    run = _run(tmp_path, perception_script(14, None, message_bytes=9000),
               context_window_tokens=32000, max_output_tokens=2048)
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    projections = _projections(run)
    backstops = [payload["hard_backstop"] for payload in projections if payload["hard_backstop"]]
    assert backstops and all(item["post_estimated_input_tokens"] <= 27904 for item in backstops)
    assert {item["kind"] for payload in projections for item in payload["projection_transitions"]} == {"hard_backstop"}
    assert all(payload["request_estimate"]["tokens"] <= 27904 for payload in projections)
    assert _aged_arguments(json.loads(run["requests"][-1]))[-2:] == [False, False]


def test_a_request_that_cannot_fit_ends_the_turn_with_context_exhausted(tmp_path: Path) -> None:
    run = _run(tmp_path, perception_script(3, None, message_bytes=120000),
               context_window_tokens=32000, max_output_tokens=2048)
    assert run["result"]["termination"] == "context_exhausted", run["result"]
