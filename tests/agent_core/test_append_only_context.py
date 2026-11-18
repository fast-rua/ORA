"""Context management v8 on the Pi core: an append-only history and its camera delivery.

* ``rolling_tail_v1``: every request ends with the camera tail of the latest
  observations, each observation message keeps one reference instead of its
  camera images, and consecutive requests of one epoch are related as
  ``camera_tail_append``. No recoverable-artifact marker or reader is shown.
* ``inline_until_compaction_v1``: the camera images stay in their observation
  messages; requests are ``append_only``.
* Provider input above ``compaction_trigger_tokens`` admits a rollout
  compaction (L6): older ActionSteps become one checkpoint without an
  exact-transcript Pointer, and the camera tail keeps rolling.
* A request the provider reported over the hard limit is still sent, with a
  warning, instead of ending the episode with ``context_exhausted``.

Every run evaluates valid.
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


def management(mode: str, window: int | None, *, compaction: int = 150000) -> dict[str, Any]:
    return {
        "schema": "ora.native-context-management.v8",
        "causal_exact_tail_action_steps": 3,
        "camera_images": mode,
        "camera_tail_observations": window,
        "episode_checkpoint_trigger_tokens": 150000,
        "compaction_trigger_tokens": compaction,
        "emergency_fallback_ratio": 0.8,
        "emergency_target_ratio": 0.65,
        "checkpoint_target_estimated_tokens": 1200,
        "checkpoint_max_estimated_tokens": 1600,
        "image_estimate_tokens": 530,
    }


def _run(tmp_path: Path, mode: str, window: int | None, *, actions: int = 6, usage: list[int] | None = None,
         compactions: int = 0, **policy: Any) -> dict[str, Any]:
    script = episodes.action_script(actions)
    for response, tokens in zip(script, usage or []):
        response["usage"] = {"input_tokens": tokens, "output_tokens": 30}
    provider = types.SimpleNamespace(pi_provider_spec={
        "kind": "fake", "responses": script, "compactor_responses": compactor_responses(compactions)})
    run = episodes.run_core("pi", tmp_path / "out", actions=actions, provider=provider, policy_overrides={
        "context_management": management(mode, window, **({"compaction": policy.pop("compaction")}
                                                           if "compaction" in policy else {})),
        "context_tools": [], **policy})
    assert run["evaluation"]["status"] == "valid", run["evaluation"].get("reasons")
    return run


def _texts(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    return [block["text"] for block in content or [] if block.get("type") == "text"]


def _images(message: dict[str, Any]) -> int:
    content = message.get("content")
    return sum(block.get("type") == "image_url" for block in content) if isinstance(content, list) else 0


def _projections(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [event["payload"] for event in run["events"] if event["type"] == "context_projected"]


def _no_reader(request: dict[str, Any]) -> None:
    names = [tool["function"]["name"] for tool in request["tools"]]
    assert "artifact_read" not in names
    observe = next(tool["function"] for tool in request["tools"] if tool["function"]["name"] == "observe")
    assert "ref" not in observe["parameters"].get("properties", {})
    assert not any("<recoverable_artifact>" in text or "artifact://" in text
                   for message in request["messages"] for text in _texts(message))


def test_the_rolling_tail_shows_each_observations_camera_images_once_at_the_end(tmp_path: Path) -> None:
    run = _run(tmp_path, "rolling_tail_v1", 2)
    plan = json.loads((run["root"] / "plan.json").read_text())
    assert plan["policy"]["context_tools"] == []
    requests = [json.loads(data) for data in run["requests"]]
    cameras = None
    for request in requests:
        _no_reader(request)
        tail = request["messages"][-1]
        assert _texts(tail)[0].startswith("CAMERA TAIL · latest ")
        observations = [message for message in request["messages"][:-1]
                        if any(text.startswith("Camera images of observation step ") for text in _texts(message))]
        assert observations and all(_images(message) == 0 for message in observations)
        cameras = cameras or _images(tail)
        assert _images(tail) == cameras * min(2, len(observations))
        assert _texts(tail)[-1].endswith(" · current")
    relations = [payload["request_relation"] for payload in _projections(run)]
    assert relations[0] == "initial" and "camera_tail_append" in relations
    assert set(relations) <= {"initial", "camera_tail_append", "append_only"}
    record = _projections(run)[-1]["camera_delivery"]
    assert record["camera_images"] == "rolling_tail_v1" and record["tail_observations"] == 2
    assert [item["action_step_index"] for item in record["tail"]] == [
        item["action_step_index"] for item in _projections(run)[-1]["action_step_inventory"]][-2:]


def test_inline_until_compaction_keeps_camera_images_in_the_history(tmp_path: Path) -> None:
    run = _run(tmp_path, "inline_until_compaction_v1", None)
    requests = [json.loads(data) for data in run["requests"]]
    for request in requests:
        _no_reader(request)
        assert not _texts(request["messages"][-1])[0].startswith("CAMERA TAIL")
    first, last = requests[0], requests[-1]
    assert sum(map(_images, last["messages"])) > sum(map(_images, first["messages"]))
    relations = {payload["request_relation"] for payload in _projections(run)}
    assert relations <= {"initial", "append_only"}


def test_a_compaction_summarizes_older_steps_and_the_tail_keeps_rolling(tmp_path: Path) -> None:
    usage = [5000] * 6 + [30000] * 20  # above compaction_trigger_tokens from the fourth action on
    run = _run(tmp_path, "rolling_tail_v1", 2, actions=8, usage=usage, compactions=4, compaction=20000)
    transitions = [item for payload in _projections(run) for item in payload["projection_transitions"]]
    assert "emergency_compaction" in {item["kind"] for item in transitions}
    requests = [json.loads(data) for data in run["requests"]]
    last = requests[-1]
    checkpoint = [text for message in last["messages"] for text in _texts(message)
                  if text.startswith("<emergency_context_checkpoint>")]
    assert checkpoint and "exact_transcript" not in checkpoint[0]
    assert _texts(last["messages"][-1])[0].startswith("CAMERA TAIL · latest 2 observations")
    assert {payload["request_relation"] for payload in _projections(run)} >= {"projection_epoch", "camera_tail_append"}


def test_an_over_limit_request_is_sent_with_a_warning(tmp_path: Path) -> None:
    usage = [5000] * 8 + [17000] * 20  # hard limit 20000 - 2048 - 2048 = 15904
    run = _run(tmp_path, "inline_until_compaction_v1", None, actions=6, usage=usage, compactions=4,
               context_window_tokens=20000, max_output_tokens=2048)
    assert run["result"]["termination"] != "context_exhausted"
    warnings = [payload["context_over_limit"] for payload in _projections(run) if payload["context_over_limit"]]
    assert warnings and warnings[0]["status"] == "warning_sent_anyway"
    assert warnings[0]["provider_input_tokens"] == 17000 and warnings[0]["hard_input_tokens"] == 15904
