"""Rollout compactor and checkpoint contracts.

The action_step_pressure_* retentions these contracts were first written for
compose only at the commits that composed them; the compactor and the
checkpoint validator serve the lossless retention's L5/L6 checkpoints.
"""

from __future__ import annotations

import copy
import base64
import json
from pathlib import Path
from typing import Any

import pytest

from ora.agent.context.causal_value import (
    checkpoint_forbidden_fragments,
)
from tests.support.agent.rollout_compactor import RolloutCheckpointCompactor
from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.canonical import ArtifactStore
from tests.support.agent.agent_events import TrajectoryRecorder
from ora.agent.providers.fake import FakeProvider


CHECKPOINT = {
    "task_progress": ["The object has not yet been confirmed in the gripper."],
    "confirmed_facts": ["The latest retained view is the source of truth."],
    "invalidated_beliefs": ["A prior grasp belief was not visually confirmed."],
    "failed_attempts": [{
        "failure": "The gripper closed without confirmed transport.",
        "evidence": "A later retained image did not show the object moving with the hand.",
        "correction": "Re-ground and confirm attachment before transport.",
    }],
    "current_constraints": ["The next decision must use the retained actual view."],
    "do_not_repeat": ["Do not assume closure means attachment."],
    "re_ground_next": ["Locate the target again in the current view."],
}


def _image(label: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(
            label.encode("utf-8")
        ).decode("ascii")},
    }


def _protected() -> list[dict[str, Any]]:
    return [
        {
            "role": "system", "content": "system",
            "_protected_context": True, "_context_source": "harness_system",
        },
        {
            "role": "user", "content": "embodiment",
            "_protected_context": True, "_context_source": "project_instructions",
        },
        {
            "role": "user", "content": "pick object",
            "_protected_context": True, "_context_source": "episode_task",
        },
    ]


def _observation(step: int, *, receipt: bool) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps({"current_actual": {"step": step}})},
            _image(f"image-{step}"),
        ],
        "_observation": {
            "step": step,
            "receipt": {
                "event_id": f"receipt-{step}",
                "action_id": f"action-{step - 1}",
            } if receipt else None,
        },
        "_causal_observation": {"step": step},
        "_source_event_ids": [f"agent-event-{step * 3:06d}"],
    }


def test_qualitative_need_to_reground_coordinates_is_not_replay_leakage() -> None:
    qualitative = copy.deepcopy(CHECKPOINT)
    qualitative["current_constraints"] = [
        "Re-ground against the current observation rather than carried-over coordinates."
    ]
    assert checkpoint_forbidden_fragments(qualitative) == []

    actual_world_coordinate = copy.deepcopy(CHECKPOINT)
    actual_world_coordinate["current_constraints"] = [
        "Reuse world coordinates from the earlier move."
    ]
    assert checkpoint_forbidden_fragments(actual_world_coordinate) == [
        "world coordinates"
    ]


def test_future_task_check_is_not_mistaken_for_an_official_outcome() -> None:
    neutral = copy.deepcopy(CHECKPOINT)
    neutral["re_ground_next"] = [
        "After placement, call check_task for the official verdict."
    ]
    assert checkpoint_forbidden_fragments(neutral) == []

    asserted = copy.deepcopy(CHECKPOINT)
    asserted["confirmed_facts"] = ["The official verdict was false."]
    assert checkpoint_forbidden_fragments(asserted) == [
        "official verdict was false"
    ]

    keyed = copy.deepcopy(CHECKPOINT)
    keyed["confirmed_facts"] = ["official_success=false"]
    assert checkpoint_forbidden_fragments(keyed) == ["official_success"]


def test_compactor_prompt_forbids_numeric_geometry_and_requests_qualitative_state() -> None:
    from ora.agent.context.lossless.compactor import COMPACTOR_SYSTEM_PROMPT

    assert "Express spatial state only qualitatively" in COMPACTOR_SYSTEM_PROMPT
    assert "z=" in COMPACTOR_SYSTEM_PROMPT
    assert "not this checkpoint, own all geometry" in COMPACTOR_SYSTEM_PROMPT


@pytest.mark.parametrize("model,structured", [
    ("deepseek-flash", True), ("deepseek-v4-flash", True),
    ("deepseek-v4-pro", True), ("deepseek-v4-flash-vision-exp", True),
    ("k3", False), ("custom-model", False),
])
def test_compactor_provider_options_are_separate_from_policy_requests(tmp_path, model, structured):
    from ora.agent.providers.openai_compatible import OpenAICompatibleProvider

    class Provider(FakeProvider):
        checkpoint_request_options = staticmethod(OpenAICompatibleProvider.checkpoint_request_options)
        prepare_request = staticmethod(OpenAICompatibleProvider.prepare_request)

    provider = Provider(responses=[{"message": {
        "role": "assistant", "content": json.dumps(CHECKPOINT),
    }}])
    recorder = TrajectoryRecorder(tmp_path, "run", "session")
    compactor = RolloutCheckpointCompactor(
        provider=provider, model=model, reasoning_effort="high", stream=True,
        artifacts=ArtifactStore(tmp_path), event_sink=recorder.append,
    )
    result = compactor.summarize(
        messages=[{"role": "user", "content": "completed episode evidence"}],
        source_refs=[{"event_id": "agent-event-000001"}],
    )
    assert json.loads(result.content) == CHECKPOINT
    request = provider.requests[0]
    assert request["reasoning_effort"] == ("low" if structured else "high")
    assert request.get("response_format") == ({"type": "json_object"} if structured else None)
    assert request["max_tokens"] == 8192 and "tools" not in request
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    saved = json.loads((tmp_path / events[0]["artifact_ref"]["path"]).read_text())
    assert saved == provider.prepare_request(request)
    # The same adapter still sends the selected high effort for policy calls.
    normal = provider.prepare_request({"model": model, "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "act"}], "max_tokens": 16384})
    assert normal["reasoning_effort"] == "high" and "response_format" not in normal


def test_compactor_request_options_cannot_change_model_or_add_tools(tmp_path):
    provider = FakeProvider(responses=[])
    provider.checkpoint_request_options = lambda **kwargs: {"model": "other", "tools": []}
    compactor = RolloutCheckpointCompactor(
        provider=provider, model="deepseek-flash", reasoning_effort="high", stream=True,
        artifacts=ArtifactStore(tmp_path), event_sink=lambda *args, **kwargs: {},
    )
    with pytest.raises(PolicySessionError, match="unsupported checkpoint request options"):
        compactor.summarize(messages=[], source_refs=[])
    assert provider.requests == []


def test_compactor_is_one_fresh_tool_free_multimodal_call(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path)
    recorder = TrajectoryRecorder(tmp_path, "run", "session")
    provider = FakeProvider(responses=[{
        "message": {"role": "assistant", "content": json.dumps(CHECKPOINT)},
        "usage": {
            "input_tokens": 100, "cached_input_tokens": 0,
            "output_tokens": 80, "total_tokens": 180,
        },
    }])
    compactor = RolloutCheckpointCompactor(
        provider=provider,
        model="k3",
        reasoning_effort="medium",
        stream=True,
        artifacts=artifacts,
        event_sink=recorder.append,
    )
    refs = [{"event_id": "agent-event-000001"}]
    result = compactor.summarize(
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "historical evidence"}, _image("real-bytes"),
        ]}],
        source_refs=refs,
    )
    assert json.loads(result.content) == CHECKPOINT
    assert len(provider.requests) == 1
    assert provider.requests[0]["max_tokens"] == 8192
    assert "3,240 UTF-8 bytes" in provider.requests[0]["messages"][0]["content"]
    assert "rewrite and\nconsolidate" in provider.requests[0]["messages"][0]["content"]
    assert "tools" not in provider.requests[0]
    assert provider.requests[0]["messages"][1]["content"][1] == _image("real-bytes")
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "context_compactor_attempt", "context_compactor_response",
    ]
    truncated = RolloutCheckpointCompactor(
        provider=FakeProvider(responses=[{
            "message": {"role": "assistant", "content": json.dumps(CHECKPOINT)},
            "finish_reason": "length",
        }]),
        model="k3", reasoning_effort="medium", stream=True,
        artifacts=artifacts, event_sink=recorder.append,
    )
    with pytest.raises(PolicySessionError, match="did not complete with stop"):
        truncated.summarize(
            messages=[{"role": "user", "content": "history"}],
            source_refs=refs,
        )
    truncated_events = [
        json.loads(line) for line in recorder.path.read_text().splitlines()
    ][-3:]
    assert [event["type"] for event in truncated_events] == [
        "context_compactor_attempt",
        "context_compactor_response",
        "context_compactor_failed",
    ]
    assert truncated_events[1]["payload"]["finish_reason"] == "length"


def test_a_bound_compactor_names_its_provider_binding_in_every_event(tmp_path: Path) -> None:
    """policy.context_compactor: attempt, response and failure name {provider_profile, model, endpoint}."""
    recorder = TrajectoryRecorder(tmp_path, "run", "session")
    binding = {"provider_profile": "compactor-b", "model": "model-b", "endpoint": "http://127.0.0.1:9/b/v1"}

    def compactor(reply: dict[str, Any], provider_binding: Any) -> RolloutCheckpointCompactor:
        return RolloutCheckpointCompactor(
            provider=FakeProvider(responses=[reply]), model="model-b", reasoning_effort="low", stream=True,
            artifacts=ArtifactStore(tmp_path), event_sink=recorder.append, provider_binding=provider_binding,
        )

    ok = {"message": {"role": "assistant", "content": json.dumps(CHECKPOINT)}}
    compactor(ok, binding).summarize(messages=[{"role": "user", "content": "history"}], source_refs=[])
    with pytest.raises(PolicySessionError, match="did not complete with stop"):
        compactor({**ok, "finish_reason": "length"}, binding).summarize(
            messages=[{"role": "user", "content": "history"}], source_refs=[])
    compactor(ok, None).summarize(messages=[{"role": "user", "content": "history"}], source_refs=[])
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "context_compactor_attempt", "context_compactor_response",
        "context_compactor_attempt", "context_compactor_response", "context_compactor_failed",
        "context_compactor_attempt", "context_compactor_response",
    ]
    assert all(event["payload"]["provider_binding"] == binding for event in events[:5])
    # An unbound compactor (the default) writes no such key, not even a null one.
    assert all("provider_binding" not in event["payload"] for event in events[5:])
