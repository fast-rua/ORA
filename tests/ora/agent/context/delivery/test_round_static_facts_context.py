"""D3b in the lossless Context: the current round's slot is never aged.

One ``round_static_facts`` slot follows each round's first observation, as
``native_slots_v4`` does; a Synthetic episode and Supervisor rounds carry it on
the agent core. That it stays byte-exact through every L1-L4 aging batch of
its round and ages to a recoverable pointer when its round's media are released
at reset is held by the recorded Context fixtures
(``tests/data/context_differential/stride_v6_4_slots_reset``,
``captured_slots_*``) the TypeScript pipeline replays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ora.agent.context.delivery import round_static_facts as slot
from ora.agent.session_core.constants import TOOL_IMAGE_CALL_LABELS
from tests.support.agent.lossless_context_stride import _episode, _v6


def _slot(step: int, session_round: int) -> dict[str, Any]:
    message = slot.slot_message(round_index=session_round, actual_sequence=step, fields={
        "camera_geometry": {"head": {"intrinsic_cv": [[600.0, 0.0, 320.0]] * 24}},
        "public_robot_facts": {"controller_feedback": {"position_tolerance_m": 0.01}},
    })
    message["_source_event_ids"] = [f"agent-event-{step * 10:06d}"]
    return message


def _slot_texts(messages):
    texts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
            text = content[0].get("text", "")
            if text.startswith('{"round_static_facts"') or '"round_static_facts_externalized"' in text:
                texts.append(text)
    return texts


# -- One synthetic episode through the Supervisor --------------------------------


def _tool(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, sort_keys=True)},
    }]}


def _episode_run(root: Path, composition: str):
    from tests.support.orchestrator.native_synthetic_spec import synthetic_spec
    from ora.agent.providers.fake import FakeProvider
    from ora.orchestrator.episode.coordinator import run_episode
    from ora.orchestrator.evaluation.evaluate import evaluate
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    spec = synthetic_spec(seed=0)
    spec["policy"]["context_retention"] = "action_step_lossless_v4"
    spec["policy"]["summary_policy"] = "rollout_checkpoint_v1"
    spec["policy"]["context_management"] = _v6(4)
    plan = resolve_run_spec(spec, root=Path(__file__).resolve().parents[5])
    # RunSpec registration of native_slots_v4 is integration 0058 unit A0's
    # (orchestrator/spec/composition.py); the session and Supervisor run it now.
    plan["policy"]["context_composition"] = composition
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    provider = FakeProvider(responses=[
        _tool("c0", "close_gripper", {}), _tool("o0", "open_gripper", {}),
        _tool("c1", "close_gripper", {}),
        _tool("f0", "finish_episode", {"outcome": "failure", "reason": "done",
                                       "attempts": ["three steps"]}),
    ])
    result = run_episode(run_root=root, plan=plan, run_id=f"d3b-{composition}",
                         provider_override=provider)
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    requests = [json.loads((root / event["payload"]["canonical_request_ref"]["path"]).read_text())
                for event in events if event["type"] == "provider_request"]
    return result, events, requests, evaluate(root)


def test_a_synthetic_episode_carries_one_slot_and_stays_append_only(tmp_path: Path) -> None:
    result, events, requests, evaluation = _episode_run(tmp_path / "v4", slot.COMPOSITION)
    assert result["state"] == "completed" and evaluation["status"] == "valid"
    started = next(event for event in events if event["type"] == "session_started")
    assert started["payload"]["context_capabilities"] == [slot.CAPABILITY, TOOL_IMAGE_CALL_LABELS]
    for request in requests:
        slots = [index for index, message in enumerate(request["messages"])
                 if _slot_texts([message])]
        assert len(slots) == 1
        # The slot follows the round's first observation.
        assert "INITIAL ACTUAL" in json.dumps(request["messages"][slots[0] - 1])
    last = requests[-1]["messages"]
    observation = json.loads(next(
        message["content"][0]["text"] for message in reversed(last)
        if message["role"] == "user" and '"current_actual"' in message["content"][0]["text"]))
    fields = json.loads(_slot_texts(last)[0])["round_static_facts"]["current_actual"]
    assert observation["current_actual"]["static_facts"]["round_slot"] == 1
    assert slot.restore(observation["current_actual"], fields)["actual_sequence"] == 3
    projections = [event["payload"] for event in events if event["type"] == "context_projected"]
    assert [item["request_relation"] for item in projections[1:]] == ["append_only"] * 3
    assert any(record["kind"] == "round_static_facts" for record in projections[-1][
        "context_sources"])

    legacy, legacy_events, legacy_requests, legacy_evaluation = _episode_run(
        tmp_path / "v3", "native_slots_v3")
    assert legacy_evaluation["status"] == "valid"
    assert not any(_slot_texts(request["messages"]) for request in legacy_requests)
    assert "context_capabilities" not in next(
        event for event in legacy_events if event["type"] == "session_started")["payload"]


def test_runspec_policy_rules_for_native_slots_v4() -> None:
    from ora.orchestrator.spec.native_policy import _resolve_native_observation
    from ora.orchestrator.spec.values import RunSpecError

    policy = {"context_composition": slot.COMPOSITION, "context_profile": "native_causal_context_v1"}
    with pytest.raises(RunSpecError, match="native_slots_v4 requires context_retention"):
        _resolve_native_observation(dict(policy), "append_only_epoch_v1")
    with pytest.raises(RunSpecError, match="native_slots_v3 or native_slots_v4"):
        _resolve_native_observation({**policy, "context_composition": "native_slots_v5"}, None)


def test_each_supervisor_round_gets_its_slot_and_the_previous_one_ages(tmp_path: Path) -> None:
    from ora.orchestrator.evaluation.evaluate import evaluate
    from ora.orchestrator.spec.run_spec import resolve_run_spec
    from tests.support.orchestrator import multi_round
    from tests.support.orchestrator.multi_round import ROOT

    spec = multi_round.spec()
    spec["session"] = {"cadence": "persistent_agent_k_rounds_v1", "rounds": 3,
                       "round_seeds": [0, 1, 2]}
    spec["policy"]["context_retention"] = "action_step_lossless_v4"
    spec["policy"]["summary_policy"] = "rollout_checkpoint_v1"
    spec["policy"]["context_management"] = _v6(4)
    plan = resolve_run_spec(spec, root=ROOT)
    # Multi-round registration of native_slots_v4 (orchestrator/spec/runtime.py) is A0's.
    plan["policy"]["context_composition"] = slot.COMPOSITION
    root = tmp_path / "run"
    result = multi_round.run(root, plan, multi_round.finish_each_round(3), run_id="d3b-rounds")
    assert result["state"] == "completed" and result["round_count"] == 3
    assert evaluate(root)["status"] == "valid"
    requests = multi_round.provider_requests(root)
    assert len(requests) == 3
    for index, request in enumerate(requests):
        assert multi_round.announces_round(request, index + 1)
        texts = _slot_texts(request["messages"])
        assert [json.loads(text).get("round_static_facts_externalized", False)
                for text in texts] == [True] * index + [False]
        assert json.loads(texts[-1])["round_static_facts"]["round"] == index + 1
