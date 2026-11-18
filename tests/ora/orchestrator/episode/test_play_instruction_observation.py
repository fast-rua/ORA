"""Free play: every new instruction starts from a fresh look at the world.

When the operator's next instruction opens, the Supervisor publishes one new
observation through the opening scene's path (``publish_observation``, cause
``instruction_start``) before the instruction's first model call; the act that
delivers the instruction carries it. A message that joins a running instruction
takes no new look. When the look fails, the instruction still runs on the last
observation and its delivery carries a note that says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ora.agent.evidence.policy_state import PolicyStateStore, verify_policy_state
from ora.environment.synthetic import SyntheticSession
from ora.orchestrator.episode.interaction import EpisodeInteractionState


class _Inbox:
    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []
        self.acknowledged: list[tuple[str, str]] = []

    def poll(self) -> list[dict[str, Any]]:
        return list(self.commands)

    def acknowledge(self, command: dict[str, Any], status: str, **_fields: Any) -> None:
        self.acknowledged.append((command["id"], status))


def _waiting_loop(root: Path) -> tuple[Any, _Inbox, list[Any]]:
    """A PlayLoop waiting for an instruction on observation step 0."""

    from ora.orchestrator.episode.play import PlayLoop

    loop = object.__new__(PlayLoop)
    inbox = _Inbox()
    delivered: list[Any] = []
    root.mkdir(parents=True)
    loop.run_root, loop.events, loop.native = root, root / "events.jsonl", root / "native"
    loop.operator_inbox = inbox
    loop.planner = SimpleNamespace(accept_runtime_event=delivered.append, announce_play_state=lambda record: None)
    loop.interaction = EpisodeInteractionState(run_id="play-run")
    loop.interaction.start()
    loop.interaction.publish_observation(observation_id="obs-0", step=0)
    loop.interaction.await_instruction(reason="scene_ready")
    loop.termination, loop.interrupted, loop.scene_over = None, False, False
    loop._check_session_budget = lambda: False
    loop.play_scenes, loop.round_index = [{"seed": 0}], 0
    loop.segment, loop.segments, loop.play_pending, loop.play_seen = None, [], [], set()
    loop.play_stop, loop.play_controls, loop.operator_message_count = None, 0, 0
    loop.decisions = loop.refused_proposals = loop.model_calls = 0
    loop.instruction_budgets = {"wall_time_s": 3600, "max_model_calls": 80, "max_robot_decisions": 24}
    loop._moving = None
    loop.observation_unix_s = 1000.0
    loop.live_sequence, loop.auxiliary_failures = 0, []
    loop.operator_port = SimpleNamespace(publish=lambda event: None)
    loop.pointer, loop.environment_status = {"observation": 0}, {"terminal": False}
    return loop, inbox, delivered


def _events(loop: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in loop.events.read_text().splitlines()]


def _fake_publish(monkeypatch: pytest.MonkeyPatch, calls: list[str], *, fail: bool = False) -> None:
    from ora.orchestrator.episode import play

    def publish(loop: Any, cause: str, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append(cause)
        if fail:
            raise RuntimeError("overhead camera timed out")
        step = loop.interaction.observation_step + 1
        loop.interaction.publish_observation(observation_id=f"obs-{step}", step=step)
        loop.observation_unix_s = 2000.0
        return {"observation": step}, {"terminal": False}

    monkeypatch.setattr(play, "publish_observation", publish)


def test_a_new_instruction_takes_one_fresh_observation_before_its_first_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, inbox, delivered = _waiting_loop(tmp_path / "run")
    calls: list[str] = []
    _fake_publish(monkeypatch, calls)
    inbox.commands.append({"id": "i1", "type": "instruction", "message": "put the cup on the plate"})
    loop._await_instruction()
    assert calls == ["instruction_start"]
    assert loop.pointer == {"observation": 1} and loop.interaction.phase == "observation_ready"
    assert loop.interaction.observation_step == 1
    assert loop.segment["observation_step_at_start"] == 1
    started = [event["record"] for event in _events(loop) if event["type"] == "play_instruction_started"]
    assert started == [{"schema": "ora.play-instruction.v1", "index": 0, "instruction_id": "i1", "scene_index": 0,
                        "observation_step": 1, "observation": {"fresh": True, "observation_step": 1}}]
    # The instruction's delivery carries no note: its act's observation is the fresh one.
    loop._deliver_operator_messages(observation_step=1, before_model_call_id="play-call-000000")
    assert [event.payload["id"] for event in delivered] == ["i1"]
    assert "observation_note" not in delivered[0].payload


def test_a_message_that_joins_the_running_instruction_takes_no_new_look(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, inbox, delivered = _waiting_loop(tmp_path / "run")
    calls: list[str] = []
    _fake_publish(monkeypatch, calls)
    inbox.commands.append({"id": "i1", "type": "instruction", "message": "put the cup on the plate"})
    loop._await_instruction()
    loop._deliver_operator_messages(observation_step=1, before_model_call_id="play-call-000000")
    inbox.commands.append({"id": "m1", "type": "steer", "message": "the blue cup, not the red one"})
    loop._deliver_operator_messages(observation_step=1, before_model_call_id="play-call-000001")
    assert calls == ["instruction_start"]
    assert [(event.payload["id"], event.payload["joined"]) for event in delivered] == [("i1", False), ("m1", True)]
    assert all("observation_note" not in event.payload for event in delivered)


def test_a_failed_look_still_starts_the_instruction_with_a_note_on_the_last_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, inbox, delivered = _waiting_loop(tmp_path / "run")
    calls: list[str] = []
    _fake_publish(monkeypatch, calls, fail=True)
    monkeypatch.setattr("ora.orchestrator.episode.play.time.time", lambda: 1000.0 + 754.0)
    inbox.commands.append({"id": "i1", "type": "instruction", "message": "put the cup on the plate"})
    loop._await_instruction()
    assert calls == ["instruction_start"]
    assert loop.pointer == {"observation": 0} and loop.interaction.phase == "observation_ready"
    assert loop.segment["observation_step_at_start"] == 0
    failures = [event for event in _events(loop) if event["type"] == "auxiliary_failure"]
    assert [(item["stage"], item["error"]) for item in failures] == [
        ("play:instruction_observation", "overhead camera timed out")]
    started = [event["record"] for event in _events(loop) if event["type"] == "play_instruction_started"]
    assert started[0]["observation"] == {"fresh": False, "observation_step": 0,
                                         "error": "RuntimeError: overhead camera timed out"}
    inbox.commands.append({"id": "m1", "type": "steer", "message": "careful"})
    loop._deliver_operator_messages(observation_step=0, before_model_call_id="play-call-000000")
    opening, joined = delivered[0].payload, delivered[1].payload
    assert opening["observation_note"] == (
        "ORA could not take a fresh observation at the start of this instruction "
        "(RuntimeError: overhead camera timed out). The observation with it is the latest one, step 0, "
        "captured 12 min 34 s ago; the scene may have changed since."
    )
    assert "observation_note" not in joined


def test_a_look_that_failed_after_publishing_is_a_fault_not_a_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.orchestrator.episode import play

    loop, inbox, _delivered = _waiting_loop(tmp_path / "run")

    def half(loop: Any, cause: str, **_kwargs: Any) -> Any:
        loop.interaction.publish_observation(observation_id="obs-1", step=1)
        raise RuntimeError("live port closed")

    monkeypatch.setattr(play, "publish_observation", half)
    inbox.commands.append({"id": "i1", "type": "instruction", "message": "go"})
    with pytest.raises(RuntimeError, match="live port closed"):
        loop._await_instruction()


def test_the_policy_state_timeline_admits_an_instruction_start_observation(tmp_path: Path) -> None:
    observation_root = tmp_path / "policy-observations"
    session = SyntheticSession(tmp_path / "native", "synthetic_tool_boundary", 0)
    session.start()
    store = PolicyStateStore(tmp_path, "play-run", "play-plan")
    for cause in ("reset", "instruction_start"):
        observation, _ = session.observe(observation_root)
        store.publish(observation=observation, observation_root=observation_root / observation["observation_id"],
                      cause=cause, task_instruction="Free play.", previous_public_action=None,
                      remaining_budget={"robot_decisions": 1, "model_calls": 2})
    store.finalize(state="completed", termination="operator_quit")
    assert verify_policy_state(tmp_path)["status"] == "valid"
