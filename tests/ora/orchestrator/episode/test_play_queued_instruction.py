"""Free play: an instruction the operator queues while one runs waits for it to end, then is the next;
a queued instruction with nothing running starts at once, and a steer still joins the running one."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ora.orchestrator.episode.interaction import EpisodeInteractionState
from ora.orchestrator.episode.operator_control import PLAY_COMMANDS


class _Inbox:
    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []
        self.acknowledged: list[tuple[str, str]] = []

    def poll(self) -> list[dict[str, Any]]:
        return list(self.commands)

    def acknowledge(self, command: dict[str, Any], status: str, **_fields: Any) -> None:
        self.acknowledged.append((command["id"], status))


def _loop(root: Path) -> tuple[Any, _Inbox, list[dict[str, Any]]]:
    from ora.orchestrator.episode.play import PlayLoop

    loop = object.__new__(PlayLoop)
    inbox, announced = _Inbox(), []
    root.mkdir(parents=True)
    loop.run_root, loop.events, loop.native = root, root / "events.jsonl", root / "native"
    loop.operator_inbox = inbox
    loop.planner = SimpleNamespace(announce_play_state=announced.append)
    loop.publish_live = lambda kind, record: None
    loop.interaction = EpisodeInteractionState(run_id="play-run")
    loop.interaction.start()
    loop.interaction.publish_observation(observation_id="obs-0", step=0)
    loop.play_scenes, loop.round_index = [{"seed": 0}], 0
    loop.segment, loop.segments, loop.play_pending, loop.play_queued, loop.play_seen = None, [], [], [], set()
    loop.play_stop, loop.play_controls = None, 0
    loop.decisions = loop.refused_proposals = loop.model_calls = 0
    return loop, inbox, announced


def _command(identifier: str, kind: str) -> dict[str, Any]:
    return {"id": identifier, "type": kind, "message": f"{kind} {identifier}"}


def test_a_queued_instruction_waits_for_the_running_one(tmp_path: Path) -> None:
    assert "queued_instruction" in PLAY_COMMANDS
    loop, inbox, announced = _loop(tmp_path / "run")
    loop._begin_segment(_command("i1", "instruction"))
    loop.segment["delivered"] = True
    inbox.commands = [_command("q1", "queued_instruction"), _command("s1", "steer"), _command("q2", "queued_instruction")]
    loop._take_commands()

    assert [c["id"] for c in loop.play_pending] == ["s1"]            # the steer joins the running instruction
    assert [c["id"] for c in loop.play_queued] == ["q1", "q2"]
    assert [r["state"] for r in announced].count("message_queued_next") == 2

    loop.play_pending.clear()                                         # delivered before the next model call
    loop._finish_segment("reported")
    assert [c["id"] for c in loop.play_pending] == ["q1", "q2"] and loop.play_queued == []


def test_a_queued_instruction_with_nothing_running_is_the_next_at_once(tmp_path: Path) -> None:
    loop, inbox, _ = _loop(tmp_path / "run")
    inbox.commands = [_command("q1", "queued_instruction")]
    loop._take_commands()
    assert [c["id"] for c in loop.play_pending] == ["q1"] and loop.play_queued == []


def test_play_states_carry_wall_time_and_an_actions_approval_wait(tmp_path: Path) -> None:
    from ora.orchestrator.episode.play import _approval_wait_s

    loop, _, announced = _loop(tmp_path / "run")
    loop._announce("awaiting_instruction")
    assert isinstance(announced[-1]["unix_s"], float)
    receipt = {"receipt": {"phase_receipts": [{"receipt": {"timestamps": {"approval": {
        "created_unix_s": 10.0, "answered_unix_s": 13.25, "by": "board", "approved": True}}}}]}}
    assert _approval_wait_s(receipt) == 3.25
    assert _approval_wait_s({"execution_status": "reached"}) is None
