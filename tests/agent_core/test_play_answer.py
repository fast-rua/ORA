"""``ora play``: a text-only answer to a message that joined a running instruction does not end it.

A Synthetic free-play session on the Pi core against the scripted loopback
model (``fake_model_server --play``): the instruction makes three moves, then
reports. After its first move the operator asks a question the model answers
in words only ("just answer"). That reply is the joined message's answer
(``play_answer``), not the instruction's report: the instruction goes on with
the next act on the same observation, makes its remaining moves and ends
``reported``. The session evaluates valid and seals.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.agent_core import episodes, play_session
from tests.support.agent_core.play_session import wait

pytestmark = pytest.mark.usefixtures("pi_core")

ROOT = Path(__file__).resolve().parents[2]


def _operator(root: Path) -> None:
    wait(lambda: play_session.phase(root) == "awaiting_instruction", "the first scene")
    play_session.send(root, "i1", "instruction", "raise the left arm in three small steps")
    wait(lambda: play_session.count(root, "motion_receipt") >= 1, "the first move")
    play_session.send(root, "m1", "instruction", "just answer: how far along are you?")
    wait(lambda: play_session.count(root, "play_instruction_finished") == 1, "the instruction's report")
    play_session.send(root, "q1", "quit", "/quit")


def test_an_answer_to_a_joined_question_keeps_the_instruction_running(tmp_path: Path, monkeypatch) -> None:
    from ora.orchestrator import paths
    from ora.orchestrator.evaluation.evaluate import evaluate, seal
    from ora.orchestrator.spec.quickstart import environment_policy

    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / "state")
    port_file = tmp_path / "port"
    server = subprocess.Popen([sys.executable, "-m", "tests.support.agent_core.fake_model_server", str(port_file),
                               "--play", "--delay", "0.3"], cwd=ROOT)
    try:
        wait(lambda: port_file.is_file() and port_file.read_text().strip() != "", "the scripted model", 30)
        (tmp_path / "pi-agent").mkdir()
        with episodes.environment(PI_CODING_AGENT_DIR=str(tmp_path / "pi-agent"), PI_OFFLINE="1",
                                  ORA_CONFIG_HOME=str(tmp_path / "ora-config"),
                                  ORA_AGENT_CORE="pi", ORA_PROVIDER="openai_compatible", ORA_MODEL="fake-play",
                                  ORA_BASE_URL=f"http://127.0.0.1:{port_file.read_text().strip()}/v1",
                                  ORA_API_KEY="fake-key", ORA_SUPPORTS_IMAGES="1"):
            root = tmp_path / "play-run"
            result, failures = play_session.run_play(root, play_session.play_plan(environment_policy()), _operator)
    finally:
        server.kill()
        server.wait()
    assert not failures, failures
    (instruction,) = result["play"]["instructions"]
    assert instruction["instruction_id"] == "i1" and instruction["ended"] == "reported"
    assert instruction["joined_messages"] == ["m1"] and instruction["answered_messages"] == ["m1"]
    assert instruction["decisions"] == 3, "it went on to its remaining moves after the answer"

    events = play_session.events(root)
    (answer,) = [event["record"] for event in events if event["type"] == "play_answer"]
    assert (answer["instruction_id"], answer["message_id"], answer["instruction_index"]) == ("i1", "m1", 0)
    kinds = [event["type"] for event in events]
    # The answer came between the two moves, and the instruction ended only after the second one.
    first_move, answered = kinds.index("motion_receipt"), kinds.index("play_answer")
    assert first_move < answered < len(kinds) - 1 - kinds[::-1].index("motion_receipt") \
        < kinds.index("play_instruction_finished")
    agent = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    replies = [json.loads((root / event["artifact_ref"]["path"]).read_text()) for event in agent
               if event["type"] == "provider_response"]
    (worded,) = [reply for reply in replies if any("You asked" in str(block.get("text", ""))
                                                    for block in reply.get("content") or [])]
    assert not [block for block in worded["content"] if block.get("type") == "toolCall"], "a text-only answer"

    evaluation = evaluate(root)
    assert evaluation["status"] == "valid", evaluation["reasons"]
    assert seal(root)
