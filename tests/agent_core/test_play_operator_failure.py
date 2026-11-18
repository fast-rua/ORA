"""A free-play test whose operator gives up ends its session instead of leaving it waiting.

The Supervisor of a free-play session waits for the operator's next instruction
as long as the session lasts. A test's operator is a thread that reads the
Supervisor's events while they are appended; a read can see part of an append,
and parsing that torn last line killed the operator before it sent its quit,
so the session (and the test) waited for good with the agent core idle.
``play_session.events`` reads only complete lines, an operator that fails quits
the session, and a session past its deadline is quit with every thread's stack.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support.agent_core import episodes, play_session
from tests.support.agent_core.play_session import wait

ROOT = Path(__file__).resolve().parents[2]


def test_the_operator_reads_only_complete_event_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first, second = {"type": "observation", "seq": 1}, {"type": "motion_receipt", "seq": 2}
    path.write_text(json.dumps(first) + "\n" + json.dumps(second)[:12], encoding="utf-8")  # an append in progress
    assert play_session.events(tmp_path) == [first]
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    assert play_session.events(tmp_path) == [first, second]


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator, **options) -> tuple[dict, list, float]:
    """A Synthetic free-play session on the Pi core driven by ``operator``; its result, failures and duration."""

    from ora.orchestrator import paths
    from ora.orchestrator.spec.quickstart import environment_policy

    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / "state")
    port_file = tmp_path / "port"
    server = subprocess.Popen([sys.executable, "-m", "tests.support.agent_core.fake_model_server", str(port_file),
                               "--play", "--delay", "0.05"], cwd=ROOT)
    try:
        wait(lambda: port_file.is_file() and port_file.read_text().strip() != "", "the scripted model", 30)
        (tmp_path / "pi-agent").mkdir()
        with episodes.environment(PI_CODING_AGENT_DIR=str(tmp_path / "pi-agent"), PI_OFFLINE="1",
                                  ORA_CONFIG_HOME=str(tmp_path / "ora-config"),
                                  ORA_AGENT_CORE="pi", ORA_PROVIDER="openai_compatible", ORA_MODEL="fake-play",
                                  ORA_BASE_URL=f"http://127.0.0.1:{port_file.read_text().strip()}/v1",
                                  ORA_API_KEY="fake-key", ORA_SUPPORTS_IMAGES="1"):
            started = time.monotonic()
            result, failures = play_session.run_play(
                tmp_path / "play-run", play_session.play_plan(environment_policy()), operator, **options)
            return result, failures, time.monotonic() - started
    finally:
        server.kill()
        server.wait()


@pytest.mark.usefixtures("pi_core")
def test_a_failing_operator_quits_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "play-run"

    def operator(_root: Path) -> None:
        wait(lambda: play_session.phase(root) == "awaiting_instruction", "the first scene")
        raise RuntimeError("the operator gave up")

    result, failures, _seconds = _session(tmp_path, monkeypatch, operator)
    assert [str(failure) for failure in failures] == ["the operator gave up"]
    assert result["state"] == "completed" and result["termination"] == "operator_quit", result
    (quit,) = [event["record"] for event in play_session.events(root) if event["type"] == "play_control"]
    assert (quit["id"], quit["command"], quit["status"]) == ("test-operator-failed", "quit", "applied")


@pytest.mark.usefixtures("pi_core")
def test_a_session_past_its_deadline_is_quit_with_every_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "play-run"

    def operator(_root: Path) -> None:  # returns without the quit the session waits for
        wait(lambda: play_session.phase(root) == "awaiting_instruction", "the first scene")

    result, failures, seconds = _session(tmp_path, monkeypatch, operator, deadline_s=5.0)
    (failure,) = failures
    assert "the play session was still running 5 s after it started" in str(failure)
    assert "_await_instruction" in str(failure), "the stack says where the Supervisor waited"
    assert result["termination"] == "operator_quit", result
    assert seconds < 5.0 + play_session.QUIT_GRACE_S
