"""``ora play``: a reply the provider's output limit cut off before any tool call is no report.

A Synthetic free-play session on the Pi core against the scripted loopback
model (``fake_model_server --play``) with a nine-call session budget. The
first instruction's first reply is reasoning only, cut off at the output limit
(``finish_reason: "length"``): as in an episode, the agent core records
``provider_output_truncated`` and the model continues on the same observation
after ORA's continuation message, then moves and reports (four calls). The
second instruction's every reply is cut off: once the act's model calls (the
session's last five) are spent, the instruction ends ``output_truncated``
rather than ``reported``, and the spent session ends. (An instruction's own
call budget, which the Supervisor checks before each call, ends it as
``instruction_model_call_budget`` as before.) The session evaluates valid and
seals.
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
CONTINUATION = "reached the Provider output limit"


def _operator(root: Path) -> None:
    wait(lambda: play_session.phase(root) == "awaiting_instruction", "the first scene")
    play_session.send(root, "i1", "instruction", "think too long, then raise the left arm")
    wait(lambda: play_session.count(root, "play_instruction_finished") == 1, "the first instruction's report")
    play_session.send(root, "i2", "instruction", "always think too long about the right arm")
    wait(lambda: play_session.count(root, "play_instruction_finished") == 2, "the second instruction's end")
    # The session's model calls are spent: it ends by itself.


def test_a_cut_off_reply_continues_and_never_counts_as_the_report(tmp_path: Path, monkeypatch) -> None:
    from ora.orchestrator import paths
    from ora.orchestrator.evaluation.evaluate import evaluate, seal
    from ora.orchestrator.spec.quickstart import environment_policy

    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / "state")
    port_file = tmp_path / "port"
    server = subprocess.Popen([sys.executable, "-m", "tests.support.agent_core.fake_model_server", str(port_file),
                               "--play", "--delay", "0.2"], cwd=ROOT)
    try:
        wait(lambda: port_file.is_file() and port_file.read_text().strip() != "", "the scripted model", 30)
        (tmp_path / "pi-agent").mkdir()
        with episodes.environment(PI_CODING_AGENT_DIR=str(tmp_path / "pi-agent"), PI_OFFLINE="1",
                                  ORA_CONFIG_HOME=str(tmp_path / "ora-config"),
                                  ORA_AGENT_CORE="pi", ORA_PROVIDER="openai_compatible", ORA_MODEL="fake-play",
                                  ORA_BASE_URL=f"http://127.0.0.1:{port_file.read_text().strip()}/v1",
                                  ORA_API_KEY="fake-key", ORA_SUPPORTS_IMAGES="1"):
            root = tmp_path / "play-run"
            plan = play_session.play_plan(environment_policy(), budgets={"max_model_calls": 9},
                                          instruction_budgets={"max_model_calls": 9})
            result, failures = play_session.run_play(root, plan, _operator)
    finally:
        server.kill()
        server.wait()
    assert not failures, failures
    first, second = result["play"]["instructions"]
    assert (first["instruction_id"], first["ended"], first["decisions"]) == ("i1", "reported", 2)
    assert first["model_calls"] == 4, "one cut-off reply, two moves, the report"
    assert (second["instruction_id"], second["ended"], second["decisions"]) == ("i2", "output_truncated", 0)
    assert second["model_calls"] == 5
    assert result["termination"] == "model_call_budget"

    agent = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    truncated = [event for event in agent if event["type"] == "provider_output_truncated"]
    assert [event["payload"]["policy_outcome"] for event in truncated] == \
        ["continue_same_observation"] * 5 + ["output_truncated"]
    stops = {event["model_call_id"]: event["payload"]["stop_reason"] for event in agent
             if event["type"] == "provider_response"}
    assert all(stops[event["model_call_id"]] == "length" for event in truncated)
    requests = [event for event in agent if event["type"] == "provider_request"]
    # The call after each continued cut-off reply read ORA's continuation as its latest message.
    for event in truncated[:5]:
        after = requests[[request["model_call_id"] for request in requests].index(event["model_call_id"]) + 1]
        body = json.loads((root / after["artifact_ref"]["path"]).read_text())
        latest = body["messages"][-1]
        assert latest["role"] == "user" and CONTINUATION in json.dumps(latest["content"])

    evaluation = evaluate(root)
    assert evaluation["status"] == "valid", evaluation["reasons"]
    assert seal(root)
