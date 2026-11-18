"""``ora play`` on Pi's own agent: instructions in turn, a stopped one, a shell, a scene reset and quit, one sealed run.

A Synthetic free-play session runs in this process on the Pi core, which runs
Pi as a coding agent (its own context, provider code and bash tool) against
the scripted loopback model (``fake_model_server --play``: two moves, then a
text report per instruction; "keep" moves until stopped; "bash" runs Pi's bash
tool once; "ora-robot" drives the robot from that shell; "codemode" from a Pi
codemode script). A second thread is the operator, through the run's own
inbox (``play_session``): an instruction the agent reports, one it keeps
working on while the operator talks to it (the message joins the running
instruction and the agent keeps moving) until ``stop_instruction`` (Esc in
the terminal), one that uses the
shell, one that moves the robot with ``ora-robot`` from the shell, one that
moves it from a codemode script, a scene reset, a last instruction, quit.

Each instruction is one pair of events and ends the way it should; the stopped
one ends at a boundary (every request answered); the shell sees no credential;
each script's robot call goes through the Supervisor with its origin (the
shell's client, or the codemode call that made it) and each action has its
receipt;
the operator view (Synthetic's drawn front view) is stored for every
observation and no request, agent event or agent artifact carries it;
every model call records the provider and model it used and carries the
profile's request_options, on the wire too; the policy reads the
play framing, never a task contract or verdict; the session completes,
evaluates valid (its trace complete), seals, and is unscored and never
learnable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes, play_session
from tests.support.agent_core.play_session import wait

pytestmark = pytest.mark.usefixtures("pi_core")

ROOT = Path(__file__).resolve().parents[2]
REQUEST_OPTIONS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "provider": {"require_parameters": True}}


def _operator(root: Path) -> None:
    def send(command_id: str, kind: str, message: str) -> None:
        play_session.send(root, command_id, kind, message)

    def finished(count: int, what: str) -> None:
        wait(lambda: play_session.count(root, "play_instruction_finished") == count, what)

    wait(lambda: play_session.phase(root) == "awaiting_instruction", "the first scene")
    send("i1", "instruction", "raise the left arm a little")
    finished(1, "the first report")
    send("i2", "instruction", "keep moving the left arm back and forth")
    wait(lambda: play_session.count(root, "motion_receipt") >= 4, "the second instruction's moves")
    # Talking to the agent while it works: the message joins the running instruction, which goes on.
    send("m1", "instruction", "keep at it -- how far along are you?")
    wait(lambda: any(event["type"] == "operator_message" and event["payload"]["id"] == "m1"
                     for event in play_session.events(root)), "the message delivered into the instruction")
    moved = play_session.count(root, "motion_receipt")
    wait(lambda: play_session.count(root, "motion_receipt") >= moved + 2, "moves after the message")
    send("s1", "stop_instruction", "Esc")
    finished(2, "the stop")
    send("b1", "instruction", "use bash to look at the workspace")
    finished(3, "the shell instruction")
    send("o1", "instruction", "move the left arm with ora-robot from the shell")
    finished(4, "the ora-robot instruction")
    send("c1", "instruction", "use codemode to look and move the left arm")
    finished(5, "the codemode instruction")
    send("r1", "reset", "/reset")
    wait(lambda: play_session.count(root, "play_scene_reset") == 1
         and play_session.phase(root) == "awaiting_instruction", "the second scene")
    send("i3", "instruction", "lower the left arm")
    finished(6, "the last report")
    send("q1", "quit", "/quit")


def test_a_free_play_session_runs_instructions_in_turn_and_seals_valid(tmp_path: Path, monkeypatch) -> None:
    from ora.orchestrator import paths
    from ora.orchestrator.evaluation.evaluate import evaluate, seal
    from ora.app.play import PLAY_TASK
    from ora.orchestrator.spec.quickstart import environment_policy

    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / "state")
    port_file = tmp_path / "port"
    received = tmp_path / "requests.jsonl"
    server = subprocess.Popen([sys.executable, "-m", "tests.support.agent_core.fake_model_server", str(port_file),
                               "--play", "--delay", "0.05", "--requests", str(received)], cwd=ROOT)
    try:
        wait(lambda: port_file.is_file() and port_file.read_text().strip() != "", "the scripted model", 30)
        (tmp_path / "pi-agent").mkdir()
        # A second configured model: Pi's default is not ORA's, so the session starts by switching to it.
        (tmp_path / "pi-agent" / "models.json").write_text(json.dumps({"providers": {"demo-second": {
            "baseUrl": f"http://127.0.0.1:{port_file.read_text().strip()}/v1", "api": "openai-completions",
            "apiKey": "local", "models": [{"id": "fake-second"}]}}}))
        with episodes.environment(PI_CODING_AGENT_DIR=str(tmp_path / "pi-agent"), PI_OFFLINE="1",
                                  ORA_CONFIG_HOME=str(tmp_path / "ora-config"),
                                  ORA_AGENT_CORE="pi", ORA_PROVIDER="openai_compatible", ORA_MODEL="fake-play",
                                  ORA_BASE_URL=f"http://127.0.0.1:{port_file.read_text().strip()}/v1",
                                  ORA_API_KEY="fake-key", ORA_SUPPORTS_IMAGES="1"):
            root = tmp_path / "play-run"
            # A profile's request_options, as the Muse Spark Open and Nemotron profiles carry them.
            plan = play_session.play_plan({**environment_policy(), "request_options": REQUEST_OPTIONS})
            assert plan["policy"]["request_options"] == REQUEST_OPTIONS
            result, failures = play_session.run_play(root, plan, _operator)
    finally:
        server.kill()
        server.wait()
    assert not failures, failures
    assert episodes.ran_core(root) == "pi"

    assert result["state"] == "completed" and result["termination"] == "operator_quit", result
    assert result["official_success"] is None and result["verdict"] is None
    assert result["behavioral_sample_valid"] is False and result["human_intervention"] is True
    play = result["play"]
    assert [item["ended"] for item in play["instructions"]] == ["reported", "operator_stopped"] + ["reported"] * 4
    assert [item["instruction_id"] for item in play["instructions"]] == ["i1", "i2", "b1", "o1", "c1", "i3"]
    assert [item["joined_messages"] for item in play["instructions"]] == [[], ["m1"], [], [], [], []]
    assert play["delivered_instruction_count"] == 6
    assert [item["scene_index"] for item in play["instructions"]] == [0, 0, 0, 0, 0, 1]
    assert play["instructions"][0]["decisions"] == 2 and play["instructions"][1]["decisions"] >= 2
    # A script's action is a decision of its instruction, from the shell (ora-robot) or a codemode script.
    assert [play["instructions"][index]["decisions"] for index in (3, 4)] == [1, 1]
    assert [scene["seed"] for scene in play["scenes"]] == [0, 0]  # Synthetic's one world, reloaded
    assert play["control_count"] == 3  # stop, reset, quit

    events = play_session.events(root)
    kinds = [event["type"] for event in events]
    assert kinds.count("play_instruction_started") == kinds.count("play_instruction_finished") == 6
    messages = [event["payload"] for event in events if event["type"] == "operator_message"]
    assert [(item["id"], item["type"]) for item in messages] == [
        (identity, "instruction") for identity in ("i1", "i2", "m1", "b1", "o1", "c1", "i3")]
    assert kinds.count("script_action") == 2
    assert {event["record"]["command"]: event["record"]["status"] for event in events
            if event["type"] == "play_control"} == {"stop_instruction": "applied", "reset": "applied", "quit": "applied"}
    assert not (root / "native/official-verdict.json").exists()
    # The operator saw each scene before its first instruction (a display copy the agent process may read).
    assert sorted(path.name for path in (root / "native/play-scenes").iterdir()) == ["scene-0000.png", "scene-0001.png"]

    agent = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    assert "official_verdict" not in {event["type"] for event in agent}
    request_events = [event for event in agent if event["type"] == "provider_request"]
    requests = [(root / event["artifact_ref"]["path"]).read_text() for event in request_events]
    assert sum(event["type"] == "provider_response" for event in agent) == len(requests), \
        "the stopped instruction ended at a boundary: every request has its response"
    assert {(event["payload"]["provider"], event["payload"]["model"]) for event in request_events} == {("ora", "fake-play")}
    first = json.loads(requests[0])
    assert first["messages"][0]["role"] == "system" and "store" not in first and "max_tokens" in first, \
        "the ORA profile's request shape (system message, no store, max_tokens)"
    # The profile's request_options go with every model call, as in an episode: in the recorded
    # requests and in every request the scripted model received.
    assert all({key: json.loads(request).get(key) for key in REQUEST_OPTIONS} == REQUEST_OPTIONS
               for request in requests)
    wire = [json.loads(line) for line in received.read_text().splitlines()]
    assert len(wire) >= len(requests) and all(item["options"] == REQUEST_OPTIONS for item in wire)
    assert all(PLAY_TASK in request for request in requests)
    assert all("no benchmark task" in request for request in requests), "the play episode-task layer"
    assert all("The operator may write to you while you work" in request for request in requests), \
        "the play episode-task layer says what a message during an instruction is"
    assert any("how far along are you?" in request for request in requests), "the joined message reached the model"
    # It joined instruction 2 (recorded with its delivery), reached Pi as an `ora.play_message` the
    # terminal and the page mark as joined, and the agent answered it while it kept working.
    (joined,) = [item for item in messages if item["id"] == "m1"]
    assert (joined["joined"], joined["instruction_id"]) == (True, "i2")
    assert all(item["joined"] is False and item["instruction_id"] == item["id"] for item in messages if item["id"] != "m1")
    entries = [json.loads(line) for path in (root / "native/pi-session").glob("*.jsonl")
               for line in path.read_text().splitlines()]
    (custom,) = [entry for entry in entries if entry.get("type") == "custom_message"
                 and entry.get("customType") == "ora.play_message"]
    assert custom["details"]["ora_message_id"] == "m1" and custom["details"]["ora_instruction"] == 2
    assert custom["details"]["ora_observation_step"] == joined["observation_step"]
    answers = [(root / event["artifact_ref"]["path"]).read_text() for event in agent if event["type"] == "provider_response"]
    assert any("You asked" in answer and "how far along are you?" in answer for answer in answers)
    assert "fake-key" not in "".join(requests)

    # The operator view: Synthetic's front view stored for every observation, for the operator only.
    # No provider request, agent event or agent artifact carries its pixels, hash, camera or path.
    views = [event for event in events if event["type"] == "operator_view_frame"]
    assert len(views) == kinds.count("observation") > 0
    assert {view["camera"] for view in views} == {"operator_front_view"}
    marks = {"operator-view", "operator_view", "operator_front_view"}
    for view in views:
        data = (root / view["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == view["sha256"]
        encoded = base64.b64encode(data).decode()
        marks |= {view["sha256"], encoded[64:192], encoded[-128:-8]}
    assert not [mark for mark in marks for request in requests if mark in request], "an operator view reached a request"
    trace = [(root / "agent-events.jsonl").read_text(), *(path.read_text(errors="replace")
             for path in (root / "native/agent-artifacts").iterdir() if path.is_file())]
    assert not [mark for mark in marks for text in trace if mark in text], "an operator view reached the agent trace"

    shell =[event for event in agent if event["type"] == "tool_result" and event["payload"].get("name") == "bash"]
    assert len(shell) == 2
    output = (root / shell[0]["artifact_ref"]["path"]).read_text()
    assert "key visible: no" in output and "credential variables: 0" in output
    assert "moved: True" in (root / shell[1]["artifact_ref"]["path"]).read_text(), "ora-robot moved the arm"

    # Every robot call a script made: through the Supervisor, with its origin (the shell's client, or
    # the codemode call whose script made it), each action with its receipt.
    def scripted(kind: str) -> list[dict[str, Any]]:
        return [event for event in agent if event["type"] == "tool_call_proposed"
                and (event["payload"].get("origin") or {}).get("kind") == kind]
    assert [event["payload"]["name"] for event in scripted("script")] == ["observation.dump_state",
                                                                            "robot.submit_action"]
    assert scripted("script")[0]["payload"]["origin"]["client"] == "ora-robot"
    (codemode,) = [event for event in agent if event["type"] == "tool_call_proposed"
                   and event["payload"].get("name") == "codemode"]
    from_script = scripted("codemode")
    assert [event["payload"]["name"] for event in from_script] == ["observation.dump_state", "robot.submit_action"]
    assert {event["payload"]["origin"]["parent_tool_call_id"] for event in from_script} == {codemode["tool_call_id"]}
    (script_result,) = [event for event in agent if event["type"] == "tool_result"
                        and event.get("tool_call_id") == codemode["tool_call_id"]]
    said = "".join((root / script_result["artifact_ref"]["path"]).read_text().replace("\\", "").split())
    assert '"looked":true' in said and '"moved":true' in said, said
    script_actions = [event for event in agent if event["type"] == "action_proposed"
                      and (event["payload"].get("origin") or {}).get("kind") in {"script", "codemode"}]
    assert len(script_actions) == 2 and all(any(
        receipt["type"] == "environment_receipt" and receipt.get("causal_parent_id") == action["event_id"]
        for receipt in agent) for action in script_actions)

    evaluation = evaluate(root)
    assert evaluation["status"] == "valid", evaluation["reasons"]
    assert evaluation["official_success"] is None
    assert evaluation["learning_eligibility"]["eligible"] is False
    assert "free_play_session" in evaluation["learning_eligibility"]["reasons"]
    assert seal(root)
    assert json.loads((root / "seal.json").read_text())["episode_mode"] == "play", "sealed with the play label"
