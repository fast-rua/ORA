"""``ora play --code-mode`` on the Pi core: free play in ORA's code mode (plan ``play.code_mode``).

A Synthetic free-play session on its toolkit surface (``native_atomic_causal_v1``,
whose actions a script's parent executes; the play default's
``robot.submit_action`` is a composite the toolkit executor refuses) whose
policy names ``code_mode`` (Synthetic's baseline has none to default to)
against the scripted loopback model
(``fake_model_server --play``, its code-mode branch). The model's tools are
ORA's ``read``, ``write``, ``edit`` and ``bash`` and nothing of Pi's: no Pi
built-in tool, no ``codemode``, no direct robot tool. An instruction runs one
``bash`` command whose program closes and opens the gripper through the
``robot`` library: every call it makes is recorded under the command's call
with origin ``code_mode``, its action is executed inside that call by the
Supervisor (the call's ``ora.tool-execution.v1`` ledger), the act ends there
(``actions_executed``) and the instruction goes on with the next act on the new
observation, where the model reports. A slow program the operator stops (Esc)
is ended between its calls (``stopped``) and its instruction ends
``operator_stopped``; the session quits, evaluates valid and seals.

The command runs without bubblewrap here (``tests.support.agent_core.code_mode_shell``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.support.agent_core import episodes, play_session
from tests.support.agent_core.code_mode_shell import local_run_command
from tests.support.agent_core.play_session import wait

pytestmark = pytest.mark.usefixtures("pi_core")

ROOT = Path(__file__).resolve().parents[2]


def _toolkit() -> dict[str, Any]:
    """Synthetic's toolkit surface, without the task check and finishing (free play has neither)."""
    from ora.agent.composition import resolve_tool_surface

    surface = resolve_tool_surface("native_atomic_causal_v1")
    return {"tool_surface": {"profile": surface.profile_id, "component_overrides": {}},
            "tool_allowlist": [item["name"] for item in surface.definitions
                               if item["name"] not in {"check_task", "finish_episode"}],
            "action_codec": "runtime_action_v3", "context_profile": "native_causal_context_v1",
            "context_window_tokens": 262144}


TOOLKIT = _toolkit()


def _receipts(root: Path) -> int:
    path = root / "agent-events.jsonl"
    try:
        return sum('"environment_receipt"' in line for line in path.read_text().splitlines())
    except OSError:
        return 0


def _operator(root: Path) -> None:
    def finished(count: int, what: str) -> None:
        wait(lambda: play_session.count(root, "play_instruction_finished") == count, what)

    wait(lambda: play_session.phase(root) == "awaiting_instruction", "the first scene")
    play_session.send(root, "p1", "instruction", "move the left arm with a program")
    finished(1, "the program's report")
    before = _receipts(root)
    play_session.send(root, "p2", "instruction", "run a slow program on the left arm")
    # Its gripper closed: the program now waits a minute, until the operator's Esc.
    wait(lambda: play_session.count(root, "play_instruction_started") == 2 and _receipts(root) > before,
         "the slow program's action")
    play_session.send(root, "s1", "stop_instruction", "Esc")
    finished(2, "the stop")
    play_session.send(root, "q1", "quit", "/quit")


def test_free_play_in_code_mode_runs_programs_through_the_supervisor_and_stops_on_esc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ora.agent.pi.code_mode as session
    from ora.orchestrator import paths
    from ora.orchestrator.evaluation.evaluate import evaluate, seal
    from ora.orchestrator.spec.quickstart import environment_policy

    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(session, "run_command", local_run_command)
    port_file = tmp_path / "port"
    server = subprocess.Popen([sys.executable, "-m", "tests.support.agent_core.fake_model_server", str(port_file),
                               "--play", "--delay", "0.05"], cwd=ROOT)
    try:
        wait(lambda: port_file.is_file() and port_file.read_text().strip() != "", "the scripted model", 30)
        (tmp_path / "pi-agent").mkdir()
        port = port_file.read_text().strip()
        with episodes.environment(PI_CODING_AGENT_DIR=str(tmp_path / "pi-agent"), PI_OFFLINE="1",
                                  ORA_CONFIG_HOME=str(tmp_path / "ora-config"),
                                  ORA_AGENT_CORE="pi", ORA_PROVIDER="openai_compatible", ORA_MODEL="fake-play",
                                  ORA_BASE_URL=f"http://127.0.0.1:{port}/v1",
                                  ORA_API_KEY="fake-key", ORA_SUPPORTS_IMAGES="1"):
            # The toolkit surface moves one arm: the world's right one.
            plan = play_session.play_plan({**environment_policy(), **TOOLKIT, "code_mode": True}, code_mode=True,
                                          runtime={"controlled_end_effector": "right"})
            assert plan["play"]["code_mode"] is True and plan["policy"]["code_mode"]
            root = tmp_path / "play-run"
            result, failures = play_session.run_play(root, plan, _operator)
    finally:
        server.kill()
        server.wait()
    assert not failures, failures
    assert episodes.ran_core(root) == "pi"
    assert result["state"] == "completed" and result["termination"] == "operator_quit", result
    play = result["play"]
    assert [item["instruction_id"] for item in play["instructions"]] == ["p1", "p2"]
    assert [item["ended"] for item in play["instructions"]] == ["reported", "operator_stopped"]
    # A command whose program acted is one robot decision of its instruction.
    assert [item["decisions"] for item in play["instructions"]] == [1, 1]

    agent = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    requests = [json.loads((root / event["artifact_ref"]["path"]).read_text())
                for event in agent if event["type"] == "provider_request"]
    # ORA's coding tools are all the model has: none of Pi's own, no codemode, no direct robot tool.
    offered = {tuple(sorted(tool["function"]["name"] for tool in request.get("tools") or []))
               for request in requests if request.get("tools")}
    assert offered and all({"bash", "read", "write", "edit"} <= set(names) for names in offered)
    assert not {name for names in offered for name in names
                if name in {"codemode", "grep", "find", "ls", "observe", "close_gripper", "move_pose"}}
    system = requests[0]["messages"][0]["content"]
    assert "Free play with an operator" in (system if isinstance(system, str) else json.dumps(system)), \
        "the free-play code-mode layer"

    commands = [event for event in agent if event["type"] == "tool_call_proposed"
                and event["payload"].get("name") == "bash" and not event["payload"].get("origin")]
    assert len(commands) == 2
    for command, tools in zip(commands, (["close_gripper", "open_gripper"], ["close_gripper"])):
        nested = [event for event in agent if event["type"] == "tool_call_proposed"
                  and (event["payload"].get("origin") or {}).get("kind") == "code_mode"
                  and event["payload"]["origin"]["parent_tool_call_id"] == command["tool_call_id"]]
        assert [event["payload"]["name"] for event in nested] == tools
        actions = [event for event in agent if event["type"] == "action_proposed"
                   and event["tool_call_id"] == command["tool_call_id"]]
        assert len(actions) == len(tools), "the program's actions belong to the command's call"
        (outcome,) = [event for event in agent if event["type"] == "tool_result"
                      and event["tool_call_id"] == command["tool_call_id"]]
        # The Supervisor's ledger of the call, recorded with its result as in an episode.
        assert outcome["payload"]["execution"]["native_actions_used"] >= len(tools)

    def result_of(command: dict[str, Any]) -> dict[str, Any]:
        (event,) = [item for item in agent if item["type"] == "tool_result"
                    and item["tool_call_id"] == command["tool_call_id"]]
        return json.loads((root / event["artifact_ref"]["path"]).read_text())

    first, slow = (result_of(command) for command in commands)
    assert first["status"] == "completed" and "program moved:" in first["output"]
    assert slow["status"] == "stopped" and slow["actions_executed"] == 1
    assert "waiting" in slow["output"] and "woke up" not in slow["output"]
    assert slow["error_detail"]["code"] == "operator_instruction_stop"

    evaluation = evaluate(root)
    assert evaluation["status"] == "valid", evaluation["reasons"]
    assert evaluation["official_success"] is None
    assert seal(root)
