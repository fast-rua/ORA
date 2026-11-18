"""A free-play Synthetic session whose script drives the robot through ``ORA_ROBOT_SOCKET``, on the Pi core.

The plan is ``ora play``'s own (play_scenes' Synthetic world, ``episode_mode:
"play"``; ``play_session``), and the operator sends one instruction, then
quits. A loopback Anthropic server holds the first model request while a
script -- the Python client and the ``ora-robot`` shim the run installs --
loops over read-only and world-changing tools, as a script inside the agent's
bash call would while the turn waits. The model then calls a world-changing
tool itself (refused: the world changed under its turn) and reports. Every
script call is in the agent trace with its origin, each action has its receipt
and fresh observation, the Supervisor counted both as decisions of the
instruction and the evaluation is valid. A free-play surface has no lifecycle
tool.

A second session runs the script from the agent's own shell: Pi's agent,
against a scripted Chat Completions model, runs one ``bash`` command (an
``ora-robot`` loop and a Python script on ``ora.robot_client``), then moves the
robot itself (refused) and reports; every script call lands inside that bash
call with its origin, and the session evaluates valid and seals. A benchmark
run opens no socket.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from tests.support.agent_core import episodes, play_session
from tests.support.agent_core.fake_model_server import latest_instruction, tool_named

pytestmark = pytest.mark.usefixtures("pi_core")

API_KEY = "sk-ant-local-robot-socket-test-abcdefghij"
FINISH = {"outcome": "failure", "reason": "robot socket probe", "attempts": ["scripted"]}
LEVEL = [0, 0, 0, 1]
#: The model's own world-changing call (refused once a script has moved the world).
MOVE = {"arm": "left", "pose": {"position_m": [0, 0, 1.02], "orientation_xyzw": LEVEL}}


def sse(event: dict[str, Any]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode()


def report(index: int, text: str) -> bytes:
    """A reply with no tool call: the agent's report, which ends the instruction."""
    events = [
        {"type": "message_start", "message": {"id": f"msg_{index:03d}", "type": "message", "role": "assistant",
                                              "content": [], "usage": {"input_tokens": 3000 + 700 * index}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]
    return b"".join(sse(event) for event in events)


def reply(index: int, name: str, arguments: dict[str, Any]) -> bytes:
    text = json.dumps(arguments)
    events = [
        {"type": "message_start", "message": {"id": f"msg_{index:03d}", "type": "message", "role": "assistant",
                                              "content": [], "usage": {"input_tokens": 3000 + 700 * index}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": f"call {name}."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": f"EqQBsig{index:03d}AAAA"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": f"toolu_{index:03d}",
                                                                       "name": name, "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": text}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 40}},
        {"type": "message_stop"},
    ]
    return b"".join(sse(event) for event in events)


@pytest.fixture
def model() -> Iterator[dict[str, Any]]:
    """A loopback Anthropic server; ``state["before_first_reply"]`` runs before reply 0 is sent."""

    # The replies in turn (the last repeats): a tool call, or None for a plain report. A free-play
    # turn's model moves itself once, then reports; a test sets its own.
    state: dict[str, Any] = {"calls": 0, "before_first_reply": None, "errors": [],
                             "script": [("robot_submit_action", MOVE), None]}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("content-length", "0")))
            with lock:
                index = state["calls"]
                state["calls"] += 1
            hook: Callable[[], None] | None = state["before_first_reply"] if index == 0 else None
            if hook is not None:
                try:
                    hook()
                except BaseException as exc:  # reported by the test, the reply still goes out
                    state["errors"].append(exc)
            script = state["script"]
            step = script[min(index, len(script) - 1)]
            payload = reply(index, *step) if step is not None else report(index, "Done: the script moved the arm.")
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state["port"] = httpd.server_address[1]
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def configure_for(port: int) -> Callable[[dict[str, Any]], dict[str, str]]:
    def configure(plan: dict[str, Any]) -> dict[str, str]:
        endpoint = f"http://127.0.0.1:{port}/v1"
        policy = plan["policy"]
        policy.update({"endpoint": endpoint, "require_live_marker": False})
        return {policy["base_url_env"]: endpoint, policy["api_key_env"]: API_KEY}
    return configure


def run_script(run_root: Path, out: Path, seen: dict[str, Any]) -> None:
    """What an agent's bash command would run: the Python client, then the ``ora-robot`` shim."""

    from ora.robot_client import Robot

    pointer = json.loads((run_root / "native" / "robot-socket.json").read_text(encoding="utf-8"))
    socket_path = pointer["socket"]
    seen["socket"] = socket_path
    seen["mode"] = os.stat(socket_path).st_mode & 0o777
    with Robot(socket_path, timeout=120) as robot:
        seen["tools"] = robot.tools()
        seen["status"] = robot.status()
        seen["first"] = robot.observe()
        seen["loop"] = []
        # The second move spells the tool as the model's wire shows it (robot_submit_action).
        for position, name in (([0.03, 0, 1.04], "robot.submit_action"), ([0, 0, 1.02], "robot_submit_action")):
            looked = robot.call("observation.dump_state")
            moved = robot.call(name, arm="left", pose={"position_m": position, "orientation_xyzw": LEVEL})
            seen["loop"].append((looked, moved))
        seen["unknown"] = robot.call("fly_away")
    shim = run_root / "native" / "robot-client" / "bin" / "ora-robot"
    # -S in the shim: only the standard library and the run's client copy are importable.
    environment = {"PATH": os.environ["PATH"], "ORA_ROBOT_SOCKET": socket_path, "ORA_ROBOT_PYTHON": sys.executable}
    seen["cli_observe"] = subprocess.run([str(shim), "observe", "--save", str(out / "view.png")], env=environment,
                                         capture_output=True, text=True, timeout=120, check=False)
    seen["cli_call"] = subprocess.run([str(shim), "call", "observation.dump_state"], env=environment,
                                      capture_output=True, text=True, timeout=120, check=False)


def artifact(root: Path, event: dict[str, Any]) -> Any:
    return json.loads((root / event["artifact_ref"]["path"]).read_text(encoding="utf-8"))


def _operator(root: Path) -> None:
    play_session.wait(lambda: play_session.phase(root) == "awaiting_instruction", "the scene")
    play_session.send(root, "i1", "instruction", "move the left arm a little")
    play_session.wait(lambda: play_session.count(root, "play_instruction_finished") == 1, "the report")
    play_session.send(root, "q1", "quit", "/quit")


def test_a_play_script_drives_the_robot_through_the_supervisor(tmp_path: Path, model: dict[str, Any],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from ora.orchestrator import paths
    from ora.orchestrator.evaluation.evaluate import evaluate
    from ora.orchestrator.spec.quickstart import environment_policy

    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / "state")
    seen: dict[str, Any] = {}
    root = tmp_path / "play-run"
    model["before_first_reply"] = lambda: run_script(root, tmp_path, seen)
    with episodes.environment(PI_OFFLINE="1", PI_CODING_AGENT_DIR=str(tmp_path / "pi-agent"),
                              ORA_CONFIG_HOME=str(tmp_path / "ora-config"), ORA_AGENT_CORE="pi",
                              ORA_PROVIDER="anthropic_messages", ORA_MODEL="fake-claude",
                              ORA_BASE_URL=f"http://127.0.0.1:{model['port']}", ORA_API_KEY=API_KEY):
        result, failures = play_session.run_play(root, play_session.play_plan(environment_policy()), _operator)
    if model["errors"]:
        raise model["errors"][0]
    assert not failures, failures
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    assert episodes.ran_core(root) == "pi"
    assert result["state"] == "completed" and result["termination"] == "operator_quit", result
    evaluation = evaluate(root)
    assert evaluation["status"] == "valid", evaluation["reasons"]
    assert result["decisions"] == 2
    (instruction,) = result["play"]["instructions"]
    assert instruction["ended"] == "reported" and instruction["decisions"] == 2

    # What the script saw: the free-play surface, without a lifecycle tool or the task checker.
    assert seen["mode"] == 0o600
    effects = {tool["name"]: tool["effect"] for tool in seen["tools"]}
    assert effects["robot.submit_action"] == "mutation" and effects["observation.dump_state"] == "read_only"
    assert not {"finish_episode", "check_task"} & set(effects) and "parameters" in seen["tools"][0]
    assert {tool["name"]: tool["wire_name"] for tool in seen["tools"]}["robot.submit_action"] == "robot_submit_action"
    assert seen["status"]["turn_active"] is True and seen["status"]["episode_mode"] == "play"
    step = seen["first"]["observation_step"]
    assert seen["first"]["images"]
    for index, (looked, moved) in enumerate(seen["loop"]):
        assert looked["ok"] and looked["effect"] == "read_only"
        assert looked["observation_step"] == step + index
        assert moved["ok"] and moved["effect"] == "mutation", moved
        # robot.submit_action's composite receipt: the pose it reached, and the action it is for.
        assert moved["receipt"]["action_id"] == moved["action"]["action_id"] and moved["receipt"]["pose_reached"]
        assert moved["observation"]["observation_step"] == step + index + 1
    assert seen["unknown"]["error"]["code"] == "unknown_tool"
    assert seen["cli_observe"].returncode == 0, seen["cli_observe"].stderr
    assert (tmp_path / "view.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert seen["cli_call"].returncode == 0, seen["cli_call"].stderr
    assert json.loads(seen["cli_call"].stdout)["ok"] is True

    # The agent trace: every script call, its result, its actions and their receipts.
    def scripted(event: dict[str, Any]) -> bool:
        payload = event.get("payload")
        return isinstance(payload, dict) and (payload.get("origin") or {}).get("kind") == "script"

    proposals = [event for event in events if event["type"] == "tool_call_proposed" and scripted(event)]
    assert [event["payload"]["name"] for event in proposals] == [
        "observation.dump_state", "robot.submit_action", "observation.dump_state", "robot.submit_action",
        "fly_away", "observation.dump_state"]
    assert all(event["model_call_id"] == event["tool_call_id"] and event["tool_call_id"].startswith("script-call-")
               for event in proposals)
    results = {event["tool_call_id"]: event for event in events if event["type"] == "tool_result" and scripted(event)}
    assert set(results) == {event["tool_call_id"] for event in proposals}
    assert [results[event["tool_call_id"]]["payload"]["ok"] for event in proposals] == [
        True, True, True, True, False, True]
    python_origin = proposals[0]["payload"]["origin"]
    assert python_origin["client"] == "python" and python_origin["command"] and python_origin["peer_pid"] == os.getpid()
    cli_origin = proposals[-1]["payload"]["origin"]
    assert cli_origin["client"] == "ora-robot" and cli_origin["command"].startswith("ora-robot call observation")
    actions = [event for event in events if event["type"] == "action_proposed"]
    assert len(actions) == 2 and all(scripted(event) for event in actions)
    for action in actions:
        position = events.index(action)
        receipt = next(event for event in events[position:] if event["type"] == "environment_receipt")
        fresh = next(event for event in events[position:] if event["type"] == "fresh_observation_published")
        assert receipt["causal_parent_id"] == action["event_id"]
        assert receipt["payload"]["action_id"] == action["payload"]["action_id"]
        assert fresh["observation_step"] == action["observation_step"] + 1

    # The model's own world-changing call after the script's actions, refused.
    (direct,) = [event for event in events if event["type"] == "tool_call_proposed" and not scripted(event)
                 and event["payload"]["name"] == "robot.submit_action"]
    (refused,) = [event for event in events if event["type"] == "tool_result"
                  and event["tool_call_id"] == direct["tool_call_id"]]
    assert artifact(root, refused)["error"]["code"] == "world_changed_by_script"

    # The Supervisor's events: two script decisions, each executed and observed once.
    outer = play_session.events(root)
    types = [event["type"] for event in outer]
    assert types.count("script_action") == types.count("action_submitted") == types.count("motion_receipt") == 2
    assert not Path(seen["socket"]).exists()
    logged = [json.loads(line) for line in (root / "native" / "robot-socket.jsonl").read_text().splitlines()]
    assert {record["method"] for record in logged} == {"tools.list", "status", "observe", "tools.call"}


# -- The agent's own shell -----------------------------------------------------------------
# A free-play agent scripts the robot from its own bash tool: Pi's agent against a scripted
# Chat Completions model runs one bash command, an ora-robot loop and a Python script on
# ora.robot_client, then moves the robot itself (refused) and reports.

INSTRUCTION = "script the left arm through two small steps"
SHELL_SCRIPT = r"""set -e
ora-robot tools > tools.txt
for x in 0.02 0.04; do
  ora-robot call observation.dump_state > /dev/null
  ora-robot call robot_submit_action --json "{\"arm\": \"left\", \"pose\": {\"position_m\": [$x, 0, 1.02], \"orientation_xyzw\": [0, 0, 0, 1]}}" > "move-$x.json"
done
cat > drive.py <<'PY'
from ora.robot_client import Robot

robot = Robot()
print("turn active:", robot.status()["turn_active"])
print("observation step:", robot.observe(save="view.png")["observation_step"])
print("finish_episode:", robot.call("finish_episode")["error"]["code"])
PY
python3 drive.py
"""


def sse_reply(call: int, request: dict[str, Any], text: str, name: str | None, arguments: dict[str, Any]) -> bytes:
    """One streamed Chat Completions reply (Pi's providers stream)."""

    deltas: list[tuple[dict[str, Any], str | None]] = [({"role": "assistant", "content": text}, None)]
    if name is not None:
        deltas.append(({"tool_calls": [{"index": 0, "id": f"call-{call}", "type": "function",
                                        "function": {"name": name, "arguments": json.dumps(arguments)}}]}, None))
    deltas.append(({}, "tool_calls" if name else "stop"))
    chunks = []
    for delta, reason in deltas:
        chunk = {"id": f"fake-{call}", "object": "chat.completion.chunk", "model": request.get("model"),
                 "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
        if reason is not None:
            chunk["usage"] = {"prompt_tokens": 1500 + 200 * call, "completion_tokens": 40,
                              "total_tokens": 1540 + 200 * call}
        chunks.append(f"data: {json.dumps(chunk)}\n\n".encode())
    return b"".join(chunks) + b"data: [DONE]\n\n"


def play_model_reply(request: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    """For the latest instruction: the bash script, then a direct move (refused), then the report."""

    if not request.get("tools"):
        return "Summary: the agent scripted the left arm.", None, {}
    instruction, moves, other = latest_instruction(request)
    if instruction is None:
        return "No instruction yet.", None, {}
    if other == 0:
        return "Scripting the steps in the shell.", "bash", {"command": SHELL_SCRIPT}
    if moves == 0:
        return "One more step myself.", tool_named(request, "robot.submit_action"), {
            "arm": "left", "pose": {"position_m": [0.06, 0, 1.02], "orientation_xyzw": [0, 0, 0, 1]}}
    return f"Done with \"{instruction}\": the script moved the left arm twice.", None, {}


@pytest.fixture
def play_model() -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {"calls": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["calls"] += 1
                call = state["calls"]
            payload = sse_reply(call, request, *play_model_reply(request))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _instruct(root: Path) -> None:
    play_session.wait(lambda: play_session.phase(root) == "awaiting_instruction", "the scene", timeout=120.0)
    play_session.send(root, "i1", "instruction", INSTRUCTION)
    play_session.wait(lambda: play_session.count(root, "play_instruction_finished") == 1, "the report", timeout=120.0)
    play_session.send(root, "q1", "quit", "/quit")


def _scripted(event: dict[str, Any]) -> bool:
    payload = event.get("payload")
    return isinstance(payload, dict) and (payload.get("origin") or {}).get("kind") == "script"


def test_a_play_agent_scripts_the_robot_from_its_shell_through_the_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, play_model: dict[str, Any],
) -> None:
    from ora.orchestrator import paths
    from ora.orchestrator.evaluation.evaluate import evaluate, seal
    from ora.orchestrator.spec.quickstart import environment_policy

    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / "state")
    (tmp_path / "pi-agent").mkdir()
    root = tmp_path / "play-run"
    with episodes.environment(PI_CODING_AGENT_DIR=str(tmp_path / "pi-agent"), PI_OFFLINE="1",
                              ORA_CONFIG_HOME=str(tmp_path / "ora-config"), ORA_AGENT_CORE="pi",
                              ORA_PROVIDER="openai_compatible", ORA_MODEL="fake-play",
                              ORA_BASE_URL=play_model["base_url"], ORA_API_KEY="fake-key", ORA_SUPPORTS_IMAGES="1"):
        result, failures = play_session.run_play(root, play_session.play_plan(environment_policy()), _instruct)
    assert not failures, failures
    assert result["state"] == "completed" and result["termination"] == "operator_quit", result
    (instruction,) = result["play"]["instructions"]
    assert instruction["ended"] == "reported" and instruction["decisions"] == 2, instruction

    # What the agent's shell saw.
    workspace = root / "sandbox" / "workspace"
    listed = {line.split()[0]: line.split()[1] for line in (workspace / "tools.txt").read_text().splitlines()}
    assert listed["robot.submit_action"] == "mutation" and listed["observation.dump_state"] == "read_only"
    moves = [json.loads((workspace / f"move-{x}.json").read_text()) for x in ("0.02", "0.04")]
    for move in moves:
        assert move["ok"] and move["tool"] == "robot.submit_action" and move["effect"] == "mutation", move
        assert move["action"]["action_id"] in json.dumps(move["receipt"])  # the receipt's shape is the surface's
        assert move["observation"]["observation_step"] == move["observation_step"] + 1
    assert moves[1]["observation_step"] == moves[0]["observation_step"] + 1
    assert (workspace / "view.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    agent = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    (shell,) = [event for event in agent if event["type"] == "tool_result" and event["payload"].get("name") == "bash"]
    output = (root / shell["artifact_ref"]["path"]).read_text()
    assert "turn active: True" in output and "finish_episode: unknown_tool" in output, output
    assert f"observation step: {moves[1]['observation_step'] + 1}" in output, output

    # The agent trace: every script call with its origin, its actions and their receipts.
    proposals = [event for event in agent if event["type"] == "tool_call_proposed" and _scripted(event)]
    assert [event["payload"]["name"] for event in proposals] == [
        "observation.dump_state", "robot.submit_action", "observation.dump_state", "robot.submit_action",
        "finish_episode"]
    assert all(event["model_call_id"] == event["tool_call_id"] and event["tool_call_id"].startswith("script-call-")
               for event in proposals)
    results = {event["tool_call_id"]: event for event in agent if event["type"] == "tool_result" and _scripted(event)}
    assert [results[event["tool_call_id"]]["payload"]["ok"] for event in proposals] == [True, True, True, True, False]
    shell_origin = proposals[1]["payload"]["origin"]
    assert shell_origin["client"] == "ora-robot"
    assert shell_origin["command"].startswith("ora-robot call robot_submit_action")
    assert "ora-robot call" in shell_origin["parent_command"], shell_origin  # the agent's bash command
    script_origin = proposals[-1]["payload"]["origin"]
    assert script_origin["client"] == "python" and script_origin["script"] == str(workspace / "drive.py")
    actions = [event for event in agent if event["type"] == "action_proposed" and _scripted(event)]
    assert len(actions) == 2
    for action in actions:
        after = agent[agent.index(action):]
        receipt = next(event for event in after if event["type"] == "environment_receipt")
        fresh = next(event for event in after if event["type"] == "fresh_observation_published")
        assert receipt["causal_parent_id"] == action["event_id"]
        assert receipt["payload"]["action_id"] == action["payload"]["action_id"]
        assert fresh["observation_step"] == action["observation_step"] + 1
    assert all(agent.index(proposal) < agent.index(shell) for proposal in proposals), "inside the bash call"

    # The model's own world-changing call after the script's actions, refused.
    (direct,) = [event for event in agent if event["type"] == "tool_call_proposed" and not _scripted(event)
                 and event["payload"]["name"] == "robot.submit_action"]
    (refused,) = [event for event in agent if event["type"] == "tool_result"
                  and event["tool_call_id"] == direct["tool_call_id"]]
    assert json.loads((root / refused["artifact_ref"]["path"]).read_text())["error"]["code"] == "world_changed_by_script"

    # The Supervisor: two script decisions, each executed and observed once.
    kinds = [event["type"] for event in play_session.events(root)]
    assert kinds.count("script_action") == kinds.count("action_submitted") == kinds.count("motion_receipt") == 2
    pointer = json.loads((root / "native" / "robot-socket.json").read_text())
    assert not Path(pointer["socket"]).exists(), "the socket closes with the session"
    evaluation = evaluate(root)
    assert evaluation["status"] == "valid", evaluation["reasons"]
    assert seal(root)


def test_a_benchmark_run_opens_no_robot_socket(tmp_path: Path, model: dict[str, Any]) -> None:
    model["script"] = [("close_gripper", {}), ("finish_episode", FINISH)]  # the benchmark surface
    run = episodes.run_core("pi", tmp_path, configure=configure_for(model["port"]), actions=1,
                            run_id="robot-socket-benchmark")
    assert run["evaluation"]["status"] == "valid", run["evaluation"]["reasons"]
    native = run["root"] / "native"
    assert not any((native / name).exists() for name in ("robot.sock", "robot-socket.json", "robot-socket.jsonl",
                                                          "robot-client"))
    assert not any("origin" in (event.get("payload") or {}) for event in run["events"]
                   if isinstance(event.get("payload"), dict))
    # The model's own world-changing call runs as always.
    assert run["result"]["decisions"] == 1
