"""The free-play robot socket, its RPC multiplexing and its clients, without an agent core.

``RobotSocketServer`` answers reads from snapshots and refuses a call outside
a turn (untraced); its socket is 0600, moved to a private directory when the
run path is too long. ``ScriptAwareRpcPeer`` runs a queued script call on the
thread waiting on ``act`` -- holding the core's own requests until the call is
done and keeping an ``act`` response that arrives inside the call's nested
request -- and refuses calls once the turn ended. ``ora-robot`` parses its
arguments, saves images and reports outcomes through its exit status.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

import pytest

from ora.agent.pi.robot_socket import RobotSocketServer, ScriptAwareRpcPeer
from ora.robot_client import Robot, RobotError
from ora.robot_client.cli import main as ora_robot

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16


@pytest.fixture
def server(tmp_path: Path):
    native = tmp_path / "native"
    native.mkdir()
    robot = RobotSocketServer(native)
    try:
        yield robot
    finally:
        robot.close()


def client(robot: RobotSocketServer) -> Robot:
    return Robot(robot.path, timeout=30, origin={"client": "test"})


def test_reads_answer_from_snapshots_and_a_call_outside_a_turn_is_refused(server: RobotSocketServer) -> None:
    server.runner = lambda call: pytest.fail("a call ran outside a turn")
    server.publish(tools=[{"name": "observe", "effect": "read_only"}], status={"episode_mode": "play"},
                   observation={"observation_step": 3, "main_camera": "agentview", "images": []})
    assert os.stat(server.path).st_mode & 0o777 == 0o600
    pointer = json.loads((server.native / "robot-socket.json").read_text())
    assert pointer["socket"] == str(server.path) and pointer["schema"] == "ora.robot-socket.v1"
    with client(server) as robot:
        assert robot.tools() == [{"name": "observe", "effect": "read_only"}]
        assert robot.observe()["observation_step"] == 3
        status = robot.status()
        assert status["turn_active"] is False and status["episode_mode"] == "play" and status["observation_step"] == 3
        refused = robot.call("observe", views=["agentview"])
        assert refused["ok"] is False and refused["traced"] is False
        assert refused["error"]["code"] == "no_active_turn"
        with pytest.raises(RobotError) as unknown:
            robot.request("teleport")
        assert unknown.value.code == "method_not_found"
        with pytest.raises(RobotError) as malformed:
            robot.request("tools.call", {"name": 3})
        assert malformed.value.code == "bad_request"
    logged = [json.loads(line) for line in server.log_path.read_text().splitlines()]
    assert [record["outcome"] for record in logged] == [
        "ok", "ok", "ok", "refused:no_active_turn", "error:method_not_found", "error:bad_request"]


def test_a_long_run_path_binds_in_a_private_directory(tmp_path: Path) -> None:
    native = tmp_path / ("deep-" * 20) / "native"
    native.mkdir(parents=True)
    robot = RobotSocketServer(native)
    try:
        assert robot.path.parent != native and robot.path.parent.stat().st_mode & 0o777 == 0o700
        assert json.loads((native / "robot-socket.json").read_text())["socket"] == str(robot.path)
        with client(robot) as connection:
            assert connection.status()["socket"] == str(robot.path)
    finally:
        robot.close()
    assert not robot.path.parent.exists()


class FakeCore:
    """The TypeScript side of the RPC channel, scripted line by line."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.reader = sock.makefile("rb")

    def read(self) -> dict[str, Any]:
        return json.loads(self.reader.readline())

    def send(self, message: dict[str, Any]) -> None:
        self.sock.sendall((json.dumps(message) + "\n").encode())


def act_with_one_script_call(server: RobotSocketServer, core_script) -> dict[str, Any]:
    """``peer.call("act")`` on this thread while ``core_script`` plays the core and one client calls."""

    python_side, core_side = socket.socketpair()
    peer = ScriptAwareRpcPeer(python_side, server)
    seen: dict[str, Any] = {"order": []}

    def runner(call: Any) -> dict[str, Any]:
        seen["runner_thread"] = threading.current_thread()
        seen["record"] = peer.call("record", {"type": "tool_call_proposed"})
        seen["order"].append("script call done")
        return {"ok": True, "call_id": "script-call-000001"}

    def invoke(params: Any) -> dict[str, Any]:
        seen["order"].append("core request served")
        return {"served": True}

    server.runner = runner
    peer.on("tools.invoke", invoke)
    core = FakeCore(core_side)

    def call_from_script() -> None:
        with client(server) as robot:
            seen["outcome"] = robot.call("observe")

    def play() -> None:
        seen["act_request"] = core.read()
        script = threading.Thread(target=call_from_script)
        script.start()
        core_script(core, seen)
        script.join(timeout=30)

    playing = threading.Thread(target=play)
    playing.start()
    try:
        seen["act_result"] = peer.call("act", {"step": 0})
    finally:
        playing.join(timeout=30)
        peer.close()
        core_side.close()
    return seen


def test_a_script_call_runs_on_the_act_thread_and_holds_the_cores_requests(server: RobotSocketServer) -> None:
    def core_script(core: FakeCore, seen: dict[str, Any]) -> None:
        record = core.read()  # the script call's nested request
        assert record["method"] == "record"
        core.send({"id": "t1", "method": "tools.invoke", "params": {}})  # held until the call is done
        core.send({"id": record["id"], "result": {"event_id": "agent-event-000007"}})
        seen["invoke_answer"] = core.read()
        core.send({"id": seen["act_request"]["id"], "result": {"submitted": False}})

    seen = act_with_one_script_call(server, core_script)
    assert seen["act_result"] == {"submitted": False}
    assert seen["runner_thread"] is threading.main_thread()
    assert seen["record"] == {"event_id": "agent-event-000007"}
    assert seen["order"] == ["script call done", "core request served"]
    assert seen["invoke_answer"] == {"id": "t1", "result": {"served": True}}
    assert seen["outcome"] == {"ok": True, "call_id": "script-call-000001"}
    assert server.turn_open is False
    with client(server) as robot:
        assert robot.call("observe")["error"]["code"] == "no_active_turn"


def test_an_act_response_inside_a_script_calls_nested_request_is_kept_for_act(server: RobotSocketServer) -> None:
    def core_script(core: FakeCore, seen: dict[str, Any]) -> None:
        record = core.read()
        core.send({"id": seen["act_request"]["id"], "result": {"submitted": False, "termination": "done"}})
        core.send({"id": record["id"], "result": {"event_id": "agent-event-000003"}})

    seen = act_with_one_script_call(server, core_script)
    assert seen["record"] == {"event_id": "agent-event-000003"}
    assert seen["act_result"] == {"submitted": False, "termination": "done"}
    assert seen["outcome"]["ok"] is True


@pytest.fixture
def open_turn(server: RobotSocketServer):
    """A turn whose script calls a worker thread answers with ``server.answer(call)``."""

    stop = threading.Event()

    def work() -> None:
        while not stop.is_set():
            if server.has_calls():
                server.run_pending()
            stop.wait(0.01)

    server.runner = lambda call: server.answer(call)
    server.open_turn()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        stop.set()
        worker.join(timeout=5)


def test_ora_robot_parses_arguments_saves_images_and_reports_by_exit_status(
    open_turn: RobotSocketServer, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "rendered.png"
    image.write_bytes(PNG)
    calls: list[Any] = []

    def answer(call: Any) -> dict[str, Any]:
        calls.append(call)
        if call.name == "observe":
            return {"ok": True, "result": {"ok": True}, "images": [{"path": str(image)}]}
        return {"ok": False, "error": {"code": "out_of_bounds", "message": "position x is outside"}}

    open_turn.answer = answer
    socket_flag = ["--socket", str(open_turn.path)]
    assert ora_robot([*socket_flag, "call", "observe", "--json", '{"views": ["wrist"], "width": 1}',
                      'views=["agentview"]', "width=0.04", "label=red cup",
                      "--save", str(tmp_path / "view.png")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["saved"] == [str(tmp_path / "view.png")] and (tmp_path / "view.png").read_bytes() == PNG
    assert calls[0].arguments == {"views": ["agentview"], "width": 0.04, "label": "red cup"}
    assert calls[0].origin["client"] == "ora-robot"
    assert calls[0].origin["command"].startswith("ora-robot --socket")
    assert ora_robot([*socket_flag, "call", "move_pose", "--json", '{"arm": "right"}']) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "out_of_bounds"
    assert ora_robot(["--socket", str(tmp_path / "absent.sock"), "status"]) == 2
    assert "unreachable" in capsys.readouterr().err
    assert ora_robot([*socket_flag, "call", "observe", "not-a-pair"]) == 2
