"""``ora.robot_client``: the free-play robot client and ``ora-robot``, in process, against a scripted socket.

The client speaks ``ora.robot-socket.v1`` (the server's ``PROTOCOL`` in
``ora/agent/pi/robot_socket.py``): one JSON object per line, answered under the
same ``id`` with a ``result`` or an ``error``. Only a broken request or
connection raises ``RobotError`` (with its code); a refused or failed tool is
a ``result`` with ``ok: false``. ``ora-robot`` exits 0 when a call ran, 1 when
it was refused and 2 when the socket is unreachable or the request is
malformed. The Pi-side server is ``tests/agent_core/test_robot_socket_protocol.py``.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from ora.agent.pi import robot_socket
from ora.robot_client import PROTOCOL, SOCKET_ENV, Robot, RobotError
from ora.robot_client import client as robot_client
from ora.robot_client.cli import main as ora_robot

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 8
Handler = Callable[[dict[str, Any]], "dict[str, Any] | None"]


class ScriptedSocket:
    """A Unix socket that answers each request line with ``handler(request)``; ``None`` hangs up."""

    def __init__(self, path: Path, handler: Handler) -> None:
        self.path, self.handler, self.requests, self.connections = path, handler, [], []
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(path))
        self.server.listen()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            self.connections.append(connection)
            with connection, connection.makefile("rb") as reader:
                for line in reader:
                    request = json.loads(line)
                    self.requests.append(request)
                    answer = self.handler(request)
                    if answer is None:
                        break
                    connection.sendall((json.dumps({"id": request["id"], **answer}) + "\n").encode())

    def close(self) -> None:
        """Wake the serving thread (in ``accept`` or reading a client left open) and stop it."""

        for item in (*self.connections, self.server):
            try:
                item.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.server.close()
        self.thread.join(timeout=5)


@pytest.fixture
def socket_dir(tmp_path: Path) -> Iterator[Path]:
    """A directory short enough for a Unix socket path (about 100 bytes)."""

    if len(str(tmp_path)) < 80:
        yield tmp_path
        return
    with tempfile.TemporaryDirectory(prefix="ora-rc-") as short:
        yield Path(short)


@pytest.fixture
def serve(socket_dir: Path) -> Iterator[Callable[[Handler], ScriptedSocket]]:
    servers: list[ScriptedSocket] = []

    def start(handler: Handler) -> ScriptedSocket:
        servers.append(ScriptedSocket(socket_dir / f"robot-{len(servers)}.sock", handler))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def run_tool(view: dict[str, Any]) -> Handler:
    """The answers of a run: two tools, a refused move and an observation with one image."""

    def answer(request: dict[str, Any]) -> dict[str, Any]:
        method, params = request["method"], request["params"]
        if method == "tools.list":
            return {"result": {"protocol": PROTOCOL, "tools": [
                {"name": "observe", "effect": "read_only", "description": "Look.\nMore text."},
                {"name": "move_pose", "effect": "mutation", "description": ""}]}}
        if method == "tools.call" and params["name"] == "observe":
            return {"result": {"ok": True, "result": {"step": 3}, "images": view["images"]}}
        if method == "tools.call":
            return {"result": {"ok": False, "error": {"code": "out_of_bounds", "message": "x is outside"}}}
        if method == "observe":
            return {"result": view}
        if method == "status":
            return {"result": {"turn_active": True, "observation_step": 3}}
        return {"error": {"code": "method_not_found", "message": f"no method {method}"}}

    return answer


@pytest.fixture
def view(tmp_path: Path) -> dict[str, Any]:
    for name in ("agentview", "wrist"):
        (tmp_path / f"{name}.png").write_bytes(PNG + name.encode())
    return {"observation_step": 3, "main_camera": "wrist", "images": [
        {"camera": "agentview", "path": str(tmp_path / "agentview.png")},
        {"camera": "wrist", "path": str(tmp_path / "wrist.png")}]}


def test_the_client_and_the_server_speak_one_protocol() -> None:
    assert PROTOCOL == robot_socket.PROTOCOL == "ora.robot-socket.v1"
    assert SOCKET_ENV == robot_socket.SOCKET_ENV == "ORA_ROBOT_SOCKET"


def test_robot_reads_tools_calls_and_observations_with_its_origin(serve, view, tmp_path, monkeypatch) -> None:
    server = serve(run_tool(view))
    monkeypatch.setenv(SOCKET_ENV, str(server.path))
    with Robot(timeout=10, origin={"client": "test"}) as robot:
        assert [tool["name"] for tool in robot.tools()] == ["observe", "move_pose"]
        assert robot.call("observe", {"views": ["wrist"]}, width=4)["result"] == {"step": 3}
        refused = robot.call("move_pose", arm="left")
        assert refused["ok"] is False and refused["error"]["code"] == "out_of_bounds"
        observed = robot.observe(save=tmp_path / "saved.png")
        assert observed["saved"] == str(tmp_path / "saved.png")
        assert (tmp_path / "saved.png").read_bytes() == PNG + b"wrist"  # the main camera's
        robot.observe(save=tmp_path / "other.png", camera="agentview")
        assert (tmp_path / "other.png").read_bytes() == PNG + b"agentview"
        assert robot.status() == {"turn_active": True, "observation_step": 3}
        with pytest.raises(RobotError) as unknown:
            robot.request("teleport")
        assert unknown.value.code == "method_not_found"
    assert [request["id"] for request in server.requests] == [1, 2, 3, 4, 5, 6, 7]
    assert server.requests[1]["params"] == {"name": "observe", "arguments": {"views": ["wrist"], "width": 4}}
    assert {request["origin"]["client"] for request in server.requests} == {"test"}


def test_a_broken_connection_or_answer_raises_with_its_code(serve, socket_dir, monkeypatch) -> None:
    monkeypatch.delenv(SOCKET_ENV, raising=False)
    with pytest.raises(RobotError) as missing:
        Robot()
    assert missing.value.code == "no_socket"
    with pytest.raises(RobotError) as unreachable:
        Robot(socket_dir / "absent.sock").status()
    assert unreachable.value.code == "unreachable"
    hangs_up = serve(lambda request: None)
    with pytest.raises(RobotError) as closed:
        Robot(hangs_up.path, timeout=10).status()
    assert closed.value.code == "closed"
    wrong_id = serve(lambda request: {"id": 99, "result": {}})
    with pytest.raises(RobotError) as protocol:
        Robot(wrong_id.path, timeout=10).status()
    assert protocol.value.code == "protocol"


def test_an_image_is_picked_by_camera_then_the_main_camera_then_the_first(view) -> None:
    assert robot_client.pick_image(view).endswith("wrist.png")
    assert robot_client.pick_image(view, "agentview").endswith("agentview.png")
    assert robot_client.pick_image({**view, "main_camera": "head"}).endswith("agentview.png")
    with pytest.raises(RobotError) as absent:
        robot_client.pick_image(view, "head")
    assert absent.value.code == "no_image" and "agentview" in str(absent.value)


def test_the_origin_names_this_process_its_script_and_its_parent() -> None:
    record = robot_client.origin_record("python", ["/work/script.py", "--fast"])
    assert record["client"] == "python" and record["pid"] == os.getpid() and record["cwd"] == os.getcwd()
    assert record["command"] == "/work/script.py --fast" and record["script"] == "/work/script.py"
    assert record["parent_command"]
    assert "script" not in robot_client.origin_record("python", ["-c", "pass"])
    assert robot_client._bounded("x" * 5000).endswith("...") and len(robot_client._bounded("x" * 5000)) == 2003


def test_ora_robot_lists_calls_observes_and_reports_by_exit_status(serve, view, tmp_path, capsys) -> None:
    server = serve(run_tool(view))
    flag = ["--socket", str(server.path)]
    assert ora_robot([*flag, "tools"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert listed[0].split() == ["observe", "read_only", "Look."] and listed[1].split() == ["move_pose", "mutation"]
    assert ora_robot([*flag, "tools", "--json"]) == 0
    assert [tool["name"] for tool in json.loads(capsys.readouterr().out)] == ["observe", "move_pose"]
    assert ora_robot([*flag, "call", "observe", "--json", '{"views": ["wrist"]}', "width=4", "label=red cup",
                      "--save", str(tmp_path / "out.png")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["saved"] == [str(tmp_path / "out-1.png"), str(tmp_path / "out-2.png")]
    assert server.requests[-1]["params"]["arguments"] == {"views": ["wrist"], "width": 4, "label": "red cup"}
    assert server.requests[-1]["origin"]["client"] == "ora-robot"
    assert ora_robot([*flag, "call", "move_pose", "arm=left"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "out_of_bounds"
    assert ora_robot([*flag, "observe", "--save", str(tmp_path / "view.png"), "--camera", "agentview"]) == 0
    assert (tmp_path / "view.png").read_bytes() == PNG + b"agentview"
    capsys.readouterr()
    assert ora_robot([*flag, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["observation_step"] == 3


def test_ora_robot_refuses_a_malformed_line_or_an_unreachable_socket(serve, view, tmp_path, capsys) -> None:
    server = serve(run_tool(view))
    flag = ["--socket", str(server.path)]
    for argv in ([*flag, "call", "observe", "not-a-pair"], [*flag, "call", "observe", "--json", "[1]"]):
        assert ora_robot(argv) == 2
        assert "ora-robot:" in capsys.readouterr().err
    assert ora_robot(["--socket", str(tmp_path / "absent.sock"), "status"]) == 2
    assert "(unreachable)" in capsys.readouterr().err
    with pytest.raises(SystemExit) as usage:
        ora_robot([*flag, "status", "extra=1"])
    assert usage.value.code == 2
