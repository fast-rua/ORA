"""``Robot``: one connection to a free-play run's robot socket (protocol ``ora.robot-socket.v1``).

The protocol is one JSON object per line in each direction. A request is
``{"id", "method", "params", "origin"}``; the answer carries the same ``id``
and either ``result`` or ``error`` (``{"code", "message"}``, a request the
socket could not take). Methods: ``tools.list``, ``tools.call`` (``name``,
``arguments``), ``observe`` and ``status``. A tool that ran, failed or was
refused answers a ``result`` whose ``ok`` says which; only a broken request or
connection raises ``RobotError``.

``origin`` tells the run where a call came from: this process's command line,
its script and the command of the process that started it (the shell running
an agent's ``bash`` command). The run records it beside each call it traces.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROTOCOL = "ora.robot-socket.v1"
SOCKET_ENV = "ORA_ROBOT_SOCKET"
#: Longest command line kept in an origin record.
MAX_COMMAND_CHARS = 2000


class RobotError(RuntimeError):
    """The robot socket is unreachable or rejected the request itself (not a tool's refusal)."""

    def __init__(self, message: str, code: str = "robot_error") -> None:
        super().__init__(message)
        self.code = code


def _bounded(text: str) -> str:
    return text if len(text) <= MAX_COMMAND_CHARS else text[:MAX_COMMAND_CHARS] + "..."


def _command_of(pid: int) -> str | None:
    """The command line of ``pid`` (``/proc`` on Linux, ``ps`` elsewhere), or None."""

    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        parts = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        return _bounded(shlex.join(parts)) if parts else None
    except OSError:
        pass
    try:
        listed = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True,
                                text=True, timeout=2, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return _bounded(listed) or None


def origin_record(client: str, argv: list[str] | None = None) -> dict[str, Any]:
    """Where a call comes from: this process, its script and its parent's command."""

    argv = list(sys.argv if argv is None else argv)
    record: dict[str, Any] = {"client": client, "pid": os.getpid(), "cwd": os.getcwd()}
    if argv:
        record["command"] = _bounded(shlex.join(argv))
    if client == "python" and argv and argv[0] not in {"", "-c", "-m"}:
        record["script"] = os.path.abspath(argv[0])
    parent = _command_of(os.getppid())
    if parent is not None:
        record["parent_command"] = parent
    return record


class Robot:
    """The robot of the current free-play run; the socket comes from ``ORA_ROBOT_SOCKET``."""

    def __init__(self, socket_path: str | os.PathLike[str] | None = None, *, timeout: float | None = None,
                 client: str = "python", origin: Mapping[str, Any] | None = None) -> None:
        path = socket_path if socket_path is not None else os.environ.get(SOCKET_ENV)
        if not path:
            raise RobotError(f"{SOCKET_ENV} is not set: the robot is reachable only inside an ORA free-play run",
                             "no_socket")
        self.socket_path = os.fspath(path)
        self.timeout = timeout
        self.origin = dict(origin) if origin is not None else origin_record(client)
        self._socket: socket.socket | None = None
        self._reader: Any = None
        self._next_id = 1

    def __enter__(self) -> "Robot":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._reader.close()
                self._socket.close()
            finally:
                self._socket = self._reader = None

    def _connect(self) -> socket.socket:
        if self._socket is None:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.timeout)
            try:
                connection.connect(self.socket_path)
            except OSError as exc:
                connection.close()
                raise RobotError(f"cannot reach the robot socket {self.socket_path}: {exc}", "unreachable") from exc
            self._socket, self._reader = connection, connection.makefile("rb")
        return self._socket

    def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        """One protocol request; its ``result``, or ``RobotError`` for a rejected request."""

        connection = self._connect()
        request_id = self._next_id
        self._next_id += 1
        line = json.dumps({"id": request_id, "method": method, "params": dict(params or {}),
                           "origin": self.origin}, ensure_ascii=False) + "\n"
        try:
            connection.sendall(line.encode("utf-8"))
            answer = self._reader.readline()
        except OSError as exc:
            self.close()
            raise RobotError(f"the robot socket connection failed: {exc}", "connection") from exc
        if not answer:
            self.close()
            raise RobotError("the robot socket closed the connection (the run ended?)", "closed")
        message = json.loads(answer.decode("utf-8"))
        if message.get("id") != request_id:
            raise RobotError(f"unexpected answer id {message.get('id')!r}", "protocol")
        error = message.get("error")
        if error is not None:
            raise RobotError(str(error.get("message")), str(error.get("code", "robot_error")))
        return message.get("result")

    def tools(self) -> list[dict[str, Any]]:
        """The run's tools: ``name``, ``description``, ``parameters`` (JSON Schema) and declared ``effect``."""

        return list(self.request("tools.list")["tools"])

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None, /, **kwargs: Any) -> dict[str, Any]:
        """Call one tool through the Supervisor; arguments as a mapping, keywords, or both.

        Returns the outcome: ``ok``, the tool's ``result`` and saved ``images``;
        for an executed world-changing call also ``action``, ``receipt`` and the
        new ``observation``. A refusal is ``ok: false`` with its ``error``.
        """

        merged = {**dict(arguments or {}), **kwargs}
        return dict(self.request("tools.call", {"name": tool, "arguments": merged}))

    def observe(self, *, save: str | os.PathLike[str] | None = None, camera: str | None = None) -> dict[str, Any]:
        """The latest published observation: step, public robot state and image paths.

        ``save`` copies one image there: ``camera``'s, else the main camera's.
        """

        view = dict(self.request("observe"))
        if save is not None:
            shutil.copyfile(pick_image(view, camera), save)
            view["saved"] = os.fspath(save)
        return view

    def status(self) -> dict[str, Any]:
        """The socket's view of the run: whether a turn is active, the observation step, call counts."""

        return dict(self.request("status"))


def pick_image(view: Mapping[str, Any], camera: str | None = None) -> str:
    """The path of ``camera``'s image in an observation (default: the main camera, else the first)."""

    images = [item for item in view.get("images") or [] if isinstance(item, Mapping)]
    wanted = camera or view.get("main_camera")
    for item in images:
        if item.get("camera") == wanted:
            return str(item["path"])
    if camera is None and images:
        return str(images[0]["path"])
    raise RobotError(f"no image for camera {wanted!r}; cameras: {[item.get('camera') for item in images]}",
                     "no_image")
