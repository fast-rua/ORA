"""The robot socket of a free-play run: the agent's scripts call the robot through the Supervisor.

A free-play run (plan ``episode_mode: "play"``) opens it; a benchmark run opens
it only for its code-mode scripts (plan ``policy.code_mode``, ``code_mode.py``),
with no client installed, and calls them through ``_script_call`` directly. ``PiPlanner`` binds a unix socket (mode 0600) at
``<run>/native/robot.sock`` -- or, when that path is too long for a socket
address, in a private directory under the temporary directory -- announces it
in ``<run>/native/robot-socket.json`` and hands the agent core's environment
``ORA_ROBOT_SOCKET``, ``ora-robot`` on ``PATH`` and ``ora.robot_client`` on
``PYTHONPATH`` (a standard-library copy under ``<run>/native/robot-client``).
The protocol (``ora.robot-socket.v1``) is documented in
``ora/robot_client/client.py``.

Authority stays on the Supervisor's thread. Connection threads answer
``tools.list``, ``status`` and ``observe`` from snapshots that thread
publishes; every ``tools.call`` is queued for the Supervisor's own thread,
which runs it while it waits on the core's ``act`` and serves none of the
core's requests (``ScriptAwareRpcPeer``). A call outside a policy turn is
refused at once (``no_active_turn``). A queued call takes the path of the
model's own tool call:

- the proposal, the result (and its images) are recorded in the agent trace
  as ``tool_call_proposed``/``tool_result``/``tool_image``, the fields the
  core records for the model's calls plus ``payload.origin`` (``kind:
  "script"``, the command, script and parent command the client reported;
  ``kind: "codemode"`` and the codemode call that made it when the agent core
  itself calls from a Pi codemode script);
  ``model_call_id`` and ``tool_call_id`` are the call's own
  ``script-call-NNNNNN``;
- a read-only call runs at once;
- a world-changing call is admitted by the Supervisor before it runs
  (``ScriptActionExecutor.admit``); its persisted proposal is recorded as
  ``action_proposed`` and executed by the Supervisor inside the current turn
  (``execute``), one at a time, with the receipt and the fresh observation
  recorded as for any action; the script gets the receipt and the new
  observation;
- a lifecycle call (finishing the episode) is the agent's own decision and is
  refused, as is any call after the agent's own action ended the turn;
- once a script action ran in a turn, the model's own world-changing call in
  that turn is refused (``world_changed_by_script``): its turn is anchored to
  an observation the world has left;
- on a real cell a stop at the cell ends the whole script (``script_stop.py``).

Each request is also appended to ``<run>/native/robot-socket.jsonl``.
"""

from __future__ import annotations

import base64
import binascii
import copy
import json
import os
import re
import selectors
import shutil
import socket
import struct
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from ora.agent.contracts import PolicySessionError
from ora.agent.session_core.tool_arguments import _tool_error
from ora.agent.error_classes import ToolError
from ora.agent.composition.surface.tool_names import DOTTED_TOOL_NAMES

from . import script_stop
from .rpc import RpcError, RpcPeer

PROTOCOL = "ora.robot-socket.v1"
SOCKET_ENV = "ORA_ROBOT_SOCKET"
#: Longest socket path bound in place (macOS allows 104 bytes with the NUL, Linux 108).
MAX_SOCKET_PATH_BYTES = 103
#: One request line; a longer line closes the connection.
MAX_REQUEST_BYTES = 4 * 1024 * 1024
CLIENT_SOURCE = Path(__file__).resolve().parents[2] / "robot_client"
CLIENT_SHIM = """#!/bin/sh
# ora-robot of this free-play run: the standard-library client copied beside it.
SITE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}" exec "${ORA_ROBOT_PYTHON:-python3}" -S -m ora.robot_client "$@"
"""


class ScriptActionExecutor(Protocol):
    """The Supervisor's side of a script's world-changing call (``ora/orchestrator/script_actions.py``)."""

    def admit(self) -> str | None:
        """None to let the tool run, else the reason the Supervisor refuses it."""

    def execute(self) -> Mapping[str, Any]:
        """Execute the persisted proposal inside the current turn; its receipt summary."""


@dataclass
class ScriptCall:
    """One queued ``tools.call``; the connection thread waits on ``done``."""

    connection: int
    request_id: Any
    name: str
    arguments: dict[str, Any]
    origin: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


def _refusal(code: str, message: str) -> dict[str, Any]:
    return {**_tool_error(ToolError(message, code=code, retryable=False)), "traced": False}


def _peer_pid(connection: socket.socket) -> int | None:
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        return None
    try:
        pid, _uid, _gid = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, option, struct.calcsize("3i")))
    except OSError:
        return None
    return pid


def _wire_name(name: str) -> str:
    """A tool's name on the provider wire (ora-pi/src/provider/tool_names.ts): ``robot.submit_action`` -> ``robot_submit_action``."""

    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def _bounded_origin(value: Any) -> dict[str, Any]:
    """The client's origin record, scalars only and bounded."""

    if not isinstance(value, Mapping):
        return {}
    kept: dict[str, Any] = {}
    for key, item in list(value.items())[:16]:
        if isinstance(item, (str, int, float, bool)) or item is None:
            kept[str(key)[:64]] = item[:4000] if isinstance(item, str) else item
    return kept


def install_client(native: Path) -> Path:
    """Copy ``ora.robot_client`` and an ``ora-robot`` shim under ``native/robot-client``; its root."""

    site = native / "robot-client"
    package = site / "ora" / "robot_client"
    if site.exists():
        shutil.rmtree(site)
    package.mkdir(parents=True)
    (site / "ora" / "__init__.py").write_text('"""The ORA robot client of one free-play run."""\n', encoding="utf-8")
    for source in sorted(CLIENT_SOURCE.glob("*.py")):
        shutil.copy2(source, package / source.name)
    shim = site / "bin" / "ora-robot"
    shim.parent.mkdir()
    shim.write_text(CLIENT_SHIM, encoding="utf-8")
    shim.chmod(0o755)
    return site


class RobotSocketServer:
    """The listening socket, its connection threads and the queue of script calls."""

    def __init__(self, native: Path) -> None:
        self.native = native
        preferred = native / "robot.sock"
        self.private_dir: Path | None = None
        if len(os.fsencode(preferred)) <= MAX_SOCKET_PATH_BYTES:
            self.path = preferred
        else:
            self.private_dir = Path(tempfile.mkdtemp(prefix="ora-robot-"))
            if len(os.fsencode(self.private_dir / "robot.sock")) > MAX_SOCKET_PATH_BYTES:
                # A temporary directory as deep as the run (a deep TMPDIR): /tmp instead.
                os.rmdir(self.private_dir)
                self.private_dir = Path(tempfile.mkdtemp(prefix="ora-robot-", dir="/tmp"))
            self.path = self.private_dir / "robot.sock"
        self.log_path = native / "robot-socket.jsonl"
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.calls: deque[ScriptCall] = deque()
        self.turn_open = False
        self.closed = False
        self.actions_this_turn = 0
        self.stopped_this_turn: str | None = None   # a real cell's stop this turn (script_stop.py)
        self.script_calls = 0
        self.script_actions = 0
        self.snapshot: dict[str, Any] = {"tools": [], "observation": None, "status": {}}
        self.runner: Callable[[ScriptCall], dict[str, Any]] | None = None
        self.wake_fd, self._wake_write = os.pipe()
        os.set_blocking(self.wake_fd, False)
        os.set_blocking(self._wake_write, False)
        self.connections: list[socket.socket] = []
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self.path.exists() or self.path.is_symlink():
            self.path.unlink()
        mask = os.umask(0o177)
        try:
            self.server.bind(str(self.path))
        finally:
            os.umask(mask)
        os.chmod(self.path, 0o600)
        self.server.listen(16)
        pointer = native / "robot-socket.json"
        pointer.with_suffix(".json.tmp").write_text(json.dumps({
            "schema": PROTOCOL, "socket": str(self.path), "pid": os.getpid(), "started_unix_s": time.time(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(pointer.with_suffix(".json.tmp"), pointer)
        self.acceptor = threading.Thread(target=self._accept, name="ora-robot-socket", daemon=True)
        self.acceptor.start()

    # -- the Supervisor's thread ------------------------------------------------------

    def publish(self, **fields: Any) -> None:
        with self.lock:
            self.snapshot.update(copy.deepcopy(fields))

    def open_turn(self) -> None:
        with self.lock:
            self.turn_open = True
            self.actions_this_turn = 0
            self.stopped_this_turn = None

    def close_turn(self) -> None:
        """The turn ended: calls still queued are refused, untraced."""

        with self.lock:
            self.turn_open = False
            pending, self.calls = list(self.calls), deque()
        for call in pending:
            call.response = {"result": _refusal("turn_ended", "the agent's turn ended before this call ran")}
            call.done.set()

    def has_calls(self) -> bool:
        with self.lock:
            return bool(self.calls)

    def drain_wake(self) -> None:
        try:
            while os.read(self.wake_fd, 4096):
                pass
        except BlockingIOError:
            pass

    def run_pending(self) -> None:
        """Run every queued call on this (the Supervisor's) thread, in arrival order."""

        while True:
            with self.lock:
                if not self.calls:
                    return
                call = self.calls.popleft()
            try:
                call.response = {"result": self.runner(call)}
            except BaseException as exc:
                call.response = {"error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}}
                raise
            finally:
                call.done.set()

    def close(self) -> None:
        with self.lock:
            self.closed = True
        self.close_turn()
        try:
            self.server.close()
        finally:
            for connection in list(self.connections):
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if self.path.exists() or self.path.is_symlink():
                self.path.unlink()
            if self.private_dir is not None:
                shutil.rmtree(self.private_dir, ignore_errors=True)
            for descriptor in (self.wake_fd, self._wake_write):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    # -- connection threads -------------------------------------------------------------

    def _accept(self) -> None:
        number = 0
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            number += 1
            self.connections.append(connection)
            threading.Thread(target=self._serve, args=(connection, number), name=f"ora-robot-{number}",
                             daemon=True).start()

    def _serve(self, connection: socket.socket, number: int) -> None:
        peer = _peer_pid(connection)
        reader = connection.makefile("rb")
        try:
            while True:
                line = reader.readline(MAX_REQUEST_BYTES + 1)
                if not line:
                    return
                if len(line) > MAX_REQUEST_BYTES:
                    connection.sendall(self._line({"id": None, "error": {
                        "code": "request_too_large", "message": f"a request line is at most {MAX_REQUEST_BYTES} bytes"}}))
                    return
                connection.sendall(self._line(self._answer(line, number, peer)))
        except OSError:
            return
        finally:
            reader.close()
            connection.close()
            if connection in self.connections:
                self.connections.remove(connection)

    @staticmethod
    def _line(message: Mapping[str, Any]) -> bytes:
        return (json.dumps(message, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")

    def _answer(self, line: bytes, number: int, peer: int | None) -> dict[str, Any]:
        try:
            request = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {"id": None, "error": {"code": "bad_request", "message": f"not a JSON line: {exc}"}}
        if not isinstance(request, Mapping):
            return {"id": None, "error": {"code": "bad_request", "message": "a request is a JSON object"}}
        request_id, method, params = request.get("id"), request.get("method"), request.get("params") or {}
        origin = {**_bounded_origin(request.get("origin")), "connection": number}
        if peer is not None:
            origin["peer_pid"] = peer
        if not isinstance(params, Mapping):
            answer = {"error": {"code": "bad_request", "message": "params is a JSON object"}}
        elif method == "tools.list":
            with self.lock:
                answer = {"result": {"protocol": PROTOCOL, "tools": copy.deepcopy(self.snapshot["tools"])}}
        elif method == "observe":
            with self.lock:
                observation = copy.deepcopy(self.snapshot["observation"])
            answer = ({"result": observation} if observation is not None else
                      {"error": {"code": "no_observation", "message": "no observation is published yet"}})
        elif method == "status":
            with self.lock:
                answer = {"result": {
                    **copy.deepcopy(self.snapshot["status"]), "protocol": PROTOCOL, "socket": str(self.path),
                    "turn_active": self.turn_open, "queued_calls": len(self.calls),
                    "script_calls": self.script_calls, "script_actions": self.script_actions,
                    "observation_step": (self.snapshot["observation"] or {}).get("observation_step"),
                }}
        elif method == "tools.call":
            name, arguments = params.get("name"), params.get("arguments", {})
            if not isinstance(name, str) or not name or not isinstance(arguments, Mapping):
                answer = {"error": {"code": "bad_request",
                                    "message": "tools.call takes a tool name and an arguments object"}}
            else:
                answer = self._call(ScriptCall(number, request_id, name, dict(arguments), origin))
        else:
            answer = {"error": {"code": "method_not_found", "message": f"unknown method {method!r}; methods: "
                                "tools.list, tools.call, observe, status"}}
        result = answer.get("result")
        result = result if isinstance(result, Mapping) else {}
        outcome = ("error:" + str(answer["error"].get("code")) if "error" in answer
                   else "ok" if result.get("ok", True) is not False
                   else "refused:" + str((result.get("error") or {}).get("code")))
        self._log({"unix_s": time.time(), "connection": number, "request_id": request_id, "method": method,
                   "tool": params.get("name") if isinstance(params, Mapping) else None, "outcome": outcome,
                   "call_id": result.get("call_id"), "origin": origin})
        return {"id": request_id, **answer}

    def _call(self, call: ScriptCall) -> dict[str, Any]:
        with self.lock:
            if self.closed:
                return {"result": _refusal("run_ending", "the run is ending")}
            if not self.turn_open:
                return {"result": _refusal("no_active_turn", "robot calls run only while the agent's turn runs "
                                           "(inside its own tool call, such as a bash command)")}
            self.calls.append(call)
        try:
            os.write(self._wake_write, b"x")
        except (BlockingIOError, OSError):
            pass  # a wake is already pending (or the run is closing)
        call.done.wait()
        return dict(call.response or {"error": {"code": "internal_error", "message": "no answer"}})

    def _log(self, record: Mapping[str, Any]) -> None:
        with self.log_lock:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


class ScriptAwareRpcPeer(RpcPeer):
    """``RpcPeer`` that, while ``act`` waits, also runs the robot socket's queued calls.

    Lines are read from the socket directly (no buffered reader), so the wait
    can watch the socket and the robot server's wake pipe together. Queued
    script calls run only in the frame of the outermost ``act``, never while a
    core request is being served. While one runs, the core's own requests that
    arrive are held and served after it (the core awaits them; Python's nested
    ``record`` calls are answered independently), and a response to an outer
    request that arrives inside a nested call is kept for it.
    """

    def __init__(self, sock: socket.socket, robot: RobotSocketServer) -> None:
        super().__init__(sock)
        self.robot = robot
        self.buffer = bytearray()
        self.outstanding: list[str] = []
        self.stashed: dict[str, dict[str, Any]] = {}
        self.held: deque[dict[str, Any]] = deque()
        self.holding = False
        self.core_only = selectors.DefaultSelector()
        self.core_only.register(sock, selectors.EVENT_READ, "core")
        self.with_robot = selectors.DefaultSelector()
        self.with_robot.register(sock, selectors.EVENT_READ, "core")
        self.with_robot.register(robot.wake_fd, selectors.EVENT_READ, "robot")

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        request_id = f"p{self.next_id}"
        self.next_id += 1
        turn = method == "act" and not self.outstanding
        self.outstanding.append(request_id)
        if turn:  # open before the core can start the turn's first tool call
            self.robot.open_turn()
        try:
            self._write({"id": request_id, "method": method, "params": dict(params or {})})
            while True:
                message = self.stashed.pop(request_id, None) or self._next(scripts=turn)
                if message is None:
                    continue
                if isinstance(message.get("method"), str):
                    if self.holding:
                        self.held.append(message)
                    else:
                        self._serve(message)
                    continue
                if message.get("id") != request_id:
                    if message.get("id") in self.outstanding:
                        self.stashed[str(message["id"])] = message
                        continue
                    raise RpcError(f"unexpected RPC response {message.get('id')!r}", "protocol")
                error = message.get("error")
                if error is not None:
                    raise RpcError(str(error.get("message")), str(error.get("code")), error.get("data"))
                return message.get("result")
        finally:
            self.outstanding.pop()
            if turn:
                self.robot.close_turn()

    def _next(self, *, scripts: bool) -> dict[str, Any] | None:
        """The next core message; None after running queued script calls (``scripts`` frames only).

        The core's messages come first: a queued call runs only when no core
        message is buffered or readable. Other frames watch the core alone, so
        a wake that arrives during a nested call stays for the ``act`` frame.
        """

        if scripts and self.held:
            return self.held.popleft()
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self.buffer[:newline])
                del self.buffer[:newline + 1]
                if line.strip():
                    return json.loads(line.decode("utf-8"))
                continue
            if scripts and self.robot.has_calls() and not self.core_only.select(timeout=0):
                self._run_scripts()
                return None
            ready = {key.data for key, _ in (self.with_robot if scripts else self.core_only).select()}
            if "core" in ready:
                data = self.sock.recv(1 << 16)
                if not data:
                    self.closed = True
                    raise RpcError("TypeScript agent core closed the RPC channel", "closed")
                self.buffer += data
                continue
            self.robot.drain_wake()

    def _run_scripts(self) -> None:
        self.robot.drain_wake()
        self.holding = True
        try:
            self.robot.run_pending()
        finally:
            self.holding = False

    def close(self) -> None:
        self.core_only.close()
        self.with_robot.close()
        super().close()


class PiRobotSocket:
    """``PiPlanner``'s robot socket: open, snapshot, run script calls, close (a mixin)."""

    robot: RobotSocketServer | None = None
    _script_executor: ScriptActionExecutor | None = None

    def _open_robot_socket(self, path_env: str, *, shell: bool = True) -> dict[str, str]:
        """Bind the socket and install the client; the agent core's environment additions.

        A benchmark episode's code mode (``shell`` false) needs only the script path: no client.
        """

        native = self.run_root / "native"
        site = install_client(native) if shell else None
        self.robot = RobotSocketServer(native)
        self.robot.runner = self._run_script_call
        surface = getattr(self.tools, "surface", None)
        mutation = set(getattr(surface, "mutation_tools", ("robot.submit_action",)))
        lifecycle = set(getattr(surface, "lifecycle_tools", ()))
        listed = []
        for definition in self.gateway.script_definitions():  # code mode: also the tools only scripts call
            function = definition["function"]
            name = function["name"]
            listed.append({
                "name": name, "wire_name": _wire_name(name), "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
                "effect": "lifecycle" if name in lifecycle else "mutation" if name in mutation else "read_only",
            })
        # A script may spell a tool as listed, as the model's wire shows it, or in its other dotted form.
        self._script_names = {item["name"]: item["name"] for item in listed}
        for dotted, canonical in DOTTED_TOOL_NAMES.items():
            for spelled, listed_as in ((dotted, canonical), (canonical, dotted)):
                if listed_as in self._script_names and spelled not in self._script_names:
                    self._script_names[spelled] = listed_as
        for item in listed:
            self._script_names.setdefault(item["wire_name"], item["name"])
        self.robot.publish(tools=listed, status={"run_id": self.run_id,
                                                 "episode_mode": self.episode_mode or "benchmark"})
        if site is None:
            return {}
        return {
            SOCKET_ENV: str(self.robot.path),
            "PATH": f"{site / 'bin'}{os.pathsep + path_env if path_env else ''}",
            "PYTHONPATH": str(site),
        }

    def bind_script_action_executor(self, executor: ScriptActionExecutor) -> None:
        """The Supervisor's admission and execution of script actions (bound once, free play only)."""

        if self._script_executor is not None:
            raise PolicySessionError("the script action executor must be bound once")
        self._script_executor = executor

    def _close_robot_socket(self) -> None:
        if self.robot is not None:
            self.robot.close()

    # -- what a script sees ---------------------------------------------------------

    def _robot_observed(self, context: Mapping[str, Any]) -> None:
        """Snapshot a freshly published observation for ``observe`` (the Supervisor's pointer)."""

        if self.robot is None:
            return
        state = context.get("policy_state") if isinstance(context.get("policy_state"), Mapping) else {}
        images = []
        try:
            _ref, _snapshot, cameras = self._load_actual_state(context)
        except PolicySessionError:
            cameras = {}
        for camera, record in sorted(cameras.items()):
            files = record.get("files") if isinstance(record, Mapping) else None
            rgb = (files.get("rgb_lossless_png") or files.get("rgb")) if isinstance(files, Mapping) else None
            if isinstance(rgb, Mapping) and isinstance(rgb.get("path"), str) and (self.run_root / rgb["path"]).is_file():
                images.append({"camera": camera, "path": str(self.run_root / rgb["path"])})
        try:
            observation_id = self._current_observation_id()
        except Exception:  # the Control file is the Supervisor's; absent before the first publish
            observation_id = None
        view = {
            "observation_step": state.get("step"), "observation_id": observation_id,
            "main_camera": context.get("main_camera"), "images": images,
            **{key: context[key] for key in ("proprioception", "public_robot_facts", "previous_execution",
                                             "remaining_budget", "runtime_notice") if key in context},
        }
        self.robot.publish(observation=view)

    def _robot_step(self) -> int | None:
        observation = self.robot.snapshot.get("observation") or {}
        step = observation.get("observation_step")
        return step if isinstance(step, int) else None

    # -- one script call, on the Supervisor's thread ---------------------------------

    def _run_script_call(self, call: ScriptCall) -> dict[str, Any]:
        # A core lost while a script call records is never resumed (core_process._call).
        self._serving += 1
        try:
            return self._script_call(call)
        finally:
            self._serving -= 1

    def _script_record(self, event_type: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("record", {"type": event_type, "fields": dict(fields)})

    def _origin_kind(self, origin: Mapping[str, Any]) -> str:
        """``codemode`` for a call the agent core made from a Pi codemode script, else ``script``.

        Only the core process itself (the socket peer's pid) may name a call codemode.
        """

        process = getattr(self, "process", None)
        core = process is not None and origin.get("peer_pid") == process.pid
        return "codemode" if core and origin.get("kind") == "codemode" else "script"

    def _script_call(self, call: ScriptCall, *, ids: tuple[str, str] | None = None,
                     origin: Mapping[str, Any] | None = None,
                     preset: tuple[str | None, Exception] | None = None,
                     result_fields: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None) -> dict[str, Any]:
        """One script call; a code-mode program's names its ``(model_call_id, tool_call_id)``, its
        origin, any refusal decided before it and, with ``library: extended``, ``result_fields`` (``code_mode.py``):
        given the call's complete outcome, the fields its ``tool_result`` payload records besides its own, the result
        then recorded once the outcome is complete (after an action the call submitted ran)."""

        robot = self.robot
        robot.script_calls += 1
        model_call_id, call_id = ids or (f"script-call-{robot.script_calls:06d}",) * 2
        name = self._script_names.get(call.name, call.name)
        origin = dict(origin) if origin is not None else {**call.origin, "kind": self._origin_kind(call.origin)}
        step = self._robot_step()
        arguments: dict[str, Any] | None = dict(call.arguments)
        effect = None
        refusal: Exception | None = None
        if preset is not None:
            effect, refusal = preset
        elif self.submitted_action is not None:
            refusal = ToolError("the agent's own action ended this turn; robot calls run in the next turn",
                                code="turn_ended", retryable=False)
        elif name not in self._script_names.values():
            refusal = ToolError(f"unknown tool: {call.name}", code="unknown_tool", retryable=False, field="name",
                                received=call.name, constraints={"allowed": sorted(set(self._script_names.values()))})
        else:
            try:
                arguments = dict(self.gateway.arguments(name, arguments))
                effect = self.gateway.classify_effect(name, arguments)
            except Exception as exc:
                refusal = exc
            if refusal is None and effect == "lifecycle":
                refusal = ToolError("a lifecycle tool is the agent's own decision; a script cannot call it",
                                    code="script_lifecycle_unavailable", retryable=False)
            elif refusal is None and effect == "mutation" and robot.stopped_this_turn is not None:
                refusal = script_stop.refusal(robot.stopped_this_turn)
            elif refusal is None and effect == "mutation":
                # Code mode admits through the Supervisor's tool-action executor (code_mode.py), free play its own.
                executor = getattr(self, "tool_action_executor", None) if getattr(self, "code_mode", None) \
                    else self._script_executor
                reason = "script_actions_unbound" if executor is None else executor.admit()
                if reason is not None:
                    refusal = ToolError(f"the Supervisor refused this world-changing call: {reason}",
                                        code="script_action_refused", retryable=False,
                                        constraints={"reason": reason})
        observation_event = self.last_fresh_observation_event_id
        proposal = self._script_record("tool_call_proposed", {
            "causal_parent_id": observation_event, "observation_step": step,
            "model_call_id": model_call_id, "tool_call_id": call_id,
            "payload": {"name": name, "arguments": arguments, "effect": effect, "batch_valid": True, "origin": origin},
        })
        tool_events: list[dict[str, Any]] = []
        if refusal is not None:
            result = _tool_error(refusal)
        else:
            call_context = {
                "policy_session_id": self.policy_session_id, "model_call_id": model_call_id, "tool_call_id": call_id,
                "proposal_event_id": proposal["event_id"], "observation_step": step,
                "current_observation_event_id": observation_event, "interaction_phase": "robot_rollout",
                "preceding_action_step": self.last_action_step,
                "preceding_action_event_id": self.last_action_event_id,
            }
            self._tool_events = tool_events
            try:
                invoke = getattr(self.tools, "invoke_with_context", None)
                result = (dict(invoke(name, arguments, call_context=call_context)) if callable(invoke)
                          else dict(self.tools.invoke(name, arguments)))
            except Exception as exc:
                result = _tool_error(exc)
            finally:
                self._tool_events = None
        for record in tool_events:  # where the core appends a tool's own events: before its result
            self._script_record(str(record["type"]), record["fields"])
        public, images = self._script_result(result)
        visible_ok = public.get("ok")
        ok = visible_ok if isinstance(visible_ok, bool) else (
            "error" not in public and public.get("status") not in {"error", "failed", "invalid", "rejected"})
        common = {"observation_step": step, "model_call_id": model_call_id, "tool_call_id": call_id}

        def record_result(fields: Mapping[str, Any]) -> None:
            self._script_record("tool_result", {
                **common, "causal_parent_id": proposal["event_id"],
                "payload": {"name": name, "ok": ok, "origin": origin, **fields},
                "artifact_ref": self.artifacts.put_json(public, kind="native_tool_result"),
            })
            for index, ref in enumerate(images):
                self._script_record("tool_image", {**common, "visibility": "exact", "artifact_ref": ref,
                                                   "payload": {"image_index": index, "origin": origin}})

        if result_fields is None:
            record_result({})
        outcome: dict[str, Any] = {
            "ok": ok, "call_id": call_id, "tool": name, "effect": effect, "observation_step": step,
            "result": public, "images": [{"path": str(self.run_root / ref["path"]), "sha256": ref["sha256"],
                                          "media_type": ref.get("media_type")} for ref in images],
        }
        if isinstance(public.get("error"), Mapping):
            outcome["error"] = public["error"]
        runtime = result.get("_runtime")
        directive = runtime.get("directive") if isinstance(runtime, Mapping) else None
        directive = directive if directive is not None else result.get("directive")
        fields: Mapping[str, Any] = {}
        try:
            if effect == "mutation" and result.get("ok", True) is not False and directive == "action_submitted":
                if getattr(self, "code_mode", None):  # code mode: the script's parent invocation executes it
                    executed = self._code_mode_action()
                    outcome["images"] += executed.pop("images", [])  # a committed grasp's post-close views
                    outcome.update(executed)
                else:
                    outcome.update(self._script_action(name, call_id, step, origin, model_call_id=model_call_id))
                robot.stopped_this_turn = robot.stopped_this_turn or script_stop.stopped_by(
                    outcome, script_stop.stop_codes(getattr(self, "play", None)))
            if result_fields is not None:
                fields = result_fields(outcome)
        finally:
            if result_fields is not None:  # recorded whatever happened, without the fields if they failed
                record_result(fields)
        return outcome

    def _script_result(self, result: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """What the script receives: the tool's result without runtime metadata, images stored."""

        public = {key: copy.deepcopy(value) for key, value in result.items()
                  if key not in {"_runtime", "image_base64", "images_base64"}}
        encoded = ([result["image_base64"]] if isinstance(result.get("image_base64"), str)
                   else [item for item in result.get("images_base64") or [] if isinstance(item, str)])
        historical = result.get("media_type")
        media_type = (str(historical) if result.get("status") == "actual_historical_artifact"
                      and isinstance(historical, str) and historical.startswith("image/") else "image/png")
        images = []
        for item in encoded:
            try:
                data = base64.b64decode(item, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise PolicySessionError("tool returned invalid image_base64") from exc
            images.append(self.artifacts.put_bytes(
                data, kind="native_tool_image", suffix=".png" if media_type == "image/png" else ".img",
                media_type=media_type,
            ))
        return public, images

    def _script_action(self, name: str, call_id: str, step: int | None, origin: Mapping[str, Any], *,
                       model_call_id: str | None = None) -> dict[str, Any]:
        """Record the persisted proposal, let the Supervisor execute it, return the receipt."""

        action = json.loads((self.tools.control_root / "submitted-action.json").read_text(encoding="utf-8"))
        observation_id = self._current_observation_id()
        event = self._script_record("action_proposed", {
            "observation_step": step, "model_call_id": model_call_id or call_id, "tool_call_id": call_id,
            "payload": {"action_id": action.get("action_id"), "observation_id": observation_id, "origin": dict(origin)},
            "artifact_ref": self.artifacts.put_json(action, kind="native_action_proposal"),
        })
        self.last_action_event_id = event["event_id"]
        self.last_action_step = step
        self.last_action_id = str(action.get("action_id"))
        self.last_action_public_name = name
        executed = dict(self._script_executor.execute())
        self.robot.actions_this_turn += 1
        self.robot.script_actions += 1
        observation = copy.deepcopy(self.robot.snapshot.get("observation"))
        return {
            "action": {"action_id": action.get("action_id"), "observation_id": observation_id,
                       "event_id": event["event_id"]},
            "execution": executed,
            "receipt": (observation or {}).get("previous_execution"),
            "observation": observation,
        }

    def _script_moved_refusal(self, name: str, call_context: Mapping[str, Any]) -> dict[str, Any] | None:
        """The model's own world-changing call after script actions in the same turn, refused."""

        if self.robot is None or not self.robot.actions_this_turn:
            return None
        result = _tool_error(ToolError(
            f"this turn's world already changed through {self.robot.actions_this_turn} robot script action(s); "
            "a direct world-changing call is anchored to the turn's first observation and is refused. "
            "Continue through ora-robot, or end the turn to act on a fresh observation.",
            code="world_changed_by_script", retryable=False,
            constraints={"script_actions_this_turn": self.robot.actions_this_turn},
        ))
        projected = self._project(name, result, call_id=str(call_context["tool_call_id"]),
                                  step=int(call_context["observation_step"]))
        projected.update({"directive": None, "raw_ok": False, "runtime_events": []})
        return projected


__all__ = ["PROTOCOL", "SOCKET_ENV", "PiRobotSocket", "RobotSocketServer", "ScriptActionExecutor",
           "ScriptAwareRpcPeer", "install_client"]
