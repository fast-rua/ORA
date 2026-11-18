"""The cell board: a status page of one real cell, served on the cell's computer.

    ora cell board [--cell cell.yaml] [--port 8765] [--depth-service URL]

then open http://localhost:8765 there (or forward the port over ssh). Its periodic status
reads only ``/proc`` and the bus interfaces' packet counters, opening no socket. The live
view (``live.py``), a one-off arm read, the wrist table check and the operator's verbs
(``actions.py``) run on a button, under the cell lock where they touch the hardware.

The board is also the long-lived holder of the arms for ``hold`` and ``float``: the
``ora cell`` command and the play terminal's slash commands reach it over HTTP on the
same port (``/api/reset/<verb>``).

It answers the questions a robot's worker asks (``approval.py``): ``GET /api/approval``
lists the pending ones (also in ``/api/status``), and ``POST
/api/approval/<id>/approve`` or ``/decline`` (JSON body ``{"reason": ...}``, and ``{"value":
...}`` for a ``value`` question) answers one by its id, so a click can only answer the
question the page showed.

It shows and switches the cell's permission mode (``approval.mode_state``, ``switch_mode``):
``/api/approval`` and ``/api/status`` carry it (``mode``) with the latest motions that answered
themselves (``self_answered``: safe mode or auto-approve, with the numbers safe mode checked), and
``POST /api/mode/ask|safe|auto`` sets it, ``by: "cell board"``; the page asks first for ``auto``.
``/api/auto-approve/on|off`` (the older switch) still works.

It starts the guided hand-eye calibration (``POST /api/calibrate/start``, JSON body with the
optional ``square_mm``, ``pattern`` and ``placements``): ``ora cell calibrate --method hand-eye
--ask broker --json`` as a child process, whose every step is a question on this page like a
worker's. ``/api/status`` carries its progress (the child's stderr lines) and its answer;
``POST /api/calibrate/stop`` interrupts it (SIGINT: a motion stops and the arms go back to
where they started). The child says when SIGINT will reach it (``{"interruptible": true}`` on
stderr); a stop asked before that is held and sent then, since an earlier SIGINT can be lost.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import alerts as cell_alerts
from ..errors import CellError
from ..geometry import joint_sanity
from ..lock import CellLock, holder
from ..occupancy import motion_processes
from . import actions, approval, tools
from .context import Cell
from .live import LiveCell

PAGE = Path(__file__).with_name("board.html")
DEFAULT_PORT = 8765
#: The checkout the board runs from: the calibration child runs ``python -m ora.app`` there.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROGRESS_KEPT = 200
#: The motions that answered themselves the page lists: the latest few, of the last minutes.
SELF_ANSWERS_KEPT = 5
SELF_ANSWERS_S = 600.0
_SELF_BY = re.compile(r"^(?:safe mode|auto-approve) \(")
_ANSWER_ID = re.compile(r"[0-9a-f]{8}")


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=lambda v: np.asarray(v).tolist(), allow_nan=True))


def self_answers(now: float | None = None) -> list[dict[str, Any]]:
    """The motions that answered themselves lately, newest last: approved answers with the question's
    ``kind`` and ``title`` and ``by: "safe mode (...)"`` or ``"auto-approve (...)"``. Reads only."""

    now = time.time() if now is None else now
    found = []
    for path in approval.directory().glob("*.answer.json"):
        try:
            answer = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not (isinstance(answer, dict) and answer.get("approved") is True and isinstance(answer.get("kind"), str)
                and isinstance(answer.get("by"), str) and _SELF_BY.match(answer["by"])
                and isinstance(answer.get("id"), str) and _ANSWER_ID.fullmatch(answer["id"])):
            continue
        at = answer.get("answered_unix_s")
        if isinstance(at, (int, float)) and now - SELF_ANSWERS_S <= at:
            found.append({"id": answer["id"], "kind": answer["kind"], "title": str(answer.get("title") or ""),
                          "by": answer["by"], "answered_unix_s": at})
    return sorted(found, key=lambda item: item["answered_unix_s"])[-SELF_ANSWERS_KEPT:]


def calibrate_arguments(options: Mapping[str, Any]) -> list[str]:
    """``--square-mm``, ``--pattern`` and ``--placements`` from the page's fields; empty ones are asked later."""

    args: list[str] = []
    square = options.get("square_mm")
    if square not in (None, ""):
        try:
            square = float(square)
        except (TypeError, ValueError):
            raise CellError(f"square_mm {square!r} is not a number") from None
        if not 1 <= square <= 200:
            raise CellError(f"square_mm {square:g}: one square's side in millimetres, 1 to 200")
        args += ["--square-mm", f"{square:g}"]
    pattern = options.get("pattern")
    if pattern not in (None, ""):
        found = re.fullmatch(r"\s*(\d{1,2})\s*[xX×]\s*(\d{1,2})\s*", str(pattern))
        if found is None:
            raise CellError(f"pattern {pattern!r}: the inner corners as COLSxROWS, e.g. 11x8")
        args += ["--pattern", f"{found.group(1)}x{found.group(2)}"]
    placements = options.get("placements")
    if placements not in (None, ""):
        try:
            placements = int(placements)
        except (TypeError, ValueError):
            raise CellError(f"placements {placements!r} is not a whole number") from None
        if not 1 <= placements <= 20:
            raise CellError(f"placements {placements}: 1 to 20")
        args += ["--placements", str(placements)]
    return args


class Calibration:
    """The guided hand-eye calibration the page started: one child process at a time, its progress
    lines and its answer. The child asks every step through the approval broker, so the page's
    question card answers it as it answers a worker."""

    def __init__(self, cell: Cell, command: Sequence[str] | None = None) -> None:
        self.cell = cell
        self.command = list(command or [sys.executable, "-m", "ora.app"])
        self.process: subprocess.Popen | None = None
        self.progress: deque[dict[str, Any]] = deque(maxlen=PROGRESS_KEPT)
        self.started: float | None = None
        self.ended: float | None = None
        self.options: dict[str, Any] = {}
        self.outcome: dict[str, Any] | None = None
        self.exit_code: int | None = None
        self._interruptible = False                 # the child said SIGINT now reaches it
        self._stop_held = False                     # a stop asked before it did
        self._guard = threading.Lock()

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, options: Mapping[str, Any]) -> dict[str, Any]:
        with self._guard:
            if self.running:
                raise CellError(f"a hand-eye calibration already runs (pid {self.process.pid})")
            args = [*self.command, "cell", "calibrate", "--method", "hand-eye", "--ask", "broker", "--json",
                    "--cell", str(self.cell.config.path), *calibrate_arguments(options)]
            self.progress.clear()
            self.options = {key: options.get(key) for key in ("square_mm", "pattern", "placements")}
            self.outcome, self.exit_code, self.ended = None, None, None
            self._interruptible, self._stop_held = False, False
            self.started = time.time()
            self.process = subprocess.Popen(args, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                            start_new_session=True)
            process = self.process
        stdout: list[str] = []
        readers = [threading.Thread(target=self._read_progress, args=(process,), daemon=True),
                   threading.Thread(target=lambda: stdout.append(process.stdout.read()), daemon=True)]
        for reader in readers:
            reader.start()
        threading.Thread(target=self._wait, args=(process, readers, stdout), daemon=True).start()
        return self.state()

    def stop(self) -> dict[str, Any]:
        if not self.running:
            raise CellError("no hand-eye calibration runs")
        with self._guard:
            process, now = self.process, self._interruptible
            self._stop_held = not now
        if now:
            self._interrupt(process)
        else:
            self._note("stop asked: held until the calibration can take it")
        return self.state()

    def _interrupt(self, process: subprocess.Popen) -> None:
        process.send_signal(signal.SIGINT)
        self._note("stop asked: the calibration was interrupted from the board")

    def state(self) -> dict[str, Any]:
        return {"running": self.running, "pid": None if self.process is None else self.process.pid,
                "started_unix_s": self.started, "ended_unix_s": self.ended, "options": self.options,
                "progress": list(self.progress), "outcome": self.outcome, "exit_code": self.exit_code}

    def _note(self, text: str, raw: bool = False) -> None:
        self.progress.append({"at": time.time(), "text": text, **({"raw": True} if raw else {})})

    def _read_progress(self, process: subprocess.Popen) -> None:
        for line in process.stderr:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                said = json.loads(line)
            except ValueError:
                said = None
            if isinstance(said, dict) and said.get("interruptible") is True:
                with self._guard:
                    held = self.process is process and self._stop_held
                    if self.process is process:
                        self._interruptible, self._stop_held = True, False
                if held:
                    self._interrupt(process)
            elif isinstance(said, dict) and isinstance(said.get("progress"), str):
                self._note(said["progress"])
            else:
                self._note(line, raw=True)                 # a warning or a traceback: shown as it is

    def _wait(self, process: subprocess.Popen, readers: list[threading.Thread], stdout: list[str]) -> None:
        code = process.wait()
        for reader in readers:
            reader.join(timeout=5)
        outcome = None
        for candidate in reversed((stdout[0] if stdout else "").strip().splitlines()):
            try:
                outcome = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(outcome, dict):
                break
            outcome = None
        if outcome is None:
            outcome = {"ok": False, "error": f"ended (exit code {code}) without an answer"
                                             + (f": {self.progress[-1]['text']}" if self.progress else "")}
        with self._guard:
            if self.process is process:
                self.outcome, self.exit_code, self.ended = _plain(outcome), code, time.time()


class Board:
    def __init__(self, cell: Cell, depth_service: str | None = None) -> None:
        self.cell = cell
        self.hardware = threading.Lock()
        self.previous: dict[str, tuple[float, int | None, int | None]] = {}
        self.arms: dict[str, Any] | None = None
        self.wrist: dict[str, Any] | None = None
        self.live = LiveCell(cell, depth_service)
        self.holder = actions.Holder(cell)
        self.last_reset: dict[str, Any] | None = None
        self.calibration = Calibration(cell)
        self._kinematics = None

    # ------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        now = time.time()
        buses = {}
        for arm, channel in self.cell.adapter.buses.items():
            link = self.cell.bus.interface(channel)
            last = self.previous.get(channel)
            if link.get("present") and last is not None and now > last[0]:
                span = now - last[0]
                link["rx_per_s"] = None if last[1] is None else (link["rx_packets"] - last[1]) / span
                link["tx_per_s"] = None if last[2] is None else (link["tx_packets"] - last[2]) / span
            self.previous[channel] = (now, link.get("rx_packets"), link.get("tx_packets"))
            link["receivers"] = self.cell.bus.receivers(channel)
            buses[arm] = {"channel": channel, **link}
        try:
            calibration, calibration_error = self.cell.calibration(), None
        except (CellError, ValueError) as exc:
            calibration, calibration_error = {}, str(exc)
        return _plain({
            "now": now, "lock_holder": holder(), "buses": buses, "cell_id": self.cell.config.cell_id,
            "motion_processes": motion_processes(self.cell.adapter.foreign_controller_markers,
                                                 ignore=self.cell.config.ignore_processes),
            "calibration": self._calibration_summary(calibration, calibration_error),
            "camera_roles": list(self.cell.config.cameras), "joint_names": list(self.cell.adapter.joint_names),
            "facts": [list(f) for f in self.cell.adapter.facts()] + [["相机", " / ".join(
                f"{role} {camera.serial}" for role, camera in self.cell.config.cameras.items())]],
            "arms": self.arms, "wrist_check": self.wrist, "live": self.live.summary(),
            "limits": {arm: self._kin().arm_limits[arm] for arm in self.cell.adapter.arms},
            "holder": {"active": self.holder.active, "mode": self.holder.mode, "since": self.holder.since,
                       "note": self.holder.note},
            "last_reset": self.last_reset, "approvals": approval.pending(), "mode": approval.mode_state(),
            "calibrate": self.calibration.state(),
        })

    def _calibration_summary(self, value: dict[str, Any], error: str | None) -> dict[str, Any]:
        fixed = {role: {"method": c.get("method"), "provisional": bool(c.get("provisional")),
                        "median_residual_mm": None if c.get("median_residual_m") is None else 1000 * c["median_residual_m"],
                        "inlier_fraction": c.get("inlier_fraction"), "captured_unix_s": c.get("captured_unix_s")}
                 for role, c in (value.get("cameras") or {}).items()}
        path = self.cell.config.calibration_path
        return {"path": str(path), "exists": path.is_file(), "error": error,
                "serials": self.cell.config.serials, "fixed_cameras": fixed, "table": value.get("table"),
                "gripper_limits": value.get("gripper_limits") or {}, "rest_reference": value.get("rest_reference"),
                "rest_reference_unix_s": value.get("rest_reference_unix_s"), "notes": value.get("notes") or [],
                "updated_unix_s": value.get("updated_unix_s")}

    # ------------------------------------------------------------ hardware
    def _kin(self):
        if self._kinematics is None:
            self._kinematics = self.cell.kinematics()
        return self._kinematics

    def read_arms(self) -> dict[str, Any]:
        with self.hardware, CellLock("cell board (reading the arms)"):
            readings = self.cell.read_arms()
        kinematics = self._kin()
        open_m = self.cell.adapter.finger_open_m
        kinematics.set_state({arm: r.q for arm, r in readings.items()},
                             {arm: (r.gripper_opening if r.gripper_opening is not None else 1.0) * open_m
                              for arm, r in readings.items()})
        calibration = self.cell.calibration()
        joints = joint_sanity({arm: r.q for arm, r in readings.items()}, kinematics.arm_limits,
                              calibration.get("rest_reference"))
        arms = {arm: {"q": r.q, "gripper_raw_rad": r.gripper_raw, "gripper_opening": r.gripper_opening,
                      "faults": r.faults, "status": r.status, "max_temperature_c": r.max_temperature_c,
                      "torque_nm": r.torque_nm, "motors": r.motors, "tcp_xyz_m": kinematics.tcp(arm)[:3, 3],
                      "limits_rad": kinematics.arm_limits[arm], "chain": kinematics.chain(arm)}
                for arm, r in readings.items()}
        rest = calibration.get("rest_reference")
        if rest:
            kinematics.set_state({arm: rest[arm] for arm in arms})
            for arm in arms:
                arms[arm]["rest_chain"] = kinematics.chain(arm)
        self.arms = _plain({"at": time.time(), "arms": arms, "joints": joints})
        return self.arms

    def wrist_check(self) -> dict[str, Any]:
        with self.hardware:
            self.wrist = _plain(tools.wrist_check(self.cell))
        return self.wrist

    def start_calibration(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """The hand-eye calibration needs the arms and the cameras to itself: never over a holder or a run."""

        if self.holder.active:
            raise CellError(f"the board holds the arms ({self.holder.mode}): home them and release them first")
        if self.calibration.running:
            return self.calibration.start(options)      # refused, with its pid
        self.live.stop()
        taken = holder()
        if taken:
            raise CellError(f"the cell is held by {taken}: end it first (a play run: /quit in its terminal)")
        return self.calibration.start(options)


ANSWER_PATH = re.compile(r"/api/approval/([^/]+)/(approve|decline)")
MODE_PATH = re.compile(r"/api/mode/([a-z]+)")
MAX_BODY_BYTES = 8192


def answer(request_id: str, verdict: str, body: bytes) -> str:
    """The page's 批准 / 拒绝 for one question, by the id it showed."""

    try:
        data = json.loads(body or b"{}")
    except ValueError:
        raise CellError("the answer's body is not JSON") from None
    reason = data.get("reason") if isinstance(data, dict) else None
    value = data.get("value") if isinstance(data, dict) else None
    if any(v is not None and not isinstance(v, str) for v in (reason, value)):
        raise CellError("reason and value must be text")
    return approval.answer(request_id, verdict == "approve", reason, by="cell board", value=value)


def _options(body: bytes) -> dict[str, Any]:
    try:
        data = json.loads(body or b"{}")
    except ValueError:
        raise CellError("the request's body is not JSON") from None
    if not isinstance(data, dict):
        raise CellError("the request's body is not a JSON object")
    return data


# Verbs that need the cell free of the board's own live view.
NEEDS_FREE_CELL = {"release", "release-all", "float", "hold", "cameras"}


def reset(board: Board, action: str) -> Any:
    """One operator verb from the page or from ``ora cell``. The caller asked the person first
    for every verb that can make an arm limp; ``home`` needs no question."""

    yes = lambda _question: True  # noqa: E731 - confirmed by the caller
    cell = board.cell
    if action in {"kill"} and f"pid {os.getpid()})" in (holder() or ""):
        raise CellError("the cell is held by this board: use home or release instead")
    if (action in NEEDS_FREE_CELL or action == "home") and not board.holder.active:
        board.live.stop()
    verbs = {"diagnose": lambda: actions.diagnose(cell), "home": lambda: actions.home(cell, board.holder),
             "go-home": lambda: actions.home(cell, board.holder),
             "kill": lambda: actions.kill(yes), "stop": actions.stop, "estop": lambda: actions.estop(cell),
             "float": lambda: board.holder.start("float"), "hold": lambda: board.holder.start("hold"),
             "release": board.holder.release if board.holder.active else (lambda: actions.release_all(cell, yes)),
             "release-held": board.holder.release, "release-all": lambda: actions.release_all(cell, yes),
             "cameras": lambda: actions.reset_cameras(cell), "can": lambda: actions.reset_buses(cell)}
    if action not in verbs:
        raise CellError(f"unknown verb {action!r}")
    if action == "estop" and board.holder.active:
        board.holder.release()
    result = verbs[action]()
    board.last_reset = _plain({"action": action, "at": time.time(), "result": result})
    return result


def make_handler(board: Board):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: Any, code: int = 200) -> None:
            self._send(code, json.dumps(value).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            path, _, query = self.path.partition("?")
            if path in {"/", "/index.html"}:
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json(board.status())
            elif path == "/api/alerts":
                self._json(_plain(cell_alerts.current()))
            elif path == "/api/approval":
                self._json(_plain({"pending": approval.pending(), "now": time.time(),
                                   "auto_approve": approval.auto_approve_state(), "mode": approval.mode_state(),
                                   "self_answered": self_answers()}))
            elif path == "/api/live":
                board.live.touch()
                self._json(board.status()["live"])
            elif path == "/api/cloud":
                board.live.touch()
                stride = int(dict(item.split("=", 1) for item in query.split("&") if "=" in item).get("stride", 4))
                self._send(200, board.live.cloud(stride), "application/octet-stream")
            elif path == "/api/model":
                board.live.touch()
                version, body = board.live.model()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("X-Model-Version", str(version))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path.startswith("/frame/"):
                board.live.touch()
                role, _, kind = path.removeprefix("/frame/").removesuffix(".jpg").partition("/")
                frame = board.live.jpeg(role, kind or "rgb")
                if frame is None:
                    self._send(404, b"no frame", "text/plain")
                else:
                    self._send(200, frame[1], "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            actions = {"/api/arms": board.read_arms, "/api/wrist-check": board.wrist_check,
                       "/api/live/start": board.live.start, "/api/live/stop": board.live.stop,
                       "/api/completion/on": lambda: board.live.set_completion(True),
                       "/api/completion/off": lambda: board.live.set_completion(False)}
            answering = ANSWER_PATH.fullmatch(self.path)
            if answering is not None:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY_BYTES:
                    return self._json({"error": "the answer is too long"}, 413)
                body = self.rfile.read(length)
                action = lambda: answer(answering.group(1), answering.group(2), body)  # noqa: E731
            elif self.path in ("/api/calibrate/start", "/api/calibrate/stop"):
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY_BYTES:
                    return self._json({"error": "the request is too long"}, 413)
                body = self.rfile.read(length)
                action = (lambda: board.start_calibration(_options(body))) if self.path.endswith("/start") \
                    else board.calibration.stop
            elif self.path == "/api/live/start" and board.calibration.running:
                return self._json({"error": "the hand-eye calibration uses the cameras now"}, 409)
            elif MODE_PATH.fullmatch(self.path):
                mode = MODE_PATH.fullmatch(self.path).group(1)
                action = lambda: approval.switch_mode(mode, by="cell board")  # noqa: E731
            elif self.path in ("/api/auto-approve/on", "/api/auto-approve/off"):
                on = self.path.endswith("/on")
                action = lambda: approval.switch_auto_approve(on, by="cell board")  # noqa: E731
            elif self.path.startswith("/api/reset/"):
                action = lambda: reset(board, self.path.removeprefix("/api/reset/"))  # noqa: E731
            elif self.path in actions:
                action = actions[self.path]
                if not board.hardware.acquire(blocking=False):
                    return self._json({"error": "the board is already reading the hardware"}, 409)
                board.hardware.release()
            else:
                return self._send(404, b"not found", "text/plain")
            try:
                self._json({"ok": True, "result": action()})
            except CellError as exc:
                self._json({"error": str(exc)}, 409)
            except Exception as exc:  # noqa: BLE001 - the page shows what went wrong
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def log_message(self, *_: Any) -> None:
            pass

    return Handler


def serve(cell: Cell, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT, depth_service: str | None = None) -> int:
    depth_service = depth_service or cell.config.perception.get("depth_completion")
    server = ThreadingHTTPServer((host, port), make_handler(Board(cell, depth_service)))
    print(f"cell board for {cell.config.cell_id}: http://{'localhost' if host == '127.0.0.1' else host}:{port}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
