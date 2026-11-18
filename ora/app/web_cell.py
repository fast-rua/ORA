"""Free play's real-cell controls on `ora web`'s page: what the terminal's /approve, /decline, /yolo,
/mode, /halt, /home, /hold, /float, /release, /estop and /diagnose do, as the gateway's routes.

A run on a real cell (its plan's runtime is a real-cell backend, today `yam`, on rig_mode `read` or
`on`: ora-pi/src/play/cell.ts realCellOf) has them; any other run answers that it drives no cell.

  GET  /ora/cell                                 the cell now: the questions the robot's worker waits on
                                                 (oldest first), the auto-approve switch, the permission
                                                 mode, the latest motions that answered themselves (safe
                                                 mode or auto-approve), the motion questions answered
                                                 lately with when each was asked and answered (the page's
                                                 speed panel), the held arms' alerts and whether the
                                                 worker that held the arms ended
  POST /ora/cell/approval/<id>/approve|decline   {confirmed: true, reason?, value?}: that question, by its
                                                 id; ``value`` (a ``value`` question's answer) on approve only
  POST /ora/cell/auto-approve/on|off             {confirmed: true} turns it on; off needs nothing
  POST /ora/cell/mode/ask|safe|auto              the permission mode; auto with {confirmed: true}
  POST /ora/cell/verb/<verb>                     home, hold, diagnose; float, release, estop with
                                                 {confirmed: true}
  POST /ora/cell/stop-motion                     /halt: the running motion ends where the arms are and
                                                 they stay held there, and one the agent still plans is
                                                 refused at its approval; never asked (it holds, it never
                                                 drops). {withdraw: true} takes back a stop nobody read
                                                 (the agent answered in words: no motion came)

The page asks first exactly where the terminal asks y/N, and a request that needs the answer must
carry it. The state is read with the cell kit's own functions, the ones the cell board shows
(``approval.pending``, ``approval.auto_approve_state``, ``approval.mode_state``, ``alerts.current``) and the worker from the
cell lock's label as the terminal reads it, so the page, the terminal and the board read the same
files. Every action runs the terminal's own command line, ``scripts/ora cell <verb> --json`` (with
``--cell`` when the plan names the cell config, ``--yes`` after the page's confirmation, ``--by
"web page"`` for an answer, the switch, the mode or a stop), so `ora cell` decides, as for the terminal, whether a
verb goes through a running cell board. Nothing is proxied to the board.

Each outcome is also handed to the run's agent process (``note``: the web server's
``/api/cell-note``, which no browser reaches), which shows it in the terminal's conversation, records
the command and its output in the agent trace as the terminal's own commands are, and tells every
page. A run whose agent process has ended still takes these commands: the cell outlives the run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.orchestrator import paths

#: Runtime kinds whose rig can be a real cell, and the rig modes on the real hardware (play/cell.ts).
REAL_CELL_KINDS = frozenset({"yam"})
REAL_RIG_MODES = frozenset({"read", "on"})
#: The terminal's cell commands (play/cell.ts CELL_COMMANDS): verb -> asked first (y/N there, a dialog here).
VERBS: dict[str, bool] = {"home": False, "hold": False, "float": True, "release": True, "estop": True,
                          "diagnose": False}
#: The terminal's answers (play/cell.ts APPROVAL_COMMANDS); both are asked first.
ANSWERS = ("approve", "decline")
#: The permission modes (play/cell.ts PERMISSION_MODES); only auto is asked first.
MODES = ("ask", "safe", "auto")
#: Who answered, as the broker keeps it with each answer (the terminal's is "play terminal").
BY = "web page"
#: The motions that answered themselves the page lists: the latest few, of the last minutes.
SELF_ANSWERS_KEPT = 5
SELF_ANSWERS_S = 600.0
_SELF_BY = re.compile(r"^(?:safe mode|auto-approve) \(")
#: The answered questions the page times (its speed panel, ora-pi/web/src/lib/perf.ts), of the same minutes,
#: and the waiting questions remembered for it: a question's request file goes once it is answered.
ANSWERS_TIMED = 30
ASKED_KEPT = 200
#: Longest an `ora cell` command may run (a home drives for at most 30 s, a board takes up to 30 s to start).
COMMAND_TIMEOUT_S = 180.0
REASON_MAX_CHARS = 500
VALUE_MAX_CHARS = 200
_WORKER_HOLDER = re.compile(r"ORA (?:YAM )?worker .*\(pid (\d+)\)", re.IGNORECASE)
_REQUEST_ID = re.compile(r"^[0-9a-f]{8}$")


class CellRouteError(Exception):
    """A refused request, with its HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def real_cell(plan: Any) -> dict[str, Any] | None:
    """The real cell a plan drives, or None (a simulator, or a real-cell backend on rig_mode `sim`)."""

    runtime = plan.get("runtime") if isinstance(plan, Mapping) else None
    if not isinstance(runtime, Mapping):
        return None
    kind, mode = str(runtime.get("kind") or ""), str(runtime.get("rig_mode") or "sim")
    if kind not in REAL_CELL_KINDS or mode not in REAL_RIG_MODES:
        return None
    config = runtime.get("cell_config")
    return {"kind": kind, "rig_mode": mode,
            "cell_config": config if isinstance(config, str) and config.strip() else None}


def read_real_cell(run_root: Path) -> dict[str, Any] | None:
    try:
        return real_cell(json.loads((Path(run_root) / "plan.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError, ValueError):
        return False
    return True


def worker_pid(lock_path: Path) -> int | None:
    """The play worker holding the cell, from the lock's label (play/cell.ts cellWorkerPid); reads only."""

    try:
        found = _WORKER_HOLDER.search(Path(lock_path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return int(found.group(1)) if found else None


def self_answers(folder: Path, now: float, keep: int = SELF_ANSWERS_KEPT,
                 within_s: float = SELF_ANSWERS_S) -> list[dict[str, Any]]:
    """The motions that answered themselves lately (play/cell.ts selfAnswers), newest last: approved
    answers with the question's ``kind`` and ``title`` and ``by: "safe mode (...)"`` or ``"auto-approve
    (...)"``. Reads only."""

    found = []
    for path in Path(folder).glob("*.answer.json"):
        try:
            answer = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not (isinstance(answer, dict) and answer.get("approved") is True and isinstance(answer.get("kind"), str)
                and isinstance(answer.get("by"), str) and _SELF_BY.match(answer["by"])
                and isinstance(answer.get("id"), str) and _REQUEST_ID.fullmatch(answer["id"])):
            continue
        at = answer.get("answered_unix_s")
        if isinstance(at, (int, float)) and now - within_s <= at:
            found.append({"id": answer["id"], "kind": answer["kind"], "title": str(answer.get("title") or ""),
                          "by": answer["by"], "answered_unix_s": at})
    return sorted(found, key=lambda item: item["answered_unix_s"])[-keep:]


def timed_answers(folder: Path, now: float, asked: Mapping[str, Mapping[str, Any]], keep: int = ANSWERS_TIMED,
                  within_s: float = SELF_ANSWERS_S) -> list[dict[str, Any]]:
    """The motion questions answered lately, newest last, for the page's speed panel: ``id``, ``kind``
    (``None`` when neither the answer nor a sighting of the question says), ``by``, ``approved``,
    ``answered_unix_s`` and ``created_unix_s``: when the question was posted, from its request as
    ``asked`` saw it waiting (the request file goes once it is answered), the moment it was answered for
    one that answered itself (safe mode, auto-approve: nobody waited), else ``None``. Reads only."""

    found = []
    for path in Path(folder).glob("*.answer.json"):
        try:
            answer = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not (isinstance(answer, dict) and isinstance(answer.get("id"), str) and _REQUEST_ID.fullmatch(answer["id"])
                and isinstance(answer.get("by"), str)):
            continue
        at = answer.get("answered_unix_s")
        if not (isinstance(at, (int, float)) and now - within_s <= at):
            continue
        request = asked.get(answer["id"]) or {}
        kind = answer.get("kind") if isinstance(answer.get("kind"), str) else request.get("kind")
        if kind not in {"motion", None}:
            continue
        created = at if _SELF_BY.match(answer["by"]) else request.get("created_unix_s")
        found.append({"id": answer["id"], "kind": kind, "by": answer["by"], "approved": answer.get("approved") is True,
                      "answered_unix_s": at,
                      "created_unix_s": created if isinstance(created, (int, float)) else None})
    return sorted(found, key=lambda item: item["answered_unix_s"])[-keep:]


def outcome(code: int | None, stdout: str, stderr: str) -> tuple[bool, str]:
    """`ora cell --json`'s answer as the terminal reads it (play/cell.ts cellOutcome)."""

    answer = None
    text = stdout.strip()
    for candidate in (text, text.splitlines()[-1] if text else ""):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            answer = value
            break
    if answer is not None and answer.get("ok") is True and code == 0:
        result = answer.get("result")
        return True, "" if result is None else result if isinstance(result, str) else json.dumps(result, indent=2)
    if answer is not None and answer.get("ok") is False:
        return False, str(answer.get("error") or "the cell refused, without a reason")
    first = next((line.strip() for line in (stderr + "\n" + stdout).splitlines() if line.strip()), "no answer")
    return False, first if code is None else f"exit code {code}: {first}"


Run = Callable[[list[str], Path], tuple[int | None, str, str]]


def run_command(args: list[str], cwd: Path) -> tuple[int | None, str, str]:
    """The process, with no stdin (it can never wait on a terminal), its output kept."""

    try:
        done = subprocess.run(args, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=COMMAND_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        return None, str(exc.stdout or ""), f"no answer within {COMMAND_TIMEOUT_S:.0f} s"
    except OSError as exc:
        return None, "", str(exc)
    return done.returncode, done.stdout, done.stderr


class CellControls:
    """One run's real-cell routes. ``note`` hands each outcome to the run's agent process (best effort)."""

    def __init__(self, run_root: Path, *, project_root: Path = paths.PROJECT_ROOT, run: Run = run_command,
                 alive: Callable[[int], bool] = process_alive, note: Callable[[dict[str, Any]], None] | None = None,
                 lock_path: Path | None = None) -> None:
        self.run_root = Path(run_root)
        self.project_root = Path(project_root)
        self.run = run
        self.alive = alive
        self.note = note
        self.lock_path = lock_path
        self._guard = threading.Lock()
        self._worker: int | None = None
        self._ended: dict[str, Any] | None = None
        #: The questions seen waiting (id -> kind, created_unix_s): when each was asked, once answered.
        self._asked: dict[str, dict[str, Any]] = {}

    @property
    def cell(self) -> dict[str, Any] | None:
        return read_real_cell(self.run_root)

    def _cell(self) -> dict[str, Any]:
        cell = self.cell
        if cell is None:
            raise CellRouteError(404, "this run drives no real cell (rig_mode read or on): nothing to answer or run")
        return cell

    # -- The state ---------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        cell = self.cell
        if cell is None:
            return {"cell": None}
        from ora.cell import alerts as cell_alerts
        from ora.cell import lock as cell_lock
        from ora.cell.operator import approval

        published = cell_alerts.current()
        pid = worker_pid(self.lock_path or cell_lock.LOCK_PATH)
        holding = pid is not None and self.alive(pid)
        with self._guard:
            if holding:
                self._worker, self._ended = pid, None
            elif self._worker is not None and not self.alive(self._worker):
                # The worker this page saw holding the arms is gone: said until another one holds them.
                self._ended, self._worker = {"pid": self._worker, "seen_unix_s": time.time()}, None
            ended = self._ended
        now = time.time()
        pending = approval.pending()
        with self._guard:
            for request in pending:
                if isinstance(request.get("id"), str):
                    self._asked[request["id"]] = {"kind": request.get("kind"),
                                                  "created_unix_s": request.get("created_unix_s")}
            while len(self._asked) > ASKED_KEPT:
                del self._asked[next(iter(self._asked))]
            asked = dict(self._asked)
        folder = approval.directory()
        return {
            "cell": cell, "now": now, "by": BY,
            "pending": pending, "auto_approve": approval.auto_approve_state(),
            "mode": approval.mode_state(), "self_answered": self_answers(folder, now),
            "answers": timed_answers(folder, now, asked),
            "alerts": published.get("alerts") or [], "alerts_stale": bool(published.get("stale")),
            "worker": {"pid": pid, "alive": holding}, "worker_ended": ended,
        }

    # -- The actions -------------------------------------------------------------------------

    @staticmethod
    def _confirmed(body: Mapping[str, Any], what: str) -> None:
        if body.get("confirmed") is not True:
            raise CellRouteError(400, f"{what} is asked first: send it with confirmed true once the operator said yes")

    def answer(self, request_id: str, verb: str, body: Mapping[str, Any]) -> dict[str, Any]:
        cell = self._cell()
        if verb not in ANSWERS:
            raise CellRouteError(404, f"no answer {verb!r}: approve or decline")
        if not _REQUEST_ID.fullmatch(request_id):
            raise CellRouteError(400, f"{request_id!r} is not a request id")
        self._confirmed(body, f"/{verb}")
        reason = body.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise CellRouteError(400, "reason must be text")
        reason = (reason or "").strip()[:REASON_MAX_CHARS] or None
        value = body.get("value")
        if value is not None and not isinstance(value, str):
            raise CellRouteError(400, "value must be text")
        # `ora cell approve --value` itself refuses a value for any question but a `value` one.
        value = (value or "").strip()[:VALUE_MAX_CHARS] or None
        args = ["cell", verb, "--json", *self._cell_option(cell), "--id", request_id, "--by", BY,
                *(["--reason", reason] if verb == "decline" and reason else []),
                *(["--value", value] if verb == "approve" and value else [])]
        return self._execute(verb, verb, args, request_id=request_id)

    def auto_approve(self, state: str, body: Mapping[str, Any]) -> dict[str, Any]:
        cell = self._cell()
        if state not in {"on", "off"}:
            raise CellRouteError(404, f"auto-approve is on or off, not {state!r}")
        if state == "on":
            self._confirmed(body, "/yolo on")
        args = ["cell", "auto-approve", state, "--json", *self._cell_option(cell), "--by", BY]
        return self._execute("yolo", "auto-approve", args)

    def mode(self, mode: str, body: Mapping[str, Any]) -> dict[str, Any]:
        cell = self._cell()
        if mode not in MODES:
            raise CellRouteError(404, f"the permission mode is {', '.join(MODES)}, not {mode!r}")
        if mode == "auto":
            self._confirmed(body, "/mode auto")
        args = ["cell", "mode", mode, "--json", *self._cell_option(cell), "--by", BY]
        return self._execute("mode", "mode", args)

    def verb(self, verb: str, body: Mapping[str, Any]) -> dict[str, Any]:
        cell = self._cell()
        if verb not in VERBS:
            raise CellRouteError(404, f"no cell verb {verb!r}: {', '.join(VERBS)}")
        if VERBS[verb]:
            self._confirmed(body, f"/{verb}")
        args = ["cell", verb, "--json", *self._cell_option(cell), *(["--yes"] if VERBS[verb] else [])]
        return self._execute(verb, verb, args)

    def stop_motion(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """/halt (play/cell.ts HALT_COMMAND): ``ora cell stop-motion``; nothing to confirm. With
        ``withdraw``, ``ora cell stop-motion --withdraw`` (play/cell.ts PlanStop)."""

        cell = self._cell()
        if body.get("withdraw") is True:
            args = ["cell", "stop-motion", "--withdraw", "--json", *self._cell_option(cell)]
            return self._execute("halt", "stop-motion", args, withdraw=True)
        args = ["cell", "stop-motion", "--json", *self._cell_option(cell), "--by", BY]
        return self._execute("halt", "stop-motion", args)

    @staticmethod
    def _cell_option(cell: Mapping[str, Any]) -> list[str]:
        return [] if cell.get("cell_config") is None else ["--cell", str(cell["cell_config"])]

    def _execute(self, name: str, verb: str, args: list[str], **extra: Any) -> dict[str, Any]:
        script = self.project_root / "scripts" / "ora"
        started = time.monotonic()
        code, stdout, stderr = self.run([str(script), *args], self.project_root)
        ok, text = outcome(code, stdout, stderr)
        if ok and name in {"yolo", "mode"}:
            text = _switch_text(text)
        result = {"id": uuid.uuid4().hex, "name": name, "verb": verb, "ok": ok, "text": text,
                  "seconds": round(time.monotonic() - started, 1), "via": BY, **extra}
        if self.note is not None:
            try:
                self.note({**result, "command": _command_line(script, args), "cwd": str(self.project_root),
                           "exit_code": code, "output": stdout + stderr})
            except Exception:  # noqa: BLE001 - the command ran; the run's agent process may be gone
                pass
        return result


def _switch_text(text: str) -> str:
    """`ora cell auto-approve|mode --json`'s state, kept as JSON for the page to word."""

    try:
        return json.dumps(json.loads(text))
    except ValueError:
        return text


def _command_line(script: Path, args: list[str]) -> str:
    def quote(word: str) -> str:
        return word if re.fullmatch(r"[\w@%+=:,./-]+", word) else "'" + word.replace("'", "'\\''") + "'"

    return " ".join(quote(word) for word in [str(script), *args])


ROUTE = re.compile(r"^/ora/cell(?:/(approval)/([^/]+)/([a-z]+)|/(auto-approve)/([a-z]+)|/(verb)/([a-z]+)"
                   r"|/(mode)/([a-z]+)|/(stop-motion))?$")


def dispatch(controls: CellControls, method: str, path: str, body: bytes) -> dict[str, Any]:
    """One /ora/cell request; CellRouteError for a refusal."""

    match = ROUTE.match(path)
    if match is None:
        raise CellRouteError(404, "no such ORA cell route")
    if match.group(0) == "/ora/cell":
        if method not in {"GET", "HEAD"}:
            raise CellRouteError(405, "the cell's state is read with GET")
        return controls.state()
    if method != "POST":
        raise CellRouteError(405, "a cell command is a POST")
    try:
        data = json.loads(body or b"{}")
    except ValueError:
        raise CellRouteError(400, "the request body must be a JSON object") from None
    if not isinstance(data, dict):
        raise CellRouteError(400, "the request body must be a JSON object")
    if match.group(1):
        return controls.answer(match.group(2), match.group(3), data)
    if match.group(4):
        return controls.auto_approve(match.group(5), data)
    if match.group(8):
        return controls.mode(match.group(9), data)
    if match.group(10):
        return controls.stop_motion(data)
    return controls.verb(match.group(7), data)


__all__ = ["ANSWERS", "BY", "CellControls", "CellRouteError", "MODES", "REAL_CELL_KINDS", "REAL_RIG_MODES", "VERBS",
           "dispatch", "outcome", "read_real_cell", "real_cell", "self_answers", "timed_answers", "worker_pid"]
