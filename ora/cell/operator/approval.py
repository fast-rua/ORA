"""Questions a robot's worker asks the person at the cell, answered from any front end.

A worker that drives a real cell has no terminal: it runs in its own session with stdin
closed, so it cannot ask on ``/dev/tty``. It asks here instead. ``ask`` posts one request
and waits for its answer; the cell board, ``ora cell approve|decline`` and the play
terminal's ``/approve`` and ``/decline`` read ``pending()`` and ``answer`` it by id.

Everything is a file in ``directory()`` (``$ORA_CELL_APPROVALS``, else
``ora-cell-approvals`` beside the cell lock), so any process on the cell's computer can
take part: the worker in the robot's interpreter, the board, an ``ora cell`` command and
the play terminal (Node), which reads the requests itself.

- ``<id>.request.json`` (written whole, by rename): ``schema``, ``id``, ``kind``,
  ``title`` and ``text`` in plain words for a person who is not a roboticist,
  ``details`` (the raw plan), ``created_unix_s``, ``expires_unix_s``, ``timeout_s``, the
  asker's ``pid`` and ``asker`` label.
- ``<id>.answer.json``: ``schema``, ``id``, ``approved``, ``reason``, ``by``,
  ``answered_unix_s``, the question's ``kind`` and ``created_unix_s``, and ``value`` for
  a ``value`` question. It is created exclusively
  (a hard link of a finished file), so the first answer wins and none is ever
  overwritten.

The kinds: a worker asks ``enable`` (power the arms), ``motion`` (run one plan) and
``release`` (let go where the arms are). A guided setup step (``ora cell calibrate
--method hand-eye``) also asks ``task`` (do something by hand, such as placing a board,
then approve; decline to stop), ``value`` (approve with a ``value``, text the asker
parses, such as a measured length; its ``details`` may name a ``unit`` and an
``example``) and ``write`` (save what was measured). Only a ``value`` question takes a
value, and its approval needs one.

A request is pending while it has no answer, has not expired and its asker lives (a
request whose pid is dead is stale and ignored). An answer names the request's id, and an
id that is not pending is refused: nobody approves a plan they did not see. An asker that
gives up (timeout, or its plan no longer applies) answers its own request through the same
exclusive slot, so either the person's answer or the asker's own is the one that counts,
never both. The asker removes its request once decided; the answer stays as the record
until a later ``ask`` sweeps answers older than ``KEEP_ANSWERS_S``.

The asker polls with short sleeps, so the signal handlers of its main thread (a worker's
SIGUSR1 ``home``) still run while it waits.

Auto-approve (the operator's "YOLO" switch) is ``auto-approve.json`` in the same folder
(written whole): ``schema``, ``kinds`` (only ``motion``: enabling the arms and releasing
them are always asked), ``by``, ``since_unix_s``, ``pid`` (the one worker it answers for)
and ``run`` (a label: the run directory, or the lock holder it was set for). While it names
the asking process and the kind, ``ask`` answers itself at once: it writes the normal
answer file, ``by: "auto-approve (<by>)"`` with the question's ``kind`` and ``title``, so
the record still shows every motion and that nobody looked at it. A cancelled question is
still declined. A worker's start (``start_run``) clears any switch and sets one for itself
only when its run was started with it (``ora play --auto-approve``), so a switch never
carries into another run; ``switch_auto_approve`` turns it on for the worker that holds
the cell in ``rig_mode on``, or off; ``auto_approve_state`` reads it (the board, ``ora cell
auto-approve status``). A switch whose worker ended is stale and answers nothing.

The permission mode is what a front end shows and switches (``mode_state``,
``switch_mode``): ``ask`` (every question waits for a person), ``safe`` (a motion whose
plan stays inside ``SAFE_LIMITS`` answers itself, any other waits) or ``auto`` (the
auto-approve switch above). ``ask`` and ``safe`` are ``mode.json`` in the same folder
(``schema``, ``mode``, ``by``, ``since_unix_s``, ``safe_limits``): they belong to the cell,
not to a run, so they can be chosen before a worker starts and they stay across runs.
``auto`` belongs to one running worker, as above, and falls back to the cell's mode when
that worker ends. Whatever the mode, enabling the arms and releasing them are asked. A
self-answered motion is recorded like any answer, ``by: "safe mode (...)"`` or
``"auto-approve (...)"``.

The operator's *stop here* (``request_stop_motion``, ``ora cell stop-motion``) is
``stop-motion.json`` in the same folder: ``pid`` (the program holding the cell),
``by`` and ``at``. The worker reads it at every control tick (``take_stop_request``) and,
for its own pid, ends the motion it is executing where the arms are and keeps them held
there; the agent gets a fresh observation. A motion that is still being planned when it
arrives is refused at its approval, before anything moves (``operator_declined``, the
reason telling the agent to read the operator's message). ``withdraw_stop_request``
takes back one nobody read (the agent answered in words). A request older than
STOP_REQUEST_S is ignored.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..errors import CellError

REQUEST_SCHEMA = "ora.cell-approval-request.v1"
ANSWER_SCHEMA = "ora.cell-approval-answer.v1"
KINDS = ("enable", "motion", "release", "task", "value", "write")
POLL_S = 0.1
KEEP_ANSWERS_S = 3600.0
REASON_MAX_CHARS = 500
VALUE_MAX_CHARS = 200
_ID = re.compile(r"[0-9a-f]{8}")
AUTO_SCHEMA = "ora.cell-auto-approve.v1"
AUTO_FILE = "auto-approve.json"
#: The only questions a switch may answer: enabling and releasing the arms are always asked.
AUTO_KINDS = ("motion",)
STOP_SCHEMA = "ora.cell-stop-motion.v1"
STOP_FILE = "stop-motion.json"
STOP_REQUEST_S = 10.0
MODE_SCHEMA = "ora.cell-permission-mode.v1"
MODE_FILE = "mode.json"
MODES = ("ask", "safe", "auto")
#: A motion plan answers itself in ``safe`` mode only inside all of these (the plan summary
#: a worker attaches: ``duration_s``, ``max_joint_travel_rad``, ``min_table_distance_m``,
#: ``min_inter_arm_distance_m``); a plan that does not state one of them is asked.
SAFE_LIMITS = {"max_duration_s": 10.0, "max_joint_travel_rad": 0.6, "min_table_distance_m": 0.03,
               "min_inter_arm_distance_m": 0.15}
#: A play worker driving the real arms, by its cell-lock label (``RealRig.label``).
_ON_WORKER = re.compile(r"ORA (?:YAM )?worker .*rig_mode on\b.*\(pid (\d+)\)", re.IGNORECASE)


def directory() -> Path:
    """Where requests and answers live: ``$ORA_CELL_APPROVALS``, else beside the cell lock."""

    named = os.environ.get("ORA_CELL_APPROVALS")
    if named:
        return Path(named).expanduser()
    from .. import lock

    return lock.LOCK_PATH.parent / "ora-cell-approvals"


# ---------------------------------------------------------------- the asking side
def ask(kind: str, title: str, text: str, details: Mapping[str, Any] | None = None, *, timeout_s: float,
        asker: str = "", cancel: Callable[[], str | None] | None = None,
        poll_s: float = POLL_S) -> tuple[bool, str | None]:
    """Post a question and wait for its answer: ``(True, None)`` or ``(False, reason)``.

    ``cancel`` is called between polls; a reason from it withdraws the question (declined
    with that reason, whatever was answered meanwhile). No answer in ``timeout_s`` is a
    decline. A question that cannot be posted is a decline too: nothing is approved by
    default.
    """

    return _decision(_ask(kind, title, text, details, timeout_s=timeout_s, asker=asker, cancel=cancel, poll_s=poll_s))


def ask_record(kind: str, title: str, text: str, details: Mapping[str, Any] | None = None, *, timeout_s: float,
               asker: str = "", cancel: Callable[[], str | None] | None = None,
               poll_s: float = POLL_S) -> dict[str, Any]:
    """``ask``, returning the answer record itself: ``approved``, ``reason``, ``by``, the question's
    ``id``, ``kind`` and ``created_unix_s`` and the ``answered_unix_s``, for a caller that keeps them."""

    return _ask(kind, title, text, details, timeout_s=timeout_s, asker=asker, cancel=cancel, poll_s=poll_s)


def ask_value(title: str, text: str, details: Mapping[str, Any] | None = None, *, timeout_s: float,
              asker: str = "", cancel: Callable[[], str | None] | None = None,
              poll_s: float = POLL_S) -> tuple[str | None, str | None]:
    """Post a ``value`` question and wait: ``(value, None)``, or ``(None, reason)`` when declined."""

    record = _ask("value", title, text, details, timeout_s=timeout_s, asker=asker, cancel=cancel, poll_s=poll_s)
    approved, reason = _decision(record)
    return (str(record.get("value")), None) if approved else (None, reason)


def _ask(kind: str, title: str, text: str, details: Mapping[str, Any] | None, *, timeout_s: float, asker: str,
         cancel: Callable[[], str | None] | None, poll_s: float) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"unknown approval kind {kind!r}")
    asked = time.time()
    record = _ask_once(kind, title, text, details, timeout_s=timeout_s, asker=asker, cancel=cancel, poll_s=poll_s)
    return {"kind": kind, "created_unix_s": asked, **record}


def _ask_once(kind: str, title: str, text: str, details: Mapping[str, Any] | None, *, timeout_s: float, asker: str,
              cancel: Callable[[], str | None] | None, poll_s: float) -> dict[str, Any]:
    folder = directory()
    request_id = secrets.token_hex(4)
    automatic = _auto_by(folder, kind)
    if automatic is not None:
        return _auto_answer(folder, request_id, kind, title, f"auto-approve ({automatic})", cancel)
    safe = _safe_by(folder, kind, details)
    if safe is not None:
        return _auto_answer(folder, request_id, kind, title, safe, cancel)
    request_path = folder / f"{request_id}.request.json"
    now = time.time()
    request = {"schema": REQUEST_SCHEMA, "id": request_id, "kind": kind, "title": title, "text": text,
               "details": _plain(dict(details or {})), "created_unix_s": now, "expires_unix_s": now + timeout_s,
               "timeout_s": timeout_s, "pid": os.getpid(), "asker": asker}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _sweep(folder)
        _write_whole(request_path, request)
    except OSError as exc:
        return _answer_record(request_id, False, f"the question could not be posted in {folder} ({exc}): "
                              "nothing was approved", "the asker")
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            answer = _read(folder / f"{request_id}.answer.json")
            if answer is not None:
                return answer
            withdrawn = cancel() if cancel is not None else None
            if withdrawn:
                _claim(folder, request_id, approved=False, reason=withdrawn, by="withdrawn by the asker")
                return _answer_record(request_id, False, withdrawn, "withdrawn by the asker")
            if time.monotonic() >= deadline:
                return _claim(folder, request_id, approved=False, reason=f"no answer in {timeout_s:g} s", by="timeout")
            time.sleep(poll_s)
    finally:
        request_path.unlink(missing_ok=True)


def _decision(answer: Mapping[str, Any]) -> tuple[bool, str | None]:
    if answer.get("approved") is True:
        return True, None
    return False, answer.get("reason") or f"declined ({answer.get('by') or 'the operator'})"


def _claim(folder: Path, request_id: str, *, approved: bool, reason: str | None, by: str) -> dict[str, Any]:
    """The asker's own answer, unless someone answered first: then theirs."""

    record = _answer_record(request_id, approved, reason, by)
    try:
        if _create_exclusive(folder / f"{request_id}.answer.json", record):
            return record
    except OSError:
        return record                    # not recorded; the asker still decides by it
    return _read(folder / f"{request_id}.answer.json") or record


def _sweep(folder: Path) -> None:
    """Forget requests whose asker died, and answers and temporary files older than KEEP_ANSWERS_S."""

    now = time.time()
    for path in folder.glob("*.request.json"):
        request = _read(path)
        if request is None or not _alive(request.get("pid")):
            path.unlink(missing_ok=True)
    for path in [*folder.glob("*.answer.json"), *folder.glob(".*.tmp")]:
        try:
            old = now - path.stat().st_mtime > KEEP_ANSWERS_S
        except OSError:
            continue
        if old and not (folder / path.name.replace(".answer.json", ".request.json")).exists():
            path.unlink(missing_ok=True)


def _auto_answer(folder: Path, request_id: str, kind: str, title: str, by: str,
                 cancel: Callable[[], str | None] | None) -> dict[str, Any]:
    """The switch's answer record, written as any other; one that cannot be written is a decline."""

    withdrawn = cancel() if cancel is not None else None
    if withdrawn:
        return _answer_record(request_id, False, withdrawn, "the asker")
    record = {**_answer_record(request_id, True, None, by), "kind": kind, "title": title, "created_unix_s": time.time()}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _sweep(folder)
        _create_exclusive(folder / f"{request_id}.answer.json", record)
    except OSError as exc:
        return _answer_record(request_id, False, f"the auto-approval could not be recorded in {folder} ({exc}): "
                              "nothing was approved", "the asker")
    return record


# ---------------------------------------------------------------- auto-approve
def start_run(*, auto_kinds: Sequence[str] = (), by: str = "", run: str | None = None) -> dict[str, Any]:
    """At a worker's start: clear any switch, and set one for this process when its run asks for it."""

    kinds = _auto_kinds(auto_kinds)
    folder = directory()
    try:
        (folder / AUTO_FILE).unlink(missing_ok=True)
    except OSError:
        pass                             # another pid's switch answers nothing here anyway
    if kinds:
        _set_switch(folder, kinds, by=by, pid=os.getpid(), run=run)
    return auto_approve_state(folder)


def switch_auto_approve(on: bool, *, by: str = "operator", kinds: Sequence[str] = AUTO_KINDS) -> dict[str, Any]:
    """The operator's switch: on for the worker that drives the arms now (``CellError`` without one), or off."""

    folder = directory()
    if not on:
        try:
            (folder / AUTO_FILE).unlink(missing_ok=True)
        except OSError as exc:
            raise CellError(f"auto-approve could not be turned off in {folder} ({exc})") from None
        return auto_approve_state(folder)
    from .. import lock

    holder = lock.holder(lock.LOCK_PATH) or ""
    found = _ON_WORKER.search(holder)
    if found is None:
        raise CellError("no play worker drives the arms (rig_mode on) now"
                        + (f"; the cell is held by {holder}" if holder else "")
                        + ": start one with `ora play --env yam --rig-mode on --operator-present --auto-approve`, "
                          "or turn this on while it runs")
    _set_switch(folder, _auto_kinds(kinds), by=by, pid=int(found.group(1)), run=holder)
    return auto_approve_state(folder)


def auto_approve_state(folder: Path | None = None) -> dict[str, Any]:
    """The switch as a front end shows it: ``on`` only while its worker lives. Reads only."""

    folder = directory() if folder is None else folder
    flag = _read(folder / AUTO_FILE)
    state: dict[str, Any] = {"on": False, "kinds": [], "by": None, "since_unix_s": None, "pid": None, "run": None,
                             "stale": None}
    if flag is None or flag.get("schema") != AUTO_SCHEMA:
        return state
    kinds = flag.get("kinds")
    kinds = [kind for kind in kinds if kind in AUTO_KINDS] if isinstance(kinds, list) else []
    state.update({"kinds": kinds, "by": flag.get("by"), "since_unix_s": flag.get("since_unix_s"),
                  "pid": flag.get("pid"), "run": flag.get("run")})
    if not kinds:
        state["stale"] = "it names no question that can be auto-approved"
    elif not _alive(flag.get("pid")):
        state["stale"] = f"the worker it was set for (pid {flag.get('pid')}) has ended"
    state["on"] = state["stale"] is None
    return state


def _auto_kinds(kinds: Sequence[str]) -> list[str]:
    chosen = list(dict.fromkeys(str(kind) for kind in kinds))
    if any(kind not in AUTO_KINDS for kind in chosen):
        raise CellError(f"only {' and '.join(AUTO_KINDS)} questions can be auto-approved, not {chosen}: "
                        "enabling the arms and releasing them are always asked")
    return chosen


def _set_switch(folder: Path, kinds: list[str], *, by: str, pid: int, run: str | None) -> None:
    flag = {"schema": AUTO_SCHEMA, "kinds": kinds, "by": by, "since_unix_s": time.time(), "pid": pid, "run": run}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _write_whole(folder / AUTO_FILE, flag)
    except OSError as exc:
        raise CellError(f"auto-approve could not be turned on in {folder} ({exc})") from None


def _auto_by(folder: Path, kind: str) -> str | None:
    """Who set the switch that answers this process's ``kind`` question, or None."""

    flag = _read(folder / AUTO_FILE)
    if (flag is None or flag.get("schema") != AUTO_SCHEMA or kind not in AUTO_KINDS
            or not isinstance(flag.get("kinds"), list) or kind not in flag["kinds"] or flag.get("pid") != os.getpid()):
        return None
    return str(flag.get("by") or "operator")


# ---------------------------------------------------------------- the permission mode
def mode_state(folder: Path | None = None) -> dict[str, Any]:
    """The cell's permission mode as a front end shows it. Reads only."""

    folder = directory() if folder is None else folder
    flag = _read(folder / MODE_FILE)
    chosen = flag if flag is not None and flag.get("schema") == MODE_SCHEMA and flag.get("mode") in ("ask", "safe") \
        else None
    state: dict[str, Any] = {"mode": chosen["mode"] if chosen else "ask", "by": chosen and chosen.get("by"),
                             "since_unix_s": chosen and chosen.get("since_unix_s"), "pid": None, "run": None,
                             "stale": None, "safe_limits": _limits(chosen)}
    auto = auto_approve_state(folder)
    if auto["on"]:
        state.update({"mode": "auto", "by": auto["by"], "since_unix_s": auto["since_unix_s"], "pid": auto["pid"],
                      "run": auto["run"], "falls_back_to": state["mode"]})
    elif auto["stale"] and auto["pid"] is not None:
        state["stale"] = f"auto-approve was set for a worker that has ended: {auto['stale']}"
    return state


def switch_mode(mode: str, *, by: str = "operator") -> dict[str, Any]:
    """``ask`` or ``safe`` for the cell (any time, also with no worker), or ``auto`` for the worker
    that drives the arms now (``CellError`` without one)."""

    if mode not in MODES:
        raise CellError(f"unknown permission mode {mode!r}: one of {', '.join(MODES)}")
    folder = directory()
    if mode == "auto":
        switch_auto_approve(True, by=by)
        return mode_state(folder)
    switch_auto_approve(False, by=by)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _write_whole(folder / MODE_FILE, {"schema": MODE_SCHEMA, "mode": mode, "by": by, "since_unix_s": time.time(),
                                          "safe_limits": dict(SAFE_LIMITS)})
    except OSError as exc:
        raise CellError(f"the permission mode could not be set in {folder} ({exc})") from None
    return mode_state(folder)


def _limits(flag: Mapping[str, Any] | None) -> dict[str, float]:
    limits = dict(SAFE_LIMITS)
    stored = (flag or {}).get("safe_limits")
    if isinstance(stored, Mapping):
        # A stored limit may only be stricter than the built-in one.
        for key, value in stored.items():
            if key in limits and isinstance(value, (int, float)) and math.isfinite(value):
                limits[key] = min(limits[key], value) if key.startswith("max_") else max(limits[key], value)
    return limits


def _safe_by(folder: Path, kind: str, details: Mapping[str, Any] | None) -> str | None:
    """``safe`` mode's answer to a motion inside the limits, or None (asked)."""

    flag = _read(folder / MODE_FILE)
    if kind != "motion" or flag is None or flag.get("schema") != MODE_SCHEMA or flag.get("mode") != "safe":
        return None
    limits, plan = _limits(flag), dict(details or {})

    def values(key: str) -> list[float]:
        value = plan.get(key)
        found = list(value.values()) if isinstance(value, Mapping) else [value]
        return [float(v) for v in found if isinstance(v, (int, float)) and not isinstance(v, bool)]

    duration, travel = values("duration_s"), values("max_joint_travel_rad")
    table, apart = values("min_table_distance_m"), values("min_inter_arm_distance_m")
    if not (duration and travel and apart and table):
        return None
    if (max(duration) <= limits["max_duration_s"] and max(travel) <= limits["max_joint_travel_rad"]
            and min(table) >= limits["min_table_distance_m"] and min(apart) >= limits["min_inter_arm_distance_m"]):
        return (f"safe mode ({max(duration):.1f} s, {max(travel):.2f} rad, {min(table) * 100:.0f} cm above the table, "
                f"{min(apart) * 100:.0f} cm between the arms)")
    return None


# ---------------------------------------------------------------- stop here
def request_stop_motion(*, by: str = "operator") -> dict[str, Any]:
    """Ask the program holding the cell to end its motion where the arms are (``CellError`` without one)."""

    from .. import lock

    holder = lock.holder(lock.LOCK_PATH) or ""
    found = re.search(r"\(pid (\d+)\)", holder)
    if found is None:
        raise CellError("no program holds the arms now: nothing is moving")
    folder = directory()
    record = {"schema": STOP_SCHEMA, "pid": int(found.group(1)), "by": by, "at": time.time(), "holder": holder}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _write_whole(folder / STOP_FILE, record)
    except OSError as exc:
        raise CellError(f"the stop could not be sent in {folder} ({exc})") from None
    return record


def withdraw_stop_request() -> bool:
    """Take back a stop nobody has read yet (the agent answered in words, no motion came); True if one was."""

    path = directory() / STOP_FILE
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def take_stop_request(pid: int | None = None, folder: Path | None = None) -> dict[str, Any] | None:
    """This process's pending stop, consumed; None when there is none (one stat per call)."""

    path = (directory() if folder is None else folder) / STOP_FILE
    if not path.exists():
        return None
    record = _read(path)
    if record is None or record.get("schema") != STOP_SCHEMA or record.get("pid") != (os.getpid() if pid is None else pid):
        return None
    path.unlink(missing_ok=True)
    if time.time() - float(record.get("at") or 0.0) > STOP_REQUEST_S:
        return None
    return record


# ---------------------------------------------------------------- the answering side
def pending(folder: Path | None = None) -> list[dict[str, Any]]:
    """The questions waiting for an answer, oldest first. Reads only."""

    folder = directory() if folder is None else folder
    if not folder.is_dir():
        return []
    now = time.time()
    waiting = []
    for path in folder.glob("*.request.json"):
        request = _read(path)
        if request is not None and _is_pending(folder, request, now):
            waiting.append(request)
    return sorted(waiting, key=lambda request: request.get("created_unix_s") or 0.0)


def answer(request_id: str, approved: bool, reason: str | None = None, by: str = "operator",
           value: str | None = None) -> str:
    """Answer one pending question by its id; an id that is not pending is refused (``CellError``),
    and so is a ``value`` question approved without a value, or a value for any other question."""

    request_id = str(request_id).strip().lower()
    if not _ID.fullmatch(request_id):
        raise CellError(f"{request_id!r} is not a request id (`ora cell pending` lists them)")
    reason = (reason or "").strip()[:REASON_MAX_CHARS] or None
    folder = directory()
    request = _read(folder / f"{request_id}.request.json")
    if request is None or not _is_pending(folder, request, time.time()):
        earlier = _read(folder / f"{request_id}.answer.json")
        if earlier is not None:
            raise CellError(f"request {request_id} was already answered: "
                            f"{'approved' if earlier.get('approved') else 'declined'} ({earlier.get('by')})")
        raise CellError(f"no question {request_id} is waiting for an answer (answered, expired or its asker ended)")
    value = None if value is None else str(value).strip()[:VALUE_MAX_CHARS]
    if request.get("kind") == "value":
        if approved and not value:
            raise CellError(f"request {request_id} asks for a value: answer it with one (`--value`)")
    elif value is not None:
        raise CellError(f"request {request_id} is a {request.get('kind')} question: it takes no value")
    record = {**_answer_record(request_id, approved, None if approved else reason, by, value if approved else None),
              "kind": request.get("kind"), "created_unix_s": request.get("created_unix_s")}
    if not _create_exclusive(folder / f"{request_id}.answer.json", record):
        earlier = _read(folder / f"{request_id}.answer.json") or {}
        raise CellError(f"request {request_id} was already answered: "
                        f"{'approved' if earlier.get('approved') else 'declined'} ({earlier.get('by')})")
    return f"{'approved' if approved else 'declined'} {request_id}: {request.get('title', '')}"


def _answer_record(request_id: str, approved: bool, reason: str | None, by: str,
                   value: str | None = None) -> dict[str, Any]:
    record = {"schema": ANSWER_SCHEMA, "id": request_id, "approved": bool(approved), "reason": reason, "by": by,
              "answered_unix_s": time.time()}
    if value is not None:
        record["value"] = value
    return record


def _is_pending(folder: Path, request: Mapping[str, Any], now: float) -> bool:
    request_id, expires = request.get("id"), request.get("expires_unix_s")
    return (isinstance(request_id, str) and _ID.fullmatch(request_id) is not None
            and isinstance(expires, (int, float)) and now < expires
            and not (folder / f"{request_id}.answer.json").exists() and _alive(request.get("pid")))


# ---------------------------------------------------------------- files and processes
def _alive(pid: Any) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                      # another user's live process
    except OSError:
        return False
    return True


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _encoded(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(_plain(dict(value)), ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")


def _write_whole(path: Path, value: Mapping[str, Any]) -> None:
    """Readers see the old file or the whole new one, never a part."""

    temporary = _temporary(path)
    temporary.write_bytes(_encoded(value))
    temporary.replace(path)


def _create_exclusive(path: Path, value: Mapping[str, Any]) -> bool:
    """Create ``path`` with the whole of ``value``, unless it exists: False then."""

    temporary = _temporary(path)
    temporary.write_bytes(_encoded(value))
    try:
        os.link(temporary, path)
    except FileExistsError:
        return False
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _plain(value: Any) -> Any:
    """JSON every reader parses: NumPy values become lists and numbers, non-finite numbers null."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)
