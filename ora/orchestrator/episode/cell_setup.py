"""A real cell's setup before its play worker starts (``ora play --env yam --rig-mode on --setup``).

The worker takes the cell lock as it starts, and ``ora cell setup`` needs that lock to move the arms
and read the cameras, so the setup runs here: after the operator's terminal and ``ora web``'s page are
up, before the worker exists. The Supervisor runs ``scripts/ora cell setup --ask broker --json``
itself. Every question the setup asks goes through the cell's approval broker and is answered as a
worker's is: ``/approve`` and ``/decline`` in the play terminal, the page's question card, the cell
board or ``ora cell approve``.

What the operator sees are play states ``setup`` (display only), each carrying the whole view so
far, so a page that keeps only the latest state still draws it all: ``setup.phase`` (status,
running, done, failed), ``setup.steps`` (the checklist of ``ora cell setup --status --json``, each
step's ``state`` done, missing, stale, running or failed, and its ``why``), ``next``,
``ready_for_play``, ``lines`` (the latest progress lines), ``result`` or ``error`` at the end, and
``event``, what just happened: the checklist, one line the setup printed on stderr (``{"progress":
...}`` or ``{"step": ..., "state": "running"|"done"|"failed", "why"?}``) or its answer. The events
are kept in ``native/cell-setup.jsonl``. A cell with nothing left to do starts at once. A setup that fails, is
declined or is interrupted ends the run before any arm is enabled.
"""

from __future__ import annotations

import copy
import json
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..faults import SupervisorError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
#: The command that runs ``ora cell``; tests replace it.
COMMAND: Sequence[str] = (str(PROJECT_ROOT / "scripts" / "ora"),)
STATUS_TIMEOUT_S = 120.0
#: Progress lines a view keeps (the full record is native/cell-setup.jsonl).
LINES_KEPT = 8
#: How long an interrupted setup may take to bring its arms home and end before it is killed.
STOP_GRACE_S = 60.0


def run_cell_setup(loop: Any) -> None:
    """Show the cell's checklist, run what is missing, and return once the cell is ready for play."""

    cell = loop.plan["runtime"].get("cell_config")
    cell_args = [] if not cell else ["--cell", str(cell)]
    status = _answer(_run([*COMMAND, "cell", "setup", "--status", "--json", *cell_args], STATUS_TIMEOUT_S),
                     "ora cell setup --status")
    view: dict[str, Any] = {"phase": "status", "lines": []}
    _tell(loop, view, {"status": status})
    if status.get("next") is None:
        _tell(loop, view, {"phase": "done", "skipped": True})
        return
    view["phase"] = "running"
    try:
        result = _answer(_follow(loop, view, [*COMMAND, "cell", "setup", "--ask", "broker", "--json", *cell_args]),
                         "ora cell setup")
    except SupervisorError as exc:
        _tell(loop, view, {"phase": "failed", "error": str(exc)})
        raise
    failed = result.get("failed")
    _tell(loop, view, {"phase": "failed" if failed else "done", "result": result})
    if failed:
        raise SupervisorError(f"the cell's setup stopped at {failed}: {result.get('why') or 'no reason given'}; "
                              "no arm was enabled")
    if loop.interrupted:
        raise SupervisorError("the run was stopped during the cell's setup; no arm was enabled")


def _tell(loop: Any, view: dict[str, Any], event: Mapping[str, Any]) -> None:
    """Record one event, fold it into the view and announce the view."""

    record = {"unix_s": time.time(), **event}
    if "status" in event:
        _tell_status(view, event["status"])
    if "step" in event:
        step = next((step for step in view.setdefault("steps", []) if step.get("id") == event["step"]), None)
        if step is None:
            step = {"id": event["step"]}
            view["steps"].append(step)
        step.update(state=event.get("state", step.get("state")), why=event.get("why"))
    if "progress" in event:
        view["lines"] = (view["lines"] + [str(event["progress"])])[-LINES_KEPT:]
    if "result" in event:
        if isinstance(event["result"].get("status"), Mapping):
            _tell_status(view, event["result"]["status"])
        view["result"] = {key: value for key, value in event["result"].items() if key != "status"}
    view.update({key: event[key] for key in ("phase", "error", "skipped") if key in event})
    try:
        native = Path(loop.run_root) / "native"
        native.mkdir(parents=True, exist_ok=True)
        with (native / "cell-setup.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        loop.record_auxiliary_failure("cell_setup_record", exc)
    loop._announce("setup", setup=copy.deepcopy({**view, "event": dict(event)}))


def _tell_status(view: dict[str, Any], status: Mapping[str, Any]) -> None:
    view.update(steps=[dict(step) for step in status.get("steps") or []], next=status.get("next"),
                ready_for_play=status.get("ready_for_play"))


def _run(args: Sequence[str], timeout_s: float) -> tuple[int | None, str, str]:
    try:
        done = subprocess.run(list(args), cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SupervisorError(f"{' '.join(args[-5:])} did not answer: {exc}") from None
    return done.returncode, done.stdout, done.stderr


def _follow(loop: Any, view: dict[str, Any], args: Sequence[str]) -> tuple[int | None, str, str]:
    """Run the setup, telling the operator each stderr line as it comes; SIGINT once the run is stopped."""

    try:
        process = subprocess.Popen(list(args), cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                   start_new_session=True)
    except OSError as exc:
        raise SupervisorError(f"ora cell setup could not start: {exc}") from None
    lines: queue.Queue[str | None] = queue.Queue()
    stdout: list[str] = []

    def read_stderr() -> None:
        for line in process.stderr:
            lines.put(line)
        lines.put(None)

    readers = [threading.Thread(target=read_stderr, daemon=True),
               threading.Thread(target=lambda: stdout.append(process.stdout.read()), daemon=True)]
    for reader in readers:
        reader.start()
    tail: list[str] = []
    stopped_at: float | None = None
    while True:
        if loop.interrupted and stopped_at is None:
            stopped_at = time.monotonic()
            process.send_signal(signal.SIGINT)         # a motion stops and goes home; otherwise the setup ends
        if stopped_at is not None and time.monotonic() - stopped_at > STOP_GRACE_S and process.poll() is None:
            process.kill()
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        if line is None:
            break
        line = line.strip()
        if not line:
            continue
        tail = (tail + [line])[-5:]
        try:
            said = json.loads(line)
        except ValueError:
            said = None
        if isinstance(said, dict) and ("progress" in said or "step" in said):
            _tell(loop, view, {key: said[key] for key in ("progress", "step", "state", "why") if key in said})
    code = process.wait()
    for reader in readers:
        reader.join(timeout=5)
    return code, stdout[0] if stdout else "", "\n".join(tail)


def _answer(done: tuple[int | None, str, str], what: str) -> dict[str, Any]:
    """The ``result`` of ``{"ok": true, "result": ...}`` on stdout's last JSON line; else why it failed."""

    code, stdout, stderr = done
    for line in reversed(stdout.strip().splitlines()):
        try:
            answer = json.loads(line)
        except ValueError:
            continue
        if isinstance(answer, dict):
            if answer.get("ok") is True and isinstance(answer.get("result"), dict):
                return answer["result"]
            raise SupervisorError(f"{what} failed: {answer.get('error') or 'no reason given'}")
    last = (stderr.strip().splitlines() or ["no answer"])[-1]
    raise SupervisorError(f"{what} ended (exit code {code}) without an answer: {last}")
