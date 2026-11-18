"""JSON-lines ORA interaction: prepared tasks, audited steering and explicit stops.

The pi-tui front end (``ora-pi/src/watch/index.mjs``) draws what this process
writes; it starts it as

    python -m ora.app.backend watch RUN --emit jsonl [watch and world options]
    python -m ora.app.backend interactive --spec SPEC [--spec ...] [--confirm-launch] --emit jsonl

``watch`` writes the UI operations of ``ora watch`` (``ora.agent.tui``) and
reads nothing; ``interactive`` serves the task picker (``serve``): it launches a
selected task through the normal resolver, preflight and Supervisor, and
delivers queued messages only to the run it launched.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TextIO

from ora.agent.tui.follow import EventTail, run_is_finished
from ora.agent.tui.transcript import RunInfo, Transcript
from ora.version import ora_command, version_label

if TYPE_CHECKING:  # the catalog resolves RunSpecs; the watch child never loads it
    from .task_catalog import PreparedTask


class Emitter:
    def __init__(self, stream: TextIO):
        self.stream = stream
        self.lock = threading.Lock()

    def __call__(self, op: Mapping[str, Any]) -> None:
        with self.lock:
            self.stream.write(json.dumps(dict(op), ensure_ascii=False) + "\n")
            self.stream.flush()

    def notice(self, message: str) -> None:
        self({"op": "notice", "level": "info", "mark": "·", "text": message})


def launch_prepared(task: PreparedTask, initial: Sequence[Mapping[str, Any]],
                    cancelled: threading.Event, report: Callable[[str], None]) -> Path:
    """Use the normal resolver, preflight and detached Supervisor; never run a policy here."""
    from ora.orchestrator.launch import _preflight, _spawn
    from ora.orchestrator.paths import resolve_run_root
    from ora.orchestrator.spec.run_spec import resolve_run_spec

    spec = task.interactive_spec(uuid.uuid4().hex, initial)
    plan = resolve_run_spec(spec)
    if plan["policy"]["kind"] != "native_api":
        raise RuntimeError("interactive tasks must resolve to a Native policy")
    if cancelled.is_set():
        raise RuntimeError("launch cancelled")
    preflight = _preflight(plan, report=report, compact=True)  # one TUI line per passed check
    if not preflight["passed"]:
        raise RuntimeError("launch blocked by preflight; inspect the reported checks")
    if cancelled.is_set():
        raise RuntimeError("launch cancelled")
    run_id = _spawn(plan, preflight, yes=True, report=report)
    return resolve_run_root(run_id)


class InteractiveBackend:
    """One UI process owns its queue and only runs it explicitly selected."""

    def __init__(self, tasks: Sequence[PreparedTask], emit: Callable[[Mapping[str, Any]], None],
                 *, confirm_launch: bool, launch: Callable[..., Path] = launch_prepared, observer=None):
        self.tasks = {task.value: task for task in tasks}
        self.emit = emit
        self.confirm_launch = confirm_launch
        self.launch = launch
        self.observer = observer
        self.state = "idle"
        self.root: Path | None = None
        self.transcript: Transcript | None = None
        self.tail: EventTail | None = None
        self.followups: list[dict[str, Any]] = []
        self.run_messages: dict[str, dict[str, Any]] = {}
        self.submitted: set[str] = set()
        self.seen: dict[str, dict[str, Any]] = {}
        self.last_queue: str | None = None
        self.cancelled = threading.Event()
        self.completed: queue.Queue = queue.Queue()
        self.job: threading.Thread | None = None
        self.initial: list[dict[str, Any]] = []
        self.launch_lock = threading.Lock()
        self.abort_requested = False
        self.launch_root: Path | None = None
        self.launch_stop_claimed = False

    def start(self) -> None:
        if self.observer is not None:
            self.observer.poll()
        self.emit({"op": "header", "version": version_label(), "mode": "interactive", "run_id": "choose a task",
                   "run_root": "", "benchmark": "", "task": "", "experiment": "ora-interactive",
                   "rounds": 0, **self.provider()})
        self.change_state("idle")
        self.show_tasks()
        self.show_queue()

    def provider(self) -> dict[str, str]:
        """The picker header's model line, from the prepared RunSpecs' Provider profiles.

        A field the specs disagree on lists every value, so the header never names one
        task's model for another.
        """
        bound = [task.provider() for task in self.tasks.values()]
        return {key: " / ".join(dict.fromkeys(item[key] for item in bound if item[key])) for key in bound[0]}

    def change_state(self, state: str, message: str = "") -> None:
        self.state = state
        self.emit({"op": "session_state", "state": state, "run_id": self.root.name if self.root else None,
                   "message": message})

    def show_tasks(self) -> None:
        self.emit({"op": "tasks", "items": [task.item() for task in self.tasks.values()],
                   "message": "Select a prepared benchmark task. Its instruction and evaluator stay fixed."})

    def show_queue(self) -> None:
        items = [*self.run_messages.values(), *self.followups]
        encoded = json.dumps(items, sort_keys=True)
        if encoded != self.last_queue:
            self.last_queue = encoded
            self.emit({"op": "queue", "items": items})

    def reply(self, request: Mapping[str, Any], *, error: str | None = None) -> None:
        self.emit({"op": "response", "id": request.get("id"), "type": request.get("type"),
                   "success": error is None, **({"error": error} if error else {})})

    def request(self, request: Any) -> None:
        if not isinstance(request, dict):
            self.reply({}, error="request must be a JSON object")
            return
        try:
            identity, kind = request.get("id"), request.get("type")
            if not isinstance(identity, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identity) is None:
                raise RuntimeError("request needs a stable id of at most 128 letters, digits, dots, underscores or hyphens")
            if kind not in {"prompt", "steer", "follow_up", "abort"}:
                raise RuntimeError("unknown request type")
            allowed = {"id", "type", "task"} if kind == "prompt" else {"id", "type"}
            if kind in {"steer", "follow_up"}:
                allowed.add("message")
            if set(request) != allowed:
                raise RuntimeError("request fields do not match its type")
            if identity in self.seen:
                if self.seen[identity] != request:
                    raise RuntimeError("request id was already used with different contents")
                self.reply(request)
                return
            if kind in {"steer", "follow_up"}:
                if not isinstance(request["message"], str) or not request["message"].strip():
                    raise RuntimeError("message must be non-empty text")
                if len(request["message"].encode("utf-8")) > 60000:
                    raise RuntimeError("message exceeds the operator-control size limit")
            self._dispatch(request)
            self.seen[identity] = dict(request)
            self.reply(request)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.reply(request, error=str(exc))
        self.show_queue()

    def _dispatch(self, request: Mapping[str, Any]) -> None:
        kind = request["type"]
        if kind == "prompt":
            self._launch(request)
        elif kind == "follow_up":
            if len(self.followups) >= 64:
                raise RuntimeError("follow-up queue is full (64 messages)")
            self.followups.append({**request, "status": "queued"})
        elif kind == "steer":
            if self.state != "running" or self.root is None:
                raise RuntimeError("steer requires a running interactive task")
            from ora.orchestrator.episode.operator_control import submit_command
            try:
                submit_command(self.root, request)
                self.submitted.add(request["id"])
            except FileNotFoundError:
                pass  # preserve the message while the Supervisor binds its inbox
            self.run_messages[request["id"]] = {**request, "status": "queued"}
        elif self.state == "launching":
            with self.launch_lock:
                self.cancelled.set()
                self.abort_requested = True
                root = self.launch_root if not self.launch_stop_claimed else None
                if root is not None:
                    self.launch_stop_claimed = True
            if root is not None:
                self.stop_root(root)
        elif self.state == "running" and self.root is not None:
            self.stop()
        else:
            raise RuntimeError("there is no active task to abort")

    def _launch(self, request: Mapping[str, Any]) -> None:
        if self.state in {"launching", "running"}:
            raise RuntimeError("finish or abort the current task before choosing another")
        if not self.confirm_launch:
            raise RuntimeError("launch requires --confirm-launch; task selection is the final launch action")
        task = self.tasks.get(request["task"])
        if task is None:
            raise RuntimeError("choose a task from the supplied RunSpecs")
        self.initial = [{key: message[key] for key in ("id", "type", "message")} for message in self.followups]
        if self.observer is not None:
            self.observer.bind(None)
        initial = list(self.initial)
        self.cancelled = threading.Event()
        with self.launch_lock:
            self.abort_requested = False
            self.launch_root = None
            self.launch_stop_claimed = False
        self.change_state("launching", f"Preflight: {task.item()['label']}")

        def launch() -> None:
            try:
                root = self.launch(task, initial, self.cancelled, lambda message: self.emit(
                    {"op": "notice", "level": "info", "mark": "·", "text": message}))
                with self.launch_lock:
                    self.launch_root = Path(root)
                    stop = self.abort_requested and not self.launch_stop_claimed
                    if stop:
                        self.launch_stop_claimed = True
                if stop:
                    try:
                        self.stop_root(Path(root))
                    except (OSError, RuntimeError) as exc:
                        self.emit({"op": "notice", "level": "warning", "mark": "·", "text": str(exc)})
                self.completed.put((root, None))
            except Exception as exc:
                self.completed.put((None, str(exc)))

        self.job = threading.Thread(target=launch, name="ora-preflight-launch")
        self.job.start()

    def stop(self) -> None:
        assert self.root is not None
        self.stop_root(self.root)

    def stop_root(self, root: Path) -> None:
        from ora.orchestrator.launch import request_stop
        # Only this backend's newly launched run, through the existing exact-node lease guard.
        request_stop(root.name, report=lambda text: self.emit(
            {"op": "notice", "level": "info", "mark": "·", "text": text}))

    def read_events(self) -> None:
        assert self.tail is not None and self.transcript is not None
        for event in self.tail.read():
            for op in self.transcript.feed(event):
                self.emit(op)
            if event.get("type") == "session_started":
                self.emit(self.transcript.header("interactive", version_label()))

    def tick(self) -> None:
        if self.observer is not None:
            self.observer.poll()
        if self.state == "launching":
            try:
                root, error = self.completed.get_nowait()
            except queue.Empty:
                return
            if error:
                self.change_state("idle", error)
                self.show_tasks()
                return
            self.root = Path(root)
            if self.observer is not None:
                self.observer.bind(self.root)
            self.transcript = Transcript(self.root, RunInfo.from_run(self.root))
            self.tail = EventTail(self.root / "agent-events.jsonl")
            used = {message["id"] for message in self.initial}
            self.run_messages = {message["id"]: {**message, "status": "queued"} for message in self.initial}
            self.submitted = set(self.run_messages)  # initial follow-ups are seeded by the RunPlan
            self.followups = [message for message in self.followups if message["id"] not in used]
            self.emit(self.transcript.header("interactive", version_label()))
            self.emit(self.transcript.user())
            self.change_state("running")
        if self.state != "running" or self.transcript is None or self.tail is None:
            return
        self.read_events()
        from ora.orchestrator.episode.operator_control import OperatorControlError, command_status, submit_command
        try:
            status = command_status(self.root)
            for identity, message in self.run_messages.items():
                if identity not in self.submitted and message["status"] == "queued":
                    try:
                        submit_command(self.root, {key: message[key] for key in ("id", "type", "message")})
                        self.submitted.add(identity)
                    except OperatorControlError as exc:
                        message.update(status="rejected", error=str(exc))
            for record in status["acknowledgements"]:
                if record["id"] in self.run_messages:
                    self.run_messages[record["id"]].update(
                        status="delivered" if record["status"] == "applied" else "rejected",
                        **({"error": record["reason"]} if record.get("reason") else {}),
                    )
        except FileNotFoundError:
            pass  # the Supervisor is still binding its sessions
        except OperatorControlError as exc:
            for message in self.run_messages.values():
                if message["status"] == "queued":
                    message.update(status="rejected", error=str(exc))
            self.emit({"op": "notice", "level": "warning", "mark": "·", "text": str(exc)})
        self.show_queue()
        current = self.transcript.status
        self.emit({"op": "status", "text": None if current is None else
                   f"{current.verb}… ({max(0, time.time() - current.started_unix):.0f}s)"})
        self.emit(self.transcript.footer_state())
        if self.transcript.finished or run_is_finished(self.root):
            self.read_events()
            for op in self.transcript.flush():
                self.emit(op)
            landing = landing_state(self.root)
            if landing == "landing":
                # The Supervisor still writes metrics, evaluates, seals and releases its lease;
                # leaving now (or ending the cluster job step that owns it) would cut that short.
                self.emit({"op": "status", "text": "Landing… (metrics, evaluation, seal, lease release)"})
                return
            for message in self.run_messages.values():
                if message["status"] == "queued":
                    message.update(status="rejected", error="run ended before delivery")
            self.show_queue()
            if landing == "unsealed":
                self.emit({"op": "notice", "level": "warning", "mark": "·", "text": (
                    f"The Supervisor exited before sealing {self.root.name}; on this node run "
                    "`ora metrics`, `ora evaluate` and `ora seal` for it.")})
            self.emit({"op": "status", "text": None})
            self.emit({"op": "end", "ended": self.transcript.ended, "landing": landing})
            self.change_state("finished", "Choose another benchmark task when ready.")
            self.show_tasks()


def landing_state(root: Path) -> str:
    """``landing`` while this node's Supervisor still runs; after it exits, ``sealed`` or
    ``seal_refused`` from its landing record, else ``unsealed``.

    The seal is not the last step: the Supervisor then releases its lease and run scratch, so
    a client must wait for the process itself.
    """
    from ora.orchestrator.suite.queue import supervisor_alive

    if supervisor_alive(root, root.name):
        return "landing"
    if (root / "seal.json").exists():
        return "sealed"
    if (root / "seal-refusal.json").exists():
        return "seal_refused"
    return "unsealed"


def serve(specs: Sequence[Path], *, confirm_launch: bool, poll: float = 0.2,
          input_stream: TextIO | None = None, output_stream: TextIO | None = None, observer_factory=None) -> int:
    if not math.isfinite(poll) or poll <= 0:
        raise RuntimeError("poll must be a positive finite number")
    emitter = Emitter(output_stream or sys.stdout)
    from .task_catalog import load_tasks

    tasks = load_tasks(specs)
    observer = observer_factory(root=None, finished=False, emit=emitter) if observer_factory else None
    backend = InteractiveBackend(tasks, emitter, confirm_launch=confirm_launch, observer=observer)
    requests: queue.Queue = queue.Queue()

    def read() -> None:
        stream = input_stream or sys.stdin
        while line := stream.readline(65537):
            if len(line.encode("utf-8")) > 65536 or not line.endswith("\n"):
                requests.put({"invalid": "request must be a complete JSON line of at most 64 KiB"})
                if not line.endswith("\n"):
                    while line and not line.endswith("\n"):
                        line = stream.readline(65537)
                continue
            try:
                requests.put(json.loads(line))
            except ValueError:
                requests.put({"invalid": "invalid JSON request"})
        requests.put(None)

    threading.Thread(target=read, daemon=True, name="ora-ui-input").start()
    backend.start()
    try:
        while True:
            try:
                request = requests.get(timeout=max(0.05, poll))
                if request is None:
                    backend.cancelled.set()  # cancel unlaunched preparation; never stop a live run on detach
                    return 0
                backend.request(request)
            except queue.Empty:
                pass
            backend.tick()
    except (KeyboardInterrupt, BrokenPipeError):
        backend.cancelled.set()
        return 0
    finally:
        if observer is not None:
            observer.close()


def build_parser() -> argparse.ArgumentParser:
    """The front end's two child commands; their options are ``ora watch``'s and ``ora interactive``'s."""

    from .interactive import add_interactive_arguments
    from .runs import add_watch_arguments

    parser = argparse.ArgumentParser(prog="python -m ora.app.backend",
                                     description="The JSON-lines backend of ORA's pi-tui front end.")
    sub = parser.add_subparsers(dest="command", required=True)
    watch = sub.add_parser("watch", help="the UI operations of one run's transcript, read-only")
    watch.add_argument("run", help="run directory (outputs/<experiment>/runs/<run_id>)")
    add_watch_arguments(watch)
    add_interactive_arguments(sub.add_parser("interactive", help="the task picker over prepared RunSpecs"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.emit != "jsonl":
        print("ora backend: the front end's child writes JSON lines; pass --emit jsonl", file=sys.stderr)
        return 2
    from .world import watch_observer

    try:
        world = watch_observer(args)
        if args.command == "watch":
            from ora.agent.tui.cli import watch_run

            return watch_run(args.run, args, observer_factory=world)
        if not args.spec:
            raise ValueError("the task picker needs --spec")
        return serve(args.spec, confirm_launch=args.confirm_launch, poll=args.poll, observer_factory=world)
    except ModuleNotFoundError as exc:
        print(f"ora: dependency {exc.name!r} is unavailable; run {ora_command()} setup first", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ora: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
