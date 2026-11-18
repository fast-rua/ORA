"""Formal runs: ``ora launch``, ``retry``, ``status``, ``stop``, ``watch``, ``workspace`` and ``task``.

``launch`` and ``retry`` print the plan, run its preflight, ask for the plan's
confirmation code (``--yes`` skips the question) and hand the plan to
``orchestrator.launch``, which stages the run, claims its lease and starts its
Supervisor. ``status`` reads the run registry and leases only; ``watch`` is
read-only and takes no lease; ``stop`` signals exactly one run's Supervisor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ora.version import ora_command

COMMANDS = ("launch", "retry", "status", "stop", "watch", "workspace", "task")


def add_watch_arguments(parser: argparse.ArgumentParser) -> None:
    """``ora watch``'s display and world-view options (``python -m ora.app.backend watch`` takes them too).

    ``ora.agent.tui.cli.watch_run`` reads them.
    """

    from .world import add_world_options

    parser.add_argument("--speed", type=float, default=None,
                        help="replay speed of a finished run (default 10 in a terminal, "
                             "0 = print at once, the default when not a terminal)")
    parser.add_argument("--max-gap", type=float, default=30.0,
                        help="longest recorded pause replayed, in seconds (default 30)")
    parser.add_argument("--plain", action="store_true", help="no colours and no live status region")
    parser.add_argument("--ui", choices=("auto", "pi", "ansi", "plain"), default="auto",
                        help="front end: pi (pi-tui, needs Node.js >= 22.19 and `npm ci` in ora-pi/), ansi "
                             "(built in), plain; auto picks pi in a terminal when installed")
    parser.add_argument("--emit", choices=("jsonl",), default=None,
                        help="write UI operations as JSON lines instead of drawing (the pi front end reads them)")
    parser.add_argument("--poll", type=float, default=0.2,
                        help="seconds between reads of a live run's events (default 0.2)")
    add_world_options(parser)


def _overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", dest="cell", help="the task name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--layout", type=int)
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                        help="any other task key (policy.*, budgets.* included); VALUE is YAML")


def add_task_parser(sub: Any) -> None:
    """``ora task resolve|run|submit`` (ora.app.task)."""
    task = sub.add_parser("task", help="resolve, run or submit a task file",
                          description="run a task file (ora.task.v1) many times, changing only what each run needs")
    commands = task.add_subparsers(dest="task_command", required=True)
    resolve = commands.add_parser("resolve", help="print the task after overrides")
    resolve.add_argument("file", type=Path)
    _overrides(resolve)
    run = commands.add_parser("run", help="run the task once on its binding and wait for it")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("file", type=Path, nargs="?")
    source.add_argument("--resolved", help="the resolved task as JSON (what submit sends)")
    _overrides(run)
    run.add_argument("--binding", type=Path, help="an ora.run-binding.v1 or allocation file "
                                                  "(default: ORA_ROUTER_ALLOCATION)")
    run.add_argument("--summary", type=Path, help="also write the task record here")
    run.add_argument("--no-wait", action="store_true", help="return once the run is launched")
    submit = commands.add_parser("submit", help="send one run of the task to the scheduler")
    submit.add_argument("file", type=Path)
    _overrides(submit)
    submit.add_argument("--defaults", type=Path, help="standing request fields (YAML)")
    submit.add_argument("--scheduler-url")
    submit.add_argument("--owner")
    submit.add_argument("--name")
    submit.add_argument("--priority")
    submit.add_argument("--job-type")
    submit.add_argument("--window")
    submit.add_argument("--workspace")
    submit.add_argument("--pin")
    submit.add_argument("--cpu", action="store_true", default=None)
    submit.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    submit.add_argument("--env-file", action="append", default=[], metavar="NAME=FILE")
    submit.add_argument("--summary", type=Path, help="the run also writes its task record here (shared storage)")
    submit.add_argument("--print", action="store_true", help="print the request instead of sending it")


def register(sub: Any) -> None:
    from ora.orchestrator.episode import agent_core

    launch = sub.add_parser("launch", help="preflight and launch one confirmed run")
    launch.add_argument("run_spec", type=Path)
    launch.add_argument("--yes", action="store_true")
    agent_core.add_cli_option(launch)
    retry = sub.add_parser("retry", help="retry an unsealed action-free incomplete run")
    retry.add_argument("run_id")
    retry.add_argument("--yes", action="store_true")
    status = sub.add_parser("status", help="show one run or the complete local run registry")
    status.add_argument("run_id", nargs="?")
    stop = sub.add_parser("stop", help="request SIGTERM for the exact active run lease")
    stop.add_argument("run_id")
    watch = sub.add_parser(
        "watch", help="follow a live run, or replay a finished one, as ORA's transcript",
        description="Follow a live Native run, or replay a finished one, as ORA's terminal "
                    "transcript. Read-only: writes nothing under the run and takes no lease; "
                    "Ctrl-C stops watching, not the run.",
    )
    watch.add_argument("run_id", help="run ID or run directory")
    add_watch_arguments(watch)
    watch.add_argument("--raw", action="store_true",
                       help="print the run's canonical causal events as JSON lines until it terminates")
    sub.add_parser("workspace", help="show the shared evidence/state roots and legacy discovery roots")
    add_task_parser(sub)


def _confirmed(plan: dict[str, Any], *, yes: bool, verb: str) -> bool:
    if yes:
        return True
    answer = input(f"Type confirmation code {plan['confirmation_code']} to {verb}: ").strip()
    if answer != plan["confirmation_code"]:
        print(f"{verb} cancelled")
        return False
    return True


def command_launch(args: argparse.Namespace) -> int:
    from ora.orchestrator.launch import _preflight, _spawn
    from ora.orchestrator.spec.run_spec import load_run_spec, resolve_run_spec

    from .plan import display_plan

    from .run import bind_prepared_specs

    try:
        bind_prepared_specs([args.run_spec])  # a spec ora run --prepare-only wrote: its .env, marker and services
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ora launch: {exc}", file=sys.stderr)
        return 2
    plan = resolve_run_spec(load_run_spec(args.run_spec))
    display_plan(plan)
    print(f"command              {ora_command()} launch {args.run_spec}")
    preflight = _preflight(plan)
    if not preflight["passed"]:
        return 2
    if not _confirmed(plan, yes=args.yes, verb="launch"):
        return 3
    _spawn(plan, preflight, yes=bool(args.yes))
    return 0


def command_retry(args: argparse.Namespace) -> int:
    from ora.orchestrator.launch import _preflight, _require_launchable, _run_root, _spawn
    from ora.orchestrator.leases import _load
    from ora.orchestrator.spec.run_spec import validate_run_plan

    from .plan import display_plan

    previous = _run_root(args.run_id)
    if (previous / "seal.json").exists():
        raise RuntimeError("sealed attempts cannot be retried")
    result = _load(previous / "result.json")
    # A refused proposal reached the environment although it spent no decision.
    dispatched = int(result.get("decisions", -1)) + int(result.get("refused_proposals", 0))
    if result.get("state") != "incomplete" or dispatched != 0:
        raise RuntimeError("retry is allowed only before any robot action was published")
    plan = _load(previous / "plan.json")
    validate_run_plan(plan)
    _require_launchable(plan, retry=True)
    display_plan(plan)
    print(f"retry of             {args.run_id}")
    preflight = _preflight(plan)
    if not preflight["passed"]:
        return 2
    if not _confirmed(plan, yes=args.yes, verb="retry"):
        return 3
    _spawn(plan, preflight, yes=bool(args.yes), retry_of=args.run_id)
    return 0


def command_status(run_id: str | None) -> int:
    """One run's persisted status, else the active leases and every run; reads leases and runs only."""

    import json

    from ora.orchestrator import paths
    from ora.orchestrator.leases import _active_leases, _load, _public_lease

    if run_id:
        root = paths.resolve_run_root(run_id)
        print(json.dumps(_load(root / "status.json"), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    active = [_public_lease(item) for item in _active_leases()]
    print(f"active: {json.dumps(active, ensure_ascii=False, sort_keys=True) if active else 'none'}")
    for root in paths.iter_run_roots():
        status = _load(root / "status.json") if (root / "status.json").is_file() else {"state": "staged"}
        plan = _load(root / "plan.json") if (root / "plan.json").is_file() else {}
        print(f"{root.name}\t{plan.get('experiment', '—')}\t{status.get('state')}\t{status.get('official_success', '—')}")
    return 0


def command_watch(args: argparse.Namespace) -> int:
    import json
    import time

    from ora.orchestrator import paths
    from ora.orchestrator.leases import _load

    candidate = Path(args.run_id).expanduser()
    root = candidate.resolve() if candidate.is_dir() else paths.resolve_run_root(args.run_id)
    if not args.raw:
        from ora.agent.tui.cli import watch_run

        from .world import watch_observer

        return watch_run(root, args, observer_factory=watch_observer(args))
    events = root / "events.jsonl"
    offset = 0
    try:
        while True:
            if events.is_file():
                with events.open("r", encoding="utf-8") as stream:
                    stream.seek(offset)
                    for line in stream:
                        event = json.loads(line)
                        print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)
                    offset = stream.tell()
            status = _load(root / "status.json") if (root / "status.json").is_file() else {}
            if status.get("state") in {"completed", "incomplete", "invalid", "blocked"}:
                return 0
            time.sleep(1)
    except KeyboardInterrupt:
        return 130


def run(args: argparse.Namespace) -> int:
    if args.command == "task":
        from .task import run as run_task

        return run_task(args)
    if args.command == "launch":
        return command_launch(args)
    if args.command == "retry":
        return command_retry(args)
    if args.command == "status":
        return command_status(args.run_id)
    if args.command == "watch":
        return command_watch(args)
    if args.command == "stop":
        from ora.orchestrator.launch import request_stop

        request_stop(args.run_id)
        return 0
    import json

    from ora.orchestrator import paths

    print(json.dumps(paths.workspace_topology(), indent=2, sort_keys=True))
    return 0


__all__ = ["COMMANDS", "add_watch_arguments", "command_launch", "command_retry", "command_status",
           "command_watch", "register", "run"]
