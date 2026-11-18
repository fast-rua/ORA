"""``ora task``: run a task file (``ora.task.v1``) many times, changing only what each run needs.

    ora task resolve FILE [--task T --seed S --layout L --set KEY=VALUE ...]
    ora task run (FILE | --resolved JSON) [overrides] [--binding FILE] [--summary FILE] [--no-wait]
    ora task submit FILE [overrides] [--defaults FILE] [--owner ...] [--print]

``resolve`` prints the task after overrides. ``run`` binds one run (``--binding``
or the allocation ``ORA_ROUTER_ALLOCATION`` names), composes its RunSpec,
preflights and launches it and waits for its Supervisor, then writes the task
record. ``submit`` sends a scheduler request whose command is ``ora task run``
on the resolved task in this checkout. See docs/reference/tasks.md. The
parser is ``ora.app.runs``'s (``add_task_parser``), so ``ora --help`` does not import this.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _task(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    from ora.orchestrator.task.task_file import apply_overrides, load_task, overrides_record, validate_task

    base = validate_task(json.loads(args.resolved)) if getattr(args, "resolved", None) else load_task(args.file)
    given = dict(cell=args.cell, seed=args.seed, layout=args.layout, sets=args.sets)
    return apply_overrides(base, **given), overrides_record(**given)


def run(args: argparse.Namespace) -> int:
    task, overrides = _task(args)
    if args.task_command == "resolve":
        print(json.dumps(task, ensure_ascii=False, indent=2))
        return 0
    if args.task_command == "run":
        from ora.orchestrator.task.binding import load_binding
        from ora.orchestrator.task.run import run_task

        record = run_task(task, load_binding(args.binding), overrides=overrides,
                          summary_path=args.summary, wait=not args.no_wait)
        summary = record["summary"]
        return 0 if summary.get("state") not in ("preflight_blocked",) else 2
    from ora.orchestrator.task import submit as scheduler

    defaults = scheduler.load_defaults(args.defaults)
    request = scheduler.build_request(
        task, repo=scheduler.checkout_root(Path.cwd()), defaults=defaults, name=args.name,
        env=scheduler.pairs(args.env, "--env"), env_files=scheduler.pairs(args.env_file, "--env-file"),
        summary=args.summary.resolve() if args.summary else None,
        owner=args.owner, priority=args.priority, job_type=args.job_type, window=args.window,
        workspace=args.workspace, pin=args.pin, cpu=args.cpu,
    )
    if args.print:
        print(json.dumps(request, ensure_ascii=False, indent=2))
        return 0
    reply = scheduler.post(scheduler.scheduler_url(defaults, args.scheduler_url), request)
    print(json.dumps(reply, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    return 0


__all__ = ["run"]
