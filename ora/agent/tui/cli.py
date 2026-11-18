"""``ora watch``'s front ends: the pi-tui transcript (Node) when installed, else the built-in one.

The ``ora`` command parses the options (``ora.app.runs.add_watch_arguments``); ``watch_run``
watches one run directory with them. Read-only: nothing under the run is written.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

from .follow import watch

#: The pi-tui front end lives in the Pi agent core's package and shares its installed packages.
FRONT_END = Path(__file__).resolve().parents[3] / "ora-pi"
FRONT_END_SCRIPT = FRONT_END / "src" / "watch" / "index.mjs"


def _run_root(value: str) -> Path:
    """A run directory; run IDs are not resolved here (the agent package stays below the orchestrator)."""
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"{value} is not a run directory (pass outputs/<experiment>/runs/<run_id>)")
    return path.resolve()


#: pi-tui 1.3.0 (the version ora-pi locks) declares Node.js >= 22.19, the Pi agent core's own floor.
MIN_NODE = (22, 19)
MIN_NODE_TEXT = "{}.{}".format(*MIN_NODE)


def _node_version(path: str) -> tuple[int, int, int] | None:
    import re
    import subprocess

    try:
        version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    return tuple(int(part) for part in match.groups()) if match else None


def _node_candidates() -> list[str]:
    """``ORA_NODE``, else ``node`` on PATH, ``~/.local/bin/node`` and nvm installs, newest nvm first."""
    if os.environ.get("ORA_NODE"):
        return [os.environ["ORA_NODE"]]
    found = [path for path in (shutil.which("node"), os.path.expanduser("~/.local/bin/node")) if path]
    nvm = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/v*/bin/node")),
                 key=lambda path: [int(part) for part in Path(path).parts[-3].lstrip("v").split(".")])
    return list(dict.fromkeys([*found, *reversed(nvm)]))


def _node() -> str | None:
    """The first candidate that is Node.js 22.19 or newer.

    A login shell often finds ``~/.local/bin/node`` while a tmux pane or a new
    SSH session finds the distribution's older Node first; that one would start
    the terminal UI and fail inside pi-tui.
    """
    for candidate in _node_candidates():
        if os.access(candidate, os.X_OK) and (_node_version(candidate) or (0, 0, 0))[:2] >= MIN_NODE:
            return candidate
    return None


def node_diagnosis() -> str:
    """What was found instead of a usable Node, for error messages."""
    seen = [f"{'v{}.{}.{}'.format(*version) if (version := _node_version(path)) else 'unknown version'} at {path}"
            for path in _node_candidates() if os.access(path, os.X_OK)]
    return "found " + ", ".join(seen) if seen else "no node on PATH, ORA_NODE, ~/.local/bin or ~/.nvm"


def _pi_front_end() -> tuple[str, Path] | None:
    """Node and the front end script once ``npm ci`` in ora-pi installed the packages it imports."""
    node = _node()
    modules = FRONT_END / "node_modules"
    if node and (modules / "@earendil-works" / "pi-tui").is_dir() and (modules / "chalk").is_dir():
        return node, FRONT_END_SCRIPT
    return None


def _exec_pi(run_root: Path, args: argparse.Namespace, front_end: tuple[str, Path]) -> None:
    node, script = front_end
    passed = []
    for flag, value in (("--speed", args.speed), ("--max-gap", args.max_gap), ("--poll", args.poll)):
        if value is not None:
            passed += [flag, str(value)]
    for name in ("world_view", "world_port", "world_browser", "world_public_url"):
        value = getattr(args, name, None)
        if value is not None:
            passed += ["--" + name.replace("_", "-"), str(value)]
    environment = dict(os.environ)
    root = str(FRONT_END.parent)
    environment["PYTHONPATH"] = root + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    os.execve(node, [node, str(script), str(run_root), "--python", sys.executable, *passed], environment)


def watch_run(run: str | Path, args: argparse.Namespace, *, observer_factory=None) -> int:
    """Watch one run directory with ``ora watch``'s parsed options (``ora.app.runs.add_watch_arguments``)."""
    try:
        run_root = _run_root(str(run))
        if not (run_root / "agent-events.jsonl").is_file() and not (run_root / "plan.json").is_file():
            raise RuntimeError(f"{run_root} has neither agent-events.jsonl nor plan.json")
        ui = "plain" if args.plain else args.ui
        if args.emit is None and ui in ("auto", "pi"):
            front_end = _pi_front_end() if (ui == "pi" or sys.stdout.isatty()) else None
            if front_end:
                _exec_pi(run_root, args, front_end)
            elif ui == "pi":
                raise RuntimeError(f"the pi front end needs node and `npm ci` in {FRONT_END}")
        return watch(run_root, speed=args.speed, plain={"plain": True, "ansi": False}.get(ui),
                     emit=args.emit, poll=max(0.05, args.poll), max_gap=max(0.0, args.max_gap),
                     observer_factory=observer_factory)
    except BrokenPipeError:              # e.g. piped into head: stop quietly
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"ora: {exc}", file=sys.stderr)
        return 2
