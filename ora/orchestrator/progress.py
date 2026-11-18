"""``ora run``'s long producer steps: a subprocess with its log, a progress line and its stdout.

A backend's run cell (``backends/registry.py``, ``run_cell``) runs its
one-time qualification producers through ``run_with_progress``, with their
logs under ``outputs/ora-interactive/logs`` (``log_path``). The module stays
in the orchestrator core because those backend modules use it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .paths import PROJECT_ROOT, link_logical_workspace


def clock(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def run_with_progress(command: Sequence[str], *, label: str, estimate: str, log: Path, stream=None,
                      env: Mapping[str, str] | None = None, poll_s: float = 1.0,
                      popen=subprocess.Popen, accepted_codes: Sequence[int] = (0,)) -> str:
    """Run ``command`` with its output in ``log``; show elapsed time against ``estimate``; return stdout."""
    output = stream or sys.stderr
    log.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log.with_suffix(".stdout")
    live = hasattr(output, "isatty") and output.isatty()
    started = time.monotonic()
    last_line = started
    with log.open("ab") as errors, stdout_path.open("wb") as captured:
        process = popen([str(item) for item in command], cwd=PROJECT_ROOT, stdout=captured, stderr=errors,
                        env={**os.environ, **dict(env or {})})
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if live:
                print(f"\r  {label}: {clock(elapsed)} elapsed ({estimate}; log {log})", end="", file=output,
                      flush=True)
            elif time.monotonic() - last_line >= 60:
                print(f"  {label}: {clock(elapsed)} elapsed ({estimate})", file=output, flush=True)
                last_line = time.monotonic()
            time.sleep(poll_s)
    elapsed = time.monotonic() - started
    if live:
        print("", file=output)
    result = stdout_path.read_text(errors="replace")
    if process.returncode not in accepted_codes:
        tail = (log.read_text(errors="replace").splitlines() + result.splitlines())[-20:]
        raise RuntimeError(f"{label} failed after {clock(elapsed)} (exit {process.returncode}; log {log})\n"
                           + "\n".join(tail))
    print(f"  {label}: done in {clock(elapsed)}", file=output, flush=True)
    return result


def log_path(name: str) -> Path:
    """Where one producer's log goes: ``outputs/ora-interactive/logs/<UTC time>-<name>.log``."""
    return link_logical_workspace("ora-interactive") / "logs" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{name}.log"


def newest(paths: Iterable[Path]) -> list[Path]:
    """``paths``, newest modification first."""
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


__all__ = ["clock", "log_path", "newest", "run_with_progress"]
