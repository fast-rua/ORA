"""Opt-in reset crash diagnostics; no simulator, budget or rendering changes."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import faulthandler
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback
from typing import TextIO

from .common import write_json

TIMER_ENV = "ORA_ROBODOJO_RESET_TRACEBACK"
FATAL_ENV = "ORA_ROBODOJO_RESET_FATAL_TRACEBACK"


def _enabled(name: str, default: str) -> bool:
    value = os.environ.get(name, default)
    if value not in {"on", "off"}:
        raise ValueError(f"{name} must be on or off")
    return value == "on"


def _record(path: Path, value: dict) -> None:
    try:
        write_json(path, value)
    except OSError as exc:
        # Diagnostics must not mask an environment exception or prevent reset.
        print(f"reset diagnostic evidence unavailable: {exc}", file=sys.stderr, flush=True)


@contextmanager
def stack_timer(seconds: float, file: TextIO):
    """Write every thread's Python stack to ``file`` once, if the block outlasts ``seconds``.

    ``faulthandler.dump_traceback_later`` reads other threads' frames without the GIL, so it can
    crash the worker it observes (SIGSEGV). This reads them from a Python thread instead; a C call
    that keeps the GIL delays the dump until it returns.
    """

    def dump() -> None:
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        lines = [f"Timeout after {seconds:g} s\n"]
        for ident, frame in sys._current_frames().items():
            lines.append(f"\nThread {ident:#x} {names.get(ident, 'unknown')} (most recent call last):\n")
            lines.extend(traceback.format_stack(frame))
        file.write("".join(lines))
        file.flush()

    timer = threading.Timer(seconds, dump)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join()


@contextmanager
def reset_diagnostics(stack_path: Path):
    """Keep the released 60s timer by default; optional fatal log owns its FD.

    A pre-existing Python fatal handler is preserved because its original file
    descriptor cannot be recovered through faulthandler's public API. Neither
    core limits nor the host's core collector configuration are changed.
    """
    timer = _enabled(TIMER_ENV, "on")
    fatal = _enabled(FATAL_ENV, "off")
    evidence_path = stack_path.with_suffix(".diagnostics.json")
    fatal_path = stack_path.with_suffix(".fatal.txt")
    record = {
        "schema": "ora.robodojo-reset-diagnostics.v1",
        "worker_pid": os.getpid(), "started_unix_s": time.time(),
        "timer_enabled": timer, "timer_seconds": 60,
        "timer_file": stack_path.name, "fatal_requested": fatal,
        "fatal_handler": "disabled", "fatal_file": None,
        "core_limit_bytes": list(resource.getrlimit(resource.RLIMIT_CORE)),
        "outcome": "in_progress",
    }
    try:
        record["core_pattern"] = Path("/proc/sys/kernel/core_pattern").read_text().strip()
    except OSError:
        record["core_pattern"] = None
    owned_fatal = False
    with ExitStack() as files:
        # Preserve the historical empty timer file even with the timer disabled.
        stack_file = files.enter_context(stack_path.open("w"))
        try:
            if fatal:
                if faulthandler.is_enabled():
                    record["fatal_handler"] = "existing_handler_preserved"
                else:
                    try:
                        fatal_file = files.enter_context(fatal_path.open("w"))
                        faulthandler.enable(file=fatal_file, all_threads=True)
                        owned_fatal = True
                        record.update(fatal_handler="independent_reset_log", fatal_file=fatal_path.name)
                    except (OSError, RuntimeError) as exc:
                        record.update(fatal_handler="setup_failed", fatal_error=str(exc))
            if timer:
                files.enter_context(stack_timer(60, stack_file))
            _record(evidence_path, record)
            try:
                yield
            except BaseException as exc:
                record.update(outcome="exception", exception_type=type(exc).__name__)
                raise
            else:
                record["outcome"] = "completed"
        finally:
            if owned_fatal:
                faulthandler.disable()
            record["finished_unix_s"] = time.time()
            _record(evidence_path, record)
