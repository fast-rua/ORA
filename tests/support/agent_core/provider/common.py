"""Shared helpers for the provider differential scripts (Python side).

The TypeScript tests in ora-pi/test/provider*.test.ts spawn these scripts and
compare their output byte for byte with the TypeScript port.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time as _real_time
from pathlib import Path
from typing import Any, Mapping

from ora.agent.evidence.canonical import canonical_bytes

TIMING_FIELDS = ("started_unix_s", "finished_unix_s", "duration_s", "ttft_s", "no_content_elapsed_s")


def normalize_timing(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replace wall/monotonic timing values (never comparable) with a marker."""
    out = dict(value)
    for key in TIMING_FIELDS:
        if key in out and out[key] is not None:
            out[key] = "<t>"
    return out


def canonical_text(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def describe_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "retryable": getattr(exc, "retryable", None),
        "http_status": getattr(exc, "http_status", None),
        "response_started": getattr(exc, "response_started", None),
        "credential_echo": getattr(exc, "credential_echo", None),
        "provider_failure": getattr(exc, "provider_failure", None),
        "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
    }


class SleepRecorder:
    """A `time` module stand-in whose sleep() records and returns at once.

    With ``advance`` set, each recorded sleep also moves monotonic() on by its
    length, as the backoff would have, so a window measured across retries
    sees it.
    """

    def __init__(self) -> None:
        self.sleeps: list[Any] = []
        self.advance = False
        self.offset = 0.0

    def sleep(self, seconds: Any) -> None:
        self.sleeps.append(seconds)
        if self.advance:
            self.offset += seconds

    def monotonic(self) -> float:
        return _real_time.monotonic() + self.offset

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


#: The command recovery.bounded_request starts for each deadline-policy attempt.
WORKER_COMMAND = (sys.executable, "-m", "ora.agent.providers.recovery_worker")
_READY = b"ready\n"
# The worker module's own main(), its imports done first: the worker's module imports, and
# httpx, which the worker imports inside the attempt.
_LAUNCHER = (
    "import sys, httpx\n"
    "from ora.agent.providers import recovery_worker\n"
    "sys.stderr.buffer.write(b'ready\\n')\n"
    "sys.stderr.flush()\n"
    "recovery_worker.main()\n"
)


class PrestartedWorkers:
    """``subprocess`` as ``recovery.bounded_request`` uses it, with every worker started before the window.

    The parent starts ``python -m ora.agent.providers.recovery_worker`` for each
    attempt, inside the attempt's incident window. A worker's start-up (an
    interpreter and its imports from a shared file system) is Python's alone:
    the TypeScript port runs the attempt in process. On a loaded host it took
    longer than a scenario's window (6 s, 30 s), so Python's attempt never
    reached the server and ended "recovery deadline exhausted", while
    TypeScript's did; which scenario lost the race varied from run to run.

    Here each worker runs the same main() and is started, and waits on its
    stdin with its imports done, before the scenario's window opens
    (``prepare``); the parent's own code hands it the attempt as before, so the
    window covers the attempt, as it does in TypeScript. ``started_in_window``
    counts the workers an attempt had to start itself, which the callers
    report: a scenario that needs more than were prepared shows it.
    """

    PIPE = subprocess.PIPE
    DEVNULL = subprocess.DEVNULL
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self.started_in_window = 0
        self._ready: list[subprocess.Popen] = []

    @classmethod
    def for_recovery(cls) -> "PrestartedWorkers":
        from ora.agent.providers import recovery

        return cls(Path(recovery.__file__).resolve().parents[3])  # the cwd bounded_request gives its worker

    def _start(self) -> subprocess.Popen:
        return subprocess.Popen([sys.executable, "-c", _LAUNCHER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=self.cwd)

    @staticmethod
    def _await_ready(process: subprocess.Popen) -> subprocess.Popen:
        signal = b""
        while len(signal) < len(_READY):
            chunk = os.read(process.stderr.fileno(), len(_READY) - len(signal))
            if not chunk:
                process.kill()
                process.wait()
                raise RuntimeError(f"a recovery worker exited before it was ready ({process.returncode})")
            signal += chunk
        if signal != _READY:
            raise RuntimeError(f"a recovery worker announced {signal!r}")
        return process

    def prepare(self, count: int) -> None:
        """Have ``count`` workers ready, started side by side; reset ``started_in_window``."""

        started = [self._start() for _ in range(count - len(self._ready))]
        self._ready.extend(self._await_ready(process) for process in started)
        self.started_in_window = 0

    def run(self, command: Any, *, input: bytes | None = None, stdout: Any = None, stderr: Any = None,
            cwd: Any = None, timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess:
        """``subprocess.run`` on a ready worker, with run's own timeout handling."""

        if tuple(command) != WORKER_COMMAND or stdout != subprocess.PIPE or stderr != subprocess.PIPE \
                or Path(cwd) != self.cwd or check:
            raise AssertionError(f"recovery.bounded_request changed how it starts its worker: {command!r}")
        if self._ready:
            process = self._ready.pop(0)
        else:
            self.started_in_window += 1
            process = self._await_ready(self._start())
        with process:
            try:
                output, errors = process.communicate(input, timeout=timeout)
            except subprocess.TimeoutExpired:
                # As subprocess.run: the output so far is on the exception; kill and reap the child.
                process.kill()
                process.wait()
                raise
            except BaseException:
                process.kill()
                raise
            return subprocess.CompletedProcess(list(command), process.poll(), output, errors)

    def Popen(self, command: Any, *, stdin: Any = None, stdout: Any = None, stderr: Any = None,  # noqa: N802
              cwd: Any = None) -> subprocess.Popen:
        """``subprocess.Popen`` as a first_token_deadline_v1 attempt starts its worker: a ready one."""

        if tuple(command) != WORKER_COMMAND or stdin != subprocess.PIPE or stdout != subprocess.PIPE \
                or stderr != subprocess.DEVNULL or Path(cwd) != self.cwd:
            raise AssertionError(f"recovery.bounded_request changed how it starts its worker: {command!r}")
        if self._ready:
            return self._ready.pop(0)
        self.started_in_window += 1
        return self._await_ready(self._start())

    def close(self) -> None:
        for process in self._ready:
            process.kill()
            process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
        self._ready = []


def python_complete(request: Mapping[str, Any], events: list[Any], *, api_key: str = "sk-replay-key",
                    reasoning_field: str = "reasoning_content", model_call_id: str = "model-call-000000",
                    recovery_policy: str = "same_credential_deadline_v1") -> dict[str, Any]:
    """Run OpenAICompatibleProvider.complete() over recorded events (no network).

    The transport (`bounded_request` / `_request_once`) is replaced by one that
    returns the recorded events, so the real parse phase, its errors and its
    attempt records run unchanged.
    """
    import copy
    from ora.agent.providers import openai_compatible

    records: list[dict[str, Any]] = []
    provider = openai_compatible.OpenAICompatibleProvider(
        base_url="https://replay.invalid/v1", api_key=api_key, client=object(),
        recovery_policy=recovery_policy, reasoning_field=reasoning_field,
        attempt_observer=lambda record: records.append(dict(record)),
    )
    replay = lambda *args, **kwargs: (copy.deepcopy(events), _real_time.monotonic())  # noqa: E731
    original = openai_compatible.bounded_request
    openai_compatible.bounded_request = replay
    provider._request_once = replay
    try:
        response = provider.complete(request=request, model_call_id=model_call_id)
    except Exception as exc:  # noqa: BLE001 - every outcome is compared
        return {"error": describe_error(exc), "records": [normalize_timing(r) for r in records]}
    finally:
        openai_compatible.bounded_request = original
    return {"response": canonical_text(normalize_timing(response)),
            "records": [normalize_timing(r) for r in records]}
