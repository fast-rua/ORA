"""``ora task run``: one run of a task on its binding, from RunSpec to result summary.

The simulator's backend composes the RunSpec (``OrchestratorBackend.compose_task``);
then the run is resolved, preflighted and launched exactly as ``ora launch
--yes`` does, and this waits for its Supervisor. The task record
(``outputs/<experiment>/tasks/<run_id>.json``, ``ora.task-run.v1``) keeps the
task, the overrides, the binding, the RunSpec and the result summary; the run
directory itself is not touched.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .binding import BindingError, check_model, provider_profile, run_environment

RECORD_SCHEMA = "ora.task-run.v1"
#: Variables ORA reads once when it starts (workspace paths, host profiles): a binding may
#: restate them but not change them.
STARTUP_VARIABLES = ("ORA_WORKSPACE_ROOT", "ORA_OUTPUTS_ROOT", "ORA_HOSTS_DIR")
#: Model-request settings a task may set over its Provider profile's values.
MODEL_REQUEST_FIELDS = ("reasoning_effort", "max_output_tokens")
#: ``off`` stops ``ora task run`` from following its run's transcript on its own output.
TRANSCRIPT_VARIABLE = "ORA_TASK_TRANSCRIPT"
#: Seconds the transcript may take to print the last events after the Supervisor exits.
TRANSCRIPT_DRAIN_S = 15.0


def compose_spec(task: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    """The RunSpec of ``task`` on ``binding``, from the task's simulator backend."""

    from ora.orchestrator.backends.registry import backend

    compose = backend(task["simulator"]).compose_task
    if compose is None:
        raise ValueError(f"ora task does not compose {task['simulator']} runs yet")
    spec = compose(task, binding)
    if task.get("save_sandbox", True) is False:  # every simulator: sandbox/ keeps only hashes after sealing
        spec["save_sandbox"] = False
    return spec


def apply_model_request(task: Mapping[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Set the task's model-request fields on ``spec``; returns ``{field: {profile, run}}`` for the record."""

    changed = {}
    for field in MODEL_REQUEST_FIELDS:
        if field in (task.get("policy") or {}):
            changed[field] = {"profile": spec["policy"].get(field), "run": task["policy"][field]}
            spec["policy"][field] = task["policy"][field]
    return changed


def prepare_environment(task: Mapping[str, Any], binding: Mapping[str, Any],
                        environ: Mapping[str, str] = os.environ) -> dict[str, str]:
    """The run's environment; refuses a model mismatch or a startup variable this process did not start with."""

    profile = provider_profile(binding)
    check_model(task, binding, profile)
    environment = run_environment(binding, profile, environ)
    for variable in STARTUP_VARIABLES:
        if environment.get(variable) != environ.get(variable):
            raise BindingError(f"{variable} must be set before `ora task run` starts "
                               f"({environ.get(variable)!r} here, {environment.get(variable)!r} in the binding)")
    return environment


def _wait(pid: int, *, poll_s: float) -> None:
    while True:
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)  # the Supervisor is this process's child
            if done:
                return
        except ChildProcessError:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
        time.sleep(poll_s)


class _StopForwarding:
    """While ``ora task run`` waits, a stop signal it receives is passed on as a graceful stop of its run."""

    SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

    def __init__(self, run_id: str, report: Callable[[str], None]) -> None:
        self.run_id, self.report, self.signal = run_id, report, None
        self.previous = {number: signal.signal(number, self._forward) for number in self.SIGNALS}

    def _forward(self, number: int, frame: Any) -> None:
        if self.signal is not None:
            return  # one stop request is enough; keep waiting for the run to seal
        self.signal = number
        from ora.orchestrator.launch import request_stop
        self.report(f"{signal.Signals(number).name}: stopping run {self.run_id} and waiting for it to seal")
        try:
            request_stop(self.run_id, report=self.report)
        except Exception as error:  # noqa: BLE001 - the run may already be ending
            self.report(f"stop request for {self.run_id} failed: {error}")

    def restore(self) -> None:
        for number, handler in self.previous.items():
            signal.signal(number, handler)


def forward_stop_signals(run_id: str, report: Callable[[str], None]) -> _StopForwarding:
    """The Supervisor runs in its own session (``launch._spawn``), so a scheduler that stops this task by
    signalling its process group does not reach it. Forward SIGTERM / SIGINT / SIGHUP to the run as ORA's
    graceful stop (``ora stop``): the run seals as interrupted and releases its lease and GPU."""
    return _StopForwarding(run_id, report)


def follow_transcript(root: Path) -> subprocess.Popen | None:
    """Print the run's plain transcript (``ora watch --plain``) on this process's output while it runs.

    A scheduler that keeps a task's output (ora-router's task log) then shows what the run does: the task
    instruction, the model's replies and each tool call with its result summary. Images appear only as a count
    and the model's reasoning is left out. ``ora watch`` only reads the run; ``ORA_TASK_TRANSCRIPT=off`` skips it.
    """
    if os.environ.get(TRANSCRIPT_VARIABLE, "on").lower() == "off":
        return None
    command = [sys.executable, "-c", "import sys; from ora.app.main import main; sys.exit(main())",
               "watch", "--plain", "--ui", "plain", "--speed", "0", str(root)]
    try:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL)
    except OSError:
        return None


def stop_transcript(watcher: subprocess.Popen | None) -> None:
    if watcher is None:
        return
    try:
        watcher.wait(timeout=TRANSCRIPT_DRAIN_S)
    except subprocess.TimeoutExpired:
        watcher.terminate()
        try:
            watcher.wait(timeout=5)
        except subprocess.TimeoutExpired:
            watcher.kill()


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarize(run_id: str, root: Path, workspace: Path) -> dict[str, Any]:
    """What a caller needs from a finished run: where it is and its official outcome."""

    result, evaluation = _read(root / "result.json") or {}, _read(root / "evaluation.json") or {}
    ground_truth = workspace / "ground-truth" / run_id
    return {
        "run_id": run_id, "run_dir": str(root),
        "state": result.get("state"), "official_success": result.get("official_success"),
        "termination": result.get("termination"), "decisions": result.get("decisions"),
        "model_calls": result.get("model_calls"), "evaluation_status": evaluation.get("status"),
        "sealed": (root / "seal.json").is_file(),
        "ground_truth_dir": str(ground_truth) if ground_truth.is_dir() else None,
    }


def run_task(task: Mapping[str, Any], binding: Mapping[str, Any], *, overrides: Mapping[str, Any],
             summary_path: Path | None = None, wait: bool = True, poll_s: float = 10.0,
             report: Callable[[str], None] = print) -> dict[str, Any]:
    """Compose, preflight, launch and (``wait``) await one run; returns its task record."""

    from ora.orchestrator import paths
    from ora.orchestrator.launch import _preflight, _spawn
    from ora.orchestrator.spec.inputs import bind_service_environment
    from ora.orchestrator.spec.run_spec import load_run_spec, resolve_run_spec

    os.environ.update(prepare_environment(task, binding))
    spec = compose_spec(task, binding)
    model_request = apply_model_request(task, spec)
    bind_service_environment(spec)
    workspace = paths.experiment_workspace(spec["experiment"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    spec_path = workspace / "specs" / f"{spec['config']}--{stamp}-{os.getpid()}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report(f"spec {spec_path}")
    record: dict[str, Any] = {
        "schema": RECORD_SCHEMA, "task": dict(task), "overrides": dict(overrides),
        "binding": dict(binding), "spec_path": str(spec_path), "spec": spec,
        "model_request_overrides": model_request,
    }
    plan = resolve_run_spec(load_run_spec(spec_path))
    record.update(plan_sha256=plan["plan_sha256"], source=plan["source"])
    preflight = _preflight(plan, compact=True, report=report)
    if not preflight["passed"]:
        record["summary"] = {"run_id": None, "state": "preflight_blocked",
                             "failed_checks": [c["check"] for c in preflight["checks"] if not c["passed"]]}
        return _write(record, workspace / "tasks" / f"blocked--{spec_path.stem}.json", summary_path, report)
    # Pacing simulator starts per node is the scheduler's job (a router admits a task only with a start quota);
    # `ora task run` itself starts at once.
    run_id = _spawn(plan, preflight, yes=True, report=report)
    root = paths.resolve_run_root(run_id)
    record["summary"] = {"run_id": run_id, "run_dir": str(root), "state": "launched"}
    if wait:
        watcher = follow_transcript(root)
        stopping = forward_stop_signals(run_id, report)
        try:
            _wait(int(_read(root / "process.json")["pid"]), poll_s=poll_s)
        finally:
            stopping.restore()
            stop_transcript(watcher)
        if stopping.signal is not None:
            record["stopped_by_signal"] = stopping.signal
        record["summary"] = summarize(run_id, root, workspace)
    return _write(record, workspace / "tasks" / f"{run_id}.json", summary_path, report)


def _write(record: dict[str, Any], path: Path, summary_path: Path | None,
           report: Callable[[str], None]) -> dict[str, Any]:
    text = json.dumps(record, ensure_ascii=False, indent=1) + "\n"
    for target in (path, summary_path):
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    report(f"task record {path}")
    report(json.dumps(record["summary"], ensure_ascii=False))
    return record
