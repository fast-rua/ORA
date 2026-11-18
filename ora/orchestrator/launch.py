"""Launch one RunPlan and supervise it to its landing seal.

``_preflight`` runs the launch gates and persists their record. ``_spawn``
re-checks the plan, its preflight and the clean source, stages the run
directory, claims the plan's resource lease and starts the supervisor process
(``python -m ora.orchestrator.supervise RUN_ID CLAIM_TOKEN``). ``supervise`` is
that process: it runs the episode, writes the Runtime metrics and context
audit, seals the run at landing and releases its lease. ``request_stop``
signals exactly one run's supervisor. The ``ora`` CLI prints, confirms and
dispatches; it does not hold this code.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ora.agent.evidence import request_archive, sandbox_retention
from ora.agent.providers.credentials import launch_environment
from ora.version import ora_command

from . import paths
from .preflight import host_storage
from .evaluation.context_audit import write_context_audit
from .episode.coordinator import run_episode
from .evaluation.evaluate import seal
from .leases import (
    _assign_lease, _atomic, _create_lease, _lease_for_run, _load, _plan_resource_binding,
    _release_lease, lease_off_node,
)
from .evaluation.metrics import write_runtime_metrics
from .preflight.gates import run_preflight
from .preflight.provider_gateway import gateway_preflight
from .spec.run_spec import PROJECT_ROOT, validate_run_plan
from .spec.source_overlay import matches_plan, source_state


def _operator_status(root: Path) -> dict[str, Any]:
    """The run's persisted status."""

    return _load(root / "status.json")


def _run_root(run_id: str) -> Path:
    return paths.resolve_run_root(run_id)


def _preflight(plan: Mapping[str, Any], *, persist: bool = True, report: Any = print,
               compact: bool = False) -> dict[str, Any]:
    """``compact`` reports a failed check's detail only; the persisted record keeps every detail."""
    result = gateway_preflight(plan, run_preflight)  # loopback gateway client first
    if persist:
        _atomic(paths.workspace_from_plan(plan) / "preflight" / f"{plan['plan_sha256']}.json", result)
    for check in result["checks"]:
        line = f"[{'PASS' if check['passed'] else 'FAIL'}] {check['check']}"
        report(line if compact and check["passed"] else f"{line}: {check['detail']}")
    report(f"preflight: {'PASS' if result['passed'] else 'BLOCKED'}")
    return result


def _require_launchable(plan: Mapping[str, Any], *, retry: bool) -> None:
    admission = plan.get("launch_admission")
    if not isinstance(admission, Mapping):
        return
    mode = admission.get("mode")
    if mode == "disabled":
        raise RuntimeError("this RunPlan is context-only and cannot be launched")
    if mode == "single_attempt_cell" and retry:
        raise RuntimeError("this RunPlan forbids retries after its cohort claim")


def _claim_launch_admission(
    plan: Mapping[str, Any], run_id: str | None = None,
) -> dict[str, Any] | None:
    """Atomically consume an optional plan-bound single-attempt cell."""

    admission = plan.get("launch_admission")
    if not isinstance(admission, Mapping):
        return None
    if admission.get("mode") != "single_attempt_cell":
        return None
    root = (
        paths.workspace_from_plan(plan) / "launch-admissions"
        / str(admission["cohort_id"])
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{admission['cell_id']}.json"
    claim = {
        "schema": "ora.run-launch-claim.v1",
        "run_id": run_id,
        "experiment": plan["experiment"],
        "config": plan["config"],
        "cohort_id": admission["cohort_id"],
        "cell_id": admission["cell_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source"]["commit"],
        "created_unix_s": time.time(),
    }
    encoded = (
        json.dumps(claim, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{admission['cell_id']}.", suffix=".tmp", dir=root,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            # The immutable prior claim identifies the consumed attempt. A
            # corrupt record still blocks admission, but must not hide why.
            try:
                previous = _load(path)
                if not isinstance(previous, Mapping):
                    raise ValueError("claim is not an object")
                owner = (
                    f"run_id={previous.get('run_id') or 'unrecorded'}, "
                    f"plan_id={previous.get('plan_id', 'unknown')}, "
                    f"created_unix_s={previous.get('created_unix_s', 'unknown')}"
                )
            except (OSError, ValueError):
                owner = "existing claim metadata unavailable"
            raise RuntimeError(
                "single-attempt cohort cell was already claimed: "
                f"{admission['cohort_id']}/{admission['cell_id']}; {owner}; "
                "inspect that attempt; a new attempt requires separate authorization"
            ) from exc
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": claim["schema"],
        "path": str(path),
        "claim_sha256": hashlib.sha256(encoded).hexdigest(),
        "plan_sha256": plan["plan_sha256"],
    }


def _seal_at_landing(root: Path) -> None:
    """Seal the landed run in the same operator action, fail-open to manual.

    A refusal is recorded as evidence instead of failing the landing; no
    output may be written after a successful seal because the supervisor log
    is part of the sealed inventory.
    """

    if (root / "seal.json").exists():
        return
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        seal(root, mode="landing")
    except Exception as exc:
        _atomic(root / "seal-refusal.json", {
            "schema": "ora.seal-refusal.v1",
            "run_id": root.name, "stage": "landing", "reason": str(exc),
        })
        print(f"landing seal refused: {exc}")
        return
    request_archive.compact_after_seal(root)  # lossless, silent, fail-open (agent/evidence/request_archive.py)
    sandbox_retention.omit_after_seal(root)  # save_sandbox: false keeps only hashes (agent/evidence/sandbox_retention.py)


def _spawn(plan: Mapping[str, Any], preflight: Mapping[str, Any], *, yes: bool, retry_of: str | None = None, report: Any = print) -> str:
    validate_run_plan(plan)
    _require_launchable(plan, retry=retry_of is not None)
    if (
        preflight.get("schema") != "ora.preflight.v1"
        or preflight.get("passed") is not True
        or preflight.get("plan_id") != plan["plan_id"]
        or preflight.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise RuntimeError("launch preflight is not bound to this passing RunPlan")
    if not matches_plan(source_state(PROJECT_ROOT), plan):
        raise RuntimeError("source changed after preflight; refusing to stage run")
    credentialed = launch_environment(plan, preflight, os.environ)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"--retry-{int(time.time() * 1000) % 1_000_000:06d}" if retry_of else ""
    run_id = f"{plan['plan_id']}--{stamp}{suffix}"
    root = paths.new_run_root(plan, run_id)
    if paths.existing_run_location(plan, run_id) is not None:
        raise RuntimeError(f"run-id already exists: {run_id}")
    # Per-run TMPDIR off the system partition and own dead-PID carb segments
    # swept, both after confirmation and before the lease.
    staged = host_storage.stage_launch(plan, run_id)
    child_environment = {**credentialed, **host_storage.scratch_environment(staged)}
    lease: dict[str, Any] | None = None
    try:
        lease = _create_lease(plan, run_id, os.getpid())
        launch_admission = _claim_launch_admission(plan, run_id)
        root.mkdir(parents=True)
        _atomic(root / "plan.json", plan)
        from ora.contracts.toolkit_protocol import metadata as toolkit_protocol_metadata
        _atomic(root / "toolkit-protocol.json", toolkit_protocol_metadata(
            recording_enabled=bool(plan.get("record_toolkit_io")),
            implementation={"harness_commit": plan["source"]["commit"]}))
        _atomic(root / "preflight.json", preflight)
        _atomic(root / "launch.json", {
            "run_id": run_id, "non_interactive_confirmation": bool(yes),
            "launcher_pid": os.getpid(), "created_unix_s": time.time(),
            "retry_of": retry_of,
            "workspace": paths.workspace_topology(),
            "execution_resource": _plan_resource_binding(plan),
            "launch_admission": launch_admission,
            **staged,
        })
        paths.register_run(run_id, plan, root)
        log = (root / "supervisor.log").open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "ora.orchestrator.supervise",
                    run_id, str(lease["claim_token"]),
                ],
                cwd=PROJECT_ROOT, env=child_environment, stdout=log,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
        finally:
            log.close()
        lease = _assign_lease(lease, process.pid)
    except Exception:
        if lease is not None:
            _release_lease(run_id, claim_token=str(lease["claim_token"]))
        # The staged scratch goes; unlinked dead-PID segments stay unlinked.
        host_storage.discard_staged(staged)
        raise
    _atomic(root / "process.json", {"run_id": run_id, "pid": process.pid, "process_group": process.pid})
    host_storage.bind_scratch_owner(staged, process.pid)
    report(f"launched {run_id} pid={process.pid}")
    report(f"watch: {ora_command()} watch {run_id}")
    return run_id


def supervise(run_id: str, claim_token: str) -> int:
    """Run one launched episode as its supervisor process, seal it at landing and release its lease."""

    root = _run_root(run_id)
    plan = _load(root / "plan.json")
    formal_plan = (
        plan.get("schema") == "ora.run-plan.v1"
        and isinstance(plan.get("plan_sha256"), str)
        and len(plan["plan_sha256"]) == 64
    )
    if formal_plan:
        validate_run_plan(plan)
        preflight = _load(root / "preflight.json")
        if (
            preflight.get("passed") is not True
            or preflight.get("plan_sha256") != plan["plan_sha256"]
            or preflight.get("plan_id") != plan["plan_id"]
        ):
            raise RuntimeError("supervisor preflight is not bound to its RunPlan")
        if not matches_plan(source_state(PROJECT_ROOT), plan):
            raise RuntimeError("supervisor source differs from the admitted clean RunPlan")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        pending = _lease_for_run(run_id)
        if (
            pending is not None
            and pending.get("claim_token") == claim_token
            and pending.get("supervisor_pid") == os.getpid()  # null until the launcher records it
        ):
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("supervisor did not receive its exact resource lease")
    try:
        result = run_episode(run_root=root, plan=plan, run_id=run_id)
        write_runtime_metrics(root)
        write_context_audit(root)
        _seal_at_landing(root)
        return 0 if result["state"] == "completed" else 1
    finally:
        _release_lease(
            run_id, supervisor_pid=os.getpid(), claim_token=claim_token,
        )
        host_storage.release_run_scratch(root)


def request_stop(run_id: str, *, report: Any = print) -> None:
    """Ask exactly this run's supervisor, on this node, to stop gracefully (SIGTERM to its group)."""

    root = _run_root(run_id)
    process = _load(root / "process.json")
    lease = _lease_for_run(run_id)
    if not lease or lease.get("run_id") != run_id or process.get("run_id") != run_id:
        raise RuntimeError("the exact run-id is not the active lease")
    if node := lease_off_node(lease):
        raise RuntimeError(f"{run_id} is supervised on {node}; run stop on that node")
    pid = int(process["pid"])
    probe = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False)
    if probe.returncode or run_id not in probe.stdout:
        raise RuntimeError("refusing to signal a PID that is not this supervisor")
    os.killpg(int(process["process_group"]), signal.SIGTERM)
    report(f"graceful stop requested for {run_id}")
