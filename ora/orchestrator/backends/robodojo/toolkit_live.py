"""Lease-bound toolkit diagnostic worker; no change to the qualified worker route."""

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from ora.agent.toolkit.interface.types import Divergence
from ora.environment.rpc import LocalRpcEnvironmentSession
from ora.orchestrator.paths import PROJECT_ROOT
from ora.orchestrator.preflight import host_storage
from ora.orchestrator.preflight.resources import resolve_execution_resource
from ora.orchestrator.leases import (
    assign_resource_lease,
    claim_resource_lease,
    release_resource_lease,
)
from .motion.replay import (
    _worker_config,
    launch_identity,
    physics_step_divergence,
    replay_runtime,
    harness_identity,
    _run_identity,
    ReplayLaunchError,
)

WORKER_MODULE = "ora.environment.backends.robodojo.toolkit_worker"
DIAGNOSTIC_SOURCES = (
    "ora/environment/backends/robodojo/toolkit_worker.py",
    "ora/environment/backends/robodojo/toolkit_restore.py",
    "ora/environment/backends/robodojo/toolkit_render.py",
    "ora/environment/backends/robodojo/control_replay.py",
    "ora/orchestrator/backends/robodojo/toolkit_live.py",
)

NATIVE_DIAGNOSTIC_SOURCES = (*DIAGNOSTIC_SOURCES, "ora/environment/backends/robodojo/toolkit_native_state.py")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Session(LocalRpcEnvironmentSession):
    """The existing transport lifecycle, with only its private entry point replaced."""

    def _spawn_worker(self, log_path, config_path, environment):
        with log_path.open("a") as log:
            self.process = subprocess.Popen(
                [
                    str(self.python_executable),
                    "-m",
                    WORKER_MODULE,
                    "--socket",
                    str(self.socket_path),
                    "--config",
                    str(config_path),
                ],
                cwd=self.project_root,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        self._persist_record(state="starting", diagnostic_entry=WORKER_MODULE)

    def diagnostic(self, method, **params):
        return self._request("toolkit_" + method, params)


def state_differences(expected, actual):
    """Exact public robot/clock checks; optional object poses remain verifier-only."""
    differences = [
        key
        for key in (
            "round_index",
            "physics_clock",
            "native_actions",
            "sim_steps",
            "proprioception",
        )
        if expected[key] != actual.get(key)
    ]
    objects = expected.get("objects")
    if objects is not None:
        actual_objects = actual.get("objects") or {}
        if set(objects) != set(actual_objects):
            differences.append("objects.keys")
        for name in sorted(set(objects) & set(actual_objects)):
            for recorded, fresh in (
                ("position_m", "position_m"),
                ("orientation_wxyz", "orientation"),
            ):
                if objects[name].get(recorded) != actual_objects[name].get(fresh):
                    differences.append(f"objects.{name}.{recorded}")
    return differences


def verify_prefix(boundary, restored, worker_root):
    expected = boundary["expected"]
    replay = restored["replay"]
    failures = state_differences(expected, restored["actual"])
    if (
        replay["ticks_recorded"] != boundary["native_ticks"]
        or replay["ticks_replayed"] != boundary["native_ticks"]
    ):
        failures.append("control_prefix.tick_count")
    if replay["applied_target_mismatches"] or any(
        (replay.get("max_divergence") or {}).values()
    ):
        failures.append("control_prefix.native_state")
    physics = physics_step_divergence(
        Path(boundary["source_native"]), worker_root, upto=expected["native_receipts"]
    )
    if (
        physics["actions_compared"] != expected["native_receipts"]
        or physics["line_count_mismatches"]
    ):
        failures.append("control_prefix.physics_count")
    if any((physics.get("max_divergence") or {}).values()):
        failures.append("control_prefix.physics_state")
    # The qualification comparator checks controls and joints, not clocks.
    for i in range(expected["native_receipts"]):
        name = f"controller/action-{i:04d}.jsonl"
        left = [
            json.loads(line)
            for line in (Path(boundary["source_native"]) / name)
            .read_bytes()
            .splitlines()
        ]
        right = [
            json.loads(line) for line in (worker_root / name).read_bytes().splitlines()
        ]
        if any(
            any(a.get(k) != b.get(k) for k in ("physics_clock", "native_action"))
            for a, b in zip(left, right)
        ):
            failures.append(f"control_prefix.clock[{i}]")
    return dict(
        passed=not failures,
        differences=failures,
        physics_prefix=physics,
        expected=expected,
        actual=restored["actual"],
        replay=replay,
    )


def native_identity(source):
    """New snapshot path: source versions warn, clean host/checkout still required."""
    plan = json.loads((source / "plan.json").read_bytes())
    host = plan["runtime"].get("required_host")
    if host is not None and os.environ.get("ORA_EXECUTION_HOST") != host:
        raise ReplayLaunchError("the recorded execution host marker is required")
    identity = harness_identity(_run_identity(source)["harness_commit"])
    if identity["commit"] is None or identity["tracked_worktree_dirty"] is not False:
        raise ReplayLaunchError("replay launches only from a clean committed Git checkout")
    return identity


def bind_timing(runtime, root):
    """Write what a run's preflight leaves its worker (``motion_timing.bind_preflight``).

    A diagnostic has no preflight, and its isolated worker's environment holds no timing
    interpreter, so without this record a move falls back to ``motion_pace``'s timing.
    """
    timing = runtime.get("motion_timing")
    if not isinstance(timing, Mapping):
        return
    from ora.orchestrator.preflight.checks import motion_timing_check

    started = motion_timing_check(timing, environment_overrides=runtime["resource_binding"]["worker_environment"])
    if not started["passed"]:
        raise ReplayLaunchError(f"the run's timing worker did not start: {started['detail']}")
    write(root / "preflight.json", dict(checks=[started]))


@contextmanager
def open_live(*, source, output, boundary, profile, execute_action, progress, stack=None):
    """Yield only a worker whose controls and selected physical state matched exactly.

    ``stack`` replaces runtime fields for this worker alone; the sealed plan stays as it is.
    """
    native_snapshot = boundary.get("schema") == "ora.toolkit.native-boundary.v1"
    identity = (native_identity(source) if native_snapshot
                else launch_identity(source))
    plan = json.loads((source / "plan.json").read_bytes())
    binding = resolve_execution_resource(
        profile,
        runtime_kind="robodojo",
        required_host=plan["runtime"].get("required_host"),
    )
    runtime = {**replay_runtime(plan["runtime"], binding), **deepcopy((stack or {}).get("runtime", {}))}
    root = output / "live-worker"
    root.mkdir()
    template, config_path = _worker_config(
        runtime, plan, root, Path(boundary["source_native"])
    )
    config = json.loads(config_path.read_bytes())
    if native_snapshot and execute_action:
        raise ValueError("native snapshot restoration never grants physical actions")
    from ora.agent.toolkit.interface.protocol import audit
    config["protocol_compatibility"] = audit(source)
    config.update(
        toolkit_boundary=boundary,
        toolkit_progress=str(progress.path),
        toolkit_execute_action=bool(execute_action),
    )
    worker_output = (output if native_snapshot else root) / "environment"
    if native_snapshot:
        # RPC start derives shared_root from output_root.parent. Keep both the
        # rendered publication and simulator output in this independent root.
        config["output_root"] = str(worker_output.resolve())
        destination = (output / "policy-observations").resolve()
        if not destination.is_relative_to(worker_output.parent.resolve()):
            raise ValueError("native observation destination escapes the replay workspace")
    bind_timing(runtime, worker_output.parent)
    owner = "toolkit-live-" + uuid.uuid4().hex
    staged = host_storage.stage_launch({"runtime": runtime}, owner)
    lease, session, assigned = None, None, False
    launch_published = False
    legacy_cleanup = False
    try:
        launch = dict(
            schema="ora.toolkit.live-launch.v1",
            run_id=owner,
            profile=profile,
            identity=identity,
            diagnostic_sources={
                path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
                for path in (NATIVE_DIAGNOSTIC_SOURCES if native_snapshot else DIAGNOSTIC_SOURCES)
            },
            **staged,
        )
        if identity.get("worker_kernel_matches_run") is not True:
            launch["warnings"] = [dict(kind="kernel_admission", evidence=identity.get("kernel_admission"),
                                       physical_verification_required=True)]
        write(root / "launch.json", launch)
        launch_published = True
        lease_plan = dict(
            schema="ora.robodojo-infrastructure-plan.v1",
            runtime=runtime,
            harness_commit=identity["commit"],
            episodes=1,
            task_solving=False,
            diagnostic="toolkit live restoration",
        )
        lease_plan["plan_sha256"] = hashlib.sha256(
            json.dumps(lease_plan, sort_keys=True).encode()
        ).hexdigest()
        legacy_cleanup = True
        lease = claim_resource_lease(lease_plan, owner, os.getpid())
        assign_resource_lease(lease, os.getpid())
        assigned = True
        session = Session(
            python_executable=template.python_executable,
            project_root=template.project_root,
            output_root=worker_output,
            worker_config=config,
            socket_root=template.requested_socket_root,
            startup_timeout_s=template.startup_timeout_s,
            request_timeout_s=7200,
            close_timeout_s=20,
            worker_environment={
                **template.worker_environment,
                **host_storage.scratch_environment(staged),
            },
        )
        from ora.orchestrator.episode.execution_state import bind_execution_state
        bind_execution_state(session)
        progress.start("boot_worker", "Start independent diagnostic worker")
        session.start()
        write(root / "launch.json", dict(launch, worker_pid=session.process.pid))
        progress.start("replay_controls", "Replay the selected native control prefix")
        progress.emit(
            "replay_controls",
            "Control prefix ready",
            done=0,
            total=boundary["native_ticks"],
        )
        restored = session.diagnostic("restore")
        progress.start(
            "verify_state", "Verify controls, physics clock and restored state"
        )
        verification = verify_prefix(
            boundary, restored, session.output_root / "restored-prefix"
        )
        write(output / "verification.json", verification)
        if not verification["passed"]:
            raise Divergence(
                "restored boundary differs: " + ", ".join(verification["differences"])
            )
        yield session
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            try:
                if lease is not None:
                    release_resource_lease(
                        owner,
                        owner_pid=os.getpid() if assigned else None,
                        claim_token=lease["claim_token"],
                    )
            finally:
                if native_snapshot:
                    if launch_published:
                        host_storage.release_run_scratch(root)
                    else:
                        host_storage.discard_staged(staged)
                elif legacy_cleanup:
                    # Frozen legacy lifecycle: early launch failures predate
                    # its resource-cleanup scope. Do not change that route.
                    host_storage.release_run_scratch(root)
