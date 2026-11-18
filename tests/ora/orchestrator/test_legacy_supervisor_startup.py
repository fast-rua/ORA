"""Enter the real Supervisor, Pi host and episode startup without a simulator."""
import json
import os
import signal
import subprocess
from types import SimpleNamespace

import pytest

from ora.agent.model_services import clients
from ora.orchestrator import launch, supervise
from ora.orchestrator.episode import coordinator
from ora.orchestrator.spec.run_spec import resolve_run_spec
from tests.support.orchestrator.run_specs import robodojo_native_spec
from tests.support.orchestrator.control_plane import _service_set_plan


# The legacy-visible surface left is v5 (RoboDojo's default, tool_output receipts); legacy-visible-v1 and its replay
# baseline, with v3's observation delivery, were retired on 2026-10-31.
@pytest.mark.parametrize("baseline", ["robodojo_toolkit_v6"])
@pytest.mark.parametrize("delivery", ["tool_output"])
def test_supervisor_reaches_environment_start_with_granted_legacy_surface(tmp_path, monkeypatch, baseline, delivery):
    # External source/lease admission is synthetic. All plan validation, real
    # Supervisor entry, tool runtime assembly, Pi process startup, manifest
    # writing and episode lifecycle code remain production implementations.
    real_run = subprocess.run

    def git_clean(argv, *args, **kwargs):
        if argv[:1] == ["git"] and "status" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", git_clean)
    # Resident HTTP health checks are external; retain real client construction
    # and binding validation while substituting only service readiness.
    for client in (clients.Sam3ApiSegmenter, clients.ContactGraspNetServiceClient):
        monkeypatch.setattr(client, "require_ready", lambda self: dict(self.expected_binding))
    spec = robodojo_native_spec("general_pickup")
    spec["policy"]["baseline"] = baseline
    spec["prerequisite"] = _service_set_plan()["prerequisite"]
    spec["prerequisite"]["capability_scope"] = [
        "resident_sam3_point_segmentation_v1", "resident_sam3_text_grounding_v1", "resident_contact_graspnet_v1"]
    for child in spec["prerequisite"]["services"].values():
        child.update(
            tool_surface="robodojo_placement_toolkit_v5", pid=4242,
            started_unix_s=1.0, adapter_sha256="a" * 64,
            model={"family": "cpu-fixture"}, load_count=1, warmup_complete=True,
            qualification={"status": "passed", "sha256": "b" * 64, "created_unix_s": 1.0},
        )
    spec["toolkit_grants"] = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    spec["toolkit_receipt_delivery"] = delivery
    plan = resolve_run_spec(spec)
    run_root = tmp_path / "outputs/experiment/runs/startup"
    run_root.mkdir(parents=True)
    (run_root / "plan.json").write_text(json.dumps(plan))
    (run_root / "preflight.json").write_text(json.dumps({
        "passed": True, "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"],
        "checks": [{"check": "robodojo_infrastructure_qualification", "passed": True,
                    "detail": {"schema": "ora.robodojo-unqualified-admission.v1", "qualification_available": False}}]}))
    monkeypatch.setattr(launch, "_run_root", lambda _: run_root)
    monkeypatch.setattr(launch, "_lease_for_run", lambda _: {"claim_token": "cpu", "supervisor_pid": os.getpid()})
    released = []
    monkeypatch.setattr(launch, "_release_lease", lambda *a, **k: released.append(k))

    class CPUEnvironment:
        started = False
        closed = False

        def start(self):
            self.started = True
            # This boundary is reached only after manifest publication and
            # status=running. Deliberate fault avoids any rendering or physics.
            raise RuntimeError("cpu_environment_start_reached")

        def close(self):
            self.closed = True

    environment = CPUEnvironment()
    planners = []

    def policy_factory(**kwargs):
        planner = coordinator.build_policy_session(**kwargs)
        planners.append(planner)
        return planner

    monkeypatch.setattr(launch, "run_episode", lambda **kwargs: coordinator.run_episode(
        **kwargs, environment_session_factory=lambda **_: environment,
        policy_session_factory=policy_factory, provider_override=SimpleNamespace(responses=[])))
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        code = supervise.main(["startup", "cpu"])
    finally:
        for planner in planners:
            planner.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    assert code == 1
    assert environment.started and environment.closed
    assert released and released[0]["claim_token"] == "cpu"
    manifest = json.loads((run_root / "context-manifest.json").read_text())
    definitions = manifest["tool_surface"]["advertised_in_order"]
    assert [{k: v for k, v in d.items() if k != "state_effect"} for d in definitions] == (
        plan["policy_composition"]["tool_surface"]["definitions"]
    )
    assert "check_task" in [d["name"] for d in definitions]
    result = json.loads((run_root / "result.json").read_text())
    assert "cpu_environment_start_reached" in json.dumps(result)
    assert json.loads((run_root / "status.json").read_text())["state"] == "incomplete"
    grants = json.loads((run_root / "native/toolkit-state-grants.json").read_text())
    assert grants["effective_keys"] == ["privileged.task_check"]
    assert grants["tool_grants"] == ["check_task"]
