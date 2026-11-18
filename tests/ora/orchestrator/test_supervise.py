"""The supervisor process entry: what `launch._spawn` starts, and how it reports a refusal."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ora.orchestrator import launch, paths, supervise


def test_a_refused_supervisor_exits_2_with_the_reason_after_applying_stored_keys(monkeypatch, capsys):
    applied, seen = [], []
    monkeypatch.setattr(supervise, "apply_stored_credentials", lambda: applied.append(True))

    def refuse(run_id, claim_token):
        seen.append((run_id, claim_token))
        raise RuntimeError("supervisor did not receive its exact resource lease")

    monkeypatch.setattr(supervise, "supervise", refuse)
    assert supervise.main(["run-1", "token-1"]) == 2
    assert applied == [True] and seen == [("run-1", "token-1")]
    assert capsys.readouterr().err == "ora: supervisor did not receive its exact resource lease\n"


def test_the_supervisor_returns_the_episode_exit_code(monkeypatch):
    monkeypatch.setattr(supervise, "apply_stored_credentials", lambda: None)
    monkeypatch.setattr(supervise, "supervise", lambda run_id, claim_token: 1)
    assert supervise.main(["run-1", "token-1"]) == 1
    with pytest.raises(SystemExit):
        supervise.main(["run-1"])


@pytest.mark.parametrize("recording", [False, True])
def test_spawn_starts_this_module_with_the_run_and_its_claim_token(monkeypatch, tmp_path, recording):
    # `request_stop` signals a PID only when its command line names the run.
    plan = {"plan_id": "demo-m0-deadbeef0000", "plan_sha256": "a" * 64, "source": {"commit": "c" * 40},
            "experiment": "demo", "output_workspace": "outputs/demo", "record_toolkit_io": recording}
    preflight = {"schema": "ora.preflight.v1", "passed": True,
                 "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"]}
    monkeypatch.setattr(launch, "validate_run_plan", lambda *_: None)
    monkeypatch.setattr(launch, "_require_launchable", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(
        command, 0, stdout=plan["source"]["commit"] if "rev-parse" in command else ""))
    monkeypatch.setattr(launch, "launch_environment", lambda *_args: {})
    for name, value in (("new_run_root", lambda *_args: tmp_path / "run"), ("existing_run_location", lambda *_args: None),
                        ("workspace_topology", lambda: {}), ("register_run", lambda *_args: None)):
        monkeypatch.setattr(paths, name, value)
    monkeypatch.setattr(launch, "_create_lease", lambda *_args: {"claim_token": "claim-1"})
    monkeypatch.setattr(launch, "_claim_launch_admission", lambda *_args: None)
    monkeypatch.setattr(launch, "_plan_resource_binding", lambda *_args: {})
    monkeypatch.setattr(launch, "_assign_lease", lambda lease, pid: lease)
    started = []
    monkeypatch.setattr(launch.subprocess, "Popen",
                        lambda command, **_kwargs: started.append(command) or SimpleNamespace(pid=4242))
    run_id = launch._spawn(plan, preflight, yes=True, report=lambda line: None)
    assert started == [[sys.executable, "-m", "ora.orchestrator.supervise", run_id, "claim-1"]]
    protocol = json.loads((tmp_path / "run/toolkit-protocol.json").read_bytes())
    assert protocol["protocol_version"] == 1
    assert protocol["recording_enabled"] is recording
    assert protocol["implementation"]["harness_commit"] == plan["source"]["commit"]
