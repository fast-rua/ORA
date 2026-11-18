"""Exact physical verification and cleanup of the separate live diagnostic lane."""

from copy import deepcopy
from types import SimpleNamespace
import json
import pytest

from ora.agent.toolkit.interface.types import Divergence
from ora.orchestrator.backends.robodojo import toolkit_live as live
from ora.orchestrator.toolkit.progress import Progress


def expected():
    return dict(
        round_index=0,
        physics_clock=100,
        native_actions=2,
        sim_steps=20,
        proprioception=dict(end_effectors=dict(left=dict(arm_joint_qpos=[0.1]))),
        objects=dict(
            object_a=dict(position_m=[1, 2, 3], orientation_wxyz=[1, 0, 0, 0])
        ),
    )


def actual():
    value = expected()
    value["objects"]["object_a"]["orientation"] = value["objects"]["object_a"].pop(
        "orientation_wxyz"
    )
    return value


@pytest.mark.parametrize(
    "field",
    [
        "round_index",
        "physics_clock",
        "native_actions",
        "sim_steps",
        "proprioception",
        "objects",
    ],
)
def test_state_verification_rejects_every_boundary_dimension(field):
    state = actual()
    state[field] = None
    assert live.state_differences(expected(), state)
    assert live.state_differences(expected(), actual()) == []


@pytest.mark.parametrize("native_snapshot", [False, True])
@pytest.mark.parametrize("failure", ["launch", "boot", "restore", "mismatch", "tool", "none"])
def test_diagnostic_lease_and_scratch_release_even_on_failure(
    tmp_path, monkeypatch, failure, native_snapshot
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "plan.json").write_text(json.dumps(dict(runtime=dict(kind="robodojo"))))
    output = tmp_path / "out"
    output.mkdir()
    order = []
    monkeypatch.setattr(
        live,
        "native_identity" if native_snapshot else "launch_identity",
        lambda source, **kw: dict(commit="a" * 40, worker_kernel_matches_run=not native_snapshot,
                                    kernel_admission=dict(status="control_path_changed")),
    )
    monkeypatch.setattr(
        live,
        "resolve_execution_resource",
        lambda *args, **kw: dict(
            worker_environment=dict(CUDA_VISIBLE_DEVICES="0"), profile_id="fake"
        ),
    )
    monkeypatch.setattr(live.host_storage, "stage_launch", lambda *args: {})
    monkeypatch.setattr(live.host_storage, "discard_staged", lambda staged: order.append("scratch_discard"))
    if failure == "launch":
        def failed_write(*args):
            raise RuntimeError("launch evidence unavailable")
        monkeypatch.setattr(live, "write", failed_write)
    monkeypatch.setattr(live.host_storage, "scratch_environment", lambda staged: {})
    monkeypatch.setattr(
        live.host_storage,
        "release_run_scratch",
        lambda root: order.append("scratch_release"),
    )
    monkeypatch.setattr(
        live, "claim_resource_lease", lambda *args: dict(claim_token="t")
    )
    monkeypatch.setattr(
        live, "assign_resource_lease", lambda *args: order.append("assign")
    )
    monkeypatch.setattr(
        live,
        "release_resource_lease",
        lambda *args, **kw: order.append("lease_release"),
    )

    def worker_config(runtime, plan, root, native):
        path = root / "worker-config.json"
        path.write_text("{}")
        return (
            SimpleNamespace(
                python_executable="unused",
                project_root=tmp_path,
                requested_socket_root=tmp_path,
                startup_timeout_s=1,
                worker_environment={},
            ),
            path,
        )

    monkeypatch.setattr(live, "_worker_config", worker_config)

    class Session:
        def __init__(self, **kw):
            self.output_root = kw["output_root"]
            if native_snapshot:
                assert self.output_root.parent == output
                assert kw["worker_config"]["output_root"] == str(self.output_root)
            self.process = SimpleNamespace(pid=1)

        def start(self):
            order.append("boot")
            if failure == "boot":
                raise RuntimeError("boot")

        def diagnostic(self, method):
            order.append(method)
            if failure == "restore":
                raise RuntimeError("restore")
            return {}

        def close(self):
            order.append("close")

    monkeypatch.setattr(live, "Session", Session)
    monkeypatch.setattr(
        live,
        "verify_prefix",
        lambda *args: dict(
            passed=failure != "mismatch",
            differences=["clock"] if failure == "mismatch" else [],
        ),
    )

    def run():
        with live.open_live(
            source=source,
            output=output,
            boundary=dict(source_native=str(source), native_ticks=2,
                          schema="ora.toolkit.native-boundary.v1" if native_snapshot else "ora.toolkit.live-boundary.v1"),
            profile="fake",
            execute_action=False,
            progress=Progress(output),
        ):
            order.append("tool")
            if failure == "tool":
                raise RuntimeError("tool")

    if failure == "none":
        run()
    else:
        with pytest.raises((RuntimeError, Divergence)):
            run()
    if native_snapshot and failure != "launch":
        launch = json.loads((output / "live-worker/launch.json").read_bytes())
        assert launch["warnings"][0]["kind"] == "kernel_admission"
    if failure == "launch":
        assert order == (["scratch_discard"] if native_snapshot else [])
    else:
        assert order[-3:] == ["close", "lease_release", "scratch_release"]
        launch = json.loads((output / "live-worker/launch.json").read_bytes())
        native_helper = "ora/environment/backends/robodojo/toolkit_native_state.py"
        assert (native_helper in launch["diagnostic_sources"]) == native_snapshot
    assert ("tool" in order) == (failure in {"tool", "none"})


@pytest.mark.parametrize("change", ["clock", "applied", "missing_tick", "none"])
def test_full_physics_prefix_checks_more_than_endpoint(tmp_path, change):
    import shutil
    from tests.support.robodojo.journal_fixture import (
        write_native,
        action,
        receipt,
        tick,
        control,
    )

    source = (
        write_native(
            tmp_path / "source",
            [[action(receipt("move_pose"), [tick("move", control())])]],
        )
        / "native"
    )
    restored = tmp_path / "restored"
    shutil.copytree(source, restored)
    expected_state = expected()
    expected_state.update(
        native_receipts=1, native_actions=1, sim_steps=10, physics_clock=610
    )
    current = actual()
    current.update(native_actions=1, sim_steps=10, physics_clock=610)
    report = dict(
        actual=current,
        replay=dict(
            ticks_recorded=1,
            ticks_replayed=1,
            applied_target_mismatches=0,
            max_divergence={},
        ),
    )
    path = restored / "controller/action-0000.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "clock":
        rows[0]["physics_clock"] += 1
    elif change == "applied":
        rows[0]["applied_position_targets"]["left_arm_joint_state"][0] += 0.01
    elif change == "missing_tick":
        rows.pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    verified = live.verify_prefix(
        dict(expected=expected_state, source_native=str(source), native_ticks=1),
        report,
        restored,
    )
    assert verified["passed"] == (change == "none")


def test_a_named_timing_is_bound_for_the_worker_as_a_preflight_would(tmp_path, monkeypatch):
    started = dict(check="motion_timing_worker", passed=True, detail=dict(python="/p", python_sha256="s"))
    calls = []

    def check(timing, *, environment_overrides):
        calls.append((timing, environment_overrides))
        return started

    monkeypatch.setattr("ora.orchestrator.preflight.checks.motion_timing_check", check)
    runtime = dict(motion_timing=dict(implementation="toppra_ruckig", python_env="ORA_TIMING_PYTHON"),
                   resource_binding=dict(worker_environment=dict(CUDA_VISIBLE_DEVICES="3")))
    live.bind_timing(runtime, tmp_path)
    assert json.loads((tmp_path / "preflight.json").read_bytes()) == dict(checks=[started])
    assert calls == [(runtime["motion_timing"], dict(CUDA_VISIBLE_DEVICES="3"))]


def test_no_named_timing_leaves_no_record(tmp_path):
    live.bind_timing(dict(), tmp_path)
    assert not (tmp_path / "preflight.json").exists()


def test_a_timing_worker_that_will_not_start_stops_the_launch(tmp_path, monkeypatch):
    monkeypatch.setattr("ora.orchestrator.preflight.checks.motion_timing_check",
                        lambda *a, **k: dict(check="motion_timing_worker", passed=False, detail="ORA_TIMING_PYTHON is missing"))
    runtime = dict(motion_timing=dict(python_env="ORA_TIMING_PYTHON"), resource_binding=dict(worker_environment={}))
    with pytest.raises(live.ReplayLaunchError, match="ORA_TIMING_PYTHON"):
        live.bind_timing(runtime, tmp_path)
