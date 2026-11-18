"""Optional developer-only ground-truth capture of a RoboDojo episode.

The switch is ``runtime.ground_truth_capture``; when on, the worker writes
``<workspace>/ground-truth/<run_id>/`` beside ``runs/`` and nothing inside the
run directory changes. No simulator: the plant and the scene readers are fakes.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robodojo import ground_truth
from ora.environment.backends.robodojo.session import RobodojoSession
from ora.environment.benchmarks import robodojo


class FakeReaders:
    def __init__(self, fail_tick: bool = False, bad_field: bool = False):
        self.fail_tick = fail_tick
        self.bad_field = bad_field
        self.ticks = 0

    def physics(self):
        return {"instances": {"block_0": {"kind": "rigid", "simulator_readback": {"mass_kg": 0.1}}},
                "articulations": {}}

    def tick(self):
        self.ticks += 1
        if self.fail_tick:
            raise RuntimeError("reader broke")
        block = {"kind": "rigid", "position_m": [0.0, 0.0, 0.01 * self.ticks]}
        if self.bad_field:
            block["linear_velocity"] = {"error": "AttributeError: no view"}
        return {"objects": {"block_0": block}, "articulations": {}}


def test_capture_is_off_unless_the_runtime_says_true():
    assert ground_truth.enabled({}) is False
    assert ground_truth.enabled({"ground_truth_capture": False}) is False
    assert ground_truth.enabled({"ground_truth_capture": True}) is True
    with pytest.raises(ground_truth.GroundTruthError):
        ground_truth.enabled({"ground_truth_capture": "yes"})


def test_capture_lands_beside_runs_never_inside_the_run(tmp_path):
    run = tmp_path / "ws/runs/run-1"
    assert ground_truth.capture_root({"shared_root": str(run)}) == (tmp_path / "ws/ground-truth/run-1").resolve()
    with pytest.raises(ground_truth.GroundTruthError, match="runs/<run_id>"):
        ground_truth.capture_root({"shared_root": str(tmp_path / "replay-pass")})


def test_the_recorder_writes_physics_per_round_and_one_row_per_tick(tmp_path):
    recorder = ground_truth.GroundTruthRecorder(tmp_path / "gt", run_id="run-1", readers=FakeReaders())
    recorder.round_started(round_index=0, layout_id=3, physics_clock=0)
    for native in (1, 2):
        recorder.tick(round_index=0, action_sequence=0, action_id="a0", segment="command",
                      native_action=native, physics_clock=10 * native)
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path / "gt/ticks.jsonl").read_text().splitlines()]
    assert [row["native_action"] for row in rows] == [1, 2]
    assert rows[1]["objects"]["block_0"]["position_m"] == [0.0, 0.0, 0.02]
    physics = json.loads((tmp_path / "gt/physics/round-0000.json").read_text())
    assert physics["schema"] == ground_truth.PHYSICS_SCHEMA and physics["layout_id"] == 3
    manifest = json.loads((tmp_path / "gt/manifest.json").read_text())
    assert manifest["schema"] == ground_truth.CAPTURE_SCHEMA and manifest["ticks"]["rows"] == 2
    assert manifest["physics"]["rounds"] == [{"round_index": 0, "layout_id": 3, "path": "physics/round-0000.json"}]


def test_a_reader_failure_is_recorded_never_raised(tmp_path):
    recorder = ground_truth.GroundTruthRecorder(tmp_path / "gt", run_id="run-1", readers=FakeReaders(fail_tick=True))
    recorder.tick(round_index=0, action_sequence=0, action_id=None, segment=None, native_action=1, physics_clock=10)
    recorder.close()
    row = json.loads((tmp_path / "gt/ticks.jsonl").read_text())
    assert row["error"] == "RuntimeError: reader broke"
    assert json.loads((tmp_path / "gt/manifest.json").read_text())["errors"] == 1


def test_a_failed_field_inside_a_good_row_is_counted_apart_from_whole_read_errors(tmp_path):
    recorder = ground_truth.GroundTruthRecorder(tmp_path / "gt", run_id="run-1", readers=FakeReaders(bad_field=True))
    for native in (1, 2):
        recorder.tick(round_index=0, action_sequence=0, action_id=None, segment=None,
                      native_action=native, physics_clock=10 * native)
    recorder.close()
    manifest = json.loads((tmp_path / "gt/manifest.json").read_text())
    assert (manifest["errors"], manifest["field_errors"]) == (0, 2)
    assert manifest["first_field_error"] == "AttributeError: no view"


def test_the_recorder_refuses_a_non_empty_directory(tmp_path):
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt/ticks.jsonl").write_text("{}\n")
    with pytest.raises(ground_truth.GroundTruthError, match="non-empty"):
        ground_truth.GroundTruthRecorder(tmp_path / "gt", run_id="run-1", readers=FakeReaders())


def _session(tmp_path: Path, capture: bool | None) -> tuple[RobodojoSession, Path]:
    from tests.support.robodojo.backend import Plant, _fake_source_manifest

    run = tmp_path / "ws/runs/run-1"
    config = {"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}, "shared_root": str(run)}
    if capture is not None:
        config["ground_truth_capture"] = capture
    session = RobodojoSession(
        config=config, output_root=run / "native",
        environment_factory=lambda config: (Plant(), SimpleNamespace(close=lambda: None),
                                            _fake_source_manifest(tmp_path)))
    session._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    return session, run


@pytest.mark.parametrize("capture", [None, False, True])
def test_the_session_writes_ground_truth_only_when_asked_and_only_outside_the_run(tmp_path, monkeypatch, capture):
    monkeypatch.setattr(ground_truth, "SceneReaders", lambda session: FakeReaders())
    session, run = _session(tmp_path, capture)
    try:
        session.start()
        session._conformance_tick({})
        session._conformance_tick({})
    finally:
        session.close()
    gt = tmp_path / "ws/ground-truth/run-1"
    assert not any("ground-truth" in str(path) or path.name == "ticks.jsonl" for path in run.rglob("*"))
    if capture is not True:
        assert not gt.exists()
        return
    rows = [json.loads(line) for line in (gt / "ticks.jsonl").read_text().splitlines()]
    assert [(row["segment"], row["round_index"]) for row in rows] == [
        ("round_start", 0), ("conformance", 0), ("conformance", 0)]
    assert [row["native_action"] for row in rows[1:]] == [rows[0]["native_action"] + 1, rows[0]["native_action"] + 2]
    assert rows[0]["action_id"] is None
    assert (gt / "physics/round-0000.json").exists()
    assert json.loads((gt / "manifest.json").read_text())["ticks"]["rows"] == 3


def _runtime(task: str) -> dict:
    """An otherwise admissible runtime for ``task`` (the admission fixture's shape)."""
    from ora.environment.benchmarks.robodojo_physics import physics_device_for

    return {
        "kind": "robodojo", "transport": "local_rpc", "root_env": "ROBODOJO_ROOT",
        "python_env": "ROBODOJO_PYTHON", "asset_root_env": "ROBODOJO_ASSET_ROOT",
        **{field: robodojo.REVISIONS[name] for name, field in robodojo.REVISION_FIELDS.items()},
        "task_variant": task, "eval_seed": 0, "layout_id": 0,
        "layout_file": f"Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json",
        "layout_sha256": "a" * 64, "num_envs": 1, "env_config": "arx_x5",
        "sensor_profile": "rgbd_metric_v1", "physics_dt": 0.004, "control_hz": 25,
        "native_action_limit": robodojo.NATIVE_ACTION_LIMITS[task], "native_actions_per_command": 20,
        "main_camera": "head", "policy_cameras": list(robodojo.PUBLIC_CAMERAS),
        "required_host": "test-host", "require_gpu": True, "physics_device": physics_device_for(task),
        "renderer_gpu": 4, "cuda_visible_devices": "4", "startup_timeout_s": 600,
        "request_timeout_s": 180, "close_timeout_s": 30,
        "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
        "infrastructure_qualification": {"path": "outputs/x/qualification/fixture.json", "sha256": "b" * 64},
    }


def test_the_runtime_admits_only_a_boolean_switch():
    robodojo.validate_runtime_extension({**_runtime("stack_bowls"), "ground_truth_capture": True})
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="ground_truth_capture"):
        robodojo.validate_runtime_extension({**_runtime("stack_bowls"), "ground_truth_capture": "on"})
