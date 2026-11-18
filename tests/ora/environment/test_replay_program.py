"""The replay program: a run's actions as code that reproduces it (``ora/environment/replay_program.py``)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ora.environment import replay_program as rp
from ora.environment import replay_program_worker as program_worker
from ora.environment import replay_worker as worker
from ora.environment.controller_replay import ControllerReplayRecorder, load_controller_replay_manifest
from ora.environment.robolab_replay import receipt_clock_failures, replay_in_environment_worker

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POSE = {"left": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], "right": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]}


def _dense(sequence: int, steps: int, **fields: Any) -> dict[str, Any]:
    return {"action_sequence": sequence, "action_id": f"a{sequence}", "action": "move_pose", "round_index": 0,
            "controller": "robotwin_dense_joint_targets", "replayable": True, "physics_steps": steps,
            "execution": {"status": "reached", "settle_steps": 0}, **fields}


def _fault(sequence: int, steps: int) -> dict[str, Any]:
    tracking = {"physics_step_attempts": steps, "sampled_steps": steps, "controller_write_attempts": steps,
                "mode": "global", "abort": {"code": "inter_arm_collision"}}
    return _dense(sequence, steps, replayable=False,
                  execution={"status": "execution_fault", "settle_steps": 0, "fault": "guard"},
                  metadata={"controller_tracking": tracking, "control_sequence": {
                      "left_arm": None, "left_gripper": None, "right_gripper": None,
                      "right_arm": {"position": {"array": "p"}, "velocity": {"array": "v"}}}})


ENTRIES = [
    _dense(0, 3),
    {"action_sequence": 1, "action_id": "a1", "action": "wait", "round_index": 0, "replayable": False,
     "controller": "robotwin_hold_current_drive_targets", "physics_steps": 5, "execution": {"status": "executed"}},
    _fault(2, 4),
    _dense(3, 0, replayable=False, execution={"status": "planner_failed"}),
]


def _robotwin_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: list[dict[str, Any]] = ENTRIES,
                  *, kind: str = "robotwin") -> Path:
    run = tmp_path / "runs" / "ora-test-robotwin-0001"
    native = run / "native"
    (native / "controller-replay").mkdir(parents=True)
    (native / "receipts").mkdir()
    (run / "plan.json").write_text(json.dumps({"task": "adjust_bottle", "seed": 7, "source": {"commit": "c0ffee"},
                                               "runtime": {"kind": kind}}))
    (native / "controller-replay" / "manifest.json").write_text(json.dumps({"backend": "robotwin"}))
    for entry in entries:
        (native / "receipts" / f"action-{entry['action_sequence']:04d}.json").write_text(json.dumps(
            {"action_id": entry["action_id"], "action": entry["action"],
             "after": {arm: {"pose_xyz_xyzw": pose} for arm, pose in POSE.items()}}))
    (native / "official-verdict.json").write_text(json.dumps(
        {"official_success": True, "official_terminal": {"action_sequence": 2}}))
    monkeypatch.setattr("ora.environment.controller_replay.load_controller_replay_manifest",
                        lambda _path: {"backend": "robotwin", "entries": entries})
    monkeypatch.setattr(worker, "_artifact_arrays", lambda _path, entry: {
        "p": np.zeros((entry["physics_steps"], 6)), "v": np.zeros((entry["physics_steps"], 6))})
    return run


def test_every_backend_has_a_program_and_a_scene_and_every_record_a_verdict() -> None:
    assert rp.PROGRAM_BACKENDS == {"robotwin", "libero_pro", "robodojo", "robocasa", "robolab"}
    assert set(rp.SCENE_BACKENDS) == rp.PROGRAM_BACKENDS
    assert set(rp.VERDICT_KEYS) == rp.CONTROLLER_REPLAY_BACKENDS == rp.PROGRAM_BACKENDS - {"robodojo"}


def test_each_recorded_action_becomes_one_step_of_its_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    program = rp.build_program(_robotwin_run(tmp_path, monkeypatch))

    assert program["schema"] == rp.PROGRAM_SCHEMA and program["status"] == "replayable"
    assert [step["kind"] for step in program["steps"]] == ["controls", "hold", "controls", "skip"]
    assert "inter_arm_collision" in program["steps"][2]["note"]
    assert program["steps"][1]["physics_steps"] == 5
    assert program["steps"][0]["expect"] == POSE
    assert program["expect"] == {"official_success": True, "first_success_action": 2}
    assert (program["task"], program["seed"], program["source_commit"]) == ("adjust_bottle", 7, "c0ffee")


def test_receipts_are_matched_by_sequence_when_two_decisions_share_an_action_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    joint = _dense(1, 3, action_id="shared")
    refine = _dense(2, 2, action_id="shared")
    run = _robotwin_run(tmp_path, monkeypatch, [ENTRIES[0], joint, refine])
    later = {"left": POSE["right"], "right": POSE["left"]}
    (run / "native" / "receipts" / "action-0002.json").write_text(json.dumps(
        {"action_id": "shared", "action": "move_pose", "after": {arm: {"pose_xyz_xyzw": pose} for arm, pose in later.items()}}))
    (run / "native" / "receipts" / "action-0000.json").write_text(json.dumps({"action_id": "another", "after": {}}))
    steps = rp.build_program(run)["steps"]

    assert steps[1]["expect"] == POSE and steps[2]["expect"] == later
    assert steps[0]["expect"] == {}


def test_an_action_whose_steps_cannot_be_replayed_makes_the_program_unreplayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lost = _dense(4, 6, replayable=False, execution={"status": "executed"})
    program = rp.build_program(_robotwin_run(tmp_path, monkeypatch, [*ENTRIES, lost]))

    assert program["status"] == "unreplayable"
    assert program["steps"][-1]["kind"] == "unreplayable"
    assert "6 physics steps" in program["problems"][0]


def test_program_py_is_the_steps_as_code_and_loads_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _robotwin_run(tmp_path, monkeypatch)
    out = tmp_path / "exported"
    rp.write_program(run, out)
    code = (out / "program.py").read_text()

    assert "program.hold(1, steps=5)" in code and "program.skip(3)" in code and code.count("program.controls(") == 2
    loaded = rp.load_program(out)
    assert loaded["recorded"] is True
    assert [step["kind"] for step in loaded["steps"]] == ["controls", "hold", "controls", "skip"]
    assert loaded["steps"][0]["expect"] == POSE

    (out / "program.py").write_text(code.replace("program.hold(1, steps=5)", "program.hold(1, steps=9)"))
    edited = rp.load_program(out)
    assert edited["recorded"] is False
    assert edited["steps"][1]["physics_steps"] == 9
    assert all("expect" not in step for step in edited["steps"])


def test_program_py_runs_as_a_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "exported"
    rp.write_program(_robotwin_run(tmp_path, monkeypatch), out)
    shown = subprocess.run([sys.executable, str(out / "program.py"), "--help"], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}, check=False, cwd=tmp_path)

    assert shown.returncode == 0, shown.stderr
    assert "ora-test-robotwin-0001" in shown.stdout and "--check" in shown.stdout


def test_a_program_is_written_inside_an_unsealed_run_or_outside_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _robotwin_run(tmp_path, monkeypatch)
    with pytest.raises(rp.ReplayProgramError, match="outside the run"):
        rp.write_program(run, run / "native" / "elsewhere")
    assert rp.write_program(run)["status"] == "replayable"
    assert (run / "replay-program" / "program.json").is_file()
    (run / "seal.json").write_text("{}")
    with pytest.raises(rp.ReplayProgramError, match="sealed"):
        rp.write_program(run)


def test_the_seal_records_why_a_program_is_missing_and_skips_other_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _robotwin_run(tmp_path, monkeypatch)
    (run / "native" / "controller-replay" / "manifest.json").unlink()
    written = rp.write_program_at_seal(run)
    assert written["status"] == "unavailable"
    reason = json.loads((run / "replay-program" / "unavailable.json").read_text())
    assert reason["schema"] == rp.UNAVAILABLE_SCHEMA and "no controller replay" in reason["reason"]

    synthetic = _robotwin_run(tmp_path / "other", monkeypatch, kind="synthetic")
    assert rp.write_program_at_seal(synthetic)["status"] == "not_applicable"
    assert not (synthetic / "replay-program").exists()


def _robolab_run(tmp_path: Path, moves: tuple[int, ...] = (3, 0, 2), *, success_at: int | None = 2) -> Path:
    """A RoboLab run's evidence as ``ControlRecord`` writes it: ``moves`` native actions per action, 0 for a refusal."""
    run = tmp_path / "runs" / "ora-test-robolab-0001"
    native = run / "native"
    (native / "receipts").mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({"task": "BananaInBowlTask", "seed": 1, "source": {"commit": "c0ffee"},
                                               "runtime": {"kind": "robolab"}}))
    recorder = ControllerReplayRecorder(output_root=native, backend="robolab", environment_identity={},
                                        camera_panels=["agentview"])
    for sequence, ticks in enumerate(moves):
        action = {"action_id": f"a{sequence}", "action": "move_pose"}
        arrays = ({"native_actions": np.float32(np.full((ticks, 8), sequence)),
                   "joint_targets": np.float32(np.full((ticks, 8, 13), sequence))} if ticks else {})
        recorder.record(action_sequence=sequence, action=action, controller="robolab_joint_targets", arrays=arrays,
                        physics_steps=ticks * 8, replayable=bool(ticks))
        recorder.update(sequence, execution_status="reached" if ticks else "plan_rejected")
        (native / "receipts" / f"action-{sequence:04d}.json").write_text(json.dumps(
            {**action, "round_index": 0, "steps_used": ticks, "execution_status": "reached",
             "after": {"right": {"pose_xyz_xyzw": POSE["right"]}}}))
    (native / "official-verdict.json").write_text(json.dumps(
        {"official_success": success_at is not None, "first_success": None if success_at is None else {
            "action_sequence": success_at, "native_action": 7}}))
    return run


def test_a_robolab_program_replays_controls_and_skips_what_moved_nothing(tmp_path: Path) -> None:
    program = rp.build_program(_robolab_run(tmp_path))

    assert program["backend"] == "robolab" and program["status"] == "replayable"
    assert [step["kind"] for step in program["steps"]] == ["controls", "skip", "controls"]
    assert [step["physics_steps"] for step in program["steps"]] == [24, 0, 16]
    assert program["steps"][0]["expect"] == {"right": POSE["right"]}
    assert program["expect"] == {"official_success": True, "first_termination_action": 2}


@pytest.mark.parametrize("marker", [None, "partial"])
def test_a_robolab_run_without_a_whole_record_is_unavailable_never_a_program(
    tmp_path: Path, marker: str | None,
) -> None:
    run = _robolab_run(tmp_path)
    if marker is None:  # sealed before the recording existed: receipts only
        shutil.rmtree(run / "native" / "controller-replay")
    else:  # a recording that failed part way
        (run / "native" / "receipts" / "action-0003.json").write_text("{}")
    with pytest.raises(rp.ReplayProgramUnavailable):
        rp.build_program(run)
    written = rp.write_program_at_seal(run)

    assert written["status"] == "unavailable"
    assert sorted(path.name for path in (run / "replay-program").iterdir()) == ["unavailable.json"]


def test_an_unavailable_marker_for_a_sealed_run_lands_outside_it(tmp_path: Path) -> None:
    run = _robolab_run(tmp_path)
    shutil.rmtree(run / "native" / "controller-replay")
    (run / "seal.json").write_text("{}")
    with pytest.raises(rp.ReplayProgramUnavailable, match="never rebuilt from receipts") as refused:
        rp.prepare_check(run, tmp_path / "out")

    written = rp.write_unavailable(tmp_path / "exported", run.name, str(refused.value))

    assert json.loads((tmp_path / "exported" / "unavailable.json").read_text())["schema"] == rp.UNAVAILABLE_SCHEMA
    assert written["status"] == "unavailable" and not (run / "replay-program").exists()


class RobolabStub:
    """The session the environment worker holds: ``replay_tick`` records what it was fed, the TCP moves with it."""

    def __init__(self, *, succeed_on: int | None = None) -> None:
        self.bridge = self
        self.action_sequence, self.terminated, self.first_success = 0, False, None
        self.fed: list[tuple[float, float]] = []
        self.rounds: list[tuple[int, int | None]] = []
        self.succeed_on = succeed_on
        self.tracer = None

    def replay_tick(self, native: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
        assert native.shape == (8,) and targets.shape == (8, 13) and targets.dtype == np.float32
        self.fed.append((float(native[0]), float(targets[0, 0])))
        return {"official_result": self.action_sequence == self.succeed_on}

    def after_native_action(self, result: dict[str, Any]) -> None:
        if result["official_result"] and not self.terminated:
            self.terminated = True
            self.first_success = {"action_sequence": self.action_sequence}

    def take_ticks(self) -> list[Any]:
        return []

    def tcp_pose(self) -> list[float]:
        return list(POSE["right"])

    def restart_round(self, round_index: int, *, seed: int | None = None) -> None:
        self.rounds.append((round_index, seed))

    def scene_trace(self, directory: Path, run_id: str) -> Any:
        return self.tracer


def _robolab_measured(run: Path, session: RobolabStub, *, scene: dict[str, str] | None = None) -> dict[str, Any]:
    manifest_path = run / "native" / "controller-replay" / "manifest.json"
    return program_worker._run_robolab_program(
        session=session, manifest=load_controller_replay_manifest(manifest_path), manifest_path=manifest_path,
        run_root=run, program=rp.build_program(run), scene=scene)


def test_the_recorded_tensors_reach_the_bridge_in_order_and_the_verdict_latches_on_its_action(tmp_path: Path) -> None:
    run = _robolab_run(tmp_path)
    session = RobolabStub(succeed_on=2)

    measured = _robolab_measured(run, session)

    assert session.fed == [(0.0, 0.0)] * 3 + [(2.0, 2.0)] * 2
    assert measured["official_success"] is True and measured["first_termination_action"] == 2
    assert [row["kind"] for row in measured["steps"]] == ["controls", "skip", "controls"]
    assert all(row["arms"]["right"]["bit_identical"] for row in measured["steps"])
    comparison = rp._compare({**rp.build_program(run), "recorded": True}, measured)
    assert comparison["failures"] == [] and comparison["verdict"]["expected"] == comparison["verdict"]["replayed"]


def test_one_bit_off_a_tcp_or_a_verdict_lost_fails_the_robolab_check(tmp_path: Path) -> None:
    run = _robolab_run(tmp_path)
    session = RobolabStub()  # never succeeds
    off = list(POSE["right"])
    off[2] = float(np.nextafter(off[2], 1.0))
    session.tcp_pose = lambda: off

    comparison = rp._compare({**rp.build_program(run), "recorded": True}, _robolab_measured(run, session))

    assert comparison["failures"] == ["3 action end poses differ from their receipts",
                                      "the replay's verdict differs from the run's"]


def test_a_later_round_restarts_on_the_seed_its_reset_recorded(tmp_path: Path) -> None:
    run = _robolab_run(tmp_path, (2, 1))
    native = run / "native"
    manifest_path = native / "controller-replay" / "manifest.json"
    manifest = load_controller_replay_manifest(manifest_path)
    manifest["entries"][1]["round_index"] = 1
    (native / "round-resets").mkdir()
    (native / "round-resets" / "round-0001.json").write_text(json.dumps({"seed": 41}))
    session = RobolabStub()

    program_worker._run_robolab_program(session=session, manifest=manifest, manifest_path=manifest_path, run_root=run,
                                        program=rp.build_program(run), scene=None)

    assert session.rounds == [(1, 41)]


def test_a_robolab_tracer_sees_each_round_action_and_native_action_and_reports_in_the_result(tmp_path: Path) -> None:
    run = _robolab_run(tmp_path, (2, 0, 1))
    native = run / "native"
    manifest_path = native / "controller-replay" / "manifest.json"
    manifest = load_controller_replay_manifest(manifest_path)
    manifest["entries"][2]["round_index"] = 1
    (native / "round-resets").mkdir()
    (native / "round-resets" / "round-0001.json").write_text(json.dumps({"seed": 3}))
    calls: list[Any] = []

    class Tracer:
        def begin_round(self, _session: Any, round_index: int) -> None:
            calls.append(("round", round_index))

        def begin_action(self, entry: dict[str, Any]) -> None:
            calls.append(("action", entry["action_sequence"]))

        def after_tick(self) -> None:
            calls.append("tick")

        def end_action(self) -> None:
            calls.append("end")

        def finish(self) -> dict[str, Any]:
            return {"shape": [1, 1, 7]}

    session = RobolabStub()
    session.tracer = Tracer()
    measured = program_worker._run_robolab_program(
        session=session, manifest=manifest, manifest_path=manifest_path, run_root=run,
        program=rp.build_program(run), scene={"directory": str(tmp_path / "trace")})

    assert calls == [("round", 0), ("action", 0), "tick", "tick", "end", ("action", 1), "end",
                     ("round", 1), ("action", 2), "tick", "end"]
    assert measured["scene_trace"] == {"shape": [1, 1, 7]}


def test_a_step_that_disagrees_with_its_record_stops_the_replay(tmp_path: Path) -> None:
    run = _robolab_run(tmp_path)
    program = rp.build_program(run)
    program["steps"][0]["kind"] = "skip"
    manifest_path = run / "native" / "controller-replay" / "manifest.json"

    with pytest.raises(worker.ReplayWorkerError, match="replays as controls"):
        program_worker._run_robolab_program(session=RobolabStub(), manifest=load_controller_replay_manifest(manifest_path),
                                            manifest_path=manifest_path, run_root=run, program=program, scene=None)


def test_the_robolab_worker_session_keeps_the_plan_without_its_planner_on_the_caller_s_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: dict[str, Any] = {}
    calls: list[Any] = []

    class Session:
        def start(self) -> None:
            calls.append("start")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr("ora.environment.factory.build_robolab_worker_session",
                        lambda **kwargs: built.update(kwargs) or Session())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    plan = {"task": "T", "seed": 3, "policy": {"action_codec": "motion_action_v3"},
            "runtime": {"renderer_gpu": 1, "motion_planner": {}, "resource_binding": {}, "kind": "robolab"}}

    code = replay_in_environment_worker(plan, tmp_path / "job.json", tmp_path / "worker", tmp_path / "worker.log", 60.0)

    assert code == 0 and calls == ["start", "close"]
    assert built["runtime"] == {"renderer_gpu": 5, "kind": "robolab", "request_timeout_s": 60.0,
                                "replay_program_job": str(tmp_path / "job.json")}
    assert (built["task"], built["seed"], built["output_root"]) == ("T", 3, tmp_path / "worker" / "environment")


def test_a_robolab_worker_that_fails_closes_and_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class Session:
        def start(self) -> None:
            raise RuntimeError("no card")

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("ora.environment.factory.build_robolab_worker_session", lambda **_kwargs: Session())
    plan = {"task": "T", "seed": 3, "runtime": {"renderer_gpu": 1}}

    assert replay_in_environment_worker(plan, tmp_path / "j", tmp_path / "w", tmp_path / "log", 60.0) == 1
    assert closed == [True] and "no card" in (tmp_path / "log").read_text()


def _manifest(spans: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    return {"clock": {"stride": 8}, "spans": [
        {"round": round_index, "action_sequence": sequence, "first_step": first, "last_step": last}
        for round_index, sequence, first, last in spans]}


def test_a_robolab_trace_s_clock_is_checked_against_the_receipts_steps_round_by_round() -> None:
    receipts = {0: {"steps_used": 3}, 1: {"steps_used": 0}, 2: {"steps_used": 2}, 3: {"steps_used": 1}}
    good = [(0, 0, 0, 24), (0, 1, 24, 24), (0, 2, 24, 40), (1, 3, 0, 8)]

    assert receipt_clock_failures(receipts, _manifest(good)) == []
    bad = receipt_clock_failures(receipts, _manifest([(0, 0, 0, 24), (0, 1, 24, 24), (0, 2, 24, 48), (1, 3, 8, 16)]))
    assert bad == ["the scene trace's steps differ from the receipts' at actions [2, 3]"]


def test_a_robodojo_program_feeds_each_action_its_recorded_ticks(tmp_path: Path) -> None:
    run = tmp_path / "ora-test-robodojo-0001"
    native = run / "native"
    (native / "controller").mkdir(parents=True)
    (native / "receipts").mkdir()
    (run / "plan.json").write_text(json.dumps({"task": "push_T", "seed": 0, "runtime": {"kind": "robodojo"}}))
    for sequence, ticks, action in ((0, 3, "move_pose"), (1, 0, "move_pose"), (2, 2, "set_gripper")):
        (native / "controller" / f"action-{sequence:04d}.json").write_text(json.dumps({"samples": [{}] * ticks}))
        (native / "receipts" / f"action-{sequence:04d}.json").write_text(json.dumps(
            {"action_id": f"a{sequence}", "action": action, "round_index": 0, "execution_status": "reached"}))
    (native / "official-verdict.json").write_text(json.dumps(
        {"terminal": True, "official_success": False, "native_actions": 5}))
    program = rp.build_program(run)

    assert program["backend"] == "robodojo"
    assert [(step["kind"], step["ticks"]) for step in program["steps"]] == [("ticks", 3), ("skip", 0), ("ticks", 2)]
    assert program["expect"] == {"terminal": True, "official_success": False, "native_actions": 5}
    assert "program.ticks(2)" in rp.render_program(program)


def test_the_check_fails_on_any_pose_bit_or_a_different_verdict() -> None:
    program = {"backend": "robotwin", "recorded": True, "expect": {"official_success": True, "first_success_action": 2}}
    same = {"steps": [{"action_sequence": 0, "arms": {"left": {"bit_identical": True}}}],
            "official_success": True, "first_success_action": 2}
    assert rp._compare(program, same)["failures"] == []
    moved = {**same, "steps": [{"action_sequence": 0, "arms": {"left": {"bit_identical": False, "position_mm": 0.1}}}]}
    assert rp._compare(program, moved)["pose_mismatches"] == [{"action_sequence": 0, "arm": "left", "position_mm": 0.1}]
    late = {**same, "first_success_action": 3}
    assert rp._compare(program, late)["failures"] == ["the replay's verdict differs from the run's"]


def test_the_worker_runs_program_steps_and_measures_poses_and_the_success_latch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment.backends.robotwin import session

    physics: list[str] = []
    environment = SimpleNamespace(eval_success=False, _update_render=lambda: None)
    environment.scene = SimpleNamespace(step=lambda: physics.append("step"))
    environment.check_success = lambda: len(physics) >= 9

    def dense_action(sequence: Any, save_freq: int) -> None:
        for _row in sequence["right_arm"]["position"]:
            environment.scene.step()
            environment._update_render()

    environment.take_dense_action = dense_action
    wxyz = {arm: [*pose[:3], pose[6], *pose[3:6]] for arm, pose in POSE.items()}
    fake = SimpleNamespace(start=lambda: None, close=lambda: None, _require_environment=lambda: environment,
                           robotwin_root=tmp_path, _public_pose_wxyz=lambda _environment, arm: wxyz[arm])
    entries = [{**_fault(0, 3), "replayable": True, "execution": {"status": "reached", "settle_steps": 0}},
               ENTRIES[1], _fault(2, 4)]
    monkeypatch.setattr(session, "RoboTwinLiveSession", lambda **_kwargs: fake)
    monkeypatch.setattr(worker, "_robotwin_replay_needs_planner", lambda _manifest: True)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_artifact_arrays", lambda _path, entry: {
        "p": np.zeros((entry["physics_steps"], 6)), "v": np.zeros((entry["physics_steps"], 6))})
    steps = [{"action_sequence": 0, "kind": "controls", "expect": POSE},
             {"action_sequence": 1, "kind": "hold", "physics_steps": 5, "expect": {"left": POSE["right"]}},
             {"action_sequence": 2, "kind": "controls"}]
    measured = program_worker._run_robotwin_program(
        plan={"runtime": {"task_config": "t", "required_revision": "r", "workspace_bounds_m": []},
              "task": "t", "seed": 0, "budgets": {"max_sim_steps": 100}},
        manifest={"entries": entries}, manifest_path=tmp_path / "m.json", project_root=tmp_path,
        output_root=tmp_path, program={"steps": steps})

    assert len(physics) == 3 + 5 + 4
    assert measured["steps"][0]["arms"]["left"]["bit_identical"] is True
    assert measured["steps"][1]["arms"]["left"]["bit_identical"] is False
    assert measured["first_success_action"] == 2 and measured["official_success"] is True


def test_the_worker_traces_the_scene_of_the_same_steps_it_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment.backends.robotwin import scene_trace as robotwin_trace
    from ora.environment.backends.robotwin import session
    from ora.environment.scene_trace import Body

    class Scene:
        """A class of its own: the trace's step tap wraps ``type(scene).step``, as on SAPIEN's."""

        steps = 0

        def step(self) -> None:
            Scene.steps += 1

        def get_timestep(self) -> float:
            return 0.004

    environment = SimpleNamespace(eval_success=False, _update_render=lambda: None, scene=Scene(),
                                  check_success=lambda: False)

    def dense_action(sequence: Any, save_freq: int) -> None:
        for _row in sequence["right_arm"]["position"]:
            environment.scene.step()
            environment._update_render()

    environment.take_dense_action = dense_action
    body = Body("cup", "cup", "object", lambda: [float(Scene.steps), 0, 0, 0, 0, 0, 1], None)
    monkeypatch.setattr(robotwin_trace, "scene_bodies", lambda _environment: [body])
    fake = SimpleNamespace(start=lambda: None, close=lambda: None, _require_environment=lambda: environment,
                           robotwin_root=tmp_path, _public_pose_wxyz=lambda _environment, arm: [0.0] * 7)
    monkeypatch.setattr(session, "RoboTwinLiveSession", lambda **_kwargs: fake)
    monkeypatch.setattr(worker, "_robotwin_replay_needs_planner", lambda _manifest: True)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_artifact_arrays", lambda _path, entry: {
        "p": np.zeros((entry["physics_steps"], 6)), "v": np.zeros((entry["physics_steps"], 6))})
    entries = [{**_fault(0, 3), "replayable": True, "execution": {"status": "reached", "settle_steps": 0}},
               ENTRIES[1], _dense(2, 0, replayable=False, execution={"status": "planner_failed"})]
    tracer = robotwin_trace.RoboTwinSceneTrace(tmp_path / "trace", run_id="r")
    measured = program_worker._run_robotwin_program(
        plan={"runtime": {"task_config": "t", "required_revision": "r", "workspace_bounds_m": []},
              "task": "t", "seed": 0, "budgets": {"max_sim_steps": 100}},
        manifest={"entries": entries}, manifest_path=tmp_path / "m.json", project_root=tmp_path,
        output_root=tmp_path, scene_trace=tracer,
        program={"steps": [{"action_sequence": 0, "kind": "controls"},
                           {"action_sequence": 1, "kind": "hold", "physics_steps": 5},
                           {"action_sequence": 2, "kind": "skip"}]})
    manifest = tracer.finish()
    poses = np.frombuffer((tmp_path / "trace" / "poses.bin").read_bytes(), dtype="<f4").reshape(manifest["shape"])

    assert [row["kind"] for row in measured["steps"]] == ["controls", "hold", "skip"]
    assert [(s["action_sequence"], s["first_step"], s["last_step"]) for s in manifest["spans"]] == [
        (0, 0, 3), (1, 3, 8), (2, 8, 8)]
    assert poses[:, 0, 0].tolist() == [0.0, 3.0, 5.0, 8.0]      # a pose is read after the step the row names


def test_the_worker_traces_the_libero_steps_it_checks_and_no_round_s_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment.backends.libero_pro.scene_trace import LiberoSceneTrace
    from ora.environment.scene_trace import Body

    class Sim:
        """A class of its own: the tap wraps ``type(sim).step``. A round's reset steps this same simulator."""

        steps = 0
        model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        data = SimpleNamespace()

        def step(self) -> None:
            Sim.steps += 1

    sim = Sim()

    def command(name: str, _value: Any) -> None:
        if name == "harness_replay_command":
            for _ in range(25):                 # one control tick, as robosuite's env.step
                sim.step()

    def restart_round(_round_index: int) -> None:
        for _ in range(375):                    # the reset's 15 settling ticks
            sim.step()

    panda = SimpleNamespace(robot_model=SimpleNamespace(root_body="robot0_base"),
                            gripper=SimpleNamespace(important_sites={"grip_site": "gripper0_grip_site"}))
    attributes = {"sim": sim, "robots": [panda], "harness_replay_observation": {"frame": 0},
                  "harness_replay_step_result": {"terminated": False}}
    environment = SimpleNamespace(env=SimpleNamespace(set_env_attr=command))
    session = SimpleNamespace(start=lambda: None, close=lambda: None, restart_round=restart_round,
                              _require_environment=lambda: environment, _worker_attr=attributes.__getitem__,
                              _grip_site_pose=lambda: ([0.0] * 3, [0.0, 0.0, 0.0, 1.0]))
    monkeypatch.setattr(worker, "_libero_replay_session", lambda *_args: session)
    monkeypatch.setattr(worker, "_artifact_arrays", lambda _path, entry: {
        "commands": np.zeros((entry["physics_steps"], 7))})
    monkeypatch.setattr("ora.environment.mujoco_scene_trace.scene_bodies", lambda *_args, **_kwargs: [
        Body("cup", "cup", "object", lambda: [float(Sim.steps), 0, 0, 0, 0, 0, 1], None)])
    monkeypatch.setattr("ora.environment.mujoco_scene_trace.tcp_links", lambda *_args: {})

    def entry(sequence: int, ticks: int, round_index: int = 0, **fields: Any) -> dict[str, Any]:
        return {"action_sequence": sequence, "action_id": f"a{sequence}", "round_index": round_index,
                "controller": "libero_osc_pose", "physics_steps": ticks, "replayable": True, **fields}

    entries = [entry(0, 2), entry(1, 0, replayable=False), entry(2, 1, 1)]
    tracer = LiberoSceneTrace(tmp_path / "trace", run_id="r")
    measured = program_worker._run_libero_program(
        plan={}, manifest={"entries": entries}, manifest_path=tmp_path / "m.json", output_root=tmp_path,
        scene_trace=tracer, program={"steps": [{"action_sequence": 0, "kind": "controls"},
                                               {"action_sequence": 1, "kind": "skip"},
                                               {"action_sequence": 2, "kind": "controls"}]})
    manifest = tracer.finish()
    poses = np.frombuffer((tmp_path / "trace" / "poses.bin").read_bytes(), dtype="<f4").reshape(manifest["shape"])

    assert [row["kind"] for row in measured["steps"]] == ["controls", "skip", "controls"]
    assert [(s["round"], s["action_sequence"], s["first_step"], s["last_step"]) for s in manifest["spans"]] == [
        (0, 0, 0, 50), (0, 1, 50, 50), (1, 2, 0, 25)]       # the 375 settling steps count in no round
    assert poses[:, 0, 0].tolist() == [float(n) for n in (0, *range(10, 51, 10), 425, *range(435, 455, 10), 450)]


def test_a_move_s_refinement_nested_in_its_frame_ends_the_move_s_joint_phase(tmp_path: Path) -> None:
    """The rows of a LIBERO-Pro run: 1 is one OSC action, 5 a joint phase then a refinement that takes a new sequence."""
    root = tmp_path / "native" / "execution-capture"
    root.mkdir(parents=True)
    rows = [("reset", None, 0), ("action_start", 1, 100), ("action_start", 1, 100), ("action_end", 1, 175),
            ("action_end", None, 175), ("action_start", 5, 175), ("control", 5, 200), ("action_start", 6, 300),
            ("action_end", 6, 350), ("action_end", None, 350), ("terminal", None, 350)]
    (root / "timeline.jsonl").write_text("".join(json.dumps(
        {"phase": phase, "action_sequence": sequence, "round_index": 0, "sim_step": step}) + "\n"
        for phase, sequence, step in rows))

    assert rp._capture_spans(tmp_path) == {(0, 1): (100, 175), (0, 5): (175, 300), (0, 6): (300, 350)}


def _scene_outcome(*, failures: list[str] | None = None, last_step: int = 7, verdict: Any = None) -> Any:
    """``controller_replay_outcome`` as the worker ends it: a trace of one body and one action, and its result."""
    from ora.environment.scene_trace import Body, SceneTraceWriter

    def outcome(run_root: Path, output: Path, program: Any, timeout_s: float, *, scene_output: Path) -> dict[str, Any]:
        writer = SceneTraceWriter(scene_output, run_id=run_root.name, backend="robotwin", clock_unit="u")
        writer.begin_round(0, [Body("cup", "cup", "object", lambda: [0, 0, 0, 0, 0, 0, 1], None)], timestep_s=0.004)
        writer.begin_action({"action_sequence": 0, "action_id": "a0"})
        for _ in range(last_step):
            writer.after_step()
        writer.end_action()
        (output / "worker").mkdir()
        (output / "worker" / "replay-program-result.json").write_text(json.dumps({"scene_trace": writer.finish()}))
        return {"worker_returncode": 0, "steps": 1, "failures": list(failures or []), "verdict": verdict}
    return outcome


def _capture_timeline(run: Path, steps: tuple[int, int]) -> None:
    root = run / "native" / "execution-capture"
    root.mkdir()
    (root / "timeline.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"phase": "reset", "action_sequence": None, "round_index": 0, "sim_step": 0},
        {"phase": "action_start", "action_sequence": 0, "round_index": 0, "sim_step": steps[0]},
        {"phase": "control", "action_sequence": 0, "round_index": 0, "sim_step": steps[0] + 1},
        {"phase": "action_end", "action_sequence": 0, "round_index": 0, "sim_step": steps[1]}]))


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_a_passing_replay_publishes_its_scene_trace_beside_runs_and_leaves_the_run_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _robotwin_run(tmp_path, monkeypatch)
    _capture_timeline(run, (0, 7))
    before = _files(run)
    monkeypatch.setattr(rp, "controller_replay_outcome", _scene_outcome())
    result = rp.export_scene_trace(run)
    final = tmp_path / "scene-trace" / run.name

    assert result["passed"] is True and result["scene_trace"] == str(final) and result["failures"] == []
    assert (result["rows"], result["bodies"]) == (3, 1) and result["bytes"] > 0
    assert sorted(path.name for path in final.iterdir()) == [
        "manifest.json", "meshes", "poses.bin", "replay-program-check.json", "steps.jsonl"]
    assert json.loads((final / "replay-program-check.json").read_text())["passed"] is True
    assert [path.name for path in final.parent.iterdir()] == [run.name]      # no work directory left
    assert _files(run) == before


def test_a_replay_that_diverges_gets_no_scene_trace_and_keeps_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _robotwin_run(tmp_path, monkeypatch)
    monkeypatch.setattr(rp, "controller_replay_outcome",
                        _scene_outcome(failures=["1 action end poses differ from their receipts"]))
    result = rp.export_scene_trace(run)

    assert result["passed"] is False and result["scene_trace"] is None
    assert result["failures"] == ["1 action end poses differ from their receipts"]
    assert not (tmp_path / "scene-trace" / run.name).exists()
    assert json.loads(Path(result["report"]).read_text())["passed"] is False
    assert not list((tmp_path / "scene-trace").glob("*/trace"))


def test_a_replay_that_matches_every_bit_keeps_its_trace_when_only_the_verdict_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _robotwin_run(tmp_path, monkeypatch)
    _capture_timeline(run, (0, 7))
    verdict = {"expected": {"terminal": True}, "replayed": {"terminal": False}}
    monkeypatch.setattr(rp, "controller_replay_outcome", _scene_outcome(failures=[rp.VERDICT_DIFFERS], verdict=verdict))
    result = rp.export_scene_trace(run)
    final = tmp_path / "scene-trace" / run.name

    assert result["passed"] is True and result["scene_trace"] == str(final) and result["failures"] == []
    assert result["verdict_differs"] is True and result["verdict"] == verdict
    report = json.loads((final / "replay-program-check.json").read_text())
    assert report["passed"] is True and report["verdict_differs"] is True and report["verdict"] == verdict


def test_a_replay_that_diverges_and_reads_another_verdict_still_gets_no_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _robotwin_run(tmp_path, monkeypatch)
    failures = ["1 action end poses differ from their receipts", rp.VERDICT_DIFFERS]
    monkeypatch.setattr(rp, "controller_replay_outcome", _scene_outcome(failures=failures))
    result = rp.export_scene_trace(run)

    assert result["passed"] is False and result["scene_trace"] is None
    assert result["failures"] == failures[:1]
    assert json.loads(Path(result["report"]).read_text())["verdict_differs"] is True
    assert not (tmp_path / "scene-trace" / run.name).exists()


def test_a_trace_off_the_execution_capture_s_clock_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _robotwin_run(tmp_path, monkeypatch)
    _capture_timeline(run, (0, 9))
    monkeypatch.setattr(rp, "controller_replay_outcome", _scene_outcome(last_step=7))
    result = rp.export_scene_trace(run)

    assert result["passed"] is False and "at actions [0]" in result["failures"][0]
    assert not (tmp_path / "scene-trace" / run.name).exists()


def test_a_scene_trace_is_refused_where_it_could_not_be_this_run_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _robotwin_run(tmp_path, monkeypatch)
    monkeypatch.setattr(rp, "controller_replay_outcome", _scene_outcome())

    with pytest.raises(rp.ReplayProgramError, match="outside the run"):
        rp.export_scene_trace(run, run / "native" / "traces")
    assert not (run / "native" / "traces").exists()

    (tmp_path / "scene-trace" / run.name).mkdir(parents=True)
    with pytest.raises(rp.ReplayProgramError, match="already holds"):
        rp.export_scene_trace(run)
    (tmp_path / "scene-trace" / run.name).rmdir()

    monkeypatch.setattr(rp, "load_program", lambda directory: {
        "status": "replayable", "steps": [], "backend": "yam", "recorded": True})
    with pytest.raises(rp.ReplayProgramError, match="this run is yam"):
        rp.export_scene_trace(run)
    monkeypatch.setattr(rp, "load_program", lambda directory: {
        "status": "replayable", "steps": [], "backend": "robotwin", "recorded": False})
    with pytest.raises(rp.ReplayProgramError, match="differs from its recording"):
        rp.export_scene_trace(run)
    assert list((tmp_path / "scene-trace").iterdir()) == []


def test_a_round_s_reset_does_not_end_the_last_action_of_the_round_before(tmp_path: Path) -> None:
    root = tmp_path / "native" / "execution-capture"
    root.mkdir(parents=True)
    (root / "timeline.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"phase": "action_start", "action_sequence": 0, "round_index": 0, "sim_step": 0},
        {"phase": "action_end", "action_sequence": 0, "round_index": 0, "sim_step": 20},
        {"phase": "action_start", "action_sequence": 1, "round_index": 1, "sim_step": 0},
        {"phase": "action_end", "action_sequence": 1, "round_index": 1, "sim_step": 10}]))

    assert rp._capture_spans(tmp_path) == {(0, 0): (0, 20), (1, 1): (0, 10)}


def _robodojo_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "ora-test-robodojo-0001"
    (run / "native" / "controller").mkdir(parents=True)
    (run / "native" / "receipts").mkdir()
    (run / "plan.json").write_text(json.dumps({"task": "stack_blocks", "seed": 0, "runtime": {"kind": "robodojo"}}))
    (run / "native" / "controller" / "action-0000.json").write_text(json.dumps({"samples": [{}]}))
    (run / "native" / "receipts" / "action-0000.json").write_text(json.dumps({"action_id": "a0", "action": "move_pose"}))
    (run / "native" / "official-verdict.json").write_text(json.dumps(
        {"terminal": True, "official_success": True, "native_actions": 1}))
    return run


def _robodojo_replay(*, last_step: int = 10, failures: list[str] | None = None, rounds: list[dict] | None = None) -> Any:
    """What ``ora.app.replay`` passes: the control replay's report, worker report and trace."""
    from ora.environment.scene_trace import Body, SceneTraceWriter

    def replay(run_root: Path, motion: Path, trace: Path) -> None:
        writer = SceneTraceWriter(trace, run_id=run_root.name, backend="robodojo", clock_unit="u", stride=10)
        writer.begin_round(0, [Body("cup", "cup", "object", lambda: [0, 0, 0, 0, 0, 0, 1], None)], timestep_s=0.004)
        writer.begin_action({"action_sequence": 0, "action_id": "a0"})
        for _ in range(last_step):
            writer.after_step()
        writer.end_action()
        (motion / "pass-a").mkdir(parents=True)
        (motion / "robodojo-motion-replay-report.json").write_text(json.dumps({"failures": failures or []}))
        worker_rounds = rounds if rounds is not None else [
            {"terminal_after_round": True, "official_terminal": {"official_success": True, "native_actions": 1}}]
        (motion / "pass-a" / "replay-worker.json").write_text(json.dumps(
            {"replay": {"rounds": worker_rounds}, "scene_trace": writer.finish()}))
    return replay


def test_a_robodojo_replay_that_passes_its_check_publishes_the_trace_its_worker_wrote(tmp_path: Path) -> None:
    run = _robodojo_run(tmp_path)
    _capture_timeline(run, (0, 10))
    before = _files(run)
    calls: list[tuple[Path, Path, Path]] = []
    replay = _robodojo_replay()

    def recorded(*args: Path) -> None:
        calls.append(args)
        replay(*args)

    result = rp.export_scene_trace(run, robodojo_replay=recorded)
    final = tmp_path / "scene-trace" / run.name

    assert result["passed"] is True and result["scene_trace"] == str(final) and result["rows"] == 2
    assert calls[0][0] == run and calls[0][1].name == "motion-replay"
    check = json.loads((final / "replay-program-check.json").read_text())
    assert check["passed"] is True and check["verdict"]["replayed"]["native_actions"] == 1
    assert [path.name for path in final.parent.iterdir()] == [run.name] and _files(run) == before


def test_a_robodojo_replay_that_diverges_or_leaves_the_capture_s_clock_gets_no_trace(tmp_path: Path) -> None:
    run = _robodojo_run(tmp_path)
    _capture_timeline(run, (0, 20))

    diverged = rp.export_scene_trace(run, robodojo_replay=_robodojo_replay(
        last_step=20, failures=["pass-a joint_rad diverges"]))
    assert diverged["passed"] is False and diverged["failures"][0] == "control replay: pass-a joint_rad diverges"
    off_clock = rp.export_scene_trace(run, robodojo_replay=_robodojo_replay(last_step=10))
    assert off_clock["passed"] is False and "at actions [0]" in off_clock["failures"][0]
    verdict = rp.export_scene_trace(run, robodojo_replay=_robodojo_replay(last_step=20, rounds=[]))
    assert "the RoboDojo control replay recorded no rounds" in verdict["failures"]
    assert not (tmp_path / "scene-trace" / run.name).exists()


def test_a_robodojo_trace_needs_a_slot_and_a_launch_refused_leaves_no_work_directory(tmp_path: Path) -> None:
    run = _robodojo_run(tmp_path)

    with pytest.raises(rp.ReplayProgramError, match="--resource-profile"):
        rp.export_scene_trace(run)

    def refused(*_args: Path) -> None:
        raise RuntimeError("replay launches only from a clean committed Git checkout")

    with pytest.raises(RuntimeError, match="clean committed"):
        rp.export_scene_trace(run, robodojo_replay=refused)
    assert list((tmp_path / "scene-trace").iterdir()) == []


def test_a_robodojo_worker_that_wrote_no_trace_publishes_nothing(tmp_path: Path) -> None:
    run = _robodojo_run(tmp_path)
    replay = _robodojo_replay()

    def without_trace(run_root: Path, motion: Path, trace: Path) -> None:
        replay(run_root, motion, trace)
        worker_report = motion / "pass-a" / "replay-worker.json"
        worker_report.write_text(json.dumps({"replay": json.loads(worker_report.read_text())["replay"]}))

    result = rp.export_scene_trace(run, robodojo_replay=without_trace)

    assert result["passed"] is False and result["failures"] == ["the worker wrote no scene trace"]


def test_a_program_step_that_disagrees_with_its_record_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment.backends.robotwin import session

    environment = SimpleNamespace(eval_success=False, _update_render=lambda: None,
                                  scene=SimpleNamespace(step=lambda: None), check_success=lambda: False)
    fake = SimpleNamespace(start=lambda: None, close=lambda: None, _require_environment=lambda: environment,
                           robotwin_root=tmp_path)
    monkeypatch.setattr(session, "RoboTwinLiveSession", lambda **_kwargs: fake)
    monkeypatch.setattr(worker, "_robotwin_replay_needs_planner", lambda _manifest: True)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_artifact_arrays", lambda *_: {})
    with pytest.raises(worker.ReplayWorkerError, match="is controls but its record replays as hold"):
        program_worker._run_robotwin_program(
            plan={"runtime": {"task_config": "t", "required_revision": "r", "workspace_bounds_m": []},
                  "task": "t", "seed": 0, "budgets": {"max_sim_steps": 100}},
            manifest={"entries": [ENTRIES[1]]}, manifest_path=tmp_path / "m.json", project_root=tmp_path,
            output_root=tmp_path, program={"steps": [{"action_sequence": 1, "kind": "controls"}]})


def test_a_script_s_skill_sequence_stages_are_grouped_under_the_script(tmp_path: Path) -> None:
    """A code-mode script's grasp_execute proposes action-7; the controller records its stages as action-7:NN-<stage>."""
    from ora.environment.replay_program import _script_groups

    rows = [
        {"type": "tool_call_proposed", "tool_call_id": "call-1",
         "payload": {"name": "bash", "arguments": {"command": "python -m policy"}}},
        {"type": "action_proposed", "tool_call_id": "call-1", "payload": {"action_id": "action-7"}},
    ]
    (tmp_path / "agent-events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    steps = [{"action_id": "action-7:00-open"}, {"action_id": "action-7:01-pregrasp"}, {"action_id": "action-9"}]
    assert _script_groups(tmp_path, steps) == {"call-1": {"code": "python -m policy"}}
    assert [step.get("script") for step in steps] == ["call-1", "call-1", None]
