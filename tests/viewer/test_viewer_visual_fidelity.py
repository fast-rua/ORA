"""Original visual evidence is the default; frame sampling never skips physics."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from viewer import trajectory as video


@pytest.mark.parametrize("backend", ["robotwin", "libero_pro", "robodojo"])
def test_review_uses_original_images_even_with_replay_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    run = tmp_path / "runs" / backend
    (run / "policy-state/steps").mkdir(parents=True)
    (run / "native/controller-replay").mkdir(parents=True)
    # Deliberately unusable: default review must never need replay setup.
    (run / "native/controller-replay/manifest.json").write_text("{}")
    (run / "plan.json").write_text(json.dumps({"runtime": {"kind": backend}}))
    Image.new("RGB", (32, 32), "red").save(run / "original.png")
    (run / "policy-state/steps/step-0000.json").write_text(json.dumps({
        "step": 0, "observation": {"id": "original-0"},
        "cameras": {"agentview": {"files": {
            "rgb_lossless_png": {"path": "original.png"},
        }}},
    }))
    (run / "policy-state/manifest.json").write_text(json.dumps({"steps": [
        {"step": 0, "path": "policy-state/steps/step-0000.json"},
    ]}))

    def forbidden(*_args, **_kwargs):
        pytest.fail("default review attempted simulator reconstruction")

    monkeypatch.setattr(video, "_controller_replay_source", forbidden)
    trajectory = video.compile_trajectory(run)
    assert trajectory["video_source"]["kind"] == "original_observation_sequence"
    assert video.pending_replay_source(run) is None
    output, manifest = video.compile_trajectory_video(run, allow_replay_compile=False)
    assert output.is_file()
    assert manifest["visual_fidelity"] == "original_frames_only"
    assert manifest["video_source_kind"] == "original_observation_sequence"
    assert manifest["controller_replay"] is False
    assert manifest["frame_count"] == 1
    assert video.compile_trajectory_video(run)[0] == output

    # A stale render cannot silently accept an image changed after projection.
    Image.new("RGB", (32, 32), "blue").save(run / "original.png")
    with pytest.raises(RuntimeError, match="hash drifted"):
        video._render_trajectory_frame(run, trajectory, 0)
    assert video.trajectory_video_status(run)["cached"] is False


def test_robotwin_frame_sampling_executes_every_target_and_settle_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment import replay_worker as worker
    from ora.environment.backends.robotwin import session

    applied = []
    physics = []
    captures = []
    environment = SimpleNamespace()
    environment.scene = SimpleNamespace(step=lambda: physics.append("step"))
    environment._update_render = lambda: None

    def dense_action(targets, save_freq):
        assert save_freq == 1
        for target in targets:
            applied.append(target)
            environment.scene.step()
            environment._update_render()

    environment.take_dense_action = dense_action
    fake_session = SimpleNamespace(
        start=lambda: None, close=lambda: None,
        _require_environment=lambda: environment,
    )

    class Writer:
        sequence = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def append(self, _images, **_kwargs):
            captures.append(len(physics))
            self.sequence += 1

    monkeypatch.setattr(session, "RoboTwinLiveSession", lambda **_kwargs: fake_session)
    monkeypatch.setattr(worker, "_FrameWriter", Writer)
    monkeypatch.setattr(worker, "_robotwin_replay_needs_planner", lambda _: True)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_artifact_arrays", lambda *_: {})
    monkeypatch.setattr(worker, "_recorded_control_step_count", lambda *_: 7)
    monkeypatch.setattr(worker, "_robotwin_control_sequence", lambda *_: list(range(7)))
    monkeypatch.setattr(worker, "_robotwin_images", lambda _: {})
    worker._replay_robotwin(
        plan={"runtime": {"task_config": "test", "required_revision": "source",
                          "workspace_bounds_m": []},
              "task": "test", "seed": 0, "budgets": {"max_sim_steps": 100}},
        manifest={"camera_panels": [], "entries": [{
            "replayable": True, "controller": "robotwin_dense_joint_targets",
            "execution": {"settle_steps": 2},
        }]},
        manifest_path=tmp_path / "manifest.json", project_root=tmp_path,
        output_root=tmp_path,
        sampling={"mode": "action_keyframes_v1", "max_frames_per_action": 4,
                  "hold_steps_per_sample": 8},
    )
    assert applied == list(range(7))
    assert len(physics) == 9
    assert captures == [0, 1, 4, 6, 9]


@pytest.mark.parametrize(("total", "expected"), [(None, 2), (7, 0), (9, 2), (12, 2), (5, 0)])
def test_robotwin_settle_steps_are_only_what_the_recorded_total_has_beyond_the_sequence(
    monkeypatch: pytest.MonkeyPatch, total, expected,
) -> None:
    """A global-precontact-then-local action records its local phase in the sequence and as settle steps."""
    from ora.environment import replay_worker as worker

    monkeypatch.setattr(worker, "_recorded_control_step_count", lambda *_: 7)
    entry = {"execution": {"settle_steps": 2}}
    if total is not None:
        entry["physics_steps"] = total
    assert worker._extra_settle_steps(entry, {}) == expected
    assert worker._extra_settle_steps({"physics_steps": 9, "execution": {"settle_steps": 0}}, {}) == 0


def _ramp(n: int = 200, pad: int = 100):
    import numpy as np

    return np.concatenate([np.linspace(1.0, 0.0, n), np.zeros(pad)])


@pytest.mark.parametrize(("values", "recorded", "expected"), [
    ("full", 0.0, -1.0 / 200),        # a planned close and its constant tail
    ("cut", 0.0, -1.0 / 200),         # the same ramp ended early by the action
    ("still", 0.0, 0.0),              # nothing moved: nothing to limit
    ("full", -0.005, -0.005),         # a current trace keeps its recorded step
])
def test_legacy_gripper_step_is_recovered_from_the_planned_ramp(values, recorded, expected) -> None:
    import numpy as np
    from ora.environment import replay_worker as worker

    result = {"full": _ramp(), "cut": np.linspace(1.0, 0.0, 200)[:161], "still": np.ones(300)}[values]
    sequence = {"left_arm": None, "right_arm": None, "right_gripper": None,
                "left_gripper": {"num_step": len(result), "per_step": recorded, "result": result}}
    restored = worker._legacy_gripper_steps(sequence, planner_steps=lambda: 200)
    assert restored["left_gripper"]["per_step"] == pytest.approx(expected, abs=1e-15)
    assert restored["left_gripper"]["result"] is result
    assert sequence["left_gripper"]["per_step"] == recorded
    assert worker._legacy_gripper_steps(list(range(3))) == [0, 1, 2]


def test_legacy_gripper_path_that_is_not_a_ramp_is_refused() -> None:
    import numpy as np
    from ora.environment import replay_worker as worker

    sequence = {"left_gripper": {"num_step": 4, "per_step": 0.0, "result": np.asarray([1.0, 0.9, 0.5, 0.4])}}
    with pytest.raises(worker.ReplayWorkerError, match="not a planned ramp"):
        worker._legacy_gripper_steps(sequence, planner_steps=lambda: 200)


def test_robotwin_gripper_planner_steps_come_from_its_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    from ora.environment import replay_worker as worker

    class MplibPlanner:
        def plan_grippers(self, now_val, target_val):
            return {"num_step": 200, "per_step": (target_val - now_val) / 200}

    worker._robotwin_gripper_planner_steps.cache_clear()
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: SimpleNamespace(MplibPlanner=MplibPlanner))
    try:
        assert worker._robotwin_gripper_planner_steps() == 200
    finally:
        worker._robotwin_gripper_planner_steps.cache_clear()


def test_robotwin_replay_runs_a_local_phase_inside_the_sequence_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment import replay_worker as worker
    from ora.environment.backends.robotwin import session

    physics = []
    environment = SimpleNamespace()
    environment.scene = SimpleNamespace(step=lambda: physics.append("step"))
    environment._update_render = lambda: None

    def dense_action(targets, save_freq):
        for _target in targets:
            environment.scene.step()
            environment._update_render()

    environment.take_dense_action = dense_action
    fake_session = SimpleNamespace(start=lambda: None, close=lambda: None, _require_environment=lambda: environment)

    class Writer:
        sequence = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def append(self, _images, **_kwargs):
            self.sequence += 1

    monkeypatch.setattr(session, "RoboTwinLiveSession", lambda **_kwargs: fake_session)
    monkeypatch.setattr(worker, "_FrameWriter", Writer)
    monkeypatch.setattr(worker, "_robotwin_replay_needs_planner", lambda _: True)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_artifact_arrays", lambda *_: {})
    monkeypatch.setattr(worker, "_recorded_control_step_count", lambda *_: 7)
    monkeypatch.setattr(worker, "_robotwin_control_sequence", lambda *_: list(range(7)))
    monkeypatch.setattr(worker, "_robotwin_images", lambda _: {})
    worker._replay_robotwin(
        plan={"runtime": {"task_config": "test", "required_revision": "source", "workspace_bounds_m": []},
              "task": "test", "seed": 0, "budgets": {"max_sim_steps": 100}},
        manifest={"camera_panels": [], "entries": [{
            "replayable": True, "controller": "robotwin_global_precontact_then_local",
            "physics_steps": 7, "execution": {"settle_steps": 2},
        }]},
        manifest_path=tmp_path / "manifest.json", project_root=tmp_path, output_root=tmp_path,
        sampling={"mode": "action_keyframes_v1", "max_frames_per_action": 4, "hold_steps_per_sample": 8},
    )
    assert len(physics) == 7


def test_robotwin_wait_steps_are_replayed_and_other_skipped_entries_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wait ran its physics steps with no new target; skipping them shifts every later action."""
    from ora.environment import replay_worker as worker
    from ora.environment.backends.robotwin import session

    physics, phases = [], []
    environment = SimpleNamespace()
    environment.scene = SimpleNamespace(step=lambda: physics.append("step"))
    environment._update_render = lambda: None

    def dense_action(targets, save_freq):
        for _target in targets:
            environment.scene.step()
            environment._update_render()

    environment.take_dense_action = dense_action
    fake_session = SimpleNamespace(start=lambda: None, close=lambda: None, _require_environment=lambda: environment)

    class Writer:
        sequence = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def append(self, _images, **kwargs):
            phases.append(kwargs["phase"])
            self.sequence += 1

    monkeypatch.setattr(session, "RoboTwinLiveSession", lambda **_kwargs: fake_session)
    monkeypatch.setattr(worker, "_FrameWriter", Writer)
    monkeypatch.setattr(worker, "_robotwin_replay_needs_planner", lambda _: True)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_artifact_arrays", lambda *_: {})
    monkeypatch.setattr(worker, "_recorded_control_step_count", lambda entry, _arrays: entry["physics_steps"])
    monkeypatch.setattr(worker, "_robotwin_control_sequence", lambda entry, _arrays: list(range(entry["physics_steps"])))
    monkeypatch.setattr(worker, "_robotwin_images", lambda _: {})
    dense = {"replayable": True, "action": "move_pose", "controller": "robotwin_dense_joint_targets",
             "execution": {"settle_steps": 0}}
    worker._replay_robotwin(
        plan={"runtime": {"task_config": "test", "required_revision": "source", "workspace_bounds_m": []},
              "task": "test", "seed": 0, "budgets": {"max_sim_steps": 100}},
        manifest={"camera_panels": [], "entries": [
            {**dense, "physics_steps": 3},
            {"replayable": False, "action": "wait", "controller": "robotwin_hold_current_drive_targets",
             "physics_steps": 5, "execution": {"settle_steps": 0}},
            {"replayable": False, "action": "move_pose", "controller": "robotwin_dense_joint_targets",
             "physics_steps": 0, "execution": {"status": "planner_failed"}},
            {**dense, "physics_steps": 2},
        ]},
        manifest_path=tmp_path / "manifest.json", project_root=tmp_path, output_root=tmp_path,
        sampling={"mode": "physics_stride_v1", "capture_stride": 1},
    )
    assert len(physics) == 3 + 5 + 2
    assert phases.count("hold") == 5 and phases.count("not_replayed") == 1


def test_robotwin_hold_steps_name_the_live_wait_controller() -> None:
    from ora.environment import replay_worker as worker
    from ora.environment.backends.robotwin.wait import WAIT_CONTROLLER

    assert worker.ROBOTWIN_HOLD_CONTROLLER == WAIT_CONTROLLER
    wait = {"action": "wait", "controller": WAIT_CONTROLLER}
    assert worker._robotwin_hold_steps({**wait, "physics_steps": 40}) == 40
    assert worker._robotwin_hold_steps({**wait, "physics_steps": 0}) == 0
    assert worker._robotwin_hold_steps({"action": "move_pose", "controller": "robotwin_dense_joint_targets",
                                        "physics_steps": 9}) == 0
    with pytest.raises(worker.ReplayWorkerError, match="no physics step count"):
        worker._robotwin_hold_steps({**wait, "physics_steps": None})


def _fault_entry(steps: int = 3, **changes):
    import numpy as np

    tracking = {"physics_step_attempts": steps, "sampled_steps": steps, "controller_write_attempts": steps,
                "mode": "global", "abort": {"code": "inter_arm_collision", "stage_index": steps - 1}}
    tracking.update(changes.pop("tracking", {}))
    entry = {
        "action_sequence": 54, "action": "move_pose", "replayable": False, "physics_steps": steps,
        "controller": "robotwin_dense_joint_targets",
        "execution": {"status": "execution_fault", "settle_steps": 0,
                      "fault": "RoboTwinControllerTrackingError: motion execution guard inter_arm_collision"},
        "metadata": {"controller_tracking": tracking, "control_sequence": {
            "left_arm": None, "left_gripper": None, "right_gripper": None,
            "right_arm": {"position": {"array": "right_arm__position"}, "velocity": {"array": "right_arm__velocity"}}}},
    }
    entry.update(changes)
    arrays = {"right_arm__position": np.zeros((steps, 6)), "right_arm__velocity": np.zeros((steps, 6))}
    return entry, arrays


def test_robotwin_fault_after_the_physics_step_replays_as_its_recorded_prefix() -> None:
    from ora.environment import replay_worker as worker

    assert worker._robotwin_fault_prefix_steps(*_fault_entry()) == 3
    entry, arrays = _fault_entry(0, tracking={"abort": {"code": "inter_arm_collision"}})
    assert worker._robotwin_fault_prefix_steps(entry, arrays) == 0
    entry, arrays = _fault_entry()
    assert worker._robotwin_fault_prefix_steps({**entry, "execution": {"status": "reached"}}, arrays) == 0


@pytest.mark.parametrize(("changes", "message"), [
    ({"tracking": {"abort": {"code": "before_step_fault"}}}, "without its physics step"),
    ({"tracking": {"abort": {"code": "backend_write_fault"}}}, "without its physics step"),
    ({"tracking": {"abort": {"code": "backend_step_fault"}}}, "without its physics step"),
    ({"tracking": {"physics_step_attempts": 4}}, "step counters disagree"),
    ({"physics_steps": 4}, "step counters disagree"),
    ({"tracking": {"controller_write_attempts": 4}}, "target writes do not match"),
    ({"tracking": {"abort": None}}, "no controller abort"),
    ({"controller": "robotwin_global_precontact_then_local", "tracking": {"mode": "local"}}, "global phase"),
    ({"controller": "robotwin_ee_action"}, "no dense controller sequence"),
])
def test_robotwin_fault_record_that_is_not_a_step_prefix_is_refused(changes, message) -> None:
    from ora.environment import replay_worker as worker

    entry, arrays = _fault_entry(**changes)
    with pytest.raises(worker.ReplayWorkerError, match=message):
        worker._robotwin_fault_prefix_steps(entry, arrays)


def test_robotwin_fault_rows_must_match_its_physics_steps() -> None:
    import numpy as np
    from ora.environment import replay_worker as worker

    entry, arrays = _fault_entry()
    arrays["right_arm__velocity"] = np.zeros((4, 6))
    with pytest.raises(worker.ReplayWorkerError, match="positions and velocities differ"):
        worker._robotwin_fault_prefix_steps(entry, arrays)
    arrays["right_arm__position"] = np.zeros((4, 6))
    with pytest.raises(worker.ReplayWorkerError, match="holds 4 rows for 3 physics steps"):
        worker._robotwin_fault_prefix_steps(entry, arrays)


def test_robotwin_replay_runs_a_fault_prefix_and_refuses_unreplayable_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment import replay_worker as worker
    from ora.environment.backends.robotwin import session

    physics = []
    environment = SimpleNamespace()
    environment.scene = SimpleNamespace(step=lambda: physics.append("step"))
    environment._update_render = lambda: None

    def dense_action(sequence, save_freq):
        for _row in sequence["right_arm"]["position"]:
            environment.scene.step()
            environment._update_render()

    environment.take_dense_action = dense_action
    fake_session = SimpleNamespace(start=lambda: None, close=lambda: None, _require_environment=lambda: environment)

    class Writer:
        sequence = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def append(self, _images, **_kwargs):
            self.sequence += 1

    fault, arrays = _fault_entry()
    monkeypatch.setattr(session, "RoboTwinLiveSession", lambda **_kwargs: fake_session)
    monkeypatch.setattr(worker, "_FrameWriter", Writer)
    monkeypatch.setattr(worker, "_robotwin_replay_needs_planner", lambda _: True)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_artifact_arrays", lambda *_: arrays)
    monkeypatch.setattr(worker, "_robotwin_images", lambda _: {})
    plan = {"runtime": {"task_config": "test", "required_revision": "source", "workspace_bounds_m": []},
            "task": "test", "seed": 0, "budgets": {"max_sim_steps": 100}}
    sampling = {"mode": "physics_stride_v1", "capture_stride": 1}
    worker._replay_robotwin(
        plan=plan, manifest={"camera_panels": [], "entries": [fault]},
        manifest_path=tmp_path / "manifest.json", project_root=tmp_path, output_root=tmp_path, sampling=sampling,
    )
    assert len(physics) == 3
    unknown = {"action_sequence": 7, "action": "move_pose", "replayable": False, "physics_steps": 5,
               "controller": "robotwin_dense_joint_targets", "execution": {"status": "executed"}}
    with pytest.raises(worker.ReplayWorkerError, match="ran 5 physics steps that its record cannot replay"):
        worker._replay_robotwin(
            plan=plan, manifest={"camera_panels": [], "entries": [unknown]},
            manifest_path=tmp_path / "manifest.json", project_root=tmp_path, output_root=tmp_path, sampling=sampling,
        )


@pytest.mark.parametrize("controller", ["libero_osc_pose", "libero_joint_position"])
def test_libero_legacy_replay_frames_use_each_actual_step_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, controller: str,
) -> None:
    """Direct worker steps must not render an unchanged outer reset cache."""
    import numpy as np
    from ora.environment import replay_worker as worker
    from ora.environment.backends.libero_pro import session as backend

    def observation(number):
        return {"agentview_image": np.full((2, 3, 3), number, dtype=np.uint8),
                "robot0_eye_in_hand_image": np.full((2, 3, 3), number + 10, dtype=np.uint8)}

    actual_steps = []
    frames = []
    environment = SimpleNamespace(current_raw_obs=[observation(0)])
    current = observation(0)

    def setter(name, value):
        nonlocal current
        if name == "harness_replay_command":
            actual_steps.append(value)
            current = observation(len(actual_steps))

    environment.env = SimpleNamespace(set_env_attr=setter)
    fake_session = SimpleNamespace(
        start=lambda: None, close=lambda: None,
        _require_environment=lambda: environment,
        _raw_observation=lambda: environment.current_raw_obs[0],
        _worker_attr=lambda name: current if name == "harness_replay_observation" else None,
    )

    class Writer:
        sequence = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def append(self, images, **kwargs):
            frames.append((kwargs["phase"], int(images["agentview"][0, 0, 0]),
                           int(images["wrist"][0, 0, 0])))
            self.sequence += 1

    arrays = {"commands": np.zeros((2, 8 if controller == "libero_joint_position" else 7))}
    if controller == "libero_joint_position":
        arrays["refresh_osc_command"] = np.zeros((1, 7))
    monkeypatch.setattr(backend, "LiberoProSession", lambda **_kwargs: fake_session)
    monkeypatch.setattr(worker, "_FrameWriter", Writer)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_libero_qualified_controller_binding", lambda *_: {})
    monkeypatch.setattr(worker, "_artifact_arrays", lambda *_: arrays)
    worker._replay_libero(
        plan={"runtime": {"suite": "libero_10", "task_id": 0,
                          "expected_task_language": "test",
                          "required_libero_pro_revision": "l", "required_data_revision": "d",
                          "workspace_bounds_m": []},
              "task": "test", "seed": 0, "budgets": {"max_sim_steps": 100}},
        manifest={"camera_panels": ["agentview", "wrist"], "entries": [{
            "replayable": True, "controller": controller,
        }]}, manifest_path=tmp_path / "manifest.json", output_root=tmp_path,
    )
    assert frames[:3] == [("reset", 0, 10), ("control", 1, 11), ("control", 2, 12)]
    if controller == "libero_joint_position":
        assert frames[3:] == [("refresh", 3, 13)]
    assert len(actual_steps) == len(frames) - 1
