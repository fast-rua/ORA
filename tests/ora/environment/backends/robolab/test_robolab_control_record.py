"""RoboLab's record of what reached the simulator (``backends/robolab/control_record.py``), on stand-ins of the
articulation: the tap is neutral while recording, a replay's rows reach the write, and a failed recording stops the
record and never the run. Measured neutrality on the real simulator is in the design note."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robolab.control_record import CONTROLLER, FAILURE_FILE, ControlRecord, JointTargetTap
from ora.environment.controller_replay import load_controller_replay_arrays, load_controller_replay_manifest


class Robot:
    """An articulation's buffer and its write: ``written`` is what would have reached PhysX."""

    def __init__(self, joints: int = 3) -> None:
        self.data = SimpleNamespace(joint_pos_target=np.zeros((1, joints), dtype=np.float32))
        self.written: list[np.ndarray] = []

    def write_data_to_sim(self) -> None:
        self.written.append(self.data.joint_pos_target[0].copy())

    def set_joint_position_target(self, target: np.ndarray) -> None:
        self.data.joint_pos_target[:] = target


def test_recording_copies_the_buffer_and_leaves_what_is_written_alone() -> None:
    robot = Robot()
    tap = JointTargetTap(robot)
    before = np.random.get_state()[1].copy()

    for step in range(2):
        robot.data.joint_pos_target[:] = step + 0.1
        robot.write_data_to_sim()  # the attribute the scene's write calls
    rows = tap.take()

    assert rows.dtype == np.float32 and rows.tolist() == np.float32([[0.1] * 3, [1.1] * 3]).tolist()
    assert [row.tolist() for row in robot.written] == rows.tolist()
    assert (np.random.get_state()[1] == before).all()
    robot.data.joint_pos_target[:] = 9.0
    assert rows[0][0] != 9.0, "a row is a copy, not a view of the buffer"


def test_a_replay_writes_its_recorded_row_over_whatever_the_terms_left() -> None:
    robot = Robot()
    tap = JointTargetTap(robot)
    recorded = np.float32([[1, 2, 3], [4, 5, 6]])
    tap.feed = iter(recorded[:, None])

    for _ in range(2):
        robot.data.joint_pos_target[:] = -7.0  # what a recomputing IK term would have set
        robot.write_data_to_sim()

    assert [row.tolist() for row in robot.written] == recorded.tolist()
    assert tap.take().tolist() == recorded.tolist()


def _ticks(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.float32(np.arange(8) + tick), np.float32(np.full((8, 13), tick))) for tick in range(count)]


ACTION = {"action_id": "a0", "action": "move_pose", "arguments": {}}


def test_each_action_lands_in_the_controller_replay_layout(tmp_path: Path) -> None:
    record = ControlRecord(tmp_path, identity={"task": "T", "seed": 1})
    record.action(0, ACTION, _ticks(3), {"execution_status": "reached"})
    record.action(1, {**ACTION, "action_id": "a1"}, [], {"execution_status": "plan_rejected"})

    path = tmp_path / "controller-replay" / "manifest.json"
    manifest = load_controller_replay_manifest(path)
    moved, refused = manifest["entries"]
    assert manifest["backend"] == "robolab" and moved["controller"] == CONTROLLER
    assert moved["replayable"] and moved["physics_steps"] == 24 and moved["execution"]["status"] == "reached"
    assert not refused["replayable"] and refused["physics_steps"] == 0 and refused["artifact"] is None
    arrays = load_controller_replay_arrays(path, moved)
    assert arrays["native_actions"].shape == (3, 8) and arrays["joint_targets"].shape == (3, 8, 13)
    assert arrays["joint_targets"].dtype == np.float32


def test_a_failed_recording_ends_the_record_and_says_why_without_raising(tmp_path: Path) -> None:
    record = ControlRecord(tmp_path, identity={"task": "T", "seed": 1})
    native, targets = _ticks(1)[0]
    targets[0, 0] = np.nan  # the recorder refuses a non-finite array

    record.action(0, ACTION, [(native, targets)], {"execution_status": "reached"})
    record.action(1, ACTION, _ticks(1), {"execution_status": "reached"})
    record.begin_round(1)

    assert record.recorder is None
    assert "non-finite" in json.loads((tmp_path / "controller-replay" / FAILURE_FILE).read_text())["reason"]
    assert load_controller_replay_manifest(tmp_path / "controller-replay" / "manifest.json")["entries"] == []


def test_ticks_of_unequal_shape_end_the_record_too(tmp_path: Path) -> None:
    record = ControlRecord(tmp_path, identity={})
    uneven = [_ticks(1)[0], (np.float32(np.zeros(8)), np.float32(np.zeros((9, 13))))]

    record.action(0, ACTION, uneven, {"execution_status": "reached"})

    assert record.recorder is None and (tmp_path / "controller-replay" / FAILURE_FILE).is_file()


def test_a_record_that_cannot_open_does_not_stop_the_session(tmp_path: Path) -> None:
    ControlRecord(tmp_path, identity={"task": "T", "seed": 1})
    second = ControlRecord(tmp_path, identity={"task": "T", "seed": 1})  # the recorder never overwrites

    assert second.recorder is None
    second.action(0, ACTION, _ticks(1), {"execution_status": "reached"})


def test_the_session_records_each_action_after_its_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ora.environment.backends.robolab import session as module

    session = module.RobolabSession(config={"task": "BananaInBowlTask", "seed": 0}, output_root=tmp_path / "native")
    sent = _ticks(2)
    session.bridge = SimpleNamespace(take_ticks=lambda: sent)
    session.control_record = ControlRecord(session.output_root, identity={})
    monkeypatch.setattr(module, "validate_runtime_action", lambda action: {**action, "schema": "ora-motion-action.v3"})
    monkeypatch.setattr(module, "execute_motion_v3", lambda _session, _action: {"execution_status": "reached"})
    session.must_stop = lambda: False

    session.execute(ACTION)

    entry = load_controller_replay_manifest(session.output_root / "controller-replay" / "manifest.json")["entries"][0]
    assert entry["action_sequence"] == 0 and entry["physics_steps"] == 16 and session.action_sequence == 1


class StartBridge:
    """What ``RobolabSession.start`` reaches on a bridge."""

    def reset(self, _seed: int) -> dict:
        return {"frozen": False, "episode_steps": 0}

    def tcp_pose(self) -> list[float]:
        return [0.0] * 7

    def step(self, _target: list[float], _finger: float) -> dict:
        return {}

    def take_ticks(self) -> list:
        return []

    def contact_geometry(self) -> dict:
        return {}

    def finger_qpos(self) -> float:
        return 0.0

    def jaw_separation_m(self) -> float:
        return 0.08


@pytest.mark.parametrize("named", [True, False])
def test_start_runs_the_replay_job_its_configuration_names_after_the_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: bool,
) -> None:
    from ora.environment.backends.robolab import session as module

    ran: list[tuple] = []
    job = tmp_path / "job.json"
    job.write_text('{"schema": "ora.replay-program-job.v1"}')
    monkeypatch.setattr("ora.environment.backends.robolab.bridge.build_environment",
                        lambda _config, _root: (StartBridge(), {}))
    monkeypatch.setattr("ora.environment.replay_program_worker.run_program",
                        lambda value, *, session: ran.append((value, session.home_pose)))
    config = {"task": "BananaInBowlTask", "seed": 0, **({"replay_program_job": str(job)} if named else {})}

    module.RobolabSession(config=config, output_root=tmp_path / "native").start()

    assert ran == ([({"schema": "ora.replay-program-job.v1"}, [0.0] * 7)] if named else [])
