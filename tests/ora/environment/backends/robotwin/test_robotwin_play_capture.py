"""A RoboTwin scene of ORA's play scene sets: its operator camera, 30 fps capture and post-action settle."""
from __future__ import annotations

import json
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robotwin import execution_capture as robotwin_capture
from ora.environment.backends.robotwin.common import PLAY_SIGNAL_POLICY
from ora.environment.backends.robotwin.play_settle import (
    POST_ACTION_SETTLE_SCHEMA, rotation_deg, settle_after_action, window_motion,
)
from ora.environment.backends.robotwin.session import RoboTwinLiveSession
from ora.environment.benchmarks.robotwin_play_scenes import PLAY_EXECUTION_CAPTURE, PLAY_POST_ACTION_SETTLE
from ora.environment.execution_capture import resolve_execution_capture

TABLE_TOP = 0.74


class Scene:
    """A SAPIEN scene stand-in: a 4 ms step that moves the world, and a render sync."""

    def __init__(self, world):
        self.world = world
        self.steps = 0
        self.renders = 0

    def step(self):
        self.steps += 1
        self.world.advance()

    def get_timestep(self):
        return 0.004

    def update_render(self):
        self.renders += 1


class World:
    """Objects that keep moving for ``moving_steps`` physics steps after each kick, then rest."""

    def __init__(self, *, moving_steps=0, speed=0.002, off_table=()):
        self.positions = {"cup": np.array([0.2, -0.1, TABLE_TOP]), "bowl": np.array([0.0, -0.1, TABLE_TOP])}
        for name in off_table:
            self.positions[name] = np.array([3.0, 0.0, 0.05])
        self.moving_steps, self.speed, self.left = moving_steps, speed, 0
        self.rolling = set(off_table)

    def kick(self):
        self.left = self.moving_steps

    def advance(self):
        if self.left > 0:
            self.positions["cup"] = self.positions["cup"] + np.array([self.speed, 0.0, 0.0])
            self.left -= 1
        for name in self.rolling:  # rolls on the floor forever
            self.positions[name] = self.positions[name] + np.array([0.01, 0.0, 0.0])


class PlayTask:
    """The worker's play scene task object (play_scenes.base.PlaySceneTask), as the session sees it."""

    operator_camera_name = "operator_camera"
    capture_cameras = ("head_camera", "front_camera", "operator_camera")
    table_top_z = TABLE_TOP
    take_action_cnt, step_lim, eval_success = 0, 1000, False

    def __init__(self, world):
        self.world = world
        self.scene = Scene(world)
        self.rendered: list[tuple[str, ...]] = []
        self.cameras = SimpleNamespace(update_picture=lambda: pytest.fail("a play scene renders only its cameras"))

    def object_poses(self):
        return {name: (position.copy(), np.array([1.0, 0.0, 0.0, 0.0])) for name, position in self.world.positions.items()}

    def capture_images(self, names):
        self.rendered.append(tuple(names))
        shapes = {"operator_camera": (18, 32, 3)}
        return {name: np.full(shapes.get(name, (6, 8, 3)), self.scene.steps % 255, dtype=np.uint8) for name in names}

    def operator_rgb(self):
        return np.full((18, 32, 3), 90, dtype=np.uint8)


def _session(tmp_path, task, *, play=True, fps=30):
    session = RoboTwinLiveSession.__new__(RoboTwinLiveSession)
    session.output_root = tmp_path / "native"
    session.output_root.mkdir(parents=True)
    session.task, session.source_manifest = "table_setting_sequence", {"manifest_sha256": "0" * 64}
    session.execution_capture = resolve_execution_capture({"enabled": True, "fps": fps})
    session.play_scene_set = "tabletop-v1" if play else None
    session.environment = task
    session.round_index, session.action_sequence = 0, 0
    session.environment_signal_policy = PLAY_SIGNAL_POLICY if play else "lifecycle_only"
    session.max_sim_steps, session.abort_reason = None, None
    session._execution_recorder = session._execution_step_tap = None
    session._execution_capture_error = None
    session._require_environment = lambda: task
    return session


def _execute(session, task, action_id, *, steps=25):
    def act(payload):
        task.world.kick()
        for _ in range(steps):
            task.scene.step()
        session.action_sequence += 1
        return {"receipt": {"status": "executed", "action_id": payload["action_id"]}}

    session._execute_without_capture = act
    return session.execute({"action_id": action_id, "action": "move_pose"})


def _rows(tmp_path):
    path = tmp_path / "native" / "execution-capture" / "timeline.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _settles(tmp_path):
    path = tmp_path / "native" / "post-action-settle.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def test_a_play_scene_records_its_operator_camera_at_30_fps_and_settles_inside_the_action(tmp_path):
    assert PLAY_EXECUTION_CAPTURE == {"enabled": True, "fps": 30}
    world = World(moving_steps=60)
    task = PlayTask(world)
    session = _session(tmp_path, task)
    session._start_execution_capture()
    try:
        receipt = _execute(session, task, "action-1")
        assert receipt["receipt"]["status"] == "executed"
        _execute(session, task, "action-2:03-close")
    finally:
        session._close_execution_capture()
    rows = _rows(tmp_path)
    assert all(set(row["images"]) == {"head_camera", "front_camera", "operator_camera"} for row in rows)
    assert all(row["images"]["operator_camera"]["shape"] == [18, 32, 3] for row in rows)
    assert all(names == task.capture_cameras for names in task.rendered)
    settles = _settles(tmp_path)
    assert [record["action_id"] for record in settles] == ["action-1", "action-2"]
    assert settles[1]["dispatched_action_id"] == "action-2:03-close"
    first = settles[0]
    assert first["schema"] == POST_ACTION_SETTLE_SCHEMA and first["ended"] == "at_rest" and first["moving_at_start"]
    # The cup moved 25 steps inside the action and 35 more in the settle: windows of ten steps, the fourth
    # still moving for five; the settle stops on the first window in which nothing moved more than 1 mm.
    assert first["physics_step_start"] == 25 and first["physics_steps"] == 50 and first["windows"] == 5
    assert first["simulated_s"] == pytest.approx(0.2) and first["action_sequence"] == 0
    # Frames inside the settle carry the action's id and come before its action_end boundary.
    action_one = [row for row in rows if row["action_id"] == "action-1"]
    assert [row["phase"] for row in action_one][-1] == "action_end"
    settle_frames = [row for row in action_one if row["phase"] == "control" and row["sim_step"] > 25]
    assert settle_frames and all(row["sim_step"] <= 75 for row in settle_frames)
    # 30 fps on the 4 ms grid: within an action, a control frame every ninth physics step.
    controls = [row["sim_step"] for row in action_one if row["phase"] == "control"]
    assert controls[0] == 9 and {b - a for a, b in zip(controls, controls[1:])} == {9}
    manifest = json.loads((tmp_path / "native" / "execution-capture" / "manifest.json").read_text())
    assert manifest["camera_panels"] == ["head_camera", "front_camera", "operator_camera"]
    assert manifest["config"]["fps"] == 30 and manifest["dropped_by_reason"] == {}


def test_a_benchmark_scene_keeps_its_two_cameras_and_never_settles(tmp_path):
    world = World(moving_steps=60)
    task = PlayTask(world)
    manager_renders = []
    task.cameras = SimpleNamespace(
        update_wrist_camera=lambda *poses: None,
        update_picture=lambda: manager_renders.append("all"),
        get_rgb=lambda: {name: {"rgb": np.zeros((6, 8, 3), dtype=np.uint8)}
                         for name in ("head_camera", "front_camera", "left_camera", "right_camera")})
    task.robot = SimpleNamespace(left_camera=SimpleNamespace(get_pose=lambda: None),
                                 right_camera=SimpleNamespace(get_pose=lambda: None))
    session = _session(tmp_path, task, play=False, fps=10)
    session._start_execution_capture()
    try:
        _execute(session, task, "action-1")
    finally:
        session._close_execution_capture()
    assert task.scene.steps == 25 and not _settles(tmp_path) and not task.rendered and manager_renders
    assert all(set(row["images"]) == {"head_camera", "front_camera"} for row in _rows(tmp_path))


def test_a_settle_stops_at_its_cap_and_ignores_what_left_the_table(tmp_path):
    world = World(moving_steps=10_000)
    world.kick()
    task = PlayTask(world)
    session = _session(tmp_path, task)
    record = settle_after_action(session, {"action_id": "action-9", "action": "move_pose"})
    assert record["ended"] == "step_limit" and record["physics_steps"] == PLAY_POST_ACTION_SETTLE["max_physics_steps"]
    assert record["windows"] == 50 and record["simulated_s"] == pytest.approx(2.0)
    rolling = PlayTask(World(moving_steps=0, off_table=("plate",)))
    settled = settle_after_action(_session(tmp_path / "b", rolling), {"action_id": "action-10", "action": "set_gripper"})
    assert settled["ended"] == "at_rest" and settled["physics_steps"] == 10 and settled["off_table"] == ["plate"]
    assert not settled["moving_at_start"]


def test_no_settle_for_a_benchmark_a_terminal_scene_or_a_scene_without_poses(tmp_path):
    task = PlayTask(World())
    assert settle_after_action(_session(tmp_path / "a", task, play=False), {"action_id": "a"}) is None
    terminal = _session(tmp_path / "b", task)
    terminal.abort_reason = "operator_quit"
    assert terminal.is_terminal
    assert settle_after_action(terminal, {"action_id": "a"}) is None
    bare = SimpleNamespace(scene=task.scene, table_top_z=TABLE_TOP)
    session = _session(tmp_path / "c", task)
    session._require_environment = lambda: bare
    assert settle_after_action(session, {"action_id": "a"}) is None
    assert task.scene.steps == 0


def test_window_motion_reports_the_largest_move_and_turn():
    still = {"cup": (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))}
    turned = {"cup": (np.array([0.0, 0.0005, 0.0]), np.array([np.cos(np.radians(1)), 0.0, 0.0, np.sin(np.radians(1))]))}
    motion = window_motion(still, turned)
    assert motion["max_displacement_m"] == pytest.approx(0.0005) and motion["max_rotation_deg"] == pytest.approx(2.0)
    assert motion["object"] == "cup" and rotation_deg([1, 0, 0, 0], [1, 0, 0, 0]) == 0.0
    assert window_motion(still, turned, ignore={"cup"})["object"] is None


def test_the_operator_view_of_a_play_scene_is_its_operator_camera():
    session = RoboTwinLiveSession.__new__(RoboTwinLiveSession)
    session.environment = PlayTask(World())
    session.environment.cameras = SimpleNamespace(get_observer_rgb=lambda: pytest.fail("observer for a play scene"))
    session.play_scene_set = "tabletop-v1"
    view = session.operator_view()
    assert view["camera"] == "operator_camera" and (view["width"], view["height"]) == (32, 18)
    session.play_scene_set = None
    session.environment.cameras = SimpleNamespace(get_observer_rgb=lambda: np.zeros((24, 32, 3), dtype=np.uint8))
    assert session.operator_view()["camera"] == "observer_camera"


def test_the_capture_waits_for_its_writer_before_rendering(monkeypatch):
    backlog: queue.Queue = queue.Queue()
    for item in range(6):
        backlog.put(item)
    recorder = SimpleNamespace(_queue=backlog, error=None)

    def drain():
        time.sleep(0.05)
        while backlog.qsize() > 0:
            backlog.get()

    threading.Thread(target=drain).start()
    started = time.monotonic()
    robotwin_capture._await_capture_writer(recorder)
    assert backlog.qsize() <= robotwin_capture.PLAY_CAPTURE_WRITER_BACKLOG
    assert time.monotonic() - started >= 0.04
    stalled = SimpleNamespace(_queue=queue.Queue(), error="writer failed")
    for item in range(6):
        stalled._queue.put(item)
    started = time.monotonic()
    robotwin_capture._await_capture_writer(stalled)  # a failed writer is not waited for
    assert time.monotonic() - started < 0.5
