"""An in-memory real cell for the real-cell kit's tests: no bus, camera, MuJoCo or lock file of the host.

``FakeAdapter`` implements ``ora.cell.adapter.CellAdapter`` over a dictionary of joint
positions. Its ``Held`` follows every command exactly, or (``blocked``) never moves, so a
test can watch the operator verbs drive, hold and release the arms. ``FakeKinematics`` is
the part of ``Kinematics`` the verbs use: it records the poses it is set to and answers
``inter_arm_distance`` and ``table_distance`` from settable values. ``write_config``
writes a valid ``ora.cell-config.v1`` file for those arms, and ``redirect_lock`` points
``ora.cell.lock`` at a file under the test's own directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pytest

ARMS = ("left", "right")
REST = {"left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
CAMERA_MOUNTS = {"overhead": "fixed_world", "wrist_left": "end_effector", "wrist_right": "end_effector"}
LIMITS = np.array([[-3.0, 3.0]] * 6)

CONFIG = """schema: ora.cell-config.v1
cell_id: fake-cell-01
robot:
  environment: fake
  model: fake_station
  buses: {left: fake_left, right: fake_right}
cameras:
  overhead:    {serial: "100", mount: fixed_world}
  wrist_left:  {serial: "200", mount: end_effector}
  wrist_right: {serial: "300", mount: end_effector}
calibration: CALIBRATION
safety:
  max_joint_speed_rad_s: 0.5
  operator_present_required: true
  foreign_controllers: {ignore: [my_teleop.py]}
perception:
  depth_completion: null
policy:
  provider_profile: test
"""


def write_config(directory: Path, text: str = CONFIG) -> Path:
    """A cell config whose calibration lives beside it; ``CALIBRATION`` in ``text`` names that file."""

    path = directory / "cell.yaml"
    path.write_text(text.replace("CALIBRATION", str(directory / "calibration.json")))
    return path


def redirect_lock(monkeypatch: pytest.MonkeyPatch, path: Path) -> Path:
    """Every ``CellLock`` and ``holder()`` that takes the default path uses ``path`` instead.

    The default is bound when ``ora.cell.lock`` is imported, so the functions' defaults are
    replaced, not the module constant alone.
    """

    from ora.cell import lock

    monkeypatch.setattr(lock, "LOCK_PATH", path)
    monkeypatch.setattr(lock.CellLock.__init__, "__defaults__", (path,))
    monkeypatch.setattr(lock.holder, "__defaults__", (path,))
    return path


class FakeKinematics:
    def __init__(self, inter_arm_m: float = 0.5, table_m: float | None = 0.2) -> None:
        self.mujoco = self.model = self.data = None
        self.world_offset = np.zeros(3)
        self.arm_limits = {arm: LIMITS.copy() for arm in ARMS}
        self.inter_arm_m, self.table_m = inter_arm_m, table_m
        self.states: list[dict[str, np.ndarray]] = []

    def set_state(self, arm_q: Mapping[str, Sequence[float]], finger_m: Mapping[str, float] | None = None) -> None:
        self.states.append({arm: np.asarray(q, float) for arm, q in arm_q.items()})

    def tcp(self, arm: str) -> np.ndarray:
        return np.eye(4)

    def chain(self, arm: str) -> list[list[float]]:
        return []

    def wrist_camera(self, arm: str) -> np.ndarray:
        return np.eye(4)

    def inter_arm_distance(self) -> float:
        return self.inter_arm_m

    def table_distance(self, arm: str) -> float | None:
        return self.table_m


class FakeReader:
    def __init__(self, adapter: "FakeAdapter") -> None:
        self.adapter, self.closed = adapter, False

    def read(self) -> dict[str, Any]:
        from ora.cell.adapter import ArmReading

        return {arm: ArmReading(q=q.copy(), gripper_raw=self.adapter.gripper_raw.get(arm))
                for arm, q in self.adapter.pose.items()}

    def close(self) -> None:
        self.closed = True


class FakeHeld:
    """Arms under the fake driver: commands move ``adapter.pose`` unless ``blocked``."""

    def __init__(self, adapter: "FakeAdapter", arms: Sequence[str], mode: str) -> None:
        self.adapter, self.arms = adapter, tuple(arms)
        self.modes = [mode]
        self.commands: list[dict[str, np.ndarray]] = []
        self.released = False

    def joints(self) -> dict[str, np.ndarray]:
        return {arm: self.adapter.pose[arm].copy() for arm in self.arms}

    def command(self, targets: Mapping[str, np.ndarray]) -> None:
        self.commands.append({arm: np.asarray(q, float).copy() for arm, q in targets.items()})
        if not self.adapter.blocked:
            for arm, q in targets.items():
                self.adapter.pose[arm] = np.asarray(q, float).copy()

    def alive(self) -> dict[str, bool]:
        return {arm: True for arm in self.arms}

    def alerts(self) -> list[dict[str, Any]]:
        return [dict(alert) for alert in self.adapter.alerts]

    def set_mode(self, mode: str) -> None:
        self.modes.append(mode)

    def release(self) -> None:
        self.released = True


class FakeAdapter:
    environment = "fake"
    model = "fake_station"
    arms = ARMS
    camera_mounts = CAMERA_MOUNTS
    wrist_cameras = {"wrist_left": "left", "wrist_right": "right"}
    joint_names = ("j1", "j2", "j3", "j4", "j5", "j6")
    control_hz = 100.0
    tracking_stop_rad = 0.05
    finger_open_m = 0.04
    foreign_controller_markers = ("--allow-motion", "fake_teleop")

    def __init__(self, pose: Mapping[str, Sequence[float]] | None = None, *, blocked: bool = False,
                 kinematics: FakeKinematics | None = None,
                 take_over_error: Callable[[], BaseException] | None = None) -> None:
        self.buses = {"left": "fake_left", "right": "fake_right"}
        self.pose = {arm: np.asarray((pose or REST)[arm], float).copy() for arm in ARMS}
        self.blocked = blocked
        self.fake_kinematics = kinematics or FakeKinematics()
        self.take_over_error = take_over_error
        self.held: list[FakeHeld] = []
        self.alerts: list[dict[str, Any]] = []      # what every FakeHeld reports
        self.gripper_raw: dict[str, float] = {}     # each gripper motor's reading, when it has one
        self.readers: list[FakeReader] = []
        self.disabled = 0

    def open_reader(self, gripper_limits: Mapping[str, Sequence[float]]) -> FakeReader:
        reader = FakeReader(self)
        self.readers.append(reader)
        return reader

    def take_over(self, arms: Sequence[str], *, mode: str, gripper_limits: Mapping[str, Sequence[float]]) -> FakeHeld:
        if self.take_over_error is not None:
            raise self.take_over_error()
        held = FakeHeld(self, arms, mode)
        self.held.append(held)
        return held

    def disable_all(self) -> None:
        self.disabled += 1

    def calibrate_grippers(self, arms: Sequence[str], should_stop: Callable[[], bool]) -> dict[str, list[float]]:
        return {arm: [0.0, 1.0] for arm in arms}

    def kinematics(self, table: Mapping[str, Any] | None = None) -> FakeKinematics:
        return self.fake_kinematics

    def calibration_offsets(self) -> list[np.ndarray]:
        return [np.full(6, 0.1)]

    def facts(self) -> list[tuple[str, str]]:
        return [("robot", "fake")]


def make_cell(directory: Path, adapter: FakeAdapter | None = None, *, rest: Mapping[str, Sequence[float]] | None = REST):
    """A ``Cell`` on ``adapter`` (a fresh ``FakeAdapter`` by default) with ``rest`` recorded."""

    from ora.cell import calibration, config
    from ora.cell.operator.context import Cell

    loaded = config.load(write_config(directory))
    if rest is not None:
        calibration.save(loaded.calibration_path, {**calibration.empty(), "rest_reference": dict(rest)})
    return Cell(loaded, adapter or FakeAdapter())
