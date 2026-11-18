"""RoboCasa365's part of the scene trace: its simulator, robot and TCP, its clock and what is backdrop in a kitchen."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ora.environment import mujoco_scene_trace
from ora.environment.backends.robocasa.scene_trace import RoboCasaSceneTrace
from ora.environment.controllers import robocasa_omron as C
from ora.environment.scene_trace import POSES, STEPS, Body, Mesh


def _mesh(*extents: float) -> Mesh:
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64) * extents
    return Mesh(corners, np.full((4, 4), 200, dtype=np.uint8), np.array([[0, 1, 2], [0, 2, 3]]))


class Sim:
    """Steps through ``step1``/``step2`` as robosuite's lite physics does: it has a ``step`` that never runs."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.data = SimpleNamespace()

    def step(self) -> None:
        raise AssertionError("lite physics does not call MjSim.step")


def _session() -> SimpleNamespace:
    robot = SimpleNamespace(robot_model=SimpleNamespace(root_body="robot0_base"))
    return SimpleNamespace(environment=SimpleNamespace(sim=Sim(), robots=[robot]), physics_steps_per_tick=25)


@pytest.fixture()
def trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RoboCasaSceneTrace, dict[str, Any]]:
    seen: dict[str, Any] = {"pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}

    def bodies(_model: Any, _data: Any, *, robot_roots: Any) -> list[Body]:
        seen["roots"] = robot_roots
        return [
            Body("mobilebase0_base", "base", "robot", lambda: seen["pose"], _mesh(0.5, 0.5, 0.3)),
            Body("wall_room_main", "wall", "static", lambda: seen["pose"], _mesh(6.0, 0.04, 3.0)),
            Body("floor_room_main", "floor", "static", lambda: seen["pose"], _mesh(5.0, 6.0, 0.04)),
            Body("counter_main_main_group_main", "counter", "static", lambda: seen["pose"], _mesh(3.8, 0.65, 0.03)),
        ]

    def links(_model: Any, sites: Any) -> dict[str, Any]:
        seen["sites"] = sites
        return {"right": {"body": "gripper0_right_eef", "offset_xyz_xyzw": [0, 0, 0, 0, 0, 0, 1]}}

    monkeypatch.setattr(mujoco_scene_trace, "scene_bodies", bodies)
    monkeypatch.setattr(mujoco_scene_trace, "tcp_links", links)
    return RoboCasaSceneTrace(tmp_path / "t", run_id="r"), seen


def test_a_round_reads_the_session_s_kitchen_robot_and_grip_site(trace) -> None:
    writer, seen = trace
    writer.begin_round(_session(), 0)

    assert seen["roots"] == ["robot0_base"] and seen["sites"] == {"right": C.GRIP_SITE}
    assert writer.finish()["backend"] == "robocasa"


def test_every_control_step_is_a_row_on_the_mujoco_clock_with_no_step_tap(trace, tmp_path: Path) -> None:
    writer, seen = trace
    original = Sim.step
    writer.begin_round(_session(), 0)
    assert Sim.step is original  # the class is left alone: lite physics would never call a tap on ``step``

    writer.begin_action({"action_sequence": 0, "action_id": "a0"})
    for tick in range(1, 4):
        seen["pose"] = [float(tick), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        writer.after_tick()
    writer.end_action()
    manifest = writer.finish()

    steps = [json.loads(line) for line in (tmp_path / "t" / STEPS).read_text().splitlines()]
    assert manifest["clock"] == {"unit": "mujoco_step_since_round_reset", "timestep_s": 0.002, "hz": 500.0, "stride": 25}
    assert [row["physics_step"] for row in steps] == [0, 25, 50, 75]
    assert manifest["spans"] == [{"round": 0, "action_sequence": 0, "action_id": "a0", "first_step": 0, "last_step": 75}]
    poses = np.frombuffer((tmp_path / "t" / POSES).read_bytes(), dtype="<f4").reshape(manifest["shape"])
    assert poses[:, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]  # the state after each control step, read at that step


def test_the_walls_and_floor_are_backdrop_and_the_long_counter_stays_in_the_view(trace) -> None:
    writer, _seen = trace
    writer.begin_round(_session(), 0)

    assert {body["id"]: body["kind"] for body in writer.finish()["bodies"]} == {
        "mobilebase0_base": "robot", "wall_room_main": "backdrop", "floor_room_main": "backdrop",
        "counter_main_main_group_main": "static"}


def test_a_new_round_restarts_the_clock_and_keeps_the_stride(trace) -> None:
    writer, _seen = trace
    writer.begin_round(_session(), 0)
    writer.begin_action({"action_sequence": 0, "action_id": "a0"})
    writer.after_tick()
    writer.end_action()
    writer.begin_round(_session(), 1)
    writer.begin_action({"action_sequence": 1, "action_id": "a1"})
    writer.after_tick()
    writer.end_action()
    manifest = writer.finish()

    assert manifest["rounds"] == [0, 1] and manifest["clock"]["stride"] == 25
    assert [span["last_step"] for span in manifest["spans"]] == [25, 25]
