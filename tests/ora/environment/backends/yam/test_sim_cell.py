"""The simulated YAM cell without MuJoCo: a ``robot.sim`` config loads the simulated adapter and
keeps its lock away from the real cell's; its buses count the handles and frames of every
process, so the kit's occupancy checks and a held arm's guards work as on socketcan; held arms
follow commands and stop following once another process disables them; the simulated operator
answers each kind of question; and ``recovery`` measures a calibration against the truth."""

from __future__ import annotations

import subprocess
import sys
import threading

import numpy as np
import pytest

from ora.cell import config as cell_config
from ora.cell.adapter import load_adapter
from ora.cell.errors import CellError
from ora.environment.backends.yam import sim_cell
from ora.environment.backends.yam.sim_cell import DEFAULT_WORLD, SimCell, SimState, fixed_camera_pose, recovery
from tests.support.cell.fake import redirect_lock

SIM_CONFIG = """schema: ora.cell-config.v1
cell_id: sim-test-01
robot:
  environment: yam
  model: yam_station_linear_4310_d405
  sim: true
  buses: {left: sim_left, right: sim_right}
cameras:
  overhead:    {serial: "sim-overhead", mount: fixed_world}
  wrist_left:  {serial: "sim-wrist-left", mount: end_effector}
  wrist_right: {serial: "sim-wrist-right", mount: end_effector}
calibration: calibration.json
"""


@pytest.fixture
def sim_config(tmp_path, monkeypatch):
    for name in cell_config.ISOLATED:
        monkeypatch.setenv(name, "")               # unset for ``isolate``, and restored after the test
    monkeypatch.setenv("MUJOCO_GL", "egl")          # the adapter would set it for the whole session
    path = tmp_path / "sim.yaml"
    path.write_text(SIM_CONFIG.replace("calibration.json", str(tmp_path / "calibration.json")))
    config = cell_config.load(path)
    chosen = cell_config.isolate(config)
    redirect_lock(monkeypatch, config.sim_state / "cell.lock")
    return config, chosen


@pytest.fixture
def adapter(sim_config):
    return load_adapter(sim_config[0])


def test_a_sim_config_isolates_its_lock_approvals_and_alerts(sim_config, tmp_path) -> None:
    config, chosen = sim_config
    assert config.sim_state == tmp_path / "sim.sim"
    assert chosen == {"ORA_CELL_LOCK": str(tmp_path / "sim.sim" / "cell.lock"),
                      "ORA_CELL_APPROVALS": str(tmp_path / "sim.sim" / "approvals"),
                      "ORA_CELL_ALERTS": str(tmp_path / "sim.sim" / "alerts.json")}
    assert cell_config.isolate(config) == {}                     # what the environment names is kept


def test_the_sim_adapter_refuses_the_real_cells_lock(sim_config, monkeypatch) -> None:
    from ora.cell import lock

    monkeypatch.setattr(lock, "LOCK_PATH", lock.DEFAULT_LOCK_PATH)
    with pytest.raises(CellError, match="real cell's lock"):
        load_adapter(sim_config[0])


def test_a_real_config_has_no_sim_state(tmp_path) -> None:
    path = tmp_path / "real.yaml"
    path.write_text(SIM_CONFIG.replace("  sim: true\n", ""))
    assert cell_config.load(path).sim_state is None
    path.write_text(SIM_CONFIG.replace("sim: true", "sim: {state: elsewhere}"))
    assert cell_config.load(path).sim_state == tmp_path / "elsewhere"
    path.write_text(SIM_CONFIG.replace("sim: true", "sim: maybe"))
    with pytest.raises(CellError, match="robot.sim"):
        cell_config.load(path)


def test_the_arms_start_at_rest_with_their_encoders_off_by_the_injected_zeros(adapter) -> None:
    reader = adapter.open_reader({"left": [6.37, 1.11]})
    try:
        readings = reader.read()
    finally:
        reader.close()
    expected = -np.asarray(DEFAULT_WORLD["joint_zero_offsets"]["left"])
    assert np.allclose(readings["left"].q, expected)
    assert readings["left"].gripper_opening == pytest.approx(1.0) and readings["right"].gripper_opening is None
    assert np.all(readings["left"].q[1:3] >= 0)                  # inside the model's joint limits at rest


def test_open_handles_are_receivers_and_reads_are_traffic(adapter) -> None:
    bus, channel = adapter.bus, adapter.buses["left"]
    assert bus.receivers(channel) == 0 and bus.traffic(channel, 0.05) == 0
    reader = adapter.open_reader({})
    assert bus.receivers(channel) == 1
    reading = threading.Thread(target=lambda: [reader.read() for _ in range(5)])
    reading.start()
    heard = bus.traffic(channel, 0.3)
    reading.join()
    reader.close()
    assert heard >= 1 and bus.receivers(channel) == 0
    assert bus.interface(channel)["up"] is True


def test_a_handle_of_a_dead_process_is_no_receiver(adapter) -> None:
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    folder = adapter.state.directory / "bus" / "sim_left"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{dead.stdout.strip()}-1-1").write_text("")
    assert adapter.bus.receivers("sim_left") == 0 and not any(folder.iterdir())


def test_held_arms_follow_until_another_process_disables_them(adapter) -> None:
    held = adapter.take_over(["left", "right"], mode="hold", gripper_limits={"left": [6.37, 1.11]})
    with pytest.raises(CellError, match="in use by another program"):
        adapter.take_over(["left"], mode="hold", gripper_limits={})
    target = held.joints()["left"] + 0.2
    held.command({"left": target})
    assert np.allclose(held.joints()["left"], target) and held.alive() == {"left": True, "right": True}
    # the left gripper closes as far as a hold goes; the right has no measured limits to move by
    assert held.grip({"left": 0.0, "right": 0.0}) == {"left": pytest.approx(0.03)}
    adapter.disable_all()                                         # ``ora cell estop`` from elsewhere
    assert held.alive() == {"left": False, "right": False}
    held.command({"left": target + 0.2})
    assert np.allclose(held.joints()["left"], target)             # a disabled motor follows nothing
    held.release()
    assert adapter.bus.receivers(adapter.buses["left"]) == 0


def test_the_gripper_calibration_names_the_stops_by_the_motors_direction(adapter) -> None:
    found = adapter.calibrate_grippers(["left", "right"], lambda: False)
    assert found == {"left": [6.37, 1.11], "right": [0.21, -5.16]}  # the right gripper's: swapped
    assert adapter.state.arms()["arms"]["right"]["gripper_raw"] == 0.21   # it ends open


def test_the_simulated_operator_answers_as_the_person_would(adapter, tmp_path) -> None:
    state = adapter.state
    person = SimCell(state)
    assert person.answer({"kind": "value", "details": {"unit": "mm"}})[2] == "20"
    assert person.answer({"kind": "value", "details": {"unit": "columns x rows"}})[2] == "11x8"
    adapter.calibrate_grippers(["right"], lambda: False)
    assert person.answer({"kind": "value", "details": {"arm": "right", "reading": 0.21}})[2] == "开"
    assert person.answer({"kind": "task", "details": {"for": "devices"}})[0] is False
    with state.changing() as value:
        value["arms"]["left"]["q"] = [0.3] * 6
    assert person.answer({"kind": "task", "details": {"for": "rest"}})[0] is True
    assert np.allclose(state.arms()["arms"]["left"]["q"], state.rest_readings("left"))
    def placed(number: int) -> np.ndarray:
        assert person.answer({"kind": "task", "details": {"placement": number, "arms": ["left"]}})[0] is True
        pose = state.board()
        return pose[:2, :3] @ [0.1, 0.07, 0.0] + pose[:2, 3]        # the middle of the 11x8 board

    first, again, third = placed(2), placed(2), placed(3)
    base = np.asarray(sim_cell.Y.BASE_IN_WORLD_XYZ_M["left"][:2])
    assert np.linalg.norm(again - base) < np.linalg.norm(first - base)       # asked again: nearer its arm
    assert not np.allclose(third, first)                                     # the arm's next turn: elsewhere
    assert person.answer({"kind": "motion", "details": {}}) == (True, None, None)


def test_the_operator_answers_the_broker_until_stopped(adapter, sim_config) -> None:
    from ora.cell.operator import approval

    stop = threading.Event()
    answering = threading.Thread(target=SimCell(adapter.state).operate, args=(stop,), kwargs={"poll_s": 0.02})
    answering.start()
    try:
        value = approval.ask_value("标定板：内角点数", "", {"unit": "columns x rows"}, timeout_s=10.0, poll_s=0.02)
        moved = approval.ask("motion", "move", "", {}, timeout_s=10.0, poll_s=0.02)
    finally:
        stop.set()
        answering.join()
    assert value == ("11x8", None) and moved == (True, None)
    assert str(approval.directory()).startswith(str(sim_config[0].sim_state))


def test_recovery_of_the_truth_itself_is_exact() -> None:
    world = DEFAULT_WORLD
    calibration = {"arm_base_corrections": world["arm_base_corrections"],
                   "wrist_camera_corrections": world["wrist_camera_corrections"],
                   "joint_zero_offsets": {a: {"rad": v} for a, v in world["joint_zero_offsets"].items()},
                   "cameras": {r: {"cam2world_cv": fixed_camera_pose(s).tolist()} for r, s in world["fixed_cameras"].items()},
                   "table": {"top_z_m": world["table"]["top_z_m"]}}
    found = recovery(world, calibration)
    assert all(g == {"mm": 0.0, "deg": 0.0} for part in ("base", "wrist_mount", "fixed_cameras")
               for g in found[part].values())
    assert all(not any(z["error"]) for z in found["joint_zero_offsets_deg"].values()) and found["table_mm"] == 0.0
    nothing = recovery(world, {})
    assert nothing["base"]["right"]["mm"] == pytest.approx(7.48, abs=0.01) and nothing["fixed_cameras"]["front"] is None


def test_a_fixed_camera_pose_looks_where_it_is_told() -> None:
    pose = fixed_camera_pose({"position_m": [0, 0, 1], "look_at_m": [0, 0, 0], "up_hint": [1, 0, 0]})
    assert np.allclose(pose[:3, 2], [0, 0, -1]) and np.allclose(pose[:3, 1], [-1, 0, 0])   # image down: -x
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)
