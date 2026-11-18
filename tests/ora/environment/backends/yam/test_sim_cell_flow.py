"""A new simulated YAM cell set up end to end with no hardware: ``ora cell setup --ask broker --json``
records rest, measures and orients the grippers, places the fixed cameras, runs the hand-eye
calibration, measures the table and verifies it, every question answered by the simulated
operator through the approval broker, and the calibration it writes finds the error the
simulation injected (the right arm's base, the joints' encoder zeros, the wrist mounts). The
cell board's status and live view run on the same cell. Needs MuJoCo, mink, i2rt and a
headless OpenGL (the YAM interpreter); the motions run fast and the cameras unthrottled."""

from __future__ import annotations

import json
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mink")
pytest.importorskip("i2rt")

from ora.app import cell as command  # noqa: E402
from ora.app.main import main  # noqa: E402
from ora.cell import config as cell_config  # noqa: E402
from ora.cell.operator import handeye, motion  # noqa: E402
from ora.cell.operator.context import Cell  # noqa: E402
from ora.environment.backends.yam import sim_cell_adapter  # noqa: E402
from tests.support.cell.fake import redirect_lock  # noqa: E402


@pytest.fixture
def sim_cell(tmp_path, monkeypatch):
    """A fresh simulated cell under ``tmp_path``: its own lock, approvals and alerts, fast motions."""

    monkeypatch.setenv("MUJOCO_GL", "egl")
    for name in cell_config.ISOLATED:
        monkeypatch.setenv(name, "")
    path = tmp_path / "sim.yaml"
    path.write_text(command.SIM_EXAMPLE.format(state=tmp_path / "sim.sim", python=sys.executable,
                                               calibration=tmp_path / "sim.calibration.json"))
    config = cell_config.load(path)
    cell_config.isolate(config)
    redirect_lock(monkeypatch, config.sim_state / "cell.lock")
    monkeypatch.setattr(motion, "SPEED_RAD_S", 3.0)
    monkeypatch.setattr(motion, "HOLD_S", 0.0)
    monkeypatch.setattr(handeye, "HOLD_S", 0.0)
    monkeypatch.setattr(motion, "RECORD_ROOT", tmp_path / "records")
    monkeypatch.setattr(handeye, "RECORD_ROOT", tmp_path / "records")
    monkeypatch.setattr(sim_cell_adapter, "FRAME_PERIOD_S", 0.0)
    monkeypatch.setattr(sim_cell_adapter, "GRIPPER_MOVE_S", 0.0)
    cell = Cell.load(path)
    try:
        camera = cell.open_camera("overhead")
        camera.capture()
        camera.close()
    except Exception as exc:  # noqa: BLE001 - no headless OpenGL here
        pytest.skip(f"MuJoCo cannot render here: {exc}")
    return path, cell


@pytest.mark.timeout(900)
def test_setup_through_the_broker_finds_the_injected_error(sim_cell, capsys) -> None:
    path, cell = sim_cell
    stop = threading.Event()
    operator = threading.Thread(target=cell.adapter.sim.operate, args=(stop,), kwargs={"poll_s": 0.05}, daemon=True)
    operator.start()
    try:
        code = main(["cell", "setup", "--cell", str(path), "--ask", "broker", "--json"])
    finally:
        stop.set()
        operator.join(timeout=5.0)
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and result["ok"], result
    assert result["result"]["failed"] is None
    assert result["result"]["done"] == ["rest", "grippers", "fixed_cameras", "handeye", "table", "verify"]
    assert result["result"]["status"]["ready_for_play"] is True

    calibration = cell.calibration()
    assert calibration["gripper_limits"]["right"] == [-5.16, 0.21]       # swapped once the person said "open"
    found = cell.adapter.sim.status(calibration)["recovery"]
    assert found["base"]["right"]["mm"] < 2.0 and found["base"]["right"]["deg"] < 0.3   # injected: 7.5 mm, 0.73 deg
    for arm, zeros in found["joint_zero_offsets_deg"].items():
        assert np.max(np.abs(zeros["error"])) < 0.2, (arm, zeros)       # injected: up to 1.4 deg
    assert all(m["mm"] < 2.0 and m["deg"] < 0.3 for m in found["wrist_mount"].values())
    assert all(c["mm"] < 6.0 and c["deg"] < 0.3 for c in found["fixed_cameras"].values())
    assert abs(found["table_mm"]) < 3.0


@pytest.mark.timeout(120)
def test_the_board_shows_the_simulated_cell_and_its_live_view(sim_cell) -> None:
    from ora.cell.operator.board import Board

    _, cell = sim_cell
    board = Board(cell)
    status = board.status()
    assert all(bus["present"] and bus["up"] for bus in status["buses"].values())
    assert any(fact[0] == "仿真" for fact in status["facts"])
    board.live.start()
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            summary = board.live.summary()
            if summary["arms"] and all(c["size"] for c in summary["cameras"].values()):
                break
            time.sleep(0.2)
        assert summary["arms"] and summary["arms_error"] is None
        assert {role: c["size"] for role, c in summary["cameras"].items()} == {
            role: [848, 480] for role in cell.config.cameras}
        assert all(c["error"] is None and c["source"].startswith("sim:") for c in summary["cameras"].values())
        assert cell.bus.receivers(cell.adapter.buses["left"]) == 1     # the live view's passive reader
    finally:
        board.live.stop()
