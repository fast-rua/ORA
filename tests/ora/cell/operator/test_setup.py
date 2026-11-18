"""``ora cell setup`` says what a cell still needs and does it in order: a step done is skipped unless
redone, the first that fails stops the rest, a gripper whose reading left its limits after a power
cycle is stale, and a table measured before the hand-eye calibration is stale."""

from __future__ import annotations

import time

import pytest

from ora.cell.errors import CellError
from ora.cell.operator import setup


@pytest.fixture
def devices_ok(monkeypatch):
    monkeypatch.setattr(setup, "_devices", lambda cell: setup._step("devices", "done", "ok"))


def _states(cell) -> dict[str, str]:
    return {step["id"]: step["state"] for step in setup.status(cell)["steps"]}


def test_a_cell_with_only_its_rest_pose_starts_at_the_grippers(cell, devices_ok) -> None:
    found = setup.status(cell)
    assert _states(cell) == {"devices": "done", "rest": "done", "grippers": "missing", "fixed_cameras": "missing",
                             "handeye": "missing", "table": "missing", "verify": "missing"}
    assert found["next"] == "grippers" and found["ready_for_play"] is False
    assert all(step["title"] and step["detail"] for step in found["steps"])


def test_a_gripper_reading_outside_its_limits_is_stale(cell, adapter, devices_ok) -> None:
    value = cell.calibration()
    value["gripper_limits"] = {"left": [6.38, 1.11], "right": [6.40, 0.98]}
    cell.save_calibration(value)
    adapter.gripper_raw = {"left": 1.27, "right": 6.2}
    assert _states(cell)["grippers"] == "done"
    adapter.gripper_raw = {"left": 1.27, "right": -5.16}           # the right reading jumped a turn
    step = next(s for s in setup.status(cell)["steps"] if s["id"] == "grippers")
    assert step["state"] == "stale" and "右臂读数 -5.16" in step["detail"]


def test_a_table_measured_before_the_calibration_is_stale(cell, devices_ok) -> None:
    value = cell.calibration()
    value["bundle_fit"] = {"fitted_unix_s": time.time(), "held_out": {"1": {"wrist_left": [{"mm": 4.0}]}}}
    value["table"] = {"top_z_m": -0.09, "center_xy_m": [0.4, 0.0], "size_xy_m": [1.2, 1.2]}
    cell.save_calibration(value)
    assert _states(cell)["table"] == "stale"
    value["table"]["measured_unix_s"] = time.time() + 1
    cell.save_calibration(value)
    assert _states(cell)["table"] == "done"


def test_run_does_what_is_not_done_in_order_and_stops_at_a_failure(cell, devices_ok, monkeypatch) -> None:
    did = []

    def step(name, result=None):
        def do(cell, asker, progress):
            did.append(name)
            return result or {}
        return do

    monkeypatch.setattr(setup, "DO", {**{name: step(name) for name in setup.DO},
                                      "handeye": step("handeye", {"failed": "没有批准：不写"})})
    events = []
    report = setup.run(cell, asker=None, event=events.append)
    assert did == ["grippers", "fixed_cameras", "handeye"]
    assert (report["done"], report["failed"], report["why"]) == (["grippers", "fixed_cameras"], "handeye", "没有批准：不写")
    assert events[-1] == {"step": "handeye", "state": "failed", "why": "没有批准：不写"}

    did.clear()
    setup.run(cell, asker=None, only=("rest",), redo=("rest",))
    assert did == ["rest"]
    with pytest.raises(CellError, match="unknown setup step"):
        setup.run(cell, asker=None, redo=("calibrate",))
