"""The operator verbs on a fake cell: ``stop`` signals only holders that home on a signal and recognises one
at rest; the ``Holder`` takes the lock with the arms, switches modes without letting go and frees both on
``release``; ``drive_home`` holds, never releases, when the arms lag or the path is refused; ``home`` leaves
arms already at rest alone, and releases only once at rest."""

from __future__ import annotations

import os
import signal
import time
from types import SimpleNamespace

import numpy as np
import pytest

from ora.cell.errors import CellError
from ora.cell.lock import AT_REST, HOMES_ON_SIGNAL, HOMING, CellLock, holder
from ora.cell.operator import actions
from tests.support.cell.fake import FakeAdapter, FakeKinematics

AWAY = {"left": [0.0, 0.3, 0.2, 0.0, 0.0, 0.0], "right": [0.0, 0.3, 0.2, 0.0, 0.0, 0.0]}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """The verbs' waits return at once; their deadlines still run on the real clock."""

    monkeypatch.setattr(actions, "time", SimpleNamespace(sleep=lambda _s: None, monotonic=time.monotonic,
                                                         time=time.time))


@pytest.fixture
def signals(monkeypatch):
    """Signals the verbs send, recorded instead of sent; ``on_signal`` plays the receiving program."""

    sent: list[tuple[int, int]] = []
    state = SimpleNamespace(sent=sent, on_signal=lambda pid, number: None)

    def kill(pid: int, number: int) -> None:
        sent.append((pid, number))
        state.on_signal(pid, number)

    monkeypatch.setattr(actions, "os", SimpleNamespace(kill=kill, getpid=os.getpid))
    return state


# ---------------------------------------------------------------- stop
def test_stop_without_a_holder_sends_nothing(signals) -> None:
    assert actions.stop() == "no program holds the cell"
    assert signals.sent == []


def test_stop_leaves_a_holder_that_does_not_home_on_a_signal_alone(signals) -> None:
    with CellLock("cell board (live view)"):
        answer = actions.stop()
    assert "does not home on a signal" in answer
    assert signals.sent == []


def test_stop_signals_a_homing_holder_and_recognises_it_at_rest(signals) -> None:
    lock = CellLock(f"ora worker {HOMES_ON_SIGNAL}").acquire()
    signals.on_signal = lambda pid, number: lock.relabel(f"ora worker {HOMES_ON_SIGNAL} {AT_REST}")
    try:
        answer = actions.stop(timeout_s=5.0)
    finally:
        lock.release()
    assert signals.sent == [(os.getpid(), signal.SIGUSR1)]
    assert answer == f"pid {os.getpid()} went home and holds the arms at rest"


def test_stop_reports_a_holder_that_went_home_and_released(signals) -> None:
    lock = CellLock(f"calibration motion {HOMES_ON_SIGNAL}").acquire()
    signals.on_signal = lambda pid, number: lock.release()
    assert actions.stop(timeout_s=5.0) == f"pid {os.getpid()} went home and released the cell"
    assert signals.sent == [(os.getpid(), signal.SIGUSR1)]
    assert holder() is None


def test_stop_waits_longer_for_a_holder_driving_home(signals, monkeypatch) -> None:
    monkeypatch.setattr(actions, "HOMING_WAIT_S", 0.3)
    with CellLock(f"ora worker {HOMES_ON_SIGNAL} {HOMING}"):
        begun = time.monotonic()
        answer = actions.stop(timeout_s=0.05)
    assert time.monotonic() - begun >= 0.3
    assert answer.endswith("still holds the cell after 0 s")


def test_stop_says_when_the_holder_does_not_answer(signals) -> None:
    with CellLock(f"ora worker {HOMES_ON_SIGNAL}"):
        answer = actions.stop(timeout_s=0.05)
    assert answer == f"pid {os.getpid()} still holds the cell after 0 s"
    assert signals.sent == [(os.getpid(), signal.SIGUSR1)]


# ---------------------------------------------------------------- the Holder
def test_the_holder_takes_the_lock_with_the_arms_and_frees_both_on_release(cell, adapter) -> None:
    held = actions.Holder(cell)

    assert held.start("float") == "arms floating (gravity compensation)"
    assert held.active and held.mode == "float"
    assert holder().startswith("cell operator (holding the arms)")
    assert adapter.held[0].modes == ["float"]

    assert held.start("hold") == "switched to hold"                     # no second take-over
    assert len(adapter.held) == 1 and adapter.held[0].modes == ["float", "hold"]
    assert held.mode == "hold"

    assert held.release() == "released: the arms are limp"
    assert adapter.held[0].released
    assert not held.active and holder() is None
    assert held.release() == "nothing held"


def test_a_failed_take_over_leaves_the_cell_free(cell, adapter) -> None:
    adapter.take_over_error = lambda: CellError("fake_left is in use by another program")
    with pytest.raises(CellError, match="in use"):
        actions.Holder(cell).start("hold")
    assert holder() is None


def test_drive_home_reaches_rest_and_keeps_holding(cell, adapter) -> None:
    adapter.pose = {arm: np.asarray(q) for arm, q in AWAY.items()}
    held = actions.Holder(cell)
    held.start("float")

    assert held.drive_home() == "at rest, holding"

    assert held.note == "at rest" and held.mode == "hold" and held.active
    assert not adapter.held[0].released
    assert all(np.max(np.abs(q)) < actions.HOME_TOLERANCE_RAD for q in adapter.pose.values())
    steps = [np.max(np.abs(b["left"] - a["left"])) for a, b in zip(adapter.held[0].commands, adapter.held[0].commands[1:])]
    assert max(steps) <= actions.HOME_SPEED_RAD_S / adapter.control_hz + 1e-12
    held.release()


def test_drive_home_holds_where_the_arms_are_when_they_do_not_follow(cell, adapter) -> None:
    adapter.pose = {arm: np.asarray(q) for arm, q in AWAY.items()}
    adapter.blocked = True
    held = actions.Holder(cell)
    held.start("hold")

    answer = held.drive_home()

    assert answer.endswith("does not follow (blocked?): holding where it is")
    assert held.active and not adapter.held[0].released
    last = adapter.held[0].commands[-1]
    assert all(np.allclose(last[arm], AWAY[arm]) for arm in last)          # the measured pose, held
    held.release()


@pytest.mark.parametrize(("kinematics", "problem"), [
    (FakeKinematics(inter_arm_m=0.01), "the arms come within 0.06 m of each other at 0%"),
    (FakeKinematics(table_m=-0.01), "arm passes through the table at 0%"),
])
def test_drive_home_holds_where_the_arms_are_when_the_path_is_refused(cell, adapter, kinematics, problem) -> None:
    adapter.pose = {arm: np.asarray(q) for arm, q in AWAY.items()}
    adapter.fake_kinematics = kinematics
    held = actions.Holder(cell)
    held.start("float")

    answer = held.drive_home()

    assert answer.startswith("a straight path home is not clear: ") and problem in answer
    assert answer.endswith("Holding here; float the arms closer to rest by hand and try again")
    assert held.active and held.mode == "hold" and adapter.held[0].modes == ["float", "hold"]
    assert adapter.held[0].commands == [] and not adapter.held[0].released
    held.release()


# ---------------------------------------------------------------- home
def test_home_leaves_arms_already_at_rest_alone(cell, adapter) -> None:
    assert actions.home(cell, actions.Holder(cell)) == "already at rest: nothing moved"
    assert adapter.held == [] and adapter.readers[0].closed
    assert holder() is None


def test_home_drives_to_rest_then_releases(cell, adapter) -> None:
    adapter.pose = {arm: np.asarray(q) for arm, q in AWAY.items()}
    held = actions.Holder(cell)

    assert actions.home(cell, held) == "at rest and released"

    assert adapter.held[0].released and not held.active and holder() is None


def test_home_keeps_holding_when_the_way_is_blocked(cell, adapter) -> None:
    adapter.pose = {arm: np.asarray(q) for arm, q in AWAY.items()}
    adapter.blocked = True
    held = actions.Holder(cell)

    answer = actions.home(cell, held)

    assert "does not follow" in answer
    assert held.active and not adapter.held[0].released
    assert holder().startswith("cell operator (holding the arms)")
    held.release()


def test_home_needs_a_recorded_rest_pose(tmp_path, quiet_buses) -> None:
    from tests.support.cell.fake import make_cell

    cell = make_cell(tmp_path, rest=None)
    with pytest.raises(CellError, match="no rest pose recorded"):
        actions.home(cell, actions.Holder(cell))
    assert holder() is None


# ---------------------------------------------------------------- the motors-off verbs
def test_release_all_asks_first(cell, adapter) -> None:
    assert actions.release_all(cell, confirm=lambda _q: False) == "nothing sent"
    assert adapter.disabled == 0
    assert actions.release_all(cell, confirm=lambda _q: True) == "every motor disabled"
    assert adapter.disabled == 1 and holder() is None


def test_estop_disables_every_motor_twice_whoever_holds_the_cell(cell, adapter, signals) -> None:
    assert actions.estop(cell) == "every motor disabled"
    assert adapter.disabled == 2 and signals.sent == []


def test_the_holder_publishes_the_held_arms_alerts_for_the_board(cell, adapter, monkeypatch) -> None:
    from ora.cell import alerts as cell_alerts

    monkeypatch.setattr(actions, "ALERT_PERIOD_S", 0.01)
    adapter.alerts = [{"code": "motor_warm", "level": "warning", "arm": "right", "motor": "夹爪", "text": "右臂夹爪电机 52 °C，在升温。"}]
    held = actions.Holder(cell)
    held.start("hold")
    end = time.monotonic() + 2.0
    while time.monotonic() < end and not cell_alerts.current()["alerts"]:
        time.sleep(0.01)
    assert [a["code"] for a in cell_alerts.current()["alerts"]] == ["motor_warm"]
    held.release()
