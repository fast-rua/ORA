"""A supervised calibration motion watches the held arms: their alerts reach the board, and a
critical one (a motor near its protection) stops the motion like the operator's home."""

from __future__ import annotations

import numpy as np
import pytest

from ora.cell import alerts as cell_alerts
from ora.cell import occupancy
from ora.cell.operator import motion


@pytest.fixture
def mover(cell, adapter, monkeypatch):
    monkeypatch.setattr(occupancy, "can_receivers", lambda channel: 0)
    monkeypatch.setattr(motion.time, "sleep", lambda seconds: None)
    moving = motion.Mover(cell)
    yield moving
    if not moving.released:
        moving.held.release()


def test_a_warning_is_published_and_the_motion_goes_on(mover, adapter) -> None:
    adapter.alerts = [{"code": "motor_warm", "level": "warning", "arm": "left", "text": "左臂肘电机 52 °C，在升温。"}]
    mover.move({arm: np.full(6, 0.01) for arm in adapter.arms})
    assert [a["code"] for a in cell_alerts.current()["alerts"]] == ["motor_warm"]
    assert np.allclose(adapter.pose["left"], 0.01)


def test_a_critical_alert_stops_the_motion_and_stays_on_the_board(mover, adapter) -> None:
    adapter.alerts = [{"code": "motor_hot", "level": "critical", "arm": "right", "motor": "夹爪",
                       "text": "右臂夹爪电机 61 °C，接近过温保护"}]
    with pytest.raises(motion.Stop, match="接近过温保护"):
        mover.move({arm: np.full(6, 0.3) for arm in adapter.arms})
    start = {arm: np.zeros(6) for arm in adapter.arms}
    mover.back_and_release(start)
    assert adapter.held[0].released
    assert [a["code"] for a in cell_alerts.current()["alerts"]] == ["motor_hot"]
