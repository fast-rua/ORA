"""The published alerts (``ora.cell.alerts``): what a publisher wrote, and how an ended publisher's file reads."""

from __future__ import annotations

import json

from ora.cell import alerts


def test_published_alerts_read_back_and_clear(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORA_CELL_ALERTS", str(tmp_path / "alerts.json"))
    alert = {"code": "motor_warm", "level": "warning", "text": "右臂夹爪电机 52 °C，在升温。", "arm": "right"}
    alerts.publish([alert, {"code": "x", "level": "debug", "text": "dropped"}], source="worker")

    read = alerts.current()
    assert read["alerts"] == [alert] and read["stale"] is False and read["source"] == "worker"
    alerts.clear(source="worker")
    assert alerts.current()["alerts"] == []


def test_an_ended_publisher_keeps_only_its_word_that_the_arms_are_uncontrolled(tmp_path, monkeypatch) -> None:
    target = tmp_path / "alerts.json"
    monkeypatch.setenv("ORA_CELL_ALERTS", str(target))
    target.write_text(json.dumps({"pid": 2 ** 22 + 12345, "alerts": [
        {"code": "motor_warm", "level": "warning", "text": "warm"},
        {"code": "arms_not_controlled", "level": "critical", "text": "not controlled"}]}))

    read = alerts.current()
    assert read["stale"] is True
    assert [a["code"] for a in read["alerts"]] == ["arms_not_controlled"]


def test_no_file_reads_as_no_alerts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORA_CELL_ALERTS", str(tmp_path / "missing.json"))
    assert alerts.current()["alerts"] == []
