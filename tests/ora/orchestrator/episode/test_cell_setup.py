"""A real cell's setup before its play worker starts: the checklist, every progress line and the answer reach
the operator as play states ``setup`` and native/cell-setup.jsonl; a setup that does not finish ends the run."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from ora.orchestrator.episode import cell_setup
from ora.orchestrator.faults import SupervisorError

FAKE = r'''
import json, signal, sys, time
# As `ora cell setup` does: SIGINT stops it even when the Supervisor ignores SIGINT (a background start).
signal.signal(signal.SIGINT, signal.default_int_handler)
args = sys.argv[1:]
state = json.load(open(sys.argv[0] + ".state.json"))
if "--status" in args:
    print("initializing motorchain robot...")             # a driver's print before the answer
    print(json.dumps({"ok": True, "result": {"steps": [{"id": "table", "title": "桌面高度", "state": "stale",
          "detail": "标定之前测的"}], "next": state["next"], "ready_for_play": False, "argv": args}}))
    sys.exit(0)
print(json.dumps({"step": "table", "state": "running"}), file=sys.stderr, flush=True)
print(json.dumps({"progress": "测桌面高度"}), file=sys.stderr, flush=True)
print("UserWarning: not a setup line", file=sys.stderr, flush=True)
if state.get("wait"):
    time.sleep(30)
failed = state.get("failed")
print(json.dumps({"step": "table", "state": "failed" if failed else "done"}), file=sys.stderr, flush=True)
print(json.dumps({"ok": True, "result": {"done": [] if failed else ["table"], "failed": failed,
                                         "why": "没有看到桌面" if failed else None, "argv": args}}))
'''


class Loop:
    def __init__(self, run_root: Path) -> None:
        self.plan = {"runtime": {"kind": "yam", "rig_mode": "on", "cell_config": "/cells/cell.yaml"},
                     "play": {"cell_setup": True}}
        self.run_root = run_root
        self.interrupted = False
        self.said: list[dict] = []

    def _announce(self, state: str, **fields) -> None:
        assert state == "setup"
        self.said.append(fields["setup"])

    def record_auxiliary_failure(self, stage: str, exc: BaseException) -> None:
        raise AssertionError(f"{stage}: {exc}")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_ora.py"
    script.write_text(FAKE)

    def configure(**state) -> Loop:
        (tmp_path / "fake_ora.py.state.json").write_text(json.dumps({"next": "table", **state}))
        monkeypatch.setattr(cell_setup, "COMMAND", (sys.executable, str(script)))
        return Loop(tmp_path / "run")
    return configure


def _recorded(loop: Loop) -> list[dict]:
    path = loop.run_root / "native" / "cell-setup.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_ready_cell_shows_its_checklist_and_starts_at_once(fake) -> None:
    loop = fake(next=None)
    cell_setup.run_cell_setup(loop)
    assert [said["phase"] for said in loop.said] == ["status", "done"]
    assert loop.said[0]["event"]["status"]["argv"] == ["cell", "setup", "--status", "--json", "--cell",
                                                       "/cells/cell.yaml"]
    assert loop.said[1]["skipped"] is True and loop.said[1]["steps"][0]["state"] == "stale"


def test_the_setup_runs_what_is_missing_and_each_line_reaches_the_operator(fake) -> None:
    loop = fake()
    cell_setup.run_cell_setup(loop)
    assert [said["phase"] for said in loop.said] == ["status", "running", "running", "running", "done"]
    assert [said["event"] for said in loop.said[1:4]] == [{"step": "table", "state": "running"},
                                                         {"progress": "测桌面高度"},
                                                         {"step": "table", "state": "done"}]
    # Every announcement carries the whole view: the checklist with live states and the latest lines.
    assert loop.said[1]["steps"] == [{"id": "table", "title": "桌面高度", "state": "running", "detail": "标定之前测的",
                                      "why": None}]
    assert loop.said[-1]["steps"][0]["state"] == "done" and loop.said[-1]["lines"] == ["测桌面高度"]
    assert loop.said[-1]["result"]["argv"] == ["cell", "setup", "--ask", "broker", "--json", "--cell",
                                               "/cells/cell.yaml"]
    recorded = _recorded(loop)
    assert [r.get("phase") for r in recorded] == [None, None, None, None, "done"]
    assert recorded[0]["status"]["next"] == "table" and all("unix_s" in r for r in recorded)


def test_a_setup_that_stops_ends_the_run_before_any_arm_is_enabled(fake) -> None:
    loop = fake(failed="table")
    with pytest.raises(SupervisorError, match="stopped at table: 没有看到桌面; no arm was enabled"):
        cell_setup.run_cell_setup(loop)
    assert loop.said[-1]["phase"] == "failed" and loop.said[-1]["steps"][0] == {
        "id": "table", "title": "桌面高度", "state": "failed", "detail": "标定之前测的", "why": None}


def test_stopping_the_run_interrupts_the_setup(fake) -> None:
    loop = fake(wait=True)
    stop = threading.Timer(1.0, lambda: setattr(loop, "interrupted", True))
    stop.start()
    started = time.monotonic()
    with pytest.raises(SupervisorError, match="without an answer"):
        cell_setup.run_cell_setup(loop)
    assert time.monotonic() - started < 15, "SIGINT ended the setup"
    assert loop.said[-1]["phase"] == "failed"
