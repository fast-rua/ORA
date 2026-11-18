"""The cell board answers a worker's questions: ``/api/approval`` and ``/api/status`` list what waits, and
``POST /api/approval/<id>/approve|decline`` answers that id only, with the reason from the page."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from ora.cell import occupancy
from ora.cell.operator import approval, board as board_module
from ora.cell.errors import CellError
from ora.cell.lock import CellLock
from ora.cell.operator.board import Board, Calibration, calibrate_arguments, make_handler


@pytest.fixture
def board_served(cell, monkeypatch):
    """A board on the fake cell, on a free local port, and the board; its status reads no host interface."""

    monkeypatch.delenv("ORA_CELL_APPROVALS", raising=False)
    monkeypatch.setattr(occupancy, "can_interface", lambda channel: {"present": False})
    monkeypatch.setattr(occupancy, "can_receivers", lambda channel: 0)
    monkeypatch.setattr(board_module, "motion_processes", lambda markers, ignore=(): [])
    board = Board(cell)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(board))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", board
    server.shutdown()
    server.server_close()


@pytest.fixture
def served(board_served):
    return board_served[0]


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as reply:
        return json.loads(reply.read())


def _post(url: str, body: bytes = b"") -> tuple[int, dict]:
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return exc.code, json.loads(body) if body.startswith(b"{") else {"text": body.decode()}


def _asking(kind: str = "motion", timeout_s: float = 10.0) -> tuple[threading.Thread, list]:
    result: list = []
    thread = threading.Thread(target=lambda: result.append(approval.ask(kind, "左臂动作等待批准", "左臂移动，约 1.5 秒",
                                                                       {"arms": ["left"]}, timeout_s=timeout_s)),
                              daemon=True)
    thread.start()
    for _ in range(500):
        if approval.pending():
            break
        time.sleep(0.01)
    return thread, result


def test_the_page_lists_what_waits_and_declines_it_with_a_reason(served) -> None:
    assert _get(served + "/api/approval")["pending"] == []
    thread, result = _asking()

    [request] = _get(served + "/api/approval")["pending"]
    assert (request["kind"], request["title"], request["details"]) == ("motion", "左臂动作等待批准", {"arms": ["left"]})
    assert [r["id"] for r in _get(served + "/api/status")["approvals"]] == [request["id"]]

    code, reply = _post(f"{served}/api/approval/{request['id']}/decline", json.dumps({"reason": "杯子挡着"}).encode())
    thread.join(timeout=5)
    assert (code, reply) == (200, {"ok": True, "result": f"declined {request['id']}: 左臂动作等待批准"})
    assert result == [(False, "杯子挡着")]
    assert _get(served + "/api/approval")["pending"] == []


def test_only_the_id_on_the_page_is_answered(served) -> None:
    thread, result = _asking()
    [request] = approval.pending()

    code, reply = _post(f"{served}/api/approval/deadbeef/approve")
    assert code == 409 and "no question deadbeef is waiting" in reply["error"]
    code, reply = _post(f"{served}/api/approval/{request['id']}/approve", b"not json")
    assert code == 409 and reply["error"] == "the answer's body is not JSON"
    code, _ = _post(f"{served}/api/approval/{request['id']}/approve", b"x" * (board_module.MAX_BODY_BYTES + 1))
    assert code == 413
    assert [r["id"] for r in approval.pending()] == [request["id"]], "still waiting"

    code, reply = _post(f"{served}/api/approval/{request['id']}/approve")
    thread.join(timeout=5)
    assert code == 200 and reply["result"].startswith("approved ") and result == [(True, None)]

    code, reply = _post(f"{served}/api/approval/{request['id']}/approve")
    assert code == 409 and "already answered: approved (cell board)" in reply["error"]
    assert _post(f"{served}/api/approval/{request['id']}/maybe")[0] == 404


def test_the_page_shows_the_alerts_and_the_auto_approve_switch(served, tmp_path, monkeypatch) -> None:
    from ora.cell import alerts

    monkeypatch.setenv("ORA_CELL_ALERTS", str(tmp_path / "alerts.json"))
    alerts.publish([{"code": "motor_warm", "level": "warning", "text": "右臂夹爪电机 52 °C，在升温。"}], source="worker")
    assert [a["code"] for a in _get(f"{served}/api/alerts")["alerts"]] == ["motor_warm"]

    assert _get(f"{served}/api/approval")["auto_approve"]["on"] is False
    code, body = _post(f"{served}/api/auto-approve/on")     # no play worker drives the arms: refused
    assert code == 409 and "error" in body


def test_the_page_shows_and_switches_the_permission_mode(served) -> None:
    state = _get(f"{served}/api/approval")
    assert (state["mode"]["mode"], state["self_answered"]) == ("ask", [])
    assert state["mode"]["safe_limits"] == approval.SAFE_LIMITS
    code, reply = _post(f"{served}/api/mode/safe")
    assert code == 200 and (reply["result"]["mode"], reply["result"]["by"]) == ("safe", "cell board")
    assert _get(f"{served}/api/status")["mode"]["mode"] == "safe"

    plan = {"duration_s": 2.5, "max_joint_travel_rad": 0.3, "min_table_distance_m": 0.08, "min_inter_arm_distance_m": 0.4}
    assert approval.ask("motion", "左臂动作等待批准", "左臂移动", plan, timeout_s=5.0) == (True, None)
    [answered] = _get(f"{served}/api/approval")["self_answered"]
    assert (answered["title"], answered["by"]) == (
        "左臂动作等待批准", "safe mode (2.5 s, 0.30 rad, 8 cm above the table, 40 cm between the arms)")

    code, reply = _post(f"{served}/api/mode/auto")          # no play worker drives the arms: refused
    assert code == 409 and "no play worker drives the arms" in reply["error"]
    code, reply = _post(f"{served}/api/mode/yolo")
    assert code == 409 and "unknown permission mode" in reply["error"]
    assert _post(f"{served}/api/mode/ask")[1]["result"]["mode"] == "ask"
    assert _post(f"{served}/api/auto-approve/off")[0] == 200, "the older switch still answers"


def test_a_value_question_is_answered_with_the_value_from_the_page(served) -> None:
    result: list = []
    thread = threading.Thread(target=lambda: result.append(approval.ask_value("一个格子的边长", "量 10 格", timeout_s=10.0)),
                              daemon=True)
    thread.start()
    for _ in range(500):
        if approval.pending():
            break
        time.sleep(0.01)
    [request] = approval.pending()
    code, reply = _post(f"{served}/api/approval/{request['id']}/approve", json.dumps({"value": 20}).encode())
    assert code == 409 and reply["error"] == "reason and value must be text"
    code, _ = _post(f"{served}/api/approval/{request['id']}/approve", json.dumps({"value": "20"}).encode())
    thread.join(timeout=5)
    assert code == 200 and result == [("20", None)]


# ---------------------------------------------------------------- the hand-eye calibration
FAKE_CALIBRATE = """
import json, signal, sys, time
# As `ora cell calibrate` does: SIGINT stops it even when its parent ignores SIGINT (a board started in
# the background passes that on).
signal.signal(signal.SIGINT, signal.default_int_handler)
print(json.dumps({"interruptible": True}), file=sys.stderr, flush=True)
print(json.dumps({"progress": "placement 1: locating the board"}), file=sys.stderr, flush=True)
print("UserWarning: a library said something", file=sys.stderr, flush=True)
if "--placements" in sys.argv and sys.argv[sys.argv.index("--placements") + 1] == "9":
    time.sleep(30)
print(json.dumps({"ok": True, "result": {"saved": False, "not_saved": "declined", "argv": sys.argv[1:]}}))
"""


#: A calibration that ignores SIGINT, as a child of a board started in the background does until
#: ``ora cell`` takes interrupts, and takes them once ``go`` (its first argument) exists.
LATE_CALIBRATE = """
import json, os, signal, sys, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
while not os.path.exists(sys.argv[1]):
    time.sleep(0.01)
signal.signal(signal.SIGINT, signal.default_int_handler)
print(json.dumps({"interruptible": True}), file=sys.stderr, flush=True)
time.sleep(30)
"""


def _finished(board: Board, seconds: float = 10.0) -> dict:
    for _ in range(int(seconds / 0.02)):
        state = board.calibration.state()
        if not state["running"] and state["outcome"] is not None:
            return state
        time.sleep(0.02)
    raise AssertionError(f"the calibration did not end: {board.calibration.state()}")


def test_the_page_fields_become_the_calibrate_options() -> None:
    assert calibrate_arguments({}) == []
    assert calibrate_arguments({"square_mm": "20", "pattern": " 11 × 8 ", "placements": "5"}) == [
        "--square-mm", "20", "--pattern", "11x8", "--placements", "5"]
    assert calibrate_arguments({"square_mm": "", "pattern": None, "placements": ""}) == []
    for bad in ({"square_mm": "abc"}, {"square_mm": "0"}, {"pattern": "11-8"}, {"pattern": "--yes"},
                {"placements": "2.5"}, {"placements": "50"}):
        with pytest.raises(CellError):
            calibrate_arguments(bad)


def test_the_board_starts_the_calibration_and_keeps_its_progress_and_answer(board_served, tmp_path) -> None:
    served, board = board_served
    script = tmp_path / "fake_calibrate.py"
    script.write_text(FAKE_CALIBRATE)
    board.calibration = Calibration(board.cell, command=[sys.executable, str(script)])

    code, reply = _post(f"{served}/api/calibrate/start", json.dumps({"square_mm": "20", "pattern": "11x8"}).encode())
    assert code == 200 and reply["result"]["running"] in (True, False)
    state = _finished(board)
    assert [p["text"] for p in state["progress"]] == ["placement 1: locating the board",
                                                      "UserWarning: a library said something"]
    assert state["progress"][1]["raw"] is True and "raw" not in state["progress"][0]
    assert state["exit_code"] == 0 and state["outcome"]["ok"] is True
    assert state["outcome"]["result"]["argv"] == ["cell", "calibrate", "--method", "hand-eye", "--ask", "broker", "--json",
                                                  "--cell", str(board.cell.config.path), "--square-mm", "20",
                                                  "--pattern", "11x8"]
    assert _get(f"{served}/api/status")["calibrate"]["outcome"]["result"]["not_saved"] == "declined"

    code, reply = _post(f"{served}/api/calibrate/start", json.dumps({"pattern": "11-8"}).encode())
    assert code == 409 and "COLSxROWS" in reply["error"]
    code, reply = _post(f"{served}/api/calibrate/stop")
    assert code == 409 and reply["error"] == "no hand-eye calibration runs"


def test_one_calibration_at_a_time_never_over_another_holder_and_stoppable(board_served, tmp_path) -> None:
    served, board = board_served
    script = tmp_path / "fake_calibrate.py"
    script.write_text(FAKE_CALIBRATE)
    board.calibration = Calibration(board.cell, command=[sys.executable, str(script)])

    with CellLock("ORA worker yam rig_mode on (pid 1)"):
        code, reply = _post(f"{served}/api/calibrate/start")
        assert code == 409 and "the cell is held by ORA worker yam rig_mode on" in reply["error"]
    assert board.calibration.process is None, "nothing started"

    code, _ = _post(f"{served}/api/calibrate/start", json.dumps({"placements": "9"}).encode())   # waits 30 s
    assert code == 200 and board.calibration.running
    code, reply = _post(f"{served}/api/calibrate/start")
    assert code == 409 and "already runs" in reply["error"]
    assert _post(f"{served}/api/live/start")[0] == 409
    code, _ = _post(f"{served}/api/calibrate/stop")             # SIGINT ends the fake at once
    state = _finished(board)
    assert code == 200 and state["outcome"]["ok"] is False and "without an answer" in state["outcome"]["error"]
    assert any("interrupted from the board" in p["text"] for p in state["progress"])


def test_a_stop_asked_before_the_calibration_takes_interrupts_is_held_until_it_does(board_served, tmp_path) -> None:
    served, board = board_served
    script, go = tmp_path / "late_calibrate.py", tmp_path / "go"
    script.write_text(LATE_CALIBRATE)
    board.calibration = Calibration(board.cell, command=[sys.executable, str(script), str(go)])

    assert _post(f"{served}/api/calibrate/start")[0] == 200
    code, reply = _post(f"{served}/api/calibrate/stop")
    assert code == 200 and reply["result"]["running"] is True
    assert reply["result"]["progress"][-1]["text"] == "stop asked: held until the calibration can take it"
    time.sleep(0.3)
    assert board.calibration.running, "nothing was sent that it would have lost"
    go.touch()
    state = _finished(board)
    assert state["outcome"]["ok"] is False and state["exit_code"] != 0
    assert [p["text"] for p in state["progress"]][:2] == ["stop asked: held until the calibration can take it",
                                                          "stop asked: the calibration was interrupted from the board"]
