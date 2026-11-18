"""The approval broker: a question posted by one process, answered by id from another; an answer for an id
that is not pending does nothing; a dead asker's question is ignored; no answer in time is a decline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ora.cell.errors import CellError
from ora.cell.operator import approval

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True)
def beside_the_lock(monkeypatch):
    """The broker's folder follows the test's private cell lock (tests/ora/cell/conftest.py)."""

    monkeypatch.delenv("ORA_CELL_APPROVALS", raising=False)


def _answer_when_asked(*, approved: bool, reason: str | None = None, by: str = "test") -> threading.Thread:
    def run() -> None:
        for _ in range(200):
            waiting = approval.pending()
            if waiting:
                approval.answer(waiting[0]["id"], approved, reason, by=by)
                return
            time.sleep(0.01)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_the_folder_is_beside_the_cell_lock_unless_named(cell_lock_path, monkeypatch, tmp_path) -> None:
    assert approval.directory() == cell_lock_path.parent / "ora-cell-approvals"
    monkeypatch.setenv("ORA_CELL_APPROVALS", str(tmp_path / "elsewhere"))
    assert approval.directory() == tmp_path / "elsewhere"


def test_an_approval_reaches_the_asker_and_leaves_its_record() -> None:
    thread = _answer_when_asked(approved=True, by="board")
    assert approval.ask("motion", "左臂动作等待批准", "左臂移动", {"arms": ["left"]}, timeout_s=5.0) == (True, None)
    thread.join()

    folder = approval.directory()
    assert approval.pending() == [] and list(folder.glob("*.request.json")) == []
    [record] = [json.loads(path.read_text()) for path in folder.glob("*.answer.json")]
    assert (record["schema"], record["approved"], record["by"]) == (approval.ANSWER_SCHEMA, True, "board")
    assert list(folder.glob(".*.tmp")) == []


def test_a_decline_carries_its_reason() -> None:
    thread = _answer_when_asked(approved=False, reason="  the cup is in the way  ")
    assert approval.ask("motion", "t", "x", timeout_s=5.0) == (False, "the cup is in the way")
    thread.join()

    thread = _answer_when_asked(approved=False, by="ora cell")
    assert approval.ask("motion", "t", "x", timeout_s=5.0) == (False, "declined (ora cell)")
    thread.join()


def test_the_request_names_what_is_asked_and_who_asks(monkeypatch) -> None:
    seen = []

    def look() -> str | None:
        seen.extend(approval.pending())
        return "seen"

    approval.ask("enable", "要给两条手臂通电吗？", "通电后锁住姿态", {"q": np.array([0.5, -0.25]), "far": np.inf,
                                                         "pair": (1, np.float32(2.5))},
                 timeout_s=5.0, asker="ORA worker yam", cancel=look)
    [request] = seen[:1]
    assert request["schema"] == approval.REQUEST_SCHEMA and len(request["id"]) == 8
    assert (request["kind"], request["title"], request["asker"], request["pid"]) == (
        "enable", "要给两条手臂通电吗？", "ORA worker yam", os.getpid())
    assert request["details"] == {"q": [0.5, -0.25], "far": None, "pair": [1, 2.5]}, "plain JSON for every reader"
    assert request["expires_unix_s"] - request["created_unix_s"] == pytest.approx(5.0)


def test_an_answer_for_another_id_does_nothing() -> None:
    answered = []

    def wrong_then_right() -> None:
        for _ in range(200):
            waiting = approval.pending()
            if waiting:
                with pytest.raises(CellError, match="no question deadbeef is waiting"):
                    approval.answer("deadbeef", True)
                with pytest.raises(CellError, match="is not a request id"):
                    approval.answer("../../etc", True)
                assert [r["id"] for r in approval.pending()] == [waiting[0]["id"]], "still waiting"
                answered.append(approval.answer(waiting[0]["id"].upper(), False, "not this one"))
                return
            time.sleep(0.01)

    thread = threading.Thread(target=wrong_then_right, daemon=True)
    thread.start()
    assert approval.ask("motion", "右臂动作", "x", timeout_s=5.0) == (False, "not this one")
    thread.join()
    assert answered and answered[0].startswith("declined ") and answered[0].endswith(": 右臂动作")


def test_the_first_answer_wins() -> None:
    results = []

    def twice() -> None:
        for _ in range(200):
            waiting = approval.pending()
            if waiting:
                results.append(approval.answer(waiting[0]["id"], False, "stop"))
                try:
                    approval.answer(waiting[0]["id"], True)
                except CellError as exc:
                    results.append(str(exc))
                return
            time.sleep(0.01)

    thread = threading.Thread(target=twice, daemon=True)
    thread.start()
    assert approval.ask("motion", "t", "x", timeout_s=5.0) == (False, "stop")
    thread.join()
    assert "was already answered: declined (operator)" in results[1]


def test_no_answer_in_time_declines_and_a_late_answer_is_refused() -> None:
    started = time.monotonic()
    assert approval.ask("release", "t", "x", timeout_s=0.3) == (False, "no answer in 0.3 s")
    assert 0.3 <= time.monotonic() - started < 2.0
    assert approval.pending() == []

    [record] = [json.loads(p.read_text()) for p in approval.directory().glob("*.answer.json")]
    assert (record["approved"], record["by"]) == (False, "timeout")
    with pytest.raises(CellError, match="already answered: declined \\(timeout\\)"):
        approval.answer(record["id"], True)


def test_a_withdrawn_question_is_declined_whatever_was_answered() -> None:
    calls = []

    def withdraw() -> str | None:
        calls.append(1)
        if len(calls) == 3:
            approval.answer(approval.pending()[0]["id"], True)   # an approval that arrives as the plan goes stale
            return "the arms went home: the plan no longer starts where they are"
        return None

    assert approval.ask("motion", "t", "x", timeout_s=5.0, cancel=withdraw) == (
        False, "the arms went home: the plan no longer starts where they are")


def test_a_question_whose_asker_died_is_ignored_and_swept() -> None:
    folder = approval.directory()
    folder.mkdir(parents=True)
    stale = {"schema": approval.REQUEST_SCHEMA, "id": "0badf00d", "kind": "motion", "title": "old", "text": "",
             "details": {}, "created_unix_s": time.time(), "expires_unix_s": time.time() + 600, "timeout_s": 600,
             "pid": _dead_pid(), "asker": "a worker that crashed"}
    (folder / "0badf00d.request.json").write_text(json.dumps(stale))

    assert approval.pending() == []
    with pytest.raises(CellError, match="no question 0badf00d is waiting"):
        approval.answer("0badf00d", True)
    assert not (folder / "0badf00d.answer.json").exists()

    expired = {**stale, "id": "00c0ffee", "pid": os.getpid(), "expires_unix_s": time.time() - 1}
    (folder / "00c0ffee.request.json").write_text(json.dumps(expired))
    assert approval.pending() == []

    approval.ask("motion", "t", "x", timeout_s=0.05)
    assert not (folder / "0badf00d.request.json").exists(), "the next question sweeps the dead asker's"


def test_a_question_that_cannot_be_posted_is_declined(monkeypatch, tmp_path) -> None:
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("")
    monkeypatch.setenv("ORA_CELL_APPROVALS", str(blocker))
    approved, reason = approval.ask("enable", "t", "x", timeout_s=5.0)
    assert approved is False and reason.startswith("the question could not be posted")


def test_a_question_from_another_process_is_answered_here(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "shared"
    monkeypatch.setenv("ORA_CELL_APPROVALS", str(folder))
    script = ("import json; from ora.cell.operator import approval; "
              "print(json.dumps(approval.ask('motion', 'from a worker', 'x', {'arms': ['left']}, timeout_s=20.0)))")
    asker = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, cwd=ROOT,
                             env={**os.environ, "ORA_CELL_APPROVALS": str(folder), "PYTHONPATH": str(ROOT)})
    try:
        for _ in range(500):
            waiting = approval.pending()
            if waiting:
                break
            time.sleep(0.02)
        assert [(r["title"], r["pid"]) for r in waiting] == [("from a worker", asker.pid)]
        approval.answer(waiting[0]["id"], True, by="test")
        out, _ = asker.communicate(timeout=20)
    finally:
        asker.kill()
    assert json.loads(out) == [True, None]


# ---------------------------------------------------------------- auto-approve (YOLO)
def _answers() -> list[dict]:
    return [json.loads(path.read_text()) for path in approval.directory().glob("*.answer.json")]


def test_a_run_started_with_auto_approve_answers_its_own_motions_and_records_each() -> None:
    state = approval.start_run(auto_kinds=("motion",), by="ora play --auto-approve", run="outputs/ora-play/r1")
    assert state["on"] is True and state["kinds"] == ["motion"] and state["pid"] == os.getpid()
    flag = json.loads((approval.directory() / approval.AUTO_FILE).read_text())
    assert (flag["schema"], flag["kinds"], flag["by"], flag["pid"], flag["run"]) == (
        approval.AUTO_SCHEMA, ["motion"], "ora play --auto-approve", os.getpid(), "outputs/ora-play/r1")
    assert isinstance(flag["since_unix_s"], float)

    started = time.monotonic()
    assert approval.ask("motion", "左臂动作等待批准", "左臂移动", {"arms": ["left"]}, timeout_s=30.0) == (True, None)
    assert time.monotonic() - started < 1.0, "nobody is asked"
    assert approval.pending() == [] and list(approval.directory().glob("*.request.json")) == []
    [record] = _answers()
    assert (record["schema"], record["approved"], record["reason"], record["by"], record["kind"], record["title"]) == (
        approval.ANSWER_SCHEMA, True, None, "auto-approve (ora play --auto-approve)", "motion", "左臂动作等待批准")


def test_enable_and_release_are_always_asked() -> None:
    approval.start_run(auto_kinds=("motion",), by="ora play --auto-approve")
    assert approval.ask("enable", "要给两条手臂通电吗？", "x", timeout_s=0.2) == (False, "no answer in 0.2 s")
    assert approval.ask("release", "手臂没能回到起始位", "x", timeout_s=0.2) == (False, "no answer in 0.2 s")
    for kinds in (("enable",), ("release",), ("motion", "enable"), ("anything",)):
        with pytest.raises(CellError, match="only motion questions can be auto-approved"):
            approval.start_run(auto_kinds=kinds, by="x")


def test_a_withdrawn_motion_is_declined_even_under_auto_approve() -> None:
    approval.start_run(auto_kinds=("motion",), by="ora play --auto-approve")
    assert approval.ask("motion", "t", "x", timeout_s=5.0, cancel=lambda: "the arms went home") == (
        False, "the arms went home")


def test_a_worker_start_clears_a_switch_it_was_not_started_with() -> None:
    approval.start_run(auto_kinds=("motion",), by="ora play --auto-approve")
    state = approval.start_run(auto_kinds=(), by="")
    assert state["on"] is False and not (approval.directory() / approval.AUTO_FILE).exists()
    assert approval.ask("motion", "t", "x", timeout_s=0.2) == (False, "no answer in 0.2 s")


def test_a_switch_set_for_another_process_never_answers_here() -> None:
    folder = approval.directory()
    folder.mkdir(parents=True)
    dead = _dead_pid()
    flag = {"schema": approval.AUTO_SCHEMA, "kinds": ["motion"], "by": "ora cell", "since_unix_s": time.time(),
            "pid": dead, "run": "ORA worker yam rig_mode on (pid %d)" % dead}
    (folder / approval.AUTO_FILE).write_text(json.dumps(flag))
    assert approval.ask("motion", "t", "x", timeout_s=0.2) == (False, "no answer in 0.2 s")
    state = approval.auto_approve_state()
    assert state["on"] is False and state["pid"] == dead and "ended" in state["stale"]

    alive_elsewhere = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (folder / approval.AUTO_FILE).write_text(json.dumps({**flag, "pid": alive_elsewhere.pid}))
        assert approval.auto_approve_state()["on"] is True
        assert approval.ask("motion", "t", "x", timeout_s=0.2) == (False, "no answer in 0.2 s"), "not this asker's"
    finally:
        alive_elsewhere.kill()


def test_the_state_of_no_switch_and_of_a_broken_one() -> None:
    assert approval.auto_approve_state() == {"on": False, "kinds": [], "by": None, "since_unix_s": None, "pid": None,
                                             "run": None, "stale": None}
    folder = approval.directory()
    folder.mkdir(parents=True)
    (folder / approval.AUTO_FILE).write_text("{not json")
    assert approval.auto_approve_state()["on"] is False
    (folder / approval.AUTO_FILE).write_text(json.dumps({"schema": approval.AUTO_SCHEMA, "kinds": ["enable"],
                                                         "pid": os.getpid()}))
    assert approval.auto_approve_state()["on"] is False
    assert approval.ask("enable", "t", "x", timeout_s=0.2) == (False, "no answer in 0.2 s")


def test_the_operator_switches_the_running_worker_on_and_off() -> None:
    from ora.cell.lock import CellLock

    with pytest.raises(CellError, match="no play worker drives the arms"):
        approval.switch_auto_approve(True, by="ora cell")
    with CellLock("ora cell board"):
        with pytest.raises(CellError, match="no play worker drives the arms"):
            approval.switch_auto_approve(True, by="ora cell")
    with CellLock("ORA worker yam rig_mode read [homes on SIGUSR1]"):
        with pytest.raises(CellError, match="no play worker drives the arms"):
            approval.switch_auto_approve(True, by="ora cell")
    with CellLock("ORA worker yam rig_mode on [homes on SIGUSR1]"):
        state = approval.switch_auto_approve(True, by="play terminal")
    assert (state["on"], state["kinds"], state["by"], state["pid"]) == (True, ["motion"], "play terminal", os.getpid())
    assert state["run"] == f"ORA worker yam rig_mode on [homes on SIGUSR1] (pid {os.getpid()})"
    assert approval.ask("motion", "t", "x", timeout_s=5.0) == (True, None)
    assert _answers()[0]["by"] == "auto-approve (play terminal)"

    state = approval.switch_auto_approve(False, by="ora cell")
    assert state["on"] is False and not (approval.directory() / approval.AUTO_FILE).exists()
    assert approval.switch_auto_approve(False, by="ora cell")["on"] is False, "off twice is fine"
    assert approval.ask("motion", "t", "x", timeout_s=0.2) == (False, "no answer in 0.2 s")


def test_auto_approve_in_a_worker_process_from_its_own_start(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "shared"
    monkeypatch.setenv("ORA_CELL_APPROVALS", str(folder))
    script = ("import json; from ora.cell.operator import approval; "
              "approval.start_run(auto_kinds=('motion',), by='ora play --auto-approve'); "
              "print(json.dumps([approval.ask('motion', 'm', 'x', timeout_s=20.0), "
              "approval.ask('enable', 'e', 'x', timeout_s=0.2)]))")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT, timeout=60,
                         env={**os.environ, "ORA_CELL_APPROVALS": str(folder), "PYTHONPATH": str(ROOT)})
    assert json.loads(out.stdout) == [[True, None], [False, "no answer in 0.2 s"]], out.stderr
    assert approval.auto_approve_state()["on"] is False, "its worker ended"


def _value_when_asked(value: str | None, *, approved: bool = True) -> threading.Thread:
    def run() -> None:
        for _ in range(200):
            waiting = approval.pending()
            if waiting:
                approval.answer(waiting[0]["id"], approved, None if approved else "不量了", by="test", value=value)
                return
            time.sleep(0.01)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_a_value_question_returns_the_value_and_keeps_it_with_the_answer() -> None:
    thread = _value_when_asked(" 20.1 ")
    assert approval.ask_value("一个格子的边长", "量 10 格", {"unit": "mm"}, timeout_s=5.0) == ("20.1", None)
    thread.join()
    [record] = [json.loads(p.read_text()) for p in approval.directory().glob("*.answer.json")]
    assert (record["approved"], record["value"]) == (True, "20.1")


def test_a_declined_value_question_returns_no_value() -> None:
    thread = _value_when_asked(None, approved=False)
    assert approval.ask_value("一个格子的边长", "量 10 格", timeout_s=5.0) == (None, "不量了")
    thread.join()


def test_a_value_is_needed_for_a_value_question_and_refused_for_any_other() -> None:
    for kind, value, refusal in (("value", None, "asks for a value"), ("task", "20", "takes no value")):
        result: list = []
        asking = threading.Thread(target=lambda: result.append(approval.ask(kind, "t", "x", timeout_s=5.0)), daemon=True)
        asking.start()
        for _ in range(200):
            if approval.pending():
                break
            time.sleep(0.01)
        [request] = approval.pending()
        with pytest.raises(CellError, match=refusal):
            approval.answer(request["id"], True, value=value)
        approval.answer(request["id"], False, "stop")
        asking.join()
        assert result == [(False, "stop")]
    with pytest.raises(ValueError, match="unknown approval kind"):
        approval.ask("calibrate", "t", "x", timeout_s=1.0)


SMALL = {"arms": ["left"], "duration_s": 2.5, "max_joint_travel_rad": {"left": 0.3},
         "min_table_distance_m": {"left": 0.08}, "min_inter_arm_distance_m": 0.4}


def test_the_mode_is_ask_until_chosen_and_safe_can_be_chosen_with_no_worker() -> None:
    assert approval.mode_state()["mode"] == "ask"
    state = approval.switch_mode("safe", by="setup")
    assert (state["mode"], state["by"], state["pid"]) == ("safe", "setup", None)
    assert state["safe_limits"] == approval.SAFE_LIMITS
    approval.start_run()                                  # a worker's start keeps the cell's mode
    assert approval.mode_state()["mode"] == "safe"
    with pytest.raises(CellError, match="no play worker drives the arms"):
        approval.switch_mode("auto")
    with pytest.raises(CellError, match="unknown permission mode"):
        approval.switch_mode("yolo")


def test_safe_mode_answers_a_small_motion_and_asks_anything_else() -> None:
    approval.switch_mode("safe", by="setup")
    assert approval.ask("motion", "左臂动作等待批准", "x", SMALL, timeout_s=30.0) == (True, None)
    [record] = _answers()
    assert record["by"].startswith("safe mode (2.5 s, 0.30 rad, 8 cm above the table, 40 cm between the arms)")

    for change in ({"max_joint_travel_rad": {"left": 0.9}}, {"duration_s": 14.0},
                   {"min_table_distance_m": {"left": 0.01}}, {"min_inter_arm_distance_m": 0.1},
                   {"duration_s": None}):
        plan = {**SMALL, **change}
        assert approval.ask("motion", "左臂动作等待批准", "x", plan, timeout_s=0.2) == (False, "no answer in 0.2 s")
    assert approval.ask("enable", "要给两条手臂通电吗？", "x", SMALL, timeout_s=0.2) == (False, "no answer in 0.2 s")
    approval.switch_mode("ask")
    assert approval.ask("motion", "左臂动作等待批准", "x", SMALL, timeout_s=0.2) == (False, "no answer in 0.2 s")


def test_a_stored_limit_may_only_be_stricter() -> None:
    approval.switch_mode("safe")
    path = approval.directory() / approval.MODE_FILE
    flag = json.loads(path.read_text())
    flag["safe_limits"] = {"max_joint_travel_rad": 3.0, "min_table_distance_m": 0.05}
    path.write_text(json.dumps(flag))
    limits = approval.mode_state()["safe_limits"]
    assert limits["max_joint_travel_rad"] == approval.SAFE_LIMITS["max_joint_travel_rad"]
    assert limits["min_table_distance_m"] == 0.05


def test_auto_is_the_running_workers_and_falls_back_to_the_cells_mode() -> None:
    from ora.cell.lock import CellLock

    approval.switch_mode("safe")
    with CellLock("ORA worker yam rig_mode on [homes on SIGUSR1]"):
        state = approval.switch_mode("auto", by="board")
        assert (state["mode"], state["pid"], state["falls_back_to"]) == ("auto", os.getpid(), "safe")
        assert approval.switch_mode("ask")["mode"] == "ask"
        assert approval.auto_approve_state()["on"] is False


def test_stop_here_reaches_only_the_holder_and_only_while_fresh(monkeypatch) -> None:
    from ora.cell.lock import CellLock

    with pytest.raises(CellError, match="nothing is moving"):
        approval.request_stop_motion()
    with CellLock("ORA worker yam rig_mode on [homes on SIGUSR1]"):
        record = approval.request_stop_motion(by="board")
    assert record["pid"] == os.getpid() and record["by"] == "board"
    assert approval.take_stop_request(pid=os.getpid() + 1) is None      # another process's stop
    assert approval.take_stop_request()["by"] == "board"
    assert approval.take_stop_request() is None                          # consumed once
    with CellLock("ORA worker yam rig_mode on [homes on SIGUSR1]"):
        approval.request_stop_motion()
    monkeypatch.setattr(approval.time, "time", lambda: 1e12)
    assert approval.take_stop_request() is None                          # stale
