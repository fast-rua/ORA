"""``ora cell``: with ``--json`` every outcome is one object, ``{"ok": true, "result": ...}`` with exit 0 or
``{"ok": false, "error": ...}`` with exit 1 for a refusal; a running board takes the verbs; ``init --example``
writes a config that loads. The cell is the fake one of ``tests/support/cell``; nothing re-executes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ora.app.main import main
from ora.cell import config as cell_config
from ora.cell.errors import CellError
from tests.support.cell.fake import FakeAdapter, make_cell, redirect_lock


@pytest.fixture(autouse=True)
def private_cell(tmp_path, monkeypatch):
    """No re-execution in a robot interpreter, a private cell lock, and no board on this machine."""

    from ora.app import cell as command

    monkeypatch.setenv("ORA_CELL_REEXEC", "1")
    redirect_lock(monkeypatch, tmp_path / "ora-cell.lock")
    monkeypatch.setattr(command, "_board", lambda port, cell_id: None)


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    """The adapter of the fake cell ``Cell.load`` returns."""

    from ora.cell.operator.context import Cell

    fake = FakeAdapter()
    cell = make_cell(tmp_path, fake)
    monkeypatch.setattr(Cell, "load", lambda path=None: cell)
    return fake


def _json(capsys) -> dict:
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1, out
    return json.loads(out[0])


def test_a_cell_that_does_not_load_is_refused_as_json(monkeypatch, capsys) -> None:
    from ora.cell.operator.context import Cell

    def refuse(path=None):
        raise CellError("no cell config at /nowhere/cell.yaml")

    monkeypatch.setattr(Cell, "load", refuse)

    assert main(["cell", "diagnose", "--json"]) == 1
    assert _json(capsys) == {"ok": False, "error": "no cell config at /nowhere/cell.yaml"}
    assert main(["cell", "diagnose"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.strip() == "refused: no cell config at /nowhere/cell.yaml"


def test_a_verb_prints_its_result(adapter, capsys) -> None:
    assert main(["cell", "estop", "--json", "--yes"]) == 0
    assert _json(capsys) == {"ok": True, "result": "every motor disabled"}
    assert adapter.disabled == 2

    assert main(["cell", "stop"]) == 0
    assert capsys.readouterr().out.strip() == "no program holds the cell"


def test_numpy_results_print_as_lists(adapter, monkeypatch, capsys) -> None:
    from ora.cell.operator import actions

    monkeypatch.setattr(actions, "diagnose", lambda cell: {"arms": {"left": {"q": np.array([0.5, -0.25])}}})

    assert main(["cell", "diagnose", "--json"]) == 0
    assert _json(capsys) == {"ok": True, "result": {"arms": {"left": {"q": [0.5, -0.25]}}}}


def test_an_unconfirmed_verb_does_nothing(adapter, monkeypatch, capsys) -> None:
    from ora.cell.operator import actions

    questions = []
    monkeypatch.setattr(actions, "ask_terminal", lambda question: questions.append(question) and False)

    assert main(["cell", "release", "--json"]) == 1
    assert _json(capsys) == {"ok": False, "error": "not confirmed: nothing done"}
    assert adapter.disabled == 0 and len(questions) == 1


@pytest.mark.parametrize("verb", ["home", "hold", "float"])
def test_take_over_verbs_without_a_board_start_one_and_go_through_it(adapter, monkeypatch, capsys, verb) -> None:
    # A command that took the arms itself would let them fall when it exits on a blocked
    # path; the board it starts holds them for as long as needed.
    from ora.app import cell as command

    started, sent = [], []
    monkeypatch.setattr(command, "_board", lambda port, cell_id: None)
    monkeypatch.setattr(command, "_start_board", lambda port, cell: started.append(port) or f"http://127.0.0.1:{port}")
    monkeypatch.setattr(command, "_via_board", lambda url, action: sent.append((url, action)) or "done")
    assert main(["cell", verb, "--json", "--yes", "--board-port", "9002"]) == 0
    assert _json(capsys) == {"ok": True, "result": "done"}
    assert started == [9002] and sent == [("http://127.0.0.1:9002", verb)]
    assert adapter.held == []


def test_a_running_board_takes_every_verb(adapter, monkeypatch, capsys) -> None:
    from ora.app import cell as command

    sent = []
    monkeypatch.setattr(command, "_board", lambda port, cell_id: f"http://127.0.0.1:{port}")
    monkeypatch.setattr(command, "_via_board", lambda url, verb: sent.append((url, verb)) or "switched to hold")

    assert main(["cell", "hold", "--json", "--board-port", "9001"]) == 0
    assert _json(capsys) == {"ok": True, "result": "switched to hold"}
    assert sent == [("http://127.0.0.1:9001", "hold")] and adapter.held == []


def test_a_board_refusal_is_refused_as_json(adapter, monkeypatch, capsys) -> None:
    from ora.app import cell as command

    def refuse(url, verb):
        raise CellError("the cell is held by this board: use home or release instead")

    monkeypatch.setattr(command, "_board", lambda port, cell_id: "http://127.0.0.1:8765")
    monkeypatch.setattr(command, "_via_board", refuse)

    assert main(["cell", "kill", "--json", "--yes"]) == 1
    assert _json(capsys)["error"].startswith("the cell is held by this board")


def test_init_example_writes_a_config_that_loads(tmp_path, capsys) -> None:
    path = tmp_path / "configs" / "cell.yaml"

    assert main(["cell", "init", "--example", str(path)]) == 0

    assert str(path) in capsys.readouterr().out
    loaded = cell_config.load(path)
    assert (loaded.cell_id, loaded.environment) == ("my-cell-01", "yam")
    assert set(loaded.buses) == {"left", "right"}
    assert {role: camera.mount for role, camera in loaded.cameras.items()} == {
        "overhead": "fixed_world", "front": "fixed_world", "wrist_left": "end_effector", "wrist_right": "end_effector"}


# ------------------------------------------------------------------ the worker's questions
@pytest.fixture
def asking(monkeypatch):
    """A question waiting in the private approval folder, as a worker would post it; never a board or a cell."""

    import threading
    import time

    from ora.app import cell as command
    from ora.cell.operator import approval
    from ora.cell.operator.context import Cell

    monkeypatch.delenv("ORA_CELL_APPROVALS", raising=False)
    monkeypatch.setattr(Cell, "load", lambda path=None: pytest.fail("answering needs no cell"))
    monkeypatch.setattr(command, "_board", lambda port, cell_id: pytest.fail("answering goes through no board"))
    monkeypatch.setattr(command, "_reexec_in_robot_interpreter", lambda args: pytest.fail("no re-execution"))
    result: list = []
    thread = threading.Thread(target=lambda: result.append(
        approval.ask("motion", "左臂动作等待批准", "左臂移动，约 1.5 秒", {"arms": ["left"]}, timeout_s=10.0)), daemon=True)
    thread.start()
    for _ in range(500):
        if approval.pending():
            break
        time.sleep(0.01)
    [request] = approval.pending()
    yield request, thread, result
    thread.join(timeout=12)


def test_pending_lists_the_questions(asking, capsys) -> None:
    request, _, _ = asking
    assert main(["cell", "pending", "--json", "--cell", "/nowhere.yaml"]) == 0
    [listed] = _json(capsys)["result"]
    assert (listed["id"], listed["kind"], listed["title"]) == (request["id"], "motion", "左臂动作等待批准")

    assert main(["cell", "pending"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"[{request['id']}] motion · 左臂动作等待批准 · ") and "  左臂移动，约 1.5 秒" in out
    assert main(["cell", "decline", "--id", request["id"], "--json"]) == 0
    capsys.readouterr()


def test_approve_and_decline_answer_by_id(asking, capsys) -> None:
    request, thread, result = asking
    assert main(["cell", "approve", "--id", "deadbeef", "--json"]) == 1
    assert "no question deadbeef is waiting" in _json(capsys)["error"]
    assert main(["cell", "approve", "--json"]) == 1
    assert _json(capsys)["error"].startswith("--id is required here")

    assert main(["cell", "decline", "--id", request["id"], "--reason", "杯子挡着", "--by", "play terminal", "--json"]) == 0
    assert _json(capsys) == {"ok": True, "result": f"declined {request['id']}: 左臂动作等待批准"}
    thread.join(timeout=5)
    assert result == [(False, "杯子挡着")]
    assert main(["cell", "approve", "--id", request["id"], "--json"]) == 1
    assert "already answered: declined (play terminal)" in _json(capsys)["error"]


def test_on_a_terminal_the_one_waiting_question_is_shown_and_answered_after_a_yes(asking, monkeypatch, capsys) -> None:
    from ora.app import cell as command
    from ora.cell.operator import actions

    request, thread, result = asking
    questions = []
    monkeypatch.setattr(command, "_interactive", lambda: True)
    monkeypatch.setattr(actions, "ask_terminal", lambda question: questions.append(question) and False)
    assert main(["cell", "approve"]) == 1
    assert questions == [f"Approve {request['id']}?"]
    captured = capsys.readouterr()
    assert "左臂移动" in captured.out and captured.err.strip() == "refused: not answered: nothing sent"

    monkeypatch.setattr(actions, "ask_terminal", lambda question: True)
    assert main(["cell", "approve"]) == 0
    assert capsys.readouterr().out.strip().endswith(f"approved {request['id']}: 左臂动作等待批准")
    thread.join(timeout=5)
    assert result == [(True, None)]
    assert main(["cell", "approve"]) == 1
    assert capsys.readouterr().err.strip() == "refused: nothing is waiting for an answer"


# ------------------------------------------------------------------ auto-approve (YOLO)
def test_auto_approve_switches_the_running_worker_without_a_lock_a_board_or_a_cell(monkeypatch, capsys) -> None:
    import os

    from ora.app import cell as command
    from ora.cell.lock import CellLock
    from ora.cell.operator import approval
    from ora.cell.operator.context import Cell

    monkeypatch.delenv("ORA_CELL_APPROVALS", raising=False)
    monkeypatch.setattr(Cell, "load", lambda path=None: pytest.fail("the switch needs no cell"))
    monkeypatch.setattr(command, "_board", lambda port, cell_id: pytest.fail("the switch goes through no board"))
    monkeypatch.setattr(command, "_reexec_in_robot_interpreter", lambda args: pytest.fail("no re-execution"))

    assert main(["cell", "auto-approve", "--json"]) == 0
    assert _json(capsys)["result"]["on"] is False
    assert main(["cell", "auto-approve", "on", "--json"]) == 1
    assert "no play worker drives the arms" in _json(capsys)["error"]

    with CellLock("ORA worker yam rig_mode on [homes on SIGUSR1]"):
        assert main(["cell", "auto-approve", "on", "--by", "play terminal", "--json"]) == 0
    state = _json(capsys)["result"]
    assert (state["on"], state["kinds"], state["by"], state["pid"]) == (True, ["motion"], "play terminal", os.getpid())
    assert main(["cell", "auto-approve", "status"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"auto-approve is on: motion questions of pid {os.getpid()} answer themselves")
    assert "set by play terminal" in out and "enabling and releasing the arms are still asked" in out
    assert approval.ask("motion", "t", "x", timeout_s=5.0) == (True, None)

    assert main(["cell", "auto-approve", "off"]) == 0
    assert capsys.readouterr().out.strip() == "auto-approve is off: every question waits for an answer"
    assert approval.auto_approve_state()["on"] is False


def test_what_the_drivers_print_does_not_break_the_json(adapter, monkeypatch, capsys) -> None:
    def calibrate(arms, should_stop):
        print("initializing motorchain robot, gripper_limits: None")       # as i2rt does
        return {arm: [0.0, 1.0] for arm in arms}

    monkeypatch.setattr(adapter, "calibrate_grippers", calibrate)
    monkeypatch.setattr(adapter, "gripper_raw", {"left": 1.0, "right": 1.0})
    from ora.app import cell as command
    from ora.cell import occupancy

    monkeypatch.setattr(occupancy, "can_receivers", lambda channel: 0)
    monkeypatch.setattr(occupancy, "can_traffic", lambda channel, seconds=0.3: 0)

    monkeypatch.setattr(command, "_asker", lambda args, label: type("Yes", (), {
        "value": lambda self, *a: ("开", None)})())
    assert main(["cell", "grippers", "--json", "--yes"]) == 0
    captured = capsys.readouterr()
    assert "initializing motorchain robot" in captured.err
    result = json.loads(captured.out)["result"]
    assert result["gripper_limits"] == {"left": [0.0, 1.0], "right": [0.0, 1.0]}
    assert result["orientation"]["saved"] is False            # both read open, both said open


def test_a_long_step_takes_sigint_even_when_started_with_it_ignored(tmp_path) -> None:
    """A shell's background job inherits SIGINT ignored; ``ora cell setup`` is stoppable all the same."""

    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = tmp_path / "long_step.py"
    script.write_text(
        "import sys, time\n"
        "from ora.app import cell\n"
        "cell._take_interrupts()\n"
        "try:\n"
        "    print('ready', flush=True)\n"
        "    time.sleep(30)\n"
        "except KeyboardInterrupt:\n"
        "    print('stopped', flush=True)\n"
        "    sys.exit(3)\n")
    child = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True, cwd=root,
                             env={**os.environ, "PYTHONPATH": str(root)},
                             preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGINT)
        assert child.wait(timeout=10) == 3 and child.stdout.read().strip() == "stopped"
    finally:
        child.kill()
    assert {"calibrate", "setup", "grippers"} <= __import__("ora.app.cell", fromlist=["x"]).INTERRUPTIBLE


def test_with_json_the_step_says_when_a_stop_reaches_it(capsys, monkeypatch) -> None:
    import signal

    from ora.app import cell

    monkeypatch.setattr(signal, "signal", lambda *_: None)
    cell._take_interrupts()
    assert capsys.readouterr().err == ""
    cell._take_interrupts(say=True)
    assert [json.loads(line) for line in capsys.readouterr().err.splitlines()] == [{"interruptible": True}]
