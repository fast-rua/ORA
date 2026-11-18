"""`ora web`'s real-cell routes (ora/app/web_cell.py through the gateway): what the terminal's /approve,
/decline, /yolo, /home ... do, behind the gateway's cookie, loopback Host and same-origin checks. `ora cell`
never runs: a stand-in answers its command line. The parity guard reads the terminal's command table."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn, UnixStreamServer

import pytest
from werkzeug.test import Client

from ora.app import web_cell
from ora.app.web_cell import CellControls, outcome, real_cell
from ora.app.web_gateway import WebGateway, allowed
from tests.support.cell.fake import redirect_lock

TOKEN = "launch-token-0123456789abcdef"
HOST = {"Host": "127.0.0.1:43211"}
POST = {**HOST, "Origin": "http://127.0.0.1:43211"}
ROOT = Path(__file__).resolve().parents[3]


def plan(rig_mode: str, kind: str = "yam", cell_config: str | None = "/cells/bench.yaml") -> dict:
    runtime = {"kind": kind, "rig_mode": rig_mode, **({"cell_config": cell_config} if cell_config else {})}
    return {"runtime": runtime}


@pytest.fixture
def cell_files(tmp_path, monkeypatch):
    """The cell's files under the test's own directory: lock, approvals and alerts."""

    lock = redirect_lock(monkeypatch, tmp_path / "cell" / "ora-cell.lock")
    lock.parent.mkdir()
    monkeypatch.setenv("ORA_CELL_APPROVALS", str(tmp_path / "cell" / "approvals"))
    monkeypatch.setenv("ORA_CELL_ALERTS", str(tmp_path / "cell" / "alerts.json"))
    (tmp_path / "cell" / "approvals").mkdir()
    return tmp_path / "cell"


def make_run(tmp_path: Path, rig_mode: str = "on", **kwargs) -> Path:
    root = tmp_path / "run-1"
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(plan(rig_mode, **kwargs)))
    return root


class FakeOraCell:
    """`scripts/ora cell ...`: records each command line and answers as `ora cell --json` does."""

    def __init__(self, answer: tuple[int | None, str, str] = (0, '{"ok": true, "result": "done"}', "")) -> None:
        self.calls: list[tuple[list[str], Path]] = []
        self.answer = answer

    def __call__(self, args: list[str], cwd: Path) -> tuple[int | None, str, str]:
        self.calls.append((args, cwd))
        return self.answer


def signed_in(gateway: WebGateway) -> Client:
    client = Client(gateway)
    assert client.get(f"/?ora_token={TOKEN}", headers=HOST).status_code == 303
    return client


def gateway_with(run: Path, ora_cell: FakeOraCell, *, upstream: str | None = None, alive=None,
                 lock: Path | None = None) -> tuple[WebGateway, Client]:
    gateway = WebGateway(run_root=run, upstream=upstream, world_app=None, launch_token=TOKEN)
    gateway.cell = CellControls(run, run=ora_cell, note=gateway._cell_note, lock_path=lock,
                                **({"alive": alive} if alive else {}))
    return gateway, signed_in(gateway)


def ask(folder: Path, request_id: str = "0a1b2c3d", *, kind: str = "motion", seconds: float = 120.0,
        pid: int | None = None) -> dict:
    now = time.time()
    request = {"schema": "ora.cell-approval-request.v1", "id": request_id, "kind": kind,
               "title": "左臂动作：下降 4 cm", "text": "左臂从桌面上方 12 cm 下降到 8 cm。\n速度：慢。",
               "details": {"arms": ["left"]}, "created_unix_s": now, "expires_unix_s": now + seconds,
               "timeout_s": seconds, "pid": os.getpid() if pid is None else pid, "asker": "ORA YAM worker"}
    (folder / "approvals" / f"{request_id}.request.json").write_text(json.dumps(request))
    return request


# -- Which runs have them --------------------------------------------------------------------------

@pytest.mark.parametrize("mode,kind,expected", [
    ("on", "yam", True), ("read", "yam", True), ("sim", "yam", False), ("on", "robotwin", False),
])
def test_only_a_real_cell_run_has_the_cell_s_controls(mode, kind, expected):
    found = real_cell(plan(mode, kind))
    assert (found is not None) is expected
    if expected:
        assert found == {"kind": "yam", "rig_mode": mode, "cell_config": "/cells/bench.yaml"}
    assert real_cell({}) is None and real_cell({"runtime": "x"}) is None


def test_a_simulated_run_answers_that_it_drives_no_cell_and_runs_nothing(tmp_path, cell_files):
    ora_cell = FakeOraCell()
    _, client = gateway_with(make_run(tmp_path, "sim"), ora_cell)
    assert client.get("/ora/cell", headers=HOST).json == {"cell": None}
    response = client.post("/ora/cell/verb/home", json={}, headers=POST)
    assert response.status_code == 404 and "drives no real cell" in response.json["error"]
    assert ora_cell.calls == []


def test_the_routes_keep_the_gateway_s_cookie_host_and_origin_checks(tmp_path, cell_files):
    ora_cell = FakeOraCell()
    gateway, client = gateway_with(make_run(tmp_path), ora_cell)
    assert Client(gateway).get("/ora/cell", headers=HOST).status_code == 403, "no cookie, no cell"
    assert client.get("/ora/cell", headers={"Host": "attacker.example:43211"}).status_code == 403
    assert client.post("/ora/cell/verb/home", json={}, headers=HOST).status_code == 403, "no Origin"
    assert client.post("/ora/cell/verb/home", json={}, headers={
        **HOST, "Origin": "http://evil.example"}).status_code == 403
    assert ora_cell.calls == []
    assert client.get("/ora/cell/verb/home", headers=HOST).status_code == 405
    assert client.post("/ora/cell/verb/rm", json={}, headers=POST).status_code == 404
    assert client.post("/ora/cell/elsewhere", json={}, headers=POST).status_code == 404
    assert not allowed("POST", "/api/cell-note"), "the agent's note is the gateway's own, never a page's"
    assert ora_cell.calls == []


def test_a_replay_has_no_cell_routes(tmp_path, cell_files):
    gateway = WebGateway(run_root=make_run(tmp_path), upstream=None, world_app=None, launch_token=TOKEN, replay=True)
    response = signed_in(gateway).get("/ora/cell", headers=HOST)
    assert response.status_code == 404


# -- The state ------------------------------------------------------------------------------------

def test_the_state_is_the_cell_s_own_files(tmp_path, cell_files):
    request = ask(cell_files)
    ask(cell_files, "0badf00d", pid=2 ** 22 + 12345)              # a dead asker's question is stale
    (cell_files / "alerts.json").write_text(json.dumps({
        "schema": "ora.cell-alerts.v1", "source": "worker", "pid": os.getpid(), "alerts": [
            {"code": "motor_hot", "level": "warning", "text": "左臂肘电机 52°C"},
            {"code": "bogus", "level": "info", "text": "not an alert"}]}))
    (cell_files / "ora-cell.lock").write_text(f"ORA YAM worker rig_mode on (pid {os.getpid()})")
    _, client = gateway_with(make_run(tmp_path), FakeOraCell(), lock=cell_files / "ora-cell.lock")
    state = client.get("/ora/cell", headers=HOST).json
    assert state["cell"] == {"kind": "yam", "rig_mode": "on", "cell_config": "/cells/bench.yaml"}
    assert [item["id"] for item in state["pending"]] == [request["id"]]
    assert state["pending"][0]["text"] == request["text"]
    assert [alert["code"] for alert in state["alerts"]] == ["motor_hot"]
    assert state["auto_approve"]["on"] is False
    assert state["mode"]["mode"] == "ask" and state["mode"]["safe_limits"]["max_duration_s"] == 10.0
    assert state["self_answered"] == []
    assert state["worker"] == {"pid": os.getpid(), "alive": True} and state["worker_ended"] is None


def test_the_state_has_the_mode_and_the_motions_that_answered_themselves(tmp_path, cell_files):
    from ora.cell.operator import approval

    approval.switch_mode("safe", by="cell board")
    plan = {"duration_s": 2.5, "max_joint_travel_rad": 0.3, "min_table_distance_m": 0.08, "min_inter_arm_distance_m": 0.4}
    assert approval.ask("motion", "左臂动作：下降 4 cm", "下降", plan, timeout_s=5.0) == (True, None)
    approved = cell_files / "approvals" / "0a1b2c3d.answer.json"          # a person's answer is not listed
    approved.write_text(json.dumps({"id": "0a1b2c3d", "approved": True, "by": "web page", "kind": "motion",
                                    "answered_unix_s": time.time()}))
    _, client = gateway_with(make_run(tmp_path), FakeOraCell())
    state = client.get("/ora/cell", headers=HOST).json
    assert (state["mode"]["mode"], state["mode"]["by"]) == ("safe", "cell board")
    assert [(item["kind"], item["title"], item["by"]) for item in state["self_answered"]] == [
        ("motion", "左臂动作：下降 4 cm", "safe mode (2.5 s, 0.30 rad, 8 cm above the table, 40 cm between the arms)")]


def test_the_state_times_each_motion_question_from_when_it_was_seen_waiting_to_its_answer(tmp_path, cell_files):
    from ora.cell.operator import approval

    request = ask(cell_files)
    _, client = gateway_with(make_run(tmp_path), FakeOraCell())
    assert client.get("/ora/cell", headers=HOST).json["answers"] == []
    approval.answer(request["id"], True, by="play terminal")
    (cell_files / "approvals" / f"{request['id']}.request.json").unlink()     # the asker removes it once decided
    approval.switch_mode("safe", by="cell board")
    plan = {"duration_s": 2.5, "max_joint_travel_rad": 0.3, "min_table_distance_m": 0.08, "min_inter_arm_distance_m": 0.4}
    assert approval.ask("motion", "左臂动作：下降 4 cm", "下降", plan, timeout_s=5.0) == (True, None)
    (cell_files / "approvals" / "0badf00d.answer.json").write_text(json.dumps({    # never seen waiting
        "id": "0badf00d", "approved": False, "by": "web page", "answered_unix_s": time.time()}))
    (cell_files / "approvals" / "0e0e0e0e.answer.json").write_text(json.dumps({    # not a motion
        "id": "0e0e0e0e", "approved": True, "by": "auto-approve (x)", "kind": "enable", "answered_unix_s": time.time()}))
    answers = client.get("/ora/cell", headers=HOST).json["answers"]
    person, safe, unseen = answers
    assert (person["id"], person["kind"], person["by"], person["approved"]) == (request["id"], "motion", "play terminal", True)
    assert person["created_unix_s"] == pytest.approx(request["created_unix_s"])
    assert person["answered_unix_s"] >= person["created_unix_s"]
    assert safe["by"].startswith("safe mode (") and safe["created_unix_s"] == safe["answered_unix_s"]
    assert (unseen["id"], unseen["kind"], unseen["created_unix_s"], unseen["approved"]) == ("0badf00d", None, None, False)


def test_the_end_of_the_worker_that_held_the_arms_is_said_until_another_holds_them(tmp_path, cell_files):
    lock = cell_files / "ora-cell.lock"
    living = {4242: True, 5151: True}
    _, client = gateway_with(make_run(tmp_path), FakeOraCell(), alive=lambda pid: living.get(pid, False), lock=lock)
    assert client.get("/ora/cell", headers=HOST).json["worker_ended"] is None, "never seen: nothing to say"
    lock.write_text("ORA YAM worker rig_mode on (pid 4242)")
    assert client.get("/ora/cell", headers=HOST).json["worker"] == {"pid": 4242, "alive": True}
    living[4242] = False
    for _ in range(2):
        state = client.get("/ora/cell", headers=HOST).json
        assert state["worker_ended"]["pid"] == 4242 and state["worker"]["alive"] is False
    lock.write_text("ORA YAM worker rig_mode on (pid 5151)")
    assert client.get("/ora/cell", headers=HOST).json["worker_ended"] is None


# -- The actions ----------------------------------------------------------------------------------

def test_home_runs_at_once_and_estop_only_after_the_page_s_confirmation(tmp_path, cell_files):
    ora_cell = FakeOraCell()
    _, client = gateway_with(make_run(tmp_path), ora_cell)
    script = str(web_cell.paths.PROJECT_ROOT / "scripts" / "ora")
    result = client.post("/ora/cell/verb/home", json={}, headers=POST).json
    assert ora_cell.calls[-1] == ([script, "cell", "home", "--json", "--cell", "/cells/bench.yaml"],
                                  web_cell.paths.PROJECT_ROOT)
    assert (result["name"], result["verb"], result["ok"], result["text"], result["via"]) == (
        "home", "home", True, "done", "web page")
    for verb in ("float", "release", "estop"):
        refused = client.post(f"/ora/cell/verb/{verb}", json={}, headers=POST)
        assert refused.status_code == 400 and "asked first" in refused.json["error"]
        assert client.post(f"/ora/cell/verb/{verb}", json={"confirmed": True}, headers=POST).status_code == 200
        assert ora_cell.calls[-1][0] == [script, "cell", verb, "--json", "--cell", "/cells/bench.yaml", "--yes"]
    assert len(ora_cell.calls) == 4


def test_an_answer_names_the_question_s_id_and_carries_the_reason(tmp_path, cell_files):
    ora_cell = FakeOraCell((0, '{"ok": true, "result": "declined 0a1b2c3d: 左臂动作"}', ""))
    _, client = gateway_with(make_run(tmp_path, cell_config=None), ora_cell)
    assert client.post("/ora/cell/approval/0a1b2c3d/decline", json={"reason": "too close"},
                       headers=POST).status_code == 400, "an answer is confirmed first"
    assert client.post("/ora/cell/approval/not-an-id/approve", json={"confirmed": True}, headers=POST).status_code == 400
    assert client.post("/ora/cell/approval/0a1b2c3d/maybe", json={"confirmed": True}, headers=POST).status_code == 404
    result = client.post("/ora/cell/approval/0a1b2c3d/decline", json={"confirmed": True, "reason": " too close "},
                         headers=POST).json
    assert ora_cell.calls[-1][0][1:] == ["cell", "decline", "--json", "--id", "0a1b2c3d", "--by", "web page",
                                         "--reason", "too close"]
    assert result["ok"] is True and result["request_id"] == "0a1b2c3d"
    client.post("/ora/cell/approval/0a1b2c3d/approve", json={"confirmed": True, "reason": "ignored"}, headers=POST)
    assert ora_cell.calls[-1][0][1:] == ["cell", "approve", "--json", "--id", "0a1b2c3d", "--by", "web page"]

    # A value question's answer goes with an approve only, as `--value`; it is text or refused.
    client.post("/ora/cell/approval/0a1b2c3d/approve", json={"confirmed": True, "value": " 20 "}, headers=POST)
    assert ora_cell.calls[-1][0][1:] == ["cell", "approve", "--json", "--id", "0a1b2c3d", "--by", "web page",
                                         "--value", "20"]
    client.post("/ora/cell/approval/0a1b2c3d/decline", json={"confirmed": True, "value": "20"}, headers=POST)
    assert "--value" not in ora_cell.calls[-1][0]
    calls = len(ora_cell.calls)
    refused = client.post("/ora/cell/approval/0a1b2c3d/approve", json={"confirmed": True, "value": 20}, headers=POST)
    assert refused.status_code == 400 and "value must be text" in refused.json["error"]
    assert len(ora_cell.calls) == calls


def test_auto_approve_on_is_confirmed_first_and_off_is_not(tmp_path, cell_files):
    state = {"on": True, "kinds": ["motion"], "pid": 4242, "by": "web page"}
    ora_cell = FakeOraCell((0, json.dumps({"ok": True, "result": state}), ""))
    _, client = gateway_with(make_run(tmp_path), ora_cell)
    assert client.post("/ora/cell/auto-approve/on", json={}, headers=POST).status_code == 400
    result = client.post("/ora/cell/auto-approve/on", json={"confirmed": True}, headers=POST).json
    assert ora_cell.calls[-1][0][1:] == ["cell", "auto-approve", "on", "--json", "--cell", "/cells/bench.yaml",
                                         "--by", "web page"]
    assert (result["name"], result["verb"], json.loads(result["text"])) == ("yolo", "auto-approve", state)
    assert client.post("/ora/cell/auto-approve/off", json={}, headers=POST).status_code == 200
    assert ora_cell.calls[-1][0][3] == "off"
    assert client.post("/ora/cell/auto-approve/maybe", json={"confirmed": True}, headers=POST).status_code == 404


def test_the_mode_is_switched_with_the_terminal_s_command_and_auto_is_confirmed_first(tmp_path, cell_files):
    state = {"mode": "safe", "by": "web page", "safe_limits": {"max_duration_s": 10.0}}
    ora_cell = FakeOraCell((0, json.dumps({"ok": True, "result": state}), ""))
    _, client = gateway_with(make_run(tmp_path), ora_cell)
    result = client.post("/ora/cell/mode/safe", json={}, headers=POST).json
    assert ora_cell.calls[-1][0][1:] == ["cell", "mode", "safe", "--json", "--cell", "/cells/bench.yaml", "--by", "web page"]
    assert (result["name"], result["verb"], result["ok"], json.loads(result["text"])) == ("mode", "mode", True, state)
    calls = len(ora_cell.calls)
    refused = client.post("/ora/cell/mode/auto", json={}, headers=POST)
    assert refused.status_code == 400 and "/mode auto is asked first" in refused.json["error"]
    assert len(ora_cell.calls) == calls
    assert client.post("/ora/cell/mode/auto", json={"confirmed": True}, headers=POST).status_code == 200
    assert ora_cell.calls[-1][0][3] == "auto"
    assert client.post("/ora/cell/mode/status", json={}, headers=POST).status_code == 404
    assert client.get("/ora/cell/mode/ask", headers=HOST).status_code == 405


def test_halt_stops_the_running_motion_at_once_without_a_confirmation(tmp_path, cell_files):
    record = {"schema": "ora.cell-stop-motion.v1", "pid": 4242, "by": "web page"}
    ora_cell = FakeOraCell((0, json.dumps({"ok": True, "result": record}), ""))
    _, client = gateway_with(make_run(tmp_path), ora_cell)
    result = client.post("/ora/cell/stop-motion", json={}, headers=POST).json
    assert ora_cell.calls[-1][0][1:] == ["cell", "stop-motion", "--json", "--cell", "/cells/bench.yaml", "--by", "web page"]
    assert (result["name"], result["verb"], result["ok"], result["via"]) == ("halt", "stop-motion", True, "web page")
    assert client.get("/ora/cell/stop-motion", headers=HOST).status_code == 405
    # A stop sent while the agent planned is taken back when no motion came (the reply was words).
    result = client.post("/ora/cell/stop-motion", json={"withdraw": True}, headers=POST).json
    assert ora_cell.calls[-1][0][1:] == ["cell", "stop-motion", "--withdraw", "--json", "--cell", "/cells/bench.yaml"]
    assert (result["name"], result["verb"], result["withdraw"]) == ("halt", "stop-motion", True)
    nothing = FakeOraCell((1, '{"ok": false, "error": "no program holds the arms now: nothing is moving"}', ""))
    _, client = gateway_with(make_run(tmp_path / "idle"), nothing)
    result = client.post("/ora/cell/stop-motion", json={}, headers=POST).json
    assert result["ok"] is False and "nothing is moving" in result["text"]


@pytest.mark.parametrize("answer,expected", [
    ((0, '{"ok": true, "result": "at rest"}', ""), (True, "at rest")),
    ((0, 'board says hi\n{"ok": true, "result": {"faults": []}}', ""), (True, '{\n  "faults": []\n}')),
    ((1, '{"ok": false, "error": "the cell is held by the board"}', ""), (False, "the cell is held by the board")),
    ((2, "", "Traceback (most recent call last):\nboom"), (False, "exit code 2: Traceback (most recent call last):")),
    ((None, "", "no answer within 180 s"), (False, "no answer within 180 s")),
])
def test_ora_cell_s_answer_is_read_as_the_terminal_reads_it(answer, expected):
    assert outcome(*answer) == expected


# -- The run's agent process hears each outcome ---------------------------------------------------

class Upstream(ThreadingMixIn, UnixStreamServer):
    """ORA's web server, as far as the note goes: records each request."""
    daemon_threads = True

    def __init__(self, path: str) -> None:
        self.seen: list[tuple[str, dict]] = []

        class Handler(BaseHTTPRequestHandler):
            def address_string(self) -> str:
                return "unix"

            def log_message(self, *args) -> None:
                pass

            def do_POST(handler) -> None:  # noqa: N802
                length = int(handler.headers.get("Content-Length", 0))
                self.seen.append((handler.path, json.loads(handler.rfile.read(length))))
                handler.send_response(200)
                handler.send_header("Content-Length", "2")
                handler.end_headers()
                handler.wfile.write(b"{}")

        super().__init__(path, Handler)


@pytest.fixture
def upstream():
    directory = Path(tempfile.mkdtemp(prefix="ora-web-cell-", dir="/tmp"))
    server = Upstream(str(directory / "web.sock"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, str(directory / "web.sock")
    server.shutdown()
    server.server_close()
    shutil.rmtree(directory, ignore_errors=True)


def test_each_outcome_is_handed_to_the_run_s_agent_process_and_a_gone_one_changes_nothing(tmp_path, cell_files,
                                                                                         upstream):
    ora_cell = FakeOraCell((0, '{"ok": true, "result": "at rest"}', "board log\n"))
    _, client = gateway_with(make_run(tmp_path), ora_cell, upstream=upstream[1])
    result = client.post("/ora/cell/verb/home", json={}, headers=POST).json
    deadline = time.monotonic() + 5
    while not upstream[0].seen and time.monotonic() < deadline:
        time.sleep(0.01)
    path, note = upstream[0].seen[-1]
    assert path == "/api/cell-note"
    assert {key: note[key] for key in ("id", "name", "verb", "ok", "text", "via", "exit_code")} == {
        "id": result["id"], "name": "home", "verb": "home", "ok": True, "text": "at rest", "via": "web page",
        "exit_code": 0}
    assert note["command"].endswith("scripts/ora cell home --json --cell /cells/bench.yaml")
    assert note["output"] == '{"ok": true, "result": "at rest"}board log\n'

    _, gone = gateway_with(make_run(tmp_path / "other"), ora_cell, upstream=str(tmp_path / "no.sock"))
    assert gone.post("/ora/cell/verb/home", json={}, headers=POST).json["ok"] is True


# -- Parity with the terminal ----------------------------------------------------------------------

def _terminal_commands() -> dict[str, list[tuple[str, str, bool]]]:
    """ora-pi/src/play/cell.ts: each table's (name, verb, confirm)."""

    source = (ROOT / "ora-pi" / "src" / "play" / "cell.ts").read_text(encoding="utf-8")
    entry = re.compile(r'\{\s*name:\s*"([^"]+)",\s*verb:\s*"([^"]+)",\s*confirm:\s*(true|false)')
    tables = {}
    for table in ("CELL_COMMANDS", "APPROVAL_COMMANDS", "YOLO_COMMAND", "MODE_COMMAND", "HALT_COMMAND"):
        start = source.index(f"export const {table}")
        end = source.index(";\n", start)
        tables[table] = [(name, verb, confirm == "true") for name, verb, confirm in entry.findall(source[start:end])]
    return tables


def test_every_terminal_cell_command_has_a_gateway_route():
    """A command added to the terminal (play/cell.ts) without the page's route fails here: add it to
    web_cell.py (VERBS or ANSWERS) and to the page (ora-pi/web/src/lib/cell.ts; its own test guards it)."""

    tables = _terminal_commands()
    assert {verb: confirm for _, verb, confirm in tables["CELL_COMMANDS"]} == web_cell.VERBS
    assert tuple(verb for _, verb, _ in tables["APPROVAL_COMMANDS"]) == web_cell.ANSWERS
    assert all(confirm for _, _, confirm in tables["APPROVAL_COMMANDS"])
    assert tables["YOLO_COMMAND"] == [("yolo", "auto-approve", True)]
    assert tables["MODE_COMMAND"] == [("mode", "mode", True)]
    assert tables["HALT_COMMAND"] == [("halt", "stop-motion", False)]
