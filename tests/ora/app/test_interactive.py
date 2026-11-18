"""`ora interactive` in Pi's terminal UI, `ora attach` and `ora sessions` (ora.app.interactive)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ora.app import interactive
from ora.app.main import build_parser, main as ora_main
from ora.orchestrator import paths
from ora.orchestrator.spec.run_spec import resolve_run_spec

QUICKSTART_ENV = ("ORA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ORA_MODEL", "ORA_PROVIDER", "ORA_BASE_URL")


@pytest.fixture
def quickstart_env(monkeypatch):
    for key in QUICKSTART_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ORA_API_KEY", "placeholder-not-a-secret")
    monkeypatch.setenv("ORA_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("ORA_MODEL", "local-model")


def test_interactive_spec_is_a_steerable_synthetic_run(quickstart_env):
    plan = resolve_run_spec(interactive.interactive_spec(provider="env", task=None, actions=4))
    assert plan["experiment"] == "ora-interactive"
    assert plan["operator_control"] == {"mode": "file_inbox_v1"}
    assert plan["runtime"]["terminal_actions"] == plan["budgets"]["max_robot_decisions"] == 4
    assert plan["policy"]["model"] == "local-model"
    with pytest.raises(interactive.PiTuiError):
        interactive.interactive_spec(provider="env", task=None, actions=0)


def test_interactive_spec_binds_the_chosen_provider_profile():
    plan = resolve_run_spec(interactive.interactive_spec(
        provider="deepseek_flash_openai_high_v1", task="Move once.", actions=1))
    assert plan["policy"]["provider_profile"] == "deepseek_flash_openai_high_v1"
    assert plan["policy"]["api_key_env"] == "DEEPSEEK_API_KEY"


def test_provider_choices_name_credentials_without_their_values(monkeypatch, quickstart_env):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-must-never-be-shown")
    choices = interactive.provider_choices()
    assert "sk-must-never-be-shown" not in json.dumps(choices)
    assert "placeholder-not-a-secret" not in json.dumps(choices)
    deepseek = next(item for item in choices if item["value"] == "deepseek_flash_openai_high_v1")
    assert deepseek["ready"] and "ready" in deepseek["description"]
    assert choices[-1]["value"] == interactive.ENV_PROVIDER and choices[-1]["ready"]


def test_endpoint_needs_the_live_agent_process_and_its_socket(tmp_path):
    root = tmp_path / "run"
    (root / "native").mkdir(parents=True)
    assert interactive.endpoint(root) is None
    socket = tmp_path / "tui.sock"
    socket.touch()
    pointer = root / interactive.ENDPOINT
    pointer.write_text(json.dumps({"socket": str(socket), "pid": os.getpid()}))
    assert interactive.endpoint(root)["socket"] == str(socket)
    socket.unlink()
    assert interactive.endpoint(root) is None
    socket.touch()
    pointer.write_text(json.dumps({"socket": str(socket), "pid": 2 ** 22 + 1}))  # beyond pid_max: never alive
    assert interactive.endpoint(root) is None


def test_sessions_list_runs_that_have_a_pi_session(monkeypatch, tmp_path, capsys):
    finished, native_only = tmp_path / "run-a", tmp_path / "run-b"
    (finished / "native" / "pi-session").mkdir(parents=True)
    (finished / "native" / "pi-session" / "s.jsonl").write_text("{}\n")
    (finished / "plan.json").write_text(json.dumps({"experiment": "ora-interactive", "policy": {"model": "m"}}))
    (finished / "result.json").write_text(json.dumps({
        "state": "completed", "decisions": 3, "model_calls": 5, "human_intervention": True}))
    (finished / "status.json").write_text(json.dumps({"state": "completed"}))
    (finished / "seal.json").write_text("{}")
    (native_only / "native").mkdir(parents=True)
    monkeypatch.setattr(paths, "iter_run_roots", lambda: iter([finished, native_only]))
    rows = interactive.session_rows()
    assert [row["run_id"] for row in rows] == ["run-a"]
    assert rows[0] | {"updated_unix_s": 0} == rows[0] | {
        "state": "completed", "attachable": False, "decisions": 3, "model_calls": 5,
        "human_intervention": True, "sealed": True, "updated_unix_s": 0,
    }
    assert interactive.command_sessions(argparse_namespace(action="list", json=False)) == 0
    assert "run-a" in capsys.readouterr().out


def argparse_namespace(**values):
    import argparse

    return argparse.Namespace(**{"run_id": None, "output": None, **values})


def test_interactive_without_a_spec_attach_and_sessions_reach_the_pi_terminal_ui(monkeypatch):
    seen = []
    for command in ("interactive", "attach", "sessions"):
        monkeypatch.setattr(interactive, f"command_{command}",
                            lambda args, _command=command: seen.append((_command, vars(args))) or 0)
    assert ora_main(["interactive", "--provider", "env", "--actions", "2"]) == 0
    assert ora_main(["interactive", "--spec", "task.json", "--ui", "pi"]) == 0
    assert ora_main(["attach", "run-1", "--last"]) == 0
    assert ora_main(["sessions", "export", "run-1", "-o", "x.html"]) == 0
    assert [(command, {key: args[key] for key in keys}) for (command, args), keys in zip(seen, (
        ("provider", "actions", "spec", "ui"), ("provider", "actions", "spec", "ui"),
        ("run_id", "last"), ("action", "run_id", "output"),
    ))] == [
        ("interactive", {"provider": "env", "actions": 2, "spec": None, "ui": None}),
        ("interactive", {"provider": None, "actions": 3, "spec": [Path("task.json")], "ui": "pi"}),
        ("attach", {"run_id": "run-1", "last": True}),
        ("sessions", {"action": "export", "run_id": "run-1", "output": "x.html"}),
    ]
    assert ora_main(["interactive", "--emit", "jsonl"]) == 2  # the JSON-lines backend needs --spec


def landed(root: Path) -> None:
    (root / "status.json").write_text(json.dumps({"state": "incomplete"}))
    (root / "result.json").write_text(json.dumps({"state": "incomplete", "decisions": 2, "model_calls": 4}))
    (root / "evaluation.json").write_text(json.dumps({"status": "incomplete"}))
    (root / "seal.json").write_text("{}")


def finished_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    (root / "native" / "pi-session").mkdir(parents=True)
    (root / "native" / "pi-session" / "s.jsonl").write_text("{}\n")
    (root / interactive.ENDPOINT).write_text(json.dumps({"socket": str(tmp_path / "gone.sock"), "pid": 2 ** 22 + 1}))
    return root


def test_attach_to_a_landed_run_opens_it_read_only(monkeypatch, tmp_path, capsys):
    root = finished_run(tmp_path)
    landed(root)
    monkeypatch.setattr(interactive, "node_binary", lambda: "node")
    opened = []
    monkeypatch.setattr(interactive.subprocess, "call", lambda argv, **_: opened.append(argv) or 0)
    monkeypatch.setattr(interactive, "_terminal", lambda: True)
    assert interactive.command_attach(argparse_namespace(run_id=str(root), last=False)) == 0
    assert opened == [["node", str(interactive.TUI / "view.ts"), str(root)]]
    monkeypatch.setattr(interactive, "_terminal", lambda: False)  # no terminal: the outcome instead
    assert interactive.command_attach(argparse_namespace(run_id=str(root), last=False)) == 0
    assert "state incomplete · 2 decisions" in capsys.readouterr().out and len(opened) == 1


def test_attach_while_the_run_lands_says_so_and_opens_it_once_landed(monkeypatch, tmp_path, capsys):
    """The window between the end of the episode and the seal: the Supervisor still runs, the
    agent process has closed its session, no result yet. `ora attach` must say it is landing,
    wait, then open the landed run, never wait silently for an agent process that is gone."""

    root = finished_run(tmp_path)
    (root / "agent-events.jsonl").write_text(json.dumps({"type": "session_closed", "payload": {}}) + "\n")
    supervisor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (root / "process.json").write_text(json.dumps({"pid": supervisor.pid}))
    assert interactive.run_phase(root) == interactive.LANDING

    def land() -> None:
        time.sleep(1.0)
        landed(root)
        supervisor.terminate()

    lander = threading.Thread(target=land)
    lander.start()
    opened = []
    monkeypatch.setattr(interactive, "node_binary", lambda: "node")
    monkeypatch.setattr(interactive.subprocess, "call", lambda argv, **_: opened.append((argv, time.monotonic())) or 0)
    monkeypatch.setattr(interactive, "_terminal", lambda: True)
    started = time.monotonic()
    try:
        assert interactive.command_attach(argparse_namespace(run_id=str(root), last=False)) == 0
    finally:
        lander.join()
        supervisor.wait(timeout=10)
    assert "evaluating and sealing the run" in capsys.readouterr().out
    assert [argv for argv, _ in opened] == [["node", str(interactive.TUI / "view.ts"), str(root)]]
    assert opened[0][1] - started >= 0.9  # opened only once landed
    assert interactive.run_phase(root) == interactive.FINISHED


def test_attach_reports_a_run_whose_supervisor_is_gone(monkeypatch, tmp_path):
    root = finished_run(tmp_path)
    (root / "process.json").write_text(json.dumps({"pid": 2 ** 22 + 1}))
    monkeypatch.setattr(interactive, "node_binary", lambda: "node")
    assert interactive.run_phase(root) == interactive.LOST
    with pytest.raises(interactive.PiTuiError, match="ora retry"):
        interactive.command_attach(argparse_namespace(run_id=str(root), last=False))


# -- The world view beside the terminal, and ora-remote's sessions -----------------------------

class FakeObserver:
    """WorldObserver's surface: emits snapshots (whose frame ages tick) and closes."""

    def __init__(self, emit):
        self.emit, self.closed, self.snapshots = emit, False, []

    def poll(self):
        if self.snapshots:
            self.emit(self.snapshots.pop(0))

    def close(self):
        self.closed = True


READ_WORLD_PIPE = ("import os, sys; fd = int(os.environ['ORA_WORLD_FD']); data = b''\n"
                   "while data.count(b'\\n') < int(sys.argv[1]):\n"
                   "    data += os.read(fd, 4096)\n"
                   "sys.stdout.write(data.decode())")


def test_the_world_feed_gives_each_attach_client_the_current_state_and_its_changes(tmp_path, capfd):
    from ora.app.world import TerminalWorldFeed

    observers = []
    feed = TerminalWorldFeed(lambda **kwargs: observers.append(FakeObserver(kwargs["emit"])) or observers[-1],
                             tmp_path)
    observer = observers[0]
    url = "http://127.0.0.1:40201/live"
    observer.snapshots = [{"phase": "waiting", "url": url, "frame": None, "age_s": 1},
                          {"phase": "waiting", "url": url, "frame": None, "age_s": 2}]  # only an age ticked
    assert feed.url(wait_s=2) == url
    observer.snapshots.append({"phase": "holding", "url": url, "connected": True, "visible": True,
                               "frame": {"index": 4, "cameras": []}, "message": "Holding"})
    assert feed.call([sys.executable, "-c", READ_WORLD_PIPE, "2"]) == 0
    lines = [json.loads(line) for line in capfd.readouterr().out.splitlines()]
    assert [line["phase"] for line in lines] == ["waiting", "holding"]
    assert lines[1] == {"browser_message": None, "connected": True, "frame_index": 4, "message": "Holding",
                        "phase": "holding", "run_id": None, "url": url, "visible": True}
    # A second client (a re-attach) starts from the current state.
    assert feed.call([sys.executable, "-c", READ_WORLD_PIPE, "1"]) == 0
    assert json.loads(capfd.readouterr().out)["frame_index"] == 4
    feed.close()
    assert observer.closed


def test_without_a_world_view_the_attach_client_runs_as_before(monkeypatch, tmp_path):
    from ora.app.world import TerminalWorldFeed

    feed = TerminalWorldFeed(None, tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "call", lambda argv, **kwargs: calls.append((argv, kwargs)) or 0)
    assert feed.url() is None and feed.call(["node", "attach.ts", "s"]) == 0
    assert calls == [(["node", "attach.ts", "s"], {})]
    feed.close()


def test_terminal_world_view_follows_the_terminal_and_explicit_options(monkeypatch):
    from ora.app.world import terminal_observer

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    assert terminal_observer(argparse_namespace()) is None, "not a terminal and nothing asked"
    assert terminal_observer(argparse_namespace(world_port=40201)) is not None
    assert terminal_observer(argparse_namespace(world_port=40201, world_view="off")) is None
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert terminal_observer(argparse_namespace()) is not None


def test_attach_parses_the_world_options_ora_remote_passes():
    args = build_parser().parse_args(["attach", "run-1", "--world-port", "40201", "--world-browser", "off",
                                              "--world-public-url", "http://127.0.0.1:5000/live"])
    assert (args.world_port, args.world_browser, args.world_public_url) == (40201, "off", "http://127.0.0.1:5000/live")


def live_run(tmp_path: Path) -> tuple[Path, subprocess.Popen]:
    root = tmp_path / "outputs" / "runs" / "run-9"
    (root / "native").mkdir(parents=True)
    supervisor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (root / "process.json").write_text(json.dumps({"pid": supervisor.pid}))
    return root, supervisor


def test_the_same_remote_command_attaches_to_the_run_it_left_going(monkeypatch, tmp_path, capsys):
    from ora.app import remote_session

    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / ".harness-runtime")
    monkeypatch.delenv("ORA_REMOTE_SESSION", raising=False)
    root, supervisor = live_run(tmp_path)
    command = ["play", "--env", "synthetic", "--world-port", "40201", "--world-browser", "off",
               "--world-public-url=http://127.0.0.1:40201/live"]
    try:
        remote_session.record(root)  # outside ora-remote: nothing is recorded
        assert remote_session.resume_arguments(command) is None
        monkeypatch.setenv("ORA_REMOTE_SESSION", "play-0123456789abcdef")
        assert remote_session.resume_arguments(command) is None, "nothing recorded yet: launch"
        remote_session.record(root)
        record = json.loads((tmp_path / ".harness-runtime/ora-remote/play-0123456789abcdef.json").read_text())
        assert record["run_id"] == "run-9"
        assert remote_session.resume_arguments(command) == [
            "attach", str(root), "--world-port", "40201", "--world-browser", "off",
            "--world-public-url=http://127.0.0.1:40201/live"]
        assert "run-9 is still going" in capsys.readouterr().out
        assert remote_session.resume_arguments(["attach", "run-9"]) is None, "attach names its run"
        monkeypatch.setenv("ORA_REMOTE_FRESH", "1")
        assert remote_session.resume_arguments(command) is None, "--new launches another run"
        monkeypatch.delenv("ORA_REMOTE_FRESH")
        record["host"] = "another-node"
        (tmp_path / ".harness-runtime/ora-remote/play-0123456789abcdef.json").write_text(json.dumps(record))
        assert remote_session.resume_arguments(command) is None, "another node's process ids mean nothing here"
        record["host"] = __import__("socket").gethostname()
        (tmp_path / ".harness-runtime/ora-remote/play-0123456789abcdef.json").write_text(json.dumps(record))
    finally:
        supervisor.terminate()
        supervisor.wait(timeout=10)
    assert remote_session.resume_arguments(command) is None, "a finished run: the command starts a new one"
    monkeypatch.setenv("ORA_REMOTE_SESSION", "../escape")
    with pytest.raises(ValueError):
        remote_session.resume_arguments(command)


def test_ora_routes_a_resumed_remote_command_to_attach(monkeypatch, tmp_path):
    from ora.app import remote_session

    seen = []
    monkeypatch.setattr(remote_session, "resumable_run", lambda: tmp_path / "run-9")
    monkeypatch.setattr(interactive, "command_attach", lambda args: seen.append(("attach", args)) or 0)
    monkeypatch.setenv("ORA_REMOTE_SESSION", "interactive-0123456789abcdef")
    assert ora_main(["interactive", "--provider", "env", "--world-port", "40201"]) == 0
    assert [(command, args.run_id, args.world_port) for command, args in seen] == [
        ("attach", str(tmp_path / "run-9"), 40201)]
