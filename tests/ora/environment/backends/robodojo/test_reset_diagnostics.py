"""CPU-only reset diagnostics, including a deliberately signaled child process."""
import json
import os
from contextlib import contextmanager
from pathlib import Path
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from ora.environment.backends.robodojo import reset_diagnostics as diagnostics
from ora.environment.backends.robodojo.round_lifecycle import RobodojoRoundMixin
from ora.environment.rpc import _worker_environment


@pytest.fixture
def handler(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        is_enabled=lambda: False,
        enable=lambda **kw: calls.append(("enable", kw)),
        disable=lambda: calls.append(("disable", {})),
    )

    @contextmanager
    def timer(seconds, file):
        calls.append(("timer", {"seconds": seconds, "file": file}))
        try:
            yield
        finally:
            calls.append(("cancel", {}))

    monkeypatch.setattr(diagnostics, "faulthandler", fake)
    monkeypatch.setattr(diagnostics, "stack_timer", timer)
    monkeypatch.delenv(diagnostics.TIMER_ENV, raising=False)
    monkeypatch.delenv(diagnostics.FATAL_ENV, raising=False)
    return calls, fake


def evidence(path):
    return json.loads(path.with_suffix(".diagnostics.json").read_text())


def test_default_preserves_timer_and_records_effective_settings(tmp_path, handler):
    calls, _ = handler
    path = tmp_path / "reset-initialization-stack.txt"
    with diagnostics.reset_diagnostics(path):
        assert evidence(path)["outcome"] == "in_progress"
        assert [name for name, _ in calls] == ["timer"]
        assert calls[0][1]["seconds"] == 60
        assert not calls[0][1]["file"].closed
    assert [name for name, _ in calls] == ["timer", "cancel"]
    assert calls[0][1]["file"].closed
    assert evidence(path)["timer_enabled"] is True
    assert evidence(path)["fatal_handler"] == "disabled"
    assert evidence(path)["outcome"] == "completed"


@pytest.mark.parametrize("timer", ["on", "off"])
@pytest.mark.parametrize("fatal", ["on", "off"])
def test_exception_identity_and_owned_handler_cleanup(tmp_path, monkeypatch, handler, timer, fatal):
    calls, _ = handler
    monkeypatch.setenv(diagnostics.TIMER_ENV, timer)
    monkeypatch.setenv(diagnostics.FATAL_ENV, fatal)
    path = tmp_path / "reset.txt"
    error = RuntimeError("reset failed")
    with pytest.raises(RuntimeError) as caught:
        with diagnostics.reset_diagnostics(path):
            raise error
    assert caught.value is error
    names = [name for name, _ in calls]
    assert ("timer" in names) == ("cancel" in names) == (timer == "on")
    assert ("enable" in names) == ("disable" in names) == (fatal == "on")
    assert all(kw["file"].closed for _, kw in calls if "file" in kw)
    assert evidence(path)["outcome"] == "exception"
    assert evidence(path)["exception_type"] == "RuntimeError"


def test_preexisting_fatal_handler_is_not_replaced(tmp_path, monkeypatch, handler):
    calls, fake = handler
    fake.is_enabled = lambda: True
    monkeypatch.setenv(diagnostics.FATAL_ENV, "on")
    path = tmp_path / "reset.txt"
    with diagnostics.reset_diagnostics(path):
        pass
    assert [name for name, _ in calls] == ["timer", "cancel"]
    assert evidence(path)["fatal_handler"] == "existing_handler_preserved"
    assert evidence(path)["fatal_file"] is None


@pytest.mark.parametrize("timer", ["on", "off"])
def test_real_reset_entry_and_round_relaunch_keep_arguments_and_result(tmp_path, monkeypatch, handler, timer):
    monkeypatch.setenv(diagnostics.TIMER_ENV, timer)
    calls = []
    env = SimpleNamespace(
        step_lim=100, reward_registered=True,
        reset=lambda **kw: calls.append(("reset", kw)),
        relaunch_for_layout=lambda layout: calls.append(("relaunch", layout)) or {"layout": layout},
    )
    owner = SimpleNamespace(environment=env, layout_schedule=None, layout_id=7,
                            output_root=tmp_path, native_action_limit=100, official_limit_multiple=3)
    assert RobodojoRoundMixin._reset_environment(owner, "reset.txt") is None
    assert env.harness_official_limit_multiple == 3
    assert RobodojoRoundMixin._reset_environment(owner, "round.txt", relaunch_layout=8) == {"layout": 8}
    assert calls == [("reset", {"seed": [7]}), ("relaunch", 8)]
    assert env.harness_layout_ids == (7,)
    for name in ("reset.txt", "round.txt"):
        assert evidence(tmp_path / name)["timer_enabled"] == (timer == "on")


def test_worker_allowlist_forwards_only_explicit_diagnostic_controls(monkeypatch, tmp_path):
    monkeypatch.setenv(diagnostics.TIMER_ENV, "off")
    monkeypatch.setenv(diagnostics.FATAL_ENV, "on")
    monkeypatch.setenv("DIAGNOSTIC_API_KEY", "secret")
    env = _worker_environment(tmp_path)
    assert env[diagnostics.TIMER_ENV] == "off"
    assert env[diagnostics.FATAL_ENV] == "on"
    assert "DIAGNOSTIC_API_KEY" not in env


@pytest.mark.parametrize("name", [diagnostics.TIMER_ENV, diagnostics.FATAL_ENV])
def test_invalid_control_is_explicit_configuration_error(name, monkeypatch, tmp_path, handler):
    monkeypatch.setenv(name, "maybe")
    with pytest.raises(ValueError, match=f"{name} must be on or off"):
        with diagnostics.reset_diagnostics(tmp_path / "reset.txt"):
            pytest.fail("invalid configuration reached reset")


def test_evidence_write_failure_does_not_mask_reset_error(monkeypatch, tmp_path, handler, capsys):
    def fail(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(diagnostics, "write_json", fail)
    error = RuntimeError("original reset error")
    with pytest.raises(RuntimeError) as caught:
        with diagnostics.reset_diagnostics(tmp_path / "reset.txt"):
            raise error
    assert caught.value is error
    assert "diagnostic evidence unavailable" in capsys.readouterr().err


def test_stack_timer_writes_every_thread_once_it_expires(tmp_path):
    release = threading.Event()

    def blocked_worker_thread():
        release.wait(5)

    other = threading.Thread(target=blocked_worker_thread, name="reset-helper")
    other.start()
    try:
        with (tmp_path / "stack.txt").open("w") as file, diagnostics.stack_timer(0.01, file):
            while not (tmp_path / "stack.txt").stat().st_size:
                pass
    finally:
        release.set()
        other.join()
    stack = (tmp_path / "stack.txt").read_text()
    assert "Timeout after 0.01 s" in stack
    assert "reset-helper" in stack
    assert "blocked_worker_thread" in stack
    assert "test_stack_timer_writes_every_thread_once_it_expires" in stack


def test_stack_timer_stays_silent_and_stops_when_the_block_ends_first(tmp_path):
    before = threading.active_count()
    with (tmp_path / "stack.txt").open("w") as file, diagnostics.stack_timer(60, file):
        assert threading.active_count() == before + 1
    assert threading.active_count() == before
    assert (tmp_path / "stack.txt").read_bytes() == b""


def test_stack_timer_survives_what_crashes_the_faulthandler_timer(tmp_path):
    """Code objects and threads come and go under the dump; ``dump_traceback_later`` segfaults on this."""

    code = """
import sys, threading, time
from pathlib import Path
from ora.environment.backends.robodojo.reset_diagnostics import stack_timer
source = "def f(n):\\n    return n if n == 0 else f(n - 1)\\nf(20)\\n"
end = time.time() + 2
with open(sys.argv[1], "w") as file:
    index = 0
    while time.time() < end:
        with stack_timer(0.002, file):
            for _ in range(40):
                index += 1
                exec(compile(source + f"# {index}\\n", f"<dyn-{index}>", "exec"), {})
            threading.Thread(target=time.sleep, args=(0.0005,)).start()
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path / "stack.txt")],
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert "Timeout after 0.002 s" in (tmp_path / "stack.txt").read_text()


def test_fatal_stack_survives_child_sigsegv_with_timer_disabled(tmp_path):
    # No simulator and no core artifact: the signal is confined to this child.
    # Testing a real fatal signal verifies FD lifetime and persisted pre-crash settings.
    path = tmp_path / "reset.txt"
    code = """
import os, resource, signal, sys
from pathlib import Path
from ora.environment.backends.robodojo.reset_diagnostics import reset_diagnostics
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
def crash_during_reset():
    os.kill(os.getpid(), signal.SIGSEGV)
with reset_diagnostics(Path(sys.argv[1])):
    crash_during_reset()
"""
    env = dict(os.environ, ORA_ROBODOJO_RESET_TRACEBACK="off", ORA_ROBODOJO_RESET_FATAL_TRACEBACK="on")
    env.pop("PYTHONFAULTHANDLER", None)
    child = subprocess.run([sys.executable, "-c", code, str(path)], env=env,
                           capture_output=True, text=True, timeout=30)
    assert child.returncode == -signal.SIGSEGV, child.stderr
    value = evidence(path)
    assert value["timer_enabled"] is False
    assert value["outcome"] == "in_progress"
    assert "finished_unix_s" not in value
    assert value["fatal_handler"] == "independent_reset_log"
    assert value["core_limit_bytes"] == [0, 0]
    stack = (tmp_path / value["fatal_file"]).read_text()
    assert "Fatal Python error: Segmentation fault" in stack
    assert "crash_during_reset" in stack
    assert path.read_bytes() == b""
