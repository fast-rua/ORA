"""A RoboLab session built by the RPC worker ends its process on close instead of closing Kit in-process.

Kit's in-process teardown segfaulted on 2026-10-18 and left that card's Vulkan device lost for every later
start; RoboDojo's worker has ended with ``os._exit`` since. In-process owners keep the full teardown.
No simulator: the bridge is a fake.
"""
from __future__ import annotations

from types import SimpleNamespace

from ora.environment import worker
from ora.environment.backends.robolab import session as session_module
from ora.environment.backends.robolab.session import RobolabSession


def _session(tmp_path, monkeypatch, **kwargs):
    session = RobolabSession(config={"task": "BananaInBowlTask", "seed": 0}, output_root=tmp_path / "native", **kwargs)
    calls: list[str] = []
    session.bridge = SimpleNamespace(close=lambda: calls.append("kit-close"))
    monkeypatch.setattr(session_module.shared_motion, "close", lambda owner: calls.append("motion-close"))
    exits: list[str] = []
    monkeypatch.setattr(session_module, "atexit", SimpleNamespace(register=lambda fn: exits.append("atexit")))
    fake_thread = lambda target, args, daemon: SimpleNamespace(start=lambda: exits.append(f"timer:{args[0]}"))
    monkeypatch.setattr(session_module, "threading", SimpleNamespace(Thread=fake_thread))
    return session, calls, exits


def test_an_in_process_session_keeps_the_full_teardown(tmp_path, monkeypatch):
    session, calls, exits = _session(tmp_path, monkeypatch)
    session.close()
    assert calls == ["motion-close", "kit-close"] and exits == []


def test_a_worker_session_ends_its_process_instead_of_closing_kit(tmp_path, monkeypatch):
    session, calls, exits = _session(tmp_path, monkeypatch, exit_process_on_close=True)
    session.close()
    session.close()
    assert calls == ["motion-close", "motion-close"]
    assert exits == ["atexit", f"timer:{session_module.CLOSE_EXIT_GRACE_S}"]


def test_the_worker_builds_robolab_sessions_that_exit_on_close(tmp_path, monkeypatch):
    monkeypatch.setattr("ora.environment.backends.robolab.session.RobolabSession", lambda **kwargs: kwargs)
    built = worker._build_session({"backend": "robolab", "output_root": str(tmp_path / "run/native"),
                                   "shared_root": str(tmp_path / "run")})
    assert built["exit_process_on_close"] is True and "exit_process_on_close" not in built["config"]
