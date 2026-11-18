"""A RoboDojo session built by the RPC worker ends its process on close instead of tearing Kit down.

Kit's graceful shutdown unloads ~290 extensions at ~0.3 s each, so the worker
always outlived the client's close deadline (measured on an H200 host,
2026-10-23). In-process owners keep the full teardown.
No simulator: the plant and Kit app are fakes.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment import worker
from ora.environment.backends.robodojo import session as session_module
from ora.environment.backends.robodojo.session import RobodojoSession


def _session(tmp_path, monkeypatch, **kwargs):
    from tests.support.robodojo.backend import Plant, _fake_source_manifest

    calls: list[str] = []
    plant = Plant()
    plant.close = lambda: calls.append("plant-close")
    app = SimpleNamespace(close=lambda: calls.append("kit-close"))
    session = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (plant, app, _fake_source_manifest(tmp_path)), **kwargs)
    session._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    exits: list[str] = []
    # Only this module's view: reset's stack timer and lazy imports need the real ones.
    monkeypatch.setattr(session_module, "atexit", SimpleNamespace(register=lambda fn: exits.append("atexit")))
    fake_thread = lambda target, args, daemon: SimpleNamespace(start=lambda: exits.append(f"timer:{args[0]}"))
    monkeypatch.setattr(session_module, "threading", SimpleNamespace(Thread=fake_thread))
    return session, calls, exits


def test_an_in_process_session_keeps_the_full_teardown(tmp_path, monkeypatch):
    session, calls, exits = _session(tmp_path, monkeypatch)
    session.start()
    session.close()
    assert calls == ["plant-close", "kit-close"] and exits == []


def test_a_worker_session_ends_its_process_instead_of_tearing_kit_down(tmp_path, monkeypatch):
    session, calls, exits = _session(tmp_path, monkeypatch, exit_process_on_close=True)
    session.start()
    session.close()
    session.close()
    assert calls == [] and exits == ["atexit", f"timer:{session_module.CLOSE_EXIT_GRACE_S}"]
    assert session.closed


def test_the_worker_builds_robodojo_sessions_that_exit_on_close(tmp_path, monkeypatch):
    monkeypatch.setattr("ora.environment.backends.robodojo.session.RobodojoSession", lambda **kwargs: kwargs)
    built = worker._build_session({"backend": "robodojo", "output_root": str(tmp_path / "run/native"),
                                   "shared_root": str(tmp_path / "run")})
    assert built["exit_process_on_close"] is True and "exit_process_on_close" not in built["config"]
