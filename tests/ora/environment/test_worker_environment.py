"""A real cell's worker shares the cell lock and the approval broker's folder with the operator's tools."""

from __future__ import annotations

from pathlib import Path

from ora.environment.rpc import _worker_environment

ROOT = Path(__file__).resolve().parents[3]


def test_the_cell_lock_and_the_approval_folder_reach_the_worker(monkeypatch) -> None:
    monkeypatch.setenv("ORA_CELL_LOCK", "/run/cells/bench-a.lock")
    monkeypatch.setenv("ORA_CELL_APPROVALS", "/run/cells/bench-a-questions")
    monkeypatch.setenv("ORA_CELL_BOARD_TOKEN", "not for the worker")

    environment = _worker_environment(ROOT)

    assert environment["ORA_CELL_LOCK"] == "/run/cells/bench-a.lock"
    assert environment["ORA_CELL_APPROVALS"] == "/run/cells/bench-a-questions"
    assert "ORA_CELL_BOARD_TOKEN" not in environment


def test_without_them_the_worker_uses_the_defaults_like_everyone_else(monkeypatch) -> None:
    monkeypatch.delenv("ORA_CELL_LOCK", raising=False)
    monkeypatch.delenv("ORA_CELL_APPROVALS", raising=False)

    environment = _worker_environment(ROOT)

    assert "ORA_CELL_LOCK" not in environment and "ORA_CELL_APPROVALS" not in environment
