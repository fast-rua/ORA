"""A real-cell worker's start sets the broker's auto-approve switch for itself only when its run was started
with it (``runtime.auto_approve_motions``, ``ora play --auto-approve``), and clears any other: a switch never
carries into another run. The simulated station never touches the cell's files. No rig is built: ``build_rig``
answers with a stand-in."""

from __future__ import annotations

import json
import os
import types

import pytest

from ora.cell.operator import approval
from ora.environment.backends.yam import rig as yam_rig
from ora.environment.backends.yam.session import YamSession


@pytest.fixture
def started(tmp_path, monkeypatch):
    monkeypatch.setenv("ORA_CELL_APPROVALS", str(tmp_path / "approvals"))
    built = []

    def build_rig(config):
        built.append(approval.auto_approve_state())       # what the rig sees when it first asks
        raise RuntimeError("no rig in this test")

    monkeypatch.setattr(yam_rig, "build_rig", build_rig)

    def start(**config):
        session = YamSession(config={"task_instruction": "play", **config}, output_root=tmp_path / "run")
        with pytest.raises(RuntimeError, match="no rig"):
            session.start()
        return built[-1]

    return start


def _flag(pid: int) -> None:
    folder = approval.directory()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / approval.AUTO_FILE).write_text(json.dumps({
        "schema": approval.AUTO_SCHEMA, "kinds": ["motion"], "by": "an earlier run", "since_unix_s": 0.0,
        "pid": pid, "run": "earlier"}))


def test_a_run_started_with_auto_approve_switches_it_on_for_its_own_worker(started, tmp_path) -> None:
    _flag(1)
    state = started(rig_mode="on", auto_approve_motions=True)
    assert (state["on"], state["kinds"], state["pid"], state["by"]) == (True, ["motion"], os.getpid(),
                                                                         "ora play --auto-approve")
    assert state["run"] == str((tmp_path / "run").resolve())


@pytest.mark.parametrize("mode", ["on", "read"])
def test_any_other_real_worker_clears_the_switch(started, mode) -> None:
    _flag(os.getpid())
    state = started(rig_mode=mode)
    assert state["on"] is False and not (approval.directory() / approval.AUTO_FILE).exists()


def test_the_simulated_station_leaves_the_cell_alone(started) -> None:
    _flag(os.getpid())
    started(rig_mode="sim")
    assert (approval.directory() / approval.AUTO_FILE).exists()
