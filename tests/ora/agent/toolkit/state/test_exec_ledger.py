"""The exec ledger keeps every IK generator change, whatever request made it, without toolkit I/O."""
import json

import pytest

from ora.agent.toolkit.state import exec_ledger
from ora.agent.toolkit.state.deltas import bind_journal
from ora.agent.toolkit.state.keys import StateError
from ora.agent.toolkit.state.store import StateStore


def ik(step):
    return {"schema": "ora.robodojo-ik-sampler-state.v1",
            "planners": {"x5": {"generator": "seed_ik_solver", "state_hex": f"{step:02x}"}}}


class Owner:
    """The Supervisor's execution owner as it applies one RPC response (orchestrator/episode/execution_state.py)."""

    def __init__(self, store):
        self.store, self.writer = store, store.access(reader="supervisor", writer="execution")

    def accept(self, method, values):
        previous = dict(self.store._values)
        self.writer.update(values)
        if self.store.exec_ledger is not None:
            self.store.exec_ledger.accepted(method, previous)


def parent(call):
    return {"kind": "parent", "run_id": "run", "round_index": 0, "model_call_id": f"m-{call}",
            "tool_call_id": call}


def make_run(tmp_path):
    store = StateStore()
    owner = Owner(store)
    owner.accept("start", {"exec.ik_sampler_state": ik(0)})
    journal = bind_journal(store, tmp_path)
    exec_ledger.bind_exec_ledger(store, tmp_path)
    ordinal = {"value": 0}
    with journal.boundary(parent("t-1"), lambda: ordinal["value"]):
        owner.accept("align_plan", {"exec.ik_sampler_state": ik(1)})   # a reach check
        ordinal["value"] = 1
        owner.accept("align_plan", {"exec.ik_sampler_state": ik(1)})   # no IK solved: no row
        ordinal["value"] = 2
        owner.accept("execute", {"exec.ik_sampler_state": ik(2), "exec.native_actions": {"used": 3}})
    owner.accept("toolkit_status", {"exec.ik_sampler_state": ik(3)})   # between calls
    with journal.boundary(parent("t-2"), lambda: 0):
        owner.accept("align_plan", {"exec.ik_sampler_state": ik(4)})
    return store


def test_one_hash_chained_row_per_change_with_its_call_scope(tmp_path):
    make_run(tmp_path)
    rows = exec_ledger.read_rows(tmp_path)
    assert [(row["kind"], row["method"], row["value"]["planners"]["x5"]["state_hex"]) for row in rows] == [
        ("bound", None, "00"), ("change", "align_plan", "01"), ("change", "execute", "02"),
        ("change", "toolkit_status", "03"), ("change", "align_plan", "04")]
    assert [row["context"] and row["context"]["tool_call_id"] for row in rows] == [None, "t-1", "t-1", None, "t-2"]
    assert [row["interface_ordinal"] for row in rows] == [None, 0, 2, None, 0]
    assert rows[2]["previous_value_sha256"] == rows[1]["value_sha256"]


def test_a_tampered_row_or_a_recording_gap_is_refused(tmp_path):
    make_run(tmp_path)
    path = tmp_path / exec_ledger.PATH
    lines = path.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[1])
    row["value"]["planners"]["x5"]["state_hex"] = "ff"
    tampered = tmp_path / "tampered"
    (tampered / "state").mkdir(parents=True)
    (tampered / exec_ledger.PATH).write_bytes(lines[0] + json.dumps(row).encode() + b"\n")
    with pytest.raises(StateError, match="identity"):
        exec_ledger.read_rows(tampered)
    with pytest.raises(StateError, match="未记录"):
        exec_ledger.read_rows(tmp_path / "absent")
    # After a recording gap the writer never reopens: later changes are absent, never guessed.
    (tmp_path / exec_ledger.GAPS).write_text(json.dumps({"sequence": 1, "tool_delta_count": 0}) + "\n")
    assert exec_ledger.ExecLedger(tmp_path, StateStore()).closed


def test_evidence_failure_never_changes_the_run(tmp_path):
    store = StateStore()
    ledger = exec_ledger.bind_exec_ledger(store, tmp_path)
    (tmp_path / exec_ledger.PATH).write_bytes(b"not json\n")
    ledger.size = -1  # the file changed outside its owner
    Owner(store).accept("execute", {"exec.ik_sampler_state": ik(5)})
    assert store.access(reader="supervisor").read("exec.ik_sampler_state") == ik(5)
    assert ledger.closed and (tmp_path / exec_ledger.GAPS).is_file()
    with pytest.raises(StateError):
        exec_ledger.bind_exec_ledger(store, tmp_path / "other")
