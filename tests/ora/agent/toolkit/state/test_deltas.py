"""Exact pre-parent tool state, independent of rendering and interface I/O."""

import json

import pytest

from ora.agent.toolkit.state.deltas import PATH, before_call, bind_journal, journal_state, read_rows
from ora.agent.toolkit.state.keys import StateError
from ora.agent.toolkit.state.store import StateStore


def parent(model, tool="reused"):
    return {"kind": "parent", "run_id": "test", "round_index": 0,
            "model_call_id": model, "tool_call_id": tool}


def setup(root):
    state = StateStore()
    journal = bind_journal(state, root)
    writer = state.access(writer="tool")
    with journal.boundary({"kind": "owner", "operation": "initialize"}):
        writer.write("tool.marks", {})
    snapshot = StateStore(read_only=True)
    snapshot._values = dict(state._values)
    snapshot.tool_watermark = journal.watermark()
    snapshot.complete_boundary = True
    return state, journal, writer, snapshot


def test_read_only_calls_restore_at_parent_not_next_publication(tmp_path):
    state, journal, tool, snapshot = setup(tmp_path)
    ordinal = 0
    with journal.boundary(parent("m0"), lambda: ordinal):
        tool.write("tool.marks", {"rim": [1, 2, 3]})
        tool.write("tool.held.left", {"intent": True})
        ordinal = 2
        tool.write("tool.marks", {"rim": [1, 2, 4]})
    with journal.boundary(parent("m1")):
        tool.write("tool.marks", {"future": True})
    rows = read_rows(tmp_path)
    writes = [row for row in rows if row["kind"] == "write" and row["context"].get("model_call_id") == "m0"]
    assert [row["interface_ordinal"] for row in writes] == [0, 0, 2]
    assert [row["state_write_ordinal"] for row in writes] == [0, 1, 2]
    first = before_call(snapshot, tmp_path, model_call_id="m0", tool_call_id="reused")
    assert first.access().read("tool.marks") == {}
    second = before_call(snapshot, tmp_path, model_call_id="m1", tool_call_id="reused")
    assert second.access().read("tool.marks") == {"rim": [1, 2, 4]}
    original = (tmp_path / PATH).read_bytes()
    second.access(reader="rsi", writer="tool").write("tool.marks", {"experiment": True})
    assert state.access().read("tool.marks") == {"future": True}
    assert snapshot.access().read("tool.marks") == {}
    assert (tmp_path / PATH).read_bytes() == original


def test_a_journal_alone_gives_the_state_before_a_parent(tmp_path):
    """An owner with no snapshots: the journal starts at ``initialize``, so nothing is what comes before it."""
    _, journal, writer, _ = setup(tmp_path)
    with journal.boundary(parent("first")):
        writer.write("tool.marks", {"a": 1})
    with journal.boundary({"kind": "owner", "operation": "reset", "round_index": 1}):
        writer.write("tool.marks", {})
    with journal.boundary(parent("second")):
        writer.write("tool.marks", {"b": 2})
    with journal.boundary(parent("third")):
        pass

    def before(model):
        return journal_state(tmp_path, model_call_id=model, tool_call_id="reused")

    assert before("first").access(reader="rsi").read("tool.marks") == {}
    assert before("second").access(reader="rsi").read("tool.marks") == {}, "a reset is in the journal too"
    third = before("third")
    assert third.access(reader="rsi").read("tool.marks") == {"b": 2}
    third.access(reader="rsi", writer="tool").write("tool.marks", {"private": True})
    assert before("third").access(reader="rsi").read("tool.marks") == {"b": 2}, "a replay writes on its own copy"
    with pytest.raises(StateError, match="未记录"):
        before("never-called")


def test_a_journal_that_is_gone_is_not_an_empty_state(tmp_path):
    with pytest.raises(StateError, match="未记录"):
        journal_state(tmp_path, model_call_id="model", tool_call_id="tool")


def test_snapshot_watermark_in_middle_of_parent_and_explicit_reset(tmp_path):
    state, journal, tool, snapshot = setup(tmp_path)
    with journal.boundary(parent("action")):
        tool.write("tool.held.left", {"intent": True})
        snapshot._values = dict(state._values)
        snapshot.tool_watermark = journal.watermark()
        tool.write("tool.held.left", None)  # verify after the published action
    with journal.boundary({"kind": "owner", "operation": "reset", "round_index": 1}):
        tool.write("tool.marks", {})
    with journal.boundary(parent("next")):
        pass
    restored = before_call(snapshot, tmp_path, model_call_id="next", tool_call_id="reused")
    assert restored.access().read("tool.held.left") is None
    with pytest.raises(StateError, match="later"):
        before_call(snapshot, tmp_path, model_call_id="action", tool_call_id="reused")


@pytest.mark.parametrize("stage", ["begin", "write", "end"])
def test_disk_failure_preserves_tool_value_exception_and_marks_gap(tmp_path, monkeypatch, stage):
    state, journal, tool, snapshot = setup(tmp_path)
    original = type(journal.path).open
    def failed(*args, **kwargs):
        raise OSError("disk full")
    def inject():
        monkeypatch.setattr(type(journal.path), "open", lambda path, *a, **kw:
                            failed() if path == journal.path else original(path, *a, **kw))
    expected = RuntimeError("original tool exception")
    if stage == "begin":
        inject()
    with pytest.raises(RuntimeError) as raised:
        with journal.boundary(parent("failed")):
            if stage == "write":
                inject()
            tool.write("tool.marks", {"kept": True})
            if stage == "end":
                inject()
            raise expected
    monkeypatch.undo()
    assert raised.value is expected
    assert tool.read("tool.marks") == {"kept": True}
    assert journal.watermark()["complete"] is False
    assert json.loads((tmp_path / "state/tool-deltas.gaps.jsonl").read_text())["stage"] == stage
    with journal.boundary(parent("after-gap")):
        tool.write("tool.marks", {})
    with pytest.raises(StateError, match="未记录"):
        before_call(snapshot, tmp_path, model_call_id="after-gap", tool_call_id="reused")


def test_missing_watermark_tamper_unscoped_and_writer_identity(tmp_path):
    state, journal, tool, snapshot = setup(tmp_path)
    assert bind_journal(state, tmp_path) is journal
    with pytest.raises(StateError, match="another run"):
        bind_journal(state, tmp_path / "another")
    with journal.boundary(parent("valid")):
        pass
    snapshot.tool_watermark = None
    with pytest.raises(StateError, match="未记录"):
        before_call(snapshot, tmp_path, model_call_id="valid", tool_call_id="reused")
    tool.write("tool.marks", {"unscoped": True})
    assert journal.closed
    assert tool.read("tool.marks") == {"unscoped": True}
    path = tmp_path / PATH
    path.write_bytes(path.read_bytes().replace(b'"initialize"', b'"forged"'))
    with pytest.raises(StateError, match="identity"):
        read_rows(tmp_path)


def test_gap_failure_and_recursive_scope_do_not_change_live_state(tmp_path, monkeypatch):
    _, journal, tool, _ = setup(tmp_path)
    def failed(*args, **kwargs):
        raise OSError("all recording unavailable")
    monkeypatch.setattr(type(journal.path), "open", failed)
    with journal.boundary(parent("outer")):
        with journal.boundary(parent("nested")):
            tool.write("tool.result", 42)
    assert tool.read("tool.result") == 42
    assert journal.closed


def test_conflicting_snapshot_value_and_ambiguous_identity_are_rejected(tmp_path):
    _, journal, tool, snapshot = setup(tmp_path)
    with journal.boundary(parent("first")):
        tool.write("tool.marks", {"valid": True})
    with journal.boundary(parent("next")):
        pass
    snapshot._values["tool.marks"] = b'{"not_the_snapshot":true}'
    with pytest.raises(StateError, match="does not follow"):
        before_call(snapshot, tmp_path, model_call_id="next", tool_call_id="reused")
    with journal.boundary(parent("next")):
        pass
    with pytest.raises(StateError, match="ambiguous"):
        before_call(snapshot, tmp_path, model_call_id="next", tool_call_id="reused")


def test_completed_append_failure_cannot_reopen_recorder(tmp_path, monkeypatch):
    from ora.agent.toolkit.state.deltas import ToolJournal
    state, journal, tool, snapshot = setup(tmp_path)
    from ora.agent.toolkit.state import deltas
    sync = deltas.os.fsync
    def fail_after_append(fd):
        if read_rows(tmp_path)[-1]["kind"] == "end":
            raise OSError("durability uncertain after completed write")
        return sync(fd)
    monkeypatch.setattr(deltas.os, "fsync", fail_after_append)
    with journal.boundary(parent("uncertain")):
        tool.write("tool.marks", {"x": True})
    assert journal.closed
    reopened = ToolJournal(tmp_path)
    assert reopened.closed and not reopened.watermark()["complete"]


def test_before_interface_preserves_same_gap_order_and_excludes_later_writes(tmp_path):
    from ora.agent.toolkit.state.deltas import before_interface
    state, journal, tool, snapshot = setup(tmp_path)
    ordinal = 0
    with journal.boundary(parent("action"), lambda: ordinal):
        tool.write("tool.marks", {"first": True})
        tool.write("tool.held.left", {"intent": "close"})
        ordinal = 1
        tool.write("tool.held.left", None)
    restored = before_interface(snapshot, tmp_path, model_call_id="action", tool_call_id="reused", interface_ordinal=0)
    assert restored.access().read("tool.held.left") == {"intent": "close"}
    assert restored.access().read("tool.marks") == {"first": True}
    assert state.access().read("tool.held.left") is None
    # A recorded failure in this interval makes exact restoration unavailable.
    (tmp_path / "state/tool-deltas.gaps.jsonl").write_text(json.dumps({"sequence": snapshot.tool_watermark["count"]}) + "\n")
    with pytest.raises(StateError, match="未记录"):
        before_interface(snapshot, tmp_path, model_call_id="action", tool_call_id="reused", interface_ordinal=0)
