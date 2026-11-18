"""The physics journal kept open for an action (``backends/robodojo/common.py`` ``JournalLines``).

One open, line-buffered handle per action writes the bytes the per-step open, append and close wrote, and every line
is on the file as it is written, before the next physics step, so a hash or a reader at any moment sees the same file.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from ora.environment.backends.robodojo.common import JournalLines, journal_lines, json_default

LINES = [{"action_id": "action-1", "physics_clock": step, "measured": {"left": [0.1 * step, float("1e-17")]}}
         for step in range(25)]


def _text(sample):
    return json.dumps(sample, default=json_default, allow_nan=False, sort_keys=True) + "\n"


def test_one_open_handle_writes_the_bytes_the_per_step_append_wrote(tmp_path):
    before, after = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    before.touch(), after.touch()
    for sample in LINES:
        with before.open("a") as stream:
            stream.write(_text(sample))
    writer = JournalLines()
    for index, sample in enumerate(LINES):
        writer.write(after, _text(sample))
        assert after.read_bytes() == b"".join(_text(item).encode() for item in LINES[:index + 1]), "on the file as written"
    writer.close()
    assert after.read_bytes() == before.read_bytes()


def test_the_next_action_s_journal_closes_the_last_and_the_session_keeps_one_writer(tmp_path):
    session = SimpleNamespace()
    first, second = tmp_path / "action-0000.jsonl", tmp_path / "action-0001.jsonl"
    journal_lines(session).write(first, _text(LINES[0]))
    stream = journal_lines(session)._stream
    journal_lines(session).write(second, _text(LINES[1]))
    assert stream.closed and journal_lines(session) is session._journal_lines
    journal_lines(session).close()
    journal_lines(session).close()
    assert first.read_text() == _text(LINES[0]) and second.read_text() == _text(LINES[1])
