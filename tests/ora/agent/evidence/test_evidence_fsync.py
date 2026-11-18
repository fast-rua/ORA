"""Evidence writes fsync before they return.

``ArtifactStore`` fsyncs each artifact before linking it into place, and the
agent-event recorder fsyncs each line before returning; the Context Artifact
index is rewritten once per projection however many references it registers.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.memory.catalog import ContextArtifactCatalog
from tests.support.agent.agent_events import TrajectoryRecorder


def test_artifacts_and_events_fsync_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        calls.append(threading.get_ident())
        real_fsync(descriptor)

    monkeypatch.setattr("ora.agent.evidence.canonical.os.fsync", recording_fsync)
    store = ArtifactStore(tmp_path)
    recorder = TrajectoryRecorder(tmp_path, "run", "policy-session")
    store.put_json({"a": 1}, kind="x")
    recorder.append("session_started", payload={})
    assert calls == [threading.get_ident(), threading.get_ident()]


def test_context_index_batches_one_rewrite_per_projection(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    catalog = ContextArtifactCatalog(tmp_path, "policy-session", store)
    writes: list[int] = []
    original = catalog.__class__._write_index

    def counting(self: ContextArtifactCatalog) -> None:
        writes.append(len(self._entries))
        original(self)

    catalog.__class__._write_index = counting  # type: ignore[method-assign]
    try:
        with catalog.batched_persistence():
            refs = [
                catalog.put_bytes(
                    f"payload-{index}".encode(), kind="context_tool_result",
                    suffix=".json", media_type="application/json",
                    round_index=1, action_step=index,
                )
                for index in range(4)
            ]
            # In-memory resolution is immediate inside the batch.
            assert catalog.resolve(refs[0]["ref"])[1] == b"payload-0"
        assert writes == [4]
    finally:
        catalog.__class__._write_index = original  # type: ignore[method-assign]
    index = json.loads((tmp_path / "native/context-artifacts/index.json").read_text())
    assert sorted(index["artifacts"]) == sorted(ref["ref"] for ref in refs)
