"""Snapshot service calls never read recorded simulator responses or execute actions."""

import pytest

from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.runtime import Runtime
from ora.agent.toolkit.interface.snapshot_services import SnapshotServices
from ora.agent.toolkit.interface.types import AccessDenied, ServiceUnavailable, decode
from tests.support.agent.toolkit_lib import BINDINGS, make_runtime


def test_snapshot_service_recomputes_and_compares_without_cursor(tmp_path):
    old, sam, _ = make_runtime(tmp_path / "source")
    observed = old.observation().json()["data"]
    rgb = next(ref for ref in observed["files"] if ref["media_type"] == "image/png")
    expected = old.ground(rgb=rgb, text="cup")
    store = BlobStore(old.store.root)
    store.register(rgb)
    evaluation = Runtime(context=old.context, store=BlobStore(tmp_path / "out", writable=True),
                         journal=Journal(tmp_path / "out"), adapters={"ground": old.adapters["ground"]},
                         bindings=BINDINGS)
    backend = SnapshotServices(context=old.context, store=store, evaluation=evaluation,
                               history=old.journal.events)
    actual = backend.ground(rgb=rgb, text="cup")
    assert actual.payload == expected.payload
    assert actual.comparison is None
    row = decode((evaluation.store.root / "native/toolkit-io/comparisons.jsonl").read_bytes())
    assert decode((evaluation.store.root / row["reference"]["path"]).read_bytes())["status"] == "equal"
    with pytest.raises(AccessDenied):
        backend.read_blob(row["reference"])
    assert sam.calls == [("ground", "cup"), ("ground", "cup")]
    assert backend.ground(rgb=rgb, text="different").comparison is None
    with pytest.raises(AccessDenied, match="no simulator"):
        backend.probe_motion(request={})
    with pytest.raises(AccessDenied, match="no simulator"):
        backend.execute_action(action={}, budget=20)
    with pytest.raises(ServiceUnavailable):
        backend.segment(rgb=rgb, depth=rgb, point_uv=(0, 0))


def test_future_service_response_and_comparison_blobs_never_grant_script_reads(tmp_path):
    import numpy as np
    from ora.agent.toolkit.interface.types import PublicData
    old, _, _ = make_runtime(tmp_path / "source")
    rgb = old.observation().json()["data"]["files"][0]
    old.adapters["ground"] = lambda p, i, s: PublicData.from_json({
        "secret": "future-only-value", "mask": s.put_array(np.zeros((2, 2), dtype=bool))})
    response = old.ground(rgb=rgb, text="cup")
    end = old.journal.events[-1]
    source_store = BlobStore(old.store.root)
    source_store.register(rgb)
    evaluation = Runtime(context=old.context, store=BlobStore(tmp_path / "out", writable=True),
                         journal=Journal(tmp_path / "out"), bindings=BINDINGS,
                         adapters={"ground": lambda p, i, s: PublicData.from_json({
                             "secret": "current-value", "mask": s.put_array(np.ones((2, 2), dtype=bool))})})
    backend = SnapshotServices(context=old.context, store=source_store, evaluation=evaluation,
                               history=old.journal.events)
    actual = backend.ground(rgb=rgb, text="cup")
    assert actual.comparison is None
    assert b"future-only-value" not in actual.payload
    row = decode((evaluation.store.root / "native/toolkit-io/comparisons.jsonl").read_bytes())
    comparison = decode((evaluation.store.root / row["reference"]["path"]).read_bytes())
    assert comparison["status"] == "different"
    assert any(d.get("reference") == "future-only-value" for d in comparison["differences"])
    for ref in (end["response_ref"], response.json()["data"]["mask"], row["reference"]):
        with pytest.raises(AccessDenied, match="unregistered"):
            backend.read_blob(ref)
    assert backend.read_blob(actual.json()["data"]["mask"])
