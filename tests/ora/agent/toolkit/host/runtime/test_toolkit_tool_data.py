"""Compatibility artifacts publish atomically and preserve immutable cache bytes."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.agent.toolkit.interface.types import canonical, digest
from ora.agent.toolkit.host.runtime import tool_data as adapter


def artifact():
    identity = dict(observation_id='observation-0', camera='head')
    key = 'grounding-query-' + digest(canonical(identity))[:24]
    overlay = dict(path='overlay.png', sha256=digest(b'png'), bytes=3)
    manifest = dict(schema='ora.grounding-query.v2', grounding_ref=key, identity=identity, overlay=overlay,
        visualization=dict(renderer='grounding-candidate-inspection.v2', overview=overlay, candidate_previews=[]))
    return key, {'overlay.png': b'png', 'manifest.json': canonical(manifest)}, identity


def test_atomic_publication_and_immutable_reuse(tmp_path):
    key, files, meta = artifact()
    host = SimpleNamespace(artifact_root=tmp_path)
    adapter.publish(host, 'grounding_query', key, files, meta)
    root = tmp_path/'grounding-queries'/key
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
    adapter.publish(host, 'grounding_query', key, files, meta)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
    assert sorted(p.name for p in root.parent.iterdir()) == [key]
    (root/'overlay.png').chmod(0o644)
    (root/'overlay.png').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='content drift'):
        adapter.publish(host, 'grounding_query', key, files, meta)


@pytest.mark.parametrize('stage', ['fsync', 'rename'])
def test_failed_publication_exposes_no_partial_artifact(tmp_path, monkeypatch, stage):
    key, files, meta = artifact()
    def fail(*args):
        raise OSError('injected publication failure')
    monkeypatch.setattr(adapter.os, stage, fail)
    with pytest.raises(OSError, match='injected'):
        adapter.publish(SimpleNamespace(artifact_root=tmp_path), 'grounding_query', key, files, meta)
    assert not (tmp_path/'grounding-queries'/key).exists()
    assert not list((tmp_path/'grounding-queries').iterdir())
