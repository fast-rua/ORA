"""Live sandbox mounts resolve to hash-bound immutable publication references."""

import shutil

import pytest

from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.gaps import require_no_gaps
from ora.agent.toolkit.interface.types import AccessDenied, NotRecorded, canonical
from ora.agent.toolkit.host import recording as toolkit_recording
from tests.support.agent.toolkit_atomic import complete_cameras
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_tools import make_host


def sandbox_mount(host):
    control, published, _ = host._current()
    sandbox = host.control_root.parent / 'sandbox/workspace/inputs/current'
    shutil.copytree(published, sandbox)
    control.update(source_root=str(sandbox), observation_manifest=str(sandbox / 'observation.json'))
    (host.control_root / 'current-observation.json').write_bytes(canonical(control))
    event = dict(type='fresh_observation_published', seq=1, run_id='run',
                 payload={'observation_id': control['observation_id']})
    (host.control_root.parent / 'agent-events.jsonl').write_bytes(canonical(event) + b'\n')
    return published, sandbox


@pytest.mark.parametrize('fault', ['missing', 'manifest_hash', 'camera_hash', 'symlink'])
def test_bad_immutable_publication_is_not_replaced_with_sandbox(tmp_path, monkeypatch, fault):
    host, client, _ = make_host(tmp_path, monkeypatch)
    try:
        complete_cameras(host)
        published, sandbox = sandbox_mount(host)
        original_value = host._current()
        if fault == 'missing':
            shutil.rmtree(published)
        elif fault == 'manifest_hash':
            (published / 'observation.json').write_text('{}')
        elif fault == 'camera_hash':
            (published / 'head/rgb.png').write_bytes(b'bad pixels')
        else:
            shutil.rmtree(published)
            published.symlink_to(sandbox, target_is_directory=True)
        with pytest.raises((AccessDenied, FileNotFoundError)):
            toolkit_recording._publication(original_value, BlobStore(tmp_path, writable=True))
        # Instrumentation still returns the original object, with a gap.
        host.invoke = lambda name, args: host._current()
        parent = proposal(host, 'exercise', {})
        toolkit_recording.install(host)
        assert host.invoke_with_context('exercise', {}, call_context=parent) == original_value
        with pytest.raises(NotRecorded):
            require_no_gaps(tmp_path, 'call-0')
    finally:
        client.close()
