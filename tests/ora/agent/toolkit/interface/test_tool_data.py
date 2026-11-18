"""Only boundary-bound, hash-verified public tool artifacts cross the interface."""
import json
import pytest

from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.tool_data import ToolData
from ora.agent.toolkit.interface.types import AccessDenied, CorruptRecord, canonical, digest

def artifact(kind='target', pixels=b'pixels'):
    identity = dict(observation_id='obs', camera='head', test=True)
    prefix = 'target-' if kind == 'target' else 'segment-query-' if kind == 'segmentation_query' else 'grounding-query-'
    key = prefix + digest(canonical(identity))[:24]
    files = {'overlay.png': pixels}
    def ref(name): return dict(path=name, sha256=digest(files[name]), bytes=len(files[name]))
    manifest = dict(identity=identity, overlay=ref('overlay.png'))
    if kind == 'target':
        files['mask.npy'] = b'mask'
        manifest.update(schema='ora.segmented-target.v2', target_ref=key, private_mask=ref('mask.npy'))
    elif kind == 'segmentation_query':
        files['overview.png'] = files.pop('overlay.png')
        files['raw-masks.npz'] = b'masks'
        manifest.pop('overlay')
        manifest.update(schema='ora.point-segmentation-batch.v2', query_ref=key,
                        images=[ref('overview.png')], raw_masks=ref('raw-masks.npz'))
    else:
        manifest.update(schema='ora.grounding-query.v2', grounding_ref=key, visualization=dict(
            renderer='grounding-candidate-inspection.v2', overview=ref('overlay.png'), candidate_previews=[]))
    files['manifest.json'] = canonical(manifest)
    return key, files, dict(observation_id='obs', camera='head')

KEY, FILES, META = artifact()
QUERY, QUERY_FILES, QUERY_META = artifact('grounding_query')


def test_input_output_and_configuration_are_owned_bytes(tmp_path):
    store = BlobStore(tmp_path, writable=True)
    config = dict(cameras=['head'])
    data = ToolData(store=store, configuration=config)
    config['cameras'].append('future')
    assert data.context() == dict(cameras=['head'])
    record = data.publish('target', KEY, FILES, META)
    record['metadata']['camera'] = 'future'
    assert data.lookup('target', KEY)['metadata']['camera'] == 'head'
    inputs = {f'target:{KEY}': data.lookup('target', KEY)}
    second = ToolData(store=store, configuration={}, inputs=inputs)
    inputs.clear()
    assert second.lookup('target', KEY) == data.lookup('target', KEY)
    assert second.lookup('target', 'target-' + 'c'*24) is None


@pytest.mark.parametrize("kind", ["grounding_query", "segmentation_query"])
def test_replay_never_bypasses_service_with_query_cache(tmp_path, kind):
    store = BlobStore(tmp_path, writable=True)
    QUERY, QUERY_FILES, QUERY_META = artifact(kind)
    data = ToolData(store=store, configuration={})
    record = data.publish(kind, QUERY, QUERY_FILES, QUERY_META)
    replay = ToolData(store=store, configuration={}, inputs={f'{kind}:{QUERY}': record}, replay=True)
    assert replay.lookup(kind, QUERY, cache=True) is None
    _, files, meta = artifact(kind, b'fresh')
    fresh = replay.publish(kind, QUERY, files, meta)
    assert replay.lookup(kind, QUERY, cache=True) is None
    assert store.read(fresh['files']['manifest.json']) == files['manifest.json']
    with pytest.raises(AccessDenied, match='cannot publish live'):
        ToolData(store=store, configuration={}, replay=True, publisher=lambda *a: None)


@pytest.mark.parametrize('kind,key,name', [('target', '../secret', 'mask.npy'),
    ('truth', KEY, 'manifest.json'), ('target', KEY, '../ground-truth.json'),
    ('target', KEY, '/tmp/secret.json'), ('target', KEY, 'arbitrary.json')])
def test_no_arbitrary_path_or_category(tmp_path, kind, key, name):
    store = BlobStore(tmp_path, writable=True)
    data = ToolData(store=store, configuration={})
    with pytest.raises(AccessDenied):
        data.publish(kind, key, {name: b'x'}, {})
    assert not list(tmp_path.rglob('blobs'))


def test_bad_input_hash_and_unregistered_ref_rejected(tmp_path):
    store = BlobStore(tmp_path, writable=True)
    ref = store.put(b'{}')
    data = dict(files={'manifest.json': ref}, metadata={})
    (tmp_path/ref['path']).write_bytes(b'corrupt')
    with pytest.raises(CorruptRecord):
        ToolData(store=store, configuration={}, inputs={f'target:{KEY}': data})


@pytest.mark.parametrize('kind', ['target', 'segmentation_query'])
@pytest.mark.parametrize('fault', ['identity', 'member', 'metadata', 'missing', 'undeclared'])
def test_cached_cross_references_fail_closed(tmp_path, fault, kind):
    store = BlobStore(tmp_path, writable=True)
    key, files, meta = artifact(kind)
    manifest = json.loads(files['manifest.json'])
    if fault == 'identity': manifest['identity']['test'] = False
    if fault == 'member': manifest['overlay' if kind == 'target' else 'raw_masks']['sha256'] = '0'*64
    if fault == 'metadata': meta['camera'] = 'future'
    if fault == 'missing': files.pop('mask.npy' if kind == 'target' else 'raw-masks.npz')
    if fault == 'undeclared': files['candidate-T0.png'] = b'unknown'
    files['manifest.json'] = canonical(manifest)
    refs = {name: store.put(raw) for name, raw in files.items()}
    with pytest.raises((CorruptRecord, AccessDenied)):
        ToolData(store=store, configuration={}, inputs={f'{kind}:{key}': dict(files=refs, metadata=meta)})
