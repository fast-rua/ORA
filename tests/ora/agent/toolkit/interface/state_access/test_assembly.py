"""Category grants, dynamic state reads and snapshot-only capability refusals."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import pytest
from ora.agent.toolkit.interface.assemble import assemble_robot, assemble_rsi, assemble_snapshot
from ora.agent.toolkit.interface.types import AccessDenied, SimulatorRequired, NotRecorded
from ora.agent.toolkit.state.store import StateStore
from ora.agent.toolkit.state.keys import StateAccessDenied


def store():
    state = StateStore()
    state.access(reader='supervisor', writer='environment').update({
        'env.terminal': {'terminal': False, 'reason': None},
        'privileged.task_check': {'task_success': False},
    })
    return state


def test_live_state_is_not_cached_and_only_tool_keys_are_writable():
    owner = store()
    api = assemble_robot(backend=object(), state=owner.access(writer='tool'))
    assert not hasattr(api, 'privileged')
    with pytest.raises(FrozenInstanceError):
        api.state = None
    api.state.write('tool.example', {'count': 1})
    result = api.state.read('tool.example')
    result['count'] = 99
    assert api.state.read('tool.example') == {'count': 1}
    owner.access(reader='supervisor', writer='environment').write('env.terminal', {'terminal': True})
    assert api.state.read('env.terminal') == {'terminal': True}
    for key in ['env.terminal', 'exec.round', 'privileged.task_check']:
        with pytest.raises(AccessDenied):
            api.state.write(key, {})
    with pytest.raises(AccessDenied):
        api.state.read('privileged.task_check')


@pytest.mark.parametrize('tools,keys,allowed', [
    ((), (), False), (('check_task',), (), False), ((), ('privileged.task_check',), False),
    (('check_task',), ('privileged.task_check',), True),
])
def test_robot_task_check_requires_both_grants(tools, keys, allowed):
    calls = []
    backend = SimpleNamespace(check_task=lambda: calls.append('check') or {'original': True})
    access = store().access(privileged_tools=tools, privileged_keys=keys if allowed else ())
    api = assemble_robot(backend=backend, state=access,
                         privileged_tools=tools, privileged_keys=keys)
    assert not hasattr(api, 'privileged')
    if allowed:
        assert api.state.check_task() == {'original': True}
    else:
        with pytest.raises(AccessDenied, match='check_task capability disabled'):
            api.state.check_task()
        assert calls == []


def test_rsi_and_snapshot_keep_privilege_but_never_execution():
    state = store()
    api = assemble_rsi(backend=object(), state=state.access(reader='rsi'))
    assert api.privileged.read('privileged.task_check') == {'task_success': False}
    with pytest.raises(AccessDenied):
        assemble_robot(backend=object(), state=state.access(reader='supervisor'))
    snapshot = assemble_snapshot(backend=object(), state=state.access(reader='rsi'))
    for method in ('probe_motion', 'align_plan', 'reach_check'):
        with pytest.raises(SimulatorRequired, match='需要仿真器'):
            getattr(snapshot.planner, method)(request={})
    with pytest.raises(AccessDenied):
        snapshot.act.execute_action({}, 1)
    with pytest.raises(AccessDenied, match='output recorder'):
        snapshot.episode.finish_episode(request={})
    controls = []
    output_only = assemble_snapshot(backend=object(), state=state.access(reader='rsi'),
                                    control_recorder=lambda method, args: controls.append((method, args)))
    output_only.episode.report_issue(issue={'message': 'example'})
    assert controls == [('report_issue', {'issue': {'message': 'example'}})]
    with pytest.raises(StateAccessDenied):
        snapshot.state.write('tool.example', 1)


def test_snapshot_category_reads_original_bytes_without_a_recorded_cursor(tmp_path):
    from ora.agent.toolkit.interface.blobs import BlobStore
    from ora.agent.toolkit.interface.types import CallContext, canonical, digest
    from ora.agent.toolkit.state.snapshot import file_ref, export_snapshot, load_snapshot
    root = tmp_path/'policy-observations/obs-0'
    (root/'head').mkdir(parents=True)
    (root/'observation.json').write_bytes(canonical({'observation_id': 'obs-0'}))
    (root/'proprioception.json').write_bytes(b'{"joint":[0.123456789]}')
    (root/'head/rgb.png').write_bytes(b'original lossless pixels')
    manifest = file_ref(tmp_path, 'policy-observations/obs-0/observation.json')
    proprio = file_ref(tmp_path, 'policy-observations/obs-0/proprioception.json')
    rgb = file_ref(tmp_path, 'policy-observations/obs-0/head/rgb.png')
    state = store()
    camera = {'files': {'rgb': rgb}}
    state.access(reader='supervisor', writer='environment').update({
        'env.observation': {'observation_id': 'obs-0', 'observation_sha256': manifest['sha256'],
                            'manifest': manifest, 'proprioception_file': proprio, 'step': 0},
        'env.camera.head': camera,
    })
    policy = {'schema': 'ora.policy-state.v1', 'step': 0, 'run_id': 'run',
              'previous_state_sha256': None, 'observation': {'id': 'obs-0', 'manifest': manifest},
              'cameras': {'head': camera}}
    policy['state_sha256'] = digest(canonical(policy))
    path = tmp_path/'policy-state/steps/step-0000.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical(policy))
    reference = {**file_ref(tmp_path, 'policy-state/steps/step-0000.json', prefix='policy-state/'),
                 'step': 0, 'state_sha256': policy['state_sha256']}
    state.publication_completed()
    export_snapshot(state, tmp_path, reference)
    loaded = load_snapshot(tmp_path, 0)
    context = CallContext.create(run_id='run', round_index=0, tool_call_id='test',
                                 boundary={'kind': 'before_call', 'call': 0}, state={}, check_task=True)
    backend = SimpleNamespace(context=context, bindings={}, store=BlobStore(tmp_path))
    api = assemble_snapshot(backend=backend, state=loaded.access(reader='rsi'))
    response = api.state.observation().json()
    assert response['data']['manifest']['sha256'] == manifest['sha256']
    for ref in response['data']['files']:
        assert backend.store.read(ref) == loaded.read_artifact(ref['path'])
    ref = api.state.proprioception().json()['data']['proprioception']
    assert backend.store.read(ref) == b'{"joint":[0.123456789]}'
    assert api.state.check_task().json()['data']['task_success'] is False
    assert not hasattr(backend, 'cursor')


def test_snapshot_without_published_task_check_is_not_recorded():
    owner = store()
    owner.access(reader='supervisor', writer='environment').write('privileged.task_check', None)
    api = assemble_snapshot(backend=object(), state=owner.access(reader='rsi'))
    with pytest.raises(NotRecorded, match='no official task check'):
        api.state.check_task()


def test_legacy_rsi_missing_privilege_retains_state_error_snapshot_uses_not_recorded():
    from ora.agent.toolkit.state.keys import StateError
    from ora.agent.toolkit.interface.types import NotRecorded
    state = StateStore()
    legacy = assemble_rsi(backend=object(), state=state.access(reader="rsi"))
    with pytest.raises(StateError, match="state not recorded"):
        legacy.privileged.read("privileged.task_check")
    snapshot = assemble_snapshot(backend=object(), state=state.access(reader="rsi"))
    with pytest.raises(NotRecorded, match="未记录"):
        snapshot.privileged.read("privileged.task_check")
