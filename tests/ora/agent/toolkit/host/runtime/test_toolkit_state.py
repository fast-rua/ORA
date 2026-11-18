"""Live data reads follow the owner, rather than the parent-call opening publication."""
import json
from types import SimpleNamespace
from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal, read_events
from ora.agent.toolkit.interface.types import CallContext, canonical, digest
from ora.agent.toolkit.state.store import StateStore
from ora.agent.toolkit.host.adapter import bind_runtime


def test_new_publication_is_read_without_reassembling_data_api(tmp_path):
    store = StateStore()
    owner = store.access(reader='supervisor', writer='environment')
    live = {}
    host = SimpleNamespace(toolkit_state_store=store, toolkit_state_access=store.access(),
                           _current=lambda: (live['identity'], live['root'], live['manifest']))

    def publish(number):
        root = tmp_path / 'policy-observations' / f'observation-{number:04d}'
        root.mkdir(parents=True)
        manifest = {'observation_id': root.name, 'round_index': 0, 'cameras': {}}
        raw = canonical(manifest)
        (root / 'observation.json').write_bytes(raw)
        (root / 'proprioception.json').write_bytes(canonical({'joint': number}))
        identity = dict(observation_id=root.name, observation_sha256=digest(raw))
        owner.write('env.observation', identity)
        live.update(identity=identity, root=root, manifest=manifest)
        return identity

    first = publish(0)
    context = CallContext.create(run_id='run', round_index=0, tool_call_id='call',
                                 boundary={'kind': 'before_call', 'call': 0},
                                 state={'observation': first}, check_task=False)
    api = bind_runtime(runtime=host, context=context, store=BlobStore(tmp_path, writable=True),
                       journal=Journal(tmp_path), bindings={})
    assert api.state.observation().json()['data']['observation'] == first
    second = publish(1)
    response = api.state.observation().json()
    assert response['data']['observation'] == second
    assert response['provenance']['observation'] == second
    ref = api.state.proprioception().json()['data']['proprioception']
    assert json.loads(api.artifacts.read_blob(ref)) == {'joint': 1}
    begins = [e for e in read_events(api.journal.path) if e['event'] == 'begin']
    assert [e['identity']['interface_ordinal'] for e in begins] == [0, 1, 2]
    assert begins[0]['state_before']['observation'] == first
    assert begins[1]['state_before']['observation'] == second


def test_official_read_uses_granted_state_not_control_file(tmp_path):
    from ora.agent.toolkit.host.state import bind
    from ora.agent.toolkit.host.lifecycle import check_adapter
    from ora.agent.toolkit.state.keys import StateAccessDenied
    import pytest
    for tool, capability in [(False, False), (False, True), (True, False), (True, True)]:
        store = StateStore()
        store.access(reader='supervisor', writer='environment').write(
            'privileged.task_check', {'task_success': True, 'terminal': True, 'reason': 'official_success'})
        host = SimpleNamespace(toolkit_state_store=store, allowed_tools=['check_task'] if tool else [],
                               surface=SimpleNamespace(supports=lambda name: capability and name == "faithful_check_task"),
                               control_root=tmp_path/'no-control-files')
        evidence = bind(host)
        assert evidence['privileged_category'] is False
        assert evidence['effective_keys'] == (['privileged.task_check'] if tool and capability else [])
        read = check_adapter(host)
        if tool and capability:
            result = json.loads(read({}, {}, None).data)
            assert result == {'success': True, 'task_success': True,
                              'verdict': {'task_success': True, 'terminal': True}}
        else:
            with pytest.raises(StateAccessDenied):
                read({}, {}, None)
