"""Recorder faults never become tool/RPC errors or repeat an original side effect."""
import json
from types import SimpleNamespace

import pytest

from ora.agent.pi.tool_rpc import PiToolRpc
from ora.agent.toolkit.interface.types import canonical, digest
from ora.agent.toolkit.host import recording as recording
from ora.agent.error_classes import ToolError
from tests.support.agent.grounding_grasp_runtime import _runtime
from tests.support.agent.toolkit_recording import publication, proposal


class Rpc(PiToolRpc):
    """Real RPC run/error/projection path; only transport/artifact storage is fake."""
    context_composition = 'test_without_causal_slots'
    supports_images = False
    last_action_step = last_action_event_id = None

    def __init__(self, host):
        self.tools = host
        self.artifacts = SimpleNamespace(put_json=lambda value, **kw: {'sha256':digest(canonical(value))},
                                         put_bytes=lambda value, **kw: {'sha256':digest(value)})

    def _once(self, key, fn):
        return fn()

    def _script_moved_refusal(self, name, context):
        return None

    def _current_observation_id(self):
        return 'fixture-observation'


def execute(host, delivery, *, name='check_task', args=None, mutation=False):
    args = args or {}
    if delivery == 'direct':
        return host.invoke(name, args)
    return Rpc(host)._rpc_invoke(dict(name=name, arguments=args, effect_is_mutation=mutation,
        call_context=dict(tool_call_id='call-0',model_call_id='model-0',observation_step=0)))


def gaps(root):
    return [json.loads(line) for line in (root/'native/toolkit-io/gaps.jsonl').read_bytes().splitlines()]


def fail(message='injected recorder fault', kind=OSError):
    def raise_error(*args, **kwargs):
        raise kind(message)
    return raise_error


def host_fixture(root, *, original_error=None):
    host,_ = _runtime(root)
    publication(host)
    proposal(host,'check_task',{})
    value = dict(success=True, task_success=False)
    calls = []
    def check(args):
        calls.append(1)
        if original_error is not None:
            raise original_error
        return value
    host._check_task = check
    return host,value,calls


@pytest.mark.parametrize('original_fails',[False,True])
def test_gap_write_failure_is_swallowed_and_original_error_survives(tmp_path,monkeypatch,original_fails):
    error=ToolError('original refusal',code='original_refusal') if original_fails else None
    host,value,calls=host_fixture(tmp_path,original_error=error)
    recording.install(host)
    monkeypatch.setattr(host.invoke.__self__.journal,'append',fail())
    import ora.agent.toolkit.interface.gaps as module
    monkeypatch.setattr(module.os,'write',fail('disk full even for gaps'))
    if error:
        with pytest.raises(ToolError) as caught:host.invoke('check_task',{})
        assert caught.value is error
    else:assert host.invoke('check_task',{}) is value
    assert len(calls)==1


@pytest.mark.parametrize('original_fails',[False,True])
def test_initialization_failure_preserves_rpc_and_records_unbound_gap(tmp_path,monkeypatch,original_fails):
    error=ToolError('original refusal',code='original_refusal') if original_fails else None
    baseline,_,_=host_fixture(tmp_path/'off',original_error=error)
    expected=execute(baseline,'rpc')
    host,_,calls=host_fixture(tmp_path/'on',original_error=error)
    monkeypatch.setattr(recording,'Journal',fail('cannot open journal'))
    recording.install(host)
    assert canonical(execute(host,'rpc'))==canonical(expected)
    assert len(calls)==1
    row=gaps(tmp_path/'on')[0]
    assert row['tool_call_id'] is None and row['method'] is None and row['stage']=='context'
