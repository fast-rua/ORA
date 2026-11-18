"""Only the gated same-boundary data method can provide official judgment.

Each answer is checked against its digest in ``tests/data/toolkit-goldens``, recorded while the handlers answered it
alike.
"""
import pytest
from tests.support.agent.toolkit_tools import install
from ora.agent.toolkit.interface.types import canonical
from tests.support.agent.toolkit_tools import make_host
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_goldens import Goldens

GOLDEN = Goldens(__file__)


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


@pytest.mark.parametrize('verdict', [None, {}, {'task_success': False}, {'task_success': True, 'terminal': True}])
def test_check_task_answers_the_recorded_verdict(golden, tmp_path, monkeypatch, verdict):
    host, client, _ = make_host(tmp_path / 'run', monkeypatch)
    try:
        if verdict is not None:
            (host.control_root / 'policy-task-check.json').write_bytes(canonical(verdict))
        parent = proposal(host, 'check_task', {})
        install(host)
        golden('check', host.invoke_with_context('check_task', {}, call_context=parent), tmp_path / 'run')
    finally:
        client.close()


def test_disabled_capability_refuses_before_reading_verdict(tmp_path, monkeypatch):
    from ora.agent.toolkit.interface.types import AccessDenied, CallContext
    from ora.agent.toolkit.host.adapter import bind_runtime
    from ora.agent.toolkit.interface.blobs import BlobStore
    from ora.agent.toolkit.interface.journal import Journal
    from ora.agent.toolkit.tools.check_task.tool import run
    from ora.agent.toolkit.host.lifecycle import check_adapter
    host, client, _ = make_host(tmp_path/'run',monkeypatch)
    try:
        control = host._current()[0]
        context = CallContext.create(run_id='run', round_index=0, tool_call_id='call-0',
            boundary=dict(kind='before_call',call=0),
            state=dict(observation={k:control[k] for k in ('observation_id','observation_sha256')}),check_task=False)
        api = bind_runtime(runtime=host,context=context,store=BlobStore(tmp_path/'run',writable=True),journal=Journal(tmp_path/'run'),bindings={})
        def forbidden(*args):
            pytest.fail('disabled capability touched official judgment')
        api.adapters['check_task'] = forbidden
        with pytest.raises(AccessDenied,match='capability disabled'):
            run(api,{})
    finally:
        client.close()
