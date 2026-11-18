"""Compute stays bounded and independent of the simulator's current observation.

Each call is checked against its digest in ``tests/data/toolkit-goldens``, recorded while the handlers answered it alike.
"""
import pytest
from ora.agent.pi.tool_rpc import _tool_error
from tests.support.agent.toolkit_tools import install
from tests.support.agent.toolkit_tools import make_host
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_goldens import Goldens

GOLDEN = Goldens(__file__)


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


@pytest.mark.parametrize('arguments', [
    {'expressions': 'x = np.array([1, 2, 3]) * 2', 'outputs': ['x']},
    {'expressions': 'x = a + 1\\ny = x * 2', 'bindings': [{'name': 'a', 'value': 3, 'from': 'manual'}]},
    {'expressions': 'import os'},
    {'expressions': 'x = 1', 'bindings': [{'name': '_x', 'value': 1}]},
    {'expressions': 'x = np.ones(10000000)'},
])
@pytest.mark.parametrize('closed', [False, True])
def test_compute_results_and_refusals(golden, tmp_path, monkeypatch, arguments, closed):
    host, client, _ = make_host(tmp_path / 'run', monkeypatch)
    try:
        if closed:
            (host.control_root / 'current-observation.json').unlink()
        parent = proposal(host, 'compute', arguments)
        install(host)
        try:
            result = host.invoke_with_context('compute', arguments, call_context=parent)
            # Kernel child timings are volatile.
            result.get('_runtime', {}).pop('elapsed_ms', None)
            outcome = ('ok', result)
        except Exception as error:
            outcome = ('error', _tool_error(error))
    finally:
        client.close()
    golden('compute', outcome, tmp_path / 'run')
