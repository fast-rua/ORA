"""Finish refusals, ordering and publication, as the toolkit answered them beside the handlers.

Each call is checked against its digest in ``tests/data/toolkit-goldens``, recorded while the handlers answered it alike.
"""
import pytest
from ora.agent.pi.tool_rpc import _tool_error
from tests.support.agent.toolkit_tools import install
from ora.agent.toolkit.interface.types import canonical
from tests.support.agent.toolkit_tools import make_host
from tests.support.agent.toolkit_recording import proposal
from tests.support.agent.toolkit_goldens import Goldens

GOLDEN = Goldens(__file__)


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


@pytest.mark.parametrize('arguments', [
    {'outcome': 'success', 'reason': 'check_task confirms the object has reached its destination'},
    {'outcome': 'give_up', 'category': 'unreachable', 'reason': 'The arm cannot reach the remaining object pose'},
    {'outcome': 'abort', 'reason': 'The camera service has stopped responding to requests'},
    {'outcome': 'failure'}, {'outcome': 'success', 'reason': 'done'},
])
@pytest.mark.parametrize('verdict', [None, {'task_success': False, 'terminal': False}, {'task_success': False, 'terminal': True}, {'task_success': True, 'terminal': True}])
@pytest.mark.parametrize('pending', [False, True])
def test_finish_refusals_order_and_publication(golden, tmp_path, monkeypatch, arguments, verdict, pending):
    monkeypatch.setattr('time.time', lambda: 123.)
    host, client, _ = make_host(tmp_path / 'run', monkeypatch)
    try:
        if verdict is not None:
            (host.control_root / 'policy-task-check.json').write_bytes(canonical(verdict))
        if pending:
            (host.control_root / 'submitted-action.json').write_text('{}')
        parent = proposal(host, 'finish_episode', arguments)
        install(host)
        try:
            value = {'result': host.invoke_with_context('finish_episode', arguments, call_context=parent)}
        except Exception as error:
            value = {'error': _tool_error(error)}
        path = host.control_root / 'policy-finish.json'
        golden('finish', (canonical(value), path.read_bytes() if path.exists() else None), tmp_path / 'run')
    finally:
        client.close()


def test_official_verdict_context_pack_has_no_judgment_query(golden, tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    host, client, _ = make_host(tmp_path / 'run', monkeypatch)
    try:
        host.surface = replace(host.surface, capabilities=(*host.surface.capabilities,'finish_success_official_verdict_v1'))
        host.allowed_tools = tuple(name for name in host.allowed_tools if name != 'check_task')
        (host.control_root/'policy-task-check.json').write_text('not readable JSON')
        args = dict(outcome='success',reason='The camera image shows the object in its final location')
        parent = proposal(host,'finish_episode',args)
        install(host)
        golden('finish', host.invoke_with_context('finish_episode',args,call_context=parent), tmp_path / 'run')
        rows = [json.loads(row) for row in (host.control_root.parent/'native/toolkit-io/calls.jsonl').read_bytes().splitlines()]
        assert {row['request']['identity']['method'] for row in rows} == {'finish_episode'}
        index = (host.control_root.parent/'native/toolkit-io/tool-inputs.jsonl').read_text()
        assert 'task_success' not in index and 'terminal' not in index
    finally:
        client.close()


def test_without_the_official_check_finish_accepts_no_success():
    from ora.agent.toolkit.host.finish.outcomes import accepted_outcomes
    from tests.support.agent.surface_catalog import surface_of

    assert accepted_outcomes(surface_of("robodojo_placement_toolkit_v6")) == ("give_up", "abort")


def test_a_give_up_category_is_accepted_only_where_the_surface_declares_it():
    """insufficient_budget (2026-10-28) is declared by the file-backed surfaces only; a configuration recorded
    before the categories key existed accepts the released list, as it did then."""
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    from tests.support.agent.surface_catalog import surface_of
    from ora.agent.toolkit.tools.errors import ToolRefusal
    from ora.agent.toolkit.tools.finish_episode._arguments import CATEGORIES, _check_category
    from ora.agent.toolkit.host.lifecycle import declared_categories

    grants = {"tools": ["check_task"], "state_keys": ["privileged.task_check"]}
    released = declared_categories(surface_of("robotwin_toolchain_b2_closed_noservo_v2"))
    files = declared_categories(resolve_tool_surface("robodojo_placement_toolkit_v5", toolkit_grants=grants))
    assert released == list(CATEGORIES) and files == [*CATEGORIES][:-1] + ["insufficient_budget", "other"]
    _check_category("give_up", "insufficient_budget", files)
    for declared in (released, None):
        with pytest.raises(ToolRefusal) as refused:
            _check_category("give_up", "insufficient_budget", declared)
        assert refused.value.constraints["allowed"] == list(CATEGORIES)
