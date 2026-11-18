"""Observe pixels, validation and compatibility overlays, as the toolkit answered them beside the handlers.

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


@pytest.mark.parametrize('arguments', [{}, {'views':['head']}, {'views':['right_wrist','head']},
    {'views':['wrong']}, {'views':['head','head']}, {'history_point_ids':['missing']},
    {'step': 0}, {'step': -1}, {'step': 1, 'history_point_ids':['point']},
    {'history_point_ids':['point']}])
@pytest.mark.parametrize('projector', [False, True])
def test_observe_pixels_validation_and_overlays(golden, tmp_path, monkeypatch, arguments, projector):
    host, client, _ = make_host(tmp_path / 'run', monkeypatch)
    try:
        if projector:
            from ora.agent.context.observation.contact_points import project_policy_public_facts
            host.public_facts_projector = project_policy_public_facts
        parent = proposal(host, 'observe', arguments)
        install(host)  # the toolkit keeps its marks in tool state, as its mark_point writes them
        host.toolkit_state_access.write('tool.marks', {'point': {'status': 'solved', 'xyz_m': [0., 0., .8]}})
        try:
            value = {'result': host.invoke_with_context('observe', arguments, call_context=parent)}
        except Exception as error:
            value = {'error': _tool_error(error)}
        overlays = {p.relative_to(host.artifact_root).as_posix(): p.read_bytes()
                    for p in host.artifact_root.glob('*/overlays/*.png')}
        golden('observe', (canonical(value), overlays), tmp_path / 'run')
    finally:
        client.close()


@pytest.mark.parametrize('step', [None, 0, 1, -1, 2, False])
def test_verified_history_and_current_pixels(golden, tmp_path, monkeypatch, step):
    from tests.support.agent.toolkit_atomic import timeline
    from ora.agent.context.observation.contact_points import project_policy_public_facts
    host, client, _ = make_host(tmp_path / 'run', monkeypatch)
    try:
        timeline(host)
        host.public_facts_projector = project_policy_public_facts
        args = dict(step=step, views=['head','right_wrist'])
        parent = proposal(host, 'observe', args)
        install(host)
        try:
            outcome = {'result': host.invoke_with_context('observe', args, call_context=parent)}
        except Exception as error:
            outcome = {'error': _tool_error(error)}
    finally:
        client.close()
    golden('observe', outcome, tmp_path / 'run')
    if step is None or type(step) is int and step in (0, 1):
        assert 'result' in outcome, outcome
        value = outcome['result']
        assert len(value['images_base64']) == 2
        assert value['observation_id'].endswith('observation-0000' if step == 0 else 'observation-0001')
        assert value['public_robot_facts']['snapshot_role'] == ('historical' if step == 0 else 'current')


def test_robodojo_tool_context_carries_none_of_the_other_backends_keys(tmp_path, monkeypatch):
    from ora.agent.toolkit.host.observe import configuration as observe_configuration
    from ora.agent.toolkit.host.runtime.tool_data import configuration
    host, client, _ = make_host(tmp_path, monkeypatch)
    try:
        install(host)
        assert not {"pointcloud", "camera_list"} & observe_configuration(host)[0].keys()
        assert "overhead_view_pose" not in configuration(host)
    finally:
        client.close()
