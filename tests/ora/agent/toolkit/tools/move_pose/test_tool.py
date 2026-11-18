"""Motion tools' public compilation, IK references and persisted action bytes, as the toolkit answered them beside
the handlers.

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


def moved(host, name, args):
    """The toolkit's answer to ``name(args)`` on ``host`` and the action it persisted."""
    parent = proposal(host, name, args)
    install(host)
    try:
        result = {'result': host.invoke_with_context(name, args, call_context=parent)}
    except Exception as error:
        result = {'error': _tool_error(error)}
    path = host.control_root / 'submitted-action.json'
    return canonical(result), path.read_bytes() if path.exists() else None


@pytest.mark.parametrize('name', ['move_pose', 'move_to'])
@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('failure', [None, 'workspace', 'arguments'])
def test_motion_dispatch_compiles_and_persists(golden, tmp_path, monkeypatch, name, arm, failure):
    host, client, _ = make_host(tmp_path / 'run', monkeypatch)
    try:
        measured = host._current()[2]['proprioception']['end_effectors'][arm]['pose_xyz_xyzw']
        position = [measured[0] + .01, measured[1], measured[2]]
        if failure == 'workspace':
            position = [3., 3., 3.]
        target = {'position_m': position}
        if name == 'move_pose':
            target.update(position_mode='absolute', orientation_mode='hold_current')
        args = {'targets': {arm: target}}
        if failure == 'arguments':
            target.pop('position_m')
        outcome = moved(host, name, args)
    finally:
        client.close()
    golden('motion', outcome, tmp_path / 'run')
    if failure is None:
        assert outcome[1] is not None


@pytest.mark.parametrize('mode', ['absolute','delta','axes','both','unreachable','backend_error'])
def test_motion_orientations_and_ik_references(golden, tmp_path, monkeypatch, mode):
    host, client, _ = make_host(tmp_path / 'run', monkeypatch)
    try:
        original = host.align_plan_callback
        if mode == 'unreachable':
            def unreachable(request):
                value = original(request)
                for row in value['results']:
                    row.update(reachable=False, reason='no_solution')
                return value
            host.align_plan_callback = unreachable
        if mode == 'backend_error':
            def broken(request):
                raise RuntimeError('fixture IK failure')
            host.align_plan_callback = broken
        targets = {}
        for arm in ('left','right') if mode == 'both' else ('left',):
            measured = host._current()[2]['proprioception']['end_effectors'][arm]['pose_xyz_xyzw']
            target = dict(position_mode='absolute', position_m=[measured[0]+.01,*measured[1:3]],
                          orientation_mode='absolute_quaternion', orientation_xyzw=measured[3:])
            if mode == 'delta':
                target = dict(position_mode='delta',translation_m=[0.,0.,.01],position_frame='world',
                              orientation_mode='axis_angle_delta',rotation_axis='z',rotation_angle_deg=5.,orientation_frame='world')
            if mode == 'axes':
                target.pop('orientation_xyzw')
                target.update(orientation_mode='world_axes',approach_world=[0.,0.,-1.],jaw_line_world=[1.,0.,0.])
            targets[arm] = target
        outcome = moved(host, 'move_pose', dict(targets=targets))
    finally:
        client.close()
    golden('motion', outcome, tmp_path / 'run')
    assert outcome[1] is not None, outcome[0]
