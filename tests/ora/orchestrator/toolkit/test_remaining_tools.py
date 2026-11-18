"""All six M4b entries replay at their own recorded boundary without source writes."""
import json
import pytest
from ora.agent.toolkit.interface.types import canonical, digest
from tests.support.agent.toolkit_tools import install
from ora.agent.toolkit.host.recording import install as record
from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.orchestrator.toolkit.replay import experiment
from tests.support.agent.toolkit_tools import make_host
from tests.support.agent.toolkit_atomic import timeline
from tests.support.agent.toolkit_recording import proposal


@pytest.mark.parametrize('name', ['move_pose','move_to','compute','check_task','finish_episode','observe'])
@pytest.mark.parametrize('recording', [False,True])
def test_record_and_replay_remaining_tools(tmp_path, monkeypatch, name, recording):
    root = tmp_path/'run'
    host, client, _ = make_host(root, monkeypatch)
    try:
        timeline(host)
        host.public_facts_projector = project_policy_public_facts
        if name in {'move_pose','move_to'}:
            measured = host._current()[2]['proprioception']['end_effectors']['left']['pose_xyz_xyzw']
            target = dict(position_m=[measured[0]+.01, measured[1], measured[2]])
            if name == 'move_pose':
                target.update(position_mode='absolute', orientation_mode='hold_current')
            args = dict(targets=dict(left=target))
        elif name == 'compute':
            args = dict(expressions='x = np.array([1, 2, 3]) * 2', outputs=['x'])
        elif name == 'finish_episode':
            (host.control_root/'policy-task-check.json').write_bytes(canonical(dict(task_success=True,terminal=True)))
            args = dict(outcome='success',reason='check_task confirms completion at this observation boundary')
        elif name == 'observe':
            args = dict(step=0,views=['head'])
        else:
            (host.control_root/'policy-task-check.json').write_bytes(canonical(dict(task_success=False,terminal=False)))
            args = {}
        if recording:
            record(host)
        install(host)
        parent = proposal(host, name, args)
        expected = host.invoke_with_context(name, args, call_context=parent)
        before = {str(p.relative_to(root)):digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
        result = experiment(source=root, at='0', tool=name, output=tmp_path/'replay')
        assert result['status'] == 'ok', result.get('error')
        assert result['result'] == expected
        assert result['mode'] == 'recorded'
        assert result['tool_inputs']['mode'] == 'recorded'
        assert result['next_interface_ordinal'] == (0 if name == 'compute' else result['next_interface_ordinal'])
        assert {str(p.relative_to(root)):digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()} == before
        assert not (root/'native/toolkit-io/gaps.jsonl').exists()
        if name in {'move_pose','move_to'}:
            changed = json.loads(canonical(args))
            changed['targets']['left']['position_m'][0] += .02
            diverged = experiment(source=root, at='0', tool=name, arguments=changed, output=tmp_path/'changed')
            assert diverged['error']['type'] in {'NotRecorded','Divergence'}, diverged
        if name == 'finish_episode':
            diverged = experiment(source=root, at='0', tool=name,
                arguments=dict(args,reason='check_task reports the same success with another explanation'),output=tmp_path/'changed')
            assert diverged['error']['type'] == 'Divergence'
    finally:
        client.close()
