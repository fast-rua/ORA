"""A prepared skill's commitment, action and result from public fixture pixels.

Each answer is checked against its digest in ``tests/data/toolkit-goldens``, recorded while the handlers answered it
alike.
"""
import json
import pytest
from ora.agent.toolkit.tools.vision_ground.tool import run as ground
from ora.agent.toolkit.tools.grasp_plan.tool import run as plan
from ora.agent.toolkit.tools.grasp_execute.tool import run
from ora.agent.toolkit.host.failures import invoke
from tests.support.agent.toolkit_tools import make_host,api_for
from tests.support.agent.toolkit_goldens import Goldens

GOLDEN = Goldens(__file__)
#: What one prepared skill leaves in the control root.
WRITTEN = ('submitted-action.json','manipulation-skill-ledger.json','manipulation-skill-commitment.json')


@pytest.fixture
def golden(request):
    return GOLDEN.checker(request)


@pytest.mark.parametrize('legacy_visible', [False, True])
@pytest.mark.parametrize('arm',['left','right'])
def test_prepared_commitment_action_and_result(golden,tmp_path,monkeypatch,arm,legacy_visible):
    host,client,_=make_host(tmp_path/'run',monkeypatch)
    try:
        grounded=ground(api_for(host),dict(camera='head',text='object',max_candidates=1))
        planned=plan(api_for(host,'call-1'),dict(target_ref=grounded['candidates'][0]['target_ref'],arm=arm,max_candidates=3))
        assert planned['candidates']
        args=dict(candidate_id=planned['candidates'][0]['candidate_id'],rationale='  fixed public evidence  ')
        def probe(request):
            return dict(schema='ora.motion-probe-result.v1',observation_id=request['observation_id'],
                world_mutation_sequence=0,variant_policy=request['variant_policy'],robot_decisions_consumed=0,world_state_changed=False,
                results=[dict(query_id=q['query_id'],arm=q['arm'],reachable=True,variants=[])
                         for q in request['queries']])
        host.motion_probe_callback=probe
        api=api_for(host,'call-2')
        if legacy_visible:
            config = json.loads(api.tool_data.configuration)
            config.update(synchronous_execution=True, legacy_visible=True)
            api.tool_data.configuration = json.dumps(config).encode()
            seen = []
            def stages(api, config, commitment, sequence):
                seen.append((commitment, sequence))
                return {"status": "executed", "_runtime": {"skill_feedback": {}}}
            monkeypatch.setattr("ora.agent.toolkit.tools.grasp_execute._stages.run", stages)
            actual = invoke(run, api, args)["_runtime"]["legacy_submission"]
            golden('legacy_submission', actual, tmp_path/'run')
            assert len(seen) == 1
            golden('staged_commitment', seen[0][0], tmp_path/'run')
            assert seen[0][1] == actual["committed_sequence"]
            return
        actual=invoke(run,api,args)
        golden('result',actual,tmp_path/'run')
        golden('written',{name:(host.control_root/name).read_bytes() for name in WRITTEN},tmp_path/'run')
    finally:client.close()
