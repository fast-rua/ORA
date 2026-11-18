"""Recorded --tool replay uses frozen inputs, fresh fake perception and no source writes."""
import json
from contextlib import contextmanager
from pathlib import Path
import pytest
from ora.agent.toolkit.interface.services import sam3_adapters,grasp_adapter
from ora.agent.toolkit.interface.types import digest
from tests.support.agent.toolkit_tools import install
from ora.orchestrator.toolkit.replay import experiment
from tests.support.agent.toolkit_tools import make_host
from tests.support.agent.toolkit_recording import proposal


def tree(root):return {str(p.relative_to(root)):digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}


def test_replay_ground_plan_and_execute(tmp_path,monkeypatch):
    source=tmp_path/'run'
    host,client,_=make_host(source,monkeypatch)
    try:
        install(host)
        ground_args=dict(camera='head',text='object',max_candidates=1)
        first=host.invoke_with_context('vision_ground',ground_args,call_context=proposal(host,'vision_ground',ground_args))
        plan_args=dict(target_ref=first['candidates'][0]['target_ref'],arm='left',max_candidates=3)
        second=host.invoke_with_context('grasp_plan',plan_args,call_context=proposal(host,'grasp_plan',plan_args,index=1))
        def probe(request):
            return dict(schema='ora.motion-probe-result.v1',observation_id=request['observation_id'],
                world_mutation_sequence=0,variant_policy=request['variant_policy'],robot_decisions_consumed=0,world_state_changed=False,
                results=[dict(query_id=q['query_id'],arm=q['arm'],reachable=True,variants=[]) for q in request['queries']])
        host.motion_probe_callback=probe
        execute_args=dict(candidate_id=second['candidates'][0]['candidate_id'],rationale='public evidence')
        third=host.invoke_with_context('grasp_execute',execute_args,call_context=proposal(host,'grasp_execute',execute_args,index=2))
        events=[json.loads(line) for line in (source/'native/toolkit-io/calls.jsonl').read_bytes().splitlines()]
        bindings={e['request']['identity']['method']:e['request']['identity']['binding'] for e in events
                  if e['request']['identity']['method'] in {'ground','grasp_candidates'}}
        @contextmanager
        def services(report):
            def propose(p,scene,target):return client.propose(scene_points_camera_m=scene,target_points_camera_m=target,
                camera_to_world=p['camera_to_world'],target_ref='toolkit-raw-pool',workspace_bounds_m=p['workspace_bounds_m'],
                max_candidates=p['max_candidates'])
            yield dict(sam3_adapters(host.grounding_backend),grasp_candidates=grasp_adapter(propose)),bindings
        monkeypatch.setattr('ora.orchestrator.toolkit.replay.service_session',services)
        before=tree(source)
        for index,(name,expected) in enumerate(zip(('vision_ground','grasp_plan','grasp_execute'),(first,second,third))):
            result=experiment(source=source,at=str(index),tool=name,output=tmp_path/f'replay-{index}')
            assert result['status']=='ok',result.get('error')
            assert result['result']==expected
            assert result['tool_inputs']['mode']=='recorded'
            assert result['next_interface_ordinal']>0
            comparisons=[json.loads(line) for line in (tmp_path/f'replay-{index}'/result['comparisons']).read_bytes().splitlines()]
            if index<2:
                assert comparisons
                for row in comparisons:
                    comparison=json.loads((tmp_path/f'replay-{index}'/row['reference']['path']).read_bytes())
                    assert comparison['status']=='equal',comparison
        assert tree(source)==before
        different=experiment(source=source,at='2',tool='grasp_execute',arguments=dict(execute_args,rationale='different'),
            output=tmp_path/'diverged')
        assert different['error']['type']=='Divergence'
        assert tree(source)==before
        from ora.agent.toolkit.tools.grasp_execute import _measure
        original_semantics=_measure.semantics
        monkeypatch.setattr(_measure,'semantics',lambda *a:dict(original_semantics(*a),phrase='changed preparation'))
        changed=experiment(source=source,at='2',tool='grasp_execute',output=tmp_path/'preparation-diverged')
        assert changed['error']['type']=='Divergence'
        script=tmp_path/'inspect_inputs.py'
        script.write_text('def run(api,args):\n    return api.artifacts.tool_context()["grasp_query_refs"]\n')
        inspected=experiment(source=source,at='2',script=script,output=tmp_path/'script-inputs')
        assert inspected['status']=='ok'
        assert inspected['result'] and inspected['tool_inputs']['mode']=='recorded'
        assert tree(source)==before
    finally:client.close()
