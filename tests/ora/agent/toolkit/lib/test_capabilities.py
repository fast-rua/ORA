"""Exercise public lib against real interface runtime/replay and fake resident services."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import numpy as np
import pytest

from ora.agent.toolkit.interface.api import SIMULATOR, EFFECTS
from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.legacy import LegacyReplay
from ora.agent.toolkit.interface.runtime import Runtime
from ora.agent.toolkit.interface.types import AccessDenied, CallContext, Divergence, NotRecorded, PublicData
from ora.agent.toolkit.lib import blobs, control, geometry, grasp, motion, observation, perception
from ora.agent.toolkit.lib.values import Array, JsonValue, PointCloudM, PublicResult
from tests.support.agent.toolkit_lib import make_runtime, make_replay


def test_complete_ground_grasp_chain_and_replay_new_perception(tmp_path):
    runtime,sam,cgn_calls=make_runtime(tmp_path/'source')
    def chain(api):
        obs=observation.observe(api)
        refs=obs.camera('head');camera=observation.load_camera(api,refs)
        masks=perception.ground(api,rgb=refs.rgb,text='cup')
        cloud=geometry.mask_to_cloud(camera,masks.candidates[0].mask)
        pool=grasp.candidates(api,scene=cloud,target=cloud,camera_to_world=camera.camera_to_world,
                              workspace_bounds_m=JsonValue.of({'x':[-5,5],'y':[-5,5],'z':[0,5]}))
        return obs,camera,masks,pool
    online=chain(runtime)
    before={p.relative_to(runtime.store.root):p.read_bytes() for p in runtime.store.root.rglob('*') if p.is_file()}
    replay=make_replay(runtime,tmp_path/'evaluation')
    repeated=chain(replay)
    assert online[0].result.envelope.encoded==repeated[0].result.envelope.encoded
    assert online[0].robot.arms[0].joint_positions_rad==(.1,.2)
    assert online[0].robot.arms[0].public.json()['gripper_feedback']['measurement_source']=='test'
    assert online[1].rgb.data==repeated[1].rgb.data
    assert len(repeated[2].candidates)==3 and len(repeated[3].scores.numpy())==32
    assert repeated[2].result.comparison.json()['status']=='equal'
    assert repeated[3].result.comparison.json()['status']=='equal'
    np.testing.assert_array_equal(repeated[3].scores.numpy(),np.linspace(.2,.9,32))
    assert len(sam.calls)==2 and len(cgn_calls)==2 and replay.replay_state.cursor==3
    assert cgn_calls[0][1].dtype==np.float32 and cgn_calls[0][1].flags.c_contiguous
    new=perception.ground(replay,rgb=repeated[0].camera('head').rgb,text='another cup')
    assert new.result.comparison.json()['status']=='unavailable' and replay.replay_state.cursor==3
    with pytest.raises(NotRecorded):motion.probe_motion(replay,JsonValue.of({'new_target':[1,2,3]}))
    assert before=={p.relative_to(runtime.store.root):p.read_bytes() for p in runtime.store.root.rglob('*') if p.is_file()}
    with pytest.raises(FrozenInstanceError):repeated[0].cameras=()


@pytest.mark.parametrize('name',['segment','segment_points','ground'])
def test_all_supported_prompt_routes_keep_every_alternative(tmp_path,name):
    runtime,sam,_=make_runtime(tmp_path/'source')
    obs=observation.observe(runtime)
    refs=obs.camera('head')
    kwargs={'rgb':refs.rgb}
    if name!='ground':kwargs['depth_m']=refs.depth_m
    kwargs.update({'segment':{'point_uv':(1,1)},'segment_points':{'points_uv':((1,1),(2,2))},
                   'ground':{'text':'cup','max_candidates':3}}[name])
    result=getattr(perception,name)(runtime,**kwargs)
    replay=make_replay(runtime,tmp_path/'evaluation')
    observation.observe(replay)
    repeated=getattr(perception,name)(replay,**kwargs)
    assert len(result.candidates)=={'segment':1,'segment_points':6,'ground':3}[name]
    assert len(sam.calls)==2 and repeated.result.comparison.json()['status']=='equal'
    assert repeated.result.envelope.encoded==result.result.envelope.encoded
    if name=='segment_points':
        assert [c.fields.json()['source_index'] for c in result.candidates]==[0,1,2,0,1,2]
        assert 'query_results' in result.metadata.json()


@pytest.mark.parametrize('method',sorted((SIMULATOR|EFFECTS)-{'observation','proprioception'}))
def test_all_nonperception_capabilities_keep_full_public_envelope(tmp_path,method):
    runtime,_,_=make_runtime(tmp_path/'source')
    expected={'diagnostic':{'all_fields':[1,2,3]},'method':method}
    runtime.adapters[method]=lambda p,i,s:PublicData.from_json(expected)
    module=motion if method in ('probe_motion','reach_check','align_plan','kinematics') else control
    def invoke(api):
        if method=='check_task':return control.check_task(api)
        if method=='receipt':return control.receipt(api,action_id='action-0')
        return getattr(module,method)(api,JsonValue.of({'schema':'fixture','position_m':[1,2,3]}))
    original=invoke(runtime)
    replay=make_replay(runtime,tmp_path/'evaluation')
    result=invoke(replay)
    assert result.data.json()==expected and result.envelope.encoded==original.envelope.encoded
    assert replay.replay_state.cursor==1
    if method in ('probe_motion','reach_check','align_plan','kinematics'):
        with pytest.raises(NotRecorded):getattr(module,method)(replay,JsonValue.of({'new':True}))
    if method=='submit_action':
        second=make_replay(runtime,tmp_path/'other')
        with pytest.raises(Divergence):control.submit_action(second,JsonValue.of({'new_action':True}))
        assert second.replay_state.cursor==0


def test_blob_capability_is_granted_only_and_replay_writes_only_evaluation(tmp_path):
    runtime,_,_=make_runtime(tmp_path/'source')
    hidden=runtime.store.put(b'not granted by selected call')
    observation.observe(runtime)
    replay=make_replay(runtime,tmp_path/'evaluation')
    with pytest.raises(AccessDenied):replay.read_blob(hidden)
    obs=observation.observe(replay)
    assert blobs.read_json(replay,obs.manifest_ref).encoded==obs.manifest.encoded
    derived=blobs.save_array(replay,Array.of(np.arange(15).reshape(5,3)))
    assert (replay.evaluation.store.root/derived.json()['path']).exists()
    assert not (runtime.store.root/derived.json()['path']).exists()
    with pytest.raises(AccessDenied):
        replay.read_blob(dict(derived.json(),path='../secret'))


def test_real_legacy_manifest_state_and_missing_pixels(tmp_path):
    root=Path(__file__).resolve().parents[4]/'data/toolkit-general-pickup'
    snapshot=json.loads((root/'snapshot.json').read_bytes())
    available=snapshot['available_at_boundary']['observation']
    ctx=CallContext.create(run_id=snapshot['run_id'],round_index=0,tool_call_id=snapshot['identity']['tool_call_id'],
        boundary=snapshot['boundary'],state={'observation':{'observation_id':available['id'],
        'observation_sha256':available['manifest']['sha256']}},check_task=False)
    evaluation=Runtime(context=ctx,store=BlobStore(tmp_path,writable=True),journal=Journal(tmp_path),adapters={},bindings={})
    api=LegacyReplay(snapshot=snapshot,store=BlobStore(root),context=ctx,evaluation=evaluation,bindings={})
    obs=observation.observe(api)
    assert len(obs.cameras)==3 and len(obs.robot.arms)==2
    assert obs.manifest.encoded==(root/available['manifest']['path']).read_bytes()
    state=observation.proprioception(api)
    # The standalone file retains its pretty-printed bytes; the embedded manifest
    # subtree is a separate JSON encoding scope. Compare its fields, not its encoding.
    assert state.public.json()==obs.robot.public.json()
    assert state.public.encoded==(root/available['manifest']['path']).with_name('proprioception.json').read_bytes()
    assert obs.robot.arms[0].pose.frame=='robodojo_episode_local_ee_link'
    assert observation.load_camera(api,obs.camera('head')).rgb.shape == (480,640,3)
    with pytest.raises(NotRecorded):observation.load_camera(api,obs.cameras[1])
    with pytest.raises(AccessDenied):control.check_task(api)


def test_public_error_is_preserved_without_hidden_fallback(tmp_path):
    runtime,_,_=make_runtime(tmp_path)
    def timeout(*args):raise TimeoutError('private service detail')
    runtime.adapters['check_task']=timeout
    result=control.check_task(runtime)
    assert result.envelope.json()['status']=='timeout'
    assert b'private service detail' not in result.envelope.encoded
    with pytest.raises(ValueError):result.data
