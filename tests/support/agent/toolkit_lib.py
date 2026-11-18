"""Small public RGB-D fixture and fake services; no simulator, network or GPU."""
from io import BytesIO
import numpy as np
from PIL import Image

from ora.agent.toolkit.interface.api import EXTERNAL
from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.interface.replay import Replay
from ora.agent.toolkit.interface.runtime import Runtime
from ora.agent.toolkit.interface.services import sam3_adapters, grasp_adapter
from ora.agent.toolkit.interface.types import CallContext, PublicData, canonical

BINDINGS={m:dict(backend='fake',service_identity={'id':'test'},model_identity={'id':'test'},
                configuration_identity={'version':1}) for m in EXTERNAL}


class Sam:
    def __init__(self):self.calls=[]

    def segment(self, *, rgb_u8, depth_m, point_uv):
        self.calls.append(('segment',point_uv))
        return np.ones(depth_m.shape,dtype=bool),{'model_score':.8}

    def segment_points(self, *, rgb_u8, depth_m, points_uv):
        self.calls.append(('segment_points',points_uv))
        return [dict(query_index=i,source_index=j,point_uv=list(point),backend_score=.8,
                     mask=np.ones(depth_m.shape,dtype=bool))
                for i,point in enumerate(points_uv) for j in range(3)], {
                    'query_results':[{'query_index':0,'point_uv':list(points_uv[0]),'raw_candidate_count':3,'error_code':None}]}

    def ground(self, *, rgb_u8, text, max_candidates):
        self.calls.append(('ground',text))
        return [dict(rank=i,source_index=i,backend_score=.9-i*.1,bbox_xyxy=[0,0,4,4],
                     mask=np.ones(rgb_u8.shape[:2],dtype=bool)) for i in range(max_candidates)], {'request_id':'fake'}


def make_runtime(root):
    store=BlobStore(root,writable=True)
    rgb=np.full((4,4,3),100,dtype=np.uint8)
    stream=BytesIO();Image.fromarray(rgb).save(stream,format='PNG')
    rgb_ref=store.put(stream.getvalue(),media_type='image/png')
    stream=BytesIO();np.save(stream,np.ones((4,4),dtype=np.float32),allow_pickle=False)
    depth_ref=store.put(stream.getvalue(),media_type='application/x-npy')
    calibration=dict(intrinsic_cv=[[2.,0,1.],[0,2.,1.],[0,0,1.]],
                     extrinsic_cv=[[1.,0,0,-2.],[0,1.,0,0],[0,0,1.,0],[0,0,0,1.]])
    calib_ref=store.put(canonical(calibration),media_type='application/json')
    proprio=dict(end_effectors={'left':dict(arm_joint_qpos=[.1,.2],gripper=.7,
        pose_frame='world',pose_xyz_xyzw=[1.,2.,3.,0.,0.,0.,1.],gripper_feedback={'measurement_source':'test'})})
    proprio_ref=store.put(canonical(proprio),media_type='application/json')
    manifest=dict(observation_id='obs-0',round_index=0,proprioception=proprio,cameras={
        'head':dict(width=4,height=4,files={'rgb_lossless_png':rgb_ref,
                    'depth_numpy_m_f32':depth_ref,'calibration':calib_ref})})
    manifest_ref=store.put(canonical(manifest),media_type='application/json')
    observed=dict(observation_id='obs-0',observation_sha256=manifest_ref['sha256'])
    context=CallContext.create(run_id='run',round_index=0,tool_call_id='call-0',
                              boundary={'kind':'before_call','call':0},state={'observation':observed},check_task=True)
    sam=Sam()
    calls=[]
    def propose(params,scene,target):
        calls.append((params,scene.copy(),target.copy()))
        n=params['max_candidates']
        poses=np.tile(np.eye(4),(n,1,1));poses[:,0,3]=np.arange(n)*.01
        return [],{'_canonical_backend_output':dict(poses_camera=poses,scores=np.linspace(.2,.9,n),
            contact_points_camera=poses[:,:3,3],openings_m=np.full(n,.04)),'raw_candidate_limit':n}
    adapters=dict(sam3_adapters(sam),grasp_candidates=grasp_adapter(propose),
        observation=lambda p,i,s:PublicData.from_json(dict(observation=observed,manifest=manifest_ref,
                                  files=[rgb_ref,depth_ref,calib_ref,proprio_ref])),
        proprioception=lambda p,i,s:PublicData.from_json({'proprioception':proprio_ref}))
    runtime=Runtime(context=context,store=store,journal=Journal(root),adapters=adapters,bindings=BINDINGS)
    return runtime,sam,calls


def make_replay(runtime, root):
    evaluation=Runtime(context=runtime.context,store=BlobStore(root,writable=True),journal=Journal(root),
        adapters={k:v for k,v in runtime.adapters.items() if k in EXTERNAL},bindings=BINDINGS)
    return Replay(context=runtime.context,store=BlobStore(runtime.store.root),services=evaluation.adapters,
                  bindings=BINDINGS,evaluation=evaluation)
