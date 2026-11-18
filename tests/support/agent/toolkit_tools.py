"""Public recorded pixels and in-process fake service transport for tool equivalence."""
import json
import shutil
from dataclasses import replace
from pathlib import Path
import httpx
import numpy as np
from ora.agent.model_services.clients import CONTACT_GRASPNET_MODEL_IDENTITY, SAM3_MODEL_IDENTITY, _encode_npy
from ora.agent.toolkit.host.grasp.contact_graspnet_client import ContactGraspNetApiProposer
from ora.agent.toolkit.host.grasp.proposers import AlohaHybridGraspProposer, PcaTopDownGraspProposer
from ora.agent.toolkit.host.runtime import ManipulationToolRuntime
from ora.agent.toolkit.interface.types import canonical, digest, CallContext
from ora.agent.toolkit.interface.blobs import BlobStore
from ora.agent.toolkit.interface.journal import Journal
from ora.agent.toolkit.host.adapter import bind_runtime
from ora.agent.toolkit.host.runtime.dispatch import install as install_toolkit
from tests.support.agent.surface_catalog import handlers_only
from ora.agent.toolkit.host.runtime.tool_data import bind

ROOT=Path(__file__).resolve().parents[3]
FIXTURE=ROOT/'tests/data/toolkit-general-pickup'
#: The grants toolkit v5 requires (its surface.json ``required_grants``).
V5_GRANTS=dict(tools=['check_task'],state_keys=['privileged.task_check'])

def use_toolkit_v5(host):
    """Put ``host`` on RoboDojo's toolkit surface v5 with the grants it requires (the toolkit tools' test surface since
    the layered v2 retired on 2026-11-01). The caller sets the executor and installs the toolkit."""
    from ora.agent.composition.surface.tool_surface import resolve_tool_surface
    host.toolkit_grants=V5_GRANTS
    host.surface=resolve_tool_surface('robodojo_placement_toolkit_v5',toolkit_grants=V5_GRANTS)
    host.allowed_tools=tuple(d['name'] for d in host.surface.definitions)
    return host

def install(host):
    """Move ``host`` from the handlers to the toolkit on the same surface."""
    if not host.surface.supports('toolkit_only_v1'):
        host.surface=replace(host.surface,capabilities=(*host.surface.capabilities,'toolkit_only_v1'))
    install_toolkit(host)

class Grounder:
    backend_id='sam3';backend_version='test'
    expected_binding={'fixture':'sam'}
    def configuration(self):return dict(endpoint='http://fake.invalid',test=True)
    def ground(self, *,rgb_u8,text,max_candidates):
        mask=np.zeros(rgb_u8.shape[:2],bool);mask[180:210,290:310]=True
        return [dict(rank=0,source_index=0,backend_score=.9,bbox_xyxy=[290,180,310,210],mask=mask)],dict(
            request_id='fixture',returned_candidate_count=1,_service_identity={'test':True},model=SAM3_MODEL_IDENTITY,latency_ms=dict(roundtrip=1.,service_queue=0.,service_inference=1.,service_total=1.))


def grasp_service(received,count=3):
    """The in-process grasp service's transport; ``received`` collects its requests."""
    def transport(request):
        received.append(json.loads(request.content))
        poses=np.tile(np.eye(4),(count,1,1));poses[:,2,3]=.7
        scores=np.linspace(.9,.5,count);openings=np.full(count,.04)
        return httpx.Response(200,json=dict(schema='ora.model-service.grasp-proposals.v1',request_id='fixed',
            model=CONTACT_GRASPNET_MODEL_IDENTITY,service_identity=dict(schema='ora.model-service-response-identity.v1',
            instance_id='fake',pid=123,started_unix_s=1.,adapter_sha256='a'*64),
            poses_camera_npy_base64=_encode_npy(poses),scores_npy_base64=_encode_npy(scores),
            openings_m_npy_base64=_encode_npy(openings),contact_points_camera_npy_base64=_encode_npy(poses[:,:3,3]),
            metrics=dict(queue_ms=.2,inference_ms=20.,total_ms=20.2)))
    return transport


def make_host(root, monkeypatch, count=3):
    shutil.copytree(FIXTURE,root,dirs_exist_ok=True)
    snapshot=json.loads((root/'snapshot.json').read_bytes())
    oid=snapshot['available_at_boundary']['observation']['id']
    observation_root=root/'policy-observations'/oid
    control=root/'control';control.mkdir()
    pointer=dict(open=True,observation_id=oid,observation_sha256=digest((observation_root/'observation.json').read_bytes()),
        source_root=str(observation_root),generation=0)
    (control/'current-observation.json').write_bytes(canonical(pointer))
    received=[]
    transport=grasp_service(received,count)
    client=ContactGraspNetApiProposer(endpoint='http://127.0.0.1:1',hover_offset_m=.04,
        expected_binding=dict(schema='ora.grasp-proposal-service-binding.v1',tool_surface='robodojo_placement_toolkit_v5',
            endpoint='http://127.0.0.1:1',instance_id='fake',pid=123,started_unix_s=1.,adapter_sha256='a'*64,
            model=CONTACT_GRASPNET_MODEL_IDENTITY,load_count=1,warmup_complete=True,
            qualification=dict(status='passed',sha256='b'*64,created_unix_s=1.)))
    client._client.close();object.__setattr__(client,'_client',httpx.Client(transport=httpx.MockTransport(transport)))
    monkeypatch.setattr('ora.agent.toolkit.host.grasp.contact_graspnet_client.time.perf_counter',lambda:1.)
    monkeypatch.setattr('ora.agent.toolkit.tools.grasp_plan._reach.time.monotonic',lambda:1.)
    monkeypatch.setattr(ContactGraspNetApiProposer, 'require_ready', lambda self: {'fixture': True})
    backend=AlohaHybridGraspProposer(client,PcaTopDownGraspProposer())
    host=ManipulationToolRuntime(control_root=control,manual_path=ROOT/'tests/data/prompts/tool-schema-delivery-v1.md',
        event_path=root/'events.jsonl',workspace_bounds_m=dict(x=[-.75,.75],y=[-.65,.55],z=[.72,1.4]),
        state_root=None,tool_surface_profile='robodojo_placement_toolkit_v5',toolkit_grants=V5_GRANTS,controlled_end_effector='right',
        segmentation_backend=Grounder(),grounding_backend=Grounder(),grasp_backend=backend)
    def reach(request):
        return dict(schema='ora.reach-check-result.v1',arm=request['arm'],ik_model='fixture',
            matches_move_pose_admission=True,motion_planner_called=False,claim_boundary='fake IK only',
            results=[dict(reachable=True,reason=None,route='global',position_error_m=0.,orientation_error_rad=0.,
                joint_limit_margin_rad=.7,joint_delta_rad=.2) for _ in request['queries']])
    host.align_plan_callback=reach
    return handlers_only(host),client,received


def api_for(host,tool_call_id='call-0',inputs=None):
    control,root,obs=host._current()
    context=CallContext.create(run_id='fixture',round_index=0,tool_call_id=tool_call_id,
        boundary=dict(kind='before_call',call=0),state=dict(observation=dict(observation_id=control['observation_id'],
        observation_sha256=control['observation_sha256'])),check_task=False)
    api=bind_runtime(runtime=host,context=context,store=BlobStore(host.control_root.parent,writable=True),
        journal=Journal(host.control_root.parent),bindings={},sam3=host.grounding_backend,
        grasp=host.grasp_backend.contact_backend)
    from ora.agent.toolkit.host.runtime.inputs import freeze
    from ora.agent.toolkit.host.runtime.tool_data import publish
    from ora.agent.toolkit.host.execution import submit_adapter
    from ora.agent.toolkit.interface.tool_data import ToolData
    config,granted=freeze(host,api.store)
    api.tool_data=ToolData(store=api.store,configuration=config,inputs=granted if inputs is None else inputs,
        publisher=lambda *args:publish(host,*args))
    api.adapters['submit_action']=submit_adapter(host)
    return api
