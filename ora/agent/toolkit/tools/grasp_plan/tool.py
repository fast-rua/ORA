"""Plan against published RGB-D, a newly requested learned pool and recorded IK only."""
import base64
import json
import re
from ...lib import observation
from ...lib.values import PublicResult
from ...interface.types import InterfaceError, PublicToolFailure
from ...lib._grasp_frames import GRIP_SITE_STYLE_COMPILER_IDS
from ...lib._pose_math import quaternion_matrix
from ..errors import ToolRefusal
from . import _scene, _reach, _jaw, _materialize
from ._topdown import TargetInsufficient

#: A surface whose one arm is the right arm takes no ``arm`` and publishes none. A context without the capability, every
#: RoboDojo and sealed one, plans for the arm a call names.
SINGLE_ARM = 'single_arm_flat_targets_v1'


def target_input(api, target_ref, obs, configuration):
    observed = obs.identity.json()['observation_id']
    malformed = not isinstance(target_ref, str) or re.fullmatch(r'target-[0-9a-f]{24}',target_ref) is None
    record = None if malformed else api.artifacts.tool_artifact(kind='target',key=target_ref)
    if record is None:
        refs = configuration.get('current_target_refs', [])
        text = 'target_ref is malformed: not a segmented-target reference' if malformed else 'target_ref is unavailable: no segmented target in this run carries it'
        raise ToolRefusal('target_ref: '+text,code='target_ref_unavailable',field='target_ref',received=target_ref,
            constraints=dict(pattern='^target-[0-9a-f]{24}$',current_observation_id=observed,
                current_target_refs=refs[:32],current_target_ref_count=len(refs)))
    target = json.loads(api.artifacts.read_blob(record['files']['manifest.json']))
    identity=target.get('identity',{})
    if identity.get('observation_id') != observed:
        bound=identity.get('observation_id')
        raise ToolRefusal(f'target_ref is stale: it was segmented on observation {bound}, and the current observation is {observed}; '
            'a target_ref is valid only on the observation it was segmented on',code='stale_observation',field='target_ref',
            received=target_ref,retry_requires_changed_arguments=True,
            constraints=dict(bound_observation_id=bound,current_observation_id=observed))
    if not isinstance(identity.get('camera'),str):
        raise ToolRefusal('target_ref lacks a source camera',code='artifact_hash_mismatch',retryable=False)
    return target, record


def selection(api, config, arm, obs):
    if not config['reach_annotation']:
        return None
    identity=dict(schema='ora.grasp-ik-candidate-selection.v1',pool_limit=_reach.POOL_LIMIT,order=_reach.ORDER,
        members=list(_reach.MEMBERS),vertical_preference_deg=_reach.VERTICAL_PREFERENCE_DEG,arm=arm)
    rule=None
    if config['jaw_symmetry']:
        bound=_jaw.ISOLATION_BOUNDS.get(config['controlled_frame'])
        rule=dict(identity=dict(capability=_jaw.CAPABILITY,rule=_jaw.RULE,margin_tie_rad=_jaw.MARGIN_TIE_RAD,
            joint_delta_tie_rad=_jaw.DELTA_TIE_RAD,joint6_isolation_bound_rad=bound),isolation_bound_rad=bound,
            measured_joints=_jaw.measured_arm_joints(obs.manifest.json(),arm) if bound is not None else None)
        identity['jaw_symmetry']=rule['identity']
    def choose(compiled):
        return _reach.select_grasp_candidates(compiled,arm=arm,
            bridge=lambda request: PublicResult.of(api.planner.reach_check(request=request)).data.json(),
            workspace_bounds_m=config['workspace_bounds_m'],approach_axis_local=config['approach_axis'],jaw_symmetry=rule)
    return dict(identity=identity,select=choose)


def run(api,args):
    """Replay: recorded observation/IK or 未记录; CGN re-requested, geometry recomputed.

    Returns the complete legacy public result; it submits no robot action.
    """
    configuration=api.artifacts.tool_context()
    single=SINGLE_ARM in (configuration.get('motion') or {}).get('capabilities',())
    allowed={'target_ref','max_candidates'}|(set() if single else {'arm'})
    if set(args)-allowed or not allowed-{'max_candidates'} <= set(args):
        raise ToolRefusal('grasp_plan requires target_ref; the Runtime binds the current observation automatically')
    obs=observation.observe(api)
    config=configuration['grasp']
    target_ref,maximum=args['target_ref'],args.get('max_candidates',3)
    if not isinstance(target_ref,str) or not target_ref:
        raise ToolRefusal('target_ref must be a non-empty string',field='target_ref')
    if type(maximum) is not int or not 1<=maximum<=3:
        raise ToolRefusal('max_candidates must be an integer in [1,3]',field='max_candidates',received=maximum)
    target,record=target_input(api,target_ref,obs,configuration)
    arm='right' if single else args['arm']
    if arm not in {'left','right'}:
        raise ToolRefusal('grasp_plan arm must be left or right',field='arm')
    camera=target['identity']['camera']
    names=config['scene_cameras'][arm]
    def usable(name):
        return _scene.fuse_rgbd_pointcloud_sources(api=api,observation=obs,workspace_bounds_m=config['workspace_bounds_m'],
            camera_names=(name,),voxel_size_m=.005,max_depth_m=2.5)['sources'][0]
    try:
        primary=usable(camera)
    except (ValueError,TypeError,KeyError,OSError,EOFError) as error:
        if isinstance(error,InterfaceError): raise
        raise ToolRefusal(f'grasp primary RGB-D source is unusable: camera {camera} of target_ref {target_ref}: {error}',
                          code='grasp_proposal_failed') from error
    binding=dict(requested_cameras=names,available_sources=[],unavailable_sources=[])
    available,unavailable=[],[]
    for name in names:
        try:
            source=primary if name==camera else usable(name)
        except (ValueError,TypeError,KeyError,OSError,EOFError):
            unavailable.append(name)
            binding['unavailable_sources'].append(dict(camera=name,reason='unusable_public_rgbd_review_source'))
        else:
            available.append(name);binding['available_sources'].append(source)
    try:
        manifest,record=_materialize.materialize(api,obs,target,record,maximum=maximum,config=config,
            compiler=config['compilers'][arm],selection=selection(api,config,arm,obs),
            review_names=tuple(available),review_binding=binding)
    except PublicToolFailure as error:
        raise ToolRefusal.from_public(error.public()) from error
    except InterfaceError:
        raise
    except TargetInsufficient as error:
        raise ToolRefusal(str(error),code='grasp_proposal_failed',field='target_ref',received=target_ref,
            retry_requires_changed_arguments=True) from error
    except ValueError as error:
        raise ToolRefusal(str(error),code='grasp_proposal_failed') from error
    return public_result(api,manifest,record,config,None if single else arm,names,available,unavailable)


def public_compiler(compiler,arm):
    """A grip-site compiler publishes its complete registered identity; the others a restricted projection for their arm."""
    if compiler['id'] in GRIP_SITE_STYLE_COMPILER_IDS:
        return dict(compiler)
    public={k:compiler[k] for k in ('id','embodiment_descriptor_id','controlled_point','controlled_point_semantics')}
    public['arm']=arm
    for key in ('physical_jaw_aperture_calibrated','aperture_caveat','maximum_opening_source','predicted_opening_source'):
        if key in compiler: public[key]=compiler[key]
    return public


def world_axes(orientation_xyzw,local):
    """``approach_world`` and ``jaw_line_world`` of one pose: its rotation applied to the controlled point's local
    approach and closing axes, the two directions move_pose's world_axes mode takes."""
    matrix=quaternion_matrix(orientation_xyzw)
    def rotate(axis):return [round(sum(matrix[row][k]*float(axis[k]) for k in range(3)),6)+0.0 for row in range(3)]
    return dict(approach_world=rotate(local['approach']),jaw_line_world=rotate(local['closing']))


def public_result(api,manifest,record,config,arm,names,available,unavailable):
    """``arm`` is the arm a dual-arm surface planned for, and None on a single arm, whose result does not name it."""
    identity=manifest['identity']
    candidates=[dict(row) for row in manifest['compiled_candidates']]
    result=dict(success=True,status='candidates' if candidates else 'no_candidate',
        observation_id=identity['observation_id'],target_ref=identity['target_ref'],camera=identity['camera'],
        grasp_query_ref=manifest['query_ref'],backend=manifest['backend'],compiler=public_compiler(identity['compiler'],arm),
        candidates=candidates,diagnostics=manifest['diagnostics'],
        overlay=dict(sha256=manifest['overlay']['sha256'],width=1440,height=1080))
    if arm is not None:result['arm']=arm
    review=manifest['review_visualization']
    claim='optional_review_only;primary_rgbd_input_remains_required;not_grasp_success'
    if config['fused_backend']:
        backend=manifest['input_pointcloud']['backend_input_profile']
        claim='primary_rgbd_input_remains_required;backend_input_scope='+backend['scope']+';backend_cameras='+','.join(backend.get('contributing_cameras',[]))+';review_is_not_grasp_success'
    result['review_evidence']=dict(status='available' if review.get('status')=='available' else 'unavailable',
        primary_camera=identity['camera'],requested_cameras=names,available_cameras=available,unavailable_cameras=unavailable,
        primary_input_required=True,claim_boundary=claim)
    if config['reach_annotation']:result['ik_reachability']=manifest['candidate_selection']['summary']
    axes=config.get('pose_world_axes')  # RO-27: only a grasp_pose_world_axes_v1 surface has the key
    for row in candidates:
        if config['execution_tool']=='move_pose':row['compiler_checks']['reachability']='delegated_to_move_pose'
        for key in ('pregrasp_pose','contact_pose') if axes else ():
            if key in row:row[key]=dict(row[key],**world_axes(row[key]['orientation_xyzw'],axes))
        if config['copyable_approach']:
            delta=row['approach_delta']
            if config['flat_targets']:
                value=dict(position_mode='delta',translation_m=delta['translation_m'],position_frame='end_effector',orientation_mode='hold_current')
            else:
                value=dict(position=dict(mode='delta',translation_m=delta['translation_m'],frame='end_effector'),orientation=dict(mode='hold_current'))
            if config['semantic_interaction']:value['interaction']='contact_approach'
            row['approach_delta']=value
    result['images_base64']=[base64.b64encode(api.artifacts.read_blob(record['files']['overlay.png'])).decode('ascii')]
    if config['perception_feedback']:
        def label(index,role):return dict(image_index=index,role=role,camera='',candidate_id='',target_ref='',observation_step=None)
        result['image_mapping']=[label(0,'local_contact_candidates')]
        result['scene_review']=dict(status='unavailable',reason='public_scene_review_unavailable',
            scope='observed_public_rgbd_only;unobserved_space_unknown;axes_and_ids_match_local_candidates')
        if review.get('status')=='available':
            result['images_base64'].append(base64.b64encode(api.artifacts.read_blob(record['files'][review['artifact']['path']])).decode('ascii'))
            result['image_mapping'].append(label(1,'global_public_scene_candidates'))
            result['scene_review'].update(status='available',reason='public_source_bound_review')
    return result
