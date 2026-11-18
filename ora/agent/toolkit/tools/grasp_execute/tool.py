"""Commit a prepared grasp and propose exactly one action; replay never executes it."""
import json
from ...interface import dry_run
from ...lib.observation import observe
from ...lib.values import PublicResult
from ...interface.types import canonical, digest
from ..errors import ToolRefusal
from ..vision_ground.tool import json_bytes
from ..grasp_plan._lift import _pose, _lifted, validate_prepared_targets
from . import _candidates, _measure, _precommit

CLAIM_BOUNDARY = ('one committed sequence of trusted executions and the public measurements '
    'taken after it; motion arrival and jaw separation beside visible extents '
    'are evidence, not grasp, placement, retention or official task success')


def run(api,args):
    """Replay: recorded precommit probe or 未记录; compare action AND prepared blob hashes.

    Geometry and sequence are computed locally. A different action/preparation
    is Divergence, never online execution. One invocation submits at most once.
    """
    config=api.artifacts.tool_context()
    execution=config['execution']
    if execution.get('mutation_refusal') is not None:
        raise ToolRefusal.from_public(execution['mutation_refusal'])
    if set(args)!={'candidate_id','rationale'}:
        raise ToolRefusal('grasp_execute requires candidate_id and rationale')
    rationale=args['rationale']
    if not isinstance(rationale,str) or not rationale.strip():
        raise ToolRefusal('rationale is required: state briefly why this candidate was chosen',field='rationale')
    if len(rationale.strip())>200:
        raise ToolRefusal('rationale must be at most 200 characters',field='rationale',received=len(rationale.strip()),constraints=dict(maximum=200))
    obs=observe(api);observed=obs.identity.json();oid=observed['observation_id']
    state=execution.get('ledger')
    if state is not None and state['phase'] not in {'completed','aborted','terminated'}:
        if execution['action_pending'] or state['observation_id']==oid:
            sequence=state['sequence'];done=len(state['completed'])
            raise ToolRefusal('a committed skill sequence is already in flight; resume or abandon it before starting another',
                code='skill_sequence_active',retryable=False,constraints=dict(skill=state['skill'],phase=state['phase'],
                    pending_stage=sequence[done] if done<len(sequence) else None))
    candidate_id=str(args['candidate_id'])
    resolved=_candidates.resolve(api,config,candidate_id,oid)
    contact=_pose(resolved['candidate'].get('contact_pose'),'contact_pose',candidate_id)
    prepared=execution['prepared_grasp']
    poses=dict(approach=contact)
    if prepared:
        poses.update(pregrasp=_pose(resolved['candidate'].get('pregrasp_pose'),'pregrasp_pose',candidate_id),
                     lift=_lifted(contact,execution['lift_m']))
        validate_prepared_targets(poses,execution['workspace_bounds_m'],candidate_id)
    localized,retention=_measure.selected_surface(api,obs,resolved['target_ref'],config)
    commitment=dict(skill='grasp_execute',candidate_id=candidate_id,arm=resolved['arm'],observation_id=oid,
        source_ref=resolved['target_ref'],poses=poses,grippers=dict(close=0.,**({'open':1.} if prepared else {})),
        localization=localized,semantics=_measure.semantics(api,config,resolved['target_ref']),
        **({'retention_reference':retention} if prepared else {}))
    if prepared and execution['precommit']:
        commitment['precommit_motion_check']=_precommit.check_pregrasp(api,commitment,execution,observed)
    sequence=['open','pregrasp','approach','close','lift','verify'] if prepared else ['approach','close','verify']
    if dry_run.requested():
        if config.get('synchronous_execution'):
            from . import _preview
            return _preview.record(api,config,commitment,sequence,obs)
        raise dry_run.Proposed(dict(action='execute_skill_sequence',arguments=dict(sequence=sequence),
            observation_id=oid,bound_observation_sha256=observed['observation_sha256'],
            action_id=f"action-{execution['generation']}"),commitment)
    if config.get('synchronous_execution'):
        from . import _stages
        result = _stages.run(api, config, commitment, sequence)
        if config.get("legacy_visible"):
            result["_runtime"]["legacy_submission"] = dict(
                success=True, status="submitted", skill="grasp_execute", candidate_id=candidate_id,
                committed_sequence=sequence, driving_stages=sequence,
                cost=dict(robot_decisions=1, dispatched_stage_actions=sum(stage != "verify" for stage in sequence)),
                **({"precommit_motion_check": commitment["precommit_motion_check"]}
                   if "precommit_motion_check" in commitment else {}),
                target_ref=commitment["source_ref"], claim_boundary=CLAIM_BOUNDARY)
        return result
    state=dict(schema='ora.manipulation-skill-ledger.v1',skill='grasp_execute',candidate_id=candidate_id,
        rationale=rationale.strip(),stop_before_commit=False,sequence=sequence,completed=[],phase='running',
        observation_id=oid,observation_sha256=observed['observation_sha256'])
    files={'ledger.json':json_bytes(state),'commitment.json':json_bytes(commitment)}
    identity=dict(observation_id=oid,candidate_id=candidate_id,
        ledger_sha256=digest(files['ledger.json']),commitment_sha256=digest(files['commitment.json']))
    key='skill-'+digest(canonical(identity))[:24]
    files['manifest.json']=json_bytes(dict(schema='ora.toolkit.skill-preparation.v1',identity=identity,key=key))
    record=api.artifacts.publish_tool_artifact(kind='skill_preparation',key=key,files=files,metadata=dict(observation_id=oid))
    action=dict(schema='ora-skill-sequence-action.v1',observation_id=oid,action_id=f"action-{execution['generation']}",
        action='execute_skill_sequence',arguments=dict(skill='grasp_execute',candidate_id=candidate_id,
            rationale=state['rationale'],stop_before_commit=False,sequence=sequence,resume_from=0),
        bound_observation_sha256=observed['observation_sha256'])
    PublicResult.of(api.act.submit_action(action=action,preparation={
        'skill_ledger':record['files']['ledger.json'],'skill_commitment':record['files']['commitment.json']})).data
    return dict(success=True,status='submitted',skill='grasp_execute',candidate_id=candidate_id,
        committed_sequence=sequence,driving_stages=sequence,cost=dict(robot_decisions=1,
            dispatched_stage_actions=sum(stage!='verify' for stage in sequence)),
        **({'precommit_motion_check':commitment['precommit_motion_check']} if 'precommit_motion_check' in commitment else {}),
        target_ref=commitment['source_ref'],claim_boundary=CLAIM_BOUNDARY,
        _runtime=dict(directive='action_submitted',action_id=action['action_id']))
