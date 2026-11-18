"""Selected-pregrasp admission (``grasp_execute/_precommit.py``) and mode-companion delivery without simulator steps."""
import json

import pytest

from ora.agent.composition import resolve_tool_surface
from tests.support.agent.surface_catalog import surface_of
from ora.agent.toolkit.tools.grasp_execute import _precommit as precommit
from ora.agent.toolkit.host.motion_rules import arguments as motion_arguments
from ora.agent.error_classes import ToolError


@pytest.mark.parametrize('profile', ['robotwin_toolchain_b2_closed_noservo_v2',
    'robodojo_placement_toolkit_v5'])
def test_current_defaults_deliver_mode_companions_and_precommit_check(profile):
    surface = surface_of(profile)
    assert surface.supports(precommit.CAPABILITY)
    move = next(t for t in surface.definitions if t['name'] == 'move_pose')
    props = move['inputSchema']['properties']
    targets = props.get('targets', {}).get('properties', {'right': move['inputSchema']})
    for schema in targets.values():
        fields = schema['properties']
        assert 'translation_m AND position_frame' in fields['position_mode']['description']
        assert 'orientation_frame' in fields['orientation_mode']['description']
        assert 'Never inferred' in fields['position_frame']['description']
    grasp = next(t for t in surface.definitions if t['name'] == 'grasp_execute')
    assert 'before opening' in grasp['description'].lower() or 'Before opening' in grasp['description']


def test_all_missing_companions_are_returned_in_one_refusal():
    with pytest.raises(ToolError) as exc:
        motion_arguments.arm_target({'position_mode': 'delta',
            'orientation_mode': 'axis_angle_delta'}, arm='targets.left')
    assert exc.value.field == 'targets.left.translation_m'
    assert exc.value.constraints['missing_fields'] == ['translation_m', 'position_frame',
        'rotation_axis', 'rotation_angle_deg', 'orientation_frame']
    assert exc.value.constraints['effect'] == 'no_action_submitted'


def test_the_compiled_pose_is_recognised_in_its_normalised_backend_echo():
    # motion.probe normalises and the backend may sign-flip the quaternion;
    # prepared geometry admits a norm within 1e-3 of one.
    compiled = [.2, .1, .9, 0., 0., 0., .9995]
    assert precommit._same_pose(compiled, [.2, .1, .9, 0., 0., 0., 1.])
    assert precommit._same_pose(compiled, [.2, .1, .9, -0., -0., -0., -1.])
    assert not precommit._same_pose(compiled, [.2, .1, .9, 0., 0., 1., 0.])
    assert not precommit._same_pose(compiled, [.2, .1, .91, 0., 0., 0., 1.])


def test_terminal_settle_preserves_joint_failure_when_tcp_arrived():
    from ora.agent.session_core.receipt_projection import _public_terminal_settle, public_motion_failure_code
    result = _public_terminal_settle({'segment': {'execution': {
        'tracking_error_max_abs_rad': .01,
        'terminal_settle': {'joint_tolerance_rad': .005, 'ticks': 64, 'max_ticks': 64,
            'exhausted': True, 'endpoint_error': {'position_m': .0001, 'orientation_rad': .001, 'object_gt': 'SECRET'},
            'endpoint_tolerance': {'position_m': .004, 'orientation_rad': .02}},
    }}})['terminal_settle']
    assert result['joint_within_tolerance'] is False
    assert result['budget_exhausted'] is True
    assert result['joint_error_max_abs_rad'] == .01
    assert result['tcp_endpoint_error']['position_m'] == .0001
    assert 'SECRET' not in json.dumps(result)
    assert public_motion_failure_code('execution', 'terminal_tracking_timeout') == 'controller_timeout'


def test_rejected_link_spheres_keep_names_and_world_centers_through_public_projection():
    from tests.support.robotwin.contact_recovery import binding, path
    from ora.environment.backends.robotwin.planning import validate_inter_arm_clearance, RoboTwinPlanRejected
    from ora.agent.evidence.plan_feedback import public_plan_validation
    with pytest.raises(RoboTwinPlanRejected) as exc:
        validate_inter_arm_clearance({'right': path([.2, .09])},
            measured_joints={'left': [0, 0, 1], 'right': [.2, 0, 1]},
            fk_chain_points=lambda arm, joints: [list(joints)],
            profile=binding()['profile']['dual_arm_collision_check'],
            link_names={'left': ['fl_link6'], 'right': ['fr_link6']})
    details = exc.value.details
    details['link_identity']['object_gt'] = 'SECRET'
    result = public_plan_validation({'status': 'rejected', 'code': exc.value.code, 'details': details})
    spheres = result['clearance']['link_identity']
    assert spheres['frame'] == 'world'
    assert spheres['left']['name'] == 'fl_link6'
    assert spheres['right']['name'] == 'fr_link6'
    assert spheres['left']['center_m'] == [0, 0, 1]
    assert spheres['right']['center_m'] == [.09, 0, 1]
    assert spheres['left']['radius_m'] == .045
    assert 'SECRET' not in json.dumps(result)


@pytest.mark.parametrize('learned', [False, True])
def test_aloha_oversize_rejection_discloses_its_compiler_bound(monkeypatch, learned):
    import numpy as np
    from types import SimpleNamespace
    from ora.agent.model_services import clients
    from ora.agent.toolkit.host.grasp import aloha_gripper
    from ora.agent.toolkit.host.grasp import compiler_registry
    from ora.agent.toolkit.host.grasp import contact_graspnet_client
    from ora.agent.toolkit.host.grasp import proposers
    # The bound is derived from the pinned URDF and config rather than written
    # as a literal, so the test asks for the ALOHA compiler by name and reads
    # the number off it. Pinning the digits here is what made the previous bound
    # look deliberate when it was a stale literal.
    bound = aloha_gripper.ALOHA_NOMINAL_MAXIMUM_OPENING_M
    compiler = next(dict(v) for v in compiler_registry.GRASP_COMPILER_IDENTITIES.values()
                    if v.get('maximum_opening_m') == bound
                    and v.get('embodiment_descriptor_id') == 'aloha_agilex')
    x, y = np.meshgrid(np.linspace(-.1, .1, 12), np.linspace(-.1, .1, 12))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, .9)])
    if learned:
        proposer = contact_graspnet_client.ContactGraspNetApiProposer()
        body = {'schema': 'ora.model-service.grasp-proposals.v1',
            'request_id': 'offline', 'metrics': {'queue_ms': 0, 'inference_ms': 0, 'total_ms': 0},
            'poses_camera_npy_base64': clients._encode_npy(np.eye(4)[None]),
            'scores_npy_base64': clients._encode_npy(np.array([.9])),
            'openings_m_npy_base64': clients._encode_npy(np.array([.12])),
            'contact_points_camera_npy_base64': clients._encode_npy(np.zeros((1, 3)))}
        monkeypatch.setattr(contact_graspnet_client, "_require_response_service_identity", lambda **kwargs: None)
        monkeypatch.setattr(contact_graspnet_client, "_verify_model_identity", lambda *args: {})
        monkeypatch.setattr(proposer._client, 'post', lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: body))
    else:
        proposer = proposers.PcaTopDownGraspProposer()
    candidates, diagnostics = proposer.propose(scene_points_camera_m=points,
        target_points_camera_m=points, camera_to_world=np.eye(4),
        target_ref='test', workspace_bounds_m={'x': [-2, 2], 'y': [-2, 2], 'z': [-2, 2]},
        max_candidates=3, compiler_identity=compiler)
    assert not candidates
    assert diagnostics['rejected']
    for rejection in diagnostics['rejected']:
        assert rejection['maximum_opening_m'] == bound
        assert rejection['predicted_opening_m' if learned else 'required_opening_m'] > bound
        assert rejection['limit_semantics'] == 'compiler_admission_not_measured_physical_aperture'


def test_both_arm_omissions_preserve_values_and_accept_one_complete_correction():
    from copy import deepcopy
    request = {'left': {'position_mode': 'delta', 'translation_m': [0, 0, -.04],
                       'orientation_mode': 'hold_current'},
               'right': {'position_mode': 'hold_current', 'orientation_mode': 'axis_angle_delta',
                         'rotation_axis': 'z', 'rotation_angle_deg': 15}}
    original = deepcopy(request)
    with pytest.raises(ToolError) as exc:
        motion_arguments.arm_targets(request)
    detail = exc.value.constraints
    assert detail['missing_paths'] == ['targets.left.position_frame', 'targets.right.orientation_frame']
    assert detail['preserved_targets'] == original == request
    assert detail['effect'] == 'no_action_submitted'
    corrected = deepcopy(detail['preserved_targets'])
    corrected['left']['position_frame'] = 'world'
    corrected['right']['orientation_frame'] = 'end_effector'
    result = motion_arguments.arm_targets(corrected)
    assert result['left']['position']['translation_m'] == [0, 0, -.04]
    assert result['right']['orientation']['angle_deg'] == 15


def test_developing_dojo_default_delivers_source_controller_rules():
    from pathlib import Path
    from ora.environment.controllers.robodojo_arx import public_feedback_contract
    root = Path(__file__).resolve().parents[5]
    # The RoboDojo default instance (integration 0059 unit A0) replaced the
    # developing-surface entry; it restates angles in degrees.
    requirement = json.loads((root / 'tests/data/default-expectations.json').read_text())['instances']['robodojo']
    from ora.agent.composition.surface.toolkit_surface import declared_required_grants
    surface = resolve_tool_surface(requirement['tool_surface'],
                                   toolkit_grants=declared_required_grants(requirement['tool_surface']))
    assert all(surface.supports(cap) for cap in requirement['required_capabilities'])
    gripper = next(t for t in surface.definitions if t['name'] == 'set_gripper')
    delivered, _ = json.JSONDecoder().raw_decode(gripper['description'].split('Controller feedback contract: ', 1)[1])
    from ora.agent.composition.surface.units import policy_degrees
    assert surface.supports('policy_angles_degrees_v1')
    assert delivered == policy_degrees(public_feedback_contract())
    from ora.contracts.robodojo_arx import ARX_X5_CONTRACT
    from ora.environment.controllers.robodojo_arx import ARX_X5_CONTRACT as executor_contract
    assert executor_contract is ARX_X5_CONTRACT
    assert delivered['gripper_tolerance_normalized'] == .03
    assert delivered['gripper_drive_limit']['maximum_normalized_offset'] == .2
