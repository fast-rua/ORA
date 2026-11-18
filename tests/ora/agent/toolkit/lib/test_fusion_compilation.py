"""Parameterized pure computations preserve the existing fusion/compiler recipes."""
import json
from pathlib import Path

import numpy as np
import pytest

from ora.agent.toolkit.interface.snapshot import open_replay
from ora.agent.toolkit.lib import observation, compilation, fusion
from ora.agent.toolkit.lib.values import JsonValue
from ora.agent.toolkit.host.shared.rgbd_fusion import fuse_rgbd_pointcloud_sources
from ora.agent.toolkit.host.grasp.compiler import (
    PANDA_GRIP_SITE_COMPILER_IDENTITY, ROBOTIQ_TCP_COMPILER_IDENTITY, _compile_grasp_candidate,
)
from ora.agent.toolkit.host.grasp.x5_topdown import X5_TOPDOWN_COMPILER_IDENTITIES

ROOT = Path(__file__).resolve().parents[5]
FIXTURE = ROOT/'tests/data/toolkit-general-pickup'


@pytest.mark.parametrize('voxel', [.005, .01])
def test_fusion_matches_original_bytes_and_attribution(tmp_path, voxel):
    snapshot = json.loads((FIXTURE/'snapshot.json').read_bytes())
    api = open_replay(run_root=FIXTURE, snapshot=snapshot, output_root=tmp_path, services={}, bindings={})
    obs = observation.observe(api)
    refs = obs.camera('head')
    camera = observation.load_camera(api, refs)
    files = refs.declaration.json()['files']
    files = {k: {field: value[field] for field in ('path', 'sha256', 'bytes')} for k, value in files.items()}
    bounds = dict(x=[-2, 2], y=[-2, 2], z=[-1, 3])
    actual = fusion.fuse((fusion.FusionSource(camera, JsonValue.of(files)),), JsonValue.of(bounds),
                         voxel_size_m=voxel, max_depth_m=2.5)
    old = fuse_rgbd_pointcloud_sources(observation_root=FIXTURE/'policy-observations'/obs.identity.json()['observation_id'],
        observation=obs.manifest.json(), workspace_bounds_m=bounds, camera_names=('head',), voxel_size_m=voxel, max_depth_m=2.5)
    for key, value in [('xyz_world_m', actual.points.xyz_m), ('rgb_u8', actual.rgb), ('source_index', actual.source_index)]:
        expected = old.pop(key)
        assert expected.dtype.str == value.numpy().dtype.str
        assert expected.shape == value.numpy().shape
        assert expected.tobytes() == value.numpy().tobytes()
    assert actual.metadata.json() == old
    with pytest.raises(ValueError, match='unique'):
        fusion.fuse((fusion.FusionSource(camera, JsonValue.of(files)),)*2, JsonValue.of(bounds))


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('public_pose', [False, True])
def test_x5_frame_compilation_matches_original(arm, public_pose):
    compiler = X5_TOPDOWN_COMPILER_IDENTITIES[arm]
    raw = dict(candidate_id='grasp-example', rank=0, source_gripper_origin_xyz_m=[0.,0.,1.],
        source_pregrasp_origin_xyz_m=[0.,0.,1.04], approach_world=[0.,0.,-1.], jaw_world=[1.,0.,0.],
        gripper_opening_m=.04, backend_score=.8, pose_convention=compiler['source_pose_convention'],
        checks=dict(workspace='passed', aperture='passed'))
    actual = compilation.compile_contact(JsonValue.of(raw), JsonValue.of(compiler), public_contact_pose=public_pose)
    assert actual.json() == _compile_grasp_candidate(raw, compiler_identity=compiler, public_contact_pose=public_pose)
    raw['jaw_world'] = raw['approach_world']
    with pytest.raises(ValueError, match='degenerate'):
        compilation.compile_contact(JsonValue.of(raw), JsonValue.of(compiler))


GRIP_SITE = [PANDA_GRIP_SITE_COMPILER_IDENTITY, ROBOTIQ_TCP_COMPILER_IDENTITY]


def _back(point, approach, distance=.04):
    return [value - axis*distance for value, axis in zip(point, approach)]


def _learned(compiler, *, approach=(0., 0., -1.), jaw=(1., 0., 0.), distance=.04, opening=.04):
    origin = [.1, -.2, 1.]
    return dict(candidate_id='grasp-example', rank=2, source_gripper_origin_xyz_m=origin,
        source_pregrasp_origin_xyz_m=_back(origin, approach, distance), approach_world=list(approach), jaw_world=list(jaw),
        gripper_opening_m=opening, backend_score=.8, pose_convention=compiler['source_pose_convention'],
        checks=dict(workspace='passed', aperture='passed'))


def _topdown(compiler, **overrides):
    centre = [.1, -.2, .9]
    return dict(_learned(compiler, **overrides), pose_convention='pca_topdown_parallel_jaw_center_v1',
                grasp_center_xyz_m=centre, pregrasp_center_xyz_m=_back(centre, overrides.get('approach', (0., 0., -1.))))


@pytest.mark.parametrize('compiler', GRIP_SITE, ids=lambda item: item['controlled_point'])
@pytest.mark.parametrize('raw', [_learned, _topdown])
@pytest.mark.parametrize('axes', [dict(), dict(approach=(.6, 0., -.8), jaw=(0., 1., 0.))])
def test_grip_site_compilation_matches_the_handlers(compiler, raw, axes):
    candidate = raw(compiler, **axes)
    assert compilation.compile_contact(JsonValue.of(candidate), JsonValue.of(compiler)).json() == (
        _compile_grasp_candidate(candidate, compiler_identity=compiler))


@pytest.mark.parametrize('compiler', GRIP_SITE, ids=lambda item: item['controlled_point'])
def test_a_grip_site_candidate_keeps_its_own_limits(compiler):
    wide = _learned(compiler, opening=.12)  # the proposal's admission owns the opening
    assert compilation.compile_contact(JsonValue.of(wide), JsonValue.of(compiler)).json()['gripper_opening_m'] == .12
    far = _learned(compiler, distance=.06)
    with pytest.raises(ValueError, match='exceeds bounded servo distance'):
        compilation.compile_contact(JsonValue.of(far), JsonValue.of(compiler))


def test_a_compiler_the_library_cannot_map_is_refused():
    compiler = dict(PANDA_GRIP_SITE_COMPILER_IDENTITY, id='unregistered', controlled_point='wrist_flange')
    with pytest.raises(ValueError, match='controlled-point mapping'):
        compilation.compile_contact(JsonValue.of(_learned(compiler)), JsonValue.of(compiler))
