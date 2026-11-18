"""The public in-memory review recipe retains the released image bytes."""
import hashlib
import json
from pathlib import Path
from ora.agent.toolkit.interface.snapshot import open_replay
from ora.agent.toolkit.interface.types import digest
from ora.agent.toolkit.lib import observation, geometry, fusion, grasp_review, vision
from ora.agent.toolkit.lib.values import Array, JsonValue
from ora.agent.toolkit.host.grasp.x5_topdown import X5_TOPDOWN_COMPILER_IDENTITIES

FIXTURE = Path(__file__).resolve().parents[5]/'tests/data/toolkit-general-pickup'
# What the legacy renderers drew for this fixture: ora.agent.tools.grasp.review's
# _render_grasp_overlay, and _render_grasp_global_review given scene_extent_review.cell.
# Pixels are RGB uint8. Metadata is canonical JSON with floats rounded to 10
# significant digits: the observed-extent bounds differ in the last bit between
# BLAS kernels.
CONTACT_SHEET = dict(dtype='|u1', shape=[1080, 1440, 3],
                     sha256='fb1d78d7b166b2298d01655dffedccdd4f5079a73b0cfe8fb15ad5f949ec68d1',
                     metadata='f1dc3d0a620931f563d573b5276e5647bfa3dedd195999c78530bb9988ab346b')
GLOBAL_REVIEW = dict(dtype='|u1', shape=[768, 1168, 3],
                     sha256='2e52e83a1547f7e103f8964d929b5fa63434dc1f5f74534b2258206c42e58630',
                     metadata='49cd0470b3482e16ba0906d41806f38cc1ab53c65e65722f29de10c3a8d1272b')


def _rounded(value):
    if isinstance(value, float):
        return float(f'{value:.10g}')
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


def _record(image):
    array = image.rgb.numpy()
    text = json.dumps(_rounded(image.metadata.json()), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                metadata=hashlib.sha256(text.encode()).hexdigest())


def test_contact_sheet_and_extent_match_original(tmp_path):
    api = open_replay(run_root=FIXTURE, snapshot=json.loads((FIXTURE/'snapshot.json').read_bytes()),
                      output_root=tmp_path/'replay', services={}, bindings={})
    obs = observation.observe(api)
    refs = obs.camera('head')
    camera = observation.load_camera(api, refs)
    mask = Array.of((camera.depth_m.numpy() > .1) & (camera.depth_m.numpy() < 2.5))
    target = geometry.transform_points(geometry.mask_to_cloud(camera, mask), camera.camera_to_world)
    compiler = JsonValue.of(X5_TOPDOWN_COMPILER_IDENTITIES['left'])
    # Empty candidates exercises both the contact sheet and unclipped extent
    # independently of proposal policy; candidate projections are covered below.
    rows = JsonValue.of([])
    identity = obs.identity.json()['observation_id']
    new = grasp_review.contact_sheet(camera, mask, target, observation_id=identity,
        target_ref='target-test', candidates=rows, compiler=compiler,
        mask_sha256=digest(vision.npy_bytes(mask)), execution_tool='move_pose')
    assert _record(new) == CONTACT_SHEET
    source = fusion.FusionSource(camera, JsonValue.of(refs.declaration.json()['files']))
    workspace = JsonValue.of(dict(x=[-.75,.75],y=[-.65,.55],z=[.72,1.4]))
    fused = fusion.fuse((source,), workspace, voxel_size_m=.005, max_depth_m=2.5)
    labels = dict(observation_id=identity, target_ref='target-test', camera='head',
                  backend_input_scope='target_source_camera_rgbd_only', backend_cameras=['head'],
                  backend_points_by_camera={'head': len(fused.points.xyz_m.numpy())},
                  backend_excluded_cameras=[], policy_visible=True)
    actual = grasp_review.global_review((source,), fused, target, overlay=new, workspace=workspace,
        candidates=rows, compiler=compiler, labels=JsonValue.of(labels), observed_extent=True)
    assert _record(actual) == GLOBAL_REVIEW
