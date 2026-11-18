"""One text query; masks, measurements and derived artifacts stay public-only.

Replay reads the published observation, re-requests grounding, and computes
measurements/images locally. It never samples the simulator or reads truth.
"""
import base64
import hashlib
import json
import math

import numpy as np

from ...lib import _vision_footprint, observation, perception, vision
from ...lib.blobs import read_json
from ...lib.values import Array, JsonValue
from ...interface.stages import stage
from ...interface.types import InterfaceError, PublicToolFailure, canonical, digest
from ..errors import ToolRefusal

RENDERER = 'grounding-candidate-inspection.v2'
PROFILE = 'grounding-candidate-depth-profile.v1'
LOCALIZATION = 'mask_depth_primary_component_robust_pca_v1'


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()


def artifact(name, data):
    return dict(path=name, sha256=digest(data), bytes=len(data))


def source_identity(obs, camera, refs):
    return dict(policy_observation_path=f'policy-observations/{obs}/{camera}', files={
        name: dict(name=filename, sha256=ref.json()['sha256'], bytes=ref.json()['bytes'])
        for name, filename, ref in [('rgb', 'rgb.png', refs.rgb), ('depth', 'depth-m.npy', refs.depth_m),
                                    ('calibration', 'calibration.json', refs.calibration)]})


def materialize(api, obs, camera, *, text, maximum, config):
    refs = obs.camera(camera)
    pixels = observation.load_camera(api, refs)
    # The old measurement recipe casts depth to float64 before measuring.
    depth = Array.of(pixels.depth_m.numpy().astype(np.float64))
    observed = obs.identity.json()['observation_id']
    source = source_identity(observed, camera, refs)
    backend = config['grounding_backend']
    profile = config.get('depth_profile')
    identity = dict(observation_id=observed, camera=camera, text=text, max_candidates=maximum,
                    source=source, backend=backend, visualization=dict(renderer=RENDERER,
                        **({} if profile is None else dict(depth_profile=profile))))
    key = 'grounding-query-' + digest(canonical(identity))[:24]
    cached = api.artifacts.tool_artifact(kind='grounding_query', key=key, cache=True)
    if cached is not None:
        manifest = json.loads(api.artifacts.read_blob(cached['files']['manifest.json']))
        if manifest.get('identity') != identity or manifest.get('schema') != 'ora.grounding-query.v2':
            raise ValueError('grounding query identity drift')
        for candidate in manifest['candidates']:
            if api.artifacts.tool_artifact(kind='target', key=candidate['target_ref']) is None:
                raise ValueError('cached grounding target unavailable')
        stage('query_cache', hit=True)
        return manifest, cached, pixels
    stage('query_cache', hit=False)
    segmented = perception.ground(api, rgb=JsonValue.of(api.artifacts.save_array(pixels.rgb.numpy())),
                                  text=text, max_candidates=maximum)
    diagnostics = segmented.metadata.json()
    service_identity = diagnostics.pop('service_identity', None)
    candidates, rendered, files = [], [], {}
    for raw in segmented.candidates[:maximum]:
        mask = Array.of(raw.mask.numpy().astype(bool))
        if mask.numpy().shape != depth.numpy().shape:
            raise ValueError('grounded mask shape differs from aligned depth')
        fields = raw.fields.json()
        summary = vision.mask_summary(mask, depth).json()
        localized = vision.localize_mask(mask, depth, pixels.calibration).json()
        target_identity = dict(observation_id=observed, camera=camera, prompt_kind='text', text=text,
            source_index=int(fields['source_index']), mask_sha256=hashlib.sha256(
                mask.numpy().astype(np.uint8).tobytes(order='C')).hexdigest(), source=source,
            backend=backend, localization_method=LOCALIZATION)
        target_key = 'target-' + digest(canonical(target_identity))[:24]
        drawing = perception.MaskCandidate(mask, JsonValue.of(dict(rank=int(fields['rank']),
            bbox_xyxy=[int(v) for v in fields['bbox_xyxy']], target_ref=target_key,
            backend_score=float(fields['backend_score']))))
        target_files = dict()
        target_files['mask.npy'] = vision.npy_bytes(Array.of(mask.numpy().astype(np.uint8)))
        target_files['overlay.png'] = vision.png_bytes(vision.mask_overview(pixels.rgb, (drawing,)))
        target = dict(schema='ora.segmented-target.v2', target_ref=target_key, identity=target_identity,
            prompt=dict(kind='text', text=text, source_index=int(fields['source_index'])), backend=backend,
            summary=summary, localization=localized, diagnostics=dict(candidate_metadata_scope='grounding_query_manifest'),
            private_mask=artifact('mask.npy', target_files['mask.npy']), overlay=artifact('overlay.png', target_files['overlay.png']))
        target_files['manifest.json'] = json_bytes(target)
        api.artifacts.publish_tool_artifact(kind='target', key=target_key, files=target_files,
                                  metadata=dict(observation_id=observed, camera=camera))
        candidates.append(dict(rank=int(fields['rank']), target_ref=target_key,
            backend_score=float(fields['backend_score']), bbox_xyxy=[int(v) for v in fields['bbox_xyxy']],
            mask_summary=summary, localization=localized, overlay_sha256=target['overlay']['sha256']))
        rendered.append(drawing)
    files['overlay.png'] = vision.png_bytes(vision.mask_overview(pixels.rgb, tuple(rendered)))
    previews = []
    for drawing, candidate in zip(rendered[:3], candidates[:3], strict=True):
        card = vision.candidate_inspection(pixels.rgb, drawing, text=text,
            **({} if profile is None else dict(depth_m=depth, calibration=pixels.calibration,
                localization=JsonValue.of(candidate['localization']), support_line=profile['support_line'])))
        name = f"candidate-T{candidate['rank']}.png"
        files[name] = vision.png_bytes(card.rgb)
        previews.append(dict(card.metadata.json(), artifact=artifact(name, files[name])))
    overlay = artifact('overlay.png', files['overlay.png'])
    manifest = dict(schema='ora.grounding-query.v2', grounding_ref=key, identity=identity, backend=backend,
        status='candidates' if candidates else 'no_candidate', candidates=candidates,
        diagnostics=diagnostics, response_service_identity=service_identity, overlay=overlay,
        visualization=dict(renderer=RENDERER, policy_visible=True,
            image_order=['overview', *[f"T{row['rank']}" for row in previews]], overview=overlay,
            candidate_previews=previews, claim_boundary='overview plus candidate-local enlargements from exact public '
                'RGB pixels; no new sensor detail or semantic ground truth'))
    files['manifest.json'] = json_bytes(manifest)
    record = api.artifacts.publish_tool_artifact(kind='grounding_query', key=key, files=files,
                                       metadata=dict(observation_id=observed, camera=camera))
    return manifest, record, pixels


def overhead_view_poses(api, result, config):
    """Add the overhead wrist pose to oblique footprints, on a surface that delivers them.

    The pose comes from the current grip-site pose and wrist calibration, whichever observation
    measured the candidates; without those the measurement simply carries no pose.
    """
    if not config.get('overhead_view_pose'):
        return
    obs = observation.observe(api)
    effectors = obs.manifest.json().get('proprioception', {}).get('end_effectors', {})
    state = effectors.get('right') or effectors.get('left')
    pose = state.get('pose_xyz_xyzw') if isinstance(state, dict) else None
    wrist = next((camera for camera in obs.cameras if camera.name == 'wrist'), None)
    if (not isinstance(pose, list) or len(pose) != 7 or wrist is None
            or any(ref is None for ref in (wrist.rgb, wrist.depth_m, wrist.calibration))):
        return
    camera_to_world = read_json(api, wrist.calibration).json().get('cam2world_cv')
    if camera_to_world is not None:
        _vision_footprint.annotate(result, grip_pose_xyz_xyzw=[float(item) for item in pose],
                                   wrist_camera_to_world=camera_to_world)


def selection_evidence(api, manifest, size):
    records = []
    for candidate in manifest['candidates']:
        record = api.artifacts.tool_artifact(kind='target', key=candidate['target_ref'])
        target = json.loads(api.artifacts.read_blob(record['files']['manifest.json']))
        from ...lib.blobs import read_array
        mask = read_array(api, JsonValue.of(record['files']['mask.npy'])).numpy().astype(bool)
        summary, localization = target['summary'], target.get('localization', {})
        def number(value):
            return float(value) if type(value) in (int, float) and math.isfinite(value) else None
        extent = localization.get('extent_xyz_m')
        if not (isinstance(extent, list) and len(extent) == 3 and all(number(v) is not None for v in extent)):
            extent = None
        records.append(dict(target_ref=candidate['target_ref'], mask_touches_image_boundary=bool(
            mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
            mask_pixel_count=int(np.count_nonzero(mask)), depth_valid_fraction=number(summary.get('depth_valid_fraction')),
            depth_inlier_fraction=number(localization.get('depth_inlier_fraction')),
            primary_component_fraction=number(localization.get('primary_component_fraction')), visible_extent_xyz_m=extent))
    return dict(schema='ora.perception-selection-evidence.v1', prompt_kind='text', source_image_size=list(size),
                point_uv=None, point_inside_returned_mask=None, semantic_identity_verified=False, targets=records)


def run(api, args):
    """Run vision_ground; replay re-requests SAM3 and computes derived images locally."""
    if set(args) - {'camera', 'text', 'max_candidates'} or not {'camera', 'text'} <= set(args):
        raise ToolRefusal('vision_ground requires camera and text; the Runtime binds the current observation automatically')
    obs = observation.observe(api)
    config = api.artifacts.tool_context()
    camera, text, maximum = args['camera'], args['text'], args.get('max_candidates', 3)
    if camera not in config['grounding_cameras']:
        raise ToolRefusal('camera is not available to text grounding', field='camera', received=camera,
                          constraints=dict(enum=sorted(config['grounding_cameras'])))
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 120:
        raise ToolRefusal('text must contain 1..120 non-whitespace characters', field='text')
    if type(maximum) is not int or not 1 <= maximum <= 3:
        raise ToolRefusal('max_candidates must be an integer in [1,3]; the grounding service answers at most that many '
                          'ranked regions per call', field='max_candidates', received=maximum,
                          constraints=dict(minimum=1, maximum=3))
    if camera not in obs.manifest.json().get('cameras', {}):
        raise ToolRefusal('camera is not policy-visible')
    if config['grounding_backend'] is None:
        raise ToolRefusal('vision_ground backend is unavailable', code='grounding_unavailable', retryable=True)
    try:
        manifest, record, pixels = materialize(api, obs, camera, text=text.strip(), maximum=maximum, config=config)
    except PublicToolFailure as error:
        raise ToolRefusal.from_public(error.public()) from error
    except InterfaceError:
        raise
    except ValueError as error:
        raise ToolRefusal(str(error), code='grounding_failed') from error
    identity = manifest['identity']
    previews = [dict(image_index=i, rank=row['rank'], target_ref=row['target_ref'], sha256=row['artifact']['sha256'],
        **{k: row[k] for k in ('source_dimensions', 'source_bbox_xyxy', 'crop_xyxy', 'render_dimensions', 'resampling')})
        for i, row in enumerate(manifest['visualization']['candidate_previews'], 1)]
    height, width = pixels.rgb.numpy().shape[:2]
    result = dict(success=True, status=manifest['status'], observation_id=identity['observation_id'],
        camera=camera, text=identity['text'], grounding_ref=manifest['grounding_ref'], backend=manifest['backend'],
        candidates=json.loads(canonical(manifest['candidates'])), diagnostics=manifest['diagnostics'],
        candidate_previews=previews, overlay=dict(sha256=manifest['overlay']['sha256'], width=width, height=height, image_index=0))
    for row in result['candidates']:
        for key in config['withheld_localization_fields']:
            row['localization'].pop(key, None)
    overhead_view_poses(api, result, config)
    if config['perception_evidence']:
        result['perception_evidence'] = selection_evidence(api, manifest, (width, height))
    advertised = config.get('grounding_result_fields')
    if advertised is not None:
        result = {key: value for key, value in result.items() if key in advertised}
    image_names = ['overlay.png', *[row['artifact']['path'] for row in manifest['visualization']['candidate_previews']]]
    result['images_base64'] = [base64.b64encode(api.artifacts.read_blob(record['files'][name])).decode('ascii') for name in image_names]
    if config['perception_feedback']:
        for candidate in result['candidates']:
            candidate['candidate_id'] = f"T{candidate['rank']}"
        mapping = []
        for index in range(len(image_names)):
            candidate = result['candidates'][index - 1] if index else {}
            mapping.append(dict(image_index=index, role='source_overview' if index == 0 else 'candidate_source_crop',
                camera=camera, candidate_id=candidate.get('candidate_id', ''), target_ref=candidate.get('target_ref', ''), observation_step=None))
        result['image_mapping'] = mapping
    return result
