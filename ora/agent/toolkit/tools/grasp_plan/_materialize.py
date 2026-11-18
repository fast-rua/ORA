"""Bind one query, request its raw pool and publish complete public derived artifacts."""
import json
import numpy as np
from ...interface.types import canonical, digest, InterfaceError
from ...lib import observation, geometry, fusion, grasp_review, vision
from ...lib import _rigid
from ...lib.blobs import read_array
from ...lib.values import Array, JsonValue, PointCloudM, TransformM
from ..vision_ground.tool import source_identity, artifact, json_bytes
from . import _scene, _records, _proposals


def world_to_camera(points, transform):
    return _rigid.apply(points, _rigid.invert(transform))


def materialize(api, obs, target, target_record, *, maximum, config, compiler, selection, review_names, review_binding):
    identity = target['identity']
    camera = identity['camera']
    observed = obs.identity.json()['observation_id']
    refs = obs.camera(camera)
    pixels = observation.load_camera(api, refs)
    source = source_identity(observed, camera, refs)
    legacy_source = {role+'_sha256': source['files'][role]['sha256'] for role in ('rgb', 'depth', 'calibration')}
    if identity['source'] not in (source, legacy_source):
        raise ValueError('target_ref source artifacts do not match the current observation')
    workspace = JsonValue.of(config['workspace_bounds_m'])
    proposer = _scene._grasp_backend_input(api=api, observation=obs, workspace_bounds_m=workspace.json(),
        target_source_camera=camera, backend_scene_cameras=review_names if config['fused_backend'] else None)
    backend = config['backend']
    query_identity = _records._grasp_query_identity(observation_id=observed, target_ref=target['target_ref'],
        identity=identity, actual_source=source, backend_input_profile=proposer.profile,
        workspace_bounds_m=workspace.json(), max_candidates=maximum, backend_record=backend, compiler=compiler,
        compile_for_execution=True, execution_tool=config['execution_tool'], target=target,
        public_contact_pose=config['public_contact_pose'], review_source_binding=review_binding,
        candidate_selection=selection, observed_extent=config['observed_extent'])
    key = 'grasp-query-' + digest(canonical(query_identity))[:24]
    cached = api.artifacts.tool_artifact(kind='grasp_query', key=key, cache=True)
    if cached is not None:
        manifest = json.loads(api.artifacts.read_blob(cached['files']['manifest.json']))
        if manifest['identity'] != query_identity:
            raise ValueError('grasp query identity drift')
        return manifest, cached
    mask = read_array(api, JsonValue.of(target_record['files']['mask.npy']))
    # Preserve the released float64 back-projection before CGN's float32 request.
    pixels = observation.CameraRGBD(pixels.name, pixels.rgb, Array.of(pixels.depth_m.numpy().astype(np.float64)),
        pixels.intrinsic_px, pixels.camera_to_world, pixels.calibration)
    target_cloud = geometry.mask_to_cloud(pixels, mask)
    if len(target_cloud.xyz_m.numpy()) < 16:
        from ..errors import ToolRefusal
        raise ToolRefusal('segmented target has too few valid depth pixels', code='grasp_proposal_failed',
            field='target_ref', received=target['target_ref'], retry_requires_changed_arguments=True)
    transform = pixels.camera_to_world
    scene_world = proposer.fused['xyz_world_m']
    scene_camera = world_to_camera(scene_world, transform.matrix.numpy())
    scene = PointCloudM(target_cloud.frame, Array.of(scene_camera))
    result, pool = _proposals.proposals(api, scene=scene, target=target_cloud, transform=transform,
        target_ref=target['target_ref'], workspace=workspace, maximum=maximum, config=config,
        compiler=compiler, selection=selection)
    proposals = _records._GraspProposals(**result, backend_output=None)
    world = geometry.transform_points(target_cloud, transform)
    points = _records._GraspQueryPoints(target_cloud.xyz_m.numpy(), transform.matrix.numpy(), scene_world, scene_camera)
    files = {}
    arrays = dict(backend_scene_points_camera_m=Array.of(scene_camera), backend_scene_points_world_m=Array.of(scene_world),
        target_points_camera_m=target_cloud.xyz_m, target_camera_to_world=transform.matrix,
        target_points_world_m=world.xyz_m)
    files['input-pointcloud.npz'] = grasp_review.npz_bytes(tuple(arrays.items()))
    overlay = grasp_review.contact_sheet(pixels, mask, world, observation_id=observed, target_ref=target['target_ref'],
        candidates=JsonValue.of(proposals.candidates), compiler=JsonValue.of(compiler),
        mask_sha256=target['private_mask']['sha256'], execution_tool=config['execution_tool'])
    files['overlay.png'] = vision.png_bytes(overlay.rgb)
    review, review_cloud = review_artifacts(api, obs, world, overlay, compiler, proposals.candidates,
        proposer, review_names, config, files, transform.matrix.numpy(), target['target_ref'])
    backend_arrays = dict(poses_camera=pool.poses_m, scores=pool.scores,
        contact_points_camera=pool.contact_points_m, openings_m=pool.openings_m)
    files['backend-output.npz'] = grasp_review.npz_bytes(tuple(backend_arrays.items()))
    backend_output = dict(schema='ora.contact-graspnet-backend-output.v1',
        arrays={name: dict(shape=list(value.shape), dtype=str(value.numpy().dtype)) for name,value in backend_arrays.items()},
        artifact=artifact('backend-output.npz', files['backend-output.npz']), input_regime=proposals.input_regime, policy_visible=False)
    manifest = _records._grasp_query_manifest(query_ref=key, query_identity=query_identity, backend_record=backend,
        proposals=proposals, backend_output_record=backend_output, visualization=overlay.metadata.json(),
        review_visualization=review, review_pointcloud=review_cloud,
        input_pointcloud=_records._grasp_input_pointcloud_record(
            input_pointcloud_artifact=artifact('input-pointcloud.npz',files['input-pointcloud.npz']),
            identity=identity,target_source_camera=camera,proposer_input=proposer,points=points,target_world=world.xyz_m.numpy()),
        overlay=artifact('overlay.png',files['overlay.png']))
    files['manifest.json'] = json_bytes(manifest)
    record = api.artifacts.publish_tool_artifact(kind='grasp_query', key=key, files=files,
        metadata=dict(observation_id=observed,camera=camera))
    return manifest, record


def review_artifacts(api, obs, world, overlay, compiler, candidates, proposer, names, config, files, transform, target_ref):
    review = dict(renderer='grasp-plan-global-pointcloud-review.v3',status='unavailable',policy_visible=False,
                  claim_boundary='best_effort_review_only_not_required_for_grasp_plan')
    record = None
    try:
        sources = _scene.fusion_sources(api, obs, names)
        fused = fusion.fuse(sources, JsonValue.of(config['workspace_bounds_m']),voxel_size_m=.005,max_depth_m=2.5)
        files['review-pointcloud.npz'] = grasp_review.npz_bytes(tuple(dict(
            review_scene_points_camera_m=Array.of(world_to_camera(fused.points.xyz_m.numpy(),transform)),
            review_scene_points_world_m=fused.points.xyz_m,review_scene_rgb_u8=fused.rgb,
            review_scene_source_index=fused.source_index).items()))
        image = grasp_review.global_review(sources,fused,world,overlay=overlay,
            workspace=JsonValue.of(config['workspace_bounds_m']),candidates=JsonValue.of(candidates),compiler=JsonValue.of(compiler),
            labels=JsonValue.of(dict(observation_id=obs.identity.json()['observation_id'],target_ref=target_ref,
                camera=proposer.profile['camera'],backend_input_scope=proposer.profile['scope'],
                backend_cameras=proposer.contributing_cameras,
                backend_points_by_camera={s['camera']:s['retained_voxel_points'] for s in proposer.fused['sources']},
                backend_excluded_cameras=proposer.admission['excluded_cameras'],policy_visible=config['perception_feedback'])),
            observed_extent=config['observed_extent'])
        files['global-review.png'] = vision.png_bytes(image.rgb)
        review = dict(image.metadata.json(),status='available',artifact=artifact('global-review.png',files['global-review.png']))
        data = fused.metadata.json()
        record = dict(schema='ora.grasp-review-pointcloud.v1',sources=data['sources'],camera_order=list(names),
            voxel_size_m=.005,max_depth_m=2.5,**{k:data[k] for k in ('input_point_count','workspace_point_count','filtered_voxel_count')},
            artifact=artifact('review-pointcloud.npz',files['review-pointcloud.npz']),policy_visible=False)
    except InterfaceError:
        raise
    except Exception as error:
        files.pop('review-pointcloud.npz', None)
        files.pop('global-review.png', None)
        review['failure'] = dict(type=type(error).__name__,message=str(error)[:512])
    return review, record
