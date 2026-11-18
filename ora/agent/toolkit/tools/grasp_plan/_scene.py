"""Tool-local camera admission policy, using only granted public observations."""
from __future__ import annotations
from typing import Any, Mapping, NamedTuple, Sequence
from ...interface.types import NotRecorded, CorruptRecord
from ...lib.observation import load_camera
from ...lib.fusion import fuse, FusionSource
from ...lib.values import JsonValue

GRASP_SCENE_CAMERAS = ("agentview", "wrist")
GRASP_FUSION_VOXEL_SIZE_M = 0.005
GRASP_FUSION_MAX_DEPTH_M = 2.5
GRASP_BACKEND_INPUT_FUSION_SCHEMA = "ora.grasp-backend-input-fusion.v1"
GRASP_BACKEND_SINGLE_VIEW_SCOPE = "target_source_camera_rgbd_only"
GRASP_BACKEND_FUSED_SCOPE = "multiview_fused_rgbd"
GRASP_BACKEND_SINGLE_VIEW_INPUT_SCOPE = "target_source_camera_for_backend_only"
GRASP_BACKEND_FUSED_INPUT_SCOPE = "multiview_fused_rgbd_for_backend"
# One camera must place at least this many deduplicated points inside the
# workspace before it is allowed to alter the proposer's input geometry.
GRASP_BACKEND_MINIMUM_CAMERA_POINTS = 16
# Closed vocabulary: a camera that cannot enter the fused proposer input is
# always recorded with exactly one of these reasons, never silently dropped.
GRASP_BACKEND_EXCLUSION_REASONS = (
    "camera_absent_from_observation",
    "rgbd_source_reference_missing",
    "rgbd_source_file_unavailable",
    "rgbd_source_hash_drift",
    "rgbd_calibration_invalid",
    "rgbd_depth_unusable",
    "insufficient_workspace_depth_points",
    "duplicate_camera_request",
)


def _camera_rgbd_admission(*, api, observation, camera_name):
    record = observation.manifest.json().get('cameras', {}).get(camera_name)
    if not isinstance(record, Mapping):
        return 'camera_absent_from_observation', None
    files = record.get('files')
    if not isinstance(files, Mapping) or any(not isinstance(files.get(k), Mapping) for k in
            ('rgb_lossless_png', 'depth_numpy_m_f32', 'calibration')):
        return 'rgbd_source_reference_missing', None
    try:
        refs = observation.camera(camera_name)
        camera = load_camera(api, refs)
    except NotRecorded:
        return 'rgbd_source_file_unavailable', None
    except CorruptRecord:
        return 'rgbd_source_hash_drift', None
    except (ValueError, TypeError, KeyError, IndexError):
        return 'rgbd_calibration_invalid', None
    import numpy as np
    intrinsic = camera.intrinsic_px.numpy()
    camera_to_world = camera.camera_to_world.matrix.numpy()
    extrinsic = np.asarray(camera.calibration.json()['extrinsic_cv'], dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.,0.,0.,1.]])
    return None, dict(camera=camera_name, calibration=dict(
        path=files['calibration']['path'], sha256=files['calibration']['sha256'],
        convention='opencv_camera_optical', intrinsic_cv=intrinsic.tolist(),
        extrinsic_cv_world_to_camera=extrinsic.tolist(), camera_to_world=camera_to_world.tolist()),
        files={k: {n: files[k][n] for n in ('path','sha256')} for k in
            ('rgb_lossless_png','depth_numpy_m_f32','calibration')})


def fusion_sources(api, observation, names):
    return tuple(FusionSource(load_camera(api, observation.camera(name)),
        JsonValue.of({role: {field: value[field] for field in ('path', 'sha256', 'bytes')}
                     for role, value in observation.camera(name).declaration.json()['files'].items()}))
                 for name in names)


def fused_dict(fused):
    return dict(fused.metadata.json(), xyz_world_m=fused.points.xyz_m.numpy(),
                rgb_u8=fused.rgb.numpy(), source_index=fused.source_index.numpy())


def fuse_rgbd_pointcloud_sources(*, api, observation, workspace_bounds_m, camera_names,
                                 voxel_size_m, max_depth_m):
    return fused_dict(fuse(fusion_sources(api, observation, camera_names), JsonValue.of(workspace_bounds_m),
                           voxel_size_m=voxel_size_m, max_depth_m=max_depth_m))


def _admit_backend_scene_cameras(
    *, api, observation,
    workspace_bounds_m: Mapping[str, list[float]],
    target_source_camera: str, requested_cameras: Sequence[str],
    voxel_size_m: float, max_depth_m: float,
) -> dict[str, Any]:
    """Fuse every usable requested camera into one proposer input cloud.

    The target source camera is always first, so it wins voxel ties and the
    fused cloud stays consistent with the single-camera target mask.  Every
    other camera either contributes or is excluded with a named reason; the
    target source camera failing is a hard error, exactly as before.
    """

    seen: set[str] = set()
    candidates: list[str] = []
    excluded: list[dict[str, Any]] = []
    for name in [target_source_camera, *requested_cameras]:
        text = str(name)
        if text in seen:
            if text != target_source_camera:
                excluded.append({
                    "camera": text, "reason": "duplicate_camera_request",
                })
            continue
        seen.add(text)
        candidates.append(text)

    calibrations: dict[str, Any] = {}
    source_files: dict[str, Any] = {}
    admitted: list[str] = []
    for name in candidates:
        reason, record = _camera_rgbd_admission(
            api=api, observation=observation,
            camera_name=name,
        )
        if reason is None and len(candidates) > 1:
            # A camera only earns a place in the fused geometry once its own
            # depth actually lands inside the workspace.
            try:
                probe = fuse_rgbd_pointcloud_sources(
                    api=api, observation=observation,
                    workspace_bounds_m=workspace_bounds_m, camera_names=(name,),
                    voxel_size_m=voxel_size_m, max_depth_m=max_depth_m,
                )
            except (ValueError, TypeError, KeyError, OSError, EOFError):
                reason = "rgbd_depth_unusable"
            else:
                if int(probe["sources"][0].get("workspace_points", 0)) < (
                    GRASP_BACKEND_MINIMUM_CAMERA_POINTS
                ):
                    reason = "insufficient_workspace_depth_points"
        if reason is not None:
            if name == target_source_camera:
                raise ValueError(
                    "grasp backend input requires the target source camera "
                    f"{name}: {reason}"
                )
            excluded.append({"camera": name, "reason": reason})
            continue
        admitted.append(name)
        calibrations[name] = record["calibration"]
        source_files[name] = record["files"]

    fused = fuse_rgbd_pointcloud_sources(
        api=api, observation=observation,
        workspace_bounds_m=workspace_bounds_m, camera_names=tuple(admitted),
        voxel_size_m=voxel_size_m, max_depth_m=max_depth_m,
    )
    return {
        "fused": fused,
        "requested_cameras": [str(name) for name in requested_cameras],
        "contributing_cameras": admitted,
        "excluded_cameras": excluded,
        "calibrations": calibrations,
        "source_files": source_files,
    }


class _GraspBackendInput(NamedTuple):
    """The admitted proposer input cloud and the profile its identity binds."""

    admission: dict[str, Any]
    fused: dict[str, Any]
    contributing_cameras: list[str]
    multiview: bool
    profile: dict[str, Any]


def _grasp_backend_input(
    *, api, observation,
    workspace_bounds_m: Mapping[str, list[float]], target_source_camera: str,
    backend_scene_cameras: Sequence[str] | None,
) -> _GraspBackendInput:
    """Admit the proposer's cameras and describe the input its identity binds."""

    # Default: the historical single-view input, whose query identity stays
    # byte-identical.  An explicit request opts into the multi-view fusion the
    # review canvas has always computed and then thrown away.
    requested_backend_cameras = (
        (target_source_camera,) if backend_scene_cameras is None
        else tuple(str(name) for name in backend_scene_cameras)
    )
    admission = _admit_backend_scene_cameras(
        api=api,
        observation=observation,
        workspace_bounds_m=workspace_bounds_m,
        target_source_camera=target_source_camera,
        requested_cameras=requested_backend_cameras,
        voxel_size_m=GRASP_FUSION_VOXEL_SIZE_M,
        max_depth_m=GRASP_FUSION_MAX_DEPTH_M,
    )
    backend_input = admission["fused"]
    contributing_cameras = list(admission["contributing_cameras"])
    multiview_backend_input = len(contributing_cameras) > 1
    backend_input_profile = {
        "scope": (
            GRASP_BACKEND_FUSED_SCOPE if multiview_backend_input
            else GRASP_BACKEND_SINGLE_VIEW_SCOPE
        ),
        "camera": target_source_camera,
        "voxel_size_m": GRASP_FUSION_VOXEL_SIZE_M,
        "max_depth_m": GRASP_FUSION_MAX_DEPTH_M,
        "workspace_bounds_m": dict(workspace_bounds_m),
        "robot_self_filter": False,
        "multiview_fusion": multiview_backend_input,
    }
    if backend_scene_cameras is not None:
        # Only an opt-in request extends the identity, so frozen single-view
        # query refs keep their exact hashes.  Every camera whose RGB-D
        # actually determines the proposer input is bound by content here.
        backend_input_profile.update({
            "fusion_schema": GRASP_BACKEND_INPUT_FUSION_SCHEMA,
            "requested_cameras": list(admission["requested_cameras"]),
            "contributing_cameras": contributing_cameras,
            "excluded_cameras": [dict(item) for item in admission["excluded_cameras"]],
            "contributing_sources": [
                {
                    "camera": name,
                    "files": admission["source_files"][name],
                    "calibration_sha256": admission["calibrations"][name]["sha256"],
                }
                for name in contributing_cameras
            ],
            "deduplication": backend_input["deduplication"],
            "point_frame": "target_source_camera_opencv",
        })
    return _GraspBackendInput(
        admission, backend_input, contributing_cameras,
        multiview_backend_input, backend_input_profile,
    )
