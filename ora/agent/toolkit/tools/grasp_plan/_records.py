"""Versioned public grasp artifact records; no I/O or candidate policy."""
from __future__ import annotations
from typing import Any, Mapping, NamedTuple
from ._scene import (GRASP_BACKEND_FUSED_INPUT_SCOPE, GRASP_BACKEND_SINGLE_VIEW_INPUT_SCOPE,
    GRASP_BACKEND_INPUT_FUSION_SCHEMA, GRASP_BACKEND_EXCLUSION_REASONS,
    GRASP_BACKEND_MINIMUM_CAMERA_POINTS, _GraspBackendInput)
GRASP_QUERY_SCHEMA = 'ora.grasp-query.v7'
GRASP_PLAN_VISUALIZATION_ID = 'grasp-plan-target-local-contact-sheet.v2'
GRASP_INPUT_POINTCLOUD_SCHEMA = 'ora.grasp-input-pointcloud.v4'

class _GraspQueryPoints(NamedTuple):
    """Target points and the proposer scene in the target source camera frame."""

    target_camera: Any
    camera_to_world: Any
    backend_scene_world: Any
    backend_scene_camera: Any



class _GraspProposals(NamedTuple):
    """Compiled, selected candidates and what the backend reported beside them."""

    candidates: list[dict[str, Any]]
    compiled_candidates: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    response_service_identity: Any
    backend_output: Any
    input_regime: Any
    selection_record: Any



def _grasp_query_identity(
    *, observation_id: str, target_ref: str, identity: Mapping[str, Any],
    actual_source: Mapping[str, Any], backend_input_profile: dict[str, Any],
    workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
    backend_record: dict[str, Any], compiler: Mapping[str, Any],
    compile_for_execution: bool, execution_tool: str,
    target: Mapping[str, Any], public_contact_pose: bool,
    review_source_binding: Mapping[str, Any] | None,
    candidate_selection: Mapping[str, Any] | None, observed_extent: bool,
) -> dict[str, Any]:
    """The hashed identity whose digest names the query directory."""

    query_identity = {
        "observation_id": observation_id,
        "target_ref": target_ref,
        "camera": identity["camera"],
        "source": actual_source,
        "backend_input_profile": backend_input_profile,
        "workspace_bounds_m": dict(workspace_bounds_m),
        "max_candidates": max_candidates,
        "backend": backend_record,
        "compiler": dict(compiler),
        "compiler_mode": (
            "execution_plan" if compile_for_execution else "review_geometry_only"
        ),
        "execution_tool": execution_tool,
        "target_mask_sha256": target["private_mask"]["sha256"],
        "visualization_renderer": GRASP_PLAN_VISUALIZATION_ID,
        "input_pointcloud_schema": GRASP_INPUT_POINTCLOUD_SCHEMA,
    }
    if public_contact_pose:
        # Conditional so frozen earlier query identities keep their exact
        # hashes; surfaces without the capability never set the flag.
        query_identity["public_contact_pose"] = True
    if review_source_binding is not None:
        # Correctness surfaces bind optional review availability and source
        # hashes, so a degraded view cannot reuse a previously fused artifact.
        query_identity["review_source_binding"] = dict(review_source_binding)
    if candidate_selection is not None:  # released query refs keep their hashes
        query_identity["candidate_selection"] = dict(candidate_selection["identity"])
    query_identity.update({"review_renderer": "grasp-review-observed-extent.v1"} if observed_extent else {})
    return query_identity



def _grasp_input_pointcloud_record(
    *, input_pointcloud_artifact: dict, identity: Mapping[str, Any],
    target_source_camera: str, proposer_input: _GraspBackendInput,
    points: _GraspQueryPoints, target_world: Any,
) -> dict[str, Any]:
    """The manifest record of the proposer input cloud and its fusion."""

    admission = proposer_input.admission
    backend_input = proposer_input.fused
    contributing_cameras = proposer_input.contributing_cameras
    target_camera, camera_to_world = points.target_camera, points.camera_to_world
    backend_scene_camera = points.backend_scene_camera
    backend_scene_world = points.backend_scene_world
    return {
        "schema": GRASP_INPUT_POINTCLOUD_SCHEMA,
        "input_scope": (
            GRASP_BACKEND_FUSED_INPUT_SCOPE if proposer_input.multiview
            else GRASP_BACKEND_SINGLE_VIEW_INPUT_SCOPE
        ),
        "target_source_camera": identity["camera"],
        "backend_input_profile": proposer_input.profile,
        "backend_source": backend_input["sources"][0],
        "backend_input_sources": [
            dict(source) for source in backend_input["sources"]
        ],
        "backend_input_fusion": {
            "schema": GRASP_BACKEND_INPUT_FUSION_SCHEMA,
            "requested_cameras": list(admission["requested_cameras"]),
            "contributing_cameras": contributing_cameras,
            "excluded_cameras": [
                dict(item) for item in admission["excluded_cameras"]
            ],
            "exclusion_vocabulary": list(GRASP_BACKEND_EXCLUSION_REASONS),
            "minimum_camera_points": GRASP_BACKEND_MINIMUM_CAMERA_POINTS,
            "deduplication": backend_input["deduplication"],
            # The fused world cloud is expressed in the target camera's
            # OpenCV frame before it reaches the proposer; both directions
            # of that transform are recorded so a replay can undo it.
            "backend_point_frame": {
                "camera": target_source_camera,
                "convention": "opencv_camera_optical",
                "world_to_backend_camera": admission["calibrations"][
                    target_source_camera
                ]["extrinsic_cv_world_to_camera"],
                "backend_camera_to_world": [
                    [float(value) for value in row]
                    for row in camera_to_world.tolist()
                ],
            },
            "cameras": [
                {
                    "camera": str(source["camera"]),
                    "source_index": int(source["source_index"]),
                    "source_resolution": list(source["source_resolution"]),
                    "valid_depth_points": int(source["valid_depth_points"]),
                    "workspace_points": int(source["workspace_points"]),
                    "retained_voxel_points": int(source["retained_voxel_points"]),
                    "calibration": admission["calibrations"][
                        str(source["camera"])
                    ],
                }
                for source in backend_input["sources"]
            ],
        },
        "input_point_count": backend_input["input_point_count"],
        "workspace_point_count": backend_input["workspace_point_count"],
        "filtered_voxel_count": backend_input["filtered_voxel_count"],
        "arrays": {
            "backend_scene_points_camera_m": {
                "shape": list(backend_scene_camera.shape),
                "dtype": str(backend_scene_camera.dtype),
            },
            "backend_scene_points_world_m": {
                "shape": list(backend_scene_world.shape),
                "dtype": str(backend_scene_world.dtype),
            },
            "target_points_camera_m": {
                "shape": list(target_camera.shape), "dtype": str(target_camera.dtype),
            },
            "target_camera_to_world": {
                "shape": list(camera_to_world.shape), "dtype": str(camera_to_world.dtype),
            },
            "target_points_world_m": {
                "shape": list(target_world.shape), "dtype": str(target_world.dtype),
            },
        },
        "artifact": input_pointcloud_artifact,
    }



def _grasp_query_manifest(
    *, query_ref: str, query_identity: dict[str, Any],
    backend_record: dict[str, Any], proposals: _GraspProposals,
    backend_output_record: dict[str, Any] | None,
    visualization: dict[str, Any], review_visualization: dict[str, Any],
    review_pointcloud: dict[str, Any] | None,
    input_pointcloud: dict[str, Any], overlay: dict[str, Any],
) -> dict[str, Any]:
    candidates = proposals.candidates
    selection_record = proposals.selection_record
    return {
        "schema": GRASP_QUERY_SCHEMA,
        "query_ref": query_ref,
        "identity": query_identity,
        "backend": backend_record,
        "status": "candidates" if candidates else "no_candidate",
        "candidates": candidates,
        "compiled_candidates": proposals.compiled_candidates,
        "diagnostics": dict(proposals.diagnostics),
        "response_service_identity": proposals.response_service_identity,
        "backend_output": backend_output_record,
        "visualization": visualization,
        "review_visualization": review_visualization,
        "review_pointcloud": review_pointcloud,
        "input_pointcloud": input_pointcloud,
        "overlay": overlay,
        **({"candidate_selection": selection_record} if selection_record is not None else {}),
    }

