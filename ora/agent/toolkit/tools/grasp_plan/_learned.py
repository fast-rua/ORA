"""Learned candidate admission and deterministic ordering stay with grasp_plan."""
from __future__ import annotations
import math
import numpy as np
from ...interface.types import canonical, digest
from ._topdown import GRIP_SITE_STYLE_COMPILER_IDS


def _canonical_hash(value):
    return digest(canonical(value))


def _inside(point, bounds):
    return all(float(bounds[axis][0]) <= float(point[i]) <= float(bounds[axis][1]) for i, axis in enumerate('xyz'))


def learned(pool, *, camera_to_world, target_ref, workspace_bounds_m, maximum, backend, compiler):
    """Admit the complete service pool against this tool's compiler and workspace."""
    settings = backend['configuration']
    transform = np.asarray(camera_to_world, dtype=np.float64)
    poses = pool.poses_m.numpy().astype(np.float64)
    scores = pool.scores.numpy().astype(np.float64)
    openings = pool.openings_m.numpy().astype(np.float64)
    controlled_offset = float(
        compiler["source_origin_to_controlled_point_m"]
    )
    maximum_opening = float(
        compiler.get("maximum_opening_m", settings['maximum_aperture_m'])
    )
    candidates = []
    rejected = []
    for source_index, (pose_camera, score, opening) in enumerate(
        zip(poses, scores, openings, strict=True)
    ):
        pose_world = transform @ pose_camera
        source_origin = pose_world[:3, 3]
        jaw = pose_world[:3, 0]
        approach = pose_world[:3, 2]
        identity = {
            "target_ref": target_ref,
            "pose_world": np.round(pose_world, 9).tolist(),
            "opening_m": round(float(opening), 9),
            "backend": backend['id'],
            "version": backend['version'],
            "compiler": compiler["id"],
        }
        candidate_id = "grasp-" + _canonical_hash(identity)[:24]
        reasons = []
        jaw_norm = float(np.linalg.norm(jaw))
        approach_norm = float(np.linalg.norm(approach))
        if jaw_norm <= 1e-6 or approach_norm <= 1e-6:
            reasons.append("degenerate_gripper_axis")
        else:
            jaw = jaw / jaw_norm
            approach = approach / approach_norm
            if abs(float(np.dot(jaw, approach))) > 0.05:
                reasons.append("non_orthogonal_gripper_axes")
        if reasons:
            rejected.append({"candidate_id": candidate_id, "reasons": reasons})
            continue
        controlled_point = source_origin + approach * controlled_offset
        source_pregrasp = source_origin - approach * settings['hover_offset_m']
        controlled_pregrasp = (
            controlled_point - approach * settings['hover_offset_m']
        )
        reasons = []
        if not math.isfinite(float(opening)) or not 0 < opening <= maximum_opening:
            reasons.append(
                "aperture_exceeds_panda_limit"
                if compiler["id"] in GRIP_SITE_STYLE_COMPILER_IDS
                else "aperture_exceeds_embodiment_limit"
            )
        if not _inside(controlled_point, workspace_bounds_m):
            reasons.append("compiled_controlled_point_outside_workspace")
        if not _inside(controlled_pregrasp, workspace_bounds_m):
            reasons.append(
                "compiled_pregrasp_controlled_point_outside_workspace"
            )
        if reasons:
            rejected.append({"candidate_id": candidate_id, "reasons": reasons,
                             "predicted_opening_m": float(opening) if math.isfinite(float(opening)) else None,
                             "maximum_opening_m": maximum_opening,
                             "limit_semantics": "compiler_admission_not_measured_physical_aperture"})
            continue
        candidates.append({
            "candidate_id": candidate_id,
            "source_index": source_index,
            "source_gripper_origin_xyz_m": [
                float(item) for item in source_origin
            ],
            "source_pregrasp_origin_xyz_m": [
                float(item) for item in source_pregrasp
            ],
            # Legacy public aliases retained for grasp.query consumers.
            # grasp.plan compiles only the explicit source-origin fields.
            "grasp_center_xyz_m": [float(item) for item in source_origin],
            "pregrasp_center_xyz_m": [
                float(item) for item in source_pregrasp
            ],
            "approach_world": [float(item) for item in approach],
            "jaw_world": [float(item) for item in jaw],
            "gripper_opening_m": float(opening),
            "backend_score": float(score),
            "pose_convention": (
                "contact_graspnet_panda_gripper_origin_opencv_v2"
            ),
            "checks": {
                "workspace": "passed",
                "aperture": "passed",
                "ik": "not_checked",
                "collision": "not_checked",
            },
            "execution_ready": False,
        })
    candidates.sort(
        key=lambda item: (-float(item["backend_score"]), item["candidate_id"])
    )
    candidates = candidates[:maximum]
    for rank, candidate in enumerate(candidates):
        candidate["rank"] = rank
    diagnostics = pool.metadata.json()
    service_identity = diagnostics.pop('service_identity', None)
    diagnostics.update(compiled_candidate_count=len(candidates), rejected=rejected,
        warning="aperture passed means only that the predicted opening is positive "
                "and within the compiler's admission limit; it does not establish "
                "object-width fit or measured jaw clearance. Model candidates "
                "require embodiment compilation, IK and collision checks")
    if service_identity is not None:
        diagnostics['_service_identity'] = service_identity
    return candidates, diagnostics
