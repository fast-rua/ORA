"""Bounded lift evidence from measured TCP, jaws and public RGB-D surfaces.

The reference is the Agent's selected target mask before grasp execution.
We require that surface to remain visible in the post-close baseline, then
compare its stationary and measured-TCP rigid-motion hypotheses against the
post-lift depth cloud. This is a scene-correspondence diagnostic, not a fresh
segmentation, semantic re-identification, contact sensor or attachment oracle.
Occlusion and indistinguishable hypotheses produce unknown.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ora.contracts.pose import quaternion_matrix

from ..shared import rgbd_fusion as geometry
from ..perception import localization
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.tools.grasp_execute._lift_evidence import LIFT_M, _effector, compare_surfaces

CAPABILITY = "prepared_grasp_execution_v2"


def capture_baseline(observation: Mapping[str, Any], root: Path) -> dict[str, Any]:
    manifest = root / "observation.json"
    return {
        "role": "post_close_before_lift",
        "observation": dict(observation),
        "observation_root": str(root.resolve()),
        "observation_id": observation["observation_id"],
        "observation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def _pose(observation: Mapping[str, Any], arm: str) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(_effector(observation, arm).get("pose_xyz_xyzw"), dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("measured_tcp_unavailable")
    norm = np.linalg.norm(pose[3:])
    if not 0.99 <= norm <= 1.01:
        raise ValueError("measured_tcp_orientation_unavailable")
    return pose[:3], np.asarray(quaternion_matrix((pose[3:] / norm).tolist()))


def measure(
    host: Any, *, commitment: Mapping[str, Any],
    observation: Mapping[str, Any], observation_root: Path,
) -> dict[str, Any]:
    baseline = commitment.get("lift_baseline")
    arm = commitment["arm"]
    result: dict[str, Any] = {
        "requested_world_lift_m": LIFT_M,
        "measured_tcp_translation_m": None, "measured_tcp_lift_m": None,
        "jaw_before_lift_m": None,
        "jaw_after_lift_m": localization.measured_jaw_separation_m(_effector(observation, arm)),
        "surface_motion": {"status": "unknown", "reason": "post_close_baseline_unavailable"},
        "claim_boundary": "short_lift_evidence_only;not_attachment_or_stable_transport_or_task_success",
    }
    if not isinstance(baseline, Mapping):
        return result
    before = baseline["observation"]
    result["jaw_before_lift_m"] = localization.measured_jaw_separation_m(_effector(before, arm))
    try:
        p0, r0 = _pose(before, arm)
        p1, r1 = _pose(observation, arm)
        result.update(measured_tcp_translation_m=(p1 - p0).tolist(), measured_tcp_lift_m=float(p1[2] - p0[2]))
        selected = commitment.get("retention_reference")
        if not isinstance(selected, Mapping):
            result["surface_motion"]["reason"] = "selected_public_target_surface_unavailable"
            return result
        scenes = [geometry.fuse_rgbd_pointcloud_sources(
            observation_root=root, observation=frame,
            workspace_bounds_m=host.workspace_bounds_m,
            camera_names=(selected["camera"],), voxel_size_m=0.003,
        )["xyz_world_m"] for frame, root in (
            (before, Path(baseline["observation_root"])), (observation, observation_root),
        )]
        result["surface_motion"] = compare_surfaces(
            points=np.asarray(selected["points"]), before_scene=scenes[0], after_scene=scenes[1],
            before_position=p0, before_rotation=r0, after_position=p1, after_rotation=r1,
        )
    except (ToolError, KeyError, ValueError, OSError, np.linalg.LinAlgError):
        result["surface_motion"] = {"status": "unknown", "reason": "public_rgbd_or_tcp_evidence_unavailable"}
    return result
