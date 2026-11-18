"""The Contact-GraspNet grasp proposer: the proposal request over the loopback
service client and the embodiment compiler's admission of its candidates.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from ora.agent.model_services.clients import (
    CONTACT_GRASPNET_MODEL_IDENTITY,
    ContactGraspNetServiceClient,
    _decode_npy,
    _encode_npy,
    _require_response_service_identity,
    _verify_model_identity,
    http_failure,
    service_identity_drift,
)
from ora.agent.toolkit.host.shared import artifact_errors as _errors
from ora.agent.toolkit.host.grasp.compiler import GRIP_SITE_STYLE_COMPILER_IDS, PANDA_GRIP_SITE_COMPILER_IDENTITY
from .compiler_registry import GRASP_COMPILER_IDENTITIES
from ora.agent.toolkit.host.shared.artifact_io import _canonical_hash


@dataclass(frozen=True)
class ContactGraspNetApiProposer(ContactGraspNetServiceClient):
    """Contact-GraspNet client for one already-warm loopback service.

    The loopback transport, binding and readiness check are
    ``model_services.clients.ContactGraspNetServiceClient``; this adds the
    proposal request and the embodiment compiler's admission of its candidates.
    """

    forward_passes: int = 1
    hover_offset_m: float = 0.08
    maximum_aperture_m: float = 0.08
    raw_candidate_limit: int = 32
    backend_id: str = "contact_graspnet_api"
    backend_version: str = "v1"

    def _check_settings(self) -> None:
        if self.forward_passes != 1:
            raise ValueError("the first Contact-GraspNet treatment fixes one forward pass")
        if (
            isinstance(self.raw_candidate_limit, bool)
            or not isinstance(self.raw_candidate_limit, int)
            or not 1 <= self.raw_candidate_limit <= 32
        ):
            raise ValueError("Contact-GraspNet raw candidate limit must be in [1,32]")

    def configuration(self) -> Mapping[str, Any]:
        return {
            "endpoint": self.endpoint,
            "timeout_s": self.timeout_s,
            "forward_passes": self.forward_passes,
            "hover_offset_m": self.hover_offset_m,
            "maximum_aperture_m": self.maximum_aperture_m,
            "raw_candidate_limit": self.raw_candidate_limit,
            "expected_model": dict(CONTACT_GRASPNET_MODEL_IDENTITY),
            "api_schema": "ora.model-service.grasp-proposals.v1",
            "seed_policy": "canonical_scene_target_pointcloud_v1",
            "expected_binding_sha256": (
                None if self.expected_binding is None
                else _canonical_hash(self.expected_binding)
            ),
        }

    @staticmethod
    def _inside(point: Any, bounds: Mapping[str, list[float]]) -> bool:
        return all(
            float(bounds[axis][0]) <= float(point[index]) <= float(bounds[axis][1])
            for index, axis in enumerate(("x", "y", "z"))
        )

    def propose(
        self, *, scene_points_camera_m: Any, target_points_camera_m: Any,
        camera_to_world: Any, target_ref: str,
        workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
        compiler_identity: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        import httpx
        import numpy as np

        scene = np.ascontiguousarray(
            np.asarray(scene_points_camera_m, dtype=np.float32)
        )
        target = np.ascontiguousarray(
            np.asarray(target_points_camera_m, dtype=np.float32)
        )
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if scene.ndim != 2 or scene.shape[1] != 3 or len(scene) < 16:
            raise ValueError("Contact-GraspNet requires an Nx3 scene point cloud")
        if target.ndim != 2 or target.shape[1] != 3 or len(target) < 16:
            raise _errors.TargetInsufficient("Contact-GraspNet requires an Nx3 target point cloud of at least 16 points")
        if transform.shape != (4, 4):
            raise ValueError("Contact-GraspNet requires one 4x4 camera-to-world transform")
        seed_identity = {
            "schema": "ora.grasp-inference-seed.v1",
            "scene": {
                "shape": list(scene.shape),
                "dtype": str(scene.dtype),
                "sha256": hashlib.sha256(scene.tobytes(order="C")).hexdigest(),
            },
            "target": {
                "shape": list(target.shape),
                "dtype": str(target.dtype),
                "sha256": hashlib.sha256(target.tobytes(order="C")).hexdigest(),
            },
        }
        inference_seed = int(_canonical_hash(seed_identity)[:8], 16)
        payload = {
            "schema": "ora.model-service.grasp-proposals-request.v1",
            "scene_points_npy_base64": _encode_npy(scene),
            "target_points_npy_base64": _encode_npy(target),
            # The policy-visible limit is applied only after embodiment
            # compilation.  Request a wider bounded backend pool so invalid
            # high-score proposals cannot hide a valid lower-ranked grasp.
            "max_candidates": self.raw_candidate_limit,
            "forward_passes": self.forward_passes,
            "seed": inference_seed,
        }
        if self.expected_binding is not None:
            self.require_ready()
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self.endpoint}/v1/grasp-proposals", json=payload,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise http_failure(exc, service="Contact-GraspNet", what="service request") from exc
        roundtrip_ms = (time.perf_counter() - started) * 1000
        if result.get("schema") != "ora.model-service.grasp-proposals.v1":
            raise service_identity_drift("Contact-GraspNet service response schema mismatch", service="Contact-GraspNet")
        service_identity = _require_response_service_identity(
            result=result, binding=self.expected_binding,
            service="Contact-GraspNet",
        )
        model = _verify_model_identity(
            result.get("model"), CONTACT_GRASPNET_MODEL_IDENTITY,
        )
        poses = _decode_npy(result.get("poses_camera_npy_base64")).astype(np.float64)
        scores = _decode_npy(result.get("scores_npy_base64")).astype(np.float64).reshape(-1)
        openings = _decode_npy(
            result.get("openings_m_npy_base64"),
        ).astype(np.float64).reshape(-1)
        contacts = _decode_npy(
            result.get("contact_points_camera_npy_base64"),
        ).astype(np.float64)
        if poses.size == 0:
            poses = np.empty((0, 4, 4), dtype=np.float64)
            contacts = np.empty((0, 3), dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("Contact-GraspNet returned malformed poses")
        if contacts.ndim != 2 or contacts.shape[1:] != (3,):
            raise ValueError("Contact-GraspNet returned malformed contact points")
        if not (len(poses) == len(scores) == len(openings) == len(contacts)):
            raise ValueError("Contact-GraspNet returned unaligned candidate arrays")
        if not all(
            np.isfinite(array).all()
            for array in (poses, scores, openings, contacts)
        ):
            raise ValueError("Contact-GraspNet returned non-finite candidate arrays")
        compiler = dict(
            compiler_identity or PANDA_GRIP_SITE_COMPILER_IDENTITY
        )
        if compiler.get("id") not in GRASP_COMPILER_IDENTITIES:
            raise ValueError("Contact-GraspNet compiler identity is unregistered")
        controlled_offset = float(
            compiler["source_origin_to_controlled_point_m"]
        )
        maximum_opening = float(
            compiler.get("maximum_opening_m", self.maximum_aperture_m)
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
                "backend": self.backend_id,
                "version": self.backend_version,
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
            source_pregrasp = source_origin - approach * self.hover_offset_m
            controlled_pregrasp = (
                controlled_point - approach * self.hover_offset_m
            )
            reasons = []
            if not math.isfinite(float(opening)) or not 0 < opening <= maximum_opening:
                reasons.append(
                    "aperture_exceeds_panda_limit"
                    if compiler["id"] in GRIP_SITE_STYLE_COMPILER_IDS
                    else "aperture_exceeds_embodiment_limit"
                )
            if not self._inside(controlled_point, workspace_bounds_m):
                reasons.append("compiled_controlled_point_outside_workspace")
            if not self._inside(controlled_pregrasp, workspace_bounds_m):
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
        candidates = candidates[:max_candidates]
        for rank, candidate in enumerate(candidates):
            candidate["rank"] = rank
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("Contact-GraspNet response lacks timing metrics")
        return candidates, {
            "model": model,
            "_service_identity": service_identity,
            "request_id": str(result["request_id"]),
            "scene_point_count": int(len(scene)),
            "target_point_count": int(len(target)),
            "returned_candidate_count": int(len(poses)),
            "raw_candidate_limit": self.raw_candidate_limit,
            "compiled_candidate_count": len(candidates),
            "inference_seed": inference_seed,
            "seed_policy": "canonical_scene_target_pointcloud_v1",
            "seed_identity_sha256": _canonical_hash(seed_identity),
            "_backend_input_regime": result.get("input_regime") if isinstance(result.get("input_regime"), Mapping) else None,
            "rejected": rejected,
            # Consumed by materialize_grasp_query and removed before the
            # policy-visible diagnostics projection.  These exact aligned
            # arrays make the learned proposal reproducible without exposing
            # an additional policy authority.
            "_canonical_backend_output": {
                "poses_camera": poses,
                "scores": scores,
                "contact_points_camera": contacts,
                "openings_m": openings,
            },
            "latency_ms": {
                "roundtrip": roundtrip_ms,
                "service_queue": float(metrics["queue_ms"]),
                "service_inference": float(metrics["inference_ms"]),
                "service_total": float(metrics["total_ms"]),
            },
            "warning": (
                "aperture passed means only that the predicted opening is positive "
                "and within the compiler's admission limit; it does not establish "
                "object-width fit or measured jaw clearance. Model candidates "
                "require embodiment compilation, IK and collision checks"
            ),
        }
