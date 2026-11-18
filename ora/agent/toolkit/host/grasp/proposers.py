"""Grasp proposal backends: the protocol, the deterministic PCA top-down proposer,
and the hybrid that adds public-RGB-D top-down candidates to Contact-GraspNet's:
first on ALOHA and the X5, as a fallback on the Panda (``panda_topdown.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .x5_topdown import topdown_contact_height
from ora.agent.toolkit.host.shared import artifact_errors as _errors
from ora.agent.toolkit.host.grasp.diagnostics import hybrid_grasp_diagnostics
from ora.agent.toolkit.host.grasp.compiler import GRIP_SITE_STYLE_COMPILER_IDS, PANDA_GRIP_SITE_COMPILER_IDENTITY
from .compiler_registry import GRASP_COMPILER_IDENTITIES
from ora.agent.toolkit.host.shared.artifact_io import _canonical_hash


_TOPDOWN_MODEL = {
    "family": "deterministic_pca_test_double",
    "source_commit": "repository-source",
    "checkpoint_sha256": "not_applicable",
}


class GraspProposalBackend(Protocol):
    backend_id: str
    backend_version: str

    def configuration(self) -> Mapping[str, Any]: ...

    def propose(
        self, *, scene_points_camera_m: Any, target_points_camera_m: Any,
        camera_to_world: Any, target_ref: str,
        workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
        compiler_identity: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]: ...


@dataclass(frozen=True)
class PcaTopDownGraspProposer:
    """Simple Panda-width top-down candidates from one segmented point cloud."""

    backend_id: str = "pca_topdown_parallel_jaw"
    backend_version: str = "v1"
    maximum_aperture_m: float = 0.08
    aperture_clearance_m: float = 0.006
    hover_offset_m: float = 0.08
    #: A height profile for a compiler identity that carries none, whose
    #: controlled point is the contact centre (``panda_topdown.PROFILE``).
    topdown_profile: Mapping[str, Any] | None = field(default=None, hash=False)

    def configuration(self) -> Mapping[str, Any]:
        return {
            "maximum_aperture_m": self.maximum_aperture_m,
            "aperture_clearance_m": self.aperture_clearance_m,
            "hover_offset_m": self.hover_offset_m,
            "proposal_family": "top_down_only",
            **({"topdown_profile": dict(self.topdown_profile)} if self.topdown_profile is not None else {}),
        }

    @staticmethod
    def _canonical_axis(axis: Any) -> Any:
        import numpy as np

        value = np.asarray(axis, dtype=np.float64)
        value = value / np.linalg.norm(value)
        if value[0] < -1e-9 or (abs(value[0]) <= 1e-9 and value[1] < 0):
            value = -value
        return value

    @staticmethod
    def _inside(
        point: Any, bounds: Mapping[str, list[float]],
    ) -> bool:
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
        import numpy as np
        from ora.agent.toolkit.lib._principal_axes import principal_axes

        points_camera = np.asarray(target_points_camera_m, dtype=np.float64)
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if points_camera.ndim != 2 or points_camera.shape[1] != 3:
            raise ValueError("grasp proposal requires Nx3 target points")
        if transform.shape != (4, 4):
            raise ValueError("grasp proposal requires one 4x4 camera-to-world transform")
        points_h = np.column_stack([points_camera, np.ones(len(points_camera))])
        points = (transform @ points_h.T).T[:, :3]
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 16:
            raise _errors.TargetInsufficient("grasp proposal has too few valid target points")
        low = np.percentile(points, 2, axis=0)
        high = np.percentile(points, 98, axis=0)
        retained = np.all((points >= low) & (points <= high), axis=1)
        filtered = points[retained]
        if len(filtered) < 16:
            filtered = points
        xy = filtered[:, :2]
        covariance = np.cov(xy - np.mean(xy, axis=0), rowvar=False)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise _errors.TargetInsufficient("target geometry has invalid horizontal covariance")
        values, vectors = principal_axes(covariance)
        minor = self._canonical_axis(vectors[:, int(np.argmin(values))])
        rotations = (0.0, math.pi / 4, -math.pi / 4, math.pi / 2)
        axis_options = []
        for angle in rotations:
            c, s = math.cos(angle), math.sin(angle)
            rotated = np.asarray([
                c * minor[0] - s * minor[1],
                s * minor[0] + c * minor[1],
            ])
            axis_options.append(self._canonical_axis(rotated))
        center = np.median(filtered, axis=0)
        approach = np.asarray([0.0, 0.0, -1.0])
        compiler = dict(
            compiler_identity or PANDA_GRIP_SITE_COMPILER_IDENTITY
        )
        if compiler.get("id") not in GRASP_COMPILER_IDENTITIES:
            raise ValueError("grasp compiler identity is unregistered")
        topdown_profile = self.topdown_profile or compiler.get("public_rgbd_topdown_profile")
        height = None  # the support-aware rule (x5_topdown.py); ALOHA below is unchanged
        if isinstance(topdown_profile, Mapping) and "height_rule" in topdown_profile:
            height = topdown_contact_height(filtered, scene_points_camera_m, transform, topdown_profile)
            center[2], hover_offset = height["contact_centre_z_m"], float(topdown_profile["pregrasp_standoff_m"])
            max_candidates = min(max_candidates, int(topdown_profile["maximum_candidates"]))
            if self.topdown_profile is None:  # the X5 names link6, behind the contact centre
                center = center - approach * float(compiler["controlled_point_to_contact_baseline_m"])
        elif topdown_profile is None:
            center[2] = float(
                (
                    np.percentile(filtered[:, 2], 5)
                    + np.percentile(filtered[:, 2], 95)
                )
                / 2
            )
            hover_offset = self.hover_offset_m
        else:
            if (
                not isinstance(topdown_profile, Mapping)
                or set(topdown_profile) != {
                    "surface_percentile", "contact_clearance_m",
                    "pregrasp_standoff_m", "maximum_candidates",
                }
            ):
                raise ValueError("ALOHA public RGB-D top-down profile is invalid")
            percentile = float(topdown_profile["surface_percentile"])
            clearance = float(topdown_profile["contact_clearance_m"])
            hover_offset = float(topdown_profile["pregrasp_standoff_m"])
            maximum_topdown = int(topdown_profile["maximum_candidates"])
            if (
                not 50.0 <= percentile <= 99.0
                or not 0.0 <= clearance <= 0.02
                or not 0.02 <= hover_offset <= 0.10
                or not 1 <= maximum_topdown <= 4
            ):
                raise ValueError("ALOHA public RGB-D top-down profile is out of bounds")
            center[2] = float(
                np.percentile(filtered[:, 2], percentile) + clearance
            )
            max_candidates = min(max_candidates, maximum_topdown)
        hover = center - approach * hover_offset
        maximum_opening = float(
            compiler.get("maximum_opening_m", self.maximum_aperture_m)
        )
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for option_index, axis_xy in enumerate(axis_options):
            projection = xy @ axis_xy
            required = float(
                np.percentile(projection, 98) - np.percentile(projection, 2)
                + self.aperture_clearance_m
            )
            jaw = np.asarray([axis_xy[0], axis_xy[1], 0.0])
            reasons = []
            if required > maximum_opening:
                reasons.append(
                    "aperture_exceeds_panda_limit"
                    if compiler["id"] in GRIP_SITE_STYLE_COMPILER_IDS
                    else "aperture_exceeds_embodiment_limit"
                )
            if not self._inside(center, workspace_bounds_m):
                reasons.append("contact_pose_outside_workspace")
            if not self._inside(hover, workspace_bounds_m):
                reasons.append("hover_pose_outside_workspace")
            if height is not None and not height["reaches_object"]:
                reasons.append("fingertips_above_robust_top")
            identity = {
                "target_ref": target_ref,
                "center": [round(float(item), 9) for item in center],
                "jaw": [round(float(item), 9) for item in jaw],
                "approach": approach.tolist(),
                "required_aperture_m": round(required, 9),
                "backend": self.backend_id,
                "version": self.backend_version,
                "compiler": compiler["id"],
            }
            candidate_id = "grasp-" + _canonical_hash(identity)[:24]
            if reasons:
                rejected.append({"candidate_id": candidate_id, "reasons": reasons,
                                 "required_opening_m": required,
                                 "maximum_opening_m": maximum_opening,
                                 "limit_semantics": "compiler_admission_not_measured_physical_aperture"})
                continue
            score = max(0.0, 1.0 - required / maximum_opening)
            candidates.append({
                "candidate_id": candidate_id,
                "source_index": option_index,
                "grasp_center_xyz_m": [float(item) for item in center],
                "pregrasp_center_xyz_m": [float(item) for item in hover],
                "approach_world": approach.tolist(),
                "jaw_world": [float(item) for item in jaw],
                "gripper_opening_m": required,
                "backend_score": score,
                "pose_convention": "pca_topdown_parallel_jaw_center_v1",
                **({"topdown_height": dict(height)} if height is not None else {}),
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
        return candidates, {
            "model": dict(_TOPDOWN_MODEL),
            "target_point_count": int(len(points)),
            "filtered_point_count": int(len(filtered)),
            "rejected": rejected,
            "warning": "geometric candidates are not IK or collision qualified",
        }


#: How a hybrid fills its list. Under ``CONTACT_FIRST`` top-down is a fallback: it
#: is shown only when the learned family returned no candidate at all.
TOPDOWN_FIRST = "up_to_two_topdown_then_contact_fill_v1"
CONTACT_FIRST = "topdown_only_without_contact_v1"


@dataclass(frozen=True)
class AlohaHybridGraspProposer:
    """Fuse learned CGN candidates with public-RGB-D top-down poses.

    The learned service remains source-bound and mandatory.  The deterministic
    family adds a geometry-first candidate when the target mask supplies a
    closure-compatible top surface.  Neither family receives simulator GT.
    Under ``CONTACT_FIRST`` a target too sparse for the top-down family costs
    only that family: the learned candidates still stand.
    """

    contact_backend: GraspProposalBackend
    topdown_backend: PcaTopDownGraspProposer = field(
        default_factory=PcaTopDownGraspProposer,
    )
    backend_id: str = "contact_graspnet_plus_aloha_rgbd_topdown"
    backend_version: str = "v1"
    selection: str = TOPDOWN_FIRST

    def configuration(self) -> Mapping[str, Any]:
        return {
            "contact_backend": {
                "id": self.contact_backend.backend_id,
                "version": self.contact_backend.backend_version,
                "configuration": dict(self.contact_backend.configuration()),
            },
            "topdown_backend": {
                "id": self.topdown_backend.backend_id,
                "version": self.topdown_backend.backend_version,
                "configuration": dict(self.topdown_backend.configuration()),
            },
            "selection": self.selection,
        }

    def close(self) -> None:
        closer = getattr(self.contact_backend, "close", None)
        if callable(closer):
            closer()

    def require_ready(self) -> Mapping[str, Any]:
        readiness = getattr(self.contact_backend, "require_ready", None)
        if not callable(readiness):
            raise ValueError("hybrid grasp proposer lacks learned-service readiness")
        return readiness()

    def propose(
        self, *, scene_points_camera_m: Any, target_points_camera_m: Any,
        camera_to_world: Any, target_ref: str,
        workspace_bounds_m: Mapping[str, list[float]], max_candidates: int,
        compiler_identity: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        learned, learned_raw_diagnostics = self.contact_backend.propose(
            scene_points_camera_m=scene_points_camera_m,
            target_points_camera_m=target_points_camera_m,
            camera_to_world=camera_to_world,
            target_ref=target_ref,
            workspace_bounds_m=workspace_bounds_m,
            max_candidates=max_candidates,
            compiler_identity=compiler_identity,
        )
        # The learned service's private response identity and canonical
        # backend arrays must surface at the top level so the shared
        # materializer can strip them before the policy-visible projection.
        learned_diagnostics = dict(learned_raw_diagnostics)
        service_identity = learned_diagnostics.pop("_service_identity", None)
        backend_output = learned_diagnostics.pop("_canonical_backend_output", None)
        input_regime = learned_diagnostics.pop("_backend_input_regime", None)
        try:
            geometric, geometric_diagnostics = self.topdown_backend.propose(
                scene_points_camera_m=scene_points_camera_m,
                target_points_camera_m=target_points_camera_m,
                camera_to_world=camera_to_world,
                target_ref=target_ref,
                workspace_bounds_m=workspace_bounds_m,
                max_candidates=max_candidates,
                compiler_identity=compiler_identity,
            )
        except _errors.TargetInsufficient as exc:
            if self.selection != CONTACT_FIRST:
                raise
            geometric, geometric_diagnostics = [], {
                "model": dict(_TOPDOWN_MODEL), "rejected": [], "warning": str(exc),
            }
        if self.selection == CONTACT_FIRST and learned:
            geometric_shown: list[dict[str, Any]] = []
        else:
            geometric_shown = list(geometric)
        selected_geometric_count = (
            0 if self.selection == CONTACT_FIRST else min(2, max_candidates, len(geometric_shown))
        )
        selected = [dict(item) for item in geometric_shown[:selected_geometric_count]]
        selected.extend(
            dict(item) for item in learned[: max_candidates - len(selected)]
        )
        if len(selected) < max_candidates:
            selected.extend(
                dict(item)
                for item in geometric_shown[
                    selected_geometric_count:
                    selected_geometric_count + max_candidates - len(selected)
                ]
            )
        seen: set[str] = set()
        deduplicated: list[dict[str, Any]] = []
        for candidate in selected:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            candidate["rank"] = len(deduplicated)
            deduplicated.append(candidate)
        diagnostics: dict[str, Any] = hybrid_grasp_diagnostics(
            learned_diagnostics=learned_diagnostics,
            geometric_diagnostics=geometric_diagnostics,
            learned=learned, geometric=geometric, selected=deduplicated,
        )
        if service_identity is not None:
            diagnostics["_service_identity"] = service_identity
        if backend_output is not None:
            diagnostics["_canonical_backend_output"] = backend_output
        if input_regime is not None:
            diagnostics["_backend_input_regime"] = input_regime
        return deduplicated, diagnostics
