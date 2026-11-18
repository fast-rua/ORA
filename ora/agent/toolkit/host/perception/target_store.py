"""The segmented-target store: content-addressed target manifests under one
observation, the run's target index, and the source binding every ``target_ref``
is checked against.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..shared import artifact_errors as _errors
from ..shared.artifact_io import _atomic_json, _canonical_hash, _file_hash, _verify_artifact
from ora.agent.toolkit.lib._vision_measurement import DEPTH_INLIER_SELECTION_METHOD, LOCALIZATION_METHOD


TARGET_SCHEMA = "ora.segmented-target.v2"
TARGET_INDEX_SCHEMA = "ora.segmented-target-index.v1"
TARGET_REF_PATTERN = re.compile(r"target-[0-9a-f]{24}")
DEPTH_INLIER_GEOMETRY_CAPABILITY = "public_target_geometry_depth_inliers_v2"


def _safe_path_component(value: str, *, field_name: str) -> str:
    if (
        not value or value in {".", ".."}
        or Path(value).name != value
        or "/" in value or "\\" in value
    ):
        raise ValueError(f"{field_name} is not a safe artifact path component")
    return value


def _segmentation_source_identity(
    *, observation_id: str, camera: str, rgb_path: Path,
    depth_path: Path, calibration_path: Path,
) -> dict[str, Any]:
    observation_id = _safe_path_component(
        observation_id, field_name="observation_id",
    )
    camera = _safe_path_component(camera, field_name="camera")
    source_root = rgb_path.resolve().parent
    expected = {
        "rgb": (rgb_path.resolve(), "rgb.png"),
        "depth": (depth_path.resolve(), "depth-m.npy"),
        "calibration": (calibration_path.resolve(), "calibration.json"),
    }
    files: dict[str, Any] = {}
    for role, (path, expected_name) in expected.items():
        if path.parent != source_root or path.name != expected_name or not path.is_file():
            raise ValueError(
                "segmentation source files must use rgb.png, depth-m.npy and "
                "calibration.json in one camera directory"
            )
        files[role] = {
            "name": expected_name,
            "sha256": _file_hash(path),
            "bytes": path.stat().st_size,
        }
    return {
        "policy_observation_path": (
            Path("policy-observations") / observation_id / camera
        ).as_posix(),
        "files": files,
    }


def _segmented_target_root(
    *, artifact_root: Path, observation_id: str, camera: str, target_ref: str,
) -> Path:
    return (
        artifact_root / "policy-observations" / observation_id / camera
        / "segmentation-queries" / target_ref
    )


def _target_index_path(artifact_root: Path, target_ref: str) -> Path:
    return artifact_root / "segmented-target-index" / f"{target_ref}.json"


def _materialize_target_index(
    *, artifact_root: Path, target_root: Path, manifest_path: Path,
    observation_id: str, camera: str, target_ref: str,
) -> None:
    index_path = _target_index_path(artifact_root, target_ref)
    record = {
        "schema": TARGET_INDEX_SCHEMA,
        "target_ref": target_ref,
        "observation_id": observation_id,
        "camera": camera,
        "artifact_path": target_root.resolve().relative_to(
            artifact_root.resolve(),
        ).as_posix(),
        "manifest_sha256": _file_hash(manifest_path),
    }
    if index_path.is_file():
        if json.loads(index_path.read_text(encoding="utf-8")) != record:
            raise _errors.ArtifactHashMismatch("segmented target index drift")
        return
    _atomic_json(index_path, record)
    index_path.chmod(0o444)


def _resolve_segmented_target_root(
    *, artifact_root: Path, target_ref: str,
) -> Path:
    index_path = _target_index_path(artifact_root, target_ref)
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            index.get("schema") != TARGET_INDEX_SCHEMA
            or index.get("target_ref") != target_ref
        ):
            raise _errors.ArtifactIntegrityError("segmented target index is invalid")
        relative = index.get("artifact_path")
        if not isinstance(relative, str):
            raise _errors.ArtifactIntegrityError("segmented target index lacks an artifact path")
        target_root = (artifact_root / relative).resolve()
        if not target_root.is_relative_to(artifact_root.resolve()):
            raise _errors.ArtifactIntegrityError("segmented target index escapes the artifact root")
        manifest_path = target_root / "manifest.json"
        if (
            not manifest_path.is_file()
            or _file_hash(manifest_path) != index.get("manifest_sha256")
        ):
            raise _errors.ArtifactHashMismatch("segmented target index manifest hash drift")
        return target_root

    # Read-only compatibility for pre-normalization qualification artifacts.
    legacy_root = artifact_root / "segmented-targets" / target_ref
    if legacy_root.joinpath("manifest.json").is_file():
        return legacy_root
    raise _errors.TargetRefUnknown("target_ref is unavailable: no segmented target in this run carries it")


def load_segmented_target(
    *, artifact_root: Path, target_ref: str,
) -> tuple[dict[str, Any], Path, Path]:
    if TARGET_REF_PATTERN.fullmatch(target_ref) is None:
        raise _errors.TargetRefMalformed("target_ref is malformed: not a segmented-target reference")
    target_root = _resolve_segmented_target_root(
        artifact_root=artifact_root, target_ref=target_ref,
    )
    manifest_path = target_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != TARGET_SCHEMA or manifest.get("target_ref") != target_ref:
        raise _errors.ArtifactIntegrityError("target_ref manifest is invalid")
    identity = manifest.get("identity")
    if (
        not isinstance(identity, Mapping)
        or "target-" + _canonical_hash(identity)[:24] != target_ref
    ):
        raise _errors.ArtifactHashMismatch("target_ref does not match its content identity")
    mask = _verify_artifact(target_root, manifest["private_mask"])
    overlay = _verify_artifact(target_root, manifest["overlay"])
    return manifest, mask, overlay


def validate_segmented_target_source(
    target: Mapping[str, Any], *, observation_id: str, rgb_path: Path,
    depth_path: Path, calibration_path: Path,
) -> None:
    """Bind derived target geometry to the exact source RGB, depth and calibration."""
    identity = target.get("identity", {})
    if identity.get("observation_id") != observation_id:
        raise ValueError("target_ref is stale for the selected observation")
    source = _segmentation_source_identity(
        observation_id=observation_id, camera=identity.get("camera", ""),
        rgb_path=rgb_path, depth_path=depth_path, calibration_path=calibration_path,
    )
    legacy = {f"{name}_sha256": source["files"][name]["sha256"]
              for name in ("rgb", "depth", "calibration")}
    if identity.get("source") not in (source, legacy):
        raise ValueError("target_ref source artifacts do not match the selected observation")


def perception_selection_evidence(
    *, result: Mapping[str, Any], artifact_root: Path,
    source_size: tuple[int, int], point_uv: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Expose selection/depth limitations without declaring task identity."""
    import numpy as np

    width, height = source_size
    references = [result["target_ref"]] if point_uv is not None else [
        candidate["target_ref"] for candidate in result["candidates"]
    ]
    records = []
    point_inside = None
    for reference in references:
        target, mask_path, _ = load_segmented_target(artifact_root=artifact_root, target_ref=reference)
        mask = np.load(mask_path, allow_pickle=False).astype(bool)
        if mask.shape != (height, width):
            raise ValueError("selection evidence dimensions differ from source image")
        if point_uv is not None:
            point_inside = bool(mask[point_uv[1], point_uv[0]])
        summary, localization = target["summary"], target.get("localization", {})
        def number(value: Any) -> float | None:
            return float(value) if type(value) in (int, float) and math.isfinite(value) else None
        extent = localization.get("extent_xyz_m")
        if not (isinstance(extent, list) and len(extent) == 3
                and all(number(value) is not None for value in extent)):
            extent = None
        records.append({
            "target_ref": reference,
            "mask_touches_image_boundary": bool(mask[0].any() or mask[-1].any()
                                                or mask[:, 0].any() or mask[:, -1].any()),
            "mask_pixel_count": int(np.count_nonzero(mask)),
            "depth_valid_fraction": number(summary.get("depth_valid_fraction")),
            "depth_inlier_fraction": number(localization.get("depth_inlier_fraction")),
            "primary_component_fraction": number(localization.get("primary_component_fraction")),
            "visible_extent_xyz_m": extent,
        })
    return {
        "schema": "ora.perception-selection-evidence.v1",
        "prompt_kind": "point" if point_uv is not None else "text",
        "source_image_size": [width, height],
        "point_uv": None if point_uv is None else list(point_uv),
        "point_inside_returned_mask": point_inside,
        "semantic_identity_verified": False,
        "targets": records,
    }
