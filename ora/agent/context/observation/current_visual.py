"""Complete current public camera delivery, shared by all policy runtimes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from ora.agent.evidence.image_binding import FULL_VISUAL_SCHEMA, PostActionObservationError, _safe_file


def full_visual_enabled(view_config: Mapping[str, Any] | None) -> bool:
    return bool(view_config and view_config.get("current_visual_delivery") == "full_rgb_v1")


def _read(root: Path, ref: Mapping[str, Any]) -> Path:
    path = _safe_file(root, ref["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref.get("sha256"):
        raise PostActionObservationError("current visual artifact hash mismatch")
    return path


def camera_geometry(root: Path, camera: Mapping[str, Any]) -> dict[str, Any]:
    """Publish only calibrated sensor geometry; never arbitrary artifact fields."""
    raw = json.loads(_read(root, camera["files"]["calibration"]).read_text())
    intrinsic = np.asarray(raw.get("intrinsic_cv"), dtype=float)
    extrinsic = np.asarray(raw.get("extrinsic_cv"), dtype=float)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack((extrinsic, [0.0, 0.0, 0.0, 1.0]))
    if (intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4)
            or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-8)):
        raise PostActionObservationError("invalid current camera calibration")
    camera_to_world = _affine_inverse(extrinsic)
    return {"intrinsic_cv": intrinsic.tolist(), "extrinsic_cv": extrinsic.tolist(),
        "cam2world_cv": camera_to_world,
        "camera_origin_world_m": [row[3] for row in camera_to_world[:3]],
        "camera_axes_world": {name: [row[index] for row in camera_to_world[:3]]
            for index, name in enumerate(("right", "down", "forward"))},
        "depth_unit": "metres",
        "pixel_coordinates": "origin top-left; u right, v down",
        "camera_coordinates": "OpenCV: x right, y down, z forward",
        "extrinsic_convention": "world_to_camera; cam2world_cv is its inverse",
        "depth_semantics": "optical-axis z distance in metres"}


def _affine_inverse(extrinsic: np.ndarray) -> list[list[float]]:
    """Invert a world-to-camera transform in plain Python arithmetic.

    These digits are rendered into the observation the model reads, so they
    must be the same on every host. ``np.linalg.inv`` is not: OpenBLAS picks a
    LAPACK kernel per CPU, and the HASWELL and SKYLAKEX kernels disagree in the
    last digit of the same inverse (the robocasa golden replayed to two
    different first requests on GitHub's AMD runners and on an AVX-512 host).
    Plain Python floats are IEEE-754 doubles evaluated in one fixed order
    everywhere.

    The caller has checked that the bottom row is ``[0, 0, 0, 1]``, so the
    inverse of ``[A | t]`` is ``[A^-1 | -A^-1 t]`` with ``A^-1`` by adjugate.
    """
    rows = [[float(value) for value in row] for row in extrinsic[:3]]
    a = [row[:3] for row in rows]
    t = [row[3] for row in rows]
    cofactor = [[a[(i + 1) % 3][(j + 1) % 3] * a[(i + 2) % 3][(j + 2) % 3]
                 - a[(i + 1) % 3][(j + 2) % 3] * a[(i + 2) % 3][(j + 1) % 3]
                 for j in range(3)] for i in range(3)]
    determinant = a[0][0] * cofactor[0][0] + a[0][1] * cofactor[0][1] + a[0][2] * cofactor[0][2]
    if determinant == 0.0 or not math.isfinite(determinant):
        raise PostActionObservationError("singular current camera calibration")
    inverse = [[cofactor[j][i] / determinant for j in range(3)] for i in range(3)]
    translation = [0.0 - (inverse[i][0] * t[0] + inverse[i][1] * t[1] + inverse[i][2] * t[2])
                   for i in range(3)]
    # ``+ 0.0`` turns a negative zero into ``0.0``: the model reads these digits,
    # and ``-0.0`` says nothing that ``0.0`` does not.
    result = [[value + 0.0 for value in (*inverse[i], translation[i])] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    if not all(math.isfinite(value) for row in result for value in row):
        raise PostActionObservationError("singular current camera calibration")
    return result


def materialize_current_visual(
    *, run_root: Path, workspace: Path, source_root: Path,
    cameras: Mapping[str, Any], policy_state: Mapping[str, Any],
    observation_id: str, observation_sha256: str | None,
    required_views: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Deliver every declared public RGB camera and its current calibration."""
    run_root, workspace, source_root = run_root.resolve(), workspace.resolve(), source_root.resolve()
    if not workspace.is_relative_to(run_root) or not source_root.is_relative_to(run_root):
        raise PostActionObservationError("current visual evidence escapes run")
    if not cameras:
        raise PostActionObservationError("current visual observation lacks cameras")
    if set(required_views) - set(cameras):
        raise PostActionObservationError("declared current camera is unavailable")
    if required_views and set(cameras) - set(required_views):
        raise PostActionObservationError("current observation contains an undeclared camera")
    identity = hashlib.sha256(json.dumps({"cameras": cameras, "state": dict(policy_state)},
        sort_keys=True).encode()).hexdigest()
    destination = workspace / "inputs/current-visual" / identity
    destination.mkdir(parents=True, exist_ok=True)
    images, geometry = [], {}
    for index, (name, camera) in enumerate(sorted(cameras.items())):
        files = camera["files"]
        rgb = _read(source_root, files.get("rgb_lossless_png") or files["rgb"])
        geometry[name] = camera_geometry(source_root, camera)
        with Image.open(rgb) as source_image:
            width, height = source_image.size
        target = destination / f"{index:02d}-rgb.png"
        data = rgb.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise PostActionObservationError("immutable current visual artifact drift")
        if not target.exists():
            target.write_bytes(data)
            target.chmod(0o444)
        images.append({"camera": name, "modality": "rgb",
            "path": target.relative_to(workspace).as_posix(),
            "run_path": target.relative_to(run_root).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(), "width": width, "height": height,
            "derivation": None})
    return {"schema": FULL_VISUAL_SCHEMA, "observation_id": observation_id,
        "observation_sha256": observation_sha256, "observation_step": policy_state["step"],
        "state_sha256": policy_state.get("state_sha256"), "images": images,
        "camera_geometry": geometry, "depth_delivery": "on_demand"}


def visual_feedback(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"camera_geometry": record["camera_geometry"],
        "views": [image["camera"] for image in record["images"]],
        "delivery": "complete_current_rgb_and_calibration", "depth_delivery": "on_demand", "observation_step": record["observation_step"]}
