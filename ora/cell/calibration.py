"""One installation's calibration: what the robot model cannot know, written by tools only.

The robot model fixes the arms relative to each other and the wrist cameras relative to
their flanges. An installation adds where its fixed cameras stand, where its table top is,
the grippers' measured [closed, open] limits (so a driver need not re-run a gripper
calibration, which moves the jaws, at every start), the arms' joint readings at their
mechanical rest pose (a later reading that disagrees means an encoder zero has moved) and
the operator's notes. Which device plays which camera is the cell config's
(``config.py``), not this file's.

Where the installation departs from the robot model, it also holds the measured
corrections the kinematics apply (``apply``): per arm, a correction of its base pose in the
cell's world (``arm_base_corrections``, ``{arm: {rotvec_rad, translation_m}}``, corrected =
correction @ model), of each joint's encoder zero (``joint_zero_offsets``, ``{arm: [rad,
...]}``, model angle = reading + offset) and of its wrist camera's mount
(``wrist_camera_corrections``, camera frame). The robot's own encoder zeros are never
written; a recorded rest pose and every joint reading stay raw encoder values.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import CellError
from .io import write_json

SCHEMA = "ora.cell-calibration.v1"
LEGACY_SCHEMAS = ("ora.yam-cell-calibration.v1",)


def empty() -> dict[str, Any]:
    return {"schema": SCHEMA, "cameras": {}, "table": None, "gripper_limits": {}, "rest_reference": None,
            "notes": [], "updated_unix_s": time.time()}


def load(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return empty()
    path = Path(path).expanduser()
    if not path.is_file():
        return empty()
    value = json.loads(path.read_text())
    if value.get("schema") not in (SCHEMA, *LEGACY_SCHEMAS):
        raise CellError(f"{path} is not a {SCHEMA} file")
    value.pop("serials", None)                   # legacy: serials now live in the cell config
    value["schema"] = SCHEMA
    for role, camera in value.get("cameras", {}).items():
        matrix = np.asarray(camera.get("cam2world_cv"), dtype=float)
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise CellError(f"camera {role} in {path} has no valid 4x4 cam2world_cv")
        if not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-4):
            raise CellError(f"camera {role} in {path} has a non-rigid rotation")
    for arm, limits in value.get("gripper_limits", {}).items():
        if len(limits) != 2 or limits[0] == limits[1]:
            raise CellError(f"gripper limits of {arm} in {path} must be [closed, open] motor readings")
    for field in ("arm_base_corrections", "wrist_camera_corrections"):
        for arm, correction in (value.get(field) or {}).items():
            parts = [np.asarray(correction.get(key, [0, 0, 0]), dtype=float) for key in ("rotvec_rad", "translation_m")]
            if any(p.shape != (3,) or not np.all(np.isfinite(p)) for p in parts):
                raise CellError(f"{field} of {arm} in {path} needs rotvec_rad and translation_m of 3 numbers")
    for arm, offsets in (value.get("joint_zero_offsets") or {}).items():
        offsets = np.asarray(offsets.get("rad") if isinstance(offsets, Mapping) else offsets, dtype=float)
        if offsets.ndim != 1 or not np.all(np.isfinite(offsets)):
            raise CellError(f"joint_zero_offsets of {arm} in {path} must be a list of radians, one per joint")
    return value


def save(path: str | Path, value: Mapping[str, Any]) -> None:
    out = dict(value)
    out["schema"] = SCHEMA
    out["updated_unix_s"] = time.time()
    write_json(Path(path).expanduser(), out)


def table_for_model(calibration: Mapping[str, Any]) -> dict[str, Any] | None:
    """The scene builder's table from a calibrated plane, or None when the cell has none yet."""

    table = calibration.get("table")
    if not table:
        return None
    return {"center_xy_m": list(table["center_xy_m"]), "size_xy_m": list(table["size_xy_m"]),
            "top_z_m": float(table["top_z_m"])}


def _corrections(calibration: Mapping[str, Any], field: str) -> dict[str, np.ndarray]:
    from scipy.spatial.transform import Rotation

    out = {}
    for arm, value in (calibration.get(field) or {}).items():
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_rotvec(value.get("rotvec_rad", [0, 0, 0])).as_matrix()
        matrix[:3, 3] = value.get("translation_m", [0, 0, 0])
        out[arm] = matrix
    return out


def camera_corrections(calibration: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Per arm, the measured correction of its wrist camera's CAD mount as a 4x4 in the camera
    frame (stored as ``wrist_camera_corrections: {arm: {rotvec_rad, translation_m}}``)."""

    return _corrections(calibration, "wrist_camera_corrections")


def base_corrections(calibration: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Per arm, the measured correction of its base pose as a 4x4 in the cell's world
    (corrected = correction @ model)."""

    return _corrections(calibration, "arm_base_corrections")


def joint_offsets(calibration: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Per arm, each joint's encoder zero error in radians: model angle = reading + offset."""

    out = {}
    for arm, value in (calibration.get("joint_zero_offsets") or {}).items():
        out[arm] = np.asarray(value.get("rad") if isinstance(value, Mapping) else value, dtype=float)
    return out


#: How far past its recorded [closed, open] a gripper motor may read (rad; about 1.5 mm of a YAM jaw's aperture).
#: The stops give a little, a moved encoder zero does not: on 2026-11-14 a reading 0.30 rad outside passed a
#: 0.3 rad margin while the opening it gave was wrong by the whole travel.
GRIPPER_READING_MARGIN_RAD = 0.1


def gripper_readings_off(limits: Mapping[str, Any] | None, readings: Mapping[str, float | None],
                         margin: float = GRIPPER_READING_MARGIN_RAD) -> dict[str, dict[str, float]]:
    """The grippers whose motor reading lies outside its recorded limits by more than ``margin``."""

    off: dict[str, dict[str, float]] = {}
    for arm, pair in (limits or {}).items():
        raw = readings.get(arm)
        if raw is None:
            continue
        low, high = min(pair), max(pair)
        if not low - margin <= float(raw) <= high + margin:
            off[arm] = {"reading_rad": float(raw), "low_rad": float(low), "high_rad": float(high)}
    return off


def apply(kinematics: Any, calibration: Mapping[str, Any]) -> Any:
    """Give a robot model the installation's measured corrections it supports; returns it."""

    if hasattr(kinematics, "camera_corrections"):
        kinematics.camera_corrections = camera_corrections(calibration)
    if hasattr(kinematics, "joint_offsets"):
        kinematics.joint_offsets = joint_offsets(calibration)
    if hasattr(kinematics, "set_base_corrections"):
        kinematics.set_base_corrections(base_corrections(calibration))
    return kinematics
