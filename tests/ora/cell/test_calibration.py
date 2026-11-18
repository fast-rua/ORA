"""The tool-written calibration (``ora.cell-calibration.v1``): absent files read as empty, legacy YAM files load
without their serials, invalid extrinsics and gripper limits are refused, and a save is atomic JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ora.cell import calibration
from ora.cell.errors import CellError

CAMERA = {"cam2world_cv": np.eye(4).tolist(), "method": "charuco"}


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def test_no_path_or_no_file_reads_as_an_empty_calibration(tmp_path: Path) -> None:
    for value in (calibration.load(None), calibration.load(tmp_path / "absent.json")):
        assert value["schema"] == calibration.SCHEMA
        assert value["cameras"] == {} and value["gripper_limits"] == {}
        assert value["table"] is None and value["rest_reference"] is None


def test_a_legacy_yam_file_loads_as_the_current_schema_without_its_serials(tmp_path: Path) -> None:
    path = _write(tmp_path / "yam-cell.json", {
        "schema": "ora.yam-cell-calibration.v1", "serials": {"front": "317422075675"},
        "cameras": {"front": CAMERA}, "gripper_limits": {"left": [0.0, -2.6]},
        "rest_reference": {"left": [0.0] * 6}})

    value = calibration.load(path)

    assert value["schema"] == calibration.SCHEMA
    assert "serials" not in value
    assert value["cameras"]["front"]["method"] == "charuco"
    assert value["gripper_limits"] == {"left": [0.0, -2.6]}


def test_another_schema_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CellError, match="is not a ora.cell-calibration.v1 file"):
        calibration.load(_write(tmp_path / "c.json", {"schema": "something.else"}))


@pytest.mark.parametrize(("camera", "message"), [
    ({"cam2world_cv": np.eye(3).tolist()}, "no valid 4x4 cam2world_cv"),
    ({"cam2world_cv": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]}, "no valid 4x4 cam2world_cv"),
    ({}, "no valid 4x4 cam2world_cv"),
    ({"cam2world_cv": np.diag([2.0, 1.0, 1.0, 1.0]).tolist()}, "non-rigid rotation"),
])
def test_an_invalid_extrinsic_is_refused(tmp_path: Path, camera: dict, message: str) -> None:
    path = _write(tmp_path / "c.json", {"schema": calibration.SCHEMA, "cameras": {"front": camera}})
    with pytest.raises(CellError, match=f"camera front in .*{message}"):
        calibration.load(path)


@pytest.mark.parametrize("limits", [[0.5, 0.5], [0.1], [0.0, 1.0, 2.0]])
def test_gripper_limits_must_be_two_different_readings(tmp_path: Path, limits: list) -> None:
    path = _write(tmp_path / "c.json", {"schema": calibration.SCHEMA, "gripper_limits": {"left": limits}})
    with pytest.raises(CellError, match="gripper limits of left"):
        calibration.load(path)


def test_save_writes_the_current_schema_and_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "cells" / "c.json"
    value = {**calibration.empty(), "schema": "ora.yam-cell-calibration.v1", "cameras": {"front": CAMERA},
             "rest_reference": {"left": np.zeros(6)}, "updated_unix_s": 0.0}

    calibration.save(path, value)

    written = json.loads(path.read_text())
    assert written["schema"] == calibration.SCHEMA
    assert written["updated_unix_s"] > 0.0
    assert written["rest_reference"] == {"left": [0.0] * 6}
    assert not path.with_suffix(".json.tmp").exists()
    assert calibration.load(path)["cameras"]["front"]["cam2world_cv"] == np.eye(4).tolist()
    assert value["schema"] == "ora.yam-cell-calibration.v1"          # the caller's mapping is left as it was


def test_save_refuses_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        calibration.save(tmp_path / "c.json", {**calibration.empty(), "table": {"top_z_m": float("nan")}})
    assert not (tmp_path / "c.json").exists()


def test_the_scene_table_comes_from_a_calibrated_plane_only() -> None:
    assert calibration.table_for_model(calibration.empty()) is None
    table = {"center_xy_m": (0.4, 0.0), "size_xy_m": (1.2, 1.2), "top_z_m": "-0.01", "source": "front"}
    assert calibration.table_for_model({"table": table}) == {
        "center_xy_m": [0.4, 0.0], "size_xy_m": [1.2, 1.2], "top_z_m": -0.01}


def test_measured_corrections_load_as_matrices_and_offsets(tmp_path: Path) -> None:
    path = _write(tmp_path / "c.json", {
        "schema": calibration.SCHEMA,
        "arm_base_corrections": {"right": {"rotvec_rad": [0.0, 0.0, np.pi / 2], "translation_m": [0.0, 0.08, 0.02]}},
        "joint_zero_offsets": {"left": {"rad": [0.0, 0.04, 0.0, -0.04, 0.05, 0.0], "method": "hand_eye"},
                               "right": [0.0] * 6},
        "wrist_camera_corrections": {"left": {"translation_m": [0.01, 0.0, 0.0]}}})

    value = calibration.load(path)
    base = calibration.base_corrections(value)["right"]
    offsets = calibration.joint_offsets(value)

    assert np.allclose(base[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert np.allclose(base[:3, 3], [0.0, 0.08, 0.02])
    assert offsets["left"].tolist() == [0.0, 0.04, 0.0, -0.04, 0.05, 0.0] and offsets["right"].tolist() == [0.0] * 6
    assert np.allclose(calibration.camera_corrections(value)["left"][:3, 3], [0.01, 0.0, 0.0])
    assert calibration.base_corrections(calibration.empty()) == {} == calibration.joint_offsets(calibration.empty())


@pytest.mark.parametrize(("field", "entry", "message"), [
    ("arm_base_corrections", {"translation_m": [0.0, 0.1]}, "arm_base_corrections of right"),
    ("wrist_camera_corrections", {"rotvec_rad": [0.0, float("nan"), 0.0]}, "wrist_camera_corrections of right"),
    ("joint_zero_offsets", [[0.0, 0.1]], "joint_zero_offsets of right"),
    ("joint_zero_offsets", {"rad": [0.0, float("inf")]}, "joint_zero_offsets of right"),
])
def test_a_malformed_correction_is_refused(tmp_path: Path, field: str, entry, message: str) -> None:
    path = _write(tmp_path / "c.json", {"schema": calibration.SCHEMA, field: {"right": entry}})
    with pytest.raises(CellError, match=message):
        calibration.load(path)


class _Model:
    """Only what ``apply`` looks for."""

    def __init__(self) -> None:
        self.camera_corrections, self.joint_offsets, self.bases = {}, {}, None

    def set_base_corrections(self, corrections) -> None:
        self.bases = corrections


def test_apply_hands_the_model_every_correction_it_supports() -> None:
    value = {"arm_base_corrections": {"right": {"translation_m": [0.0, 0.08, 0.0]}},
             "joint_zero_offsets": {"left": [0.0, 0.04, 0.0, 0.0, 0.0, 0.0]}}
    model = calibration.apply(_Model(), value)
    assert set(model.bases) == {"right"} and model.joint_offsets["left"][1] == 0.04 and model.camera_corrections == {}

    class Plain:                                       # a robot model without corrections is left alone
        pass

    assert isinstance(calibration.apply(Plain(), value), Plain)


def test_a_gripper_reading_outside_its_recorded_limits_is_named():
    """2026-11-14: after a power cycle an open jaw read 0.506 rad against recorded [0.207, -5.164], 0.30 rad past
    the closed stop; the old 0.3 rad margin let it through and the jaw was taken as closed."""
    from ora.cell.calibration import GRIPPER_READING_MARGIN_RAD, gripper_readings_off

    limits = {"left": [6.413, 1.111], "right": [0.207, -5.164]}
    assert gripper_readings_off(limits, {"left": 1.119, "right": -5.10}) == {}
    off = gripper_readings_off(limits, {"left": 1.119, "right": 0.506})
    assert list(off) == ["right"] and off["right"]["reading_rad"] == 0.506
    assert gripper_readings_off(limits, {"right": 0.207 + GRIPPER_READING_MARGIN_RAD * 0.9}) == {}
    assert gripper_readings_off(limits, {"left": None}) == {} and gripper_readings_off(None, {"left": 9.0}) == {}
