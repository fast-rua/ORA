"""The cell config (``ora.cell-config.v1``): what a valid file yields, what each mistake is refused with, and
``check_against`` naming what the config gets wrong about its robot."""

from __future__ import annotations

from pathlib import Path

import pytest

from ora.cell import config
from ora.cell.errors import CellError
from tests.support.cell.fake import CONFIG, FakeAdapter, write_config


def test_a_valid_config_loads_every_section(tmp_path: Path) -> None:
    loaded = config.load(write_config(tmp_path))

    assert loaded.cell_id == "fake-cell-01"
    assert (loaded.environment, loaded.model) == ("fake", "fake_station")
    assert dict(loaded.buses) == {"left": "fake_left", "right": "fake_right"}
    assert loaded.serials == {"overhead": "100", "wrist_left": "200", "wrist_right": "300"}
    assert loaded.cameras["wrist_left"].mount == "end_effector"
    assert loaded.calibration_path == tmp_path / "calibration.json"
    assert loaded.python is None
    assert loaded.max_joint_speed_rad_s == 0.5
    assert loaded.operator_present_required is True
    assert loaded.ignore_processes == ("my_teleop.py",)
    assert loaded.perception == {"depth_completion": None}
    assert loaded.policy == {"provider_profile": "test"}


def test_the_path_comes_from_ora_cell_and_user_paths_expand(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    text = CONFIG.replace("  environment: fake\n", "  environment: fake\n  python: ~/envs/fake/bin/python\n")
    path = write_config(tmp_path, text)
    monkeypatch.setenv("ORA_CELL", str(path))

    assert config.default_path() == path
    loaded = config.load()
    assert loaded.path == path
    assert loaded.python == tmp_path / "envs" / "fake" / "bin" / "python"


def test_a_missing_file_names_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(CellError, match="no cell config at .*nowhere.yaml"):
        config.load(tmp_path / "nowhere.yaml")


@pytest.mark.parametrize(("old", "new", "message"), [
    ("schema: ora.cell-config.v1", "schema: ora.cell-config.v0", "schema must be ora.cell-config.v1"),
    ("cell_id: fake-cell-01\n", "", "'cell_id' is required"),
    ("calibration: CALIBRATION\n", "", "'calibration' is required"),
    ("  environment: fake\n", "", "robot: 'environment' is required"),
    ("  model: fake_station\n", "", "robot: 'model' is required"),
    ("  buses: {left: fake_left, right: fake_right}\n", "", "robot: 'buses' is required"),
    ('{serial: "100", mount: fixed_world}', '{serial: "100", mount: ceiling}', "cameras.overhead: mount must be one of"),
    ('{serial: "100", mount: fixed_world}', "{mount: fixed_world}", "cameras.overhead: 'serial' is required"),
    ('{serial: "100", mount: fixed_world}', '{serial: "100"}', "cameras.overhead: 'mount' is required"),
    ("max_joint_speed_rad_s: 0.5", "max_joint_speed_rad_s: 0", "max_joint_speed_rad_s must be a positive number"),
    ("max_joint_speed_rad_s: 0.5", "max_joint_speed_rad_s: fast", "max_joint_speed_rad_s must be a positive number"),
])
def test_each_mistake_is_refused_with_where_it_is(tmp_path: Path, old: str, new: str, message: str) -> None:
    assert old in CONFIG
    path = write_config(tmp_path, CONFIG.replace(old, new))
    with pytest.raises(CellError, match=message) as refused:
        config.load(path)
    assert str(path) in str(refused.value)


def test_an_empty_file_is_refused_not_crashed(tmp_path: Path) -> None:
    path = tmp_path / "cell.yaml"
    path.write_text("")
    with pytest.raises(CellError, match="schema must be"):
        config.load(path)


def test_check_against_accepts_a_config_that_matches_its_robot(tmp_path: Path) -> None:
    assert config.check_against(config.load(write_config(tmp_path)), FakeAdapter()) == []


def test_check_against_names_unknown_arms_cameras_and_wrong_mounts(tmp_path: Path) -> None:
    text = (CONFIG.replace("buses: {left: fake_left, right: fake_right}", "buses: {left: fake_left}")
            .replace('wrist_left:  {serial: "200", mount: end_effector}', 'wrist_left:  {serial: "200", mount: fixed_world}')
            .replace('  wrist_right: {serial: "300", mount: end_effector}\n',
                     '  wrist_right: {serial: "300", mount: end_effector}\n  side: {serial: "400", mount: fixed_world}\n'))
    problems = config.check_against(config.load(write_config(tmp_path, text)), FakeAdapter())

    assert len(problems) == 3
    assert "robot.buses must name exactly the arms ['left', 'right']" in problems[0]
    assert "camera 'wrist_left' is mounted end_effector on this robot, not fixed_world" in problems
    assert any(problem.startswith("camera 'side' is not one of the robot's cameras") for problem in problems)


def test_a_widened_wrist_table_check_needs_a_reason_and_stays_bounded(tmp_path: Path) -> None:
    def load(block: str):
        text = CONFIG.replace("safety:\n", "safety:\n  wrist_table_check:\n" + block)
        return config.load(write_config(tmp_path, text))

    widened = load("    max_tilt_deg: 4.5\n    max_height_gap_m: 0.035\n    reason: arm chain under recalibration\n")
    assert widened.wrist_table_check["max_height_gap_m"] == 0.035
    assert config.load(write_config(tmp_path)).wrist_table_check == {}
    with pytest.raises(CellError, match="needs a 'reason'"):
        load("    max_tilt_deg: 4.5\n")
    with pytest.raises(CellError, match="at most 6 deg"):
        load("    max_tilt_deg: 9\n    reason: testing\n")
