"""The guided hand-eye calibration asks in order and keeps the person in charge: a placement is asked
again with the reason when the board is not seen, a motion is declined or stopped; declining a
placement ends the captures; nothing is written unless approved, and the old file is kept."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ora.cell import bundle
from ora.cell import calibration as cell_calibration
from ora.cell.errors import CellError
from ora.cell.lock import holder
from ora.cell.operator import handeye

K = [[430.0, 0.0, 424.0], [0.0, 430.0, 240.0], [0.0, 0.0, 1.0]]


class Scripted:
    """An asker that answers from a list and keeps every question."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.asked: list[tuple[str, str, str]] = []

    def confirm(self, kind, title, text, details, timeout_s):
        self.asked.append((kind, title, text))
        answer = self.answers.pop(0)
        return (True, None) if answer is True else (False, answer)

    def value(self, title, text, details, timeout_s):
        self.asked.append(("value", title, text))
        answer = self.answers.pop(0)
        return (answer, None) if answer is not None else (None, "no")


@pytest.fixture
def placed(cell):
    """The fake cell with its overhead camera roughly placed."""

    value = cell.calibration()
    value["cameras"] = {"overhead": {"cam2world_cv": np.eye(4).tolist()}}
    cell.save_calibration(value)
    return cell


@pytest.fixture
def stage(monkeypatch, cell):
    """Cameras, planning, motion and the fit replaced: what each was asked is kept in ``calls``."""

    calls = {"locate": 0, "runs": [], "labels": []}
    found = {"overhead": {"board2world": np.eye(4), "reprojection_px": 0.3, "center": [0.4, 0.0, 0.0]}}
    sightings = [{"overhead": None}]

    def locate(cell, board):
        calls["locate"] += 1
        return sightings.pop(0) if sightings else found

    def run(cell, planned, record, lock, label):
        calls["labels"].append(holder())
        calls["runs"].append(sorted(planned["arms"]))
        record.mkdir(parents=True)
        return calls.pop("stop", None)

    monkeypatch.setattr(handeye, "RECORD_ROOT", cell.config.calibration_path.parent / "records")
    monkeypatch.setattr(handeye, "capture", lambda cell, roles, frames=1: {
        r: {"intrinsic_cv": K, "rgb": np.zeros((480, 848, 3), np.uint8)} for r in roles})
    monkeypatch.setattr(handeye, "locate", locate)
    monkeypatch.setattr(handeye, "plan", lambda cell, arms, board2world, board, wrist: {
        "arms": {a: {"views": calls.get("views", 6), "route": [], "max_offset_deg": [11.5] * 6} for a in arms},
        "skipped": []})
    monkeypatch.setattr(handeye, "run_placement", run)
    monkeypatch.setattr(bundle, "load_views", lambda records, roles, wrist_arm, board, arms: (
        [bundle.View(i, r, {}, np.zeros((88, 2)), np.eye(3), np.zeros(5)) for i, _ in enumerate(records)
         for r in ("wrist_left", "wrist_right")], []))
    monkeypatch.setattr(bundle, "solve", lambda views, kinematics, board, **_: bundle.Fit(
        ["left", "right"], {}, {}, {}, {}, [], {"views": {"wrist_left": [{"px": 1.0, "mm": 0.8, "capture": "x"}]}}))
    monkeypatch.setattr(bundle, "validate", lambda *a, **k: {"1": {"wrist_left": [{"px": 2.0, "mm": 1.5}]}})
    monkeypatch.setattr(bundle, "apply_fit", lambda calibration, fit, **k: {**calibration, "bundle_fit": {"marker": 1}})
    return calls


def _run(cell, asker, **options):
    return handeye.calibrate(cell, asker=asker, square_mm=20.0, pattern=(11, 8), **options)


def test_the_placements_are_asked_in_turn_and_the_result_is_written_after_a_yes(placed, stage) -> None:
    before = placed.config.calibration_path.read_text()
    asker = Scripted(True, True, True, True, True, "够了", "不加", True)
    report = _run(placed, asker, placements=3)

    kinds = [kind for kind, _, _ in asker.asked]
    assert kinds == ["task", "task", "motion", "task", "motion", "task", "task", "write"]
    assert "两臂之间正前方" in asker.asked[0][2] and "左臂前方近处" in asker.asked[3][2]
    assert asker.asked[6][1].startswith("再加一个摆位？")
    assert "没有一台固定相机" in asker.asked[1][2]          # the first board was not seen: asked again, with why
    assert stage["runs"] == [["left", "right"], ["left"]]   # the first placement for both arms, then one in turn
    assert all("hand-eye calibration" in label for label in stage["labels"])
    assert holder() is None                                 # the lock is released after the captures
    assert len(report["records"]) == 2 and report["saved"] is True
    assert Path(report["backup"]).read_text() == before
    assert cell_calibration.load(placed.config.calibration_path)["bundle_fit"] == {"marker": 1}


def test_a_declined_or_stopped_motion_asks_the_placement_again(placed, stage) -> None:
    stage["stop"] = "stop requested (operator home, or Ctrl-C)"
    asker = Scripted(True, True, True, True, "不行", True, True, "够了", "不加", "不写")
    report = _run(placed, asker, placements=2)

    texts = [text for kind, _, text in asker.asked if kind == "task"]
    assert "动作中途停止（stop requested" in texts[2]
    assert "动作被拒绝（不行）" in texts[3]
    assert report["records"] == [str(Path(report["session"]) / "placement-1")]
    assert report["saved"] is False and report["not_saved"] == "不写"
    assert "bundle_fit" not in cell_calibration.load(placed.config.calibration_path)


def test_too_few_safe_views_ask_to_move_the_board_closer(placed, stage) -> None:
    stage["views"] = 2
    asker = Scripted(True, True, "停")
    with pytest.raises(CellError, match="no placement was captured"):
        _run(placed, asker, placements=1)
    assert "左臂和右臂只找到 2 个安全的拍摄姿势" in asker.asked[2][2]


def test_without_a_placed_fixed_camera_nothing_is_asked(cell, stage) -> None:
    asker = Scripted()
    with pytest.raises(CellError, match="no fixed camera has a pose yet"):
        _run(cell, asker)
    assert asker.asked == []


def test_no_save_reports_without_asking_to_write(placed, stage) -> None:
    asker = Scripted(True, True, True)
    report = _run(placed, asker, placements=1, save=False)
    assert [kind for kind, _, _ in asker.asked] == ["task", "task", "motion"]
    assert report["saved"] is False and report["concerns"]


def test_the_board_is_asked_until_its_answers_make_sense() -> None:
    asker = Scripted("twenty", "200", "20 mm", "8x8", "11×8")
    assert handeye.ask_board(asker, None, None) == bundle.Board(0.02, (11, 8))
    assert [kind for kind, _, _ in asker.asked] == ["value"] * 5


def test_each_arm_looks_at_the_first_placement_then_in_turn() -> None:
    assert handeye.schedule(["left", "right"], 5) == [("left", "right"), ("left",), ("right",), ("left",), ("right",)]
    assert handeye.schedule(["left"], 2) == [("left",), ("left",)]
    assert handeye.parse_pattern("11x8") == (11, 8) and handeye.parse_pattern("8x8") is None


def test_a_view_counts_only_with_the_whole_board_clear_of_the_jaws() -> None:
    board = bundle.Board(0.02, (11, 8))
    corners = board.points() - board.points().mean(0)
    camera = handeye._look_at(np.array([0.0, 0.0, 0.3]), np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.0)
    assert handeye.sees_board(camera, np.asarray(K), (848, 480), corners, [])
    assert not handeye.sees_board(camera, np.asarray(K), (848, 480), corners, [(300, 200, 500, 300)])
    assert not handeye.sees_board(camera, np.asarray(K), (300, 200), corners, [])


def test_the_concerns_name_what_should_make_the_person_hesitate() -> None:
    report = {"fit": {"views": {"wrist_left": [{"px": 6.0}, {"px": 5.0}]}},
              "held_out": {"2": {"wrist_right": [{"mm": 14.0}]}}, "placements_per_arm": {"left": 3, "right": 1}}
    text = json.dumps(handeye.concerns(report), ensure_ascii=False)
    assert "wrist_left 的画面拟合误差中位 5.5 像素" in text
    assert "第 2 个摆位的留一验证误差中位 14 mm" in text
    assert "右臂只在 1 个摆位" in text and "左臂" not in text


def test_a_poorly_predicted_arm_is_offered_one_more_placement(placed, stage) -> None:
    asker = Scripted(True, True, True, True, True, True, True, "不加", True)
    report = _run(placed, asker, placements=1)
    titles = [title for _, title, _ in asker.asked]
    assert titles[3].startswith("再加一个摆位？") and "第 2/2 次" in titles[4]
    assert stage["runs"] == [["left", "right"], [titles[3][-3:-1] == "左臂" and "left" or "right"]]
    assert report["saved"] is True and len(report["records"]) == 2


def test_placement_hints_differ_for_each_turn_of_an_arm() -> None:
    turns = handeye.schedule(["left", "right"], 7)
    hints = [handeye.placement_hint(turns, i) for i in range(7)]
    assert hints[0] == handeye.BOTH_HINT
    assert hints[1] != hints[3] != hints[5] and "左臂" in hints[1] and "右臂" in hints[2]


def test_weak_arms_name_the_worse_predicted_first() -> None:
    report = {"held_out": {"1": {"wrist_left": [{"mm": 14.0}], "wrist_right": [{"mm": 4.0}]}},
              "placements_per_arm": {"left": 3, "right": 2}}
    assert handeye.weak_arms(report, {"wrist_left": "left", "wrist_right": "right"}) == ["right", "left"]
    assert report["weak"] == {"left": "留一验证误差中位 14 mm", "right": "只拍了 2 个摆位"}
