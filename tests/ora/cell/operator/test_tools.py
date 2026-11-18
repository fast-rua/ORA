"""Which way each gripper opens is what the person sees: a reading the answer contradicts swaps
that gripper's [closed, open] limits; nothing moves, and a gripper stopped midway is not judged."""

from __future__ import annotations

import pytest

from ora.cell.errors import CellError
from ora.cell.operator import tools


class Answers:
    def __init__(self, *values) -> None:
        self.values, self.asked = list(values), []

    def value(self, title, text, details, timeout_s):
        self.asked.append((details["arm"], title))
        answer = self.values.pop(0)
        return (answer, None) if answer is not None else (None, "no answer")


@pytest.fixture
def limited(cell):
    value = cell.calibration()
    value["gripper_limits"] = {"left": [6.38, 1.11], "right": [6.40, 0.98]}
    cell.save_calibration(value)
    return cell


def test_a_gripper_the_person_sees_closed_at_its_open_end_is_swapped(limited, adapter) -> None:
    adapter.gripper_raw = {"left": 1.27, "right": 0.73}      # the station on 2026-10-29: left open, right closed
    answers = Answers("开", "合")
    report = tools.orient_grippers(limited, answers)

    assert [arm for arm, _ in answers.asked] == ["left", "right"]
    assert report["saved"] is True and report["grippers"]["left"].startswith("matches: open")
    assert report["grippers"]["right"].startswith("swapped")
    assert limited.gripper_limits() == {"left": [6.38, 1.11], "right": [0.98, 6.40]}
    assert any("swapped" in note for note in limited.calibration()["notes"])


def test_an_unclear_answer_is_asked_again_and_a_match_saves_nothing(limited, adapter) -> None:
    adapter.gripper_raw = {"left": 6.30, "right": 6.35}
    answers = Answers("half", "closed", "合")
    report = tools.orient_grippers(limited, answers)
    assert [arm for arm, _ in answers.asked] == ["left", "left", "right"]
    assert report["saved"] is False
    assert limited.gripper_limits() == {"left": [6.38, 1.11], "right": [6.40, 0.98]}


def test_a_gripper_stopped_midway_is_not_judged(limited, adapter) -> None:
    adapter.gripper_raw = {"left": 3.7, "right": 0.98}
    answers = Answers("开")
    report = tools.orient_grippers(limited, answers)
    assert [arm for arm, _ in answers.asked] == ["right"]
    assert "between its limits" in report["grippers"]["left"]
    assert limited.gripper_limits()["left"] == [6.38, 1.11]


def test_without_an_answer_nothing_is_saved(limited, adapter) -> None:
    adapter.gripper_raw = {"left": 1.2, "right": 1.0}
    with pytest.raises(CellError, match="no answer for the left gripper"):
        tools.orient_grippers(limited, Answers(None))
    assert limited.gripper_limits()["right"] == [6.40, 0.98]
