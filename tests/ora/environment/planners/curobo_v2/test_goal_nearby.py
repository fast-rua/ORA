"""The tilted goals a refused pose is tried at, and the words for what they showed."""
from __future__ import annotations

import numpy as np
import pytest

from ora.environment.controllers.depth_world import pose_matrix, tool_from_hand
from ora.environment.planners.curobo_v2.goal_nearby import (
    TILT_DIRECTIONS, TILTS_DEG, nearby_sentence, tilted_goals, what_blocks,
)

GRIP_M = 0.1576
#: The X5's tool frame, hand pointing straight down, out from the base along +x.
POSE = [0.4, 0.0, 0.3, *np.asarray([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)]).tolist()]
TOOL = tool_from_hand("arx_x5")


def _hand(pose):
    return pose_matrix(pose) @ TOOL


def test_tilts_run_smallest_first_and_the_search_is_bounded():
    goals = list(tilted_goals(POSE, TOOL, GRIP_M))
    assert len(goals) == len(TILTS_DEG) * TILT_DIRECTIONS
    assert [goal["tilt_deg"] for goal in goals] == sorted(goal["tilt_deg"] for goal in goals)


def test_every_goal_keeps_the_grip_point_and_turns_the_approach_by_its_tilt():
    hand = _hand(POSE)
    grip = hand[:3, 3] + hand[:3, 2] * GRIP_M
    for goal in tilted_goals(POSE, TOOL, GRIP_M):
        moved = _hand(goal["pose_base_xyzw"])
        assert moved[:3, 3] + moved[:3, 2] * GRIP_M == pytest.approx(grip)
        assert moved[:3, 2] == pytest.approx(np.asarray(goal["approach_base"]))
        cosine = float(moved[:3, 2] @ hand[:3, 2])
        assert np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) == pytest.approx(goal["tilt_deg"], abs=1e-6)


def test_the_first_direction_leans_outward_from_the_base():
    first = next(iter(tilted_goals(POSE, TOOL, GRIP_M)))
    assert first["approach_base"][0] > 0.1 and abs(first["approach_base"][1]) < 1e-9


def test_a_goal_over_the_base_still_has_directions():
    above = [0.0, 0.0, 0.3, *POSE[3:]]
    assert len(list(tilted_goals(above, TOOL, 0.0))) == len(TILTS_DEG) * TILT_DIRECTIONS


@pytest.mark.parametrize(("checks", "expected"), [
    ([{"self_collision": 1.0, "world_collision": 0.0}] * 2, "self"),
    ([{"self_collision": 0.0, "world_collision": 1.0}, {"self_collision": 1.0, "world_collision": 1.0}], "world"),
    ([{"self_collision": 1.0, "world_collision": 0.0}, {"self_collision": 0.0, "world_collision": 1.0}], None),
    ([{"self_collision": 0.0, "world_collision": None}], None), ([], None),
])
def test_what_stops_the_seeds_is_what_stops_them_alone(checks, expected):
    assert what_blocks(checks) == expected


def test_the_sentence_gives_the_tilt_in_world_coordinates():
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    found = {"tilt_deg": 20.0, "approach_base": [0.34, 0.0, -0.94]}
    assert nearby_sentence(found, [0.0, 0.0, -1.0], turn) == (
        "with the fingertips at the same point, an approach tilted 20 degrees plans: "
        "(0.00, 0.34, -0.94) in world coordinates instead of (0.00, 0.00, -1.00)")


def test_a_rounded_zero_has_no_sign():
    found = {"tilt_deg": 10.0, "approach_base": [-0.001, 0.0, -1.0]}
    assert "(0.00, 0.00, -1.00) in world coordinates" in nearby_sentence(found, [0.0, 0.0, -1.0], np.eye(3))


def test_nothing_nearby_is_said_plainly():
    assert nearby_sentence(None, [0.0, 0.0, -1.0], np.eye(3)) == (
        "no approach tilted up to 45 degrees was found to plan here either")
