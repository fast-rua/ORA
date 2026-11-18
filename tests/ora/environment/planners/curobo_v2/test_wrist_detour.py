"""The straight joint lines that may replace a plan turning the wrist the long way (``curobo_v2/wrist_detour.py``)."""
from __future__ import annotations

import math

import pytest

from ora.environment.planners.curobo_v2 import wrist_detour

ALOHA = [[-4.2] * 2, [4.2] * 2]
PANDA = [[-2.8973] * 2, [2.8973] * 2]


def test_a_goal_takes_the_2_pi_equivalent_nearest_its_start_only_inside_the_limits():
    assert wrist_detour.nearest_alias([0.0, -2.0], [0.1, 3.5], ALOHA) == pytest.approx([0.1, 3.5 - 2 * math.pi])
    # Already the nearest, or the nearer one out of range: no alias.
    assert wrist_detour.nearest_alias([0.0, 1.0], [0.1, 3.5], ALOHA) is None
    assert wrist_detour.nearest_alias([0.0, -2.0], [0.1, 2.5], PANDA) is None


def test_lines_are_ranked_within_the_rule_then_by_the_smallest_turn_and_the_plan_wins_a_tie():
    def turns(path: float, requested: float, source: str) -> dict:
        return {"source": source, "path_turn_deg": path, "requested_turn_deg": requested}

    candidates = [turns(40.0, 30.0, "planned"), turns(300.0, 10.0, "alias"), turns(40.0, 40.0, "same_goal")]
    blocked_asked = []
    chosen = wrist_detour.choose(candidates, {1: "alias line", 2: "same line"},
                                 lambda line: blocked_asked.append(line))
    assert chosen == 0 and blocked_asked == []
    assert [item["outcome"] for item in candidates] == ["chosen", "unwinds", "ranked_lower"]
    # A blocked line yields to the next; a line the rule refuses is never answered.
    candidates = [turns(20.0, 20.0, "alias"), turns(25.0, 25.0, "same_goal"), turns(400.0, 0.0, "ik_pool")]
    chosen = wrist_detour.choose(candidates, {0: "a", 1: "b", 2: "c"},
                                 lambda line: {"check": "world"} if line == "a" else None)
    assert chosen == 1 and [item["outcome"] for item in candidates] == ["blocked", "chosen", "unwinds"]
    assert wrist_detour.choose([turns(400.0, 0.0, "same_goal")], {0: "a"}, lambda line: None) is None


def test_the_pool_is_the_accepted_in_range_solutions_once_each_nearest_first():
    attempts = [{"status": "captured", "accepted": [True, True, True, False, True, True],
                 "joint_positions": [[0.0, 3.5], [0.0, 3.5 - 2 * math.pi], [0.4, 0.2], [0.1, 0.1], [9.0, 0.0],
                                     [0.0, 3.0]]}, {"status": "unavailable"}]
    goals = wrist_detour.pool_goals(attempts, [0.0, -2.0], ALOHA, known=[[0.0, 3.0]])
    # 3.5 and its equivalent are one goal, taken at the equivalent, which is the nearer; 0.1 was not accepted, 9.0
    # is out of range and 3.0 is the plan's own (its equivalent 3.0 - 2 pi is the same goal).
    assert len(goals) == 2 and goals[0] == pytest.approx([0.0, 3.5 - 2 * math.pi])
    assert goals[1] == pytest.approx([0.4, 0.2])


def test_a_line_is_a_trajectory_at_rest_at_both_ends():
    line = wrist_detour.straight_line([0.0, 0.0], [0.1, -0.05])
    assert len(line) == 6 and line[-1].tolist() == [0.1, -0.05]
    trajectory = wrist_detour.as_trajectory(line, ["a", "b"], 0.02)
    assert trajectory["velocities"][0] == [0.0, 0.0] == trajectory["velocities"][-1]
    assert trajectory["velocities"][2] == pytest.approx([1.0, -0.5]) and trajectory["dt_s"] == 0.02
