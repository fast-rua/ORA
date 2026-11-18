"""A call's budget is large enough that the call finishes: only what the episode has left cuts a move."""

import pytest

from ora.agent.toolkit.tools.budgets import MOVE_BUDGET, NATIVE_ACTION_BUDGETS, native_action_budget
from ora.agent.toolkit.tools.move_pose import _smooth
from ora.contracts.joint_trajectory import MAX_TRAJECTORY_NATIVE_ACTIONS


@pytest.mark.parametrize("name", ["move_pose", "move_to", "rotate_wrist", "rotate_pitch", "recover_pose", "return_home"])
def test_a_move_may_use_all_one_trajectory_may(name):
    assert native_action_budget(name) == MOVE_BUDGET == MAX_TRAJECTORY_NATIVE_ACTIONS


def test_a_grasp_covers_each_stage_at_its_own_limit():
    assert native_action_budget("grasp_execute") == 3 * MOVE_BUDGET + 2 * native_action_budget("set_gripper")


def test_one_trajectory_never_asks_for_more_than_it_may_use():
    assert _smooth.allowance(NATIVE_ACTION_BUDGETS["grasp_execute"], {}) == (MOVE_BUDGET, "tool_native_action_budget")


def test_the_episode_and_the_script_cut_a_move_only_at_what_they_have_left():
    config = {"episode_native_actions_remaining": 120, "script_native_actions_remaining": 90}
    assert _smooth.allowance(MOVE_BUDGET, config) == (90, "script_native_actions_remaining")
    assert _smooth.allowance(MOVE_BUDGET, {"episode_native_actions_remaining": 120}) == (
        120, "episode_native_actions_remaining")
