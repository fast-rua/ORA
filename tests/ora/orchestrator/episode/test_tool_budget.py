"""Measured native spend, independent caps and one count for a compound tool."""

import pytest

from ora.orchestrator.episode.tool_budget import ToolBudget


def test_budget_charges_actual_spend_and_keeps_refusals_available():
    budget = ToolBudget("call-1", 42)
    assert budget.admit(100, 200) == 20
    assert not budget.charge(0, 20)
    assert budget.remaining == 42
    assert budget.charge(3, 20)
    assert not budget.charge(20, 20)
    assert budget.remaining == 19
    assert budget.admit(20, 200) == 19
    assert budget.admit(8, 200) == 8
    assert budget.admit(20, 2) == 2
    assert not budget.charge(19, 19)
    assert budget.admit(20, 200) == 0
    assert budget.used == 42


@pytest.mark.parametrize("value", [-1, True, 1.5, None, "20"])
def test_invalid_budgets_are_not_coerced(value):
    with pytest.raises(ValueError):
        ToolBudget("call", value)
    budget = ToolBudget("call", 20)
    with pytest.raises(ValueError):
        budget.admit(value, 100)
    with pytest.raises(ValueError):
        budget.charge(value, 20)


def test_overconsumption_is_a_contract_failure():
    budget = ToolBudget("call", 3)
    with pytest.raises(ValueError, match="exceeds"):
        budget.charge(4, 3)
    assert budget.used == 0


def test_a_trajectory_or_rate_limited_gripper_may_use_more_than_twenty_native_actions():
    from ora.contracts.joint_trajectory import native_action_cap
    from ora.orchestrator.episode.execution_state import ExecutionStateOwner
    from ora.orchestrator.episode.tool_budget import ToolBudget

    budget = ToolBudget("call", 60)
    trajectory = {"action": "follow_joint_trajectory", "arguments": {}}
    assert budget.admit(55, 100, native_action_cap(trajectory)) == 55
    assert budget.admit(55, 100, native_action_cap({"action": "move_pose", "arguments": {}})) == 20
    assert budget.admit(55, 100) == 20
    with ExecutionStateOwner().limited_action(55):
        pass
