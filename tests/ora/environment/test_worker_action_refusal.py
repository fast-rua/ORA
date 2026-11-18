"""Only RoboDojo executes joint trajectories and rate-limited jaws; every other backend refuses them typed."""
from types import SimpleNamespace

import pytest

from ora.contracts.refusal import EnvironmentRefusal
from ora.environment import worker


def test_other_backends_refuse_the_per_physics_step_actions_before_executing():
    executed = []
    session = SimpleNamespace(execute=lambda action: executed.append(action) or {"ok": True})
    trajectory = {"action": "follow_joint_trajectory", "arguments": {"arms": {}}}
    with pytest.raises(EnvironmentRefusal) as error:
        worker._dispatch(session, {}, "execute", {"action": trajectory})
    assert error.value.field == "action" and not executed
    ramp = {"action": "set_gripper", "arguments": {"targets": {"left": {"opening": "close"}}}}
    with pytest.raises(EnvironmentRefusal):
        worker._dispatch(session, {}, "execute", {"action": ramp})
    legacy = {"action": "set_gripper", "arguments": {"targets": {"left": "close"}}}
    assert worker._dispatch(session, {}, "execute", {"action": legacy}) == ({"ok": True}, False)
