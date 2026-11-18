"""The read-only plan of one action (``contracts/action_plan.py``)."""
import pytest

from ora.contracts import action_plan


def test_a_plan_request_is_told_apart_on_the_read_only_bridge():
    assert action_plan.is_request(action_plan.request({"action": "move_pose"}))
    assert not action_plan.is_request({"schema": "ora.reach-check-request.v1"}) and not action_plan.is_request(None)


def test_a_kind_of_segment_is_one_of_the_four():
    with pytest.raises(ValueError):
        action_plan.segment("wrist", "left", "x", poses=[], planned=True)
