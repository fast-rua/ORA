"""Where a joint trajectory stopped and which joint fell behind, from the receipt alone."""
from __future__ import annotations

import pytest

from ora.agent.session_core.stop_evidence import stop_evidence, with_stop_evidence

KNOTS = [[0.1 * row, 0.0, 0.2 * row] for row in range(1, 11)]


def _record(*, stop, residual=(0.5, 0.0, 1.2), planned=10):
    return {"arguments": {"arms": {"left": {"knots_rad": KNOTS}}},
            "trajectory": {"knots_planned": planned, "stop": stop,
                           "arms": {"left": {"final_joint_error_per_joint_rad": list(residual)}}}}


def test_the_joint_furthest_from_the_commanded_joints_at_the_stop_fell_behind():
    arm, evidence = stop_evidence(_record(stop={"arm": "left", "reason": "tracking", "knot": 5, "physics_step": 10}))
    assert arm == "left"
    # Commanded at knot 5: joint 3 at 1.0; measured: 2.0 - 1.2 = 0.8.
    assert evidence == {"path_fraction": 0.5, "lagging_joint": 3, "lag_rad": pytest.approx(0.2)}


def test_the_commanded_joints_are_read_inside_the_knot_the_stop_names():
    _arm, evidence = stop_evidence(_record(stop={"arm": "left", "reason": "tracking", "knot": 5, "physics_step": 5}))
    assert evidence["lag_rad"] == pytest.approx(0.1)  # halfway from knot 4 (0.8) to knot 5 (1.0)


def test_the_first_knot_has_no_knot_before_it():
    _arm, evidence = stop_evidence(_record(stop={"arm": "left", "reason": "progress", "knot": 1, "physics_step": 3},
                                           residual=(0.9, 0.0, 1.8)))
    assert evidence["path_fraction"] == 0.1 and evidence["lag_rad"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("record", [
    {}, {"trajectory": {"stop": None}}, _record(stop={"arm": "left", "reason": "terminal", "knot": 5}),
    _record(stop={"arm": "right", "reason": "tracking", "knot": 5}),
    _record(stop={"arm": "left", "reason": "tracking", "knot": 11}),
    _record(stop={"arm": "left", "reason": "tracking", "knot": 5}, residual=(0.0,)),
])
def test_a_receipt_that_cannot_say_gives_nothing(record):
    assert stop_evidence(record) is None


def test_only_the_stopped_arm_carries_it():
    record = _record(stop={"arm": "left", "reason": "tracking", "knot": 5, "physics_step": 10})
    arms = with_stop_evidence({"left": {"targeted": True}, "right": {"targeted": False}}, record)
    assert "stop_evidence" in arms["left"] and "stop_evidence" not in arms["right"]
    untouched = {"left": {"targeted": True}}
    assert with_stop_evidence(untouched, {}) is untouched


def _endpoint(**endpoint):
    return {"target_arms": ["right"], "endpoint": {"stop_reason": "settle_stalled", "path_samples": 25,
                                                  "samples_commanded": 25, "ticks": 27, **endpoint}}


def test_an_executor_that_names_the_joint_itself_is_read_as_it_recorded_it():
    assert stop_evidence(_endpoint(lagging_joint=4, lag_rad=0.1274)) == (
        "right", {"lagging_joint": 4, "lag_rad": 0.1274})
    # A stop before the last knot also says how far along its knots it was.
    assert stop_evidence(_endpoint(stop_reason="joint_lag", path_fraction=0.52, lagging_joint=2, lag_rad=0.26)) == (
        "right", {"path_fraction": 0.52, "lagging_joint": 2, "lag_rad": 0.26})


@pytest.mark.parametrize("record", [
    _endpoint(), _endpoint(stop_reason="contact_detected"), _endpoint(lagging_joint=0, lag_rad=0.1),
    _endpoint(lagging_joint=True, lag_rad=0.1), _endpoint(lagging_joint=4, lag_rad=float("nan")),
    {**_endpoint(lagging_joint=4, lag_rad=0.1), "target_arms": ["left", "right"]},
])
def test_an_endpoint_that_names_no_joint_gives_nothing(record):
    assert stop_evidence(record) is None
