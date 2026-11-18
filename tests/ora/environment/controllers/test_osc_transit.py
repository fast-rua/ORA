"""The straight OSC line of a clear free-space move: its route choice, reference and profile section."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ora.environment.controllers import osc_transit
from ora.environment.controllers.hybrid_motion import (
    HybridMotionProfileError, libero_osc_transit, validate_hybrid_motion_profile,
)

ROOT = Path(__file__).resolve().parents[4]
V3 = json.loads((ROOT / "ora/policy/embodiments/franka-panda/motion-profile-v3.json").read_text())
V4 = json.loads((ROOT / "ora/policy/embodiments/franka-panda/motion-profile-v4.json").read_text())
SECTION = V4["osc_transit"]
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _yaw(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def test_v4_is_v3_plus_its_osc_transit_section() -> None:
    assert validate_hybrid_motion_profile(V4) == V4
    assert libero_osc_transit(V4) == SECTION and libero_osc_transit(V3) is None
    assert {k: v for k, v in V4.items() if k not in {"profile_id", "osc_transit"}} == {
        k: v for k, v in V3.items() if k != "profile_id"}
    missing = json.loads(json.dumps(V4))
    missing.pop("osc_transit")
    with pytest.raises(HybridMotionProfileError, match="fields drifted"):
        validate_hybrid_motion_profile(missing)
    # 2 / sqrt(150) x 0.35 m/s = 0.057 m, past the OSC controller's 0.05 m reach.
    fast = json.loads(json.dumps(V4))
    fast["osc_transit"]["follow_speed_m_s"] = 0.35
    with pytest.raises(HybridMotionProfileError, match="past its reach"):
        validate_hybrid_motion_profile(fast)


@pytest.mark.parametrize("kwargs,route,reason", [
    ({}, osc_transit.ROUTE_OSC, "planner_path_clear_and_straight"),
    ({"interaction": "contact_approach"}, osc_transit.ROUTE_JOINT, "not_free_space"),
    ({"slow_approaching": True}, osc_transit.ROUTE_JOINT, "slow_approaching_requested"),
    ({"plan_reference": True}, osc_transit.ROUTE_JOINT, "previewed_plan_runs_as_planned"),
    ({"rotation_rad": math.radians(46)}, osc_transit.ROUTE_JOINT, "turn_above_osc_limit"),
    ({"planner_path_positions": None}, osc_transit.ROUTE_JOINT, "planner_path_unavailable"),
    # A detour 3 cm off the straight line, past the 2 cm the section admits.
    ({"planner_path_positions": [[0, 0, 0], [0.1, 0.03, 0], [0.2, 0, 0]]}, osc_transit.ROUTE_JOINT,
     "planner_path_not_straight"),
])
def test_the_route_choice_names_its_reason(kwargs, route, reason) -> None:
    arguments = {"interaction": "free_space", "slow_approaching": False, "plan_reference": False,
                 "planner_path_positions": [[0, 0, 0], [0.1, 0.015, 0], [0.2, 0, 0]], "rotation_rad": 0.3}
    choice = osc_transit.route_choice(SECTION, **{**arguments, **kwargs})
    assert (choice["route"], choice["reason"]) == (route, reason)
    assert choice["collision_checked"] == (
        "planner_gripper_path_only" if route == osc_transit.ROUTE_OSC else "planner_joint_path")
    assert osc_transit.route_choice(None, **arguments) is None


def test_path_deviation_measures_from_the_straight_segment() -> None:
    assert osc_transit.path_deviation([[0, 0, 0], [0.5, 0.02, 0], [1, 0, 0]]) == pytest.approx(0.02)
    # A point beyond an end is measured from that end.
    assert osc_transit.path_deviation([[0, 0, 0], [1.03, 0, 0], [1, 0, 0]]) == pytest.approx(0.03)


def test_the_line_ends_on_its_goal_within_its_caps() -> None:
    start, goal = [0.0, 0.0, 1.0, *IDENTITY], [0.3, 0.0, 0.9, *_yaw(30)]
    lead = osc_transit.lead_s(SECTION)
    line = osc_transit.straight_line_reference(
        start, goal, linear_m_s=0.25, angular_rad_s=1.5, linear_accel_m_s2=2.0, angular_accel_rad_s2=12.0,
        tick_s=0.05, lead_s=lead)
    length = math.dist(start[:3], goal[:3])
    assert line["length_m"] == pytest.approx(length) and line["turn_rad"] == pytest.approx(math.radians(30))
    # Cruise at 0.25 m/s with 0.125 s ramps at each end: about 1.39 s, whole ticks.
    assert line["ticks"] == math.ceil((length / 0.25 + 0.125) / 0.05 - 1e-9)
    assert line["aims"][-1] == pytest.approx(goal)
    progress = [math.dist(start[:3], aim[:3]) / length for aim in line["aims"]]
    assert progress == sorted(progress) and max(progress) <= 1.0 + 1e-12
    # Mid-line, each aim leads the line by the controller's lag times the cruise speed.
    assert progress[len(progress) // 2] - progress[len(progress) // 2 - 1] == pytest.approx(
        0.25 * 0.05 / length, rel=0.05)


def test_a_slower_request_takes_more_ticks_and_a_pure_turn_is_paced_by_its_turn() -> None:
    start, goal = [0.0, 0.0, 1.0, *IDENTITY], [0.2, 0.0, 1.0, *IDENTITY]
    common = dict(angular_rad_s=1.5, angular_accel_rad_s2=12.0, tick_s=0.05, lead_s=0.163)
    fast = osc_transit.straight_line_reference(start, goal, linear_m_s=0.25, linear_accel_m_s2=2.0, **common)
    careful = osc_transit.straight_line_reference(start, goal, linear_m_s=0.1, linear_accel_m_s2=0.5, **common)
    assert careful["ticks"] > 2 * fast["ticks"]
    turn = osc_transit.straight_line_reference(start, [0.0, 0.0, 1.0, *_yaw(40)], linear_m_s=0.25,
                                               linear_accel_m_s2=2.0, **common)
    assert turn["ticks"] >= math.ceil(math.radians(40) / 1.5 / 0.05)
    assert osc_transit.straight_line_reference(start, start, linear_m_s=0.25, linear_accel_m_s2=2.0,
                                               **common)["ticks"] == 0
