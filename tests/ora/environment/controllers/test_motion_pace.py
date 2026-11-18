"""Planner-neutral pacing: caps at the controlled point, joints as the ceiling, the wrist check and the budget."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ora.contracts.motion_speed import PRESET_CAPS, SLOW_APPROACH_M, speed_caps
from ora.environment.controllers import motion_pace as mp

TICK = 0.05
LIMITS = (-2.8973, 2.8973)


def fk(rows):
    """A test arm: joints 0-2 move the point in x, y, z (0.3 m per rad), joint 6 turns it about z."""

    poses = []
    for q in rows:
        poses.append([0.4 + 0.3 * q[0], 0.3 * q[1], 0.5 + 0.3 * q[2], 0.0, 0.0,
                      math.sin(q[6] / 2), math.cos(q[6] / 2)])
    return poses


def line(start, end, points=5):
    start, end = np.asarray(start, float), np.asarray(end, float)
    return [(start + k / (points - 1) * (end - start)).tolist() for k in range(points)]


def paced(path, speed=None, **options):
    arguments = dict(tick_s=TICK, joint_speed_ceiling_rad_s=1.0, wrist_joint_index=6, wrist_limits_rad=LIMITS,
                     budget_ticks=2000)
    arguments.update(options)
    return mp.pace_path(path, fk, speed=speed, **arguments)


def steps(result):
    poses = fk([result.grid.joints["arm"][0].tolist(), *result.rows()])
    linear = [math.dist(a[:3], b[:3]) for a, b in zip(poses, poses[1:])]
    joints = np.abs(np.diff(np.asarray([result.grid.joints["arm"][0], *result.rows()]), axis=0)).max(axis=1)
    return np.asarray(linear), joints


Q0 = [0.0] * 7
QX = [2.0, 0, 0, 0, 0, 0, 0]  # 0.6 m along x


@pytest.mark.parametrize("speed,cap", [
    ({"speed": "normal", "slow_approaching": False}, 0.3),
    ({"speed": "careful", "slow_approaching": False}, 0.1),
    ({"speed": "custom", "speed_m_s": 0.05, "slow_approaching": False}, 0.05),
])
def test_the_controlled_point_keeps_the_requested_speed(speed, cap):
    result = paced(line(Q0, QX), speed)
    linear, joints = steps(result)
    assert linear.max() <= cap * TICK + 1e-9
    assert joints.max() <= 1.0 * TICK + 1e-9
    assert result.speed["planned_peak_linear_m_s"] == pytest.approx(cap, abs=1e-4)
    assert result.knots["ticks"] >= math.ceil(0.6 / cap / TICK)
    assert result.rows()[-1] == pytest.approx(QX)


def test_fast_has_no_cartesian_cap_and_the_joint_ceiling_limits_it():
    result = paced(line(Q0, QX))
    linear, joints = steps(result)
    # 1 rad/s on joint 0 is 0.3 m/s of this arm's point: the joints, not a Cartesian cap, set the speed.
    assert joints.max() <= 1.0 * TICK + 1e-9 and linear.max() == pytest.approx(0.3 * TICK, rel=0.01)
    assert result.speed == {"speed": "fast", "slow_approaching": False, "linear_m_s": None, "angular_deg_s": None,
                            "joint_speed_deg_s": round(math.degrees(1.0), 3),
                            "planned_peak_linear_m_s": pytest.approx(0.3, abs=1e-3),
                            "planned_peak_angular_deg_s": 0.0}
    # Without a backend ceiling the toolkit's 4 rad/s joint cap holds.
    free = paced(line(Q0, QX), joint_speed_ceiling_rad_s=None)
    assert free.knots["ticks"] < result.knots["ticks"]


def test_slow_approaching_holds_the_last_two_centimetres_to_careful():
    result = paced(line(Q0, QX), {"speed": "normal", "slow_approaching": True})
    linear, _ = steps(result)
    poses = fk(result.rows())
    left = [math.dist(pose[:3], fk([QX])[0][:3]) for pose in poses]
    # The slow segment starts at the grid node at or before 2 cm from the end, so every tick that starts
    # within the last 2 cm moves at most careful's 0.1 m/s.
    starts = [0.6] + left[:-1]
    assert max(step for step, remaining in zip(linear, starts) if remaining <= 0.02 + 1e-9) <= 0.1 * TICK + 1e-9
    assert linear.max() > 0.1 * TICK
    assert result.slow_from_knot is not None
    assert result.speed["slow"]["linear_m_s"] == 0.1 and result.speed["slow_approach_m"] == 0.02
    unzoned = paced(line(Q0, QX), {"speed": "normal", "slow_approaching": True}, slow_approach=False)
    assert unzoned.slow_pace is None and unzoned.knots["ticks"] < result.knots["ticks"]


def test_a_turn_the_move_did_not_ask_for_is_refused_before_anything_moves():
    half = math.pi
    unwind = [Q0, [0, 0, 0, 0, 0, 0, half / 2], [0, 0, 0, 0, 0, 0, half], [0, 0, 0, 0, 0, 0, half / 2], Q0]
    with pytest.raises(mp.PaceRefusal) as error:
        paced(unwind)
    assert error.value.code == "paced_path_wrist_unwind"
    assert error.value.constraints["path_turn_deg"] == pytest.approx(360.0, abs=0.1)
    assert error.value.constraints["requested_turn_deg"] == pytest.approx(0.0, abs=0.1)
    # Read at the knots, which need not land on the turning point.
    assert error.value.constraints["wrist"]["travel_deg"] == pytest.approx(360.0, abs=5.0)
    # A planner's small detour, and a large turn the move does ask for, both pass.
    detour = [Q0, [0.5, 0, 0, 0, 0, 0, 0.3], [1.0, 0, 0, 0, 0, 0, 0]]
    assert paced(detour).wrist["path_turn_deg"] == pytest.approx(34.38, abs=0.01)
    with pytest.raises(mp.PaceRefusal):
        paced([Q0, [0.5, 0, 0, 0, 0, 0, 0.4], [1.0, 0, 0, 0, 0, 0, 0]])  # 45.8 degrees beyond the asked 0
    asked = paced([Q0, [0, 0, 0, 0, 0, 0, 2.5]])
    assert asked.wrist["requested_turn_deg"] == pytest.approx(143.24, abs=0.01)
    assert asked.wrist["travel_deg"] == pytest.approx(143.24, abs=0.01)
    assert asked.wrist["min_limit_margin_deg"] == pytest.approx(math.degrees(2.8973 - 2.5), abs=0.01)


def test_the_long_way_round_at_the_joint_limit_is_an_unwind():
    # Asked: a 146 degree turn. Planned: the wrist turns 214 degrees the other way.
    start, end = math.radians(-74.0), math.radians(140.0)
    with pytest.raises(mp.PaceRefusal) as error:
        paced([[0, 0, 0, 0, 0, 0, start], [0, 0, 0, 0, 0, 0, end]])
    assert error.value.constraints["requested_turn_deg"] == pytest.approx(146.0, abs=0.01)
    assert error.value.constraints["unrequested_turn_deg"] == pytest.approx(68.0, abs=0.01)


def test_a_move_over_its_budget_is_refused_whole_or_cut_outside_a_wrist_turn():
    path = line(Q0, QX)
    with pytest.raises(mp.PaceRefusal) as error:
        paced(path, {"speed": "careful", "slow_approaching": False}, budget_ticks=20)
    assert error.value.code == "paced_motion_budget"
    assert error.value.constraints["tick_budget"] == 20 and error.value.constraints["needed_ticks"] > 20
    cut = paced(path, {"speed": "careful", "slow_approaching": False}, budget_ticks=20, cut_to_budget=True)
    assert cut.knots["ticks"] <= 20 and 0 < cut.fraction < 1
    assert cut.report()["budget_limited"]["needed_ticks"] == error.value.constraints["needed_ticks"]
    # A wrist turn of 90 degrees or more that starts the move is never cut part way.
    turn = [Q0, [0, 0, 0, 0, 0, 0, 2.0], [0.5, 0, 0, 0, 0, 0, 2.0]]
    with pytest.raises(mp.PaceRefusal) as error:
        paced(turn, {"speed": "careful", "slow_approaching": False}, budget_ticks=30, cut_to_budget=True)
    assert error.value.constraints["unsplit_span"] == {"from_fraction": 0.0, "to_fraction": pytest.approx(0.8)}


def test_a_budget_cut_moves_back_to_the_start_of_a_wrist_turn():
    path = [Q0, [1.0, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 2.0]]
    grid = mp.joint_path_grid(path, fk, wrist_joint_index=6).grid
    assert grid.atomic == ((50, 150),)
    pace, _ = mp.paces({"speed": "careful", "slow_approaching": False})
    whole = mp.knots(grid, pace, tick_s=TICK)
    part, planned, fraction = mp.fit_budget(grid, pace, whole["ticks"] - 5, tick_s=TICK)
    assert part.segments == 50 and fraction == pytest.approx(1 / 3)
    assert planned["joints"]["arm"][-1] == pytest.approx([1.0, 0, 0, 0, 0, 0, 0])


def test_wrist_turn_spans_are_monotonic_runs_of_a_quarter_turn_or_more():
    assert mp.wrist_turn_spans([0, 1, 2, 2, 1]) == ((0, 2),)
    assert mp.wrist_turn_spans([0, 1, 0.9, 2.6]) == ((2, 3),)
    assert mp.wrist_turn_spans([0, 0.5, 1.0, 0.2]) == ()


def test_a_still_path_is_one_tick():
    result = paced([Q0, Q0])
    assert result.knots["ticks"] == 1 and result.rows() == [Q0]
    assert result.wrist["travel_deg"] == 0.0


def test_forward_kinematics_must_answer_every_row():
    with pytest.raises(ValueError):
        mp.pace_path(line(Q0, QX), lambda rows: fk(rows)[:-1], speed=None, tick_s=TICK,
                     joint_speed_ceiling_rad_s=1.0, wrist_joint_index=6, wrist_limits_rad=LIMITS, budget_ticks=100)


def test_a_local_executor_caps_its_step_per_tick():
    assert mp.step_caps(None, tick_s=TICK) is None
    assert mp.step_caps({"speed": "fast", "slow_approaching": False}, tick_s=TICK) is None
    careful = mp.step_caps({"speed": "careful", "slow_approaching": False}, tick_s=TICK)
    assert careful.linear_m == pytest.approx(0.005) and careful.angular_rad == pytest.approx(math.radians(1.0))
    translation, rotation = careful.apply([0.02, 0.0, 0.0], [0.0, 0.0, 0.1], remaining_m=0.3)
    assert np.linalg.norm(translation) == pytest.approx(0.005) and np.linalg.norm(rotation) == pytest.approx(
        math.radians(1.0))
    small, _ = careful.apply([0.001, 0.0, 0.0], [0.0, 0.0, 0.0], remaining_m=0.001)
    assert small.tolist() == [0.001, 0.0, 0.0]
    assert careful.min_ticks(0.03, 0.0) == 6
    # fast with slow_approaching: uncapped until the last 2 cm, careful inside them.
    slow = mp.step_caps({"speed": "fast", "slow_approaching": True}, tick_s=TICK)
    assert slow.limits(0.05) == (None, None)
    assert slow.limits(0.02) == pytest.approx((0.005, math.radians(1.0)))
    assert slow.min_ticks(0.05, 0.0) == 4
    normal = mp.step_caps({"speed": "normal", "slow_approaching": True}, tick_s=TICK)
    assert normal.limits(0.05) == pytest.approx((0.015, math.radians(3.0)))
    assert normal.min_ticks(0.05, 0.0) == math.ceil(0.05 / 0.015 + 0.02 / 0.005)


# --------------------------------------------------------------- ramps and the slow zone (a TOPP-RA scoping
# on 219 recorded LIBERO plans, 2026-10-31: repeated planner rows let a move skip its 0.25 s ramp, and the
# slow zone began up to one grid node inside the last 2 cm)

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = json.loads((ROOT / "tests/data/motion-pace-paths.json").read_text(encoding="utf-8"))["paths"]
RAMPED = {"normal": {"speed": "normal", "slow_approaching": False},
          "careful": {"speed": "careful", "slow_approaching": False},
          "normal_slow": {"speed": "normal", "slow_approaching": True},
          "custom_0.05": {"speed": "custom", "speed_m_s": 0.05, "slow_approaching": False},
          "custom_0.5": {"speed": "custom", "speed_m_s": 0.5, "slow_approaching": False}}


def _tick_speeds(fk_rows, start, rows):
    points = np.asarray([pose[:3] for pose in fk_rows([list(start), *rows])])
    travel = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return travel / TICK, np.concatenate([[0.0], np.cumsum(travel)])


def _check_ramp_and_zone(fk_rows, path, request):
    """Tick by tick: the controlled point's speed change stays within the ramp (from and to rest included),
    and every tick that starts within the last 2 cm keeps the careful speed."""

    caps, slow = speed_caps(request)
    result = mp.pace_path(path, fk_rows, speed=request, tick_s=TICK, joint_speed_ceiling_rad_s=1.0,
                          wrist_joint_index=6, wrist_limits_rad=LIMITS, budget_ticks=10 ** 6)
    speeds, reached = _tick_speeds(fk_rows, path[0], result.rows())
    tangential = np.abs(np.diff(np.concatenate([[0.0], speeds, [0.0]]))) / TICK
    assert tangential.max() <= caps["linear_accel_m_s2"] * (1 + 1e-6), (request, tangential.max())
    assert speeds.max() <= caps["linear_m_s"] * (1 + 1e-6)
    if slow is not None:
        inside = reached[:-1] >= reached[-1] - SLOW_APPROACH_M - 1e-9
        assert inside.any() and speeds[inside].max() <= slow["linear_m_s"] * (1 + 1e-6), speeds[inside].max()
    return result


def test_repeated_rows_are_dropped_and_the_endpoints_kept():
    a, b = np.zeros(7), np.full(7, 0.5)
    rows = mp.distinct_rows(np.asarray([a, a, a + 1e-7, b - 1e-7, b, b]))
    assert rows.tolist() == [a.tolist(), b.tolist()]
    assert mp.distinct_rows(np.asarray([a, a])).tolist() == [a.tolist()]


@pytest.mark.parametrize("label", sorted(RAMPED))
def test_the_ramp_holds_with_repeated_and_uneven_planner_rows(label):
    """A planner's timed output repeats its start and goal and steps short where its own profile is slow."""

    steps = [0.0, 0.0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.05, 0.05, 0.05, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4,
             0.0, 0.0]
    path = [[float(sum(steps[:index + 1])), 0, 0, 0, 0, 0, 0] for index in range(len(steps))]
    result = _check_ramp_and_zone(fk, path, RAMPED[label])
    assert result.rows()[-1] == pytest.approx(path[-1])


def _two_rate_fk(rows):
    """The controlled point moves 0.3 m per rad of joint 0 and 0.1 m per rad of joint 1, along x."""

    return [[0.4 + 0.3 * q[0] + 0.1 * q[1], 0.0, 0.5, 0.0, 0.0, 0.0, 1.0] for q in rows]


@pytest.mark.parametrize("label", sorted(RAMPED))
def test_the_ramp_holds_where_the_point_moves_unevenly_per_joint_step(label):
    """Half the path turns joint 0, the other half joint 1: the same joint step moves the point a third as far.

    On even joint spacing the node speed (segments per second) would carry a threefold step in the point's
    speed across the corner; the grid is spread by the binding acceleration length instead."""

    path = [Q0, [0.4, 0, 0, 0, 0, 0, 0], [0.4, 1.2, 0, 0, 0, 0, 0]]
    result = _check_ramp_and_zone(_two_rate_fk, path, RAMPED[label])
    assert len(set(np.round(result.grid.linear_m, 9))) <= 3


def _fixture_fk(path):
    """The recorded grid's own lengths: along it the point moves its recorded linear length (on x) and turns its
    recorded angle (about z), so a row anywhere on the grid gets the pose it has on the recorded path."""

    joints = np.asarray(path["joints"])
    along = np.concatenate([[0.0], np.cumsum(path["linear_m"])])
    turned = np.concatenate([[0.0], np.cumsum(path["angular_rad"])])

    def pose(row):
        row = np.asarray(row, dtype=float)
        best = None
        for index in range(len(joints) - 1):
            segment = joints[index + 1] - joints[index]
            weight = float(np.clip((row - joints[index]) @ segment / float(segment @ segment), 0.0, 1.0))
            error = float(np.max(np.abs(joints[index] + weight * segment - row)))
            if best is None or error < best[0]:
                best = (error, index, weight)
        _, index, weight = best
        x = along[index] + weight * (along[index + 1] - along[index])
        angle = turned[index] + weight * (turned[index + 1] - turned[index])
        return [0.4 + x, 0.0, 0.5, 0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]

    return lambda rows: [pose(row) for row in rows]


@pytest.mark.parametrize("label", sorted(RAMPED))
@pytest.mark.parametrize("path", FIXTURE, ids=[path["label"] for path in FIXTURE])
def test_the_ramp_and_the_slow_zone_hold_on_recorded_paths(path, label):
    _check_ramp_and_zone(_fixture_fk(path), path["joints"], RAMPED[label])


def test_the_slow_zone_starts_at_the_node_at_or_before_the_boundary():
    pace = mp.paces({"speed": "careful", "slow_approaching": False})[0]
    linear = (0.012, 0.012, 0.012, 0.012)  # 2 cm from the end falls inside the third segment
    grid = mp.Grid((0.0, 0.25, 0.5, 0.75, 1.0), {"arm": np.zeros((5, 7))}, linear, (0.0,) * 4)
    assert mp.zone_starts(mp.slow_final(grid, {"arm": (linear, (0.0,) * 4)}, pace)) == {"arm": 2}
    exact = (0.01, 0.01, 0.01, 0.01)  # the boundary on a node: the zone starts there
    assert mp.zone_starts(mp.slow_final(grid, {"arm": (exact, (0.0,) * 4)}, pace)) == {"arm": 2}
    short = (0.005,) * 4  # a path shorter than 2 cm is slow throughout
    assert mp.zone_starts(mp.slow_final(grid, {"arm": (short, (0.0,) * 4)}, pace)) == {"arm": 0}
