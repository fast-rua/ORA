"""lib.motion_plan: paths, timing under speed caps, IK holes and budget fit, against a stub kinematics query.

The stub answers like the RoboDojo fake plant's toy kinematics: joints 0-2
are the link6 position, so a straight Cartesian line is a straight joint line.
"""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.agent.paced_kinematics import paced_response
from ora.agent.toolkit.interface.types import Response
from ora.agent.toolkit.lib.motion_plan import checks, execute, kinematics, paths, planner, timing

START = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
NORMAL = timing.Pace(0.3, 60.0, 1.0, 200.0, 2.5, 15.0)
CAREFUL = timing.Pace(0.1, 20.0, 0.5, 100.0, 1.2, 8.0)
FULL = timing.Pace(None, None, None, None, 4.0, 40.0)


class StubKinematics:
    """api.planner.kinematics answering with toy kinematics; ``holes`` lists x ranges without IK."""

    def __init__(self, holes=()):
        self.holes, self.requests = list(holes), []

    def kinematics(self, *, request):
        self.requests.append(request)
        if request["operation"] == "pace_path":
            return paced_response(request)
        if request["operation"] == "fk":
            rows = [{"index": i, "pose_xyz_xyzw": [*q[:3], 0.0, 0.0, 0.0, 1.0]} for i, q in enumerate(request["joints_rad"])]
        else:
            rows = []
            for index, pose in enumerate(request["poses_xyz_xyzw"]):
                failed = any(low <= pose[0] <= high for low, high in self.holes)
                rows.append({"index": index, "status": "ik_failed" if failed else "ok",
                             "joints_rad": None if failed else [*pose[:3], 0.0, 0.0, 0.0]})
        data = {"schema": "ora.kinematics-result.v1", "operation": request["operation"], "rows": rows,
                "seeding": "explicit_solver_seed_sequence_v1"}
        return Response(json.dumps({"status": "ok", "method": "kinematics", "data": data}).encode())


def _api(**kwargs):
    return SimpleNamespace(planner=StubKinematics(**kwargs))


def _line_plan(api, goal):
    return planner.plan_line(api, starts={"left": START}, goals={"left": goal},
                             start_joints={"left": [*START[:3], 0.0, 0.0, 0.0]})


def _speeds(knots, start):
    xs = [start] + [row[0] for row in knots["joints"]["left"]]
    return np.diff(xs) / timing.NATIVE_ACTION_S


def test_a_line_is_gridded_by_the_path_not_the_speed():
    line = paths.line(START, [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    grid = paths.line_grid(line)
    assert len(grid) == 101 and grid[0] == 0.0 and grid[-1] == 1.0
    assert line.pose(0.5)[:3] == pytest.approx([0.45, 0.0, 0.2])


@pytest.mark.parametrize("pace,cap", [(NORMAL, 0.3), (CAREFUL, 0.1)])
def test_a_long_move_ramps_smoothly_and_never_exceeds_its_caps(pace, cap):
    plan = _line_plan(_api(), [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    timed = timing.timed(_api(), plan.grid, pace)
    speeds = _speeds({"joints": timed.joints}, 0.3)
    assert speeds.max() <= cap + 1e-9 and speeds.max() > 0.9 * cap
    assert speeds[0] < 0.3 * cap and speeds[-1] < 0.3 * cap  # ramps at both ends
    assert timed.joints["left"][-1][0] == pytest.approx(0.6) and timed.fraction == 1.0
    assert timed.peaks["planned_peak_linear_m_s"] <= cap + 1e-9


def test_full_speed_respects_the_joint_cap_and_the_admitted_step():
    start, goal = np.zeros(6), np.asarray([2.0, -1.0, 0.5, 0.0, 0.0, 0.0])
    plan = planner.plan_joint(_api(), goals={"left": START}, start_joints={"left": start}, goal_joints={"left": goal})
    rows = np.vstack([start, timing.timed(_api(), plan.grid, FULL).joints["left"]])
    assert np.max(np.abs(np.diff(rows, axis=0))) <= 4.0 * timing.NATIVE_ACTION_S + 1e-9
    assert rows[-1] == pytest.approx(goal)


def test_an_ik_hole_inside_the_line_is_bridged_in_joint_space():
    plan = _line_plan(_api(holes=[(0.40, 0.42)]), [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert plan.bridged and plan.bridged[0]["deviation_m"] < 1e-9 and plan.bridged[0]["turn_deg"] < 1e-6
    assert not np.isnan(plan.grid.joints["left"]).any()


class WrappingKinematics(StubKinematics):
    """Toy kinematics whose joint 5 is the wrist roll about the approach axis: FK turns the orientation by it,
    and beyond x = 0.45 the IK answer comes a full turn away (a wrap at the joint's half-turn limit)."""

    def kinematics(self, *, request):
        if request["operation"] == "pace_path":
            return paced_response(request)
        if request["operation"] == "fk":
            rows = [{"index": i, "pose_xyz_xyzw": [*q[:3], math.sin(q[5] / 2), 0.0, 0.0, math.cos(q[5] / 2)]}
                    for i, q in enumerate(request["joints_rad"])]
            data = {"schema": "ora.kinematics-result.v1", "operation": "fk", "rows": rows}
            return Response(json.dumps({"status": "ok", "method": "kinematics", "data": data}).encode())
        rows, wrapped = [], False
        for index, pose in enumerate(request["poses_xyz_xyzw"]):
            jump = pose[0] > 0.45 and not wrapped
            wrapped = wrapped or pose[0] > 0.45
            rows.append({"index": index, "status": "discontinuous" if jump else "ok",
                         "joints_rad": None if jump else [*pose[:3], 0.0, 0.0, 2 * math.pi if wrapped else 0.0]})
        data = {"schema": "ora.kinematics-result.v1", "operation": "ik_chain", "rows": rows}
        return Response(json.dumps({"status": "ok", "method": "kinematics", "data": data}).encode())


def test_a_bridge_that_spins_the_wrist_is_refused_although_the_tool_point_stays_on_the_line():
    api = SimpleNamespace(planner=WrappingKinematics())
    with pytest.raises(checks.PlanRefusal) as refused:
        _line_plan(api, [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert refused.value.code == "paced_path_wrist_unwind"
    assert refused.value.constraints["joint_line_bridge_turn_deg"] > 90


def test_a_target_without_ik_is_refused_before_anything_moves():
    with pytest.raises(checks.PlanRefusal) as error:
        _line_plan(_api(holes=[(0.55, 1.0)]), [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert error.value.code == "paced_path_unreachable" and error.value.constraints["fraction_of_move"] > 0.8


def test_over_the_budget_the_first_part_of_the_path_is_kept():
    plan = _line_plan(_api(), [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    timed = timing.timed(_api(), plan.grid, CAREFUL, budget=20)
    assert timed.native_actions == 20 and timed.needed_native_actions > 20 and 0 < timed.fraction < 1
    assert timed.joints["left"][-1][0] == pytest.approx(0.3 + 0.3 * timed.fraction, abs=0.01)


def test_a_budget_that_covers_nothing_is_refused():
    plan = _line_plan(_api(), [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    with pytest.raises(checks.PlanRefusal) as refused:
        timing.timed(_api(), plan.grid, CAREFUL, budget=1)
    assert refused.value.code == "paced_motion_budget"


def test_two_arms_share_one_timeline_the_shorter_move_slowed_uniformly():
    left, right = np.linspace(0.0, 0.3, 31), np.linspace(0.0, 0.1, 31)
    grid = timing.Grid(tuple(index / 30 for index in range(31)),
                       {"left": np.column_stack([left, *[np.zeros(31)] * 5]),
                        "right": np.column_stack([right, *[np.zeros(31)] * 5])})
    timed = timing.timed(_api(), grid, CAREFUL)
    assert len(timed.joints["left"]) == len(timed.joints["right"]) == timed.native_actions
    assert timed.joints["right"][-1][0] == pytest.approx(0.1)
    assert np.max(np.diff([0.0, *[row[0] for row in timed.joints["right"]]])) / timing.NATIVE_ACTION_S <= 0.1 + 1e-9


def test_an_arm_that_holds_still_stays_where_it_is():
    moving = np.column_stack([np.linspace(0.0, 0.1, 11), *[np.zeros(11)] * 5])
    grid = timing.Grid(tuple(index / 10 for index in range(11)), {"left": moving, "right": np.zeros((11, 6))})
    timed = timing.timed(_api(), grid, CAREFUL)
    assert timed.joints["right"] == [[0.0] * 6] * timed.native_actions


def test_joint6_extremes_report_a_swing_both_ends_hide():
    rows = [[0.0] * 5 + [math.radians(v)] for v in (60.0, 120.0, 170.0, 100.0, 72.5)]
    report = checks.joint6_extremes([0.0] * 6, rows)
    assert report == {"start_deg": 0.0, "end_deg": 72.5, "min_deg": 0.0, "max_deg": 170.0, "travel_deg": 267.5,
                      "min_limit_margin_deg": round(math.degrees(3.14) - 170.0, 2)}


def test_a_staged_concatenation_stops_where_its_parts_meet():
    api = _api()
    first = planner.plan_joint(api, goals={"left": START}, start_joints={"left": np.zeros(6)},
                               goal_joints={"left": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]})
    second = _line_plan(api, [0.35, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    grid = timing.concatenate(first.grid, second.grid)
    assert grid.stops == (first.grid.segments,)
    api.planner.requests.clear()
    timed = timing.timed(api, grid, NORMAL, slow_pace=CAREFUL)
    asked = [request for request in api.planner.requests if request["operation"] == "pace_path"]
    assert [request["slow_approach"] for request in asked] == [False, True]
    assert timed.joints["left"][-1][0] == pytest.approx(0.35)


def test_the_action_is_one_trajectory_with_its_settle_in_the_budget():
    knots = {"left": [[0.0] * 6, [0.1] * 6]}
    action = execute.trajectory_action(action_id="action-3", observation={"observation_id": "o",
                                       "observation_sha256": "s"}, knots=knots,
                                       settle={"native_actions": 3, "joint_tolerance_rad": 0.002,
                                               "joint_speed_rad_s": 0.05})
    calls = []
    api = SimpleNamespace(act=SimpleNamespace(execute_action=lambda action, budget: calls.append(budget) or Response(
        json.dumps({"status": "ok", "method": "execute_action", "data": {"receipt": {}, "remaining_budget": 0}}).encode())))
    execute.follow(api, action, settle_native_actions=3)
    assert action["action"] == "follow_joint_trajectory" and calls == [5]


def test_a_chain_longer_than_one_request_is_seeded_from_the_last_solution():
    api = _api()
    poses = [[0.3 + 0.001 * k, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0] for k in range(300)]
    result = kinematics.ik_chain(api, arm="left", poses=poses, seed=None, max_joint_step_rad=0.25)
    assert len(result["rows"]) == 300 and len(api.planner.requests) == 2
    assert api.planner.requests[1]["seed_joints_rad"][0] == pytest.approx(poses[255][0])
    assert math.isclose(result["rows"][-1]["joints_rad"][0], poses[-1][0])


def test_a_blocked_screen_is_refused_where_it_happens():
    checks.screened({"status": "clear"}, [0.5, 1.0])
    with pytest.raises(checks.PlanRefusal) as error:
        checks.screened({"status": "table_collision", "arm": "right", "first_blocked_segment": 2,
                         "min_table_clearance_m": -0.004, "limiting_link": "link5"}, [0.25, 0.5, 1.0])
    assert error.value.code == "table_collision"
    assert error.value.constraints == {"fraction_of_move": 0.5, "native_action": 2, "arm": "right",
                                       "min_table_clearance_m": -0.004, "limiting_link": "link5"}
    with pytest.raises(checks.PlanRefusal) as error:
        checks.screened({"status": "odd"}, [1.0])
    assert error.value.code == "path_blocked"


def test_a_cut_move_names_its_goal_in_the_action():
    action = execute.trajectory_action(action_id="a", observation={"observation_id": "o", "observation_sha256": "s"},
                                       knots={"left": [[0.0] * 6]},
                                       part_of_move={"fraction": 0.4, "goals_xyz_xyzw": {"left": START}})
    assert action["arguments"]["part_of_move"] == {"fraction": 0.4, "goals_xyz_xyzw": {"left": START}}
    from ora.contracts.joint_trajectory import validate_trajectory_arguments
    assert validate_trajectory_arguments(action["arguments"])["part_of_move"]["fraction"] == 0.4


def test_warnings_of_kinematics_answers_reach_the_plan():
    class Warning(StubKinematics):
        def kinematics(self, *, request):
            if request["operation"] == "pace_path":
                return paced_response(request)
            response = super().kinematics(request=request)
            data = response.json()["data"]
            data.update(ik_sampler_state="unrecorded", warning="restored without generator state")
            return Response(json.dumps({"status": "ok", "method": "kinematics", "data": data}).encode())

    plan = _line_plan(SimpleNamespace(planner=Warning()), [0.35, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert plan.warnings and plan.warnings[0]["ik_sampler_state"] == "unrecorded"
    assert not _line_plan(_api(), [0.35, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]).warnings


def test_a_tool_of_its_own_plans_with_any_pace_and_runs_one_trajectory():
    from ora.agent.toolkit.lib.motion_plan import move

    api = _api()
    executed = []

    def execute_action(action, budget):
        executed.append((action, budget))
        count = len(action["arguments"]["arms"]["left"]["knots_rad"])
        receipt = {"status": "reached", "stage_evidence": {"trajectory": {
            "status": "completed", "knots_planned": count, "knots_executed": count, "stop": None}}}
        return Response(json.dumps({"status": "ok", "method": "execute_action",
                                    "data": {"receipt": receipt, "remaining_budget": 0}}).encode())

    def kinematics(*, request):
        if request["operation"] == "pace_path":
            return paced_response(request)
        if request["operation"] == "screen_path":
            return Response(json.dumps({"status": "ok", "method": "kinematics",
                                        "data": {"status": "clear"}}).encode())
        return StubKinematics.kinematics(api.planner, request=request)

    api = SimpleNamespace(planner=SimpleNamespace(kinematics=kinematics, requests=[]),
                          act=SimpleNamespace(execute_action=execute_action))
    api.planner.holes = []
    mine = timing.Pace(linear_m_s=0.2, angular_deg_s=40.0, linear_accel_m_s2=0.8, angular_accel_deg_s2=150.0,
                       joint_speed_rad_s=2.0, joint_accel_rad_s2=12.0)
    slow = timing.Pace(0.05, 10.0, 0.3, 60.0, 0.8, 5.0)
    planned = move.plan(api, starts={"left": START}, goals={"left": [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]},
                        start_joints={"left": [*START[:3], 0, 0, 0]}, pace=mine,
                        settle={"native_actions": 2, "joint_tolerance_rad": 0.002, "joint_speed_rad_s": 0.05},
                        native_action_budget=80, slow_pace=slow)
    result = move.run(api, planned, action_id="action-1", observation={"observation_id": "o", "observation_sha256": "s"},
                      labels={"move": "mine", "slow": "gentle"})
    action, budget = executed[0]
    speeds = _speeds(planned.knots, 0.3)
    assert speeds.max() <= 0.2 + 1e-9 and budget == planned.knots["native_actions"] + 2
    gentle = action["arguments"]["speed_segments"]["left"][1]
    assert gentle["speed"] == "gentle" and gentle["path_left_m"] == timing.SLOW_APPROACH_M
    assert speeds[gentle["first_knot"]:].max() <= 0.05 + 1e-9
    assert result["motion"]["status"] == "completed" and result["receipt"]["status"] == "reached"


def _labelled_move(interactions, *, expect_contact=False):
    """A short move through ``move.plan`` and ``move.run``: the action it sent, its ``motion`` and its call's note."""
    from ora.agent.toolkit.interface import call_note
    from ora.agent.toolkit.lib.motion_plan import move

    executed = []

    def execute_action(action, budget):
        executed.append((action, call_note.current()))
        receipt = {"stage_evidence": {"trajectory": {
            "status": "stopped", "stop": {"reason": "tracking", "rule": "bound", "arm": "left"}}}}
        return Response(json.dumps({"status": "ok", "method": "execute_action",
                                    "data": {"receipt": receipt, "remaining_budget": 40}}).encode())

    def kinematics(*, request):
        if request["operation"] in ("screen_path", "contact_path"):
            return Response(json.dumps({"status": "ok", "method": "kinematics",
                                        "data": {"status": "clear"}}).encode())
        return StubKinematics.kinematics(api.planner, request=request)

    api = SimpleNamespace(planner=SimpleNamespace(kinematics=kinematics, requests=[]),
                          act=SimpleNamespace(execute_action=execute_action))
    api.planner.holes = []
    planned = move.plan(api, starts={"left": START}, goals={"left": [0.32, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]},
                        start_joints={"left": [*START[:3], 0, 0, 0]}, pace=NORMAL,
                        settle={"native_actions": 2, "joint_tolerance_rad": 0.002, "joint_speed_rad_s": 0.05},
                        native_action_budget=80, route_kind="interaction", interactions=interactions)
    result = move.run(api, planned, action_id="action-1", expect_contact=expect_contact,
                      observation={"observation_id": "o", "observation_sha256": "s"})
    return executed[0][0], result["motion"], executed[0][1], planned


def test_a_contact_approach_leaves_the_flag_to_the_caller():
    action, motion, _, _ = _labelled_move({"left": "contact_approach"})
    assert "expect_contact" not in action["arguments"] and motion["expect_contact"] is False
    action, motion, _, _ = _labelled_move({"left": "contact_approach"}, expect_contact=True)
    assert action["arguments"]["expect_contact"] is True and motion["expect_contact"] is True


def test_a_move_s_route_goes_on_its_toolkit_call_and_stays_out_of_its_action():
    action, motion, note, _ = _labelled_move({"left": "contact_approach"})
    assert note == {"route": motion["route"]} and "route" not in action["arguments"]
    from ora.contracts.action import validate_runtime_action

    validate_runtime_action(action)     # the kernel contract's fields are untouched


def test_a_jaw_command_with_its_own_rate_and_lead():
    action = execute.gripper_action(action_id="a", observation={"observation_id": "o", "observation_sha256": "s"},
                                    targets={"left": {"opening": "close", "rate_per_s": 1.0, "max_lead_m": 0.004}})
    from ora.contracts.action import validate_runtime_action
    from ora.contracts.joint_trajectory import native_action_cap

    validated = validate_runtime_action(action)
    assert validated["arguments"]["targets"]["left"]["max_lead_m"] == 0.004 and native_action_cap(validated) == 250


def test_the_nearest_jaw_twin_rule():
    from ora.agent.toolkit.lib.motion_plan import orientation

    current = [0.0, 0.0, 0.0, 1.0]
    far = [math.sin(math.radians(85)), 0.0, 0.0, math.cos(math.radians(85))]  # 170 degrees about +X
    near = [math.sin(math.radians(20)), 0.0, 0.0, math.cos(math.radians(20))]  # 40 degrees
    choice = orientation.nearest_jaw_twin(current, far)
    assert choice["jaw_symmetry"] == "twin_nearer_current"
    assert choice["turn_deg"] == {"requested": pytest.approx(170.0), "twin": pytest.approx(10.0)}
    assert orientation.turn_deg(current, choice["orientation_xyzw"]) == pytest.approx(10.0)
    assert orientation.nearest_jaw_twin(current, near)["jaw_symmetry"] == "requested"
    assert orientation.nearest_jaw_twin(current, far, held=True)["jaw_symmetry"] == "held_no_flip"
    exact = orientation.nearest_jaw_twin(current, far, held=True, exact=True)
    assert exact["jaw_symmetry"] == "exact" and orientation.turn_deg(current, exact["orientation_xyzw"]) == pytest.approx(170)
    # The twin of the twin is the request; the twin is the same jaw line turned about the approach axis.
    assert orientation.turn_deg(orientation.jaw_twin(orientation.jaw_twin(far)), far) == pytest.approx(0.0, abs=1e-6)
    assert orientation.turn_deg(orientation.jaw_twin(far), far) == pytest.approx(180.0)
