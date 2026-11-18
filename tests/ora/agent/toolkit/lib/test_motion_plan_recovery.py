"""lib.motion_plan.recovery: the measured retrace, the screened joint unwind, the home lift, the budget.

Toy kinematics as in test_motion_plan: joints 0-2 are the link6 position, FK reads them back, IK writes
the position into them; ``screens`` queues the screen answers (clear when empty).
"""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.agent.paced_kinematics import paced_response
from ora.agent.toolkit.interface.types import Response
from ora.agent.toolkit.lib.motion_plan import checks, recovery, timing

NORMAL = timing.Pace(0.3, 60.0, 1.0, 200.0, 2.5, 15.0)
FULL = timing.Pace(None, None, None, None, 4.0, 40.0)
PACES = {recovery.RETRACE: NORMAL, recovery.JOINT: FULL}
SETTLE = {"native_actions": 2, "joint_tolerance_rad": 0.003, "joint_speed_rad_s": 0.05}
BOUNDS = {"x": [-1, 1], "y": [-1, 1], "z": [0, 0.6]}
ANCHOR = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]


class Stub:
    def __init__(self, screens=()):
        self.screens, self.requests = list(screens), []

    def kinematics(self, *, request):
        if request["operation"] == "pace_path":
            return paced_response(request)
        self.requests.append(request)
        operation = request["operation"]
        if operation == "screen_path":
            data = self.screens.pop(0) if self.screens else {"status": "clear"}
        elif operation == "fk":
            data = {"rows": [{"index": i, "pose_xyz_xyzw": [*q[:3], 0, 0, 0, 1]}
                             for i, q in enumerate(request["joints_rad"])]}
        else:
            data = {"rows": [{"index": i, "status": "ok", "joints_rad": [*pose[:3], 0.0, 0.0, 0.0]}
                             for i, pose in enumerate(request["poses_xyz_xyzw"])]}
        return Response(json.dumps({"status": "ok", "method": "kinematics", "data": data}).encode())


def _api(screens=()):
    return SimpleNamespace(planner=Stub(screens))


def _recorded(count=12, step=0.02, stall_after=None):
    """Measured joints at the end of each native action of a move along joint 0 (optionally stalled)."""
    rows, current = [list(ANCHOR)], list(ANCHOR)
    for tick in range(count):
        if stall_after is None or tick < stall_after:
            current = [current[0] + step, *current[1:]]
        else:
            current = [current[0] + 1e-4 * (-1) ** tick, *current[1:]]  # stalled: jitter in place
        rows.append(list(current))
    return rows


def _plan(api, options, start, goal=None, budget=20):
    return recovery.plan(api, options=options, start_joints=start,
                         goals={arm: goal or [*ANCHOR[:3], 0, 0, 0, 1] for arm in options}, paces=PACES,
                         settle=SETTLE, native_action_budget=budget)


def test_the_retrace_runs_back_along_the_measured_path_to_the_anchor():
    path = _recorded(count=6)
    nodes, info = recovery.retrace_nodes(ANCHOR, path, path[-1])
    assert info["status"] == "retraced" and info["recorded_native_actions"] == 6
    assert nodes[0] == pytest.approx(path[-1]) and nodes[-1] == pytest.approx(ANCHOR)
    assert np.all(np.diff(nodes[:, 0]) < 0)  # backwards, node by recorded node
    planned = _plan(_api(), {"left": [recovery.Option(recovery.RETRACE, (nodes,))]}, {"left": path[-1]})
    knots = planned.knots["joints"]["left"]
    assert planned.fraction == 1.0 and knots[-1] == pytest.approx(ANCHOR)
    assert planned.pace is NORMAL and planned.targets["left"] == pytest.approx(ANCHOR)
    speeds = np.diff([path[-1][0]] + [row[0] for row in knots]) / timing.NATIVE_ACTION_S
    assert np.max(np.abs(speeds)) <= 0.3 + 1e-9  # link6 at most 0.3 m/s (normal) on the toy kinematics


def test_a_stalled_move_is_retraced_along_where_the_arm_was_not_where_it_was_commanded():
    # Commanded 12 x 0.02 rad, but the arm stalled after 4 native actions and jittered in place.
    path = _recorded(stall_after=4)
    nodes, info = recovery.retrace_nodes(ANCHOR, path, path[-1])
    assert info["status"] == "retraced"
    assert np.max(nodes[:, 0]) <= ANCHOR[0] + 4 * 0.02 + 1e-3  # never beyond the stall point
    assert len(nodes) <= 6  # the stalled jitter is thinned out
    planned = _plan(_api(), {"left": [recovery.Option(recovery.RETRACE, (nodes,))]}, {"left": path[-1]})
    assert planned.knots["joints"]["left"][-1] == pytest.approx(ANCHOR)


def test_a_retrace_continues_from_where_the_arm_stands_on_the_path():
    path = _recorded()
    here = [path[7][0] + 0.005, *path[7][1:]]
    nodes, info = recovery.retrace_nodes(ANCHOR, path, here)
    assert info["status"] == "retraced" and 7 <= info["from_native_action"] <= 8
    assert nodes[0] == pytest.approx(here) and nodes[1] == pytest.approx(path[7])


def test_no_retrace_without_a_record_of_this_move_or_once_the_arm_left_it():
    path = _recorded()
    assert recovery.retrace_nodes(ANCHOR, None, path[-1])[1]["status"] == "not_recorded"
    other = [[ANCHOR[0] + 0.5, *ANCHOR[1:]], *path[1:]]
    assert recovery.retrace_nodes(ANCHOR, other, path[-1])[1]["status"] == "record_of_another_move"
    moved = [path[-1][0], 0.3, *path[-1][2:]]
    nodes, info = recovery.retrace_nodes(ANCHOR, path, moved)
    assert nodes is None and info["status"] == "arm_moved_since" and info["joint_distance_before_rad"] > 0.05


def test_the_unwind_target_takes_the_nearest_whole_turn_but_never_winds_a_pitch_joint():
    measured = [3.0, 0.0, 3.0, 0.0, 0.0, 0.0]
    stored = [-3.0, 0.0, -3.0, 0.0, 0.0, 0.0]
    target = recovery.unwind_target(stored, measured)
    assert target["equivalent_turns"] == [1, 0, 0, 0, 0, 0]  # joint1 unwinds; joint3 would pass pi
    assert target["joint_target"][0] == pytest.approx(-3.0 + math.tau)
    assert target["pitch_vetoed_turns"] == [0, 0, 1, 0, 0, 0]
    assert recovery.unwind_target(ANCHOR, ANCHOR)["status"] == "not_needed"


def test_joint_orders_are_screened_and_the_first_clear_one_runs():
    start, target = [1.0, 0.5, 0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]
    options = recovery.joint_options(start, target)
    assert [option.detail["unwind_order"] for option in options] == list(recovery.ORDERS)
    api = _api([{"status": "self_collision", "arm": "left", "first_blocked_segment": 3}])
    planned = _plan(api, {"left": options}, {"left": start})
    assert planned.options["left"].detail["unwind_order"] == "fold_then_yaw" and planned.pace is FULL
    assert planned.grid.stops  # one trajectory, stopping at the corner between fold and yaw
    corner = planned.grid.joints["left"][planned.grid.stops[0]]
    assert corner == pytest.approx([1.0, 0.0, 0.2, 0, 0, 0])  # the fold ends before the yaw starts
    assert planned.knots["joints"]["left"][-1] == pytest.approx(target)
    assert [screen["status"] for screen in planned.screens] == ["self_collision", "clear"]


def test_every_order_blocked_refuses_before_anything_moves_with_the_released_codes():
    start, target = [1.0, 0.5, 0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]
    options = {"left": recovery.joint_options(start, target)}
    with pytest.raises(checks.PlanRefusal) as error:
        _plan(_api([{"status": "self_collision"}, {"status": "inter_arm_collision"}, {"status": "self_collision"}]),
              options, {"left": start})
    assert error.value.code == "recovery_inter_arm_collision_path"
    assert error.value.constraints["native_actions"] == 0 and len(error.value.constraints["screens"]) == 3
    with pytest.raises(checks.PlanRefusal) as error:
        _plan(_api([{"status": "self_collision"}] * 3), options, {"left": start})
    assert error.value.code == "recovery_self_collision_path"
    with pytest.raises(checks.PlanRefusal) as error:
        _plan(_api([{"status": "table_collision"}, {"status": "self_collision"}] + [{"status": "self_collision"}]),
              options, {"left": start})
    assert error.value.code == "recovery_table_collision_path"


def test_a_blocked_retrace_falls_back_to_the_joint_unwind():
    path = _recorded()
    nodes, info = recovery.retrace_nodes(ANCHOR, path, path[-1])
    options = [recovery.Option(recovery.RETRACE, (nodes,), {"retrace": info}),
               *recovery.joint_options(path[-1], ANCHOR, detail={"retrace": info})]
    planned = _plan(_api([{"status": "self_collision"}]), {"left": options}, {"left": path[-1]})
    assert planned.options["left"].method == recovery.JOINT and planned.pace is FULL


def test_home_lifts_straight_up_then_unwinds_in_one_trajectory():
    api = _api()
    start = [0.3, 0.0, 0.2, 0.8, 0.0, 0.0]
    lift, report, _ = recovery.plan_lift(api, arm="left", pose=[0.3, 0.0, 0.2, 0, 0, 0, 1], joints=start,
                                         bounds=BOUNDS)
    assert report == {"status": "planned", "lift_m": pytest.approx(0.1)}
    assert lift[-1][:3] == pytest.approx([0.3, 0.0, 0.3]) and np.allclose(lift[:, :2], [0.3, 0.0])
    home = [0.0, 0.0, 0.25, 0.0, 0.0, 0.0]
    options = recovery.joint_options(start, home, before=(lift,), detail={"retract": report})
    planned = _plan(api, {"left": options}, {"left": start}, goal=[0.0, 0.0, 0.25, 0, 0, 0, 1])
    knots = np.asarray(planned.knots["joints"]["left"])
    assert len(planned.grid.stops) == 1  # lift, a stop, then the straight joint unwind
    grid = planned.grid.joints["left"]
    stop = planned.grid.stops[0]
    assert grid[stop][:3] == pytest.approx([0.3, 0.0, 0.3])  # straight up, then the unwind
    assert np.allclose(grid[:stop + 1, :2], [0.3, 0.0])
    assert knots[-1] == pytest.approx(home) and planned.fraction == 1.0
    assert planned.knots["native_actions"] + SETTLE["native_actions"] <= 20


def test_no_lift_at_the_workspace_ceiling():
    goal, report = recovery.lift_goal([0.3, 0.0, 0.595, 0, 0, 0, 1], BOUNDS)
    assert goal is None and report["reason"] == "workspace_ceiling"
    goal, report = recovery.lift_goal([0.3, 0.0, 0.55, 0, 0, 0, 1], BOUNDS)
    assert goal[2] == pytest.approx(0.6) and report["lift_m"] == pytest.approx(0.05)


def test_two_arms_recover_in_lockstep_on_one_grid():
    left_path = _recorded(count=5)
    nodes, _ = recovery.retrace_nodes(ANCHOR, left_path, left_path[-1])
    right_start, right_home = [0.3, -0.4, 0.2, 0.5, 0.0, 0.0], [0.3, -0.4, 0.2, 0.0, 0.0, 0.0]
    options = {"left": [recovery.Option(recovery.RETRACE, (nodes,))],
               "right": recovery.joint_options(right_start, right_home)}
    api = _api()
    planned = _plan(api, options, {"left": left_path[-1], "right": right_start})
    joints = planned.knots["joints"]
    assert len(joints["left"]) == len(joints["right"]) and planned.pace is NORMAL
    assert joints["left"][-1] == pytest.approx(ANCHOR) and joints["right"][-1] == pytest.approx(right_home)
    screen = next(r for r in api.planner.requests if r["operation"] == "screen_path")
    assert set(screen["knots_rad"]) == {"left", "right"}


def test_the_budget_of_20_and_what_it_covers():
    """Measurement for the budget decision: settle 2 leaves 18 native actions for the path."""
    # Joint interpolation at the fast pace (4 rad/s, 40 rad/s^2): about 2.5 rad of the largest joint fits.
    for travel, fits in ((2.4, True), (3.0, False)):
        start = [travel, 0.0, 0.2, 0.0, 0.0, 0.0]
        planned = _plan(_api(), {"left": recovery.joint_options(start, ANCHOR[:1] + start[1:])}, {"left": start})
        assert (planned.fraction == 1.0) is fits
        assert planned.knots["native_actions"] + planned.settle["native_actions"] <= 20
    # A retrace at the normal pace (0.3 m/s, 1 m/s^2 for link6): about 0.12 m of path fits; longer is cut.
    for length, fits in ((0.12, True), (0.14, False)):
        path = _recorded(count=int(round(length / 0.01)), step=0.01)
        nodes, _ = recovery.retrace_nodes(ANCHOR, path, path[-1])
        planned = _plan(_api(), {"left": [recovery.Option(recovery.RETRACE, (nodes,))]}, {"left": path[-1]})
        assert (planned.fraction == 1.0) is fits
        assert planned.knots["native_actions"] + planned.settle["native_actions"] <= 20


def test_a_cut_recovery_reports_its_progress_honestly():
    path = _recorded(count=30, step=0.01)
    nodes, _ = recovery.retrace_nodes(ANCHOR, path, path[-1])
    planned = _plan(_api(), {"left": [recovery.Option(recovery.RETRACE, (nodes,))]}, {"left": path[-1]})
    assert planned.fraction < 1.0
    end = planned.knots["joints"]["left"][-1]
    settled = {"status": "completed", "settle": {"settled": True}}
    progress = recovery.outcome(planned, "left", path[-1], end, settled)
    assert progress["outcome"] == "cut_short" and progress["cut_short"]["native_action_budget"] == 20
    assert progress["joint_distance_end_rad"] < progress["joint_distance_start_rad"]
    assert progress["joint6"] == checks.joint6_extremes(path[-1], planned.knots["joints"]["left"])
    stalled = recovery.outcome(planned, "left", path[-1], path[-1], {"status": "stopped", "stop": {"reason": "progress"}})
    assert stalled["outcome"] == "stalled"


def test_the_outcome_follows_the_measured_evidence_and_never_guesses():
    path = _recorded(count=6)
    nodes, _ = recovery.retrace_nodes(ANCHOR, path, path[-1])
    planned = _plan(_api(), {"left": [recovery.Option(recovery.RETRACE, (nodes,))]}, {"left": path[-1]})
    residual = [0.00015, -0.0001, 0.0, 0.0, 0.0, 0.0]
    evidence = {"status": "completed", "settle": {"settled": True, "joint_speed_rad_s": 0.034},
                "arms": {"left": {"final_joint_error_per_joint_rad": residual}}}
    end = recovery.measured_end(planned, "left", evidence)
    assert end == pytest.approx([a - b for a, b in zip(ANCHOR, residual)])
    arrived = recovery.outcome(planned, "left", path[-1], end, evidence)
    assert arrived["outcome"] == "arrived" and arrived["joint_distance_end_rad"] == pytest.approx(0.00015)
    unsettled = {**evidence, "settle": {"settled": False}}
    assert recovery.outcome(planned, "left", path[-1], end, unsettled)["outcome"] == "arrived_not_settled"
    far = [ANCHOR[0] + 0.1, *ANCHOR[1:]]
    assert recovery.outcome(planned, "left", path[-1], far, unsettled)["outcome"] == "still_moving_at_window_end"
    assert recovery.outcome(planned, "left", path[-1], end, {"status": "budget"})["outcome"] == \
        "still_moving_at_window_end"
    assert recovery.outcome(planned, "left", path[-1], end, {"status": "terminal"})["outcome"] == "episode_terminal"
    # Without the evidence the outcome is unknown and the end distance null, never "still moving".
    assert recovery.measured_end(planned, "left", {"status": "completed"}) is None
    missing = recovery.outcome(planned, "left", path[-1], None, {"status": "completed"})
    assert missing["outcome"] == "unknown" and missing["joint_distance_end_rad"] is None
    assert recovery.outcome(planned, "left", path[-1], None, {})["outcome"] == "unknown"


def test_the_run_restates_the_anchors_and_keeps_the_settle_in_the_budget():
    executed = []

    def execute_action(action, budget):
        executed.append((action, budget))
        trajectory = {"status": "completed", "knots_planned": 1, "knots_executed": 1, "stop": None}
        return Response(json.dumps({"status": "ok", "method": "execute_action", "data": {
            "remaining_budget": 20 - budget,
            "receipt": {"status": "reached", "stage_evidence": {"trajectory": trajectory}}}}).encode())

    api = SimpleNamespace(planner=Stub(), act=SimpleNamespace(execute_action=execute_action))
    start = [0.4, 0.0, 0.2, 0.0, 0.0, 0.0]
    planned = _plan(api, {"left": recovery.joint_options(start, ANCHOR)}, {"left": start})
    anchor = {"strategy": "home", "arm_joint_qpos": ANCHOR, "pose_xyz_xyzw": [0.3, 0, 0.2, 0, 0, 0, 1]}
    result = recovery.run(api, planned, action_id="action-3", observation={"observation_id": "o",
                                                                           "observation_sha256": "s"},
                          recovery={"targets": {"left": "home"}, "recovery_targets": {"left": anchor}})
    action, budget = executed[0]
    assert action["action"] == "follow_joint_trajectory" and action["arguments"]["recovery"]["targets"] == {
        "left": "home"}
    assert action["arguments"]["recovery"]["recovery_targets"]["left"] == anchor
    assert budget == len(action["arguments"]["arms"]["left"]["knots_rad"]) + SETTLE["native_actions"] <= 20
    assert result["summary"]["status"] == "completed"
