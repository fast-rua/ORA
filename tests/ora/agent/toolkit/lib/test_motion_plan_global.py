"""lib.motion_plan on the shared motion base: the planner's route, its fallbacks, and a deliberate contact."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.agent.paced_kinematics import paced_response
from ora.agent.toolkit.interface.types import Response
from ora.agent.toolkit.lib.motion_plan import checks, move, planner, recovery, route, timing

START = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
FAR = [0.6, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0]
Q0 = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
NORMAL = timing.Pace(0.3, 60.0, 1.0, 200.0, 2.5, 15.0)
SETTLE = {"native_actions": 3, "joint_tolerance_rad": 0.002, "joint_speed_rad_s": 0.05}


class Stub:
    """Toy kinematics (joints 0-2 are the position); the shared planner answers as told."""

    def __init__(self, plan=None, contact=None, holes=()):
        self.plan, self.contact, self.holes, self.requests = plan, contact, list(holes), []

    def kinematics(self, *, request):
        self.requests.append(request)
        operation = request["operation"]
        if operation == "pace_path":
            return paced_response(request)
        if operation == "plan_path":
            data = dict(self.plan or {"status": "planned", "reason": None, "start_joints": Q0,
                                      "path_joints": np.linspace(Q0, [0.6, 0.1, 0.2, 0.0, 0.0, 0.0], 12).tolist()})
        elif operation == "contact_path":
            data = dict(self.contact or {"status": "clear"})
        elif operation == "screen_path":
            data = {"status": "clear"}
        elif operation == "fk":
            data = {"rows": [{"index": i, "pose_xyz_xyzw": [*q[:3], 0, 0, 0, 1]}
                             for i, q in enumerate(request["joints_rad"])]}
        else:
            rows = []
            for index, pose in enumerate(request["poses_xyz_xyzw"]):
                failed = any(low <= pose[0] <= high for low, high in self.holes)
                rows.append({"index": index, "status": "ik_failed" if failed else "ok",
                             "joints_rad": None if failed else [*pose[:3], 0.0, 0.0, 0.0]})
            data = {"rows": rows}
        return Response(json.dumps({"status": "ok", "method": "kinematics",
                                    "data": {"operation": operation, **data}}).encode())


def _api(**kwargs):
    return SimpleNamespace(planner=Stub(**kwargs))


def _plan(api, goal=FAR, *, interaction="free_space", route_kind="interaction", **options):
    return move.plan(api, starts={"left": START}, goals={"left": goal}, start_joints={"left": Q0}, pace=NORMAL,
                     settle=SETTLE, native_action_budget=120, route_kind=route_kind,
                     interactions={"left": interaction}, **options)


def _asked(api, operation):
    return [request for request in api.planner.requests if request["operation"] == operation]


def test_the_route_rule_takes_the_planner_for_every_free_route():
    far, near = {"left": planner.paths.line(START, FAR)}, {"left": planner.paths.line(START, START)}
    assert route.choose("free_space", far, staged_available=True) == "joint"
    assert route.choose("free_space", far, staged_available=True, global_available=True) == "global"
    assert route.choose("precision", far, staged_available=True, global_available=True) == "global"
    assert route.choose("precision", near, staged_available=True, global_available=True) == "line"
    assert route.choose("contact_approach", far, staged_available=True, global_available=True) == "staged"


def test_a_free_move_runs_the_planners_path_from_the_measured_joints():
    api = _api()
    planned = _plan(api, global_planner=True)
    assert planned.path.route == "global" and planned.path.grid.joints["left"][0].tolist() == Q0
    assert planned.knots["joints"]["left"][-1][:3] == pytest.approx([0.6, 0.1, 0.2], abs=1e-6)
    assert _asked(api, "screen_path"), "the timed path is re-checked in lockstep"


def test_without_a_planner_the_free_route_stays_the_joint_line():
    api = _api()
    assert _plan(api).path.route == "joint" and not _asked(api, "plan_path")
    api = _api(plan={"status": "unavailable", "reason": "stale_observation"})
    assert _plan(api, global_planner=True).path.route == "joint" and _asked(api, "plan_path")


@pytest.mark.parametrize("reason", ["path_blocked", "goal_in_collision", "start_in_collision"])
def test_a_planner_refusal_is_the_moves_refusal_with_its_neutral_reason(reason):
    with pytest.raises(checks.PlanRefusal) as refused:
        _plan(_api(plan={"status": "refused", "reason": reason}), global_planner=True)
    assert refused.value.code == reason and "nothing was executed" in str(refused.value)


def test_an_arm_past_the_planners_limits_comes_back_by_the_joint_route():
    api = _api(plan={"status": "refused", "reason": "start_outside_limits"})
    assert _plan(api, global_planner=True).path.route == "joint" and _asked(api, "plan_path")


def test_the_worlds_goal_refusal_keeps_its_own_words():
    message = "the goal puts a fingertip 3.0 cm below the supporting surface"
    with pytest.raises(checks.PlanRefusal) as refused:
        _plan(_api(plan={"status": "refused", "reason": "goal_below_support", "goal_refusal": {"message": message}}),
              global_planner=True)
    assert refused.value.code == "goal_below_support" and message in str(refused.value)


def test_a_goal_below_the_support_names_its_heights_in_the_world_frame():
    """The worker's goal refusal carries its heights in the world frame of the policy's poses (``in_world``)."""
    refusal = {"reason": "fingertips_below_support", "fingertip_z_m": -0.011, "support_z_m": -0.0, "depth_m": 0.011,
               "tolerance_m": 0.01, "message": "the goal puts a fingertip 1.1 cm below the supporting surface at z "
                                              "0.766 m (world frame) seen in the depth under it",
               "in_world": {"frame": "world", "support_z_m": 0.766, "fingertip_z_m": 0.755}}
    with pytest.raises(checks.PlanRefusal) as refused:
        _plan(_api(plan={"status": "refused", "reason": "goal_below_support", "goal_refusal": refusal}),
              global_planner=True)
    assert refused.value.constraints == {"arm": "left", "frame": "world", "support_z_m": 0.766, "fingertip_z_m": 0.755,
                                         "depth_m": 0.011, "tolerance_m": 0.01}
    assert "(world frame)" in str(refused.value)


def test_a_line_that_cannot_be_followed_goes_around_unless_it_is_a_contact():
    api = _api(holes=[(0.55, 2.0)])
    assert _plan(api, route_kind="line", global_planner=True).path.route == "global"
    with pytest.raises(checks.PlanRefusal):
        _plan(_api(holes=[(0.55, 2.0)]), route_kind="line", global_planner=True, expect_contact=True)
    with pytest.raises(checks.PlanRefusal):
        _plan(_api(holes=[(0.55, 2.0)]), route_kind="line")


def test_a_contact_takes_the_contact_route_whatever_its_label():
    near = [0.33, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    api = _api()
    planned = _plan(api, near, interaction="free_space", global_planner=True, expect_contact=True)
    assert planned.path.route == "line" and not _asked(api, "plan_path") and _asked(api, "contact_path")
    api = _api()
    planned = _plan(api, interaction="free_space", global_planner=True, expect_contact=True,
                    capabilities=[route.STAGED_CAPABILITY])
    assert planned.path.route == "line" and not _asked(api, "plan_path") and _asked(api, "contact_path")
    api = _api()
    planned = _plan(api, interaction="contact_approach", global_planner=True, capabilities=[route.STAGED_CAPABILITY])
    assert planned.path.route == "staged" and _asked(api, "plan_path")
    # Only the straight part is checked: its first row is the hand-over.
    rows = _asked(api, "contact_path")[0]["joints_rad"]
    assert rows[-1][:3] == pytest.approx(FAR[:3]) and np.linalg.norm(np.subtract(rows[0][:3], rows[-1][:3])) < 0.05


def test_a_contact_approach_label_is_a_contact_without_expect_contact():
    near = [0.33, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    api = _api()
    assert _plan(api, near, interaction="contact_approach", global_planner=True).path.route == "line"
    assert _asked(api, "contact_path")
    api = _api()
    planned = _plan(api, interaction="contact_approach", global_planner=True, capabilities=[route.STAGED_CAPABILITY])
    assert planned.path.route == "staged" and _asked(api, "contact_path")
    with pytest.raises(checks.PlanRefusal) as refused:
        _plan(_api(contact={"status": "blocked", "limiting_link": "link6", "penetration_m": 0.021}), near,
              interaction="contact_approach", global_planner=True)
    assert refused.value.code == "contact_path_blocked"


def test_only_a_straight_contact_is_checked_and_only_with_a_planner():
    near = [0.33, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    api = _api()
    _plan(api, near, interaction="contact_approach")
    assert not _asked(api, "contact_path"), "no planner, no world to check against"
    api = _api()
    _plan(api, near, interaction="precision", global_planner=True)
    assert not _asked(api, "contact_path")
    api = _api(holes=[(0.31, 2.0)])
    assert _plan(api, near, interaction="contact_approach", global_planner=True).path.route == "global"
    assert not _asked(api, "contact_path"), "a line that went around is the planner's own path"


def test_the_straight_route_is_the_whole_line_and_never_another_path():
    api = _api()
    planned = _plan(api, route_kind="straight", global_planner=True)
    assert planned.path.route == "line" and not _asked(api, "plan_path") and _asked(api, "contact_path")
    api = _api(holes=[(0.55, 2.0)])
    with pytest.raises(checks.PlanRefusal):
        _plan(api, route_kind="straight", global_planner=True)
    assert not _asked(api, "plan_path")


def test_a_contact_a_link_behind_the_fingers_would_make_is_refused():
    api = _api(contact={"status": "blocked", "limiting_link": "link6", "penetration_m": 0.021, "first_blocked_row": 3})
    with pytest.raises(checks.PlanRefusal) as refused:
        _plan(api, [0.33, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], global_planner=True, expect_contact=True)
    assert refused.value.code == "contact_path_blocked"
    assert refused.value.constraints == {"arm": "left", "limiting_link": "link6", "penetration_m": 0.021}
    assert not _asked(api, "pace_path")


def test_a_contact_where_nothing_can_check_it_still_runs():
    api = _api(contact={"status": "unavailable", "reason": "no_planner"})
    assert _plan(api, [0.33, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], global_planner=True, expect_contact=True).fraction == 1.0


def test_home_goes_by_the_planner_when_one_answers():
    api = _api()
    option = recovery.global_option(api, arm="left", target=[0.6, 0.1, 0.2, 0.0, 0.0, 0.0], detail={"unwind_target": {}})
    assert option.method == recovery.PLANNED and option.detail["planned_path"] is True
    assert option.phases[0][0].tolist() == Q0
    assert _asked(api, "plan_path")[0]["goal_joints"] == [0.6, 0.1, 0.2, 0.0, 0.0, 0.0]
    assert recovery.global_option(_api(plan={"status": "refused", "reason": "path_blocked"}), arm="left",
                                  target=Q0) is None
