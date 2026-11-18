"""move_pose's speeds: one joint trajectory per call, at fast (default), normal, careful or speed_m_s, an
optional slow final approach, screened before it moves, and cut to its budget (never reported as an arrival)."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.agent.paced_kinematics import paced_response
from ora.agent.toolkit.interface.types import Response
from ora.agent.toolkit.lib.motion_plan.route import STAGED_CAPABILITY
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.move_pose import _smooth, tool
from ora.contracts.joint_trajectory import speed_segment_spend

START = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
RIGHT = [0.3, -0.4, 0.2, 0.0, 0.0, 0.0, 1.0]
NEAR = [0.33, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
FAR = [0.45, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
PLANNER = {"synchronous_execution": True, "motion": {"joint_trajectory": True, "global_planner": True,
                                                     "capabilities": [STAGED_CAPABILITY], "examples": []}}


def _response(data, method):
    return Response(json.dumps({"status": "ok", "method": method, "data": data}).encode())


screens = []


def _api(executed, config, seen=None, extra=None, contact=None, holes=()):
    def kinematics(*, request):
        if request["operation"] == "pace_path":
            return paced_response(request)
        if seen is not None:
            seen.append(request)
        if request["operation"] == "plan_path":
            first = [*START[:3], 0, 0, 0]
            path = np.linspace(first, [*request["goal_pose_xyz_xyzw"][:3], 0, 0, 0], 12).tolist()
            return _response({"status": "planned", "reason": None, "start_joints": first, "path_joints": path},
                             "kinematics")
        if request["operation"] == "contact_path":
            return _response(contact or {"status": "clear"}, "kinematics")
        if request["operation"] == "screen_path":
            return _response(screens.pop(0) if screens else {"status": "clear"}, "kinematics")
        if request["operation"] == "fk":
            rows = [{"index": i, "pose_xyz_xyzw": [*q[:3], 0, 0, 0, 1]} for i, q in enumerate(request["joints_rad"])]
        else:
            rows = [{"index": i, "status": "ik_failed", "joints_rad": None}
                    if any(low <= pose[0] <= high for low, high in holes)
                    else {"index": i, "status": "ok", "joints_rad": [*pose[:3], 0, 0, 0]}
                    for i, pose in enumerate(request["poses_xyz_xyzw"])]
            return _response({"rows": rows, **(extra or {})}, "kinematics")
        return _response({"rows": rows}, "kinematics")

    def execute_action(action, budget):
        executed.append((action, budget))
        knots = next(iter(action["arguments"]["arms"].values()))["knots_rad"]
        trajectory = {"status": "completed", "knots_planned": len(knots), "knots_executed": len(knots), "stop": None}
        if "speed_segments" in action["arguments"]:
            trajectory["speed_segments"] = {arm: speed_segment_spend(rows, len(knots))
                                            for arm, rows in action["arguments"]["speed_segments"].items()}
        status = "not_reached" if "part_of_move" in action["arguments"] else "reached"
        return _response({"remaining_budget": 50 - budget, "receipt": {
            "status": status, "native_actions": len(knots), "stage_evidence": {"trajectory": trajectory}}},
            "execute_action")

    return SimpleNamespace(artifacts=SimpleNamespace(tool_context=lambda: config),
                           planner=SimpleNamespace(kinematics=kinematics),
                           act=SimpleNamespace(execute_action=execute_action))


def _run(monkeypatch, goal, config=None, *, speed="normal", right=None, interaction="free_space", jaw=None,
         **options):
    goals = {"left": goal, **({"right": right} if right is not None else {})}
    targets = {arm: {"position_m": pose[:3], "orientation_xyzw": pose[3:], "controlled_point": "tcp"}
               for arm, pose in goals.items()}
    action = {"action_id": "action-7", "arguments": {"targets": targets,
                                                     "interactions": {arm: interaction for arm in goals}}}
    starts = {"left": START, "right": RIGHT}
    obs = SimpleNamespace(
        manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": {
            arm: {"pose_xyz_xyzw": pose, "arm_joint_qpos": [*pose[:3], 0, 0, 0]} for arm, pose in starts.items()}}}),
        identity=SimpleNamespace(json=lambda: {"observation_id": "o", "observation_sha256": "s"}))
    received = []
    monkeypatch.setattr("ora.agent.toolkit.tools.move_pose.tool.compile_action",
                        lambda api, arguments, **kw: received.append(arguments) or (action, {"targets": jaw or {}}))
    monkeypatch.setattr(_smooth.observation, "observe", lambda api: obs)
    executed = []
    config = config or {"synchronous_execution": True,
                        "motion": {"joint_trajectory": True, "capabilities": [], "examples": []}}
    api_options = {key: options.pop(key) for key in ("seen", "extra", "contact", "holes") if key in options}
    arguments = {"targets": {}, **({"speed": speed} if speed is not None else {}), **options}
    result = tool.run(_api(executed, config, **api_options), arguments)
    assert received[0] == {"targets": {}}  # speed and slow_approaching never reach the compiler
    return result, executed


def test_a_normal_move_runs_as_one_trajectory_at_about_0_3_m_s(monkeypatch):
    result, executed = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert result["status"] == "executed" and len(executed) == 1
    action, budget = executed[0]
    knots = action["arguments"]["arms"]["left"]["knots_rad"]
    assert action["action"] == "follow_joint_trajectory" and action["action_id"] == "action-7"
    assert budget == len(knots) + 3 and knots[-1][0] == pytest.approx(0.4)
    steps = [b[0] - a[0] for a, b in zip([[0.3]] + knots, knots)]
    assert max(steps) <= 0.3 * 0.04 + 1e-9  # 0.3 m/s at one knot per 40 ms
    assert result["motion"]["route"] == "joint" and result["motion"]["status"] == "completed"
    assert result["action"] == "move_pose" and result["motion"]["speed"]["speed"] == "normal"


def test_a_move_over_the_budget_runs_its_first_part(monkeypatch):
    result, executed = _run(monkeypatch, [4.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    limited = result["motion"]["budget_limited"]
    assert result["status"] == "partial" and 0 < limited["fraction_of_move"] < 1
    assert limited["native_action_budget"] == 250 and limited["needed_native_actions"] > 247
    action, budget = executed[0]
    assert budget <= 250 and limited["remaining_m"]["left"] > 0.4


def test_without_a_trajectory_backend_the_paced_tool_is_unavailable(monkeypatch):
    from ora.agent.toolkit.tools.errors import ToolRefusal

    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
             config={"synchronous_execution": True, "motion": {"examples": []}})
    assert error.value.code == "tool_unavailable"


def test_a_path_the_screen_blocks_is_refused_before_anything_moves(monkeypatch):
    from ora.agent.toolkit.tools.errors import ToolRefusal

    screens[:] = [{"status": "inter_arm_collision", "first_blocked_segment": 2, "min_inter_arm_clearance_m": 0.001,
                   "limiting_links": {"left": "link6", "right": "link5"}}]
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert error.value.code == "inter_arm_collision"
    assert 0 < error.value.constraints["fraction_of_move"] < 1 and error.value.constraints["native_action"] == 2
    screens[:] = [{"status": "table_collision", "arm": "left", "first_blocked_segment": 1}]
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert error.value.code == "table_collision" and error.value.constraints["arm"] == "left"


def test_the_screen_sees_the_measured_start_and_every_knot(monkeypatch):
    seen = []
    _, executed = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], seen=seen)
    screen = [request for request in seen if request["operation"] == "screen_path"][0]
    knots = executed[0][0]["arguments"]["arms"]["left"]["knots_rad"]
    assert screen["knots_rad"]["left"] == [[*START[:3], 0, 0, 0], *knots]


def test_a_cut_move_says_so_in_the_action_and_the_result(monkeypatch):
    result, executed = _run(monkeypatch, [4.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    part = executed[0][0]["arguments"]["part_of_move"]
    assert 0 < part["fraction"] < 1 and part["goals_xyz_xyzw"]["left"][0] == pytest.approx(4.0)
    limited = result["motion"]["budget_limited"]
    assert limited["residuals_against"] == "cut_point" and "NOT reached" in limited["note"]
    assert limited["limited_by"] == "tool_native_action_budget"


def test_near_the_episode_limit_the_move_fits_what_is_left(monkeypatch):
    config = {"synchronous_execution": True, "episode_native_actions_remaining": 12,
              "motion": {"joint_trajectory": True, "capabilities": [], "examples": []}}
    result, executed = _run(monkeypatch, [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], config=config)
    action, budget = executed[0]
    assert budget <= 12 and len(action["arguments"]["arms"]["left"]["knots_rad"]) + 3 == budget
    limited = result["motion"]["budget_limited"]
    assert limited["limited_by"] == "episode_native_actions_remaining" and limited["native_action_budget"] == 12


def test_inside_a_code_mode_script_the_move_fits_what_the_script_has_left(monkeypatch):
    """A script's calls share its parent's allowance: a move plans against what the script has left (the
    parent the Supervisor admits it from), so it ends at rest on a reported cut instead of being stopped
    mid-path by the parent's admission."""
    config = {"synchronous_execution": True, "episode_native_actions_remaining": 400,
              "script_native_actions_remaining": 12,
              "motion": {"joint_trajectory": True, "capabilities": [], "examples": []}}
    result, executed = _run(monkeypatch, [0.6, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], config=config)
    action, budget = executed[0]
    assert budget <= 12 and len(action["arguments"]["arms"]["left"]["knots_rad"]) + 3 == budget
    limited = result["motion"]["budget_limited"]
    assert limited["limited_by"] == "script_native_actions_remaining" and limited["native_action_budget"] == 12
    assert "script" in limited["note"]


@pytest.mark.parametrize("episode,script,expected", [(None, None, 20), (15, None, 15), (400, 9, 9), (6, 9, 6)])
def test_a_compiled_move_plans_against_the_tightest_allowance(monkeypatch, episode, script, expected):
    """move_to, rotate_* and grasp_execute's arm stages (``run_action``): the call's budget, at most what the
    episode and, inside a script, the script has left."""
    from ora.agent.toolkit.lib.motion_plan import checks

    seen = []

    def plan(api, **kwargs):
        seen.append(kwargs["native_action_budget"])
        raise checks.PlanRefusal("stop here", code="fixture")

    monkeypatch.setattr(_smooth.move, "plan", plan)
    obs = SimpleNamespace(
        manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": {
            "left": {"pose_xyz_xyzw": START, "arm_joint_qpos": [*START[:3], 0, 0, 0]}}}}),
        identity=SimpleNamespace(json=lambda: {"observation_id": "o", "observation_sha256": "s"}))
    monkeypatch.setattr(_smooth.observation, "observe", lambda api: obs)
    config = {"synchronous_execution": True, "motion": {"joint_trajectory": True, "capabilities": []},
              **({"episode_native_actions_remaining": episode} if episode is not None else {}),
              **({"script_native_actions_remaining": script} if script is not None else {})}
    action = {"action_id": "a", "arguments": {"targets": {"left": {"position_m": START[:3],
                                                                   "orientation_xyzw": START[3:]}},
                                              "interactions": {"left": "free_space"}}}
    with pytest.raises(checks.PlanRefusal):
        _smooth.run_action(_api([], config), action, config, budget=20)
    assert seen == [expected]


def test_an_unrecorded_ik_generator_warning_reaches_the_result(monkeypatch):
    result, _ = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
                     extra={"ik_sampler_state": "unrecorded", "warning": "environment text"})
    assert result["motion"]["ik_warning"] == _smooth.IK_UNRECORDED_WARNING
    assert "predates IK-generator recording" in result["motion"]["ik_warning"]
    clean, _ = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert "ik_warning" not in clean["motion"]


def test_stop_and_settle_angles_are_labelled():
    summary = {"stop": {"reason": "tracking", "error": 0.1, "bound": 0.05, "knot": 3},
               "settle": {"settled": False, "joint_speed_rad_s": 0.5, "joint_error_rad": 0.01}}
    from ora.agent.toolkit.lib.motion_plan import move

    labelled = move.outcome(summary)
    assert labelled["stop"] == {"reason": "tracking", "joint_error_rad": 0.1, "tracking_bound_rad": 0.05, "knot": 3}
    assert "joint_speed_rad_s" not in labelled["settle"]
    assert labelled["settle"]["joint_speed_deg_s"] == pytest.approx(28.648, abs=1e-3)


def _knot_speeds(action, arm="left", start=START):
    rows = np.asarray([[*start[:3], 0, 0, 0], *action["arguments"]["arms"][arm]["knots_rad"]])
    return np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1) / 0.04


def test_fast_is_the_default_and_a_wrong_speed_is_refused(monkeypatch):
    result, executed = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed=None)
    report = dict(result["motion"]["speed"])
    assert report.pop("planned_peak_linear_m_s") > 0 and report.pop("planned_peak_angular_deg_s") == 0
    assert report == {"speed": "fast", "slow_approaching": False, "linear_m_s": None, "angular_deg_s": None,
                      "joint_speed_deg_s": pytest.approx(229.183, abs=1e-3)}
    assert result["motion"]["route"] == "joint" and executed[0][1] == len(
        executed[0][0]["arguments"]["arms"]["left"]["knots_rad"]) + 5
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed="slow")
    assert error.value.code == "invalid_motion_arguments" and error.value.fields["field"] == "speed"
    assert error.value.constraints["allowed"] == ["fast", "normal", "careful"]


def test_fast_runs_at_the_pace_of_the_context_s_motion_profile(monkeypatch):
    import hashlib

    from ora.agent.toolkit.host.motion import profile_entry
    from ora.orchestrator.paths import PROJECT_ROOT

    path = "ora/policy/embodiments/arx-x5/motion-profile-v2.json"
    record = {"path": path, "sha256": hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()}
    config = {"synchronous_execution": True, "motion": {"joint_trajectory": True, "capabilities": [], "examples": [],
                                                        "profile": profile_entry(record, root=PROJECT_ROOT)}}
    goal = [1.2, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    v1, before = _run(monkeypatch, goal, speed=None)
    v2, after = _run(monkeypatch, goal, config=config, speed=None)
    assert v2["motion"]["speed"]["joint_speed_deg_s"] == pytest.approx(257.831, abs=1e-3)
    knots = [len(run[0][0]["arguments"]["arms"]["left"]["knots_rad"]) for run in (before, after)]
    assert knots[1] < knots[0]
    # normal is the same pace whatever the profile.
    normal_v1, normal_before = _run(monkeypatch, goal)
    normal_v2, normal_after = _run(monkeypatch, goal, config=config)
    assert normal_before[0][0]["arguments"] == normal_after[0][0]["arguments"]
    assert normal_v1["motion"]["speed"] == normal_v2["motion"]["speed"]


def test_without_a_trajectory_backend_only_a_fast_move_runs(monkeypatch):
    config = {"synchronous_execution": True, "motion": {"examples": []}}
    for options in ({"speed": "normal"}, {"speed": None, "slow_approaching": True}):
        with pytest.raises(ToolRefusal) as error:
            _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], config=config, **options)
        assert error.value.code == "tool_unavailable"


def test_slow_approaching_slows_the_last_two_centimetres_without_a_stop(monkeypatch):
    result, executed = _run(monkeypatch, [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], slow_approaching=True)
    action, budget = executed[0]
    speeds = _knot_speeds(action)
    segments = action["arguments"]["speed_segments"]["left"]
    assert [segment["speed"] for segment in segments] == ["normal", "careful"]
    switch = segments[1]["first_knot"]
    # The segment: at most 2 cm of path left where it starts, a little less at the knot that enters it.
    assert segments[1]["path_left_m"] == 0.02
    assert 0.2 * (1 - segments[1]["from_fraction_of_move"]) == pytest.approx(0.02, abs=0.004)
    # Velocity is continuous: it decelerates into the slow segment at the normal pace's acceleration, never
    # dropping to rest before the end, and stays at or below 0.1 m/s inside it.
    assert np.all(speeds[:-1] > 0)
    assert np.all(np.abs(np.diff(speeds)) <= 1.0 * 0.04 + 1e-6)
    # The first slow knot straddles the boundary, where the speed has just come down to 0.1 m/s.
    assert speeds[switch:].max() <= 0.1 + 1e-9 and speeds[switch - 1] <= 0.12 and speeds[:switch - 1].max() > 0.25
    spent = result["motion"]["speed_segments"]["left"]
    assert [item["native_actions"] for item in spent] == [switch - 1, len(speeds) - switch + 1]
    assert result["motion"]["speed"]["slow_approaching"] is True and result["motion"]["speed"]["slow_approach_m"] == 0.02
    assert budget == len(speeds) + 3  # the careful settle


def test_a_path_shorter_than_the_slow_segment_is_careful_throughout(monkeypatch):
    result, executed = _run(monkeypatch, [0.315, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed=None, slow_approaching=True)
    action, _ = executed[0]
    segments = action["arguments"]["speed_segments"]["left"]
    assert segments[0]["speed"] == "careful" and segments[0]["first_knot"] == 1
    assert _knot_speeds(action).max() <= 0.1 + 1e-9


def test_two_arms_each_slow_over_their_own_last_two_centimetres(monkeypatch):
    result, executed = _run(monkeypatch, [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], right=[0.35, -0.4, 0.2, 0.0, 0.0, 0.0, 1.0],
                            slow_approaching=True)
    action, _ = executed[0]
    segments = action["arguments"]["speed_segments"]
    left, right = segments["left"][-1], segments["right"][-1]
    assert left["speed"] == right["speed"] == "careful"
    # The shorter right path (5 cm) starts its slow segment earlier along the shared timeline.
    assert right["first_knot"] < left["first_knot"]
    assert right["path_left_m"] <= 0.02 and left["path_left_m"] <= 0.02
    assert _knot_speeds(action, "right", RIGHT)[right["first_knot"]:].max() <= 0.1 + 1e-9
    assert _knot_speeds(action)[left["first_knot"]:].max() <= 0.1 + 1e-9


def test_a_budget_cut_inside_the_slow_segment_keeps_its_knots_and_says_so(monkeypatch):
    config = {"synchronous_execution": True, "episode_native_actions_remaining": 18,
              "motion": {"joint_trajectory": True, "capabilities": [], "examples": []}}
    # 6 cm at careful-free normal speed needs about 22 knots; 18 - 3 settle = 15 cut inside the slow part.
    whole, _ = _run(monkeypatch, [0.36, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], slow_approaching=True)
    switch = whole["motion"]["speed_segments"]["left"][1]["first_knot"]
    result, executed = _run(monkeypatch, [0.36, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], config=config, slow_approaching=True)
    action, budget = executed[0]
    count = len(action["arguments"]["arms"]["left"]["knots_rad"])
    assert budget == count + 3 and count <= 15 and count < whole["motion"]["planned_native_actions"]
    assert result["status"] == "partial" and "NOT reached" in result["motion"]["budget_limited"]["note"]
    assert _knot_speeds(action)[-1] < 0.1  # it slows down at the cut


def test_slow_approaching_with_careful_is_refused(monkeypatch):
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed="careful", slow_approaching=True)
    assert error.value.code == "invalid_motion_arguments" and error.value.fields["field"] == "slow_approaching"
    assert str(error.value) == "slow_approaching has no effect with speed=careful"
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], slow_approaching="yes")
    assert error.value.fields["field"] == "slow_approaching"


@pytest.mark.parametrize("primitive,interaction,route", [("move_to", "free_space", "joint"),
                                                         ("rotate_wrist", "precision", "line")])
def test_move_to_and_rotate_run_their_compiled_action_as_one_fast_trajectory(monkeypatch, primitive, interaction,
                                                                             route):
    from ora.agent.toolkit.tools.move_pose.tool import compile_and_submit

    target = {"position_m": [0.32, 0.0, 0.2], "orientation_xyzw": [0, 0, 0, 1]}
    action = {"action_id": "action-9", "arguments": {"targets": {"left": target},
                                                     "interactions": {"left": interaction}}}
    obs = SimpleNamespace(
        manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": {
            "left": {"pose_xyz_xyzw": START, "arm_joint_qpos": [*START[:3], 0, 0, 0]}}}}),
        identity=SimpleNamespace(json=lambda: {"observation_id": "o", "observation_sha256": "s"}))
    monkeypatch.setattr("ora.agent.toolkit.tools.move_pose.tool.compile_action",
                        lambda api, arguments, **kw: (action, {"targets": {}, "semantic_primitive": primitive}))
    monkeypatch.setattr(_smooth.observation, "observe", lambda api: obs)
    executed, seen = [], []
    config = {"synchronous_execution": True, "motion": {"joint_trajectory": True, "capabilities": [], "examples": []}}
    result = compile_and_submit(_api(executed, config, seen=seen), {}, config=config, primitive=primitive)
    assert set(result) == {"status", "action", "compiled_pose", "receipt", "remaining_budget"}
    assert result["status"] == "executed" and result["action"] == primitive
    sent, budget = executed[0]
    assert sent["action"] == "follow_joint_trajectory" and sent["action_id"] == "action-9"
    assert budget <= 20  # the tool's own budget, unchanged
    assert any(request["operation"] == "screen_path" for request in seen)
    # The interaction picks the route as for move_pose: free_space a joint line (one endpoint IK), precision the line.
    rankings = {request.get("ranking") for request in seen if request["operation"] == "ik_chain"}
    assert rankings == ({"canonical_first"} if route == "joint" else {"nearest"})


def test_expect_contact_reaches_the_action_and_the_result(monkeypatch):
    result, executed = _run(monkeypatch, NEAR, expect_contact=True, slow_approaching=True)
    assert executed[0][0]["arguments"]["expect_contact"] is True and result["motion"]["expect_contact"] is True
    plain, executed = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert "expect_contact" not in executed[0][0]["arguments"] and plain["motion"]["expect_contact"] is False
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], expect_contact="yes")
    assert error.value.fields["field"] == "expect_contact"


def test_move_to_passes_expect_contact_to_its_trajectory(monkeypatch):
    from ora.agent.toolkit.tools.move_pose.tool import compile_and_submit

    target = {"position_m": [0.32, 0.0, 0.2], "orientation_xyzw": [0, 0, 0, 1]}
    action = {"action_id": "action-9", "arguments": {"targets": {"left": target},
                                                     "interactions": {"left": "free_space"}}}
    obs = SimpleNamespace(
        manifest=SimpleNamespace(json=lambda: {"proprioception": {"end_effectors": {
            "left": {"pose_xyz_xyzw": START, "arm_joint_qpos": [*START[:3], 0, 0, 0]}}}}),
        identity=SimpleNamespace(json=lambda: {"observation_id": "o", "observation_sha256": "s"}))
    received = []
    monkeypatch.setattr("ora.agent.toolkit.tools.move_pose.tool.compile_action",
                        lambda api, arguments, **kw: received.append(arguments) or (action, {"targets": {}}))
    monkeypatch.setattr(_smooth.observation, "observe", lambda api: obs)
    executed = []
    config = {"synchronous_execution": True, "motion": {"joint_trajectory": True, "capabilities": [], "examples": []}}
    compile_and_submit(_api(executed, config), {"targets": {}, "expect_contact": True}, config=config,
                       primitive="move_to")
    assert received == [{"targets": {}}] and executed[0][0]["arguments"]["expect_contact"] is True


def test_the_jaw_choice_reaches_the_action(monkeypatch):
    result, executed = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], exact_orientation=True,
                            jaw={"left": {"jaw_symmetry": "exact"}})
    assert executed[0][0]["arguments"]["jaw_symmetry"] == {"left": "exact"}


def test_speed_m_s_caps_the_controlled_point(monkeypatch):
    result, executed = _run(monkeypatch, [0.33, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed=None, speed_m_s=0.05)
    action, budget = executed[0]
    speeds = _knot_speeds(action)
    assert speeds.max() <= 0.05 + 1e-9 and budget == len(speeds) + 3  # the paced settle
    assert result["status"] == "executed" and result["motion"]["route"] == "joint"
    report = dict(result["motion"]["speed"])
    assert 0 < report.pop("planned_peak_linear_m_s") <= 0.05 + 1e-9
    assert report == {"speed": "custom", "speed_m_s": 0.05, "slow_approaching": False, "linear_m_s": 0.05,
                      "angular_deg_s": 20.0, "joint_speed_deg_s": pytest.approx(229.183, abs=1e-3),
                      "planned_peak_angular_deg_s": 0.0}


def test_the_joints_stay_the_ceiling_of_a_high_speed_m_s(monkeypatch):
    # The fake arm's first joints are the hand's position, so 4 rad/s and 40 rad/s^2 are 4 m/s and 40 m/s^2 here.
    result, executed = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed=None, speed_m_s=100)
    peak = result["motion"]["speed"]["planned_peak_linear_m_s"]
    assert peak < 100 and peak <= 4.0 + 1e-9
    assert _knot_speeds(executed[0][0]).max() <= 4.0 + 1e-9


def test_normal_reports_the_peak_its_path_reaches(monkeypatch):
    result, _ = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert result["motion"]["speed"]["planned_peak_linear_m_s"] == pytest.approx(0.3, abs=0.03)


def test_slow_approaching_with_speed_m_s_slows_to_careful_from_above_it(monkeypatch):
    result, executed = _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed=None, speed_m_s=0.2,
                            slow_approaching=True)
    segments = executed[0][0]["arguments"]["speed_segments"]["left"]
    assert [segment["speed"] for segment in segments] == ["0.2 m/s", "careful"]
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed=None, speed_m_s=0.1, slow_approaching=True)
    assert error.value.fields["field"] == "slow_approaching"


@pytest.mark.parametrize("value", [0, -0.1, True, "fast"])
def test_a_speed_m_s_that_is_not_a_positive_number_is_refused(monkeypatch, value):
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed=None, speed_m_s=value)
    assert error.value.code == "invalid_motion_arguments" and error.value.fields["field"] == "speed_m_s"


def test_speed_and_speed_m_s_together_are_refused(monkeypatch):
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], speed="normal", speed_m_s=0.2)
    assert error.value.fields["field"] == "speed_m_s" and str(error.value) == "give speed or speed_m_s, not both"


@pytest.mark.parametrize("pace", [{"speed": None}, {"speed": "normal"}, {"speed": "careful"},
                                  {"speed": None, "speed_m_s": 0.2}])
@pytest.mark.parametrize("goal,interaction,route", [(FAR, "free_space", "global"), (NEAR, "free_space", "global"),
                                                    (FAR, "precision", "global"), (NEAR, "precision", "line"),
                                                    (FAR, "contact_approach", "staged"),
                                                    (NEAR, "contact_approach", "line")])
def test_the_label_picks_the_route_and_the_speed_only_paces_it(monkeypatch, pace, goal, interaction, route):
    result, _ = _run(monkeypatch, goal, config=PLANNER, interaction=interaction, **pace)
    assert result["motion"]["route"] == route
    assert 0 < result["motion"]["speed"]["planned_peak_linear_m_s"]


def test_a_slower_speed_paces_the_planned_route(monkeypatch):
    fast, _ = _run(monkeypatch, FAR, config=PLANNER, speed=None)
    careful, _ = _run(monkeypatch, FAR, config=PLANNER, speed="careful")
    assert careful["motion"]["route"] == fast["motion"]["route"] == "global"
    assert careful["motion"]["planned_native_actions"] > fast["motion"]["planned_native_actions"]
    assert careful["motion"]["speed"]["planned_peak_linear_m_s"] <= 0.1 + 1e-9


def _operations(seen):
    return [request["operation"] for request in seen]


def test_a_contact_approach_is_checked_without_expect_contact(monkeypatch):
    seen = []
    _run(monkeypatch, NEAR, config=PLANNER, interaction="contact_approach", seen=seen)
    assert "contact_path" in _operations(seen)
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, NEAR, config=PLANNER, interaction="contact_approach",
             contact={"status": "blocked", "limiting_link": "link6", "penetration_m": 0.02})
    assert error.value.code == "contact_path_blocked" and error.value.constraints["limiting_link"] == "link6"


def test_a_short_precision_line_has_no_contact_check(monkeypatch):
    seen = []
    result, _ = _run(monkeypatch, NEAR, config=PLANNER, interaction="precision", seen=seen)
    assert result["motion"]["route"] == "line" and "contact_path" not in _operations(seen)


@pytest.mark.parametrize("pace", [{"speed": None}, {"speed": "normal"}, {"speed": "careful"}])
def test_a_push_stays_on_its_line_at_every_length_and_speed(monkeypatch, pace):
    seen = []
    result, _ = _run(monkeypatch, FAR, config=PLANNER, expect_contact=True, seen=seen, **pace)
    assert result["motion"]["route"] == "line" and result["motion"]["expect_contact"] is True
    assert "plan_path" not in _operations(seen) and "contact_path" in _operations(seen)


def test_a_push_is_refused_rather_than_replaced(monkeypatch):
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, FAR, config=PLANNER, expect_contact=True, holes=[(0.4, 2.0)])
    assert error.value.code == "paced_path_unreachable"
    with pytest.raises(ToolRefusal) as error:
        _run(monkeypatch, FAR, config=PLANNER, expect_contact=True,
             contact={"status": "blocked", "limiting_link": "link5", "penetration_m": 0.03})
    assert error.value.code == "contact_path_blocked"
