"""dry_run on move_pose, move_to and the rotations (``ora/agent/toolkit/interface/dry_run.py``): the plan the real call
runs before it acts, returned instead of executed.

The same fake kinematics serves a real call and its dry run, so the dry run is held to the real call's own plan: its
predicted end pose is the last pose the executed action expects, its IK solution the action's last knot, its native
actions the action's knots. A dry run executes nothing, whatever the tool does: the data interface it is given has no
action capability. A plan the real call would refuse is the dry run's answer; a malformed request stays a refusal.
"""
import json
import math
import os
from types import SimpleNamespace

import pytest

from ora.agent.toolkit import explain
from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.interface.types import Response
from ora.agent.toolkit.tools import _dry_run
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.tools.move_pose import tool
from ora.agent.toolkit.tools.move_to import tool as move_to
from tests.support.agent.toolkit_dry_run import GOAL, START, Fixture, compiled as _compiled, measured, response as _response


@pytest.fixture(autouse=True)
def current(monkeypatch):
    return measured(monkeypatch)


def _dry(function, *arguments):
    with dry_run.dry():
        return function(*arguments)


def test_a_dry_run_is_the_plan_the_real_call_runs_and_executes_nothing():
    real = Fixture()
    result = tool.run(real.api, {"targets": {}, "speed": "normal"})
    action, budget = real.executed[0]
    knots = action["arguments"]["arms"]["left"]["knots_rad"]
    expected = action["arguments"]["arms"]["left"]["expected_link6_xyz_xyzw"]
    plain = Fixture()
    record = _dry(tool.run, plain.api, {"targets": {}, "speed": "normal"})
    assert plain.executed == [] and "execute_action" not in plain.queries
    assert record["dry_run"] is True and record["status"] == "planned" and record["feasible"] is True
    assert record["action"] == "move_pose" and record["failure_code"] is None and record["failure_stage"] is None
    left = record["arms"]["left"]
    assert left["predicted_pose"] == pytest.approx(expected[-1], abs=1e-5)
    assert left["target_pose"] == pytest.approx(GOAL, abs=1e-5)
    assert left["end_joints_deg"] == pytest.approx([math.degrees(value) for value in knots[-1]], abs=1e-3)
    assert record["route"] == result["motion"]["route"] == "joint"
    assert record["motion"]["native_actions"] == len(knots) and budget == len(knots) + 3
    assert record["motion"]["fraction_of_move"] == 1.0 and "remaining_m" not in record["motion"]
    assert record["motion"]["duration_s"] == result["motion"]["duration_s"]
    assert record["speed"] == result["motion"]["speed"]
    assert record["compiled_pose"] == result["compiled_pose"]
    assert record["screening"]["status"] == "clear" and record["screening"]["segments_screened"] == len(knots)
    assert record["screening"]["min_inter_arm_clearance_m"] == 0.125
    assert "world's response" in record["claim_boundary"]
    assert json.loads(json.dumps(record, allow_nan=False)) == record, "plain JSON"
    assert "_rad" not in json.dumps(record), "every angle a policy sees is in degrees"
    explained = explain.render("move_pose", arguments={"targets": {}}, result=record, strict=True)
    assert explained.text.splitlines()[0] == "move_pose()  dry run: feasible" and not explained.failed
    assert "least inter-arm clearance 12.5 cm" in explained.text and "left ends at (0.400, 0.000, 0.200" in explained.text


def test_nothing_in_a_dry_run_can_act_even_if_the_tool_tried():
    fixture = Fixture()
    api = SimpleNamespace(**{**vars(fixture.api), "act": dry_run.NoActions()})
    assert _dry(tool.run, api, {"targets": {}})["feasible"] is True
    with pytest.raises(dry_run.DryRunAction):
        api.act.execute_action({}, 3)
    with pytest.raises(dry_run.DryRunAction):
        api.act.submit_action(action={})
    with pytest.raises(dry_run.DryRunAction):  # the real call, given the same interface, cannot run
        tool.run(api, {"targets": {}})
    assert not dry_run.requested()


def test_a_dry_run_leaves_the_tool_as_it_was_for_the_next_real_call():
    fixture = Fixture()
    first = tool.run(fixture.api, {"targets": {}})
    _dry(tool.run, fixture.api, {"targets": {}})
    second = tool.run(Fixture().api, {"targets": {}})
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(fixture.executed) == 1 and not dry_run.requested()


def test_a_move_cut_to_its_budget_is_feasible_and_says_where_it_stops(monkeypatch):
    far = [4.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
    monkeypatch.setattr("ora.agent.toolkit.tools.move_pose.tool.compile_action",
                        lambda api, arguments, **kwargs: _compiled(far))
    fixture = Fixture()
    record = _dry(tool.run, fixture.api, {"targets": {}, "speed": "normal"})
    assert record["feasible"] is True and record["motion"]["fraction_of_move"] < 1.0
    assert record["motion"]["remaining_m"]["left"] > 0.4
    assert record["arms"]["left"]["predicted_pose"][0] < far[0], "the cut point, not the target"
    assert record["motion"]["whole_native_actions"] > record["motion"]["native_action_budget"] - 3
    assert fixture.executed == []


GLOBAL_CONFIG = {"synchronous_execution": True,
                 "motion": {"joint_trajectory": True, "capabilities": [], "examples": [], "global_planner": True}}


def test_a_dry_run_takes_the_planner_s_route_as_the_real_call_does():
    path = [[0.3, 0.0, 0.2, 0, 0, 0], [0.35, 0.02, 0.2, 0, 0, 0], [*GOAL[:3], 0, 0, 0]]
    planned = {"status": "planned", "reason": None, "start_joints": path[0], "path_joints": path}
    fixture = Fixture(config=GLOBAL_CONFIG, planned=planned)
    record = _dry(tool.run, fixture.api, {"targets": {}})
    assert record["feasible"] is True and record["route"] == "global" and "plan_path" in fixture.queries
    assert record["arms"]["left"]["predicted_pose"][:3] == pytest.approx(GOAL[:3])
    assert record["failure_code"] is None and not fixture.executed
    real = Fixture(config=GLOBAL_CONFIG, planned=planned)
    action = tool.run(real.api, {"targets": {}})
    assert real.executed and action["motion"]["route"] == "global"
    end = real.executed[0][0]["arguments"]["arms"]["left"]["knots_rad"][-1]
    assert [math.degrees(value) for value in end] == pytest.approx(record["arms"]["left"]["end_joints_deg"], abs=1e-3)


@pytest.mark.parametrize("reason", ["goal_unreachable", "goal_in_collision", "goal_below_support", "start_in_collision"])
def test_a_planner_refusal_is_the_dry_run_s_answer_with_the_real_call_s_code(reason):
    refused = {"status": "refused", "reason": reason}
    record = _dry(tool.run, Fixture(config=GLOBAL_CONFIG, planned=refused).api, {"targets": {}})
    assert record["feasible"] is False and (record["failure_code"], record["failure_stage"]) == (reason, "plan")
    explained = explain.render("move_pose", arguments={"targets": {}}, result=record, strict=True)
    assert explained.failed and reason in explained.text
    with pytest.raises(ToolRefusal) as real:
        tool.run(Fixture(config=GLOBAL_CONFIG, planned=refused).api, {"targets": {}})
    assert real.value.code == reason


@pytest.mark.parametrize("status,code", [("inter_arm_collision", "inter_arm_collision"),
                                         ("table_collision", "table_collision"), ("self_collision", "self_collision")])
def test_a_blocked_path_is_the_answer_and_the_real_call_refuses_it_with_the_same_code(status, code):
    screen = {"status": status, "arm": "left", "first_blocked_segment": 2, "min_inter_arm_clearance_m": 0.001}
    record = _dry(tool.run, Fixture(screens=[screen]).api, {"targets": {}})
    assert record["feasible"] is False and record["status"] == "infeasible"
    assert (record["failure_code"], record["failure_stage"]) == (code, "screening")
    assert record["refusal"]["code"] == code and record["refusal"]["constraints"]["native_action"] == 2
    assert "arms" not in record and "route" not in record, "no plan was made"
    explained = explain.render("move_pose", arguments={"targets": {}}, result=record, strict=True)
    assert explained.failed and explained.text.splitlines()[0] == f"move_pose()  dry run: would be refused, {code}"
    assert f"{code} at the screening stage" in explained.text and explained.text.splitlines()[-1].startswith("next: choose targets")
    with pytest.raises(ToolRefusal) as error:
        tool.run(Fixture(screens=[screen]).api, {"targets": {}})
    assert error.value.code == code


def test_an_ik_hole_is_the_answer_with_the_real_calls_code_and_how_far_into_the_move():
    fixture = Fixture()
    original = fixture.api.planner.kinematics

    def kinematics(*, request):  # an IK that fails from the third pose of the chain on
        if request["operation"] != "ik_chain":
            return original(request=request)
        return _response({"rows": [{"index": i, "status": "ok", "joints_rad": [*pose[:3], 0, 0, 0]} if i < 2
                                   else {"index": i, "status": "ik_failed", "joints_rad": None}
                                   for i, pose in enumerate(request["poses_xyz_xyzw"])]})

    fixture.api.planner = SimpleNamespace(kinematics=kinematics)
    record = _dry(tool.run, fixture.api, {"targets": {}, "expect_contact": True})
    assert (record["feasible"], record["failure_code"], record["failure_stage"]) == (False, "paced_path_unreachable", "ik")
    assert 0 < record["refusal"]["constraints"]["fraction_of_move"] < 1
    assert record["refusal"]["constraints"]["arm"] == "left" and fixture.executed == []
    explained = explain.render("move_pose", arguments={"targets": {}}, result=record)
    assert explained.failed and "1. plan: " in explained.text and "no admissible IK solution from" in explained.text
    assert explained.text.endswith("dry_run the new target\n")
    with pytest.raises(ToolRefusal) as error:
        tool.run(fixture.api, {"targets": {}, "expect_contact": True})
    assert error.value.code == "paced_path_unreachable" and fixture.executed == []


def test_a_target_outside_the_workspace_is_a_request_stage_answer(monkeypatch):
    def outside(api, arguments, **kwargs):
        raise ToolRefusal("target pose is outside the configured action workspace", code="workspace_bounds",
                          field="targets.left.position_m", constraints={"workspace_bounds_m": {"z": [0.7, 1.4]}})

    monkeypatch.setattr("ora.agent.toolkit.tools.move_pose.tool.compile_action", outside)
    record = _dry(tool.run, Fixture().api, {"targets": {}})
    assert (record["feasible"], record["failure_code"], record["failure_stage"]) == (False, "workspace_bounds", "request")
    assert record["refusal"]["field"] == "targets.left.position_m"


def test_a_malformed_request_or_a_planner_that_cannot_answer_is_a_refusal_not_an_answer(monkeypatch):
    for arguments in ({"targets": {}, "speed": "warp"}, {"targets": {}, "slow_approaching": "yes"}):
        with pytest.raises(ToolRefusal) as error:
            _dry(tool.run, Fixture().api, arguments)
        assert error.value.code == "invalid_motion_arguments"
    fixture = Fixture()

    def broken(*, request):
        return Response(json.dumps({"status": "ok", "method": "kinematics", "data": {}, "error": "bridge down"}).encode())

    fixture.api.planner = SimpleNamespace(kinematics=broken)
    with pytest.raises(ToolRefusal) as error:
        _dry(tool.run, fixture.api, {"targets": {}})
    assert error.value.code == "planner_failed" and fixture.executed == []


def test_where_the_environment_plans_the_move_the_tool_stops_at_the_action_it_would_hand_over():
    fixture = Fixture(config={"synchronous_execution": True, "motion": {"examples": []}})
    api = SimpleNamespace(**{**vars(fixture.api), "act": dry_run.NoActions()})
    for run in (lambda: tool.run(api, {"targets": {}}), lambda: move_to.run(api, {"targets": {}})):
        with pytest.raises(dry_run.Proposed) as stopped:
            _dry(run)
        assert stopped.value.action["arguments"]["targets"]["left"]["position_m"] == GOAL[:3]
    assert fixture.executed == [] and fixture.queries == []
    assert _dry_run.verdict(ToolRefusal("x", code="self_collision"))
    assert not _dry_run.verdict(ToolRefusal("x", code=dry_run.UNAVAILABLE))


def test_move_to_and_the_rotations_dry_run_through_the_same_plan(monkeypatch):
    real = Fixture()
    monkeypatch.setattr("ora.agent.toolkit.tools.move_pose.tool.compile_action",
                        lambda api, arguments, **kwargs: _compiled(GOAL))
    result = move_to.run(real.api, {"targets": {}})
    action, _ = real.executed[0]
    plain = Fixture()
    record = _dry(move_to.run, plain.api, {"targets": {"left": {"position_m": GOAL[:3]}}})
    assert plain.executed == [] and record["action"] == "move_to" and record["feasible"] is True
    assert record["arms"]["left"]["predicted_pose"] == pytest.approx(
        action["arguments"]["arms"]["left"]["expected_link6_xyz_xyzw"][-1], abs=1e-5)
    assert record["motion"]["native_actions"] == len(action["arguments"]["arms"]["left"]["knots_rad"])
    assert result["status"] == "executed" and "dry_run" not in result, "a real call's result has no dry-run field"
    blocked = Fixture(screens=[{"status": "table_collision", "arm": "left", "first_blocked_segment": 1}])
    verdict = _dry(move_to.run, blocked.api, {"targets": {}})
    assert verdict["failure_code"] == "table_collision" and blocked.executed == []


def test_every_action_that_moves_the_robot_takes_dry_run_and_nothing_else_does():
    """One rule, read off the surfaces: moving tools have it, on every default surface; reads and wait_ticks have none;
    the Synthetic ladder, which moves no robot, is untouched. A tool's own description file does not carry it: the
    surface adds the one parameter."""
    from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG
    from ora.agent.composition.surface.tool_surface import DRY_RUN_CAPABILITY, resolve_tool_surface
    from tests.support.agent.surface_catalog import catalog_grants

    served = set()
    for profile in PROFILE_CATALOG:
        surface = resolve_tool_surface(profile, toolkit_grants=catalog_grants(profile))
        takes = {item["name"] for item in surface.definitions if "dry_run" in item["inputSchema"].get("properties", {})}
        if DRY_RUN_CAPABILITY not in surface.capabilities:
            assert not takes, profile
            continue
        served.add(profile)
        assert takes == set(surface.mutation_tools) & set(dry_run.TOOLS), profile
        assert takes == set(surface.mutation_tools) - {"wait_ticks"}, profile
    assert served == {"robodojo_placement_toolkit_v5", "robodojo_placement_toolkit_v6",
                      "robotwin_toolchain_b2_closed_noservo_v2",
                      "libero_placement_direct_overhead_v1", "robocasa_mobile_direct_v1", "robolab_direct_v1",
                      "yam_direct_v1", "yam_perception_v1"}
    for name in dry_run.TOOLS:
        path = f"ora/agent/toolkit/tools/{name}/description.json"
        if os.path.exists(path):
            text = open(path, encoding="utf-8").read()
            assert "dry_run" not in text and "dry run" not in text.lower(), path


def test_a_real_call_is_the_same_bytes_with_a_dry_run_before_it_and_without():
    plain, asked = Fixture(), Fixture()
    first = tool.run(plain.api, {"targets": {}, "speed": "normal"})
    _dry(tool.run, asked.api, {"targets": {}, "speed": "normal"})
    again = tool.run(asked.api, {"targets": {}, "speed": "normal"})
    assert again["receipt"].pop("ran_previewed_plan") is True, "the one difference, and the receipt says it"
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert json.dumps(plain.executed, sort_keys=True) == json.dumps(asked.executed, sort_keys=True)


def test_on_recorded_runs_the_predicted_end_pose_is_where_the_arm_ended():
    """68 moves of 5 sealed RoboDojo toolkit-v5 runs ran whole and reached: the plan's expected end pose (what a dry run
    returns as ``predicted_pose``) against the receipt's measured pose. Largest seen: 0.81 mm and 0.12 deg."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[6] / "tests/data/code_mode/motion-predictions.json"
    moves = json.loads(root.read_text(encoding="utf-8"))["moves"]
    assert len(moves) >= 68 and len({move["run"] for move in moves}) >= 5
    for move in moves:
        predicted, measured = move["predicted_pose"], move["measured_pose"]
        assert math.dist(predicted[:3], measured[:3]) <= 0.002, move["run"]
        dot = abs(sum(a * b for a, b in zip(predicted[3:], measured[3:]))) / (
            math.hypot(*predicted[3:]) * math.hypot(*measured[3:]))
        assert math.degrees(2 * math.acos(min(1.0, dot))) <= 0.5, move["run"]


@pytest.mark.parametrize("name", ["rotate_wrist", "rotate_pitch"])
def test_the_rotations_dry_run_through_the_same_plan_and_the_same_request_check(name, monkeypatch):
    import importlib

    module = importlib.import_module(f"ora.agent.toolkit.tools.{name}.tool")
    real, plain, asked = Fixture(), Fixture(), []
    inner = tool.compile_action
    monkeypatch.setattr(tool, "compile_action", lambda api, arguments, **kwargs: (asked.append(kwargs), inner(api, arguments, **kwargs))[1])
    module.run(real.api, {"targets": {"left": 30.0}})
    record = _dry(module.run, plain.api, {"targets": {"left": 30.0}})
    assert plain.executed == [] and record["action"] == name and record["feasible"] is True
    assert record["arms"]["left"]["predicted_pose"] == pytest.approx(
        real.executed[0][0]["arguments"]["arms"]["left"]["expected_link6_xyz_xyzw"][-1], abs=1e-5)
    assert asked[0]["primitive"] == asked[1]["primitive"] == name, "compiled by the real call's own compiler"
    assert asked[0]["request_for_target"] is not asked[1]["request_for_target"]  # one per call, the same check
    with pytest.raises(ValueError):  # the angle bound the real call enforces
        asked[1]["request_for_target"](270.0)
