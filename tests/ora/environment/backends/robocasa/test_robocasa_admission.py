"""RoboCasa's admission of an arm move (``session.arm_path_plan``): the lift admitted on the whole move, and the base
poses a refusal carries.

``shared_motion``'s planner questions are stood in for; the session's own rules are what is tested.
"""
from __future__ import annotations

from typing import Any

import pytest

from ora.environment.backends.robocasa import session as session_module, shared_motion
from ora.environment.controllers import robocasa_omron as C
from ora.environment.controllers.motion_pace import PaceRefusal

IDENTITY = [0.0, 0.0, 0.0, 1.0]
TARGET = [1.6, 0.0, 1.0, *IDENTITY]
FOUND = {"suggestions": [{"x_m": 1.0, "y_m": 0.0, "yaw_deg": 0.0, "distance_m": 0.5, "turn_deg": 0.0,
                          "arm_path": "admitted"}], "searched": {"asked": 3}}


class Io:
    gripper_directive = C.GRIPPER_OPEN_COMMAND

    def grip_pose(self) -> list[float]:
        return [1.0, 0.0, 1.0, *IDENTITY]

    def torso_height(self) -> float:
        return 0.1

    def arm_base_position(self) -> list[float]:
        return [0.5, 0.0, 0.8]

    def base_pose(self) -> list[float]:
        return [0.5, 0.0, 0.0]


def _refused(code: str, **detail: Any) -> dict[str, Any]:
    return {"refusal": {"path_kind": "line", **detail}, "code": code, "path": None, "poses": None,
            "torso_height_m": None, "kind": "line"}


def _admitted() -> dict[str, Any]:
    return {"refusal": None, "code": None, "path": [[0.0] * 7], "poses": [TARGET], "torso_height_m": None,
            "kind": "line"}


class Paced:
    def report(self) -> dict[str, Any]:
        return {"planned_ticks": 4, "duration_s": 0.2}


@pytest.fixture
def robocasa(monkeypatch):
    session = session_module.RoboCasaSession.__new__(session_module.RoboCasaSession)
    session.io = Io()
    calls: dict[str, list[Any]] = {"arm": [], "search": [], "paced": []}
    state: dict[str, Any] = {"first": _refused("target_unreachable_at_base_pose", cause="ik_unsolved"),
                             "lifted": {}, "heights": [0.14, 0.18, 0.22, 0.26], "paced": lambda plan: Paced()}

    def arm_path(_session, pose, kind, interaction, stand=None, *, explain=False, kept=True):
        calls["arm"].append((stand, explain))
        calls.setdefault("kept", []).append(kept)
        return state["first"] if stand is None else state["lifted"].get(round(stand.torso_m, 3), state["first"])

    def base_search(_session, pose, move, admits):
        calls["search"].append(pose)
        calls.setdefault("move", []).append(move)
        calls.setdefault("admits", []).append(admits)
        return FOUND

    def paced(_session, plan, interaction, speed):
        calls["paced"].append(speed)
        return state["paced"](plan)

    monkeypatch.setattr(shared_motion, "arm_path", arm_path)
    monkeypatch.setattr(shared_motion, "torso_heights", lambda _session, _pose: list(state["heights"]))
    monkeypatch.setattr(shared_motion, "lifted", lambda _session, height: shared_motion.Stand(0.5, 0.0, 0.0, height))
    monkeypatch.setattr(shared_motion, "base_search", base_search)
    monkeypatch.setattr(session_module, "_paced_plan", paced)
    return session, state, calls


def test_a_lift_is_offered_only_from_a_height_the_whole_move_is_admitted_and_paced_from(robocasa):
    session, state, calls = robocasa
    state["lifted"] = {0.18: _admitted(), 0.22: _admitted()}
    speed = {"preset": "careful"}
    plan = session.arm_path_plan(TARGET, "precision", speed=speed)
    assert plan["refusal"] is None and plan["torso_height_m"] == 0.18 and plan["lift_m"] == pytest.approx(0.08)
    assert plan["path"] == [[0.0] * 7] and isinstance(plan["lift_paced"][0], Paced) and plan["lift_paced"][1] is None
    assert [round(stand.torso_m, 2) for stand, _ in calls["arm"][1:]] == [0.14, 0.18]
    assert calls["paced"] == [speed], "the lift's pacing is the real call's, at the call's speed"
    assert not calls["search"]


def test_a_lift_whose_path_does_not_pace_is_not_offered_and_only_so_many_heights_are_tried(robocasa):
    session, state, calls = robocasa
    state["lifted"] = {height: _admitted() for height in state["heights"]}

    def refuse(_plan):
        raise PaceRefusal("too long", code="paced_motion_budget")

    state["paced"] = refuse
    plan = session.arm_path_plan(TARGET, "precision")
    assert plan["code"] == "target_unreachable_at_base_pose" and len(calls["arm"]) == 1 + C.TORSO_ADMIT_HEIGHTS
    assert plan["refusal"]["base_suggestions"] == FOUND["suggestions"] and plan["refusal"]["base_search"] == {
        "asked": 3}
    assert plan["refusal"]["torso_admission"]["tried"] == [
        {"torso_height_m": height, "code": "paced_motion_budget", "stage": "pacing"}
        for height in state["heights"][:C.TORSO_ADMIT_HEIGHTS]]


def test_a_refused_lift_says_where_each_height_failed(robocasa):
    """The 2026-11-06 qualification's torso raise was refused with only the measured height's reason: the receipt
    now carries what the lift's admission asked and why each height was refused."""
    session, state, calls = robocasa
    state["lifted"] = {height: _refused("path_in_collision_at_base_pose", robot_part="arm", path_fraction=0.48,
                                        penetration_m=0.012, scope="arm_and_torso")
                       for height in state["heights"]}
    plan = session.arm_path_plan(TARGET, "precision")
    admission = plan["refusal"]["torso_admission"]
    assert plan["code"] == "target_unreachable_at_base_pose" and plan["refusal"]["cause"] == "ik_unsolved"
    assert admission["measured_m"] == 0.1 and admission["reaching_m"] == state["heights"]
    assert admission["tried"] == [{"torso_height_m": height, "code": "path_in_collision_at_base_pose",
                                   "path_kind": "line", "robot_part": "arm", "path_fraction": 0.48,
                                   "penetration_m": 0.012} for height in state["heights"][:C.TORSO_ADMIT_HEIGHTS]]
    state["heights"] = []
    assert session.arm_path_plan(TARGET, "precision")["refusal"]["torso_admission"] == {
        "measured_m": 0.1, "reaching_m": [], "tried": []}, "no height reaches the target: nothing was tried"


def test_a_line_whose_target_the_arm_reaches_from_here_neither_lifts_nor_searches(robocasa):
    session, state, calls = robocasa
    state["first"] = _refused("target_unreachable_at_base_pose", cause="ik_unsolved",
                              endpoint=shared_motion.ENDPOINT_REACHABLE)
    plan = session.arm_path_plan(TARGET, "precision")
    assert plan is state["first"] and len(calls["arm"]) == 1 and calls["arm"][0] == (None, True)
    assert not calls["search"]


@pytest.mark.parametrize("code", ["target_in_collision_at_base_pose", "path_in_collision_at_base_pose"])
def test_a_refusal_a_base_move_may_answer_carries_the_base_poses_without_a_lift(robocasa, code):
    session, state, calls = robocasa
    state["first"] = _refused(code, robot_part="arm", path_fraction=0.4)
    plan = session.arm_path_plan(TARGET, "precision")
    assert plan["code"] == code and plan["refusal"]["robot_part"] == "arm"
    assert plan["refusal"]["base_suggestions"] == FOUND["suggestions"] and len(calls["arm"]) == 1


def test_the_arm_alone_is_asked_without_a_lift_or_a_search(robocasa):
    session, state, calls = robocasa
    plan = session.arm_path_plan(TARGET, "precision", allow_torso=False)
    assert plan is state["first"] and calls["arm"] == [(None, False)] and not calls["search"]


def test_an_unavailable_planner_or_a_search_with_no_answer_adds_nothing(robocasa, monkeypatch):
    session, state, _calls = robocasa
    state["first"] = _refused(shared_motion.UNAVAILABLE)
    assert "base_suggestions" not in session.arm_path_plan(TARGET, "free_space")["refusal"]
    state["first"] = _refused("target_in_collision_at_base_pose")
    monkeypatch.setattr(shared_motion, "base_search", lambda _session, _pose, _move, _admits: None)
    assert "base_suggestions" not in session.arm_path_plan(TARGET, "precision")["refusal"]


def test_a_target_out_of_reach_says_how_far_and_where_to_stand(robocasa):
    session, _state, calls = robocasa
    far = [2.0, 0.0, 0.9, *IDENTITY]
    assert session.workspace_refusal(far) == "target_out_of_reach_from_base"
    detail = session.reach_detail(far, "target_out_of_reach_from_base", "contact_approach", {"preset": "normal"})
    assert detail["path_kind"] == "reach" and detail["reach_m"] == C.ARM_MAX_REACH_FROM_SHOULDER_M
    assert detail["distance_from_mount_m"] == pytest.approx(1.5) and detail["nearest_mount_height_m"] == 0.9
    assert detail["base_suggestions"] == FOUND["suggestions"] and calls["search"] == [far]
    assert calls["move"] == [("line", "contact_approach", {"preset": "normal"})], "the search asks the call's own move"
    assert session.reach_detail(far, "target_outside_kitchen_height", "free_space") == {}


def test_a_base_move_beyond_the_servo_s_bound_is_left_to_the_servo(robocasa, monkeypatch):
    session, _state, _calls = robocasa
    asked = []
    monkeypatch.setattr(shared_motion, "base_path_refusal", lambda _session, target: asked.append(target) or {"x": 1})
    assert session.base_path_refusal({"x_m": 0.5 + C.BASE_MAX_TRAVEL_M + 0.1, "y_m": 0.0, "yaw_rad": 0.0}) is None
    assert session.base_path_refusal({"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0}) == {"x": 1} and len(asked) == 1


def test_a_line_whose_gripper_enters_the_world_gets_no_search(robocasa):
    """What lies around the target stays where it is whatever the base does (smoke 11-06: Store [78], Blender [42])."""
    session, state, calls = robocasa
    state["first"] = _refused("path_in_collision_at_base_pose", robot_part="gripper", path_fraction=0.93)
    plan = session.arm_path_plan(TARGET, "precision")
    assert plan is state["first"] and "base_suggestions" not in plan["refusal"] and not calls["search"]


@pytest.mark.parametrize("interaction, kind", [("contact_approach", "line"), ("precision", "line"),
                                               ("free_space", "joint")])
def test_a_stand_is_judged_by_the_move_s_own_admission(robocasa, interaction, kind):
    """The search's stands are asked the refused call's own question there: its line or route, then its pacing at its
    speed, none of it kept for a dry run (smoke 11-06: a contact line refused at a suggested stand on the branch jump
    it had here, because only a joint route had been asked)."""
    session, state, calls = robocasa
    state["first"] = _refused("path_in_collision_at_base_pose", robot_part="arm", path_fraction=0.4)
    session.arm_path_plan(TARGET, interaction, speed={"preset": "careful"})
    assert calls["move"] == [(kind, interaction, {"preset": "careful"})]
    stand = shared_motion.Stand(1.0, 0.0, 0.0, 0.1)
    admits = calls["admits"][0]
    state["lifted"] = {0.1: _admitted()}
    calls["paced"].clear()
    assert admits(stand) is True and calls["arm"][-1] == (stand, False) and calls["kept"][-1] is False
    assert calls["paced"] == [{"preset": "careful"}], "paced at the call's speed"

    def refuse(_plan):
        raise PaceRefusal("too long", code="paced_motion_budget")

    state["paced"] = refuse
    assert admits(stand) is False, "a path its pacing refuses is no stand"
    state["lifted"] = {}
    assert admits(stand) is False, "nor one whose path is refused"
