"""The internal reach check and grasp_plan's pooled IK selection, on both dual-arm backends.

RoboDojo g003 recorded 13 of 24 grasp_execute calls refused at the precommit
check. grasp_plan now checks its whole compiled pool with each backend's
existing read-only IK -- on RoboDojo the SDK IK on the move_pose route the
interaction selects, the approach chained onto its pregrasp -- before the
display cap. The check is internal: by decision (2026-09-29) no ToolSurface
declares ``reach_check`` as a policy tool, and these tests hold that too.

Everything here is offline: the RoboDojo session is the fake SDK plant, the
RoboTwin adapter gets a toy forward-kinematics model, and the tool runtime is
driven through its real dispatcher and result-schema admission.
"""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.agent.surface_catalog import surface_of
from tests.support.agent.reach_stacks import policy_reach
from ora.agent.error_classes import ToolError
from ora.contracts.alignment import REQUEST_SCHEMA as ALIGN_REQUEST_SCHEMA
from ora.contracts.reach_check import (
    CLAIM_BOUNDARY, REQUEST_SCHEMA, RESULT_SCHEMA, ReachCheckRequestDrift,
)
from ora.contracts.refusal import EnvironmentRefusal
from ora.environment.controllers.alignment import plan_alignment
from ora.environment.controllers.reach_check import check_reach
from tests.support.agent.reach_check import planner_ik, ARRIVAL, _robotwin_adapter, _dojo_runtime, _stub_bridge

LIVE = (
    "robotwin_toolchain_b2_closed_noservo_v2",
)
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _query(pose, interaction="free_space", from_index=None):
    return {"pose_xyzw": list(pose), "interaction": interaction, "from_index": from_index}


def _request(queries, *, arm="left"):
    return {"schema": REQUEST_SCHEMA, "arm": arm, "queries": [dict(item) for item in queries]}


# --------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------

@pytest.mark.parametrize("profile", LIVE)
def test_the_live_surfaces_select_the_pooled_ik_check_without_declaring_reach_check(profile):
    from ora.agent.composition.surface.profile_catalog import PROFILE_CATALOG

    surface = surface_of(profile)
    names = [item["name"] for item in surface.definitions]
    assert "reach_check" not in names and "reach_check" not in surface.result_schemas
    assert not any("reach_check" in capability for capability in surface.capabilities)
    assert surface.supports(policy_reach.GRASP_ANNOTATION_CAPABILITY)
    manifest = json.loads(PROFILE_CATALOG[profile].read_text())
    assert "reach_check" not in manifest["tools"]  # file-backed: it lists only its tools
    grasp_help = next(entry["description"] for entry in surface.definitions if entry["name"] == "grasp_plan")
    assert "IK reference at planning time, not a guarantee" in grasp_help
    assert "pool of up to 32" in grasp_help
    for definition in surface.definitions:  # help never names the undeclared tool
        assert "reach_check" not in definition["description"], definition["name"]


def test_the_annotation_reason_set_is_the_backend_check_s_own_set():
    from ora.contracts.reach_check import ROBODOJO_REASONS, SHARED_REASONS
    from ora.contracts.robotwin_reach import REASONS

    for profile in LIVE:
        surface = surface_of(profile)
        stage = surface.result_schemas["grasp_plan"]["$defs"]["candidate"]["properties"][
            "ik_reachability"]["properties"]["pregrasp"]["properties"]["reason"]
        backend = ROBODOJO_REASONS if profile.startswith("robodojo") else REASONS
        assert stage["enum"] == [*SHARED_REASONS, *backend, None]


# --------------------------------------------------------------------------
# Shared controller
# --------------------------------------------------------------------------

class _Adapter:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows

    def reach_check_batch(self, arm, queries, **tolerances):
        self.calls.append((arm, copy.deepcopy(queries), tolerances))
        rows = self.rows or [{"reachable": True, "reason": None, "route": None, "position_error_m": 0.001,
                              "orientation_error_rad": 0.002, "joint_limit_margin_rad": 0.5,
                              "joint_delta_rad": 0.1} for _ in queries]
        return {"ik_model": "stub_ik", "matches_move_pose_admission": False,
                "claim_boundary": "stub", "results": rows}


def test_the_bridge_routes_reach_checks_and_leaves_alignment_requests_alone():
    adapter = _Adapter()
    answer = plan_alignment(
        _request([_query([0.1, 0, 0.9, *IDENTITY]), _query([0.1, 0, 0.85, *IDENTITY], "contact_approach", 0)]),
        adapter=adapter, control_profile=ARRIVAL, available_arms=("left", "right"))
    assert answer["schema"] == RESULT_SCHEMA and answer["motion_planner_called"] is False
    assert answer["matches_move_pose_admission"] is False
    assert adapter.calls[0][1][1]["from_index"] == 0
    assert adapter.calls[0][2] == {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05}
    align = SimpleNamespace(solve_batch_reachability=lambda arm, poses, **kw: [
        {"reachable": True, "position_error_m": 0.0, "orientation_error_rad": 0.0,
         "iterations": 1, "claim_boundary": "IK"} for _ in poses])
    aligned = plan_alignment(
        {"schema": ALIGN_REQUEST_SCHEMA, "target_ref": "t", "approach_side": "top", "arm": "left",
         "target_poses_xyzw": [[0.1, 0, 0.9, *IDENTITY]]},
        adapter=align, control_profile=ARRIVAL, available_arms=("left", "right"))
    assert aligned["schema"] == "ora.align-plan-result.v1"


@pytest.mark.parametrize(("request_", "field"), [
    (_request([_query([0, 0, 1, *IDENTITY])], arm="middle"), "arm"),
    (_request([_query([0, 0, 1, *IDENTITY])] * 17), "queries"),
    (_request([_query([0, 0, 1, *IDENTITY], "sprint")]), "queries[0].interaction"),
    (_request([_query([0, 0, 1, *IDENTITY], from_index=0)]), "queries[0].from_index"),
])
def test_the_controller_refuses_correctable_input(request_, field):
    with pytest.raises(EnvironmentRefusal) as refused:
        check_reach(request_, adapter=_Adapter(), control_profile=ARRIVAL, available_arms=("left", "right"))
    assert refused.value.field == field


def test_the_controller_rejects_backend_drift():
    with pytest.raises(ReachCheckRequestDrift):
        check_reach({**_request([_query([0, 0, 1, *IDENTITY])]), "extra": 1}, adapter=_Adapter(),
                    control_profile=ARRIVAL, available_arms=("left",))
    bad = _Adapter(rows=[{"reachable": False, "reason": "made_up", "route": None, "position_error_m": None,
                          "orientation_error_rad": None, "joint_limit_margin_rad": None,
                          "joint_delta_rad": None}])
    with pytest.raises(ReachCheckRequestDrift, match="undeclared"):
        check_reach(_request([_query([0, 0, 1, *IDENTITY])]), adapter=bad, control_profile=ARRIVAL,
                    available_arms=("left",))


# --------------------------------------------------------------------------
# RoboTwin: the IK a global move_pose route plans through
# --------------------------------------------------------------------------


def test_robotwin_answers_chains_and_names_its_claim(monkeypatch):
    from ora.environment.backends.robotwin import session as robotwin_session

    limits = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.2], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]]
    adapter = _robotwin_adapter(limits)
    monkeypatch.setattr(robotwin_session, "RoboTwinMotionBackendAdapter", lambda **kwargs: adapter)
    # Since the RoboTwin session split every motion mixin builds its adapter
    # through ``_motion_backend_adapter``, so the stand-in session carries it.
    fake = SimpleNamespace(_require_environment=lambda: object(), control_profile=ARRIVAL,
                           _motion_backend_adapter=lambda **kwargs: adapter)
    answer = robotwin_session.RoboTwinLiveSession.align_plan(fake, _request([
        _query([0.1, 0.05, 0.95, *IDENTITY]),                                # inside the toy limits
        _query([0.1, 0.05, 1.5, *IDENTITY]),                                 # z beyond the joint stop
        _query([0.12, 0.05, 0.95, *IDENTITY], "contact_approach", 0),       # chained onto 0
        _query([0.1, 0.05, 1.0, *IDENTITY], "contact_approach", 1),         # chained onto a refusal
    ]))
    assert answer["ik_model"] == "robotwin_reach_ik_depth_world_v1"
    # The admission also plans a trajectory and validates it, whatever the interaction.
    assert answer["matches_move_pose_admission"] is False
    reachable, blocked, chained, orphan = answer["results"]
    assert reachable["reachable"] is True and reachable["route"] is None
    assert reachable["position_error_m"] <= 0.005
    assert reachable["joint_limit_margin_rad"] == pytest.approx(0.25, abs=1e-3)
    assert reachable["joint_delta_rad"] == pytest.approx(0.1, abs=1e-3)
    assert blocked["reachable"] is False and blocked["reason"] == "ik_failed"
    assert blocked["position_error_m"] is None and blocked["joint_limit_margin_rad"] is None
    assert chained["reachable"] is True
    assert chained["joint_delta_rad"] == pytest.approx(0.02, abs=1e-3)  # from the chained start
    assert orphan == {**orphan, "reachable": False, "reason": "start_unreachable", "position_error_m": None}
    assert "It plans no trajectory" in answer["claim_boundary"]
    free_space = robotwin_session.RoboTwinLiveSession.align_plan(fake, _request([_query([0.1, 0.05, 0.95, *IDENTITY])]))
    assert free_space["matches_move_pose_admission"] is False


# --------------------------------------------------------------------------
# RoboDojo: the shared planner's IK, in the world a plan would see
# --------------------------------------------------------------------------


def _dojo(dojo, *queries):
    answer = dojo.align_plan(_request(queries))
    assert answer["schema"] == RESULT_SCHEMA and answer["arm"] == "left"
    assert answer["ik_model"] == "robodojo_reach_ik_depth_world_v1"
    assert answer["matches_move_pose_admission"] is False
    return answer["results"]


def test_robodojo_answers_from_the_planners_ik_without_writing(dojo, monkeypatch):
    requests = planner_ik(dojo, monkeypatch)
    [row] = _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]))
    assert row["reachable"] is True and row["reason"] is None and row["route"] is None
    assert row["position_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["joint_limit_margin_rad"] == pytest.approx(3.14 - 0.8)
    assert row["joint_delta_rad"] == pytest.approx(0.02)
    assert dojo.environment.commands == [] and dojo.sim_steps == 0
    assert requests[0]["world"]["input"] == {"path": "world.npz"} and requests[0]["held_object"] == {"source": "world"}


def test_robodojo_gives_every_interaction_the_same_answer(dojo, monkeypatch):
    planner_ik(dojo, monkeypatch)
    pose = [0.2, 0.0, 0.8, *IDENTITY]
    rows = _dojo(dojo, _query(pose, "free_space"), _query(pose, "precision"), _query(pose, "contact_approach"))
    assert [row["reachable"] for row in rows] == [True] * 3 and len({row["joint_delta_rad"] for row in rows}) == 1


def test_robodojo_chains_a_query_onto_the_earlier_solution(dojo, monkeypatch):
    requests = planner_ik(dojo, monkeypatch)
    pregrasp, approach = [0.3, 0.0, 0.8, *IDENTITY], [0.3, 0.0, 0.76, *IDENTITY]
    rows = _dojo(dojo, _query(pregrasp), _query(approach, "contact_approach", 0))
    assert [query["from_index"] for query in requests[0]["queries"]] == [None, 0]
    assert rows[1]["reachable"] is True and rows[1]["joint_delta_rad"] == pytest.approx(0.04)
    planner_ik(dojo, monkeypatch, refuse=lambda pose: "unsolved")
    rows = _dojo(dojo, _query(pregrasp), _query(approach, "contact_approach", 0))
    assert [row["reason"] for row in rows] == ["ik_failed", "start_unreachable"]


def test_robodojo_names_what_the_planner_refused_and_a_pose_it_missed(dojo, monkeypatch):
    planner_ik(dojo, monkeypatch, refuse=lambda pose: "goal_below_support")
    assert _dojo(dojo, _query([0.02, 0.0, 0.2, *IDENTITY]))[0]["reason"] == "goal_below_support"
    from ora.environment.backends.robodojo import reach_check as dojo_reach, reachability

    planner_ik(dojo, monkeypatch)
    monkeypatch.setattr(dojo_reach, "fk_calibrated", lambda *a: True)
    monkeypatch.setattr(reachability, "fk_link6", lambda session, arm, joints: [0.5, 0.0, 0.8, *IDENTITY])
    [missed] = _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]))
    assert missed["reason"] == "fk_residual_exceeded" and missed["position_error_m"] == pytest.approx(0.48)


@pytest.mark.parametrize("reason,arrange", [
    ("no_planner", lambda dojo, shared: setattr(dojo, "motion_planner", None)),
    ("stale_observation", lambda dojo, shared: setattr(shared, "world_input", lambda session, arm: (None, []))),
    ("planner_error", lambda dojo, shared: setattr(dojo.motion_planner, "solve_ik", lambda request: {
        "status": "error", "reason": "world_build_failed"})),
])
def test_robodojo_says_why_it_has_no_answer(dojo, monkeypatch, reason, arrange):
    from ora.environment.backends.robodojo import shared_motion

    planner_ik(dojo, monkeypatch)
    with monkeypatch.context() as local:
        local.setattr(shared_motion, "_evidence", lambda *arguments: None)
        arrange(dojo, shared_motion)
        rows = _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]), _query([0.03, 0.0, 0.8, *IDENTITY]))
    assert [row["reason"] for row in rows] == [reason, reason] and not any(row["reachable"] for row in rows)


def test_an_unsettled_measured_frame_is_a_typed_answer_not_an_internal_fault(dojo, monkeypatch):
    from ora.environment.backends.robodojo import reach_check as dojo_reach

    requests = planner_ik(dojo, monkeypatch)
    monkeypatch.setattr(dojo_reach, "fk_calibrated", lambda *a: False)
    rows = _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]), _query([0.03, 0.0, 0.8, *IDENTITY]))
    assert {row["reason"] for row in rows} == {"fk_calibration_mismatch"} and requests == []


def _align(dojo, *poses):
    return dojo.align_plan({"schema": ALIGN_REQUEST_SCHEMA, "target_ref": "target", "approach_side": "top",
                            "arm": "left", "target_poses_xyzw": [list(pose) for pose in poses]})


def test_robodojo_align_plan_answers_from_the_planners_ik(dojo, monkeypatch):
    requests = planner_ik(dojo, monkeypatch, refuse=lambda pose: "goal_below_support" if pose[2] < 0.5 else None)
    answer = _align(dojo, [0.02, 0.0, 0.8, *IDENTITY], [0.02, 0.0, 0.2, *IDENTITY])
    assert answer["motion_planner_called"] is False
    assert [query["from_index"] for query in requests[0]["queries"]] == [None, None]
    held, low = answer["candidates"]
    assert held["reachable"] is True and held["rejection_reason"] is None and held["iterations"] == 1
    assert held["position_error_m"] == pytest.approx(0.0, abs=1e-9) and held["claim_boundary"] == CLAIM_BOUNDARY
    assert low["reachable"] is False and low["rejection_reason"] == "goal_below_support"
    assert dojo.environment.commands == [] and dojo.sim_steps == 0


def test_robodojo_align_plan_with_no_planner_answers_no_pose(dojo, monkeypatch):
    """A runtime that names no planner gets no answer, typed, rather than another model's; no IK runs."""
    from ora.environment.backends.robodojo import ik

    monkeypatch.setattr(ik, "sdk_candidates", lambda *arguments, **keywords: pytest.fail("the SDK's IK ran"))
    held, low = _align(dojo, [0.02, 0.0, 0.8, *IDENTITY], [0.02, 0.0, 0.2, *IDENTITY])["candidates"]
    assert [held["reachable"], low["reachable"]] == [False, False]
    assert {held["rejection_reason"], low["rejection_reason"]} == {"no_planner"}
    assert held["position_error_m"] is None and held["claim_boundary"] == CLAIM_BOUNDARY
    assert dojo.environment.commands == [] and dojo.sim_steps == 0


# --------------------------------------------------------------------------
# Policy edge, through the real dispatcher and result admission
# --------------------------------------------------------------------------


def test_the_policy_cannot_call_the_undeclared_tool(tmp_path):
    runtime = _dojo_runtime(tmp_path)
    runtime.align_plan_callback = _stub_bridge([])
    with pytest.raises(ToolError) as refused:
        runtime.invoke("reach_check", {"arm": "left", "poses": [
            {"position_m": [0.1, 0.0, 0.8], "orientation_xyzw": IDENTITY}]})
    assert refused.value.code == "tool_unavailable"


def test_the_check_answers_in_order_and_never_asks_ik_outside_the_workspace(tmp_path):
    runtime = _dojo_runtime(tmp_path)
    requests = []
    runtime.align_plan_callback = _stub_bridge(requests, reachable=lambda query: query["pose_xyzw"][0] < 0.3)
    outside = [runtime.workspace_bounds_m["x"][1] + 1.0, 0.0, 0.8]
    result = policy_reach.check_queries("right", [
        _query([0.1, 0.0, 0.8, *IDENTITY]),
        _query([*outside, *IDENTITY]),
        _query([0.35, 0.0, 0.8, *IDENTITY], "precision"),
        _query([0.1, 0.0, 0.76, *IDENTITY], "contact_approach", 1),
        _query([0.1, 0.0, 0.76, *IDENTITY], "contact_approach", 2),
    ], bridge=runtime.align_plan_callback, workspace_bounds_m=runtime.workspace_bounds_m)
    assert [row["index"] for row in result["results"]] == [0, 1, 2, 3, 4]
    assert result["reachable_indices"] == [0]
    assert result["matches_move_pose_admission"] is True
    assert result["results"][0]["position_error_m"] == 0.00123
    assert [row["reason"] for row in result["results"]] == [
        None, "outside_workspace", "ik_failed", "start_unreachable", "start_unreachable"]
    [sent] = requests
    assert len(sent["queries"]) == 3  # 0, 2 and 4; 1 is outside and 3 is chained onto it
    assert sent["queries"][1]["interaction"] == "precision"
    assert sent["queries"][2]["from_index"] == 1  # renumbered from 2
    assert not (runtime.control_root / "submitted-action.json").exists()


# --------------------------------------------------------------------------
# grasp_plan: check the whole pool before the cap, order, never drop
# --------------------------------------------------------------------------


def test_the_jaw_flip_is_the_precommit_s_rotation_about_local_x():
    from ora.environment.backends.robodojo.sparse_motion import (
        JAW_SYMMETRY_AXIS, rotate_about_local_axis,
    )
    from ora.agent.toolkit.host.reach.alignment import frame_local_axes

    orientation = [0.1, 0.2, 0.3, math.sqrt(1 - 0.14)]
    mine = policy_reach._member(orientation, "jaw_flip", frame_local_axes("arx_x5_link6")["approach"])
    theirs = rotate_about_local_axis(orientation, JAW_SYMMETRY_AXIS, math.pi)
    assert abs(float(np.dot(mine, theirs))) == pytest.approx(1.0)
