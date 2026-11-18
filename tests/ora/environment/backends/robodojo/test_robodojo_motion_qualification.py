"""Deterministic layer of the RoboDojo motion qualification runner."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from ora.contracts.action import validate_runtime_action
from ora.contracts.execution_state import FIELDS, pack
from ora.contracts.joint_trajectory import native_action_cap
from ora.environment.backends.robodojo import session as session_module
from ora.environment.backends.robodojo.gripper_ramp import GRIPPER_RAMP_EXECUTOR
from ora.environment.backends.robodojo.motion_qualification import (
    ALL_CASE_IDS,
    MOTION_CASES,
    SCHEMA,
    SEQUENCE_CASES,
    UNREACHABLE_POSITION,
    X5_BASE_MOUNTS_M,
    X5_LINK6_REACH_BOUND_M,
    build_case_knots,
    build_case_request,
    build_qualification_record,
    judge_gripper_case,
    judge_motion_case,
    judge_recovery_case,
    judge_refusal,
    run_cases,
    summarize_receipt,
)
from ora.environment.backends.robodojo.motion_checks import REFUSED, knot_arguments
from ora.environment.backends.robodojo.common import workspace_violation
from ora.environment.backends.robodojo.session import RobodojoSession
from ora.environment.backends.robodojo.trajectory import TRAJECTORY_EXECUTOR, TRAJECTORY_STRATEGY
from ora.environment.workspace_profiles import (
    ROBODOJO_ARX_X5_TABLETOP_V1,
    resolve_workspace_profile,
)
from ora.orchestrator.backends.robodojo.qualification import MatrixPlanner
from tests.support.robodojo.backend import Plant

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")


START = {"left": [-0.3, -0.35, 0.92, 0.0, 0.0, 0.707107, 0.707107],
         "right": [0.3, -0.35, 0.92, 0.0, 0.0, 0.707107, 0.707107]}
JOINTS = [0.1, 1.0, 1.0, 0.0, 0.0, 0.0]
RECEIPT_SCHEMA = "ora.robodojo-sparse-motion-receipt.v1"
CASES = {case["case_id"]: case for case in MOTION_CASES}


def _receipt(case: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    """A truthful synthetic trajectory receipt for one case's expected happy path."""
    status = sorted(case["expected_status"] - {REFUSED})[0]
    reached = status == "reached"
    arms = {}
    for arm in ("left", "right"):
        targeted = arm in targets
        arms[arm] = {
            "targeted": targeted,
            "pose_error": {"translation_m": 0.002, "orientation_rad": 0.01} if targeted and reached else None,
            "inactive_arm_drift": None if targeted else {"translation_m": 0.0, "orientation_rad": 0.0},
        }
    after = {arm: {"pose_xyz_xyzw": list(targets[arm]) if arm in targets and reached else list(START[arm])}
             for arm in ("left", "right")}
    return {
        "schema": RECEIPT_SCHEMA, "action": "follow_joint_trajectory", "execution_status": status,
        "motion_strategy": TRAJECTORY_STRATEGY if reached else None, "failure_stage": None,
        "failure_code": case.get("expected_failure_code"),
        "executor": [TRAJECTORY_EXECUTOR] if reached else None,
        "state_changed": reached, "native_actions": 2 if reached else 0,
        "arms": arms, "before": {arm: {"pose_xyz_xyzw": list(START[arm])} for arm in START}, "after": after,
        "command_invariant": {"verified": True},
    }


def test_planned_cases_are_moves_and_knots_cases_are_valid_trajectories() -> None:
    for case in MOTION_CASES:
        if case.get("knots"):
            action = validate_runtime_action({
                "schema": "ora-motion-action.v3", "observation_id": "observation-0",
                "action_id": "action-0", "bound_observation_sha256": "0" * 64,
                "action": "follow_joint_trajectory", "arguments": knot_arguments(
                    build_case_knots(case["case_id"], JOINTS)),
            })
            assert set(action["arguments"]["arms"]) == {"left"}
            continue
        request = build_case_request(str(case["case_id"]), START)
        expected_arms = {"left", "right"} if case.get("dual_arm") else {"left"}
        assert request["kind"] == "move" and set(request["targets"]) == set(request["interactions"]) == expected_arms
        assert all(len(pose) == 7 for pose in request["targets"].values())
    with pytest.raises(ValueError, match="unknown qualification case"):
        build_case_request("nope", START)
    with pytest.raises(ValueError, match="unknown qualification case"):
        build_case_knots("nope", JOINTS)
    assert set(ALL_CASE_IDS) == set(CASES) | set(SEQUENCE_CASES)
    # The plan_ref cases left with the route that minted plan references.
    assert not {"probe_then_plan_ref_execution", "plan_ref_survives_gripper_only",
                "plan_ref_expires_after_motion", "free_space_lift_translate_descend"} & set(ALL_CASE_IDS)


def test_the_admission_knots_break_exactly_the_rule_they_name() -> None:
    from ora.contracts.joint_trajectory import MAX_KNOT_STEP_RAD

    step = build_case_knots("trajectory_step_exceeds_limit", JOINTS)
    assert step[0] == JOINTS and step[1][0] - JOINTS[0] == pytest.approx(1.5 * MAX_KNOT_STEP_RAD)
    limit = build_case_knots("trajectory_joint_limit", JOINTS)
    assert len(limit) == 1 and limit[0][0] > 10.0 and limit[0][1:] == JOINTS[1:]


def test_judges_flag_drift_route_executor_residual_and_inactive_arm_motion() -> None:
    near = CASES["orientation_only_near_180deg"]
    request = build_case_request("orientation_only_near_180deg", START)
    drifted = _receipt(near, request["targets"])
    drifted["after"]["left"]["pose_xyz_xyzw"][:3] = [-0.3, -0.35, 0.98]
    judgement = judge_motion_case(near, drifted, targets=request["targets"], route="global")
    assert judgement["passed"] is False
    assert any("translation drift" in note for note in judgement["notes"])
    precise = CASES["precision_small_local_6mm"]
    request = build_case_request("precision_small_local_6mm", START)
    wrong = _receipt(precise, request["targets"])
    assert judge_motion_case(precise, wrong, targets=request["targets"], route="line")["passed"] is True
    wrong["executor"] = ["robodojo_sdk_ik_joint_interpolation"]
    wrong["motion_strategy"] = "global"
    notes = judge_motion_case(precise, wrong, targets=request["targets"], route="global")["notes"]
    assert any("executor" in note for note in notes) and any("strategy" in note for note in notes)
    assert any("route global is not line" in note for note in notes)
    translation = CASES["free_space_translation_8cm"]
    request = build_case_request("free_space_translation_8cm", START)
    moved = _receipt(translation, request["targets"])
    # The free route is the run's: the planner's path where one is bound, else the joint line.
    assert judge_motion_case(translation, moved, targets=request["targets"], route="joint",
                             free_route="joint")["passed"] is True
    assert any("route joint is not global" in note for note in judge_motion_case(
        translation, moved, targets=request["targets"], route="joint")["notes"])
    moved["arms"]["right"]["inactive_arm_drift"] = {"translation_m": 0.02, "orientation_rad": 0.0}
    assert any("while not targeted" in note for note in judge_motion_case(
        translation, moved, targets=request["targets"], route="global")["notes"])
    short = _receipt(translation, request["targets"])
    short["after"]["left"]["pose_xyz_xyzw"][2] -= 0.03
    assert any("from its requested target" in note for note in judge_motion_case(
        translation, short, targets=request["targets"], route="global")["notes"])
    refusal = CASES["trajectory_step_exceeds_limit"]
    wrote = _receipt(refusal, {"left": START["left"]})
    assert judge_motion_case(refusal, wrote)["passed"] is True
    wrote["native_actions"] = 1
    assert any("wrote the world" in note for note in judge_motion_case(refusal, wrote)["notes"])
    assert judge_motion_case(refusal, {"schema": "other"})["passed"] is False


def test_a_refusal_passes_only_where_the_case_expects_it_and_for_the_reason_it_names() -> None:
    unreachable = CASES["planner_refusal_unreachable"]
    assert judge_refusal(unreachable, {"code": "goal_unreachable"})["passed"] is True
    assert judge_refusal(unreachable, {"code": "paced_path_unreachable"}, free_route="joint")["passed"] is True
    assert judge_refusal(unreachable, {"code": "path_blocked"})["notes"] == [
        "refusal code 'path_blocked' is not one of ['goal_unreachable']"]
    assert judge_refusal(CASES["orientation_only_90deg"], {"code": "paced_path_wrist_unwind"})["passed"] is True
    assert "refused before any action" in judge_refusal(
        CASES["free_space_translation_8cm"], {"code": "path_blocked", "message": "blocked"})["notes"][0]


def test_gripper_and_recovery_judges_read_the_trajectory_route_receipts() -> None:
    gripper = {"schema": RECEIPT_SCHEMA, "execution_status": "reached", "executor": [GRIPPER_RAMP_EXECUTOR],
               "arms": {"left": {"gripper_error": 0.01}}, "command_invariant": {"verified": True},
               "pose_targets_before": {}, "pose_targets_after": {}}
    assert judge_gripper_case("close", gripper)["passed"] is True
    assert judge_gripper_case("close", {**gripper, "arms": {"left": {"gripper_error": 0.1}}})["notes"] == [
        "gripper residual 0.1"]
    assert judge_gripper_case("close", {**gripper, "executor": ["robodojo_sdk_gripper_actuator"]})["passed"] is False
    recovered = {"schema": RECEIPT_SCHEMA, "execution_status": "reached", "executor": [TRAJECTORY_EXECUTOR],
                 "trajectory": {"recovery": {"targets": {"left": "home"}, "previous_anchor": "kept"}},
                 "after": {"left": {"pose_xyz_xyzw": [0, 0, 0.8, 0, 0, 0, 1]}}}
    assert judge_recovery_case(recovered, [0, 0, 0.8, 0, 0, 0, 1], strategy="home")["passed"] is True
    assert judge_recovery_case(recovered, [0, 0, 0.8, 0, 0, 0, 1], strategy="retrace")["passed"] is False
    assert judge_recovery_case(recovered, [0, 0, 0.9, 0, 0, 0, 1], strategy="home")["passed"] is False


def test_the_refusal_target_is_in_envelope_and_out_of_reach() -> None:
    """The unreachable cases must refuse for reach, not at the envelope.

    ``probe_motion`` raises on an out-of-envelope query, and move_pose refuses a
    target outside the live ``robodojo_arx_x5_tabletop_v1`` box before it plans,
    so such a target would never ask the planner anything. Pin both halves:
    inside the envelope, past the X5's reach.
    """

    bounds = resolve_workspace_profile(ROBODOJO_ARX_X5_TABLETOP_V1)["bounds_m"]
    assert workspace_violation(UNREACHABLE_POSITION, bounds) is None
    for axis, index in (("x", 0), ("y", 1), ("z", 2)):
        low, high = bounds[axis]
        assert min(UNREACHABLE_POSITION[index] - low, high - UNREACHABLE_POSITION[index]) >= 0.005
    for base in X5_BASE_MOUNTS_M:
        assert math.dist(UNREACHABLE_POSITION, base) > 1.5 * X5_LINK6_REACH_BOUND_M


def _source_manifest(tmp_path: Path) -> dict[str, Any]:
    """A manifest whose layout file exists, as the round-isolation audit needs."""

    layout = tmp_path / "layout" / "hang_mugs_0.json"
    layout.parent.mkdir(parents=True, exist_ok=True)
    if not layout.is_file():
        layout.write_text('{"objects": []}', encoding="utf-8")
    return {
        "schema": "ora.robodojo-source-manifest.v1",
        "source_revisions": {"robodojo": "1" * 40},
        "layout_file": {
            "path": str(layout),
            "sha256": hashlib.sha256(layout.read_bytes()).hexdigest(),
        },
    }


def test_run_cases_completes_the_full_matrix_against_the_fake_plant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole gate plans with the host's planner and runs every move as a trajectory over the fake SDK.

    The fake plant has no shared planner, so free-space moves take the joint route; the card run's record
    names the planner's (``free_route``)."""

    limit = 400
    monkeypatch.setitem(session_module.NATIVE_ACTION_LIMITS, "arrange_largest_number", limit)
    plant = Plant()
    plant.step_lim = limit
    # Drive the matrix under the *live* action envelope the resolver
    # hands the worker, from a home link6 pose that sits inside it: the gate
    # once passed 24/24 only because the compatibility runtime's +/-1.5 m box
    # let the refusal cases aim outside any real workspace.
    bounds = resolve_workspace_profile(ROBODOJO_ARX_X5_TABLETOP_V1)["bounds_m"]
    plant.robot_manager.set_home([0.0, -0.35, 0.95, 0.0, 0.0, 0.0])
    # The fake SDK refuses exactly what the real arm cannot reach: a link6
    # target farther than the URDF chain length from both mounted bases.
    plant.robot_manager.unreachable = lambda target: all(
        math.dist([float(value) for value in target[:3]], base) > X5_LINK6_REACH_BOUND_M
        for base in X5_BASE_MOUNTS_M
    )
    session = RobodojoSession(
        config={"workspace_bounds_m": bounds, "action_codec": "motion_action_v3"},
        output_root=tmp_path / "environment",
        environment_factory=lambda config: (
            plant, SimpleNamespace(close=lambda: None), _source_manifest(tmp_path),
        ),
    )
    session._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    session.start()
    observation_root = tmp_path / "observations"

    def observe() -> Mapping[str, Any]:
        observation, _images = session.observe(observation_root)
        return {**observation, "observation_sha256": f"{session.observation_sequence:064d}"[:64]}

    def execute(action: Mapping[str, Any]) -> Mapping[str, Any]:
        # What the host's ExecutionStateOwner lends one action (limited_action).
        session.request_action_budget = native_action_cap(action)
        try:
            return session.execute(action)
        finally:
            del session.request_action_budget

    def restart_round(index: int) -> Mapping[str, Any]:
        verdict = session.output_root / "official-verdict.json"
        archive = session.output_root / "rounds" / f"round-{index - 1:04d}" / "official-verdict.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        verdict.replace(archive)
        return session.restart_round(index, seed=0)

    planner = MatrixPlanner(kinematics=session.align_plan, global_planner=False, workspace_bounds_m=bounds,
                            state=lambda: pack({name: getattr(session, name, {}) for name in FIELDS}))
    results = run_cases(
        observe=observe, execute=execute, plan=planner, probe=session.probe_motion, restart_round=restart_round,
        status=session.status, persist_verdict=session.persist_official_verdict,
        current_poses=lambda observation: {
            arm: state["pose_xyz_xyzw"] for arm, state in observation["proprioception"]["end_effectors"].items()
        },
        native_action_limit=limit, free_route=planner.free_route,
    )
    by_id = {result["case_id"]: result for result in results}
    assert set(by_id) == set(ALL_CASE_IDS)
    failed = {case_id: result["notes"] for case_id, result in by_id.items() if not result["passed"]}
    assert failed == {}
    executed = [result for result in results if result["receipt"] is not None
                and result["receipt"]["action"] == "follow_joint_trajectory"]
    assert all(result["receipt"]["schema"] == RECEIPT_SCHEMA for result in executed)
    assert {by_id[case_id]["route"] for case_id in ("free_space_translation_8cm", "dual_arm_synchronized_lift")} == {
        "joint"}
    assert by_id["precision_small_local_6mm"]["route"] == "line"
    assert by_id["contact_approach_staged_80mm"]["route"] == "staged"
    assert by_id["planner_refusal_unreachable"]["refusal"]["code"] == "paced_path_unreachable"
    for case_id in ("trajectory_step_exceeds_limit", "trajectory_joint_limit"):
        assert by_id[case_id]["receipt"]["native_actions"] == 0
        assert by_id[case_id]["receipt"]["trajectory"]["admission"]["code"] == case_id
    assert by_id["recover_home_after_matrix"]["receipt"]["trajectory"]["recovery"]["targets"] == {
        "left": "home", "right": "home"}
    assert by_id["recover_retrace_after_move"]["receipt"]["trajectory"]["undo"]["arms"] == {"left": "pop"}
    assert by_id["gripper_close_holds_pose"]["receipt"]["executor"] == [GRIPPER_RAMP_EXECUTOR]
    assert by_id["round_reset_isolation"]["reset"]["round_index"] == 1
    assert by_id["official_terminal_boundary_intact"]["verdict"]["official_success"] is False
    assert by_id["official_terminal_boundary_intact"]["verdict"]["native_actions"] == limit
    assert by_id["official_terminal_boundary_intact"]["receipt"]["round_index"] == 0
    assert by_id["round_reset_isolation"]["receipt"]["round_index"] == 1
    assert session.round_index == 1 and not session.is_terminal
    assert (session.output_root / "rounds/round-0000/official-verdict.json").exists()
    assert (session.output_root / "round-resets/round-0001.json").exists()
    # Nothing in the gate reaches the SDK executor's actions any more.
    actions = {json.loads(path.read_text())["action"] for path in (session.output_root / "receipts").glob("*.json")}
    assert actions == {"follow_joint_trajectory", "set_gripper", "wait"}

    record = build_qualification_record(
        source_commit="a" * 40, host={"execution_host": "test", "cuda_visible_devices": "3"},
        session_identity={"task": "arrange_largest_number", "rounds": 2}, case_results=results,
        evidence={"environment_root": str(session.output_root)},
    )
    assert record["schema"] == SCHEMA and record["passed"] is True and record["missing_cases"] == []
    assert len(record["record_sha256"]) == 64
    json.dumps(record, allow_nan=False)
    partial = build_qualification_record(source_commit="a" * 40, host={}, session_identity={}, case_results=results[:3])
    assert partial["passed"] is False and partial["missing_cases"]
    summary = summarize_receipt(session.last_receipt["receipt"])
    assert "control_trace" not in json.dumps(summary) and summary["schema"] == RECEIPT_SCHEMA


def test_run_cases_records_boundary_errors_instead_of_dropping_cases() -> None:
    calls = {"observe": 0}

    def observe() -> Mapping[str, Any]:
        calls["observe"] += 1
        return {"observation_id": f"o{calls['observe']}", "observation_sha256": "0" * 64,
                "proprioception": {"end_effectors": {arm: {"pose_xyz_xyzw": list(pose), "arm_joint_qpos": JOINTS}
                                                     for arm, pose in START.items()}}}

    def execute(action: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError("worker died")

    def plan(observation: Mapping[str, Any], request: Mapping[str, Any], action_id: str) -> Mapping[str, Any]:
        return {"action": {"action_id": action_id}, "route": "global"}

    results = run_cases(
        observe=observe, execute=execute, plan=plan,
        probe=lambda request: (_ for _ in ()).throw(RuntimeError("no probe")),
        restart_round=lambda index: (_ for _ in ()).throw(RuntimeError("no reset")),
        status=lambda: {"terminal": False}, persist_verdict=lambda: {"terminal": False, "official_success": None},
        current_poses=lambda observation: {
            arm: state["pose_xyz_xyzw"] for arm, state in observation["proprioception"]["end_effectors"].items()
        },
        native_action_limit=3, free_route="global",
    )
    by_id = {result["case_id"]: result for result in results}
    assert set(by_id) == set(ALL_CASE_IDS)
    assert all(result["passed"] is False for case_id, result in by_id.items()
               if case_id != "official_preterminal_boundary")
    assert by_id["free_space_translation_8cm"]["notes"] == ["execution error: worker died"]
    assert by_id["trajectory_joint_limit"]["notes"] == ["execution error: worker died"]
    assert by_id["official_preterminal_boundary"]["passed"] is True
    assert build_qualification_record(source_commit="b" * 40, host={}, session_identity={},
                                      case_results=results)["passed"] is False
    with pytest.raises(ValueError, match="free_route"):
        run_cases(observe=observe, execute=execute, plan=plan, probe=None, restart_round=None, status=None,
                  persist_verdict=None, current_poses=None, native_action_limit=3, free_route="sdk")


def test_rotation_cases_compose_about_the_link6_approach_axis() -> None:
    request = build_case_request("orientation_only_90deg", {"left": [0, 0, 0.8, 0, 0, 0, 1], "right": START["right"]})
    orientation = request["targets"]["left"][3:]
    assert orientation == pytest.approx([math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)], abs=1e-6)
    assert request["interactions"] == {"left": "free_space"}


def test_legacy_host_reports_migration_before_starting_a_worker(tmp_path):
    from ora.environment.backends.robodojo.motion_qualification import main
    with pytest.raises(SystemExit, match="No worker was started"):
        main(["--runtime-config", str(tmp_path / "absent.json"), "--output", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()
