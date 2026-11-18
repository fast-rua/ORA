"""Deterministic layer of the LIBERO sparse-motion qualification runner."""

from __future__ import annotations

import math
from typing import Any, Mapping

import pytest

from ora.contracts.action import validate_runtime_action
from ora.orchestrator.backends.libero_pro.qualification import (
    ALL_CASE_IDS,
    MOTION_CASES,
    SCHEMA,
    _qualification_probe_row,
    build_case_arguments,
    build_qualification_record,
    judge_motion_case,
    run_cases,
)


START_POSE = [0.2, 0.0, 0.95, 0.0, 0.0, 0.0, 1.0]


def _receipt(case_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """A truthful synthetic receipt for one case's expected happy path."""

    case = next(item for item in MOTION_CASES if item["case_id"] == case_id)
    target = arguments.get("targets", {}).get("right", {})
    status = "reached" if "reached" in case["expected_status"] else sorted(case["expected_status"])[0]
    execution = {"steps_used": 0, "terminal_after_action": False}
    if case_id == "planner_refusal_unreachable":
        return {
            **execution,
            "schema": "ora.libero-sparse-motion-receipt.v1",
            "execution_status": "planner_failed",
            "failure_stage": "planner", "failure_code": "goal_unreachable",
            "executor": None, "state_changed": False,
            "arms": {"right": {"pose_error": None}},
            "before": {"right": {"pose_xyz_xyzw": list(START_POSE)}},
            "after": {"right": {"pose_xyz_xyzw": list(START_POSE)}},
            "segment": {"planner": {
                "status": "not_reachable", "reachable": False,
                "valid_query": True, "planner_status": "goal_unreachable",
                "canonical_artifact": {"path": "curobo/plans/action-0000.json", "sha256": "a" * 64},
            }},
        }
    after = list(target.get("position_m", START_POSE[:3])) + list(
        target.get("orientation_xyzw", START_POSE[3:])
    )
    return {
        **execution, "steps_used": 1,
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "execution_status": status,
        "failure_stage": None, "failure_code": None,
        "executor": (
            [case["expected_executor"]] if case.get("expected_executor")
            else ["libero_curobo_joint_position"]
        ),
        "state_changed": True,
        **({"motion_strategy": case["expected_strategy"]}
           if case.get("expected_strategy") else {}),
        "arms": {"right": {
            "pose_error": {"translation_m": 0.002, "orientation_rad": 0.01},
        }},
        "before": {"right": {"pose_xyz_xyzw": list(START_POSE)}},
        "after": {"right": {"pose_xyz_xyzw": after}},
    }


def test_case_arguments_are_valid_v3_actions() -> None:
    for case in MOTION_CASES:
        arguments = build_case_arguments(
            str(case["case_id"]),
            position=START_POSE[:3], orientation=START_POSE[3:],
        )
        action = validate_runtime_action({
            "schema": "ora-motion-action.v3",
            "observation_id": "observation-0",
            "action_id": "action-0",
            "bound_observation_sha256": "0" * 64,
            "action": "move_pose",
            "arguments": arguments,
        })
        assert set(action["arguments"]["targets"]) == {"right"}


def test_judge_flags_alias_drift_and_executor_mismatch() -> None:
    case = next(
        item for item in MOTION_CASES
        if item["case_id"] == "orientation_only_near_180deg"
    )
    drifted = {
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "execution_status": "reached",
        "executor": ["libero_curobo_joint_position"],
        "state_changed": True,
        "arms": {"right": {"pose_error": {
            "translation_m": 0.001, "orientation_rad": 0.01,
        }}},
        "before": {"right": {"pose_xyz_xyzw": list(START_POSE)}},
        "after": {"right": {"pose_xyz_xyzw": [
            0.2, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
        ]}},
    }
    judgement = judge_motion_case(case, drifted)
    assert judgement["passed"] is False
    assert any("2pi-alias" in note for note in judgement["notes"])

    precise = next(
        item for item in MOTION_CASES
        if item["case_id"] == "precision_small_local_6mm"
    )
    wrong_executor = _receipt(
        "precision_small_local_6mm",
        build_case_arguments(
            "precision_small_local_6mm",
            position=START_POSE[:3], orientation=START_POSE[3:],
        ),
    )
    wrong_executor["executor"] = ["libero_curobo_joint_position"]
    judgement = judge_motion_case(precise, wrong_executor)
    assert judgement["passed"] is False
    assert any("executor" in note for note in judgement["notes"])


def test_run_cases_completes_the_full_matrix_deterministically() -> None:
    observations = {"count": 0}
    world = {"pose": list(START_POSE)}

    def observe() -> Mapping[str, Any]:
        observations["count"] += 1
        return {
            "observation_id": f"observation-{observations['count']}",
            "observation_sha256": f"{observations['count']:064d}"[:64],
            "proprioception": {"end_effectors": {"right": {
                "pose_xyz_xyzw": list(world["pose"]),
            }}},
        }

    plans = {"issued": 0, "stale": set()}

    def execute(action: Mapping[str, Any]) -> Mapping[str, Any]:
        arguments = action["arguments"]
        if set(arguments) == {"plan_ref"}:
            if arguments["plan_ref"] in plans["stale"]:
                raise RuntimeError("plan_ref is stale for the current world state")
            return {
                "schema": "ora.libero-sparse-motion-receipt.v1",
                "execution_status": "reached",
                "failure_stage": None, "failure_code": None,
                "executor": ["libero_curobo_joint_position"],
                "state_changed": True,
                "steps_used": 1, "terminal_after_action": False,
                "arms": {"right": {"pose_error": {
                    "translation_m": 0.002, "orientation_rad": 0.005,
                }, "plan_ref": arguments["plan_ref"]}},
                "before": {"right": {"pose_xyz_xyzw": list(START_POSE)}},
                "after": {"right": {"pose_xyz_xyzw": list(START_POSE)}},
            }
        if action["action"] == "set_gripper":
            return {"schema": "gripper", "state_changed": True}
        case_id = next(
            (
                item["case_id"] for item in MOTION_CASES
                if build_case_arguments(
                    str(item["case_id"]),
                    position=START_POSE[:3], orientation=START_POSE[3:],
                ) == arguments
            ),
            None,
        )
        # any executed non-gripper write expires every issued plan,
        # mirroring the real freshness semantics
        plans["stale"].update(
            f"plan-{index + 1:032d}" for index in range(plans["issued"])
        )
        return _receipt(case_id or "precision_small_local_6mm", arguments)

    def probe(request: Mapping[str, Any]) -> Mapping[str, Any]:
        plans["issued"] += 1
        plan_ref = f"plan-{plans['issued']:032d}"
        variants = [{"variant_id": "variant-00-exact-+0deg"}]
        if request["variant_policy"] == "parallel_jaw_equivalents":
            variants.append({"variant_id": "variant-01-jaw-flip-180deg"})
        for variant in variants:
            variant.update(reachable=True, plan_ref=plan_ref, planner={
                "called": True, "valid_query": True, "status": "success", "error": None,
            })
        return {
            "schema": "ora.motion-probe-result.v1",
            "world_state_changed": False, "robot_decisions_consumed": 0,
            "variant_policy": request["variant_policy"],
            "results": [{
            "query_id": "q0", "arm": "right", "reachable": True,
            "selected_variant_id": "variant-00-exact-+0deg",
            "plan_ref": plan_ref, "variants": variants,
        }]}

    results = run_cases(
        observe=observe, execute=execute, probe=probe,
        current_pose=lambda observation: observation["proprioception"][
            "end_effectors"]["right"]["pose_xyz_xyzw"],
    )
    by_id = {result["case_id"]: result for result in results}
    assert set(by_id) == set(ALL_CASE_IDS)
    failed = [case_id for case_id, result in by_id.items() if not result["passed"]]
    assert failed == []

    record = build_qualification_record(
        source_commit="a" * 40, host={"execution_host": "test"},
        session_identity={"task": "t", "seed": 0, "suite": "s"},
        case_results=results,
    )
    assert record["schema"] == SCHEMA
    assert record["passed"] is True
    assert record["missing_cases"] == []
    assert len(record["record_sha256"]) == 64

    partial = build_qualification_record(
        source_commit="a" * 40, host={}, session_identity={},
        case_results=results[:3],
    )
    assert partial["passed"] is False
    assert partial["missing_cases"]


def test_local_route_tolerance_matches_the_servo_contract() -> None:
    """A boundary-exact 15-degree request must never route local yet fail
    contract validation as a fatal fault; the routing tolerance is exactly
    the contract's +1e-12."""

    import math

    from ora.contracts.action import (
        SERVO_MAX_ROTATION_RAD,
        SERVO_MAX_TRANSLATION_M,
    )
    from ora.environment.backends.libero_pro.common import _within_local_servo_envelope

    assert _within_local_servo_envelope(0.01, math.radians(10.0))
    assert _within_local_servo_envelope(
        SERVO_MAX_TRANSLATION_M, SERVO_MAX_ROTATION_RAD
    )
    # The live failure: a model-computed "exactly 15 degrees" rotation landed
    # ~1e-10 rad above the limit. It must route global, not local.
    assert not _within_local_servo_envelope(0.01, SERVO_MAX_ROTATION_RAD + 1e-10)
    assert not _within_local_servo_envelope(
        SERVO_MAX_TRANSLATION_M + 1e-10, 0.0
    )


@pytest.mark.parametrize("case_id", ["planner_refusal_unreachable", "orientation_only_near_180deg"])
def test_qualification_rejects_a_planner_error_receipt(case_id: str) -> None:
    """A recorded action's no-write AttributeError must not qualify reachability.

    Canonical planner artifact SHA256:
    809a8c188c5269f749baf9f7679699f645ba921c2861a3716e63643451bc2318.
    The fixture retains its classification and evidence identity, not run data.
    """
    case = next(row for row in MOTION_CASES if row["case_id"] == case_id)
    receipt = _receipt("planner_refusal_unreachable", {})
    receipt["failure_code"] = "planner_error"
    receipt["segment"]["planner"].update(
        status="planner_error", planner_status=None, valid_query=None,
        canonical_artifact={
            "path": "curobo/plans/action-0124.json",
            "sha256": "809a8c188c5269f749baf9f7679699f645ba921c2861a3716e63643451bc2318",
        },
    )
    result = judge_motion_case(case, receipt)
    assert result["passed"] is False
    assert any("implementation error" in note for note in result["notes"])


@pytest.mark.parametrize("defect", ["invalid_start", "missing_planner", "missing_artifact", "writes", "missing_steps", "missing_state"])
def test_qualification_refusal_requires_valid_start_and_nonexecution(defect: str) -> None:
    case = next(row for row in MOTION_CASES if row["case_id"] == "planner_refusal_unreachable")
    receipt = _receipt("planner_refusal_unreachable", {})
    assert judge_motion_case(case, receipt)["passed"] is True
    if defect == "invalid_start":
        receipt["segment"]["planner"]["valid_query"] = False
    elif defect == "missing_planner":
        del receipt["segment"]
    elif defect == "missing_artifact":
        del receipt["segment"]["planner"]["canonical_artifact"]
    elif defect == "writes":
        receipt.update(state_changed=True, steps_used=1)
    else:
        del receipt["steps_used" if defect == "missing_steps" else "state_changed"]
    assert judge_motion_case(case, receipt)["passed"] is False


@pytest.mark.parametrize("defect", ["missing", "nan", "negative", "wrong_shape", "missing_terminal"])
def test_qualification_reached_requires_finite_complete_measurements(defect: str) -> None:
    case = next(row for row in MOTION_CASES if row["case_id"] == "precision_small_local_6mm")
    receipt = _receipt(case["case_id"], {})
    assert judge_motion_case(case, receipt)["passed"] is True
    if defect == "missing":
        del receipt["arms"]["right"]["pose_error"]
    elif defect == "wrong_shape":
        receipt["arms"]["right"] = []
    elif defect == "missing_terminal":
        del receipt["terminal_after_action"]
    else:
        receipt["arms"]["right"]["pose_error"]["translation_m"] = float("nan") if defect == "nan" else -0.1
    assert judge_motion_case(case, receipt)["passed"] is False


@pytest.mark.parametrize("defect", ["planner_error", "missing", "invalid_start", "duplicates", "world_write"])
def test_qualification_jaw_flip_rejects_two_invalid_variant_placeholders(defect: str) -> None:
    probed = {
        "schema": "ora.motion-probe-result.v1",
        "world_state_changed": False, "robot_decisions_consumed": 0,
        "variant_policy": "parallel_jaw_equivalents",
        "results": [{
            "reachable": False, "plan_ref": None, "selected_variant_id": None,
            "variants": [{
                "variant_id": variant_id, "reachable": False,
                "planner": {"called": True, "valid_query": True, "status": "goal_unreachable", "error": None},
                "failure": {"code": "goal_unreachable"},
            } for variant_id in ("variant-00-exact-+0deg", "variant-01-jaw-flip-180deg")],
        }],
    }
    assert _qualification_probe_row(
        probed, policy="parallel_jaw_equivalents", require_plan=False,
    )["reachable"] is False
    variants = probed["results"][0]["variants"]
    if defect == "planner_error":
        for variant in variants:
            variant["planner"].update(status="planner_error", error={"type": "AttributeError"})
            variant["failure"]["code"] = "planner_error"
    elif defect == "missing":
        for variant in variants:
            del variant["planner"]
    elif defect == "invalid_start":
        variants[0]["planner"]["valid_query"] = False
    elif defect == "duplicates":
        variants[1]["variant_id"] = variants[0]["variant_id"]
    else:
        probed["world_state_changed"] = True
    with pytest.raises(ValueError):
        _qualification_probe_row(probed, policy="parallel_jaw_equivalents", require_plan=False)


def test_large_contact_approach_case_requires_the_staged_route() -> None:
    """Arriving by some other route does not qualify the staged one."""

    case = next(item for item in MOTION_CASES
                if item["case_id"] == "contact_approach_large_staged")
    arguments = build_case_arguments(
        case["case_id"], position=START_POSE[:3], orientation=START_POSE[3:],
    )
    staged = _receipt(case["case_id"], arguments)
    assert judge_motion_case(case, staged)["passed"] is True
    rerouted = {**staged, "motion_strategy": "global_then_local_settle"}
    verdict = judge_motion_case(case, rerouted)
    assert verdict["passed"] is False
    assert "motion_strategy global_then_local_settle" in verdict["notes"]


def test_the_worker_environment_of_a_lane_replaces_the_run_s_values() -> None:
    from ora.orchestrator.backends.libero_pro.qualification import _lane_runtime, _worker_environment

    lane = _worker_environment(["CUDA_VISIBLE_DEVICES=3"])
    runtime = {"resource_binding": {"worker_environment": {"CUDA_VISIBLE_DEVICES": "1", "MUJOCO_GL": "egl"}}}
    assert _lane_runtime(runtime, lane)["resource_binding"]["worker_environment"] == {
        "CUDA_VISIBLE_DEVICES": "3", "MUJOCO_GL": "egl"}
    assert runtime["resource_binding"]["worker_environment"]["CUDA_VISIBLE_DEVICES"] == "1"
    with pytest.raises(ValueError, match="NAME=VALUE"):
        _worker_environment(["CUDA_VISIBLE_DEVICES"])


def test_a_free_space_case_accepts_the_osc_line_only_with_its_route_record() -> None:
    """Motion profile v4 runs a clear, straight free-space plan as the OSC line; the receipt must say so."""

    case = next(item for item in MOTION_CASES if item["case_id"] == "free_space_translation_10cm")
    arguments = build_case_arguments(case["case_id"], position=START_POSE[:3], orientation=START_POSE[3:])
    receipt = _receipt(case["case_id"], arguments)
    receipt["executor"] = ["libero_robosuite_osc_pose"]
    assert judge_motion_case(case, receipt)["passed"] is False
    receipt["arms"]["right"]["route_choice"] = {
        "route": "osc_straight_line", "reason": "planner_path_clear_and_straight",
        "planner": {"canonical_artifact": {"path": "curobo/plans/action-0000.json", "sha256": "a" * 64}}}
    assert judge_motion_case(case, receipt)["passed"] is True
    receipt["arms"]["right"]["route_choice"]["planner"] = {}
    assert judge_motion_case(case, receipt)["passed"] is False
