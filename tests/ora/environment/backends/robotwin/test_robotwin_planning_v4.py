"""The vendored RoboTwin planner v4: candidate search, admission, fallback and endpoint contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ora.environment.backends.robotwin.planning import (
    RoboTwinPlanRejected,
    ValidatedRoboTwinPlan,
    bounded_dls_joint_delta,
    cartesian_pose_error_twist,
    numerical_cartesian_jacobian,
    planner_public_summary,
    public_tcp_point_jacobian,
    select_topk_validated_plan,
    split_batch_plan_result,
    validate_inter_arm_clearance,
    validate_single_arm_plan,
    weighted_svd_dls_joint_delta,
    write_plan_artifact,
)
from ora.contracts import (
    ActionValidationError,
    MOTION_ACTION_V3_SCHEMA,
    validate_runtime_action,
)
from ora.environment.backends.robotwin.session import (
    RoboTwinLiveSession,
)
from ora.environment.backends.robotwin.motion_adapter import (
    RoboTwinMotionBackendAdapter,
)
from ora.environment.controllers.embodiment_control import (
    validate_embodiment_control_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
PROFILE_PATH = (
    PROJECT_ROOT
    / "ora/policy/embodiments/aloha-agilex/control-profile-v10.json"
)


def _profile() -> dict:
    return validate_embodiment_control_profile(
        json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    )


def _fk(joints: list[float] | np.ndarray) -> list[float]:
    values = np.asarray(joints, dtype=np.float64)
    return [float(values[0]), float(values[1]), 0.9, 0.0, 0.0, 0.0, 1.0]


def _result() -> dict:
    position = np.asarray([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.01, 0.02, 0.0, 0.0, 0.0, 0.0],
    ])
    return {
        "schema": "robotwin.curobo-plan-result.v1",
        "status": "Success",
        "planner_status": "SUCCESS",
        "valid_query": True,
        "attempts": 1,
        "trajopt_attempts": 1,
        "used_graph": False,
        "goalset_index": None,
        "path_buffer_last_tstep": None,
        "errors": {"position": 0.0, "rotation": 0.0, "cspace": None},
        "timings_s": {
            "solve": 0.01, "ik": 0.001, "graph": 0.0,
            "trajopt": 0.009, "finetune": 0.0, "total": 0.01,
        },
        "interpolation_dt_s": 0.004,
        "joint_names": [f"fl_joint{index}" for index in range(1, 7)],
        "position": position,
        "velocity": np.zeros_like(position),
    }


def _validate(result: dict):
    return validate_single_arm_plan(
        result,
        expected_joint_names=[f"fl_joint{index}" for index in range(1, 7)],
        measured_start_joints=[0.0] * 6,
        measured_start_pose_xyzw=_fk([0.0] * 6),
        target_pose_xyzw=_fk([0.01, 0.02, 0.0, 0.0, 0.0, 0.0]),
        joint_limits=[[-1.0, 1.0]] * 6,
        fk_public_pose=_fk,
        workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45],
        },
        profile=_profile()["plan_validation"],
    )


def test_profile_binds_validation_execution_and_local_limits() -> None:
    profile = _profile()
    assert profile["profile_id"] == "robotwin_aloha_contact_recovery_dual_arm_v10"
    assert profile["plan_validation"]["maximum_start_joint_error_rad"] == 0.02
    assert profile["waypoint_execution"]["maximum_physics_steps_per_waypoint"] == 64
    assert profile["local_cartesian"]["global_planner_fallback"] is False


def test_numeric_jacobian_and_bounded_dls_respect_cartesian_and_joint_steps() -> None:
    def linear_fk(joints) -> list[float]:
        q = np.asarray(joints, dtype=np.float64)
        half = q[3] * 0.5
        return [q[0], q[1], 0.9 + q[2], np.sin(half), 0.0, 0.0, np.cos(half)]

    joints = np.zeros(6)
    target = [0.02, 0.0, 0.9, np.sin(0.1), 0.0, 0.0, np.cos(0.1)]
    error = cartesian_pose_error_twist(target, linear_fk(joints))
    jacobian = numerical_cartesian_jacobian(
        joints, fk_public_pose=linear_fk, epsilon_rad=1e-5,
    )
    delta = bounded_dls_joint_delta(
        error, jacobian, damping=0.01,
        maximum_translation_step_m=0.005,
        maximum_rotation_step_rad=np.deg2rad(2.0),
        maximum_joint_step_rad=0.04,
    )
    assert np.max(np.abs(delta)) <= 0.04 + 1e-12
    assert 0.0 < delta[0] <= 0.0051
    assert 0.0 < delta[3] <= np.deg2rad(2.0) + 1e-4


def test_public_tcp_adjoint_matches_finite_difference_point_velocity() -> None:
    link = np.zeros((6, 1), dtype=np.float64)
    link[5, 0] = 1.0
    shifted = public_tcp_point_jacobian(link, [0.12, 0.0, 0.0])
    assert shifted[:, 0] == pytest.approx([0.0, 0.12, 0.0, 0.0, 0.0, 1.0])

    epsilon = 1e-6
    before = np.asarray([0.12, 0.0, 0.0])
    after = np.asarray([
        0.12 * np.cos(epsilon), 0.12 * np.sin(epsilon), 0.0,
    ])
    assert (after - before) / epsilon == pytest.approx(
        shifted[:3, 0], abs=1e-6,
    )


def test_weighted_svd_dls_is_bounded_and_predicts_progress() -> None:
    error = np.asarray([0.01, -0.005, 0.0, 0.02, 0.0, 0.0])
    jacobian = np.eye(6)
    delta, diagnostics = weighted_svd_dls_joint_delta(
        error,
        jacobian,
        current_joints=[0.0] * 6,
        joint_limits=[[-1.0, 1.0]] * 6,
        base_damping=0.02,
        singular_value_threshold=0.05,
        maximum_damping=0.3,
        joint_limit_margin_rad=0.2,
        minimum_joint_mobility=0.1,
        maximum_translation_step_m=0.005,
        maximum_rotation_step_rad=np.deg2rad(2.0),
        maximum_joint_step_rad=0.04,
    )
    assert np.max(np.abs(delta)) <= 0.04
    assert np.linalg.norm(error - jacobian @ delta) < np.linalg.norm(error)
    assert diagnostics["adaptive_damping"] >= 0.02


def test_analytic_local_line_search_stalls_without_predicted_fk_progress() -> None:
    adapter = object.__new__(RoboTwinMotionBackendAdapter)
    adapter.public_tcp_jacobian = lambda _arm, _joints: np.eye(6)
    adapter.joint_limits = lambda _arm: [[-1.0, 1.0]] * 6
    current_pose = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]
    adapter.fk_public_tcp_pose = lambda _arm, _joints: current_pose
    local = json.loads((
        Path(__file__).resolve().parents[5]
        / "ora/policy/embodiments/aloha-agilex/control-profile-v10.json"
    ).read_text(encoding="utf-8"))["local_cartesian"]
    with pytest.raises(RoboTwinPlanRejected) as caught:
        adapter.compute_local_joint_target(
            "left", [0.0] * 6, current_pose,
            [0.005, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
            local,
        )
    assert caught.value.code == "local_servo_stalled"
    assert caught.value.details["maximum_line_search_halvings"] == 5


def test_topk_selects_later_admitted_candidate_and_records_nearest_failure() -> None:
    rejected = _result()
    rejected["position"] = rejected["position"].copy()
    rejected["position"][0, 0] = 0.03
    accepted = _result()
    topk = {
        "schema": "robotwin.curobo-plan-result.v2",
        "status": "Success",
        "joint_names": accepted["joint_names"],
        "ik": {"seed_count": 128},
        "candidates": [
            {**rejected, "ik_candidate_index": 0, "ik_distance_rad": 0.1},
            {**accepted, "ik_candidate_index": 1, "ik_distance_rad": 0.3},
        ],
    }
    selected, validated, selection = select_topk_validated_plan(
        topk, validator=_validate,
    )
    assert selected["ik_candidate_index"] == 1
    assert validated.summary["status"] == "accepted"
    assert selection["selected_candidate_index"] == 1
    assert selection["candidates"][0]["admission"] == "plan_rejected"
    assert selection["candidates"][1]["admission"] == "accepted"


def test_topk_all_failed_or_rejected_fails_closed() -> None:
    failed = {
        "schema": "robotwin.curobo-plan-result.v2",
        "status": "Fail",
        "joint_names": _result()["joint_names"],
        "candidates": [{
            "status": "Fail", "planner_status": "TRAJOPT_FAIL",
            "ik_candidate_index": 0,
        }],
    }
    with pytest.raises(RoboTwinPlanRejected) as caught:
        select_topk_validated_plan(failed, validator=_validate)
    assert caught.value.code == "topk_no_admitted_plan"


def test_topk_all_failed_preserves_bounded_candidate_diagnostics() -> None:
    failed = {
        "schema": "robotwin.curobo-plan-result.v2",
        "status": "Fail",
        "planner_status": "TOP_K_PLANNING_FAILED",
        "joint_names": _result()["joint_names"],
        "candidates": [
            {
                "status": "Fail",
                "planner_status": "TRAJOPT_FAIL",
                "planning_attempt": "graph_only_fallback",
                "ik_candidate_index": 0,
                "ik_distance_rad": 0.42,
                "failed_attempts": [{
                    "planning_attempt": "trajopt",
                    "planner_status": "TRAJOPT_FAIL",
                }],
            },
            {
                "status": "Fail",
                "planner_status": "GRAPH_FAIL",
                "planning_attempt": "validated_linear_joint_fallback",
                "ik_candidate_index": 1,
                "ik_distance_rad": 0.18,
                "failed_attempts": [{
                    "planning_attempt": "graph_fallback",
                    "planner_status": "GRAPH_FAIL",
                }],
            },
        ],
    }

    summary = planner_public_summary(failed)

    assert summary["stage"] == "TOP_K_PLANNING_FAILED"
    assert summary["topk_selection"] == {
        "candidate_count": 2,
        "admitted_count": 0,
        "selected_candidate_index": None,
        "candidate_outcomes": [
            {
                "candidate_index": 0,
                "status": "Fail",
                "admission": "planner_failed",
                "planning_attempt": "graph_only_fallback",
                "planner_status": "TRAJOPT_FAIL",
                "ik_distance_rad": 0.42,
                "failure_stages": [{
                    "planning_attempt": "trajopt",
                    "planner_status": "TRAJOPT_FAIL",
                }],
            },
            {
                "candidate_index": 1,
                "status": "Fail",
                "admission": "planner_failed",
                "planning_attempt": "validated_linear_joint_fallback",
                "planner_status": "GRAPH_FAIL",
                "ik_distance_rad": 0.18,
                "failure_stages": [{
                    "planning_attempt": "graph_fallback",
                    "planner_status": "GRAPH_FAIL",
                }],
            },
        ],
    }
    assert RoboTwinLiveSession._planner_record(failed, called=True) == summary


def test_valid_plan_records_independent_fk_and_compact_artifact(tmp_path: Path) -> None:
    validated = _validate(_result())
    assert validated.summary["status"] == "accepted"
    assert validated.summary["planning_endpoint_error"]["translation_m"] == 0.0
    reference = write_plan_artifact(
        tmp_path, action_sequence=3, arm="left", validated=validated,
        binding={"action_id": "a3", "source": "test"},
    )
    manifest = json.loads((tmp_path / reference["path"]).read_text())
    assert manifest["schema"] == "ora.robotwin-plan-artifact.v1"
    arrays = np.load((tmp_path / reference["path"]).parent / "arrays.npz")
    assert set(arrays.files) == {
        "planner_joint_position", "planner_joint_velocity",
        "planned_link6_pose_xyzw", "planned_public_tcp_pose_xyzw",
    }


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda value: value.update(schema="old"), "planner_schema"),
        (lambda value: value.update(joint_names=list(reversed(value["joint_names"]))), "joint_order"),
        (lambda value: value["position"].__setitem__((0, 0), 0.03), "start_joint_error"),
        (lambda value: value["position"].__setitem__((1, 0), np.nan), "non_finite_plan"),
        (lambda value: value["position"].__setitem__((1, 0), 0.05), "planning_endpoint_error"),
    ],
)
def test_plan_validator_rejects_before_execution(mutate, code: str) -> None:
    result = _result()
    mutate(result)
    with pytest.raises(RoboTwinPlanRejected) as caught:
        _validate(result)
    assert caught.value.code == code


def _validated_path(x_values: list[float]) -> ValidatedRoboTwinPlan:
    position = np.zeros((len(x_values), 6), dtype=np.float64)
    position[:, 0] = x_values
    return ValidatedRoboTwinPlan(
        position=position,
        velocity=np.zeros_like(position),
        planned_link6_pose_xyzw=np.zeros((len(x_values), 7), dtype=np.float64),
        planned_tcp_pose_xyzw=np.zeros((len(x_values), 7), dtype=np.float64),
        summary={"status": "accepted"},
    )


def test_inter_arm_clearance_checks_static_other_arm_and_pair() -> None:
    def chain(arm: str, joints) -> list[list[float]]:
        x = float(joints[0])
        base = -0.2 if arm == "left" else 0.2
        return [[base + x, 0.0, 1.0]]

    profile = {"link_radius_m": 0.045, "minimum_surface_clearance_m": 0.0}
    accepted = validate_inter_arm_clearance(
        {"left": _validated_path([0.0, 0.05])},
        measured_joints={"left": [0.0] * 6, "right": [0.0] * 6},
        fk_chain_points=chain,
        profile=profile,
    )
    assert accepted["status"] == "accepted"
    with pytest.raises(RoboTwinPlanRejected) as caught:
        validate_inter_arm_clearance(
            {
                "left": _validated_path([0.0, 0.16]),
                "right": _validated_path([0.0, -0.16]),
            },
            measured_joints={"left": [0.0] * 6, "right": [0.0] * 6},
            fk_chain_points=chain,
            profile=profile,
        )
    assert caught.value.code == "inter_arm_collision"


def test_rich_planner_summary_is_closed_and_batch_rows_are_arm_local() -> None:
    first = _result()
    second = _result()
    second["status"] = "Fail"
    batch = {
        **first,
        "status": ["Success", "Fail"],
        "planner_status": ["SUCCESS", "IK_FAIL"],
        "valid_query": [True, True],
        "goalset_index": [0, 1],
        "errors": {
            "position": [0.001, 0.2],
            "rotation": [0.002, 0.3],
            "cspace": [None, None],
        },
        "position": np.stack([first["position"], second["position"]]),
        "velocity": np.stack([first["velocity"], second["velocity"]]),
    }
    row = split_batch_plan_result(batch, 0)
    assert row["status"] == "Success"
    assert row["goalset_index"] == 0
    assert row["position"].shape == (2, 6)
    summary = planner_public_summary(row)
    assert set(summary) == {
        "called", "status", "stage", "valid_query", "attempts",
        "trajopt_attempts", "used_graph", "goalset_index", "path_steps",
        "interpolation_dt_s", "errors", "timings_s", "error",
    }
    assert summary["errors"] == {
        "position": 0.001, "rotation": 0.002, "cspace": None,
    }
    assert summary["interpolation_dt_s"] == pytest.approx(0.004)

    ragged = dict(batch)
    ragged["position"] = [first["position"], None]
    ragged["velocity"] = [first["velocity"], None]
    ragged_row = split_batch_plan_result(ragged, 0)
    assert ragged_row["position"].shape == (2, 6)
    failed_row = split_batch_plan_result(ragged, 1)
    assert failed_row["status"] == "Fail"
    assert "position" not in failed_row


def test_motion_v3_plan_ref_is_exactly_hash_shaped() -> None:
    payload = {
        "schema": MOTION_ACTION_V3_SCHEMA,
        "observation_id": "observation-0",
        "action_id": "action-0",
        "action": "move_pose",
        "arguments": {"plan_ref": "plan-" + "a" * 32},
        "bound_observation_sha256": "0" * 64,
    }
    assert validate_runtime_action(payload)["arguments"] == payload["arguments"]
    for invalid in ("plan-short", "plan-" + "g" * 32, "x" * 37):
        payload["arguments"] = {"plan_ref": invalid}
        with pytest.raises(ActionValidationError):
            validate_runtime_action(payload)


def test_topk_untimed_plan_ranks_strictly_behind_timed_plan() -> None:
    untimed = _result()
    untimed["interpolation_dt_s"] = None
    timed = _result()
    topk = {
        "schema": "robotwin.curobo-plan-result.v2",
        "status": "Success",
        "joint_names": timed["joint_names"],
        "ik": {"seed_count": 128},
        "candidates": [
            {**untimed, "ik_candidate_index": 0, "ik_distance_rad": 0.1},
            {**timed, "ik_candidate_index": 1, "ik_distance_rad": 0.3},
        ],
    }
    selected, validated, selection = select_topk_validated_plan(
        topk, validator=_validate,
    )
    assert selected["ik_candidate_index"] == 1
    assert selection["candidates"][0]["admission"] == "accepted"
    assert selection["selected_candidate_index"] == 1


def test_topk_boolean_dt_is_untimed_not_one_second() -> None:
    bool_dt = _result()
    bool_dt["interpolation_dt_s"] = True
    timed = _result()
    topk = {
        "schema": "robotwin.curobo-plan-result.v2",
        "status": "Success",
        "joint_names": timed["joint_names"],
        "ik": {"seed_count": 128},
        "candidates": [
            {**bool_dt, "ik_candidate_index": 0, "ik_distance_rad": 0.1},
            {**timed, "ik_candidate_index": 1, "ik_distance_rad": 0.3},
        ],
    }
    selected, _, _ = select_topk_validated_plan(topk, validator=_validate)
    assert selected["ik_candidate_index"] == 1


def test_probe_variant_sort_key_ranks_untimed_last_and_shortest_timed_first() -> None:
    from ora.environment.backends.robotwin.session import (
        probe_variant_sort_key,
    )

    def row(variant_id, duration, translation=0.001):
        return {
            "variant_id": variant_id,
            "trajectory_duration_s": duration,
            "endpoint_error": {
                "translation_m": translation, "orientation_rad": 0.001,
            },
        }

    rows = [row("untimed", 0.0, 0.0), row("slow", 2.0), row("fast", 0.5)]
    rows.sort(key=probe_variant_sort_key)
    assert [item["variant_id"] for item in rows] == ["fast", "slow", "untimed"]
