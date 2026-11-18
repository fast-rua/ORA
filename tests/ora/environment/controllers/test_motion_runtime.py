"""Shared hybrid-motion Runtime contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from ora.agent.evidence.policy_state import _forbidden_policy_key
from ora.agent.evidence.trace_privacy import _forbidden_path
from ora.agent.model_services.clients import Sam3ApiSegmenter
from ora.contracts import (
    MOTION_ACTION_SCHEMA,
    compile_servo_delta,
    validate_runtime_action,
)
from ora.contracts.action import ActionValidationError
from ora.environment.backends.libero_pro.planner_port import PLAN_RECORD_SCHEMA
from ora.environment.backends.libero_pro.planner_request import PlannerRequestError, validate_libero_plan_request
from ora.environment.backends.libero_pro.session import LiberoProSession
from ora.environment.controllers.hybrid_motion import (
    HybridMotionProfileError,
    aggregate_motion_contact,
    bounded_joint_position_command,
    load_hybrid_motion_profile,
    motion_stop_reason,
    validate_hybrid_motion_profile,
)
from ora.environment.workspace_profiles import resolve_workspace_profile
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from ora.orchestrator.episode.stagnation import StagnationGuard
from tests.support.agent.motion_runtime import _action, _pose


ROOT = Path(__file__).resolve().parents[4]
MOTION_PROFILE = (
    ROOT / "ora/policy/embodiments/franka-panda/motion-profile-v2.json"
)
MOTION_PROFILE_SHA256 = (
    "30df8c9ba91abc23c8ebdbf3977bf20604441e6bbe853ee6dad618074f8548ea"
)


@pytest.mark.parametrize("outcome", [
    "ready", "timeout", "invalid_json", "response_identity", "close_error", "interrupt",
])
def test_curobo_preflight_closes_its_worker_even_when_readiness_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str,
) -> None:
    from ora.environment.controllers import motion_planner
    from ora.orchestrator.preflight import gates as preflight

    absent = {
        "libero_pro_root_env": "TEST_CUROBO_PREFLIGHT_ABSENT_LIBERO",
        "data_root_env": "TEST_CUROBO_PREFLIGHT_ABSENT_DATA",
        "python_env": "TEST_CUROBO_PREFLIGHT_ABSENT_SIMULATOR",
    }
    for name in absent.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TEST_CUROBO_PREFLIGHT_PYTHON", str(tmp_path / "planner-python"))
    monkeypatch.setenv("TEST_CUROBO_PREFLIGHT_ROOT", str(tmp_path / "planner-root"))
    events = []
    worker_outputs = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.output = kwargs["output_root"]
            worker_outputs.append(self.output)
            events.append("created")

        def start(self) -> dict[str, Any]:
            events.append("spawned")
            if outcome == "interrupt":
                raise KeyboardInterrupt("CPU readiness interruption")
            if outcome not in {"ready", "close_error"}:
                raise motion_planner.MotionPlannerError(f"CPU readiness {outcome}")
            return {"worker": {"pid": 123}, "test_fixture_only": True}

        def close(self) -> None:
            assert self.output.is_dir(), "close must precede temporary workspace cleanup"
            events.append("closed")
            if outcome == "close_error":
                raise motion_planner.MotionPlannerError("CPU readiness cleanup failed")

    def forbidden_process(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CPU preflight fixture must not launch any process")

    monkeypatch.setattr(motion_planner, "MotionPlannerClient", FakeClient)
    monkeypatch.setattr(preflight.subprocess, "run", forbidden_process)
    monkeypatch.setattr(preflight, "_git", forbidden_process)
    plan = {
        "policy": {"action_codec": "motion_action_v3"},
        "runtime": {**absent, "kind": "libero_pro", "require_gpu": False,
            "motion_planner": {
                "implementation": "curobo_v2",
                "python_env": "TEST_CUROBO_PREFLIGHT_PYTHON",
                "root_env": "TEST_CUROBO_PREFLIGHT_ROOT",
                "required_revision": "a" * 40,
            }},
    }
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt, match="CPU readiness interruption"):
            preflight._runtime(plan)
    else:
        checks = preflight._runtime(plan)
        planner = next(row for row in checks if row["check"] == "curobo_motion_planner")
        assert planner["passed"] is (outcome == "ready")
        if outcome != "ready":
            assert "CPU readiness" in planner["detail"]
    assert events == ["created", "spawned", "closed"]
    assert len(worker_outputs) == 1 and not worker_outputs[0].exists()


def test_hybrid_contact_estimate_is_safe_aggregate_not_contact_truth() -> None:
    public_receipt = {"feedback": {"contact_estimated": True}}
    assert _forbidden_path(public_receipt) is None
    assert _forbidden_policy_key(public_receipt) is None

    privileged = {"feedback": {"contact_estimated": {"body": "object-1"}}}
    assert _forbidden_path(privileged) == "$.feedback.contact_estimated"
    assert _forbidden_policy_key(privileged) == "$.feedback.contact_estimated"

    compiler_geometry = {
        "compiler": {
            "contact_graspnet_gripper_depth_m": 0.1034,
            "controlled_point_to_contact_baseline_m": 0.0064,
        },
        "candidate": {"compiler_checks": {
            "contact_baseline_reconstruction": "passed",
        }},
    }
    assert _forbidden_path(compiler_geometry) == (
        "$.compiler.contact_graspnet_gripper_depth_m"
    )
    assert _forbidden_path(
        compiler_geometry, allow_public_grasp_contact_pose=True,
    ) is None
    assert _forbidden_path({
        "compiler": {"contact_graspnet_gripper_depth_m": 0.1035},
    }, allow_public_grasp_contact_pose=True) == (
        "$.compiler.contact_graspnet_gripper_depth_m"
    )
    assert _forbidden_path({
        "elsewhere": {"controlled_point_to_contact_baseline_m": 0.0064},
    }, allow_public_grasp_contact_pose=True) == (
        "$.elsewhere.controlled_point_to_contact_baseline_m"
    )
    assert _forbidden_path({
        "candidate": {"compiler_checks": {
            "contact_baseline_reconstruction": {"body": "object-1"},
        }},
    }, allow_public_grasp_contact_pose=True) == (
        "$.candidate.compiler_checks.contact_baseline_reconstruction"
    )


def test_forbidden_terms_match_word_boundaries_not_substrings() -> None:
    # Regression for the v13 seal incident: the policy-safe staged-motion
    # receipt fields embed the controller phase noun "precontact", which the
    # bare substring rule misread as contact-sensing evidence and used to
    # invalidate trace evaluation for nine completed runs.
    policy_safe_receipt = {
        "motion_strategy": {
            "precontact_public_tcp_poses": [[0.1, 0.2, 0.3]],
            "mode": "global_precontact_then_local",
        },
        "controller_tracking": {"phases": {"global_precontact": {"completed": True}}},
    }
    assert _forbidden_path(policy_safe_receipt) is None

    # Boundary-delimited terms stay forbidden, including plural forms.
    assert _forbidden_path({"contact": {"body": "object-1"}}) == "$.contact"
    assert _forbidden_path({"contact_force_n": 1.2}) == "$.contact_force_n"
    assert _forbidden_path({"contacts": []}) == "$.contacts"
    assert _forbidden_path({"task_sources": ["x"]}) == "$.task_sources"
    assert _forbidden_path({"object_gt": {}}) == "$.object_gt"
    assert _forbidden_path({"segmentation_mask": "m"}) == "$.segmentation_mask"

    # Word-internal embeddings are public vocabulary, not evidence leaks.
    assert _forbidden_path({"presegmentation_stage": 1}) is None
    assert _forbidden_path({"precontact_offset_m": 0.05}) is None

    # The policy-state scanner shares the boundary semantics.
    assert _forbidden_policy_key({
        "motion_strategy": {"precontact_public_tcp_poses": [[0.1, 0.2, 0.3]]},
        "controller_tracking": {"phases": {"global_precontact": {}}},
    }) is None
    assert _forbidden_policy_key({"contact": {}}) == "$.contact"
    assert _forbidden_policy_key({"official_success": True}) == "$.official_success"

    # The policy-authored staged-approach argument is public motion
    # vocabulary in both scanners; values beneath it are still scanned.
    assert _forbidden_path({"targets": {"left": {"contact_approach": "top_down"}}}) is None
    assert _forbidden_policy_key({"contact_approach": "top_down"}) is None
    assert _forbidden_path(
        {"contact_approach": {"object_gt": {}}}
    ) == "$.contact_approach.object_gt"

    # The staged phase label `local_contact` is exempt only under `phases`;
    # elsewhere the standalone term stays forbidden, and phase values are
    # still scanned.
    phases_receipt = {"controller_tracking": {"phases": {"local_contact": {"completed": True}}}}
    assert _forbidden_path(phases_receipt) is None
    assert _forbidden_policy_key(phases_receipt) is None
    assert _forbidden_path({"local_contact": True}) == "$.local_contact"
    assert _forbidden_policy_key({"local_contact": True}) == "$.local_contact"
    assert _forbidden_path({
        "controller_tracking": {"phases": {"local_contact": {"object_gt": {}}}},
    }) == "$.controller_tracking.phases.local_contact.object_gt"


def test_libero_90_range_is_validated_by_runspec() -> None:
    from tests.support.orchestrator.run_specs import native_spec

    spec = native_spec("libero_pro")
    spec["runtime"].update(suite="libero_90", task_id=89)
    assert resolve_run_spec(spec, root=ROOT)["runtime"]["task_id"] == 89
    spec["runtime"]["task_id"] = 90
    with pytest.raises(RunSpecError, match=r"\[0,89\]"):
        resolve_run_spec(spec, root=ROOT)


def test_resident_sam_binding_accepts_the_motion_surface() -> None:
    segmenter = Sam3ApiSegmenter(expected_binding={
        "schema": "ora.segmentation-service-binding.v1",
        "tool_surface": "libero_placement_direct_overhead_v1",
        "endpoint": "http://127.0.0.1:8114",
        "instance_id": "model-instance-test",
        "pid": 123,
        "started_unix_s": 1.0,
        "adapter_sha256": "a" * 64,
        "model": {"family": "sam3"},
        "load_count": 1,
        "warmup_complete": True,
        "qualification": {
            "sha256": "b" * 64,
            "status": "passed",
            "created_unix_s": 1.0,
        },
    })
    try:
        assert segmenter.expected_binding["tool_surface"] == (
            "libero_placement_direct_overhead_v1"
        )
    finally:
        segmenter.close()


def test_hybrid_workspace_profile_names_the_actual_controller_pair() -> None:
    profile = resolve_workspace_profile("libero_pro_panda_hybrid_motion_v1")
    assert profile["provenance"]["controller"] == (
        "CUROBO_JOINT_POSITION+OSC_POSE"
    )
    assert profile["semantics"] == "hard_action_envelope_not_reachability"


def test_servo_delta_compiles_world_and_grip_site_frames() -> None:
    current = [
        0.4, 0.0, 0.5, 0.0, 0.0,
        math.sin(math.pi / 4), math.cos(math.pi / 4),
    ]
    tool = compile_servo_delta({
        "translation_m": [0.01, 0.0, 0.0],
        "rotation_rotvec_rad": [0.0, 0.0, 0.0],
        "frame": "grip_site",
    }, current_pose_xyz_xyzw=current)
    assert tool["resolved_translation_world_m"] == pytest.approx([0.0, 0.01, 0.0])
    world = compile_servo_delta({
        "translation_m": [0.01, 0.0, 0.0],
        "rotation_rotvec_rad": [0.0, 0.0, 0.0],
        "frame": "world",
    }, current_pose_xyz_xyzw=current)
    assert world["resolved_translation_world_m"] == pytest.approx([0.01, 0.0, 0.0])


@pytest.mark.parametrize(
    ("delta", "iterations"),
    [([0.051, 0.0, 0.0], 1), ([0.0, 0.0, 0.0], 21)],
)
def test_motion_action_rejects_unbounded_servo(
    delta: list[float], iterations: int,
) -> None:
    with pytest.raises(ActionValidationError):
        validate_runtime_action(_action("servo_delta", {
            "arm": "right", "target_pose": _pose(),
            "delta_pose": {
                "translation_m": delta,
                "rotation_rotvec_rad": [0.0, 0.0, 0.0],
            },
            "frame": "world", "max_iterations": iterations,
        }))


def test_gripper_and_recovery_contracts_are_discrete() -> None:
    close = validate_runtime_action(_action(
        "set_gripper", {"arm": "right", "gripper": 0.0},
    ))
    assert close["arguments"]["gripper"] == 0.0
    with pytest.raises(ActionValidationError):
        validate_runtime_action(_action(
            "set_gripper", {"arm": "right", "gripper": 0.4},
        ))
    recovered = validate_runtime_action(_action(
        "recover_motion", {"arm": "right", "strategy": "backtrack_last_safe"},
    ))
    assert recovered["arguments"]["strategy"] == "backtrack_last_safe"


def test_planner_request_is_single_goal_and_has_no_scene_fields() -> None:
    request = validate_libero_plan_request({
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_positions": [0.0] * 7,
        "target_pose_base_xyzw": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
    })
    assert set(request) == {
        "joint_names", "joint_positions", "target_pose_base_xyzw",
    }
    with pytest.raises(PlannerRequestError, match="fields drifted"):
        validate_libero_plan_request({**request, "voxel_pitch_m": 0.01})
    with pytest.raises(PlannerRequestError, match="exactly one goal"):
        validate_libero_plan_request({
            **request, "goal_joint_positions": [0.0] * 7,
        })


def test_planner_request_ignores_closed_gripper_and_has_no_payload(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class Planner:
        implementation = "curobo_v2"

        def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
            captured.update(request)
            return {
                "schema": PLAN_RECORD_SCHEMA,
                "status": "not_reachable", "reachable": False,
                "world_model": "none", "planner_status": "goal_unreachable",
                "start_ee_pose_base_xyzw": [
                    0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0,
                ],
            }

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.motion_planner = Planner()
    session.output_root = tmp_path
    session.action_sequence = 0
    session.sim_steps = 0
    session.last_receipt = None
    session.workspace_bounds_m = {
        "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
    }
    session._require_environment = lambda: SimpleNamespace()
    session._current_state = lambda: (
        np.array([0.4, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        0.2,
    )
    session._motion_state = lambda: {
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_positions": [0.0] * 7,
        "base_pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "hand_pose_world_xyzw": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        "grip_site_pose_world_xyzw": [
            0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0,
        ],
    }
    receipt = session._execute_planned_motion(_action(
        "move_free_space", {"arm": "right", "target_pose": _pose()},
    ))
    assert set(captured) == {
        "joint_names", "joint_positions", "target_pose_base_xyzw",
    }
    assert receipt["motion_status"] == "refused"
    assert receipt["feedback"]["planning_failure_reason"] == "goal_unreachable"


@pytest.mark.parametrize("reason,valid_query", [
    ("planner_error", None),
    ("start_outside_limits", False),
    ("start_in_collision", False),
    ("goal_unreachable", True),
])
def test_planner_failure_without_fk_is_consistent_in_probe_and_execute(
    tmp_path: Path, reason: str, valid_query: bool | None,
) -> None:
    response = {
        "schema": PLAN_RECORD_SCHEMA,
        "status": "planner_error" if reason == "planner_error" else "not_reachable",
        "reachable": False, "world_model": "none", "valid_query": valid_query,
    }
    if reason == "planner_error":
        response["error"] = {
            "type": "AttributeError",
            "message": "'ArmReacher' object has no attribute 'primitive_collision_constraint'",
        }
    else:
        response["planner_status"] = reason

    class Planner:
        implementation = "curobo_v2"

        def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
            return {**response, "request": dict(request)}

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.motion_planner = Planner()
    session.output_root = tmp_path
    session.action_sequence = 0
    session.sim_steps = 0
    session.last_receipt = None
    session.world_mutation_sequence = 0
    session.motion_probe_sequence = 0
    session.plan_references = {}
    session.workspace_bounds_m = {
        "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
    }
    session._require_environment = lambda: SimpleNamespace()
    session._current_state = lambda: (
        np.array([0.4, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        0.2,
    )
    session._motion_state = lambda: {
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_positions": [0.0] * 7,
        "base_pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "hand_pose_world_xyzw": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        "grip_site_pose_world_xyzw": [
            0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0,
        ],
    }
    receipt = session._execute_planned_motion(_action(
        "move_free_space", {"arm": "right", "target_pose": _pose()},
    ))
    assert receipt["motion_status"] == "refused"
    assert receipt["state_changed"] is False
    assert receipt["steps_used"] == 0
    assert receipt["feedback"]["planning_failure_reason"] == reason
    assert receipt["planner"]["status"] == response["status"]
    assert receipt["planner"]["valid_query"] is valid_query
    execution_evidence = json.loads((
        tmp_path / receipt["planner"]["canonical_artifact"]["path"]
    ).read_text())
    assert execution_evidence["request"]["joint_positions"] == [0.0] * 7
    for sequence in range(2):
        result = session.probe_motion({
            "schema": "ora.motion-probe-request.v1",
            "observation_id": "observation-0", "observation_sha256": "0" * 64,
            "queries": [{"query_id": "query-0", "arm": "right", "pose": _pose()}],
            "variant_policy": "exact",
        })
        assert result["schema"] == "ora.motion-probe-result.v1"
        assert result["robot_decisions_consumed"] == 0
        assert result["world_state_changed"] is False
        row = result["results"][0]
        assert row["reachable"] is False and row["plan_ref"] is None
        variant = row["variants"][0]
        assert variant["failure"]["code"] == reason
        assert variant["planner"]["status"] == reason
        assert variant["planner"]["valid_query"] is valid_query
        if reason == "planner_error":
            assert variant["planner"]["error"] == {"stage": "planner", **response["error"]}
        probe_evidence = json.loads((
            tmp_path / f"curobo/probes/request-{sequence:06d}.json"
        ).read_text())
        assert probe_evidence == execution_evidence
    assert session.world_mutation_sequence == session.sim_steps == 0
    assert session.plan_references == {}


def test_full_joint_plan_is_canonical_but_not_policy_visible(tmp_path: Path) -> None:
    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.output_root = tmp_path
    session.action_sequence = 0
    summary = session._persist_planner_evidence({
        "schema": PLAN_RECORD_SCHEMA,
        "status": "planned", "reachable": True,
        "world_model": "none", "planner_status": "success",
        "trajectory": {
            "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
            "positions": [[0.0] * 7, [0.05] * 7], "dt_s": 0.02,
        },
    })
    assert summary["world_model"] == "none"
    assert summary["trajectory"]["waypoint_count"] == 2
    assert "positions" not in summary["trajectory"]
    artifact = tmp_path / summary["canonical_artifact"]["path"]
    assert "0.05" in artifact.read_text(encoding="utf-8")


def test_world_plan_summary_records_its_world_but_projects_none(tmp_path: Path) -> None:
    from ora.agent.session_core.receipt_projection import _public_planner_summary

    session = object.__new__(LiberoProSession)
    session.output_root = tmp_path
    session.action_sequence = 0
    world = {"world_model": "public_depth_voxel_v1", "sha256": "b" * 64, "inputs_sha256": "c" * 64,
             "occupied_voxels": 12}
    summary = session._persist_planner_evidence({
        "schema": PLAN_RECORD_SCHEMA, "status": "not_reachable", "reachable": False,
        "world_model": "public_depth_voxel_v1", "planner_status": "goal_unreachable", "world": world,
    })
    assert summary["world_model"] == "public_depth_voxel_v1" and summary["world"] == world
    assert "world" not in _public_planner_summary(summary)
    assert "world_model" not in _public_planner_summary(summary)


def test_each_plan_record_names_its_planner_but_the_summary_does_not(tmp_path: Path) -> None:
    """A move's canonical plan says which planner made it: countable."""
    from ora.agent.session_core.receipt_projection import _public_planner_summary

    session = object.__new__(LiberoProSession)
    session.output_root = tmp_path
    session.action_sequence, session.motion_planner = 0, SimpleNamespace(implementation="curobo_v2")
    summary = session._persist_planner_evidence({
        "schema": PLAN_RECORD_SCHEMA, "status": "planned", "reachable": True,
        "world_model": "none", "planner_status": "success"})
    record = json.loads((tmp_path / summary["canonical_artifact"]["path"]).read_text(encoding="utf-8"))
    assert record["planner_implementation"] == "curobo_v2"
    assert "planner_implementation" not in summary and "planner_implementation" not in _public_planner_summary(summary)


def test_motion_profile_is_hash_bound_and_gripper_is_fixed() -> None:
    binding = load_hybrid_motion_profile(MOTION_PROFILE, MOTION_PROFILE_SHA256)
    profile = binding["profile"]
    assert profile["gripper"]["open_directive"] == 0.0
    assert profile["gripper"]["close_directive"] == 1.0
    assert profile["servo"]["max_iterations"] == 20
    assert profile["sensing"]["force_telemetry_threshold_n"] == 40.0
    drifted = dict(profile)
    drifted["gripper"] = {**profile["gripper"], "close_directive": 0.5}
    with pytest.raises(HybridMotionProfileError):
        validate_hybrid_motion_profile(drifted)


def test_contact_aggregates_blocked_closure_without_payload_inference() -> None:
    assert aggregate_motion_contact(
        external_force_xyz=[0.0, 0.0, 5.0], force_threshold_n=10.0,
        closing=True, measured_aperture_normalized=0.4,
        closure_contact_min_normalized=0.08,
    ) is True
    assert aggregate_motion_contact(
        external_force_xyz=[0.0, 0.0, 5.0], force_threshold_n=10.0,
        closing=False, measured_aperture_normalized=0.4,
        closure_contact_min_normalized=0.08,
    ) is False


def test_joint_executor_stop_is_operator_owned_and_force_is_telemetry(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "control/stop.request"
    assert motion_stop_reason(
        stop_request_path=sentinel,
    ) is None
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("operator\n", encoding="utf-8")
    assert motion_stop_reason(
        stop_request_path=sentinel,
    ) == "operator_stop_requested"


def test_joint_position_command_is_bounded() -> None:
    command = bounded_joint_position_command(
        [0.0] * 7, [0.2, -0.2, 0.025, 0.0, 0.0, 0.0, 0.0],
        maximum_step_rad=0.05,
    )
    assert command == pytest.approx([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0])


def test_backtrack_state_follows_each_non_recovery_action_start() -> None:
    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.last_backtrack_joint_state = None
    state = {"joint_positions": [0.1] * 7}
    session._motion_state = lambda: dict(state)
    session._remember_backtrack_state()
    assert session.last_backtrack_joint_state == [0.1] * 7
    state["joint_positions"] = [-0.2] * 7
    session._remember_backtrack_state()
    assert session.last_backtrack_joint_state == [-0.2] * 7


def test_the_observation_delta_no_longer_compares_motion_layers(
    tmp_path: Path,
) -> None:
    """Two actions are never judged equivalent by the Harness.

    The retired guard decided whether a `move_free_space`, `servo_delta` or
    `recover_motion` proposal repeated the previous one, using position and
    orientation epsilons that were bound to no profile. That comparison, and
    the repetition streak built on it, are gone; what the Supervisor records
    after an action is one image distance.
    """

    guard = StagnationGuard(tmp_path / "stagnation.json")
    assert not hasattr(guard, "_equivalent")
    assert not hasattr(guard, "is_high_confidence")
    assert not any(
        hasattr(guard, name) for name in (
            "position_epsilon_m", "orientation_epsilon_deg", "gripper_epsilon",
            "eef_motion_epsilon_m", "image_phash_hamming_max",
            "warning_at", "high_confidence_at",
        )
    )
