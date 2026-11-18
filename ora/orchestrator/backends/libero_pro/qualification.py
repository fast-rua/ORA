"""Scripted LIBERO sparse-motion qualification through the real worker path.

The record this writes is what ``definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS``
cites for ``("motion_action_v3", "libero_pro")``: a closed table of cases against
the real Panda plant through the public motion action, with no policy and no
provider. Case construction and the pass predicates read only public receipt
fields, so they are unit tested; ``qualify`` drives one worker.

The worker needs a LIBERO run's own bindings, so the tool starts from a run that
passed preflight on this host: its ``plan.json`` gives the runtime, task, seed and
step budget, and its ``preflight.json`` the controller and timing bindings the
Supervisor would pass. ``--worker-env NAME=VALUE`` replaces a lane's value when
the qualification runs on another lane.

The large-rotation cases are the 2 pi alias monitors: a near-180-degree request
stays legal and may honestly end ``not_reached``, but must never realize a
translation drift past the bound, the signature of an aliased IK branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.environment.controllers.hybrid_motion import libero_global_endpoint_tolerances

SCHEMA = "ora.libero-sparse-motion-qualification.v1"
RECORD_NAME = "libero-sparse-motion-qualification.json"
LIBERO_SPARSE_RECEIPT_SCHEMA = "ora.libero-sparse-motion-receipt.v1"
MOTION_ACTION_V3_SCHEMA = "ora-motion-action.v3"

# Alias-monitor drift bound: a pure rotation that realizes more translation
# than this is treated as an aliased or degenerate execution (M3 monitor).
ROTATION_ONLY_DRIFT_BOUND_M = 0.03
REACH_POSITION_TOLERANCE_M = 0.01
# All global routes share the 15 mm / 5 degree qualification bound. Local OSC
# execution retains its tighter action-contract tolerances.
FREE_SPACE_POSITION_TOLERANCE_M, REACH_ORIENTATION_TOLERANCE_RAD = (
    libero_global_endpoint_tolerances()
)
#: The route a v4 free-space move takes when its plan is clear and nearly straight (``controllers/osc_transit.py``).
OSC_LINE_ROUTE = "osc_straight_line"
#: The planner-neutral reasons of a refused goal (``controllers/motion_planner.py``).
PLANNING_REFUSAL_CODES = frozenset({"goal_unreachable", "goal_in_collision", "goal_below_support", "path_blocked"})


def _rotate_local_z(orientation_xyzw: Sequence[float], degrees: float) -> list[float]:
    """Compose a local-frame rotation about the grip-site approach axis."""

    half = math.radians(degrees) / 2.0
    rx, ry, rz, rw = 0.0, 0.0, math.sin(half), math.cos(half)
    x, y, z, w = (float(value) for value in orientation_xyzw)
    return [
        w * rx + rw * x + (y * rz - z * ry),
        w * ry + rw * y + (z * rx - x * rz),
        w * rz + rw * z + (x * ry - y * rx),
        w * rw - (x * rx + y * ry + z * rz),
    ]


def _target(position: Sequence[float], orientation: Sequence[float],
            interaction: str) -> dict[str, Any]:
    return {
        "targets": {"right": {
            "position_m": [round(float(v), 6) for v in position],
            "orientation_xyzw": [round(float(v), 8) for v in orientation],
        }},
        "interactions": {"right": interaction},
    }


def build_case_arguments(
    case_id: str, *, position: Sequence[float], orientation: Sequence[float],
) -> dict[str, Any]:
    """Compile one case's v3 arguments from the measured start pose."""

    x, y, z = (float(v) for v in position)
    quat = [float(v) for v in orientation]
    if case_id == "free_space_translation_10cm":
        return _target([x - 0.10, y + 0.05, z + 0.05], quat, "free_space")
    if case_id == "combined_translation_rotation_45deg":
        return _target(
            [x, y - 0.08, z + 0.06], _rotate_local_z(quat, 45.0), "free_space",
        )
    if case_id == "orientation_only_small_12deg":
        # Comfortably inside the 15-degree local envelope: a boundary-exact
        # request routes either way under microscopic passive drift between
        # observation and execution, so the routing case stays interior.
        return _target([x, y, z], _rotate_local_z(quat, 12.0), "precision")
    if case_id == "orientation_only_90deg":
        return _target([x, y, z], _rotate_local_z(quat, 90.0), "free_space")
    if case_id == "orientation_only_near_180deg":
        return _target([x, y, z], _rotate_local_z(quat, 179.0), "free_space")
    if case_id == "precision_small_local_6mm":
        return _target([x, y, z + 0.006], quat, "precision")
    if case_id == "contact_approach_small_8mm":
        return _target([x, y, z - 0.008], quat, "contact_approach")
    if case_id == "contact_approach_large_staged":
        # Horizontal, so the staged route moves through free air: a downward
        # approach this long would end in the table. Lateral, because the
        # plan_ref cases that follow step along -x from wherever this leaves
        # the arm, and an x offset here compounds with theirs.
        return _target([x, y + 0.10, z], quat, "contact_approach")
    if case_id == "planner_refusal_unreachable":
        # Inside the action workspace, kinematically hopeless for a mounted
        # Panda: full stretch to a far corner with an inverted approach.
        return _target(
            [0.58, -0.48, 1.28], _rotate_local_z([1.0, 0.0, 0.0, 0.0], 90.0),
            "free_space",
        )
    if case_id == "below_tolerance_noop":
        return _target([x, y, z], quat, "precision")
    raise ValueError(f"unknown qualification case: {case_id}")


MOTION_CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "free_space_translation_10cm",
     "expected_status": {"reached"},
     "expected_executor": "libero_curobo_joint_position"},
    {"case_id": "combined_translation_rotation_45deg",
     "expected_status": {"reached"},
     "expected_executor": "libero_curobo_joint_position"},
    {"case_id": "orientation_only_small_12deg",
     "expected_status": {"reached"},
     "expected_executor": "libero_robosuite_osc_pose",
     "rotation_only": True},
    {"case_id": "orientation_only_90deg",
     "expected_status": {"reached", "not_reached"},
     "expected_executor": "libero_curobo_joint_position",
     "rotation_only": True},
    {"case_id": "orientation_only_near_180deg",
     "expected_status": {"reached", "not_reached", "planner_failed"},
     "expected_executor": "libero_curobo_joint_position",
     "rotation_only": True},
    {"case_id": "precision_small_local_6mm",
     "expected_status": {"reached"},
     "expected_executor": "libero_robosuite_osc_pose"},
    {"case_id": "contact_approach_small_8mm",
     "expected_status": {"reached", "not_reached"},
     "expected_executor": "libero_robosuite_osc_pose"},
    {"case_id": "contact_approach_large_staged",
     "expected_status": {"reached", "not_reached"},
     "expected_executor": "libero_robosuite_osc_pose",
     "expected_strategy": "global_precontact_then_local"},
    {"case_id": "planner_refusal_unreachable",
     "expected_status": {"planner_failed"},
     "expected_executor": None,
     "expected_failure_codes": PLANNING_REFUSAL_CODES,
     "expect_no_write": True},
    {"case_id": "below_tolerance_noop",
     "expected_status": {"reached"},
     "expected_executor": None},
)

PLAN_REF_CASES: tuple[str, ...] = (
    "probe_then_plan_ref_execution",
    "plan_ref_survives_gripper_only",
    "plan_ref_expires_after_motion",
    "jaw_flip_variants_probed",
)

ALL_CASE_IDS: tuple[str, ...] = tuple(
    [case["case_id"] for case in MOTION_CASES] + list(PLAN_REF_CASES)
)


def judge_motion_case(
    case: Mapping[str, Any], receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Pass/fail one executed case using only public receipt fields."""

    notes: list[str] = []
    passed = True
    if receipt.get("schema") != LIBERO_SPARSE_RECEIPT_SCHEMA:
        return {
            "passed": False,
            "notes": [f"unexpected receipt schema: {receipt.get('schema')}"],
        }
    def reject(note: str) -> None:
        nonlocal passed
        passed = False
        notes.append(note)

    def finite_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)
        )

    if not isinstance(receipt.get("state_changed"), bool):
        reject("missing or malformed state_changed evidence")
    steps = receipt.get("steps_used")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        reject("missing or malformed steps_used evidence")
    if not isinstance(receipt.get("terminal_after_action"), bool):
        reject("missing or malformed terminal evidence")
    elif receipt["terminal_after_action"]:
        reject("qualification action unexpectedly terminated the environment")
    poses = []
    for key in ("before", "after"):
        side = receipt.get(key)
        right = side.get("right") if isinstance(side, Mapping) else None
        pose = right.get("pose_xyz_xyzw") if isinstance(right, Mapping) else None
        if not isinstance(pose, list) or len(pose) != 7 or not all(
            finite_number(value) for value in pose
        ):
            reject(f"missing or malformed {key} pose")
        else:
            poses.append(pose)
    segment = receipt.get("segment")
    planner = segment.get("planner") if isinstance(segment, Mapping) else None
    if receipt.get("failure_code") == "planner_error" or (
        isinstance(planner, Mapping) and planner.get("status") == "planner_error"
    ):
        reject("planner implementation error is not motion qualification")
    status = str(receipt.get("execution_status"))
    if status not in case["expected_status"]:
        passed = False
        notes.append(
            f"status {status} not in {sorted(case['expected_status'])}"
        )
    expected_executor = case.get("expected_executor")
    executor = receipt.get("executor")
    route = ((receipt.get("arms") or {}).get("right") or {}).get("route_choice")
    if (expected_executor == "libero_curobo_joint_position" and isinstance(route, Mapping)
            and route.get("route") == OSC_LINE_ROUTE):
        # Motion profile v4 runs a clear, nearly straight free-space plan as the OSC line: the
        # planner's plan is still the move's evidence, and the receipt names the route and why.
        planner_record = route.get("planner") if isinstance(route.get("planner"), Mapping) else {}
        if route.get("reason") != "planner_path_clear_and_straight" or not planner_record.get("canonical_artifact"):
            reject("osc line route lacks its planner evidence and reason")
        expected_executor = "libero_robosuite_osc_pose"
    if expected_executor is not None and status in {"reached", "not_reached"} and (
        not isinstance(executor, list) or expected_executor not in executor
    ):
        passed = False
        notes.append(f"executor {executor} lacks {expected_executor}")
    if case.get("expected_strategy") is not None and (
        receipt.get("motion_strategy") != case["expected_strategy"]
    ):
        reject(f"motion_strategy {receipt.get('motion_strategy')}")
    if case.get("expected_failure_code") is not None and (
        receipt.get("failure_code") != case["expected_failure_code"]
    ):
        passed = False
        notes.append(f"failure_code {receipt.get('failure_code')}")
    if case.get("expected_failure_codes") is not None and (
        receipt.get("failure_code") not in case["expected_failure_codes"]
    ):
        reject(f"unexpected planner failure_code {receipt.get('failure_code')}")
    if status == "planner_failed":
        if receipt.get("failure_stage") != "planner" or (
            receipt.get("failure_code") not in PLANNING_REFUSAL_CODES
        ):
            reject("planner refusal lacks a recognized ordinary planning failure")
        if not isinstance(planner, Mapping) or (
            planner.get("status") != "not_reachable"
            or planner.get("reachable") is not False
            or planner.get("valid_query") is not True
            or planner.get("planner_status") != receipt.get("failure_code")
        ):
            reject("planner refusal lacks valid-start, non-execution planner evidence")
        artifact = planner.get("canonical_artifact") if isinstance(planner, Mapping) else None
        if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str) or (
            not artifact["path"] or not isinstance(artifact.get("sha256"), str)
            or len(artifact["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in artifact["sha256"])
        ):
            reject("planner refusal lacks canonical artifact identity")
    if case.get("expect_no_write") or status in {"planner_failed", "plan_rejected"}:
        if receipt.get("state_changed") is not False or steps != 0:
            reject("refusal wrote the world or lacks zero-step evidence")
        if len(poses) != 2 or poses[0] != poses[1]:
            reject("refusal before/after poses differ or are missing")
    arms = receipt.get("arms")
    right = arms.get("right") if isinstance(arms, Mapping) else None
    pose_error = right.get("pose_error") if isinstance(right, Mapping) else None
    if status == "reached" and (
        not isinstance(pose_error, Mapping) or any(
            not finite_number(pose_error.get(key)) or pose_error[key] < 0
            for key in ("translation_m", "orientation_rad")
        )
    ):
        reject("reached result lacks finite measured pose error")
    elif status == "reached":
        position_tolerance = (
            FREE_SPACE_POSITION_TOLERANCE_M
            if case.get("expected_executor") == "libero_curobo_joint_position"
            else REACH_POSITION_TOLERANCE_M
        )
        if float(pose_error["translation_m"]) > position_tolerance:
            passed = False
            notes.append(f"reached with {pose_error['translation_m']:.4f} m residual")
        if float(
            pose_error["orientation_rad"]
        ) > REACH_ORIENTATION_TOLERANCE_RAD:
            passed = False
            notes.append(
                f"reached with {pose_error['orientation_rad']:.4f} rad residual"
            )
    if case.get("rotation_only"):
        if len(poses) == 2:
            before, after = poses
            drift = math.sqrt(sum(
                (float(after[index]) - float(before[index])) ** 2
                for index in range(3)
            ))
            if drift > ROTATION_ONLY_DRIFT_BOUND_M:
                passed = False
                notes.append(
                    f"2pi-alias monitor: pure rotation realized {drift:.3f} m "
                    "translation drift"
                )
        else:
            passed = False
            notes.append("rotation-only case lacks before/after poses")
    return {"passed": passed, "notes": notes}


def summarize_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Public, prompt-safe summary persisted per case."""

    arms = receipt.get("arms") if isinstance(receipt.get("arms"), Mapping) else {}
    right = arms.get("right") if isinstance(arms.get("right"), Mapping) else {}
    return {
        "schema": receipt.get("schema"),
        "execution_status": receipt.get("execution_status"),
        "motion_strategy": receipt.get("motion_strategy"),
        "failure_stage": receipt.get("failure_stage"),
        "failure_code": receipt.get("failure_code"),
        "executor": receipt.get("executor"),
        "state_changed": receipt.get("state_changed"),
        "pose_error": right.get("pose_error"),
        "endpoint_contract": right.get("endpoint_contract"),
        "plan_ref": right.get("plan_ref"),
        "steps_used": receipt.get("steps_used"),
        "terminal_after_action": receipt.get("terminal_after_action"),
        "planner": (
            receipt["segment"].get("planner")
            if isinstance(receipt.get("segment"), Mapping) else None
        ),
    }


def build_qualification_record(
    *, source_commit: str, host: Mapping[str, Any],
    session_identity: Mapping[str, Any],
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_ids = [str(result.get("case_id")) for result in case_results]
    executed = set(case_ids)
    missing = [case_id for case_id in ALL_CASE_IDS if case_id not in executed]
    record = {
        "schema": SCHEMA,
        "source_commit": source_commit,
        "host": dict(host),
        "session": dict(session_identity),
        "cases": [dict(result) for result in case_results],
        "missing_cases": missing,
        "invalid_case_inventory": (
            len(case_ids) != len(executed) or bool(executed - set(ALL_CASE_IDS))
        ),
        "passed": not missing and len(case_ids) == len(ALL_CASE_IDS) and all(
            result.get("passed") is True for result in case_results
        ),
        "created_unix_s": time.time(),
    }
    body = {key: value for key, value in record.items() if key != "created_unix_s"}
    record["record_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return record


def _v3_action(step: int, observation: Mapping[str, Any],
               arguments: Mapping[str, Any], name: str = "move_pose") -> dict[str, Any]:
    return {
        "schema": MOTION_ACTION_V3_SCHEMA,
        "observation_id": str(observation["observation_id"]),
        "action_id": f"qualification-{step:04d}",
        "bound_observation_sha256": str(observation["observation_sha256"]),
        "action": name,
        "arguments": dict(arguments),
    }


def _qualification_probe_row(
    probed: Mapping[str, Any], *, policy: str, require_plan: bool,
) -> Mapping[str, Any]:
    """Require actual, valid-query planner calls, not just variant placeholders."""
    if (
        probed.get("schema") != "ora.motion-probe-result.v1"
        or probed.get("world_state_changed") is not False
        or type(probed.get("robot_decisions_consumed")) is not int
        or probed["robot_decisions_consumed"] != 0
        or probed.get("variant_policy") != policy
    ):
        raise ValueError("probe lacks non-mutating qualification evidence")
    rows = probed.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError("probe omitted its query result")
    row = rows[0]
    variants = row.get("variants")
    expected = {"variant-00-exact-+0deg"}
    if policy == "parallel_jaw_equivalents":
        expected.add("variant-01-jaw-flip-180deg")
    if not isinstance(variants, list) or len(variants) != len(expected) or any(
        not isinstance(variant, Mapping) for variant in variants
    ) or {variant.get("variant_id") for variant in variants} != expected:
        raise ValueError("probe variant inventory is incomplete or duplicated")
    admitted = []
    for variant in variants:
        planner = variant.get("planner")
        if not isinstance(planner, Mapping) or (
            planner.get("called") is not True or planner.get("valid_query") is not True
            or planner.get("error") is not None
        ):
            raise ValueError("probe lacks a valid-query, implementation-error-free planner call")
        if variant.get("reachable") is True:
            if planner.get("status") not in {"success", "SUCCESS", "MotionGenStatus.SUCCESS"} or (
                not isinstance(variant.get("plan_ref"), str) or not variant["plan_ref"]
            ):
                raise ValueError("probe admitted a plan without successful planner evidence")
            admitted.append(variant)
        elif variant.get("reachable") is False:
            failure = variant.get("failure")
            if planner.get("status") not in PLANNING_REFUSAL_CODES or (
                not isinstance(failure, Mapping) or failure.get("code") != planner["status"]
            ):
                raise ValueError("probe refusal is not an ordinary planning failure")
        else:
            raise ValueError("probe variant omitted typed reachability")
    if row.get("reachable") is not bool(admitted) or (
        require_plan and not admitted
    ):
        raise ValueError("probe omitted the required admitted plan")
    if admitted and not any(
        variant["variant_id"] == row.get("selected_variant_id")
        and variant["plan_ref"] == row.get("plan_ref") for variant in admitted
    ):
        raise ValueError("probe selected plan does not match an admitted variant")
    return row


def run_cases(
    *, observe: Callable[[], Mapping[str, Any]],
    execute: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    probe: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    current_pose: Callable[[Mapping[str, Any]], Sequence[float]],
) -> list[dict[str, Any]]:
    """Drive every case through the injected live boundary.

    ``observe`` returns {observation_id, observation_sha256, ...} and the
    fresh public observation; ``current_pose`` extracts the measured right
    grip-site pose [x,y,z,qx,qy,qz,qw] from it. Recovery to a home-adjacent
    configuration between cases is the caller's responsibility via the same
    public recover_pose contract when a case ends displaced.
    """

    results: list[dict[str, Any]] = []
    step = 0
    for case in MOTION_CASES:
        observation = observe()
        pose = [float(v) for v in current_pose(observation)]
        arguments = build_case_arguments(
            str(case["case_id"]), position=pose[:3], orientation=pose[3:],
        )
        step += 1
        try:
            receipt = execute(_v3_action(step, observation, arguments))
            judgement = judge_motion_case(case, receipt)
            results.append({
                "case_id": case["case_id"], **judgement,
                "receipt": summarize_receipt(receipt),
            })
        except Exception as error:  # recorded, never silently dropped
            results.append({
                "case_id": case["case_id"], "passed": False,
                "notes": [f"execution error: {error}"], "receipt": None,
            })

    # plan_ref lifecycle cases
    def _probe_request(observation: Mapping[str, Any], pose: Sequence[float],
                       policy: str) -> dict[str, Any]:
        return {
            "schema": "ora.motion-probe-request.v1",
            "observation_id": str(observation["observation_id"]),
            "observation_sha256": str(observation["observation_sha256"]),
            "variant_policy": policy,
            "queries": [{
                "query_id": "q0", "arm": "right",
                "pose": {
                    "position_m": [
                        float(pose[0]) - 0.06, float(pose[1]), float(pose[2]),
                    ],
                    "orientation_xyzw": [float(v) for v in pose[3:]],
                },
            }],
        }

    try:
        observation = observe()
        pose = [float(v) for v in current_pose(observation)]
        probed = probe(_probe_request(observation, pose, "exact"))
        row = _qualification_probe_row(probed, policy="exact", require_plan=True)
        plan_ref = row.get("plan_ref")
        step += 1
        receipt = execute(_v3_action(
            step, observation, {"plan_ref": plan_ref},
        ))
        judgement = judge_motion_case(
            {"case_id": "probe_then_plan_ref_execution",
             "expected_status": {"reached"},
             "expected_executor": "libero_curobo_joint_position"},
            receipt,
        )
        passed = judgement["passed"] and bool(row.get("reachable"))
        results.append({
            "case_id": "probe_then_plan_ref_execution", "passed": passed,
            "notes": judgement["notes"], "receipt": summarize_receipt(receipt),
        })
    except Exception as error:
        results.append({
            "case_id": "probe_then_plan_ref_execution", "passed": False,
            "notes": [f"execution error: {error}"], "receipt": None,
        })

    try:
        observation = observe()
        pose = [float(v) for v in current_pose(observation)]
        probed = probe(_probe_request(observation, pose, "exact"))
        plan_ref = _qualification_probe_row(
            probed, policy="exact", require_plan=True,
        )["plan_ref"]
        step += 1
        execute(_v3_action(step, observation, {
            "targets": {"right": "close"},
        }, name="set_gripper"))
        fresh = observe()
        step += 1
        receipt = execute(_v3_action(step, fresh, {"plan_ref": plan_ref}))
        judgement = judge_motion_case(
            {"expected_status": {"reached"},
             "expected_executor": "libero_curobo_joint_position"}, receipt,
        )
        passed = judgement["passed"]
        results.append({
            "case_id": "plan_ref_survives_gripper_only", "passed": passed,
            "notes": judgement["notes"],
            "receipt": summarize_receipt(receipt),
        })
    except Exception as error:
        results.append({
            "case_id": "plan_ref_survives_gripper_only", "passed": False,
            "notes": [f"execution error: {error}"], "receipt": None,
        })

    try:
        observation = observe()
        pose = [float(v) for v in current_pose(observation)]
        probed = probe(_probe_request(observation, pose, "exact"))
        plan_ref = _qualification_probe_row(
            probed, policy="exact", require_plan=True,
        )["plan_ref"]
        step += 1
        execute(_v3_action(step, observation, _target(
            [pose[0], pose[1], pose[2] + 0.006], pose[3:], "precision",
        )))
        stale = observe()
        step += 1
        try:
            execute(_v3_action(step, stale, {"plan_ref": plan_ref}))
            results.append({
                "case_id": "plan_ref_expires_after_motion", "passed": False,
                "notes": ["stale plan executed"], "receipt": None,
            })
        except Exception as error:
            # The worker's error crosses the RPC boundary with its message.
            expected = "plan_ref is stale for the current world state" in str(error)
            results.append({
                "case_id": "plan_ref_expires_after_motion", "passed": expected,
                "notes": ["stale plan_ref failed closed"] if expected else [
                    f"unexpected stale-plan error: {type(error).__name__}: {error}"
                ], "receipt": None,
            })
    except Exception as error:
        results.append({
            "case_id": "plan_ref_expires_after_motion", "passed": False,
            "notes": [f"setup error: {error}"], "receipt": None,
        })

    try:
        observation = observe()
        pose = [float(v) for v in current_pose(observation)]
        probed = probe(_probe_request(
            observation, pose, "parallel_jaw_equivalents",
        ))
        _qualification_probe_row(
            probed, policy="parallel_jaw_equivalents", require_plan=False,
        )
        results.append({
            "case_id": "jaw_flip_variants_probed", "passed": True,
            "notes": [],
            "receipt": None,
        })
    except Exception as error:
        results.append({
            "case_id": "jaw_flip_variants_probed", "passed": False,
            "notes": [f"execution error: {error}"], "receipt": None,
        })
    return results


def _worker_environment(values: Sequence[str]) -> dict[str, str]:
    pairs = [value.partition("=") for value in values]
    if any(not separator or not name for name, separator, _ in pairs):
        raise ValueError("--worker-env takes NAME=VALUE")
    return {name: value for name, _, value in pairs}


def _lane_runtime(runtime: Mapping[str, Any], lane: Mapping[str, str]) -> dict[str, Any]:
    """The run's runtime on this lane: ``lane`` replaces values of its worker environment."""
    binding = dict(runtime.get("resource_binding") or {})
    binding["worker_environment"] = {**(binding.get("worker_environment") or {}), **lane}
    return {**runtime, "resource_binding": binding}


def qualify(*, run_root: Path, output_root: Path, lane: Mapping[str, str]) -> dict[str, Any]:
    """Drive the case table through one worker built from ``run_root``'s plan and preflight."""
    from ora.environment.factory import build_environment_session
    from ora.orchestrator.backends.libero_pro.supervision import supervisor_binding
    from ora.orchestrator.episode.execution_state import bind_execution_state

    run_root, output_root = run_root.resolve(), output_root.resolve()
    plan = json.loads((run_root / "plan.json").read_text(encoding="utf-8"))
    runtime = _lane_runtime(plan["runtime"], lane)
    output_root.mkdir(parents=True, exist_ok=False)
    session = build_environment_session(
        runtime_config=runtime, output_root=output_root / "native", task=plan["task"], seed=plan["seed"],
        max_sim_steps=plan["budgets"]["max_sim_steps"], action_codec="motion_action_v3",
        qualified_runtime_binding=supervisor_binding(run_root, runtime),
    )
    bind_execution_state(session)
    observation_root = output_root / "native" / "observations"

    def observe() -> Mapping[str, Any]:
        observation, _images = session.observe(observation_root)
        # The Runtime's identity of an observation: the hash of its persisted file.
        manifest = observation_root / str(observation["observation_id"]) / "observation.json"
        return {**observation, "observation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}

    try:
        session.start()
        results = run_cases(
            observe=observe, execute=session.execute, probe=session.probe_motion,
            current_pose=lambda observation: observation["proprioception"]["end_effectors"]["right"]["pose_xyz_xyzw"],
        )
    finally:
        session.close()
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                   check=True).stdout.strip()
    record = build_qualification_record(
        source_commit=source_commit, host={"required_host": runtime.get("required_host")},
        session_identity={"task": plan["task"], "seed": plan["seed"], "suite": runtime.get("suite"),
                          "world_model": (runtime.get("motion_planner") or {}).get("world_model"),
                          "planner": (runtime.get("motion_planner") or {}).get("implementation")},
        case_results=results,
    )
    path = output_root / RECORD_NAME
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**record, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="a LIBERO run that passed preflight on this host")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-env", action="append", default=[], metavar="NAME=VALUE",
                        help="replace a value of the run's lane environment (repeatable)")
    args = parser.parse_args(argv)
    record = qualify(run_root=args.run, output_root=args.output, lane=_worker_environment(args.worker_env))
    for item in record["cases"]:
        print(("PASS " if item["passed"] else "FAIL ") + str(item["case_id"]) + " " + "; ".join(item.get("notes", [])))
    print(json.dumps({"passed": record["passed"], "missing": record["missing_cases"],
                      "path": record["path"], "sha256": record["sha256"]}))
    return 0 if record["passed"] else 1


__all__ = [
    "ALL_CASE_IDS", "MOTION_CASES", "PLAN_REF_CASES", "RECORD_NAME", "SCHEMA", "build_case_arguments",
    "build_qualification_record", "judge_motion_case", "main", "qualify", "run_cases", "summarize_receipt",
]
