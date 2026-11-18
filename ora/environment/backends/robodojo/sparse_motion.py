"""Sparse motion v3 on RoboDojo: the evidence and budget every executed action shares, and the read-only probe.

A move is a ``follow_joint_trajectory`` the toolkit planned (``trajectory``), a
jaw command a ``set_gripper`` (``gripper_ramp`` for the rate-limited form, the
open/close form here) and a hold a ``wait``; ``_SparseAction`` finishes every
receipt. The SDK executor that solved one IK endpoint per segment and
interpolated the joints to it is gone, so ``move_pose``, ``move_free_space``
and ``recover_motion`` are refused before anything is written. Receipts sealed
on that route keep their schema; nothing here reads them.
"""

from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.planning_diagnostics import PROBE_FEEDBACK_REQUEST_SCHEMA, PROBE_IK_CANDIDATE_FEEDBACK
from ora.contracts.refusal import EnvironmentRefusal
from ora.contracts.robodojo_arx import ARX_X5_CONTRACT

from .bridge import RobodojoError
from .ik import solve_ik
from ora.environment.controllers.gripper_stall import (
    GRIPPER_STALL_REASON, gripper_outcome, gripper_outcome_state, gripper_stalled,
)

from .common import (
    ARX_GRIPPER_STALL, GRIPPER_TOLERANCE, ORIENTATION_TOLERANCE_RAD, POSITION_TOLERANCE_M, array,
    gripper_opening, physical_gripper, pose_error, primary_limit_holds_close, workspace_violation,
)


SPARSE_RECEIPT_SCHEMA = "ora.robodojo-sparse-motion-receipt.v1"
PROBE_REQUEST_SCHEMA = "ora.motion-probe-request.v1"
PROBE_RESULT_SCHEMA = "ora.motion-probe-result.v1"
GRIPPER_EXECUTOR = "robodojo_sdk_gripper_actuator"
#: Bounds a runtime may still name: ``session.__init__`` validates all three, and
#: the two joint-jump bounds limit ``ik.solve_ik``.
FREE_SPACE_CLEARANCE_M = 0.05
LOCAL_JOINT_JUMP_LIMIT_RAD = 1.0
# IK endpoints are not paths. Preserve the half-turn raw-coordinate bound
# after legal equivalent-coordinate selection; a live 179-degree probe once
# returned a 6.5 rad jump and drove the arm 0.5 m off.
GLOBAL_JOINT_JUMP_LIMIT_RAD = math.pi
#: The staged contact route a runtime may name (``session.__init__`` validates it);
#: the toolkit chooses every route now (``lib.motion_plan.route``).
STAGED_CONTACT_STRATEGY = "global_precontact_then_local"
ARMS = ("left", "right")
# Parallel-jaw symmetry axis: joint7/joint8 close along link6 +/-Y from an
# origin 0.087 m along link6 +X (X5A.urdf), so the approach axis is link6 +X.
JAW_SYMMETRY_AXIS = (1.0, 0.0, 0.0)


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    value = np.asarray([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise RobodojoError("rotation composition produced an invalid orientation")
    return (value / norm).tolist()


def rotate_about_local_axis(orientation_xyzw: Sequence[float], axis: Sequence[float], angle_rad: float) -> list[float]:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    half = angle_rad / 2.0
    delta = [*(unit * math.sin(half)).tolist(), math.cos(half)]
    return _quaternion_multiply(orientation_xyzw, delta)


def gripper_closed_floor(session: Any, arm: str) -> float:
    """Normalized opening at the primary finger joint's lower limit.

    The command scale starts below the X5 finger joint's lower limit, so a
    ``close`` directive of 0.0 is never measurable; the plant stops at the
    articulation's own limit and that is the honest closed target.
    """
    manager, robot = session.environment.robot_manager, session.robots[arm]
    data = manager.robot_key[manager.robot_list.index(robot)].data
    for name in ("joint_pos_limits", "joint_limits"):
        if hasattr(data, name):
            limits = array(getattr(data, name))[0]
            return gripper_opening(robot, float(limits[int(robot.gripper_joint_indices[0])][0]))
    return 0.0


class _SparseAction:
    """Evidence and budget of one sparse v3 action across its native ticks."""

    def __init__(self, session: Any, action: Mapping[str, Any], arms: Sequence[str]) -> None:
        self.session = session
        self.action = action
        self.arms = tuple(arms)
        self.manager = session.environment.robot_manager
        self.before = session._proprioception()["end_effectors"]
        self.command_before = deepcopy(self.manager.control_manager.prev_control[0])
        if getattr(session.environment, "interact", False):
            from .interaction_control import policy_targets
            self.command_before = policy_targets(session, self.command_before)
        self.held_targets: dict[str, Any] = deepcopy(self.command_before)
        self.pose_targets_before = deepcopy(session.pose_targets)
        self.gripper_directives_before = deepcopy(session.gripper_directives)
        self.native_before = int(session.environment.take_action_cnt[0])
        self.steps_before = session.sim_steps
        self.started = time.time()
        self.budget = min(int(session.native_actions_per_command),
                          getattr(session, "request_action_budget", 20))
        self.used = 0
        self.tracking: list[dict[str, Any]] = []
        self.segments: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        self.journal = session._open_journal()

    def begin(self, controlled_keys: Sequence[str]) -> None:
        self.held_targets = {key: value for key, value in self.command_before.items() if key not in set(controlled_keys)}

    def arm_key(self, arm: str) -> str:
        return self.manager.process_name(self.session.robots[arm].arm_name)

    def gripper_key(self, arm: str) -> str:
        return self.manager.process_name(self.session.robots[arm].gripper_name)

    def remaining(self) -> int:
        return self.budget - self.used

    def run_gripper(self, directives: Mapping[str, float], effective: Mapping[str, float]) -> dict[str, str]:
        """Drive the jaws until each arrives or stalls; name how each one ended.

        A jaw stopped on an object holds the SDK-limited drive target 0.2 below
        its primary finger's encoder and never arrives, so the command ends once
        every targeted jaw has arrived or stalled, both read on the published
        two-finger mean (E11G12-001), instead of spending the budget.
        """
        control = {self.gripper_key(arm): {"position": physical_gripper(self.session.robots[arm], value)}
                   for arm, value in directives.items()}
        traces = {arm: [float(self.before[arm]["gripper"])] for arm in effective}

        def arrived(current: Mapping[str, Any], arm: str) -> bool:
            return abs(current[arm]["gripper"] - effective[arm]) <= GRIPPER_TOLERANCE

        def settled(current: Mapping[str, Any]) -> bool:
            # Called exactly once after every native action.
            for arm in effective:
                traces[arm].append(float(current[arm]["gripper"]))
            return all(arrived(current, arm) or gripper_stalled(
                traces[arm], target=effective[arm], rule=ARX_GRIPPER_STALL) for arm in effective)

        budget = self.remaining()
        used = self.session._run_native_control(
            action_id=str(self.action["action_id"]), control=control, held_targets=self.held_targets,
            journal=self.journal, tracking=self.tracking, segment="gripper", reached=settled,
            budget=budget,
        )
        self.used += used
        after = self.session._proprioception()["end_effectors"]
        return {arm: gripper_outcome_state(
            reached=arrived(after, arm),
            stalled=gripper_stalled(traces[arm], target=effective[arm], rule=ARX_GRIPPER_STALL),
            ticks_used=used, ticks_budget=budget,
        ) for arm in effective}

    def finish(
        self, *, status: str, strategy: str | None, stage: str | None, code: str | None,
        executor: list[str] | None, target_poses: Mapping[str, Sequence[float]] | None = None,
        gripper_targets: Mapping[str, float] | None = None, gripper_states: Mapping[str, str] | None = None,
        arm_extras: Mapping[str, Mapping[str, Any]] | None = None, extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.session
        after = session._proprioception()["end_effectors"]
        command_after = deepcopy(self.manager.control_manager.prev_control[0])
        if getattr(session.environment, "interact", False):
            from .interaction_control import policy_targets
            command_after = policy_targets(session, command_after)
        held_verified = session._held_targets_unchanged(self.held_targets)
        if not held_verified:
            session._fault("robodojo_held_command_changed")
            raise RobodojoError("action changed a held SDK command")
        pose_action = gripper_targets is None
        hold = not self.arms  # ``wait``: no dimension is controlled, every one is held
        gripper_floors = {arm: gripper_closed_floor(session, arm) for arm in (gripper_targets or {})}
        arms_effect: dict[str, Any] = {}
        for arm in ARMS:
            targeted = arm in self.arms
            effect: dict[str, Any] = {
                "targeted": targeted, "interaction": None,
                "pose_error": None, "gripper_error": None, "plan_validation": None,
                "plan_ref": None, "segments": self.segments[arm],
                "inactive_arm_drift": None,
            }
            if targeted and target_poses is not None and arm in target_poses:
                distance, angle = pose_error(target_poses[arm], after[arm]["pose_xyz_xyzw"])
                effect["pose_error"] = {"translation_m": distance, "orientation_rad": angle}
            if targeted and gripper_targets is not None and arm in gripper_targets:
                effect["gripper_error"] = abs(after[arm]["gripper"] - gripper_targets[arm])
            if targeted and gripper_states and arm in gripper_states:
                effect["gripper_outcome"] = _gripper_outcome(state=gripper_states[arm], after=after[arm],
                                                             budget=self.budget, used=self.used,
                                                             robot=session.robots[arm])
            if not targeted:
                distance, angle = pose_error(self.before[arm]["pose_xyz_xyzw"], after[arm]["pose_xyz_xyzw"])
                effect["inactive_arm_drift"] = {"translation_m": distance, "orientation_rad": angle}
            arms_effect[arm] = {**effect, **deepcopy(dict((arm_extras or {}).get(arm, {})))}
        native_used = int(session.environment.take_action_cnt[0]) - self.native_before
        receipt = {
            "schema": SPARSE_RECEIPT_SCHEMA, "action_id": self.action["action_id"], "action": self.action["action"],
            "arguments": deepcopy(dict(self.action["arguments"])), "consumed": True,
            "state_changed": native_used > 0,
            "execution_status": status, "motion_strategy": strategy,
            "failure_stage": stage, "failure_code": code, "executor": executor,
            "target_arms": list(self.arms),
            "synchronization": {
                "mode": "hold" if hold else "single_arm" if len(self.arms) == 1 else "dual_arm_common_native_tick",
                "requested": len(self.arms) == 2, "executed": native_used > 0,
            },
            "arms": arms_effect, "before": self.before, "after": after,
            "ik_attempts": [],
            "native_actions": native_used, "physics_steps": session.sim_steps - self.steps_before,
            "native_action_budget": self.budget, "steps_used": native_used, "sim_steps_total": session.sim_steps,
            "gripper_directives": deepcopy(session.gripper_directives),
            "gripper_floors": gripper_floors, "gripper_targets_effective": dict(gripper_targets or {}),
            "command_invariant": {
                "schema": "ora.command-invariant.v2",
                "controlled_dimension": "none" if hold else "eef_pose_target" if pose_action else "gripper",
                "held_dimension": ("eef_pose_target_and_gripper_directive" if hold else "gripper_directive"
                                   if pose_action else "eef_pose_target"),
                "verified": held_verified and (
                    session.gripper_directives == self.gripper_directives_before or not pose_action) and (
                    session.pose_targets == self.pose_targets_before or pose_action and not hold),
                "applied_targets": "per_physics_step_sdk_joint_position_buffer",
            },
            "command_targets_before": self.command_before, "command_targets_after": command_after,
            "pose_targets_before": self.pose_targets_before, "pose_targets_after": deepcopy(session.pose_targets),
            "round_index": session.round_index, "world_mutation_sequence": session.world_mutation_sequence,
            "official_evaluator_queried": True, "terminal_after_action": bool(session.is_terminal),
            "started_unix_s": self.started, "finished_unix_s": time.time(), **dict(extra or {}),
        }
        return session._persist_receipt(receipt, tracking=self.tracking, journal=self.journal)


def _remember_backtrack(session: Any, before: Mapping[str, Any], arms: Sequence[str]) -> None:
    for arm in arms:
        session.last_safe_joints[arm] = list(before[arm]["arm_joint_qpos"])
        session.last_safe_poses[arm] = list(before[arm]["pose_xyz_xyzw"])


def _gripper_outcome(*, state: str, after: Mapping[str, Any], budget: int, used: int, robot: Any) -> dict[str, Any]:
    """One arm's jaw outcome: request, target, two-finger reading, SDK-limited drive and its hold flag."""
    feedback = after["gripper_feedback"]
    fingers = [float(value) for value in after["gripper_finger_qpos"]]
    return gripper_outcome(
        rule=ARX_GRIPPER_STALL, state=state,
        target=feedback["requested_opening_normalized"],
        measured=feedback["measured_opening_normalized"],
        ticks_used=used, ticks_budget=budget,
        extra={
            "effective_target_opening_normalized": feedback["effective_target_opening_normalized"],
            "sdk_limited_drive_target_normalized": feedback["drive_target_opening_normalized"],
            "sdk_drive_limit_normalized": float(
                ARX_X5_CONTRACT["gripper_drive_limit"]["maximum_normalized_offset"]),
            "measured_finger_displacements_m": fingers,
            "finger_lateral_offset_m": feedback["finger_lateral_offset_m"],
            "primary_finger_limit_holds_close": primary_limit_holds_close(
                robot, fingers, requested=feedback["requested_opening_normalized"],
                drive=feedback["drive_target_opening_normalized"],
                effective=feedback["effective_target_opening_normalized"]),
            "tick_unit": ARX_GRIPPER_STALL.tick_unit,
            "tick_period_s": ARX_GRIPPER_STALL.tick_period_s,
        },
    )


def gripper_failure_code(states: Mapping[str, str], *, terminal: bool) -> str | None:
    """One public code for a jaw command; ``None`` only when every jaw arrived."""
    if all(state == "reached_target" for state in states.values()):
        return None
    if terminal:
        return "episode_terminal_during_execution"
    if all(state in {"reached_target", "stalled_before_target"} for state in states.values()):
        return GRIPPER_STALL_REASON
    return "gripper_not_reached"


def _execute_gripper(execution: _SparseAction, targets: Mapping[str, str]) -> dict[str, Any]:
    session = execution.session
    directives = {arm: 1.0 if targets[arm] == "open" else 0.0 for arm in execution.arms}
    execution.begin([execution.gripper_key(arm) for arm in execution.arms])
    for arm, value in directives.items():
        session.gripper_directives[arm] = value
    effective = {arm: max(value, gripper_closed_floor(session, arm)) for arm, value in directives.items()}
    states = execution.run_gripper(directives, effective)
    code = gripper_failure_code(states, terminal=session.is_terminal)
    return execution.finish(
        status="reached" if code is None else "not_reached", strategy=None,
        stage=None if code is None else "execution", code=code,
        executor=[GRIPPER_EXECUTOR], gripper_targets=effective, gripper_states=states,
    )


def execute_sparse_motion_v3(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one validated sparse v3 action to its RoboDojo executor; any other is refused unwritten."""
    kind = action["action"]
    if kind == "wait":
        from .wait import execute_wait
        return execute_wait(_SparseAction(session, action, ()))
    if kind != "set_gripper":
        raise EnvironmentRefusal(
            f"RoboDojo executes no {kind}: a move runs as the follow_joint_trajectory the toolkit plans "
            "(unsupported_action_for_backend)", field="action", received=kind,
            allowed=["follow_joint_trajectory", "set_gripper", "wait"])
    targets = action["arguments"]["targets"]
    arms = tuple(arm for arm in ARMS if arm in targets)
    return _execute_gripper(_SparseAction(session, action, arms), targets)


def _probe_variant(variant_id: str, pose: list[float], row: Mapping[str, Any]) -> dict[str, Any]:
    entry = {
        "variant_id": variant_id, "reachable": row["reachable"], "target_pose_xyzw": pose,
        "planner": {"kind": "shared_ik", "plan": "none"},
        "endpoint_error": {"translation_m": row["position_error_m"], "orientation_rad": row["orientation_error_rad"]},
        "trajectory_steps": 0, "trajectory_duration_s": 0.0, "joint_delta_rad": row["joint_delta_rad"],
        "plan_ref": None,
    }
    if not row["reachable"]:
        entry["failure"] = {"code": row["reason"], "message": f"IK admitted no joint target: {row['reason']}"}
    return entry


def probe_motion(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Whether IK holds each pose from the measured joints (``reachability.solve_batch``); plans nothing.

    A surface may still send the v2 request. It names per-seed IK feedback, and one
    IK solve per pose has none to give, so both requests get this record.
    """
    from .reachability import solve_batch

    fields = {"schema", "observation_id", "observation_sha256", "queries", "variant_policy"}
    if not isinstance(request, Mapping) or not (
        (request.get("schema") == PROBE_REQUEST_SCHEMA and set(request) == fields)
        or (request.get("schema") == PROBE_FEEDBACK_REQUEST_SCHEMA
            and set(request) == fields | {"planner_feedback"}
            and request.get("planner_feedback") == PROBE_IK_CANDIDATE_FEEDBACK)
    ):
        raise RobodojoError("motion probe request fields drifted")
    observation_id, observation_sha256 = request.get("observation_id"), request.get("observation_sha256")
    queries, variant_policy = request.get("queries"), request.get("variant_policy")
    if (not isinstance(observation_id, str) or not observation_id
            or not isinstance(observation_sha256, str) or not observation_sha256
            or not isinstance(queries, list) or not 1 <= len(queries) <= 12
            or variant_policy not in {"exact", "parallel_jaw_equivalents"}):
        raise RobodojoError("motion probe request is invalid")
    clock_before = session._clock()
    results: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for query in queries:
        if not isinstance(query, Mapping) or set(query) != {"query_id", "arm", "pose"}:
            raise RobodojoError("motion probe query fields drifted")
        query_id, arm, pose = query.get("query_id"), query.get("arm"), query.get("pose")
        if (not isinstance(query_id, str) or not query_id or query_id in query_ids or arm not in ARMS
                or not isinstance(pose, Mapping) or set(pose) != {"position_m", "orientation_xyzw"}):
            raise RobodojoError("motion probe query is invalid")
        query_ids.add(query_id)
        position = [float(value) for value in pose["position_m"]]
        orientation = [float(value) for value in pose["orientation_xyzw"]]
        if (len(position) != 3 or len(orientation) != 4
                or not all(math.isfinite(value) for value in [*position, *orientation])
                or abs(float(np.linalg.norm(orientation)) - 1.0) > 0.02):
            raise RobodojoError("motion probe pose is invalid")
        if workspace_violation(position, session.config.get("workspace_bounds_m")) is not None:
            raise RobodojoError("motion probe target is outside the action workspace")
        variants = [("variant-00-exact-+0deg", orientation)]
        if variant_policy == "parallel_jaw_equivalents":
            variants.append(("variant-01-jaw-flip-180deg", rotate_about_local_axis(orientation, JAW_SYMMETRY_AXIS, math.pi)))
        poses = [position + list(variant_orientation) for _variant_id, variant_orientation in variants]
        solved = solve_batch(session, arm, poses, position_tolerance_m=POSITION_TOLERANCE_M,
                             orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD)
        rows = [_probe_variant(variant_id, pose, row)
                for (variant_id, _orientation), pose, row in zip(variants, poses, solved)]
        selected = min((row for row in rows if row["reachable"]), key=lambda row: row["joint_delta_rad"], default=None)
        results.append({
            "query_id": query_id, "arm": arm, "reachable": selected is not None,
            "selected_variant_id": selected["variant_id"] if selected else None, "plan_ref": None,
            "selected_target_pose_xyzw": selected["target_pose_xyzw"] if selected else None,
            "variants": rows,
        })
    if session._clock() != clock_before:
        session._fault("motion_probe_advanced_physics")
        raise RobodojoError("read-only motion probe advanced physics")
    return {
        "schema": PROBE_RESULT_SCHEMA, "observation_id": observation_id,
        "world_mutation_sequence": session.world_mutation_sequence, "variant_policy": variant_policy,
        "robot_decisions_consumed": 0, "world_state_changed": False, "results": results,
    }


__all__ = [
    "GLOBAL_JOINT_JUMP_LIMIT_RAD", "GRIPPER_EXECUTOR", "SPARSE_RECEIPT_SCHEMA",
    "STAGED_CONTACT_STRATEGY",
    "execute_sparse_motion_v3", "gripper_closed_floor", "gripper_failure_code",
    "probe_motion",
    "rotate_about_local_axis",
    "solve_ik",
]
