"""The global CuRobo / JOINT_POSITION route and its world-to-planner frames.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. Shared constants and helpers come from ``common``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

from ora.environment.controllers import jaw_flip_retry
from ora.environment.controllers.motion_pace import PacedPath, PaceRefusal, pace_path, paces, speed_report
from ora.environment.controllers.hybrid_motion import (
    LIBERO_ENDPOINT_ROUTE_GLOBAL,
    libero_endpoint_contract,
    libero_global_endpoint_tolerances,
    libero_global_joint_ceilings,
    libero_joint_controller,
    libero_osc_transit,
    precision_force_stop_enabled,
)
from ora.contracts.action import RUNTIME_ACTION_SCHEMA
from ora.contracts.workspace import workspace_state
from ora.environment.controllers import osc_transit

from .common import (
    HYBRID_RECEIPT_SCHEMA,
    CONTROL_HEALTH_SCHEMA,
    LiberoProError,
    _quaternion_error_rotvec,
)
from .motion_settle import LIBERO_CONTROL_PERIOD_S
from .planner_request import failure_reason

#: The Panda's last wrist joint (panda_joint7), whose turns the pacing reports and checks.
WRIST_JOINT_INDEX = 6


class LiberoPlannedMotionMixin:
    """The global CuRobo / JOINT_POSITION route and its world-to-planner frames."""

    def _remember_backtrack_state(self) -> None:
        """Freeze the state before a non-recovery motion mutation."""

        state = self._motion_state()
        joints = state["joint_positions"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in joints
        ):
            raise LiberoProError("backtrack state has invalid Panda joints")
        self.last_backtrack_joint_state = [float(value) for value in joints]
        pose = state.get("grip_site_pose_world_xyzw")
        self.last_backtrack_tcp_pose = list(pose) if pose is not None else None

    def _refresh_joint_observation(self, steps: int) -> None:
        """Update the driver's observation cache from the last dispatched step, without stepping."""

        if steps == 0:
            return
        raw = self._worker_attr("harness_joint_execution_observation")
        if not isinstance(raw, Mapping):
            raise LiberoProError("joint execution observation is unavailable")
        environment = self._require_environment()
        environment.current_raw_obs = [dict(raw)]
        # Direct joint dispatch bypasses the driver's step, including its counter. The
        # observation transfer itself consumes zero simulation steps.
        if hasattr(environment, "_elapsed_steps"):
            environment._elapsed_steps += steps
        self.last_raw_observation = dict(raw)

    def _validate_planned_start(
        self, planned: Mapping[str, Any], motion_state: Mapping[str, Any],
    ) -> None:
        """Apply identical measured-start admission to direct/probe/reference."""

        import numpy as np
        from scipy.spatial.transform import Rotation

        trajectory = planned.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise LiberoProError("CuRobo omitted its trajectory")
        names = [f"panda_joint{index}" for index in range(1, 8)]
        if trajectory.get("joint_names") != names or motion_state.get("joint_names") != names:
            raise LiberoProError("CuRobo Panda joint identity drifted")
        positions = np.asarray(trajectory.get("positions"), dtype=np.float64)
        measured_joints = np.asarray(motion_state["joint_positions"], dtype=np.float64)
        if (
            positions.ndim != 2 or positions.shape[1] != 7 or not len(positions)
            or measured_joints.shape != (7,)
            or not np.isfinite(positions).all()
            or not np.isfinite(measured_joints).all()
        ):
            raise LiberoProError("CuRobo trajectory start is malformed")
        if float(np.max(np.abs(positions[0] - measured_joints))) > 0.02:
            raise LiberoProError("CuRobo trajectory start joints no longer match")
        planner_start = planned.get("start_ee_pose_base_xyzw")
        if not isinstance(planner_start, list) or len(planner_start) != 7:
            raise LiberoProError("CuRobo omitted start FK cross-validation")
        fk = np.asarray(planner_start, dtype=np.float64)
        if not np.isfinite(fk).all() or not np.isclose(
            float(np.linalg.norm(fk[3:])), 1.0, atol=1e-5, rtol=0,
        ):
            raise LiberoProError("CuRobo start FK is malformed")
        measured_start = self._matrix_pose(
            np.linalg.inv(self._pose_matrix(motion_state["base_pose_world_xyzw"]))
            @ self._pose_matrix(motion_state["hand_pose_world_xyzw"])
        )
        position_error = float(np.linalg.norm(np.asarray(measured_start[:3]) - fk[:3]))
        orientation_error = float((
            Rotation.from_quat(measured_start[3:]).inv()
            * Rotation.from_quat(fk[3:])
        ).magnitude())
        if position_error > 0.005 or orientation_error > 0.05:
            raise LiberoProError("CuRobo Panda FK does not align with MuJoCo panda_hand")

    @staticmethod
    def _pose_matrix(pose_xyzw: Any) -> Any:
        import numpy as np
        from scipy.spatial.transform import Rotation

        pose = np.asarray(pose_xyzw, dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise LiberoProError("rigid pose is malformed")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        transform[:3, 3] = pose[:3]
        return transform

    @staticmethod
    def _matrix_pose(transform: Any) -> list[float]:
        import numpy as np
        from scipy.spatial.transform import Rotation

        matrix = np.asarray(transform, dtype=np.float64)
        return matrix[:3, 3].tolist() + Rotation.from_matrix(
            matrix[:3, :3]
        ).as_quat().tolist()

    def _grip_target_planner_pose(
        self, motion_state: Mapping[str, Any], target_pose: list[float],
    ) -> tuple[list[float], list[float]]:
        """Compile a world grip-site target into the CuRobo hand-frame request."""

        import numpy as np

        world_from_base = self._pose_matrix(motion_state["base_pose_world_xyzw"])
        world_from_hand = self._pose_matrix(motion_state["hand_pose_world_xyzw"])
        world_from_grip = self._pose_matrix(
            motion_state["grip_site_pose_world_xyzw"]
        )
        hand_from_grip = np.linalg.inv(world_from_hand) @ world_from_grip
        world_from_target_grip = self._pose_matrix(target_pose)
        base_from_target_hand = (
            np.linalg.inv(world_from_base)
            @ world_from_target_grip
            @ np.linalg.inv(hand_from_grip)
        )
        return (
            self._matrix_pose(base_from_target_hand),
            self._matrix_pose(np.linalg.inv(world_from_base) @ world_from_hand),
        )

    def _grip_site_kinematics(self, rows: list[list[float]]) -> dict[str, Any]:
        """Grip-site poses at hypothetical arm joints and the arm's joint limits, without moving the plant."""

        environment = self._require_environment()
        setter = getattr(getattr(environment, "env", None), "set_env_attr", None)
        if not callable(setter):
            raise LiberoProError("grip-site kinematics transport is unavailable")
        setter("harness_grip_site_fk", {"joints": [list(map(float, row)) for row in rows]})
        answer = self._worker_attr("harness_grip_site_fk")
        if (not isinstance(answer, Mapping) or len(answer.get("poses") or ()) != len(rows)
                or len(answer.get("joint_limits_rad") or ()) != 7):
            raise LiberoProError("grip-site kinematics answer is malformed")
        return dict(answer)

    def _pace_global_route(self, positions: Any) -> PacedPath:
        """Time a planner's joint path for the active speed request and check its wrist (``motion_pace``).

        One knot per 50 ms control tick; every joint at most what the executor's
        controller follows (``hybrid_motion.libero_global_joint_ceilings``: v3's
        ``joint_follow_speed_rad_s`` and ``joint_follow_accel_rad_s2``, v2's
        ``joint_command_step_rad`` per tick); the whole path within what one
        action may spend (``max_waypoints``, and ``max_execution_ticks`` less the
        terminal settle), refused rather than cut. Raises ``PaceRefusal`` before
        anything moves.
        """

        global_motion = self.motion_profile["global_motion"]
        speed_ceiling, accel_ceiling = libero_global_joint_ceilings(self.motion_profile, tick_s=LIBERO_CONTROL_PERIOD_S)
        rows = [[float(value) for value in row] for row in positions]
        limits = self._grip_site_kinematics(rows[:1])["joint_limits_rad"]
        return pace_path(
            rows, lambda joints: self._grip_site_kinematics(joints)["poses"],
            speed=getattr(self, "_active_motion_speed", None), tick_s=LIBERO_CONTROL_PERIOD_S,
            joint_speed_ceiling_rad_s=speed_ceiling, joint_accel_ceiling_rad_s2=accel_ceiling,
            wrist_joint_index=WRIST_JOINT_INDEX,
            wrist_limits_rad=(float(limits[WRIST_JOINT_INDEX][0]), float(limits[WRIST_JOINT_INDEX][1])),
            budget_ticks=min(int(global_motion["max_waypoints"]), int(global_motion["max_execution_ticks"])
                             - int(global_motion["max_terminal_settle_ticks"])),
            slow_approach=bool(getattr(self, "_active_motion_slow_final", True)), arm="right",
            robot="franka_panda")

    def _jaws_open(self) -> bool:
        """The jaws are commanded open, so they hold nothing a half-turn would turn.

        The Panda's hold directive opens (negative: the sign the planner's world reads as
        no load, ``world_inputs.py``) and the last accepted jaw target was not ``close``;
        the directive ramps, so a close just commanded may still read open on its own.
        """

        return self.public_gripper_command_target() != "close" and self._gripper_hold_directive() < 0.0

    def _jaw_twin_route(
        self, refusal: PaceRefusal, request: Mapping[str, Any], motion_state: Mapping[str, Any],
        goal: list[float], *, persist: bool,
    ) -> tuple[list[float], dict[str, Any], dict[str, Any] | None, PacedPath] | None:
        """The route to ``goal`` that ``refusal`` stopped, planned and paced once more to its jaw twin.

        ``controllers/jaw_flip_retry.py`` says when: a wrist refusal, with the jaws open
        (``_jaws_open``). Returns ``(twin goal, plan, its evidence, paced)``, the pacing's
        wrist report naming the twin, or None when the refusal stands. ``request`` is the
        refused plan's own, its world snapshot included, so the twin is asked in the same
        world; the real call keeps the twin's plan as ``action-NNNN-jaw-twin.json``
        (``persist``), and its dry run (``action_plan.py``) asks the same question.
        """

        kept: dict[str, Any] = {"evidence": None}

        def admit(twin: list[float]) -> Any:
            target_base = self._grip_target_planner_pose(motion_state, twin)[0]
            planned = self.motion_planner.plan({**request, "target_pose_base_xyzw": target_base})
            if persist:
                kept["evidence"] = self._persist_planner_evidence(planned, variant="jaw-twin")
            if not planned.get("reachable") or not isinstance(planned.get("trajectory"), Mapping):
                return failure_reason(planned)
            self._validate_planned_start(planned, motion_state)
            return planned, self._pace_global_route(planned["trajectory"]["positions"])

        retried = jaw_flip_retry.retry(goal, refusal, admit, jaws_open=self._jaws_open)
        if retried is None:
            return None
        twin, (planned, paced), record = retried
        return twin, planned, kept["evidence"], jaw_flip_retry.recorded(paced, record)

    def _execute_planned_motion(
        self, action: Mapping[str, Any], *,
        preplanned: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        if self.motion_planner is None and preplanned is None:
            raise LiberoProError("CuRobo planner is not active")
        environment = self._require_environment()
        arguments = action["arguments"]
        before_position, before_orientation, before_gripper = self._current_state()
        motion_state = self._motion_state()
        request: dict[str, Any] = {
            "joint_names": motion_state["joint_names"],
            "joint_positions": motion_state["joint_positions"],
        }
        target_pose = None
        strategy = None
        if preplanned is not None:
            target_pose = preplanned[0].get("target_pose_world_xyzw")
        elif action["action"] == "move_free_space":
            target_pose = (
                arguments["target_pose"]["position_m"]
                + arguments["target_pose"]["orientation_xyzw"]
            )
            if not workspace_state(target_pose[:3], self.workspace_bounds_m)["inside"]:
                raise LiberoProError("move_free_space target is outside the workspace")
            target_base, _measured_start = self._grip_target_planner_pose(
                motion_state, target_pose,
            )
            request["target_pose_base_xyzw"] = target_base
        else:
            strategy = str(arguments["strategy"])
            goal = (
                self.home_joint_state
                if strategy == "home" else self.last_backtrack_joint_state
            )
            if goal is None:
                return self._planned_refusal_receipt(
                    action, before_position, before_orientation, before_gripper,
                    "recovery_state_unavailable",
                )
            request["goal_joint_positions"] = list(goal)
            if precision_force_stop_enabled(self.motion_profile):
                target_pose = (self.home_tcp_pose if strategy == "home"
                               else self.last_backtrack_tcp_pose)
                if target_pose is None:
                    return self._planned_refusal_receipt(
                        action, before_position, before_orientation, before_gripper,
                        "recovery_state_unavailable",
                    )
        if preplanned is not None:
            planned, planner_evidence = (
                dict(preplanned[0]), dict(preplanned[1]),
            )
        else:
            request.update(self._world_input(motion_state, "action"))
            planned = self.motion_planner.plan(request)
            planner_evidence = self._persist_planner_evidence(planned)
        if not planned.get("reachable") or not isinstance(
            planned.get("trajectory"), Mapping
        ):
            return self._planned_refusal_receipt(
                action, before_position, before_orientation, before_gripper,
                failure_reason(planned),
                planner=planner_evidence,
            )
        self._validate_planned_start(planned, motion_state)
        try:
            paced = self._pace_global_route(planned["trajectory"]["positions"])
        except PaceRefusal as refusal:
            # A pose move may run its jaw twin instead; a recovery's joint goal and a
            # probe's immutable plan never do.
            twin = (None if preplanned is not None or action["action"] != "move_free_space"
                    else self._jaw_twin_route(refusal, request, motion_state, target_pose, persist=True))
            if twin is None:
                speed = getattr(self, "_active_motion_speed", None)
                move_pace, slow_pace = paces(speed)
                return self._planned_refusal_receipt(
                    action, before_position, before_orientation, before_gripper,
                    refusal.code, planner=planner_evidence,
                    pacing={"speed": speed_report(speed, move_pace, slow_pace),
                            "refused": {"code": refusal.code, "message": str(refusal), **refusal.constraints}},
                )
            target_pose, planned, planner_evidence, paced = twin
        choice = self._free_space_route_choice(action, preplanned is not None, paced, target_pose)
        if choice is not None and choice["route"] == osc_transit.ROUTE_OSC:
            return self._run_osc_transit(action, target_pose, planner_evidence, choice)
        vector_environment = getattr(environment, "env", None)
        setter = getattr(vector_environment, "set_env_attr", None)
        if not callable(setter):
            raise LiberoProError("joint trajectory command transport is unavailable")
        trajectory = planned["trajectory"]
        execution_id = f"{self.round_index}:{self.action_sequence}:{action['action_id']}"
        try:
            setter("harness_joint_trajectory_command", {
                "execution_id": execution_id,
                "positions": paced.rows(),
                "joint_names": trajectory["joint_names"],
                "dt_s": LIBERO_CONTROL_PERIOD_S,
                # A path that does not move (a no-op plan) keeps the untimed rule:
                # its one waypoint is already within tolerance and costs no tick.
                "timed": bool(float(np.sum(paced.grid.joint_steps())) > 0.0),
                **({"target_pose_world_xyzw": target_pose}
                   if precision_force_stop_enabled(self.motion_profile) else {}),
                "stop_request_path": str(
                    (self.output_root / "control/stop.request").resolve()
                ),
            })
            execution = self._worker_attr("harness_joint_execution_result")
            if not isinstance(execution, Mapping) or execution.get("execution_id") != execution_id:
                raise LiberoProError("joint trajectory execution identity is unavailable")
        except Exception:
            # Controller restoration or observation formatting may fail after a
            # successful official step. Read the separate early-latched outcome
            # before allowing that infrastructure failure to propagate.
            try:
                outcome = self._worker_attr("harness_joint_execution_status")
            except Exception:
                outcome = None
            # A failed command transport can leave the preceding command's
            # outcome behind. Never count it again or attach it to this action.
            if isinstance(outcome, Mapping) and outcome.get("execution_id") == execution_id:
                if outcome["steps_used"] and action["action"] != "recover_motion":
                    self.last_backtrack_joint_state = list(motion_state["joint_positions"])
                    self.last_backtrack_tcp_pose = list(motion_state["grip_site_pose_world_xyzw"])
                self._latch_environment_outcome(
                    terminated=outcome["terminated"], truncated=outcome["truncated"],
                    steps=int(outcome["steps_used"]), action=action,
                    controller="JOINT_POSITION",
                )
            raise
        steps_used = int(execution["steps_used"])
        if steps_used and action["action"] != "recover_motion":
            self.last_backtrack_joint_state = list(motion_state["joint_positions"])
            self.last_backtrack_tcp_pose = list(motion_state["grip_site_pose_world_xyzw"])
        self._latch_environment_outcome(
            terminated=execution.get("terminated"), truncated=execution.get("truncated"),
            steps=steps_used, action=action, controller="JOINT_POSITION",
        )
        execution_evidence = self._persist_joint_execution_evidence(execution)
        issued_commands = execution.get("commands")
        if not isinstance(issued_commands, list) or len(issued_commands) != steps_used:
            raise LiberoProError("joint trajectory omitted issued controller commands")
        self.replay_recorder.record(
            action_sequence=self.action_sequence,
            action=action,
            controller="libero_joint_position",
            arrays={"commands": np.asarray(issued_commands).reshape((-1, 8))},
            physics_steps=steps_used,
            replayable=steps_used > 0,
            metadata={
                "controller_switch": "JOINT_POSITION_then_OSC_POSE",
                "recorded_plan_artifact": planner_evidence.get(
                    "canonical_artifact"
                ),
                "paced_ticks": int(paced.knots["ticks"]),
                # The gains a replay loads (``harness_replay_controller_mode``); a run
                # bound to v1 or v2 records none and replays on RoboSuite's defaults.
                **({"joint_controller": gains}
                   if (gains := libero_joint_controller(self.motion_profile)) is not None else {}),
            },
        )
        self._refresh_joint_observation(steps_used)
        after_position, after_orientation, after_gripper = self._current_state()
        after_motion_state = self._motion_state()
        after_wrench = after_motion_state.get("external_wrench_clipped")
        force_magnitude_n, force_threshold_exceeded = self._force_telemetry(
            after_motion_state
        )
        contact_threshold = float(
            self.motion_profile["sensing"]["contact_force_threshold_n"]
        )
        not_predicted = (
            None if after_wrench is None
            else float(np.linalg.norm(
                np.asarray(after_wrench[:3], dtype=np.float64)
            )) > contact_threshold
        )
        tracking_error = float(execution["tracking_error_l2_rad"])
        tracking_limit = float(
            self.motion_profile["global_motion"]["joint_tracking_error_limit_rad"]
        )
        stopped = execution.get("stop_reason") is not None
        endpoint_error = None
        endpoint_reached = True
        position_tolerance, orientation_tolerance = libero_global_endpoint_tolerances()
        if target_pose is not None:
            endpoint_error = {
                "position_m": float(np.linalg.norm(
                    np.asarray(target_pose[:3]) - after_position
                )),
                "orientation_rad": float(np.linalg.norm(_quaternion_error_rotvec(
                    np.asarray(target_pose[3:]), after_orientation,
                ))),
            }
            endpoint_reached = (
                endpoint_error["position_m"] <= position_tolerance
                and endpoint_error["orientation_rad"] <= orientation_tolerance
            )
        healthy = tracking_error <= tracking_limit and endpoint_reached and not stopped
        endpoint_failure = (
            "endpoint_residual_exceeded" if not endpoint_reached
            else "joint_tracking_error"
        )
        settle_record = execution.get("terminal_settle")
        settle_record = settle_record if isinstance(settle_record, Mapping) else None
        # The global route is judged by the CuRobo transport bound and its last
        # millimetres are owned by the terminal settle. Compile that verdict
        # once, here, so no projection layer has to guess which bound applied.
        endpoint_contract = libero_endpoint_contract(
            route=LIBERO_ENDPOINT_ROUTE_GLOBAL,
            position_tolerance_m=position_tolerance,
            orientation_tolerance_rad=orientation_tolerance,
            position_residual_m=(
                None if endpoint_error is None else endpoint_error["position_m"]
            ),
            orientation_residual_rad=(
                None if endpoint_error is None
                else endpoint_error["orientation_rad"]
            ),
            within_tolerance=(None if endpoint_error is None else endpoint_reached),
            convergence_budget="terminal_settle",
            execution_phase=execution.get("execution_phase", "unknown"),
            stop_reason=execution.get("stop_reason"),
            waypoints_reached=execution.get("waypoints_reached"),
            waypoints_total=execution.get("waypoints_total"),
            ticks_used=(None if settle_record is None else settle_record.get("ticks")),
            ticks_budget=(
                None if settle_record is None else settle_record.get("max_ticks")
            ),
            budget_exhausted=(
                False if settle_record is None
                else bool(settle_record.get("exhausted"))
            ),
        )
        receipt = {
            "schema": HYBRID_RECEIPT_SCHEMA,
            "action_id": action["action_id"],
            "action": action["action"],
            "arguments": action["arguments"],
            "consumed": True,
            "state_changed": steps_used > 0,
            "motion_status": (
                "reached" if healthy else "stopped" if stopped else "not_reached"
            ),
            **({"commanded_pose_xyz_xyzw": list(target_pose)}
               if precision_force_stop_enabled(self.motion_profile) and target_pose is not None else {}),
            "planner": planner_evidence,
            "execution": execution_evidence,
            "pacing": paced.report(),
            "endpoint_contract": endpoint_contract,
            "feedback": {
                "endpoint_contract": endpoint_contract,
                "measured_pose_xyz_xyzw": after_position.tolist() + after_orientation.tolist(),
                "joint_state": {
                    "names": after_motion_state["joint_names"],
                    "positions": execution["joint_positions"],
                },
                "tracking_error_l2_rad": tracking_error,
                "endpoint_error": endpoint_error,
                "endpoint_tolerance": {
                    "position_m": position_tolerance,
                    "orientation_rad": orientation_tolerance,
                } if target_pose is not None else None,
                "endpoint_comparison": "<=",
                "contact_estimated": not_predicted,
                "collision": None,
                "reflex": False,
                "external_wrench_clipped": after_wrench,
                "external_force_magnitude_n": force_magnitude_n,
                "force_threshold_exceeded": force_threshold_exceeded,
                **self._force_direction_feedback(after_motion_state),
                "gripper_aperture": after_gripper,
                "motion_state": "stopped" if stopped else "idle",
                "reachability": True,
                "recovery_reason": strategy,
                "stop_reason": execution.get("stop_reason"),
            },
            "before": {"right": {"pose_xyz_xyzw": before_position.tolist() + before_orientation.tolist(), "gripper": before_gripper}},
            "after": {"right": {"pose_xyz_xyzw": after_position.tolist() + after_orientation.tolist(), "gripper": after_gripper}},
            "steps_used": steps_used,
            "sim_steps_total": self.sim_steps,
            "control_health": {
                "schema": CONTROL_HEALTH_SCHEMA,
                "status": "healthy" if healthy else "degraded",
                "codes": (
                    [] if healthy else [
                        str(execution.get("stop_reason") or endpoint_failure)
                    ]
                ),
            },
            "robustness_event": self._persist_new_robustness_event(),
            "finished_unix_s": time.time(),
        }
        self.replay_recorder.update(
            self.action_sequence,
            execution_status=str(receipt["motion_status"]),
            execution_fault=(
                str(execution.get("stop_reason"))
                if execution.get("stop_reason") is not None else None
            ),
        )
        return self._persist_receipt(receipt)

    def _free_space_route_choice(
        self, action: Mapping[str, Any], plan_reference: bool, paced: PacedPath, target_pose: Any,
    ) -> dict[str, Any] | None:
        """Which route this planned move runs (``osc_transit.route_choice``); None under v1 to v3.

        The record goes on the move's sparse receipt (``arms.right.route_choice``) whichever route
        it names. The planner's grip-site path is read at the paced knots, from the measured start.
        """

        section = libero_osc_transit(self.motion_profile)
        plan = getattr(self, "_active_sparse_plan", None)
        if section is None or action["action"] != "move_free_space" or target_pose is None:
            return None
        speed = getattr(self, "_active_motion_speed", None) or {}
        _position, start_orientation, _gripper = self._current_state()
        rotation = osc_transit.rotation_angle(list(start_orientation), list(target_pose[3:]))
        interaction = getattr(self, "_active_motion_interaction", None)
        positions = None
        if interaction == "free_space" and not speed.get("slow_approaching") and not plan_reference \
                and rotation <= float(section["max_rotation_rad"]):
            start_joints = self._motion_state()["joint_positions"]
            rows = [list(map(float, start_joints)), *paced.rows()]
            positions = [pose[:3] for pose in self._grip_site_kinematics(rows)["poses"]]
        choice = osc_transit.route_choice(
            section, interaction=interaction, slow_approaching=bool(speed.get("slow_approaching")),
            plan_reference=plan_reference, planner_path_positions=positions, rotation_rad=rotation)
        if plan is not None:
            plan.route_choice = choice
        return choice

    def _osc_line_reference(self, target_pose: list[float]) -> dict[str, Any]:
        """The straight OSC line from the measured pose to ``target_pose`` (``osc_transit.straight_line_reference``).

        Paced at the request's Cartesian caps, no faster than the OSC controller follows
        (``osc_transit.follow_*``); each tick aims ahead by the controller's lag (``osc_transit.lead_s``).
        """

        section = libero_osc_transit(self.motion_profile)
        position, orientation, _gripper = self._current_state()
        move, _slow = paces(getattr(self, "_active_motion_speed", None))
        degrees = math.pi / 180.0

        def cap(value: float | None, ceiling: float) -> float:
            return ceiling if value is None else min(float(value), ceiling)

        return osc_transit.straight_line_reference(
            [*map(float, position), *map(float, orientation)], list(map(float, target_pose)),
            linear_m_s=cap(move.linear_m_s, float(section["follow_speed_m_s"])),
            angular_rad_s=cap(None if move.angular_deg_s is None else move.angular_deg_s * degrees,
                              float(section["follow_angular_rad_s"])),
            linear_accel_m_s2=cap(move.linear_accel_m_s2, float(section["accel_m_s2"])),
            angular_accel_rad_s2=cap(None if move.angular_accel_deg_s2 is None
                                     else move.angular_accel_deg_s2 * degrees, float(section["angular_accel_rad_s2"])),
            tick_s=LIBERO_CONTROL_PERIOD_S, lead_s=osc_transit.lead_s(section))

    def _run_osc_transit(
        self, action: Mapping[str, Any], target_pose: list[float], planner_evidence: Mapping[str, Any],
        choice: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a clear, nearly straight free-space move as one straight OSC line (``osc_transit``).

        The local OSC loop follows the line (``_osc_line_reference``), then settles onto the target
        within the local bound. The plan stays the move's collision evidence (``choice['planner']``).
        """

        section = libero_osc_transit(self.motion_profile)
        reference = self._osc_line_reference(target_pose)
        choice.update(planner=dict(planner_evidence), line={
            "length_m": reference["length_m"], "turn_rad": reference["turn_rad"],
            "planned_ticks": reference["ticks"], "settle_ticks_budget": int(section["settle_ticks"])})
        plan = getattr(self, "_active_sparse_plan", None)
        if plan is not None and plan.routes:
            plan.routes[-1] = "libero_robosuite_osc_pose"
        previous = getattr(self, "_active_osc_transit", None)
        self._active_osc_transit = {**reference, "settle_ticks": int(section["settle_ticks"])}
        try:
            # The runtime action whose OSC loop runs toward a pose (``ora-action.v3`` move_pose);
            # the motion action v1 vocabulary has no unbounded OSC move.
            return self.execute({
                "schema": RUNTIME_ACTION_SCHEMA, "observation_id": action["observation_id"],
                "action_id": action["action_id"], "bound_observation_sha256": action["bound_observation_sha256"],
                "action": "move_pose",
                "arguments": {"arm": "right", "position_m": list(target_pose[:3]),
                              "orientation_xyzw": list(target_pose[3:])},
            })
        finally:
            self._active_osc_transit = previous

    def _planned_refusal_receipt(
        self, action: Mapping[str, Any], position: Any, orientation: Any,
        gripper: float, reason: str, *,
        planner: Mapping[str, Any] | None = None,
        pacing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        recorder = getattr(self, "replay_recorder", None)
        if recorder is not None:
            recorder.record(
                action_sequence=self.action_sequence,
                action=action,
                controller="libero_joint_position",
                arrays={},
                physics_steps=0,
                replayable=False,
                metadata={"reason": reason},
            )
        receipt = {
            "schema": HYBRID_RECEIPT_SCHEMA,
            "action_id": action["action_id"], "action": action["action"],
            "arguments": action["arguments"], "consumed": True,
            "state_changed": False, "motion_status": "refused",
            "planner": None if planner is None else dict(planner),
            **({"pacing": dict(pacing)} if pacing is not None else {}),
            "feedback": {
                "measured_pose_xyz_xyzw": position.tolist() + orientation.tolist(),
                "joint_state": None, "tracking_error_l2_rad": None,
                "contact_estimated": None, "collision": None, "reflex": False,
                "external_wrench_clipped": None, "gripper_aperture": gripper,
                "external_force_magnitude_n": None,
                "force_threshold_exceeded": False,
                "motion_state": "refused",
                "reachability": False, "planning_failure_reason": reason,
            },
            "before": {"right": {"pose_xyz_xyzw": position.tolist() + orientation.tolist(), "gripper": gripper}},
            "after": {"right": {"pose_xyz_xyzw": position.tolist() + orientation.tolist(), "gripper": gripper}},
            "steps_used": 0, "sim_steps_total": self.sim_steps,
            "control_health": {"schema": CONTROL_HEALTH_SCHEMA, "status": "degraded", "codes": [reason]},
            "robustness_event": None, "finished_unix_s": time.time(),
        }
        if recorder is not None:
            recorder.update(
                self.action_sequence, execution_status="refused",
                execution_fault=reason,
            )
        return self._persist_receipt(receipt)
