"""Sparse motion v3: route one semantic target over the two qualified executors.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. A ``move_pose`` is planned into a ``_SparseMotionV3`` route plan, run
as one or two executor segments, and every segment receipt is projected into
the canonical sparse receipt through that plan. Shared constants and helpers
come from ``common``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from ora.contracts import MOTION_ACTION_SCHEMA
from ora.environment.backends.libero_pro.motion_settle import (
    LIBERO_CONTROL_PERIOD_S,
    merge_settled_receipt,
    precontact_pose,
    precontact_refusal,
    settle_refusal,
)
from ora.environment.controllers.hybrid_motion import (
    libero_global_joint_ceilings,
    libero_local_endpoint_tolerances,
)
from ora.environment.controllers.jaw_flip_retry import jaw_twin
from ora.environment.controllers.motion_pace import paces, speed_report

from .common import (
    LIBERO_SPARSE_RECEIPT_SCHEMA,
    _sparse_planning_failure,
    _within_local_servo_envelope,
    LiberoProError,
    _quaternion_error_rotvec,
)


@dataclass
class _SparseMotionV3:
    """One sparse v3 ``move_pose``: its route and what its segments disclosed.

    Routes are recorded as they execute rather than predicted, because a
    settling route runs one or two of them depending on what the global
    segment actually measured.
    """

    action: Mapping[str, Any]
    arguments: Mapping[str, Any]
    identity: dict[str, Any]
    target: Mapping[str, Any]
    interaction: str
    target_position: Any
    before_position: Any
    before_orientation: Any
    strategy: str
    precontact: list[float] | None
    #: The requested end-effector speed (``ora.contracts.motion_speed`` wire form; None: the default).
    speed: Mapping[str, Any] | None = None
    #: ``motion`` of the receipt: the global segment's pacing, or the request's caps on a local route.
    motion: dict[str, Any] | None = None
    routes: list[str] = field(default_factory=list)
    settle_disclosure: dict[str, Any] | None = None
    precontact_disclosure: dict[str, Any] | None = None
    #: Whether the global segment ran the jaw twin of its goal, and ``target`` is the twin since.
    jaw_twin: bool = False
    #: Which route a planned free-space segment ran and why (``osc_transit.route_choice``; v4).
    route_choice: dict[str, Any] | None = None


def _sparse_segment_outcome(
    plan: _SparseMotionV3, inner: Mapping[str, Any],
) -> tuple[str, str | None, Any]:
    """Map one segment receipt to the sparse status, failure stage and code."""

    motion_status = str(inner.get("motion_status"))
    feedback = inner.get("feedback")
    feedback = feedback if isinstance(feedback, Mapping) else {}
    if motion_status == "refused":
        reason = str(feedback.get("planning_failure_reason"))
        status, stage, code = _sparse_planning_failure(reason)
    elif motion_status == "reached":
        status, stage, code = "reached", None, None
    elif motion_status == "stopped":
        status, stage, code = (
            "not_reached", "execution",
            feedback.get("stop_reason"),
        )
    else:
        status, stage, code = (
            "not_reached", "endpoint_verification",
            "endpoint_residual_exceeded",
        )
    if (status == "reached" and plan.precontact_disclosure is not None
            and not plan.precontact_disclosure["attempted"]):
        # The segment reached the pre-contact pose, not the target.
        status, stage, code = (
            "not_reached", "precontact", plan.precontact_disclosure["reason"],
        )
    return status, stage, code


def _sparse_pose_error(
    plan: _SparseMotionV3, inner: Mapping[str, Any],
) -> dict[str, float] | None:
    import numpy as np

    after = inner.get("after", {}).get("right", {})
    after_pose = after.get("pose_xyz_xyzw")
    pose_error = None
    if isinstance(after_pose, list) and len(after_pose) == 7:
        pose_error = {
            "translation_m": float(np.linalg.norm(
                plan.target_position
                - np.asarray(after_pose[:3], dtype=np.float64)
            )),
            "orientation_rad": float(np.linalg.norm(
                _quaternion_error_rotvec(
                    np.asarray(
                        plan.target["orientation_xyzw"], dtype=np.float64,
                    ),
                    np.asarray(after_pose[3:], dtype=np.float64),
                )
            )),
        }
    return pose_error


class LiberoSparseMotionMixin:
    """Sparse motion v3: route one semantic target over the two qualified executors."""

    def _execute_sparse_motion_v3(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Single-arm sparse motion v3 over the two qualified executors.

        The v3 semantic route delegates every physical
        segment to the exact executors this backend already qualified on real
        hardware — the CuRobo/JOINT_POSITION planned path and the bounded
        OSC servo loop — instead of re-plumbing them through the shared
        measured-convergence abstraction before its LIBERO qualification.
        Each action produces exactly one canonical sparse receipt
        (``ora.libero-sparse-motion-receipt.v1``) that embeds the
        legacy executor evidence under ``segment``. A ``contact_approach``
        larger than the local envelope is staged: a global segment to a
        pre-contact pose, then the bounded local approach.
        """

        arguments = action["arguments"]
        if action["action"] == "move_pose" and set(arguments) == {"plan_ref"}:
            return self._execute_plan_reference(action)
        targets = arguments["targets"]
        if set(targets) != {"right"}:
            raise LiberoProError(
                "LIBERO-Pro executes only the right single arm"
            )
        identity = {
            "schema": MOTION_ACTION_SCHEMA,
            "observation_id": action["observation_id"],
            "action_id": action["action_id"],
            "bound_observation_sha256": action["bound_observation_sha256"],
        }
        if action["action"] == "set_gripper":
            return self.execute({
                **identity,
                "action": "set_gripper",
                "arguments": {
                    "arm": "right",
                    "gripper": 1.0 if targets["right"] == "open" else 0.0,
                },
            })
        if action["action"] == "recover_motion":
            return self.execute({
                **identity,
                "action": "recover_motion",
                "arguments": {"arm": "right", "strategy": targets["right"]},
            })
        return self._run_sparse_motion_v3(
            self._plan_sparse_motion_v3(action, identity)
        )

    def _plan_sparse_motion_v3(
        self, action: Mapping[str, Any], identity: dict[str, Any],
    ) -> _SparseMotionV3:
        """Choose the route from the measured start and the interaction."""

        import numpy as np

        arguments = action["arguments"]
        target = arguments["targets"]["right"]
        interaction = str(arguments["interactions"]["right"])
        before_position, before_orientation, _gripper = self._current_state()
        target_position = np.asarray(target["position_m"], dtype=np.float64)
        translation_m = float(np.linalg.norm(target_position - before_position))
        rotation_rad = float(np.linalg.norm(_quaternion_error_rotvec(
            np.asarray(target["orientation_xyzw"], dtype=np.float64),
            before_orientation,
        )))
        local_route = _within_local_servo_envelope(translation_m, rotation_rad)
        if interaction == "free_space":
            strategy = "global_then_local_settle"
        elif interaction == "precision":
            strategy = "local" if local_route else "global_then_local_settle"
        else:
            strategy = "local" if local_route else "global_precontact_then_local"
        precontact = (
            precontact_pose(
                [*before_position, *before_orientation],
                [*target["position_m"], *target["orientation_xyzw"]],
            ) if strategy == "global_precontact_then_local" else None
        )
        return _SparseMotionV3(
            action=action, arguments=arguments, identity=identity,
            target=target, interaction=interaction,
            target_position=target_position,
            before_position=before_position,
            before_orientation=before_orientation,
            strategy=strategy, precontact=precontact,
            speed=arguments.get("speed"),
        )

    def _run_sparse_motion_v3(self, plan: _SparseMotionV3) -> dict[str, Any]:
        """Run the planned segments with the sparse receipt projection installed."""

        import numpy as np

        previous_transform = getattr(self, "_sparse_receipt_transform", None)
        previous_interaction = getattr(self, "_active_motion_interaction", None)
        previous_speed = getattr(self, "_active_motion_speed", None)
        previous_slow_final = getattr(self, "_active_motion_slow_final", True)
        previous_plan = getattr(self, "_active_sparse_plan", None)
        self._active_sparse_plan = plan
        self._sparse_receipt_transform = lambda inner: self._sparse_v3_receipt(plan, inner)
        self._active_motion_interaction = plan.interaction
        # Every segment keeps the move's speed; a pre-contact segment ends short
        # of the goal, so the slow final segment belongs to the local approach.
        self._active_motion_speed = plan.speed
        self._active_motion_slow_final = plan.strategy != "global_precontact_then_local"
        try:
            if plan.strategy in {"global_then_local_settle", "global_precontact_then_local"}:
                plan.routes.append("libero_curobo_joint_position")
                goal = plan.precontact or [*plan.target["position_m"], *plan.target["orientation_xyzw"]]
                first = self._execute_planned_motion({
                    **plan.identity,
                    "action": "move_free_space",
                    "arguments": {
                        "arm": "right",
                        "target_pose": {
                            "position_m": list(goal[:3]),
                            "orientation_xyzw": list(goal[3:]),
                        },
                    },
                })
                # The disclosure refuses the hand-off once the session reports
                # any terminal status after the first segment (official
                # success, LIBERO truncation, the sim-step cap, an abort or a
                # control fault), so the action ends with the first segment's
                # receipt and the loop meets the terminal status, rather than
                # the next segment raising on an already terminal episode.
                handoff = plan.precontact_disclosure if plan.precontact else plan.settle_disclosure
                if handoff is None or not handoff.get("attempted"):
                    return first
                landed = first["segments"][-1]["after"]["right"][
                    "pose_xyz_xyzw"
                ]
                second = self._run_sparse_local_segment(
                    plan,
                    np.asarray(landed[:3], dtype=np.float64),
                    np.asarray(landed[3:], dtype=np.float64),
                    settle_from=None if plan.precontact else first,
                )
                return merge_settled_receipt(first, second)

            return self._run_sparse_local_segment(
                plan, plan.before_position, plan.before_orientation,
            )
        finally:
            self._active_sparse_plan = previous_plan
            self._sparse_receipt_transform = previous_transform
            self._active_motion_interaction = previous_interaction
            self._active_motion_speed = previous_speed
            self._active_motion_slow_final = previous_slow_final

    def _run_sparse_local_segment(
        self, plan: _SparseMotionV3, from_position: Any, from_orientation: Any,
        *, settle_from: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The bounded OSC segment from a measured pose to the target."""

        import numpy as np

        plan.routes.append("libero_robosuite_osc_pose")
        plan.settle_disclosure = (
            {"attempted": True, "reason": None}
            if settle_from is not None else None
        )
        target = plan.target
        delta_rotation = _quaternion_error_rotvec(
            np.asarray(target["orientation_xyzw"], dtype=np.float64),
            from_orientation,
        )
        return self.execute({
            **plan.identity,
            "action": "servo_delta",
            "arguments": {
                "arm": "right",
                "target_pose": {
                    "position_m": list(target["position_m"]),
                    "orientation_xyzw": list(
                        target["orientation_xyzw"]
                    ),
                },
                "delta_pose": {
                    "translation_m": [
                        float(value)
                        for value in (plan.target_position - from_position)
                    ],
                    "rotation_rotvec_rad": [
                        float(value) for value in delta_rotation
                    ],
                },
                "frame": "world",
                "max_iterations": 20,
            },
        })

    def _disclose_sparse_precontact(
        self, plan: _SparseMotionV3, inner: Mapping[str, Any],
    ) -> None:
        """Record once whether the pre-contact segment may hand off locally."""

        import numpy as np

        if plan.precontact is None or plan.precontact_disclosure is not None:
            return
        target = plan.target
        landed = (inner.get("after") or {}).get("right", {}).get("pose_xyz_xyzw")
        measured = isinstance(landed, list) and len(landed) == 7
        reason = precontact_refusal(
            terminated=self.is_terminal,
            motion_status=str(inner.get("motion_status")),
            endpoint_measured=measured,
            within_local_envelope=measured and _within_local_servo_envelope(
                float(np.linalg.norm(plan.target_position - np.asarray(landed[:3]))),
                float(np.linalg.norm(_quaternion_error_rotvec(
                    np.asarray(target["orientation_xyzw"], dtype=np.float64),
                    np.asarray(landed[3:], dtype=np.float64)))),
            ),
        )
        plan.precontact_disclosure = {"pose_xyz_xyzw": plan.precontact,
                                      "attempted": reason is None, "reason": reason}

    def _disclose_sparse_settle(
        self, plan: _SparseMotionV3, inner: Mapping[str, Any],
    ) -> None:
        """Record once whether the global segment may be settled locally."""

        import numpy as np

        if not (plan.strategy == "global_then_local_settle"
                and plan.settle_disclosure is None):
            return
        # Only the global segment decides this. The settling segment
        # sets the disclosure before it runs, so this never fires
        # twice for one action.
        landed_pose = (inner.get("after") or {}).get("right", {}).get(
            "pose_xyz_xyzw"
        )
        measured = (
            isinstance(landed_pose, list) and len(landed_pose) == 7
        )
        residual_m = residual_rad = None
        if measured:
            residual_m = float(np.linalg.norm(
                plan.target_position
                - np.asarray(landed_pose[:3], dtype=np.float64)
            ))
            residual_rad = float(np.linalg.norm(
                _quaternion_error_rotvec(
                    np.asarray(
                        plan.target["orientation_xyzw"], dtype=np.float64,
                    ),
                    np.asarray(landed_pose[3:], dtype=np.float64),
                )
            ))
        bound_m, bound_rad = libero_local_endpoint_tolerances()
        reason = settle_refusal(
            strategy=plan.strategy,
            terminated=self.is_terminal,
            motion_status=str(inner.get("motion_status")),
            endpoint_measured=measured, residual_m=residual_m,
            residual_rad=residual_rad, bound_m=bound_m,
            bound_rad=bound_rad,
            within_local_envelope=bool(measured)
            and _within_local_servo_envelope(
                residual_m or 0.0, residual_rad or 0.0,
            ),
        )
        plan.settle_disclosure = {
            "attempted": reason is None, "reason": reason,
        }

    def _adopt_jaw_twin(
        self, plan: _SparseMotionV3, inner: Mapping[str, Any],
    ) -> None:
        """Once the global segment ran the jaw twin of its goal, the move's target is the twin too.

        ``planned_motion._jaw_twin_route`` runs the twin of a goal the pacing refused
        for its wrist (``controllers/jaw_flip_retry.py``) and says so in the
        segment's ``pacing.wrist.jaw_flip_retry``. The same grasp is then the
        target turned a half-turn about its approach axis, so the settle, the
        local approach after a pre-contact pose and the pose error all read that
        twin; a settle back to the request would turn the hand half a turn.
        """

        record = ((inner.get("pacing") or {}).get("wrist") or {}).get("jaw_flip_retry")
        if plan.jaw_twin or not isinstance(record, Mapping) or record.get("applied") is not True:
            return
        target = [*plan.target["position_m"], *plan.target["orientation_xyzw"]]
        plan.target = {**plan.target, "orientation_xyzw": jaw_twin(target)[3:]}
        plan.jaw_twin = True

    def _sparse_motion_report(
        self, plan: _SparseMotionV3, inner: Mapping[str, Any],
    ) -> dict[str, Any]:
        """The receipt's ``motion``: the global segment's pacing once it ran (or refused), else the caps."""

        pacing = inner.get("pacing")
        if isinstance(pacing, Mapping):
            plan.motion = dict(pacing)
        elif plan.motion is None:
            move, slow = paces(plan.speed)
            profile = getattr(self, "motion_profile", None)
            ceiling = (libero_global_joint_ceilings(profile, tick_s=LIBERO_CONTROL_PERIOD_S)
                       if isinstance(profile, Mapping) else (None, None))
            plan.motion = {"speed": speed_report(
                plan.speed, move.within(*ceiling), None if slow is None else slow.within(*ceiling),
            )}
        line = (plan.route_choice or {}).get("line")
        if line is not None and "planned_ticks" not in plan.motion:
            # The OSC line's own plan: its ticks before the settle (``planned_motion._osc_line_reference``).
            plan.motion = {**plan.motion, "route": "osc_straight_line", "planned_ticks": int(line["planned_ticks"]),
                           "duration_s": int(line["planned_ticks"]) * LIBERO_CONTROL_PERIOD_S}
        return dict(plan.motion)

    def _sparse_v3_receipt(
        self, plan: _SparseMotionV3, inner: dict[str, Any],
    ) -> dict[str, Any]:
        """Project one executor segment receipt into the canonical sparse receipt."""

        motion = self._sparse_motion_report(plan, inner)
        self._adopt_jaw_twin(plan, inner)
        self._disclose_sparse_precontact(plan, inner)
        self._disclose_sparse_settle(plan, inner)
        status, stage, code = _sparse_segment_outcome(plan, inner)
        pose_error = _sparse_pose_error(plan, inner)
        # The applied endpoint bound is compiled by the executing route and
        # only lifted here. This projection maps every non-refused,
        # non-reached, non-stopped result to endpoint_residual_exceeded
        # without re-comparing, so it must never re-derive a bound of its
        # own: an absent contract stays absent.
        endpoint_contract = inner.get("endpoint_contract")
        endpoint_contract = (
            dict(endpoint_contract)
            if isinstance(endpoint_contract, Mapping) else None
        )
        return {
            "schema": LIBERO_SPARSE_RECEIPT_SCHEMA,
            "action_id": plan.action["action_id"],
            "action": "move_pose",
            "arguments": dict(plan.arguments),
            "consumed": True,
            "state_changed": bool(inner.get("state_changed")),
            "execution_status": status,
            "motion_strategy": plan.strategy,
            "failure_stage": stage,
            "failure_code": code,
            "executor": list(plan.routes) or None,
            "target_arms": ["right"],
            "synchronization": {"mode": "single_arm"},
            "arms": {"right": {
                "targeted": True,
                "interaction": plan.interaction,
                "pose_error": pose_error,
                "endpoint_contract": endpoint_contract,
                **({"settle": dict(plan.settle_disclosure)}
                   if plan.settle_disclosure is not None else {}),
                **({"precontact": dict(plan.precontact_disclosure)}
                   if plan.precontact_disclosure is not None else {}),
                **({"route_choice": dict(plan.route_choice)} if plan.route_choice is not None else {}),
                "plan_validation": (
                    {"status": "kinematics_only"}
                    if plan.strategy == "global" else None
                ),
            }},
            "before": inner.get("before"),
            "after": inner.get("after"),
            "steps_used": inner.get("steps_used"),
            "sim_steps_total": inner.get("sim_steps_total"),
            "control_health": inner.get("control_health"),
            "motion": motion,
            "terminal_after_action": self.is_terminal,
            "segment": inner,
            "segments": [inner],
            "finished_unix_s": time.time(),
        }
