"""RoboTwin's read-only answer to a dry run (``contracts/action_plan.py``): what the sparse route would run.

A mixin of ``session.RoboTwinLiveSession``. It admits and routes the action as ``_execute_sparse_motion`` does, plans,
validates and times it with the same code and stops before any controller write. The shared planner keeps the answers,
so the real call runs the path shown. A local move runs on the Cartesian servo, which has no path ahead of time: the
answer draws the line it is commanded along, marked unplanned, as it does for the settle after a planned path. A
recovery after a blocked move retreats first and plans from where that stops, which no plan can answer before it runs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.action_plan import body_points, decimated, planned, refused, segment, unavailable
from ora.environment.controllers.kept_plans import previewing

from .common import RoboTwinLiveError, _validate_workspace_target
from .sparse_planning import RetreatFirst

SETTLE = "the servo settles onto the goal"
APPROACH = "the servo approaches along this line"
SERVO = "the servo is commanded along this line; it has no path"


def _target(pose: Mapping[str, Any]) -> list[float]:
    return [*pose["position_m"], *pose["orientation_xyzw"]]


class RoboTwinActionPlanMixin:
    """Serve ``ora.action-plan-request.v1`` on the read-only bridge."""

    def plan_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        state = self._admit_sparse_motion(action=action, receipt_path=Path(os.devnull), started=time.time())
        state.dry_run = True
        self._route_motion_strategy(state)
        if state.motion_strategy_error is not None:
            return refused("motion_intent_rejected", "route", state.motion_strategy_error)
        if not state.pose_action:
            return unavailable(f"{action['action']} moves nothing a plan can answer for")
        if state.local_cartesian_requested:
            return self._planned_servo(state)
        try:
            with previewing(getattr(getattr(self, "shared_motion", None), "client", None)):
                admitted = self._plan_and_validate(state)
        except RetreatFirst:
            return unavailable("a recovery after a blocked move first retreats along its approach, and its plan starts "
                               "from where that stops")
        except RoboTwinLiveError as error:
            return refused("plan_rejected", "admission", str(error))
        if not admitted:
            return self._plan_refusal(state)
        if not self._pace_validated_plans(state):
            return self._plan_refusal(state)
        return self._planned_paths(state)

    def _planned_servo(self, state: Any) -> dict[str, Any]:
        segments = []
        for arm in state.target_arms:
            try:
                _validate_workspace_target(list(state.targets[arm]["position_m"]), self.workspace_bounds_m)
            except RoboTwinLiveError as error:
                return refused("plan_rejected", "admission", str(error))
            segments.append(segment("arm", arm, SERVO, planned=False,
                                    poses=[state.before[arm]["pose_xyz_xyzw"], _target(state.targets[arm])]))
        return planned("servo", segments, body=self._arm_chains(state))

    @staticmethod
    def _plan_refusal(state: Any) -> dict[str, Any]:
        for arm in state.target_arms:
            rejection = state.plan_rejections.get(arm)
            if rejection is not None:
                detail = dict(rejection.get("details") or {})
                return refused(rejection["code"], str(detail.pop("stage", "plan_validation")), rejection.get("message"),
                               **detail)
        result = next((state.planner_results[arm] for arm in state.target_arms
                       if not isinstance(state.planner_results[arm], Mapping)
                       or state.planner_results[arm].get("status") != "Success"), None)
        code = result.get("planner_status") or result.get("status") if isinstance(result, Mapping) else None
        return refused(str(code or "planner_failed"), "planner")

    def _planned_paths(self, state: Any) -> dict[str, Any]:
        settle = state.global_then_local_settle_requested
        approach = state.global_precontact_then_local_requested
        motion = state.pacing["motion"] or {}
        duration = motion.get("duration_s")
        segments = []
        for arm, rows in state.pacing["rows"].items():
            poses = [self._pinocchio_public_pose_xyzw(state.environment, arm, row) for row in decimated(rows)]
            segments.append(segment("arm", arm, "planned path", planned=True,
                                    poses=[state.before[arm]["pose_xyz_xyzw"], *poses], native_actions=len(rows),
                                    duration_s=duration))
            if settle or approach:
                segments.append(segment("arm", arm, APPROACH if approach else SETTLE, planned=False,
                                        poses=[poses[-1], _target(state.targets[arm])]))
        return planned("global", segments, body=self._arm_chains(state))

    def _arm_chains(self, state: Any) -> dict[str, list[list[float]]]:
        environment = state.environment
        return {arm: body_points(lambda rows, selected=arm: [
            self._pinocchio_public_pose_xyzw(environment, selected, row) for row in rows],
            self._measured_arm_joints(environment, arm)) for arm in state.target_arms}


__all__ = ["RoboTwinActionPlanMixin"]
