"""LIBERO's read-only answer to a dry run (``contracts/action_plan.py``): what the sparse route would run.

A mixin of ``session.LiberoProSession``. It picks the route as ``_plan_sparse_motion_v3`` does, asks the planner the
question ``_execute_planned_motion`` asks and times the answer with the same pacing, then stops before the first step.
The plan is kept, so the real call runs the path shown; under motion profile v4 a clear, nearly straight
free-space plan is answered as the straight OSC line the real call then runs (``planned_motion._free_space_route_choice``). A move whose path the pacing refuses for its wrist is retried to
its jaw twin exactly as the real call retries it (``planned_motion._jaw_twin_route``), and the answer then shows the
twin's path and settle, its first segment naming the retry (``jaw_flip_retry``). A local move runs on the OSC servo,
which has no path ahead of time: the answer draws the line it is commanded along, marked unplanned, as it does for the
settle after a planned path.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.contracts.workspace import workspace_state
from ora.contracts.action_plan import body_points, planned, refused, segment, unavailable
from ora.environment.controllers.jaw_flip_retry import jaw_twin
from ora.environment.controllers.kept_plans import previewing
from ora.environment.controllers import osc_transit
from ora.environment.controllers.motion_pace import PaceRefusal

from .planner_request import failure_reason

SETTLE = "the servo settles onto the goal"
APPROACH = "the servo approaches along this line"
SERVO = "the servo is commanded along this line; it has no path"
OSC_LINE = ("the servo follows this straight line; the planner checked the gripper's path on it, "
            "not the arm's posture")


class LiberoActionPlanMixin:
    """Serve ``ora.action-plan-request.v1`` on the read-only bridge."""

    def plan_action(self, action: Mapping[str, Any]) -> dict[str, Any]:
        name, arguments = action["action"], action["arguments"]
        if name == "move_pose" and set(arguments) == {"plan_ref"}:
            return unavailable("a plan reference was planned by its probe")
        if set(arguments["targets"]) != {"right"}:
            return refused("arm_not_available", "admission")
        if name == "recover_motion":
            return self._plan_recovery(arguments["targets"]["right"])
        if name != "move_pose":
            return unavailable(f"{name} moves nothing a plan can answer for")
        plan = self._plan_sparse_motion_v3(action, {})
        if plan.strategy == "local":
            here = [*plan.before_position.tolist(), *plan.before_orientation.tolist()]
            there = [*plan.target["position_m"], *plan.target["orientation_xyzw"]]
            return planned("servo", [segment("arm", "right", SERVO, planned=False, poses=[here, there])],
                           body={"right": self._arm_chain(self._motion_state())})
        speed, slow = getattr(self, "_active_motion_speed", None), getattr(self, "_active_motion_slow_final", True)
        interaction = getattr(self, "_active_motion_interaction", None)
        self._active_motion_speed = plan.speed
        self._active_motion_slow_final = plan.strategy != "global_precontact_then_local"
        self._active_motion_interaction = plan.interaction
        try:
            goal = plan.precontact or [*plan.target["position_m"], *plan.target["orientation_xyzw"]]
            if not workspace_state(goal[:3], self.workspace_bounds_m)["inside"]:
                return refused("target_outside_workspace", "admission")
            state = self._motion_state()
            request = {"joint_names": state["joint_names"], "joint_positions": state["joint_positions"],
                       "target_pose_base_xyzw": self._grip_target_planner_pose(state, goal)[0]}
            return self._planned_route(request, state, plan, goal)
        finally:
            self._active_motion_speed, self._active_motion_slow_final = speed, slow
            self._active_motion_interaction = interaction

    def _plan_recovery(self, strategy: str) -> dict[str, Any]:
        goal = self.home_joint_state if strategy == "home" else self.last_backtrack_joint_state
        if goal is None:
            return refused("recovery_state_unavailable", "plan_validation")
        state = self._motion_state()
        return self._planned_route({"joint_names": state["joint_names"], "joint_positions": state["joint_positions"],
                                    "goal_joint_positions": list(goal)}, state, None)

    def _planned_route(self, request: dict[str, Any], state: Mapping[str, Any], plan: Any,
                       goal: list[float] | None = None) -> dict[str, Any]:
        """The global segment to ``goal`` (None: a recovery's joint goal, never retried) and what follows it."""
        request.update(self._world_input(state, "preview"))  # the real call writes its own: a plan is keyed by the hash
        retried = None
        with previewing(self.motion_planner.client):  # the twin's plan is kept too
            answer = self.motion_planner.plan(request)
            if not answer.get("reachable") or not isinstance(answer.get("trajectory"), Mapping):
                return refused(failure_reason(answer), "planner")
            self._validate_planned_start(answer, state)
            try:
                paced = self._pace_global_route(answer["trajectory"]["positions"])
            except PaceRefusal as refusal:
                twin = None if goal is None else self._jaw_twin_route(refusal, request, state, goal, persist=False)
                if twin is None:
                    return refused(refusal.code, "pacing", str(refusal), **refusal.constraints)
                paced = twin[3]
                retried = dict(paced.wrist["jaw_flip_retry"])
        start = state["grip_site_pose_world_xyzw"]
        if plan is not None and goal is not None and retried is None:
            # v4 runs a clear, nearly straight free-space plan as the OSC line: show that line.
            choice = self._free_space_route_choice({"action": "move_free_space"}, False, paced, goal)
            if choice is not None and choice["route"] == osc_transit.ROUTE_OSC:
                line = self._osc_line_reference(goal)
                return planned("osc_line", [segment(
                    "arm", "right", OSC_LINE, planned=True, poses=[start, *line["aims"]] if line["aims"] else [start, goal],
                    native_actions=line["ticks"], duration_s=line.get("duration_s"), route_choice=choice)],
                    body={"right": self._arm_chain(state)})
        report, rows = paced.report(), paced.rows()
        poses = self._grip_site_kinematics(rows)["poses"]
        segments = [segment("arm", "right", "planned path", planned=True, poses=[start, *poses],
                            native_actions=report["planned_ticks"], duration_s=report["duration_s"],
                            **({"jaw_flip_retry": retried} if retried is not None else {}))]
        if plan is not None:
            target = [*plan.target["position_m"], *plan.target["orientation_xyzw"]]
            segments.append(segment("arm", "right", APPROACH if plan.precontact else SETTLE, planned=False,
                                    poses=[poses[-1], jaw_twin(target) if retried is not None else target]))
        return planned("global", segments, body={"right": self._arm_chain(state)})

    def _arm_chain(self, state: Mapping[str, Any]) -> list[list[float]]:
        return body_points(lambda rows: self._grip_site_kinematics(rows)["poses"], state["joint_positions"])


__all__ = ["LiberoActionPlanMixin"]
