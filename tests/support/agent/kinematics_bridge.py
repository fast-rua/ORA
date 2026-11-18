"""A toy environment bridge for the code-mode query tests: the read-only IK bridge and the motion probe.

IK is a linear map of the position, so a query's answer is checkable by hand. The answers have the shape of the
environment's (``ora.kinematics-result.v1``, ``ora.reach-check-result.v1``, ``ora.align-plan-result.v1``,
``ora.motion-probe-result.v1``); the repository holds no recorded kinematics answer.
"""

from __future__ import annotations

import json
import math
from typing import Any

from tests.support.agent.paced_kinematics import paced


class Bridge:
    """The environment's read-only IK bridge and motion probe, as a toy: IK is a linear map of the position."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.probed: list[dict[str, Any]] = []
        #: The planner's answer by goal x (``plan_path``): a refusal's reason, or planned; none answers without it.
        self.refusals: dict[float, str] = {}
        self.planner = False

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(json.loads(json.dumps(request)))
        schema = request["schema"]
        if schema == "ora.kinematics-request.v1":
            return getattr(self, request["operation"])(request)
        if schema == "ora.reach-check-request.v1":
            return {"schema": "ora.reach-check-result.v1", "arm": request["arm"], "ik_model": "toy",
                    "matches_move_pose_admission": True, "motion_planner_called": False, "claim_boundary": "toy",
                    "results": [{"reachable": query["pose_xyzw"][0] < 0.5, "reason": None, "route": "global",
                                 "position_error_m": 0.0003, "orientation_error_rad": 0.01,
                                 "joint_limit_margin_rad": 0.5, "joint_delta_rad": 0.2 * (index + 1)}
                                for index, query in enumerate(request["queries"])]}
        return {"schema": "ora.align-plan-result.v1", "target_ref": request["target_ref"],
                "approach_side": request["approach_side"], "requested_arm": request["arm"],
                "motion_planner_called": False,
                "candidates": [{"source_index": index, "arm": request["arm"], "reachable": True,
                                "position_error_m": 0.0002, "orientation_error_rad": 0.02, "iterations": 7,
                                "rejection_reason": None, "claim_boundary": "toy",
                                "align_pose": {"position_m": pose[:3], "orientation_xyzw": pose[3:]}}
                               for index, pose in enumerate(request["target_poses_xyzw"])]}

    def ik_chain(self, request: dict[str, Any]) -> dict[str, Any]:
        rows = [{"index": index, "status": "ok" if pose[0] < 0.5 else "ik_failed",
                 "joints_rad": [pose[0] * 10, pose[1] * 10, (pose[2] - 0.8) * 10, 0.1 * index, 0.0, 0.2]
                 if pose[0] < 0.5 else None, "joint_step_rad": 0.05 * (index + 1), "position_error_m": 0.0001,
                 "orientation_error_rad": 0.002, "limit_margin_rad": 0.4, "canonical": True,
                 "pitch_joint_wrap": False, "path": "clear", "valid_candidates": 2, "seeding": None}
                for index, pose in enumerate(request["poses_xyz_xyzw"])]
        failed = [row["index"] for row in rows if row["status"] != "ok"]
        return {"schema": "ora.kinematics-result.v1", "operation": "ik_chain", "arm": request["arm"],
                "seed_joints_rad": request["seed_joints_rad"], "rows": rows,
                "first_failure_index": failed[0] if failed else None, "ik_model": "toy", "seeding": "toy",
                "timings_s": {"ik": 0.01, "fk": 0.0, "screen": 0.0}, "total_s": 0.01, "claim_boundary": "toy"}

    def fk(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"schema": "ora.kinematics-result.v1", "operation": "fk", "arm": request["arm"],
                "rows": [{"index": index, "pose_xyz_xyzw": [joints[0] / 10, joints[1] / 10, 0.8 + joints[2] / 10,
                                                            0.0, 0.0, math.sin(joints[5] / 2), math.cos(joints[5] / 2)]}
                         for index, joints in enumerate(request["joints_rad"])]}

    def screen_path(self, request: dict[str, Any]) -> dict[str, Any]:
        knots = len(next(iter(request["knots_rad"].values())))
        return {"schema": "ora.kinematics-result.v1", "operation": "screen_path", "status": "clear",
                "first_blocked_segment": None, "segments_screened": knots - 1, "min_inter_arm_clearance_m": 0.21,
                "claim_boundary": "toy"}

    def plan_path(self, request: dict[str, Any]) -> dict[str, Any]:
        common = {"schema": "ora.kinematics-result.v1", "operation": "plan_path", "arm": request["arm"]}
        if not self.planner:
            return {**common, "status": "unavailable", "reason": "no_planner"}
        goal = request["goal_pose_xyz_xyzw"]
        if goal[0] in self.refusals:
            return {**common, "status": "refused", "reason": self.refusals[goal[0]]}
        start = [0.0] * 6
        end = [goal[0] * 10, goal[1] * 10, (goal[2] - 0.8) * 10, 0.0, 0.0, 0.0]
        return {**common, "status": "planned", "reason": None, "start_joints": start,
                "path_joints": [[a + (b - a) * index / 8 for a, b in zip(start, end)] for index in range(9)]}

    def pace_path(self, request: dict[str, Any]) -> dict[str, Any]:
        return paced(request, lambda rows: [[row[0] / 10, row[1] / 10, 0.8 + row[2] / 10, 0.0, 0.0,
                                             math.sin(row[5] / 2), math.cos(row[5] / 2)] for row in rows])

    def probe(self, request: dict[str, Any]) -> dict[str, Any]:
        self.probed.append(json.loads(json.dumps(request)))
        results = []
        for item in request["queries"]:
            target = [*item["pose"]["position_m"], *item["pose"]["orientation_xyzw"]]
            results.append({
                "query_id": item["query_id"], "arm": item["arm"], "reachable": True, "selected_variant_id": "exact",
                "plan_ref": f"plan-{item['query_id']}", "selected_target_pose_xyzw": target,
                "variants": [{"variant_id": "exact", "reachable": True, "target_pose_xyzw": target,
                              "trajectory_steps": 12, "trajectory_duration_s": 1.5,
                              "joint_delta_rad": [0.1, 0.2, 0.3, 0.0, 0.0, 0.5], "plan_ref": f"plan-{item['query_id']}",
                              "planner": {"called": True, "status": "success"}}]})
        return {"schema": "ora.motion-probe-result.v1", "observation_id": request["observation_id"],
                "variant_policy": request["variant_policy"], "robot_decisions_consumed": 0,
                "world_state_changed": False, "results": results}
