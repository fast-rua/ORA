"""Test helpers for reach check, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/reach/test_reach_check.py``; test modules
import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace


from tests.support.agent.surface_catalog import surface_of
from ora.contracts.reach_check import REQUEST_SCHEMA, RESULT_SCHEMA


ARRIVAL = {"profile": {"arrival_check": {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05}}}


def planner_ik(session, monkeypatch, refuse=lambda pose: None):
    """The shared planner's IK on the fake plant: joints are the pose, as the fake FK reads them.

    ``refuse(pose)`` returns a refusal, ``"unsolved"`` for no solution, or ``None``. Returns the requests."""
    from ora.environment.backends.robodojo import shared_motion

    requests = []

    def solve_ik(request):
        requests.append(request)
        results = []
        for query in request["queries"]:
            pose, start = query["pose_base_xyzw"], query["from_index"]
            blank = {"solved": False, "skipped": False, "refusal": None, "joint_positions": None}
            if start is not None and results[start]["joint_positions"] is None:
                results.append({**blank, "skipped": True})
                continue
            refusal = refuse(pose)
            if refusal is not None:
                results.append({**blank, "refusal": None if refusal == "unsolved" else refusal})
                continue
            results.append({**blank, "solved": True, "joint_positions": [float(item) for item in pose[:6]]})
        return {"status": "solved", "reason": None, "joint_limits": [[-3.14] * 6, [3.14] * 6], "results": results,
                "timing_s": {"total": 0.0}}

    session.motion_planner = SimpleNamespace(solve_ik=solve_ik, close=lambda: None)
    monkeypatch.setattr(shared_motion, "world_input", lambda session, arm: (
        {"path": "world.npz"}, [float(item) for item in session._proprioception()["end_effectors"][arm]["arm_joint_qpos"]]))
    monkeypatch.setattr(shared_motion, "_base_pose", lambda session, arm, pose: [float(item) for item in pose])
    return requests


def toy_reach(limits, measured):
    """``_reach`` of a toy arm whose joints are its pose: that one solution, or none out of range."""
    from scipy.spatial.transform import Rotation

    from ora.contracts.reach_check import START_UNREACHABLE
    from ora.contracts.robotwin_reach import CLAIM_BOUNDARY, IK_FAILED, IK_MODEL
    from ora.environment.controllers.reach_check import joint_delta_rad, joint_limit_margin_rad

    blank = {"reachable": False, "route": None, "position_error_m": None, "orientation_error_rad": None,
             "joint_limit_margin_rad": None, "joint_delta_rad": None}

    def reach(arm, queries, *, position_tolerance_m, orientation_tolerance_rad):
        rows, ends = [], []
        for query in queries:
            start = query["from_index"]
            seed = measured if start is None else ends[start]
            if seed is None:
                rows.append({**blank, "reason": START_UNREACHABLE})
                ends.append(None)
                continue
            pose = query["pose_xyzw"]
            joints = [*map(float, pose[:3]), *map(float, Rotation.from_quat(pose[3:]).as_rotvec())]
            if not all(low <= value <= high for value, (low, high) in zip(joints, limits)):
                rows.append({**blank, "reason": IK_FAILED})
                ends.append(None)
                continue
            rows.append({"reachable": True, "reason": None, "route": None, "position_error_m": 0.0,
                         "orientation_error_rad": 0.0, "joint_limit_margin_rad": joint_limit_margin_rad(joints, limits),
                         "joint_delta_rad": joint_delta_rad(joints, seed)})
            ends.append(joints)
        return {"ik_model": IK_MODEL, "matches_move_pose_admission": False, "claim_boundary": CLAIM_BOUNDARY,
                "results": rows}
    return reach


def _robotwin_adapter(limits):
    """The real adapter methods over a toy arm whose joints are its pose."""
    from ora.environment.backends.robotwin.motion_adapter import RoboTwinMotionBackendAdapter
    from scipy.spatial.transform import Rotation

    adapter = object.__new__(RoboTwinMotionBackendAdapter)
    adapter.read_joint_positions = lambda arm: [0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
    adapter.joint_limits = lambda arm: limits
    adapter.fk_public_tcp_pose = lambda arm, joints: [*joints[:3], *Rotation.from_rotvec(joints[3:6]).as_quat()]
    adapter._reach = toy_reach(limits, [0.0, 0.0, 0.9, 0.0, 0.0, 0.0])
    return adapter


def _dojo_runtime(tmp_path, grasp_backend=None):
    import tests.support.robodojo.closed_skills_default as parity
    runtime = parity.tool_runtime(tmp_path, handlers=True)  # its tests leave with the handlers (P21)
    runtime.surface = surface_of("robodojo_placement_toolkit_v5")
    runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    if grasp_backend is not None:
        runtime.grasp_backend = grasp_backend
    return runtime


def _local_x(orientation):
    x, y, z, w = orientation
    return [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]


def _local_y(orientation):
    x, y, z, w = orientation
    return [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)]


def _stub_bridge(requests, *, reachable=lambda query: True, reason="ik_failed",
                 ik_model="robodojo_sdk_ik_endpoint_selection_v1", parity=True):
    def bridge(request):
        requests.append(copy.deepcopy(request))
        assert request["schema"] == REQUEST_SCHEMA
        rows = []
        for query in request["queries"]:
            start = query["from_index"]
            if start is not None and not rows[start]["reachable"]:
                rows.append({"reachable": False, "reason": "start_unreachable", "route": None,
                             "position_error_m": None, "orientation_error_rad": None,
                             "joint_limit_margin_rad": None, "joint_delta_rad": None})
                continue
            ok = reachable(query)
            rows.append({"reachable": ok, "reason": None if ok else reason,
                         "route": "global" if parity else None,
                         "position_error_m": 0.0012345678 if ok else None,
                         "orientation_error_rad": 0.0 if ok else None,
                         "joint_limit_margin_rad": 1.0 if ok else None,
                         "joint_delta_rad": 0.2 if ok else None})
        return {"schema": RESULT_SCHEMA, "arm": request["arm"], "ik_model": ik_model,
                "matches_move_pose_admission": parity, "motion_planner_called": False,
                "claim_boundary": "stub IK", "results": rows}
    return bridge


def _tilted(angle_deg):
    radians = math.radians(angle_deg)
    return [math.sin(radians), 0.0, -math.cos(radians)]


#: (score, approach tilt from straight down in degrees, x offset)
POOL = [(0.9, 180.0, 0.0), (0.8, 40.0, 0.0), (0.7, 0.0, 0.0), (0.6, 20.0, 0.0), (0.5, 0.0, 0.1)]


class PoolProposer:
    """Five X5 candidates; honours the requested count and records it."""
    backend_id = "contact_graspnet_test_double"
    backend_version = "v1"

    def __init__(self):
        self.requested = []

    def configuration(self):
        return {"pregrasp_offset_m": 0.04}

    def propose(self, **kwargs):
        self.requested.append(kwargs["max_candidates"])
        candidates = []
        for index, (score, tilt, x) in enumerate(POOL):
            approach = _tilted(tilt)
            origin = [x, 0.0, 0.8]
            candidates.append({
                "candidate_id": f"grasp-pool{index:020d}",
                "source_index": index, "rank": index,
                "source_gripper_origin_xyz_m": origin,
                "source_pregrasp_origin_xyz_m": [o - a * 0.04 for o, a in zip(origin, approach)],
                "approach_world": approach, "jaw_world": [0.0, 1.0, 0.0],
                "gripper_opening_m": 0.05, "backend_score": score,
                "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
                "checks": {"workspace": "passed", "aperture": "passed",
                           "ik": "not_checked", "collision": "not_checked"},
                "execution_ready": False,
            })
        return candidates[:kwargs["max_candidates"]], {"request_id": "grasp-pool", "rejected": []}


def _pool_ik(query):
    """Everything but the upward approach; the x=0.1 candidate only as its jaw flip."""
    orientation = query["pose_xyzw"][3:]
    if _local_x(orientation)[2] > 0.5:
        return False
    if abs(query["pose_xyzw"][0] - 0.1) < 0.05 and _local_y(orientation)[1] > 0:
        return False
    return True


def _pool_grasp(tmp_path, bridge, arm="left"):
    backend = PoolProposer()
    runtime = _dojo_runtime(tmp_path, grasp_backend=backend)
    runtime.align_plan_callback = bridge
    grounded = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
    result = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"], "arm": arm})
    return runtime, backend, result
