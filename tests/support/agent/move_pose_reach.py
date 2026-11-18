"""Test helpers for move pose reach, shared by several test modules.

They were defined in ``tests/ora/agent/toolkit/host/reach/test_move_pose_reach.py``; test
modules import them from here, never from another test module (rule L8 of
``docs/developer/repository-structure.md``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from tests.support.agent.surface_catalog import surface_of
from ora.agent.composition.schemas.move_pose_reach import LINE_FRACTIONS
from ora.contracts.reach_check import RESULT_SCHEMA
from ora.environment.controllers.reach_check import joint_delta_rad, joint_limit_margin_rad


DOJO = "robodojo_placement_toolkit_v5"

X5_LIMITS = [[-9.999999046325684, 9.999999046325684]] * 5 + [[-3.1399998664855957, 3.1399998664855957]]
HOME = [0.0] * 6

#: The round-1 absolute quaternion every move used.
REQUESTED = [-0.428955228, -0.473037167, 0.570079933, -0.51695466]
RIGHT_FROM_HOME = [0.676, 2.179, 1.97, -1.445, -0.192, -2.465]


def _local_y(orientation):
    x, y, z, w = orientation
    return np.array([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)])


def _turned(joints):
    shifted = joints[5] + math.pi if joints[5] < 0 else joints[5] - math.pi
    return [*joints[:5], shifted]


def _row(joints, reference, *, reason=None):
    if joints is None:
        return {"reachable": False, "reason": reason or "ik_failed", "route": None, "position_error_m": None,
                "orientation_error_rad": None, "joint_limit_margin_rad": None, "joint_delta_rad": None}
    return {"reachable": True, "reason": None, "route": "global", "position_error_m": 0.0001,
            "orientation_error_rad": 0.001,
            "joint_limit_margin_rad": round(joint_limit_margin_rad(joints, X5_LIMITS), 5),
            "joint_delta_rad": round(joint_delta_rad(joints, reference), 5)}


def _bridge(requests, *, requested=REQUESTED, exact=None, turned=None, measured=HOME, line=None):
    """Answer by member: the requested jaw direction is ``exact``, its reverse ``turned``.

    ``line`` answers the reach-along samples (a fraction -> reachable map);
    a sample is recognised by its position between start and target.
    """
    reference_y = _local_y(requested)

    def bridge(request):
        requests.append(copy.deepcopy(request))
        rows = []
        for query in request["queries"]:
            if line is not None and len(request["queries"]) == len(LINE_FRACTIONS):
                index = len(rows)
                rows.append(_row(HOME if line(LINE_FRACTIONS[index]) else None, measured))
                continue
            is_exact = float(np.dot(_local_y(query["pose_xyzw"][3:]), reference_y)) > 0
            joints, reason = (exact if is_exact else turned) or (None, None)
            rows.append(_row(joints, measured, reason=reason))
        return {"schema": RESULT_SCHEMA, "arm": request["arm"],
                "ik_model": "robodojo_sdk_ik_endpoint_selection_v2", "matches_move_pose_admission": True,
                "motion_planner_called": False, "claim_boundary": "stub IK", "results": rows}
    return bridge


def _dojo_runtime(tmp_path):
    import tests.support.robodojo.closed_skills_default as parity
    runtime = parity.tool_runtime(tmp_path, handlers=True)  # its tests leave with the handlers (P21)
    runtime.surface = surface_of(DOJO)
    runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    pointer, source, observation = runtime._current()
    # A submission binds the observation generation and digest the runtime publishes.
    pointer = {**pointer, "generation": 3, "observation_sha256": hashlib.sha256(
        json.dumps(observation, sort_keys=True).encode()).hexdigest()}
    runtime._current = lambda: (pointer, source, observation)
    return runtime
