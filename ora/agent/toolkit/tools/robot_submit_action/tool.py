"""``robot.submit_action``: one observation-bound target state, optional pose and gripper, for the Supervisor to execute."""

import math
from collections.abc import Mapping

from ...lib import observation
from ...lib._workspace import is_monotonic_workspace_recovery, workspace_state
from ...lib.values import PublicResult
from ..errors import ToolRefusal
from ..geometry_query._arguments import _finite_vector


def _measured_position(manifest, arm):
    """The arm's measured position, or None where the observation publishes no whole pose."""
    state = manifest.get("proprioception", {}).get("end_effectors", {}).get(arm)
    pose = state.get("pose_xyz_xyzw") if isinstance(state, Mapping) else None
    if not isinstance(pose, list) or len(pose) != 7:
        return None
    try:
        return [float(item) for item in pose[:3]]
    except (TypeError, ValueError):
        return None


def _target_pose(pose, manifest, arm, bounds):
    """The pose inside the workspace, or on the way back into it; metres and xyzw."""
    if not isinstance(pose, Mapping) or set(pose) != {"position_m", "orientation_xyzw"}:
        raise ToolRefusal("pose must contain position_m and orientation_xyzw", field="pose")
    position = _finite_vector(pose["position_m"], 3, "position_m")
    orientation = _finite_vector(pose["orientation_xyzw"], 4, "orientation_xyzw")
    measured = _measured_position(manifest, arm)
    workspace = workspace_state(position, bounds)
    recovery = measured is not None and is_monotonic_workspace_recovery(measured, position, bounds)
    if not workspace["inside"] and not recovery:
        for index, axis in enumerate(("x", "y", "z")):
            details = workspace["violations"].get(axis)
            if details is not None:
                interval = details["allowed"]
                raise ToolRefusal(f"position {axis} is outside workspace bounds", code="out_of_bounds",
                                  field=f"pose.position_m[{index}]", received=position[index],
                                  constraints={"minimum": interval[0], "maximum": interval[1]})
    return {"position_m": position, "orientation_xyzw": orientation}


def run(api, arguments):
    """Replay: recorded observation and action; the receipt comes with the next observation. Never executes here.

    Pose and gripper are independently optional; what a call omits holds at execution time.
    """
    config = api.artifacts.tool_context()["robot_submit_action"]
    if config["action_pending"]:
        raise ToolRefusal("an action is already submitted for this observation", code="duplicate_action", retryable=False)
    if set(arguments) - {"arm", "pose", "gripper"}:
        raise ToolRefusal("target state fields are arm, optional pose and gripper")
    arm = arguments.get("arm")
    if arm not in {"left", "right"}:
        raise ToolRefusal("arm must be left or right", field="arm", received=arm, constraints={"enum": ["left", "right"]})
    normalized = {"arm": arm}
    obs = observation.observe(api)
    pose = arguments.get("pose")
    if pose is not None:
        normalized["pose"] = _target_pose(pose, obs.manifest.json(), str(arm), config["workspace_bounds_m"])
    if arguments.get("gripper") is not None:
        gripper = arguments["gripper"]
        if isinstance(gripper, bool) or not isinstance(gripper, (int, float)) or not math.isfinite(gripper) or not 0 <= gripper <= 1:
            raise ToolRefusal("gripper must be in [0,1]", field="gripper", received=gripper,
                              constraints={"minimum": 0, "maximum": 1})
        normalized["gripper"] = float(gripper)
    identity = obs.identity.json()
    action = {"schema": "ora-action.v4", "observation_id": identity["observation_id"],
              "action_id": f"action-{config['generation']}", "action": "target_state", "arguments": normalized,
              "bound_observation_sha256": identity["observation_sha256"]}
    return PublicResult.of(api.act.submit_action(action=action)).data.json()
