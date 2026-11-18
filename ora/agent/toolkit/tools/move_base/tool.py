"""Drive the mobile base to a pose on the floor (after ASPIRE's navigate_to_pose).

The target is either absolute in the world frame (``x_m``, ``y_m``, ``yaw_deg``) or relative to the base pose in the
current observation (``forward_m``, ``left_m``, ``turn_deg``). Either form becomes one motion-v3 ``move_base`` action
with a world ``(x, y, yaw)`` target; the backend's base servo owns the tolerances, speed and contact stop. Only a surface
declaring ``mobile_base_navigation_v1`` has the tool, and only where the environment plans each move: the tool hands the
action over and returns, and the receipt comes with the next observation.
"""
import math

from ...lib.observation import observe
from ..errors import ToolRefusal
from ..move_pose import _context
from ..move_pose.tool import propose

CAPABILITY = "mobile_base_navigation_v1"
ABSOLUTE = frozenset({"x_m", "y_m", "yaw_deg"})
RELATIVE = frozenset({"forward_m", "left_m", "turn_deg"})
#: The base travel bound, refused here before submission. The RoboCasa base servo refuses the same bound
#: (controllers/robocasa_omron.BASE_MAX_TRAVEL_M); tests/ora/agent/toolkit/tools/move_base keeps the two equal.
MAX_TRAVEL_M = 6.0


def _number(arguments, key):
    value = arguments.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolRefusal(f"{key} must be a finite number", field=key, received=value)
    return float(value)


def _base_pose(manifest):
    """The measured base pose; a call with no measured arm pose is refused first, as every move is."""
    proprioception = manifest.get("proprioception", {})
    effectors = proprioception.get("end_effectors", {})
    state = effectors.get("right") or effectors.get("left")
    if not isinstance(state, dict):
        raise ToolRefusal("current grip-site state is unavailable", code="pose_state_unavailable")
    pose = state.get("pose_xyz_xyzw")
    if not isinstance(pose, list) or len(pose) != 7:
        raise ToolRefusal("current grip-site pose is unavailable", code="pose_state_unavailable")
    base = proprioception.get("base")
    pose = base.get("pose_xy_yaw") if isinstance(base, dict) else None
    if not isinstance(pose, list) or len(pose) != 3:
        raise ToolRefusal("the current base pose is unavailable", code="pose_state_unavailable")
    return float(pose[0]), float(pose[1]), float(pose[2])


def run(api, arguments):
    """Replay: the recorded observation and proposal; a different target is divergence.

    Live hands one ``move_base`` action to the environment, which drives the base to it. Metres and degrees in, the
    action's yaw in radians, wrapped to (-pi, pi].
    """
    config = api.artifacts.tool_context()
    refusal = config["motion"]["mutation_refusals"].get("move_base")
    if refusal is not None:
        raise ToolRefusal.from_public(refusal)
    if CAPABILITY not in config["motion"]["capabilities"] or not _context.environment_plans(config):
        raise ToolRefusal("move_base is not selected on this surface", code="tool_unavailable", retryable=False)
    provided = {key for key, value in arguments.items() if value is not None}
    obs = observe(api)
    x, y, yaw = _base_pose(obs.manifest.json())
    if provided == ABSOLUTE:
        target_x, target_y = _number(arguments, "x_m"), _number(arguments, "y_m")
        target_yaw = math.radians(_number(arguments, "yaw_deg"))
        bound_field = "x_m"
    elif provided and provided <= RELATIVE:
        forward = _number(arguments, "forward_m") if "forward_m" in provided else 0.0
        left = _number(arguments, "left_m") if "left_m" in provided else 0.0
        turn = math.radians(_number(arguments, "turn_deg")) if "turn_deg" in provided else 0.0
        target_x = x + math.cos(yaw) * forward - math.sin(yaw) * left
        target_y = y + math.sin(yaw) * forward + math.cos(yaw) * left
        target_yaw = yaw + turn
        bound_field = "forward_m"
    else:
        missing = [key for key in ("x_m", "y_m", "yaw_deg") if key not in provided]
        partial = (f"the absolute target is missing {' and '.join(missing)}: " if provided and provided <= ABSOLUTE
                   else "")
        raise ToolRefusal(
            f"{partial}move_base takes either all of x_m, y_m and yaw_deg (world frame) or any of "
            "forward_m, left_m and turn_deg (relative to the current base pose), never both",
            received=sorted(str(key) for key in provided),
            constraints={"absolute_world": sorted(ABSOLUTE), "relative_to_base": sorted(RELATIVE)})
    # The base servo refuses the same bound; refusing here spends no robot decision.
    if math.hypot(target_x - x, target_y - y) > MAX_TRAVEL_M:
        raise ToolRefusal("the base target is too far from the current base pose", field=bound_field,
                          constraints={"max_travel_m": MAX_TRAVEL_M})
    target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
    identity = obs.identity.json()
    action = {"schema": "ora-motion-action.v3", "action": "move_base",
              "action_id": f"action-{config['execution']['generation']}",
              "observation_id": identity["observation_id"],
              "bound_observation_sha256": identity["observation_sha256"],
              "arguments": {"target": {"x_m": target_x, "y_m": target_y, "yaw_rad": target_yaw}}}
    return propose(api, action, "move_base")
