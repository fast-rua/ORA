"""The atomic forms: one pose or one jaw directive per call, for the one gripper, on ``ora-action.v3``.

Only an atomic surface's tool context has the ``atomic`` section (``toolkit_observe.atomic_configuration``), so every other
context reads as it always has. Synthetic Native and ``ora local`` take their pose as ``xyz_m``/``delta_mm`` with axes, or,
under ``pose_frame_contract_v2``, as the frame-explicit request of ``lib._pose_math``; the arguments of ``move_pose`` may
instead be the ``preview_id`` of a ``preview_move`` on this observation.
"""

import hashlib
import json
from collections.abc import Mapping

import numpy as np

from ...lib import _pose_math, observation
from ..errors import ToolRefusal
from ..geometry_query._arguments import _finite_vector, _refuse

POSE_FRAME_CONTRACT = "pose_frame_contract_v2"
POSE_FIELDS = {"xyz_m", "delta_mm", "delta_frame", "approach_world", "jaw_world"}
ACTION_SCHEMA = "ora-action.v3"


def declared(config):
    return "atomic" in config


def grip_state(manifest):
    """The one gripper's measured pose, ``[x, y, z, qx, qy, qz, qw]``, and normalized opening."""
    end_effectors = manifest.get("proprioception", {}).get("end_effectors", {})
    state = end_effectors.get("right") or end_effectors.get("left")
    if not isinstance(state, Mapping):
        raise ToolRefusal("current grip-site state is unavailable", code="pose_state_unavailable")
    pose = state.get("pose_xyz_xyzw")
    if not isinstance(pose, list) or len(pose) != 7:
        raise ToolRefusal("current grip-site pose is unavailable", code="pose_state_unavailable")
    return [float(item) for item in pose], float(state.get("gripper", 0))


def arm(manifest, config):
    """The arm a call moves: the one the environment bound, or the only one the observation publishes."""
    end_effectors = manifest.get("proprioception", {}).get("end_effectors", {})
    available = [str(name) for name, state in end_effectors.items()
                 if isinstance(state, Mapping)] if isinstance(end_effectors, Mapping) else []
    controlled = config["atomic"]["controlled_end_effector"]
    if controlled is not None:
        if controlled not in available:
            raise ToolRefusal("environment-bound end effector is absent from the current observation",
                              code="embodiment_binding", retryable=False,
                              constraints={"available_effectors": sorted(available)})
        return controlled
    if len(available) != 1:
        raise ToolRefusal("atomic single-effector surface requires exactly one public end effector",
                          code="embodiment_binding", retryable=False,
                          constraints={"available_effectors": sorted(available)})
    return available[0]


def _axes_orientation(orientation, approach, jaw):
    """The quaternion that puts local +Z on ``approach`` and +X nearest ``jaw``; an axis left out keeps the measured one."""
    rotation = np.asarray(_pose_math.quaternion_matrix(orientation), dtype=float)
    z_axis = rotation[:, 2] if approach is None else np.asarray(_finite_vector(approach, 3, "approach_world"))
    x_axis = rotation[:, 0] if jaw is None else np.asarray(_finite_vector(jaw, 3, "jaw_world"))
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
    if np.linalg.norm(x_axis) < 1e-6:
        raise ToolRefusal("jaw_world must not be parallel to approach_world")
    x_axis = x_axis / np.linalg.norm(x_axis)
    return _pose_math.matrix_quaternion(np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis]).tolist())


def _pose_arguments(arguments, current, grip_arm):
    """The target of ``xyz_m`` or ``delta_mm`` and the approach and jaw axes, in world metres."""
    if set(arguments) - POSE_FIELDS:
        raise ToolRefusal("pose request contains unknown fields")
    xyz, delta = arguments.get("xyz_m"), arguments.get("delta_mm")
    present = [name for name in ("xyz_m", "delta_mm") if arguments.get(name) is not None]
    if len(present) > 1:
        raise _refuse(arguments, "provide xyz_m or delta_mm, not both",
                      {"at_most_one_of": ["xyz_m", "delta_mm"], "received": present})
    position = list(current[:3])
    if xyz is not None:
        position = _finite_vector(xyz, 3, "xyz_m")
    elif delta is not None:
        displacement = np.asarray(_finite_vector(delta, 3, "delta_mm"), dtype=float) / 1000.0
        frame = arguments.get("delta_frame", "world")
        if frame not in {"world", "grip_site"}:
            raise ToolRefusal("delta_frame must be world or grip_site")
        if frame == "grip_site":
            displacement = np.asarray(_pose_math.quaternion_matrix(current[3:]), dtype=float) @ displacement
        position = (np.asarray(current[:3], dtype=float) + displacement).tolist()
    orientation = list(current[3:])
    approach, jaw = arguments.get("approach_world"), arguments.get("jaw_world")
    if approach is not None or jaw is not None:
        orientation = _axes_orientation(orientation, approach, jaw)
    if xyz is None and delta is None and approach is None and jaw is None:
        raise ToolRefusal("pose request requires a position, delta, or orientation")
    return {"arm": grip_arm, "position_m": position, "orientation_xyzw": orientation}


def _frame_contract_pose(arguments, current, grip_arm):
    """``pose_frame_contract_v2``: a frame-explicit request, compiled once, with the record the model gets back."""
    try:
        compiled = _pose_math.compile_pose_request(arguments, current_pose_xyz_xyzw=current,
                                                   controlled_point="robosuite_grip_site")
    except _pose_math.PoseCompilationError as error:
        raise ToolRefusal(str(error), code="invalid_pose_contract_v2") from error
    pose = compiled["target_pose_xyz_xyzw"]
    return {"arm": grip_arm, "position_m": list(pose[:3]), "orientation_xyzw": list(pose[3:])}, compiled


def compile_target(arguments, *, current, grip_arm, config):
    """The action arguments of a pose request and, under the frame contract, its compiled pose; inside the workspace."""
    from .tool import validate_workspace

    if POSE_FRAME_CONTRACT in config["motion"]["capabilities"]:
        target, compiled = _frame_contract_pose(arguments, current, grip_arm)
    else:
        target, compiled = _pose_arguments(arguments, current, grip_arm), None
    validate_workspace(target["position_m"], current[:3], config["motion"]["workspace_bounds_m"], "xyz_m")
    return target, compiled


def preview_id(observation_id, target):
    """The name of one candidate on one observation; the same candidate has the same name."""
    return "preview-" + hashlib.sha256(json.dumps({"observation": observation_id, "action": "move_pose",
                                                   "arguments": target}, sort_keys=True).encode()).hexdigest()[:16]


def _previewed(api, arguments, config, observation_id):
    """The target and compiled pose ``preview_move`` froze under this ``preview_id``."""
    preview = api.state.read("tool.previews").get(arguments["preview_id"])
    if (not isinstance(preview, Mapping) or preview.get("observation_id") != observation_id
            or preview.get("action") != "move_pose" or not isinstance(preview.get("arguments"), Mapping)):
        raise ToolRefusal("preview_id is unknown or stale", code="stale_preview")
    compiled = dict(preview["compiled_pose"]) if isinstance(preview.get("compiled_pose"), Mapping) else None
    if POSE_FRAME_CONTRACT in config["motion"]["capabilities"] and compiled is None:
        raise ToolRefusal("preview lacks its frozen compiled pose", code="stale_preview")
    return dict(preview["arguments"]), compiled


def refuse_terminal(name, config):
    """The trusted terminal boundary, before anything else of a mutation."""
    refusal = config["motion"]["mutation_refusals"].get(name)
    if refusal is not None:
        raise ToolRefusal.from_public(refusal)


def _propose(api, config, obs, *, action, arguments, name, compiled=None):
    from .tool import propose

    identity = obs.identity.json()
    return propose(api, {"schema": ACTION_SCHEMA, "observation_id": identity["observation_id"],
                         "action_id": f"action-{config['execution']['generation']}", "action": action,
                         "arguments": dict(arguments), "bound_observation_sha256": identity["observation_sha256"]},
                   name, **({} if compiled is None else {"compiled_pose": dict(compiled)}))


def move_pose(api, arguments, config):
    """Propose one pose, from direct inputs or a ``preview_id`` alone; the jaws hold their directive."""
    from .tool import validate_workspace

    refuse_terminal("move_pose", config)
    obs = observation.observe(api)
    manifest = obs.manifest.json()
    current, _ = grip_state(manifest)
    if set(arguments) == {"preview_id"}:
        target, compiled = _previewed(api, arguments, config, obs.identity.json()["observation_id"])
    else:
        target, compiled = compile_target(arguments, current=current, grip_arm=arm(manifest, config), config=config)
    validate_workspace(target["position_m"], current[:3], config["motion"]["workspace_bounds_m"], "xyz_m")
    return _propose(api, config, obs, action="move_pose", arguments=target, name="move_pose", compiled=compiled)


def set_jaws(api, arguments, *, name, opening):
    """Propose one jaw command while the pose target holds; ``opening`` is normalized, 0 closed and 1 open."""
    config = api.artifacts.tool_context()
    refuse_terminal(name, config)
    if arguments:
        raise ToolRefusal("gripper action accepts no arguments")
    obs = observation.observe(api)
    manifest = obs.manifest.json()
    grip_state(manifest)
    return _propose(api, config, obs, action="set_gripper", name=name,
                    arguments={"arm": arm(manifest, config), "gripper": opening})
