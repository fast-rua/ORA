"""Freeze public motion configuration and validate proposals at the trusted control edge."""

from collections.abc import Mapping
from dataclasses import asdict

from ora.contracts.action import validate_runtime_action
from ora.agent.toolkit.interface.types import (
    AccessDenied,
    PublicData,
    PublicToolFailure,
)
from ora.agent.error_classes import ToolError
from ora.agent.toolkit.host.motion_rules import clearance_runtime, refusals
from ora.agent.toolkit.host.reach.alignment import frame_local_axes
from ora.agent.error_classes import PublicFactsError
from ora.agent.toolkit.host.reach.jaw_center import JawCenterError, jaw_center_offsets, load_descriptor
from ora.agent.composition.embodiment.jaw_models import SUPPORTED_CONVENTIONS
from ora.agent.toolkit.host.motion_rules.refusals import ENHANCED_MOVE_POSE_EXAMPLES
from .state import ATOMIC_CAPABILITY

#: The one action each proposing tool may hand over (``submit_adapter``).
PROPOSED_ACTIONS = {"move_pose": "move_pose", "move_to": "move_pose", "rotate_wrist": "move_pose",
                    "rotate_pitch": "move_pose", "set_gripper": "set_gripper", "wait_ticks": "wait",
                    "recover_pose": "recover_motion", "move_base": "move_base", "close_gripper": "set_gripper",
                    "open_gripper": "set_gripper"}
#: The capability of a surface with a mobile base, the only one whose tool context names ``move_base``.
MOBILE_BASE = "mobile_base_navigation_v1"


def joint_trajectory_surface(surface) -> bool:
    """Whether this surface's motion tools plan each move on the policy side and run it as one joint trajectory:
    synchronous execution (the toolkit surfaces that froze their help for the released primitives, v1 to v3, were
    retired by 2026-11-01). It names no backend: a surface of any backend that runs them says so with the capability.
    The one judgement behind what a resolved contract declares (``contract.with_reads``) and what a running host
    serves (``plans_moves``)."""
    return bool(surface.supports("toolkit_synchronous_execution_v1"))


def plans_moves(host) -> bool:
    """Whether this host's motion tools plan on the policy side: the surface does, and the backend's read-only IK
    bridge is there to answer them."""
    return host.align_plan_callback is not None and joint_trajectory_surface(host.surface)


def profile_entry(record, *, root) -> dict | None:
    """The ``motion.profile`` a tool context carries for a plan's ``runtime.motion_profile`` (``{path, sha256}``).

    ``None`` for a plan that binds none. The file must be the one the plan hashed and listed in ``root``'s
    ``policy.lock``; raises ``ValueError`` otherwise (``tools/move_pose/_profile.context_entry``)."""
    if record is None:
        return None
    import json

    from ora.agent.composition.locked_assets import locked_asset
    from ora.agent.toolkit.tools.move_pose._profile import context_entry

    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError("runtime.motion_profile must be a {path, sha256} record")
    asset = locked_asset(root, record["path"], label="runtime.motion_profile")
    if asset.sha256 != record["sha256"]:
        raise ValueError(f"runtime.motion_profile {record['path']} changed after the plan hashed it")
    return context_entry(json.loads(asset.data.decode("utf-8")), path=asset.relative, sha256=asset.sha256)


def _with_profile(value, host):
    """``value`` with the host's motion profile under ``profile``; unchanged on a host without one."""
    profile = getattr(host, "motion_profile", None)
    return value if profile is None else {**value, "profile": dict(profile)}


def planned_motion(host) -> dict | None:
    """What a read-only query needs to plan a move as this host's motion tools do; ``None`` where they do not."""
    if not plans_moves(host):
        return None
    surface = host.surface
    return _with_profile({"capabilities": sorted(surface.capabilities),
                          "global_planner": bool(getattr(host, "motion_planner_available", False)),
                          "local_axes": frame_local_axes(surface.view_config["controlled_frame_convention"]),
                          "workspace_bounds_m": dict(host.workspace_bounds_m)}, host)


def _preview_hands(view_config):
    """Each arm's jaw centre, opening and approach axes and widest aperture, from the public descriptor and the surface's
    frame convention; nothing where it has none."""
    try:
        descriptor, offsets = load_descriptor(view_config), jaw_center_offsets(view_config)
    except (JawCenterError, PublicFactsError, ValueError, OSError, KeyError, TypeError):
        return {}
    try:
        approach = [float(value) for value in frame_local_axes(view_config["controlled_frame_convention"])["approach"]]
    except (KeyError, TypeError, ValueError):
        approach = None
    points = {point["id"]: point for point in descriptor["control"]["controlled_points"]}
    hands = {}
    for gripper in descriptor["geometry"]["grippers"]:
        contact = gripper.get("contact_geometry") or {}
        point = points.get(contact.get("controlled_point")) or {}
        arm = str(point.get("manipulator_id", "")).removesuffix("_arm")
        fingers = contact.get("fingers") or ()
        if arm in offsets and fingers:
            hands[arm] = {"contact_center_local": offsets[arm],
                          "opening_axis_local": [float(value) for value in fingers[0]["opening_direction_local"]],
                          "approach_axis_local": approach,
                          "maximum_aperture_m": (gripper.get("maximum_aperture_m") or (gripper.get(
                              "reference_parameters") or {}).get("nominal_maximum_aperture_m"))}
    return hands


def preview_geometry(host) -> tuple[dict, dict | None, dict]:
    """What a dry run's preview reads of the robot on any host: each hand's jaw geometry, the jaws for clearance and
    the held extents; what a surface does not bind is simply missing, and the preview says so."""
    surface = host.surface
    hands = _preview_hands(surface.view_config or {})
    try:
        geometry = clearance_runtime._geometry(host)
    except (ValueError, OSError, KeyError, TypeError):
        return hands, None, {}
    jaws = {name: asdict(value) for name, value in geometry.items()}
    held = {}
    for name, value in geometry.items():
        extent = clearance_runtime._held_extent(host, name, value)
        if extent is not None:
            held[name] = {"target_ref": extent.target_ref, "recentered": extent.recentered,
                          "local_corners": None if extent.local_corners is None else extent.local_corners.tolist()}
    return hands, jaws, held


def environment_plans(surface) -> bool:
    """Whether the toolkit's motion tools propose each move and the environment plans it as it executes it."""
    return surface.supports("toolkit_only_v1") and not joint_trajectory_surface(surface)


def _jaw_offsets(view_config):
    """Each arm's jaw centre in its controlled frame; a frame with no jaw-centre rule (YAM's) has none."""
    if view_config.get("controlled_frame_convention") not in SUPPORTED_CONVENTIONS:
        return {}
    return jaw_center_offsets(view_config)


def configuration(host):
    """All quantities are published robot geometry or policy-owned held-target state.

    ``environment_plans`` is there only where the environment plans each move when it executes it, so a context without
    it, every RoboDojo one and every sealed one, is a tool that runs its own plan (``tools/move_pose/_context.py``)."""
    surface = host.surface
    view = surface.view_config or {}
    convention = view.get("controlled_frame_convention")
    value = {
        "reach_available": host.align_plan_callback is not None,
        # One follow_joint_trajectory per move and rate-limited jaws: RoboDojo's
        # SDK executor with synchronous execution (move_pose and its speed tools).
        # The manifest surface v2 carries frozen tool help written for the
        # released primitives, so it keeps them.
        "joint_trajectory": plans_moves(host),
        # Free routes through the shared motion planner, when the run's backend has one (runtime.motion_planner).
        "global_planner": bool(getattr(host, "motion_planner_available", False)),
        "capabilities": sorted(surface.capabilities),
        "view_config": dict(view),
        "local_axes": {} if convention is None else frame_local_axes(convention),
        "workspace_bounds_m": dict(host.workspace_bounds_m),
        "examples": refusals.move_pose_examples(surface, ENHANCED_MOVE_POSE_EXAMPLES),
        "jaw_offsets": _jaw_offsets(view),
        "preview_hands": _preview_hands(view),
        "jaws": {},
        "held": {},
        "mutation_refusals": {},
    }
    if environment_plans(surface):
        value["environment_plans"] = True
    if getattr(host, "motion_profile", None) is not None:
        # The plan's motion profile (fast and the jaw forms, tools/move_pose/_profile.py); a context without
        # it, every one sealed before profile v2, runs the toolkit's built-in values.
        value["profile"] = dict(host.motion_profile)
    atomic = ("close_gripper", "open_gripper") if surface.supports(ATOMIC_CAPABILITY) else ()
    for name in ("move_pose", "move_to", "set_gripper", "recover_pose", "return_home", "rotate_wrist", "rotate_pitch",
                 "wait_ticks", *atomic, *(("move_base",) if surface.supports(MOBILE_BASE) else ())):
        try:
            host._ensure_mutation_available(name)
        except ToolError as error:
            value["mutation_refusals"][name] = error.as_dict()
    try:
        geometry = clearance_runtime._geometry(host)
        value["jaws"] = {name: asdict(jaws) for name, jaws in geometry.items()}
        for name, jaws in geometry.items():
            if surface.supports("toolkit_synchronous_execution_v1"):
                continue
            extent = clearance_runtime._held_extent(host, name, jaws)
            if extent is not None:
                value["held"][name] = {
                    "target_ref": extent.target_ref,
                    "local_corners": (
                        None
                        if extent.local_corners is None
                        else extent.local_corners.tolist()
                    ),
                    "recentered": extent.recentered,
                }
    except (ValueError, OSError, KeyError, TypeError):
        value["jaws"] = None
    # Immutable wire projection uses JSON lists for descriptor tuples.
    import json

    return json.loads(json.dumps(value))


def admitted(action):
    """``action`` as a proposal is admitted, which is what a dry run asks the environment about."""
    try:
        return validate_runtime_action(action)
    except ValueError as error:
        raise ToolError(str(error)) from error


def submit_adapter(host, tool, *, target_ref=None):
    """Accept only this call's validated motion; never run the simulator here.

    ``target_ref`` is what the call named as it closed a jaw: the host keeps what each arm holds for move clearance, as
    it does for a ``grasp_execute``."""

    def submit(parameters, inputs, store):
        try:
            host._ensure_mutation_available(tool)
            control, _, observation, _, _ = host._current_grip_state()
            action = validate_runtime_action(parameters["action"])
            if (
                inputs
                or action["action"] != PROPOSED_ACTIONS[tool]
                or action["observation_id"] != control["observation_id"]
                or action["bound_observation_sha256"] != control["observation_sha256"]
                or action["action_id"] != f"action-{control['generation']}"
            ):
                raise AccessDenied("motion action boundary mismatch")
            host._persist_action(
                host.control_root / "submitted-action.json", dict(control), action
            )
            host._submitted_mutations.append(action["action"])
            if action["action"] == "set_gripper" and "targets" in action["arguments"]:  # an atomic jaw holds no target
                clearance_runtime.note_gripper(host, action["arguments"]["targets"], target_ref=target_ref,
                                               observation=observation)
            return PublicData.from_json(
                {"accepted": True, "action_id": action["action_id"]}
            )
        except ToolError as error:
            raise PublicToolFailure(error.as_dict()) from error
        except ValueError as error:
            if isinstance(error, AccessDenied):
                raise
            raise PublicToolFailure(ToolError(str(error)).as_dict()) from error

    return submit


def reach_adapter(host):
    """Keep only the reference's public failure class; original messages are private."""
    from ora.agent.toolkit.interface.projections import reach
    from ora.agent.toolkit.interface.types import ReferenceFailure
    from ora.contracts.refusal import EnvironmentRefusal

    def query(parameters, inputs, store):
        try:
            host._current()
            return reach(host.align_plan_callback(parameters["request"]))
        except ToolError as error:
            raise PublicToolFailure(error.as_dict()) from error
        except EnvironmentRefusal as error:  # a declared refusal keeps its code, as the handlers' reach check does
            raise PublicToolFailure(ToolError(
                str(error), code=error.code, retryable=error.retryable, field=error.field, received=error.received,
                constraints={"allowed": error.allowed}).as_dict()) from error
        except Exception as error:
            raise ReferenceFailure(type(error).__name__) from error

    return query


def kinematics_adapter(host):
    """The read-only kinematics query over the align bridge; argument refusals stay public."""
    from ora.agent.toolkit.interface.projections import kinematics
    from ora.contracts.refusal import EnvironmentRefusal

    def query(parameters, inputs, store):
        host._current()
        if host.align_plan_callback is None:
            raise ToolError("the environment serves no read-only kinematics bridge", code="tool_unavailable",
                            retryable=False)
        try:
            return kinematics(host.align_plan_callback(parameters["request"]))
        except EnvironmentRefusal as error:
            raise ToolError(str(error), code=error.code, retryable=error.retryable,
                            field=error.field, received=error.received,
                            constraints={"allowed": error.allowed}) from error

    return query


def alignment_adapter(host):
    """Expose only public planner fields and declared argument refusals."""
    from ora.agent.toolkit.interface.projections import alignment
    from ora.contracts.refusal import EnvironmentRefusal

    def query(parameters, inputs, store):
        host._current()
        try:
            return alignment(host.align_plan_callback(parameters["request"]))
        except EnvironmentRefusal as error:
            raise ToolError(str(error), code=error.code, retryable=error.retryable,
                            field=error.field, received=error.received,
                            constraints={"allowed": error.allowed}) from error

    return query
