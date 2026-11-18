"""Compile measured-frame requests, attach public references, and propose one motion."""

import math
from collections.abc import Mapping

from ...interface import dry_run, stages
from ...interface.stages import stage
from ...lib import clearance, observation, pose
from ...lib._workspace import is_monotonic_workspace_recovery, workspace_state
from ...lib.values import JsonValue, PoseM, PublicResult
from ..errors import ToolRefusal
from .. import _dry_run, _held, _latency
from .._receipts import model_receipt
from . import _arguments, _atomic, _axes, _context, _reach, _refusals

#: Where the environment plans, what the surface publishes of a compiled target: its depth clearance, and that its
#: controlled point is the grip centre.
CLEARANCE_CAPABILITY = "observed_clearance_reference_v2"
GRIP_CENTRE_COMPILER = "contact_graspnet_robotwin_aloha_grip_center_compiler_v3"


def sparse_targets(arguments):
    if set(arguments) != {"targets"}:
        raise ToolRefusal(
            "sparse motion arguments contain unexpected fields",
            code="invalid_motion_arguments",
        )
    targets = arguments["targets"]
    if (
        not isinstance(targets, Mapping)
        or not targets
        or set(targets) - {"left", "right"}
    ):
        raise ToolRefusal(
            "targets must specify left, right, or both", code="invalid_motion_arguments"
        )
    return dict(targets)


def surface_targets(arguments, config, *, scalar_key=None):
    """The call's targets per arm: its sparse ``targets``, or on a single-arm surface the arguments themselves.

    A single-arm surface's arguments are the right arm's request, or the one value ``scalar_key`` names."""
    if _context.SINGLE_ARM not in (config.get("motion") or {}).get("capabilities", ()):
        return sparse_targets(arguments)
    if "targets" in arguments:
        raise ToolRefusal(
            "this single-arm surface takes flat arguments without a targets wrapper",
            code="invalid_motion_arguments",
        )
    if scalar_key is not None:
        if set(arguments) != {scalar_key}:
            raise ToolRefusal(
                f"arguments must contain exactly {scalar_key}",
                code="invalid_motion_arguments",
            )
        return {"right": arguments[scalar_key]}
    if not arguments:
        raise ToolRefusal(
            "flat single-arm arguments are required", code="invalid_motion_arguments"
        )
    return {"right": dict(arguments)}


def measured_states(manifest, arms=_context.BOTH_ARMS):
    effectors = manifest.get("proprioception", {}).get("end_effectors", {})
    if arms == _context.BOTH_ARMS and (
        not isinstance(effectors, Mapping) or set(effectors) != {"left", "right"}
    ):
        raise ToolRefusal(
            "dual-arm sparse surface requires left and right end effectors",
            code="embodiment_binding",
            retryable=False,
        )
    result = {}
    for arm in arms:
        measured = effectors.get(arm, {}).get("pose_xyz_xyzw")
        if not isinstance(measured, list) or len(measured) != 7:
            raise ToolRefusal(
                f"current {arm} end-effector pose is unavailable",
                code="embodiment_binding",
                retryable=False,
            )
        result[arm] = [float(item) for item in measured]
    return result


def validate_workspace(target, current, bounds, field):
    state = workspace_state(target, bounds)
    recovery = is_monotonic_workspace_recovery(current, target, bounds)
    if not state["inside"] and not recovery:
        raise ToolRefusal(
            "target pose is outside the configured action workspace",
            code="workspace_bounds",
            field=field,
            received=target,
            constraints={
                "workspace_bounds_m": bounds,
                "violations": state["violations"],
                "monotonic_recovery_allowed": True,
            },
        )


def public_compilation(compiled):
    return {
        key: compiled[key]
        for key in (
            "controlled_point",
            "current_pose_xyz_xyzw",
            "target_pose_xyz_xyzw",
            "translation_world_m",
            "translation_distance_m",
            "rotation_angle_rad",
            "position_mode",
            "orientation_mode",
            "position_delta_frame",
            "orientation_delta_frame",
        )
    }


def run(api, arguments):
    """Replay: recorded observation, kinematics answers and the one recorded trajectory primitive.

    One smooth joint trajectory at ``speed`` (fast by default; ``_smooth``), optionally slowing over the
    last 2 cm (``slow_approaching``); without a backend that executes joint trajectories, a fast move is the
    released move_pose primitive."""
    from . import _smooth

    config = api.artifacts.tool_context()
    if _atomic.declared(config):
        return _atomic.move_pose(api, arguments, config)
    try:
        if dry_run.requested() and not _context.environment_plans(config):  # a refused plan is the dry run's answer
            return _dry_run.answered("move_pose", lambda: _smooth.run_move(api, arguments, name="move_pose"),
                                     view=lambda: _dry_run.view(api, observation.observe(api), config))
        return _smooth.run_move(api, arguments, name="move_pose")
    except ToolRefusal as error:
        if (
            error.code == "invalid_motion_arguments"
            and "examples" not in error.constraints
        ):
            error.constraints["examples"] = config["motion"]["examples"]
        raise


def propose(api, action, name, **fields):
    """Hand ``action`` to the environment that plans it; the result says so, and its receipt comes with the next turn."""
    PublicResult.of(api.act.submit_action(action=action)).data
    return {
        "status": "submitted",
        "action": name,
        **fields,
        "_runtime": {"directive": "action_submitted", "action_id": action["action_id"]},
    }


def compile_and_submit(api, arguments, *, config, primitive, request_for_target=None, scalar_key=None):
    """Compile the request into one move_pose action (``compile_action``) and execute or submit it.

    Where joint trajectories run (``_smooth.available``), move_to, rotate_wrist and rotate_pitch run their
    compiled action as one trajectory at move_pose's fast speed (``_smooth.run_action``); a path refused
    before anything moves is a ToolRefusal with the planner's code; ``expect_contact`` turns the
    growing-error tracking rule off for that trajectory. Elsewhere, and for move_pose itself, the released
    primitive (which has no growing-error rule, so ``expect_contact`` changes nothing there)."""
    from . import _smooth

    if dry_run.requested():
        plan = lambda: _dry(api, arguments, config=config, primitive=primitive, request_for_target=request_for_target,
                            scalar_key=scalar_key)
        if _context.environment_plans(config):  # the environment answers: what is refused here stays a refusal
            return plan()
        return _dry_run.answered(primitive, plan, view=lambda: _dry_run.view(api, observation.observe(api), config))
    speed, contact, exact, arguments = _smooth.call_options(arguments, config)
    action, compiled = compile_action(api, arguments, config=config, primitive=primitive,
                                      request_for_target=request_for_target, exact_orientation=exact, speed=speed,
                                      scalar_key=scalar_key)
    if primitive != "move_pose" and _smooth.available(config):
        from ...lib.motion_plan import checks, kinematics
        from ..budgets import native_action_budget

        try:
            receipt, remaining, _ = _smooth.run_action(api, action, config, budget=native_action_budget(primitive),
                                                       expect_contact=contact,
                                                       jaw_symmetry=_smooth.jaw_labels(compiled))
        except checks.PlanRefusal as error:
            refusal = ToolRefusal(str(error), code=error.code, constraints=error.constraints)
            try:  # the picture needs the observation the plan read; without it the refusal stands alone
                obs = observation.observe(api)
                starts, goals = _smooth.request_poses(obs, action)
            except Exception:  # noqa: BLE001 - a refusal's picture is a courtesy, its answer stands
                raise refusal from error
            raise _dry_run.pictured(primitive, refusal, starts, goals,
                                    lambda: _dry_run.view(api, obs, config)) from error
        except kinematics.KinematicsUnavailable as error:
            raise ToolRefusal(f"the read-only kinematics query failed: {error}", code="planner_failed") from error
        return {"status": "executed", "action": primitive, "compiled_pose": compiled,
                "receipt": model_receipt(receipt), "remaining_budget": remaining}
    if config.get("synchronous_execution"):
        execution = PublicResult.of(api.act.execute_action(action, 20)).data.json()
        return {"status": "executed", "action": primitive, "compiled_pose": compiled, **execution,
                "receipt": model_receipt(execution["receipt"])}
    return propose(api, action, primitive, compiled_pose=compiled)


def _dry(api, arguments, *, config, primitive, request_for_target, scalar_key=None):
    """A dry run of move_to, rotate_wrist or rotate_pitch: the request compiled and planned as the real call does it,
    stopping where the real call would execute the trajectory (``dry_run``)."""
    from ...lib.motion_plan import checks, kinematics
    from ..budgets import native_action_budget
    from . import _smooth

    speed, contact, exact, arguments = _smooth.call_options(arguments, config)
    action, compiled = compile_action(api, arguments, config=config, primitive=primitive,
                                      request_for_target=request_for_target, exact_orientation=exact, speed=speed,
                                      scalar_key=scalar_key)
    if not _smooth.available(config):  # the environment plans this move: it answers for it (``Proposed``)
        PublicResult.of(api.act.submit_action(action=action)).data
    try:
        obs, planned, _ = _smooth.plan_action(api, action, config, budget=native_action_budget(primitive),
                                              expect_contact=contact)
    except checks.PlanRefusal as error:
        refusal = ToolRefusal(str(error), code=error.code, constraints=error.constraints)
        try:  # what was asked, for the dry run's picture of it
            starts, goals = _smooth.request_poses(observation.observe(api), action)
        except Exception:  # noqa: BLE001 - a refusal's picture is a courtesy, its answer stands
            raise refusal from error
        raise _dry_run.with_request(refusal, starts, goals) from error
    except kinematics.KinematicsUnavailable as error:
        raise ToolRefusal(f"the read-only kinematics query failed: {error}", code="planner_failed") from error
    return _dry_run.planned(primitive, planned, compiled, expect_contact=contact, view=_dry_run.view(api, obs, config))


def _holding(api, effector, arm):
    """The jaw holds (or may hold) something: its last command was not open, or the tools noted a held target."""
    if effector.get("gripper_command_target") != "open":
        return True
    try:
        return api.state.read(f"tool.held.{arm}") is not None
    except Exception:  # an old run without the key: the gripper command decides
        return False


def compile_action(api, arguments, *, config, primitive, request_for_target=None, exact_orientation=False, speed=None,
                   scalar_key=None):
    """The validated move_pose action for the request, and the public compilation of each target; nothing
    is executed (``_smooth`` plans and runs it as one joint trajectory).

    Where joint trajectories run (``_smooth.available``), each target's orientation follows the jaw-symmetry
    rule (``lib.motion_plan.orientation.nearest_jaw_twin``) on every interaction and orientation form: an
    empty jaw takes whichever of the request and its half-turn twin is nearer the measured wrist, a jaw that
    holds something takes it only for ``world_axes`` (a jaw line has no sign), and ``exact_orientation`` keeps the
    request; the target's
    ``jaw_symmetry`` names the choice. Elsewhere the released free-space jaw turn applies. ``speed`` is the
    proposed speed the action carries (``_smooth.proposed_speed``); a target's field names are the surface's own."""
    from . import _smooth
    from ...lib._motion_math import compile_enhanced_move_pose
    from ...lib.motion_plan import orientation as jaw_rule

    motion = config["motion"]
    twin_rule = _smooth.available(config)
    proposes = _context.environment_plans(config)
    refusal = motion["mutation_refusals"].get(primitive)
    if refusal is not None:
        raise ToolRefusal.from_public(refusal)
    if primitive == "move_pose":
        if "plan_ref" in arguments:
            raise ToolRefusal(
                "Submit an explicit current pose target",
                code="invalid_motion_arguments",
                field="plan_ref",
            )
        if arguments.get("payload_ref") is not None:
            raise ToolRefusal(
                "payload_ref is deferred from the motion-intent surface",
                code="unsupported_motion_extension",
                field="payload_ref",
            )
        arguments = {
            key: value for key, value in arguments.items() if key != "payload_ref"
        }
    context = _context.from_configuration(api, motion, reach_reference=_latency.on(config, "reach_reference"))
    targets = surface_targets(arguments, config, scalar_key=scalar_key)
    jaw_lines: set[str] = set()  # arms that asked by jaw line, which has no sign; resolved to a quaternion just below
    if primitive == "move_pose":
        named = _arguments.arm_targets(targets)
        jaw_lines = {arm for arm, raw in named.items() if isinstance(raw, Mapping)
                     and isinstance(raw.get("orientation"), Mapping) and raw["orientation"].get("mode") == _axes.MODE}
        targets = _axes.resolve_targets(context.surface, named)
    obs = observation.observe(api)
    manifest = obs.manifest.json()
    current = measured_states(manifest, _context.arms(config))
    action_targets, interactions, compiled_targets = {}, {}, {}
    for arm, raw in targets.items():
        if primitive != "move_pose":
            try:
                if request_for_target is not None:
                    request = request_for_target(raw)
                else:
                    if not isinstance(raw, Mapping) or set(raw) != {"position_m"}:
                        raise ValueError("each move_to target must contain only position_m")
                    request = {
                        "position": {"mode": "absolute", "position_m": raw["position_m"]},
                        "orientation": {"mode": "hold_current"},
                    }
                compiled = pose.compile_target(
                    JsonValue.of(request),
                    PoseM("world", tuple(current[arm][:3]), tuple(current[arm][3:])),
                ).json()
            except (TypeError, ValueError) as error:
                raise _refusals.primitive_refusal(
                    error, primitive=primitive, field=_refusals.primitive_field(context.surface, arm, primitive)
                ) from error
            interaction = "free_space" if primitive == "move_to" else "precision"
            field = (_refusals.target_field(context.surface, arm, "position_m") if primitive == "move_to"
                     else _refusals.primitive_field(context.surface, arm, primitive))
        else:
            if not isinstance(raw, Mapping):
                raise ToolRefusal(
                    f"{arm} move_pose target must be an object",
                    code="invalid_motion_arguments",
                )
            interaction = raw.get("interaction", "free_space")
            if interaction not in {"free_space", "precision", "contact_approach"}:
                raise ToolRefusal(
                    f"{arm} interaction must be free_space, precision, or contact_approach",
                    code="invalid_motion_arguments",
                    field="interaction",
                )
            request = {key: value for key, value in raw.items() if key != "interaction"}
            try:
                compiled = pose.compile_target(
                    JsonValue.of(request),
                    PoseM("world", tuple(current[arm][:3]), tuple(current[arm][3:])),
                ).json()
            except (TypeError, ValueError) as error:
                raise ToolRefusal(
                    str(error), code="invalid_motion_arguments",
                    field=_refusals.target_field(context.surface, arm),
                ) from error
            suffix = {"absolute": "position_m", "delta": "translation_m"}.get(
                compiled["position_mode"]
            )
            field = _refusals.target_field(context.surface, arm, suffix)
        target = compiled["target_pose_xyz_xyzw"]
        validate_workspace(
            target[:3], current[arm][:3], motion["workspace_bounds_m"], field
        )
        jaw_label = None
        if twin_rule:
            effector = ((manifest.get("proprioception") or {}).get("end_effectors") or {}).get(arm) or {}
            choice = jaw_rule.nearest_jaw_twin(
                current[arm][3:], target[3:], held=_holding(api, effector, arm), exact=bool(exact_orientation),
                signless=arm in jaw_lines,
                approach_local=context.surface.local_axes["approach"])
            jaw_label = choice["jaw_symmetry"]
            if jaw_label == "twin_nearer_current":
                compiled = {
                    **compile_enhanced_move_pose(
                        {"position_m": target[:3], "orientation_xyzw": choice["orientation_xyzw"]},
                        current_pose_xyz_xyzw=current[arm], controlled_point="end_effector"),
                    "position_mode": compiled["position_mode"],
                    "position_delta_frame": compiled["position_delta_frame"],
                }
                target = compiled["target_pose_xyz_xyzw"]
        if primitive == "move_pose":
            compiled, action_pose, reference = _reach.apply_reference(
                context,
                arm,
                compiled,
                current_pose=current[arm],
                interaction=interaction,
                observation=manifest,
                jaw_turn=not twin_rule,
            )
        else:
            action_pose = {"position_m": target[:3], "orientation_xyzw": target[3:]}
            reference = _reach.primitive_reference(
                context,
                arm,
                compiled,
                current_pose=current[arm],
                interaction=interaction,
            )
        target = compiled["target_pose_xyz_xyzw"]
        public = public_compilation(compiled)
        if primitive != "move_pose":
            public["semantic_primitive"] = primitive
        public.update(reference)
        if jaw_label is not None:
            public["jaw_symmetry"] = jaw_label
        public.update(_axes.echo(context.surface, action_pose))
        if not proposes or context.surface.supports(CLEARANCE_CAPABILITY):
            public["observed_clearance"] = clearance.estimate(
                api,
                obs,
                arm=arm,
                target=PoseM("world", tuple(target[:3]), tuple(target[3:])),
                jaws=JsonValue.of(motion["jaws"]),
                held=JsonValue.of(_held.extents(api, motion["jaws"])
                                  if config.get("synchronous_execution") else motion["held"]),
            ).json()
        if proposes and context.surface.supports(GRIP_CENTRE_COMPILER):
            public["controlled_point_semantics"] = "parallel_jaw_grip_center_tcp"
        if stages.active():  # what compiling found, for explain: it survives a call that then fails
            reach = public.get("reach_along_request") or {}
            nearest = public.get("nearest_reachable_orientation") or {}
            seen = public.get("observed_clearance") or {}
            angle = compiled.get("rotation_angle_rad")
            stage("compiled", arm=arm, interaction=str(interaction),
                  translation_distance_m=compiled.get("translation_distance_m"),
                  rotation_angle_deg=None if angle is None else math.degrees(angle), jaw_symmetry=jaw_label,
                  target_reachable=reach.get("target_reachable"), reachable_fraction=reach.get("farthest_reachable_fraction"),
                  reachable_translation_m=reach.get("farthest_reachable_translation_m"),
                  nearest_turn_deg=nearest.get("rotation_from_request_deg"),
                  nearest_tilt_deg=nearest.get("tilt_from_straight_down_deg"), clearance_status=seen.get("status"),
                  clearance_m=seen.get("clearance_m"), clearance_observed_fraction=seen.get("footprint_observed_fraction"))
        action_targets[arm], interactions[arm], compiled_targets[arm] = (
            action_pose,
            str(interaction),
            public,
        )
    observed = obs.identity.json()
    action_id = f"action-{config['execution']['generation']}"
    action = {
        "schema": "ora-motion-action.v3",
        "observation_id": observed["observation_id"],
        "action_id": action_id,
        "action": "move_pose",
        "arguments": {"targets": action_targets, "interactions": interactions,
                      **({"speed": speed} if speed is not None else {})},
        "bound_observation_sha256": observed["observation_sha256"],
    }
    compiled = {"targets": compiled_targets}
    if primitive != "move_pose":
        compiled.update(semantic_primitive=primitive, canonical_action="move_pose")
    return action, compiled
