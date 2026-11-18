"""move_pose's request executed as one smooth joint trajectory at the speed the call names.

``speed`` (fast, the default; normal; careful) chooses a ``lib.motion_plan.timing.Pace`` and a settle,
nothing else: the interaction label picks the route at every speed. ``speed_m_s`` instead names the
controlled point's top speed in m/s (``custom_pace``), with fast's joint caps, so the joints are the only
other limit. ``slow_approaching`` adds the careful pace over the last ``timing.SLOW_APPROACH_M`` of each
arm's path, inside the same trajectory.
The request is compiled exactly as move_pose compiles it; ``lib.motion_plan.move`` plans, times, fits the
budget, screens and runs it (with the shared planner's collision-free path for free routes where the backend
answers ``plan_path``, and ``expect_contact`` planned as a contact); this module only maps the call onto it.
Without synchronous execution on a
backend that executes joint trajectories (``motion.joint_trajectory`` in the tool context), a fast move
submits the released move_pose primitive and the other options are unavailable. Where the environment plans
the move (``_context.environment_plans``), the speed travels in the action (``proposed_speed``) and every pose
tool proposes it through ``tool.compile_and_submit``.
"""

import math

from ...interface import dry_run
from ...lib import observation
from ...lib.motion_plan import checks, kept, kinematics, move, timing
from .. import _dry_run
from ..budgets import MOVE_BUDGET, native_action_budget
from ..errors import ToolRefusal
from .._receipts import model_receipt
from . import _context, _profile

#: Full speed: no Cartesian cap; 80% of the 5 rad/s joint limit, so tracking keeps headroom. A context's motion
#: profile names its own fast (``_profile.fast``: profile v2's 4.5 rad/s and 60 rad/s^2); this is profile v1's,
#: what a context without one runs, and what ``custom_pace`` and the recovery retrace keep.
FULL = timing.Pace(linear_m_s=None, angular_deg_s=None, linear_accel_m_s2=None, angular_accel_deg_s2=None,
                   joint_speed_rad_s=4.0, joint_accel_rad_s2=40.0)
NORMAL = timing.Pace(linear_m_s=0.3, angular_deg_s=60.0, linear_accel_m_s2=1.0, angular_accel_deg_s2=200.0,
                     joint_speed_rad_s=2.5, joint_accel_rad_s2=15.0)
CAREFUL = timing.Pace(linear_m_s=0.1, angular_deg_s=20.0, linear_accel_m_s2=0.5, angular_accel_deg_s2=100.0,
                      joint_speed_rad_s=1.2, joint_accel_rad_s2=8.0)
#: Explicit settles: hold the last knot until every joint is this close and this slow.
FULL_SETTLE = {"native_actions": 5, "joint_tolerance_rad": 0.003, "joint_speed_rad_s": 0.05}
PACED_SETTLE = {"native_actions": 3, "joint_tolerance_rad": 0.002, "joint_speed_rad_s": 0.05}
#: speed -> (pace, settle) without a motion profile. These are defaults; a tool of its own may use any Pace
#: (lib.motion_plan.move). ``speeds`` is a call's: fast from its context's profile.
SPEEDS = {"fast": (FULL, FULL_SETTLE), "normal": (NORMAL, PACED_SETTLE), "careful": (CAREFUL, PACED_SETTLE)}
DEFAULT_SPEED = "fast"


def speeds(config):
    """Pure computation: speed -> (pace, settle) of this call: ``SPEEDS`` with fast from the context's motion profile.

    Example: speeds({})["fast"] == SPEEDS["fast"]."""
    return {**SPEEDS, DEFAULT_SPEED: _profile.fast(config)}
#: What a cut move's ``budget_limited.note`` adds, by the limit that cut it (``allowance``).
LIMITED_NOTES = {
    "tool_native_action_budget": " Call again with the same target for the rest, or use a faster speed.",
    "episode_native_actions_remaining": " The episode has no native actions left for the rest.",
    "script_native_actions_remaining": (" This script's allowance has no native actions left for the rest; call "
                                        "again with the same target from the next script."),
}
#: speed_m_s: the turn cap per m/s (normal's 60 deg/s at 0.3 m/s, careful's 20 at 0.1), never below careful's,
#: and the time both caps are reached in from rest.
TURN_DEG_S_PER_M_S = 200.0
MIN_TURN_DEG_S = CAREFUL.angular_deg_s
RAMP_S = 0.25
#: speed_m_s's name in motion.speed.speed.
CUSTOM_SPEED = "custom"
#: The surface capability that publishes a pose tool's speed options, and the options.
SPEED_CAPABILITY = "end_effector_speed_v1"
SPEED_FIELDS = ("speed", "speed_m_s", "slow_approaching")
IK_UNRECORDED_WARNING = move.IK_UNRECORDED_WARNING


def custom_pace(speed_m_s):
    """Pure computation: the Pace of a numeric ``speed_m_s``.

    That linear cap, a turn cap of ``TURN_DEG_S_PER_M_S`` per m/s (at least ``MIN_TURN_DEG_S``), both reached
    in ``RAMP_S`` from rest, and fast's joint caps. Example: custom_pace(0.05).linear_m_s == 0.05."""
    turn = max(MIN_TURN_DEG_S, TURN_DEG_S_PER_M_S * speed_m_s)
    return timing.Pace(linear_m_s=speed_m_s, angular_deg_s=turn, linear_accel_m_s2=speed_m_s / RAMP_S,
                       angular_accel_deg_s2=turn / RAMP_S, joint_speed_rad_s=FULL.joint_speed_rad_s,
                       joint_accel_rad_s2=FULL.joint_accel_rad_s2)


def available(config):
    """Whether this call can run one joint trajectory (synchronous execution and a backend that runs it)."""
    return bool(config.get("synchronous_execution") and config["motion"].get("joint_trajectory"))


def global_planner(config):
    """Whether this call's backend answers ``kinematics.plan_path`` (the shared motion planner)."""
    return bool(config["motion"].get("global_planner"))


def contact_option(arguments):
    """The validated ``expect_contact`` of a call (default false) and the arguments without it.

    True marks a deliberate push or press: the growing-error tracking rule is off for this move; the tracking
    bound and the progress guard still stop a real block."""
    contact = arguments.get("expect_contact", False)
    contact = False if contact is None else contact
    if not isinstance(contact, bool):
        raise ToolRefusal("expect_contact must be true or false", code="invalid_motion_arguments",
                          field="expect_contact", received=contact)
    return contact, {key: value for key, value in arguments.items() if key != "expect_contact"}


def jaw_option(arguments):
    """The validated ``exact_orientation`` of a call (default false) and the arguments without it.

    True uses the requested orientation exactly, never its half-turn jaw twin (``compile_action``)."""
    exact = arguments.get("exact_orientation", False)
    exact = False if exact is None else exact
    if not isinstance(exact, bool):
        raise ToolRefusal("exact_orientation must be true or false", code="invalid_motion_arguments",
                          field="exact_orientation", received=exact)
    return exact, {key: value for key, value in arguments.items() if key != "exact_orientation"}


def jaw_labels(compiled):
    """Each target's ``jaw_symmetry`` choice from a compilation (``compile_action``), or None without the rule."""
    labels = {arm: target["jaw_symmetry"] for arm, target in (compiled.get("targets") or {}).items()
              if isinstance(target.get("jaw_symmetry"), str)}
    return labels or None


def options(arguments):
    """The validated ``speed``, ``speed_m_s`` and ``slow_approaching`` of a call, and the arguments without them.

    ``speed`` is ``CUSTOM_SPEED`` when the call gives ``speed_m_s`` (a float), else ``speed_m_s`` is None."""
    speed = arguments.get("speed", DEFAULT_SPEED)
    speed = DEFAULT_SPEED if speed is None else speed
    speed_m_s = arguments.get("speed_m_s")
    if speed_m_s is not None:
        if arguments.get("speed") is not None:
            raise ToolRefusal("give speed or speed_m_s, not both", code="invalid_motion_arguments",
                              field="speed_m_s", received=speed_m_s, constraints={"speed": speed})
        if (isinstance(speed_m_s, bool) or not isinstance(speed_m_s, (int, float))
                or not math.isfinite(speed_m_s) or speed_m_s <= 0):
            raise ToolRefusal("speed_m_s must be a number above 0 (m/s)", code="invalid_motion_arguments",
                              field="speed_m_s", received=speed_m_s, constraints={"exclusive_minimum": 0})
        speed, speed_m_s = CUSTOM_SPEED, float(speed_m_s)
    elif speed not in SPEEDS:
        raise ToolRefusal("speed must be fast, normal or careful", code="invalid_motion_arguments", field="speed",
                          received=speed, constraints={"allowed": list(SPEEDS), "default": DEFAULT_SPEED})
    slow = arguments.get("slow_approaching", False)
    slow = False if slow is None else slow
    if not isinstance(slow, bool):
        raise ToolRefusal("slow_approaching must be true or false", code="invalid_motion_arguments",
                          field="slow_approaching", received=slow)
    if slow and speed == "careful":
        raise ToolRefusal("slow_approaching has no effect with speed=careful", code="invalid_motion_arguments",
                          field="slow_approaching", constraints={"speed": speed, "allowed_speeds": ["fast", "normal"]})
    if slow and speed_m_s is not None and speed_m_s <= CAREFUL.linear_m_s:
        raise ToolRefusal(f"slow_approaching has no effect with speed_m_s at or below {CAREFUL.linear_m_s} m/s",
                          code="invalid_motion_arguments", field="slow_approaching",
                          constraints={"speed_m_s": speed_m_s, "speed_m_s_above": CAREFUL.linear_m_s})
    rest = {key: value for key, value in arguments.items() if key not in {"speed", "speed_m_s", "slow_approaching"}}
    return speed, speed_m_s, slow, rest


def proposed_speed(arguments, config):
    """The speed a proposed action carries (``None`` when the call names none) and the arguments without it.

    The wire form of ``ora.contracts.motion_speed``, under ``options``' rules. A surface that does not publish speed
    leaves the fields where they are, and they are refused like any other unknown field."""
    if SPEED_CAPABILITY not in config["motion"].get("capabilities", ()):
        return None, dict(arguments)
    if all(arguments.get(key) is None for key in SPEED_FIELDS):
        return None, {key: value for key, value in arguments.items() if key not in SPEED_FIELDS}
    speed, speed_m_s, slow, rest = options(arguments)
    return ({"speed": speed, "slow_approaching": slow, **({"speed_m_s": speed_m_s} if speed_m_s is not None else {})},
            rest)


def call_options(arguments, config):
    """``(speed, expect_contact, exact_orientation, the arguments without them)`` of a call that compiles an action.

    The two planner options have no meaning where the environment plans, so they stay unknown fields there."""
    if _context.environment_plans(config):
        speed, rest = proposed_speed(arguments, config)
        return speed, False, False, rest
    contact, arguments = contact_option(arguments)
    exact, arguments = jaw_option(arguments)
    return None, contact, exact, arguments


#: The limits a call's plan fits besides its own budget: what the episode has left and, inside a code-mode
#: script, what the script's parent allowance has left (the Supervisor admits no more than either).
ALLOWANCE_LIMITS = ("episode_native_actions_remaining", "script_native_actions_remaining")


def allowance(budget, config):
    """What one trajectory of a call may spend: ``budget`` (its own, at most ``MOVE_BUDGET``), at most each of
    ``ALLOWANCE_LIMITS`` the tool context carries; and which limit that is (``tool_native_action_budget`` when its
    own budget is the least)."""
    value, limited_by = min(budget, MOVE_BUDGET), "tool_native_action_budget"
    for key in ALLOWANCE_LIMITS:
        limit = config.get(key)
        if type(limit) is int and limit >= 0 and limit < value:
            value, limited_by = limit, key
    return value, limited_by


def planned_move(api, obs, **inputs):
    """``move.plan`` of ``inputs``, and whether it is the plan a dry run kept for exactly them on ``obs``.

    A dry run plans and keeps; a real call runs what was kept, or plans as ever (``lib.motion_plan.kept``)."""
    identity = obs.identity.json()
    key = kept.key(identity, **inputs)
    record = None if dry_run.requested() else kept.recall(api, identity, key)
    if record is not None:
        return kept.move_plan(record), True
    planned = move.plan(api, **inputs)
    if dry_run.requested():
        kept.keep(api, identity, key, kept.move_record(planned))
    return planned, False


def request_poses(obs, action):
    """Each targeted arm's measured start pose and its compiled goal, world metres and xyzw."""
    effectors = obs.manifest.json()["proprioception"]["end_effectors"]
    targets = action["arguments"]["targets"]
    return ({arm: [float(v) for v in effectors[arm]["pose_xyz_xyzw"]] for arm in targets},
            {arm: [*target["position_m"], *target["orientation_xyzw"]] for arm, target in targets.items()})


def plan_action(api, action, config, *, budget, expect_contact=False):
    """The plan of one compiled move_pose action at the fast speed (what ``run_action`` runs), nothing executed.

    Returns ``(observation, planned, reused)``; raises ``checks.PlanRefusal`` / ``kinematics.KinematicsUnavailable``."""
    obs = observation.observe(api)
    effectors = obs.manifest.json()["proprioception"]["end_effectors"]
    targets = action["arguments"]["targets"]
    pace, settle = speeds(config)[DEFAULT_SPEED]
    planned, reused = planned_move(
        api, obs, starts={arm: [float(v) for v in effectors[arm]["pose_xyz_xyzw"]] for arm in targets},
        goals={arm: [*target["position_m"], *target["orientation_xyzw"]] for arm, target in targets.items()},
        start_joints={arm: [float(v) for v in effectors[arm]["arm_joint_qpos"]] for arm in targets},
        pace=pace, settle=settle, native_action_budget=allowance(budget, config)[0],
        route_kind="interaction", interactions=action["arguments"]["interactions"],
        capabilities=config["motion"].get("capabilities", ()), global_planner=global_planner(config),
        expect_contact=expect_contact)
    return obs, planned, reused


def run_action(api, action, config, *, budget, expect_contact=False, jaw_symmetry=None):
    """Replay: recorded observation, kinematics answers and the one recorded trajectory primitive.

    One compiled move_pose action (its targets and interactions) run as one trajectory at the fast speed,
    the route picked by its interaction label, as move_pose runs it; for the motion tools that compile a
    move_pose action of their own (move_to, rotate_wrist, rotate_pitch, grasp_execute's arm stages).
    ``budget`` is the native actions it may spend (capped by what the episode and, inside a code-mode script, the
    script has left: ``allowance``); ``expect_contact``
    turns the growing-error tracking rule off (``contact_option``). Returns
    ``(receipt, remaining_budget, planned)``; raises ``checks.PlanRefusal`` / ``kinematics.KinematicsUnavailable``
    before anything moves."""
    obs, planned, reused = plan_action(api, action, config, budget=budget, expect_contact=expect_contact)
    result = move.run(api, planned, action_id=action["action_id"], observation=obs.identity.json(),
                      labels={"move": DEFAULT_SPEED}, expect_contact=expect_contact, jaw_symmetry=jaw_symmetry)
    return (_previewed(result["receipt"], reused), result["remaining_budget"], planned)


def _previewed(receipt, reused):
    """``receipt`` saying the action ran the trajectory a dry run planned, where it did."""
    return {**receipt, "ran_previewed_plan": True} if reused else receipt


def run_move(api, arguments, *, name="move_pose"):
    """Replay: recorded observation, kinematics answers and the one recorded trajectory primitive."""
    from .tool import compile_action, compile_and_submit

    config = api.artifacts.tool_context()
    if _context.environment_plans(config):
        return {**compile_and_submit(api, arguments, config=config, primitive="move_pose"), "action": name}
    contact, arguments = contact_option(arguments)
    exact, arguments = jaw_option(arguments)
    speed, speed_m_s, slow, arguments = options(arguments)
    pace, settle = (custom_pace(speed_m_s), PACED_SETTLE) if speed_m_s is not None else speeds(config)[speed]
    label = speed if speed_m_s is None else f"{speed_m_s:g} m/s"
    if not available(config):
        if speed == DEFAULT_SPEED and not slow:
            return {**compile_and_submit(api, arguments, config=config, primitive="move_pose"), "action": name}
        raise ToolRefusal(f"speed={label}{' with slow_approaching' if slow else ''} needs synchronous execution "
                          "on a backend that executes joint trajectories", code="tool_unavailable", retryable=False)
    action, compiled = compile_action(api, arguments, config=config, primitive="move_pose", exact_orientation=exact)
    obs = observation.observe(api)
    effectors = obs.manifest.json()["proprioception"]["end_effectors"]
    targets = action["arguments"]["targets"]
    budget = native_action_budget(name)
    # Fit the plan to what this call can really spend, so neither the episode's last native actions nor a
    # script's spent allowance cuts a move mid-cruise: the covered part ends at rest and the shortfall is reported.
    fitted, limited_by = allowance(budget, config)
    starts = {arm: [float(v) for v in effectors[arm]["pose_xyz_xyzw"]] for arm in targets}
    goals = {arm: [*target["position_m"], *target["orientation_xyzw"]] for arm, target in targets.items()}
    try:
        planned, reused = planned_move(
            api, obs, starts=starts, goals=goals,
            start_joints={arm: [float(v) for v in effectors[arm]["arm_joint_qpos"]] for arm in targets},
            pace=pace, settle=PACED_SETTLE if slow else settle, native_action_budget=fitted,
            route_kind="interaction", interactions=action["arguments"]["interactions"],
            capabilities=config["motion"].get("capabilities", ()), slow_pace=CAREFUL if slow else None,
            global_planner=global_planner(config), expect_contact=contact)
    except checks.PlanRefusal as error:
        raise _dry_run.pictured(name, ToolRefusal(str(error), code=error.code, constraints=error.constraints),
                                starts, goals, lambda: _dry_run.view(api, obs, config)) from error
    except kinematics.KinematicsUnavailable as error:
        raise ToolRefusal(f"the read-only kinematics query failed: {error}", code="planner_failed") from error
    chosen = {"speed": speed, **({"speed_m_s": speed_m_s} if speed_m_s is not None else {}),
              "slow_approaching": slow, **move.caps(pace), **move.peaks(planned),
              **({"slow_approach_m": timing.SLOW_APPROACH_M, "slow": move.caps(CAREFUL)} if slow else {})}
    if dry_run.requested():  # the plan the call would run, not its trajectory
        return _dry_run.planned(name, planned, compiled, speed=chosen, expect_contact=contact,
                                view=_dry_run.view(api, obs, config))
    result = move.run(api, planned, action_id=action["action_id"], observation=obs.identity.json(),
                      name=f"this move at speed {label}", labels={"move": label, "slow": "careful"},
                      expect_contact=contact, jaw_symmetry=jaw_labels(compiled))
    receipt, motion = result["receipt"], result["motion"]
    motion = {"route": motion.pop("route"), "speed": chosen, **motion}
    limited = motion.get("budget_limited")
    if limited is not None:
        limited["limited_by"] = limited_by
        limited["note"] += LIMITED_NOTES[limited_by]
    # A fast move keeps move_pose's status (the action ran; the receipt and motion say how it ended); normal,
    # careful and speed_m_s name a stop or a partial move in their status, as the separate speed tools did.
    completed = motion.get("status") == "completed"
    status = ("executed" if speed == DEFAULT_SPEED else "partial" if planned.fraction < 1.0 and completed
              else "stopped" if receipt.get("status") != "reached" else "executed")
    return {"status": status, "action": name, "compiled_pose": compiled,
            "receipt": _previewed(model_receipt(receipt), reused), "remaining_budget": result["remaining_budget"],
            "motion": motion}


__all__ = ["ALLOWANCE_LIMITS", "CAREFUL", "CUSTOM_SPEED", "DEFAULT_SPEED", "FULL", "FULL_SETTLE",
           "IK_UNRECORDED_WARNING", "LIMITED_NOTES", "NORMAL", "PACED_SETTLE", "SPEEDS", "SPEED_CAPABILITY",
           "SPEED_FIELDS", "allowance", "available", "call_options", "contact_option", "custom_pace", "global_planner",
           "jaw_labels", "jaw_option", "options", "plan_action", "planned_move", "proposed_speed", "run_action",
           "run_move", "speeds"]
