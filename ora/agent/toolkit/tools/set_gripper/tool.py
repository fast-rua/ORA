"""Explicit jaw commands, preserving the public close-intent annotation."""

from ...interface import dry_run
from ...lib.motion_plan.execute import gripper_action
from ...lib.observation import observe
from .. import _dry_run, _held
from .._receipts import model_receipt
from ..budgets import native_action_budget
from ..errors import ToolRefusal
from ..move_pose import _context, _profile
from ..move_pose.tool import measured_states, propose, surface_targets

#: The surface capability that publishes the close-intent annotation; a surface without it refuses ``target_ref``.
ANNOTATION_CAPABILITY = "public_rgbd_retention_advisory_v1"

#: Rate-limited jaws (environment ``gripper_ramp``) without a motion profile (profile v1), chosen from the
#: measured full-drive behaviour (fingers about 0.24 m/s, squeeze about 25 N per finger
#: at the SDK's 10.8 mm lead, 73% of closes starting on a moving arm):
#: a close moves at 2 openings/s (about 0.11 m/s of finger travel), never
#: leads the measured finger by more than 5 mm (about half the squeeze and
#: approach speed of the SDK lead), and first waits up to 5 native actions for
#: the arm to move slower than 0.05 rad/s; an open is at full rate with an
#: empty jaw and at the close rate when it releases a held close. A context's
#: motion profile names its own forms (``move_pose._profile``): under profile v2
#: a close approaches at the SDK lead and squeezes at 5 mm once its finger stops
#: (``hold_lead_m``), skips the settle look on an arm already at rest, a release
#: opens at full rate, and every command ends after 2 still native actions.
CLOSE = _profile.LEGACY["gripper"]["close"]
RELEASE = _profile.LEGACY["gripper"]["release"]
OPEN = _profile.LEGACY["gripper"]["open"]


def rate_limited_targets(targets, effectors, config=None):
    """The object form of each directive (``joint_trajectory.validate_gripper_target``), as the call's motion profile
    forms it (``CLOSE``, ``RELEASE``, ``OPEN`` without one)."""
    result = {}
    for arm, command in targets.items():
        held_close = (effectors.get(arm) or {}).get("gripper_command_target") == "close"
        form = "close" if command == "close" else "release" if held_close else "open"
        result[arm] = {"opening": command, **_profile.gripper_form(config or {}, form)}
    return result


def run(api, arguments):
    """Replay: use recorded observation/state/action; missing evidence fails.

    `targets` maps left/right to open/close. `target_ref` names intended contents,
    not measured attachment. Pose units are episode-local metres / xyzw.
    """
    config = api.artifacts.tool_context()
    refusal = config["motion"]["mutation_refusals"].get("set_gripper")
    if refusal is not None:
        raise ToolRefusal.from_public(refusal)
    proposes = _context.environment_plans(config)
    annotated = not proposes or ANNOTATION_CAPABILITY in config["motion"]["capabilities"]
    reference = arguments.get("target_ref") if annotated else None
    if reference is not None and (not isinstance(reference, str) or not reference.strip()):
        raise ToolRefusal("target_ref must be a non-empty intent annotation or null",
                          code="invalid_arguments", field="target_ref")
    targets = surface_targets({key: value for key, value in arguments.items() if key != "target_ref" or not annotated},
                              config, scalar_key="action")
    if any(value not in ("open", "close") for value in targets.values()):
        raise ToolRefusal("gripper targets must be open or close", code="invalid_motion_arguments")
    obs = observe(api)
    manifest = obs.manifest.json()
    current = measured_states(manifest, _context.arms(config))
    identity = obs.identity.json()
    if dry_run.requested() and not proposes:  # the jaw's own motion; where a closing jaw stops is the world's
        return _dry_run.jaws("set_gripper", targets, _dry_run.view(api, obs, config))
    rate_limited = bool(config.get("synchronous_execution") and config["motion"].get("joint_trajectory"))
    effectors = manifest.get("proprioception", {}).get("end_effectors", {})
    action = gripper_action(action_id=f"action-{config['execution']['generation']}", observation=identity,
                            targets=rate_limited_targets(targets, effectors, config) if rate_limited else targets)
    closing = [arm for arm, command in targets.items() if command == "close"]
    if proposes:  # what the jaw holds stays the host's, as it does for a grasp (``toolkit_motion.submit_adapter``)
        result = propose(api, action, "set_gripper")
        return {**result, "intent": {"target_ref": reference, "role": "agent_annotation_only"}} if (
            closing and annotated) else result
    updates = {arm: None for arm, command in targets.items() if command == "open"}
    if reference is not None and len(closing) == 1:
        arm = closing[0]
        updates[arm] = _held.intent("set_gripper", reference, current[arm])
    previous = _held.stage(api, updates)
    value = _held.execute(api, action, previous, native_action_budget("set_gripper") if rate_limited else 20)
    result = {"status": "executed", "action": "set_gripper", **value,
              "receipt": model_receipt(value["receipt"])}
    if closing:
        result["intent"] = {"target_ref": reference, "role": "agent_annotation_only"}
    return result
