"""A dry run of a committed grasp: every planned segment up to contact, and what follows it as not predicted.

Each arm stage before the close is planned with the tools' own planner, the first from the measured state and the next
from where the one before ends, and the jaw stages are the jaws' own motion. The first contact is the close: where the
fingers stop, whether anything is held and how the lift goes are the world's, so the lift is drawn as what is commanded
and marked not predicted. Nothing here runs physics.
"""

from ...interface.dry_run import segment
from .. import _dry_run
from ..budgets import MOVE_BUDGET, native_action_budget
from ..errors import ToolRefusal
from ..move_pose import _smooth
from ...lib.motion_plan import checks, kinematics, move

AFTER_CONTACT = "after contact: not predicted"


def _pose(target):
    return [*target["position_m"], *target["orientation_xyzw"]]


def _first(api, commitment, stage, config, budget):
    arm, target = commitment["arm"], commitment["poses"][stage]
    interaction = "contact_approach" if stage == "approach" else "free_space"
    action = {"action_id": "preview", "arguments": {"targets": {arm: target}, "interactions": {arm: interaction}}}
    _obs, planned, _ = _smooth.plan_action(api, action, config, budget=budget)
    return planned


def _next(api, commitment, stage, start, joints, budget, config):
    arm = commitment["arm"]
    pace, settle = _smooth.speeds(config)[_smooth.DEFAULT_SPEED]
    return move.plan(api, starts={arm: start}, goals={arm: _pose(commitment["poses"][stage])},
                     start_joints={arm: joints}, pace=pace, settle=settle, native_action_budget=min(budget, MOVE_BUDGET),
                     route_kind="straight")


def record(api, config, commitment, sequence, obs):
    """The dry-run record of ``sequence`` (``_stages.run`` without its actions)."""
    seen = _dry_run.view(api, obs, config)
    arm, grippers = commitment["arm"], commitment["grippers"]
    effector = seen.manifest["proprioception"]["end_effectors"][arm]
    pose = [float(value) for value in effector["pose_xyz_xyzw"]]
    joints, left, native = None, native_action_budget("grasp_execute"), 0
    segments, contact = [], False
    for stage in sequence:
        target = commitment["poses"].get(stage)
        if target is None:
            if stage in grippers:
                segments.extend(_dry_run.jaw_segments({arm: grippers[stage]},
                                                      {arm: {**effector, "pose_xyz_xyzw": pose}}, seen.hands))
                contact = contact or stage == "close"
            continue
        goal = _pose(target)
        if contact:
            segments.append(segment("arm", arm, AFTER_CONTACT, planned=False, poses=[pose, goal], not_predicted=True))
            pose = goal
            continue
        try:
            planned = (_first(api, commitment, stage, config, left) if joints is None
                       else _next(api, commitment, stage, pose, joints, left, config))
        except checks.PlanRefusal as error:
            refusal = _dry_run.with_request(ToolRefusal(f"{stage}: {error}", code=error.code,
                                                        constraints={"stage": stage, **error.constraints}),
                                            {arm: pose}, {arm: goal})
            return _dry_run.refused("grasp_execute", refusal, seen)
        except kinematics.KinematicsUnavailable as error:
            raise ToolRefusal(f"the read-only kinematics query failed: {error}", code="planner_failed") from error
        poses = planned.expected[arm]
        segments.append(segment("arm", arm, f"{stage}: planned path", planned=True, poses=[pose, *poses],
                                native_actions=planned.knots["native_actions"],
                                duration_s=planned.knots["duration_s"]))
        pose, joints = [float(value) for value in poses[-1]], [float(value) for value in planned.knots["joints"][arm][-1]]
        native += planned.knots["native_actions"] + planned.settle["native_actions"]
        left -= planned.knots["native_actions"] + planned.settle["native_actions"]
    return _dry_run.composed("grasp_execute", segments, seen, route="grasp_sequence",
                             arms={arm: {"target_pose": _pose(commitment["poses"]["approach"])}},
                             motion={"native_actions": native, "native_action_budget": native_action_budget("grasp_execute")},
                             stages=list(sequence))
