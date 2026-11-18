"""recover_pose: undo the latest moves along their measured paths, or the released anchor recovery.

``{"arms": [...]}`` (the trajectory surfaces, v4/v5): each arm retraces its
latest not-yet-undone move back to that move's start, one move per call
(``_trajectory.run_retrace``). ``{"targets": {arm: previous|home}}`` (the
released surfaces v1-v3): read the selected safe/home anchor and request the
environment's released ``recover_motion`` route. Where the environment plans
(``_context.environment_plans``) the strategies alone are proposed: it holds the anchors.
"""

from ...interface import dry_run
from ...lib.observation import observe
from ...lib.values import PublicResult
from ...interface.types import NotRecorded
from .._receipts import model_receipt
from .. import _dry_run
from ..errors import ToolRefusal
from ..move_pose import _context
from ..move_pose.tool import measured_states, propose, surface_targets
from . import _trajectory


def run(api, arguments):
    """Replay: read recorded recovery anchors; absent state fails, no online fallback.

    Anchors contain six joint radians and episode-local link6 metres / xyzw.
    The owner admits only its current anchor; stale or fabricated values spend 0.
    With ``arms``: recorded retrace stacks, kinematics answers and the one recorded trajectory primitive.
    """
    config = api.artifacts.tool_context()
    refusal = config["motion"]["mutation_refusals"].get("recover_pose")
    if refusal is not None:
        raise ToolRefusal.from_public(refusal)
    proposes = _context.environment_plans(config)
    if not proposes and isinstance(arguments, dict) and "arms" in arguments:
        arms = _trajectory.arms_argument(arguments, "recover_pose")
        if dry_run.requested():  # a plan the real call would refuse is the dry run's answer
            return _dry_run.answered("recover_pose", lambda: _trajectory.run_retrace(api, config, arms))
        return _trajectory.run_retrace(api, config, arms)
    targets = surface_targets(arguments, config, scalar_key="strategy")
    if any(value not in ("previous", "home") for value in targets.values()):
        raise ToolRefusal("recover_pose targets must be previous or home", code="invalid_motion_arguments")
    obs = observe(api)
    measured_states(obs.manifest.json(), _context.arms(config))
    identity = obs.identity.json()
    strategies = {arm: "backtrack_last_safe" if strategy == "previous" else "home"
                  for arm, strategy in targets.items()}
    compiled = {"semantic_primitive": "recover_pose", "canonical_action": "recover_motion",
                "targets": {arm: {"strategy": strategy, "canonical_strategy": strategies[arm]}
                            for arm, strategy in targets.items()}}
    if proposes:
        return propose(api, _action(config, identity, {"targets": strategies}), "recover_pose", compiled_pose=compiled)
    anchors = {}
    for arm, strategy in targets.items():
        try:
            anchor = api.state.read(f"exec.recovery.{arm}.{strategy}")
            anchors[arm] = {"strategy": strategies[arm],
                            "arm_joint_qpos": anchor["arm_joint_qpos"],
                            "pose_xyz_xyzw": anchor["pose_xyz_xyzw"]}
        except (NotRecorded, KeyError, TypeError) as error:
            raise ToolRefusal("stored recovery state is unavailable", code="plan_rejected",
                              constraints={"failure_code": "recovery_state_unavailable",
                                           "arm": arm, "strategy": strategy, "native_actions": 0}) from error
    action = _action(config, identity, {"targets": strategies, "recovery_targets": anchors})
    if dry_run.requested():  # the environment plans this route: it answers for it (``Proposed``)
        PublicResult.of(api.act.submit_action(action=action)).data
    value = PublicResult.of(api.act.execute_action(action, 20)).data.json()
    return {"status": "executed", "action": "recover_pose", "compiled_pose": compiled,
            **value, "receipt": model_receipt(value["receipt"])}


def _action(config, identity, arguments):
    return {"schema": "ora-motion-action.v3", "action": "recover_motion",
            "action_id": f"action-{config['execution']['generation']}",
            "observation_id": identity["observation_id"],
            "bound_observation_sha256": identity["observation_sha256"],
            "arguments": arguments}
