"""One native tick fed a target per physics step (``EvaluationLifecycle.native_tick_sequence``).

``native_tick`` (kept byte-identical for old-run restore) reproduces the
official ``process_control_info``: every native action restarts from the
measured joints, ramps over 8 physics steps, holds the target for the last 2
and writes a joint velocity target of 0, so a chain of native actions stops
and starts every 40 ms. This tick instead takes the 10 requested control dicts
from its caller -- generated at run time by an executor, or the recorded ones
during a restore -- and may carry a per-step arm ``velocity``. Everything else
is the official lifecycle unchanged: the terminal and native-action-limit
checks, ``take_action_cnt``, the interactive support-arm queue (popped at the
tick start exactly as ``interpolate`` pops it), ``_fixed_meta_control`` with
the SDK's own gripper limiter, the capture and trace callbacks after every
physics step, the reward step, the support-arm hooks, the unstable check and
the official terminal transition.

It copies less than ``native_tick``: the requested dicts are already the
caller's own (``native_sequence.run_tick`` copies each, a restore copies the
recording) and the SDK only reads what it is pushed; ``pop`` rebinds the keys
of ``prev_control`` without writing into a target, so its bindings are the
held targets; ``pop`` leaves the popped instance's ``get_action`` result in
``prev_control`` for every dimension (``update_prev_control``), limited against
the same finger reading as no physics step comes between, so
``_fixed_meta_control`` takes it instead of reading the fingers a third time;
and the trace callback receives the live resolved command, which it must
serialize or copy before the next physics step.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .bridge import RobodojoError, _fixed_meta_control

Requested = Callable[[int], Mapping[str, Any]] | Sequence[Mapping[str, Any]]


def _support_steps(environment: Any, count: int) -> list[dict[str, Any]]:
    """The support-arm entries ``interpolate`` would merge into each step of this tick."""
    steps: list[dict[str, Any]] = [{} for _ in range(count)]
    for robot in environment.robot_manager.robot_list:
        if robot.type != "target" and hasattr(environment, "support_arm_action"):
            for step in steps:
                if environment.support_arm_action[0]:
                    step.update(environment.support_arm_action[0].pop(0))
    return steps


def run_native_tick_sequence(environment: Any, requested: Requested, step: Callable[..., Any]) -> dict[str, Any] | None:
    """Advance one native action of 10 physics steps with caller-supplied targets.

    ``requested`` is either the 10 control dicts or a function of the step
    index (0..9) called just before that step, after the previous step's
    measurement, so an executor can react to it. ``step`` is the task class's
    original ``step`` (``super().step`` of the lifecycle mixin).
    """
    if environment._official_terminal is not None or environment.end_flag[0]:
        raise RobodojoError("an officially terminated episode cannot accept an action")
    if not environment.reward_registered:
        raise RobodojoError("reward registration is required before action execution")
    if environment.take_action_cnt[0] >= environment.native_action_ceiling:
        raise RobodojoError("native action limit was reached without a terminal transition")
    count = int(environment.obs_manager.collect_interval)
    if count != 10:
        raise RobodojoError("native control cadence must be 10 physics steps")
    if not callable(requested) and len(requested) != count:
        raise RobodojoError("a recorded native tick must hold exactly 10 physics-step targets")
    environment.take_action_cnt[0] += 1
    support = _support_steps(environment, count)
    manager = environment.robot_manager.control_manager
    for index in range(count):
        merged = {**support[index], **(requested(index) if callable(requested) else requested[index])}
        manager.push([0], [[merged]])
        previous = dict(manager.prev_control[0])
        commands = manager.pop([0])
        action = {key: manager.prev_control[0][key] for key in environment.robot_manager.get_robot_obs_name()}
        commands[0], resolved = _fixed_meta_control(commands[0], environment.robot_manager, previous, merged, action)
        manager.prev_control[0] = resolved
        step(meta_control_list=commands)
        environment.sim_step(render=False)
        capture = getattr(environment, "execution_capture_callback", None)
        if capture is not None:
            capture()
        callback = getattr(environment, "controller_trace_callback", None)
        if callback is not None:
            callback(manager.prev_control[0])
    if not manager.get_empty([0]):
        raise RobodojoError("native control queue did not drain at the fixed cadence")
    environment.reward_manager.step(env_idx_list=[0])
    if getattr(environment, "interact", False):
        if hasattr(environment, "query_support_arm_traj"):
            environment.query_support_arm_traj(env_idx=0)
        if hasattr(environment, "check_support_arm_stable"):
            environment.check_support_arm_stable(env_idx=0)
    if environment.unstable_envs:
        raise RobodojoError("official environment became unstable")
    return environment.advance_official_terminal()


__all__ = ["run_native_tick_sequence"]
