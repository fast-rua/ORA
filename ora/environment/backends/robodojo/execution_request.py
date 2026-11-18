"""Request-scoped controller bookkeeping; nothing survives an RPC return.

The dictionaries keep the existing numerical execution path unchanged, but
are borrowed from the host for this call and returned as explicit updates.
No agent/state-store import belongs in the worker.
"""

from copy import deepcopy
from ora.contracts.execution_state import FIELDS, SCHEMA, pack, unpack, validate

#: The CuRobo IK seed generators' state, part of every publication snapshot (``ik_sampler``).
SAMPLER_KEY = "exec.ik_sampler_state"


class ExecutionRequest:
    def __init__(self, session, value):
        self.session = session
        self.value = validate(value)
        self.updates = None

    def __enter__(self):
        session = self.session
        if any(hasattr(session, field) for field in FIELDS):
            raise RuntimeError("worker already holds an execution request")
        fields = unpack(self.value["values"])
        if session.started and not session.closed:
            if any(set(fields[name]) != {"left", "right"} for name in ("pose_targets", "gripper_directives")):
                raise ValueError("execution request lacks both arms' holding directives")
        for field, values in fields.items():
            setattr(session, field, values)
        session.request_action_budget = self.value.get("action_budget", 20)
        # The tool invocation this action belongs to (one undo entry per arm-moving tool call).
        session.tool_call = deepcopy(self.value.get("tool_call"))
        session.recovery_targets = deepcopy(self.value.get("recovery_targets", {}))
        sampler = self.value["values"].get(SAMPLER_KEY)
        if sampler is not None and session.started and not session.closed:
            # The host's snapshot is the authority: a restored or forked run
            # continues from the generator state its boundary recorded.
            from .ik_sampler import restore_sampler_states
            restore_sampler_states(session, sampler)
        return self

    def __exit__(self, *exc):
        session = self.session
        try:
            if getattr(session, "started", False) and not session.closed and session.environment is not None:
                session.joint_targets = retained_joint_targets(session, session.joint_targets)
            # A recorded path belongs to the anchor it starts from: a later move that set a new anchor
            # without recording one (a released move_pose, a restore) leaves none.
            session.last_safe_paths = {arm: path for arm, path in session.last_safe_paths.items()
                                       if path and path[0] == list(session.last_safe_joints.get(arm, ()))}
            values = pack({name: getattr(session, name) for name in FIELDS})
            if getattr(session, "started", False) and not session.closed:
                values.update({
                    "exec.world_mutation_sequence": session.world_mutation_sequence,
                    "exec.last_receipt": deepcopy(session.last_receipt["receipt"]) if session.last_receipt else None,
                    "exec.native_actions": {
                        "limit": min(session.extended_native_action_limit, session.max_sim_steps // 10),
                        "used": int(session.environment.take_action_cnt[0]),
                    },
                    "exec.round": {"index": session.round_index, "task_instruction": session._task_instruction()},
                })
                from .ik_sampler import sampler_states
                sampler = sampler_states(session)
                if sampler is not None:
                    values[SAMPLER_KEY] = sampler
            self.updates = {"schema": SCHEMA, "values": values}
        finally:
            for field in (*FIELDS, "recovery_targets", "request_action_budget", "tool_call"):
                delattr(session, field)
        return False


def retained_joint_targets(session, previous):
    """Each target arm's joint target the plant keeps applying (``exec.command.<arm>.joint_target_rad``).

    Read from the SDK's retained command after every request, so it is what
    the arm holds whichever executor wrote it: a trajectory's last knot, the
    measured joints a guard stopped at, a legacy IK target or a stall hold.
    """
    import math

    manager = session.environment.robot_manager
    retained = manager.control_manager.prev_control[0]
    result = dict(previous)
    for arm, robot in getattr(session, "robots", {}).items():
        entry = retained.get(manager.process_name(robot.arm_name))
        position = entry.get("position") if isinstance(entry, dict) else None
        if position is None:
            continue
        values = [float(value) for value in (position.tolist() if hasattr(position, "tolist") else position)]
        if len(values) == 6 and all(math.isfinite(value) for value in values):
            result[arm] = values
    return result
