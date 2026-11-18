"""Diagnostic-only control restoration; the qualified session and kernels stay unchanged.

No tool can invoke this module. The host supplies a frozen native prefix, then
must authorize the verified boundary separately before regular RPC is enabled.
Private object poses are returned only to that verifier, never to toolkit APIs.
"""

from copy import deepcopy
import json
from pathlib import Path
import time

from .toolkit_render import RenderHistory

from .control_replay import (
    ReplayError,
    ReplayJournal,
    _require_same_sources,
    object_pose_reader,
    recorded_actions,
    replay_control,
)


def load(path):
    return json.loads(Path(path).read_bytes())


class Journal(ReplayJournal):
    """Keep the existing physics journal and publish per-native-tick progress."""

    def __init__(self, root, progress, total, renders=None):
        super().__init__(root)
        self.progress, self.total, self.lines = progress, total, 0
        self.renders = renders

    def open(self, sequence):
        if self.renders is not None:
            self.renders.attach()
        return super().open(sequence)

    def append(self, path, line):
        super().append(path, line)
        self.lines += 1
        if self.lines % 10 == 0:
            with self.progress.open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            t=time.time(),
                            stage="replay_controls",
                            done=self.lines // 10,
                            total=self.total,
                            message="Replayed recorded native controls",
                        )
                    )
                    + "\n"
                )
                stream.flush()


def inspect(session, *, objects=False):
    """Operator-only boundary evidence; no physical-state setter."""
    return dict(
        round_index=session.round_index,
        physics_clock=session._clock(),
        sim_steps=session.sim_steps,
        native_actions=int(session.environment.take_action_cnt[0]),
        proprioception=session._proprioception(),
        objects=object_pose_reader(session)() if objects else None,
    )


def restore(session, config, reset):
    boundary = config["toolkit_boundary"]
    native = Path(boundary["source_native"])
    expected = boundary["expected"]
    warnings = _require_same_sources(session, reset, native)
    actions = recorded_actions(native, upto=expected["native_receipts"])
    if len(actions) != expected["native_receipts"]:
        raise ReplayError("native prefix length differs from selected boundary")
    renders = RenderHistory(session, boundary.get("render_history", {}))
    journal = Journal(
        session.output_root / "restored-prefix",
        Path(config["toolkit_progress"]),
        boundary["native_ticks"],
        renders,
    )
    try:
        renders.attach()
        report = replay_control(session, actions, journal=journal)
        # A selected call may follow a fresh round reset without controls.
        for index in range(session.round_index + 1, expected["round_index"] + 1):
            session.restart_round(index, seed=0)
            renders.attach()
        renders.synchronize()
    finally:
        rendering = renders.close()
    report["render_history"] = rendering
    if warnings:
        report["warnings"] = warnings
    report["not_replayed"]["probes_and_observations"] = (
        "probes omitted; recorded render synchronizations are replayed separately "
        "at their original physics clocks (see render_history)"
    )
    if boundary.get("schema") == "ora.toolkit.native-boundary.v1":
        from .toolkit_native_state import bind
        bind(session, boundary)
    else:
        bind_bookkeeping(session, boundary)
    # The IK seed generators continue from the boundary's snapshot; a run
    # recorded before that state existed restarts them, with a warning.
    from .ik_sampler import UNRECORDED, UNRECORDED_WARNING, resume
    report["ik_sampler_state"] = resume(session, (boundary.get("exec_values") or {}).get("exec.ik_sampler_state"))
    if report["ik_sampler_state"] == UNRECORDED:
        report.setdefault("warnings", []).append({"kind": "ik_sampler_state", "state": UNRECORDED,
                                                  "message": UNRECORDED_WARNING})
    actual = inspect(session, objects=expected["objects"] is not None)
    return dict(replay=report, actual=actual)


#: The executors that move the previous recovery anchor (``sparse_motion._remember_backtrack``): a pose
#: move or plan reference once routed, and a joint trajectory once admitted. ``recover_motion`` and a
#: recovery trajectory (``trajectory.recovery``) keep it; gripper and wait actions never touch it.
ANCHOR_ACTIONS = frozenset({"move_pose", "move_free_space", "follow_joint_trajectory"})


def sets_previous_anchor(receipt):
    """Whether the live executor of this receipt moved the previous anchor to its pre-action joints."""
    return (receipt.get("action") in ANCHOR_ACTIONS and receipt.get("motion_strategy") is not None
            and "before" in receipt and not (receipt.get("trajectory") or {}).get("recovery"))


def undo_stacks(native, receipts, arms):
    """Each arm's undo stack as this round's live trajectories left it.

    Every trajectory receipt names its undo operation (``trajectory.undo``: push or extend from its recorded
    trace, pop, or a cut retrace's progress); ``trajectory.apply_undo`` replays them in order.
    """
    from .trajectory import apply_undo

    stacks = {arm: [] for arm in arms}
    for receipt in receipts:
        undo = (receipt.get("trajectory") or {}).get("undo")
        if undo is None:
            continue
        pushes = any(op in ("push", "extend") for op in undo["arms"].values())
        samples = load(native / receipt["control_trace"]["path"]).get("samples", []) if pushes else []
        apply_undo(stacks, undo, receipt.get("before") or {}, samples)
    return stacks


def bind_bookkeeping(session, boundary):
    """Restore recorded counters/targets after verification, never robot/object poses.

    Fresh sensors are captured for diagnostics before granting tool authority.
    Tools retain the original published bytes for frozen candidates and inputs.
    The fresh worker starts with no old plan-reference cache.
    """
    expected, native = boundary["expected"], Path(boundary["source_native"])
    receipts = [
        load(native / "receipts" / f"action-{i:04d}.json")
        for i in range(expected["native_receipts"])
    ]
    current = [
        r for r in receipts if r.get("round_index", 0) == expected["round_index"]
    ]
    session.action_sequence = expected["native_receipts"]
    session.observation_sequence = expected["observation_sequence"]
    session.world_mutation_actions = [
        str(r["action"]) for r in current if r.get("state_changed")
    ]
    session.world_mutation_sequence = len(session.world_mutation_actions)
    session.last_receipt = deepcopy(expected["previous_execution_receipt"])
    paths = {}
    for receipt in current:
        if "pose_targets_after" in receipt:
            session.pose_targets = deepcopy(receipt["pose_targets_after"])
        if sets_previous_anchor(receipt):
            for arm in receipt.get("target_arms", ()):
                state = receipt["before"][arm]
                session.last_safe_joints[arm] = list(state["arm_joint_qpos"])
                session.last_safe_poses[arm] = list(state["pose_xyz_xyzw"])
            if receipt.get("action") == "follow_joint_trajectory":
                # trajectory.remember_measured_path: the anchor, then the joints measured at the end of
                # every native action it ran, which the recorded trace samples hold.
                samples = load(native / receipt["control_trace"]["path"]).get("samples", [])
                for arm in receipt.get("target_arms", ()):
                    paths[arm] = [[float(v) for v in receipt["before"][arm]["arm_joint_qpos"]]] + [
                        [float(v) for v in sample["measured"][arm]["arm_joint_qpos"]] for sample in samples]
        # execution_request drops a path whose first row is no longer its anchor after every request.
        paths = {arm: path for arm, path in paths.items()
                 if path and path[0] == list(session.last_safe_joints.get(arm, ()))}
    if hasattr(session, "last_safe_paths"):
        session.last_safe_paths = paths
    if hasattr(session, "undo_stacks"):
        session.undo_stacks = undo_stacks(native, current, getattr(session, "home_joints", {}))
    if current:
        last = current[-1]
        if "gripper_directives" in last:
            session.gripper_directives = deepcopy(last["gripper_directives"])
        if "command_targets_after" in last:
            # An executor can change retained targets after its last physics
            # tick (e.g. a zero-budget hold). Restore the recorded command
            # buffer, never write joints, velocities or object transforms.
            session.environment.robot_manager.control_manager.prev_control[0] = (
                deepcopy(last["command_targets_after"])
            )
    # Receipt paths now refer to the diagnostic output, not the source. Only
    # completed receipts are copied; replay controller journals remain fresh.
    for i, receipt in enumerate(receipts):
        target = session.output_root / "receipts" / f"action-{i:04d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((native / "receipts" / target.name).read_bytes())
        for suffix in ("json", "jsonl"):
            name = f"action-{i:04d}.{suffix}"
            controller = session.output_root / "controller" / name
            controller.parent.mkdir(parents=True, exist_ok=True)
            controller.write_bytes((native / "controller" / name).read_bytes())
