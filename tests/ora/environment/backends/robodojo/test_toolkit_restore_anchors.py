"""A restore rebuilds the recovery anchor and its measured path exactly as the live executors left them.

The fake plant runs, live, a trajectory and a recovery trajectory (the executors RoboDojo has since the SDK
executor's move_pose and recover_motion went); after each, ``bind_bookkeeping`` over the receipts and traces
recorded so far must restore the same ``exec.recovery.<arm>.previous`` (anchor joints, pose and
``measured_path_rad``) as the live session holds.
"""
from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest

from ora.contracts.execution_state import FIELDS, pack
from ora.environment.backends.robodojo.toolkit_restore import bind_bookkeeping
from tests.support.robodojo.backend import v3

pytestmark = pytest.mark.usefixtures("fake_sdk_candidate_boundary")


def _knots(start, step=0.04, count=4, joint=0):
    rows, current = [], list(start)
    for _ in range(count):
        current = list(current)
        current[joint] += step
        rows.append(current)
    return rows


def _trajectory(session, tmp_path, knots, **arguments):
    action = v3(session, tmp_path, kind="follow_joint_trajectory", targets={"left": {}})
    action["arguments"] = {"arms": {"left": {"knots_rad": knots}}, **arguments}
    return action


def _previous(values):
    return {key: value for key, value in values.items() if key.startswith("exec.recovery.") and key.endswith(".previous")}


def _restored(sparse, tmp_path, count, name):
    """bind_bookkeeping over the first ``count`` recorded receipts, on a fresh session at the round start."""
    native = tmp_path / f"prefix-{name}"
    for kind in ("receipts", "controller"):
        (native / kind).mkdir(parents=True)
        for index in range(count):
            for suffix in ("json", "jsonl") if kind == "controller" else ("json",):
                shutil.copyfile(sparse.output_root / kind / f"action-{index:04d}.{suffix}",
                                native / kind / f"action-{index:04d}.{suffix}")
    fresh = SimpleNamespace(output_root=tmp_path / f"restored-{name}", last_safe_joints=dict(sparse.home_joints),
                            last_safe_poses=dict(sparse.home_poses), last_safe_paths={}, sim_steps=0,
                            environment=SimpleNamespace(robot_manager=SimpleNamespace(
                                control_manager=SimpleNamespace(prev_control=[{}]))))
    for name_ in FIELDS:
        if not hasattr(fresh, name_):
            setattr(fresh, name_, getattr(sparse, name_))
    bind_bookkeeping(fresh, dict(source_native=str(native), expected=dict(
        native_receipts=count, observation_sequence=0, round_index=0, previous_execution_receipt=None)))
    return _previous(pack({field: getattr(fresh, field) for field in FIELDS}))


def test_restore_matches_the_live_anchor_and_path_after_every_executor(sparse, tmp_path):
    def live():
        return _previous(pack({field: getattr(sparse, field) for field in FIELDS}))

    start = list(sparse._proprioception()["end_effectors"]["left"]["arm_joint_qpos"])
    steps = []
    sparse.execute(_trajectory(sparse, tmp_path, _knots(start)))
    steps.append(("trajectory", live()))
    path = [list(row) for row in sparse.last_safe_paths["left"]]
    recovery = {"targets": {"left": "backtrack_last_safe"},
                "recovery_targets": {"left": {"strategy": "backtrack_last_safe",
                                              "arm_joint_qpos": list(sparse.last_safe_joints["left"]),
                                              "pose_xyz_xyzw": list(sparse.last_safe_poses["left"])}}}
    sparse.execute(_trajectory(sparse, tmp_path, [list(row) for row in reversed(path[:-1])], recovery=recovery))
    steps.append(("recovery trajectory", live()))
    assert steps[0][1]["exec.recovery.left.previous"]["measured_path_rad"]  # the trajectory recorded its path
    for count, (name, expected) in enumerate(steps, start=1):
        assert _restored(sparse, tmp_path, count, count) == expected, name
