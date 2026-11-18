"""Recorded bookkeeping is restored without a robot/object state setter."""

import json
from types import SimpleNamespace
from ora.environment.backends.robodojo.toolkit_restore import bind_bookkeeping


def test_bookkeeping_preserves_last_command_and_exact_receipt_bytes(tmp_path):
    native = tmp_path / "native"
    (native / "receipts").mkdir(parents=True)
    (native / "controller").mkdir()
    pose = [0, 0, 1, 0, 0, 0, 1]
    receipt = dict(
        round_index=0,
        action="move_pose",
        state_changed=True,
        motion_strategy="global",
        target_arms=["left"],
        before=dict(left=dict(arm_joint_qpos=[0.1] * 6, pose_xyz_xyzw=pose)),
        pose_targets_after=dict(left=pose),
        gripper_directives=dict(left=0.0),
        command_targets_after=dict(left_arm_joint_state=dict(position=[0.2] * 6)),
    )
    data = json.dumps(receipt, indent=2).encode()
    (native / "receipts/action-0000.json").write_bytes(data)
    for suffix in ("json", "jsonl"):
        (native / f"controller/action-0000.{suffix}").write_text("{}\n")
    session = SimpleNamespace(
        output_root=tmp_path / "output",
        last_safe_joints={},
        last_safe_poses={},
        sim_steps=10,
        environment=SimpleNamespace(
            robot_manager=SimpleNamespace(
                control_manager=SimpleNamespace(prev_control=[{}])
            )
        ),
    )
    boundary = dict(
        source_native=str(native),
        expected=dict(
            native_receipts=1,
            observation_sequence=2,
            round_index=0,
            previous_execution_receipt=dict(receipt=receipt),
        ),
    )
    bind_bookkeeping(session, boundary)
    assert session.action_sequence == 1 and session.observation_sequence == 2
    assert (
        session.world_mutation_actions == ["move_pose"]
        and session.world_mutation_sequence == 1
    )
    assert session.last_safe_joints["left"] == [0.1] * 6
    assert (
        session.pose_targets["left"] == pose
        and session.gripper_directives["left"] == 0.0
    )
    assert (
        session.environment.robot_manager.control_manager.prev_control[0]
        == receipt["command_targets_after"]
    )
    assert session.sim_steps == 10
    assert (session.output_root / "receipts/action-0000.json").read_bytes() == data


def test_native_bookkeeping_uses_pre_action_anchors_and_tick_directives_without_future_targets(tmp_path):
    from ora.environment.backends.robodojo.toolkit_native_state import bind
    from ora.contracts.execution_state import pack
    fields = {name: {} for name in ("pose_targets", "gripper_directives", "home_joints", "home_poses",
                                    "last_safe_joints", "last_safe_poses", "session_home_joints", "session_home_poses")}
    fields.update(pose_targets=dict(left=[0, 0, 1, 0, 0, 0, 1]),
                  gripper_directives=dict(left=1.0), last_safe_joints=dict(left=[0.5] * 6))
    command = {"tick": [0.2] * 6}
    session = SimpleNamespace(gripper_directives=dict(left=0.0), sim_steps=10,
                              environment=SimpleNamespace(robot_manager=SimpleNamespace(
                                  control_manager=SimpleNamespace(prev_control=[command]))))
    boundary = dict(exec_values=pack(fields), action_mapping=dict(action_id="in-progress"),
                    expected=dict(native_actions=1, observation_sequence=3, native_receipts=1,
                                  previous_execution_receipt=None))
    boundary["exec_values"]["exec.world_mutation_sequence"] = 7
    bind(session, boundary)
    assert session.world_mutation_sequence == 7
    assert session.last_safe_joints == {"left": [0.5] * 6}
    assert session.gripper_directives == {"left": 0.0}
    assert session.pose_targets == fields["pose_targets"]
    assert session.environment.robot_manager.control_manager.prev_control[0] is command
    assert session.last_receipt is None and session.sim_steps == 10


def test_native_source_versions_only_warn_and_keep_both_identities(tmp_path):
    from ora.environment.backends.robodojo.toolkit_native_state import source_warnings
    (tmp_path / "source-manifest.json").write_text(json.dumps(dict(
        source_revisions={"sdk": "old"}, layout_file={"sha256": "old-layout"})))
    (tmp_path / "reset.json").write_text(json.dumps(dict(actual_source_fingerprint="old-fingerprint")))
    session = SimpleNamespace(source=dict(source_revisions={"sdk": "new"}, layout_file={"sha256": "new-layout"}))
    warnings = source_warnings(session, {"actual_source_fingerprint": "new-fingerprint"}, tmp_path)
    assert {item["field"] for item in warnings} == {"source_revisions", "layout_file.sha256", "actual_source_fingerprint"}
    assert all(item["physical_verification_required"] for item in warnings)
    assert warnings[0]["recorded"] == {"sdk": "old"}
    assert warnings[0]["current"] == {"sdk": "new"}


def _anchor_receipts(tmp_path, receipts):
    """Receipts (with controller traces) in a source native root, as bind_bookkeeping reads them."""
    native = tmp_path / "native"
    (native / "receipts").mkdir(parents=True)
    (native / "controller").mkdir()
    for index, (receipt, samples) in enumerate(receipts):
        trace = native / f"controller/action-{index:04d}.json"
        trace.write_text(json.dumps({"samples": samples}))
        (native / f"controller/action-{index:04d}.jsonl").write_text("{}\n")
        receipt = dict(receipt, round_index=0, state_changed=True,
                       control_trace={"path": f"controller/action-{index:04d}.json"})
        (native / f"receipts/action-{index:04d}.json").write_text(json.dumps(receipt))
    session = SimpleNamespace(output_root=tmp_path / "output", last_safe_joints={"left": [9.0] * 6},
                              last_safe_poses={"left": [9.0] * 7}, last_safe_paths={}, sim_steps=0,
                              environment=SimpleNamespace(robot_manager=SimpleNamespace(
                                  control_manager=SimpleNamespace(prev_control=[{}]))))
    bind_bookkeeping(session, dict(source_native=str(native), expected=dict(
        native_receipts=len(receipts), observation_sequence=1, round_index=0, previous_execution_receipt=None)))
    return session


def _move(action, joints, *, strategy="global", recovery=False):
    receipt = dict(action=action, motion_strategy=strategy, target_arms=["left"],
                   before=dict(left=dict(arm_joint_qpos=[joints] * 6, pose_xyz_xyzw=[joints] * 7)))
    if recovery:
        receipt["trajectory"] = {"recovery": {"targets": {"left": "previous"}, "previous_anchor": "kept"}}
    return receipt


def _samples(*joints):
    return [{"measured": {"left": {"arm_joint_qpos": [value] * 6}}} for value in joints]


def test_recover_motion_keeps_the_previous_anchor_as_the_live_executor_does(tmp_path):
    # Live, recovery.execute_recovery never calls _remember_backtrack although its receipt has a strategy.
    session = _anchor_receipts(tmp_path, [(_move("move_pose", 0.1), []), (_move("recover_motion", 0.5), [])])
    assert session.last_safe_joints["left"] == [0.1] * 6 and session.last_safe_poses["left"] == [0.1] * 7


def test_a_trajectory_restores_its_anchor_and_measured_path(tmp_path):
    session = _anchor_receipts(tmp_path, [(_move("follow_joint_trajectory", 0.2, strategy="joint_trajectory"),
                                           _samples(0.3, 0.4))])
    assert session.last_safe_joints["left"] == [0.2] * 6
    assert session.last_safe_paths == {"left": [[0.2] * 6, [0.3] * 6, [0.4] * 6]}


def test_a_recovery_trajectory_keeps_the_anchor_and_its_path(tmp_path):
    session = _anchor_receipts(tmp_path, [
        (_move("follow_joint_trajectory", 0.2, strategy="joint_trajectory"), _samples(0.3)),
        (_move("follow_joint_trajectory", 0.3, strategy="joint_trajectory", recovery=True), _samples(0.2))])
    assert session.last_safe_joints["left"] == [0.2] * 6
    assert session.last_safe_paths == {"left": [[0.2] * 6, [0.3] * 6]}


def test_a_later_move_without_a_path_drops_the_old_path_and_a_refused_one_changes_nothing(tmp_path):
    session = _anchor_receipts(tmp_path, [
        (_move("follow_joint_trajectory", 0.2, strategy="joint_trajectory"), _samples(0.3)),
        (_move("move_pose", 0.3), []), (_move("move_pose", 0.7, strategy=None), []),
        (_move("set_gripper", 0.8), [])])
    assert session.last_safe_joints["left"] == [0.3] * 6 and session.last_safe_paths == {}
