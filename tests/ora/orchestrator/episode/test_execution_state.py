"""CPU request-boundary tests, including host state and fake physical execution."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ora.contracts.execution_state import FIELDS, SCHEMA
from ora.environment.backends.robodojo.session import RobodojoSession
from ora.orchestrator.episode.execution_state import ExecutionStateOwner, publish_state
from ora.agent.toolkit.state.snapshot import load_snapshot
from ora.agent.evidence.policy_state import PolicyStateStore
from tests.support.robodojo.backend import Plant, _fake_source_manifest, pose, v3_for, install_fake_sdk_candidate_boundary


def worker(tmp_path):
    plant = Plant()
    session = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
                "action_codec": "motion_action_v3"},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (plant, SimpleNamespace(close=lambda: None), _fake_source_manifest(tmp_path)),
    )
    session._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    return session


def trajectory_for(observation, session, joints, **arguments):
    """A one-knot follow_joint_trajectory of the left arm (the fake plant's joints 0-2 are its link6 position)."""
    action = v3_for(observation, session, kind="follow_joint_trajectory", targets={"left": {}})
    action["arguments"] = {"arms": {"left": {"knots_rad": [list(joints)],
                                             "expected_link6_xyz_xyzw": [[*joints[:3], 0.0, 0.0, 0.0, 1.0]]}},
                           **arguments}
    return action


def rpc(owner, session, method, **params):
    context = session.execution_request(owner.request(method, params))
    result = None
    try:
        with context:
            result = session.execute(params["action"]) if method == "execute" else getattr(session, method)(**params)
    finally:
        if context.updates is not None:
            owner.accept(method, params, context.updates, result)
        assert not any(hasattr(session, name) for name in (*FIELDS, "recovery_targets"))
    return result


def test_worker_has_no_retained_commands_and_recovery_is_explicit(tmp_path, monkeypatch):
    install_fake_sdk_candidate_boundary(monkeypatch)
    session, owner = worker(tmp_path), ExecutionStateOwner()
    rpc(owner, session, "start")
    initial = owner.writer.values()
    assert initial["exec.recovery.left.home"] == initial["exec.session_home.left"]
    observation, _ = rpc(owner, session, "observe", destination=tmp_path / "policy-observations")
    action = trajectory_for(observation, session, [0.1, 0, 0.8, 0, 0, 0])
    receipt = rpc(owner, session, "execute", action=action)["receipt"]
    assert receipt["native_actions"] > 0
    home = initial["exec.recovery.left.home"]
    previous = owner.writer.read("exec.recovery.left.previous")
    assert {key: previous[key] for key in home} == home  # the anchor, beside the path the trajectory measured
    assert owner.writer.read("exec.command.left")["pose_target_xyz_xyzw"][:3] == [0.1, 0, 0.8]
    assert owner.writer.read("exec.native_actions")["used"] == receipt["native_actions"]
    assert owner.store.complete_boundary is False
    observation, _ = rpc(owner, session, "observe", destination=tmp_path / "policy-observations")
    recover = trajectory_for(observation, session, home["arm_joint_qpos"], recovery={
        "targets": {"left": "home"}, "recovery_targets": {"left": {"strategy": "home", **home}}})
    request = owner.request("execute", {"action": recover})
    assert request["recovery_targets"]["left"]["arm_joint_qpos"] == home["arm_joint_qpos"]
    result = rpc(owner, session, "execute", action=recover)["receipt"]
    assert result["execution_status"] != "plan_rejected"
    home = owner.writer.read("exec.session_home.left")
    measured = session._proprioception
    def drifted_measurement():
        value = measured()
        value["end_effectors"]["left"]["arm_joint_qpos"][0] += 1e-8
        value["end_effectors"]["left"]["pose_xyz_xyzw"][0] += 1e-8
        return value
    monkeypatch.setattr(session, "_proprioception", drifted_measurement)
    rpc(owner, session, "restart_round", round_index=1, seed=0)
    assert owner.writer.read("exec.session_home.left") == home
    assert owner.writer.read("exec.recovery.left.home") != home
    assert owner.writer.read("exec.round")["index"] == 1
    rpc(owner, session, "close")


def test_exception_returns_updates_and_clears_worker_scratch(tmp_path):
    session, owner = worker(tmp_path), ExecutionStateOwner()
    rpc(owner, session, "start")
    context = session.execution_request(owner.request("execute", {}))
    original = RuntimeError("original failure")
    with pytest.raises(RuntimeError) as raised:
        with context:
            session.gripper_directives["left"] = 0.25
            raise original
    assert raised.value is original
    owner.accept("execute", {}, context.updates, None)
    assert owner.writer.read("exec.command.left")["gripper_directive"] == 0.25
    assert not any(hasattr(session, name) for name in FIELDS)
    with pytest.raises(ValueError, match="holding directives"):
        with session.execution_request({"schema": SCHEMA, "values": {}}):
            pytest.fail("missing holds entered worker")
    session.close()


@pytest.mark.parametrize("reason", [None, "official_success", "official_failure", "official_task_end"])
def test_publication_snapshot_matches_original_bytes_without_worker(tmp_path, reason):
    session, owner = worker(tmp_path), ExecutionStateOwner()
    rpc(owner, session, "start")
    observation, _ = rpc(owner, session, "observe", destination=tmp_path / "policy-observations")
    policy_store = PolicyStateStore(tmp_path, "test-run", "test-plan")
    source = tmp_path / "policy-observations" / observation["observation_id"]
    snapshot = policy_store.publish(observation=observation, observation_root=source, cause="reset",
                                    task_instruction="Task", previous_public_action=None, remaining_budget={})
    loop = SimpleNamespace(execution_state_owner=owner, round_index=0, run_root=tmp_path,
                           plan={"runtime": {}}, task_instruction="Task")
    status = {**rpc(owner, session, "status"), "terminal": reason is not None, "reason": reason}
    publish_state(loop, snapshot, policy_store.steps[-1], status)
    values = owner.store.access(reader="rsi").values()
    session.close()
    loaded = load_snapshot(tmp_path, 0)
    assert loaded.access(reader="rsi").values() == values
    for path in source.rglob("*"):
        if path.is_file():
            assert loaded.read_artifact(path.relative_to(tmp_path).as_posix()) == path.read_bytes()
    assert "privileged.task_check" not in loaded.access().values()
    public_terminal = loaded.access().read("env.terminal")
    assert public_terminal == {"terminal": reason is not None, "reason": "official_boundary" if reason else None}
    assert "official_success" not in json.dumps(public_terminal)
    assert "official_failure" not in json.dumps(public_terminal)
    private = loaded.access(reader="rsi").read("privileged.task_check")
    assert private["reason"] == reason
    assert private["task_success"] is (reason == "official_success")
    assert list((tmp_path / "state").rglob("*.png")) == []


def test_tool_call_count_counts_parent_once_and_refusals_zero():
    owner = ExecutionStateOwner()
    owner.finish_tool_call({"receipt": {"native_actions": 0}})
    assert owner.writer.read("exec.tool_calls_with_actions") == 0
    owner.finish_tool_call({"schema": "ora.composite-motion-receipt.v1", "phase_receipts": [
        {"receipt": {"receipt": {"native_actions": 2}}},
        {"receipt": {"receipt": {"native_actions": 1}}},
    ]})
    assert owner.writer.read("exec.tool_calls_with_actions") == 1


def test_actual_rpc_carries_updates_on_error_and_cleans_worker(tmp_path, monkeypatch):
    import socket
    import tempfile
    import threading
    import time
    from pathlib import Path
    from ora.environment import worker as server
    from ora.environment.rpc import LocalRpcEnvironmentSession, RPC_SCHEMA, EnvironmentRpcError

    session, owner = worker(tmp_path), ExecutionStateOwner()
    monkeypatch.setattr(server, "_build_session", lambda config: session)
    def failed_execute(action):
        session.gripper_directives["left"] = 0.125
        raise ValueError("failure after updating request state")
    monkeypatch.setattr(session, "execute", failed_execute)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema": RPC_SCHEMA, "session_id": "state-test",
                                  "shared_root": str(tmp_path)}))
    errors = []
    with tempfile.TemporaryDirectory(prefix="ora-state-rpc-") as directory:
        address = Path(directory) / "worker.sock"
        def serve():
            try:
                server.serve(address, config)
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(3)
            deadline = time.monotonic() + 3
            while True:
                try:
                    connection.connect(str(address))
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise AssertionError(errors)
                    time.sleep(0.01)
            with connection.makefile("r", encoding="utf-8") as reader, connection.makefile("w", encoding="utf-8") as writer:
                proxy = object.__new__(LocalRpcEnvironmentSession)
                proxy.process = SimpleNamespace(poll=lambda: None)
                proxy.connection, proxy.reader, proxy.writer = connection, reader, writer
                proxy.session_id, proxy.request_sequence = "state-test", 0
                proxy.execution_state_owner = owner
                proxy._request("start", {})
                assert not any(hasattr(session, name) for name in FIELDS)
                assert owner.writer.read("exec.recovery.left.home")["arm_joint_qpos"]
                # The dispatcher validates the public envelope before execute.
                observation, _ = proxy.observe(tmp_path / "policy-observations")
                action = v3_for(observation, session, targets={"left": pose([0.1, 0, 0.8])})
                with pytest.raises(EnvironmentRpcError, match="failure after updating"):
                    proxy.execute(action)
                assert owner.writer.read("exec.command.left")["gripper_directive"] == 0.125
                assert not any(hasattr(session, name) for name in FIELDS)
                proxy._request("close", {})
        thread.join(timeout=3)
        assert not thread.is_alive() and not errors


def test_per_action_budget_reaches_worker_and_is_cleared(tmp_path):
    session, owner = worker(tmp_path), ExecutionStateOwner()
    rpc(owner, session, "start")
    try:
        observation, _ = rpc(owner, session, "observe", destination=tmp_path / "observations")
        action = {
            "schema": "ora-motion-action.v3", "action": "wait",
            "action_id": "limited-wait", "arguments": {"ticks": 20},
            "observation_id": observation["observation_id"],
            "bound_observation_sha256": "0" * 64,
        }
        with owner.limited_action(3):
            assert owner.request("execute", {"action": action})["action_budget"] == 3
            result = rpc(owner, session, "execute", action=action)
        assert result["receipt"]["native_actions"] == 3
        assert not hasattr(session, "request_action_budget")
        assert "action_budget" not in owner.request("execute", {"action": action})
        with pytest.raises(RuntimeError, match="fixture"):
            with owner.limited_action(2):
                raise RuntimeError("fixture")
        assert owner.action_budget is None
    finally:
        session.close()


def test_an_action_request_lends_its_tool_call_with_its_ordinal_only_while_it_runs():
    """The worker keeps one undo entry per arm-moving tool call, so each execute names its call."""
    owner = ExecutionStateOwner()
    owner.writer.write("exec.tool_calls_with_actions", 2)
    with owner.limited_action(5, {"tool_call_id": "call-7", "tool": "grasp_execute"}):
        lent = owner.request("execute", {"action": {}})
        assert lent["tool_call"] == {"tool_call_id": "call-7", "tool": "grasp_execute", "ordinal": 3}
        assert "tool_call" not in owner.request("observe", {})
    assert "tool_call" not in owner.request("execute", {"action": {}})
    with owner.limited_action(5):
        assert "tool_call" not in owner.request("execute", {"action": {}})


def test_a_later_action_of_a_counted_parent_keeps_the_ordinal_of_its_decision():
    """A parent's first spend counts its robot decision (``finish_tool_call``); its later actions (grasp_execute's
    arm stages after the open, a code-mode script's next call) belong to that same decision, not the next one."""
    owner = ExecutionStateOwner()
    owner.writer.write("exec.tool_calls_with_actions", 3)
    with owner.limited_action(5, {"tool_call_id": "call-7#002", "tool": "move_pose", "decision_counted": True}):
        assert owner.request("execute", {"action": {}})["tool_call"] == {
            "tool_call_id": "call-7#002", "tool": "move_pose", "ordinal": 3}
    with owner.limited_action(5, {"tool_call_id": "call-8", "tool": "move_pose", "decision_counted": False}):
        assert owner.request("execute", {"action": {}})["tool_call"]["ordinal"] == 4


@pytest.mark.parametrize("mutation", ["nan", "short", "wrong_arm", "strategy", "extra"])
def test_explicit_recovery_contract_rejects_malformed_targets(mutation):
    from ora.contracts.action import validate_runtime_action, ActionValidationError
    value = {"strategy": "home", "arm_joint_qpos": [0.1] * 6,
             "pose_xyz_xyzw": [0, 0, .3, 0, 0, 0, 1]}
    targets = {"left": value}
    if mutation == "nan":
        value["arm_joint_qpos"][0] = float("nan")
    elif mutation == "short":
        value["arm_joint_qpos"].pop()
    elif mutation == "wrong_arm":
        targets = {"right": value}
    elif mutation == "strategy":
        value["strategy"] = "backtrack_last_safe"
    else:
        value["untrusted"] = 1
    with pytest.raises(ActionValidationError):
        validate_runtime_action({"schema": "ora-motion-action.v3", "action": "recover_motion",
            "action_id": "a", "observation_id": "o", "bound_observation_sha256": "a" * 64,
            "arguments": {"targets": {"left": "home"}, "recovery_targets": targets}})


def test_every_response_that_moves_the_ik_generator_reaches_the_exec_ledger(tmp_path):
    """Any method (here a read-only align_plan and an execute) that advanced the generator is recorded."""
    from ora.agent.toolkit.state.exec_ledger import bind_exec_ledger, read_rows

    owner = ExecutionStateOwner()
    state = lambda step: {"schema": "ora.robodojo-ik-sampler-state.v1", "planners": {"x5": {"state_hex": step}}}
    bind_exec_ledger(owner.store, tmp_path)
    for method, step in (("align_plan", "01"), ("align_plan", "01"), ("execute", "02")):
        owner.accept(method, {}, {"schema": SCHEMA, "values": {"exec.ik_sampler_state": state(step)}}, None)
    assert [(row["kind"], row["method"]) for row in read_rows(tmp_path)] == [
        ("bound", None), ("change", "align_plan"), ("change", "execute")]
