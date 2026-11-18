"""Parent tools execute sequential primitives; recovery never invokes them again."""

import json
from types import SimpleNamespace
import pytest

from ora.agent.toolkit.interface.types import ActionNotStarted
from ora.orchestrator.episode import tool_execution
from ora.orchestrator.episode.code_mode_execution import CodeModeActionExecutor
from ora.orchestrator.episode.execution_state import ExecutionStateOwner
from ora.orchestrator.episode.tool_execution_log import ToolExecutionInterrupted, WorldStateUnknown


def fixture(tmp_path, monkeypatch, spends):
    control = tmp_path / "control"
    control.mkdir()
    state = {"step": 0, "used": 0, "effects": 0, "counted": [], "ref": None, "last": None}
    def current():
        return {"observation_id": f"obs-{state['step']}", "observation_sha256": "a" * 64}
    def publish_file():
        (control / "current-observation.json").write_text(json.dumps(current()))
    publish_file()
    def receipt(raw):
        state["ref"] = {"action_id": raw["action_id"], "event_id": "receipt-" + raw["action_id"],
                        "receipt_ref": {"sha256": "b" * 64}}
    def execute(loop, action, pointer, **kwargs):
        state["effects"] += 1
        used = spends.pop(0)
        assert used <= loop.execution_state_owner.action_budget
        return {"action_id": action["action_id"], "native_actions": used}, None
    def publish(loop, *args, **kwargs):
        state["step"] += 1
        loop.interaction.observation_step = state["step"]
        publish_file()
        return current(), {"terminal": False}
    def record(action, raw, duration, **kwargs):
        state["counted"].append(kwargs["inline_consumed"])
        state["used"] += raw["native_actions"]
        receipt(raw)
    planner = SimpleNamespace(record_tool_action=lambda *args: None,
                              record_environment_receipt=receipt,
                              tool_receipt_binding=lambda action_id: state["ref"],
                              begin_tool_action=lambda action_id: state.update(ref=None),
                              project_tool_receipt=lambda raw: raw)
    loop = SimpleNamespace(run_root=tmp_path, control=control, termination=None, interrupted=False,
        environment_status={"terminal": False}, round_model_calls=0,
        plan={"budgets": {"max_model_calls": 200}},
        interaction=SimpleNamespace(phase="policy_active", observation_step=0, begin_policy=lambda: None),
        native_allowance=SimpleNamespace(remaining=lambda: 100 - state["used"],
                                          fields=lambda: {"native_actions_remaining": 100 - state["used"]}),
        execution_state_owner=ExecutionStateOwner(), planner=planner,
        _exhausted_budget=lambda: None, _should_stop=lambda: False,
        _accept_submitted_action=lambda calls: (json.loads((control / "submitted-action.json").read_text()), current()),
        _record_receipt=record, record_termination_decision=lambda *a, **k: None,
        runtime=SimpleNamespace(abort=lambda reason: state.update(abort=reason)))
    monkeypatch.setattr(tool_execution, "dispatch_action", execute)
    monkeypatch.setattr(tool_execution, "publish_observation", publish)
    def action(index):
        return {"schema": "ora-motion-action.v3", "action": "wait", "arguments": {"ticks": 20},
                "action_id": f"a-{index}", "observation_id": current()["observation_id"],
                "bound_observation_sha256": "a" * 64}
    return tool_execution.ToolActionExecutor(loop), state, action


def test_two_actions_consume_measured_budget_and_count_one_parent(tmp_path, monkeypatch):
    executor, state, action = fixture(tmp_path, monkeypatch, [3, 4])
    with executor.invocation("compound", {"tool_call_id": "call", "model_call_id": "model"}, 12):
        assert executor.execute_action(action(0), 20)["remaining_budget"] == 9
        assert executor.execute_action(action(1), 20)["remaining_budget"] == 5
    assert state["counted"] == [True, False]
    assert executor.completed["native_actions_used"] == 7
    assert executor.completed["final_observation"] == {"step": 2, "observation_id": "obs-2"}
    assert state["effects"] == 2


@pytest.mark.parametrize("failure", ["schema", "observation"])
def test_validation_failure_proves_zero_spend_before_intent(tmp_path, monkeypatch, failure):
    from ora.agent.toolkit.interface.types import ActionNotStarted
    from ora.agent.toolkit.interface.runtime import public_failure
    executor, state, action = fixture(tmp_path, monkeypatch, [])
    request = action(0)
    request["schema" if failure == "schema" else "observation_id"] = "invalid"
    with pytest.raises(ActionNotStarted) as caught:
        with executor.invocation("set_gripper", {"tool_call_id": "call", "model_call_id": "model"}, 20):
            executor.execute_action(request, 20)
    assert state["effects"] == 0
    assert executor.completed["actions"] == []
    assert not executor.blocked
    assert public_failure("execute_action", caught.value)[1]["execution"] == {
        "status": "not_started", "native_actions": 0, "stage": "before_intent"}


def test_single_action_error_terminates_and_retains_result(tmp_path, monkeypatch):
    executor, state, action = fixture(tmp_path, monkeypatch, [2])
    with pytest.raises(RuntimeError, match="tool failed"):
        with executor.invocation("compound", {"tool_call_id": "call", "model_call_id": "model"}, 20):
            executor.execute_action(action(0), 20)
            raise RuntimeError("tool failed")
    assert executor.completed["native_actions_used"] == 2
    assert executor.loop.termination == "tool_execution_interrupted"
    assert executor.interruption == {
        "schema": "ora.tool-execution-interruption.v1", "reason": "tool_execution_interrupted",
        "tool_call_id": "call", "model_call_id": "model", "last_action_id": "a-0",
        "world_state_certainty": "known", "cause": {"type": "RuntimeError", "message": "tool failed", "stage": "tool"},
    }
    with pytest.raises(WorldStateUnknown):
        with executor.invocation("compound", {"tool_call_id": "later", "model_call_id": "model"}, 20):
            pytest.fail("blocked tool must not run")
    assert state["effects"] == 1


def test_all_budget_refusals_leave_round_and_observation_unchanged(tmp_path, monkeypatch):
    executor, state, action = fixture(tmp_path, monkeypatch, [])
    with executor.invocation("compound", {"tool_call_id": "call", "model_call_id": "model"}, 0):
        result = executor.execute_action(action(0), 20)
    assert result["receipt"]["reason"] == "native_action_budget_exhausted"
    assert executor.completed["native_actions_used"] == 0
    assert executor.completed["initial_observation"] == executor.completed["final_observation"]
    assert state["effects"] == 0


@pytest.mark.parametrize("failure_stage", ["worker", "publication"])
def test_open_intent_aborts_infrastructure_and_blocks_all_later_physics(tmp_path, monkeypatch, failure_stage):
    executor, state, action = fixture(tmp_path, monkeypatch, [2, 1])
    with executor.invocation("compound", {"tool_call_id": "closed", "model_call_id": "model"}, 20):
        executor.execute_action(action(0), 20)
    original_execute = tool_execution.dispatch_action
    def fail(*args, **kwargs):
        if failure_stage == "worker":
            original_execute(*args, **kwargs)
        raise RuntimeError("connection lost after possible physical work")
    monkeypatch.setattr(tool_execution, "dispatch_action" if failure_stage == "worker" else "publish_observation", fail)
    with pytest.raises(WorldStateUnknown):
        with executor.invocation("compound", {"tool_call_id": "uncertain", "model_call_id": "model"}, 20):
            executor.execute_action(action(1), 20)
    assert state["abort"] == "world_state_unknown"
    assert executor.interruption["last_action_id"] == "a-1"
    assert executor.interruption["tool_call_id"] == "uncertain"
    assert executor.loop.termination_authority.value == "infrastructure"
    before = state["effects"]
    with pytest.raises(WorldStateUnknown):
        with executor.invocation("compound", {"tool_call_id": "later", "model_call_id": "model"}, 20):
            executor.execute_action(action(2), 20)
    assert state["effects"] == before
    assert tool_execution.ToolExecutionLog(tmp_path, "closed", "model").recover()["native_actions_used"] == 2
    with pytest.raises(WorldStateUnknown):
        tool_execution.ToolExecutionLog(tmp_path, "uncertain", "model").recover()


def test_ledger_io_failure_after_physics_blocks_future_execution(tmp_path, monkeypatch):
    executor, state, action = fixture(tmp_path, monkeypatch, [2])
    with pytest.raises(WorldStateUnknown, match="not durable"):
        with executor.invocation("compound", {"tool_call_id": "call", "model_call_id": "model"}, 20) as call:
            original_save = call.log._save
            def save():
                if call.log.value["actions"]:
                    raise OSError("ledger disk unavailable after publication")
                return original_save()
            monkeypatch.setattr(call.log, "_save", save)
            executor.execute_action(action(0), 20)
    assert state["effects"] == 1
    assert executor.blocked and executor.loop.stop_session
    assert state["abort"] == "world_state_unknown"
    with pytest.raises(WorldStateUnknown):
        with executor.invocation("compound", {"tool_call_id": "later", "model_call_id": "model"}, 20):
            executor.execute_action(action(1), 20)
    assert state["effects"] == 1


@pytest.mark.parametrize("failure_stage", ["worker", "publication"])
def test_unknown_execution_settles_from_real_interaction_state(tmp_path, monkeypatch, failure_stage):
    from ora.orchestrator.episode.interaction import EpisodeInteractionState
    from ora.orchestrator.episode.tool_execution_boundary import settle_inline_execution
    executor, state, action = fixture(tmp_path, monkeypatch, [2])
    interaction = EpisodeInteractionState(run_id="run")
    interaction.start()
    interaction.publish_observation(observation_id="obs-0", step=0)
    interaction.begin_policy()
    executor.loop.interaction = interaction
    executor.loop.toolkit_executor = executor
    old_accept, old_record = executor.loop._accept_submitted_action, executor.loop._record_receipt
    def accept(count):
        accepted, pointer = old_accept(count)
        interaction.record_policy_outcome(kind="action", model_calls=count, action_id=accepted["action_id"])
        interaction.begin_execution()
        return accepted, pointer
    def record(*args, **kwargs):
        old_record(*args, **kwargs)
        interaction.record_execution(receipt_id="receipt", consumed_decision=True)
    executor.loop._accept_submitted_action, executor.loop._record_receipt = accept, record
    def fail(*args, **kwargs):
        raise RuntimeError("worker or publication connection lost")
    monkeypatch.setattr(tool_execution, "dispatch_action" if failure_stage == "worker" else "publish_observation", fail)
    with pytest.raises(WorldStateUnknown):
        with executor.invocation("compound", {"tool_call_id": "call", "model_call_id": "model"}, 20):
            executor.execute_action(action(0), 20)
    assert settle_inline_execution(executor.loop, None, 2)
    assert interaction.phase == "aborted" and interaction.terminal_reason == "world_state_unknown"
    assert interaction.world_state_certainty == "unknown" and interaction.model_calls == 2
    assert executor.loop.termination_authority.value == "infrastructure"
    assert executor.loop.termination == "world_state_unknown" and executor.loop.stop_session


def test_reused_provider_tool_id_in_new_model_call_is_fresh(tmp_path, monkeypatch):
    executor, state, action = fixture(tmp_path, monkeypatch, [2, 3])
    for index in range(2):
        with executor.invocation("compound", {"tool_call_id": "reused", "model_call_id": f"model-{index}"}, 20) as call:
            assert not call.recovered
            executor.execute_action(action(index), 20)
    assert state["effects"] == 2
    assert executor.completed["model_call_id"] == "model-1"
    with pytest.raises(ToolExecutionInterrupted):
        with executor.invocation("compound", {"tool_call_id": "reused", "model_call_id": "model-1"}, 20):
            pytest.fail("cached parent must not resume")
    assert state["effects"] == 2


@pytest.mark.parametrize("binding", [{"action_id": "previous", "event_id": "old", "receipt_ref": {"sha256": "a" * 64}}, None])
def test_missing_or_stale_delivered_receipt_cannot_close_intent(tmp_path, monkeypatch, binding):
    executor, state, action = fixture(tmp_path, monkeypatch, [2])
    executor.loop.planner.tool_receipt_binding = lambda action_id: binding
    with pytest.raises(WorldStateUnknown):
        with executor.invocation("compound", {"tool_call_id": "call", "model_call_id": "model"}, 20):
            executor.execute_action(action(0), 20)
    assert executor.blocked
    with pytest.raises(WorldStateUnknown):
        tool_execution.ToolExecutionLog(tmp_path, "call", "model").recover()


def test_dropped_delivery_in_second_parent_does_not_reuse_previous_parent_receipt(tmp_path, monkeypatch):
    from ora.agent.pi.tool_rpc import PiToolRpc
    executor, state, action = fixture(tmp_path, monkeypatch, [2, 2])
    planner = executor.loop.planner
    planner._call = lambda *args: None
    planner.begin_tool_action = PiToolRpc.begin_tool_action.__get__(planner)
    planner.tool_receipt_binding = PiToolRpc.tool_receipt_binding.__get__(planner)
    record = executor.loop._record_receipt
    def delivered(*args, **kwargs):
        record(*args, **kwargs)
        planner.pending_receipt = dict(state["ref"])
    executor.loop._record_receipt = delivered
    with executor.invocation("compound", {"tool_call_id": "first", "model_call_id": "model"}, 20):
        executor.execute_action(action(0), 20)
    assert planner.pending_receipt["action_id"] == "a-0"
    # The callback fails and the host keeps running, exactly as best-effort delivery does.
    executor.loop._record_receipt = record
    with pytest.raises(WorldStateUnknown):
        with executor.invocation("compound", {"tool_call_id": "second", "model_call_id": "model"}, 20):
            executor.execute_action(action(1), 20)
    assert planner.pending_receipt is None and executor.blocked
    log = tool_execution.ToolExecutionLog(tmp_path, "second", "model")
    with pytest.raises(WorldStateUnknown):
        log.recover()
    assert log.value["actions"] == [] and log.value["pending"]["action_id"] == "a-1"


@pytest.mark.parametrize("change", [None, "stale", "forged"])
def test_explicit_recovery_anchor_is_checked_before_physical_dispatch(tmp_path, monkeypatch, change):
    from copy import deepcopy
    executor, state, make_action = fixture(tmp_path, monkeypatch, [2])
    owner = executor.loop.execution_state_owner
    anchor = {"arm_joint_qpos": [0.1] * 6, "pose_xyz_xyzw": [0, 0, .3, 0, 0, 0, 1]}
    owner.writer.write("exec.recovery.left.home", anchor)
    action = make_action(0)
    action.update(action="recover_motion", arguments={"targets": {"left": "home"},
        "recovery_targets": {"left": {**deepcopy(anchor), "strategy": "home"}}})
    if change == "stale":
        owner.writer.write("exec.recovery.left.home", {**anchor, "arm_joint_qpos": [0.2] * 6})
    elif change == "forged":
        action["arguments"]["recovery_targets"]["left"]["arm_joint_qpos"][0] += .1
    with executor.invocation("recover_pose", {"tool_call_id": "recovery", "model_call_id": "m"}, 20):
        result = executor.execute_action(action, 20)
    assert state["effects"] == (1 if change is None else 0)
    assert result["receipt"]["native_actions"] == (2 if change is None else 0)
    if change is not None:
        assert result["receipt"]["reason"] == "stale_or_mismatched_recovery_target"
        assert result["remaining_budget"] == 20
        assert state["step"] == 0
        assert executor.completed["native_actions_used"] == 0
        assert not executor.blocked
        assert not (executor.loop.control / "submitted-action.json").exists()


def test_legacy_recovery_keeps_owner_lookup_and_identical_request(tmp_path, monkeypatch):
    executor, state, make_action = fixture(tmp_path, monkeypatch, [2])
    owner = executor.loop.execution_state_owner
    anchor = {"arm_joint_qpos": [0.1] * 6, "pose_xyz_xyzw": [0, 0, .3, 0, 0, 0, 1]}
    owner.writer.write("exec.recovery.left.previous", anchor)
    action = make_action(0)
    action.update(action="recover_motion", arguments={"targets": {"left": "backtrack_last_safe"}})
    assert owner.request("execute", {"action": action})["recovery_targets"] == {
        "left": {**anchor, "strategy": "backtrack_last_safe"}}
    with executor.invocation("recover_pose", {"tool_call_id": "recovery", "model_call_id": "m"}, 20):
        executor.execute_action(action, 20)
    assert state["effects"] == 1


@pytest.mark.parametrize("persisted", [False, True])
def test_begin_failure_proves_no_dispatch_even_if_partial_ledger_blocks_session(tmp_path, monkeypatch, persisted):
    from contextlib import nullcontext
    from ora.agent.toolkit.interface.types import ActionNotStarted
    executor, state, action = fixture(tmp_path, monkeypatch, [2])
    with (pytest.raises(WorldStateUnknown) if persisted else nullcontext()):
        with executor.invocation("set_gripper", {"tool_call_id": "begin-fail", "model_call_id": "m"}, 20):
            begin = executor.active.log.begin
            def fail(**kwargs):
                if persisted:
                    begin(**kwargs)
                raise OSError("intent storage failure")
            monkeypatch.setattr(executor.active.log, "begin", fail)
            with pytest.raises(ActionNotStarted) as caught:
                executor.execute_action(action(0), 20)
            assert caught.value.stage == "begin"
            assert state["effects"] == 0
    assert executor.blocked is persisted


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("crash", ["disconnect", "cached_result"])
def test_pi_current_parent_cannot_resume_even_when_ledger_closed(tmp_path, monkeypatch, count, crash):
    from ora.agent.pi.core_process import PiCoreProcess
    from ora.agent.pi.tool_rpc import PiToolRpc
    from ora.agent.pi.rpc import RpcError
    from ora.agent.contracts import PolicySessionError

    class Bridge(PiCoreProcess, PiToolRpc):
        pass

    executor, state, action = fixture(tmp_path, monkeypatch, [1] * count)
    context = {"tool_call_id": "current", "model_call_id": "model"}
    with executor.invocation("custom", context, 20):
        for index in range(count):
            executor.execute_action(action(index), 20)
    bridge = Bridge()
    bridge.bind_tool_action_executor(executor, observation_receipts=False)
    bridge._pending_tool_deliveries[("model", "current")] = context
    bridge._executed = {("tools.invoke", "model", "current"): {"result": {"ok": True}}}
    def lost(*args):
        raise RpcError("process died", code="closed")
    bridge.rpc = SimpleNamespace(call=lost)
    with pytest.raises(PolicySessionError):
        if crash == "disconnect":
            bridge._call("act", {})
        else:
            bridge._once(("tools.invoke", "model", "current"), lambda: pytest.fail("body repeated"))
    assert executor.loop.termination == "tool_execution_interrupted"
    assert executor.interruption["last_action_id"] == (f"a-{count - 1}" if count else None)
    assert executor.interruption["tool_call_id"] == "current"
    assert state["effects"] == count
    assert json.loads((tmp_path / "native/tool-execution-interruption.json").read_text()) == executor.interruption


@pytest.mark.parametrize("spent,refusal", [(False, False), (False, True), (True, True)])
def test_read_only_python_error_and_expected_refusals_do_not_terminate(tmp_path, monkeypatch, spent, refusal):
    from ora.agent.error_classes import ToolError
    executor, state, action = fixture(tmp_path, monkeypatch, [1] if spent else [])
    error = ToolError("normal refusal", code="invalid_arguments") if refusal else ValueError("read-only error")
    with pytest.raises(type(error)):
        with executor.invocation("custom", {"tool_call_id": "current", "model_call_id": "model"}, 20):
            if spent:
                executor.execute_action(action(0), 20)
            raise error
    assert not executor.blocked
    assert executor.interruption is None


def test_pi_ended_history_is_not_retroactively_interrupted(tmp_path, monkeypatch):
    from ora.agent.pi.tool_rpc import PiToolRpc
    executor, state, action = fixture(tmp_path, monkeypatch, [1])
    context = {"tool_call_id": "old", "model_call_id": "model"}
    with executor.invocation("custom", context, 20):
        executor.execute_action(action(0), 20)
    bridge = PiToolRpc()
    bridge.run_root = tmp_path
    bridge.bind_tool_action_executor(executor, observation_receipts=False)
    bridge._pending_tool_deliveries[("model", "old")] = context
    bridge._rpc_tools_settled({"calls": [context]})
    assert bridge._interrupt_pending_tool() is False
    assert not executor.blocked


@pytest.mark.parametrize("unknown", [False, True])
def test_interruption_result_and_status_retain_parent_identity(tmp_path, monkeypatch, unknown):
    import time
    from ora.orchestrator.episode.result import _fault_result_record, _publish_fault_result, _publish_session_result
    executor, state, action = fixture(tmp_path, monkeypatch, [1])
    context = {"tool_call_id": "parent", "model_call_id": "model"}
    with executor.invocation("move_to", context, 20):
        executor.execute_action(action(0), 20)
    executor.interrupt(context, unknown=unknown)
    loop = executor.loop
    loop.toolkit_executor = executor
    loop.multi_round = False
    loop.verdict = None
    loop.started = time.time()
    loop.started_monotonic_ns = time.monotonic_ns()
    loop.run_id = "run"
    loop.plan["plan_sha256"] = "0" * 64
    loop.decisions, loop.refused_proposals, loop.model_calls = 1, 0, 1
    loop.verdict_guard = SimpleNamespace(divergence=None)
    loop.auxiliary_failures, loop.round_results = [], []
    loop.status_path = tmp_path / "status.json"
    loop.events = tmp_path / "events.jsonl"
    loop.deliver_policy_event = lambda *a, **k: None
    loop.record_auxiliary_failure = lambda *a: pytest.fail(str(a))
    loop.interaction.observation_id = "obs-1"
    loop.planner.close = lambda: None
    result, official = _fault_result_record(loop, RuntimeError("Pi lost"))
    assert result["termination"] == executor.interruption["reason"]
    assert result["behavioral_sample_valid"] is False and official is None
    assert result["tool_execution_interruption"] == executor.interruption
    for publish in (
        lambda: _publish_fault_result(loop, result, RuntimeError("Pi lost"), preserved_official=None),
        lambda: _publish_session_result(loop, result, state="incomplete", official=None,
                                       session_termination=result["termination"],
                                       session_authority=loop.termination_authority),
    ):
        publish()
        for filename in ("result.json", "status.json"):
            record = json.loads((tmp_path / filename).read_text())
            assert record["termination"] == executor.interruption["reason"]
            assert record["tool_execution_interruption"] == executor.interruption


@pytest.mark.parametrize("spent", [False, True])
def test_process_exit_is_converted_to_reportable_terminal_error(tmp_path, monkeypatch, spent):
    executor, state, action = fixture(tmp_path, monkeypatch, [1] if spent else [])
    with pytest.raises(ToolExecutionInterrupted):
        with executor.invocation("custom", {"tool_call_id": "exit", "model_call_id": "model"}, 20):
            if spent:
                executor.execute_action(action(0), 20)
            raise SystemExit(1)
    assert executor.loop.termination == "tool_execution_interrupted"
    assert executor.interruption["last_action_id"] == ("a-0" if spent else None)


def test_finish_io_failure_with_durable_closed_actions_is_known_state(tmp_path, monkeypatch):
    executor, state, action = fixture(tmp_path, monkeypatch, [1])
    with pytest.raises(ToolExecutionInterrupted, match="not durable"):
        with executor.invocation("move_to", {"tool_call_id": "finish-io", "model_call_id": "model"}, 20) as call:
            executor.execute_action(action(0), 20)
            def fail():
                raise OSError("finish failed")
            monkeypatch.setattr(call.log, "_save", fail)
    assert executor.interruption["reason"] == "tool_execution_interrupted"
    assert executor.interruption["last_action_id"] == "a-0"


def test_terminal_execution_attempt_is_recorded_but_never_dispatched(tmp_path, monkeypatch):
    executor, state, action = fixture(tmp_path, monkeypatch, [])
    executor.loop.environment_status = {"terminal": True}
    with executor.invocation("move_pose", {"tool_call_id": "call", "model_call_id": "model"}, 20):
        result = executor.execute_action(action(0), 20)
    assert result["receipt"]["reason"] == "turn_must_end"
    assert executor.completed["actions"][0]["refusal"] == "turn_must_end"
    assert executor.completed["native_actions_used"] == 0
    assert state["effects"] == state["step"] == 0


def test_code_mode_on_a_counted_backend_without_execution_state_charges_a_long_action(tmp_path, monkeypatch):
    """RoboCasa365 counts native actions but has no Supervisor-owned state: one base move used 34."""
    base, state, action = fixture(tmp_path, monkeypatch, [])
    base.loop.execution_state_owner = None
    spends = [34, 6]

    def execute(loop, request, pointer, **kwargs):
        state["effects"] += 1
        return {"action_id": request["action_id"], "native_actions": spends.pop(0)}, None

    monkeypatch.setattr(tool_execution, "dispatch_action", execute)
    executor = CodeModeActionExecutor(base.loop)
    per_action = executor.native_actions_per_action()
    assert per_action == 100, "one action may use what the episode has left"
    with executor.invocation("bash", {"tool_call_id": "call", "model_call_id": "model"}, 12 * per_action):
        assert executor.execute_action(action(0), executor.native_actions_per_action())["receipt"][
            "native_actions"] == 34
        assert executor.execute_action(action(1), executor.native_actions_per_action())["remaining_budget"] == 60
    assert executor.completed["native_actions_used"] == 40
    assert not executor.blocked and state["effects"] == 2


def test_a_tool_call_interruption_records_the_exception_that_caused_it(tmp_path, monkeypatch):
    """A RoboDojo scene the official task ruled invalid: the record names the worker's error, not only the ledger
    completion that fails after it."""
    base, state, action = fixture(tmp_path, monkeypatch, [])

    def unstable(loop, request, pointer, **kwargs):
        raise RuntimeError("official environment became unstable")

    monkeypatch.setattr(tool_execution, "dispatch_action", unstable)
    with pytest.raises(WorldStateUnknown, match="completion is not durable"):
        with base.invocation("wait_ticks", {"tool_call_id": "call", "model_call_id": "model"}, 20):
            base.execute_action(action(0), 20)
    record = json.loads((tmp_path / "native/tool-execution-interruption.json").read_text())
    assert record["cause"] == {"type": "RuntimeError", "stage": "action",
                               "message": "official environment became unstable"}


def test_a_code_mode_interruption_records_the_exception_that_caused_it(tmp_path, monkeypatch):
    """RoboCasa365's worker stepped once past what the episode had left: the record names that ValueError, not only
    the ledger completion that fails after it ("execution ledger completion is not durable")."""
    base, state, action = fixture(tmp_path, monkeypatch, [])
    base.loop.execution_state_owner = None

    def overrun(loop, request, pointer, **kwargs):
        state["effects"] += 1
        return {"action_id": request["action_id"], "native_actions": 101}, None

    monkeypatch.setattr(tool_execution, "dispatch_action", overrun)
    executor = CodeModeActionExecutor(base.loop)
    per_action = executor.native_actions_per_action()
    with pytest.raises(WorldStateUnknown, match="completion is not durable"):
        with executor.invocation("bash", {"tool_call_id": "call", "model_call_id": "model"}, 12 * per_action):
            executor.execute_action(action(0), per_action)
    record = json.loads((tmp_path / "native/tool-execution-interruption.json").read_text())
    assert record == executor.interruption
    assert record["reason"] == "world_state_unknown" and record["last_action_id"] == "a-0"
    assert record["cause"] == {"type": "ValueError", "stage": "action",
                               "message": "worker consumption exceeds the admitted action budget"}
    assert executor.blocked and state["effects"] == 1


def test_code_mode_on_a_worker_limited_backend_keeps_twenty_per_action(tmp_path, monkeypatch):
    base, _, _ = fixture(tmp_path, monkeypatch, [])
    executor = CodeModeActionExecutor(base.loop)
    assert executor.native_actions_per_action() == 20
    assert executor._action_cap({"action": "move_pose", "arguments": {}}, 100) == 20
    # A joint trajectory (RoboDojo's smooth move_pose) is capped by its own contract, as in tool-call mode.
    assert executor._action_cap({"action": "follow_joint_trajectory", "arguments": {}}, 100) == 250


def _trajectory(observation_id, knots):
    """One ``follow_joint_trajectory`` of ``knots`` knots (toolkit v4/v5's move_pose), bound to the observation."""
    from ora.agent.toolkit.lib.motion_plan.execute import trajectory_action

    return trajectory_action(action_id="move-0", observation={"observation_id": observation_id,
                                                              "observation_sha256": "a" * 64},
                             knots={"left": [[0.01 * index, 0, 0, 0, 0, 0] for index in range(knots)]})


@pytest.mark.parametrize("code_mode", [False, True])
def test_a_joint_trajectory_is_lent_its_own_cap_not_one_primitive_s(tmp_path, monkeypatch, code_mode):
    """A 50-knot move asks for 50 and the worker is lent 50 (``native_action_cap``), in tool-call mode and in a
    code-mode script on RoboDojo alike; one primitive's 20 would cut the move after 20 knots."""
    base, state, _ = fixture(tmp_path, monkeypatch, [50])
    lent = []
    owner = base.loop.execution_state_owner
    dispatch = tool_execution.dispatch_action

    def execute(loop, action, pointer, **kwargs):
        lent.append(owner.action_budget)
        return dispatch(loop, action, pointer, **kwargs)

    monkeypatch.setattr(tool_execution, "dispatch_action", execute)
    executor = CodeModeActionExecutor(base.loop) if code_mode else base
    name, requested = ("bash", 12 * executor.native_actions_per_action()) if code_mode else ("move_pose", 50)
    with executor.invocation(name, {"tool_call_id": "call", "model_call_id": "model"}, requested):
        result = executor.execute_action(_trajectory("obs-0", 50), 50)
    assert lent == [50]
    assert result["receipt"]["native_actions"] == 50 and executor.completed["native_actions_used"] == 50
    assert not executor.blocked and "abort" not in state


def test_code_mode_on_a_backend_without_execution_state_runs_the_action_its_executor_names():
    """LIBERO, RoboTwin, RoboCasa365 and RoboLab bind no execution state: the stand-in takes the tool call the
    executor lends with every action (the worker's undo key on RoboDojo) and ignores it."""
    from ora.orchestrator.episode.code_mode_execution import _NoExecutionState

    with _NoExecutionState().limited_action(20, {"tool_call_id": "call#001", "tool": "move_pose"}):
        pass


# Code mode runs a committed skill sequence (RoboTwin's closed-skill grasp_execute) as one script action.

def _sequence(observation_id, stages=("open", "close", "verify")):
    return {"schema": "ora-skill-sequence-action.v1", "action": "execute_skill_sequence", "action_id": "action-7",
            "observation_id": observation_id, "bound_observation_sha256": "a" * 64,
            "arguments": {"skill": "grasp_execute", "candidate_id": "cand-1", "rationale": "grasp the cube",
                          "stop_before_commit": False, "sequence": list(stages), "resume_from": 0}}


def _uncounted_loop(executor, state, tmp_path, *, owner=None):
    """A RoboTwin-like loop: no native-action limit, no Supervisor-owned execution state, real stage loop."""

    loop = executor.loop
    loop.execution_state_owner = owner
    loop.native_allowance = SimpleNamespace(remaining=lambda: None, fields=lambda: {})
    source = tmp_path / "obs"
    source.mkdir(exist_ok=True)
    def pointer():
        current = json.loads((loop.control / "current-observation.json").read_text())
        (source / "observation.json").write_text(json.dumps({"observation_id": current["observation_id"]}))
        return {**current, "source_root": str(source), "observation_manifest": str(source / "observation.json")}
    loop._accept_submitted_action = lambda calls: (json.loads((loop.control / "submitted-action.json").read_text()),
                                                   pointer())
    recorded = []
    def record(action, raw, duration, **kwargs):
        recorded.append((raw, kwargs))
        state["ref"] = {"action_id": raw["action_id"], "event_id": "receipt-" + raw["action_id"],
                        "receipt_ref": {"sha256": "b" * 64}}
    loop._record_receipt = record
    executed = []
    def execute(stage_action):
        executed.append(stage_action["action_id"])
        return {"schema": "ora.robotwin-gripper-receipt.test", "action_id": stage_action["action_id"]}
    running = {"schema": "ora.environment-status.v1", "phase": "running", "terminal": False, "reason": None,
               "source": "test", "policy_visibility": "control_only", "signals": []}
    loop.runtime = SimpleNamespace(execute=execute, status=lambda: dict(running),
                                   observe=lambda root: ({"observation_id": "captured"}, []),
                                   abort=lambda reason: state.update(abort=reason))
    ledger = {}
    def drive(action, *, stage, stage_index, observation, observation_root, completed, capture_observation):
        if stage == "verify":
            capture_observation()
            return {"kind": "measurement", "status": "executed", "reason": "verified", "measurements": []}
        return {"kind": "mutation", "action": {
            "schema": "ora-motion-action.v3", "observation_id": action["observation_id"],
            "action_id": f"{action['action_id']}:{stage_index:02d}-{stage}", "action": "set_gripper",
            "arguments": {"targets": {"left": stage}}, "bound_observation_sha256": action["bound_observation_sha256"]}}
    def receipt(**kwargs):
        ledger["released"] = kwargs["status"]
        return {"status": kwargs["status"]}
    loop.tools = SimpleNamespace(drive_skill_stage=drive, build_skill_receipt=receipt)
    loop.observation_root = tmp_path / "unpublished"
    loop.projector = SimpleNamespace(main_camera="head")
    loop.events = tmp_path / "events.jsonl"
    return recorded, executed, ledger


def test_a_script_s_skill_sequence_runs_its_stage_loop_as_one_action(tmp_path, monkeypatch):
    base, state, _ = fixture(tmp_path, monkeypatch, [])
    monkeypatch.undo()  # the real dispatch_action and its stage loop; publication stays faked
    monkeypatch.setattr(tool_execution, "publish_observation", lambda loop, *a, **k: (
        state.update(step=state["step"] + 1),
        setattr(loop.interaction, "observation_step", state["step"]),
        (loop.control / "current-observation.json").write_text(json.dumps(
            {"observation_id": f"obs-{state['step']}", "observation_sha256": "a" * 64})),
        ({}, {"terminal": False}))[-1])
    executor = CodeModeActionExecutor(base.loop)
    recorded, executed, ledger = _uncounted_loop(executor, state, tmp_path)
    assert executor.native_actions_per_action() == 1
    with executor.invocation("bash", {"tool_call_id": "call", "model_call_id": "model"}, 12):
        result = executor.execute_action(_sequence("obs-0"), executor.native_actions_per_action())
    assert executed == ["action-7:00-open", "action-7:01-close"], "every committed stage reached the world"
    assert ledger["released"] == "completed", "the skill ledger reported and released"
    (raw, accounting), = recorded
    assert raw["schema"] == "ora.manipulation-skill-sequence-receipt.v1"
    assert accounting == {"inline_consumed": True, "inline_used": 1}, "the parent's first spend: the script's decision"
    assert result["receipt"]["action_id"] == "action-7" and result["receipt"]["native_actions"] == 1
    assert [item["action_id"] for item in executor.completed["actions"]] == ["action-7"]
    assert executor.completed["native_actions_used"] == 1 and not executor.blocked


def test_a_skill_sequence_that_dispatched_no_stage_still_counts_one_action(tmp_path, monkeypatch):
    base, state, _ = fixture(tmp_path, monkeypatch, [])
    executor = CodeModeActionExecutor(base.loop)
    _uncounted_loop(executor, state, tmp_path)
    receipt = {"schema": "ora.manipulation-skill-sequence-receipt.v1", "action_id": "action-7",
               "stage_receipts": [{"stage": "open", "status": "refused", "action_id": None}]}
    monkeypatch.setattr(tool_execution, "dispatch_action", lambda loop, action, pointer, **k: (receipt, None))
    with executor.invocation("bash", {"tool_call_id": "call", "model_call_id": "model"}, 12):
        executor.execute_action(_sequence("obs-0"), 1)
    assert executor.completed["native_actions_used"] == 1, "the turn ends on it, as the tool-call submission does"
    # Only a skill sequence's receipt is charged one regardless; another receipt that counts keeps its count.
    assert executor._native_used({"action_id": "a-0", "native_actions": 3}) == 3
    assert executor._native_used({"action_id": "a-0"}) == 1


def test_a_grasp_after_another_action_is_one_more_script_action_and_no_further_decision(tmp_path, monkeypatch):
    """The script is one robot decision, charged at its parent's first spend; a grasp is one of its actions."""
    base, state, action = fixture(tmp_path, monkeypatch, [])
    executor = CodeModeActionExecutor(base.loop)
    recorded, _, _ = _uncounted_loop(executor, state, tmp_path)

    def dispatch(loop, request, pointer, **kwargs):
        if request["schema"] == "ora-skill-sequence-action.v1":
            return {"schema": "ora.manipulation-skill-sequence-receipt.v1", "action_id": request["action_id"],
                    "stage_receipts": []}, None
        return {"action_id": request["action_id"]}, None

    monkeypatch.setattr(tool_execution, "dispatch_action", dispatch)
    with executor.invocation("bash", {"tool_call_id": "call", "model_call_id": "model"}, 12):
        executor.execute_action(action(0), 1)
        executor.execute_action(_sequence("obs-1"), 1)
    assert [accounting["inline_consumed"] for _, accounting in recorded] == [True, False]
    assert [item["action_id"] for item in executor.completed["actions"]] == ["a-0", "action-7"]
    assert executor.completed["native_actions_used"] == 2


@pytest.mark.parametrize("code_mode", [False, True])
def test_a_worker_capped_backend_or_the_plain_executor_never_starts_a_skill_sequence(tmp_path, monkeypatch, code_mode):
    """RoboDojo's worker caps one action at a primitive's budget; the plain toolkit executor takes primitives only."""
    base, state, _ = fixture(tmp_path, monkeypatch, [])
    executor = CodeModeActionExecutor(base.loop) if code_mode else base
    if not code_mode:
        base.loop.execution_state_owner = None
    with executor.invocation("bash", {"tool_call_id": "call", "model_call_id": "model"}, 12):
        with pytest.raises(ActionNotStarted, match="not a composite"):
            executor.execute_action(_sequence("obs-0"), executor.native_actions_per_action())
    assert state["effects"] == 0 and executor.completed["actions"] == [] and not executor.blocked


def test_a_tool_inside_a_script_reads_what_the_script_has_left(tmp_path, monkeypatch):
    """A tool a code-mode script calls runs inside the script's parent and may spend only what that parent has
    left; ``nested_allowance`` tells it, so a motion tool plans no more than will be admitted. A tool that is its
    own parent (tool-call mode) reads none: its own budget is what it plans with."""
    base, _, action = fixture(tmp_path, monkeypatch, [7])
    executor = CodeModeActionExecutor(base.loop)
    assert executor.nested_allowance() is None
    with executor.invocation("bash", {"tool_call_id": "call", "model_call_id": "model"}, 30) as parent:
        assert executor.nested_allowance() is None  # not reentrant: no script runs inside it
        parent.reentrant = True
        assert executor.nested_allowance() == 30
        with executor.invocation("wait_ticks", {"tool_call_id": "call#001", "model_call_id": "model"}, 20):
            assert executor.nested_allowance() == 30
            executor.execute_action(action(0), 20)
            assert executor.nested_allowance() == 23
    assert executor.nested_allowance() is None


class _Surface:
    """A resolved surface as the executor reads it (``loop.tools.surface``)."""

    def __init__(self, profile_id, capabilities):
        self.profile_id, self.capabilities = profile_id, frozenset(capabilities)

    def supports(self, capability):
        return capability in self.capabilities


TRAJECTORY_CAPABILITIES = ("toolkit_synchronous_execution_v1", "robodojo_sdk_ik_joint_interpolation_v1_non_parity")


@pytest.mark.parametrize("profile,capabilities,per_action", [
    ("robodojo_placement_toolkit_v5", (), 20),  # the released primitives, 20 each
    ("robodojo_placement_toolkit_v5", TRAJECTORY_CAPABILITIES, 250),  # one joint trajectory per move: up to 250
])
def test_a_script_on_robodojo_is_sized_by_the_cap_of_the_actions_its_surface_runs(tmp_path, monkeypatch, profile,
                                                                                    capabilities, per_action):
    base, _, _ = fixture(tmp_path, monkeypatch, [])
    base.loop.tools = SimpleNamespace(surface=_Surface(profile, capabilities))
    assert CodeModeActionExecutor(base.loop).native_actions_per_action() == per_action


def test_each_action_names_its_own_tool_call_and_the_decision_it_belongs_to(tmp_path, monkeypatch):
    """The worker keeps one recover_pose undo entry per arm-moving tool call (the call it is lent). A tool call's
    actions name that call; a code-mode script's actions name the script's call of each tool (``<call>#NNN``),
    so undo retraces one of its moves, not the whole script. Every action carries the ordinal of the robot
    decision its parent spends: a later action of a counted parent keeps it."""
    from ora.environment.backends.robodojo.trajectory import record_undo

    base, _, action = fixture(tmp_path, monkeypatch, [3, 4, 5, 6])
    owner = base.loop.execution_state_owner
    lent = []
    dispatch, record = tool_execution.dispatch_action, base.loop._record_receipt

    def execute(loop, request, pointer, **kwargs):
        lent.append(dict(owner.tool_call))
        return dispatch(loop, request, pointer, **kwargs)

    def counted(request, raw, duration, **kwargs):  # receipt_accounting: a parent's first spend is its decision
        record(request, raw, duration, **kwargs)
        if kwargs["inline_consumed"]:
            owner.finish_tool_call(raw)

    monkeypatch.setattr(tool_execution, "dispatch_action", execute)
    base.loop._record_receipt = counted
    with base.invocation("grasp_execute", {"tool_call_id": "call-1", "model_call_id": "model"}, 40):
        base.execute_action(action(0), 20)
        base.execute_action(action(1), 20)
    executor = CodeModeActionExecutor(base.loop)
    with executor.invocation("bash", {"tool_call_id": "call-2", "model_call_id": "model"}, 40) as parent:
        parent.reentrant = True
        for seq in (1, 2):
            context = {"tool_call_id": f"call-2#{seq:03d}", "model_call_id": "model"}
            with executor.invocation("move_pose", context, 50):
                executor.execute_action(action(1 + seq), 20)
    assert lent == [
        {"tool_call_id": "call-1", "tool": "grasp_execute", "ordinal": 1},
        {"tool_call_id": "call-1", "tool": "grasp_execute", "ordinal": 1},
        {"tool_call_id": "call-2#001", "tool": "move_pose", "ordinal": 2},
        {"tool_call_id": "call-2#002", "tool": "move_pose", "ordinal": 2},
    ]
    # The worker's undo stack: the script's two moves are two entries, the latest undone first.
    stacks = {"left": []}
    before = {"left": {"arm_joint_qpos": [0.0] * 6, "pose_xyz_xyzw": [0.0] * 6 + [1.0]}}
    assert [record_undo(stacks, call, before, ["left"], [])["arms"]["left"] for call in lent[2:]] == ["push", "push"]
    assert [entry["tool_call_id"] for entry in stacks["left"]] == ["call-2#001", "call-2#002"]
