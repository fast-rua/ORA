"""Code mode on RoboDojo's synchronous toolkit surface: a script's robot calls reach the Supervisor's executor.

For the code-mode RoboDojo plan the coordinator composes
``CodeModeToolRuntime(RecoverableArtifactToolRuntime(ManipulationToolRuntime))`` (lossless retention adds the
Artifact reader, code mode its tools), and toolkit Dispatch reads ``toolkit_executor`` on the innermost runtime,
the one ``build_tool_runtime`` installed it on. No simulator, provider or agent core: the host is the recorded
general_pickup fixture, the plan the resolved code-mode RunPlan, the executor the real
``CodeModeActionExecutor`` (admission, the parent's ledger, receipt binding, the per-primitive cap) with only the
worker dispatch and the publication faked, as ``test_tool_execution.py`` fakes them, and the script's parent
invocation is opened as ``PiCodeMode`` opens it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ora.agent.code_mode.runtime import build_code_mode_runtime
from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.context.observation.contact_points import project_policy_public_facts
from ora.agent.memory.tool_runtime import build_recoverable_artifact_runtime
from ora.agent.evidence.canonical import ArtifactStore
from ora.agent.evidence.trace_verify import verify_agent_trace
from ora.agent.pi.code_mode import PiCodeMode
from ora.agent.pi.robot_socket import PiRobotSocket
from ora.agent.pi.tool_rpc import PiToolRpc
from ora.agent.session_core.gateway import NativeToolGateway
from ora.agent.toolkit.interface import dry_run
from ora.agent.toolkit.interface.types import canonical
from ora.agent.toolkit.tools.errors import ToolRefusal
from ora.agent.toolkit.host.runtime import dispatch as toolkit_dispatch
from tests.support.agent.toolkit_tools import install
from ora.agent.error_classes import ToolError
from ora.orchestrator.episode import tool_execution
from ora.orchestrator.episode.code_mode_execution import CodeModeActionExecutor
from ora.orchestrator.episode.execution_state import ExecutionStateOwner
from ora.orchestrator.episode.script_actions import bind_script_actions
from ora.orchestrator.episode.tool_execution_log import ToolExecutionLog
from ora.orchestrator.spec.run_spec import resolve_run_spec
from tests.support.agent.agent_events import TrajectoryRecorder
from tests.support.agent.paced_kinematics import paced
from tests.support.agent.toolkit_atomic import complete_cameras
from tests.support.agent.toolkit_tools import make_host
from tests.support.orchestrator.run_specs import robodojo_native_spec


def _proposed(host, tool_call_id: str, name: str, arguments: dict, seq: int) -> dict:
    """The ``tool_call_proposed`` row the Pi session writes before it invokes a call; its call context."""

    row = dict(type="tool_call_proposed", seq=seq, run_id="run", tool_call_id=tool_call_id,
               model_call_id="model-0", payload=dict(name=name, arguments=arguments))
    with (host.control_root.parent / "agent-events.jsonl").open("ab") as handle:
        handle.write(canonical(row) + b"\n")
    return {"tool_call_id": tool_call_id, "model_call_id": "model-0"}


DEFAULT_SURFACE = "robodojo_placement_toolkit_v6"


def code_mode_episode(tmp_path, monkeypatch, episode=1000, code_mode=True):
    """The code-mode RoboDojo tool stack, bound by ``bind_script_actions`` to a loop whose worker spends 3.

    The surface is toolkit v6, which runs one joint trajectory per move (RoboDojo's default baseline
    ``robodojo_toolkit_v7``; the released primitives' v2, the
    code-mode comparison's surface, was retired on 2026-11-01). ``episode`` is the episode's native-action limit,
    ``code_mode`` the plan's ``policy.code_mode``."""

    spec = robodojo_native_spec("general_pickup")
    spec["policy"]["code_mode"] = code_mode
    plan = resolve_run_spec(spec)
    assert plan["policy"]["tool_surface"]["profile"] == DEFAULT_SURFACE
    policy = plan["policy"]

    host, client, _ = make_host(tmp_path, monkeypatch)
    complete_cameras(host)  # the runtime build_tool_runtime makes for this plan, Dispatch installed
    host.public_facts_projector = project_policy_public_facts
    host.toolkit_grants = plan.get("toolkit_grants")
    host.surface = resolve_tool_surface(policy["tool_surface"]["profile"],
                                        toolkit_grants=plan.get("toolkit_grants"),
                                        toolkit_receipt_delivery=plan.get("toolkit_receipt_delivery"))
    host.allowed_tools = tuple(policy["tool_allowlist"])
    install(host)
    # coordinator._compose_tool_runtime's wrappers, in its order; the plan advertises what they do.
    tools = build_recoverable_artifact_runtime(plan, run_root=tmp_path, event_path=tmp_path / "events.jsonl",
                                               base_runtime=host)
    tools = build_code_mode_runtime(plan, base_runtime=tools)
    assert [d["name"] for d in tools.definitions()] == [
        d["name"] for d in plan["policy_composition"]["tool_surface"]["definitions"]]

    control = host.control_root
    state = {"used": 0, "effects": 0, "ref": None, "counted": [], "recorded": [], "budgets": [], "receipts": 0,
             "tool_calls": []}

    def current():
        return json.loads((control / "current-observation.json").read_text())

    def receipt(raw):  # the policy session records each receipt as an event of its own
        state["receipts"] += 1
        state["ref"] = {"action_id": raw["action_id"], "event_id": f"receipt-{state['receipts']}",
                        "receipt_ref": {"sha256": "b" * 64}}

    def dispatch(loop, action, pointer, **kwargs):  # the worker: three native actions under the lent cap
        state["effects"] += 1
        state["budgets"].append(loop.execution_state_owner.action_budget)
        state["tool_calls"].append(loop.execution_state_owner.tool_call)
        return {"action_id": action["action_id"], "native_actions": 3, "status": "reached",
                "stage_evidence": {"consumed": True}}, None

    def publish(loop, *args, **kwargs):  # the same observation bytes under a new publication generation
        loop.interaction.observation_step += 1
        pointer = current()
        pointer["generation"] = int(pointer.get("generation", 0)) + 1
        (control / "current-observation.json").write_bytes(canonical(pointer))
        return pointer, {"terminal": False}

    def record(action, raw, duration, **kwargs):
        state["counted"].append(kwargs["inline_consumed"])
        state["used"] += raw["native_actions"]
        receipt(raw)

    session = PiCodeMode()
    planner = SimpleNamespace(
        bind_tool_action_executor=lambda executor, observation_receipts: setattr(
            session, "tool_action_executor", executor),
        record_tool_action=lambda action, context, before, name: state["recorded"].append(
            (context["tool_call_id"], action["action_id"])),
        record_environment_receipt=receipt, tool_receipt_binding=lambda action_id: state["ref"],
        begin_tool_action=lambda action_id: state.update(ref=None), project_tool_receipt=lambda raw: raw)
    loop = SimpleNamespace(
        plan=plan, tools=tools, run_root=tmp_path, control=control, termination=None, interrupted=False,
        environment_status={"terminal": False}, round_model_calls=0,
        interaction=SimpleNamespace(phase="policy_active", observation_step=0, begin_policy=lambda: None),
        native_allowance=SimpleNamespace(remaining=lambda: episode - state["used"],
                                          fields=lambda: {"native_actions_remaining": episode - state["used"]}),
        execution_state_owner=ExecutionStateOwner(),  # RoboDojo's: the worker caps one primitive
        planner=planner, _exhausted_budget=lambda: None, _should_stop=lambda: False,
        _accept_submitted_action=lambda calls: (json.loads((control / "submitted-action.json").read_text()),
                                                current()),
        _record_receipt=record, record_termination_decision=lambda *args, **kwargs: None,
        runtime=SimpleNamespace(abort=lambda reason: state.update(abort=reason)))
    monkeypatch.setattr(tool_execution, "dispatch_action", dispatch)
    monkeypatch.setattr(tool_execution, "publish_observation", publish)
    bind_script_actions(loop)
    return SimpleNamespace(host=host, client=client, tools=tools, loop=loop, session=session, state=state)


def _move(host) -> dict:
    """A move_to one centimetre above the left end effector's recorded position."""

    _, _, manifest = host._current()
    position = manifest["proprioception"]["end_effectors"]["left"]["pose_xyz_xyzw"][:3]
    position[2] += .01
    return {"targets": {"left": {"position_m": position}}}


def test_a_code_mode_script_runs_robodojo_toolkit_tools_on_the_supervisors_executor(tmp_path, monkeypatch):
    episode = code_mode_episode(tmp_path, monkeypatch)
    host, tools, session, state = episode.host, episode.tools, episode.session, episode.state
    _kinematics(host)
    try:
        executor = episode.loop.toolkit_executor
        assert isinstance(executor, CodeModeActionExecutor)
        assert session.tool_action_executor is executor
        move = _move(host)
        parent = _proposed(host, "call-0", "bash", {"command": "python step.py"}, seq=2)
        with session._code_mode_parent("bash", parent, cap=2) as invocation:
            assert invocation.budget.limit == 500  # two actions, a trajectory's cap each
            observed = tools.invoke_with_context(
                "observe", {}, call_context=_proposed(host, "call-0#001", "observe", {}, seq=3))
            moved = tools.invoke_with_context(
                "move_to", move, call_context=_proposed(host, "call-0#002", "move_to", move, seq=4))
            waited = tools.invoke_with_context(
                "wait_ticks", {"ticks": 5},
                call_context=_proposed(host, "call-0#003", "wait_ticks", {"ticks": 5}, seq=5))
        assert observed.get("ok", True) is not False, observed
        assert observed.get("observation_id") or observed.get("views"), observed
        assert moved["status"] == "executed", moved
        assert waited["status"] == "executed", waited
        assert state["effects"] == 2 and all(0 < budget <= 20 for budget in state["budgets"])
        ledger = executor.completed
        assert ledger["tool_call_id"] == "call-0" and ledger["native_actions_used"] == 6
        assert [item["refusal"] for item in ledger["actions"]] == [None, None]
        assert state["counted"] == [True, False]  # one robot decision for the script's parent call
        # Both primitives are recorded under the script's own call, never its nested calls.
        assert [call for call, _ in state["recorded"]] == ["call-0", "call-0"]
        assert [action for _, action in state["recorded"]] == [item["action_id"] for item in ledger["actions"]]
        assert "abort" not in state
        assert host.toolkit_executor is executor  # where Dispatch reads it, under both wrappers
    finally:
        episode.client.close()


def test_an_action_refused_inside_a_script_leaves_the_next_action_its_identity(tmp_path, monkeypatch):
    """A refused action ran nothing and published nothing: the script's next action on the same observation
    carries the same identity, and the parent's ledger records both."""

    episode = code_mode_episode(tmp_path, monkeypatch)
    host, tools, session, state = episode.host, episode.tools, episode.session, episode.state
    _kinematics(host)
    try:
        executor = episode.loop.toolkit_executor
        refusals = ["stale_or_mismatched_recovery_target"]  # the owner refuses the script's first action
        monkeypatch.setattr(episode.loop.execution_state_owner, "recovery_refusal",
                            lambda action: refusals.pop(0) if refusals else None)
        move = _move(host)
        parent = _proposed(host, "call-0", "bash", {"command": "python step.py"}, seq=2)
        with session._code_mode_parent("bash", parent, cap=2):
            moved = tools.invoke_with_context(
                "move_to", move, call_context=_proposed(host, "call-0#001", "move_to", move, seq=3))
            waited = tools.invoke_with_context(
                "wait_ticks", {"ticks": 5},
                call_context=_proposed(host, "call-0#002", "wait_ticks", {"ticks": 5}, seq=4))
        assert moved["receipt"]["status"] == "refused", moved
        assert moved["receipt"]["reason"] == "stale_or_mismatched_recovery_target"
        assert waited["status"] == "executed" and waited["receipt"]["native_actions"] == 3, waited
        assert state["effects"] == 1
        ledger = executor.completed
        refused, executed = ledger["actions"]
        assert refused["action_id"] == executed["action_id"]  # one observation, one identity
        assert (refused["refusal"], refused["native_actions_used"]) == ("stale_or_mismatched_recovery_target", 0)
        assert (executed["refusal"], executed["native_actions_used"]) == (None, 3)
        assert refused["after_observation"] == executed["before_observation"] == ledger["initial_observation"]
        assert ledger["native_actions_used"] == 3
        assert state["counted"] == [True]  # the refusal spent no robot decision
        assert state["recorded"] == [("call-0", refused["action_id"])] * 2
        assert "abort" not in state and not executor.blocked
        # The durable ledger is valid evidence: recovery reads it back unchanged.
        assert ToolExecutionLog(tmp_path, "call-0", "model-0").recover() == ledger
    finally:
        episode.client.close()


def _inputs(host, tool_call_id: str) -> dict:
    """The tool context Dispatch froze for one call (``native/toolkit-io/tool-inputs.jsonl``)."""

    rows = [json.loads(line) for line in (host.control_root.parent / "native/toolkit-io/tool-inputs.jsonl")
            .read_text().splitlines()]
    return next(row["configuration"] for row in rows if row["tool_call_id"] == tool_call_id)


def test_a_script_s_context_is_a_tool_call_s_plus_the_script_s_allowance(tmp_path, monkeypatch):
    """A script's call records the tool context a tool call records on the same observation, byte for byte, plus
    what the script has left (``script_native_actions_remaining``), which a joint-trajectory move plans against."""

    def stop(api, arguments):  # the tool stops before it spends: both calls see one observation and allowance
        raise ToolRefusal("fixture stop", code="fixture")

    episode = code_mode_episode(tmp_path, monkeypatch)
    host, tools, session = episode.host, episode.tools, episode.session
    monkeypatch.setitem(toolkit_dispatch.TOOLS, "wait_ticks", stop)
    wait = {"ticks": 5}
    try:
        with session._code_mode_parent("bash", _proposed(host, "call-0", "bash", {}, seq=2), cap=2) as invocation:
            with pytest.raises(ToolError, match="fixture stop"):
                tools.invoke_with_context(
                    "wait_ticks", wait, call_context=_proposed(host, "call-0#001", "wait_ticks", wait, seq=3))
        with pytest.raises(ToolError, match="fixture stop"):
            tools.invoke_with_context(
                "wait_ticks", wait, call_context=_proposed(host, "call-1", "wait_ticks", wait, seq=4))
        script, direct = _inputs(host, "call-0#001"), _inputs(host, "call-1")
        assert "script_native_actions_remaining" not in direct
        assert script.pop("script_native_actions_remaining") == invocation.budget.limit  # nothing spent yet
        assert canonical(script) == canonical(direct)
    finally:
        episode.client.close()


def test_on_toolkit_v5_a_script_s_move_plans_against_what_the_script_has_left(tmp_path, monkeypatch):
    """On v5 a move is one joint trajectory fitted to what the call can spend. Inside a script that is also what
    the script's parent has left: the move's plan gets it (``script_native_actions_remaining``), so the parent's
    admission never stops the move mid-path. A tool-call context does not carry it."""
    from ora.agent.toolkit.lib.motion_plan import checks
    from ora.agent.toolkit.tools.move_pose import _smooth, tool

    episode = code_mode_episode(tmp_path, monkeypatch)
    host, tools, session, state = episode.host, episode.tools, episode.session, episode.state
    planned = []

    def plan(api, **kwargs):  # the planner stops here: what matters is what it was asked to fit
        planned.append(kwargs["native_action_budget"])
        raise checks.PlanRefusal("fixture stop", code="fixture")

    monkeypatch.setattr(_smooth.move, "plan", plan)
    monkeypatch.setattr(tool, "compile_action", lambda api, arguments, **kwargs: ({
        "action_id": "action-0", "arguments": {"targets": {"left": {"position_m": [0.1, 0.0, 0.9],
                                                                    "orientation_xyzw": [0, 0, 0, 1]}},
                                               "interactions": {"left": "free_space"}}}, {"targets": {}}))
    try:
        executor = episode.loop.toolkit_executor
        assert executor.native_actions_per_action() == 250  # a script lends each action a trajectory's cap
        with session._code_mode_parent("bash", _proposed(host, "call-0", "bash", {}, seq=2), cap=2) as invocation:
            assert invocation.budget.limit == 500
        move = {"targets": {"left": {"position_mode": "absolute", "position_m": [0.1, 0.0, 0.9],
                                     "orientation_mode": "hold_current"}}}
        parent = _proposed(host, "call-1", "bash", {"command": "python step.py"}, seq=3)
        with executor.invocation("bash", parent, 40) as invocation:  # a script with 40 native actions left
            invocation.reentrant = True
            waited = tools.invoke_with_context(
                "wait_ticks", {"ticks": 5}, call_context=_proposed(host, "call-1#001", "wait_ticks", {}, seq=4))
            with pytest.raises(ToolError, match="fixture stop"):
                tools.invoke_with_context(
                    "move_pose", move, call_context=_proposed(host, "call-1#002", "move_pose", move, seq=5))
        assert waited["status"] == "executed", waited
        assert planned == [37]  # 40 - 3: under the move's own 250 and the episode's 997
        context = _inputs(host, "call-1#002")
        assert context["script_native_actions_remaining"] == 37
        assert context["episode_native_actions_remaining"] == 997
        # A tool-call move_pose plans against its own 250: its context carries no script allowance.
        direct = _proposed(host, "call-2", "move_pose", move, seq=6)
        with pytest.raises(ToolError, match="fixture stop"):
            tools.invoke_with_context("move_pose", move, call_context=direct)
        assert planned[-1] == 250 and "script_native_actions_remaining" not in _inputs(host, "call-2")
    finally:
        episode.client.close()


# --- dry_run on the episode path: the real toolkit dispatch, the Supervisor's executor and a fake worker -------------
# A script's ``move_to`` asked for a dry run plans against a fake read-only kinematics service and executes nothing: no
# action reaches the worker, the parent's ledger holds no action, no ``action_proposed`` is recorded for it and no
# environment receipt exists, whatever the tool tries; the same call without the flag executes, and the plan the dry run
# returned is the plan it executed.

SCHEMA = "ora.kinematics-request.v1"


def _kinematics(host):
    """The backend's read-only kinematics query, as a toy: joints 0-2 follow the link6 position from the measured ones."""

    reach, effectors = host.align_plan_callback, host._current()[2]["proprioception"]["end_effectors"]
    asked = []

    def callback(request):
        if request.get("schema") != SCHEMA:
            return reach(request)
        asked.append(request["operation"])
        if request["operation"] == "screen_path":
            count = len(next(iter(request["knots_rad"].values())))
            return {"schema": "ora.kinematics-result.v1", "operation": "screen_path", "status": "clear",
                    "first_blocked_segment": None, "segments_screened": count - 1, "min_inter_arm_clearance_m": 0.2}
        pose, joints = effectors[request["arm"]]["pose_xyz_xyzw"], effectors[request["arm"]]["arm_joint_qpos"]
        if request["operation"] == "pace_path":
            return paced(request, lambda rows: [[*(pose[k] + row[k] - joints[k] for k in range(3)), *pose[3:]]
                                                for row in rows])
        if request["operation"] == "fk":
            rows = [{"index": i, "pose_xyz_xyzw": [*(pose[k] + row[k] - joints[k] for k in range(3)), *pose[3:]]}
                    for i, row in enumerate(request["joints_rad"])]
        else:
            rows = [{"index": i, "status": "ok", "joints_rad": [*(joints[k] + target[k] - pose[k] for k in range(3)), *joints[3:]]}
                    for i, target in enumerate(request["poses_xyz_xyzw"])]
        return {"schema": "ora.kinematics-result.v1", "operation": request["operation"], "arm": request["arm"], "rows": rows}

    host.align_plan_callback = callback
    return asked


def _pointer(host):
    return json.loads((host.control_root / "current-observation.json").read_text())


def test_a_script_s_dry_run_executes_nothing_and_the_same_call_then_executes_what_it_planned(tmp_path, monkeypatch):
    episode = code_mode_episode(tmp_path, monkeypatch)
    host, tools, session, state = episode.host, episode.tools, episode.session, episode.state
    asked, sent = _kinematics(host), []
    inner = tool_execution.dispatch_action
    monkeypatch.setattr(tool_execution, "dispatch_action", lambda loop, action, *args, **kwargs: (
        sent.append(action), inner(loop, action, *args, **kwargs))[1])
    try:
        executor, move = episode.loop.toolkit_executor, _move(host)
        pointer = _pointer(host)
        parent = _proposed(host, "call-0", "bash", {"command": "python step.py"}, seq=2)
        with session._code_mode_parent("bash", parent, cap=2):
            with dry_run.dry():
                record = tools.invoke_with_context(
                    "move_to", move, call_context=_proposed(host, "call-0#001", "move_to", move, seq=3))
            assert not dry_run.requested()
            assert (state["effects"], state["receipts"], state["recorded"], sent) == (0, 0, [], [])
            assert not (host.control_root / "submitted-action.json").exists()
            assert _pointer(host) == pointer, "the observation is still open and unchanged"
            moved = tools.invoke_with_context(
                "move_to", move, call_context=_proposed(host, "call-0#002", "move_to", move, seq=4))
        assert record["dry_run"] is True and record["feasible"] is True and record["action"] == "move_to"
        assert "receipt" not in record and "remaining_budget" not in record, "nothing executed, nothing spent"
        assert record["screening"]["status"] == "clear" and record["route"] == "joint", "move_to is free_space: a joint-space line"
        assert set(asked) >= {"ik_chain", "fk", "screen_path"}
        assert moved["status"] == "executed" and state["effects"] == 1 and len(sent) == 1
        arm = sent[0]["arguments"]["arms"]["left"]
        assert record["arms"]["left"]["predicted_pose"] == pytest.approx(arm["expected_link6_xyz_xyzw"][-1], abs=1e-5)
        assert record["motion"]["native_actions"] == len(arm["knots_rad"])
        ledger = executor.completed
        assert ledger["tool_call_id"] == "call-0" and len(ledger["actions"]) == 1, "only the real call is in the ledger"
        assert [call for call, _ in state["recorded"]] == ["call-0"]
    finally:
        episode.client.close()


def test_a_tool_that_reaches_for_the_executor_in_a_dry_run_cannot_act(tmp_path, monkeypatch):
    episode = code_mode_episode(tmp_path, monkeypatch)
    host, tools, session, state = episode.host, episode.tools, episode.session, episode.state
    _kinematics(host)
    reached = []

    def sneaky(api, arguments):  # a tool (or a bug in one) that executes a trajectory
        reached.append(type(api.act).__name__)
        api.act.execute_action({"schema": "ora-motion-action.v3", "action": "follow_joint_trajectory"}, 5)

    monkeypatch.setitem(toolkit_dispatch.TOOLS, "move_to", sneaky)
    try:
        move = _move(host)
        with session._code_mode_parent("bash", _proposed(host, "call-0", "bash", {}, seq=2), cap=2):
            with dry_run.dry(), pytest.raises(dry_run.DryRunAction):
                tools.invoke_with_context("move_to", move, call_context=_proposed(host, "call-0#001", "move_to", move, seq=3))
        assert reached == ["NoActions"] and (state["effects"], state["receipts"], state["recorded"]) == (0, 0, [])
        assert episode.loop.toolkit_executor.completed["actions"] == []
    finally:
        episode.client.close()


def test_every_arm_moving_tool_has_a_dry_run_and_no_other_tool_does(tmp_path, monkeypatch):
    episode = code_mode_episode(tmp_path, monkeypatch)
    host, tools, session = episode.host, episode.tools, episode.session
    try:
        assert dry_run.TOOLS == ("move_pose", "move_to", "rotate_wrist", "rotate_pitch", "recover_pose", "return_home",
                                 "set_gripper", "grasp_execute", "move_base")
        with session._code_mode_parent("bash", _proposed(host, "call-0", "bash", {}, seq=2), cap=2):
            with dry_run.dry(), pytest.raises(ToolError) as error:
                tools.invoke_with_context("wait_ticks", {"ticks": 5},
                                          call_context=_proposed(host, "call-0#001", "wait_ticks", {}, seq=3))
        assert error.value.code == dry_run.UNAVAILABLE and error.value.constraints["allowed"] == list(dry_run.TOOLS)
    finally:
        episode.client.close()


# --- the record a script leaves: the session's real script path over the same stack ---------------------------------
# With ``library: extended`` each call's result records the inputs its explanation renders from (the arguments, what
# the script read of the call, in degrees, and the stage events the tool's own code noted), and an ``explain`` request
# is a nested call of the script: the explanation of a real toolkit call renders again from the record alone.


class RecordingSession(PiCodeMode, PiRobotSocket, PiToolRpc):
    """The policy session's script path: calls recorded by the robot socket into the run's trace, results projected
    as the model would read them (degrees; the causal slots are not composed here), on the episode's tool stack."""

    context_composition, supports_images, policy_session_id = None, False, "pi-session"
    submitted_action = last_action_step = last_action_event_id = last_fresh_observation_event_id = None

    def __init__(self, episode) -> None:
        self.code_mode = episode.loop.plan["policy"]["code_mode"]
        self.tools, self.run_root = episode.tools, episode.host.control_root.parent
        self.tool_action_executor = episode.session.tool_action_executor
        self.gateway, self.artifacts = NativeToolGateway(episode.tools), ArtifactStore(self.run_root)
        self.recorder = TrajectoryRecorder(self.run_root, run_id="run", policy_session_id="pi-session")
        self.robot = SimpleNamespace(script_calls=0, snapshot={"observation": {"observation_step": 0}},
                                     stopped_this_turn=None)
        self._script_names = {item["name"]: item["name"] for item in episode.tools.script_definitions()}

    def _call(self, method, params):  # the agent core's record RPC
        return self.recorder.append(params["type"], **params["fields"])

    def _current_observation_id(self):
        return json.loads((self.tools.control_root / "current-observation.json").read_text())["observation_id"]


def test_a_toolkit_call_s_explanation_renders_again_from_what_its_result_recorded(tmp_path, monkeypatch) -> None:
    from ora.agent.code_mode import record

    episode = code_mode_episode(tmp_path, monkeypatch, code_mode={"interface": "coding", "library": "extended"})
    host = episode.host
    _kinematics(host)
    session = RecordingSession(episode)
    root, move = session.run_root, _move(host)
    try:
        session.recorder.append("tool_call_proposed", observation_step=0, model_call_id="model-0", tool_call_id="call-0",
                                payload={"name": "bash", "arguments": {}, "effect": "mutation", "batch_valid": True})
        with session._code_mode_parent("bash", {"model_call_id": "model-0", "tool_call_id": "call-0"}, 2) as parent:
            state = {"seq": 0, "actions": 0, "calls": 0, "cap": 2, "log": [], "frames": [], "uncounted_s": 0.0,
                     "queries": 0, "query_s": 0.0, "parent": parent, "model_call_id": "model-0",
                     "tool_call_id": "call-0"}
            call = dict(model_call="model-0", parent="call-0", state=state)
            planned = session._code_mode_call("move_to", {**move, "dry_run": True}, **call)
            moved = session._code_mode_call("move_to", move, **call)
            asked = session._code_mode_op({"op": "explain", "call_id": moved["call_id"]}, state)
        assert planned["result"]["feasible"] is True and moved["result"]["status"] == "executed", moved
        assert (state["actions"], state["calls"], state["queries"]) == (1, 0, 1), "the request counts nothing"
        planned_inputs, moved_inputs = (record.recorded_inputs(root, item["call_id"]) for item in (planned, moved))
        assert planned_inputs["answer"]["result"] == planned["result"], "what the script read, not the raw record"
        assert "compiled" in [event["stage"] for event in planned_inputs["events"]], "the tool's own stage event"
        assert planned["explain"] == record.attached(record.render(planned_inputs).text)
        assert moved_inputs["answer"]["result"]["receipt"] == moved["result"]["receipt"]
        assert asked["ok"] is True and asked["text"] == record.render(moved_inputs).text
        assert asked["text"].startswith("move_to(left") and "executed" in asked["text"].splitlines()[0]
        assert record.check_explanations(root) == []
        assert verify_agent_trace(root)["errors"] == ["missing_tool_result:call-0"], "only the script's own result"
    finally:
        episode.client.close()
