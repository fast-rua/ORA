"""Portable grasp -> owner ledger -> actual provider input, without a simulator.

On toolkit v5, the legacy-visible facade, with either receipt delivery (the layered v2 these tests were written for,
which took the budget from the call, was retired on 2026-11-01)."""
import base64
from copy import deepcopy
import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest

from ora.agent.composition.surface.tool_surface import resolve_tool_surface
from ora.agent.providers.openai_compatible import OpenAICompatibleProvider
from ora.agent.providers.anthropic_request import translate_anthropic_request
from ora.agent.toolkit.interface.types import Response, canonical
from ora.agent.toolkit.lib.values import JsonValue
from ora.agent.toolkit.tools.grasp_execute import _stages
from ora.agent.toolkit.host.runtime.dispatch import Dispatch
from ora.orchestrator.episode.execution_state import ExecutionStateOwner
from ora.orchestrator.episode import tool_execution
from tests.support.agent.current_visual_delivery import published
from tests.support.agent.observation_delivery import Delivery, host as planner_host
from tests.support.agent.observation_fixtures import observation_fixture

SEQUENCE = ["open", "pregrasp", "approach", "close", "lift", "verify"]


def fixture(root, monkeypatch, *, show, budget=None, corrupt=False, composition="native_slots_v3"):
    root.mkdir()
    context, _, _, _, _ = published(root, "robodojo", after_action=True)
    context["remaining_budget"]["session_round"] = 1
    surface = resolve_tool_surface("robodojo_placement_toolkit_v5", toolkit_grants={
        "tools": ["check_task"], "state_keys": ["privileged.task_check"]})
    if budget is not None:  # a budget is offline configuration (grasp_execute: 100)
        monkeypatch.setattr("ora.agent.toolkit.tools.budgets.native_action_budget", lambda name: budget)
    planner = planner_host(root, tools=SimpleNamespace(surface=surface, controlled_end_effector="right"),
                           context_composition=composition,
                           actual_sequence=0, observation_id=context["post_action_observation"]["observation_id"])
    planner.toolkit_observation_receipts = show
    planner.last_actual_snapshot = {**planner._actual_snapshot(context), "actual_sequence": 0}
    planner.last_action_id = planner.last_action_event_id = None
    events = []
    def record(method, params):
        if method == "execution_evidence_barrier":
            return {}
        events.append(params)
        return {"event_id": f"event-{len(events)}"}
    planner._call = record
    control = root / "control"
    control.mkdir(exist_ok=True)
    state = {"step": 0, "used": 0, "renders": 0, "held": None}
    observations, pointers = [], []
    for step in range(6):
        obs, source = observation_fixture(root, ("head", "left_wrist", "right_wrist"), sequence=step)
        obs["proprioception"]["end_effectors"]["left"] = {
            "pose_xyz_xyzw": [0., 0., .9, 0., 0., 0., 1.], "gripper_opening_m": .02}
        (source / "observation.json").write_bytes(canonical(obs))
        immutable = root / "policy-observations" / obs["observation_id"]
        shutil.copytree(source, immutable, dirs_exist_ok=True)
        observations.append(obs)
        pointers.append({"observation_id": obs["observation_id"], "source_root": str(source),
                         "observation_sha256": hashlib.sha256((source / "observation.json").read_bytes()).hexdigest()})
    def write_pointer():
        mount = root / "sandbox/inputs/current"
        shutil.copytree(pointers[state["step"]]["source_root"], mount, dirs_exist_ok=True)
        pointer = {**pointers[state["step"]], "source_root": str(mount)}
        (control / "current-observation.json").write_bytes(canonical(pointer))
    write_pointer()
    def observation(api):
        return SimpleNamespace(identity=JsonValue.of(pointers[state["step"]]),
                               manifest=JsonValue.of(observations[state["step"]]))
    monkeypatch.setattr(_stages, "observe", observation)
    def execute(loop, action, pointer, **kwargs):
        return {"schema": "ora.robodojo-sparse-motion-receipt.v1", "action_id": action["action_id"],
                "action": action, "native_actions": 2, "consumed": True,
                "execution_status": "reached", "command_invariant": {"verified": True},
                "gripper_directives": {"left": 0.0}}, None
    def publish(loop, *args, **kwargs):
        state["step"] += 1
        state["renders"] += 1
        loop.interaction.observation_step = state["step"]
        write_pointer()
        return pointers[state["step"]], {"terminal": False}
    def receipt(action, raw, duration, **kwargs):
        state["used"] += raw["native_actions"]
        planner.record_environment_receipt(raw)
    loop = SimpleNamespace(run_root=root, control=control, termination=None, interrupted=False,
        environment_status={"terminal": False}, round_model_calls=0,
        plan={"budgets": {"max_model_calls": 200}, "toolkit_observation_receipts": show},
        interaction=SimpleNamespace(phase="policy_active", observation_step=0, begin_policy=lambda: None),
        native_allowance=SimpleNamespace(remaining=lambda: 100 - state["used"],
                                          fields=lambda: {"native_actions_remaining": 100 - state["used"]}),
        execution_state_owner=ExecutionStateOwner(), planner=planner,
        _exhausted_budget=lambda: None, _should_stop=lambda: False,
        _accept_submitted_action=lambda n: (json.loads((control / "submitted-action.json").read_text()), pointers[state["step"]]),
        _record_receipt=receipt, record_termination_decision=lambda *a, **k: None,
        runtime=SimpleNamespace(abort=lambda reason: state.update(abort=reason)))
    from ora.agent.sessions.base import DelegatingPolicySession
    loop.planner = DelegatingPolicySession(planner, SimpleNamespace())
    loop.plan["toolkit_receipt_delivery"] = "observation" if show else "tool_output"
    monkeypatch.setattr(tool_execution, "dispatch_action", execute)
    monkeypatch.setattr(tool_execution, "publish_observation", publish)
    executor = tool_execution.ToolActionExecutor(loop)
    def call(action, admitted):
        return Response(canonical({"status": "ok", "data": executor.execute_action(action, admitted)}))
    api = SimpleNamespace(act=SimpleNamespace(execute_action=call),
                          state=SimpleNamespace(read=lambda key: state["held"],
                                                write=lambda key, value: state.update(held=value)))
    pose = {"position_m": [0., 0., .9], "orientation_xyzw": [0., 0., 0., 1.]}
    commitment = {"arm": "left", "candidate_id": "candidate", "source_ref": "target",
                  "poses": {name: pose for name in ("pregrasp", "approach", "lift")}}
    raw_result = {}
    def run(name, args):
        result = _stages.run(api, {"execution": {"generation": 0, "workspace_bounds_m": {}}}, commitment, SEQUENCE)
        result["_runtime"]["legacy_submission"] = {"success": True, "status": "submitted",
            "skill": "grasp_execute", "candidate_id": "candidate"}
        if corrupt:
            result["stages"][0]["action_id"] = "another-parent"
        raw_result.update(deepcopy(result))
        return result
    dispatch = Dispatch.__new__(Dispatch)
    dispatch.host = SimpleNamespace(surface=surface, toolkit_executor=executor)
    dispatch.supplied = SimpleNamespace(get=lambda: {"tool_call_id": "call", "model_call_id": "model"})
    dispatch._run = run
    result = dispatch.run("grasp_execute", {})
    return planner, executor, state, context, pointers, events, result, raw_result


@pytest.mark.parametrize("show", [False, True])
@pytest.mark.parametrize("composition", ["native_slots_v3", "native_slots_v4"])
def test_grasp_provider_input_matches_composite_feedback_and_switch(tmp_path, monkeypatch, show, composition):
    planner, executor, state, context, pointers, events, result, raw = fixture(
        tmp_path / "run", monkeypatch, show=show, composition=composition)
    assert state["renders"] == 5 and executor.completed["native_actions_used"] == 10
    assert "_runtime" not in result
    assert len([e for e in events if e["type"] == "environment_receipt"]) == 5
    last = executor.completed["actions"][-1]
    assert planner.pending_receipt["event_id"] == last["receipt_event_id"]
    assert planner.pending_receipt["receipt_ref"] == last["receipt_ref"]
    body = deepcopy(planner.pending_receipt_body)
    before = deepcopy(planner.last_actual_snapshot)
    delivery = Delivery(planner)
    message, _ = delivery.observation(context)
    envelope = delivery.envelopes[-1]
    baseline = pointers[4]
    from pathlib import Path
    baseline_pixels = base64.b64encode((Path(baseline["source_root"]) / "left_wrist/rgb.png").read_bytes()).decode()
    for prepare in (OpenAICompatibleProvider.prepare_request, translate_anthropic_request):
        encoded = json.dumps(prepare({"model": "offline", "max_tokens": 64, "messages": [message]}))
        assert ("historical intermediate frame" in encoded) is show
        assert (baseline_pixels in encoded) is show
        assert ("last_action_effect" in encoded) is show
        assert "stage_evidence" not in encoded and "skill_feedback" not in encoded
    if not show:
        assert not (planner.run_root / "native/tool-observation-feedback").exists()
        assert body["schema"] == "ora.robodojo-sparse-motion-receipt.v1"
        return
    detail = envelope["last_action_effect"]["detail"]
    assert [s["stage"] for s in detail["stages"]] == SEQUENCE
    # With observation delivery the legacy-visible tool returns only its submission; the stages come with the
    # next observation, compared here with what the tool body returned.
    assert result["status"] == "submitted"
    assert detail["verification"]["gripper_opening_m"] == raw["verification"]["gripper_opening_m"]
    assert detail["verification"]["jaw_hold"]["held"] == raw["verification"]["jaw_hold"]["held"]
    assert detail["stages_not_reached"] == [] and detail["stop_reason"] == "sequence_completed"
    assert body["skill_receipt"]["observation_evidence"][0]["observation_id"] == pointers[4]["observation_id"]
    # Old composite receipt shape, assembled independently from the returned
    # stages and the same public verify measurements. Private controller fields
    # and action IDs are not part of the visible skill projection.
    old = {
        "schema": "ora.manipulation-skill-sequence-receipt.v1", "action_id": "old-composite",
        "skill": "grasp_execute", "candidate_id": "candidate", "committed_sequence": SEQUENCE,
        "resumed_from": 0, "stage_receipts": raw["stages"] + [
            {"stage": "verify", **raw["_runtime"]["skill_feedback"]["verify"]}],
        "status": "completed", "stop_reason": "sequence_completed", "terminal_after_action": False,
        "skill_receipt": {"observation_evidence": [{"role": "post_close_before_lift", "arm": "left",
            "observation_root": baseline["source_root"], "observation_id": baseline["observation_id"],
            "observation_sha256": baseline["observation_sha256"]}]},
    }
    planner.pending_receipt_body = old
    planner.last_actual_snapshot, planner.actual_sequence = before, 0
    old_message, _ = Delivery(planner).observation(context)
    assert message["content"] == old_message["content"]
    sidecars = list((planner.run_root / "native/tool-observation-feedback").glob("*.json"))
    assert len(sidecars) == 1
    provenance = json.loads(sidecars[0].read_text())
    assert provenance["tool_call_id"] == "call" and len(provenance["actions"]) == 5


@pytest.mark.parametrize("budget", [0, 2, 8])
def test_budget_stop_keeps_prefix_and_no_invented_verification(tmp_path, monkeypatch, budget):
    planner, executor, state, context, pointers, events, result, raw = fixture(
        tmp_path / "run", monkeypatch, show=True, budget=budget)
    assert result["status"] == "submitted" and raw["status"] == "stopped"  # observation delivery: see above
    assert executor.completed["native_actions_used"] == budget
    assert state["renders"] == budget // 2
    if budget == 0:
        assert not (planner.run_root / "native/tool-observation-feedback").exists()
        return
    effect = planner._minimal_action_effect(planner.pending_receipt_body)
    detail = effect["detail"]
    assert "verification" not in detail
    assert [s["stage"] for s in detail["stages"]] == SEQUENCE[:budget // 2 + 1]
    assert detail["stages_not_reached"] == SEQUENCE[budget // 2 + 1:]
    assert bool(planner.pending_receipt_body["skill_receipt"]["observation_evidence"]) is (budget == 8)


def test_stage_identity_cannot_invent_execution(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="execution evidence"):
        fixture(tmp_path / "run", monkeypatch, show=True, corrupt=True)


def test_legacy_delivery_through_real_facade_and_mutable_input_mount(tmp_path, monkeypatch):
    """The legacy-visible facade on v5: tool_output receipts (v3's observation receipts were retired 2026-10-31)."""
    planner, executor, state, context, pointers, events, result, raw = fixture(
        tmp_path / "run", monkeypatch, show=False, composition="native_slots_v4")
    assert executor.completed["native_action_budget_requested"] == 810
    assert state["renders"] == 5
    assert "pending_receipt_body" not in vars(executor.loop.planner)
    message, _ = Delivery(planner).observation(context)
    blocks = message["content"]
    assert sum(b["type"] == "image_url" for b in blocks) == 3
    assert not [b["text"] for b in blocks if b["type"] == "text" and b["text"].startswith("GRASP BASELINE")]
    assert result["status"] == "executed" and "remaining_budget" in result
