"""Persistent AgentSession with Supervisor-owned environment rounds, on the Pi agent core.

The model is a FakeProvider script (``tests/support/orchestrator/multi_round.py``); what the
retired Python provider stubs asserted on each request is read back from the
canonical requests in the run's agent trace.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping

import pytest

from ora.agent.evidence.trace_verify import verify_agent_trace
from ora.environment import SyntheticSession
from ora.environment.base import (
    ENVIRONMENT_STATUS_SCHEMA, validate_environment_status,
)
from ora.orchestrator.episode.live import HarnessLiveEvent
from ora.orchestrator.evaluation.evaluate import (
    evaluate, verify_multi_round_session,
)
from ora.orchestrator.episode.policy_factory import build_policy_session
from ora.orchestrator.spec.run_spec import RunSpecError, resolve_run_spec
from tests.support.orchestrator import multi_round
from tests.support.orchestrator.multi_round import ROOT, SuccessSyntheticSession

_spec = multi_round.spec


class _ControlFaultSyntheticSession(SyntheticSession):
    """Deterministic terminal-after-action failure with a Boolean verdict."""

    def status(self) -> dict[str, Any]:
        terminal = bool(self.terminal)
        reason = self.abort_reason or (
            "libero_control_health_fault" if terminal else None
        )
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA,
            "phase": "terminal" if reason is not None else "running",
            "terminal": reason is not None,
            "reason": reason,
            "source": "synthetic_control_fault_session",
            "policy_visibility": "control_only",
            "signals": ([{
                "code": "libero_control_health_fault",
                "severity": "terminal",
                "source": "synthetic_control_health.v1",
                "policy_visibility": "control_only",
            }] if terminal and self.abort_reason is None else []),
        })

    def persist_official_verdict(self) -> dict[str, Any]:
        verdict = {
            "schema": "ora.synthetic-control-fault-verdict.v1",
            "official_success": False,
        }
        path = self.output_root / "official-verdict.json"
        path.write_text(json.dumps(verdict, sort_keys=True) + "\n", encoding="utf-8")
        return verdict


class _PostTerminalFailureSession(SuccessSyntheticSession):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.video_called = False
        self.close_called = False

    def finalize_operator_video(self) -> None:
        self.video_called = True
        raise RuntimeError("video callback failed after verdict")

    def close(self) -> None:
        self.close_called = True
        raise RuntimeError("environment close failed after verdict")


class _FailingLivePort:
    def publish(self, _event: HarnessLiveEvent) -> None:
        raise RuntimeError("viewer callback unavailable")

    def poll(self) -> tuple[Mapping[str, Any], ...]:
        return ()


class _PostTerminalFailurePolicy:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.capabilities = delegate.capabilities

    def start(self) -> None:
        self.delegate.start()

    def act(self, turn: Any) -> Any:
        if turn.terminal_lifecycle:
            raise RuntimeError("post-terminal policy callback failed")
        return self.delegate.act(turn)

    def accept_runtime_event(self, event: Any) -> None:
        if event.kind in {"official_verdict", "run_termination"}:
            raise RuntimeError(f"{event.kind} callback failed")
        self.delegate.accept_runtime_event(event)

    def interrupt(self, reason: str = "operator_interrupt") -> None:
        self.delegate.interrupt(reason)

    def close(self) -> None:
        try:
            self.delegate.close()
        finally:
            raise RuntimeError("policy close failed after verdict")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def test_multi_round_session_requires_explicit_supported_composition() -> None:
    legacy = resolve_run_spec(_spec(), root=ROOT)
    assert "session" not in legacy

    # K = 1 is a valid one-round session; 0 and 33 stay outside [1,32].
    one_round = _spec()
    one_round["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 1,
    }
    assert resolve_run_spec(one_round, root=ROOT)["session"]["rounds"] == 1
    for rounds, message in ((0, "positive integer"), (33, r"in \[1,32\]")):
        invalid = _spec()
        invalid["session"] = {
            "cadence": "persistent_agent_k_rounds_v1", "rounds": rounds,
        }
        with pytest.raises(RunSpecError, match=rf"session.rounds must be (a )?{message}"):
            resolve_run_spec(invalid, root=ROOT)
    invalid = _spec()
    invalid["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 1,
        "round_seeds": [0, 1],
    }
    with pytest.raises(RunSpecError, match="exactly session.rounds"):
        resolve_run_spec(invalid, root=ROOT)

    invalid = _spec()
    invalid["policy"]["kind"] = "scripted"
    invalid["policy"]["actions"] = [{"x": 1}]
    invalid["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 2,
    }
    with pytest.raises(RunSpecError, match="policy.kind=native_api"):
        resolve_run_spec(invalid, root=ROOT)

    invalid = _spec()
    invalid["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 3,
        "round_seeds": [0, 1],
    }
    with pytest.raises(RunSpecError, match="exactly session.rounds"):
        resolve_run_spec(invalid, root=ROOT)

    invalid = _spec()
    invalid["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 2,
        "round_seeds": [1, 2],
    }
    with pytest.raises(RunSpecError, match=r"round_seeds\[0\]"):
        resolve_run_spec(invalid, root=ROOT)


def test_one_agent_session_runs_three_independent_supervisor_reset_rounds(
    tmp_path: Path,
) -> None:
    spec = _spec()
    spec["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 3,
        "round_seeds": [0, 1, 2],
    }
    plan = resolve_run_spec(spec, root=ROOT)
    run_root = tmp_path / "run"
    result = multi_round.run(run_root, plan, multi_round.finish_each_round(3))
    requests = multi_round.provider_requests(run_root)

    assert result["schema"] == "ora.session-result.v1"
    assert result["state"] == "completed"
    assert result["termination"] == "session_round_limit"
    assert result["official_success"] is None
    assert result["round_count"] == 3
    assert result["round_success_count"] == 0
    assert [item["round_index"] for item in result["rounds"]] == [0, 1, 2]
    assert [item["seed"] for item in result["rounds"]] == [0, 1, 2]
    assert [item["termination"] for item in result["rounds"]] == [
        "policy_finished_episode", "policy_finished_episode", "policy_finished_episode",
    ]
    assert result["model_calls"] == 3
    assert result["decisions"] == 0
    assert len(requests) == 3
    assert [multi_round.announces_round(request, index + 1) for index, request in enumerate(requests)] == [True] * 3
    assert multi_round.session_closed(run_root)
    assert all(
        "advertised Memory tool" not in json.dumps(
            request["messages"], ensure_ascii=False,
        )
        for request in requests
    )
    assert [len(request["messages"]) for request in requests] == sorted(
        len(request["messages"]) for request in requests
    )
    for index in range(1, 3):
        assert f"finish-round-{index - 1}" in json.dumps(
            requests[index]["messages"], ensure_ascii=False,
        )

    events = [
        json.loads(line)
        for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event.get("type") == "reset" for event in events) == 3
    assert [
        event["record"]["seed"] for event in events
        if event.get("type") == "reset"
    ] == [0, 1, 2]
    assert sum(event.get("type") == "round_finished" for event in events) == 3
    assert all(
        (run_root / "rounds" / f"round-{index:04d}" / "result.json").is_file()
        for index in range(3)
    )
    assert verify_agent_trace(run_root)["status"] == "valid"
    agent_events = [
        json.loads(line)
        for line in (run_root / "agent-events.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event.get("type") == "session_started" for event in agent_events) == 1
    assert sum(event.get("type") == "session_closed" for event in agent_events) == 1
    assert evaluate(run_root)["status"] == "valid"
    drifted = copy.deepcopy(result)
    drifted["rounds"][1]["seed"] = 99
    assert "multi_round_result_seed_invalid:1" in verify_multi_round_session(
        run_root, plan, drifted, events,
    )
    # Refused proposals aggregate like decisions; a record written before the
    # counter existed carries neither field and still reconciles.
    assert result["refused_proposals"] == 0
    assert all(record["refused_proposals"] == 0 for record in result["rounds"])
    miscounted = copy.deepcopy(result)
    miscounted["rounds"][0]["refused_proposals"] = 1
    assert "multi_round_aggregate_mismatch" in verify_multi_round_session(
        run_root, plan, miscounted, events,
    )
    historical = copy.deepcopy(result)
    del historical["refused_proposals"]
    for record in historical["rounds"]:
        del record["refused_proposals"]
    assert "multi_round_aggregate_mismatch" not in verify_multi_round_session(
        run_root, plan, historical, events,
    )


def test_a_one_round_session_ends_after_round_0_and_evaluates_valid(
    tmp_path: Path,
) -> None:
    """K = 1 keeps the session wrapper and evidence; nothing restarts after the only round."""
    spec = _spec()
    spec["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 1,
    }
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["session"] == {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 1,
        "budget_scope": "per_round", "reset_owner": "supervisor",
    }
    run_root = tmp_path / "run"
    result = multi_round.run(run_root, plan, multi_round.finish_each_round(1), run_id="one-round-session")
    assert result["schema"] == "ora.session-result.v1"
    assert result["state"] == "completed"
    assert result["termination"] == "session_round_limit"
    assert result["official_success"] is None
    assert result["round_count"] == 1 and result["round_success_count"] == 0
    assert [item["round_index"] for item in result["rounds"]] == [0]
    assert result["rounds"][0]["termination"] == "policy_finished_episode"
    requests = multi_round.provider_requests(run_root)
    assert result["model_calls"] == 1 and len(requests) == 1
    assert multi_round.announces_round(requests[0], 1)
    assert multi_round.session_closed(run_root)
    events = [
        json.loads(line)
        for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # One reset (round 0's) and one finished round; no between-round restart.
    assert sum(event.get("type") == "reset" for event in events) == 1
    assert sum(event.get("type") == "round_finished" for event in events) == 1
    assert [event.get("round_index") for event in events
            if event.get("type") in {"round_start_isolation", "round_finished"}] == [0, 0]
    assert (run_root / "rounds" / "round-0000" / "result.json").is_file()
    assert not (run_root / "rounds" / "round-0001").exists()
    assert verify_agent_trace(run_root)["status"] == "valid"
    assert verify_multi_round_session(run_root, plan, result, events) == []
    assert evaluate(run_root)["status"] == "valid"
    # The one-round contract still fails closed on a missing or extra round.
    extra = copy.deepcopy(result)
    extra["rounds"].append(copy.deepcopy(extra["rounds"][0]))
    assert "multi_round_result_contract_invalid" in verify_multi_round_session(
        run_root, plan, extra, events,
    )


def test_trusted_success_automatically_advances_to_the_next_round(
    tmp_path: Path,
) -> None:
    spec = _spec()
    spec["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 2,
    }
    spec["budgets"]["max_model_calls"] = 3
    plan = resolve_run_spec(spec, root=ROOT)
    run_root = tmp_path / "run"

    def environment_factory(**kwargs: Any) -> SuccessSyntheticSession:
        return SuccessSyntheticSession(
            kwargs["output_root"], kwargs["task"], kwargs["seed"],
            max_actions=1, main_camera="agentview",
            camera_names=("agentview", "wrist"),
        )

    result = multi_round.run(
        run_root, plan, multi_round.act_then_succeed(2), run_id="multi-round-success",
        environment_session_factory=environment_factory,
    )
    assert result["state"] == "completed"
    assert result["round_success_count"] == 2
    assert result["decisions"] == 2
    assert result["model_calls"] == 4
    assert [item["termination"] for item in result["rounds"]] == [
        "trusted_runtime_boundary", "trusted_runtime_boundary",
    ]
    assert [item["official_success"] for item in result["rounds"]] == [True, True]
    assert verify_agent_trace(run_root)["status"] == "valid"


@pytest.mark.parametrize(
    ("runtime_kind", "verdicts", "missing_boolean"),
    [
        ("robotwin", [True, False, True], False),
        ("robotwin", [False, False, False], False),
        ("robotwin", [True, None, False], True),
        ("robotwin", [True, 1, False], True),
        ("synthetic", [None, None, None], False),
    ],
)
def test_evaluation_checks_robotwin_booleans_at_the_round_boundary(
    tmp_path: Path, runtime_kind: str, verdicts: list[Any],
    missing_boolean: bool,
) -> None:
    """Exercise the evaluator with a diagnostic trace, not a simulator claim.

    The Synthetic loop supplies a complete, action-free three-round fixture.
    Only its persisted runtime discriminator and verdict fields are varied to
    test the RoboTwin-specific Boolean rule through the real evaluator.
    """

    spec = _spec()
    spec["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 3,
    }
    plan = resolve_run_spec(spec, root=ROOT)
    run_root = tmp_path / "run"
    result = multi_round.run(run_root, plan, multi_round.finish_each_round(3), run_id="verdict-rule-fixture")
    plan["runtime"]["kind"] = runtime_kind
    (run_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    manifest_path = run_root / "context-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment"]["backend"] = runtime_kind
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for index, value in enumerate(verdicts):
        record = result["rounds"][index]
        record["official_success"] = value
        path = run_root / "rounds" / f"round-{index:04d}" / "result.json"
        path.write_text(json.dumps(record), encoding="utf-8")
    result["round_success_count"] = sum(value is True for value in verdicts)
    (run_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    events_path = run_root / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    for event in events:
        if event.get("type") == "round_finished":
            event["result"] = result["rounds"][event["round_index"]]
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8",
    )

    report = evaluate(run_root)
    assert ("missing_trusted_boolean_verdict" in report["reasons"]) is missing_boolean
    assert result["official_success"] is None
    if not missing_boolean:
        assert report["status"] == "valid", report["reasons"]

    # A valid Boolean prefix never qualifies an interrupted K3 session.
    result.update(state="incomplete", termination="trusted_fault", round_count=2)
    result["rounds"] = result["rounds"][:2]
    (run_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    interrupted = evaluate(run_root)
    assert interrupted["status"] == "invalid"
    assert "multi_round_result_contract_invalid" in interrupted["reasons"]


@pytest.mark.parametrize("verdict", [True, False, None, 1])
def test_single_episode_robotwin_still_requires_top_level_boolean(
    tmp_path: Path, verdict: Any,
) -> None:
    plan = resolve_run_spec(_spec(), root=ROOT)
    run_root = tmp_path / "run"
    result = multi_round.run(run_root, plan, multi_round.close_every_request(2, "repeat-close"),
                             run_id="single-verdict-rule-fixture")
    assert result["state"] == "completed"
    plan["runtime"]["kind"] = "robotwin"
    result["official_success"] = verdict
    (run_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (run_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    report = evaluate(run_root)
    assert ("missing_trusted_boolean_verdict" in report["reasons"]) is (
        not isinstance(verdict, bool)
    )


def test_post_terminal_auxiliary_failures_cannot_overwrite_official_verdict(
    tmp_path: Path,
) -> None:
    plan = resolve_run_spec(_spec(), root=ROOT)
    run_root = tmp_path / "run"
    environment_holder: list[_PostTerminalFailureSession] = []

    def environment_factory(**kwargs: Any) -> _PostTerminalFailureSession:
        environment = _PostTerminalFailureSession(
            kwargs["output_root"], kwargs["task"], kwargs["seed"],
            max_actions=1, main_camera="agentview",
            camera_names=("agentview", "wrist"),
        )
        environment_holder.append(environment)
        return environment

    def policy_factory(**kwargs: Any) -> _PostTerminalFailurePolicy:
        return _PostTerminalFailurePolicy(build_policy_session(**kwargs))

    result = multi_round.run(
        run_root, plan, multi_round.act_then_succeed(1),
        run_id="post-terminal-isolation",
        policy_session_factory=policy_factory,
        environment_session_factory=environment_factory,
        live_port=_FailingLivePort(),
    )

    assert result["state"] == "completed"
    assert result["termination"] == "trusted_runtime_boundary"
    assert result["termination_authority"] == "environment"
    assert result["official_success"] is True
    assert result["verdict"]["official_success"] is True
    assert result["behavioral_sample_valid"] is True
    assert environment_holder[0].video_called is True
    assert environment_holder[0].close_called is True
    events = [
        json.loads(line)
        for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failure_stages = {
        event["stage"] for event in events
        if event.get("type") == "auxiliary_failure"
    }
    assert {
        "terminal_policy_lifecycle",
        "policy_callback:official_verdict",
        "finalize:operator_video",
        "policy_callback:run_termination",
        "finalize:planner_close",
        "cleanup:environment_close",
    }.issubset(failure_stages)
    assert any(stage.startswith("live_port:") for stage in failure_stages)
    persisted = json.loads((run_root / "result.json").read_text(encoding="utf-8"))
    assert persisted["official_success"] is True
    assert persisted["verdict"]["official_success"] is True


def test_repeated_actions_execute_and_deliver_no_stagnation_judgement(
    tmp_path: Path,
) -> None:
    """A repeating policy is recorded, never diagnosed to itself.

    The Supervisor once compared each action with the previous one, counted a
    no-progress streak and emitted `action_stagnation_advisory`. The advisory
    never rejected anything, and the record it came from also reached the
    policy through `runtime_notice`: the managed compact context forwards that
    field, and the Native V1/V2 observation message serialized the whole
    context object. Both the comparison and that routing are retired, so the
    only surviving evidence is the Harness-owned observation delta.
    """

    spec = _spec()
    spec["runtime"]["terminal_actions"] = 100
    spec["budgets"] = {
        "max_robot_decisions": 6,
        "max_model_calls": 7,
        "max_sim_steps": 100,
        "wall_time_s": 30,
    }
    plan = resolve_run_spec(spec, root=ROOT)
    run_root = tmp_path / "run"
    result = multi_round.run(run_root, plan, multi_round.close_every_request(7, "repeat-close"),
                             run_id="stagnation-advisory")

    assert result["state"] == "completed"
    assert result["termination"] == "robot_decision_budget"
    assert result["termination_authority"] == "experiment_budget"
    assert result["decisions"] == 6
    events = [
        json.loads(line)
        for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert not any(
        event.get("type").startswith("action_stagnation")
        or event.get("type") == "action_rejected_stagnation"
        for event in events
        if isinstance(event.get("type"), str)
    )
    assert sum(event.get("type") == "action_submitted" for event in events) == 6
    deltas = [
        event["record"] for event in events
        if event.get("type") == "stagnation_observation"
    ]
    assert deltas and all(set(record) == {"head_phash_hamming"} for record in deltas)

    notices = [
        json.loads(path.read_text(encoding="utf-8"))["runtime_notice"]
        for path in sorted((run_root / "policy-state/steps").glob("step-*.json"))
    ]
    assert notices
    assert all(
        notice is None or notice.get("kind") == "session_round_start"
        for notice in notices
    )


def test_terminal_control_failure_preserves_actual_and_completes_five_rounds(
    tmp_path: Path,
) -> None:
    spec = _spec()
    spec["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 5,
    }
    plan = resolve_run_spec(spec, root=ROOT)
    run_root = tmp_path / "run"

    def environment_factory(**kwargs: Any) -> _ControlFaultSyntheticSession:
        return _ControlFaultSyntheticSession(
            kwargs["output_root"], kwargs["task"], kwargs["seed"],
            max_actions=1, main_camera="agentview",
            camera_names=("agentview", "wrist"),
        )

    result = multi_round.run(
        run_root, plan, multi_round.close_every_request(5),
        run_id="multi-round-control-failure-qualification",
        environment_session_factory=environment_factory,
    )

    assert result["state"] == "completed"
    assert result["termination"] == "session_round_limit"
    assert result["round_count"] == 5
    assert result["round_success_count"] == 0
    assert result["decisions"] == 5
    assert result["model_calls"] == 5
    assert [item["official_success"] for item in result["rounds"]] == [
        None, None, None, None, None,
    ]
    assert [item["termination_authority"] for item in result["rounds"]] == [
        "infrastructure", "infrastructure", "infrastructure",
        "infrastructure", "infrastructure",
    ]
    assert not any(item["behavioral_sample_valid"] for item in result["rounds"])
    requests = multi_round.provider_requests(run_root)
    assert len(requests) == 5
    assert [multi_round.announces_round(request, index + 1) for index, request in enumerate(requests)] == [True] * 5
    events = [
        json.loads(line)
        for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(
        event.get("type") == "terminal_observation_preserved" for event in events
    ) == 5
    agent_events = [
        json.loads(line)
        for line in (run_root / "agent-events.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert sum(
        event.get("type") == "observation_preserved_without_policy_turn"
        for event in agent_events
    ) == 5
    assert verify_agent_trace(run_root)["status"] == "valid"


def test_round_decision_budget_preserves_actual_and_completes_five_rounds(
    tmp_path: Path,
) -> None:
    spec = _spec()
    spec["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": 5,
    }
    spec["budgets"]["max_robot_decisions"] = 1
    plan = resolve_run_spec(spec, root=ROOT)
    run_root = tmp_path / "run"

    def environment_factory(**kwargs: Any) -> SyntheticSession:
        return SyntheticSession(
            kwargs["output_root"], kwargs["task"], kwargs["seed"],
            max_actions=100, main_camera="agentview",
            camera_names=("agentview", "wrist"),
        )

    result = multi_round.run(
        run_root, plan, multi_round.close_every_request(5),
        run_id="multi-round-decision-budget-qualification",
        environment_session_factory=environment_factory,
    )
    assert [multi_round.announces_round(request, index + 1)
            for index, request in enumerate(multi_round.provider_requests(run_root))] == [True] * 5

    assert result["state"] == "completed"
    assert result["termination"] == "session_round_limit"
    assert result["round_count"] == 5
    assert result["round_success_count"] == 0
    assert result["decisions"] == 5
    assert result["model_calls"] == 5
    assert [item["termination"] for item in result["rounds"]] == [
        "robot_decision_budget",
        "robot_decision_budget",
        "robot_decision_budget",
        "robot_decision_budget",
        "robot_decision_budget",
    ]
    events = [
        json.loads(line)
        for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sum(
        event.get("type") == "terminal_observation_preserved" for event in events
    ) == 5
    assert verify_agent_trace(run_root)["status"] == "valid"


def test_round_reset_drops_every_environment_bound_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Supervisor-owned world reset invalidates the refs the runtime cached.

    The environment clears its own plan/payload tables on reset. If the tool
    runtime kept its mirrors, a stale plan_ref could still pass the runtime's
    "only gripper writes since planning" freshness rule and reach a backend
    that can only reject it -- and that rejection is raised outside the
    recoverable-error classification, so the leak surfaces as a Runtime fault
    instead of a retryable receipt.
    """

    from ora.agent.toolkit.host.runtime import ManipulationToolRuntime

    runtime = ManipulationToolRuntime.__new__(ManipulationToolRuntime)
    control_root = tmp_path / "control"
    (control_root / "rounds" / "round-0000").mkdir(parents=True)
    runtime.control_root = control_root
    runtime.point_path = control_root / "points.json"
    runtime.preview_path = control_root / "preview.json"
    runtime.point_path.write_text('{"round_zero": true}')
    runtime._active_mark_point_batch = {"batch": 1}
    runtime._known_plan_refs = {"plan-round0": {"stale": True}}
    runtime._submitted_mutations = ["move_pose", "set_gripper"]

    parent_calls: list[int] = []
    monkeypatch.setattr(
        ManipulationToolRuntime.__mro__[1], "reset_round",
        lambda self, index: parent_calls.append(index), raising=True,
    )
    runtime.reset_round(1)

    assert parent_calls == [1]
    assert not runtime.point_path.exists()
    assert (control_root / "rounds/round-0000/points.json").read_text() == '{"round_zero": true}'
    assert runtime._known_plan_refs == {}
    assert runtime._submitted_mutations == []
    assert runtime._active_mark_point_batch is None


def test_backend_execution_fault_makes_a_run_learning_ineligible(
    tmp_path: Path,
) -> None:
    """A backend-declared execution fault is infrastructure, not behavior.

    Such a run can still carry a valid official verdict, but its trajectory
    crossed an unexplained physical discontinuity, so it must not reach
    training or Memory as if the policy had simply failed.
    """

    from ora.orchestrator.evaluation.evaluate import assess_learning_eligibility

    run_root = tmp_path / "run"
    (run_root / "native").mkdir(parents=True)
    plan = {"runtime": {"kind": "robotwin"}, "session": {"rounds": 1}}
    result = {"state": "completed", "official_success": False}

    clean = run_root / "native" / "action-0000.json"
    clean.write_text(json.dumps({"execution_status": "reached"}))
    assert assess_learning_eligibility(
        plan, result, [], run_root=run_root,
    )["eligible"] is True

    faulted = run_root / "native" / "action-0001.json"
    faulted.write_text(json.dumps({
        "execution_status": "execution_fault",
        "execution_fault": {"code": "unknown_executor_exception"},
    }))
    verdict = assess_learning_eligibility(plan, result, [], run_root=run_root)
    assert verdict["eligible"] is False
    assert any(
        reason.startswith("backend_execution_fault:") for reason in verdict["reasons"]
    )


def test_fault_after_an_unrecorded_round_verdict_keeps_that_rounds_seed() -> None:
    from types import SimpleNamespace

    from ora.orchestrator.episode.result import _fault_result_record
    from ora.orchestrator.episode.termination import TerminationAuthority

    verdict = {"official_success": True, "source": "environment"}
    loop = SimpleNamespace(
        multi_round=True, verdict=verdict,
        termination_authority=TerminationAuthority.ENVIRONMENT,
        started_monotonic_ns=time.monotonic_ns(), started=time.time(),
        run_id="run", plan={"plan_sha256": "0" * 64, "seed": 100},
        decisions=3, refused_proposals=0, model_calls=4,
        verdict_guard=SimpleNamespace(divergence=None), auxiliary_failures=[],
        round_results=[], session_config={"rounds": 2, "round_seeds": [100, 205]},
        round_index=1, expected_round_seed=lambda index: (100, 205)[index],
    )
    result, preserved = _fault_result_record(loop, RuntimeError("boom"))
    assert preserved is True
    assert result["pending_round_verdict"] == {
        "round_index": 1, "seed": 205,
        "termination_authority": "environment",
        "official_success": True, "verdict": verdict,
    }
