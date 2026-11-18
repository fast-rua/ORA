"""A RoboDojo episode that runs past the official limit: the official result stays the limit's.

``advance_official_terminal`` is the upstream ``is_episode_end`` rule, and it
is run here against the wrapper's own code with a stand-in reward manager. A
success after ``step_lim`` must land in ``success_after_official_limit`` and
leave ``official_success`` false, or every reported rate stops being comparable
with the runs before the extension.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ora.contracts.official_limit import TOOL_BUDGET_LIMIT_MULTIPLE, limit_snapshot
from ora.environment.backends.robodojo.bridge import EvaluationLifecycle, RobodojoError
from ora.environment.backends.robodojo.round_verdict import OFFICIAL_LIMIT_BASIS, TERMINAL_BASIS
from ora.environment.backends.robodojo.session import RobodojoSession
from ora.environment.benchmarks.robodojo import physics_step_ceiling
from ora.orchestrator.backends.robodojo.evaluation import (
    ROBODOJO_VERDICT_SCHEMA, round_verdict_errors, verdict_untrusted,
)
from ora.orchestrator.reports.run import classify_ending
from tests.support.robodojo.backend import Plant, _fake_source_manifest


class _Rewards:
    def __init__(self) -> None:
        self.reward = 0.0
        self.final_check_list = [[]]
        self.final_score_list = [[]]
        self.final_checks: list[bool] = []

    def get_reward(self, final_check: bool = True) -> list[float]:
        self.final_checks.append(final_check)
        return [self.reward]


class _Lifecycle(EvaluationLifecycle):
    def __init__(self, *, multiple: int) -> None:  # the upstream task class owns the real one
        self.step_lim, self.harness_official_limit_multiple = 10, multiple
        self.reward_registered = True
        self.take_action_cnt, self.success, self.end_flag = [0], [True], [False]
        self._official_terminal = None
        self.reward_manager = _Rewards()

    def to(self, count: int):
        """Count up to ``count``, returning the terminal of the last tick."""
        terminal = None
        while self.take_action_cnt[0] < count and terminal is None:
            self.take_action_cnt[0] += 1
            terminal = self.advance_official_terminal()
        return terminal


def test_at_multiple_one_the_official_limit_is_the_stop_and_the_final_check_runs() -> None:
    plant = _Lifecycle(multiple=1)
    assert plant.native_action_ceiling == 10
    assert plant.to(10) == {"official_success": False, "native_actions": 10}
    assert plant.reward_manager.final_checks == [False] * 9 + [True]
    assert plant.official_limit_snapshot is None


def test_a_success_inside_the_limit_is_official_and_ends_the_episode() -> None:
    plant = _Lifecycle(multiple=3)
    plant.to(9)
    plant.reward_manager.reward = 1.0
    assert plant.to(10) == {"official_success": True, "native_actions": 10}
    assert plant.official_limit_snapshot is None


def test_the_limit_is_snapshotted_and_the_episode_runs_on() -> None:
    plant = _Lifecycle(multiple=3)
    assert plant.to(11) is None
    assert plant.official_limit_snapshot == limit_snapshot(10)
    assert plant.end_flag == [False] and plant.success == [True]
    # Nothing registered a final check, so the limit evaluated nothing new.
    assert not any(plant.reward_manager.final_checks)


def test_a_success_past_the_limit_never_becomes_the_official_result() -> None:
    plant = _Lifecycle(multiple=3)
    plant.to(12)
    plant.reward_manager.reward = 1.0
    terminal = plant.to(13)
    assert terminal == {"official_success": False, "native_actions": 13, "success_after_official_limit": True}
    assert plant.official_limit_snapshot == limit_snapshot(10)


def test_the_ceiling_ends_the_episode_with_no_success_and_runs_the_final_check() -> None:
    plant = _Lifecycle(multiple=3)
    assert plant.to(30) == {"official_success": False, "native_actions": 30, "success_after_official_limit": False}
    assert plant.reward_manager.final_checks[-1] is True


def test_a_failure_latch_past_the_limit_ends_the_episode() -> None:
    plant = _Lifecycle(multiple=3)
    plant.to(12)
    plant.success[0] = False
    assert plant.to(13)["success_after_official_limit"] is False


def test_a_task_with_a_final_check_is_decided_at_the_official_limit() -> None:
    plant = _Lifecycle(multiple=3)
    plant.reward_manager.final_check_list = [[("registered",)]]
    assert plant.to(10) == {"official_success": False, "native_actions": 10}
    assert plant.reward_manager.final_checks[-1] is True
    assert plant.official_limit_snapshot is None


def test_a_native_tick_past_the_ceiling_is_refused_not_the_official_limit() -> None:
    plant = _Lifecycle(multiple=2)
    plant.take_action_cnt[0] = 20
    with pytest.raises(RobodojoError, match="native action limit"):
        plant.native_tick({})


def _session(tmp_path, **config) -> RobodojoSession:
    plant = Plant()
    plant.step_lim = 200
    instance = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
                "task": "align_blocks", "layout_id": 0, **config},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (
            plant, SimpleNamespace(close=lambda: None), _fake_source_manifest(tmp_path),
        ),
    )
    instance._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    return instance


def test_the_session_bound_follows_the_multiple(tmp_path) -> None:
    session = _session(tmp_path, official_limit_multiple=3)
    assert (session.native_action_limit, session.extended_native_action_limit) == (200, 600)
    assert session.max_sim_steps == physics_step_ceiling("align_blocks", 3) == 6000
    with pytest.raises(RobodojoError, match="max_sim_steps"):
        _session(tmp_path, official_limit_multiple=3, max_sim_steps=6010)
    with pytest.raises(RobodojoError, match="max_sim_steps"):
        _session(tmp_path, max_sim_steps=2010)
    with pytest.raises(RobodojoError, match="official_limit_multiple"):
        _session(tmp_path, official_limit_multiple=1001)


def test_a_session_at_the_tool_budget_multiple_has_no_stop_a_run_reaches(tmp_path) -> None:
    session = _session(tmp_path, official_limit_multiple=TOOL_BUDGET_LIMIT_MULTIPLE)
    assert (session.native_action_limit, session.extended_native_action_limit) == (200, 200_000)
    assert session.max_sim_steps == physics_step_ceiling("align_blocks", TOOL_BUDGET_LIMIT_MULTIPLE) == 2_000_000


def _extended_verdict(tmp_path, *, multiple=3):
    session = _session(tmp_path, official_limit_multiple=multiple)
    session.environment = Plant()
    session.environment.take_action_cnt[0] = 412
    return session


def test_a_harness_stop_after_the_limit_keeps_the_official_result(tmp_path) -> None:
    session = _extended_verdict(tmp_path)
    session.environment.official_limit_snapshot = limit_snapshot(200)
    session.abort("robot_decision_budget")
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False
    assert verdict["terminal"] is True
    assert verdict["verdict_basis"] == OFFICIAL_LIMIT_BASIS
    assert verdict["round_end_basis"] == "harness_budget_exhausted"
    assert verdict["protocol_comparable"] is False
    assert verdict["upstream_ending_condition"] is None
    assert "after the official limit" in verdict["comparability_basis"]
    assert verdict["success_after_official_limit"] is False
    assert verdict["official_limit_snapshot"] == limit_snapshot(200)
    assert verdict["native_action_limit"] == verdict["official_native_action_limit"] == 200
    assert verdict["extended_native_action_limit"] == 600
    assert verdict_untrusted(verdict, {"official_success": False}) is False
    assert classify_ending({**verdict, "termination": "robot_decision_budget"}) == "decision_budget"


def test_a_give_up_past_the_limit_is_the_policy_s_ending_and_the_result_stands(tmp_path) -> None:
    session = _extended_verdict(tmp_path)
    session.environment.official_limit_snapshot = limit_snapshot(200)
    session.abort("policy_finished_episode")
    verdict = session.persist_official_verdict()
    assert (verdict["official_success"], verdict["terminal"]) == (False, True)
    assert verdict["verdict_basis"] == OFFICIAL_LIMIT_BASIS
    assert verdict["round_end_basis"] == "policy_finished_episode"
    assert verdict["protocol_comparable"] is False
    assert verdict["upstream_ending_condition"] is None
    assert verdict["official_limit_snapshot"] == limit_snapshot(200)
    assert verdict_untrusted(verdict, {"official_success": False}) is False
    assert classify_ending({**verdict, "termination": "policy_finished_episode"}) == "policy_finish"


def test_a_failure_latch_past_the_limit_is_not_the_native_action_limit(tmp_path) -> None:
    session = _extended_verdict(tmp_path)
    session.environment.official_limit_snapshot = limit_snapshot(200)
    session.official_terminal = {"official_success": False, "native_actions": 412}
    assert session.status()["reason"] == "official_failure_latch"
    verdict = session.persist_official_verdict()
    assert verdict["verdict_basis"] == TERMINAL_BASIS
    assert verdict["upstream_ending_condition"] == "official_failure_latch"
    assert classify_ending({**verdict, "state": "completed"}) == "failure_latch"


def test_only_the_extended_limit_is_the_native_action_limit(tmp_path) -> None:
    session = _extended_verdict(tmp_path)
    session.environment.official_limit_snapshot = limit_snapshot(200)
    session.environment.take_action_cnt[0] = 600
    session.official_terminal = {"official_success": False, "native_actions": 600,
                                 "success_after_official_limit": False}
    assert session.status()["reason"] == "official_native_action_limit"
    verdict = session.persist_official_verdict()
    assert verdict["round_end_basis"] == "official_terminal_transition"
    assert verdict["protocol_comparable"] is True
    assert verdict["upstream_ending_condition"] == "native_action_limit"


def test_a_late_success_is_recorded_apart_and_counts_as_no_official_success(tmp_path) -> None:
    session = _extended_verdict(tmp_path)
    session.environment.official_limit_snapshot = limit_snapshot(200)
    session.official_terminal = {"official_success": False, "native_actions": 412,
                                 "success_after_official_limit": True}
    assert session.status()["reason"] == "success_after_official_limit"
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False
    assert verdict["verdict_basis"] == TERMINAL_BASIS
    assert verdict["upstream_ending_condition"] == "success_after_official_limit"
    assert verdict["success_after_official_limit"] is True
    assert verdict["success_after_official_limit_native_actions"] == 412
    assert verdict_untrusted(verdict, {"official_success": False}) is False
    assert classify_ending({**verdict, "state": "completed"}) == "success_after_official_limit"


def test_a_fault_after_the_limit_stays_an_infrastructure_result(tmp_path) -> None:
    session = _extended_verdict(tmp_path)
    session.environment.official_limit_snapshot = limit_snapshot(200)
    session._fault("native_cadence_mismatch")
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is None
    assert verdict_untrusted(verdict, {"official_success": None}) is True


def test_a_snapshot_basis_without_its_snapshot_is_refused() -> None:
    verdict = {"official_success": False, "terminal": True, "verdict_basis": OFFICIAL_LIMIT_BASIS,
               "fault_reason": None}
    assert verdict_untrusted(verdict, {"official_success": False}) is True


def test_a_verdict_at_multiple_one_carries_no_extension_fields(tmp_path) -> None:
    session = _extended_verdict(tmp_path, multiple=1)
    session.abort("robot_decision_budget")
    verdict = session.persist_official_verdict()
    assert verdict["verdict_basis"] == "standing_official_predicate_at_policy_budget"
    assert "success_after_official_limit" not in verdict
    assert "official_limit_snapshot" not in verdict


def _verdict_errors(tmp_path, runtime: dict) -> list[str]:
    native = tmp_path / "native"
    native.mkdir(exist_ok=True)
    verdict = {"schema": ROBODOJO_VERDICT_SCHEMA, "source": "RoboDojo EvalEnv.is_episode_end",
               "source_manifest_sha256": "0" * 64, "round_index": 0, "task": "align_blocks",
               "official_success": False, "terminal": True, "verdict_basis": TERMINAL_BASIS,
               "fault_reason": None, "abort_reason": None, "native_actions": 412, "physics_steps": 4120,
               "reset_physics_steps": 0, "official_limit_snapshot": limit_snapshot(200)}
    (native / "official-verdict.json").write_text(json.dumps(verdict))
    record = {"verdict": verdict, "official_success": False}
    return round_verdict_errors(
        native, {"rounds": [record]}, runtime, {"task": "align_blocks"}, "0" * 64, rounds=1, completed=True,
        reset_clocks=[0], native_totals=[412], physics_totals=[4120],
    )


def test_the_evaluator_counts_native_actions_against_the_extended_limit(tmp_path) -> None:
    assert _verdict_errors(tmp_path, {"native_action_limit": 200, "official_limit_multiple": 3}) == []
    assert "robodojo_official_counter_mismatch:0" in _verdict_errors(tmp_path, {"native_action_limit": 200})
