"""The official verdict of one RoboDojo round, and where its Boolean came from."""
from __future__ import annotations

import time
from typing import Any

from ora.contracts.official_limit import verdict_fields as official_limit_verdict_fields
from ora.environment.benchmarks.robodojo import UPSTREAM_ENDING_CONDITIONS, classify_round_ending

from . import layout_rounds
from .common import write_json

VERDICT_SCHEMA = "ora.robodojo-official-verdict.v2"
TERMINAL_BASIS = "official_terminal_transition"
OFFICIAL_LIMIT_BASIS = "official_result_at_official_limit"
POLICY_BUDGET_BASIS = "standing_official_predicate_at_policy_budget"
INFRASTRUCTURE_BASIS = "infrastructure_abort"
RUNNING_BASIS = "episode_incomplete"


class RobodojoVerdictMixin:
    """``RobodojoSession``'s verdict: read at every boundary, written once a round ends."""

    def _official_outcome(self) -> tuple[bool | None, bool, str]:
        """Resolve the official Boolean, its terminal claim and its provenance.

        RoboDojo latches success inside ``RewardManager.step``, which the
        wrapper runs after every native action, and no physics advances
        outside a native tick. So when a policy-side budget stops a round
        without an official terminal transition, the standing official
        predicate was already evaluated after the last physics step and was
        not a success. Reporting it is the same rule LIBERO applies, and it
        avoids a forced ``get_reward(final_check=True)``, which writes
        ``_gated_score_lst`` and can latch ``env.success`` false, is called
        two or three times inside the upstream step loop rather than once at
        the end of an episode, and for every shipped task evaluates nothing
        new.
        """
        if self.official_terminal is not None:
            return bool(self.official_terminal["official_success"]), True, TERMINAL_BASIS
        if self.fault_reason is not None:
            return None, False, INFRASTRUCTURE_BASIS
        if self.boundary_reason is not None or self.sim_steps >= self.max_sim_steps:
            # Past the official limit the official result is settled, and a
            # Harness budget that stops the extension does not change it.
            if self._official_limit_snapshot() is not None:
                return False, True, OFFICIAL_LIMIT_BASIS
            return False, False, POLICY_BUDGET_BASIS
        return None, False, RUNNING_BASIS

    def _official_limit_snapshot(self) -> dict[str, Any] | None:
        return getattr(self.environment, "official_limit_snapshot", None)

    def persist_official_verdict(self) -> dict[str, Any]:
        official_success, terminal, basis = self._official_outcome()
        native_actions = (
            int(self.environment.take_action_cnt[0])
            if self.environment is not None else 0
        )
        # ``verdict_basis`` says where the Boolean came from. It does not say
        # whether the round is comparable with an upstream rollout: a policy
        # ``finish_episode`` and an exhausted decision budget both arrive as
        # ``standing_official_predicate_at_policy_budget`` and only the first
        # is an ending upstream has no counterpart for. The comparability
        # fields answer that separately, so no reader has to infer it from
        # ``abort_reason``. Only the simulator's own transition is an official
        # ending: a give-up or a budget past the official limit claims
        # ``terminal`` for the result settled there, and is still a stop of ours.
        ending = classify_round_ending(
            official_terminal=self.official_terminal is not None,
            fault_reason=self.fault_reason,
            abort_reason=self.status()["reason"],
            official_success=official_success,
            success_after_official_limit=(self.official_terminal or {}).get("success_after_official_limit") is True,
            official_limit_passed=self._official_limit_snapshot() is not None,
            native_actions=native_actions,
            extended_native_action_limit=self.extended_native_action_limit,
        )
        verdict = {
            "schema": VERDICT_SCHEMA,
            "official_success": official_success,
            "terminal": terminal, "verdict_basis": basis, "fault_reason": self.fault_reason,
            **ending,
            "upstream_ending_conditions": list(UPSTREAM_ENDING_CONDITIONS),
            "native_action_limit": self.native_action_limit,
            "task": self.task, "eval_seed": self.eval_seed_selected, "seed": 0, "layout_id": self.layout_id, "round_index": self.round_index,
            "native_actions": native_actions,
            "physics_steps": self.sim_steps, "reset_physics_steps": self.reset_physics_steps,
            "abort_reason": self.abort_reason, "source_manifest_sha256": self.source_hash,
            "source": "RoboDojo EvalEnv.is_episode_end", "created_unix_s": time.time(),
            **layout_rounds.verdict_fields(self),
            **official_limit_verdict_fields(
                official_limit=self.native_action_limit, multiple=self.official_limit_multiple,
                snapshot=self._official_limit_snapshot(),
                success_native_actions=(
                    native_actions if (self.official_terminal or {}).get("success_after_official_limit") else None
                ),
            ),
        }
        self._capture_execution_boundary("terminal")
        write_json(self.output_root / "official-verdict.json", verdict)
        return verdict


__all__ = ["OFFICIAL_LIMIT_BASIS", "RobodojoVerdictMixin", "TERMINAL_BASIS", "VERDICT_SCHEMA"]
