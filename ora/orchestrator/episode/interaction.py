"""Orchestrator-owned causal state for one environment/policy episode."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


INTERACTION_STATE_SCHEMA = "ora.episode-interaction-state.v1"
InteractionPhase = Literal[
    "created",
    "starting",
    "observation_ready",
    "policy_active",
    "mutation_proposed",
    "executing",
    "fresh_observation_required",
    "world_state_unknown",
    "awaiting_instruction",
    "terminal",
    "aborted",
    "closed",
]


class InteractionStateError(RuntimeError):
    """An outer-loop transition would violate causal ordering."""


@dataclass
class EpisodeInteractionState:
    run_id: str
    evidence_path: Path | None = None
    phase: InteractionPhase = "created"
    observation_id: str | None = None
    observation_step: int | None = None
    pending_action_id: str | None = None
    pending_receipt_id: str | None = None
    fresh_observation_required: bool = False
    world_state_certainty: Literal["unchanged", "changed", "unknown"] = "unchanged"
    decisions: int = 0
    refused_proposals: int = 0
    model_calls: int = 0
    terminal_reason: str | None = None
    sequence: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)
    _policy_return_phase: InteractionPhase | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise InteractionStateError("interaction state requires a run id")
        if self.evidence_path is not None:
            self.evidence_path = self.evidence_path.resolve()
        self._persist()

    def _transition(self, target: InteractionPhase, *, reason: str) -> None:
        source = self.phase
        self.phase = target
        self.sequence += 1
        self.transitions.append({
            "sequence": self.sequence,
            "from": source,
            "to": target,
            "reason": reason,
            "time": time.time(),
        })
        self._persist()

    def _require(self, *phases: InteractionPhase) -> None:
        if self.phase not in phases:
            raise InteractionStateError(
                f"cannot transition from {self.phase}; expected one of {phases}"
            )

    def start(self) -> None:
        self._require("created")
        self._transition("starting", reason="episode_start")

    def publish_observation(self, *, observation_id: str, step: int) -> None:
        self._require("starting", "fresh_observation_required", "world_state_unknown", "awaiting_instruction")
        if not observation_id or step < 0:
            raise InteractionStateError("fresh observation identity is invalid")
        self.observation_id = observation_id
        self.observation_step = step
        self.pending_action_id = None
        self.pending_receipt_id = None
        self.fresh_observation_required = False
        self.world_state_certainty = "unchanged"
        self._transition("observation_ready", reason="fresh_observation")

    def begin_policy(self, *, terminal_lifecycle: bool = False) -> None:
        if terminal_lifecycle:
            self._require("terminal")
            self._policy_return_phase = "terminal"
        else:
            self._require("observation_ready")
            self._policy_return_phase = None
        self._transition("policy_active", reason=(
            "terminal_policy_lifecycle" if terminal_lifecycle else "policy_turn"
        ))

    def record_policy_outcome(
        self,
        *,
        kind: Literal["action", "no_action", "lifecycle"],
        model_calls: int,
        action_id: str | None = None,
        reason: str | None = None,
        terminal: bool = True,
    ) -> None:
        self._require("policy_active")
        if model_calls < 0:
            raise InteractionStateError("model-call increment cannot be negative")
        self.model_calls += model_calls
        if self._policy_return_phase == "terminal":
            if kind == "action":
                raise InteractionStateError("terminal lifecycle cannot propose mutation")
            self._policy_return_phase = None
            self._transition("terminal", reason=reason or "terminal_policy_complete")
            return
        if kind == "action":
            if not action_id:
                raise InteractionStateError("mutation proposal requires an action id")
            self.pending_action_id = action_id
            self._transition("mutation_proposed", reason="policy_action")
            return
        if not terminal:
            self._transition(
                "observation_ready",
                reason=reason or "recoverable_policy_no_action",
            )
            return
        self.terminal_reason = reason or "policy_no_action"
        self._transition("terminal", reason=self.terminal_reason)

    def await_instruction(self, *, reason: str) -> None:
        """Free play: the current instruction has ended; the world waits for the operator.

        The next instruction starts from a fresh observation of the same scene.
        """
        self._require("starting", "observation_ready", "fresh_observation_required")
        self._transition("awaiting_instruction", reason=reason)

    def begin_instruction(self) -> None:
        """Free play: the next instruction starts on the current observation (its fresh look failed)."""
        self._require("awaiting_instruction")
        if self.observation_id is None:
            raise InteractionStateError("an instruction needs a published observation")
        self._transition("observation_ready", reason="operator_instruction")

    def restart_scene(self) -> None:
        """Free play: the operator asked for a new scene; its first observation follows the reset."""
        self._require("starting", "awaiting_instruction")
        self.observation_id = None
        self.observation_step = None
        self.fresh_observation_required = True
        self.world_state_certainty = "unknown"
        self._transition("starting", reason="play_scene_reset")

    def record_model_calls(self, count: int) -> None:
        """Record provider work when a budget boundary rejects the outcome."""

        self._require("policy_active")
        if count < 0:
            raise InteractionStateError("model-call increment cannot be negative")
        self.model_calls += count
        self._persist()

    def begin_execution(self) -> None:
        self._require("mutation_proposed")
        self._transition("executing", reason="mutation_dispatch")

    def record_execution(
        self, *, receipt_id: str | None, consumed_decision: bool = True,
        world_changed: bool | None = None, refused: bool | None = None,
    ) -> None:
        """Close one dispatched proposal; a certified zero-write refusal spends
        no robot decision but still requires a fresh observation."""
        self._require("executing")
        if consumed_decision:
            self.decisions += 1
        was_refused = not consumed_decision if refused is None else refused
        if was_refused:
            self.refused_proposals += 1
        self.pending_receipt_id = receipt_id
        self.fresh_observation_required = True
        changed = consumed_decision if world_changed is None else world_changed
        self.world_state_certainty = "changed" if changed else "unchanged"
        self._transition("fresh_observation_required", reason="mutation_receipt")

    def mark_world_state_unknown(self, *, reason: str) -> None:
        self._require("executing", "fresh_observation_required")
        self.fresh_observation_required = True
        self.world_state_certainty = "unknown"
        self._transition("world_state_unknown", reason=reason)

    def mark_terminal(self, reason: str) -> None:
        self._require(
            "starting", "observation_ready", "policy_active", "mutation_proposed",
            "fresh_observation_required", "world_state_unknown", "terminal",
        )
        self.terminal_reason = reason
        if self.phase != "terminal":
            self._transition("terminal", reason=reason)
        else:
            self._persist()

    def abort(self, reason: str) -> None:
        if self.phase == "closed":
            raise InteractionStateError("closed interaction state cannot be aborted")
        self.terminal_reason = reason
        self._transition("aborted", reason=reason)

    def abort_unknown_execution(self, *, model_calls: int) -> None:
        """Settle a failed inline action from dispatch or publication, without resuming policy."""
        if self.phase == "closed" or type(model_calls) is not int or model_calls < 0:
            raise InteractionStateError("invalid unknown-execution settlement")
        self.model_calls += model_calls
        self.world_state_certainty = "unknown"
        self.fresh_observation_required = True
        self.abort("world_state_unknown")

    def close(self) -> None:
        if self.phase == "closed":
            return
        self._transition("closed", reason="episode_closed")

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": INTERACTION_STATE_SCHEMA,
            "run_id": self.run_id,
            "phase": self.phase,
            "observation_id": self.observation_id,
            "observation_step": self.observation_step,
            "pending_action_id": self.pending_action_id,
            "pending_receipt_id": self.pending_receipt_id,
            "fresh_observation_required": self.fresh_observation_required,
            "world_state_certainty": self.world_state_certainty,
            "decisions": self.decisions,
            "refused_proposals": self.refused_proposals,
            "model_calls": self.model_calls,
            "terminal_reason": self.terminal_reason,
            "sequence": self.sequence,
            "transitions": list(self.transitions),
        }

    def policy_view(self) -> Mapping[str, Any]:
        return {
            "phase": self.phase,
            "fresh_observation_required": self.fresh_observation_required,
        }

    def _persist(self) -> None:
        if self.evidence_path is None:
            return
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.evidence_path.with_suffix(self.evidence_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.evidence_path)
