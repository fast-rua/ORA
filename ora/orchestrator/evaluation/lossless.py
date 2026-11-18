"""Lossless-context evidence checks (``LOSSLESS_CONTEXT_RETENTION``).

Each policy-visible projection must keep its lossless invariants, advance its
externalization and semantic-compaction watermarks monotonically, hold the
expected semantic slots and exact tail, and anchor token estimates to recorded
provider usage. Semantic transitions must reference indexed exact
transcripts; the artifact index must belong to the one policy session;
provider requests must not expose host paths; and artifact provenance must be
complete. Each episode capsule must name its session round and time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ora.agent.memory.artifacts import CONTEXT_ARTIFACT_INDEX_SCHEMA
from ora.orchestrator.evaluation.artifact_provenance import (
    context_artifact_entry_errors, verify_artifact_provenance,
)
from ora.orchestrator.evaluation.context_coverage import coverage_violations
from ora.orchestrator.evaluation.evidence_io import EvaluationError, _load


@dataclass(frozen=True)
class LosslessContextEvidence:
    """Reasons and provenance report of one run."""

    reasons: list[str]
    artifact_provenance: dict[str, Any]


@dataclass(frozen=True)
class _ProjectionCursor:
    """Watermarks carried forward from the previous accepted projection."""

    epoch: Any = -1
    lifecycle: Any = -1
    pressure: Any = -1
    episode_through: Any = -1
    emergency_through: Any = -1
    episode_compactions: Any = 0
    emergency_compactions: Any = 0
    emergency_checkpoint_sha256: str | None = None
    emergency_exact_transcript_ref: Mapping[str, Any] | None = None
    #: Context v9's exchange frontiers (``exchange_aging``); they only grow.
    aged_exchanges: int = 0
    aged_observation_groups: int = 0
    reasoning_withheld_exchanges: int = 0
    #: An emergency checkpoint that counts in exchanges: the exchanges it covers; they only grow.
    emergency_exchanges: int = 0


#: Lossless (semantic_loss false) rewrites of Context v9 (exchange_layers.ts).
EXCHANGE_TRANSITIONS = frozenset({"exchange_aging", "reasoning_replay_window", "hard_backstop"})


@dataclass
class _SemanticTally:
    """Semantic transitions accumulated across the accepted projections."""

    episode_compactions: int = 0
    emergency_compactions: int = 0
    exact_transcript_refs: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _ProviderInputs:
    """Provider-reported input tokens keyed by call id and by wire request."""

    by_call: dict[str, Any]
    by_wire_request: dict[str, Any]



# A host path is "/home/<name>/" standing as a path, not the same characters
# inside model prose ("grasp-flag/home/center-release" invalidated a complete
# recorded session): the segment may not follow a letter, digit,
# underscore, dot or hyphen.
_HOST_HOME_PATH = re.compile(r"(?<![\w.\-])/home/[\w.\-]+/")


def host_path_visible(text: str, run_root: Path) -> bool:
    """Whether a provider request exposes this run's directory or a home path."""
    return str(run_root) in text or _HOST_HOME_PATH.search(text) is not None

def _lossless_context_evidence(run_root: Path) -> LosslessContextEvidence:
    """Check every lossless projection, transition, artifact and request."""

    reasons: list[str] = []
    try:
        lossless_events = [
            json.loads(line)
            for line in (run_root / "agent-events.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        artifact_index = _load(
            run_root / "native/context-artifacts/index.json"
        )
    except (EvaluationError, OSError, json.JSONDecodeError):
        lossless_events = []
        artifact_index = {}
        reasons.append("lossless_context_evidence_missing")
    projection_events = [
        event for event in lossless_events
        if event.get("type") == "context_projected"
        and isinstance(event.get("payload"), Mapping)
    ]
    provider_inputs = _provider_inputs(lossless_events, projection_events)
    if not projection_events:
        reasons.append("lossless_context_projection_missing")
    sequence_reasons, cursor, tally = _projection_sequence_reasons(
        projection_events, provider_inputs,
    )
    reasons.extend(sequence_reasons)
    if cursor.episode_compactions != tally.episode_compactions:
        reasons.append("episode_compaction_count_mismatch")
    if cursor.emergency_compactions != tally.emergency_compactions:
        reasons.append("emergency_compaction_count_mismatch")
    index_reasons, artifacts = _artifact_index_reasons(
        run_root, artifact_index, lossless_events, tally.exact_transcript_refs,
    )
    reasons.extend(index_reasons)
    reasons.extend(_host_path_reasons(run_root, lossless_events))
    artifact_provenance_check = verify_artifact_provenance(
        run_root, lossless_events, artifacts,
    )
    reasons.extend(artifact_provenance_check["errors"])
    if artifact_provenance_check["evidence_gaps"]:
        reasons.append("context_artifact_provenance_insufficient")
    return LosslessContextEvidence(
        reasons=reasons,
        artifact_provenance=artifact_provenance_check,
    )


def _provider_inputs(
    lossless_events: list[Any], projection_events: list[Any],
) -> _ProviderInputs:
    provider_input_by_call = {
        str(event["model_call_id"]): event["payload"]["usage"][
            "input_tokens"
        ]
        for event in lossless_events
        if event.get("type") == "provider_response"
        and isinstance(event.get("model_call_id"), str)
        and isinstance(event.get("payload"), Mapping)
        and isinstance(event["payload"].get("usage"), Mapping)
        and isinstance(
            event["payload"]["usage"].get("input_tokens"), int,
        )
        and not isinstance(
            event["payload"]["usage"].get("input_tokens"), bool,
        )
    }
    provider_input_by_wire_request = {
        str(event["payload"]["provider_wire_request_sha256"]): (
            provider_input_by_call[str(event["model_call_id"])]
        )
        for event in projection_events
        if isinstance(event.get("model_call_id"), str)
        and str(event["model_call_id"]) in provider_input_by_call
        and isinstance(
            event["payload"].get("provider_wire_request_sha256"), str,
        )
    }
    return _ProviderInputs(
        by_call=provider_input_by_call,
        by_wire_request=provider_input_by_wire_request,
    )


def _projection_sequence_reasons(
    projection_events: list[Any], provider_inputs: _ProviderInputs,
) -> tuple[list[str], _ProjectionCursor, _SemanticTally]:
    """Walk projections in order; the first malformed one stops the walk."""

    reasons: list[str] = []
    cursor = _ProjectionCursor()
    tally = _SemanticTally()
    for projection_event in projection_events:
        projection = projection_event["payload"]
        if not _projection_watermarks_valid(
            projection, cursor, provider_inputs.by_wire_request,
        ):
            reasons.append("lossless_context_projection_invalid")
            break
        reasons.extend(_projection_reasons(
            projection_event, projection, cursor, provider_inputs, tally,
        ))
        cursor = _advance_cursor(cursor, projection)
    return reasons, cursor, tally


def _projection_watermarks_valid(
    projection: Mapping[str, Any], cursor: _ProjectionCursor,
    provider_input_by_wire_request: Mapping[str, Any],
) -> bool:
    """Schema, monotonic watermarks and provider-usage anchor of one projection."""

    epoch = projection.get("projection_epoch")
    lifecycle_through = projection.get(
        "lifecycle_media_externalized_through_action_step"
    )
    pressure_through = projection.get(
        "pressure_externalized_through_action_step"
    )
    episode_through = projection.get(
        "episode_checkpointed_through_action_step"
    )
    emergency_through = projection.get(
        "emergency_summarized_through_action_step"
    )
    episode_compactions = projection.get("episode_checkpoint_count")
    emergency_compactions = projection.get(
        "emergency_compaction_count"
    )
    inventory = projection.get("action_step_inventory")
    projection_transitions = projection.get("projection_transitions")
    token_estimate_status = projection.get("token_estimate_status")
    anchor_applies = projection.get(
        "provider_usage_anchor_applies_to_projection_epoch"
    )
    anchor_epoch = projection.get(
        "provider_usage_anchor_projection_epoch"
    )
    anchor_hash = projection.get(
        "provider_usage_anchor_wire_request_sha256"
    )
    current_wire_hash = projection.get(
        "provider_wire_request_sha256"
    )
    exchange = projection.get("exchange_aging")
    if isinstance(exchange, Mapping) and not all(
        type(exchange.get(field)) is int and exchange[field] >= getattr(cursor, field)
        for field in ("aged_exchanges", "aged_observation_groups", "reasoning_withheld_exchanges")
    ):
        return False
    coverage = projection.get("emergency_exchange_coverage")
    if coverage is not None and not (
        isinstance(coverage, Mapping)
        and type(coverage.get("summarized_exchanges")) is int
        and coverage["summarized_exchanges"] >= cursor.emergency_exchanges
    ):
        return False
    return not (
        projection.get("schema")
        != "ora.native-context-projection.v6"
        or isinstance(epoch, bool) or not isinstance(epoch, int)
        or epoch < cursor.epoch
        or cursor.epoch >= 0 and epoch > cursor.epoch + 1
        or lifecycle_through is not None
        and (
            isinstance(lifecycle_through, bool)
            or not isinstance(lifecycle_through, int)
            or lifecycle_through < cursor.lifecycle
        )
        or pressure_through is not None
        and (
            isinstance(pressure_through, bool)
            or not isinstance(pressure_through, int)
            or pressure_through < cursor.pressure
        )
        or episode_through is not None
        and (
            isinstance(episode_through, bool)
            or not isinstance(episode_through, int)
            or episode_through < cursor.episode_through
        )
        or emergency_through is not None
        and (
            isinstance(emergency_through, bool)
            or not isinstance(emergency_through, int)
            or emergency_through < cursor.emergency_through
        )
        or isinstance(episode_compactions, bool)
        or not isinstance(episode_compactions, int)
        or episode_compactions < cursor.episode_compactions
        or isinstance(emergency_compactions, bool)
        or not isinstance(emergency_compactions, int)
        or emergency_compactions < cursor.emergency_compactions
        or projection.get("lossless_invariants_before_sha256")
        != projection.get("lossless_invariants_after_sha256")
        or not isinstance(inventory, list)
        or not inventory
        or not isinstance(projection_transitions, list)
        or token_estimate_status not in {
            "provider_anchored_suffix_estimate",
            "projection_epoch_unanchored",
            "provider_usage_unavailable",
        }
        or not isinstance(anchor_applies, bool)
        or not isinstance(current_wire_hash, str)
        or len(current_wire_hash) != 64
        or anchor_applies and (
            anchor_epoch != epoch
            or not isinstance(anchor_hash, str)
            or len(anchor_hash) != 64
            or provider_input_by_wire_request.get(anchor_hash)
            != projection.get("provider_usage_anchor_tokens")
        )
    )


def _projection_reasons(
    projection_event: Mapping[str, Any], projection: Mapping[str, Any],
    cursor: _ProjectionCursor, provider_inputs: _ProviderInputs,
    tally: _SemanticTally,
) -> list[str]:
    """Slots, exact tail, coverage and transitions of one accepted projection."""

    reasons: list[str] = []
    epoch = projection.get("projection_epoch")
    episode_through = projection.get(
        "episode_checkpointed_through_action_step"
    )
    emergency_through = projection.get(
        "emergency_summarized_through_action_step"
    )
    episode_compactions = projection.get("episode_checkpoint_count")
    checkpoint_slot_count = projection.get("checkpoint_slot_count")
    emergency_checkpoint_sha256 = projection.get(
        "emergency_checkpoint_sha256"
    )
    emergency_exact_transcript_ref = projection.get(
        "emergency_exact_transcript_ref"
    )
    inventory = projection.get("action_step_inventory")
    projection_transitions = projection.get("projection_transitions")
    episode_coverage_count = (
        episode_through + 1 if episode_through is not None else 0
    )
    emergency_coverage_count = (
        emergency_through + 1 if emergency_through is not None else 0
    )
    exchange_tail = projection.get("exact_tail")
    exchange_unit = isinstance(exchange_tail, Mapping) and exchange_tail.get("unit") == "exchange"
    active_emergency_projection = (
        _exchange_checkpoint_active(projection, episode_coverage_count) if exchange_unit
        else emergency_coverage_count > episode_coverage_count
    )
    expected_semantic_slots = (
        1 if active_emergency_projection else episode_compactions
    )
    if (
        checkpoint_slot_count != expected_semantic_slots
        or active_emergency_projection and (
            not isinstance(emergency_checkpoint_sha256, str)
            or len(emergency_checkpoint_sha256) != 64
            or not isinstance(emergency_exact_transcript_ref, Mapping)
            or not isinstance(
                emergency_exact_transcript_ref.get("ref"), str,
            )
        )
        or not active_emergency_projection and (
            emergency_checkpoint_sha256 is not None
            or emergency_exact_transcript_ref is not None
        )
    ):
        reasons.append("semantic_projection_slot_state_invalid")
    completed = sum(
        item.get("complete") is True
        for item in inventory if isinstance(item, Mapping)
    )
    expected_tail = min(
        3,
        max(
            0,
            completed
            - max(
                episode_through + 1 if episode_through is not None else 0,
                emergency_through + 1 if emergency_through is not None else 0,
            ),
        ),
    )
    if exchange_unit:
        reasons.extend(_exchange_tail_reasons(projection))
    elif (
        projection.get("retained_exact_causal_action_steps")
        != expected_tail
    ):
        reasons.append("lossless_context_exact_tail_invalid")
    reasons.extend(coverage_violations(projection, inventory))
    if cursor.epoch >= 0 and (
        epoch == cursor.epoch and projection_transitions
        or epoch == cursor.epoch + 1 and (
            not projection_transitions
            or any(
                not isinstance(item, Mapping)
                or item.get("from_epoch") != cursor.epoch
                or item.get("to_epoch") != epoch
                for item in projection_transitions
            )
        )
    ):
        reasons.append("lossless_context_epoch_transition_invalid")
    for transition in projection_transitions:
        reasons.extend(_projection_transition_reasons(
            transition, projection_event, cursor, provider_inputs, tally,
            active_emergency_projection=active_emergency_projection,
            episode_coverage_count=episode_coverage_count,
            emergency_coverage_count=emergency_coverage_count,
        ))
    return reasons


def _exchange_checkpoint_active(projection: Mapping[str, Any], episode_coverage_count: int) -> bool:
    """An emergency checkpoint that counts in exchanges is in the request when it covers an exchange that the
    Episode capsules do not (one whose last ActionStep is at or after their coverage)."""

    coverage = projection.get("emergency_exchange_coverage")
    inventory = (projection.get("exact_tail") or {}).get("exchange_inventory")
    if not isinstance(coverage, Mapping) or not isinstance(inventory, list):
        return False
    summarized, kept = coverage.get("summarized_exchanges"), coverage.get("kept_exchange")
    if type(summarized) is not int or summarized <= 0 or summarized > len(inventory):
        return False
    return any(
        isinstance(item, Mapping) and isinstance(item.get("action_steps"), list)
        and item["action_steps"][-1] >= episode_coverage_count
        for index, item in enumerate(inventory[:summarized]) if index != kept
    )


def _exchange_tail_reasons(projection: Mapping[str, Any]) -> list[str]:
    """An exact tail in exchanges: the latest ``exchanges`` exchanges and the anchor, the exchange where the current
    observation arrived; an emergency checkpoint in exchanges covers only exchanges before it, the one it kept
    excepted. Once there is an exchange, the inventory's images are the ActionStep inventory's: exchange 0 starts
    at the opening observation. Before the first assistant message there is none (``total_exchanges`` 0, the
    request of a run's first call), and the opening observation's images belong to no exchange yet; the exact
    tail then holds the whole history after the prefix."""

    tail = projection["exact_tail"]
    total, count, start = tail.get("total_exchanges"), tail.get("exchanges"), tail.get("from_exchange")
    anchor, inventory = tail.get("anchor_exchange"), tail.get("exchange_inventory")
    steps = projection.get("action_step_inventory") or []
    if (
        type(total) is not int or type(count) is not int or count < 1 or total < 0
        or start != max(0, total - count)
        or (anchor is None) != (total == 0)
        or anchor is not None and (type(anchor) is not int or not 0 <= anchor < total)
        or not isinstance(inventory, list) or len(inventory) != total
        or any(not isinstance(item, Mapping) or item.get("exchange_index") != index
               or type(item.get("inline_images")) is not int or item["inline_images"] < 0
               for index, item in enumerate(inventory))
        or total > 0 and sum(item["inline_images"] for item in inventory)
        != sum(item.get("inline_images", 0) for item in steps if isinstance(item, Mapping))
        or type(projection.get("retained_exact_causal_action_steps")) is not int
    ):
        return ["lossless_context_exact_tail_invalid"]
    coverage = projection.get("emergency_exchange_coverage")
    if coverage is None:
        return []
    summarized, kept = coverage.get("summarized_exchanges"), coverage.get("kept_exchange")
    if (
        coverage.get("unit") != "exchange" or type(summarized) is not int or not 0 <= summarized <= start
        or kept is not None and (type(kept) is not int or not 0 <= kept < summarized)
    ):
        return ["lossless_context_exact_tail_invalid"]
    return []


def _projection_transition_reasons(
    transition: Any, projection_event: Mapping[str, Any],
    cursor: _ProjectionCursor, provider_inputs: _ProviderInputs,
    tally: _SemanticTally, *, active_emergency_projection: bool,
    episode_coverage_count: int, emergency_coverage_count: int,
) -> list[str]:
    """Check one projection transition and tally its semantic compaction."""

    if not isinstance(transition, Mapping):
        return ["lossless_context_transition_invalid"]
    reasons: list[str] = []
    kind = transition.get("kind")
    if (
        transition.get("schema")
        != "ora.context-projection-transition.v4"
        or kind not in {
            "lifecycle_read_only_externalization",
            "lifecycle_media_externalization",
            "pressure_externalization",
            "episode_capsule",
            "emergency_compaction",
            *EXCHANGE_TRANSITIONS,
        }
    ):
        reasons.append("lossless_context_transition_invalid")
    if kind in EXCHANGE_TRANSITIONS:
        management = projection_event.get("payload", {}).get("context_management")
        return [*reasons, *_exchange_transition_reasons(transition, kind, provider_inputs, management)]
    if kind in {"episode_capsule", "emergency_compaction"}:
        exact_transcript = transition.get("exact_transcript")
        exact_transcript_ref = (
            exact_transcript.get("ref")
            if isinstance(exact_transcript, Mapping) else None
        )
        if isinstance(exact_transcript_ref, str):
            tally.exact_transcript_refs.add(exact_transcript_ref)
        else:
            reasons.append("semantic_exact_transcript_ref_invalid")
        if kind == "episode_capsule":
            tally.episode_compactions += 1
            reasons.extend(_episode_capsule_reasons(
                transition, projection_event, cursor, tally,
                active_emergency_projection=active_emergency_projection,
                episode_coverage_count=episode_coverage_count,
                emergency_coverage_count=emergency_coverage_count,
            ))
        else:
            tally.emergency_compactions += 1
        if not _semantic_watermark_valid(
            transition, kind, projection_event, provider_inputs,
        ):
            reasons.append("semantic_layer_watermark_invalid")
    return reasons


def _episode_capsule_reasons(
    transition: Mapping[str, Any], projection_event: Mapping[str, Any],
    cursor: _ProjectionCursor, tally: _SemanticTally, *,
    active_emergency_projection: bool, episode_coverage_count: int,
    emergency_coverage_count: int,
) -> list[str]:
    """An episode capsule may supersede the previous emergency checkpoint."""

    reasons: list[str] = []
    superseded = transition.get(
        "superseded_emergency_checkpoint"
    )
    if superseded is not None and (
        not isinstance(superseded, Mapping)
        or "superseded_overlapping_emergency_checkpoint"
        not in transition.get("reasons", [])
        or superseded.get("checkpoint_sha256")
        != cursor.emergency_checkpoint_sha256
        or superseded.get("exact_transcript")
        != cursor.emergency_exact_transcript_ref
        or active_emergency_projection
        or episode_coverage_count
        < emergency_coverage_count
    ):
        reasons.append(
            "episode_emergency_takeover_invalid"
        )
    capsule_round = transition.get("session_round")
    if (
        isinstance(capsule_round, int)
        and not isinstance(capsule_round, bool)
        and capsule_round > 0
    ):
        capsule_unix_s = projection_event.get("unix_s")
        if not isinstance(capsule_unix_s, (int, float)) or isinstance(
            capsule_unix_s, bool,
        ):
            reasons.append("episode_capsule_time_invalid")
    else:
        reasons.append("episode_capsule_round_invalid")
    return reasons


def _exchange_transition_reasons(
    transition: Mapping[str, Any], kind: Any, provider_inputs: _ProviderInputs, management: Any,
) -> list[str]:
    """A v9 rewrite is lossless and only grows its frontiers; an aging cites the usage at or above its
    watermark and ages at least one batch (``exact_tail_exchanges``) of exchanges, so it never rewrites every call."""

    counts = [transition.get(field) for field in (
        "aged_exchanges_before", "aged_exchanges_after",
        "reasoning_withheld_exchanges_before", "reasoning_withheld_exchanges_after",
        "aged_observation_groups_before", "aged_observation_groups_after",
    )]
    if (
        transition.get("semantic_loss") is not False
        or any(type(value) is not int for value in counts)
        or counts[1] < counts[0] or counts[3] < counts[2] or counts[5] < counts[4]
        or counts[1] == counts[0] and counts[3] == counts[2] and counts[5] == counts[4]
    ):
        return ["lossless_context_exchange_transition_invalid"]
    if kind != "exchange_aging":
        return []
    reported = transition.get("admission_provider_input_tokens")
    admission_hash = transition.get("admission_provider_wire_request_sha256")
    batch = management.get("exact_tail_exchanges") if isinstance(management, Mapping) else None
    if (
        type(batch) is not int
        or transition.get("batch_exchanges") != batch
        or counts[1] - counts[0] < batch
        or type(reported) is not int
        or type(transition.get("watermark_tokens")) is not int
        or reported < transition["watermark_tokens"]
        or transition.get("admission_provider_projection_epoch") != transition.get("from_epoch")
        or not isinstance(admission_hash, str)
        or provider_inputs.by_wire_request.get(admission_hash) != reported
    ):
        return ["exchange_aging_admission_invalid"]
    return []


def _semantic_watermark_valid(
    transition: Mapping[str, Any], kind: Any,
    projection_event: Mapping[str, Any], provider_inputs: _ProviderInputs,
) -> bool:
    """Semantic admission must cite the provider usage that triggered it."""

    reported = transition.get(
        "semantic_admission_provider_input_tokens"
    )
    admission_epoch = transition.get(
        "semantic_admission_provider_projection_epoch"
    )
    admission_hash = transition.get(
        "semantic_admission_provider_wire_request_sha256"
    )
    model_call_id = projection_event.get("model_call_id")
    post_summary_reported = (
        provider_inputs.by_call.get(str(model_call_id))
        if isinstance(model_call_id, str) else None
    )
    return not (
        isinstance(reported, bool)
        or not isinstance(reported, int)
        or reported < transition.get(
            "semantic_high_watermark_tokens", 0
        )
        or admission_epoch != transition.get("from_epoch")
        or not isinstance(admission_hash, str)
        or len(admission_hash) != 64
        or provider_inputs.by_wire_request.get(admission_hash)
        != reported
        or kind == "emergency_compaction" and (
            transition.get("target_validation")
            != "provider_pending"
            or isinstance(post_summary_reported, bool)
            or not isinstance(post_summary_reported, int)
        )
    )


def _advance_cursor(
    cursor: _ProjectionCursor, projection: Mapping[str, Any],
) -> _ProjectionCursor:
    """Carry this projection's watermarks to the next one."""

    lifecycle_through = projection.get(
        "lifecycle_media_externalized_through_action_step"
    )
    pressure_through = projection.get(
        "pressure_externalized_through_action_step"
    )
    episode_through = projection.get(
        "episode_checkpointed_through_action_step"
    )
    emergency_through = projection.get(
        "emergency_summarized_through_action_step"
    )
    emergency_checkpoint_sha256 = projection.get(
        "emergency_checkpoint_sha256"
    )
    emergency_exact_transcript_ref = projection.get(
        "emergency_exact_transcript_ref"
    )
    return _ProjectionCursor(
        epoch=projection.get("projection_epoch"),
        lifecycle=(
            cursor.lifecycle
            if lifecycle_through is None else lifecycle_through
        ),
        pressure=(
            cursor.pressure
            if pressure_through is None else pressure_through
        ),
        episode_through=(
            cursor.episode_through
            if episode_through is None else episode_through
        ),
        emergency_through=(
            cursor.emergency_through
            if emergency_through is None else emergency_through
        ),
        episode_compactions=projection.get("episode_checkpoint_count"),
        emergency_compactions=projection.get("emergency_compaction_count"),
        emergency_checkpoint_sha256=(
            emergency_checkpoint_sha256
            if isinstance(emergency_checkpoint_sha256, str) else None
        ),
        emergency_exact_transcript_ref=(
            emergency_exact_transcript_ref
            if isinstance(emergency_exact_transcript_ref, Mapping)
            else None
        ),
        **_exchange_cursor(cursor, projection),
        emergency_exchanges=(
            projection["emergency_exchange_coverage"]["summarized_exchanges"]
            if isinstance(projection.get("emergency_exchange_coverage"), Mapping) else cursor.emergency_exchanges
        ),
    )


def _exchange_cursor(cursor: _ProjectionCursor, projection: Mapping[str, Any]) -> dict[str, int]:
    fields = ("aged_exchanges", "aged_observation_groups", "reasoning_withheld_exchanges")
    exchange = projection.get("exchange_aging")
    if not isinstance(exchange, Mapping):
        return {field: getattr(cursor, field) for field in fields}
    return {field: exchange[field] for field in fields}


def _artifact_index_reasons(
    run_root: Path, artifact_index: Mapping[str, Any],
    lossless_events: list[Any], exact_transcript_refs: set[str],
) -> tuple[list[str], Any]:
    """The index must belong to the one session and hold every exact transcript."""

    reasons: list[str] = []
    artifacts = artifact_index.get("artifacts")
    policy_session_ids = {
        event.get("policy_session_id") for event in lossless_events
        if isinstance(event.get("policy_session_id"), str)
    }
    if (
        artifact_index.get("schema") != CONTEXT_ARTIFACT_INDEX_SCHEMA
        or len(policy_session_ids) != 1
        or artifact_index.get("policy_session_id")
        not in policy_session_ids
        or not isinstance(artifacts, Mapping)
    ):
        reasons.append("context_artifact_index_invalid")
        artifacts = {}
    reasons.extend(context_artifact_entry_errors(run_root, artifact_index, artifacts))
    known_refs = set(artifacts)
    if not exact_transcript_refs.issubset(known_refs):
        reasons.append("semantic_exact_transcript_ref_unknown")
    elif any(
        artifacts[logical_ref].get("media_type") != "application/json"
        for logical_ref in exact_transcript_refs
    ):
        reasons.append("semantic_exact_transcript_mime_invalid")
    return reasons, artifacts


def _host_path_reasons(run_root: Path, lossless_events: list[Any]) -> list[str]:
    """No recorded provider request may expose a host path to the model."""

    reasons: list[str] = []
    for event in lossless_events:
        if event.get("type") != "provider_request":
            continue
        artifact_ref = event.get("artifact_ref")
        relative = (
            artifact_ref.get("path")
            if isinstance(artifact_ref, Mapping) else None
        )
        request_path = (
            (run_root / relative).resolve()
            if isinstance(relative, str) else None
        )
        if request_path is None or not request_path.is_file():
            continue
        request_text = request_path.read_text(encoding="utf-8")
        if host_path_visible(request_text, run_root):
            reasons.append("host_path_visible_in_lossless_context")
    return reasons
