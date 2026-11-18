// emergency_checkpoint (phase summarize, L6): everything before the exact tail
// becomes one bounded checkpoint, admitted by the Provider-reported input of
// the request before (README.md). It counts in ActionSteps or, with `unit:
// exchange`, in exchanges: the exchanges before the exact exchange tail, the
// anchor exchange (the latest robot action's) left exact.

import { jsonBytesOf } from "../../projection.ts";
import { PolicySessionError, clonePy, errorText, isInt, num, pyInt } from "../../pyutil.ts";
import type { SummaryWriter } from "../episode_capsule/capsule.ts";
import type { EpisodeAdmission, EpisodeCapsuleState } from "../episode_capsule/stage.ts";
import { EMERGENCY_CHECKPOINT, EPISODE_CAPSULE, SUPERSEDED_BY_EPISODE_CAPSULE, TRANSITION_SCHEMA } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { SegmentUnit } from "../services/segmenter.ts";
import type { Admission, ApplyContext, SlotContribution, Stage, Transition, UsageObservation } from "../stage.ts";
import { type ExchangeSelection, writeEmergencyCheckpoint, writeExchangeCheckpoint } from "./checkpoint.ts";

export interface EmergencyCheckpointParams {
  /** Absolute watermark in Provider input tokens, capped by the ratio; null: the ratio alone. */
  trigger_tokens: number | null;
  /** The watermark as a share of the hard input limit. */
  trigger_ratio: number;
  /** Telemetry: the next request's Provider input should stay under this share of the hard limit. */
  target_ratio: number;
  /** The checkpoint length asked of the compactor, and its soft maximum (estimated tokens). */
  checkpoint_target_tokens: number;
  checkpoint_max_tokens: number;
  /** false only when a context pack turns rollout compaction off: nothing is summarized, the record says why. */
  enabled: boolean;
  /** What it absorbs: ActionSteps (absent) or exchanges; the exact tail counts in the same unit. */
  unit?: SegmentUnit;
}

export interface EmergencyCheckpointState {
  /** ActionSteps the checkpoint covers (from the first); with `unit: exchange`, those it covers whole. */
  summarizedActionSteps: number;
  /** `unit: exchange`: the exchanges it covers (from the first), and the one before them it kept exact. */
  summarizedExchanges: number;
  keptExchange: number | null;
  count: number;
  checkpoint: Record<string, any> | null;
  checkpointSha256: string | null;
  checkpointRef: Record<string, any> | null;
  transcriptRef: Record<string, any> | null;
  /** The epoch whose Provider report settles the target, and the target. */
  pendingTargetEpoch: number | null;
  pendingTargetTokens: number | null;
  lastTargetValidation: Record<string, any> | null;
}

export interface EmergencyAdmission extends Admission {
  threshold: number;
  anchorTokens: number | null;
  /**
   * Whether a checkpoint now would absorb anything the summaries do not cover
   * yet (not in the manifest): the hard backstop's watermark defers to an L6
   * that is only waiting for the Provider's report.
   */
  fresh: boolean;
}

const ENFORCEMENT = "telemetry_only_until_hard_limit";

function throughIndex(count: number): number | null {
  return count ? count - 1 : null;
}

type EmergencyStage = Stage<EmergencyCheckpointParams, EmergencyCheckpointState, EmergencyAdmission>;

export class EmergencyCheckpointStage implements EmergencyStage {
  readonly id = EMERGENCY_CHECKPOINT;
  readonly phase = "summarize" as const;
  readonly transitions: readonly string[] = ["emergency_compaction"];
  readonly requires: readonly string[] = ["materialize"];
  readonly excludes: readonly string[] = [];
  readonly after: readonly string[] = [EPISODE_CAPSULE];
  readonly params: EmergencyCheckpointParams;

  constructor(params: EmergencyCheckpointParams) {
    this.params = params;
  }

  init(): EmergencyCheckpointState {
    return {
      summarizedActionSteps: 0, summarizedExchanges: 0, keptExchange: null, count: 0, checkpoint: null,
      checkpointSha256: null, checkpointRef: null, transcriptRef: null, pendingTargetEpoch: null,
      pendingTargetTokens: null, lastTargetValidation: null,
    };
  }

  get unit(): SegmentUnit {
    return this.params.unit ?? "action_step";
  }

  stateFields(state: EmergencyCheckpointState): Record<string, unknown> {
    return {
      pendingSemanticTargetEpoch: state.pendingTargetEpoch,
      pendingSemanticTargetTokens: state.pendingTargetTokens,
      lastSemanticTargetValidation: state.lastTargetValidation,
      emergencySummarizedActionStepCount: state.summarizedActionSteps,
      emergencyCompactionCount: state.count,
      emergencyCheckpoint: state.checkpoint,
      emergencyCheckpointSha256: state.checkpointSha256,
      emergencyCheckpointRef: state.checkpointRef,
      emergencyTranscriptRef: state.transcriptRef,
      ...(this.unit === "exchange" ? {
        emergencySummarizedExchangeCount: state.summarizedExchanges,
        emergencyKeptExchange: state.keptExchange,
      } : {}),
    };
  }

  restoreFields(fields: Record<string, any>): EmergencyCheckpointState {
    return {
      summarizedActionSteps: fields.emergencySummarizedActionStepCount,
      summarizedExchanges: fields.emergencySummarizedExchangeCount ?? 0,
      keptExchange: fields.emergencyKeptExchange ?? null,
      count: fields.emergencyCompactionCount,
      checkpoint: fields.emergencyCheckpoint,
      checkpointSha256: fields.emergencyCheckpointSha256,
      checkpointRef: fields.emergencyCheckpointRef,
      transcriptRef: fields.emergencyTranscriptRef,
      pendingTargetEpoch: fields.pendingSemanticTargetEpoch,
      pendingTargetTokens: fields.pendingSemanticTargetTokens,
      lastTargetValidation: fields.lastSemanticTargetValidation,
    };
  }

  /** The watermark: the absolute tokens capped by the ratio of the hard limit (or the ratio alone). */
  threshold(hardInputTokens: number): number {
    const ratioTokens = pyInt(hardInputTokens * this.params.trigger_ratio);
    return this.params.trigger_tokens === null ? ratioTokens : Math.min(this.params.trigger_tokens, ratioTokens);
  }

  check(view: RequestView, state: EmergencyCheckpointState): void {
    if (this.unit !== view.exactTailUnit) {
      throw new PolicySessionError(
        `emergency_checkpoint counts in ${this.unit} and the exact tail in ${view.exactTailUnit}: they must agree`,
      );
    }
    if (this.unit === "exchange" ? state.summarizedExchanges > view.exchangeTail!.from
      : state.summarizedActionSteps > view.exactTailFrom) {
      throw new PolicySessionError("emergency checkpoint coverage exceeds eligible history");
    }
  }

  slots(state: EmergencyCheckpointState): SlotContribution {
    return {
      emergencySummarizedActionSteps: state.summarizedActionSteps,
      emergencyCheckpoint: state.checkpoint,
      emergencyTranscriptRef: state.transcriptRef,
      ...(this.unit === "exchange" ? {
        emergencyUnit: "exchange" as const,
        emergencySummarizedExchanges: state.summarizedExchanges,
        emergencyKeptExchange: state.keptExchange,
      } : {}),
    };
  }

  /**
   * `unit: exchange`: what the next checkpoint absorbs. Every exchange before
   * the exact tail not yet summarized (the exchange the last checkpoint kept,
   * once it is no longer the anchor, included), the anchor left out; the
   * ActionSteps it then covers whole.
   */
  exchangeSelection(view: RequestView, state: EmergencyCheckpointState): ExchangeSelection {
    const tail = view.exchangeTail!;
    const spans = view.exchanges.spans;
    const anchor = tail.anchor;
    const absorbed: number[] = [];
    if (state.keptExchange !== null && state.keptExchange !== anchor) absorbed.push(state.keptExchange);
    for (let index = state.summarizedExchanges; index < tail.from; index += 1) {
      if (index !== anchor) absorbed.push(index);
    }
    absorbed.sort((a, b) => a - b);
    const kept = anchor !== null && anchor < tail.from ? anchor : null;
    const transcriptEnd = tail.from < spans.length ? spans[tail.from][0] : view.materialized.length;
    const cutoff = kept !== null ? spans[kept][0] : transcriptEnd;
    const coveredActionSteps = Math.min(view.completedCount, view.groups.filter(([, end]) => end <= cutoff).length);
    return {
      spans, absorbed, transcriptEnd, summarizedExchangesBefore: state.summarizedExchanges,
      summarizedExchanges: Math.max(state.summarizedExchanges, tail.from), keptExchange: kept, coveredActionSteps,
    };
  }

  /** Whether the next checkpoint would absorb anything past what the summaries already cover. */
  private fresh(view: RequestView, state: EmergencyCheckpointState, episodeCovered: number): boolean {
    if (this.unit === "action_step") return view.exactTailFrom > Math.max(state.summarizedActionSteps, episodeCovered);
    const selection = this.exchangeSelection(view, state);
    const episodeBoundary = episodeCovered ? view.groups[episodeCovered][0] : 0;
    return selection.absorbed.length > 0 && selection.summarizedExchanges > state.summarizedExchanges
      && selection.transcriptEnd > episodeBoundary;
  }

  admit(view: RequestView, state: EmergencyCheckpointState): EmergencyAdmission {
    const enabled = this.params.enabled;
    const episode = view.admission<EpisodeAdmission>(EPISODE_CAPSULE);
    const episodeCovered = view.stateOf<EpisodeCapsuleState>(EPISODE_CAPSULE)?.checkpointedActionSteps ?? 0;
    const hasPriority = episode?.hasPriority ?? false;
    const threshold = this.threshold(view.limits.hardInputTokens);
    // Admitted on the usage that the request before this projection reported, unless L5 already acted on it.
    const anchorTokens = episode?.record.status !== "admitted" ? view.admissionInputTokens : null;
    let reason: string;
    if (!enabled) reason = "rollout_compaction_disabled";
    else if (hasPriority) reason = "l5_has_priority";
    else if (anchorTokens !== null && anchorTokens >= threshold) reason = "at_or_above_high_watermark";
    else if (anchorTokens !== null) reason = "below_high_watermark";
    else if (view.usage.lastProviderInputTokens !== null) reason = "projection_epoch_unanchored";
    else reason = "provider_usage_unavailable";
    const record: Record<string, any> = {
      authority: "provider_reported_input_tokens",
      basis: "projection_entry_provider_usage",
      status: "not_admitted",
      provider_input_tokens: anchorTokens,
      provider_projection_epoch: view.usage.lastProviderProjectionEpoch,
      provider_wire_request_sha256: view.usage.lastProviderWireRequestSha256,
      emergency_high_watermark_tokens: threshold,
      reason,
    };
    const fresh = this.fresh(view, state, episodeCovered);
    const admitted = enabled && view.allowTransition && !hasPriority && anchorTokens !== null
      && anchorTokens >= threshold && fresh;
    if (admitted) record.status = "admitted";
    return { record, admitted, threshold, anchorTokens, fresh };
  }

  async apply(
    view: RequestView, state: EmergencyCheckpointState, admission: EmergencyAdmission, context: ApplyContext,
  ): Promise<Transition | null> {
    const { record } = admission;
    const episode = view.stateOf<EpisodeCapsuleState>(EPISODE_CAPSULE);
    const episodeCovered = episode?.checkpointedActionSteps ?? 0;
    const beforeEmergency = Math.max(state.summarizedActionSteps, episodeCovered);
    const preEmergencyBytes = jsonBytesOf(view.requestView);
    const { services } = view;
    const writer: SummaryWriter = {
      summaryProvider: services.summaryProvider, catalog: services.catalog, artifacts: services.artifacts,
      domain: services.domain,
      limits: {
        targetEstimatedTokens: this.params.checkpoint_target_tokens,
        maxEstimatedTokens: this.params.checkpoint_max_tokens,
      },
    };
    const prior = {
      summarizedActionSteps: state.summarizedActionSteps,
      episodeCheckpointedActionSteps: episodeCovered,
      checkpoint: state.checkpoint,
      checkpointSha256: state.checkpointSha256,
      checkpointRef: state.checkpointRef,
      transcriptRef: state.transcriptRef,
      episodeCapsules: episode?.capsules ?? [],
      compactionCount: state.count,
    };
    const selection = this.unit === "exchange" ? this.exchangeSelection(view, state) : null;
    let semantic: Record<string, any>;
    try {
      semantic = selection === null
        ? await writeEmergencyCheckpoint({
          messages: view.compactorInput(), tools: view.tools, inventory: view.inventory,
          candidateAbsorbed: view.exactTailFrom,
        }, writer, prior)
        : await writeExchangeCheckpoint({ messages: view.compactorInput(), tools: view.tools }, writer, prior, selection);
    } catch (error) {
      context.summaries.rejected({
        layer: "emergency_session_l6",
        projection_epoch: view.epoch,
        provider_wire_request_sha256: view.usage.lastProviderWireRequestSha256,
        error: errorText(error),
        disposition: "current_projection_retained",
      });
      Object.assign(record, {
        status: "rejected",
        reason: "summary_rejected_projection_retained",
        error: errorText(error),
      });
      return null;
    }
    const absorbed = selection === null ? view.exactTailFrom : selection.coveredActionSteps;
    const exchangesBefore = state.summarizedExchanges;
    state.summarizedActionSteps = absorbed;
    if (selection !== null) {
      state.summarizedExchanges = selection.summarizedExchanges;
      state.keptExchange = selection.keptExchange;
    }
    state.count += 1;
    state.checkpoint = clonePy(semantic.checkpoint);
    state.checkpointSha256 = String(semantic.checkpoint_sha256);
    state.checkpointRef = clonePy(semantic.checkpoint_ref);
    state.transcriptRef = clonePy(semantic.exact_transcript);
    context.summaries.accepted();
    const epoch = context.commit();
    // Telemetry only: the Provider's report on this epoch settles it (observeUsage).
    const target = pyInt(view.limits.hardInputTokens * this.params.target_ratio);
    state.pendingTargetEpoch = epoch;
    state.pendingTargetTokens = target;
    state.lastTargetValidation = {
      status: "provider_pending",
      projection_epoch: epoch,
      target_tokens: target,
      enforcement: ENFORCEMENT,
    };
    const usage = view.usage;
    return {
      kind: "emergency_compaction",
      absorbedActionSteps: absorbed,
      record: {
        schema: TRANSITION_SCHEMA,
        kind: "emergency_compaction",
        projection_epoch: epoch,
        from_epoch: view.epochBefore,
        to_epoch: epoch,
        reasons: ["provider_reported_emergency_high_watermark"],
        emergency_summarized_action_steps_before: beforeEmergency,
        emergency_summarized_action_steps_after: absorbed,
        ...(selection === null ? {} : {
          unit: "exchange",
          emergency_summarized_exchanges_before: exchangesBefore,
          emergency_summarized_exchanges_after: selection.summarizedExchanges,
          absorbed_exchanges: [...selection.absorbed],
          kept_exchange: selection.keptExchange,
        }),
        semantic_loss: true,
        semantic_admission_provider_input_tokens: admission.anchorTokens,
        semantic_admission_provider_projection_epoch: usage.lastProviderProjectionEpoch,
        semantic_admission_provider_wire_request_sha256: usage.lastProviderWireRequestSha256,
        semantic_high_watermark_tokens: admission.threshold,
        pre_request_utf8_bytes: preEmergencyBytes,
        post_request_utf8_bytes: jsonBytesOf(view.requestView),
        packet_ref: clonePy(semantic.packet_ref),
        checkpoint_ref: clonePy(semantic.checkpoint_ref),
        exact_transcript: clonePy(semantic.exact_transcript),
        checkpoint_sha256: semantic.checkpoint_sha256,
        checkpoint_estimated_tokens: semantic.checkpoint_estimated_tokens,
        checkpoint_budget: clonePy(semantic.checkpoint_budget),
        compactor: clonePy(semantic.metadata),
        post_estimated_input_tokens: view.pressure,
        target_input_tokens: target,
        target_validation: "provider_pending",
      },
    };
  }

  /** A capsule covers the active checkpoint: it leaves the request, its evidence stays in the capsule's record. */
  notice(state: EmergencyCheckpointState, notice: string): void {
    if (notice !== SUPERSEDED_BY_EPISODE_CAPSULE) throw new PolicySessionError(`emergency_checkpoint: unknown notice ${notice}`);
    state.checkpoint = null;
    state.checkpointSha256 = null;
    state.checkpointRef = null;
    state.transcriptRef = null;
  }

  /** Settle the pending target with the Provider's report on its epoch. */
  observeUsage(state: EmergencyCheckpointState, usage: UsageObservation): void {
    if (state.pendingTargetEpoch === null) return;
    const epoch = usage.projectionEpoch;
    if (usage.inputTokens === null) {
      if (epoch !== null && isInt(epoch) && num(epoch) === state.pendingTargetEpoch) {
        state.lastTargetValidation = {
          status: "failed",
          projection_epoch: epoch,
          target_tokens: state.pendingTargetTokens,
          input_tokens: null,
          provider_wire_request_sha256: usage.providerWireRequestSha256,
          reason: "provider_input_tokens_unavailable",
          enforcement: ENFORCEMENT,
        };
        state.pendingTargetEpoch = null;
        state.pendingTargetTokens = null;
      }
      return;
    }
    if (state.pendingTargetEpoch === epoch && state.pendingTargetTokens !== null) {
      const target = state.pendingTargetTokens;
      state.lastTargetValidation = {
        status: usage.inputTokens <= target ? "passed" : "failed",
        projection_epoch: epoch,
        target_tokens: target,
        input_tokens: usage.inputTokens,
        provider_wire_request_sha256: usage.providerWireRequestSha256,
        enforcement: ENFORCEMENT,
      };
      state.pendingTargetEpoch = null;
      state.pendingTargetTokens = null;
    }
  }

  manifest(view: RequestView, state: EmergencyCheckpointState, admission: EmergencyAdmission | null): Record<string, unknown> {
    return {
      emergency_compaction_admission: admission === null ? null : admission.record,
      emergency_target_validation: clonePy(state.lastTargetValidation),
      emergency_summarized_through_action_step: throughIndex(state.summarizedActionSteps),
      emergency_compaction_count: state.count,
      emergency_checkpoint_sha256: state.checkpointSha256,
      emergency_exact_transcript_ref: clonePy(state.transcriptRef),
      // Its watermark, absolute or ratio-only; the legacy manifest carries it for append-only versions.
      resolved_compaction_trigger_tokens: this.threshold(view.limits.hardInputTokens),
      ...(this.unit === "exchange" ? {
        emergency_exchange_coverage: {
          unit: "exchange",
          summarized_exchanges: state.summarizedExchanges,
          through_exchange_index: throughIndex(state.summarizedExchanges),
          kept_exchange: state.keptExchange,
        },
      } : {}),
    };
  }
}
