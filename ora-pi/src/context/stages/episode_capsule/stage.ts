// episode_capsule (phase summarize, L5): a closed Episode becomes one capsule,
// admitted by the Provider-reported input of the request before (README.md).

import { inlineImageCount, inlineMediaBytesFromRefs, jsonBytesOf } from "../../projection.ts";
import { PolicySessionError, canonicalSha256, clonePy, eq, errorText, get, isInt, num, pyInt } from "../../pyutil.ts";
import { EMERGENCY_CHECKPOINT, EPISODE_CAPSULE, SUPERSEDED_BY_EPISODE_CAPSULE, TRANSITION_SCHEMA } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { Admission, ApplyContext, SlotContribution, Stage, Transition } from "../stage.ts";
import { type SummaryWriter, writeEpisodeCapsule } from "./capsule.ts";

export const EPISODE_TRIGGER_FORMULA = "absolute_tokens";
export const WINDOW_SCALED_EPISODE_TRIGGER_FORMULA = "max(absolute_tokens, int(hard_input_tokens * ratio))";
/** L5 by ratio alone (v10: `trigger_tokens` null). */
export const RATIO_EPISODE_TRIGGER_FORMULA = "int(hard_input_tokens * ratio)";

export interface EpisodeCapsuleParams {
  /** Provider-reported input tokens at which a closed Episode is summarized. */
  /** null: the ratio alone (v10). */
  trigger_tokens: number | null;
  /**
   * null: the watermark is `trigger_tokens` (v6 on). A ratio: the watermark
   * scales with the window, max(`trigger_tokens`, ratio x hard input limit)
   * (v4/v5).
   */
  trigger_ratio: number | null;
  /** Same-source retries: at most this many attempts of one Episode's capsule. */
  retry_attempts: number;
  /** A rejected attempt waits this many completed ActionSteps, doubling with each attempt. */
  retry_backoff_action_steps: number;
  /** The checkpoint length asked of the compactor, and its soft maximum (estimated tokens). */
  checkpoint_target_tokens: number;
  checkpoint_max_tokens: number;
  /** false only when a context pack turns rollout compaction off: nothing is summarized, the record says why. */
  enabled: boolean;
}

interface Retry {
  attempts: number;
  next_completed_action_step: number;
}

export interface EpisodeCapsuleState {
  /** ActionSteps the capsules cover (from the first). */
  checkpointedActionSteps: number;
  checkpointedRounds: number[];
  count: number;
  capsules: Record<string, any>[];
  /** Same-source retries, bounded and spaced by completed ActionSteps. */
  retry: Map<string, Retry>;
}

/** ``[start, end, round]`` of the next closed Episode. */
type Candidate = [number, number, number];

export interface EpisodeAdmission extends Admission {
  candidate: Candidate | null;
  retryKey: string | null;
  retry: Retry;
  threshold: number;
  anchorTokens: number | null;
  postLosslessPressure: number;
  /** L6 waits while a closed Episode is eligible for L5 and was not rejected. */
  hasPriority: boolean;
}

/** Identity of one candidate's exact source, which bounds its retries. */
function retryKey(candidate: Candidate | null, inventory: Record<string, any>[]): string | null {
  if (candidate === null) return null;
  const [start, end] = candidate;
  return canonicalSha256({
    coverage: [...candidate],
    source_events: inventory.slice(start, end).map((item) => item.source_event_ids),
  });
}

function throughIndex(count: number): number | null {
  return count ? count - 1 : null;
}

export class EpisodeCapsuleStage implements Stage<EpisodeCapsuleParams, EpisodeCapsuleState, EpisodeAdmission> {
  readonly id = EPISODE_CAPSULE;
  readonly phase = "summarize" as const;
  readonly transitions: readonly string[] = ["episode_capsule"];
  readonly requires: readonly string[] = ["materialize"];
  readonly excludes: readonly string[] = [];
  readonly params: EpisodeCapsuleParams;

  constructor(params: EpisodeCapsuleParams) {
    this.params = params;
  }

  init(): EpisodeCapsuleState {
    return { checkpointedActionSteps: 0, checkpointedRounds: [], count: 0, capsules: [], retry: new Map() };
  }

  stateFields(state: EpisodeCapsuleState): Record<string, unknown> {
    return {
      episodeCheckpointedActionStepCount: state.checkpointedActionSteps,
      episodeCheckpointedRounds: state.checkpointedRounds,
      episodeCheckpointCount: state.count,
      episodeCapsules: state.capsules,
      episodeRetryState: state.retry,
    };
  }

  restoreFields(fields: Record<string, any>): EpisodeCapsuleState {
    return {
      checkpointedActionSteps: fields.episodeCheckpointedActionStepCount,
      checkpointedRounds: fields.episodeCheckpointedRounds,
      count: fields.episodeCheckpointCount,
      capsules: fields.episodeCapsules,
      retry: fields.episodeRetryState,
    };
  }

  check(view: RequestView, state: EpisodeCapsuleState): void {
    if (state.checkpointedActionSteps > view.completedCount) {
      throw new PolicySessionError("Episode capsule coverage exceeds completed history");
    }
  }

  slots(state: EpisodeCapsuleState): SlotContribution {
    return { episodeCheckpointedActionSteps: state.checkpointedActionSteps, episodeCapsules: state.capsules };
  }

  /** The watermark: absolute, or scaled with the window. */
  threshold(hardInputTokens: number): number {
    const { trigger_tokens: tokens, trigger_ratio: ratio } = this.params;
    if (tokens === null) return pyInt(hardInputTokens * num(ratio));
    return ratio === null ? tokens : Math.max(tokens, pyInt(hardInputTokens * ratio));
  }

  /** The next closed Episode, wholly before the exact tail. */
  private candidate(view: RequestView, state: EpisodeCapsuleState): Candidate | null {
    const inventory = view.inventory;
    const completedCount = view.completedCount;
    const episodeStart = state.checkpointedActionSteps;
    if (episodeStart >= completedCount) return null;
    const candidateRound = get(inventory[episodeStart], "session_round");
    const startsAtRoundBoundary = episodeStart === 0
      || !eq(get(inventory[episodeStart - 1], "session_round"), candidateRound);
    if (!((isInt(candidateRound) || typeof candidateRound === "boolean")
      && !eq(candidateRound, view.currentRound) && startsAtRoundBoundary)) {
      return null;
    }
    let candidateEnd = episodeStart;
    while (candidateEnd < completedCount && eq(get(inventory[candidateEnd], "session_round"), candidateRound)) {
      candidateEnd += 1;
    }
    if (candidateEnd > episodeStart && candidateEnd <= view.exactTailFrom
      && inventory.slice(episodeStart, candidateEnd).every((item) => get(item, "complete") === true)) {
      return [episodeStart, candidateEnd, num(candidateRound)];
    }
    return null;
  }

  admit(view: RequestView, state: EpisodeCapsuleState): EpisodeAdmission {
    const enabled = this.params.enabled;
    const candidate = enabled ? this.candidate(view, state) : null;
    const key = retryKey(candidate, view.inventory);
    const anchorTokens = view.admissionInputTokens;
    const threshold = this.threshold(view.limits.hardInputTokens);
    let retry: Retry = { attempts: 0, next_completed_action_step: 0 };
    let retryReason: string | null = null;
    if (candidate !== null) {
      retry = state.retry.get(key!) ?? retry;
      if (retry.attempts >= this.params.retry_attempts) retryReason = "same_episode_retry_exhausted";
      else if (view.completedCount < retry.next_completed_action_step) retryReason = "same_episode_retry_backoff";
    }
    let reason: string;
    if (retryReason !== null) reason = retryReason;
    else if (anchorTokens !== null && anchorTokens >= threshold && candidate !== null) reason = "at_or_above_high_watermark";
    else if (candidate === null) reason = "no_closed_episode";
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
      episode_high_watermark_tokens: threshold,
      retry_policy: "same_source_four_attempts_action_backoff_v1",
      retry_source_sha256: key,
      attempts: retry.attempts,
      max_attempts: this.params.retry_attempts,
      next_retry_completed_action_step: retry.next_completed_action_step,
      eligible_session_round: candidate !== null ? candidate[2] : null,
      reason,
    };
    // A context pack may turn rollout compaction off: nothing is summarized, the reason says so.
    if (!enabled) record.reason = "rollout_compaction_disabled";
    return {
      record,
      admitted: enabled && view.allowTransition && anchorTokens !== null && anchorTokens >= threshold
        && candidate !== null && retryReason === null,
      candidate, retryKey: key, retry, threshold, anchorTokens,
      postLosslessPressure: view.pressure,
      hasPriority: enabled && candidate !== null && retryReason === null,
    };
  }

  async apply(
    view: RequestView, state: EpisodeCapsuleState, admission: EpisodeAdmission, context: ApplyContext,
  ): Promise<Transition | null> {
    const { record } = admission;
    const candidate = admission.candidate!;
    const retry = { attempts: admission.retry.attempts + 1, next_completed_action_step: view.completedCount };
    state.retry.set(admission.retryKey!, retry);
    record.attempts = retry.attempts;
    record.status = "admitted";
    const pre = view.sizeSnapshot();
    const [startActionStep, endActionStep, candidateRound] = candidate;
    const { services } = view;
    const writer: SummaryWriter = {
      summaryProvider: services.summaryProvider, catalog: services.catalog, artifacts: services.artifacts,
      domain: services.domain,
      limits: {
        targetEstimatedTokens: this.params.checkpoint_target_tokens,
        maxEstimatedTokens: this.params.checkpoint_max_tokens,
      },
    };
    let semantic: Record<string, any>;
    try {
      semantic = await writeEpisodeCapsule({
        messages: view.compactorInput(), tools: view.tools, inventory: view.inventory, startActionStep, endActionStep,
      }, writer, state.count + 1);
    } catch (error) {
      // The stored retry record is this same object, so the backoff persists.
      retry.next_completed_action_step = view.completedCount
        + this.params.retry_backoff_action_steps * 2 ** (retry.attempts - 1);
      record.next_retry_completed_action_step = retry.next_completed_action_step;
      context.summaries.rejected({
        layer: "episode_capsule_l5",
        session_round: candidateRound,
        projection_epoch: view.epoch,
        provider_wire_request_sha256: view.usage.lastProviderWireRequestSha256,
        error: errorText(error),
        disposition: "original_episode_retained",
      });
      Object.assign(record, {
        status: "rejected",
        reason: "summary_rejected_original_retained",
        error: errorText(error),
      });
      admission.hasPriority = false;
      return null;
    }
    const previousCount = state.count;
    const superseded = this.supersededEmergencyCheckpoint(view, state, endActionStep);
    state.checkpointedActionSteps = endActionStep;
    state.checkpointedRounds = [...state.checkpointedRounds, candidateRound];
    state.count += 1;
    state.capsules = [...state.capsules, clonePy(semantic)];
    if (superseded !== null) context.notify(EMERGENCY_CHECKPOINT, SUPERSEDED_BY_EPISODE_CAPSULE);
    context.summaries.accepted();
    const epoch = context.commit();
    const usage = view.usage;
    return {
      kind: "episode_capsule",
      absorbedActionSteps: endActionStep,
      record: {
        schema: TRANSITION_SCHEMA,
        kind: "episode_capsule",
        projection_epoch: epoch,
        from_epoch: view.epochBefore,
        to_epoch: epoch,
        reasons: [
          "closed_episode_provider_high_watermark",
          ...(superseded !== null ? ["superseded_overlapping_emergency_checkpoint"] : []),
        ],
        session_round: candidateRound,
        episode_checkpoint_count_before: previousCount,
        episode_checkpoint_count_after: state.count,
        checkpointed_action_steps_before: startActionStep,
        checkpointed_action_steps_after: endActionStep,
        semantic_loss: true,
        post_lossless_estimated_input_tokens: admission.postLosslessPressure,
        semantic_admission_provider_input_tokens: admission.anchorTokens,
        semantic_admission_provider_projection_epoch: usage.lastProviderProjectionEpoch,
        semantic_admission_provider_wire_request_sha256: usage.lastProviderWireRequestSha256,
        semantic_high_watermark_tokens: admission.threshold,
        pre_inline_images: pre.images,
        post_inline_images: inlineImageCount(view.requestView),
        pre_media_bytes: pre.media_bytes,
        post_media_bytes: inlineMediaBytesFromRefs(view.working),
        pre_request_utf8_bytes: pre.request_bytes,
        post_request_utf8_bytes: jsonBytesOf(view.requestView),
        packet_ref: clonePy(semantic.packet_ref),
        capsule_ref: clonePy(semantic.capsule_ref),
        capsule_sha256: semantic.capsule_sha256,
        exact_transcript: clonePy(semantic.exact_transcript),
        superseded_emergency_checkpoint: superseded,
        checkpoint_estimated_tokens: semantic.checkpoint_estimated_tokens,
        checkpoint_budget: clonePy(semantic.checkpoint_budget),
        compactor: clonePy(semantic.metadata),
        post_estimated_input_tokens: view.pressure,
      },
    };
  }

  /** The active L6 checkpoint a new capsule ending at ``endActionStep`` fully covers, if any. */
  private supersededEmergencyCheckpoint(
    view: RequestView, state: EpisodeCapsuleState, endActionStep: number,
  ): Record<string, any> | null {
    const emergency = view.stateOf<EmergencyView>(EMERGENCY_CHECKPOINT);
    if (emergency === null) return null;
    if (!(emergency.summarizedActionSteps > state.checkpointedActionSteps
      && emergency.checkpoint !== null
      && emergency.summarizedActionSteps > 0
      && endActionStep >= emergency.summarizedActionSteps)) {
      return null;
    }
    return {
      checkpoint_sha256: emergency.checkpointSha256,
      checkpoint_ref: clonePy(emergency.checkpointRef),
      exact_transcript: clonePy(emergency.transcriptRef),
      through_action_step_index: emergency.summarizedActionSteps - 1,
    };
  }

  manifest(view: RequestView, state: EpisodeCapsuleState, admission: EpisodeAdmission | null): Record<string, unknown> {
    return {
      resolved_episode_checkpoint_trigger_tokens: this.threshold(view.limits.hardInputTokens),
      resolved_episode_checkpoint_trigger_formula: this.params.trigger_tokens === null ? RATIO_EPISODE_TRIGGER_FORMULA
        : this.params.trigger_ratio === null ? EPISODE_TRIGGER_FORMULA : WINDOW_SCALED_EPISODE_TRIGGER_FORMULA,
      episode_checkpoint_admission: admission === null ? null : admission.record,
      episode_checkpointed_through_action_step: throughIndex(state.checkpointedActionSteps),
      episode_checkpointed_rounds: [...state.checkpointedRounds],
      episode_checkpoint_count: state.count,
    };
  }
}

/** What this stage reads of the emergency checkpoint's slice. */
interface EmergencyView {
  summarizedActionSteps: number;
  checkpoint: Record<string, any> | null;
  checkpointSha256: string | null;
  checkpointRef: Record<string, any> | null;
  transcriptRef: Record<string, any> | null;
}
