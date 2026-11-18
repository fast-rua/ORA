// The `ora.native-context-projection.v6` manifest of a stage-driven projection,
// field for field and in the order the legacy pipeline wrote it (native.ts
// `losslessManifest`), so sealed evidence and its checks are unchanged.
//
// Stages supply their fields by name (`Stage.manifest`); the driver supplies
// the request's sizes, identities, epochs and invariants. The fields of the
// age phase (L1-L4) take their "nothing aged" values when no stage of the
// scheme owns them, and the five L1-L4 passes are listed unchanged; the
// exchange passes (exchange_aging, reasoning_replay) follow them only when the
// scheme runs those stages. A new manifest format, with one section per stage,
// is a later, separately blessed step.

import { APPEND_ONLY_SCHEMAS } from "../management.ts";
import { inlineImageCount, inlineMediaBytesFromRefs, jsonBytesOf, providerMessage } from "../projection.ts";
import type { Message } from "../projection.ts";
import { PolicySessionError, canonicalSha256, clonePy, get, num, pyFloat } from "../pyutil.ts";
import { activeSizer, measuredPassRecord } from "../request_size.ts";
import type { LosslessInvariants, StageDriver } from "./driver.ts";
import type { RequestView } from "./request.ts";
import { losslessRequestIdentity } from "./request_identity.ts";

export const PROJECTION_MANIFEST_SCHEMA = "ora.native-context-projection.v6";

/** How the legacy manifest names "media stays until a summary absorbs it" (no media window). */
export const APPEND_ONLY_MEDIA_POLICY = "append_only_until_compaction_v1";

/**
 * The manifest's pass order: materialization, the four lossless layers' five
 * passes, the exchange passes, reasoning exclusion.
 */
const PASS_ORDER = [
  "recoverable_payload_materialization",
  "l1_large_result_reference",
  "l2_exact_duplicate_reference",
  "l3_lifecycle_media_reference",
  "l4_read_only_result_reference",
  "l4_observation_reference",
  "exchange_argument_reference",
  "exchange_tool_result_reference",
  "exchange_tool_image_reference",
  "exchange_camera_image_reference",
  "exchange_observation_reference",
  "exchange_reasoning_withheld",
  "reasoning_exclusion",
];

/** Passes no stage of the scheme runs: listed as unchanged over the materialized history. */
const UNAGED_PASSES = new Set(PASS_ORDER.slice(1, 6));
/** Passes listed only when a stage of the scheme records them (the camera pass: `exchange_window_v1` only). */
const OPTIONAL_PASSES = new Set(PASS_ORDER.slice(6, 12));

function throughIndex(count: number): number | null {
  return count ? count - 1 : null;
}

/** Every stage's manifest fields; `passes` are collected, any other field has one owner. */
function stageFields(driver: StageDriver, view: RequestView): [Record<string, unknown>, Record<string, any>[]] {
  const fields: Record<string, unknown> = {};
  const passes: Record<string, any>[] = [];
  for (const stage of driver.stages) {
    if (!stage.manifest) continue;
    const section = stage.manifest(view, driver.stateOf(stage.id) as any, view.admission(stage.id));
    for (const [field, value] of Object.entries(section)) {
      if (field === "passes") {
        passes.push(...(value as Record<string, any>[]));
        continue;
      }
      if (field in fields) throw new PolicySessionError(`two context stages write manifest field ${field}`);
      fields[field] = value;
    }
  }
  return [fields, passes];
}

function orderedPasses(view: RequestView, passes: Record<string, any>[]): Record<string, any>[] {
  const byName = new Map(passes.map((pass) => [String(pass.name), pass]));
  if (byName.size !== passes.length) throw new PolicySessionError("two context stages record one pass");
  const base = activeSizer().measure(view.materialized);
  return PASS_ORDER.flatMap((name) => {
    const pass = byName.get(name);
    if (pass !== undefined) return [pass];
    if (OPTIONAL_PASSES.has(name)) return [];
    if (!UNAGED_PASSES.has(name)) throw new PolicySessionError(`context manifest lacks the ${name} pass`);
    return [measuredPassRecord(name, base, base, {
      changed_messages: [], artifact_refs: [], externalized_event_ids: [],
    })];
  });
}

/**
 * An exact tail in exchanges (services/segmenter.ts): its size, where it
 * starts, the anchor exchange, and per exchange of the materialized history
 * its inline images and the ActionSteps it spans (what the evidence counts a
 * summary's images by).
 */
function exactTailRecord(view: RequestView): Record<string, unknown> {
  const tail = view.exchangeTail!;
  const spans = view.exchanges.spans;
  const actionStep = (index: number): number => {
    let found = 0;
    view.groups.forEach(([start], group) => {
      if (start <= index) found = group;
    });
    return found;
  };
  return {
    unit: "exchange",
    exchanges: tail.count,
    total_exchanges: spans.length,
    from_exchange: tail.from,
    anchor_exchange: tail.anchor,
    exchange_inventory: spans.map(([start, end], index) => ({
      exchange_index: index,
      action_steps: [actionStep(start), actionStep(end - 1)],
      inline_images: inlineImageCount((view.materialized as Message[]).slice(start, end)),
    })),
  };
}

export function projectionManifestV6(
  driver: StageDriver, view: RequestView, invariants: LosslessInvariants,
): Record<string, any> {
  const [fields, passes] = stageFields(driver, view);
  const unused = new Set(Object.keys(fields));
  const required = (field: string): unknown => {
    if (!(field in fields)) throw new PolicySessionError(`context manifest lacks ${field}`);
    unused.delete(field);
    return fields[field];
  };
  const optional = (field: string, fallback: unknown): unknown => {
    unused.delete(field);
    return field in fields ? fields[field] : fallback;
  };
  const tail = (field: string): Record<string, unknown> => {
    unused.delete(field);
    return field in fields ? { [field]: fields[field] } : {};
  };
  // Fields the legacy format carries only for the append-only management versions (v8, v9).
  const appendOnly = APPEND_ONLY_SCHEMAS.has(String(driver.scheme.source.schema));
  const appendOnlyTail = (field: string): Record<string, unknown> => {
    const value = tail(field);
    return appendOnly ? value : {};
  };
  const { limits, state, ledger } = driver;
  const requestView = view.requestView;
  const { estimator } = driver.services;
  const images = inlineImageCount(requestView);
  const imageTokens = estimator.imageEstimate(requestView);
  const preLosslessBytes = jsonBytesOf({ messages: view.materialized.map(providerMessage), tools: [...view.tools] });
  const postLosslessBytes = jsonBytesOf({ messages: view.aged.map(providerMessage), tools: [...view.tools] });
  const mediaFrontier = num(optional("lifecycle_media_externalized_action_steps", 0));
  const manifest = {
    schema: PROJECTION_MANIFEST_SCHEMA,
    profile: driver.identity.profile,
    context_retention: driver.identity.retention,
    summary_policy: driver.identity.summaryPolicy,
    context_management: clonePy({ ...driver.scheme.source }),
    resolved_near_tail_action_steps: optional("resolved_near_tail_action_steps", view.exactTail),
    resolved_media_retention_policy: optional("resolved_media_retention_policy", APPEND_ONLY_MEDIA_POLICY),
    ...tail("resolved_tail_aging_stride_action_steps"),
    ...tail("tail_aging"),
    resolved_inline_image_target: optional("resolved_inline_image_target", null),
    resolved_episode_checkpoint_trigger_tokens: required("resolved_episode_checkpoint_trigger_tokens"),
    resolved_episode_checkpoint_trigger_formula: required("resolved_episode_checkpoint_trigger_formula"),
    observation_step: view.observationStep,
    target_input_tokens: limits.targetInputTokens,
    hard_input_tokens: limits.hardInputTokens,
    estimated_input_tokens: view.pressure,
    raw_estimated_input_tokens: estimator.raw(requestView),
    pressure_estimated_input_tokens: view.pressureRaw,
    provider_usage_anchor_tokens: state.lastProviderInputTokens,
    provider_usage_anchor_projection_epoch: state.lastProviderProjectionEpoch,
    provider_usage_anchor_wire_request_sha256: state.lastProviderWireRequestSha256,
    provider_usage_anchor_applies_to_projection_epoch: view.providerAnchorApplies(),
    token_estimate_status: view.tokenEstimateStatus,
    estimate_calibration_delta_tokens: view.calibrationDelta,
    episode_checkpoint_admission: required("episode_checkpoint_admission"),
    emergency_compaction_admission: required("emergency_compaction_admission"),
    emergency_target_validation: required("emergency_target_validation"),
    request_utf8_bytes: jsonBytesOf(requestView),
    active_inline_images: images,
    active_media_bytes: inlineMediaBytesFromRefs(view.working),
    registered_recoverable_artifact_count: view.materializedRefs.length,
    first_delivery_advertised_message_count: view.firstDeliveryAdvertisedIndexes.length,
    pre_lossless_request_utf8_bytes: preLosslessBytes,
    post_lossless_request_utf8_bytes: postLosslessBytes,
    actual_lossless_externalized_ratio: pyFloat(Math.max(0, preLosslessBytes - postLosslessBytes) / preLosslessBytes),
    estimated_image_tokens: imageTokens,
    estimated_text_tokens: Math.max(0, view.pressureRaw - imageTokens),
    ...losslessRequestIdentity(view),
    cache_epoch: state.cacheEpoch,
    projection_epoch: state.cacheEpoch,
    projection_transitions: view.transitions,
    epoch_transition: view.transitions.length ? view.transitions[view.transitions.length - 1] : null,
    lossless_externalized_through_action_step: throughIndex(mediaFrontier),
    lifecycle_media_externalized_through_action_step: throughIndex(mediaFrontier),
    pressure_externalized_through_action_step: throughIndex(num(optional("pressure_externalized_action_steps", 0))),
    episode_checkpointed_through_action_step: required("episode_checkpointed_through_action_step"),
    episode_checkpointed_rounds: required("episode_checkpointed_rounds"),
    episode_checkpoint_count: required("episode_checkpoint_count"),
    emergency_summarized_through_action_step: required("emergency_summarized_through_action_step"),
    emergency_compaction_count: required("emergency_compaction_count"),
    semantic_rejection_count: ledger.rejectionCount,
    last_semantic_rejection: clonePy(ledger.lastRejection),
    retained_exact_causal_action_steps: Math.min(
      view.exactTail, view.completedCount - driver.summarizedActionSteps(),
    ),
    protected_inline_media_action_steps: view.inventory
      .filter((item) => get(item, "complete") !== true || num(get(item, "action_step_index") ?? -1) >= mediaFrontier)
      .map((item) => item.action_step_index),
    open_action_step: view.inventory[view.inventory.length - 1].action_step_index,
    action_step_inventory: clonePy(view.inventory),
    causal_exact_tail_sha256: canonicalSha256(invariants.canonicalCausalTail),
    lossless_invariants_before_sha256: invariants.beforeSha256,
    lossless_invariants_after_sha256: invariants.afterSha256,
    emergency_checkpoint_sha256: required("emergency_checkpoint_sha256"),
    emergency_exact_transcript_ref: required("emergency_exact_transcript_ref"),
    checkpoint_slot_count: view.working.filter((message) => get(message, "_episode_context_capsule") === true
      || get(message, "_emergency_context_checkpoint") === true).length,
    causal_envelope: "ora.causal-observation.v1",
    causal_forbidden_field_scan: "passed",
    passes: orderedPasses(view, passes),
    ...appendOnlyTail("resolved_compaction_trigger_tokens"),
    ...tail("camera_delivery"),
    ...tail("context_over_limit"),
    ...tail("resolved_exchange_aging_trigger_tokens"),
    ...tail("exchange_aging"),
    ...tail("hard_backstop"),
    ...tail("request_estimate"),
    ...tail("hard_backstop_aging"),
    ...tail("truncated_reply_reasoning"),
    ...tail("emergency_exchange_coverage"),
    ...(view.exchangeTail === null ? {} : { exact_tail: exactTailRecord(view) }),
  };
  if (unused.size) throw new PolicySessionError(`context manifest has no place for ${[...unused].sort().join(", ")}`);
  return manifest;
}
