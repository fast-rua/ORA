// A stage scheme: the ordered stages with their parameters, the segmentation
// they share and the request-size estimate. `schemeFromManagement` is the one
// place a management object (what sealed plans and the goldens carry) becomes
// a stage list; a plan resolved from an ora.context-scheme.v2 scheme carries
// that list too (`schemeFromStageList`).

import {
  CAMERA_IMAGE_MODES, CONTEXT_MANAGEMENT_V10_SCHEMA, CONTEXT_MANAGEMENT_V6_SCHEMA, CONTEXT_MANAGEMENT_V8_SCHEMA,
  CONTEXT_MANAGEMENT_V9_SCHEMA,
} from "../management.ts";
import { IMAGE_PRESSURE_ESTIMATE_TOKENS } from "../projection.ts";
import { num, pyInt } from "../pyutil.ts";
import { CameraDeliveryStage } from "./camera_delivery/stage.ts";
import { EagerExternalizeStage } from "./eager_externalize/stage.ts";
import { EmergencyCheckpointStage } from "./emergency_checkpoint/stage.ts";
import { EpisodeCapsuleStage } from "./episode_capsule/stage.ts";
import { ExchangeAgingStage } from "./exchange_aging/stage.ts";
import { HardBackstopStage } from "./hard_backstop/stage.ts";
import { MaterializeStage } from "./materialize/stage.ts";
import { MediaWindowStage } from "./media_window/stage.ts";
import {
  CAMERA_DELIVERY, EAGER_EXTERNALIZE, EMERGENCY_CHECKPOINT, EPISODE_CAPSULE, EXCHANGE_AGING, HARD_BACKSTOP,
  MATERIALIZE, MEDIA_WINDOW, OVER_LIMIT_GUARD, PRESSURE_EXTERNALIZE, REASONING_REPLAY, STRIDE_AGING,
} from "./names.ts";
import { OverLimitGuardStage } from "./over_limit_guard/stage.ts";
import { PressureExternalizeStage } from "./pressure_externalize/stage.ts";
import { ESTIMATE_PRESSURE_WINDOW_SHARE } from "./pressure_externalize/trigger.ts";
import { ReasoningReplayStage } from "./reasoning_replay/stage.ts";
import { RESERVE_TOKENS } from "./services/limits.ts";
import { NO_POINTERS, type ReferenceSettings } from "./services/references.ts";
import type { SegmentUnit } from "./services/segmenter.ts";
import type { Stage, StageSpec } from "./stage.ts";
import { StrideAgingStage } from "./stride_aging/stage.ts";

/** A plan's stage list (`policy.context_stages`), resolved by Python from an ora.context-scheme.v2 scheme. */
export const STAGE_LIST_SCHEMA = "ora.context-stages.v1";

export interface StageScheme {
  /** The management object this scheme was translated from; the manifest echoes it. */
  source: Record<string, any>;
  /**
   * The unit the age stages count in (services/segmenter.ts), the exact tail
   * (what no summary absorbs) and the unit it counts in: ActionSteps when
   * absent, or exchanges plus the latest robot action's exchange.
   */
  segments: { unit: SegmentUnit; exact_tail: number; exact_tail_unit?: SegmentUnit };
  /** Tokens kept free beside the maximum output: hard limit = window - max output - reserve. */
  limits: { reserve_tokens: number };
  /** The request-size estimate's tokens per inline image (before a Provider report anchors it). */
  estimator: { image_tokens: number };
  /** Whether and how recoverable references are shown (services/references.ts). */
  references: ReferenceSettings;
  stages: StageSpec[];
}

const FACTORIES: Record<string, (params: any) => Stage> = {
  [MATERIALIZE]: (params) => new MaterializeStage(params),
  [STRIDE_AGING]: (params) => new StrideAgingStage(params),
  [EAGER_EXTERNALIZE]: (params) => new EagerExternalizeStage(params),
  [MEDIA_WINDOW]: (params) => new MediaWindowStage(params),
  [PRESSURE_EXTERNALIZE]: (params) => new PressureExternalizeStage(params),
  [CAMERA_DELIVERY]: (params) => new CameraDeliveryStage(params),
  [EPISODE_CAPSULE]: (params) => new EpisodeCapsuleStage(params),
  [EMERGENCY_CHECKPOINT]: (params) => new EmergencyCheckpointStage(params),
  [OVER_LIMIT_GUARD]: (params) => new OverLimitGuardStage(params),
  [EXCHANGE_AGING]: (params) => new ExchangeAgingStage(params),
  [REASONING_REPLAY]: (params) => new ReasoningReplayStage(params),
  [HARD_BACKSTOP]: (params) => new HardBackstopStage(params),
};

export function buildStage(spec: StageSpec): Stage {
  const factory = FACTORIES[spec.stage];
  if (factory === undefined) throw new Error(`unknown context stage ${spec.stage}`);
  return factory(spec.params);
}

/** Pipeline options that shape the stages (context packs). */
export interface LegacyOptions {
  /**
   * `policy.context_pack` may turn L4 (`pressure_externalization: false`) and
   * L5/L6 (`rollout_compaction: false`) off (context_pack.ts
   * `packContextManagement`): the stages stay in the scheme with `enabled:
   * false`, so the manifest keeps their fields and records why nothing ran.
   */
  pressureExternalization: boolean;
  rolloutCompaction: boolean;
}

/** A context pack's switches on a stage list: each turns its stages' `enabled` off. */
function withPackSwitches(stages: StageSpec[], options: LegacyOptions): StageSpec[] {
  return stages.map((spec) => {
    const off = (spec.stage === PRESSURE_EXTERNALIZE && !options.pressureExternalization)
      || ((spec.stage === EPISODE_CAPSULE || spec.stage === EMERGENCY_CHECKPOINT) && !options.rolloutCompaction);
    return off ? { stage: spec.stage, params: { ...spec.params, enabled: false } } : spec;
  });
}

/** The stage list of a plan (`policy.context_stages`, ora.context-stages.v1) as a scheme, with the pack's switches. */
export function schemeFromStageList(
  list: Record<string, any>, source: Record<string, any>, options: LegacyOptions,
): StageScheme {
  if (list.schema !== STAGE_LIST_SCHEMA) throw new Error(`context_stages must be an ${STAGE_LIST_SCHEMA} stage list`);
  return {
    source,
    segments: list.segments, limits: list.limits, estimator: list.estimator, references: list.references,
    stages: withPackSwitches(list.stages, options),
  };
}

/** The stage scheme of a validated management object (v6, v8 or v9; management.ts refuses the rest). */
export function schemeFromManagement(management: Record<string, any>, options: LegacyOptions): StageScheme {
  if (management.schema === CONTEXT_MANAGEMENT_V6_SCHEMA) return schemeV6(management, options);
  if (management.schema === CONTEXT_MANAGEMENT_V8_SCHEMA) return schemeV8(management, options);
  if (management.schema === CONTEXT_MANAGEMENT_V9_SCHEMA) return schemeV9(management, options);
  if (management.schema === CONTEXT_MANAGEMENT_V10_SCHEMA) return schemeV10(management, options);
  throw new Error(`no stage scheme for context management ${String(management.schema)}`);
}

function checkpointLimits(management: Record<string, any>): Record<string, number> {
  return {
    checkpoint_target_tokens: pyInt(num(management.checkpoint_target_estimated_tokens)),
    checkpoint_max_tokens: pyInt(num(management.checkpoint_max_estimated_tokens)),
  };
}

/** L5's same-source retries: 4 attempts, the next after 4, 8, 16 more completed ActionSteps. */
const EPISODE_RETRIES = { retry_attempts: 4, retry_backoff_action_steps: 4 };

/**
 * v6 (`ora.native-context-management.v6`, the default of LIBERO-Pro,
 * RoboTwin, RoboCasa365 and RoboLab): materialize (Pointers over the large-
 * result size) -> stride_aging -> eager_externalize (L1/L2) -> media_window
 * (L3) -> pressure_externalize (L4) -> episode_capsule (L5) ->
 * emergency_checkpoint (L6) -> over_limit_guard (refuse). With
 * `backstop_exact_tail_exchanges` the guard is the hard backstop, aging by
 * itself (`aging: own`): every request is estimated before it is sent and, at
 * or above L6's watermark (`trigger_ratio`: emergency_fallback_ratio of the
 * hard limit), all but that many exchanges age, the open ActionStep's
 * included; with `truncated_reply_reasoning: withheld` materialization sends a
 * reply the output limit cut off before any tool call without its reasoning.
 */
function schemeV6(management: Record<string, any>, options: LegacyOptions): StageScheme {
  const largeResultBytes = pyInt(num(management.eager_large_result_bytes));
  const backstop = management.backstop_exact_tail_exchanges;
  const guard: StageSpec = backstop === undefined
    ? { stage: OVER_LIMIT_GUARD, params: { over_limit: "refuse" } }
    : {
      stage: HARD_BACKSTOP,
      // It ages at L6's watermark, by the request's own estimate, where L6 is a call late or has nothing to absorb.
      params: {
        exact_tail_exchanges: pyInt(num(backstop)), aging: "own",
        trigger_ratio: num(management.emergency_fallback_ratio),
      },
    };
  const truncated = management.truncated_reply_reasoning;
  return {
    source: management,
    segments: { unit: "action_step", exact_tail: pyInt(num(management.causal_exact_tail_action_steps)) },
    limits: { reserve_tokens: RESERVE_TOKENS },
    estimator: { image_tokens: IMAGE_PRESSURE_ESTIMATE_TOKENS },
    // Results over the L1 size advertise their Pointer at first delivery; references keep a head preview.
    references: {
      pointers: true, advertise_over_bytes: largeResultBytes,
      preview: { head_bytes: pyInt(num(management.eager_preview_bytes)) },
    },
    stages: [
      {
        stage: MATERIALIZE,
        params: { reasoning: "exclude", ...(truncated === undefined ? {} : { truncated_reply_reasoning: String(truncated) }) },
      },
      { stage: STRIDE_AGING, params: { stride: Math.trunc(num(management.tail_aging_stride_action_steps)), strided: true } },
      {
        stage: EAGER_EXTERNALIZE,
        params: { large_result_bytes: largeResultBytes, deduplicate: true },
      },
      {
        stage: MEDIA_WINDOW,
        params: { tail_action_steps: Math.trunc(num(management.inline_media_tail_action_steps)) },
      },
      {
        stage: PRESSURE_EXTERNALIZE,
        params: {
          inline_images: pyInt(num(management.trigger_inline_images)),
          estimated_tokens: pyInt(num(management.trigger_estimated_tokens)),
          window_share: ESTIMATE_PRESSURE_WINDOW_SHARE,
          provider_input_ratio: num(management.trigger_provider_input_ratio),
          enabled: options.pressureExternalization,
        },
      },
      {
        stage: EPISODE_CAPSULE,
        params: {
          trigger_tokens: pyInt(num(management.episode_checkpoint_trigger_tokens)),
          trigger_ratio: null,
          ...checkpointLimits(management),
          ...EPISODE_RETRIES,
          enabled: options.rolloutCompaction,
        },
      },
      {
        stage: EMERGENCY_CHECKPOINT,
        params: {
          trigger_tokens: null,
          trigger_ratio: num(management.emergency_fallback_ratio),
          target_ratio: num(management.emergency_target_ratio),
          ...checkpointLimits(management),
          enabled: options.rolloutCompaction,
        },
      },
      guard,
    ],
  };
}

/** v8's and v9's camera delivery and summary stages. */
function appendOnlyStages(management: Record<string, any>, options: LegacyOptions): Record<string, StageSpec> {
  if (!CAMERA_IMAGE_MODES.has(management.camera_images)) throw new Error("camera_images is not validated");
  const checkpoint = checkpointLimits(management);
  return {
    camera: {
      stage: CAMERA_DELIVERY,
      params: {
        mode: String(management.camera_images),
        tail_observations: management.camera_tail_observations === null
          ? null : pyInt(num(management.camera_tail_observations)),
      },
    },
    episode: {
      stage: EPISODE_CAPSULE,
      params: {
        // v10's L5 may be by ratio alone (tokens null); before v10 it is absolute.
        trigger_tokens: management.episode_checkpoint_trigger_tokens === null
          ? null : pyInt(num(management.episode_checkpoint_trigger_tokens)),
        trigger_ratio: management.episode_checkpoint_trigger_ratio == null
          ? null : num(management.episode_checkpoint_trigger_ratio),
        ...checkpoint,
        ...EPISODE_RETRIES,
        enabled: options.rolloutCompaction,
      },
    },
    emergency: {
      stage: EMERGENCY_CHECKPOINT,
      params: {
        // v9 may leave L6 ratio-only (null); v8's is always absolute.
        trigger_tokens: management.compaction_trigger_tokens === null
          ? null : pyInt(num(management.compaction_trigger_tokens)),
        trigger_ratio: num(management.emergency_fallback_ratio),
        target_ratio: num(management.emergency_target_ratio),
        ...checkpoint,
        enabled: options.rolloutCompaction,
      },
    },
  };
}

/**
 * v8 (`ora.native-context-management.v8`; its scheme `context_rolling_tail_v2` was retired on 2026-11-05):
 * materialize (no Pointer advertised) -> camera_delivery -> episode_capsule
 * (L5) -> emergency_checkpoint (L6) -> over_limit_guard (send and record).
 */
function schemeV8(management: Record<string, any>, options: LegacyOptions): StageScheme {
  const { camera, episode, emergency } = appendOnlyStages(management, options);
  return {
    source: management,
    segments: { unit: "action_step", exact_tail: pyInt(num(management.causal_exact_tail_action_steps)) },
    limits: { reserve_tokens: RESERVE_TOKENS },
    estimator: { image_tokens: pyInt(num(management.image_estimate_tokens)) },
    // v8 shows no artifact Pointer: no tool result is advertised, no summary slot names its transcript.
    references: NO_POINTERS,
    stages: [
      { stage: MATERIALIZE, params: { reasoning: "exclude" } },
      camera,
      episode,
      emergency,
      { stage: OVER_LIMIT_GUARD, params: { over_limit: "send_and_record" } },
    ],
  };
}

/**
 * v9 (`ora.native-context-management.v9`; its schemes were retired on 2026-11-05): v8
 * counted in exchanges, plus exchange aging. materialize -> exchange_aging ->
 * reasoning_replay -> camera_delivery (`exchange_window_v1` ages camera images
 * with their exchange) -> episode_capsule (L5) -> emergency_checkpoint (L6) ->
 * hard_backstop. Its references show the Pointers an aging leaves (and the
 * summary slots name their transcript); no result is advertised at first
 * delivery. The aged preview's budget is split head and tail, the tail the
 * smaller half.
 */
function schemeV9(management: Record<string, any>, options: LegacyOptions): StageScheme {
  const { camera, episode, emergency } = appendOnlyStages(management, options);
  const budget = pyInt(num(management.aged_preview_bytes));
  const tail = Math.floor(budget / 2);
  return {
    source: management,
    segments: { unit: "exchange", exact_tail: pyInt(num(management.causal_exact_tail_action_steps)) },
    limits: { reserve_tokens: RESERVE_TOKENS },
    estimator: { image_tokens: pyInt(num(management.image_estimate_tokens)) },
    references: { pointers: true, advertise_over_bytes: null, preview: { head_bytes: 0 } },
    stages: [
      { stage: MATERIALIZE, params: { reasoning: "exclude" } },
      {
        stage: EXCHANGE_AGING,
        params: {
          trigger_tokens: pyInt(num(management.exchange_aging_trigger_tokens)),
          trigger_ratio: num(management.exchange_aging_trigger_ratio),
          exact_tail_exchanges: pyInt(num(management.exact_tail_exchanges)),
          argument_bytes: pyInt(num(management.aged_argument_bytes)),
          result_bytes: pyInt(num(management.aged_result_bytes)),
          preview: { head_bytes: budget - tail, tail_bytes: tail },
        },
      },
      {
        stage: REASONING_REPLAY,
        params: {
          mode: String(management.reasoning_replay),
          tail_exchanges: management.reasoning_tail_exchanges === null
            ? null : pyInt(num(management.reasoning_tail_exchanges)),
        },
      },
      camera,
      episode,
      emergency,
      { stage: HARD_BACKSTOP, params: { exact_tail_exchanges: pyInt(num(management.backstop_exact_tail_exchanges)) } },
    ],
  };
}

/**
 * v10 (`ora.native-context-management.v10`): v9's stages with what v10 lets a
 * policy turn off or change: arguments that never age (`aged_argument_bytes`
 * null, so no assistant message is rewritten), aged results that keep their
 * structured fields (`aged_result_preview`), a rewritten message that keeps
 * unsigned reasoning (`reasoning_binding`), L5 by ratio, and L6 and the exact
 * tail in exchanges (`compaction_unit: exchange`: the segments count the
 * aging's `exact_tail_exchanges`, and an aging leaves the latest robot
 * action's exchange as it is, like the exact tail).
 */
function schemeV10(management: Record<string, any>, options: LegacyOptions): StageScheme {
  const v9 = schemeV9(management, options);
  const exchangeUnit = management.compaction_unit === "exchange";
  return {
    ...v9,
    segments: exchangeUnit
      ? { unit: "exchange", exact_tail: pyInt(num(management.exact_tail_exchanges)), exact_tail_unit: "exchange" }
      : v9.segments,
    stages: v9.stages.map((spec) => {
      if (spec.stage === EXCHANGE_AGING) {
        return { stage: spec.stage, params: {
          ...spec.params,
          argument_bytes: management.aged_argument_bytes === null ? null : pyInt(num(management.aged_argument_bytes)),
          result_preview: String(management.aged_result_preview),
          ...(exchangeUnit ? { keep_anchor: true } : {}),
        } };
      }
      if (spec.stage === REASONING_REPLAY) {
        return { stage: spec.stage, params: { ...spec.params, binding: String(management.reasoning_binding) } };
      }
      if (spec.stage === EMERGENCY_CHECKPOINT && exchangeUnit) {
        return { stage: spec.stage, params: { ...spec.params, unit: "exchange" } };
      }
      return spec;
    }),
  };
}
