// `ora/agent/context/lossless/management.py`: the closed configuration
// contracts of the Native Context management versions the agent core runs, v6,
// v8 and v9 (validation only; the field ledger's prose stays in Python). v6 is
// recoverable-first; v8 is append-only between rollout compactions and
// delivers fixed camera images by a camera-image mode; v9 is v8 plus exchange
// aging and the hard backstop, and adds a camera-image mode under which camera
// images age with their exchange; v10 is v9 with arguments that may never age
// and an L5 by ratio, plus the aged results' preview, the unit L6 and the
// exact tail count in, and whether a rewritten assistant message keeps its
// reasoning. v6 has two optional fields (2026-11-14): the hard backstop as its
// guard and truncated replies sent without their reasoning; a v6 object
// without them runs as before. A validated object runs as stages
// (stages/scheme.ts `schemeFromManagement`).
//
// v4, v5 and v7 are retired from the core: it refuses them. Python still
// validates them and evaluates the sealed runs that used them.

import { clonePy, hasKey, isInt, isMapping, isNumber, num, pyFloat, pyRepr, sortedStrings } from "./pyutil.ts";
import { PyFloat } from "../pyjson.ts";

export const CONTEXT_MANAGEMENT_V4_SCHEMA = "ora.native-context-management.v4";
export const CONTEXT_MANAGEMENT_V5_SCHEMA = "ora.native-context-management.v5";
export const CONTEXT_MANAGEMENT_V6_SCHEMA = "ora.native-context-management.v6";
export const CONTEXT_MANAGEMENT_V7_SCHEMA = "ora.native-context-management.v7";
export const CONTEXT_MANAGEMENT_V8_SCHEMA = "ora.native-context-management.v8";
export const CONTEXT_MANAGEMENT_V9_SCHEMA = "ora.native-context-management.v9";
export const CONTEXT_MANAGEMENT_V10_SCHEMA = "ora.native-context-management.v10";
/** The versions the agent core runs. */
export const LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS = new Set([
  CONTEXT_MANAGEMENT_V6_SCHEMA, CONTEXT_MANAGEMENT_V8_SCHEMA, CONTEXT_MANAGEMENT_V9_SCHEMA,
  CONTEXT_MANAGEMENT_V10_SCHEMA,
]);
/** The versions the agent core refuses; sealed runs that used them still evaluate in Python. */
export const RETIRED_CONTEXT_MANAGEMENT_SCHEMAS = new Set([
  CONTEXT_MANAGEMENT_V4_SCHEMA, CONTEXT_MANAGEMENT_V5_SCHEMA, CONTEXT_MANAGEMENT_V7_SCHEMA,
]);
/** Append-only between the rewrites they admit: no L1-L4, camera images by `camera_images`. */
export const APPEND_ONLY_SCHEMAS = new Set([
  CONTEXT_MANAGEMENT_V8_SCHEMA, CONTEXT_MANAGEMENT_V9_SCHEMA, CONTEXT_MANAGEMENT_V10_SCHEMA,
]);
/** Schemas that age exchanges into artifact Pointers at a watermark (v9). */
export const EXCHANGE_AGING_SCHEMAS = new Set([CONTEXT_MANAGEMENT_V9_SCHEMA, CONTEXT_MANAGEMENT_V10_SCHEMA]);
/** v10 `aged_result_preview`: the head and tail of the text, or the result's structured fields. */
export const RESULT_PREVIEWS = new Set(["head_and_tail", "structured"]);
/** v10 `compaction_unit`: what L6 absorbs and the exact tail counts in. */
export const COMPACTION_UNITS = new Set(["action_step", "exchange"]);
/** v10 `reasoning_binding`: a rewritten assistant message loses its reasoning, or keeps it when it is unsigned. */
export const REASONING_BINDINGS = new Set(["strict", "rebind_unsigned"]);
/** v9 `reasoning_replay`: every exchange, the unaged ones (an aging withholds the aged ones'), or a recent window. */
export const REASONING_REPLAY_ALL = "all";
export const REASONING_REPLAY_UNAGED = "unaged_exchanges";
export const REASONING_REPLAY_RECENT = "recent_exchanges";
export const REASONING_REPLAY_MODES = new Set([REASONING_REPLAY_ALL, REASONING_REPLAY_UNAGED, REASONING_REPLAY_RECENT]);
export const CAMERA_ROLLING_TAIL = "rolling_tail_v1";
export const CAMERA_INLINE_UNTIL_COMPACTION = "inline_until_compaction_v1";
/** v9 only: camera images stay in their observation message until its exchange ages. */
export const CAMERA_EXCHANGE_WINDOW = "exchange_window_v1";
export const CAMERA_IMAGE_MODES = new Set([CAMERA_ROLLING_TAIL, CAMERA_INLINE_UNTIL_COMPACTION, CAMERA_EXCHANGE_WINDOW]);
export const MIN_CAMERA_TAIL_OBSERVATIONS = 2;
export const TAIL_AGING_STRIDE_FIELD = "tail_aging_stride_action_steps";
/** v8: the request estimate's tokens per inline image (v6 uses the fixed default). */
export const IMAGE_ESTIMATE_TOKENS_FIELD = "image_estimate_tokens";

type Kind = "schema" | "positive_int" | "positive_int_or_null" | "nonnegative_int" | "required_true" | "ratio"
  | "ratio_or_null" | "camera_images" | "camera_window" | "reasoning_replay" | "reasoning_window" | "result_preview"
  | "compaction_unit" | "reasoning_binding" | "truncated_reply_reasoning";

export const CONTEXT_MANAGEMENT_V6_FIELD_KINDS: [string, Kind][] = [
  ["schema", "schema"],
  ["causal_exact_tail_action_steps", "positive_int"],
  ["inline_media_tail_action_steps", "nonnegative_int"],
  ["eager_large_result_bytes", "positive_int"],
  ["eager_preview_bytes", "positive_int"],
  ["eager_exact_deduplication", "required_true"],
  ["trigger_inline_images", "positive_int"],
  ["trigger_estimated_tokens", "positive_int"],
  ["trigger_provider_input_ratio", "ratio"],
  ["episode_checkpoint_trigger_tokens", "positive_int"],
  ["emergency_fallback_ratio", "ratio"],
  ["emergency_target_ratio", "ratio"],
  ["checkpoint_target_estimated_tokens", "positive_int"],
  ["checkpoint_max_estimated_tokens", "positive_int"],
  [TAIL_AGING_STRIDE_FIELD, "positive_int"],
];
/**
 * v6 fields a v6 object may leave out (v6 objects resolved before 2026-11-14 have neither):
 * `backstop_exact_tail_exchanges`, the guard estimates every request before it is sent and keeps
 * only that many exchanges exact once the estimate reaches L6's watermark (the hard backstop that
 * ages by itself; absent: a request the Provider reported over the limit is refused afterwards);
 * `truncated_reply_reasoning: withheld`, a reply the output limit cut off before any tool call is
 * sent without its reasoning (absent: replayed like any other).
 */
export const CONTEXT_MANAGEMENT_V6_OPTIONAL_FIELD_KINDS: [string, Kind][] = [
  ["backstop_exact_tail_exchanges", "positive_int"],
  ["truncated_reply_reasoning", "truncated_reply_reasoning"],
];
/** v6 `truncated_reply_reasoning`: a truncated reply's reasoning leaves the request. */
export const TRUNCATED_REPLY_REASONING_WITHHELD = "withheld";
export const CONTEXT_MANAGEMENT_V8_FIELD_KINDS: [string, Kind][] = [
  ["schema", "schema"],
  ["causal_exact_tail_action_steps", "positive_int"],
  ["camera_images", "camera_images"],
  ["camera_tail_observations", "camera_window"],
  ["episode_checkpoint_trigger_tokens", "positive_int"],
  ["compaction_trigger_tokens", "positive_int"],
  ["emergency_fallback_ratio", "ratio"],
  ["emergency_target_ratio", "ratio"],
  ["checkpoint_target_estimated_tokens", "positive_int"],
  ["checkpoint_max_estimated_tokens", "positive_int"],
  [IMAGE_ESTIMATE_TOKENS_FIELD, "positive_int"],
];
export const CONTEXT_MANAGEMENT_V9_FIELD_KINDS: [string, Kind][] = [
  // v9's L6 may be ratio-only: a null compaction_trigger_tokens leaves emergency_fallback_ratio alone.
  ...CONTEXT_MANAGEMENT_V8_FIELD_KINDS.map(([field, kind]): [string, Kind] => (
    field === "compaction_trigger_tokens" ? [field, "positive_int_or_null"] : [field, kind])),
  ["exchange_aging_trigger_tokens", "positive_int"],
  ["exchange_aging_trigger_ratio", "ratio"],
  ["exact_tail_exchanges", "positive_int"],
  ["aged_argument_bytes", "positive_int"],
  ["aged_result_bytes", "positive_int"],
  ["aged_preview_bytes", "positive_int"],
  ["backstop_exact_tail_exchanges", "positive_int"],
  ["reasoning_replay", "reasoning_replay"],
  ["reasoning_tail_exchanges", "reasoning_window"],
];

const NULLABLE_IN_V10: Record<string, Kind> = {
  episode_checkpoint_trigger_tokens: "positive_int_or_null",
  aged_argument_bytes: "positive_int_or_null",
};
export const CONTEXT_MANAGEMENT_V10_FIELD_KINDS: [string, Kind][] = [
  ...CONTEXT_MANAGEMENT_V9_FIELD_KINDS.flatMap(([field, kind]): [string, Kind][] => {
    const entry: [string, Kind] = [field, NULLABLE_IN_V10[field] ?? kind];
    if (field === "episode_checkpoint_trigger_tokens") return [entry, ["episode_checkpoint_trigger_ratio", "ratio_or_null"]];
    if (field === "aged_preview_bytes") return [entry, ["aged_result_preview", "result_preview"]];
    return [entry];
  }),
  ["compaction_unit", "compaction_unit"],
  ["reasoning_binding", "reasoning_binding"],
];

/** The closed Context configuration is incomplete, inconsistent or retired (ValueError). */
export class ContextManagementConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContextManagementConfigError";
  }
}

const VALIDATORS: Record<string, [string, [string, Kind][], [string, Kind][]?]> = {
  [CONTEXT_MANAGEMENT_V6_SCHEMA]: ["v6", CONTEXT_MANAGEMENT_V6_FIELD_KINDS, CONTEXT_MANAGEMENT_V6_OPTIONAL_FIELD_KINDS],
  [CONTEXT_MANAGEMENT_V8_SCHEMA]: ["v8", CONTEXT_MANAGEMENT_V8_FIELD_KINDS],
  [CONTEXT_MANAGEMENT_V9_SCHEMA]: ["v9", CONTEXT_MANAGEMENT_V9_FIELD_KINDS],
  [CONTEXT_MANAGEMENT_V10_SCHEMA]: ["v10", CONTEXT_MANAGEMENT_V10_FIELD_KINDS],
};

/** Validate the lossless-retention object against the schema it names; refuse a retired version. */
export function validateContextManagement(value: unknown): Record<string, any> {
  if (!isMapping(value)) throw new ContextManagementConfigError("context_management must be an object");
  const schema = value.schema;
  if (typeof schema === "string" && RETIRED_CONTEXT_MANAGEMENT_SCHEMAS.has(schema)) {
    throw new ContextManagementConfigError(
      `context_management ${schema} is retired from the agent core, which runs `
      + `${CONTEXT_MANAGEMENT_V6_SCHEMA}, ${CONTEXT_MANAGEMENT_V8_SCHEMA}, ${CONTEXT_MANAGEMENT_V9_SCHEMA} and `
      + `${CONTEXT_MANAGEMENT_V10_SCHEMA}; `
      + "sealed runs that used it still evaluate",
    );
  }
  const validator = typeof schema === "string" && hasKey(VALIDATORS, schema) ? VALIDATORS[schema] : undefined;
  if (validator === undefined) throw new ContextManagementConfigError("unsupported context_management schema");
  return validateClosed(value, validator[0], schema as string, validator[1], validator[2] ?? []);
}

function validateClosed(
  value: Record<string, any>, version: string, schema: string, required: [string, Kind][],
  optional: [string, Kind][],
): Record<string, any> {
  const fields = required.map(([field]) => field);
  const present = Object.keys(value);
  const missing = sortedStrings(fields.filter((field) => !hasKey(value, field)));
  const allowed = new Set([...fields, ...optional.map(([field]) => field)]);
  const extra = sortedStrings(present.filter((field) => !allowed.has(field)));
  // An optional field is checked like a required one when it is present.
  const specs = [...required, ...optional.filter(([field]) => hasKey(value, field))];
  if (missing.length || extra.length) {
    throw new ContextManagementConfigError(
      `context_management ${version} fields drifted; missing=${pyRepr(missing)}, extra=${pyRepr(extra)}`,
    );
  }
  if (value.schema !== schema) {
    throw new ContextManagementConfigError(`unsupported context_management ${version} schema`);
  }
  const normalized: Record<string, any> = clonePy({ ...value });
  for (const [field, kind] of specs) {
    const item = normalized[field];
    if (kind === "schema") continue;
    if (kind === "positive_int") {
      if (!isInt(item) || num(item) <= 0) {
        throw new ContextManagementConfigError(`context_management.${field} must be positive`);
      }
    } else if (kind === "positive_int_or_null") {
      if (item !== null && (!isInt(item) || num(item) <= 0)) {
        throw new ContextManagementConfigError(`context_management.${field} must be positive or null`);
      }
    } else if (kind === "nonnegative_int") {
      if (!isInt(item) || num(item) < 0) {
        throw new ContextManagementConfigError(`context_management.${field} must be non-negative`);
      }
    } else if (kind === "required_true") {
      if (item !== true) throw new ContextManagementConfigError(`context_management.${field} must be true`);
    } else if (kind === "ratio" || (kind === "ratio_or_null" && item !== null)) {
      if (!isNumber(item) || !(0 < num(item) && num(item) < 1)) {
        throw new ContextManagementConfigError(`context_management.${field} must be in (0,1)`);
      }
      normalized[field] = pyFloat(num(item));
    } else if (kind === "result_preview" || kind === "compaction_unit" || kind === "reasoning_binding") {
      const allowed = kind === "result_preview" ? RESULT_PREVIEWS
        : kind === "compaction_unit" ? COMPACTION_UNITS : REASONING_BINDINGS;
      if (typeof item !== "string" || !allowed.has(item)) {
        throw new ContextManagementConfigError(
          `context_management.${field} must be one of ${pyRepr(sortedStrings(allowed))}`,
        );
      }
    } else if (kind === "camera_images") {
      if (typeof item !== "string" || !CAMERA_IMAGE_MODES.has(item)) {
        throw new ContextManagementConfigError(
          `context_management.${field} must be one of ${pyRepr(sortedStrings(CAMERA_IMAGE_MODES))}`,
        );
      }
    } else if (kind === "camera_window" || kind === "reasoning_window") {
      if (item !== null && (!isInt(item) || num(item) <= 0)) {
        throw new ContextManagementConfigError(`context_management.${field} must be a positive integer or null`);
      }
    } else if (kind === "truncated_reply_reasoning") {
      if (item !== TRUNCATED_REPLY_REASONING_WITHHELD) {
        throw new ContextManagementConfigError(
          `context_management.${field} must be ${pyRepr(TRUNCATED_REPLY_REASONING_WITHHELD)} when present`,
        );
      }
    } else if (kind === "reasoning_replay") {
      if (typeof item !== "string" || !REASONING_REPLAY_MODES.has(item)) {
        throw new ContextManagementConfigError(
          `context_management.${field} must be one of ${pyRepr(sortedStrings(REASONING_REPLAY_MODES))}`,
        );
      }
    }
  }
  const n = (field: string) => num(normalized[field]);
  if (n("causal_exact_tail_action_steps") !== 3) {
    throw new ContextManagementConfigError("context_management.causal_exact_tail_action_steps must equal 3");
  }
  if (n("checkpoint_target_estimated_tokens") > n("checkpoint_max_estimated_tokens")) {
    throw new ContextManagementConfigError("checkpoint target must not exceed its configured soft maximum");
  }
  if (APPEND_ONLY_SCHEMAS.has(schema)) {
    checkAppendOnly(normalized, version, schema);
    if (EXCHANGE_AGING_SCHEMAS.has(schema)) checkExchangeAging(normalized, version);
    return normalized;
  }
  if (n("eager_preview_bytes") >= n("eager_large_result_bytes")) {
    throw new ContextManagementConfigError("eager preview must be smaller than the large-result threshold");
  }
  if (!(n("emergency_target_ratio") <= n("trigger_provider_input_ratio")
    && n("trigger_provider_input_ratio") < n("emergency_fallback_ratio"))) {
    throw new ContextManagementConfigError(
      `Context ${version} ratios require emergency target <= pressure trigger < emergency fallback`,
    );
  }
  if (!(n("trigger_estimated_tokens") < n("episode_checkpoint_trigger_tokens"))) {
    throw new ContextManagementConfigError(
      `Context ${version} requires the L4 estimate floor (trigger_estimated_tokens) below the absolute `
      + "L5 watermark (episode_checkpoint_trigger_tokens)",
    );
  }
  return normalized;
}

/** The camera window matches its mode, which the schema supports; L6 targets below its fallback. */
function checkAppendOnly(normalized: Record<string, any>, version: string, schema: string): void {
  const window = normalized.camera_tail_observations;
  if (normalized.camera_images === CAMERA_ROLLING_TAIL) {
    if (window === null || num(window) < MIN_CAMERA_TAIL_OBSERVATIONS) {
      throw new ContextManagementConfigError(
        `context_management.camera_tail_observations must be at least ${MIN_CAMERA_TAIL_OBSERVATIONS} for ${CAMERA_ROLLING_TAIL}`,
      );
    }
  } else if (window !== null) {
    throw new ContextManagementConfigError(
      `context_management.camera_tail_observations must be null for ${normalized.camera_images}`,
    );
  }
  if (normalized.camera_images === CAMERA_EXCHANGE_WINDOW && !EXCHANGE_AGING_SCHEMAS.has(schema)) {
    throw new ContextManagementConfigError(
      `context_management.camera_images ${CAMERA_EXCHANGE_WINDOW} needs exchange aging (Context ${version} has none)`,
    );
  }
  if (!(num(normalized.emergency_target_ratio) < num(normalized.emergency_fallback_ratio))) {
    throw new ContextManagementConfigError(`Context ${version} requires emergency target < emergency fallback`);
  }
}

/**
 * Aging precedes L6 on every window: its ratio below L6's, and its absolute
 * watermark below L6's when L6 has one (a ratio-only L6 is at least the ratio's
 * share of the window, which the aging watermark never exceeds). The tails and
 * the reasoning window agree.
 */
/** v10: L5 by tokens or by ratio, the ratio below L6's. */
function checkV10(normalized: Record<string, any>): void {
  const ratio = normalized.episode_checkpoint_trigger_ratio;
  if ((normalized.episode_checkpoint_trigger_tokens === null) === (ratio === null)) {
    throw new ContextManagementConfigError(
      "context_management sets exactly one of episode_checkpoint_trigger_tokens and episode_checkpoint_trigger_ratio",
    );
  }
  if (ratio !== null && !(num(ratio) < num(normalized.emergency_fallback_ratio))) {
    throw new ContextManagementConfigError(
      "Context v10 requires episode_checkpoint_trigger_ratio < emergency_fallback_ratio",
    );
  }
}

function checkExchangeAging(normalized: Record<string, any>, version: string): void {
  const n = (field: string) => num(normalized[field]);
  const absolute = normalized.compaction_trigger_tokens;
  if (!(n("exchange_aging_trigger_ratio") < n("emergency_fallback_ratio")
    && (absolute === null || n("exchange_aging_trigger_tokens") < num(absolute)))) {
    throw new ContextManagementConfigError(
      `Context ${version} requires the exchange aging watermark below the L6 watermark `
      + "(exchange_aging_trigger_ratio < emergency_fallback_ratio, and exchange_aging_trigger_tokens < "
      + "compaction_trigger_tokens when L6 has an absolute watermark)",
    );
  }
  if (!(n("backstop_exact_tail_exchanges") <= n("exact_tail_exchanges"))) {
    throw new ContextManagementConfigError(
      "context_management.backstop_exact_tail_exchanges must not exceed exact_tail_exchanges",
    );
  }
  if (!((normalized.aged_argument_bytes === null || n("aged_preview_bytes") < n("aged_argument_bytes"))
    && n("aged_preview_bytes") < n("aged_result_bytes"))) {
    throw new ContextManagementConfigError(
      "context_management.aged_preview_bytes must be smaller than the aged argument and result thresholds",
    );
  }
  if (hasKey(normalized, "compaction_unit")) checkV10(normalized);
  const window = normalized.reasoning_tail_exchanges;
  if (normalized.reasoning_replay !== REASONING_REPLAY_RECENT && window !== null) {
    throw new ContextManagementConfigError(
      `context_management.reasoning_tail_exchanges must be null for ${normalized.reasoning_replay}`,
    );
  }
  if (normalized.reasoning_replay === REASONING_REPLAY_RECENT && window === null) {
    throw new ContextManagementConfigError(
      `context_management.reasoning_tail_exchanges must be a positive integer for ${REASONING_REPLAY_RECENT}`,
    );
  }
}

export { PyFloat };
