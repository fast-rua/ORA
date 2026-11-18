// The pipeline-state fields of the `ora.state` snapshot (core/persistence.ts
// `PIPELINE_STATE_FIELDS`, plus the retry map and the over-limit count) at
// their initial values. A stage scheme serializes its slices into these fields
// and keeps the initial value of a field no stage of it owns; the
// `rewrite_compact_v1` pipeline owns only its usage anchor. The exchange
// stages' fields (`EXCHANGE_FIELDS`) are in the snapshot only for a scheme that
// runs them, as one `exchangeAging` object. Changing the snapshot format is a
// separate, blessed step.

import { PolicySessionError, canonicalBytesPy } from "../pyutil.ts";

export const INITIAL_PIPELINE_FIELDS: Record<string, () => unknown> = {
  lastEstimatedTokens: () => null,
  lastProviderInputTokens: () => null,
  lastProviderProjectionEpoch: () => null,
  lastProviderWireRequestSha256: () => null,
  pendingSemanticTargetEpoch: () => null,
  pendingSemanticTargetTokens: () => null,
  lastSemanticTargetValidation: () => null,
  cacheEpoch: () => 0,
  absorbedActionStepCount: () => 0,
  lifecycleMediaExternalizedActionStepCount: () => 0,
  lifecycleReadOnlyExternalizedActionStepCount: () => 0,
  lifecycleSupersededToolImageEventIds: () => [],
  pressureExternalizedActionStepCount: () => 0,
  episodeCheckpointedActionStepCount: () => 0,
  episodeCheckpointedRounds: () => [],
  episodeCheckpointCount: () => 0,
  episodeCapsules: () => [],
  emergencySummarizedActionStepCount: () => 0,
  emergencyCompactionCount: () => 0,
  emergencyCheckpoint: () => null,
  emergencyCheckpointSha256: () => null,
  emergencyCheckpointRef: () => null,
  emergencyTranscriptRef: () => null,
  lastSemanticRejection: () => null,
  semanticRejectionCount: () => 0,
  episodeRetryState: () => new Map(),
  overLimitRequestCount: () => 0,
};

/**
 * The exchange stages' fields (exchange_aging, reasoning_replay,
 * hard_backstop) at their initial values, in the snapshot's order: present
 * only when a stage of the scheme owns one, then all of them.
 */
export const EXCHANGE_FIELDS: Record<string, () => unknown> = {
  exchangeAgedCount: () => 0,
  exchangeAgedObservationGroups: () => 0,
  reasoningWithheldExchangeCount: () => 0,
  exchangeAgingCount: () => 0,
  hardBackstopCount: () => 0,
};

/**
 * An emergency checkpoint that counts in exchanges: its exchange frontier and
 * the exchange it kept exact, in the snapshot (as `emergencyExchangeCoverage`)
 * only for a scheme whose checkpoint counts in exchanges.
 */
export const EMERGENCY_EXCHANGE_FIELDS: Record<string, () => unknown> = {
  emergencySummarizedExchangeCount: () => 0,
  emergencyKeptExchange: () => null,
};

/** exchange_aging `keep_anchor`: the exchange it left as it is, in the snapshot's `exchangeAging` when kept. */
export const EXCHANGE_KEPT_FIELDS: Record<string, () => unknown> = {
  exchangeKeptExchange: () => null,
};

/** Optional field groups: present only when a stage of the scheme owns one of them, then whole. */
const OPTIONAL_GROUPS = [EXCHANGE_FIELDS, EMERGENCY_EXCHANGE_FIELDS, EXCHANGE_KEPT_FIELDS];

/** Maps as entry lists, for comparisons. */
export function serializable(value: unknown): unknown {
  if (value instanceof Map) return [...value.entries()];
  if (value !== null && typeof value === "object" && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype) {
    const out: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) out[key] = item instanceof Map ? [...item.entries()] : item;
    return out;
  }
  return value;
}

/** ``owned`` completed with the initial value of every field it lacks (exchange fields: when it has one). */
export function withInitialFields(owned: Record<string, unknown>): Record<string, any> {
  const fields: Record<string, any> = { ...owned };
  for (const [field, initial] of Object.entries(INITIAL_PIPELINE_FIELDS)) {
    if (!(field in fields)) fields[field] = initial();
  }
  for (const group of OPTIONAL_GROUPS) {
    if (!Object.keys(group).some((field) => field in fields)) continue;
    for (const [field, initial] of Object.entries(group)) {
      if (!(field in fields)) fields[field] = initial();
    }
  }
  return fields;
}

/** Refuse restored fields outside ``owned`` that moved from their initial value. */
export function checkUnownedFields(fields: Record<string, any>, owned: Set<string>, owner: string): void {
  for (const [field, initial] of Object.entries({
    ...INITIAL_PIPELINE_FIELDS, ...EXCHANGE_FIELDS, ...EMERGENCY_EXCHANGE_FIELDS, ...EXCHANGE_KEPT_FIELDS,
  })) {
    if (owned.has(field) || !(field in fields)) continue;
    if (!canonicalBytesPy(serializable(fields[field])).equals(canonicalBytesPy(serializable(initial())))) {
      throw new PolicySessionError(`context state carries ${field}, which ${owner} does not run`);
    }
  }
}
