// Context management v4 and v5 as stage schemes, for the differential fixtures
// recorded with them (tests/data/context_differential). The agent core refuses
// v4, v5 and v7 plans (src/context/management.ts); these fixtures keep
// exercising the stages on behaviour no live run reaches (L5 rejection and
// backoff, an L6 checkpoint then an L5 capsule, deferred L5/L6, hard limits).
//
// v4 and v5 differ from v6 in two stage parameters (src/context/stages/):
// `stride_aging.strided` (v4 ages every frontier at each observation, stride 1)
// and `episode_capsule.trigger_ratio` (their L5 watermark scales with the
// window). The manifest echoes the original management object.

import {
  CONTEXT_MANAGEMENT_V4_SCHEMA, CONTEXT_MANAGEMENT_V5_SCHEMA, CONTEXT_MANAGEMENT_V6_SCHEMA,
} from "../../src/context/management.ts";
import { num } from "../../src/context/pyutil.ts";
import { EPISODE_CAPSULE, STRIDE_AGING } from "../../src/context/stages/names.ts";
import { type LegacyOptions, type StageScheme, schemeFromManagement } from "../../src/context/stages/scheme.ts";

/** The stage scheme of a v4 or v5 management object; null for any other. */
export function legacyStageScheme(
  management: Record<string, any> | null | undefined,
  options: LegacyOptions = { pressureExternalization: true, rolloutCompaction: true },
): StageScheme | null {
  const schema = management?.schema;
  if (schema !== CONTEXT_MANAGEMENT_V4_SCHEMA && schema !== CONTEXT_MANAGEMENT_V5_SCHEMA) return null;
  const { episode_checkpoint_trigger_ratio: ratio, ...fields } = management!;
  const strided = schema === CONTEXT_MANAGEMENT_V5_SCHEMA;
  const scheme = schemeFromManagement({
    ...fields, schema: CONTEXT_MANAGEMENT_V6_SCHEMA,
    tail_aging_stride_action_steps: strided ? fields.tail_aging_stride_action_steps : 1,
  }, options);
  for (const spec of scheme.stages) {
    if (spec.stage === STRIDE_AGING) spec.params.strided = strided;
    if (spec.stage === EPISODE_CAPSULE) spec.params.trigger_ratio = num(ratio);
  }
  return { ...scheme, source: management! };
}
