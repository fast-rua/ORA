// The limits service: what a request may hold. The hard input limit is the
// window minus the maximum output minus a reserve; the target, telemetry only,
// is a share of the window minus the maximum output.

import { PolicySessionError, pyInt } from "../../pyutil.ts";

/** Tokens kept free beside the maximum output (`limits.reserve_tokens` of a scheme). */
export const RESERVE_TOKENS = 2048;
/** The share of the window minus the maximum output a request should stay under (telemetry). */
export const TARGET_INPUT_SHARE = 0.75;

/** The window's limits: hard = window - max output - reserve; target = TARGET_INPUT_SHARE x (window - max output). */
export interface Limits {
  contextWindowTokens: number;
  maxOutputTokens: number;
  reserveTokens: number;
  hardInputTokens: number;
  targetInputTokens: number;
}

export function resolveLimits(contextWindowTokens: number, maxOutputTokens: number, reserveTokens = RESERVE_TOKENS): Limits {
  if (contextWindowTokens <= maxOutputTokens + reserveTokens) {
    throw new PolicySessionError("native context window leaves no safe input budget");
  }
  return {
    contextWindowTokens,
    maxOutputTokens,
    reserveTokens,
    hardInputTokens: contextWindowTokens - maxOutputTokens - reserveTokens,
    targetInputTokens: pyInt((contextWindowTokens - maxOutputTokens) * TARGET_INPUT_SHARE),
  };
}
