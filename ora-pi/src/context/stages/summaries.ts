// What the summarize stages share (the plan's summary writer, G2): the ledger
// of rejected summaries, the compactor's identity checks, the checkpoint length
// telemetry and the protected task semantics every compaction packet carries.
//
// The functions are the Python Native pipeline's (ported in native.ts, moved
// here unchanged).

import type { SummaryResult } from "../native.ts";
import { type Message, providerMessage } from "../projection.ts";
import {
  ContextCompactionRejected, canonicalBytesPy, canonicalSha256, dumpsCompact, get, isInt, isMapping, num,
} from "../pyutil.ts";

/** Checkpoint length limits (estimated tokens): the target the compactor is asked for and the soft maximum. */
export interface CheckpointLimits {
  targetEstimatedTokens: number;
  maxEstimatedTokens: number;
}

/**
 * Rejected lossy summaries, shared by the summarize stages: how many, and the
 * latest (cleared when a summary is accepted). State fields
 * `semanticRejectionCount`, `lastSemanticRejection`.
 */
export class SummaryLedger {
  rejectionCount = 0;
  lastRejection: Record<string, any> | null = null;

  rejected(record: Record<string, any>): void {
    this.rejectionCount += 1;
    this.lastRejection = record;
  }

  accepted(): void {
    this.lastRejection = null;
  }
}

/** Python ``str(value)`` for the official round-result content. */
export function contentText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "None";
  return dumpsCompact(value);
}

/** The protected task and project messages before the first ActionStep. */
export function protectedSemantics(messages: Message[], groups: [number, number][]): [Message | null, Message | null] {
  const protectedMessages = messages.slice(0, groups[0][0])
    .filter((message) => get(message, "_protected_context") === true);
  const find = (source: string): Message | null => {
    const message = protectedMessages.find((item) => get(item, "_context_source") === source);
    return message === undefined ? null : providerMessage(message);
  };
  return [find("episode_task"), find("project_instructions")];
}

export function checkSummaryIdentity(summary: SummaryResult, sourceRefs: Record<string, any>[], label: string): void {
  if (!canonicalBytesPy([...summary.sourceRefs]).equals(canonicalBytesPy(sourceRefs))) {
    throw new ContextCompactionRejected(`${label} source references changed across the call`);
  }
  if (summary.sourceSetSha256 !== canonicalSha256(sourceRefs)) {
    throw new ContextCompactionRejected(`${label} source-set identity is invalid`);
  }
}

/** Telemetry of a checkpoint's length against its limits (never enforced). */
export function checkpointBudget(
  checkpointEstimatedTokens: number, metadata: Record<string, any>, limits: CheckpointLimits,
): Record<string, any> {
  const usage = get(metadata, "usage");
  let providerOutputTokens: unknown = isMapping(usage) ? get(usage, "output_tokens") : undefined;
  if (!isInt(providerOutputTokens) || num(providerOutputTokens) < 0) providerOutputTokens = null;
  const softMax = limits.maxEstimatedTokens;
  const observed = providerOutputTokens !== null ? num(providerOutputTokens) : checkpointEstimatedTokens;
  return {
    enforcement: "telemetry_only",
    authority: providerOutputTokens !== null ? "provider_reported_output_tokens" : "harness_model_dependent_estimate",
    provider_output_tokens: providerOutputTokens,
    estimated_tokens: checkpointEstimatedTokens,
    target_estimated_tokens: limits.targetEstimatedTokens,
    soft_max_estimated_tokens: softMax,
    status: observed <= softMax ? "within_soft_max" : "soft_max_exceeded",
  };
}
