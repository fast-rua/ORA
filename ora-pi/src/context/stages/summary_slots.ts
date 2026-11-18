// The summarize phase's shared layout: the exact protected prefix, the summary
// slots, the current round's static-facts slots of the summarized span, then
// the exact tail (`tieredSemanticProjection`, projection.ts). An emergency
// checkpoint that counts in exchanges (`exchangeLayout`) keeps the exchanges
// from its frontier on and, before them, the anchor exchange it left exact.
//
// The summarize stages render together: each contributes the slot inputs it
// owns (`Stage.slots`), and the layout places an active emergency checkpoint
// before the Episode capsules it does not cover.

import {
  type Message, emergencyCheckpointSlot, observationGroups, semanticCapsuleSlot, tieredSemanticProjection,
} from "../projection.ts";
import { PolicySessionError, clonePy, get, isInt, isMapping, num } from "../pyutil.ts";
import * as roundStaticFacts from "../round_static_facts.ts";
import { exchangesOf } from "./services/segmenter.ts";
import type { SlotContribution, SlotInputs } from "./stage.ts";

const EMPTY: Omit<SlotInputs, "transcriptPointers"> = {
  episodeCheckpointedActionSteps: 0,
  episodeCapsules: [],
  emergencySummarizedActionSteps: 0,
  emergencyCheckpoint: null,
  emergencyTranscriptRef: null,
  emergencyUnit: "action_step",
  emergencySummarizedExchanges: 0,
  emergencyKeptExchange: null,
};

/** Merge the summarize stages' contributions; two stages may not own one input. */
export function mergeSlotInputs(contributions: SlotContribution[], transcriptPointers: boolean): SlotInputs {
  const merged: Record<string, unknown> = {};
  for (const contribution of contributions) {
    for (const [key, value] of Object.entries(contribution)) {
      if (key in merged) throw new PolicySessionError(`two summary stages own the slot input ${key}`);
      merged[key] = value;
    }
  }
  return { ...EMPTY, ...merged, transcriptPointers } as SlotInputs;
}

/** ActionSteps the summary slots stand in for. */
export function summarizedActionSteps(inputs: SlotInputs): number {
  return Math.max(inputs.episodeCheckpointedActionSteps, inputs.emergencySummarizedActionSteps);
}

export function layoutSummarySlots(history: Message[], inputs: SlotInputs): Message[] {
  if (inputs.emergencyUnit === "exchange" && inputs.emergencySummarizedExchanges > 0) {
    return exchangeLayout(history, inputs);
  }
  return tieredSemanticProjection(history, inputs);
}

/**
 * The layout of an emergency checkpoint that counts in exchanges: the
 * protected prefix, the checkpoint (while an Episode capsule has not
 * superseded it) and the capsules it does not cover, the current round's
 * static-facts slots of the summarized span, the anchor exchange the
 * checkpoint kept exact, then every message from its frontier on.
 */
function exchangeLayout(history: Message[], inputs: SlotInputs): Message[] {
  const groups = observationGroups(history);
  const exchanges = exchangesOf(history, groups);
  const spans = exchanges.spans;
  const covered = inputs.emergencySummarizedExchanges;
  const emergencyBoundary = covered < spans.length ? spans[covered][0] : history.length;
  const episodeBoundary = inputs.episodeCheckpointedActionSteps ? groups[inputs.episodeCheckpointedActionSteps][0] : 0;
  const activeEmergency = inputs.emergencyCheckpoint !== null && emergencyBoundary > episodeBoundary;
  const tailStart = Math.max(activeEmergency ? emergencyBoundary : 0, episodeBoundary, groups[0][0]);
  const kept = inputs.emergencyKeptExchange;
  const keptSpan = kept !== null && kept < spans.length && spans[kept][1] <= tailStart ? spans[kept] : null;
  const current = groups[groups.length - 1][0];
  if (current < tailStart && !(keptSpan !== null && keptSpan[0] <= current && current < keptSpan[1])) {
    throw new PolicySessionError("semantic projection must retain the current observation");
  }
  const prefix = history.slice(0, groups[0][0])
    .filter((message) => get(message, "_protected_context") === true)
    .map((message) => clonePy({ ...message }));
  const pointers = inputs.transcriptPointers;
  const slots: Message[] = [];
  if (activeEmergency) {
    slots.push(emergencyCheckpointSlot(inputs.emergencyCheckpoint, inputs.emergencyTranscriptRef, pointers));
  }
  for (const capsule of inputs.episodeCapsules) {
    const coverage = get(capsule, "coverage");
    const through = isMapping(coverage) ? get(coverage, "through_action_step_index") : undefined;
    if (activeEmergency && (isInt(through) || typeof through === "boolean")
      && num(through) < inputs.emergencySummarizedActionSteps) {
      continue;
    }
    slots.push(semanticCapsuleSlot(capsule, pointers));
  }
  const summarized = history.slice(groups[0][0], tailStart)
    .filter((_message, offset) => keptSpan === null
      || !(keptSpan[0] <= groups[0][0] + offset && groups[0][0] + offset < keptSpan[1]));
  // native_slots_v4: a summarized span never takes the current round's slot with it.
  const carried = roundStaticFacts.currentRoundSlots(summarized, get(history[current], "_session_round") ?? null);
  const keptMessages = keptSpan === null ? [] : history.slice(...keptSpan);
  return [...prefix, ...slots, ...carried, ...clonePy([...keptMessages, ...history.slice(tailStart)])];
}
