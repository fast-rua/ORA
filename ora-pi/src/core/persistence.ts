// Durable ORA state inside the Pi session file, for a restarted agent process.
//
// Pi's session JSONL is the conversation's durable store. ORA writes three
// kinds of custom entries into it (pi.appendEntry: persisted, never sent to the
// model):
//   ora.history  canonical ORA messages, in history order, images as artifact
//                references (base64 never enters the session file)
//   ora.state    the context policy's frontiers/epochs/checkpoints and the
//                core's counters, appended at each flush (latest wins); the
//                session manifest's call records go to native/pi-calls.jsonl,
//                and ora.state counts how many of them (and bytes) are durable
// agent-events.jsonl stays the evidence authority; a resumed process derives
// event-bound metadata (assistant source event ids) from it.

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { EMERGENCY_EXCHANGE_FIELDS, EXCHANGE_FIELDS } from "../context/stages/state_fields.ts";
import { clonePy } from "../pyjson.ts";

type Json = any;

/** Pipeline fields that are state (caches and configuration excluded). */
export const PIPELINE_STATE_FIELDS = [
  "lastEstimatedTokens", "lastProviderInputTokens", "lastProviderProjectionEpoch",
  "lastProviderWireRequestSha256", "pendingSemanticTargetEpoch", "pendingSemanticTargetTokens",
  "lastSemanticTargetValidation", "cacheEpoch", "absorbedActionStepCount",
  "lifecycleMediaExternalizedActionStepCount", "lifecycleReadOnlyExternalizedActionStepCount",
  "lifecycleSupersededToolImageEventIds", "pressureExternalizedActionStepCount",
  "episodeCheckpointedActionStepCount", "episodeCheckpointedRounds", "episodeCheckpointCount",
  "episodeCapsules", "emergencySummarizedActionStepCount", "emergencyCompactionCount",
  "emergencyCheckpoint", "emergencyCheckpointSha256", "emergencyCheckpointRef", "emergencyTranscriptRef",
  "lastSemanticRejection", "semanticRejectionCount",
];

// The pipeline serializes its state into these fields (`pipelineFields`): a
// stage-driven pipeline (context/stages/driver.ts) one slice per stage, the
// rewrite pipeline its usage anchor; a field the pipeline does not keep has its
// initial value, so the snapshot format does not depend on which it is.

export function snapshotPipeline(target: Json): Record<string, Json> {
  const pipeline = target.pipelineFields();
  const state: Record<string, Json> = {};
  for (const field of PIPELINE_STATE_FIELDS) state[field] = clonePy(pipeline[field] ?? null);
  state.episodeRetryState = pipeline.episodeRetryState instanceof Map ? [...pipeline.episodeRetryState.entries()] : [];
  // Context v8 only; absent (zero) for every other retention, so their state is unchanged.
  if (pipeline.overLimitRequestCount) state.overLimitRequestCount = pipeline.overLimitRequestCount;
  // The exchange stages' frontiers and counters (v9), only for a scheme that runs them.
  if (Object.prototype.hasOwnProperty.call(pipeline, "exchangeAgedCount")) {
    state.exchangeAging = Object.fromEntries(Object.keys(EXCHANGE_FIELDS).map((field) => [field, pipeline[field]]));
    // keep_anchor's kept exchange, only for a scheme that keeps one.
    if (Object.prototype.hasOwnProperty.call(pipeline, "exchangeKeptExchange")) {
      state.exchangeAging.exchangeKeptExchange = pipeline.exchangeKeptExchange;
    }
  }
  // An emergency checkpoint that counts in exchanges, likewise.
  if (Object.prototype.hasOwnProperty.call(pipeline, "emergencySummarizedExchangeCount")) {
    state.emergencyExchangeCoverage = Object.fromEntries(
      Object.keys(EMERGENCY_EXCHANGE_FIELDS).map((field) => [field, clonePy(pipeline[field])]));
  }
  return state;
}

export function restorePipeline(target: Json, state: Record<string, Json>): void {
  const pipeline = target.pipelineFields();
  for (const field of PIPELINE_STATE_FIELDS) {
    if (Object.prototype.hasOwnProperty.call(state, field)) pipeline[field] = clonePy(state[field]);
  }
  pipeline.episodeRetryState = new Map(state.episodeRetryState ?? []);
  pipeline.overLimitRequestCount = state.overLimitRequestCount ?? 0;
  if (state.exchangeAging) Object.assign(pipeline, clonePy(state.exchangeAging));
  if (state.emergencyExchangeCoverage) Object.assign(pipeline, clonePy(state.emergencyExchangeCoverage));
  target.loadPipelineFields(pipeline);
}

function artifactFor(runRoot: string, data: Buffer, mediaType: string): Record<string, Json> | null {
  const digest = createHash("sha256").update(data).digest("hex");
  for (const suffix of [mediaType === "image/png" ? ".png" : ".img", ".png", ".img"]) {
    const path = `native/agent-artifacts/${digest}${suffix}`;
    if (existsSync(join(runRoot, path))) return { path, sha256: digest, bytes: data.length, media_type: mediaType };
  }
  return null;
}

/** A copy whose data-URL images are artifact references (every image is in the store). */
export function dehydrate(message: Json, runRoot: string): Json {
  const copy = clonePy(message);
  if (!Array.isArray(copy.content)) return copy;
  copy.content = copy.content.map((block: Json) => {
    if (block?.type !== "image_url") return block;
    const url = String(block.image_url?.url ?? "");
    const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
    if (!match) return block;
    const ref = artifactFor(runRoot, Buffer.from(match[2], "base64"), match[1]);
    return ref === null ? block : { type: "ora_image_ref", ref };
  });
  return copy;
}

export function hydrate(message: Json, runRoot: string): Json {
  const copy = clonePy(message);
  if (!Array.isArray(copy.content)) return copy;
  copy.content = copy.content.map((block: Json) => {
    if (block?.type !== "ora_image_ref") return block;
    const bytes = readFileSync(join(runRoot, String(block.ref.path)));
    return { type: "image_url", image_url: { url: `data:${block.ref.media_type};base64,${bytes.toString("base64")}` } };
  });
  return copy;
}
