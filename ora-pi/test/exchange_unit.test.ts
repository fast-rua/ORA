// The exchange unit of the exact tail and of the emergency checkpoint (L6):
// in code mode an ActionStep can hold dozens of exchanges, so an ActionStep
// unit never lets L6 absorb them (d6f33596dfed: L6 not admitted from call 106
// to 142 while input grew from 161k to 563k). With `exact_tail_unit: exchange`
// and L6 `unit: exchange`, L6 summarizes every exchange before the latest
// six, and keeps the anchor exchange (the latest action's: its call, its
// receipt and the observation after it) exact however old.
//
//   node --test ora-pi/test/exchange_unit.test.ts

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { ContextArtifactCatalog } from "../src/context/memory/artifacts.ts";
import { CHECKPOINT_FIELDS } from "../src/context/projection.ts";
import { NativeContextPipeline, pipelineState, providerEstimatedInputTokens } from "../src/context/native.ts";
import { canonicalSha256 } from "../src/context/pyutil.ts";
import { NativeRequestAssembler } from "../src/context/request_assembly.ts";
import { schemeFromManagement } from "../src/context/stages/scheme.ts";
import { exchangeTail, exchangesOf } from "../src/context/stages/services/segmenter.ts";
import { observationGroups } from "../src/context/projection.ts";
import { restorePipeline, snapshotPipeline } from "../src/core/persistence.ts";
import { ArtifactStore, NO_DURABILITY } from "../src/evidence/artifacts.ts";
import { prepareRequest } from "../src/provider/openai.ts";
import { STAGE_SCHEMAS } from "./support/stage_schemas.ts";

type Json = any;

const V9 = {
  schema: "ora.native-context-management.v9",
  causal_exact_tail_action_steps: 3, camera_images: "exchange_window_v1", camera_tail_observations: null,
  episode_checkpoint_trigger_tokens: 150_000, compaction_trigger_tokens: 150_000,
  emergency_fallback_ratio: 0.8, emergency_target_ratio: 0.65,
  checkpoint_target_estimated_tokens: 1200, checkpoint_max_estimated_tokens: 1600, image_estimate_tokens: 530,
  exchange_aging_trigger_tokens: 100_000, exchange_aging_trigger_ratio: 0.5, exact_tail_exchanges: 6,
  aged_argument_bytes: 1024, aged_result_bytes: 2048, aged_preview_bytes: 256,
  backstop_exact_tail_exchanges: 2, reasoning_replay: "all", reasoning_tail_exchanges: null,
};
const ON = { pressureExternalization: true, rolloutCompaction: true };

/** v9's stages with the exact tail and L6 in exchanges, L6 at 0.6 of the hard limit alone. */
function exchangeScheme(): Json {
  const scheme = schemeFromManagement(V9, ON);
  return {
    ...scheme,
    segments: { unit: "exchange", exact_tail: 6, exact_tail_unit: "exchange" },
    stages: scheme.stages.map((spec: Json) => spec.stage !== "emergency_checkpoint" ? spec : {
      stage: spec.stage,
      params: { ...spec.params, unit: "exchange", trigger_tokens: null, trigger_ratio: 0.6, target_ratio: 0.5 },
    }),
  };
}

let summaries = 0;
const compactor = {
  summarize: async ({ sourceRefs }: Json) => {
    summaries += 1;
    const body: Record<string, string[]> = {};
    for (const field of CHECKPOINT_FIELDS) body[field] = [];
    body.task_progress = [`checkpoint ${summaries}`];
    return {
      content: JSON.stringify(body), sourceRefs, sourceSetSha256: canonicalSha256(sourceRefs),
      metadata: { usage: { output_tokens: 40 } },
    };
  },
};

function pipeline(scheme: Json): NativeContextPipeline {
  const root = mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), "ora-exchange-unit-"));
  const artifacts = new ArtifactStore(root, { durability: NO_DURABILITY });
  return new NativeContextPipeline({
    profile: "native_causal_context_v1", artifacts, contextWindowTokens: 200_000, maxOutputTokens: 2048,
    retention: "action_step_lossless_v4", summaryPolicy: "rollout_checkpoint_v1",
    summaryProvider: compactor as Json, stageScheme: scheme, stageSchemas: STAGE_SCHEMAS,
    contextArtifacts: new ContextArtifactCatalog(root, "policy-session-1", artifacts),
    mutationTools: ["move_pose"],
  });
}

const PREFIX: Json[] = [
  { role: "system", content: "harness", _protected_context: true, _context_source: "harness_system" },
  { role: "user", content: "embodiment", _protected_context: true, _context_source: "project_instructions" },
  { role: "user", content: "task", _protected_context: true, _context_source: "episode_task" },
];

const event = (n: number): string => `agent-event-${String(n).padStart(6, "0")}`;

/** Observation ``step`` (the same round's next actual). */
function observation(step: number): Json {
  return {
    role: "user",
    content: [{ type: "text", text: JSON.stringify({ current_actual: { actual_sequence: step } }) }],
    _observation: { step, observation_id: `internal-observation-${step}`, keyframe: true },
    _causal_observation: { step, result_actual_sequence: step },
    _session_round: 1,
    _source_event_ids: [event(100_000 + step)],
  };
}

/** One exchange: a short script call and its result. */
function exchange(index: number): Json[] {
  return [
    { role: "assistant", content: "", tool_calls: [{
      id: `c${index}`, type: "function", function: { name: "report_issue", arguments: JSON.stringify({ message: `s${index}` }) },
    }], _source_event_ids: [event(2 * index)] },
    { role: "tool", tool_call_id: `c${index}`, name: "report_issue", content: JSON.stringify({ ok: true, output: `r${index}` }),
      _source_event_ids: [event(2 * index + 1)] },
  ];
}

const assemblers = new WeakMap<NativeContextPipeline, NativeRequestAssembler>();

/** Project and prepare one request (one assembler per pipeline: it checks the cache prefix); report the usage. */
async function send(context: NativeContextPipeline, messages: Json[], inputTokens: number): Promise<Json> {
  const projected = await context.project({ messages, tools: [], observationStep: 0 });
  if (!assemblers.has(context)) {
    assemblers.set(context, new NativeRequestAssembler({
      model: "m", reasoningEffort: "low", maxOutputTokens: 2048, stream: false,
      prepareRequest: (execution) => prepareRequest(execution),
    }));
  }
  assemblers.get(context)!.prepare(projected);
  context.observeProviderUsage({
    estimatedTokens: providerEstimatedInputTokens(projected.manifest), inputTokens,
    projectionEpoch: projected.manifest.projection_epoch,
    providerWireRequestSha256: projected.manifest.provider_wire_request_sha256,
  });
  return projected;
}

const callIds = (messages: Json[]): string[] => messages.filter((m) => m.role === "assistant")
  .map((m) => m.tool_calls[0].id);

test("the exchange tail is the latest exchanges plus the anchor, the exchange the current observation arrived in", () => {
  const messages = [...PREFIX, observation(0), ...exchange(0), observation(1)];
  for (let index = 1; index <= 10; index += 1) messages.push(...exchange(index));
  const groups = observationGroups(messages);
  const exchanges = exchangesOf(messages, groups);
  const tail = exchangeTail(messages, groups, exchanges, 6);
  assert.deepEqual([tail.from, tail.anchor], [5, 0]);
  // The anchor's span (the opening observation, call c0, its result, observation 1), then exchanges 5..10.
  assert.deepEqual(tail.ranges, [exchanges.spans[0], [exchanges.spans[5][0], messages.length]]);
  assert.deepEqual(exchangeTail([...PREFIX, observation(0)], observationGroups([...PREFIX, observation(0)]),
    exchangesOf([...PREFIX, observation(0)]), 6).ranges, [[3, 4]]);
});

test("L6 by exchange absorbs the exchanges of an open ActionStep and keeps the anchor exchange exact", async () => {
  const context = pipeline(exchangeScheme());
  const messages = [...PREFIX, observation(0), ...exchange(0), observation(1)];
  for (let index = 1; index <= 20; index += 1) messages.push(...exchange(index));
  await send(context, messages, 10_000);
  messages.push(...exchange(21));
  await send(context, messages, 150_000);  // over 0.6 x 189,952: L6 is admitted on the next request
  messages.push(...exchange(22));
  const projected = await send(context, messages, 20_000);
  const manifest = projected.manifest;
  const transition = manifest.projection_transitions.find((item: Json) => item.kind === "emergency_compaction");
  assert.ok(transition, "L6 ran with the ActionStep still open");
  assert.equal(manifest.open_action_step, 1);
  assert.equal(transition.unit, "exchange");
  // 23 exchanges, the latest 6 (17..22) stay; exchange 0 (the current observation's) is the anchor.
  assert.deepEqual(transition.absorbed_exchanges, Array.from({ length: 16 }, (_, k) => k + 1));
  assert.deepEqual([transition.emergency_summarized_exchanges_after, transition.kept_exchange], [17, 0]);
  assert.deepEqual(manifest.emergency_exchange_coverage,
    { unit: "exchange", summarized_exchanges: 17, through_exchange_index: 16, kept_exchange: 0 });
  assert.equal(manifest.exact_tail.from_exchange, 17);
  assert.equal(manifest.exact_tail.anchor_exchange, 0);
  assert.equal(manifest.exact_tail.exchange_inventory.length, 23);
  // The request: the checkpoint, the anchor exchange (observation 1 included), then the latest six.
  assert.deepEqual(callIds(projected.messages), ["c0", "c17", "c18", "c19", "c20", "c21", "c22"]);
  const text = (m: Json): string => typeof m.content === "string" ? m.content : JSON.stringify(m.content);
  const checkpointAt = projected.messages.findIndex((m: Json) => text(m).startsWith("<emergency_context_checkpoint>"));
  const callAt = (id: string) => projected.messages.findIndex((m: Json) => m.tool_calls?.[0]?.id === id);
  assert.ok(checkpointAt >= 0 && checkpointAt < callAt("c0") && callAt("c0") < callAt("c17"));
  // Observation 1 (the current one) follows c0's result inside the kept exchange.
  assert.equal(projected.messages[callAt("c0") + 1].role, "tool");
  assert.equal(projected.messages[callAt("c0") + 2].role, "user");
  // The state survives a restart.
  const snapshot = snapshotPipeline(context);
  assert.deepEqual(snapshot.emergencyExchangeCoverage, { emergencySummarizedExchangeCount: 17, emergencyKeptExchange: 0 });
  const restored = pipeline(exchangeScheme());
  restorePipeline(restored, snapshot);
  assert.deepEqual((pipelineState(restored) as Json).emergency_exchange_coverage,
    { summarized_exchanges: 17, kept_exchange: 0 });
});

test("after a new action, the next L6 absorbs the exchange it kept and keeps the new anchor", async () => {
  const context = pipeline(exchangeScheme());
  const messages = [...PREFIX, observation(0), ...exchange(0), observation(1)];
  for (let index = 1; index <= 20; index += 1) messages.push(...exchange(index));
  await send(context, messages, 150_000);
  messages.push(...exchange(21));
  await send(context, messages, 20_000);  // L6: 1..15 absorbed, 0 kept
  // A new action: exchange 22's call is followed by observation 2.
  messages.push(...exchange(22), observation(2));
  for (let index = 23; index <= 40; index += 1) messages.push(...exchange(index));
  await send(context, messages, 150_000);
  messages.push(...exchange(41));
  const projected = await send(context, messages, 20_000);
  const transitions = projected.manifest.projection_transitions.filter((item: Json) => item.kind === "emergency_compaction");
  assert.equal(transitions.length, 1);
  const [second] = transitions;
  // Exchange 0 is no longer the anchor: it is absorbed now, with 16..35 but not the new anchor 22.
  assert.equal(second.absorbed_exchanges[0], 0);
  assert.ok(!second.absorbed_exchanges.includes(22));
  assert.deepEqual([second.emergency_summarized_exchanges_after, second.kept_exchange], [36, 22]);
  assert.deepEqual(callIds(projected.messages), ["c22", "c36", "c37", "c38", "c39", "c40", "c41"]);
  assert.equal(projected.manifest.open_action_step, 2);
});

test("the checkpoint and the exact tail must count in the same unit", async () => {
  const scheme = exchangeScheme();
  const mixed = { ...scheme, segments: { unit: "exchange", exact_tail: 3 } };
  await assert.rejects(pipeline(mixed).project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 }),
    /emergency_checkpoint counts in exchange and the exact tail in action_step/);
});
