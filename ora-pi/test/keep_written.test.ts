// Context management v10 (src/context/management.ts, stages/scheme.ts
// schemeV10): the keep-written policy and its comparison arms. Arguments that
// never age leave every assistant message as it was written, so its reasoning
// stays bound; an aged result over the size keeps its structured fields; an
// aging leaves the latest robot action's exchange (the exact tail's anchor)
// as it is; a rewritten message keeps unsigned reasoning bound to the rewrite
// under `reasoning_binding: rebind_unsigned`; `unaged_exchanges` withholds the
// reasoning of what an aging ages, the anchor excepted. Each request goes
// through the OpenAI wire projection, which refuses a continuation that is not
// bound to its message.
//
//   node --test ora-pi/test/keep_written.test.ts

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { validateContextManagement } from "../src/context/management.ts";
import { ContextArtifactCatalog } from "../src/context/memory/artifacts.ts";
import { NativeContextPipeline, providerEstimatedInputTokens } from "../src/context/native.ts";
import { NativeRequestAssembler } from "../src/context/request_assembly.ts";
import { restorePipeline, snapshotPipeline } from "../src/core/persistence.ts";
import { schemeFromManagement } from "../src/context/stages/scheme.ts";
import { ArtifactStore, NO_DURABILITY } from "../src/evidence/artifacts.ts";
import { ANTHROPIC_CONTINUATION_KIND, ANTHROPIC_PROTOCOL } from "../src/provider/anthropic.ts";
import { ProviderContinuationState, UNSIGNED_CONTINUATION_KINDS } from "../src/provider/continuation.ts";
import { OPENAI_CONTINUATION_KIND, OPENAI_PROTOCOL, prepareRequest } from "../src/provider/openai.ts";
import { STAGE_SCHEMAS } from "./support/stage_schemas.ts";

type Json = any;

/** context_keep_written's management (ora/policy/context_schemes/keep-written.json), its watermark at 20,000. */
const KEEP = {
  schema: "ora.native-context-management.v10",
  causal_exact_tail_action_steps: 3, camera_images: "exchange_window_v1", camera_tail_observations: null,
  episode_checkpoint_trigger_tokens: null, episode_checkpoint_trigger_ratio: 0.15, compaction_trigger_tokens: null,
  emergency_fallback_ratio: 0.6, emergency_target_ratio: 0.5,
  checkpoint_target_estimated_tokens: 1200, checkpoint_max_estimated_tokens: 1600, image_estimate_tokens: 530,
  exchange_aging_trigger_tokens: 20_000, exchange_aging_trigger_ratio: 0.5, exact_tail_exchanges: 6,
  aged_argument_bytes: null, aged_result_bytes: 8192, aged_preview_bytes: 256, aged_result_preview: "structured",
  backstop_exact_tail_exchanges: 2, reasoning_replay: "all", reasoning_tail_exchanges: null,
  compaction_unit: "exchange", reasoning_binding: "strict",
};
/** context_age_scripts (arm B, retired 2026-11-05): scripts over 1,024 bytes age, their unsigned reasoning rebound. */
const SCRIPTS = { ...KEEP, aged_argument_bytes: 1024, reasoning_binding: "rebind_unsigned" };
/** context_keep_written_unaged_reasoning. */
const UNAGED = { ...KEEP, reasoning_replay: "unaged_exchanges" };

function pipeline(management: Json): NativeContextPipeline {
  const root = mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), "ora-keep-written-"));
  const artifacts = new ArtifactStore(root, { durability: NO_DURABILITY });
  return new NativeContextPipeline({
    profile: "native_causal_context_v1", artifacts, contextWindowTokens: 200_000, maxOutputTokens: 2048,
    retention: "action_step_lossless_v4", summaryPolicy: "rollout_checkpoint_v1",
    summaryProvider: { summarize: async () => { throw new Error("no compaction here"); } } as Json,
    contextManagement: management, contextArtifacts: new ContextArtifactCatalog(root, "policy-session-1", artifacts),
    mutationTools: ["move_pose"], stageSchemas: STAGE_SCHEMAS,
  });
}

/** The OpenAI wire projection; ``openai`` false: the execution request as it is (no binding check). */
function assembler(openai = true): NativeRequestAssembler {
  return new NativeRequestAssembler({
    model: "m", reasoningEffort: "low", maxOutputTokens: 2048, stream: false,
    prepareRequest: openai ? (execution) => prepareRequest(execution) : (execution) => structuredClone(execution),
  });
}

const PREFIX: Json[] = [
  { role: "system", content: "harness", _protected_context: true, _context_source: "harness_system" },
  { role: "user", content: "embodiment", _protected_context: true, _context_source: "project_instructions" },
  { role: "user", content: "task", _protected_context: true, _context_source: "episode_task" },
];
const OBSERVATION: Json = {
  role: "user",
  content: [{ type: "text", text: '{"current_actual": {"actual_sequence": 0}}' }],
  _observation: { step: 0, observation_id: "internal-observation-0", keyframe: true },
  _causal_observation: { step: 0, result_actual_sequence: 0 },
  _session_round: 1,
};

/** One exchange: a 3,000-byte script with its reasoning, and its result (a failed one names its error). */
function exchange(index: number, resultBytes: number, signed = false): Json[] {
  const assistant: Json = { role: "assistant", content: "", tool_calls: [{
    id: `c${index}`, type: "function",
    function: { name: "report_issue", arguments: JSON.stringify({ message: `script ${index} ` + "x".repeat(3000) }) },
  }] };
  assistant._provider_continuation_state = ProviderContinuationState.create(signed
    ? { providerProtocol: ANTHROPIC_PROTOCOL, model: "m", nativeKind: ANTHROPIC_CONTINUATION_KIND,
      nativePayload: { content: [{ type: "thinking", thinking: `thought ${index}`, signature: "sig" }] },
      canonicalMessage: assistant }
    : { providerProtocol: OPENAI_PROTOCOL, model: "m", nativeKind: OPENAI_CONTINUATION_KIND,
      nativePayload: { reasoning_content: `thought ${index}` }, canonicalMessage: assistant }).toDict();
  const failed = index % 2 === 1;
  const output = `run ${index} ` + "y".repeat(resultBytes) + (failed ? "\nTraceback ...\nValueError: no basket" : "");
  return [assistant, {
    role: "tool", tool_call_id: `c${index}`, name: "report_issue",
    content: JSON.stringify({ ok: !failed, status: failed ? "error" : "ok", actions_executed: index, output }),
  }];
}

/** Twelve exchanges, the eleventh reported over the watermark: the twelfth request ages exchanges 0-5. */
async function aged(management: Json, options: { signed?: boolean } = {}): Promise<Json> {
  const context = pipeline(management);
  const assembly = assembler(!options.signed);
  const messages: Json[] = [...PREFIX, OBSERVATION];
  let sent: Json = null;
  for (let index = 0; index < 12; index += 1) {
    messages.push(...exchange(index, index < 3 ? 10_000 : 100, options.signed));
    const projected = await context.project({ messages, tools: [], observationStep: 0 });
    const { wire } = assembly.prepare(projected);
    context.observeProviderUsage({
      estimatedTokens: providerEstimatedInputTokens(projected.manifest), inputTokens: index === 10 ? 25_000 : 10_000,
      projectionEpoch: projected.manifest.projection_epoch,
      providerWireRequestSha256: projected.manifest.provider_wire_request_sha256,
    });
    sent = { manifest: projected.manifest, messages: projected.messages, wire, context };
  }
  assert.equal(sent.manifest.request_relation, "projection_epoch");
  assert.equal(sent.manifest.lossless_invariants_before_sha256, sent.manifest.lossless_invariants_after_sha256);
  return sent;
}

const assistants = (messages: Json[]): Json[] => messages.filter((m) => m.role === "assistant");
const replayed = (wire: Json): boolean[] => assistants(wire.messages).map((m) => "reasoning_content" in m);

test("v10 validates and runs v9's stages with its own parameters", () => {
  for (const management of [KEEP, SCRIPTS, UNAGED]) assert.deepEqual(validateContextManagement(management), management);
  assert.throws(() => validateContextManagement({ ...KEEP, episode_checkpoint_trigger_tokens: 150_000 }),
    /exactly one of episode_checkpoint_trigger_tokens and episode_checkpoint_trigger_ratio/);
  assert.throws(() => validateContextManagement({ ...KEEP, episode_checkpoint_trigger_ratio: 0.6 }),
    /episode_checkpoint_trigger_ratio < emergency_fallback_ratio/);
  assert.throws(() => validateContextManagement({ ...KEEP, aged_result_preview: "head" }), /aged_result_preview must be one of/);
  assert.throws(() => validateContextManagement({ ...KEEP, reasoning_binding: "loose" }), /reasoning_binding must be one of/);
  const scheme = schemeFromManagement(validateContextManagement(SCRIPTS), { pressureExternalization: true, rolloutCompaction: true });
  assert.deepEqual(scheme.segments, { unit: "exchange", exact_tail: 6, exact_tail_unit: "exchange" });
  const params = Object.fromEntries(scheme.stages.map((spec) => [spec.stage, spec.params]));
  assert.deepEqual([params.exchange_aging.argument_bytes, params.exchange_aging.result_bytes], [1024, 8192]);
  assert.deepEqual([params.exchange_aging.result_preview, params.exchange_aging.keep_anchor], ["structured", true]);
  assert.equal(params.reasoning_replay.binding, "rebind_unsigned");
  assert.deepEqual([params.episode_capsule.trigger_tokens, params.episode_capsule.trigger_ratio], [null, 0.15]);
  assert.equal(params.emergency_checkpoint.unit, "exchange");
  assert.ok(UNSIGNED_CONTINUATION_KINDS.has(OPENAI_CONTINUATION_KIND) && !UNSIGNED_CONTINUATION_KINDS.has(ANTHROPIC_CONTINUATION_KIND));
});

test("keep written: an aging rewrites no assistant message, and a long result keeps its structured fields", async () => {
  const sent = await aged(KEEP);
  const record = sent.manifest.exchange_aging;
  assert.deepEqual([record.aged_exchanges, record.kept_exchange], [6, 0]);
  assert.deepEqual([record.reasoning_unbound_messages, record.reasoning_rebound_messages], [0, 0]);
  // Every assistant message goes out as written, with its reasoning.
  const written = Array.from({ length: 12 }, (_, index) => exchange(index, 0)[0].tool_calls[0].function.arguments);
  assert.deepEqual(assistants(sent.messages).map((m) => m.tool_calls[0].function.arguments), written);
  assert.deepEqual(replayed(sent.wire), Array(12).fill(true));
  // Exchange 0 holds the current observation (the anchor) and stays exact; exchanges 1 and 2 had long results.
  const results = sent.messages.filter((m: Json) => m.role === "tool").map((m: Json) => JSON.parse(m.content));
  assert.ok(!("externalized" in results[0]) && results[0].output.length > 8192);
  assert.deepEqual(Object.keys(results[1]).sort(),
    ["actions_executed", "error", "externalized", "ok", "recoverable_artifact", "result_bytes", "status"]);
  assert.deepEqual([results[1].ok, results[1].status, results[1].actions_executed], [false, "error", 1]);
  assert.deepEqual(results[1].error, { type: "ValueError", message: "no basket" });
  assert.deepEqual(Object.keys(results[2]).sort(),
    ["actions_executed", "externalized", "ok", "recoverable_artifact", "result_bytes", "status"]);
  // A result at or under 8,192 bytes never ages.
  assert.ok(results.slice(3).every((result: Json) => !("externalized" in result)));
  // The kept exchange is part of the session state, and restores.
  const snapshot = snapshotPipeline(sent.context);
  assert.equal(snapshot.exchangeAging.exchangeKeptExchange, 0);
  const restored = pipeline(KEEP);
  restorePipeline(restored, snapshot);
  assert.deepEqual(snapshotPipeline(restored), snapshot);
});

test("age scripts: an aged script's message keeps its unsigned reasoning, bound to the rewrite", async () => {
  const sent = await aged(SCRIPTS);
  const scripts = assistants(sent.messages).map((m) => JSON.parse(m.tool_calls[0].function.arguments));
  assert.deepEqual(scripts.map((args) => args.arguments_externalized === true),
    [false, ...Array(5).fill(true), ...Array(6).fill(false)]);
  // The OpenAI projection checks each continuation against its message: the rebound ones pass.
  assert.deepEqual(replayed(sent.wire), Array(12).fill(true));
  assert.equal(assistants(sent.wire.messages)[1].reasoning_content, "thought 1");
  const record = sent.manifest.exchange_aging;
  assert.deepEqual([record.reasoning_unbound_messages, record.reasoning_rebound_messages], [0, 5]);
  // Under strict the same messages lose it.
  const strict = await aged({ ...SCRIPTS, reasoning_binding: "strict" });
  assert.deepEqual(replayed(strict.wire), [true, ...Array(5).fill(false), ...Array(6).fill(true)]);
  assert.deepEqual([strict.manifest.exchange_aging.reasoning_unbound_messages,
    strict.manifest.exchange_aging.reasoning_rebound_messages], [5, 0]);
});

test("age scripts: signed reasoning is never rebound", async () => {
  const sent = await aged(SCRIPTS, { signed: true });
  const states = assistants(sent.messages).map((m) => "_provider_continuation_state" in m);
  assert.deepEqual(states, [true, ...Array(5).fill(false), ...Array(6).fill(true)]);
  assert.deepEqual([sent.manifest.exchange_aging.reasoning_unbound_messages,
    sent.manifest.exchange_aging.reasoning_rebound_messages], [5, 0]);
});

test("unaged reasoning: an aging withholds the reasoning of the exchanges it ages, the anchor's excepted", async () => {
  const sent = await aged(UNAGED);
  assert.deepEqual(replayed(sent.wire), [true, ...Array(5).fill(false), ...Array(6).fill(true)]);
  const record = sent.manifest.exchange_aging;
  assert.deepEqual([record.reasoning_withheld_exchanges, record.reasoning_unbound_messages], [6, 0]);
  // The messages themselves are as written.
  assert.ok(assistants(sent.messages).every((m) => !("arguments_externalized" in JSON.parse(m.tool_calls[0].function.arguments))));
});
