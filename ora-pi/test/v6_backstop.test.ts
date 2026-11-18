// Context management v6 with its optional fields (2026-11-14):
// `backstop_exact_tail_exchanges` (stages/hard_backstop with `aging: own`: every
// request is estimated before it is sent, and one at or above L6's watermark has
// all but the latest exchanges aged, the open ActionStep's included) and
// `truncated_reply_reasoning: withheld` (stages/materialize: a reply the output
// limit cut off before any tool call is sent without its reasoning). A v6 object
// without them keeps the reactive guard. The histories are one open ActionStep
// of read-only tool calls, the shape of the RoboLab tool-call runs that sent
// requests over the limit (ea103b427290: 19 calls, one robot action).
//
//   node --test ora-pi/test/v6_backstop.test.ts

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { afterTruncatedOutput } from "../src/core/continuations.ts";
import { validateContextManagement } from "../src/context/management.ts";
import { ContextArtifactCatalog } from "../src/context/memory/artifacts.ts";
import { NativeContextPipeline, providerEstimatedInputTokens } from "../src/context/native.ts";
import { NativeRequestAssembler } from "../src/context/request_assembly.ts";
import { schemeFromManagement } from "../src/context/stages/scheme.ts";
import { ArtifactStore, NO_DURABILITY } from "../src/evidence/artifacts.ts";
import { ProviderContinuationState } from "../src/provider/continuation.ts";
import { OPENAI_PROTOCOL, OPENAI_CONTINUATION_KIND, prepareRequest } from "../src/provider/openai.ts";
import { STAGE_SCHEMAS } from "./support/stage_schemas.ts";

type Json = any;

const V6 = {
  schema: "ora.native-context-management.v6",
  causal_exact_tail_action_steps: 3, inline_media_tail_action_steps: 2,
  eager_large_result_bytes: 50_000, eager_preview_bytes: 2_048, eager_exact_deduplication: true,
  trigger_inline_images: 24, trigger_estimated_tokens: 48_000, trigger_provider_input_ratio: 0.70,
  episode_checkpoint_trigger_tokens: 150_000, emergency_fallback_ratio: 0.80, emergency_target_ratio: 0.65,
  checkpoint_target_estimated_tokens: 1200, checkpoint_max_estimated_tokens: 1600, tail_aging_stride_action_steps: 4,
};
const BACKSTOP = { ...V6, backstop_exact_tail_exchanges: 2, truncated_reply_reasoning: "withheld" };
const ON = { pressureExternalization: true, rolloutCompaction: true };

/** 40,000 - 2,048 - 2,048, and L6's watermark, 0.8 of it (the backstop's `trigger_ratio`). */
const HARD = 35_904;
const WATERMARK = 28_723;

function pipeline(management: Json, window = 40_000): NativeContextPipeline {
  const root = mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), "ora-v6-backstop-"));
  const artifacts = new ArtifactStore(root, { durability: NO_DURABILITY });
  const catalog = new ContextArtifactCatalog(root, "policy-session-1", artifacts);
  return new NativeContextPipeline({
    profile: "native_causal_context_v1", artifacts, contextWindowTokens: window, maxOutputTokens: 2048,
    retention: "action_step_lossless_v4", summaryPolicy: "rollout_checkpoint_v1",
    summaryProvider: { summarize: async () => { throw new Error("no compaction here"); } } as Json,
    contextManagement: management, contextArtifacts: catalog,
    mutationTools: ["move_pose", "close_gripper", "open_gripper"], stageSchemas: STAGE_SCHEMAS,
  });
}

function assembler(): NativeRequestAssembler {
  return new NativeRequestAssembler({
    model: "m", reasoningEffort: "low", maxOutputTokens: 2048, stream: false,
    prepareRequest: (execution) => prepareRequest(execution),
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

/** ``assistant`` with an OpenAI-compatible (unsigned) reasoning state of ``reasoning``. */
function withReasoning(assistant: Json, reasoning: string): Json {
  assistant._provider_continuation_state = ProviderContinuationState.create({
    providerProtocol: OPENAI_PROTOCOL, model: "m", nativeKind: OPENAI_CONTINUATION_KIND,
    nativePayload: { reasoning_content: reasoning }, canonicalMessage: assistant,
  }).toDict();
  return assistant;
}

/** One read-only tool call: long reasoning, a long result and one returned image. */
function exchange(index: number, resultBytes = 4000, reasoningBytes = 6000): Json[] {
  const assistant = withReasoning({ role: "assistant", content: "", tool_calls: [{
    id: `c${index}`, type: "function",
    function: { name: "report_issue", arguments: JSON.stringify({ message: `look ${index}` }) },
  }] }, `thought ${index} ` + "r".repeat(reasoningBytes));
  return [assistant, {
    role: "tool", tool_call_id: `c${index}`, name: "report_issue",
    content: JSON.stringify({ ok: true, output: `result ${index} ` + "y".repeat(resultBytes) }),
  }, { role: "user", content: [
    { type: "text", text: `READ-ONLY tool feedback: 1 image(s) returned by report_issue (c${index}).` },
    { type: "image_url", image_url: { url: `data:image/png;base64,${Buffer.from(`png ${index}`).toString("base64")}` } },
  ] }];
}

/** A reply the output limit cut off before any tool call, and the core's continuation after it. */
function truncated(index: number, reasoningBytes = 6000): Json[] {
  const assistant = withReasoning({ role: "assistant", content: "" }, `cut ${index} ` + "z".repeat(reasoningBytes));
  return [assistant, { role: "user", content: afterTruncatedOutput().content }];
}

async function send(
  context: NativeContextPipeline, assembly: NativeRequestAssembler, messages: Json[], inputTokens: number | null,
): Promise<{ manifest: Json; wire: Json; messages: Json[] }> {
  const projected = await context.project({ messages, tools: [], observationStep: 0 });
  const prepared = assembly.prepare(projected);
  if (inputTokens !== null) {
    context.observeProviderUsage({
      estimatedTokens: providerEstimatedInputTokens(projected.manifest), inputTokens,
      projectionEpoch: projected.manifest.projection_epoch,
      providerWireRequestSha256: projected.manifest.provider_wire_request_sha256,
    });
  }
  return { manifest: projected.manifest, wire: prepared.wire, messages: projected.messages };
}

const assistants = (wire: Json): Json[] => wire.messages.filter((m: Json) => m.role === "assistant");
const hasReasoning = (message: Json): boolean => "reasoning_content" in message;
const imageCount = (message: Json): number => (Array.isArray(message.content)
  ? message.content.filter((block: Json) => block.type === "image_url").length : 0);

test("the optional v6 fields validate, and only their values", () => {
  assert.deepEqual(validateContextManagement(V6), V6);
  assert.deepEqual(validateContextManagement(BACKSTOP), BACKSTOP);
  assert.deepEqual(validateContextManagement({ ...V6, backstop_exact_tail_exchanges: 3 }),
    { ...V6, backstop_exact_tail_exchanges: 3 });
  assert.throws(() => validateContextManagement({ ...V6, backstop_exact_tail_exchanges: 0 }), /must be positive/);
  assert.throws(() => validateContextManagement({ ...V6, truncated_reply_reasoning: "kept" }), /must be 'withheld'/);
  assert.throws(() => validateContextManagement({ ...V6, other: 1 }), /extra=\['other'\]/);
});

test("v6 translates its optional fields into the backstop guard and the materialize switch", () => {
  const old = schemeFromManagement(V6, ON);
  assert.deepEqual(old.stages.at(-1), { stage: "over_limit_guard", params: { over_limit: "refuse" } });
  assert.deepEqual(old.stages[0], { stage: "materialize", params: { reasoning: "exclude" } });
  const fixed = schemeFromManagement(BACKSTOP, ON);
  assert.deepEqual(fixed.stages.at(-1),
    { stage: "hard_backstop", params: { exact_tail_exchanges: 2, aging: "own", trigger_ratio: 0.8 } });
  assert.deepEqual(fixed.stages[0],
    { stage: "materialize", params: { reasoning: "exclude", truncated_reply_reasoning: "withheld" } });
  assert.deepEqual(fixed.stages.slice(1, -1), old.stages.slice(1, -1));
});

test("without the backstop field v6 still sends a request its own estimate puts over the hard limit", async () => {
  const context = pipeline(V6);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, OBSERVATION];
  let over = 0;
  for (let index = 0; index < 14; index += 1) {
    messages.push(...exchange(index));
    const sent = await send(context, assembly, messages, null);
    assert.ok(!("request_estimate" in sent.manifest) && !("hard_backstop" in sent.manifest));
    if (sent.manifest.estimated_input_tokens > HARD) over += 1;
  }
  assert.ok(over > 0, "the open ActionStep outgrew the hard limit unchecked");
});

test("the v6 backstop ages the open ActionStep's older exchanges at L6's watermark, and a request that cannot fit is not sent", async () => {
  const context = pipeline(BACKSTOP);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, OBSERVATION];
  let backstopped: Json = null;
  let atWatermark = 0;
  for (let index = 0; index < 14; index += 1) {
    messages.push(...exchange(index));
    const sent = await send(context, assembly, messages, null);
    // The backstop keeps every request within the limit.
    assert.ok(sent.manifest.request_estimate.tokens <= HARD, String(sent.manifest.request_estimate.tokens));
    assert.equal(sent.manifest.open_action_step, 0);
    backstopped ??= sent.manifest.hard_backstop ? sent : null;
    if (sent.manifest.hard_backstop) {
      const record = sent.manifest.hard_backstop;
      assert.ok(WATERMARK <= record.pre_estimated_input_tokens && record.post_estimated_input_tokens <= HARD);
      assert.equal(record.watermark_tokens, WATERMARK);
      if (record.pre_estimated_input_tokens <= HARD) atWatermark += 1;
    }
  }
  assert.ok(backstopped, "the open ActionStep outgrew L6's watermark");
  // It ages at the watermark, before the hard limit (no usage here, so no L6 waits for a report).
  assert.ok(atWatermark > 0);
  const { manifest, wire } = backstopped;
  const transition = manifest.projection_transitions.find((item: Json) => item.kind === "hard_backstop");
  assert.equal(transition.semantic_loss, false);
  assert.deepEqual(transition.reasons, ["estimated_input_at_or_above_watermark"]);
  assert.equal(transition.aging, "own");
  assert.equal(transition.aged_exchanges_after, transition.exchanges - 2);
  assert.equal(transition.reasoning_withheld_exchanges_after, transition.aged_exchanges_after);
  const record = manifest.hard_backstop_aging;
  assert.equal(record.aged_exchanges, record.exchanges - 2);
  // Reasoning, results and tool images: only the latest two exchanges keep theirs.
  assert.deepEqual(assistants(wire).map(hasReasoning).slice(-3), [false, true, true]);
  const results = wire.messages.filter((m: Json) => m.role === "tool").map((m: Json) => JSON.parse(m.content));
  assert.deepEqual(results.slice(-3).map((item: Json) => item.externalized === true), [true, false, false]);
  assert.ok(results.slice(0, -2).every((item: Json) => item.recoverable_artifact && item.preview.length <= 2048));
  assert.equal(manifest.active_inline_images, 2);
  assert.equal(record.delivered_tool_images_externalized, record.exchanges - 2);
  // The one observation keeps its body; the exact ActionStep tail compares by identity.
  assert.ok(wire.messages.some((m: Json) => JSON.stringify(m.content).includes("current_actual")));
  assert.equal(manifest.lossless_invariants_before_sha256, manifest.lossless_invariants_after_sha256);
  // The latest exchange alone over the limit: refused before it is sent, distinctly from the reactive refusal.
  messages.push(...exchange(14, 200_000));
  await assert.rejects(send(context, assembly, messages, null),
    /context_exhausted: estimated \d+ input tokens .* after the hard backstop kept only the latest 2 exchanges exact, hard limit 35904/);
});

test("a request whose latest two exchanges alone reach the watermark is sent as it is while it fits", async () => {
  const context = pipeline(BACKSTOP);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, OBSERVATION, ...exchange(0, 18_000, 18_000)];
  await send(context, assembly, messages, null);
  messages.push(...exchange(1, 18_000, 18_000));
  const near = await send(context, assembly, messages, null);
  const tokens = near.manifest.request_estimate.tokens;
  assert.ok(WATERMARK <= tokens && tokens <= HARD, String(tokens));
  assert.equal(near.manifest.hard_backstop, null);
  assert.deepEqual(near.manifest.projection_transitions, []);
  assert.deepEqual(assistants(near.wire).map(hasReasoning), [true, true]);
});

test("between backstops a v6 request only appends, so the prefix cache holds", async () => {
  const context = pipeline(BACKSTOP);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, OBSERVATION];
  const relations: string[] = [];
  for (let index = 0; index < 20; index += 1) {
    messages.push(...exchange(index, 1000, 3000));
    const sent = await send(context, assembly, messages, null);
    relations.push(sent.manifest.hard_backstop ? "backstop" : sent.manifest.request_relation);
  }
  assert.ok(relations.includes("backstop"));
  assert.ok(relations.every((item) => item === "backstop" || item === "append_only" || item === "initial"),
    relations.join(", "));
});

test("a truncated reply is sent without its reasoning from its first request; other replies keep theirs", async () => {
  for (const [management, withheld] of [[BACKSTOP, true], [V6, false]] as const) {
    const context = pipeline(management, 400_000);
    const assembly = assembler();
    const messages: Json[] = [...PREFIX, OBSERVATION, ...exchange(0), ...truncated(1)];
    const first = await send(context, assembly, messages, null);
    assert.deepEqual(assistants(first.wire).map(hasReasoning), [true, !withheld]);
    const exclusion = first.manifest.passes.find((pass: Json) => pass.name === "reasoning_exclusion");
    assert.deepEqual(exclusion.changed_messages, withheld ? [messages.length - 2] : []);
    if (withheld) {
      assert.deepEqual(first.manifest.truncated_reply_reasoning,
        { mode: "withheld", withheld_messages: [messages.length - 2], withheld_event_ids: [] });
    } else {
      assert.ok(!("truncated_reply_reasoning" in first.manifest));
    }
    messages.push(...exchange(2));
    const next = await send(context, assembly, messages, null);
    assert.deepEqual(assistants(next.wire).map(hasReasoning), [true, !withheld, true]);
    assert.equal(next.manifest.request_relation, "append_only");
    // The model still reads that its reply was cut.
    assert.ok(next.wire.messages.some((m: Json) => m.content === afterTruncatedOutput().content));
  }
});
