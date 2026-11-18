// Context management v9 (src/context/stages/exchange_aging, reasoning_replay,
// hard_backstop): exchange aging in batches at a Provider-input watermark,
// tool-returned images (and, under exchange_window_v1, camera images) aged with
// their exchange, the reasoning-replay modes and the hard backstop, on a history
// of exchanges that complete no ActionStep (code mode's perception and failed
// scripts), through the OpenAI wire projection that replays each assistant
// message's signed reasoning unmodified.
//
//   node --test ora-pi/test/exchange_aging.test.ts

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { headTailPreview, isHeadTailPreview as isAgedPreview } from "../src/context/stages/services/references.ts";
import { validateContextManagement } from "../src/context/management.ts";
import { ContextArtifactCatalog } from "../src/context/memory/artifacts.ts";
import { NativeContextPipeline, providerEstimatedInputTokens } from "../src/context/native.ts";
import { NativeRequestAssembler } from "../src/context/request_assembly.ts";
import { ArtifactStore, NO_DURABILITY } from "../src/evidence/artifacts.ts";
import { ProviderContinuationState } from "../src/provider/continuation.ts";
import { OPENAI_PROTOCOL, OPENAI_CONTINUATION_KIND, prepareRequest } from "../src/provider/openai.ts";
import { STAGE_SCHEMAS } from "./support/stage_schemas.ts";

type Json = any;

/** v9's aged preview of a ``budget``: the head the larger half, the tail the smaller (stages/scheme.ts schemeV9). */
function agedPreview(text: string, budget: number): { preview: string; keptBytes: number } {
  return headTailPreview(text, { head_bytes: budget - Math.floor(budget / 2), tail_bytes: Math.floor(budget / 2) });
}

const V9 = {
  schema: "ora.native-context-management.v9",
  causal_exact_tail_action_steps: 3, camera_images: "rolling_tail_v1", camera_tail_observations: 2,
  episode_checkpoint_trigger_tokens: 150_000, compaction_trigger_tokens: 150_000,
  emergency_fallback_ratio: 0.8, emergency_target_ratio: 0.65,
  checkpoint_target_estimated_tokens: 1200, checkpoint_max_estimated_tokens: 1600, image_estimate_tokens: 530,
  exchange_aging_trigger_tokens: 20_000, exchange_aging_trigger_ratio: 0.5, exact_tail_exchanges: 6,
  aged_argument_bytes: 1024, aged_result_bytes: 2048, aged_preview_bytes: 256,
  backstop_exact_tail_exchanges: 2, reasoning_replay: "all", reasoning_tail_exchanges: null,
};
const RECENT = { ...V9, reasoning_replay: "recent_exchanges", reasoning_tail_exchanges: 3 };
const UNAGED = { ...V9, reasoning_replay: "unaged_exchanges" };
const WINDOW = { ...V9, camera_images: "exchange_window_v1", camera_tail_observations: null };

function pipeline(management: Json, window = 200_000): [NativeContextPipeline, ContextArtifactCatalog] {
  const root = mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), "ora-exchange-aging-"));
  const artifacts = new ArtifactStore(root, { durability: NO_DURABILITY });
  const catalog = new ContextArtifactCatalog(root, "policy-session-1", artifacts);
  return [new NativeContextPipeline({
    profile: "native_causal_context_v1", artifacts, contextWindowTokens: window, maxOutputTokens: 2048,
    retention: "action_step_lossless_v4", summaryPolicy: "rollout_checkpoint_v1",
    summaryProvider: { summarize: async () => { throw new Error("no compaction here"); } } as Json,
    contextManagement: management, contextArtifacts: catalog,
    mutationTools: ["move_pose", "close_gripper", "open_gripper"], stageSchemas: STAGE_SCHEMAS,
  }), catalog];
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

const PNG = "data:image/png;base64,iVBORw0KGgo=";

/** The images one tool call returned, delivered in their own user message. */
function toolImages(index: number, count: number): Json {
  const images = Array.from({ length: count }, (_, n) => (
    { type: "image_url", image_url: { url: `data:image/png;base64,${Buffer.from(`png ${index}.${n}`).toString("base64")}` } }));
  return { role: "user", content: [
    { type: "text", text: `READ-ONLY tool feedback: ${count} image(s) returned by report_issue (c${index}).` }, ...images] };
}

/** One exchange: a long script argument, its signed reasoning and a long result. */
function exchange(index: number, resultBytes = 4000, argumentBytes = 3000): Json[] {
  const assistant: Json = { role: "assistant", content: "", tool_calls: [{
    id: `c${index}`, type: "function",
    function: { name: "report_issue", arguments: JSON.stringify({ message: `script ${index} ` + "x".repeat(argumentBytes) }) },
  }] };
  assistant._provider_continuation_state = ProviderContinuationState.create({
    providerProtocol: OPENAI_PROTOCOL, model: "m", nativeKind: OPENAI_CONTINUATION_KIND,
    nativePayload: { reasoning_content: `thought ${index}` }, canonicalMessage: assistant,
  }).toDict();
  return [assistant, {
    role: "tool", tool_call_id: `c${index}`, name: "report_issue",
    content: JSON.stringify({ ok: true, output: `result ${index} ` + "y".repeat(resultBytes) }),
  }];
}

/** Project and prepare one request; report ``inputTokens`` as its Provider usage. */
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

const calls = (messages: Json[]): Json[] => messages.filter((m) => m.role === "assistant").map((m) => m.tool_calls[0]);
const aged = (call: Json): boolean => JSON.parse(call.function.arguments).arguments_externalized === true;

test("the scheme validates and needs its aging watermark below L6", () => {
  assert.deepEqual(validateContextManagement(V9), V9);
  assert.deepEqual(validateContextManagement(RECENT), RECENT);
  assert.deepEqual(validateContextManagement(UNAGED), UNAGED);
  assert.throws(() => validateContextManagement({ ...V9, exchange_aging_trigger_tokens: 150_000 }), /below the L6/);
  assert.throws(() => validateContextManagement({ ...V9, reasoning_tail_exchanges: 3 }), /must be null for all/);
  assert.throws(() => validateContextManagement({ ...UNAGED, reasoning_tail_exchanges: 3 }),
    /must be null for unaged_exchanges/);
  assert.throws(() => validateContextManagement({ ...V9, reasoning_replay: "none" }), /reasoning_replay must be one of/);
  assert.throws(() => validateContextManagement({ ...V9, backstop_exact_tail_exchanges: 7 }), /must not exceed/);
  assert.deepEqual(validateContextManagement(WINDOW), WINDOW);
  assert.throws(() => validateContextManagement({ ...WINDOW, camera_tail_observations: 2 }),
    /camera_tail_observations must be null for exchange_window_v1/);
  const v8Fields = ["causal_exact_tail_action_steps", "camera_images", "camera_tail_observations",
    "episode_checkpoint_trigger_tokens", "compaction_trigger_tokens", "emergency_fallback_ratio",
    "emergency_target_ratio", "checkpoint_target_estimated_tokens", "checkpoint_max_estimated_tokens",
    "image_estimate_tokens"];
  const v8 = Object.fromEntries(Object.entries(WINDOW).filter(([key]) => v8Fields.includes(key)));
  assert.throws(() => validateContextManagement({ ...v8, schema: "ora.native-context-management.v8" }),
    /exchange_window_v1 needs exchange aging/);
});

test("an aged preview keeps the head and the tail of its budget, whole characters only", () => {
  // Within the budget: unchanged.
  assert.deepEqual(agedPreview("short", 256), { preview: "short", keptBytes: 5 });
  assert.deepEqual(agedPreview("x".repeat(256), 256), { preview: "x".repeat(256), keptBytes: 256 });
  // Past it: the first and the last half, and how much lies between.
  const text = "h".repeat(300) + "t".repeat(300);
  const cut = agedPreview(text, 256);
  assert.deepEqual(cut, { preview: `${"h".repeat(128)}\n[... 344 bytes cut ...]\n${"t".repeat(128)}`, keptBytes: 256 });
  assert.ok(isAgedPreview(text, cut.preview) && isAgedPreview("short", "short"));
  assert.ok(!isAgedPreview(text, "h".repeat(128)), "a head alone is no aged preview");
  assert.ok(!isAgedPreview(text, cut.preview.replace("344", "343")), "the cut count must be exact");
  // A character is never split: two-byte and four-byte characters at either cut.
  assert.deepEqual(agedPreview("é".repeat(200), 255),
    { preview: `${"é".repeat(64)}\n[... 146 bytes cut ...]\n${"é".repeat(63)}`, keptBytes: 254 });
  const emoji = agedPreview("\u{1F600}".repeat(100), 250);
  assert.equal(emoji.preview, `${"\u{1F600}".repeat(31)}\n[... 152 bytes cut ...]\n${"\u{1F600}".repeat(31)}`);
  assert.ok(emoji.preview.isWellFormed() && !emoji.preview.includes("\uFFFD"));
});

test("one rewrite at the watermark ages every exchange before the exact tail, then the history appends again", async () => {
  const [context, catalog] = pipeline(V9);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, OBSERVATION];
  const relations: string[] = [];
  for (let index = 0; index < 10; index += 1) {
    messages.push(...exchange(index));
    const sent = await send(context, assembly, messages, 10_000);
    relations.push(sent.manifest.request_relation);
    assert.equal(sent.manifest.exchange_aging.admission.reason, index === 0 ? "provider_usage_unavailable" : "below_watermark");
  }
  assert.deepEqual(new Set(relations), new Set(["initial", "append_only"]));
  messages.push(...exchange(10));
  await send(context, assembly, messages, 25_000);  // this request crosses the watermark
  messages.push(...exchange(11));
  const crossed = await send(context, assembly, messages, 12_000);
  const manifest = crossed.manifest;
  assert.equal(manifest.request_relation, "projection_epoch");
  const [transition] = manifest.projection_transitions;
  assert.equal(transition.kind, "exchange_aging");
  assert.equal(transition.semantic_loss, false);
  assert.equal(transition.admission_provider_input_tokens, 25_000);
  assert.deepEqual([transition.aged_exchanges_before, transition.aged_exchanges_after], [0, 6]);
  assert.equal(manifest.exchange_aging.aged_exchanges, 6);
  assert.equal(manifest.lossless_invariants_before_sha256, manifest.lossless_invariants_after_sha256);
  // The six older calls carry a reference to their stored arguments; the latest six stay exact.
  const delivered = calls(crossed.messages);
  assert.deepEqual(delivered.map(aged), [...Array(6).fill(true), ...Array(6).fill(false)]);
  const marker = JSON.parse(delivered[0].function.arguments);
  const [, stored] = catalog.resolve(marker.recoverable_artifact.ref);
  const original = exchange(0)[0].tool_calls[0].function.arguments;
  assert.equal(stored.toString("utf8"), original);
  // The preview is the head and the tail of the 256-byte budget, around what was cut.
  assert.equal(marker.preview, `${original.slice(0, 128)}\n[... ${original.length - 256} bytes cut ...]\n${original.slice(-128)}`);
  const results = crossed.messages.filter((m) => m.role === "tool");
  const result = JSON.parse(results[0].content);
  const content = exchange(0)[1].content;
  assert.equal(result.externalized, true);
  assert.ok(result.preview.startsWith(content.slice(0, 128)) && result.preview.endsWith(content.slice(-128)));
  assert.deepEqual([result.preview_truncated, result.preview_bytes, result.result_bytes], [true, 256, content.length]);
  assert.ok(!("preview_is_summary" in result));
  assert.ok(!("externalized" in JSON.parse(results[11].content)));
  // A rewritten message is sent without its reasoning; every exact one replays it unmodified.
  const wired = crossed.wire.messages.filter((m: Json) => m.role === "assistant");
  assert.deepEqual(wired.map((m: Json) => "reasoning_content" in m), [...Array(6).fill(false), ...Array(6).fill(true)]);
  // Below the watermark again: the next request appends to the aged history.
  messages.push(...exchange(12));
  const next = await send(context, assembly, messages, 13_000);
  assert.equal(next.manifest.request_relation, "append_only");
  assert.equal(next.manifest.exchange_aging.admission.reason, "below_watermark");
});

test("a request that stays above the watermark ages again once a batch of exchanges has left the exact tail", async () => {
  const [context] = pipeline(V9);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, OBSERVATION];
  const aged: number[] = [];
  const relations: string[] = [];
  for (let index = 0; index < 26; index += 1) {
    messages.push(...exchange(index));
    // Every report is above the watermark: what aging leaves already exceeds it.
    const sent = await send(context, assembly, messages, 30_000);
    aged.push(sent.manifest.exchange_aging.aged_exchanges);
    relations.push(sent.manifest.request_relation);
    if (index >= 12 && sent.manifest.request_relation === "append_only") {
      assert.equal(sent.manifest.exchange_aging.admission.reason, "awaiting_exchange_batch");
    }
  }
  // 12 exchanges when the first batch is due (six past the exact tail), then every six more.
  const epochs = relations.flatMap((relation, index) => (relation === "projection_epoch" ? [index + 1] : []));
  assert.deepEqual(epochs, [12, 18, 24]);
  assert.deepEqual([aged[11], aged[17], aged[23], aged[25]], [6, 12, 18, 18]);
});

test("tool-returned images of an aged exchange become references, readable with observe(ref)", async () => {
  const [context, catalog] = pipeline(V9);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, OBSERVATION];
  let first: Json = null;
  let last: Json = null;
  for (let index = 0; index < 13; index += 1) {
    messages.push(...exchange(index, 100), toolImages(index, 2));
    last = await send(context, assembly, messages, index === 10 ? 25_000 : 10_000);
    first ??= index === 10 ? last : null;
  }
  const images = (request: Json[]) => request.flatMap((m) => (Array.isArray(m.content) ? m.content : []))
    .filter((block: Json) => block.type === "image_url").length;
  assert.equal(images(first.messages), 22);
  // Aged at the twelfth exchange: the first six exchanges' images are references now.
  assert.equal(last.manifest.request_relation, "append_only");
  assert.equal(images(last.messages), 26 - 12);
  assert.equal(last.manifest.active_inline_images, 14);
  assert.equal(last.manifest.exchange_aging.delivered_tool_images_externalized, 12);
  assert.equal(last.manifest.lossless_invariants_before_sha256, last.manifest.lossless_invariants_after_sha256);
  const feedback = last.messages.filter((m: Json) => m.role === "user" && Array.isArray(m.content)
    && String(m.content[0].text).startsWith("READ-ONLY tool feedback"));
  const markers = feedback[0].content.slice(1).map((block: Json) => block.text);
  assert.equal(markers.length, 2);
  assert.ok(markers.every((text: string) => text.startsWith("<recoverable_artifact>")));
  const pointer = JSON.parse(markers[0].slice("<recoverable_artifact>".length, -"</recoverable_artifact>".length));
  assert.equal(pointer.open, "observe");
  const [, stored] = catalog.resolve(pointer.ref);
  assert.equal(stored.toString("utf8"), "png 0.0");
  // An exact exchange's images stay inline, without a marker.
  assert.deepEqual(feedback.at(-1).content.slice(1).map((block: Json) => block.type), ["image_url", "image_url"]);
});

/** Observation ``step`` arriving after the last exchange (no action: the same round's next actual). */
function observation(step: number): Json {
  return {
    role: "user",
    content: [{ type: "text", text: JSON.stringify({ current_actual: {
      actual_sequence: step, public_robot_facts: { measured: "z".repeat(2000) } } }) }],
    _observation: { step, observation_id: `internal-observation-${step}`, keyframe: true },
    _causal_observation: { step, result_actual_sequence: step },
    _session_round: 1,
  };
}

test("an aged observation that leaves the camera tail between agings waits for the next aging", async () => {
  // Regression (2026-11-03): the latest observations' bodies are exempt from aging; when a new
  // observation pushed an aged one out of that window, its body was externalized at once, a
  // rewrite without an epoch (cache_prefix_rewritten). The exemption is fixed at each aging.
  const [context] = pipeline(V9);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, observation(0)];
  const reports = (index: number) => (index === 10 || index >= 16 ? 25_000 : 10_000);
  const relations: string[] = [];
  let sent: Json = null;
  for (let index = 0; index < 18; index += 1) {
    messages.push(...exchange(index, 100));
    if (index === 4 || index === 12) messages.push(observation(index === 4 ? 1 : 2));
    sent = await send(context, assembly, messages, reports(index));
    relations.push(sent.manifest.request_relation);
  }
  // Aged at 12 exchanges (observations 0 and 1 are the latest two then) and again at 18; between
  // them observation 2 arrived and the history only grew.
  const epochs = relations.flatMap((relation, index) => (relation === "projection_epoch" ? [index + 1] : []));
  assert.deepEqual(epochs, [12, 18]);
  assert.deepEqual(new Set(relations.filter((_, index) => ![11, 17].includes(index))), new Set(["append_only"]).add("initial"));
  // The second aging is the one that externalizes observation 0's body.
  const opening = sent.messages.find((m: Json) => m.role === "user" && Array.isArray(m.content)
    && String(m.content[0].text).includes('"actual_sequence":0'));
  assert.ok(opening && opening.content[0].text.includes('"observation_externalized":true'));
});

/** An observation with two labelled camera images and one receipt image a skill returned with it. */
function cameraObservation(step: number): Json {
  const image = (name: string) => ({ type: "image_url", image_url: {
    url: `data:image/png;base64,${Buffer.from(`${name} ${step}`).toString("base64")}` } });
  return {
    role: "user",
    content: [
      { type: "text", text: JSON.stringify({ current_actual: { actual_sequence: step } }) },
      { type: "text", text: `ACTION RESULT ACTUAL · sequence ${step} · agentview · latest at this message` },
      image("agentview"),
      { type: "text", text: `ACTION RESULT ACTUAL · sequence ${step} · wrist · latest at this message` },
      image("wrist"),
      image("receipt"),
    ],
    _observation: { step, observation_id: `internal-observation-${step}`, keyframe: true },
    _causal_observation: { step, result_actual_sequence: step, cameras: ["agentview", "wrist"] },
    _session_round: 1,
  };
}

test("an observation's receipt images age with their exchange; its camera images keep the camera tail", async () => {
  const [context] = pipeline(V9);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, cameraObservation(0)];
  let sent: Json = null;
  for (let index = 0; index < 13; index += 1) {
    messages.push(...exchange(index, 100));
    if (index === 2) messages.push(cameraObservation(1));
    sent = await send(context, assembly, messages, index === 10 ? 25_000 : 10_000);
  }
  // Aged at the twelfth exchange; both observations are in the aged span and are the camera tail's two.
  assert.equal(sent.manifest.request_relation, "camera_tail_append");
  assert.equal(sent.manifest.exchange_aging.delivered_tool_images_externalized, 2);
  const tail = sent.messages.at(-1);
  assert.ok(String(tail.content[0].text).startsWith("CAMERA TAIL"));
  assert.equal(tail.content.filter((block: Json) => block.type === "image_url").length, 4);
  const observations = sent.messages.filter((m: Json) => m.role === "user" && Array.isArray(m.content)
    && String(m.content[0].text).startsWith('{"current_actual"'));
  for (const message of observations) {
    assert.ok(!message.content.some((block: Json) => block.type === "image_url"));
    assert.equal(message.content.filter((block: Json) => String(block.text ?? "").startsWith("<recoverable_artifact>")).length, 1);
  }
  assert.equal(sent.manifest.active_inline_images, 4);
  assert.equal(sent.manifest.lossless_invariants_before_sha256, sent.manifest.lossless_invariants_after_sha256);
});

const imageCount = (message: Json): number => (Array.isArray(message.content)
  ? message.content.filter((block: Json) => block.type === "image_url").length : 0);
const markerTexts = (message: Json): string[] => (Array.isArray(message.content)
  ? message.content.map((block: Json) => String(block.text ?? "")).filter((text: string) => text.startsWith("<recoverable_artifact>"))
  : []);
const observationMessages = (messages: Json[]): Json[] => messages.filter((m: Json) => m.role === "user"
  && Array.isArray(m.content) && String(m.content[0].text).startsWith('{"current_actual"'));

test("exchange_window_v1: camera images stay where their observation arrived until its exchange ages", async () => {
  const [context, catalog] = pipeline(WINDOW);
  const assembly = assembler();
  // The opening observation and the one after the first call both belong to exchange 0; every later
  // exchange carries one observation (a turn per call), each with two camera images and a receipt image.
  const messages: Json[] = [...PREFIX, cameraObservation(0)];
  let previous: Json[] | null = null;
  let sent: Json = null;
  for (let index = 0; index < 11; index += 1) {
    messages.push(...exchange(index, 100), cameraObservation(index + 1));
    sent = await send(context, assembly, messages, index === 10 ? 25_000 : 10_000);
    // Below the watermark nothing is rewritten: every image is inline and every request extends the last.
    assert.ok(["initial", "append_only"].includes(sent.manifest.request_relation));
    assert.deepEqual(observationMessages(sent.messages).map(imageCount), Array(index + 2).fill(3));
    assert.equal(sent.manifest.active_inline_images, 3 * (index + 2));
    assert.ok(!sent.messages.some((m: Json) => markerTexts(m).length
      || (Array.isArray(m.content) && String(m.content[0].text ?? "").startsWith("CAMERA TAIL"))));
    if (previous !== null) assert.deepEqual(sent.messages.slice(0, previous.length), previous);
    previous = sent.messages;
  }
  assert.deepEqual(sent.manifest.camera_delivery, {
    schema: "ora.camera-delivery.v1", camera_images: "exchange_window_v1", tail_observations: null,
    retrieval: "observe.ref",
  });
  // The aging at twelve exchanges ages exchanges 0-5: observations 0-6, opening included.
  messages.push(...exchange(11, 100), cameraObservation(12));
  const crossed = await send(context, assembly, messages, 12_000);
  assert.equal(crossed.manifest.request_relation, "projection_epoch");
  const record = crossed.manifest.exchange_aging;
  assert.equal(record.aged_exchanges, 6);
  assert.deepEqual([record.delivered.camera_images, record.delivered_camera_images_externalized], [7, 14]);
  assert.deepEqual([record.delivered.tool_images, record.delivered_tool_images_externalized], [7, 7]);
  const observed = observationMessages(crossed.messages);
  assert.deepEqual(observed.map(imageCount), [...Array(7).fill(0), ...Array(6).fill(3)]);
  assert.equal(crossed.manifest.active_inline_images, 18);
  const pass = crossed.manifest.passes.find((item: Json) => item.name === "exchange_camera_image_reference");
  assert.deepEqual([pass.changed_messages.length, pass.artifact_refs.length], [7, 14]);
  // An aged camera image keeps its label, followed by its reference, which observe(ref) reopens.
  const opening = observed[0].content;
  assert.equal(markerTexts(observed[0]).length, 3);
  const label = opening.findIndex((block: Json) => String(block.text ?? "").includes("· agentview · latest at this message"));
  const marker = String(opening[label + 1].text);
  const pointer = JSON.parse(marker.slice("<recoverable_artifact>".length, -"</recoverable_artifact>".length));
  assert.equal(pointer.open, "observe");
  assert.equal(catalog.resolve(pointer.ref)[1].toString("utf8"), "agentview 0");
  assert.equal(crossed.manifest.lossless_invariants_before_sha256, crossed.manifest.lossless_invariants_after_sha256);
  // Below the watermark again: the aged history only grows, and the new observation's images are inline.
  previous = crossed.messages;
  for (let index = 12; index < 15; index += 1) {
    messages.push(...exchange(index, 100), cameraObservation(index + 1));
    const next = await send(context, assembly, messages, 13_000);
    assert.equal(next.manifest.request_relation, "append_only");
    assert.deepEqual(next.messages.slice(0, previous.length), previous);
    assert.equal(imageCount(observationMessages(next.messages).at(-1)), 3);
    assert.equal(next.manifest.exchange_aging.delivered_camera_images_externalized, 14);
    previous = next.messages;
  }
});

test("exchange_window_v1: the hard backstop takes the camera images outside its exact tail", async () => {
  // Hard limit 20,000 - 2,048 - 2,048 = 15,904; no Provider usage, so no watermark is ever crossed.
  const [context] = pipeline(WINDOW, 20_000);
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, cameraObservation(0)];
  let backstopped: Json = null;
  for (let index = 0; index < 12 && backstopped === null; index += 1) {
    messages.push(...exchange(index), cameraObservation(index + 1));
    const sent = await send(context, assembly, messages, null);
    assert.ok(sent.manifest.request_estimate.tokens <= 15_904, String(sent.manifest.request_estimate.tokens));
    backstopped = sent.manifest.hard_backstop ? sent : null;
  }
  assert.ok(backstopped, "the history outgrew the hard limit");
  const record = backstopped.manifest.exchange_aging;
  const exchanges = record.exchanges;
  assert.equal(record.aged_exchanges, exchanges - 2);
  // Only the observations of the last two exchanges keep their images; every older one is references.
  const counts = observationMessages(backstopped.messages).map(imageCount);
  assert.deepEqual(counts, [...Array(exchanges - 1).fill(0), 3, 3]);
  assert.equal(record.delivered_camera_images_externalized, 2 * (exchanges - 1));
  assert.equal(backstopped.manifest.active_inline_images, 6);
});

test("recent_exchanges withholds older reasoning in strides of its window; all and unaged_exchanges keep it", async () => {
  const every = [1, 2, 3, 4, 5, 6, 7, 8];
  for (const [management, expected] of [[V9, every], [UNAGED, every], [RECENT, [1, 2, 3, 4, 5, 3, 4, 5]]] as const) {
    const [context] = pipeline(management);
    const assembly = assembler();
    const messages: Json[] = [...PREFIX, OBSERVATION];
    const replayed: number[] = [];
    for (let index = 0; index < 8; index += 1) {
      messages.push(...exchange(index, 100));
      const sent = await send(context, assembly, messages, 5_000);
      replayed.push(sent.wire.messages.filter((m: Json) => "reasoning_content" in m).length);
      assert.ok(!calls(sent.messages).some(aged));
    }
    assert.deepEqual(replayed, expected, management.reasoning_replay);
  }
});

test("unaged_exchanges withholds the reasoning of exactly the exchanges an aging ages, in that rewrite", async () => {
  // Short arguments: an aging leaves the assistant messages themselves unchanged, so only the mode
  // decides whether an aged exchange keeps its reasoning (all keeps it).
  for (const management of [UNAGED, V9]) {
    const [context, catalog] = pipeline(management);
    const assembly = assembler();
    const messages: Json[] = [...PREFIX, OBSERVATION];
    const replayed = (sent: Json): boolean[] => sent.wire.messages.filter((m: Json) => m.role === "assistant")
      .map((m: Json) => "reasoning_content" in m);
    for (let index = 0; index < 11; index += 1) {
      messages.push(...exchange(index, 4000, 100));
      const sent = await send(context, assembly, messages, index === 10 ? 25_000 : 10_000);
      // Below the watermark nothing is withheld and the history only grows.
      assert.deepEqual(replayed(sent), Array(index + 1).fill(true));
      assert.equal(sent.manifest.exchange_aging.reasoning_withheld_exchanges, 0);
      assert.ok(["initial", "append_only"].includes(sent.manifest.request_relation));
    }
    messages.push(...exchange(11, 4000, 100));
    const crossed = await send(context, assembly, messages, 12_000);
    assert.equal(crossed.manifest.request_relation, "projection_epoch");
    const [transition] = crossed.manifest.projection_transitions;
    assert.equal(transition.kind, "exchange_aging");
    assert.ok(!calls(crossed.messages).some(aged));
    const withheld = management === UNAGED ? 6 : 0;
    assert.deepEqual([transition.aged_exchanges_after, transition.reasoning_withheld_exchanges_after], [6, withheld]);
    assert.deepEqual(
      [crossed.manifest.exchange_aging.aged_exchanges, crossed.manifest.exchange_aging.reasoning_withheld_exchanges],
      [6, withheld],
    );
    assert.deepEqual(replayed(crossed), [...Array(withheld).fill(false), ...Array(12 - withheld).fill(true)]);
    assert.equal(crossed.manifest.lossless_invariants_before_sha256, crossed.manifest.lossless_invariants_after_sha256);
    // Each withheld state is stored and readable.
    const pass = crossed.manifest.passes.find((item: Json) => item.name === "exchange_reasoning_withheld");
    assert.equal(pass.changed_messages.length, withheld);
    assert.equal(pass.artifact_refs.length, withheld);
    if (withheld) assert.match(catalog.resolve(pass.artifact_refs[0].ref)[1].toString("utf8"), /thought 0/);
    // Below the watermark again: the next request appends, and the new exchange keeps its reasoning.
    messages.push(...exchange(12, 4000, 100));
    const next = await send(context, assembly, messages, 13_000);
    assert.equal(next.manifest.request_relation, "append_only");
    assert.deepEqual(replayed(next), [...Array(withheld).fill(false), ...Array(13 - withheld).fill(true)]);
  }
});

test("the hard backstop ages the exact tail too, and a request that still cannot fit ends the turn", async () => {
  // Hard limit 20,000 - 2,048 - 2,048 = 15,904; no Provider usage, so no watermark is ever crossed.
  for (const management of [V9, UNAGED]) {
    const [context] = pipeline(management, 20_000);
    const assembly = assembler();
    const messages: Json[] = [...PREFIX, OBSERVATION];
    let backstopped: Json = null;
    for (let index = 0; index < 12; index += 1) {
      messages.push(...exchange(index), toolImages(index, 1));
      const sent = await send(context, assembly, messages, null);
      assert.ok(sent.manifest.request_estimate.tokens <= 15_904, String(sent.manifest.request_estimate.tokens));
      backstopped ??= sent.manifest.hard_backstop ? sent : null;
    }
    assert.ok(backstopped, "the history outgrew the hard limit");
    const transition = backstopped.manifest.projection_transitions.find((item: Json) => item.kind === "hard_backstop");
    assert.equal(transition.semantic_loss, false);
    assert.equal(backstopped.manifest.hard_backstop.status, "applied");
    assert.deepEqual(calls(backstopped.messages).map(aged).slice(-2), [false, false]);
    // Reasoning included: only the exact two keep theirs, under either mode.
    const record = backstopped.manifest.exchange_aging;
    assert.equal(record.reasoning_withheld_exchanges, record.aged_exchanges);
    const replayed = backstopped.wire.messages.filter((m: Json) => m.role === "assistant")
      .map((m: Json) => "reasoning_content" in m);
    assert.deepEqual(replayed.slice(-3), [false, true, true]);
    // The backstop's wider span takes the tool-returned images too: only the last two exchanges keep theirs.
    assert.equal(backstopped.manifest.active_inline_images, 2);
    messages.push(...exchange(12, 80_000));
    await assert.rejects(send(context, assembly, messages, null), /context_exhausted: .* hard limit 15904/);
  }
});

test("Context v8 is unchanged: no exchange is aged and no backstop runs", async () => {
  const { exchange_aging_trigger_tokens: _a, exchange_aging_trigger_ratio: _b, exact_tail_exchanges: _c,
    aged_argument_bytes: _d, aged_result_bytes: _e, aged_preview_bytes: _f, backstop_exact_tail_exchanges: _g,
    reasoning_replay: _h, reasoning_tail_exchanges: _i, ...rest } = V9;
  const [context] = pipeline({ ...rest, schema: "ora.native-context-management.v8" });
  const assembly = assembler();
  const messages: Json[] = [...PREFIX, OBSERVATION];
  for (let index = 0; index < 8; index += 1) {
    messages.push(...exchange(index));
    const sent = await send(context, assembly, messages, 30_000);
    assert.ok(!calls(sent.messages).some(aged));
    assert.ok(!("exchange_aging" in sent.manifest) && "context_over_limit" in sent.manifest);
    assert.ok(!sent.manifest.passes.some((item: Json) => String(item.name).startsWith("exchange_")));
  }
});
