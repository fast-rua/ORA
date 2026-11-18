// The context stage driver (src/context/stages/): v6, v8 and v9 as stages, v4, v5
// and v7 refused, context packs as stage parameters, the state record in the
// snapshot's pipeline fields, and the invariants the driver holds. The byte identity of their requests is
// held by context-level replays of sealed runs (test/support/context_replay.ts)
// and the differential fixtures.
//
//   node --test ora-pi/test/context_stages.test.ts

import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { ContextArtifactCatalog } from "../src/context/memory/artifacts.ts";
import { NativeContextPipeline, pipelineState } from "../src/context/native.ts";
import { StageDriver } from "../src/context/stages/driver.ts";
import { schemeFromManagement } from "../src/context/stages/scheme.ts";
import { RequestEstimator } from "../src/context/stages/services/estimator.ts";
import { resolveLimits } from "../src/context/stages/services/limits.ts";
import { NO_POINTERS } from "../src/context/stages/services/references.ts";
import { exchangesOf, segment } from "../src/context/stages/services/segmenter.ts";
import { DEFAULT_CONTEXT_DOMAIN } from "../src/context/domain_defaults.ts";
import { legacyStageScheme } from "./support/legacy_context.ts";
import { STAGE_SCHEMAS } from "./support/stage_schemas.ts";
import { PIPELINE_STATE_FIELDS, restorePipeline, snapshotPipeline } from "../src/core/persistence.ts";
import { ArtifactStore, NO_DURABILITY } from "../src/evidence/artifacts.ts";

type Json = any;

/** context_rolling_tail_v2's management (retired 2026-11-05; tests/data/context_scheme_fixtures/rolling-tail-v2.json). */
const V8 = {
  schema: "ora.native-context-management.v8",
  causal_exact_tail_action_steps: 3,
  camera_images: "rolling_tail_v1",
  camera_tail_observations: 2,
  episode_checkpoint_trigger_tokens: 150000,
  compaction_trigger_tokens: 150000,
  emergency_fallback_ratio: 0.8,
  emergency_target_ratio: 0.65,
  checkpoint_target_estimated_tokens: 1200,
  checkpoint_max_estimated_tokens: 1600,
  image_estimate_tokens: 530,
};

const V6 = {
  schema: "ora.native-context-management.v6",
  causal_exact_tail_action_steps: 3, inline_media_tail_action_steps: 2,
  eager_large_result_bytes: 50_000, eager_preview_bytes: 2_048, eager_exact_deduplication: true,
  trigger_inline_images: 24, trigger_estimated_tokens: 48_000, trigger_provider_input_ratio: 0.70,
  episode_checkpoint_trigger_tokens: 150_000, emergency_fallback_ratio: 0.80, emergency_target_ratio: 0.65,
  checkpoint_target_estimated_tokens: 1200, checkpoint_max_estimated_tokens: 1600, tail_aging_stride_action_steps: 4,
};

/** v5: refused as a plan; its differential fixtures run as a stage scheme (test/support/legacy_context.ts). */
const V5 = {
  ...V6, schema: "ora.native-context-management.v5", episode_checkpoint_trigger_ratio: 0.2,
};

/** The rewrite_compact_v1 pipeline (no management). */
function rewritePipeline(): NativeContextPipeline {
  const root = mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), "ora-context-stages-"));
  return new NativeContextPipeline({
    profile: "native_minimal_v1", artifacts: new ArtifactStore(root, { durability: NO_DURABILITY }),
    contextWindowTokens: 200_000, maxOutputTokens: 8192,
  });
}

function pipeline(management: Json = V8, pack: Json = {}): NativeContextPipeline {
  const root = mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), "ora-context-stages-"));
  const artifacts = new ArtifactStore(root, { durability: NO_DURABILITY });
  return new NativeContextPipeline({
    profile: "native_causal_context_v1", artifacts, contextWindowTokens: 200_000, maxOutputTokens: 8192,
    retention: "action_step_lossless_v4", summaryPolicy: "rollout_checkpoint_v1",
    summaryProvider: { summarize: async () => { throw new Error("no compaction here"); } } as Json,
    contextManagement: management, stageSchemas: STAGE_SCHEMAS,
    contextArtifacts: new ContextArtifactCatalog(root, "policy-session-1", artifacts),
    mutationTools: ["move_pose"], ...pack,
  });
}

function observation(step: number): Json {
  return {
    role: "user",
    content: [{ type: "text", text: `{"current_actual": {"actual_sequence": ${step}}}` }],
    _observation: { step, observation_id: `internal-observation-${step}`, keyframe: true },
    _causal_observation: { step, result_actual_sequence: step },
    _session_round: 1,
  };
}

const PREFIX: Json[] = [
  { role: "system", content: "minimal harness", _protected_context: true, _context_source: "harness_system" },
  { role: "user", content: "exact episode task", _protected_context: true, _context_source: "episode_task" },
];

const ON = { pressureExternalization: true, rolloutCompaction: true };

function driverOf(target: NativeContextPipeline): StageDriver {
  assert.ok(target.driver !== null);
  return target.driver;
}

test("v8 translates into five stages", () => {
  const scheme = schemeFromManagement(V8, ON)!;
  assert.deepEqual(scheme.stages.map((spec) => spec.stage),
    ["materialize", "camera_delivery", "episode_capsule", "emergency_checkpoint", "over_limit_guard"]);
  assert.deepEqual(scheme.segments, { unit: "action_step", exact_tail: 3 });
  assert.deepEqual(scheme.estimator, { image_tokens: 530 });
  assert.deepEqual(scheme.stages[1].params, { mode: "rolling_tail_v1", tail_observations: 2 });
  assert.equal(scheme.stages[3].params.trigger_tokens, 150000);
  assert.deepEqual(scheme.stages[4].params, { over_limit: "send_and_record" });
});

test("v6 translates into eight stages; the core refuses v4, v5 and v7", () => {
  const scheme = schemeFromManagement(V6, ON)!;
  assert.deepEqual(scheme.stages.map((spec) => spec.stage), [
    "materialize", "stride_aging", "eager_externalize", "media_window", "pressure_externalize",
    "episode_capsule", "emergency_checkpoint", "over_limit_guard",
  ]);
  assert.deepEqual(scheme.estimator, { image_tokens: 4096 });
  assert.deepEqual(scheme.stages[0].params, { reasoning: "exclude" });
  assert.deepEqual(scheme.references, { pointers: true, advertise_over_bytes: 50_000, preview: { head_bytes: 2048 } });
  assert.deepEqual(scheme.limits, { reserve_tokens: 2048 });
  assert.deepEqual(scheme.stages[1].params, { stride: 4, strided: true });
  assert.deepEqual(scheme.stages[3].params, { tail_action_steps: 2 });
  assert.equal(scheme.stages[4].params.inline_images, 24);
  assert.equal(scheme.stages[6].params.trigger_tokens, null);
  assert.equal(scheme.stages[5].params.retry_attempts, 4);
  assert.equal(scheme.stages[5].params.retry_backoff_action_steps, 4);
  assert.deepEqual(scheme.stages[7].params, { over_limit: "refuse" });
  assert.ok(pipeline(V6).driver !== null);
  for (const schema of ["v4", "v5", "v7"]) {
    assert.throws(() => pipeline({ ...V5, schema: `ora.native-context-management.${schema}` }),
      new RegExp(`context_management ora.native-context-management.${schema} is retired from the agent core`));
  }
});

test("v4 and v5 map onto two stage parameters: unstrided aging and the window-scaled L5 watermark", async () => {
  const v5 = legacyStageScheme(V5)!;
  const params = Object.fromEntries(v5.stages.map((spec) => [spec.stage, spec.params]));
  assert.deepEqual(params.stride_aging, { stride: 4, strided: true });
  assert.equal(params.episode_capsule.trigger_ratio, 0.2);
  const { tail_aging_stride_action_steps: _stride, ...v4Fields } = V5;
  const v4 = legacyStageScheme({ ...v4Fields, schema: "ora.native-context-management.v4" })!;
  assert.deepEqual(v4.stages[1].params, { stride: 1, strided: false });
  assert.equal(legacyStageScheme(V6), null);
  const target = pipeline(null, { stageScheme: v5 });
  const projected = await target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 });
  // max(150,000, 0.2 x (200,000 - 8,192 - 2,048)) = 150,000; the formula names the scaling.
  assert.equal(projected.manifest.resolved_episode_checkpoint_trigger_tokens, 150_000);
  assert.equal(projected.manifest.resolved_episode_checkpoint_trigger_formula,
    "max(absolute_tokens, int(hard_input_tokens * ratio))");
  assert.equal(projected.manifest.context_management.schema, "ora.native-context-management.v5");
  assert.throws(() => new StageDriver({ ...v4, stages: v4.stages.map((spec) => (spec.stage === "stride_aging"
    ? { stage: "stride_aging", params: { stride: 4, strided: false } } : spec)) },
  driverOf(target).services, driverOf(target).limits, driverOf(target).identity, STAGE_SCHEMAS),
  /context stage stride_aging: \$input/);
});

test("a context pack turns L4 and L5/L6 off through the stages' enabled parameter", async () => {
  const scheme = schemeFromManagement(V6, { pressureExternalization: false, rolloutCompaction: false })!;
  const enabled = Object.fromEntries(scheme.stages.filter((spec) => "enabled" in spec.params)
    .map((spec) => [spec.stage, spec.params.enabled]));
  assert.deepEqual(enabled, { pressure_externalize: false, episode_capsule: false, emergency_checkpoint: false });
  const target = pipeline(V6, { pressureExternalization: false, rolloutCompaction: false });
  const projected = await target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 });
  assert.equal(projected.manifest.resolved_inline_image_target, 24, "a disabled stage keeps its manifest fields");
  assert.equal(projected.manifest.episode_checkpoint_admission.reason, "rollout_compaction_disabled");
  assert.equal(projected.manifest.emergency_compaction_admission.reason, "rollout_compaction_disabled");
});

test("a v8 projection runs on the driver and records each stage's manifest fields", async () => {
  const target = pipeline();
  const projected = await target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 });
  const manifest = projected.manifest;
  assert.equal(manifest.schema, "ora.native-context-projection.v6");
  assert.equal(manifest.resolved_media_retention_policy, "append_only_until_compaction_v1");
  assert.equal(manifest.camera_delivery.camera_images, "rolling_tail_v1");
  assert.equal(manifest.context_over_limit, null);
  assert.equal(manifest.resolved_compaction_trigger_tokens, 150000);
  assert.equal(manifest.episode_checkpoint_admission.reason, "no_closed_episode");
  assert.equal(manifest.emergency_compaction_admission.reason, "provider_usage_unavailable");
  assert.deepEqual(manifest.passes.map((pass: Json) => pass.name), [
    "recoverable_payload_materialization", "l1_large_result_reference", "l2_exact_duplicate_reference",
    "l3_lifecycle_media_reference", "l4_read_only_result_reference", "l4_observation_reference",
    "reasoning_exclusion",
  ]);
});

test("the state record keeps the snapshot's pipeline fields, and restores", async () => {
  const target = pipeline();
  const fresh = snapshotPipeline(rewritePipeline());
  const initial = snapshotPipeline(target);
  assert.deepEqual(Object.keys(initial), [...PIPELINE_STATE_FIELDS, "episodeRetryState"]);
  assert.deepEqual(initial, fresh, "a fresh v8 driver serializes as a fresh rewrite_compact_v1 pipeline");
  const projected = await target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 });
  target.observeProviderUsage({
    estimatedTokens: projected.manifest.estimated_input_tokens, inputTokens: 1234, projectionEpoch: 0,
    providerWireRequestSha256: "a".repeat(64),
  });
  const after = snapshotPipeline(target);
  assert.equal(after.lastProviderInputTokens, 1234);
  assert.equal(pipelineState(target).last_provider_input_tokens, 1234);
  const resumed = pipeline();
  restorePipeline(resumed, after);
  assert.deepEqual(snapshotPipeline(resumed), after);
  assert.throws(() => restorePipeline(pipeline(), { ...after, lifecycleMediaExternalizedActionStepCount: 2 }),
    /carries lifecycleMediaExternalizedActionStepCount, which this stage scheme does not run/);
  const rewrite = rewritePipeline();
  restorePipeline(rewrite, { ...fresh, lastProviderInputTokens: 99, lastEstimatedTokens: 90 });
  assert.equal(snapshotPipeline(rewrite).lastProviderInputTokens, 99);
  assert.throws(() => restorePipeline(rewritePipeline(), { ...fresh, episodeCheckpointCount: 1 }),
    /carries episodeCheckpointCount, which rewrite_compact_v1 does not run/);
});

test("the guard's policy decides an over-limit request: send and record, or refuse", async () => {
  const messages = [...PREFIX, observation(0)];
  const over = async (target: NativeContextPipeline) => {
    const projected = await target.project({ messages, tools: [], observationStep: 0 });
    target.observeProviderUsage({
      estimatedTokens: projected.manifest.estimated_input_tokens, inputTokens: 500_000, projectionEpoch: 0,
      providerWireRequestSha256: "b".repeat(64),
    });
    return target.project({ messages, tools: [], observationStep: 0 });
  };
  const sent = await over(pipeline());
  assert.equal(sent.manifest.context_over_limit.status, "warning_sent_anyway");
  assert.equal(sent.manifest.context_over_limit.over_limit_request_count, 1);
  const refusing = pipeline();
  (driverOf(refusing).stage("over_limit_guard") as Json).params = { over_limit: "refuse" };
  await assert.rejects(over(refusing), /context_exhausted: Provider reported 500000 input tokens/);
});

/** Make emergency_checkpoint admit every request and apply with ``apply``. */
function misbehave(target: NativeContextPipeline, apply: (view: Json, state: Json, admission: Json, context: Json) => Promise<Json>) {
  const stage = driverOf(target).stage("emergency_checkpoint") as Json;
  stage.admit = () => ({ admitted: true, record: {} });
  stage.apply = apply;
}

test("the driver refuses a change outside a named transition", async () => {
  const request = { messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 };
  let target = pipeline();
  misbehave(target, async (_view, _state, _admission, context) => { context.commit(); return null; });
  await assert.rejects(target.project(request), /changed the request without a transition/);
  target = pipeline();
  misbehave(target, async () => ({ kind: "emergency_compaction", record: {} }));
  await assert.rejects(target.project(request), /recorded a transition it did not commit/);
  target = pipeline();
  misbehave(target, async (_view, _state, _admission, context) => {
    context.commit();
    return { kind: "exchange_aging", record: {} };
  });
  await assert.rejects(target.project(request), /recorded an undeclared transition exchange_aging/);
});

test("a stage writes only its own slice", async () => {
  const target = pipeline();
  misbehave(target, async (view) => {
    view.stateOf("episode_capsule").count += 1;
    return null;
  });
  await assert.rejects(target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 }),
    /emergency_checkpoint wrote the state of episode_capsule/);
});

test("the fixed prefix never ages", async () => {
  const target = pipeline();
  const camera = driverOf(target).stage("camera_delivery") as Json;
  const render = camera.render.bind(camera);
  let poison = false;
  camera.render = (view: Json, state: Json, history: Json[]) => {
    const rendered = render(view, state, history);
    if (poison) {
      rendered.history = rendered.history.map((message: Json) => (message._context_source === "harness_system"
        ? { ...message, content: "a rewritten system prompt" } : message));
    }
    return rendered;
  };
  misbehave(target, async (_view, _state, _admission, context) => {
    poison = true;
    context.commit();
    return { kind: "emergency_compaction", record: {} };
  });
  await assert.rejects(target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 }),
    /emergency_checkpoint changed the fixed prefix/);
});

test("a scheme must start with one materialize stage and list each stage once", () => {
  const scheme = schemeFromManagement(V8, ON)!;
  const services = driverOf(pipeline()).services;
  const limits = resolveLimits(200_000, 8192);
  assert.equal(limits.hardInputTokens, 189_760);
  assert.equal(limits.targetInputTokens, 143_856);
  const identity = { profile: "native_causal_context_v1", retention: "action_step_lossless_v4", summaryPolicy: "rollout_checkpoint_v1" };
  assert.throws(() => new StageDriver({ ...scheme, stages: scheme.stages.slice(1) }, services, limits, identity, STAGE_SCHEMAS),
    /exactly one materialize stage/);
  assert.throws(() => new StageDriver({ ...scheme, stages: [...scheme.stages, scheme.stages[1]] }, services, limits, identity, STAGE_SCHEMAS),
    /lists stage camera_delivery twice/);
  assert.throws(() => new StageDriver({ ...scheme, stages: [...scheme.stages, { stage: "action_digest", params: {} }] },
    services, limits, identity, STAGE_SCHEMAS), /no parameter schema for context stage action_digest/);
  const v6 = schemeFromManagement(V6, ON)!;
  const [materialize, stride, eager, ...rest] = v6.stages;
  assert.throws(() => new StageDriver({ ...v6, stages: [materialize, eager, stride, ...rest] }, services, limits, identity, STAGE_SCHEMAS),
    /eager_externalize must run after stride_aging/);
  assert.throws(() => new StageDriver({ ...v6, stages: [...v6.stages, scheme.stages[1]] }, services, limits, identity, STAGE_SCHEMAS),
    /excludes/);
});

test("the segmenter counts ActionSteps and exchanges", () => {
  const call = (id: string): Json => ({
    role: "assistant", content: null, tool_calls: [{ id, type: "function", function: { name: "observe", arguments: "{}" } }],
  });
  const result = (id: string): Json => ({ role: "tool", tool_call_id: id, name: "observe", content: "{}" });
  const messages = [
    ...PREFIX, observation(0), call("a"), result("a"), call("b"), result("b"), observation(1), call("c"), result("c"),
  ];
  const exchanges = exchangesOf(messages);
  // Exchange 0 starts at the opening observation; each later one at its assistant message.
  assert.deepEqual(exchanges, { from: 2, starts: [3, 5, 8], spans: [[2, 5], [5, 8], [8, 10]] });
  const actionSteps = segment(messages, { unit: "action_step", exactTail: 3, mutationTools: [], domain: DEFAULT_CONTEXT_DOMAIN });
  assert.deepEqual(actionSteps.spans, [[2, 7], [7, 10]]);
  assert.equal(actionSteps.completedCount, 1);
  assert.equal(actionSteps.exactTailFrom, 0);
  const byExchange = segment(messages, { unit: "exchange", exactTail: 3, mutationTools: [], domain: DEFAULT_CONTEXT_DOMAIN });
  assert.deepEqual(byExchange.spans, exchanges.spans);
  assert.deepEqual(byExchange.groups, actionSteps.groups, "the summaries keep counting ActionSteps");
});

test("the estimator anchors on a Provider report of this epoch, the limits keep their reserve", () => {
  const estimator = new RequestEstimator(530);
  const request = { messages: [{ role: "user", content: [{ type: "text", text: "look" },
    { type: "image_url", image_url: { url: "data:image/png;base64," + "A".repeat(4000) } }] }] };
  assert.equal(estimator.imageEstimate(request), 530);
  assert.ok(estimator.raw(request) > estimator.pressure(request), "an image is charged 530 tokens, not its bytes");
  assert.deepEqual(estimator.calibrate(100, { applies: true, inputTokens: 900, estimatedTokens: 800 }),
    { tokens: 200, delta: 100, status: "provider_anchored_suffix_estimate" });
  assert.deepEqual(estimator.calibrate(100, { applies: false, inputTokens: 900, estimatedTokens: 800 }),
    { tokens: 100, delta: 0, status: "projection_epoch_unanchored" });
  assert.equal(resolveLimits(200_000, 8192, 4096).hardInputTokens, 187_712);
  assert.throws(() => resolveLimits(10_000, 8192), /leaves no safe input budget/);
  const scheme = schemeFromManagement(V8, ON);
  assert.deepEqual(scheme.references, NO_POINTERS);
});

test("a plan's stage list runs when it is its management object's, and is refused otherwise", async () => {
  const scheme = schemeFromManagement(V8, ON);
  const list = { schema: "ora.context-stages.v1", segments: scheme.segments, limits: scheme.limits,
    estimator: scheme.estimator, references: scheme.references, stages: scheme.stages };
  const target = pipeline(V8, { contextStages: list });
  const projected = await target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 });
  assert.equal(projected.manifest.context_management.schema, "ora.native-context-management.v8");
  const changed = structuredClone(list);
  changed.stages[1].params.tail_observations = 3;
  assert.throws(() => pipeline(V8, { contextStages: changed }),
    /policy.context_stages is not the stage list of policy.context_management/);
  // A context pack's switch turns the listed stages off as it does the translated ones.
  assert.ok(pipeline(V8, { contextStages: list, rolloutCompaction: false }).driver !== null);
  assert.throws(() => pipeline(V8, { stageSchemas: null }), /needs the context stage schemas/);
  // Every stage the core builds is validated with the schemas Python sent.
  const bad = structuredClone(list);
  bad.stages[1].params.tail_observations = 1;
  assert.throws(() => pipeline(null, { stageScheme: { ...scheme, stages: bad.stages } }),
    /context stage camera_delivery: \$input/);
});

/** context_rolling_tail_v3's management (retired 2026-11-05; tests/data/context_scheme_fixtures/rolling-tail-v3.json). */
const V9 = {
  ...V8, schema: "ora.native-context-management.v9", camera_images: "exchange_window_v1", camera_tail_observations: null,
  exchange_aging_trigger_tokens: 100_000, exchange_aging_trigger_ratio: 0.5, exact_tail_exchanges: 6,
  aged_argument_bytes: 1024, aged_result_bytes: 2048, aged_preview_bytes: 256, backstop_exact_tail_exchanges: 2,
  reasoning_replay: "all", reasoning_tail_exchanges: null,
};

test("v9 runs as stages counted in exchanges, and its exchange stages need each other", async () => {
  const scheme = schemeFromManagement(V9, ON);
  assert.deepEqual(scheme.stages.map((spec) => spec.stage), [
    "materialize", "exchange_aging", "reasoning_replay", "camera_delivery", "episode_capsule",
    "emergency_checkpoint", "hard_backstop",
  ]);
  assert.equal(scheme.segments.unit, "exchange");
  assert.deepEqual(scheme.stages[1].params.preview, { head_bytes: 128, tail_bytes: 128 });
  const target = pipeline(V9);
  const projected = await target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 });
  const manifest = projected.manifest;
  assert.equal(manifest.exchange_aging.admission.reason, "provider_usage_unavailable");
  assert.deepEqual(manifest.request_estimate.hard_input_tokens, target.hardInputTokens);
  assert.equal(manifest.hard_backstop, null);
  assert.equal("context_over_limit" in manifest, false, "the backstop sizes the request; no over-limit record");
  assert.deepEqual(pipelineState(target).exchange_aging, {
    aged_exchanges: 0, aged_observation_groups: 0, reasoning_withheld_exchanges: 0, aging_count: 0,
    hard_backstop_count: 0,
  });
  const snapshot = snapshotPipeline(target);
  assert.deepEqual(snapshot.exchangeAging, {
    exchangeAgedCount: 0, exchangeAgedObservationGroups: 0, reasoningWithheldExchangeCount: 0,
    exchangeAgingCount: 0, hardBackstopCount: 0,
  });
  assert.equal("exchangeAging" in snapshotPipeline(pipeline(V8)), false, "v8's snapshot is unchanged");
  const restored = pipeline(V9);
  restorePipeline(restored, { ...snapshot, exchangeAging: { ...snapshot.exchangeAging, exchangeAgedCount: 4 } });
  assert.equal((pipelineState(restored) as Json).exchange_aging.aged_exchanges, 4);
  const without = (stage: string) => ({ ...scheme, stages: scheme.stages.filter((spec) => spec.stage !== stage) });
  assert.throws(() => pipeline(null, { stageScheme: without("reasoning_replay") }), /exchange_aging requires reasoning_replay/);
  assert.throws(() => pipeline(null, { stageScheme: without("exchange_aging") }), /requires exchange_aging/);
  // exchange_window_v1 leaves camera images to the exchange aging.
  const v8Window = schemeFromManagement(V8, ON);
  v8Window.stages[1] = { stage: "camera_delivery", params: { mode: "exchange_window_v1", tail_observations: null } };
  await assert.rejects(pipeline(null, { stageScheme: v8Window })
    .project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 }), /needs the exchange_aging stage/);
});

test("L6 may be ratio-only: trigger_tokens null with a trigger ratio, reported as its resolved watermark", async () => {
  const scheme = schemeFromManagement(V9, ON);
  const ratioOnly = {
    ...scheme,
    stages: scheme.stages.map((spec) => spec.stage !== "emergency_checkpoint" ? spec
      : { stage: spec.stage, params: { ...spec.params, trigger_tokens: null, trigger_ratio: 0.6, target_ratio: 0.5 } }),
  };
  const target = pipeline(null, { stageScheme: ratioOnly });
  const projected = await target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 });
  assert.equal(projected.manifest.resolved_compaction_trigger_tokens, Math.trunc(target.hardInputTokens * 0.6));
  // v6's ratio-only L6 keeps its manifest as it was: the legacy format has the field only for v8 and v9.
  const v6 = await pipeline(V6).project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 });
  assert.equal("resolved_compaction_trigger_tokens" in v6.manifest, false);
});

test("v9's L6 may be ratio-only in the management object; v8's may not", async () => {
  const exchange = { ...V9, compaction_trigger_tokens: null, emergency_fallback_ratio: 0.6, emergency_target_ratio: 0.5 };
  const scheme = schemeFromManagement(exchange, ON);
  const emergency = scheme.stages.find((spec) => spec.stage === "emergency_checkpoint")!;
  assert.deepEqual([emergency.params.trigger_tokens, emergency.params.trigger_ratio, emergency.params.target_ratio],
    [null, 0.6, 0.5]);
  const target = pipeline(exchange);
  const projected = await target.project({ messages: [...PREFIX, observation(0)], tools: [], observationStep: 0 });
  assert.equal(projected.manifest.resolved_compaction_trigger_tokens, Math.trunc(target.hardInputTokens * 0.6));
  // Only the ratios compare when L6 is ratio-only.
  assert.ok(pipeline({ ...exchange, exchange_aging_trigger_tokens: 900_000 }));
  assert.throws(() => pipeline({ ...exchange, exchange_aging_trigger_ratio: 0.6 }), /below the L6 watermark/);
  assert.throws(() => pipeline({ ...V8, compaction_trigger_tokens: null }), /compaction_trigger_tokens must be positive/);
});
