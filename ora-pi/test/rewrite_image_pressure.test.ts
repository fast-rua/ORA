// Toolkit's rewrite context must budget image tokens, not their transport bytes.
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deflateSync } from "node:zlib";
import { test } from "node:test";
import { contextDeps } from "../src/core/context_wiring.ts";
import { NativeContextPipeline, providerEstimatedInputTokens } from "../src/context/native.ts";
import { IMAGE_PRESSURE_ESTIMATE_TOKENS, inlineImageCount } from "../src/context/projection.ts";
import { ArtifactStore, NO_DURABILITY } from "../src/evidence/artifacts.ts";

type Json = any;
// The policy of the retired baseline robodojo_toolkit_v1 (rewrite_compact_v1 retention), kept here as the fixture:
// sealed runs replay with it, and no selectable baseline carries it since 2026-10-31.
const baseline = {
  policy: {
    context_profile: "native_causal_context_v1", context_composition: "native_slots_v3",
    context_retention: "rewrite_compact_v1", summary_policy: "disabled", memory_policy: "disabled",
    provider_continuation_replay: "exact_v1", initial_observation: "automatic", timeout_s: 900,
    max_model_calls_per_observation: 120, action_codec: "motion_action_v3", require_live_marker: true,
    provider_recovery: "first_token_deadline_v1", context_token_estimator: "image_aware_v1",
  },
};
const names = ["observe", "compute", "vision_ground", "grasp_plan", "grasp_execute", "move_to", "move_pose", "check_task", "finish_episode"];
const tools = names.map((name) => ({ type: "function", function: { name, parameters: { type: "object" } } }));
const HARD = 1_048_576 - 16_384 - 2048;

/** Valid RGB PNGs with identical 640x480 pixels, but very different transport sizes. */
function png(level: number): string {
  const chunk = (kind: string, data: Buffer) => {
    const body = Buffer.concat([Buffer.from(kind), data]);
    let crc = 0xffffffff;
    for (const byte of body) {
      crc ^= byte;
      for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
    const size = Buffer.alloc(4), check = Buffer.alloc(4);
    size.writeUInt32BE(data.length); check.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
    return Buffer.concat([size, body, check]);
  };
  const header = Buffer.alloc(13); header.writeUInt32BE(640); header.writeUInt32BE(480, 4);
  header[8] = 8; header[9] = 2;
  const rows = Buffer.alloc((640 * 3 + 1) * 480, 127);
  for (let y = 0; y < 480; y++) rows[y * (640 * 3 + 1)] = 0;
  return "data:image/png;base64," + Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", header),
    chunk("IDAT", deflateSync(rows, { level })), chunk("IEND", Buffer.alloc(0)),
  ]).toString("base64");
}

function appendResult(messages: Json[], index: number, image: string): void {
  messages.push(
    { role: "assistant", content: "", tool_calls: [{ id: `call-${index}`, type: "function",
      function: { name: "vision_ground", arguments: "{}" } }] },
    { role: "tool", tool_call_id: `call-${index}`, content: '{"success":true}' },
    { role: "user", _tool_image_step: 0, content: Array.from({ length: 4 }, () => (
      { type: "image_url", image_url: { url: image } })) },
  );
}

function wired(root: string): Json {
  return contextDeps({ runRoot: root, artifacts: new ArtifactStore(root, { durability: NO_DURABILITY }) } as Json, {
    policy: { ...baseline.policy, model: "fixture", context_window_tokens: 1_048_576, max_output_tokens: 16_384 },
    tools: { mutation_tools: ["move_pose", "move_to", "grasp_execute"] }, domain: {}, provider: { kind: "fake" },
  } as Json, {} as Json).pipeline;
}

test("three same-step four-image results fit before any usage anchor, independent of PNG bytes", async () => {
  const root = mkdtempSync(join(tmpdir(), "ora-image-pressure-"));
  try {
    const compact = png(9), large = png(0);
    assert.ok(large.length > compact.length * 100);
    const estimates: number[][] = [];
    for (const [mode, image] of [["compact", compact], ["large", large]]) {
      const pipeline = wired(join(root, mode));
      const messages: Json[] = [{ role: "system", content: "Use current public evidence." }];
      const row: number[] = [];
      for (let call = 0; call < 3; call++) {
        appendResult(messages, call, image);
        const result = await pipeline.project({ messages, tools, observationStep: 0 });
        const m = result.manifest;
        assert.equal(m.context_token_estimator, "image_aware_v1");
        assert.equal(m.estimate_calibration_delta_tokens, 0);
        assert.equal(m.provider_usage_anchor_tokens, null);
        assert.equal(m.estimated_image_tokens, (call + 1) * 4 * IMAGE_PRESSURE_ESTIMATE_TOKENS);
        assert.equal(inlineImageCount(result.messages), (call + 1) * 4);
        assert.equal(result.messages.at(-1).content[0].image_url.url, image);
        assert.deepEqual(result.tools, tools); // no reader or substitute tool
        assert.ok(m.estimated_input_tokens < HARD / 10);
        assert.equal(providerEstimatedInputTokens(m), m.pressure_estimated_input_tokens);
        if (mode === "large" && call === 2) assert.ok(m.raw_estimated_input_tokens > HARD);
        row.push(m.estimated_input_tokens);
      }
      assert.ok(row[2] - row[1] >= 4 * IMAGE_PRESSURE_ESTIMATE_TOKENS);
      assert.ok(row[2] - row[1] < 4 * IMAGE_PRESSURE_ESTIMATE_TOKENS + 256);
      estimates.push(row);
    }
    assert.deepEqual(estimates[0], estimates[1]);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("usage calibration anchors image-aware pressure and real text still hits the hard limit", async () => {
  const root = mkdtempSync(join(tmpdir(), "ora-image-anchor-"));
  try {
    const pipeline = wired(root);
    const messages: Json[] = [];
    appendResult(messages, 0, png(0));
    const first = await pipeline.project({ messages, tools, observationStep: 0 });
    pipeline.observeProviderUsage({ estimatedTokens: providerEstimatedInputTokens(first.manifest), inputTokens: 21457 });
    const second = await pipeline.project({ messages, tools, observationStep: 0 });
    assert.equal(second.manifest.estimated_input_tokens, 21457);
    appendResult(messages, 1, png(9));
    const next = await pipeline.project({ messages, tools, observationStep: 0 });
    assert.ok(next.manifest.estimated_input_tokens > 21457 + 4 * IMAGE_PRESSURE_ESTIMATE_TOKENS);
    const strict = new NativeContextPipeline({ profile: "native_causal_context_v1",
      artifacts: new ArtifactStore(root, { durability: NO_DURABILITY }), contextWindowTokens: 8192,
      maxOutputTokens: 2048, tokenEstimator: "image_aware_v1" });
    await assert.rejects(strict.project({ messages: [{ role: "system", content: "x".repeat(100_000) }],
      tools: [], observationStep: 0 }), /context_exhausted/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
