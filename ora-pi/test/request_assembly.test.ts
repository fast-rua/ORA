// Request assembly on the lossless Context (src/context/request_assembly.ts):
// a named tool choice escalates on the cached prefix, an Anthropic thinking
// request turns it into a Runtime-validated requirement, a silent history
// rewrite inside one cache epoch is refused and named, and a signed provider
// continuation survives projection, assembly and an appended turn. Ported from
// the retired Python Native core's request-assembly tests, on the same projection.
//
//   node --test ora-pi/test/request_assembly.test.ts

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { ContextArtifactCatalog } from "../src/context/memory/artifacts.ts";
import { NativeContextPipeline } from "../src/context/native.ts";
import { STAGE_SCHEMAS } from "./support/stage_schemas.ts";
import { canonicalBytesPy, clonePy } from "../src/context/pyutil.ts";
import { NativeRequestAssembler } from "../src/context/request_assembly.ts";
import { ArtifactStore, NO_DURABILITY } from "../src/evidence/artifacts.ts";
import { translateAnthropicRequest } from "../src/provider/anthropic.ts";
import { ProviderContinuationState } from "../src/provider/continuation.ts";

type Json = any;

/**
 * The lossless retention of the Python tests (`test_lossless_context_runtime._pipeline`) on v6:
 * stride 1 ages at every observation as their v4 did, and the L5 watermark sits above the L4
 * estimate floor (no request here closes an Episode).
 */
const MANAGEMENT = {
  schema: "ora.native-context-management.v6",
  causal_exact_tail_action_steps: 3, inline_media_tail_action_steps: 2,
  eager_large_result_bytes: 50_000, eager_preview_bytes: 2_048, eager_exact_deduplication: true,
  trigger_inline_images: 2, trigger_estimated_tokens: 1_000_000, trigger_provider_input_ratio: 0.70,
  episode_checkpoint_trigger_tokens: 2_000_000,
  emergency_fallback_ratio: 0.80, emergency_target_ratio: 0.65,
  checkpoint_target_estimated_tokens: 1200, checkpoint_max_estimated_tokens: 1600,
  tail_aging_stride_action_steps: 1,
};

function pipeline(): NativeContextPipeline {
  const root = mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), "ora-assembly-"));
  const artifacts = new ArtifactStore(root, { durability: NO_DURABILITY });
  return new NativeContextPipeline({
    profile: "native_causal_context_v1", artifacts, contextWindowTokens: 200_000, maxOutputTokens: 2048,
    retention: "action_step_lossless_v4", summaryPolicy: "rollout_checkpoint_v1",
    summaryProvider: {
      summarize: async () => {
        throw new Error("no compaction in these requests");
      },
    } as Json,
    contextManagement: MANAGEMENT, stageSchemas: STAGE_SCHEMAS,
    contextArtifacts: new ContextArtifactCatalog(root, "policy-session-1", artifacts),
    mutationTools: ["move_pose", "close_gripper", "open_gripper"],
  });
}

function cachePrefix(): Json[] {
  return [
    { role: "system", content: "minimal harness", _protected_context: true, _context_source: "harness_system" },
    { role: "user", content: "environment and embodiment", _protected_context: true,
      _context_source: "project_instructions" },
    { role: "user", content: "exact episode task", _protected_context: true, _context_source: "episode_task" },
  ];
}

/** A current-actual observation (`dumps(..., sort_keys=True)` text, one label, one image). */
function cacheActual(sequence: number): Json {
  return {
    role: "user",
    content: [
      { type: "text",
        text: `{"comparison": {"available": ${sequence > 0 ? "true" : "false"}}, "current_actual": {"actual_sequence": ${sequence}}}` },
      { type: "text", text: `ACTION RESULT ACTUAL · sequence ${sequence} · agentview` },
      { type: "image_url", image_url: { url: "data:image/png;base64,iVBORw0KGgo=" } },
    ],
    _observation: { step: sequence, observation_id: `internal-observation-${sequence}`, keyframe: true },
    _causal_observation: { step: sequence, result_actual_sequence: sequence },
    _session_round: 1,
  };
}

const FINISH_TOOL = [{
  type: "function",
  function: {
    name: "memory.finish", description: "Finalize the Memory transaction.",
    parameters: { type: "object", properties: {}, additionalProperties: false },
  },
}];

function assembler(reasoningEffort: string): NativeRequestAssembler {
  return new NativeRequestAssembler({
    model: "k3", reasoningEffort, maxOutputTokens: 4096, stream: true,
    prepareRequest: (execution) => translateAnthropicRequest(execution),
  });
}

const sha = (value: unknown) => createHash("sha256").update(canonicalBytesPy(value)).digest("hex");

test("a named tool choice escalates on the cached prefix", async () => {
  const context = pipeline();
  const messages = [...cachePrefix(), cacheActual(0)];
  const assembly = assembler("provider_default");
  assembly.prepare(await context.project({ messages, tools: FINISH_TOOL, observationStep: 0 }));
  const terminal = await context.project({ messages, tools: FINISH_TOOL, observationStep: 0 });
  const prepared = assembly.prepare(terminal, { type: "function", function: { name: "memory.finish" } });
  assert.equal(terminal.manifest.request_relation, "append_only");
  assert.equal(terminal.manifest.canonical_prefix_relation.status, "equal");
  assert.equal(prepared.canonical.tool_choice.function.name, "memory.finish");
  assert.equal(prepared.wire.tool_choice.type, "tool");
  assert.ok(prepared.wire.tool_choice.name.startsWith("memory_finish_"), prepared.wire.tool_choice.name);
  assert.equal(prepared.toolChoiceEvidence.mode, "provider_named");
});

test("Anthropic thinking turns a named tool choice into a Runtime-validated requirement", async () => {
  const projected = await pipeline().project({
    messages: [...cachePrefix(), cacheActual(0)], tools: FINISH_TOOL, observationStep: 0,
  });
  const requested = { type: "function", function: { name: "memory.finish" } };
  const prepared = assembler("medium").prepare(projected, requested);
  assert.equal(prepared.canonical.tool_choice, "auto");
  assert.equal(prepared.execution.tool_choice, "auto");
  assert.deepEqual(prepared.wire.tool_choice, { type: "auto" });
  assert.deepEqual(prepared.wire.thinking, { type: "adaptive" });
  assert.deepEqual(prepared.toolChoiceEvidence, {
    requested, effective: "auto", mode: "runtime_validated_auto",
    reason: "provider_thinking_incompatible_with_named_choice",
  });
});

test("a silent history rewrite inside one cache epoch is refused and named", async () => {
  const context = pipeline();
  const assembly = assembler("medium");
  const messages = [...cachePrefix(), cacheActual(0)];
  const first = await context.project({ messages, tools: [], observationStep: 0 });
  const prepared = assembly.prepare(first);
  assert.equal(first.manifest.canonical_request_sha256, sha(prepared.canonical));
  const rewritten = clonePy(messages);
  rewritten[rewritten.length - 1].content[1].text = "rewritten historical actual";
  rewritten.push(cacheActual(1));
  const projected = await context.project({ messages: rewritten, tools: [], observationStep: 1 });
  // The refusal names the message: the candidate request is discarded with the
  // run, and nothing else records what changed.
  assert.throws(() => assembly.prepare(projected), (error: Error) => {
    assert.match(error.message, /cache_prefix_rewritten/);
    assert.match(error.message, /at message /);
    assert.ok(error.message.includes("was ") && error.message.includes(", now "), error.message);
    return true;
  });
});

test("a signed provider continuation survives projection, assembly and an appended turn", async () => {
  const context = pipeline();
  const assembly = assembler("medium");
  const assistant: Json = { role: "assistant", content: "ready", reasoning_content: "private thought" };
  const signed = [
    { type: "thinking", thinking: "private thought", signature: "signed-state" },
    { type: "text", text: "ready" },
  ];
  assistant._provider_continuation_state = ProviderContinuationState.create({
    providerProtocol: "anthropic_messages", model: "k3", nativeKind: "anthropic_assistant_content_blocks.v1",
    nativePayload: { content: signed }, canonicalMessage: assistant, responseId: "msg-signed",
  }).toDict();
  const messages: Json[] = [...cachePrefix(), cacheActual(0), assistant];
  const projected = await context.project({ messages, tools: [], observationStep: 0 });
  const last = projected.messages[projected.messages.length - 1];
  assert.ok(!("reasoning_content" in last));
  assert.ok("_provider_continuation_state" in last);
  const prepared = assembly.prepare(projected);
  assert.ok(!("_provider_continuation_state" in prepared.canonical.messages.at(-1)));
  assert.ok("_provider_continuation_state" in prepared.execution.messages.at(-1));
  assert.deepEqual(prepared.wire.messages.at(-1).content, signed);

  messages.push({ role: "user", content: "continue" });
  const appended = await context.project({ messages, tools: [], observationStep: 0 });
  const next = assembly.prepare(appended);
  assert.equal(appended.manifest.canonical_prefix_relation.status, "exact_prefix");
  assert.equal(appended.manifest.provider_wire_prefix_relation.status, "exact_prefix");
  assert.equal(next.wire.messages.at(-2).content[0].signature, "signed-state");
});
