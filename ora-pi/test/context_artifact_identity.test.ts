// One Context Artifact for one content (src/context/memory/artifacts.ts).
//
// A code-mode script that acts and then observes returns, as tool images, frames that the
// next observation publishes. core.registerToolImages registers them first, with their
// native_tool_image storage ref; projecting that observation then stores the same bytes as a
// context_image (InlineMediaCache.admitImage). Both storage refs name one content-addressed
// file and differ only in the storage kind label, which is not identity drift; any other
// difference still is. ora/agent/memory/catalog.py follows the same rule
// (tests/agent_core/test_context_artifact_parity.py holds the two catalogs to it).
//
//   node --test ora-pi/test/context_artifact_identity.test.ts

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { ContextArtifactCatalog } from "../src/context/memory/artifacts.ts";
import { NativeContextPipeline } from "../src/context/native.ts";
import { STAGE_SCHEMAS } from "./support/stage_schemas.ts";
import { ArtifactStore, NO_DURABILITY } from "../src/evidence/artifacts.ts";

type Json = any;
const SESSION = "policy-session-1";

function session(): { root: string; artifacts: ArtifactStore; catalog: ContextArtifactCatalog } {
  const root = mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), "ora-artifact-identity-"));
  const artifacts = new ArtifactStore(root, { durability: NO_DURABILITY });
  return { root, artifacts, catalog: new ContextArtifactCatalog(root, SESSION, artifacts) };
}

/** core.registerToolImages for an image the catalog does not hold yet (core/context_wiring.ts). */
function registerToolImage(artifacts: ArtifactStore, catalog: ContextArtifactCatalog, data: Buffer, step: number): Json {
  const imageRef = artifacts.putBytes(data, { kind: "native_tool_image", suffix: ".png", mediaType: "image/png" });
  return catalog.registerStorageRef(imageRef, { kind: "context_image", roundIndex: null, actionStep: step });
}

function indexEntry(root: string, ref: string): Json {
  return JSON.parse(readFileSync(join(root, "native/context-artifacts/index.json"), "utf8")).artifacts[ref];
}

test("a tool image and an observation frame with the same bytes are one artifact, in either order", () => {
  const frame = Buffer.from("frame");
  const toolFirst = session();
  const first = registerToolImage(toolFirst.artifacts, toolFirst.catalog, frame, 0);
  const second = toolFirst.catalog.putBytes(frame, {
    kind: "context_image", suffix: ".png", mediaType: "image/png", roundIndex: 1, actionStep: 1,
  });
  assert.equal(second.ref, first.ref);
  const entry = indexEntry(toolFirst.root, first.ref);
  assert.equal(entry.storage_ref.kind, "native_tool_image", "the entry keeps its first storage ref");
  assert.deepEqual(entry.sources, [{ round: null, action_step: 0 }, { round: 1, action_step: 1 }]);

  const observationFirst = session();
  const frameRef = observationFirst.catalog.putBytes(frame, {
    kind: "context_image", suffix: ".png", mediaType: "image/png", roundIndex: 1, actionStep: 1,
  });
  assert.equal(registerToolImage(observationFirst.artifacts, observationFirst.catalog, frame, 2).ref, frameRef.ref);
  assert.equal(indexEntry(observationFirst.root, frameRef.ref).storage_ref.kind, "context_image");
});

test("same bytes under another artifact kind or media type are still identity drift", () => {
  const { artifacts, catalog } = session();
  const frame = Buffer.from("frame");
  registerToolImage(artifacts, catalog, frame, 0);
  const imageRef = artifacts.putBytes(frame, { kind: "native_tool_image", suffix: ".png", mediaType: "image/png" });
  assert.throws(() => catalog.registerStorageRef(imageRef, { kind: "context_tool_result", roundIndex: 1, actionStep: 1 }),
    /logical artifact identity drift/);
  assert.throws(() => catalog.putBytes(frame, {
    kind: "context_image", suffix: ".img", mediaType: "image/jpeg", roundIndex: 1, actionStep: 1,
  }), /logical artifact identity drift/);
});

/** The lossless retention of request_assembly.test.ts. */
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

function actual(sequence: number, frame: Buffer): Json {
  return {
    role: "user",
    content: [
      { type: "text", text: `{"current_actual": {"actual_sequence": ${sequence}}}` },
      { type: "text", text: `ACTION RESULT ACTUAL · sequence ${sequence} · left_wrist` },
      { type: "image_url", image_url: { url: `data:image/png;base64,${frame.toString("base64")}` } },
    ],
    _observation: { step: sequence, observation_id: `internal-observation-${sequence}`, keyframe: true },
    _causal_observation: { step: sequence, result_actual_sequence: sequence },
    _session_round: 1,
  };
}

test("projecting an observation whose frame a script already returned as a tool image", async () => {
  const { root, artifacts, catalog } = session();
  const pipeline = new NativeContextPipeline({
    profile: "native_causal_context_v1", artifacts, contextWindowTokens: 200_000, maxOutputTokens: 2048,
    retention: "action_step_lossless_v4", summaryPolicy: "rollout_checkpoint_v1",
    summaryProvider: { summarize: async () => { throw new Error("no compaction here"); } } as Json,
    contextManagement: MANAGEMENT, stageSchemas: STAGE_SCHEMAS, contextArtifacts: catalog,
    mutationTools: ["move_pose", "close_gripper", "open_gripper"],
  });
  const messages: Json[] = [
    { role: "system", content: "minimal harness", _protected_context: true, _context_source: "harness_system" },
    { role: "user", content: "exact episode task", _protected_context: true, _context_source: "episode_task" },
    actual(0, Buffer.from("frame before the script")),
  ];
  await pipeline.project({ messages, tools: [], observationStep: 0 });
  // The script acts, then observes: its last frames are the next observation's frames.
  const after = Buffer.from("frame after the script");
  const toolImage = registerToolImage(artifacts, catalog, after, 0);
  messages.push(actual(1, after));
  const projected = await pipeline.project({ messages, tools: [], observationStep: 1 });
  assert.ok(projected.messages.length > 0);
  const entry = indexEntry(root, toolImage.ref);
  assert.ok(entry.sources.some((source: Json) => source.round === 1), JSON.stringify(entry.sources));
});
