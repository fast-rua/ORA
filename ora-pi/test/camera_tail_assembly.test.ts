// Context v8's rolling camera tail and the request assembler
// (src/context/request_assembly.ts, src/context/stages/camera_delivery/delivery.ts): inside one
// cache epoch only the last (tail) message may change, recorded as
// `camera_tail_append`; any other rewrite is still refused, and so is a tail
// change after a request that ended without a tail.
//
//   node --test ora-pi/test/camera_tail_assembly.test.ts

import assert from "node:assert/strict";
import { test } from "node:test";

import { CAMERA_TAIL_HEADER } from "../src/context/stages/camera_delivery/delivery.ts";
import { NativeRequestAssembler } from "../src/context/request_assembly.ts";

type Json = any;

const PROTECTED = "a".repeat(64);
const image = (n: number) => ({ type: "image_url", image_url: { url: `data:image/png;base64,${"A".repeat(8)}${n}` } });
const tail = (n: number) => ({
  role: "user",
  content: [{ type: "text", text: `${CAMERA_TAIL_HEADER}latest 1 observation, oldest first` }, image(n)],
});
const observation = (step: number) => ({ role: "user", content: [{ type: "text", text: `observation ${step}` }] });
const call = (id: string) => ({ role: "assistant", content: "", tool_calls: [
  { id, type: "function", function: { name: "observe", arguments: "{}" } }] });
const result = (id: string) => ({ role: "tool", tool_call_id: id, name: "observe", content: "{}" });

function request(messages: Json[], tailImages: number): Json {
  return {
    messages, tools: [],
    manifest: {
      context_retention: "action_step_lossless_v4", cache_epoch: 0, protected_prefix_sha256: PROTECTED,
      camera_delivery: { schema: "ora.camera-delivery.v1", tail_images: tailImages },
    },
  };
}

function assembler(): NativeRequestAssembler {
  return new NativeRequestAssembler({ model: "m", reasoningEffort: "low", maxOutputTokens: 100, stream: false });
}

test("only the tail changes between requests of one epoch", () => {
  const assemble = assembler();
  const system = { role: "system", content: "harness" };
  const first = request([system, observation(0), tail(0)], 1);
  assemble.prepare(first);
  assert.equal(first.manifest.request_relation, "initial");
  const second = request([system, observation(0), call("c1"), result("c1"), observation(1), tail(1)], 1);
  assemble.prepare(second);
  assert.equal(second.manifest.request_relation, "camera_tail_append");
  assert.equal(second.manifest.canonical_prefix_relation.status, "rewritten");
  const rewritten = request([system, observation(9), call("c1"), result("c1"), observation(1), tail(1)], 1);
  assert.throws(() => assemble.prepare(rewritten), /cache_prefix_rewritten: canonical history changed/);
});

test("a tail change after a request without a tail is a rewrite", () => {
  const assemble = assembler();
  const system = { role: "system", content: "harness" };
  assemble.prepare(request([system, observation(0), { role: "user", content: "no tail" }], 0));
  assert.throws(() => assemble.prepare(request([system, observation(0), tail(0)], 1)), /cache_prefix_rewritten/);
});

// A malformed tool name the model emits mid-session makes the wire projection
// disclose an alias in the system prefix (provider/tool_names.ts): recorded,
// not refused, when the history itself only grew.
test("a wire alias disclosure alone is recorded as its own relation", async () => {
  const { prepareRequest } = await import("../src/provider/openai.ts");
  const assemble = new NativeRequestAssembler({
    model: "m", reasoningEffort: "low", maxOutputTokens: 100, stream: false,
    prepareRequest: (execution: Json) => prepareRequest(execution),
  });
  const system = { role: "system", content: "harness" };
  assemble.prepare(request([system, observation(0), tail(0)], 1));
  const odd = 'visions_segment_dummy" string="true';
  const second = request([system, observation(0),
    { role: "assistant", content: "", tool_calls: [{ id: "c1", type: "function", function: { name: odd, arguments: "{}" } }] },
    { role: "tool", tool_call_id: "c1", name: odd, content: "{}" }, tail(0)], 1);
  const prepared = assemble.prepare(second);
  assert.match(prepared.wire.messages[0].content, /<wire_tool_aliases>/);
  assert.equal(second.manifest.request_relation, "wire_tool_alias_disclosure");
});
