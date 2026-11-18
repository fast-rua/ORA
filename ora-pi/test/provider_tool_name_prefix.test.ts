// A tool call the provider returns under a name that is neither declared nor
// OpenAI-strict (a tool-call parse defect; a likely example is a DeepSeek DSML
// parameter header read as the function name) must not rewrite the cached
// wire prefix. Kept verbatim in the canonical history, it added a new
// <wire_tool_aliases> entry to system message 0 on the next request, and the
// lossless request assembler refused that request as `cache_prefix_rewritten`
// inside one cache epoch: a fatal fault for what the Runtime answers as an
// ordinary `tool_unavailable`. The real ReplayProvider parser and the real
// request assembler, no network.
//
//   node --test ora-pi/test/provider_tool_name_prefix.test.ts

import assert from "node:assert/strict";
import { test } from "node:test";

import { NativeRequestAssembler } from "../src/context/request_assembly.ts";
import { ReplayProvider } from "../src/provider/fake.ts";
import * as toolNames from "../src/provider/tool_names.ts";

const { toolAlias } = toolNames;

type Json = any;

const GARBLED = "approach_world\" string=\"false";

function tools(names: string[]): Json[] {
  return names.map((name) => ({
    type: "function", function: { name, description: name, parameters: { type: "object", properties: {} } },
  }));
}

/** The lossless retention's cache evidence inside one epoch (no transition, no image-window advance). */
function projected(messages: Json[], names: string[]): Json {
  return {
    messages, tools: tools(names),
    manifest: {
      context_retention: "action_step_lossless_v4", cache_epoch: 5,
      protected_prefix_sha256: "a".repeat(64), provider_image_window: { epoch: 0 },
    },
  };
}

function toolTurn(name: string): Json[][] {
  return [[
    { choices: [{ index: 0, delta: { role: "assistant", content: "", reasoning_content: "move_pose, world axes" } }] },
    { choices: [{ index: 0, delta: { tool_calls: [{
      index: 0, id: "chatcmpl-tool-1", type: "function",
      function: { name, arguments: "{\"position_m\": [4.649, -0.8, 0.79], \"position_mode\": \"absolute\"}" },
    }] }, finish_reason: "tool_calls" }] },
  ]];
}

/** One provider turn returning `returned`, its tool result, then the next request inside the same epoch. */
async function nextTurn(returned: string, declared: string[]): Promise<{ called: string; second: Json; first: Json }> {
  const provider = new ReplayProvider(toolTurn(returned));
  const assembler = new NativeRequestAssembler({
    model: "deepseek-flash", reasoningEffort: "low", maxOutputTokens: 16384, stream: true,
    prepareRequest: (request) => provider.prepareRequest(request),
  });
  const history: Json[] = [{ role: "system", content: "harness system" }, { role: "user", content: "open the drawer" }];
  const firstProjection = projected(history, declared);
  const first = assembler.prepare(firstProjection);
  const response = await provider.complete(first.execution, { modelCallId: "model-call-000037" });
  const assistant = response.message;
  const call = assistant.tool_calls[0];
  const result = {
    role: "tool", tool_call_id: call.id, name: call.function.name,
    content: JSON.stringify({ ok: false, error: { code: "tool_unavailable", message: `tool is disabled: ${call.function.name}` } }),
  };
  const secondProjection = projected([...history, assistant, result], declared);
  const second = assembler.prepare(secondProjection);
  return { called: call.function.name, second: { ...second, manifest: secondProjection.manifest }, first };
}

test("an undeclared non-strict tool name keeps the next request append-only", async () => {
  const { called, second, first } = await nextTurn(GARBLED, ["observe", "move_pose"]);
  assert.equal(called, toolAlias(GARBLED));
  assert.match(called, /^[A-Za-z][A-Za-z0-9_-]{0,63}$/);
  assert.equal(second.manifest.request_relation, "append_only");
  assert.deepEqual(second.manifest.provider_wire_prefix_relation,
    { status: "exact_prefix", common_prefix_units: 2, previous_units: 2, current_units: 4 });
  assert.deepEqual(second.wire.messages[0], first.wire.messages[0]);
  assert.ok(!JSON.stringify(second.wire).includes("<wire_tool_aliases>"));
});

test("declared, strict and already-carried names keep their canonical identity", async () => {
  const declared = ["observe", "move_pose", "vision.segment"];
  const cases: [string, string][] = [
    ["move_pose", "move_pose"],
    ["approach_world", "approach_world"],
    [toolAlias("vision.segment"), "vision.segment"],
    ["vision.segment", "vision.segment"],
  ];
  for (const [returned, expected] of cases) {
    const { called, second, first } = await nextTurn(returned, declared);
    assert.equal(called, expected, returned);
    assert.equal(second.manifest.request_relation, "append_only", returned);
    assert.deepEqual(second.wire.messages[0], first.wire.messages[0], returned);
  }
});

test("responseToolName only renames undeclared non-strict names", () => {
  const { responseToolName } = toolNames as Json;
  assert.equal(typeof responseToolName, "function");
  const aliases = new Map([["observe", "observe"], [toolAlias("vision.segment"), "vision.segment"]]);
  assert.equal(responseToolName("", aliases), "");
  assert.equal(responseToolName("observe", aliases), "observe");
  assert.equal(responseToolName("approach_world", aliases), "approach_world");
  assert.equal(responseToolName(toolAlias("vision.segment"), aliases), "vision.segment");
  assert.equal(responseToolName("vision.segment", aliases), "vision.segment");
  assert.equal(responseToolName(GARBLED, aliases), toolAlias(GARBLED));
});
