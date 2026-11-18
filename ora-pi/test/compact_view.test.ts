// A context pack's view of each request (src/context/compact_view.ts, context_pack.ts):
// the fold line grammar with Python's str()/len() semantics, the recent-turn window and
// tool-text cap, the image window and its labels, the static replacements by content hash,
// the overlaid tools, one downscale RPC per image, and the assembler's record of a
// rewritten history. The byte-for-byte check against the reference view is
// tests/agent_core/test_context_pack_view.py.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";

import {
  actionEffect, compactView, foldTurn, resultStatus, type CompactViewConfig,
} from "../src/context/compact_view.ts";
import { ContextPackView, packContextManagement } from "../src/context/context_pack.ts";
import { NativeRequestAssembler } from "../src/context/request_assembly.ts";
import { parsePy } from "../src/pyjson.ts";

type Json = any;

const CONFIG: CompactViewConfig = {
  schema: "ora.compact-view.v1", recent_turns: 1, tool_text_cap_chars: 10, image_window: 2, image_long_edge_px: 64,
  fold: { line_format: "turn_summary_v1", rationale_cap_chars: 12, arguments_cap_chars: 16 },
  texts: { earlier_steps_header: "EARLIER:", image_placeholder: "[omitted: {label}]", tool_text_omission: "\n…[{omitted} more]" },
};

const sha = (text: string) => createHash("sha256").update(text).digest("hex");
const image = (name: string) => ({ type: "image_url", image_url: { url: `data:image/png;base64,${name}` } });
const call = (id: string, name: string, args: string) => ({ id, type: "function", function: { name, arguments: args } });

test("a tool result's status reads like Python's str() of the decoded value", () => {
  assert.equal(resultStatus('{"error": {"code": "unreachable"}}'), "error unreachable");
  assert.equal(resultStatus('{"error": {"code": "", "type": "timeout"}}'), "error timeout");
  assert.equal(resultStatus('{"error": {}}'), "error");
  assert.equal(resultStatus('{"task_success": false, "status": "x"}'), "task_success False");
  assert.equal(resultStatus('{"status": null}'), "status None");
  assert.equal(resultStatus('{"success": true}'), "success True");
  assert.equal(resultStatus('{"value": 1.0}'), "ok");
  assert.equal(resultStatus('{"status": 1.0}'), "status 1.0");
  assert.equal(resultStatus("[1, 2]"), "returned a value");
  assert.equal(resultStatus("plain text"), "returned text");
  assert.equal(resultStatus(null), "returned text");
});

test("a fold line caps rationale and arguments by code points and keeps the last effect", () => {
  const turn = [
    { role: "assistant", content: "  机器人看到立方体在碗左边，所以先移动。 \n", tool_calls: [
      call("a", "move_to", '{"position_m": [0.1, 1.0, 1e-05], "note": "é"}'),
      call("b", "observe", "not json"),
      call("c", "observe", '{"views": ["front"]}'),
    ] },
    { role: "tool", tool_call_id: "a", content: '{"status": "submitted"}' },
    { role: "tool", tool_call_id: "b", content: '{"error": {"code": "bad_arguments"}}' },
    { role: "user", content: [{ type: "text", text: '{"last_action_effect": {"action_summary": "moved 1 mm"}}' }] },
    { role: "user", content: [{ type: "text", text: '{"last_action_effect": {"action_summary": "reached"}}' }, image("x")] },
  ];
  assert.equal(foldTurn(3, parsePy(JSON.stringify(turn)), CONFIG.fold),
    '3. 机器人看到立方体在碗左边… -> move_to({"position_m":[0…) -> status submitted; '
    + 'observe("not json") -> error bad_arguments; observe({"views":["front…) -> no result; effect: reached');
  assert.equal(foldTurn(1, [{ role: "assistant", content: null }], CONFIG.fold), "1. (no tool call)");
  assert.equal(foldTurn(2, [{ role: "assistant", content: "", tool_calls: [call("z", "observe", "{}")] }],
    { ...CONFIG.fold, arguments_cap_chars: null }), "2. observe({}) -> no result");
  // The JSON keeps Python's float spelling of the recorded arguments.
  assert.match(foldTurn(4, parsePy(JSON.stringify([{ role: "assistant", content: "x", tool_calls: [
    call("a", "move_to", '{"p": [1.0, 1e-05]}')] }])), { ...CONFIG.fold, arguments_cap_chars: null }), /\[1\.0,1e-05\]/);
  assert.equal(actionEffect({ role: "user", content: "not json" }), null);
  assert.equal(actionEffect({ role: "user", content: [image("x")] }), null);
});

function history(): Json[] {
  return [
    { role: "system", content: "SYSTEM" },
    { role: "user", content: "TASK" },
    { role: "user", content: [{ type: "text", text: "INITIAL front" }, { type: "text", text: "<recoverable_artifact>{}" }, image("i0")] },
    { role: "assistant", content: "Look first.", tool_calls: [call("c1", "observe", '{"views":["front"]}')] },
    { role: "tool", tool_call_id: "c1", content: '{"status": "observed"}' },
    { role: "user", content: [{ type: "text", text: "feedback c1" }, image("t1"), image("t2")] },
    { role: "assistant", content: "Now move.\n\n", tool_calls: [call("c2", "move_to", '{"p":[1]}')],
      _provider_continuation_state: { opaque: true } },
    { role: "tool", tool_call_id: "c2", content: "0123456789abcdefghij" },
    { role: "user", content: [{ type: "text", text: '{"last_action_effect": {"action_summary": "reached"}}' },
      { type: "text", text: "RESULT front" }, image("r1")] },
  ];
}

test("the view keeps the prefix and the recent turn, folds the rest and windows the images", () => {
  const view = compactView(history(), { ...CONFIG, image_window: 1 });
  assert.deepEqual(view.stats, { prefix_messages: 3, folded_turns: 1, kept_turns: 1, kept_images: 1, omitted_images: 1 });
  const [system, task, initial, folded, assistant, tool, result] = view.messages;
  assert.equal(view.messages.length, 7);
  assert.deepEqual([system, task], [{ role: "system", content: "SYSTEM" }, { role: "user", content: "TASK" }]);
  // The placeholder's label is the text before the image, never the artifact reference.
  assert.deepEqual(initial.content[2], { type: "text", text: "[omitted: INITIAL front]" });
  // The folded turn's images go with it; its line keeps the call and its result.
  assert.equal(folded.content, 'EARLIER:\n1. Look first. -> observe({"views":["front…) -> status observed');
  // Stripped text, no continuation state: the history carries no reasoning.
  assert.deepEqual(assistant, { role: "assistant", content: "Now move.", tool_calls: [call("c2", "move_to", '{"p":[1]}')] });
  assert.equal(tool.content, "0123456789\n…[10 more]");
  assert.deepEqual(result.content[2], image("r1"));
  assert.deepEqual(view.keptImages, [{ message: 6, part: 2, url: "data:image/png;base64,r1" }]);
});

test("the image window counts every image of the request, prefix included", () => {
  const messages = history().slice(0, 6);
  const view = compactView(messages, { ...CONFIG, recent_turns: 1 });
  assert.deepEqual(view.keptImages.map((slot) => slot.url.split(",")[1]), ["t1", "t2"]);
  assert.deepEqual(compactView(messages, { ...CONFIG, image_window: null }).keptImages.map((slot) => slot.url.split(",")[1]),
    ["i0", "t1", "t2"]);
  const everything = compactView(history(), { ...CONFIG, recent_turns: 0 });
  assert.equal(everything.stats.folded_turns, 2);
  assert.equal(everything.messages.at(-1)!.content,
    'EARLIER:\n1. Look first. -> observe({"views":["front…) -> status observed\n'
    + '2. Now move. -> move_to({"p":[1]}) -> returned text; effect: reached');
  assert.throws(() => compactView(history(), { ...CONFIG, fold: { ...CONFIG.fold, latest_result_tools: ["observe"] } }),
    /reserved/);
});

test("the pack replaces static messages by hash, overlays the tools and downscales each image once", async () => {
  const requests: Json[] = [];
  const rpc = {
    async request(method: string, params: Json): Promise<Json> {
      requests.push([method, params]);
      if (method === "context_pack.delivery") {
        return {
          messages: [
            { role: "system", content_sha256: sha("SYSTEM"), content: "COMPACT SYSTEM" },
            { role: "user", content_sha256: sha("PROJECT"), content: null },
            { role: "user", content_sha256: sha("TASK"), content: "bare task" },
          ],
          tools: [{ type: "function", function: { name: "observe", description: "Compact." } }],
        };
      }
      return { images: params.images.map((url: string) => `${url}-small`) };
    },
  };
  const binding = { schema: "ora.context-pack-binding.v1", name: "unit", profile: { compact_view: CONFIG } };
  const pack = new ContextPackView(binding, rpc);
  const messages = history();
  messages.splice(1, 0, { role: "user", content: "PROJECT" });
  const manifest: Json = { context_retention: "action_step_lossless_v4" };
  const first = await pack.apply({ messages, tools: [{ type: "function", function: { name: "observe" } }], manifest });
  assert.equal(first.manifest, manifest);
  assert.deepEqual(first.messages.slice(0, 2), [{ role: "system", content: "COMPACT SYSTEM" }, { role: "user", content: "bare task" }]);
  assert.deepEqual(first.tools, [{ type: "function", function: { name: "observe", description: "Compact." } }]);
  assert.deepEqual(manifest.context_pack, {
    schema: "ora.context-pack-view.v1", name: "unit", replaced_messages: 2, dropped_messages: 1,
    compact_view: { prefix_messages: 3, folded_turns: 1, kept_turns: 1, kept_images: 2, omitted_images: 0, resized_images: 2 },
    tools_sha256: manifest.context_pack.tools_sha256, messages: 7,
  });
  assert.ok(first.messages.flatMap((message: Json) => Array.isArray(message.content) ? message.content : [])
    .filter((part: Json) => part.type === "image_url").every((part: Json) => part.image_url.url.endsWith("-small")));
  await pack.apply({ messages, tools: [], manifest: {} });
  assert.deepEqual(requests.map(([method]) => method), ["context_pack.delivery", "context_pack.images"]);
  const stranger = new ContextPackView(binding, rpc);
  await assert.rejects(stranger.apply({ messages: history().slice(1), tools: [], manifest: {} }), /occurs 0 times/);
});

test("the assembler records a pack's rewritten history instead of refusing it", () => {
  const manifest = (epoch = 0): Json => ({
    context_retention: "action_step_lossless_v4", cache_epoch: epoch, protected_prefix_sha256: "a".repeat(64),
    provider_image_window: { epoch: 0 },
  });
  const request = (messages: Json[]) => ({ messages, tools: [], manifest: manifest() });
  const first = [{ role: "system", content: "s" }, { role: "user", content: "one" }];
  const rewritten = [{ role: "system", content: "s" }, { role: "user", content: "EARLIER:\n1. ..." }];
  const options = { model: "m", reasoningEffort: "provider_default", maxOutputTokens: 16, stream: false };
  const plain = new NativeRequestAssembler(options);
  plain.prepare(request(first));
  assert.throws(() => plain.prepare(request(rewritten)), /cache_prefix_rewritten/);
  const packed = new NativeRequestAssembler({ ...options, contextPackView: true });
  packed.prepare(request(first));
  const second = request(rewritten);
  packed.prepare(second);
  assert.equal(second.manifest.request_relation, "context_pack_view");
  const third = request([...rewritten, { role: "user", content: "more" }]);
  packed.prepare(third);
  assert.equal(third.manifest.request_relation, "append_only");
});

test("a pack's context_management switches reach the pipeline only when the profile states them", () => {
  assert.deepEqual(packContextManagement(null), {});
  assert.deepEqual(packContextManagement({ profile: {} }), {});
  assert.deepEqual(packContextManagement({ profile: { context_management: { rollout_compaction: false } } }),
    { pressureExternalization: true, rolloutCompaction: false });
  assert.deepEqual(packContextManagement({ profile: { context_management: {
    pressure_externalization: false, rollout_compaction: false } } }),
  { pressureExternalization: false, rolloutCompaction: false });
});
