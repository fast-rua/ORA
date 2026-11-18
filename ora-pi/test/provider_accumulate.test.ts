// Differential test: response accumulation, TypeScript vs Python.
//
// 1. Sealed runs (read-only, opt-in: ORA_TEST_SEALED_LIBERO / _ROBOTWIN /
//    _ROBODOJO, see tests/support/agent_core/sealed.py): rebuild every execution
//    request, project it and compare the wire sha256 with the sealed artifact;
//    replay every recorded `native_provider_wire_response` through the TS parser
//    and compare the response dict with Python's and the assistant message with
//    the sealed one. Unset, each is skipped with the reason.
// 2. Crafted edge cases through both complete() implementations.

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { canonicalBytesPy, dumpsPy, parsePy } from "../src/pyjson.ts";
import {
  OpenAICompatibleProvider, ReplayProvider, canonicalAssistantSha256, sha256Hex, toolAlias,
} from "../src/provider/index.ts";
import { accumulateChatCompletion } from "../src/provider/openai.ts";
import {
  canonicalText, describeError, normalizeTiming, readPy, runPython, runPythonArgs, scratchDir, sealedCache, writePy,
} from "./provider/harness.ts";

const dir = scratchDir("accumulate");
const SEALED: Record<string, string> = {
  libero: "ORA_TEST_SEALED_LIBERO", robotwin: "ORA_TEST_SEALED_ROBOTWIN", robodojo: "ORA_TEST_SEALED_ROBODOJO",
};

/** The sealed run and its extracted request archive, or why the check is skipped. */
async function sealedRun(env: string): Promise<{ run?: string; archive?: string; skip?: string }> {
  if (!process.env[SEALED[env]]) {
    return { skip: `${SEALED[env]} is not set (the sealed ${env} run lives outside the repository)` };
  }
  return parsePy(await runPythonArgs(["-m", "tests.support.agent_core.sealed", "archive", env, sealedCache()]));
}

const sealedRuns = Object.fromEntries(await Promise.all(
  Object.keys(SEALED).map(async (env) => [env, await sealedRun(env)] as const)));

/** A fetch that answers with recorded events as SSE (or one JSON body). */
export function replayFetch(events: any[], streaming: boolean): typeof fetch {
  return (async () => {
    if (!streaming) {
      return new Response(dumpsPy(events[0], { ensureAscii: false }), {
        status: 200, headers: { "content-type": "application/json" },
      });
    }
    // ensure_ascii: a raw U+2028 would split the line under httpx/LineDecoder rules.
    const text = events.map((event) => `data: ${dumpsPy(event, { ensureAscii: true })}\n\n`).join("")
      + "data: [DONE]\n\n";
    return new Response(text, { status: 200, headers: { "content-type": "text/event-stream" } });
  }) as typeof fetch;
}

/** TS twin of common.python_complete: complete() over recorded events. */
export async function tsComplete(request: any, events: any[], options: {
  apiKey?: string; reasoningField?: string; modelCallId?: string; recoveryPolicy?: string;
} = {}): Promise<{ response?: string; error?: any; records: string[] }> {
  const records: any[] = [];
  const provider = new OpenAICompatibleProvider({
    baseUrl: "https://replay.invalid/v1", apiKey: options.apiKey ?? "sk-replay-key",
    recoveryPolicy: options.recoveryPolicy ?? "same_credential_deadline_v1",
    reasoningField: options.reasoningField ?? "reasoning_content",
    fetchImpl: replayFetch(events, request.stream !== false),
  });
  try {
    const response = await provider.complete(request, {
      modelCallId: options.modelCallId ?? "model-call-000000",
      onAttemptFailure: (record) => records.push(record),
    });
    return { response: canonicalText(normalizeTiming(response)), records: records.map((r) => canonicalText(normalizeTiming(r))) };
  } catch (error) {
    return { error: describeError(error), records: records.map((r) => canonicalText(normalizeTiming(r))) };
  }
}

for (const [env, sealed] of Object.entries(sealedRuns)) test(`sealed ${env} run: wire bytes and replayed responses match Python and the sealed evidence`, {
  skip: sealed.skip ?? false,
}, async () => {
  const RUN = sealed.run!;
  const EXTRACT = sealed.archive!;
  const load = (ref: string): Buffer => {
    const direct = join(RUN, ref);
    return readFileSync(existsSync(direct) ? direct : join(EXTRACT, ref));
  };
  const out = join(dir, `sealed-${env}.json`);
  await runPython("sealed_run.py", [RUN, EXTRACT, out]);
  const python = readPy(out);
  const events = readFileSync(join(RUN, "agent-events.jsonl"), "utf8").split("\n").filter(Boolean).map(parsePy);
  const responses = new Map(events.filter((e: any) => e.type === "provider_response").map((e: any) => [e.model_call_id, e]));
  // The k-th assistant message with one canonical sha takes the k-th response with it.
  const states = new Map<string, any[]>();
  for (const event of responses.values()) {
    const state = (event as any).payload.message._provider_continuation_state;
    if (state) states.set(state.canonical_message_sha256, [...(states.get(state.canonical_message_sha256) ?? []), state]);
  }
  let wireMatches = 0;
  let responseMatches = 0;
  let sealedMessages = 0;
  for (const call of python.calls) {
    const execution = parsePy(load(call.canonical_ref).toString("utf8"));
    const seen = new Map<string, number>();
    for (const message of execution.messages) {
      if (message.role !== "assistant") continue;
      const key = canonicalAssistantSha256(message);
      const occurrence = (seen.get(key) ?? -1) + 1;
      seen.set(key, occurrence);
      const state = states.get(key)?.[occurrence];
      if (state) message._provider_continuation_state = state;
    }
    const recorded = parsePy(load(call.response_ref).toString("utf8"));
    const replay = new ReplayProvider([recorded], { reasoningField: python.reasoning_field, apiKey: "sk-replay-key" });
    let wireSha = "";
    const response = await replay.complete(execution, {
      modelCallId: call.model_call_id, onRequestBytes: (_bytes, sha) => (wireSha = sha),
    });
    assert.equal(wireSha, call.wire_sha256_sealed, `${call.model_call_id} wire sha vs sealed artifact`);
    assert.equal(wireSha, call.wire_sha256_python, `${call.model_call_id} wire sha vs Python`);
    wireMatches += 1;
    assert.equal(canonicalText(normalizeTiming(response)), call.python.response, `${call.model_call_id} response`);
    responseMatches += 1;
    const sealed = (responses.get(call.model_call_id) as any).payload;
    assert.equal(canonicalText(response.message), canonicalText(sealed.message), `${call.model_call_id} sealed message`);
    assert.equal(canonicalText(response.usage), canonicalText(sealed.usage));
    assert.equal(response.finish_reason, sealed.finish_reason);
    sealedMessages += 1;
    // The full fetch/SSE path yields the same dict as the replay path.
    const viaSse = await tsComplete(execution, recorded, {
      reasoningField: python.reasoning_field, modelCallId: call.model_call_id,
    });
    assert.equal(viaSse.response, call.python.response, `${call.model_call_id} via SSE transport`);
  }
  assert.equal(python.calls.length, events.filter((e: any) => e.type === "provider_request").length);
  assert.ok(python.calls.length > 0);
  console.log(JSON.stringify({ calls: python.calls.length, wireMatches, responseMatches, sealedMessages }));
});

// -- crafted edge cases ------------------------------------------------------------------

const KEY = "sk-edge-0123456789abcdef";
const MANUAL = toolAlias("robot.read_manual");
const SUBMIT = toolAlias("robot.submit_action");

function request(extra: any = {}): any {
  return {
    model: "deepseek-flash",
    messages: [{ role: "system", content: "s" }, { role: "user", content: "u" }],
    tools: ["robot.read_manual", "robot.submit_action", "observe"].map((name) => ({
      type: "function", function: { name, description: name, parameters: { type: "object", properties: {} } },
    })),
    tool_choice: "auto", max_tokens: 16384, stream: true, reasoning_effort: "low",
    stream_options: { include_usage: true }, ...extra,
  };
}

const chunk = (delta: any, finish: any = null, extra: any = {}) => ({
  id: "chatcmpl-1", object: "chat.completion.chunk", created: 1791924527, model: "deepseek-flash",
  choices: [{ index: 0, delta, finish_reason: finish, logprobs: null }], ...extra,
});
const usageChunk = (usage: any) => ({ id: "chatcmpl-1", object: "chat.completion.chunk", choices: [], usage });
const USAGE = { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120, prompt_cache_hit_tokens: 64, prompt_cache_miss_tokens: 36 };

function crafted(): any[] {
  const cases: any[] = [];
  const add = (id: string, events: any[], extra: any = {}) => cases.push({ id, request: request(extra.request), raw_events: events, ...extra.options });
  add("normal_tool_turn", [
    chunk({ role: "assistant", content: "" }), chunk({ reasoning_content: "Let me " }), chunk({ reasoning_content: "think." }),
    chunk({ content: "Moving." }),
    chunk({ tool_calls: [{ index: 0, id: "call_a", type: "function", function: { name: MANUAL, arguments: "" } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: "{\"section\": " } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: "\"Actions\"}" } }] }),
    chunk({}, "tool_calls"), usageChunk(USAGE),
  ]);
  add("length_with_tool_calls", [
    chunk({ reasoning_content: "r" }),
    chunk({ tool_calls: [{ index: 0, id: "call_a", function: { name: SUBMIT, arguments: "{\"arm\": \"left\", \"é😀\": [1.0, " } }] }),
    chunk({ tool_calls: [{ index: 1, function: { name: "observe", arguments: "{}" } }] }),
    chunk({}, "length"), usageChunk({ prompt_tokens: 5, completion_tokens: 16384, total_tokens: 16389 }),
  ]);
  add("length_without_tools", [chunk({ reasoning_content: "long" }), chunk({ content: "partial" }, "length")]);
  // A complete reply whose tool arguments do not decode to an object discards its tool turn (2026-11-07: a
  // Flash-Next `bash` call with "" arguments, replayed, made SGLang refuse every later request with HTTP 400).
  add("empty_tool_arguments", [
    chunk({ reasoning_content: "r" }),
    chunk({ tool_calls: [{ index: 0, id: "call_e", type: "function", function: { name: "bash", arguments: "" } }] }),
    chunk({}, "tool_calls"), usageChunk(USAGE),
  ]);
  add("invalid_tool_arguments_in_batch", [
    chunk({ tool_calls: [{ index: 0, id: "call_ok", function: { name: "observe", arguments: "{}" } }] }),
    chunk({ tool_calls: [{ index: 1, id: "call_bad", function: { name: SUBMIT, arguments: "{\"arm\": " } }] }),
    chunk({}, "tool_calls"),
  ]);
  add("array_tool_arguments", [
    chunk({ tool_calls: [{ index: 0, id: "call_arr", function: { name: "observe", arguments: "[1]" } }] }), chunk({}, "stop"),
  ]);
  add("missing_finish_reason", [chunk({ content: "hello" }), usageChunk(USAGE)]);
  add("no_events", []);
  add("multiple_choices", [{ choices: [{ index: 0, delta: { content: "a" } }, { index: 1, delta: { content: "b" } }] }]);
  add("choice_not_object", [{ choices: ["x"] }]);
  add("content_after_finish", [chunk({ content: "a" }, "stop"), chunk({ content: "b" })]);
  add("usage_after_finish_ok", [chunk({ content: "a" }, "stop"), usageChunk(USAGE), { choices: [] }]);
  add("credential_echo_content", [chunk({ content: `my key is ${KEY}` }, "stop")], { options: { api_key: KEY } });
  add("credential_echo_in_key", [chunk({ content: "x" }, "stop", { [KEY]: 1 })], { options: { api_key: KEY } });
  add("index_variants", [
    chunk({ tool_calls: [{ function: { name: "observe", arguments: "{}" } }] }),
    chunk({ tool_calls: [{ index: true, id: "", function: { name: SUBMIT, arguments: "{\"a\": 1}" } }] }),
    chunk({ tool_calls: [{ index: 1.0, function: { name: MANUAL } }] }),
    chunk({ tool_calls: [{ index: -1, id: "neg", function: { name: "obs", arguments: "{" } }] }),
    chunk({ tool_calls: [{ index: 9007199254740993, function: { name: "big", arguments: "[]" } }] }),
    chunk({ tool_calls: ["skip", { index: 7, function: "nope" }, { index: 8, function: { name: 5, arguments: null } }] }),
    chunk({}, "tool_calls"),
  ]);
  // A parameter header read as the function name takes its strict alias.
  add("undeclared_non_strict_name", [
    chunk({ reasoning_content: "r" }),
    chunk({ tool_calls: [{ index: 0, id: "c", function: { name: "approach_world\" string=\"false", arguments: "{}" } }] }),
    chunk({ tool_calls: [{ index: 1, id: "d", function: { name: "robot.submit_action", arguments: "{}" } }] }),
    chunk({}, "tool_calls"),
  ]);
  add("empty_reasoning_string", [chunk({ reasoning_content: "" }), chunk({ content: "", tool_calls: [{ index: 0, id: "c", function: { name: "observe", arguments: "{}" } }] }), chunk({}, "tool_calls")]);
  add("reasoning_field_reasoning", [chunk({ reasoning: "vllm thinks", reasoning_content: "ignored" }), chunk({ content: "ok" }, "stop")], {
    options: { reasoning_field: "reasoning" },
    request: { model: "deepseek-ai/DeepSeek-V4.1-Flash" },
  });
  add("delta_not_object", [chunk("text"), chunk(null), chunk({ content: 5 }), chunk({ content: "x" }, "stop")]);
  add("finish_not_string_then_string", [chunk({ content: "a" }, 1), chunk({ content: "b" }, "stop")]);
  add("usage_variants", [
    chunk({ content: "a" }, null, { usage: null }), chunk({ content: "b" }, null, { usage: { prompt_tokens: 3 } }),
    chunk({}, "stop", { usage: { prompt_tokens: 9, prompt_tokens_details: { cached_tokens: 4 }, completion_tokens_details: { reasoning_tokens: 2.0 } } }),
    usageChunk({ bogus: 1 }),
  ]);
  add("floats_and_unicode", [chunk({ content: "数值 1.0 ✓ line" }, null, { x: 1.0, y: -0.0, z: 1e-7 }), chunk({}, "stop")]);
  // Non-streaming replies (quickstart path): one chat.completion object.
  const smoke = (n: number, name: string, args: string) => ({
    id: `smoke-${n}`, object: "chat.completion", choices: [{ index: 0, finish_reason: "tool_calls", message: {
      role: "assistant", content: null, tool_calls: [{ id: `call-${n}`, type: "function", function: { name, arguments: args } }],
    } }], usage: { prompt_tokens: 500, completion_tokens: 50, total_tokens: 550 },
  });
  const nonstream = { stream: false, stream_options: undefined, reasoning_effort: undefined, model: "smoke-model", max_tokens: 2048 };
  add("nonstream_smoke_manual", [smoke(1, MANUAL, "{\"section\": \"Actions\"}")], { request: nonstream });
  add("nonstream_smoke_submit", [smoke(2, SUBMIT, "{\"arm\": \"left\", \"pose\": {\"position_m\": [0, 0, 1.02], \"orientation_xyzw\": [0, 0, 0, 1]}}")], { request: nonstream });
  add("nonstream_reasoning", [{ choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: "done", reasoning_content: "because" } }] }], { request: nonstream });
  add("nonstream_no_choices", [{ id: "x", object: "chat.completion", choices: [] }], { request: nonstream });
  for (const item of cases) {
    // JSON has no undefined: drop keys the request helper set to undefined.
    item.request = parsePy(dumpsPy(item.request));
  }
  return cases;
}

test("tool arguments that do not decode to an object discard the whole tool turn", () => {
  const options = { streaming: true, reasoningField: "reasoning_content", apiKey: null, modelCallId: "model-call-000001",
    toolNamesByAlias: new Map<string, string>(), model: "qwen3.8-flash-next" };
  const empty = accumulateChatCompletion([
    chunk({ reasoning_content: "r" }),
    chunk({ tool_calls: [{ index: 0, id: "call_e", type: "function", function: { name: "bash", arguments: "" } }] }),
    chunk({}, "tool_calls"),
  ], options);
  assert.equal(empty.finishReason, "invalid_tool_input");
  assert.equal(empty.message.tool_calls, undefined);
  assert.equal(empty.message.reasoning_content, undefined);
  assert.equal(empty.outputRejection?.code, "invalid_tool_input_json");
  assert.equal(empty.outputRejection?.policy_outcome, "continue_same_observation");
  assert.deepEqual(empty.outputRejection?.rejected_tool_blocks.map((b: any) => [b.index, b.tool_name, b.reason, b.fragment_chars]),
    [[0, "bash", "invalid_json", 0]]);
  const mixed = accumulateChatCompletion([
    chunk({ tool_calls: [{ index: 0, id: "call_ok", function: { name: "observe", arguments: "{}" } }] }),
    chunk({ tool_calls: [{ index: 1, id: "call_arr", function: { name: "observe", arguments: "[1]" } }] }),
    chunk({}, "tool_calls"),
  ], options);
  assert.equal(mixed.message.tool_calls, undefined);
  assert.equal(mixed.outputRejection?.discarded_tool_block_count, 2);
  assert.deepEqual(mixed.outputRejection?.rejected_tool_blocks.map((b: any) => [b.index, b.reason]), [[1, "not_an_object"]]);
  const valid = accumulateChatCompletion([
    chunk({ tool_calls: [{ index: 0, id: "call_ok", function: { name: "observe", arguments: "{\"camera\": \"head\"}" } }] }),
    chunk({}, "tool_calls"),
  ], options);
  assert.equal(valid.outputRejection, null);
  assert.equal(valid.message.tool_calls.length, 1);
});

test("crafted edge cases: complete() parse phase matches Python", async () => {
  const cases = crafted();
  const input = join(dir, "crafted-in.json");
  const output = join(dir, "crafted-out.json");
  writePy(input, { cases });
  await runPython("accumulate.py", [input, output]);
  const results = readPy(output).results;
  const mismatches: string[] = [];
  for (const [index, item] of cases.entries()) {
    const python = results[index];
    const ts = await tsComplete(item.request, item.raw_events, {
      apiKey: item.api_key, reasoningField: item.reasoning_field,
    });
    if ("response" in python) {
      if (ts.response !== python.response) mismatches.push(`${item.id}: response\n py ${python.response}\n ts ${ts.response ?? canonicalText(ts.error)}`);
    } else if (canonicalText(ts.error ?? null) !== canonicalText(python.error)) {
      mismatches.push(`${item.id}: error\n py ${canonicalText(python.error)}\n ts ${canonicalText(ts.error ?? ts.response)}`);
    }
    if (canonicalText(ts.records) !== canonicalText(python.records)) {
      mismatches.push(`${item.id}: records\n py ${python.records}\n ts ${ts.records}`);
    }
  }
  assert.deepEqual(mismatches, []);
  console.log(JSON.stringify({ crafted: cases.length, python_errors: results.filter((r: any) => "error" in r).length }));
});

export { sha256Hex, canonicalBytesPy };
