// FakeProvider parity with Python, ReplayProvider, and the TS-only call hooks
// (per-attempt request bytes, caller abort) plus the pi-ai adapter.

import assert from "node:assert/strict";
import { join } from "node:path";
import { test } from "node:test";
import { canonicalBytesPy, dumpsPy } from "../src/pyjson.ts";
import {
  FakeProvider, OpenAICompatibleProvider, ProviderAborted, ReplayProvider, sha256Hex, toolAlias, type Clock,
  SYSTEM_CLOCK,
} from "../src/provider/index.ts";
import { oraStreamSimple } from "../src/provider/pi_stream.ts";
import { canonicalText, describeError, normalizeTiming, readPy, runPython, scratchDir, writePy } from "./provider/harness.ts";

const dir = scratchDir("fake");

const RESPONSES = [
  { role: "assistant", content: "plain" },
  { message: { role: "assistant", content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "observe", arguments: "{}" } }] },
    usage: { input_tokens: 3, output_tokens: 1 } },
  { message: { role: "assistant", content: "x" }, finish_reason: "length", usage: null },
  { message: { role: "assistant", content: "" }, finish_reason: "invalid_tool_input",
    output_rejection: { code: "truncated_tool_call", world_state_changed: false } },
  { message: { role: "assistant", content: "" }, output_rejection: "not-a-mapping" },
  { message: "not an object" },
  { role: "assistant", content: "after close" },
];

test("FakeProvider matches Python FakeProvider", async () => {
  for (const closeAfter of [null, 6]) {
    const input = join(dir, `fake-in-${closeAfter}.json`);
    const output = join(dir, `fake-out-${closeAfter}.json`);
    writePy(input, { responses: RESPONSES, calls: 8, close_after: closeAfter });
    await runPython("fake_provider.py", [input, output]);
    const python = readPy(output).outcomes;
    const provider = new FakeProvider(RESPONSES);
    for (let index = 0; index < 8; index += 1) {
      if (closeAfter === index) provider.close();
      const request = { model: "m", messages: [{ role: "user", content: `call ${index}` }] };
      let got: any;
      try {
        got = { response: canonicalText(normalizeTiming(await provider.complete(request, {
          modelCallId: `model-call-${String(index).padStart(6, "0")}` }))) };
      } catch (error) {
        got = { error: describeError(error) };
      }
      assert.equal(canonicalText(got), canonicalText(python[index]), `call ${index}`);
    }
    assert.equal(canonicalText(provider.requests), python[8].requests);
  }
});

const KEY = "sk-hooks";
const OK_SSE = [
  { choices: [{ index: 0, delta: { reasoning_content: "think" } }] },
  { choices: [{ index: 0, delta: { content: "hi", tool_calls: [{ index: 0, id: "c", function: { name: toolAlias("robot.read_manual"), arguments: "{\"section\": \"Actions\", \"x\": 1.0}" } }] } }] },
  { choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15, prompt_cache_hit_tokens: 4 } },
];
const REQUEST = {
  model: "deepseek-flash", messages: [{ role: "user", content: "u" }], stream: true, max_tokens: 16384,
  tools: [{ type: "function", function: { name: "robot.read_manual", parameters: {} } }],
};

function sse(events: any[]): string {
  return events.map((event) => `data: ${dumpsPy(event)}\n\n`).join("") + "data: [DONE]\n\n";
}

test("every attempt's request bytes reach onRequestBytes; retries reuse identical bytes", async () => {
  let calls = 0;
  const fetchImpl = (async () => {
    calls += 1;
    if (calls === 1) return new Response("busy", { status: 503 });
    return new Response(sse(OK_SSE), { status: 200, headers: { "content-type": "text/event-stream" } });
  }) as typeof fetch;
  const sleeps: number[] = [];
  const clock: Clock = { ...SYSTEM_CLOCK, sleep: async (s) => { sleeps.push(s); } };
  const provider = new OpenAICompatibleProvider({
    baseUrl: "https://example.invalid/v1", apiKey: KEY, recoveryPolicy: "same_credential_deadline_v1", fetchImpl, clock,
  });
  const seen: Array<[number, string]> = [];
  const records: any[] = [];
  const response = await provider.complete(REQUEST, {
    modelCallId: "m-1", onRequestBytes: (bytes, sha) => seen.push([bytes.length, sha]),
    onAttemptFailure: (record) => records.push(record),
  });
  const expected = canonicalBytesPy(provider.prepareRequest(REQUEST));
  assert.deepEqual(seen, [[expected.length, sha256Hex(expected)], [expected.length, sha256Hex(expected)]]);
  assert.equal(records[0].request_sha256, sha256Hex(expected));
  assert.deepEqual(sleeps, [1]);
  assert.equal(response.attempts, 2);
  assert.equal(response.message.tool_calls[0].function.name, "robot.read_manual");
  // The replay provider reproduces the same dict from the recorded events.
  const replay = await new ReplayProvider([response.raw_events], { apiKey: KEY }).complete(REQUEST, { modelCallId: "m-1" });
  assert.equal(canonicalText({ ...normalizeTiming(replay), attempts: 2 }), canonicalText(normalizeTiming(response)));
});

test("a caller abort ends the in-flight attempt without retry", async () => {
  const fetchImpl = ((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
    init.signal!.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
  })) as unknown as typeof fetch;
  const provider = new OpenAICompatibleProvider({
    baseUrl: "https://example.invalid/v1", apiKey: KEY, recoveryPolicy: "same_credential_deadline_v1", fetchImpl,
  });
  const controller = new AbortController();
  const records: any[] = [];
  setTimeout(() => controller.abort(), 50);
  await assert.rejects(provider.complete(REQUEST, {
    modelCallId: "m-2", signal: controller.signal, onAttemptFailure: (record) => records.push(record),
  }), (error: any) => error instanceof ProviderAborted && error.providerFailure?.attempts === 1);
  assert.equal(records.length, 1);
  assert.equal(records[0].retry_scheduled, false);
});

test("pi-ai adapter emits thinking, text and tool call blocks with the ORA message attached", async () => {
  const replay = new ReplayProvider([OK_SSE], { apiKey: KEY });
  const streamFn = oraStreamSimple({
    provider: replay, buildRequest: () => REQUEST, nextModelCallId: () => "m-3",
  });
  const stream = streamFn({ id: "deepseek-flash", api: "openai-completions", provider: "ora" } as any, {} as any);
  const types: string[] = [];
  for await (const event of stream) types.push(event.type);
  const message: any = await stream.result();
  assert.deepEqual(types, ["start", "thinking_start", "thinking_delta", "thinking_end", "text_start", "text_delta",
    "text_end", "toolcall_start", "toolcall_delta", "toolcall_end", "done"]);
  assert.equal(message.stopReason, "toolUse");
  assert.equal(message.content[2].name, "robot.read_manual");
  assert.equal(message.content[2].rawArguments, "{\"section\": \"Actions\", \"x\": 1.0}");
  assert.deepEqual(message.content[2].arguments, { section: "Actions", x: 1 });
  assert.equal(message.ora._provider_continuation_state.native_payload.reasoning_content, "think");
  assert.deepEqual([message.usage.input, message.usage.cacheRead, message.usage.output], [6, 4, 5]);
});

test("the run budget caps the incident window (deadlineMonotonicMs)", async () => {
  const fetchImpl = ((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
    init.signal!.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
  })) as unknown as typeof fetch;
  const provider = new OpenAICompatibleProvider({
    baseUrl: "https://example.invalid/v1", apiKey: KEY, recoveryPolicy: "same_credential_deadline_v1", fetchImpl,
  });
  const records: any[] = [];
  const started = performance.now();
  await assert.rejects(provider.complete(REQUEST, {
    modelCallId: "m-4", deadlineMonotonicMs: performance.now() + 200, onAttemptFailure: (r) => records.push(r),
  }), (error: any) => error.pyType === "ProviderRecoveryDeadline");
  const elapsed = performance.now() - started;
  assert.ok(elapsed >= 190 && elapsed < 2000, `elapsed ${elapsed}`);
  assert.equal(records.length, 1);
  assert.equal(records[0].error_chain[0].type, "ProviderRecoveryDeadline");
  assert.equal(records[0].recovery_policy, "same_credential_deadline_v1");
});
