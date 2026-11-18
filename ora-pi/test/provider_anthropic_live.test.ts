// Differential test: AnthropicMessagesProvider.complete(), TS vs Python, live.
// Both clients run the same scripted scenarios against one local fake server
// (streams with signed thinking and tool use, credential-ring rotation on
// 401/429, the full-lap wait, stream breaks with a credential switch, the
// single message_stop-before-block-stop replay, overloaded error events,
// credential echo, malformed tool JSON, output-limit tool turns, non-streaming
// replies, the deadline policy); final results, attempt-failure records,
// backoff sleeps (value and int/float), rotations and the bytes and headers on
// the wire are compared.
//
// The incident windows (6 s, 30 s) are wall-clock budgets. Python's deadline
// attempt runs in a worker process that the helper starts before the window
// opens (common.PrestartedWorkers), so a window covers the attempt on both
// sides and not Python's interpreter start-up, which on a loaded host outlasted
// a window; a worker started inside a window fails the scenario.

import assert from "node:assert/strict";
import { join } from "node:path";
import { after, before, test } from "node:test";
import { dumpsPy, parsePy } from "../src/pyjson.ts";
import { AnthropicMessagesProvider, SYSTEM_CLOCK, toolAlias, type Clock } from "../src/provider/index.ts";
import { FakeServer, type Step } from "./provider/fake_server.ts";
import { canonicalText, describeError, normalizeTiming, readPy, runPython, scratchDir, writePy } from "./provider/harness.ts";

const dir = scratchDir("anthropic-live");
const KEY = "sk-ant-fake-provider-key-0123456789";
const STANDBY = ["sk-ant-standby-key-abcdefghijkl"];
const MANUAL = toolAlias("robot.read_manual");

const compact = (value: unknown) => dumpsPy(value, { ensureAscii: false, separators: [",", ":"] });
const ev = (event: any) => `event: ${event.type}\ndata: ${compact(event)}\n\n`;
const START = ev({ type: "message_start", message: { id: "msg_live_01", type: "message", role: "assistant", model: "claude-test",
  content: [], stop_reason: null, usage: { input_tokens: 812, cache_read_input_tokens: 20480, cache_creation_input_tokens: 0, output_tokens: 1 } } });
const THINK = [
  ev({ type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "", signature: "" } }),
  ev({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Read the manual " } }),
  ev({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "first." } }),
  ev({ type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "EqQBCkYIBxgCKkAsig" } }),
  ev({ type: "content_block_stop", index: 0 }),
];
const TEXT = [
  ev({ type: "content_block_start", index: 1, content_block: { type: "text", text: "" } }),
  ev({ type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "Reading the manual." } }),
  ev({ type: "content_block_stop", index: 1 }),
];
const TOOL = (json: string, index = 2) => [
  ev({ type: "content_block_start", index, content_block: { type: "tool_use", id: "toolu_live_0", name: MANUAL, input: {} } }),
  ev({ type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: json.slice(0, 5) } }),
  ev({ type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json: json.slice(5) } }),
  ev({ type: "content_block_stop", index }),
];
const END = (stop = "tool_use") => [
  ev({ type: "message_delta", delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: 214 } }),
  ev({ type: "message_stop" }),
];
const OK_EVENTS = [START, ": keep-alive\n\n", ev({ type: "ping" }), ...THINK, ...TEXT, ...TOOL("{\"section\": \"Actions\"}"), ...END()];
const OK: Step = { sse: OK_EVENTS, end: "close" };
const HEAD = [START, ...THINK];
const NO_BLOCK_STOP: Step = { sse: [START, ...TEXT.slice(0, 2), ...END("end_turn")], end: "close" };
// message_stop while a block is still open: the one safe exact replay.
const STOP_OPEN_BLOCK: Step = { sse: [START, ...TEXT.slice(0, 2), ev({ type: "message_stop" })], end: "close" };
const nonStream = JSON.stringify({ id: "msg_ns", type: "message", role: "assistant", model: "claude-test", stop_reason: "tool_use",
  content: [{ type: "text", text: "ok" }, { type: "tool_use", id: "toolu_ns", name: MANUAL, input: { section: "Actions" } }],
  usage: { input_tokens: 40, output_tokens: 9 } });

const TOOLS = ["robot.read_manual", "robot.submit_action", "observe"].map((name) => ({
  type: "function", function: { name, description: `${name} tool`, parameters: { type: "object", properties: { section: { type: "string" } } } },
}));
const REQUEST = {
  model: "claude-test", messages: [{ role: "system", content: "You control one robot." }, { role: "user", content: "Pick the soup." }],
  tools: TOOLS, tool_choice: "auto", max_tokens: 16384, stream: true, reasoning_effort: "low",
};
const NONSTREAM_REQUEST = { ...REQUEST, stream: false };

interface Scenario { name: string; policy: string; steps: Step[]; request?: any; provider?: any }
const D = "same_credential_deadline_v1";
const L = "legacy_v1";
const SCENARIOS: Scenario[] = [
  { name: "ok_stream", policy: D, steps: [OK] },
  { name: "legacy_ok_stream", policy: L, steps: [OK] },
  { name: "legacy_bearer_affinity", policy: L, steps: [OK], provider: { auth_scheme: "bearer", session_affinity_id: "ora:run-7:native" } },
  { name: "deadline_bearer_affinity", policy: D, steps: [OK], provider: { auth_scheme: "bearer", session_affinity_id: "ora:run-7:native" } },
  { name: "legacy_429_one_key", policy: L, steps: [{ status: 429, body: "{\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\"}}" }, OK] },
  { name: "legacy_429_rotate", policy: L, steps: [{ status: 429, body: `limit for ${KEY}` }, OK], provider: { two_keys: true } },
  { name: "legacy_401_rotate", policy: L, steps: [{ status: 401, body: "invalid x-api-key" }, OK], provider: { two_keys: true } },
  { name: "legacy_full_lap", policy: L, steps: [{ status: 429 }, { status: 429 }, { status: 429 }, OK], provider: { two_keys: true } },
  { name: "legacy_401_one_key", policy: L, steps: [{ status: 401, body: "invalid x-api-key" }] },
  { name: "legacy_5xx", policy: L, steps: [{ status: 529, body: "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\"}}" }] },
  { name: "legacy_break_switch", policy: L, steps: [{ sse: HEAD, end: "destroy" }, OK], provider: { two_keys: true } },
  { name: "legacy_eof_before_stop", policy: L, steps: [{ sse: HEAD, end: "close" }, OK] },
  { name: "legacy_done_marker", policy: L, steps: [{ sse: [...HEAD, "data: [DONE]\n\n"], end: "close" }, OK] },
  { name: "legacy_stop_before_block_stop", policy: L, steps: [NO_BLOCK_STOP, OK] },
  { name: "legacy_stop_before_block_stop_twice", policy: L, steps: [NO_BLOCK_STOP, NO_BLOCK_STOP, OK] },
  { name: "legacy_replay_once", policy: L, steps: [STOP_OPEN_BLOCK, OK] },
  { name: "legacy_replay_only_once", policy: L, steps: [STOP_OPEN_BLOCK, STOP_OPEN_BLOCK, OK] },
  { name: "deadline_replay_once", policy: D, steps: [STOP_OPEN_BLOCK, OK] },
  { name: "deadline_replay_only_once", policy: D, steps: [STOP_OPEN_BLOCK, STOP_OPEN_BLOCK, OK] },
  { name: "legacy_overloaded_event", policy: L, steps: [{ sse: [START, ev({ type: "error", error: { type: "overloaded_error", message: "busy" } })], end: "close" }, OK] },
  { name: "deadline_overloaded_event", policy: D, steps: [{ sse: [START, ev({ type: "error", error: { type: "overloaded_error", message: "busy" } })], end: "close" }, OK] },
  { name: "legacy_echo_complete", policy: L, steps: [{ sse: [START, ...TEXT.slice(0, 1),
    ev({ type: "content_block_delta", index: 1, delta: { type: "text_delta", text: `key ${KEY}` } }), TEXT[2], ...END("end_turn")], end: "close" }] },
  { name: "legacy_echo_partial", policy: L, steps: [{ sse: [START, ...TEXT.slice(0, 1),
    ev({ type: "content_block_delta", index: 1, delta: { type: "text_delta", text: `key ${KEY}` } })], end: "destroy" }] },
  { name: "legacy_bad_tool_json", policy: L, steps: [{ sse: [START, ...THINK, ...TOOL("{\"section\": \"Act", 1), ...END()], end: "close" }] },
  { name: "legacy_length_tool", policy: L, steps: [{ sse: [START, ...THINK, ...TOOL("{\"section\": \"Actions\"}", 1), ...END("max_tokens")], end: "close" }] },
  { name: "legacy_event_mismatch", policy: L, steps: [{ sse: ["event: message_stop\n", `data: ${compact({ type: "ping" })}\n\n`], end: "close" }] },
  { name: "legacy_empty_event_name", policy: L, steps: [{ sse: ["event: \n", "data: {}\n\n"], end: "close" }] },
  { name: "legacy_nonstream", policy: L, steps: [{ json: nonStream }], request: NONSTREAM_REQUEST },
  { name: "deadline_nonstream", policy: D, steps: [{ json: nonStream }], request: NONSTREAM_REQUEST },
  { name: "deadline_429_retry_after", policy: D, steps: [{ status: 429, headers: { "retry-after": "2" } }, OK] },
  { name: "deadline_break_after_content", policy: D, steps: [{ sse: HEAD, end: "destroy" }, OK] },
  { name: "deadline_stop_before_block_stop", policy: D, steps: [NO_BLOCK_STOP, OK] },
  { name: "deadline_401", policy: D, steps: [{ status: 401, body: `bad ${KEY}` }] },
  { name: "deadline_hang", policy: D, steps: [{ sse: HEAD, end: "hang" }], provider: { recovery_window_s: 6.0 } },
  // Read timeouts (timeout_seconds 1.0) before the headers and inside the body.
  { name: "legacy_header_stall", policy: L, steps: [{ silent: true }, OK], provider: { timeout_seconds: 1.0 } },
  { name: "deadline_header_stall", policy: D, steps: [{ silent: true }, OK], provider: { timeout_seconds: 1.0, recovery_window_s: 30.0 } },
  { name: "deadline_body_stall", policy: D, steps: [{ sse: [START], end: "hang" }, OK], provider: { timeout_seconds: 1.0, recovery_window_s: 30.0 } },
];

const server = new FakeServer(Object.fromEntries(SCENARIOS.map((s) => [s.name, s.steps])));
let base = "";
before(async () => {
  base = await server.start();
});
after(async () => {
  await server.stop();
});

const camel: Record<string, string> = {
  session_affinity_id: "sessionAffinityId", recovery_window_s: "recoveryWindowS", preflight_probe: "preflightProbe",
  auth_scheme: "authScheme", timeout_seconds: "timeoutSeconds",
};

/** anyio's cancel-scope message names id(scope), a memory address. */
function scopeFree(record: string): string {
  return record.replace(/cancel scope [0-9a-f]+;/g, "cancel scope <id>;");
}

async function runTs(scenario: Scenario): Promise<any> {
  const sleeps: any[] = [];
  const clock: Clock = { ...SYSTEM_CLOCK, sleep: async (seconds) => { sleeps.push(seconds); } };
  const records: any[] = [];
  const options: any = { baseUrl: `${base}/${scenario.name}/ts/v1`, apiKey: KEY, recoveryPolicy: scenario.policy, clock };
  for (const [key, value] of Object.entries(scenario.provider ?? {})) {
    if (key === "two_keys") options.standbyApiKeys = STANDBY;
    else options[camel[key]] = value;
  }
  const provider = new AnthropicMessagesProvider(options);
  const outcome: any = { name: scenario.name, sleeps };
  try {
    const response = await provider.complete(parsePy(dumpsPy(scenario.request ?? REQUEST)), {
      modelCallId: "model-call-000003", onAttemptFailure: (record) => records.push(record),
    });
    outcome.response = canonicalText(normalizeTiming(response));
  } catch (error) {
    outcome.error = describeError(error);
  }
  outcome.records = records.map((record) => scopeFree(canonicalText(normalizeTiming(record))));
  outcome.rotations = provider.credentialRotations().map(({ unix_s: _unix, ...rest }) => rest);
  return outcome;
}

test("Anthropic live scenarios: results, records, sleeps, rotations and wire match Python", async () => {
  const input = join(dir, "live-in.json");
  const output = join(dir, "live-out.json");
  writePy(input, {
    base, api_key: KEY, standby: STANDBY, scenarios: SCENARIOS.map((s) => ({
      name: s.name, policy: s.policy, provider: s.provider ?? {}, request: s.request ?? REQUEST, model_call_id: "model-call-000003",
    })),
  });
  const pythonRun = runPython("anthropic_live.py", [input, output], 900_000);
  const tsResults = await Promise.all(SCENARIOS.map(runTs));
  await pythonRun;
  const pyResults = readPy(output).results;
  for (const result of pyResults) result.records = result.records.map(scopeFree);
  const failures: string[] = [];
  for (const [index, scenario] of SCENARIOS.entries()) {
    const py = pyResults[index];
    const ts = tsResults[index];
    const finalPy = "response" in py ? py.response : canonicalText(py.error);
    const finalTs = "response" in ts ? ts.response : canonicalText(ts.error);
    const sleepsPy = py.sleeps.map((s: any) => (typeof s.value === "object" ? s.value.value : s.value));
    const sleepsFloatPy = py.sleeps.map((s: any) => s.float);
    const exactRecords = canonicalText(py.records) === canonicalText(ts.records);
    const chainOnly = !exactRecords && py.records.length === ts.records.length && py.records.every((record: string, i: number) => {
      const a = parsePy(record);
      const b = parsePy(ts.records[i]);
      delete a.error_chain;
      delete b.error_chain;
      return canonicalText(a) === canonicalText(b);
    });
    const seenPy = server.seen.get(`${scenario.name}/py`) ?? [];
    const seenTs = server.seen.get(`${scenario.name}/ts`) ?? [];
    const wireSame = seenPy.length === seenTs.length && seenPy.every((request, i) =>
      request.sha256 === seenTs[i].sha256 && canonicalText(request.headers) === canonicalText(seenTs[i].headers));
    const row = {
      name: scenario.name, final: finalPy === finalTs, records: exactRecords ? "exact" : chainOnly ? "error_chain_only" : "DIFF",
      sleeps: canonicalText(sleepsPy) === canonicalText(ts.sleeps), rotations: canonicalText(py.rotations) === canonicalText(ts.rotations),
      wire: wireSame, attempts: seenPy.length,
    };
    console.log(`# ${JSON.stringify(row)} py_sleeps=${canonicalText(sleepsPy)} float=${canonicalText(sleepsFloatPy)}`);
    if (!row.final) failures.push(`${scenario.name} final\n py ${finalPy.slice(0, 600)}\n ts ${finalTs.slice(0, 600)}`);
    if (row.records !== "exact") failures.push(`${scenario.name} records\n py ${py.records.join("\n    ")}\n ts ${ts.records.join("\n    ")}`);
    if (!row.sleeps) failures.push(`${scenario.name} sleeps py ${canonicalText(sleepsPy)} ts ${canonicalText(ts.sleeps)}`);
    if (!row.rotations) failures.push(`${scenario.name} rotations py ${canonicalText(py.rotations)} ts ${canonicalText(ts.rotations)}`);
    if (!row.wire) failures.push(`${scenario.name} wire py ${canonicalText(seenPy)} ts ${canonicalText(seenTs)}`);
    if (py.workers_started_in_window !== 0) {
      failures.push(`${scenario.name}: Python started ${py.workers_started_in_window} worker(s) inside the window`);
    }
  }
  assert.deepEqual(failures, []);
});
