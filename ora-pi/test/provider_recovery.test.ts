// Differential test: recovery decisions and live retry behaviour, TS vs Python.
//
// 1. Decision table: RecoveryWindow.for_provider / delay / transport_delay,
//    transient_transport and retry_after_seconds on a frozen clock.
// 2. Live: both clients run the same scripted scenarios against one local
//    fake server (429 with/without Retry-After, 5xx, disconnects before and
//    after content, no [DONE], 401, credential echo, deadline, non-streaming
//    quickstart replies; under first_token_deadline_v1 first-token timeouts,
//    stalls, exhausted retries, a slow but steady stream longer than the
//    window, a run deadline, and dead endpoints and an outage against the
//    no-content window, where backoff sleeps still advance the clock
//    (`virtualClock`)); final results, attempt-failure records, backoff
//    sleeps and the bytes on the wire are compared. Python's deadline attempt
//    runs in a worker process the helper starts before the scenario's window
//    opens (common.PrestartedWorkers), so the window covers the attempt on
//    both sides, not Python's interpreter start-up; a worker started inside a
//    window fails the scenario, unless it ends only by its window.

import assert from "node:assert/strict";
import type { ChildProcess } from "node:child_process";
import { join } from "node:path";
import { after, before, test } from "node:test";
import { dumpsPy, parsePy } from "../src/pyjson.ts";
import {
  FIRST_TOKEN_RECOVERY_POLICY, OpenAICompatibleProvider, PolicySessionError, ProviderAttemptTimeout,
  ProviderRecoveryDeadline, PyError, RecoveryWindow,
  SYSTEM_CLOCK, TransportError, httpStatusFailure, retryAfterSeconds, sha256Hex, toolAlias, transientTransport,
  type Clock,
} from "../src/provider/index.ts";
import { pythonJsonDecodeError } from "../src/provider/json_errors.ts";
import { PyEOFError, TlsError } from "../src/provider/py.ts";
import { FakeServer, type Step } from "./provider/fake_server.ts";
import {
  HELPERS, canonicalText, describeError, normalizeTiming, readPy, runPython, scratchDir, spawnPython, writePy,
} from "./provider/harness.ts";

const dir = scratchDir("recovery");

// -- 1. decision table -------------------------------------------------------------------

function frozenClock(mono: number, unix: number): Clock {
  return { monotonic: () => mono, time: () => unix, sleep: async () => { throw new Error("no sleeping"); } };
}

const FRAMES: Record<string, () => Error> = {
  connect_error: () => new TransportError("ConnectError", "All connection attempts failed"),
  remote_protocol: () => new TransportError("RemoteProtocolError", "peer closed connection"),
  read_timeout: () => new TransportError("ReadTimeout", ""),
  eof: () => new PyEOFError("eof"),
  ssl_cert: () => new TlsError("SSLCertVerificationError", "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed", ""),
  ssl_eof: () => new TlsError("SSLEOFError", "EOF occurred in violation of protocol", ""),
  ssl_bad_mac: () => new TlsError("SSLError", "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac", ""),
  ssl_want_read: () => new TlsError("SSLWantReadError", "The operation did not complete (read)", ""),
  ssl_generic: () => new TlsError("SSLError", "[SSL: WRONG_VERSION_NUMBER] wrong version number", ""),
  value_error: () => new PyError("ValueError", "v"),
  pse: () => new PolicySessionError("p"),
};
const ATTRS: Record<string, string> = {
  retryable: "retryable", retry_delay_s: "retryDelayS", response_started: "responseStarted",
  partial_events: "partialEvents", upstream_stream_break: "upstreamStreamBreak", http_status: "httpStatus",
  attempt_timeout: "attemptTimeout", attempt_timeout_s: "attemptTimeoutS",
};

function build(spec: any): any {
  if (spec.kind === "deadline") return new ProviderRecoveryDeadline(spec.events ?? []);
  if (spec.kind === "attempt_timeout") return new ProviderAttemptTimeout(spec.limit, plainNumber(spec.limit_s), spec.events ?? []);
  if (spec.kind === "http") return httpStatusFailure(spec.status, spec.detail ?? "", spec.base_url ?? "https://x/v1");
  if (spec.kind === "chain") {
    let exc: any;
    for (const name of [...spec.frames].reverse()) {
      const outer: any = FRAMES[name]();
      if (exc !== undefined) outer.cause = exc;
      exc = outer;
    }
    return exc;
  }
  const exc: any = new PolicySessionError("failure");
  for (const [key, value] of Object.entries(spec.attrs ?? {})) {
    exc[ATTRS[key]] = value instanceof Object && "value" in (value as any) ? (value as any).value : value;
  }
  return exc;
}

function plainNumber(value: any): number {
  return value !== null && typeof value === "object" && "value" in value ? value.value : value;
}

function num(value: { value: number; isFloat: boolean } | null): any {
  return value === null ? null : { value: value.value, float: value.isFloat };
}

test("recovery decisions match Python on a frozen clock", async () => {
  const out = join(dir, "decisions.json");
  await runPython("recovery_decisions.py", [out]);
  const data = readPy(out);
  const plain = (v: any) => (v !== null && typeof v === "object" && "value" in v ? v.value : v);
  const clock = frozenClock(plain(data.now_mono), plain(data.now_unix));
  for (const item of data.for_provider) {
    const run = item.run === "nan" ? NaN : item.run === "inf" ? Infinity : plain(item.run);
    let got: any;
    try {
      const window = RecoveryWindow.forProvider({
        recoveryPolicy: item.policy, recoveryWindowS: plain(item.window_s), requestDeadlineMonotonic: run,
        preflightProbe: item.preflight, maxRetries: item.max_retries,
      }, clock);
      got = { result: window === null ? null : {
        deadline: Number.isFinite(window.deadline) ? window.deadline : null, retries_enabled: window.retriesEnabled,
        policy: window.policy, first_token_s: window.firstTokenS, stall_s: window.stallS,
        no_content_window_s: window.noContentWindowS,
      } };
    } catch (error: any) {
      got = { error: error.message };
    }
    const expected = "error" in item ? { error: item.error } : { result: item.result === null ? null
      : { deadline: plain(item.result.deadline), retries_enabled: item.result.retries_enabled, policy: item.result.policy,
        first_token_s: plain(item.result.first_token_s), stall_s: plain(item.result.stall_s),
        no_content_window_s: plain(item.result.no_content_window_s) } };
    assert.deepEqual(got, expected, canonicalText(item));
  }
  for (const item of data.delays) {
    const deadline = item.remaining === null ? Infinity : plain(data.now_mono) + plain(item.remaining);
    const window = item.policy === FIRST_TOKEN_RECOVERY_POLICY
      ? new RecoveryWindow(deadline, item.enabled, clock, { policy: item.policy, firstTokenS: 30.0, stallS: 180.0 })
      : new RecoveryWindow(deadline, item.enabled, clock);
    window.streamBreaks = item.prior;
    window.firstTokenTimeouts = item.prior;
    window.noContentFailures = item.prior + 5;
    // No window open yet, one open 12.5 s (inside the 30 s window), one open exactly 30 s.
    window.noContentSince = item.prior === 0 ? null : plain(data.now_mono) - (item.prior === 2 ? 12.5 : 30.0);
    const exc = build(item.spec);
    const delay = num(window.delay(build(item.spec), item.attempt));
    const transport = num(window.transportDelay(exc, item.attempt, plain(item.started)));
    const expected = (v: any) => (v === null ? null : { value: plain(v.value), float: v.float });
    assert.deepEqual(
      { delay, transport, breaks: window.streamBreaks, exc: exc.streamBreaks,
        firstToken: window.firstTokenTimeouts, excFirstToken: exc.firstTokenTimeouts ?? null,
        noContent: window.noContentFailures, excNoContent: exc.noContentFailures ?? null,
        since: window.noContentSince, excWindow: exc.noContentWindowS ?? null, excElapsed: exc.noContentElapsedS ?? null },
      { delay: expected(item.delay), transport: expected(item.transport_delay), breaks: item.stream_breaks, exc: item.exc_stream_breaks,
        firstToken: item.first_token_timeouts, excFirstToken: item.exc_first_token_timeouts,
        noContent: item.no_content_failures, excNoContent: item.exc_no_content_failures,
        since: plain(item.no_content_since), excWindow: plain(item.exc_no_content_window_s),
        excElapsed: plain(item.exc_no_content_elapsed_s) },
      canonicalText(item),
    );
  }
  for (const item of data.transient) assert.equal(transientTransport(build(item.spec)), item.transient, canonicalText(item.spec));
  for (const item of data.retry_after) {
    const got = retryAfterSeconds(item.value, clock);
    assert.deepEqual(got === null ? null : { value: got, float: true },
      item.seconds === null ? null : { value: plain(item.seconds.value), float: item.seconds.float }, String(item.value));
  }
  for (const item of data.json_errors) {
    const error = pythonJsonDecodeError(item.text);
    if (item.ok) {
      assert.equal(error, null, item.text);
    } else {
      assert.equal(error?.message, item.message, item.text);
      const context: any = error?.context;
      assert.deepEqual(context ? [context.pyType, context.message] : null, item.context, item.text);
    }
  }
  console.log(JSON.stringify({ for_provider: data.for_provider.length, delays: data.delays.length,
    transient: data.transient.length, retry_after: data.retry_after.length, json_errors: data.json_errors.length }));
});

// -- 2. live scenarios -----------------------------------------------------------------

const KEY = "sk-fake-provider-key-0123456789";
const MANUAL = toolAlias("robot.read_manual");
const SUBMIT = toolAlias("robot.submit_action");
const compact = (value: unknown) => dumpsPy(value, { ensureAscii: false, separators: [",", ":"] });
const data = (event: unknown) => `data: ${compact(event)}\n\n`;
const chunk = (delta: any, finish: any = null) => data({
  id: "chatcmpl-live", object: "chat.completion.chunk", created: 1791924527, model: "deepseek-flash",
  choices: [{ index: 0, delta, finish_reason: finish, logprobs: null }],
});
const USAGE = data({ id: "chatcmpl-live", object: "chat.completion.chunk", choices: [],
  usage: { prompt_tokens: 22721, completion_tokens: 516, total_tokens: 23237, prompt_cache_hit_tokens: 22528,
    prompt_cache_miss_tokens: 193, completion_tokens_details: { reasoning_tokens: 351 } } });
const OK_EVENTS = [
  ": keep-alive\n\n", chunk({ role: "assistant", content: "" }), chunk({ reasoning_content: "Plan: read " }),
  chunk({ reasoning_content: "the manual." }), chunk({ content: "Reading." }),
  chunk({ tool_calls: [{ index: 0, id: "call_0", type: "function", function: { name: MANUAL, arguments: "" } }] }),
  chunk({ tool_calls: [{ index: 0, function: { arguments: "{\"section\": \"Actions\"}" } }] }),
  chunk({}, "tool_calls"), USAGE,
];
const OK: Step = { sse: [...OK_EVENTS, "data: [DONE]\n\n"], end: "close" };
const HEAD = OK_EVENTS.slice(0, 4);
const smokeJson = (n: number, name: string, args: unknown) => JSON.stringify({
  id: `smoke-${n}`, object: "chat.completion", choices: [{ index: 0, finish_reason: "tool_calls", message: {
    role: "assistant", content: null, tool_calls: [{ id: `call-${n}`, type: "function", function: { name, arguments: dumpsPy(args) } }],
  } }], usage: { prompt_tokens: 500, completion_tokens: 50, total_tokens: 550 },
});

const TOOLS = ["robot.read_manual", "robot.submit_action", "observe"].map((name) => ({
  type: "function", function: { name, description: `${name} tool`, parameters: { type: "object", properties: { x: { type: "number", default: 1.0 } } } },
}));
const DEFAULT_REQUEST = {
  model: "deepseek-flash", messages: [{ role: "system", content: "You control one robot." }, { role: "user", content: "Pick the soup." }],
  tools: TOOLS, tool_choice: "auto", max_tokens: 16384, stream: true, reasoning_effort: "low", stream_options: { include_usage: true },
};
const QUICKSTART_1 = {
  model: "smoke-model", messages: [{ role: "system", content: "Workspace." }, { role: "user", content: "Read the robot manual, then move." }],
  tools: TOOLS.slice(0, 2), tool_choice: "auto", max_tokens: 2048, stream: false,
};
const QUICKSTART_2 = {
  ...QUICKSTART_1,
  messages: [...QUICKSTART_1.messages,
    { role: "assistant", content: "", tool_calls: [{ id: "call-1", type: "function", function: { name: "robot.read_manual", arguments: "{\"section\": \"Actions\"}" } }] },
    { role: "tool", tool_call_id: "call-1", name: "robot.read_manual", content: "{\"manual\": \"move_pose ...\"}" }],
};

interface Scenario { name: string; policy: string; steps: Step[]; request?: any; provider?: any; refused?: boolean; blackhole?: boolean;
  /** first_token_deadline_v1's stall limit (STALL_S), shortened for the test. */
  stallS?: number;
  /** A run deadline this many seconds after the call starts. */
  runDeadlineS?: number;
  /** Each skipped backoff sleep still moves the monotonic clock on by its length. */
  virtualClock?: boolean;
  /** How many attempts each core must make: its failure records, and one more for a reply. */
  attempts?: number;
  /** Ends only by the incident deadline: later delays are clipped to the remaining window, a clock reading. */
  windowClipped?: boolean }
const D = "same_credential_deadline_v1";
const FT = "first_token_deadline_v1";
const L = "legacy_v1";
const SCENARIOS: Scenario[] = [
  { name: "ok_stream", policy: D, steps: [OK] },
  { name: "retry_after_429", policy: D, steps: [{ status: 429, headers: { "retry-after": "3" }, body: "{\"error\":{\"message\":\"rate limited\"}}" }, OK] },
  { name: "retry_after_float", policy: D, steps: [{ status: 429, headers: { "retry-after": "1.5" } }, OK] },
  { name: "default_429", policy: D, steps: [{ status: 429, body: "{\"error\":{\"message\":\"rate limited\"}}" }, OK] },
  { name: "backoff_5xx", policy: D, steps: [{ status: 503, body: "busy" }, { status: 500, body: "oops" }, { status: 502 }, OK] },
  { name: "retry_after_past_date", policy: D, steps: [{ status: 503, headers: { "retry-after": "Wed, 21 Oct 2015 07:28:00 GMT" } }, OK] },
  { name: "disconnect_before_content", policy: D, steps: [{ sse: [], end: "destroy" }, OK] },
  { name: "drop_before_response", policy: D, steps: [{ drop: true }, OK] },
  { name: "disconnect_after_content", policy: D, steps: [{ sse: HEAD, end: "destroy" }, OK] },
  { name: "no_done_limit", policy: D, steps: [{ sse: OK_EVENTS, end: "close" }] },
  { name: "no_done_then_ok", policy: D, steps: [{ sse: OK_EVENTS, end: "close" }, OK] },
  { name: "auth_401", policy: D, steps: [{ status: 401, body: `{"error":"bad key ${KEY}"}` }] },
  { name: "forbidden_403", policy: D, steps: [{ status: 403 }] },
  { name: "bad_request_400", policy: D, steps: [{ status: 400, body: `{"error":{"message":"bad tool ${KEY}"}}` }] },
  { name: "echo_partial", policy: D, steps: [{ sse: [chunk({ content: `key ${KEY}` })], end: "destroy" }, OK] },
  { name: "echo_complete", policy: D, steps: [{ sse: [chunk({ content: `key ${KEY}` }, "stop"), "data: [DONE]\n\n"], end: "close" }] },
  { name: "malformed_before_content", policy: D, steps: [{ sse: ["data: {bad\n\n"], end: "close" }] },
  { name: "malformed_after_content", policy: D, steps: [{ sse: [...HEAD, "data: {bad\n\n"], end: "close" }] },
  { name: "event_mismatch", policy: D, steps: [{ sse: ["event: foo\n", chunk({ content: "x" })], end: "close" }] },
  { name: "multiple_choices", policy: D, steps: [{ sse: [data({ choices: [{ index: 0, delta: {} }, { index: 1, delta: {} }] }), "data: [DONE]\n\n"], end: "close" }] },
  { name: "length_tool_calls", policy: D, steps: [{ sse: [chunk({ reasoning_content: "r" }),
    chunk({ tool_calls: [{ index: 0, id: "call_0", function: { name: SUBMIT, arguments: "{\"arm\": \"le" } }] }), chunk({}, "length"), "data: [DONE]\n\n"], end: "close" }] },
  { name: "upstream_408", policy: D, steps: [{ status: 408, body: "{\"error\":{\"message\":\"stream disconnected before completion: upstream\",\"type\":\"invalid_request_error\"}}" }, OK] },
  { name: "u2028_in_content", policy: D, steps: [{ sse: [chunk({ content: "a b" }, "stop"), "data: [DONE]\n\n"], end: "close" }, OK] },
  { name: "crlf_and_comments", policy: D, steps: [{ sse: [": c\r\n\r\n", chunk({ content: "x" }).replace("\n\n", "\r\n\r\n"), chunk({}, "stop"), "data:[DONE]\r\n\r\n"], end: "close" }] },
  { name: "affinity_header", policy: D, steps: [OK], provider: { session_affinity_id: "ora:run-42:native" } },
  { name: "deadline_hang", policy: D, steps: [{ sse: HEAD.slice(0, 2), end: "hang" }], provider: { recovery_window_s: 6.0 } },
  // Read timeouts before the headers and inside the body (timeout_seconds 1.0).
  { name: "header_stall", policy: D, steps: [{ silent: true }, OK], provider: { timeout_seconds: 1.0, recovery_window_s: 30.0 } },
  { name: "body_stall", policy: D, steps: [{ sse: [], end: "hang" }, OK], provider: { timeout_seconds: 1.0, recovery_window_s: 30.0 } },
  { name: "legacy_header_stall", policy: L, steps: [{ silent: true }, OK], provider: { timeout_seconds: 1.0 } },
  { name: "legacy_body_stall", policy: L, steps: [{ sse: [], end: "hang" }, OK], provider: { timeout_seconds: 1.0 } },
  // Connect timeouts (min(10, timeout) = 1 s) against a listener whose accept queue is full.
  { name: "connect_timeout", policy: D, steps: [OK], blackhole: true, windowClipped: true, provider: { timeout_seconds: 1.0, recovery_window_s: 6.0 } },
  { name: "legacy_connect_timeout", policy: L, steps: [OK], blackhole: true, provider: { timeout_seconds: 1.0 } },
  { name: "nonstream_deadline", policy: D, steps: [{ json: smokeJson(1, MANUAL, { section: "Actions" }) }], request: QUICKSTART_1 },
  { name: "legacy_ok_stream", policy: L, steps: [OK] },
  { name: "legacy_5xx_exhausted", policy: L, steps: [{ status: 500, body: "x" }] },
  { name: "legacy_429_then_ok", policy: L, steps: [{ status: 429, headers: { "retry-after": "30" } }, OK] },
  { name: "legacy_no_done_ok", policy: L, steps: [{ sse: OK_EVENTS, end: "close" }] },
  { name: "legacy_eof_without_finish", policy: L, steps: [{ sse: HEAD, end: "close" }, OK] },
  { name: "legacy_disconnect_after_content", policy: L, steps: [{ sse: HEAD, end: "destroy" }, OK] },
  { name: "legacy_echo_partial", policy: L, steps: [{ sse: [chunk({ content: `key ${KEY}` })], end: "destroy" }] },
  { name: "legacy_401", policy: L, steps: [{ status: 401, body: `bad key ${KEY}` }] },
  { name: "legacy_connection_refused", policy: L, steps: [OK], refused: true },
  { name: "deadline_connection_refused", policy: D, steps: [OK], refused: true, provider: { preflight_probe: true } },
  { name: "legacy_empty_event_name", policy: L, steps: [{ sse: ["event: \n", "data: {}\n\n"], end: "close" }] },
  { name: "quickstart_call_1", policy: L, steps: [{ json: smokeJson(1, MANUAL, { section: "Actions" }) }], request: QUICKSTART_1 },
  { name: "quickstart_call_2", policy: L, steps: [{ json: smokeJson(2, SUBMIT, { arm: "left", pose: { position_m: [0, 0, 1.02], orientation_xyzw: [0, 0, 0, 1] } }) }], request: QUICKSTART_2 },
  { name: "quickstart_invalid_json", policy: L, steps: [{ json: "not json" }], request: QUICKSTART_1 },
  { name: "quickstart_empty_body", policy: L, steps: [{ json: "" }], request: QUICKSTART_1 },
  // first_token_deadline_v1: a first-token limit per attempt (recovery_window_s), a stall limit, no total limit.
  { name: "ft_ok_stream", policy: FT, steps: [OK], provider: { recovery_window_s: 5.0 } },
  { name: "ft_first_token_retry", policy: FT, steps: [{ silent: true }, OK], provider: { recovery_window_s: 1.5 } },
  // Keep-alive comments are not a first event.
  { name: "ft_keepalive_only_retry", policy: FT, steps: [{ sse: [": keep-alive\n\n"], end: "hang" }, OK],
    provider: { recovery_window_s: 1.5 } },
  // The first first-token timeout opens the window; the second attempt starts inside it and is
  // retried, the third starts a whole first-token limit later and is the last.
  { name: "ft_first_token_exhausted", policy: FT, steps: [{ silent: true }], attempts: 3,
    provider: { recovery_window_s: 1.0 } },
  { name: "ft_stall_retry", policy: FT, steps: [{ sse: HEAD, end: "hang" }, OK], stallS: 1.5,
    provider: { recovery_window_s: 5.0 } },
  { name: "ft_stall_exhausted", policy: FT, steps: [{ sse: HEAD, end: "hang" }], stallS: 1.0,
    provider: { recovery_window_s: 5.0 } },
  // The regression: a healthy stream slower in total than the window succeeds in one attempt.
  // Its first content comes a few gaps in, so the window keeps a second or more of margin on a loaded host.
  { name: "ft_slow_steady_stream", policy: FT, steps: [{ ...OK, gapMs: 1200 }], stallS: 3.6,
    provider: { recovery_window_s: 4.5 } },
  { name: "ft_nonstream_ok", policy: FT, steps: [{ json: smokeJson(1, MANUAL, { section: "Actions" }) }], request: QUICKSTART_1,
    provider: { recovery_window_s: 5.0 } },
  { name: "ft_run_deadline", policy: FT, steps: [{ silent: true }], runDeadlineS: 2.0, provider: { recovery_window_s: 30.0 } },
  // A dead endpoint is retried while its attempts start inside the no-content window (30 s) its
  // first failure opened: attempts at 0, 1, 3, 7, 15 and 31 s of backoff, so six attempts.
  { name: "ft_dead_503", policy: FT, steps: [{ status: 503, body: "down" }], virtualClock: true, attempts: 6,
    provider: { recovery_window_s: 30.0 } },
  { name: "ft_dead_refused", policy: FT, steps: [OK], refused: true, virtualClock: true, attempts: 6,
    provider: { recovery_window_s: 30.0 } },
  // An outage shorter than the window recovers: nine failures (0 ... 91 s), the tenth attempt answers.
  { name: "ft_outage_recovers", policy: FT, virtualClock: true, attempts: 10, provider: { recovery_window_s: 120.0 },
    steps: [...Array.from({ length: 9 }, (): Step => ({ status: 503, body: "restarting" })), OK] },
  { name: "ft_503_then_ok", policy: FT, steps: [{ status: 503, body: "busy" }, OK], provider: { recovery_window_s: 5.0 } },
  { name: "ft_backoff_5xx", policy: FT, steps: [{ status: 503, body: "busy" }, { status: 429, headers: { "retry-after": "1.5" } }, OK],
    provider: { recovery_window_s: 5.0 } },
];

const server = new FakeServer(Object.fromEntries(SCENARIOS.map((s) => [s.name, s.steps])));
let base = "";
let closedBase = "";
let blackholeBase = "";
let backlog: ChildProcess | null = null;
before(async () => {
  base = await server.start();
  backlog = spawnPython([join(HELPERS, "full_backlog.py")], { stdio: ["pipe", "pipe", "inherit"] });
  const port = await new Promise<string>((resolve) => backlog!.stdout!.once("data", (chunk) => resolve(String(chunk).trim())));
  blackholeBase = `http://127.0.0.1:${port}`;
  const probe = new FakeServer({});
  closedBase = await probe.start();
  await probe.stop(); // nothing listens on this port any more
});
after(async () => {
  backlog?.stdin?.end();
  await server.stop();
});

const camel: Record<string, string> = {
  session_affinity_id: "sessionAffinityId", recovery_window_s: "recoveryWindowS", preflight_probe: "preflightProbe",
  timeout_seconds: "timeoutSeconds",
};

function scenarioBase(scenario: Scenario): string {
  return scenario.refused ? closedBase : scenario.blackhole ? blackholeBase : base;
}

/** anyio's cancel-scope message names id(scope), a memory address. */
function scopeFree(record: string): string {
  return record.replace(/cancel scope [0-9a-f]+;/g, "cancel scope <id>;");
}

async function runTs(scenario: Scenario): Promise<any> {
  const sleeps: any[] = [];
  let offset = 0;
  const clock: Clock = {
    ...SYSTEM_CLOCK, monotonic: () => SYSTEM_CLOCK.monotonic() + offset,
    sleep: async (seconds) => {
      sleeps.push(seconds);
      if (scenario.virtualClock) offset += seconds;
    },
  };
  const records: any[] = [];
  const options: any = {
    baseUrl: `${scenarioBase(scenario)}/${scenario.name}/ts/v1`, apiKey: KEY, recoveryPolicy: scenario.policy, clock,
  };
  for (const [key, value] of Object.entries(scenario.provider ?? {})) options[camel[key]] = value;
  if (scenario.stallS !== undefined) options.stallS = scenario.stallS;
  const provider = new OpenAICompatibleProvider(options);
  const outcome: any = { name: scenario.name, sleeps };
  try {
    const response = await provider.complete(parsePy(dumpsPy(scenario.request ?? DEFAULT_REQUEST)), {
      modelCallId: "model-call-000007", onAttemptFailure: (record) => records.push(record),
      deadlineMonotonicMs: scenario.runDeadlineS === undefined ? null : performance.now() + scenario.runDeadlineS * 1000,
    });
    outcome.response = canonicalText(normalizeTiming(response));
    outcome.ttft_negative = response.ttft_s !== null && Number(response.ttft_s) < 0;
  } catch (error) {
    outcome.error = describeError(error);
  }
  outcome.records = records.map((record) => scopeFree(canonicalText(normalizeTiming(record))));
  return outcome;
}

/** Fields of a record that legitimately depend on the HTTP stack, not ORA's logic. */
function withoutTransportChain(record: string): any {
  const value = parsePy(record);
  value.error_chain = value.error_chain.map((entry: any) => ({ type: entry.type, message: entry.message }));
  return value;
}

test("live scenarios: results, attempt records, sleeps and wire bytes match Python", async () => {
  const input = join(dir, "live-in.json");
  const output = join(dir, "live-out.json");
  writePy(input, {
    base, api_key: KEY, scenarios: SCENARIOS.map((s) => ({
      name: s.name, policy: s.policy, provider: s.provider ?? {}, request: s.request ?? DEFAULT_REQUEST,
      model_call_id: "model-call-000007", ...(s.refused || s.blackhole ? { base: scenarioBase(s) } : {}),
      ...(s.stallS !== undefined ? { stall_s: s.stallS } : {}),
      ...(s.runDeadlineS !== undefined ? { run_deadline_s: s.runDeadlineS } : {}),
      ...(s.virtualClock ? { virtual_clock: true } : {}),
    })),
  });
  const pythonRun = runPython("recovery_live.py", [input, output], 900_000);
  const tsResults = await Promise.all(SCENARIOS.map(runTs));
  await pythonRun;
  const pyResults = readPy(output).results;
  for (const result of pyResults) result.records = result.records.map(scopeFree);
  const report: any[] = [];
  const failures: string[] = [];
  for (const [index, scenario] of SCENARIOS.entries()) {
    const py = pyResults[index];
    const ts = tsResults[index];
    const finalPy = "response" in py ? py.response : scopeFree(canonicalText(py.error));
    const finalTs = "response" in ts ? ts.response : scopeFree(canonicalText(ts.error));
    if (scenario.windowClipped) {
      // How many 1 s attempts fit in the window, and the delays clipped to what
      // is left of it, are clock readings no two runs share; the first attempt
      // record and the outcome are not.
      const outcome = (error: any) => {
        const value = parsePy(scopeFree(canonicalText(error)));
        if (value.provider_failure) delete value.provider_failure.attempts;
        return canonicalText(value);
      };
      const final = outcome(py.error) === outcome(ts.error);
      const first = py.records[0] === ts.records[0];
      const last = (records: string[]) => parsePy(records[records.length - 1]).error_chain[0].type;
      report.push({ name: scenario.name, final, first_record: first, last: [last(py.records), last(ts.records)] });
      if (!final) failures.push(`${scenario.name} final\n py ${canonicalText(py.error)}\n ts ${canonicalText(ts.error)}`);
      if (!first) failures.push(`${scenario.name} first record\n py ${py.records[0]}\n ts ${ts.records[0]}`);
      if (last(py.records) !== "ProviderRecoveryDeadline" || last(ts.records) !== "ProviderRecoveryDeadline") failures.push(`${scenario.name} did not end by the deadline`);
      continue;
    }
    const sleepsPy = py.sleeps.map((s: any) => (typeof s.value === "object" ? s.value.value : s.value));
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
      sleeps: canonicalText(sleepsPy) === canonicalText(ts.sleeps), wire: wireSame, attempts: seenPy.length,
    };
    report.push(row);
    if (!row.final) failures.push(`${scenario.name} final\n py ${finalPy}\n ts ${finalTs}`);
    if (row.records !== "exact") failures.push(`${scenario.name} records\n py ${py.records.join("\n    ")}\n ts ${ts.records.join("\n    ")}`);
    if (row.records === "error_chain_only") {
      for (const [i, record] of py.records.entries()) {
        const a = parsePy(record).error_chain;
        const b = parsePy(ts.records[i]).error_chain;
        if (canonicalText(a) !== canonicalText(b)) console.log(`# chain ${scenario.name}[${i}]\n#  py ${canonicalText(a)}\n#  ts ${canonicalText(b)}`);
      }
    }
    if (!row.sleeps) failures.push(`${scenario.name} sleeps py ${canonicalText(sleepsPy)} ts ${canonicalText(ts.sleeps)}`);
    if (!row.wire) failures.push(`${scenario.name} wire py ${canonicalText(seenPy)} ts ${canonicalText(seenTs)}`);
    if (py.ttft_negative !== ts.ttft_negative && "response" in py) failures.push(`${scenario.name} ttft sign`);
    if (scenario.attempts !== undefined) {
      // Equal cores could share a regression: pin the count the scenario is about.
      const made = (result: any) => result.records.length + ("response" in result ? 1 : 0);
      if (made(py) !== scenario.attempts || made(ts) !== scenario.attempts) {
        failures.push(`${scenario.name} attempts py ${made(py)} ts ${made(ts)}, expected ${scenario.attempts}`);
      }
    }
    if (py.workers_started_in_window !== 0) {
      failures.push(`${scenario.name}: Python started ${py.workers_started_in_window} worker(s) inside the window`);
    }
  }
  for (const row of report) console.log(`# ${JSON.stringify(row)}`);
  assert.deepEqual(failures, []);
  void sha256Hex;
  void withoutTransportChain;
});
