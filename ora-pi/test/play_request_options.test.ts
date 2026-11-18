// Free play sends the provider profile's request_options as an episode does
// (src/play/runtime.ts): on every request Pi builds for the ORA model (a turn,
// also after /model went elsewhere and back, and compaction), never on another
// provider's model; refused as the episode provider refuses them.
//
// A real free-play Pi runtime (createPlayRuntime with ORA's play extension and
// the user's Pi config dir holding a second provider) talks to a loopback Chat
// Completions server that keeps every request body.

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, test } from "node:test";

import { PLAY_MESSAGE, PlayDriver } from "../src/play/driver.ts";
import { createPlayRuntime, oraProvider, playRequestOptions } from "../src/play/runtime.ts";
import { buildAnthropicMessagesProvider, OpenAICompatibleProvider, PolicySessionError } from "../src/provider/index.ts";
import { parsePy } from "../src/pyjson.ts";

type Json = any;

const scratch = (label: string) => mkdtempSync(join(process.env.ORA_TEST_SCRATCH ?? tmpdir(), `ora-play-options-${label}-`));
// Pi's catalogs and package checks stay offline.
process.env.PI_OFFLINE = "1";

/** The Muse Spark Open profile's options as the plan carries them across the RPC (1.0 is a Python float). */
const OPTIONS_WIRE = '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "provider": {"require_parameters": true}}';
const OPTIONS = { temperature: 1, top_p: 0.95, top_k: 20, provider: { require_parameters: true } };
const OPTION_KEYS = Object.keys(OPTIONS);

/** A loopback Chat Completions server: every body it receives, a short streamed text reply. */
class CaptureServer {
  bodies: Json[] = [];
  private server = createServer((req, res) => this.handle(req, res));
  async start(): Promise<string> {
    await new Promise<void>((resolve) => this.server.listen(0, "127.0.0.1", resolve));
    return `http://127.0.0.1:${(this.server.address() as AddressInfo).port}/v1`;
  }
  async stop(): Promise<void> {
    this.server.closeAllConnections();
    await new Promise<void>((resolve) => this.server.close(() => resolve()));
  }
  private handle(req: IncomingMessage, res: ServerResponse): void {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      this.bodies.push(body);
      const chunk = (delta: Json, finish: string | null, usage?: Json) => `data: ${JSON.stringify({
        id: `capture-${this.bodies.length}`, object: "chat.completion.chunk", model: body.model,
        choices: [{ index: 0, delta, finish_reason: finish }], ...(usage ? { usage } : {}),
      })}\n\n`;
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(chunk({ role: "assistant", content: "Summary: the operator said hello; nothing moved." }, null));
      res.write(chunk({}, "stop", { prompt_tokens: 40, completion_tokens: 9, total_tokens: 49 }));
      res.end("data: [DONE]\n\n");
    });
  }
}

/** The part of OraCore a free-play runtime touches outside an act; what it records is kept. */
function fakeCore(provider: Json, policy: Json) {
  const events: Json[] = [];
  const artifacts: Json[] = [];
  const core: Json = {
    config: {
      provider, tools: { definitions: [] },
      policy: { model: "fake-play", provider_profile: "test_profile", reasoning_effort: "provider_default",
        supports_images: false, context_window_tokens: 65536, max_output_tokens: 4096, ...policy },
    },
    interrupted: false,
    append: (type: string, fields: Json) => {
      const event = { type, event_id: `event-${events.length}`, ...fields };
      events.push(event);
      return event;
    },
    artifacts: { putJson: (value: Json) => {
      artifacts.push(value);
      return { path: `artifacts/${artifacts.length - 1}.json` };
    } },
    flush: () => {},
  };
  return { core, events, artifacts };
}

async function playRuntime(core: Json, extraProviders: Json = {}, init: Json = {}): Promise<Json> {
  const root = scratch("run");
  const agentDir = join(root, "pi-agent");
  const workspace = join(root, "workspace");
  mkdirSync(agentDir, { recursive: true });
  mkdirSync(workspace, { recursive: true });
  // A compaction summarizes everything but the last token's worth, so a short conversation compacts.
  writeFileSync(join(agentDir, "settings.json"), JSON.stringify({ compaction: { keepRecentTokens: 1 } }));
  writeFileSync(join(agentDir, "models.json"), JSON.stringify({ providers: extraProviders }));
  const driver = new PlayDriver(core, { workspace, auth_path: join(root, "auth.json"), agent_dir: agentDir, ...init });
  const runtime = await createPlayRuntime(core, driver, { sessionDir: join(root, "session"), sessionFile: null, extensions: [], apiKey: "ora-key" });
  return Object.assign(runtime, { oraDriver: driver });
}

const server = new CaptureServer();
let base = "";
before(async () => {
  base = await server.start();
});
after(async () => {
  await server.stop();
});

test("the profile's request_options are validated and refused as the episode provider does", () => {
  assert.equal(playRequestOptions({ kind: "openai_compatible" }), null);
  assert.equal(playRequestOptions({ kind: "openai_compatible", request_options: null }), null);
  // A Python float crosses the RPC as a PyFloat; Pi sends plain JSON.
  assert.deepEqual(playRequestOptions(parsePy(`{"kind": "openai_compatible", "request_options": ${OPTIONS_WIRE}}`)), OPTIONS);
  const refused = [
    {}, [], { seed: 7 }, { temperature: 2.5 }, { top_p: 0 }, { top_k: 0 }, { top_k: 2.5 },
    { chat_template_kwargs: {} }, { reasoning: "high" }, { provider: [] }, { temperature: true },
  ];
  for (const options of refused) {
    let episode: Error | null = null;
    try {
      new OpenAICompatibleProvider({ baseUrl: "https://example.invalid/v1", apiKey: "key", requestOptions: options });
    } catch (error) {
      episode = error as Error;
    }
    assert.ok(episode instanceof PolicySessionError, `the episode provider refuses ${JSON.stringify(options)}`);
    assert.throws(() => playRequestOptions({ kind: "openai_compatible", request_options: options }),
      (error: Error) => error instanceof PolicySessionError && error.message === episode!.message);
  }
  let anthropic: Error | null = null;
  try {
    buildAnthropicMessagesProvider({ baseUrl: "https://example.invalid/v1", apiKey: "key", requestOptions: OPTIONS } as Json);
  } catch (error) {
    anthropic = error as Error;
  }
  assert.ok(anthropic instanceof PolicySessionError);
  assert.throws(() => playRequestOptions({ kind: "anthropic_messages", request_options: OPTIONS }),
    (error: Error) => error instanceof PolicySessionError && error.message === anthropic!.message);
});

test("every request Pi builds for the ORA model carries them, and another provider's model none", async () => {
  server.bodies = [];
  const spec = parsePy(`{"kind": "openai_compatible", "endpoint": "${base}", "request_options": ${OPTIONS_WIRE}}`);
  const { core, events, artifacts } = fakeCore(spec, {});
  const runtime = await playRuntime(core, { other: {
    baseUrl: base, api: "openai-completions", apiKey: "other-key", models: [{ id: "other-model" }],
  } });
  const session = runtime.session;
  try {
    assert.equal(session.model.provider, "ora");
    const turn = async (text: string) => {
      await session.sendUserMessage(text);
      await session.waitForIdle();
    };
    await turn("hello");
    const other = session.modelRuntime.getModel("other", "other-model");
    await session.setModel(other);
    await turn("hello from the other model");
    // Back to the profile, as /model or Ctrl+P switches back.
    await session.setModel(session.modelRuntime.getModel("ora", "fake-play"));
    await turn("hello again");
    await session.compact();

    const [first, onOther, back, ...compaction] = server.bodies;
    assert.deepEqual([first, onOther, back].map((body) => body.model), ["fake-play", "other-model", "fake-play"]);
    // Pi's compaction: its summary requests, on the session's model (the ORA one), without tools.
    assert.ok(compaction.length >= 1 && compaction.every((body) => body.model === "fake-play" && !body.tools));
    assert.ok(compaction.some((body) => /summar/i.test(JSON.stringify(body.messages))), "a compaction summary request");
    for (const body of [first, back, ...compaction]) {
      assert.deepEqual(Object.fromEntries(OPTION_KEYS.map((key) => [key, body[key]])), OPTIONS);
    }
    for (const key of OPTION_KEYS) assert.equal(key in onOther, false, `${key} on another provider's model`);
    // ORA's request shape is otherwise unchanged: a system message, no store, max_tokens.
    assert.equal(first.messages[0].role, "system");
    assert.equal("store" in first, false);
    assert.equal(first.max_tokens, 4096);

    // The trace records each turn's request as sent, options included.
    const recorded = events.filter((event) => event.type === "provider_request");
    assert.equal(recorded.length, 3, "the three turns (Pi's compaction is recorded as conversation_compacted)");
    const payloads = recorded.map((event) => artifacts[Number(/(\d+)\.json$/.exec(event.artifact_ref.path)![1])]);
    assert.deepEqual(payloads.map((payload) => payload.model), ["fake-play", "other-model", "fake-play"]);
    assert.deepEqual(OPTION_KEYS.map((key) => payloads[0][key]), Object.values(OPTIONS));
    assert.deepEqual(OPTION_KEYS.map((key) => payloads[2][key]), Object.values(OPTIONS));
    assert.equal(OPTION_KEYS.some((key) => key in payloads[1]), false);
  } finally {
    await runtime.dispose?.();
  }
});

test("a profile without request_options adds nothing", async () => {
  server.bodies = [];
  const { core } = fakeCore({ kind: "openai_compatible", endpoint: base }, {});
  const runtime = await playRuntime(core);
  try {
    await runtime.session.sendUserMessage("hello");
    await runtime.session.waitForIdle();
    assert.equal(server.bodies.length, 1);
    for (const key of ["temperature", "top_p", "top_k", "chat_template_kwargs", "reasoning", "provider", "priority"]) {
      assert.equal(key in server.bodies[0], false, key);
    }
  } finally {
    await runtime.dispose?.();
  }
});

test("a greedy profile's temperature 0 is sent, not dropped as unset", async () => {
  // The B200 self-hosted DeepSeek profile samples greedily ({temperature: 0.0}, a Python float on the RPC).
  server.bodies = [];
  const { core } = fakeCore(parsePy(`{"kind": "openai_compatible", "endpoint": "${base}", "request_options": {"temperature": 0.0}}`), {});
  assert.deepEqual(playRequestOptions(core.config.provider), { temperature: 0 });
  const runtime = await playRuntime(core);
  try {
    await runtime.session.sendUserMessage("hello");
    await runtime.session.waitForIdle();
    assert.equal(server.bodies.length, 1);
    assert.equal(server.bodies[0].temperature, 0);
  } finally {
    await runtime.dispose?.();
  }
});

test("a plan's request_priority: correction on the call that first reads a joined message, default otherwise", async () => {
  server.bodies = [];
  const { core, events, artifacts } = fakeCore({ kind: "openai_compatible", endpoint: base }, {});
  const runtime = await playRuntime(core, { other: {
    baseUrl: base, api: "openai-completions", apiKey: "other-key", models: [{ id: "other-model" }],
  } }, { request_priority: { correction: -5, default: 3 } });
  const session = runtime.session;
  try {
    const turn = async (text: string) => {
      await session.sendUserMessage(text);
      await session.waitForIdle();
    };
    await turn("hello");
    // An operator message joined to the running instruction, first read by the next call (the `context` hook).
    runtime.oraDriver.noteContext([{ role: "custom", customType: PLAY_MESSAGE, details: { ora_message_id: "joined-1" } }]);
    await turn("the operator's correction");
    await turn("and on");
    await session.setModel(session.modelRuntime.getModel("other", "other-model"));
    await turn("hello from the other model");
    assert.deepEqual(server.bodies.map((body) => body.priority), [3, -5, 3, undefined]);
    // The trace records the payload as sent, priority included.
    const recorded = events.filter((event) => event.type === "provider_request")
      .map((event) => artifacts[Number(/(\d+)\.json$/.exec(event.artifact_ref.path)![1])]);
    assert.deepEqual(recorded.map((payload) => payload.priority), [3, -5, 3, undefined]);
  } finally {
    await runtime.dispose?.();
  }
});

test("with play.pipeline_predict the ORA model's requests go through the predictor, and still reach the server", async () => {
  server.bodies = [];
  const { core } = fakeCore(parsePy(`{"kind": "openai_compatible", "endpoint": "${base}", "request_options": {"temperature": 0.0}}`), {});
  const runtime = await playRuntime(core, {}, { pipeline_predict: true });
  try {
    assert.ok(runtime.oraDriver.predictor, "the predictor is installed");
    await runtime.session.sendUserMessage("hello");
    await runtime.session.waitForIdle();
    assert.equal(server.bodies.length, 1);
    assert.equal(server.bodies[0].temperature, 0, "the profile's options still apply");
  } finally {
    await runtime.dispose?.();
  }
  const plain = await playRuntime(fakeCore({ kind: "openai_compatible", endpoint: base }, {}).core);
  try {
    assert.equal(plain.oraDriver.predictor, null, "off by default");
  } finally {
    await plain.dispose?.();
  }
});

/**
 * The body Pi's request code builds for the registered ORA model on an OpenRouter endpoint, through
 * the runtime's own model registry, at the session's starting reasoning effort; the request is
 * answered here and never leaves the process.
 */
async function openRouterBody(requestOptions: Json, reasoningEffort: string): Promise<Json> {
  const { core } = fakeCore({ kind: "openai_compatible", endpoint: "https://openrouter.ai/api/v1", request_options: requestOptions },
    { reasoning_effort: reasoningEffort });
  const runtime = await playRuntime(core);
  try {
    const session = runtime.session;
    const model = session.modelRuntime.getModel("ora", "fake-play");
    let sent: Json = null;
    const capture = (async (_url: string, init: Json) => {
      sent = JSON.parse(String(init.body));
      return new Response("data: [DONE]\n\n", { headers: { "content-type": "text/event-stream" } });
    }) as unknown as typeof fetch;
    const reasoning = session.thinkingLevel === "off" ? undefined : session.thinkingLevel;
    await session.modelRuntime.streamSimple(model, { systemPrompt: "You control one robot.", messages: [
      { role: "user", content: "hello", timestamp: Date.now() }] }, { fetch: capture, reasoning }).result();
    assert.ok(sent !== null, "Pi built the request");
    return sent;
  } finally {
    await runtime.dispose?.();
  }
}

test("on an OpenRouter endpoint the profile's own reasoning object and Pi's effort do not collide", async () => {
  // Muse Spark Open: an effort, no reasoning object. Pi states the effort in its reasoning object, beside the options.
  const muse = await openRouterBody({ temperature: 1.0, top_p: 1.0, provider: { require_parameters: true } }, "high");
  assert.deepEqual(muse.reasoning, { effort: "high" });
  assert.deepEqual([muse.temperature, muse.top_p, muse.provider], [1, 1, { require_parameters: true }]);
  // A profile with an effort and its own reasoning object: the effort goes as reasoning_effort, as
  // ORA's provider sends it, and the profile's object is sent as named.
  const both = await openRouterBody({ reasoning: { enabled: true, exclude: false } }, "high");
  assert.equal(both.reasoning_effort, "high");
  assert.deepEqual(both.reasoning, { enabled: true, exclude: false });
  // Nemotron 3.5 Super VL: no effort (provider_default), thinking switched on by the reasoning object.
  const nemotron = await openRouterBody({ temperature: 0.6, top_p: 0.95, reasoning: { enabled: true },
    provider: { order: ["together"], allow_fallbacks: false, require_parameters: true } }, "provider_default");
  assert.equal("reasoning_effort" in nemotron, false);
  assert.deepEqual(nemotron.reasoning, { enabled: true });
  assert.deepEqual(nemotron.provider, { order: ["together"], allow_fallbacks: false, require_parameters: true });
});

test("the registered ORA model carries the options as its sampling parameters", () => {
  const { core } = fakeCore({ kind: "openai_compatible", endpoint: base }, {});
  const provider = oraProvider(core, "key", { top_k: 20 });
  assert.deepEqual(provider.models[0].samplingParams, { top_k: 20 });
  assert.equal("thinkingFormat" in provider.models[0].compat, false);
  assert.equal("samplingParams" in oraProvider(core, "key").models[0], false);
});

test("a models.json override of the ORA model's sampling is refused", async () => {
  const spec = parsePy(`{"kind": "openai_compatible", "endpoint": "${base}", "request_options": ${OPTIONS_WIRE}}`);
  const { core } = fakeCore(spec, {});
  await assert.rejects(playRuntime(core, { ora: { modelOverrides: { "fake-play": { samplingParams: { temperature: 0.2 } } } } }),
    (error: Error) => error instanceof PolicySessionError && /models\.json/.test(error.message));
});
