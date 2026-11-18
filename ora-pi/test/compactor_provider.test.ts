// The rollout compactor's provider is its own instance with no retries, as in Python.
//
// The retired Python NativePlanner gave the compactor a shallow copy of the policy provider
// with max_retries=0 (same client, same wire options). The Pi core used to hand the
// compactor the policy provider itself (max_retries 2). Here: the compactor gets a
// separate instance of the plan's provider with maxRetries 0, the same endpoint,
// model options and HTTP client, and a scripted 503 is not retried for it while the
// policy provider still retries the same failure. A compactor bound to its own profile
// (policy.context_compactor) gets that profile's provider instead: its own key, HTTP
// client and session-affinity id, no retries.

import assert from "node:assert/strict";
import { test } from "node:test";

import { compactorProvider } from "../src/core/context_wiring.ts";
import { createProvider } from "../src/core/deps.ts";
import { AnthropicMessagesProvider, FakeProvider, OpenAICompatibleProvider } from "../src/provider/index.ts";
import { FakeServer, type Step } from "./provider/fake_server.ts";

const KEY = "sk-fake-compactor-key-0123456789";
const REPLY = JSON.stringify({
  id: "c-1", object: "chat.completion", model: "deepseek-flash",
  choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: "{\"task_progress\":[]}" } }],
  usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
});
const BUSY: Step = { status: 503, body: "busy" };
const OK: Step = { json: REPLY };
const REQUEST = { model: "deepseek-flash", messages: [{ role: "user", content: "summarize" }], max_tokens: 64, stream: false };

function config(provider: Record<string, any>): any {
  return { provider, policy: { model: "deepseek-flash" } };
}

test("the compactor gets its own provider instance with maxRetries 0 and the policy's options", () => {
  process.env.ORA_PI_API_KEY = KEY;
  const spec = { kind: "openai_compatible", endpoint: "https://wire.invalid/v1", timeout_s: 120,
    request_options: { temperature: 0.2 }, session_affinity_id: "run-1" };
  const main = createProvider(config(spec)) as unknown as OpenAICompatibleProvider;
  const compactor = compactorProvider(config(spec), main as any) as unknown as OpenAICompatibleProvider;
  assert.ok(compactor instanceof OpenAICompatibleProvider);
  assert.notEqual(compactor, main);
  assert.deepEqual([main.maxRetries, compactor.maxRetries], [2, 0]);
  assert.equal(compactor.fetchImpl, main.fetchImpl, "the HTTP client is shared, as Python's copy shares it");
  for (const field of ["baseUrl", "timeoutSeconds", "recoveryPolicy", "sessionAffinityId", "outputTokenParameter",
    "toolStrictness", "reasoningField"] as const) {
    assert.deepEqual(compactor[field], main[field], field);
  }
  assert.deepEqual(compactor.prepareRequest(REQUEST), main.prepareRequest(REQUEST));

  const anthropicSpec = { kind: "anthropic_messages", endpoint: "https://wire.invalid", timeout_s: 120 };
  const anthropic = createProvider(config(anthropicSpec)) as unknown as AnthropicMessagesProvider;
  const anthropicCompactor = compactorProvider(config(anthropicSpec), anthropic as any) as unknown as AnthropicMessagesProvider;
  assert.ok(anthropicCompactor instanceof AnthropicMessagesProvider);
  assert.notEqual(anthropicCompactor, anthropic);
  assert.deepEqual([anthropic.maxRetries, anthropicCompactor.maxRetries], [2, 0]);

  // A scripted run keeps its scripted compactor.
  const fake = compactorProvider(config({ kind: "fake", compactor_responses: [] }),
    createProvider(config({ kind: "fake", responses: [] })) as any);
  assert.ok(fake instanceof FakeProvider);
});

test("a failed compaction request is not retried; the policy's request still is", async () => {
  process.env.ORA_PI_API_KEY = KEY;
  const server = new FakeServer({ compactor: [BUSY, OK], policy: [BUSY, OK] });
  const base = await server.start();
  try {
    const spec = (scenario: string) => ({ kind: "openai_compatible", endpoint: `${base}/${scenario}/ts/v1`, timeout_s: 30 });
    const main = createProvider(config(spec("policy")));
    const compactor = compactorProvider(config(spec("compactor")), createProvider(config(spec("compactor"))) as any);
    await assert.rejects(compactor.complete(structuredClone(REQUEST), { modelCallId: "context-compactor-000000" }),
      /HTTP 503/);
    assert.equal(server.seen.get("compactor/ts")?.length, 1, "one physical attempt");
    const reply = await main.complete(structuredClone(REQUEST), { modelCallId: "model-call-000000" } as any);
    assert.equal(reply.message.content, "{\"task_progress\":[]}");
    assert.equal(server.seen.get("policy/ts")?.length, 2, "the policy provider retried");
  } finally {
    await server.stop();
  }
});

test("a compactor bound to its own profile gets that profile's provider, key and client", () => {
  process.env.ORA_PI_API_KEY = KEY;
  process.env.ORA_PI_COMPACTOR_API_KEY = "sk-fake-bound-compactor-key-jihgfe";
  try {
    const policySpec = { kind: "anthropic_messages", endpoint: "http://127.0.0.1:9/a", timeout_s: 120 };
    const bound = { kind: "openai_compatible", endpoint: "http://127.0.0.1:9/b/v1", timeout_s: 120,
      recovery_policy: "legacy_v1", recovery_window_s: null, auth_scheme: "bearer", output_token_parameter: null,
      tool_strictness: null, reasoning_field: null, request_options: { temperature: 0.2 },
      session_affinity_id: "ora:run-1:compactor" };
    const bindingConfig = { ...config(policySpec), compactor_provider: bound };
    const main = createProvider(bindingConfig) as unknown as AnthropicMessagesProvider;
    const compactor = compactorProvider(bindingConfig, main as any) as unknown as OpenAICompatibleProvider;
    assert.ok(compactor instanceof OpenAICompatibleProvider, "the binding's protocol, not the policy's");
    assert.deepEqual([compactor.baseUrl, compactor.apiKey, compactor.maxRetries, compactor.sessionAffinityId],
      ["http://127.0.0.1:9/b/v1", "sk-fake-bound-compactor-key-jihgfe", 0, "ora:run-1:compactor"]);
    assert.notEqual(compactor.fetchImpl, (main as any).fetchImpl, "its own HTTP client");
    assert.deepEqual(compactor.requestOptions, { temperature: 0.2 });
    assert.equal((main as any).apiKey, KEY, "the policy keeps its own key");
    // A scripted run keeps its scripted compactor even with a binding.
    const fake = compactorProvider({ ...config({ kind: "fake", compactor_responses: [] }), compactor_provider: bound },
      createProvider(config({ kind: "fake", responses: [] })) as any);
    assert.ok(fake instanceof FakeProvider);
  } finally {
    delete process.env.ORA_PI_COMPACTOR_API_KEY;
  }
});
