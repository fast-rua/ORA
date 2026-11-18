// Differential test: OpenAI-compatible wire projection, TypeScript vs Python.
// Python (tests/support/agent_core/provider/wire_cases.py) generates execution requests
// and records prepare_request canonical bytes, headers and URL; the port must
// reproduce them byte for byte.

import assert from "node:assert/strict";
import { join } from "node:path";
import { test } from "node:test";
import { canonicalBytesPy } from "../src/pyjson.ts";
import {
  OpenAICompatibleProvider, PolicySessionError, checkpointRequestOptions, normalizeUsage, prepareRequest, sha256Hex,
  toolAlias,
} from "../src/provider/index.ts";
import { canonicalText, readPy, runPython, scratchDir } from "./provider/harness.ts";
import { RolloutCheckpointCompactor } from "../src/context/rollout_compactor.ts";
import { ArtifactStore } from "../src/evidence/artifacts.ts";
import { readFileSync } from "node:fs";

const dir = scratchDir("wire");
const fixture = join(dir, "wire.json");
await runPython("wire_cases.py", [fixture]);
const data = readPy(fixture);

function errorClass(error: any): string {
  return error instanceof PolicySessionError && error.pyType === "PolicySessionError" ? "PolicySessionError" : "other";
}

/** One recorded provider case: TypeScript's wire bytes, error, headers and URL against Python's. */
function compareWireCase(item: any, failures: string[]): any | null {
  const variant = item.provider;
  let provider: OpenAICompatibleProvider;
  try {
    provider = new OpenAICompatibleProvider({
      baseUrl: variant.base_url, apiKey: data.api_key, recoveryPolicy: "same_credential_deadline_v1",
      outputTokenParameter: variant.output_token_parameter, toolStrictness: variant.tool_strictness,
      reasoningField: variant.reasoning_field, sessionAffinityId: variant.session_affinity_id,
      requestOptions: variant.request_options,
    });
  } catch (error: any) {
    if (item.construct_error !== error.message) failures.push(`case ${item.id}: construct ${error.message} != ${item.construct_error}`);
    return null;
  }
  if (item.construct_error) {
    failures.push(`case ${item.id}: Python refused construction: ${item.construct_error}`);
    return null;
  }
  let body: any;
  try {
    body = provider.prepareRequest(item.request);
  } catch (error: any) {
    const expected = item.wire_error;
    if (!expected) {
      failures.push(`case ${item.id}: TS raised ${error.message}`);
    } else if (expected.type === "PolicySessionError") {
      if (errorClass(error) !== "PolicySessionError" || error.message !== expected.message) {
        failures.push(`case ${item.id}: ${error.pyType}:${error.message} != ${expected.message}`);
      }
    } else if (errorClass(error) !== "other") {
      failures.push(`case ${item.id}: expected non-policy ${expected.type}, TS raised PolicySessionError ${error.message}`);
    }
    return null;
  }
  if (item.wire_error) {
    failures.push(`case ${item.id}: Python raised ${item.wire_error.type}: ${item.wire_error.message}`);
    return null;
  }
  const bytes = canonicalBytesPy(body);
  if (bytes.toString("utf8") !== item.wire) {
    failures.push(`case ${item.id}: wire bytes differ`);
    return null;
  }
  const headers = provider.requestHeaders();
  for (const label of ["deadline_request", "legacy_request"]) {
    const captured = item[label];
    assert.deepEqual(headers, captured.headers, `case ${item.id} ${label} headers`);
    assert.equal(`${provider.baseUrl}/chat/completions`, captured.url, `case ${item.id} ${label} url`);
    assert.equal(sha256Hex(bytes), captured.content_sha256, `case ${item.id} ${label} content`);
  }
  return body;
}

test("prepare_request wire bytes, headers and URL match Python", () => {
  let compared = 0;
  let wires = 0;
  const failures: string[] = [];
  for (const item of data.cases) {
    compared += 1;
    if (compareWireCase(item, failures) !== null) wires += 1;
  }
  assert.deepEqual(failures, []);
  assert.ok(wires > 500, `only ${wires} successful wire projections`);
  console.log(JSON.stringify({ compared, wires }));
});

test("the tool-alias disclosure joins the leading system message, byte for byte as in Python", () => {
  // Qwen3.x chat templates reject a system message after index 0, so Python and Pi
  // append <wire_tool_aliases> to the last leading system message.
  const failures: string[] = [];
  const prefixes = new Map<string, string>();
  let joined = 0;
  let refused = 0;
  for (const item of data.alias_disclosure) {
    const body = compareWireCase(item, failures);
    if (item.wire_error) {
      // Only a non-text system message is refused, and only when there is a disclosure to add.
      assert.equal(item.tools, "dotted", `${item.id}: ${item.wire_error.message}`);
      assert.ok(item.shape === "object" || item.shape === "number", item.id);
      assert.equal(item.wire_error.message, "OpenAI system message content is not text", item.id);
      refused += 1;
      continue;
    }
    if (body === null) continue;
    const messages: any[] = body.messages;
    const roles = messages.map((message) => message.role);
    const leading = roles.findIndex((role) => role !== "system");
    assert.ok(leading >= 0 && !roles.slice(leading).includes("system"), `${item.id}: a system message after index 0`);
    const text = (message: any) => typeof message.content === "string" ? message.content
      : Array.isArray(message.content) ? message.content.map((part: any) => part.text ?? "").join("\n") : "";
    const holders = messages.filter((message) => text(message).includes("<wire_tool_aliases>"));
    if (item.tools === "dotted") {
      assert.equal(leading, Math.max(item.leading_systems, 1), item.id);
      assert.deepEqual(holders, [messages[leading - 1]], `${item.id}: the disclosure is not in the last leading system message`);
      assert.ok(text(messages[leading - 1]).endsWith("\n</wire_tool_aliases>"), item.id);
      joined += 1;
    } else {
      assert.equal(leading, item.leading_systems, item.id);
      assert.deepEqual(holders, [], `${item.id}: strict names need no disclosure`);
    }
    // The provider prefix is byte-identical across the session's turns (prefix caching).
    const key = `${item.shape}-${item.tools}-${item.variant}`;
    const prefix = canonicalBytesPy(messages.slice(0, leading + 1)).toString("utf8");
    if (!prefixes.has(key)) prefixes.set(key, prefix);
    else assert.equal(prefix, prefixes.get(key), `${item.id}: the prefix changed across turns`);
  }
  assert.deepEqual(failures, []);
  assert.ok(joined >= 30 && refused >= 6, `${joined} joined disclosures, ${refused} refusals`);
});

test("request_options reach the wire, or are refused, exactly as in Python", () => {
  const failures: string[] = [];
  let wires = 0;
  for (const [index, item] of data.request_options.entries()) {
    let body: any;
    try {
      body = prepareRequest(item.request, { requestOptions: item.request_options });
    } catch (error: any) {
      const expected = item.wire_error;
      if (!expected) failures.push(`case ${index}: TS raised ${error.message}`);
      else if (expected.type === "PolicySessionError"
          && (errorClass(error) !== "PolicySessionError" || error.message !== expected.message)) {
        failures.push(`case ${index}: ${error.pyType}:${error.message} != ${expected.message}`);
      }
      continue;
    }
    if (item.wire_error) {
      failures.push(`case ${index}: Python raised ${item.wire_error.type}: ${item.wire_error.message}`);
      continue;
    }
    wires += 1;
    if (canonicalBytesPy(body).toString("utf8") !== item.wire) failures.push(`case ${index}: wire bytes differ`);
  }
  assert.deepEqual(failures, []);
  const refusals = data.request_options.filter((item: any) => item.wire_error?.message?.includes("request_options"));
  assert.ok(wires >= 8 && refusals.length >= 8, `${wires} wires, ${refusals.length} request_options refusals`);
});

test("tool_alias, normalize_usage and checkpoint options match Python", () => {
  for (const { name, alias } of data.aliases) assert.equal(toolAlias(name), alias, name);
  for (const { value, normalized } of data.usage) {
    const got = normalizeUsage(value);
    assert.equal(got === null ? null : canonicalText(got), normalized, canonicalText(value));
  }
  for (const { model, options } of data.checkpoint) {
    assert.equal(canonicalText(checkpointRequestOptions(model)), canonicalText(options), model);
  }
});

test("rollout compactor requests through an OpenAI-compatible provider match Python's", async () => {
  // The compactor reads checkpointRequestOptions from the provider instance it holds,
  // as Python reads checkpoint_request_options; DeepSeek bindings add low reasoning and JSON mode.
  assert.ok(data.compactor_requests?.length >= 8, "wire.json has no compactor request cases");
  let structured = 0;
  for (const item of data.compactor_requests) {
    const label = `${item.model} ${item.reasoning_effort} stream=${item.stream}`;
    const provider = new OpenAICompatibleProvider({ baseUrl: "https://wire.invalid/v1", apiKey: data.api_key });
    (provider as any).complete = async (_request: any, options: { modelCallId: string }) => ({
      message: { role: "assistant", content: JSON.stringify({ task_progress: [] }) }, usage: null,
      finish_reason: "stop", raw_events: [], model_call_id: options.modelCallId, started_unix_s: 0, finished_unix_s: 0,
    });
    const root = scratchDir("compactor-request");
    const events: any[] = [];
    const compactor = new RolloutCheckpointCompactor({
      provider: provider as any, model: item.model, reasoningEffort: item.reasoning_effort, stream: item.stream,
      artifacts: new ArtifactStore(root),
      eventSink: async (type: string, fields: any) => {
        events.push({ type, ...fields });
        return { event_id: `agent-event-${String(events.length).padStart(6, "0")}` };
      },
    });
    await compactor.summarize({ messages: item.messages, sourceRefs: item.source_refs });
    const attempt = events.find((event) => event.type === "context_compactor_attempt");
    const canonical = readFileSync(join(root, attempt.payload.canonical_request_ref.path), "utf8");
    const wire = readFileSync(join(root, attempt.artifact_ref.path), "utf8");
    assert.equal(canonical, item.canonical, `${label}: canonical request`);
    assert.equal(wire, item.wire, `${label}: wire request`);
    if (JSON.parse(wire).response_format) structured += 1;
  }
  assert.ok(structured >= 6, `${structured} cases carried response_format`);
});
