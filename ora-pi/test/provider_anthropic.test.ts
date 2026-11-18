// Differential test: the Anthropic Messages port vs Python.
// Python (tests/support/agent_core/provider/anthropic_cases.py) records exact answers for
// translate_anthropic_request (randomized edges, plus the real requests of the
// sealed runs ORA_TEST_SEALED_* name; each unset one is skipped with the reason),
// _parse_events over valid and corrupted streams, the continuation round trip,
// normalize_anthropic_usage and the Kimi signature check.

import assert from "node:assert/strict";
import { join } from "node:path";
import { test } from "node:test";
import { canonicalBytesPy } from "../src/pyjson.ts";
import {
  PolicySessionError, declaredPropertyTypes, hasCompleteThinkingSignature, normalizeAnthropicUsage,
  parseAnthropicEvents, sha256Hex, translateAnthropicRequest,
} from "../src/provider/index.ts";
import { canonicalText, readPy, runPython, scratchDir, sealedCache } from "./provider/harness.ts";

const dir = scratchDir("anthropic");
const fixture = join(dir, "anthropic.json");
const sealed = ["ORA_TEST_SEALED_LIBERO", "ORA_TEST_SEALED_ROBOTWIN", "ORA_TEST_SEALED_ROBODOJO"]
  .some((name) => process.env[name]);
await runPython("anthropic_cases.py", [
  fixture, ...(process.env.ORA_ANTHROPIC_COUNT ? ["--count", process.env.ORA_ANTHROPIC_COUNT] : []),
  ...(sealed ? ["--sealed-cache", sealedCache()] : []),
]);
const data = readPy(fixture);

function translateAnswer(request: any): any {
  let body: any;
  try {
    body = translateAnthropicRequest(request);
  } catch (error: any) {
    if (error instanceof PolicySessionError && error.pyType === "PolicySessionError") return { error: error.message };
    return { crash: error?.pyType ?? error?.name ?? "Error", detail: String(error?.message) };
  }
  const bytes = canonicalBytesPy(body);
  return { sha256: sha256Hex(bytes), bytes: bytes.length, text: bytes.toString("utf8") };
}

function compareTranslate(cases: any[], label: string): void {
  const failures: string[] = [];
  for (const item of cases) {
    const got = translateAnswer(item.request);
    const want = item.answer;
    if (want.crash !== undefined) {
      if (got.crash === undefined && got.error === undefined) failures.push(`${item.id}: Python crashed (${want.crash}), TS returned a body`);
      continue;
    }
    if (want.error !== undefined) {
      if (got.error !== want.error) failures.push(`${item.id}: error ${JSON.stringify(got.error ?? got.crash)} != ${JSON.stringify(want.error)}`);
      continue;
    }
    if (got.sha256 !== want.sha256) {
      let detail = got.error ?? got.crash ?? "";
      if (want.text !== undefined && got.text !== undefined) {
        let index = 0;
        while (index < want.text.length && want.text[index] === got.text[index]) index += 1;
        detail = `first difference at ${index}: py ${JSON.stringify(want.text.slice(index, index + 80))} ts ${JSON.stringify(got.text.slice(index, index + 80))}`;
      }
      failures.push(`${item.id}: sha differs (${got.bytes} vs ${want.bytes} bytes) ${detail}`);
    }
  }
  assert.deepEqual(failures.slice(0, 15), [], `${label}: ${failures.length}/${cases.length} differ`);
}

test("translate_anthropic_request matches Python (randomized edges)", () => {
  const edges = data.translate.filter((item: any) => !item.id.startsWith("real:"));
  assert.ok(edges.length > 50);
  compareTranslate(edges, "translate");
});

for (const [env, source] of Object.entries(data.real_sources as Record<string, any>)) {
  test(`translate_anthropic_request matches Python (real ${env} requests)`, { skip: source.skip ?? false }, () => {
    const real = data.translate.filter((item: any) => item.id.startsWith(`real:${env}:`));
    assert.ok(real.length > 0 && real.length === source.cases, `${env}: ${real.length} cases`);
    compareTranslate(real, `translate real ${env}`);
  });
}

test("continuation round trip matches Python", () => {
  assert.ok(data.continuation.length > 5);
  compareTranslate(data.continuation, "continuation");
});

test("_parse_events matches Python on valid and corrupted streams", () => {
  const failures: string[] = [];
  for (const item of data.parse) {
    const aliases = item.aliases === null ? null : new Map<string, string>(Object.entries(item.aliases));
    const declared = item.declared_tools === null ? null : declaredPropertyTypes(item.declared_tools);
    const index = Number(item.id.split(":")[1]);
    let got: any;
    try {
      const reply = parseAnthropicEvents(item.events, {
        modelCallId: `model-call-${String(index).padStart(6, "0")}`, model: item.model,
        toolNamesByAlias: aliases, declaredTypesByTool: declared,
      });
      got = {
        message: canonicalText(reply.message), usage: canonicalText(reply.usage), finish: reply.finishReason,
        rejection: canonicalText(reply.outputRejection),
      };
    } catch (error: any) {
      got = error instanceof PolicySessionError
        ? { error: error.message, retryable: error.retryable ?? null }
        : { crash: error?.pyType ?? error?.name, message: error?.message };
    }
    const want = item.answer;
    if (want.crash !== undefined) {
      if (got.crash === undefined) failures.push(`${item.id}: Python crashed ${want.crash}: ${want.message}; TS ${JSON.stringify(got).slice(0, 120)}`);
      continue;
    }
    for (const key of Object.keys(want)) {
      if (JSON.stringify(got[key] ?? null) !== JSON.stringify(want[key] ?? null)) {
        failures.push(`${item.id}: ${key} ${JSON.stringify(got[key]).slice(0, 160)} != ${JSON.stringify(want[key]).slice(0, 160)}`);
        break;
      }
    }
  }
  assert.deepEqual(failures.slice(0, 15), [], `${failures.length}/${data.parse.length} differ`);
});

test("normalize_anthropic_usage matches Python", () => {
  const failures = data.usage.filter((item: any) => canonicalText(normalizeAnthropicUsage(item.value)) !== item.answer)
    .map((item: any) => `${JSON.stringify(item.value)}: ${canonicalText(normalizeAnthropicUsage(item.value))} != ${item.answer}`);
  assert.deepEqual(failures, []);
});

test("the Kimi signature check matches Python", () => {
  const failures = data.signature.filter((item: any) => hasCompleteThinkingSignature(item.body) !== item.answer)
    .map((item: any) => JSON.stringify(item.body).slice(0, 120));
  assert.deepEqual(failures, []);
});
