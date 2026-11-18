// The session manifest writer (src/evidence/manifest.ts): a streamed snapshot
// and the closed manifest are exactly the one-shot reference serialization
// (`dumpManifestText`, Python's json.dumps(ensure_ascii=False, indent=2,
// sort_keys=True) + "\n", which tests/agent_core/test_pyjson_parity.py holds to
// Python), mid-run snapshots keep the minimum interval, and the closed manifest
// is written whole, renamed into place and handed to durability. Ported from the
// retired Python Native core's session-manifest tests.
//
//   node --test ora-pi/test/session_manifest.test.ts

import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import type { Durability, TextSink } from "../src/evidence/durability.ts";
import { SessionManifestWriter, dumpManifestText } from "../src/evidence/manifest.ts";

type Json = any;

/** Every snapshot and submission the writer hands to durability, in order. */
class Recorded implements Durability {
  snapshots: string[] = [];
  submitted: string[] = [];
  submit(path: string): void {
    this.submitted.push(path);
  }
  replaceText(_path: string, text: string): void {
    this.snapshots.push(text);
  }
  replaceStreamed(_path: string, fill: (write: TextSink) => void): void {
    const parts: string[] = [];
    fill((text) => parts.push(text));
    this.snapshots.push(parts.join(""));
  }
  async barrier(): Promise<void> {}
  async close(): Promise<void> {}
}

/** mulberry32: a seeded generator, so a failing body reproduces. */
function generator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const STRINGS = ["", "plain", "line\nbreak", "tab\tquote\"", "ünïcødé 中文", "@@ora-splice-looks-like-a-marker@@", " ", "{\n}"];
const KEYS = ["b", "a", "ü", "calls", "z\n", ""];

function value(next: () => number, depth = 0): Json {
  const pick = (count: number) => Math.floor(next() * count);
  const choice = pick(depth < 4 ? 9 : 5);
  if (choice === 0) return pick(2_000_000) - 1_000_000;
  if (choice === 1) return next() * 1e3;
  if (choice === 2) return [true, false, null][pick(3)];
  if (choice <= 4) return STRINGS[pick(STRINGS.length)];
  if (choice <= 6) return Array.from({ length: pick(4) }, () => value(next, depth + 1));
  const out: Record<string, Json> = {};
  const size = pick(4);
  for (let index = 0; index < size; index++) out[KEYS[pick(KEYS.length)] + index] = value(next, depth + 1);
  return out;
}

/** The bodies a session writes as it goes: calls and usage events grow, are replaced and truncated. */
function bodies(seed: number, steps: number): Json[] {
  const next = generator(seed);
  const calls: Json[] = [];
  const usage: Json[] = [];
  const out: Json[] = [];
  for (let step = 0; step < steps; step++) {
    if (next() < 0.8) calls.push({ turn_id: `model-call-${String(step).padStart(6, "0")}`, x: value(next) });
    if (next() < 0.6) usage.push({ sequence: usage.length, last: value(next) });
    if (calls.length && next() < 0.1) calls[Math.floor(next() * calls.length)] = { replaced: value(next) };
    if (calls.length && next() < 0.05) calls.length = Math.floor(next() * calls.length);
    out.push({
      schema: "ora.native-policy-session.v1", calls: structuredClone(calls), usage_events: structuredClone(usage),
      context_compactor_calls: [], reported_totals: { input_tokens: Math.floor(next() * 1e6) }, exit_reason: null,
      memory_projection: value(next),
    });
  }
  return out;
}

const scratch = () => mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), "ora-manifest-"));

test("every snapshot is the reference serialization", () => {
  const durability = new Recorded();
  const writer = new SessionManifestWriter(join(scratch(), "agent-session.json"), durability, 0);
  const written = [
    ...bodies(20260927, 60),
    { calls: [], usage_events: [] }, { calls: [{}], usage_events: [[]] }, { other: 1 },
    { calls: "not-a-list", usage_events: null },
  ];
  for (const body of written) writer.write(body, false);
  assert.equal(durability.snapshots.length, written.length);
  written.forEach((body, index) => assert.equal(durability.snapshots[index], dumpManifestText(body), `body ${index}`));
  assert.equal(writer.skippedSnapshots, 0);
});

test("mid-run snapshots keep the minimum interval; the closed manifest is always written", () => {
  const directory = scratch();
  const target = join(directory, "agent-session.json");
  const durability = new Recorded();
  const writer = new SessionManifestWriter(target, durability, 60_000);
  const [first, ...later] = bodies(7, 4);
  writer.write(first, false);
  for (const body of later) writer.write(body, false);
  // The first snapshot is written; the three inside the interval are skipped.
  assert.deepEqual(durability.snapshots, [dumpManifestText(first)]);
  assert.equal(writer.skippedSnapshots, 3);
  const closed = { ...later[later.length - 1], exit_reason: "closed" };
  writer.write(closed, true);
  assert.equal(readFileSync(target, "utf8"), dumpManifestText(closed));
  assert.deepEqual(durability.submitted, [target]);
  assert.equal(writer.skippedSnapshots, 3);
  assert.deepEqual(readdirSync(directory), ["agent-session.json"]);  // no temporary file left
});
