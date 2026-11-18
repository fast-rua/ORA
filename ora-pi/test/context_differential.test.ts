// Differential test: the TypeScript native_api Context pipeline against fixtures
// recorded from the real Python pipeline before ORA removed it with the Native
// core (it remains in the robot-harness repository).
//
//   node --test ora-pi/test/context_differential.test.ts
//
// reads the fixtures committed in tests/data/context_differential/, which cannot
// be recorded again; no Python runs here. ORA_CONTEXT_FIXTURES=<dir> uses another recorded directory
// instead, and ORA_CONTEXT_FIXTURE=<name,...> replays only those fixtures. The
// committed set leaves out the sealed LIBERO-Pro replay fixtures, which are
// recorded only with ORA_TEST_SEALED_LIBERO.

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { gunzipSync } from "node:zlib";
import { test } from "node:test";

import { firstDiff, loadFixture, replayFixture } from "./context/replay.ts";
import { REPO } from "./provider/harness.ts";
import { DEFAULT_CONTEXT_DOMAIN } from "../src/context/domain_defaults.ts";
import { RETIRED_CONTEXT_MANAGEMENT_SCHEMAS, validateContextManagement } from "../src/context/management.ts";
import {
  artifactReadName, decodeRecoverableArtifactMarker, recoverableArtifactDefinitions,
  recoverableArtifactResultSchemas,
} from "../src/context/memory/artifacts.ts";
import {
  checkpointForbiddenFragments, compactorHistoryMessage, compactorPrivateViolations, packetMessageBlocks,
  validateCheckpoint,
} from "../src/context/projection.ts";
import { compactorSystemPrompt } from "../src/context/rollout_compactor.ts";
import { CPythonIntSet, isMapping, loadsPy, pyRound, range } from "../src/context/pyutil.ts";
import { dumpsPy } from "../src/pyjson.ts";

const directory = process.env.ORA_CONTEXT_FIXTURES ?? path.join(REPO, "tests/data/context_differential");
const only = process.env.ORA_CONTEXT_FIXTURE;
const files = fs.existsSync(directory)
  ? fs.readdirSync(directory).filter((file) => file.endsWith(".json.gz")).sort() : [];
const SEALED_REPLAY = ["replay_libero_native.json.gz", "replay_libero_inflated.json.gz"];

test("fixtures exist", () => {
  assert.ok(files.length > 0, `no fixtures under ${directory}`);
});

function sealedReplaySkip(): string | false {
  if (SEALED_REPLAY.every((file) => files.includes(file))) return false;
  const summary = path.join(directory, "record-all.json");
  const recorded = fs.existsSync(summary) ? JSON.parse(fs.readFileSync(summary, "utf8")).sealed_libero_replay : null;
  return recorded?.skip ?? `${SEALED_REPLAY.join(", ")} not under ${directory}`;
}

test("sealed LIBERO-Pro replay fixtures are present", { skip: only ? "ORA_CONTEXT_FIXTURE selects fixtures" : sealedReplaySkip() }, () => {
  for (const file of SEALED_REPLAY) assert.ok(files.includes(file), file);
});

for (const file of files) {
  if (file === "python_primitives.json.gz") continue;
  const name = file.replace(/\.json\.gz$/, "");
  if (only && !only.split(",").includes(name)) continue;
  test(`python parity: ${name}`, async () => {
    const report = await replayFixture(loadFixture(path.join(directory, file)));
    assert.deepEqual(report.differences, [], report.differences.join("\n"));
  });
}

test("python primitives: set order, round, json.loads", () => {
  const file = path.join(directory, "python_primitives.json.gz");
  assert.ok(fs.existsSync(file), `${file} was not recorded`);
  const data = loadsPy(gunzipSync(fs.readFileSync(file)).toString("utf8"));
  for (const item of data.sets) {
    const set = new CPythonIntSet();
    for (const [operation, start, end] of item.operations) {
      if (operation === "update") set.update(range(start, end));
      else set.differenceUpdate(range(start, end));
    }
    assert.deepEqual(set.values(), item.order, JSON.stringify(item.operations));
  }
  for (const [value, six, three] of data.rounds) {
    const x = typeof value === "number" ? value : value.value;
    assert.ok(Object.is(pyRound(x, 6), typeof six === "number" ? six : six.value), `round(${x}, 6)`);
    assert.ok(Object.is(pyRound(x, 3), typeof three === "number" ? three : three.value), `round(${x}, 3)`);
  }
  for (const item of data.loads) {
    assert.equal(dumpsPy(loadsPy(item.text), { sortKeys: true, ensureAscii: false }), item.dumps);
  }
});

function attempt(body: () => unknown): Record<string, unknown> {
  try {
    return { value: body() };
  } catch (error) {
    return { error: { message: (error as Error).message } };
  }
}

function sameOutcome(python: Record<string, any>, typescript: Record<string, any>, label: string): void {
  if (python.error) {
    assert.ok(typescript.error, `${label}: python raised ${python.error.message}, typescript returned`);
    assert.equal((typescript.error as any).message, python.error.message, label);
  } else {
    assert.ok(!typescript.error, `${label}: typescript raised ${JSON.stringify(typescript.error)}`);
    assert.equal(firstDiff(python.value, typescript.value), null, label);
  }
}

test("python contracts: checkpoint filter, validation, management, markers, surfaces, prompt, compactor history", () => {
  const file = path.join(directory, "python_primitives.json.gz");
  assert.ok(fs.existsSync(file), `${file} was not recorded`);
  const data = loadsPy(gunzipSync(fs.readFileSync(file)).toString("utf8"));
  assert.ok(data.checkpoints, "python_primitives.json.gz has no contract cases");
  for (const item of data.checkpoints) {
    assert.deepEqual(checkpointForbiddenFragments(item.value, DEFAULT_CONTEXT_DOMAIN), item.fragments,
      JSON.stringify(item.value.task_progress));
  }
  for (const item of data.validations) {
    sameOutcome(item, attempt(() => validateCheckpoint(item.content, DEFAULT_CONTEXT_DOMAIN)), item.content);
  }
  for (const item of data.management) {
    // Python still validates v4, v5 and v7; the agent core refuses them.
    if (isMapping(item.input) && RETIRED_CONTEXT_MANAGEMENT_SCHEMAS.has(item.input.schema)) {
      assert.throws(() => validateContextManagement(item.input), /is retired from the agent core/,
        JSON.stringify(item.input));
      continue;
    }
    sameOutcome(item, attempt(() => validateContextManagement(item.input)), JSON.stringify(item.input));
  }
  for (const item of data.markers) {
    sameOutcome(item, attempt(() => decodeRecoverableArtifactMarker(item.marker)), item.marker);
  }
  const names = DEFAULT_CONTEXT_DOMAIN.canonicalToolNames;
  for (const surface of data.surfaces) {
    assert.equal(firstDiff(surface.result, recoverableArtifactDefinitions(surface.definitions, names)), null);
    assert.equal(artifactReadName(surface.definitions, names), surface.reader_name);
    assert.equal(firstDiff(surface.schemas, recoverableArtifactResultSchemas(
      { observe: { type: "object", $defs: { a: {} } } }, { canonicalToolNames: names })), null);
    assert.equal(firstDiff(surface.schemas_none, recoverableArtifactResultSchemas(
      { grasp_plan: { type: "object" } }, { canonicalToolNames: names })), null);
  }
  for (const [tokens, prompt] of Object.entries(data.prompts as Record<string, string>)) {
    assert.equal(compactorSystemPrompt(Number(tokens)), prompt);
  }
  assert.ok(data.compactor_history?.length, "python_primitives.json.gz has no compactor history cases");
  for (const item of data.compactor_history) {
    const label = `${item.message.name}: ${item.message.content}`;
    assert.equal(firstDiff(item.visible, compactorHistoryMessage(item.message, DEFAULT_CONTEXT_DOMAIN)), null, label);
    const packet = [{ role: "user", content: packetMessageBlocks("steps", [item.message], DEFAULT_CONTEXT_DOMAIN) }];
    assert.deepEqual(compactorPrivateViolations(packet, DEFAULT_CONTEXT_DOMAIN), item.violations, label);
  }
});
