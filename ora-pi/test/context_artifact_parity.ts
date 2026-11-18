// tests/agent_core/test_context_artifact_parity.py's operations on the TypeScript Context
// Artifact catalog: each outcome (the public ref, or the ValueError message) and the index.
//
//   node ora-pi/test/context_artifact_parity.ts <operations.json> <run root>

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { ContextArtifactCatalog } from "../src/context/memory/artifacts.ts";
import { PyValueError } from "../src/context/pyutil.ts";
import { ArtifactStore, NO_DURABILITY } from "../src/evidence/artifacts.ts";

const [operationsPath, root] = process.argv.slice(2);
const artifacts = new ArtifactStore(root, { durability: NO_DURABILITY });
const catalog = new ContextArtifactCatalog(root, "policy-session-1", artifacts);
const outcomes: unknown[] = [];
for (const [operation, data, storageKind, suffix, mediaType, kind, roundIndex, actionStep]
  of JSON.parse(readFileSync(operationsPath, "utf8"))) {
  const bytes = Buffer.from(data, "utf8");
  try {
    if (operation === "register") {
      const storage = artifacts.putBytes(bytes, { kind: storageKind, suffix, mediaType });
      outcomes.push(catalog.registerStorageRef(storage, { kind, roundIndex, actionStep }));
    } else {
      outcomes.push(catalog.putBytes(bytes, { kind, suffix, mediaType, roundIndex, actionStep }));
    }
  } catch (error) {
    if (!(error instanceof PyValueError)) throw error;
    outcomes.push({ error: error.message });
  }
}
const index = JSON.parse(readFileSync(join(root, "native/context-artifacts/index.json"), "utf8"));
process.stdout.write(JSON.stringify({ outcomes, index }));
