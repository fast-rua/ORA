// The context stage parameter schemas (ora/policy/context_stages), as Python sends
// them to the core at session start (ora/agent/pi/domain.py context_stage_schemas),
// for tests and replays that build the context pipeline without a Python session.

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

import { parsePy } from "../../src/pyjson.ts";

const DIRECTORY = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../ora/policy/context_stages");

export function loadStageSchemas(directory = DIRECTORY): Record<string, any> {
  const schemas: Record<string, any> = {};
  for (const file of fs.readdirSync(directory).filter((name) => name.endsWith(".schema.json")).sort()) {
    schemas[file.slice(0, -".schema.json".length)] = parsePy(fs.readFileSync(path.join(directory, file), "utf8"));
  }
  return schemas;
}

export const STAGE_SCHEMAS = loadStageSchemas();
