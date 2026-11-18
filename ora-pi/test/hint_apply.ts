// Apply dumped retention hints with the TypeScript engine (tests/support/agent_core/hint_check.py drives this).
// Input: the ORA_PI_HINT_DUMP JSON lines; output: one JSON line per record with the delivered text.
import { readFileSync } from "node:fs";
import { applyHints, StaticFactsState } from "../src/core/hints.ts";
import { dumpsPy, dumpsSorted, parsePy } from "../src/pyjson.ts";

const state = new StaticFactsState();
for (const line of readFileSync(process.argv[2], "utf8").split("\n")) {
  if (!line.trim()) continue;
  const record = parsePy(line);
  const value = record.kind === "observation" ? record.envelope : record.visible;
  const local = record.kind === "observation" ? state : new StaticFactsState();
  const { delivered, slot } = applyHints(value, record.hints, local);
  process.stdout.write(dumpsPy({
    kind: record.kind, delivered_text: dumpsSorted(delivered),
    slot_text: slot === null ? null : slot.content[0].text,
  }, { ensureAscii: false }) + "\n");
}
