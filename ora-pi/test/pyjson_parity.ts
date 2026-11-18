// Reads cases written by tests/agent_core/test_pyjson_parity.py and re-serializes them.
import { readFileSync } from "node:fs";
import { canonicalBytesPy, dumpsPy, parsePy } from "../src/pyjson.ts";
import { jcsString } from "../src/jcs.ts";

const cases = JSON.parse(readFileSync(process.argv[2], "utf8")) as Array<{
  text: string; options: any; expected: string; canonical: string; jcs: string | null;
}>;
let failures = 0;
for (const [index, item] of cases.entries()) {
  const value = parsePy(item.text);
  const got = dumpsPy(value, item.options);
  const canonical = canonicalBytesPy(value).toString("utf8");
  let jcs: string | null = null;
  try { jcs = jcsString(value); } catch { jcs = null; }
  for (const [label, a, b] of [["dumps", got, item.expected], ["canonical", canonical, item.canonical], ["jcs", jcs, item.jcs]] as const) {
    if (a !== b) {
      failures += 1;
      if (failures < 10) console.log(`case ${index} ${label} mismatch\n  py: ${String(b).slice(0, 300)}\n  ts: ${String(a).slice(0, 300)}`);
    }
  }
}
console.log(JSON.stringify({ cases: cases.length, failures }));
process.exit(failures ? 1 : 0);
