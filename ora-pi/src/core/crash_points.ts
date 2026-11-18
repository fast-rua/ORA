// Test-only crash points for the agent-process restart tests.
//
// ORA_PI_CRASH_AT=<point>:<n> SIGKILLs this process the n-th time it reaches
// <point>, once per run: a marker in native/ keeps the restarted process from
// crashing again. Unset (every real run), crashPoint() does nothing.

import { existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const target = process.env.ORA_PI_CRASH_AT ?? "";
const counts = new Map<string, number>();

export function crashPoint(point: string, runRoot: string): void {
  if (!target) return;
  const [name, nth = "1"] = target.split(":");
  if (name !== point) return;
  const count = (counts.get(point) ?? 0) + 1;
  counts.set(point, count);
  if (count !== Number(nth)) return;
  const marker = join(runRoot, "native", "pi-crash-fired");
  if (existsSync(marker)) return;
  writeFileSync(marker, `${target}\n`);
  process.kill(process.pid, "SIGKILL");
}
