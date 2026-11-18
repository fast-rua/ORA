// Opt-in phase timing (ORA_PI_PROFILE=1): accumulated milliseconds per phase,
// written to stderr (native/pi-core.log) at close. Opt-in memory samples
// (ORA_PI_MEMORY_LOG=1, or =gc to collect garbage first and so measure the
// live heap): process.memoryUsage() plus the caller's size breakdown, one JSON
// line per sample in native/pi-memory.jsonl.

import { appendFileSync, writeFileSync } from "node:fs";
import { Session } from "node:inspector";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { setFlagsFromString } from "node:v8";
import { runInNewContext } from "node:vm";

const enabled = process.env.ORA_PI_PROFILE === "1";
const totals = new Map<string, { ms: number; count: number }>();

export async function timed<T>(phase: string, body: () => Promise<T> | T): Promise<T> {
  if (!enabled) return await body();
  const started = performance.now();
  try {
    return await body();
  } finally {
    const entry = totals.get(phase) ?? { ms: 0, count: 0 };
    entry.ms += performance.now() - started;
    entry.count += 1;
    totals.set(phase, entry);
  }
}

export function report(): void {
  if (!enabled) return;
  const rows = [...totals.entries()].sort((a, b) => b[1].ms - a[1].ms)
    .map(([phase, { ms, count }]) => ({ phase, ms: Math.round(ms), count, mean_ms: Math.round((ms / count) * 10) / 10 }));
  process.stderr.write(`ora-pi profile ${JSON.stringify(rows)}\n`);
}

const memoryLog = process.env.ORA_PI_MEMORY_LOG ?? "";
let collect: (() => void) | null = null;

export function sampleMemory(runRoot: string, label: string, detail: () => Record<string, unknown> = () => ({})): void {
  if (!memoryLog) return;
  if (memoryLog === "gc") {
    if (collect === null) {
      setFlagsFromString("--expose-gc");
      collect = runInNewContext("gc") as () => void;
    }
    collect();
  }
  const usage = process.memoryUsage();
  appendFileSync(join(runRoot, "native", "pi-memory.jsonl"), JSON.stringify({
    label, unix_s: Date.now() / 1000, rss: usage.rss, heap_used: usage.heapUsed, heap_total: usage.heapTotal,
    external: usage.external, array_buffers: usage.arrayBuffers, ...detail(),
  }) + "\n");
}

/**
 * ORA_PI_MEMORY_LOG=ablate, at the end of a run: release each named holder in
 * turn, collecting garbage after each, and log how much live heap it held.
 */
export function ablateMemory(runRoot: string, holders: Array<[string, () => void]>): void {
  if (memoryLog !== "ablate") return;
  setFlagsFromString("--expose-gc");
  const gc = runInNewContext("gc") as () => void;
  gc();
  let before = process.memoryUsage().heapUsed;
  const held: Record<string, number> = { live_heap: before };
  for (const [name, release] of holders) {
    release();
    gc();
    const after = process.memoryUsage().heapUsed;
    held[name] = before - after;
    before = after;
  }
  held.remaining = before;
  appendFileSync(join(runRoot, "native", "pi-memory.jsonl"), JSON.stringify({ label: "ablation", ...held }) + "\n");
}

// ORA_PI_MEMORY_LOG=alloc: V8's sampling heap profiler over the whole run,
// counting objects garbage collection already freed, written at close to
// native/pi-alloc.heapprofile (needs ORA_PI_NO_PERMISSION=1 for the inspector).
let allocations: Session | null = null;
if (memoryLog === "alloc") {
  allocations = new Session();
  allocations.connect();
  allocations.post("HeapProfiler.startSampling", {
    samplingInterval: 65536, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true,
  } as any);
}

export function writeAllocationProfile(runRoot: string): void {
  if (allocations === null) return;
  allocations.post("HeapProfiler.stopSampling", (error, result: any) => {
    if (!error) writeFileSync(join(runRoot, "native", "pi-alloc.heapprofile"), JSON.stringify(result.profile));
  });
}
