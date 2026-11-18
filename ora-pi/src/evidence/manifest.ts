// native/agent-session.json, the TypeScript twin of the retired Python Native
// core's ora/agent/native/session_manifest.py: mid-run snapshots at most every
// 120 s (latest wins, not fsynced), and the closed manifest serialized exactly
// as `json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n"`
// and fsynced behind the close barrier.

import { renameSync } from "node:fs";
import { performance } from "node:perf_hooks";

import { dumpsPy, writePy } from "../pyjson.ts";
import { writeTextFile, type Durability, type TextSink } from "./durability.ts";

export const SNAPSHOT_MIN_INTERVAL_MS = 120_000;

const MANIFEST_JSON = { ensureAscii: false, indent: 2, sortKeys: true } as const;

export function dumpManifestText(body: unknown): string {
  return dumpsPy(body, MANIFEST_JSON) + "\n";
}

/**
 * `dumpManifestText`, streamed: every call record carries its context
 * projection, so a long episode's manifest is ~100 MB and is written piece by
 * piece instead of being built whole in memory.
 */
function writeManifest(body: unknown, write: TextSink): void {
  writePy(body, MANIFEST_JSON, write);
  write("\n");
}

export class SessionManifestWriter {
  private lastSnapshot: number | null = null;
  skippedSnapshots = 0;

  readonly target: string;
  readonly durability: Durability;
  readonly minIntervalMs: number;

  constructor(target: string, durability: Durability, minIntervalMs: number = SNAPSHOT_MIN_INTERVAL_MS) {
    this.target = target;
    this.durability = durability;
    this.minIntervalMs = minIntervalMs;
  }

  write(body: unknown, final: boolean): void {
    if (!final) {
      const now = performance.now();
      if (this.lastSnapshot !== null && now - this.lastSnapshot < this.minIntervalMs) {
        this.skippedSnapshots += 1;
        return;
      }
      this.lastSnapshot = now;
      this.durability.replaceStreamed(this.target, (write) => writeManifest(body, write));
      return;
    }
    const temporary = this.target.replace(/\.json$/, ".json.tmp");
    writeTextFile(temporary, (write) => writeManifest(body, write));
    renameSync(temporary, this.target);
    this.durability.submit(this.target);
  }
}
