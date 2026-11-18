// `ora/agent/memory/index_persistence.py`: atomic, batched, deferred-durable
// rewrites of the Session Artifact index.

import * as fs from "node:fs";
import * as path from "node:path";

import { dumpsPy } from "../../pyjson.ts";
import { type Durability, makeTemp, unlinkMissingOk } from "../../evidence/artifacts.ts";
import { utf8 } from "../pyutil.ts";

/** Atomically replace ``filePath``; fsync inline or behind ``durability``. */
export function atomicJson(
  filePath: string, value: Record<string, unknown>, durability: Durability | null = null,
): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const { fd, name } = makeTemp(path.dirname(filePath), `.${path.basename(filePath)}.`);
  try {
    try {
      const data = utf8(dumpsPy(value, { ensureAscii: false, indent: 2, sortKeys: true }) + "\n");
      let offset = 0;
      while (offset < data.length) offset += fs.writeSync(fd, data, offset);
      if (durability === null) fs.fsyncSync(fd);
    } finally {
      fs.closeSync(fd);
    }
    fs.renameSync(name, filePath);
  } finally {
    unlinkMissingOk(name);
  }
  if (durability !== null) durability.submit(filePath);
}

/** ``_persist`` writes now, or once when the outermost batch exits. */
export abstract class BatchedIndexPersistence {
  private batchDepth = 0;
  private batchDirty = false;

  protected abstract writeIndex(): void;

  protected persist(): void {
    if (this.batchDepth) {
      this.batchDirty = true;
      return;
    }
    this.writeIndex();
  }

  /** Coalesce index rewrites inside one projection into a single write. */
  async batchedPersistence<T>(body: () => Promise<T>): Promise<T> {
    this.batchDepth += 1;
    try {
      return await body();
    } finally {
      this.batchDepth -= 1;
      if (this.batchDepth === 0 && this.batchDirty) {
        this.batchDirty = false;
        this.writeIndex();
      }
    }
  }

  /** Synchronous variant for callers without an async body. */
  batchedPersistenceSync<T>(body: () => T): T {
    this.batchDepth += 1;
    try {
      return body();
    } finally {
      this.batchDepth -= 1;
      if (this.batchDepth === 0 && this.batchDirty) {
        this.batchDirty = false;
        this.writeIndex();
      }
    }
  }
}
