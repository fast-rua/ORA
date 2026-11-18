// Deferred evidence durability, ported from the retired Python Native core's
// evidence/durability.py: writes, renames and links happen in order
// on the calling turn; fsync runs on libuv's thread pool and `barrier()`
// awaits every fsync submitted before it. Failures are sticky.

import { closeSync, fsyncSync, openSync, renameSync, unlinkSync, writeSync } from "node:fs";
import { open } from "node:fs/promises";
import { dirname, basename, join } from "node:path";

export class DurabilityError extends Error {}

/** A text sink over an open file that writes in bounded pieces. */
export type TextSink = (text: string) => void;

/**
 * Write a file through `fill`, buffered in pieces of about 1 MiB, so a large
 * document never exists whole in memory (as text or as its UTF-8 bytes).
 */
export function writeTextFile(path: string, fill: (write: TextSink) => void): void {
  const descriptor = openSync(path, "w");
  try {
    let pending: string[] = [];
    let size = 0;
    const flush = () => {
      if (pending.length) writeSync(descriptor, pending.join(""));
      pending = [];
      size = 0;
    };
    fill((text) => {
      pending.push(text);
      size += text.length;
      if (size >= 1 << 20) flush();
    });
    flush();
  } catch (error) {
    closeSync(descriptor);
    try {
      unlinkSync(path);
    } catch {
      // the partial file is gone already
    }
    throw error;
  }
  closeSync(descriptor);
}

export interface Durability {
  submit(path: string): void;
  replaceText(path: string, text: string): void;
  replaceStreamed(path: string, fill: (write: TextSink) => void): void;
  barrier(): Promise<void>;
  close(): Promise<void>;
}

export class DeferredDurability implements Durability {
  private pending = new Map<string, Promise<void>>();
  private failure: Error | null = null;

  private check(): void {
    if (this.failure) throw new DurabilityError(`evidence is not durable: ${this.failure.message}`);
  }

  submit(path: string): void {
    this.check();
    if (this.pending.has(path)) return;
    const task = (async () => {
      const handle = await open(path, "r");
      try {
        await handle.sync();
      } finally {
        await handle.close();
      }
    })().catch((error: Error) => {
      this.failure ??= error;
    }).finally(() => {
      this.pending.delete(path);
    });
    this.pending.set(path, task);
  }

  replaceText(path: string, text: string): void {
    this.replaceStreamed(path, (write) => write(text));
  }

  replaceStreamed(path: string, fill: (write: TextSink) => void): void {
    this.check();
    const temporary = join(dirname(path), `.${basename(path)}.${process.pid}.tmp`);
    writeTextFile(temporary, fill);
    renameSync(temporary, path);
    this.submit(path);
  }

  async barrier(): Promise<void> {
    while (this.pending.size) {
      await Promise.all([...this.pending.values()]);
    }
    this.check();
  }

  async close(): Promise<void> {
    await this.barrier();
  }
}

/** Synchronous fsync for the few places that must not return before durability. */
export function fsyncPath(path: string): void {
  const descriptor = openSync(path, "r");
  try {
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}
