// `ora/agent/context/lossless/request_size.py`: exact canonical-JSON sizes that do not
// re-serialize inline media on every projection.
//
// A value's canonical length equals the length of the same value with every
// large `data:` string replaced by "" plus, per replaced string, its own
// canonical length minus the two quote bytes. Strings are memoized by value and
// messages by identity (lossless passes are copy-on-write), so the numbers stay
// byte-exact while each projection pays only for text and newly seen media.

import { canonicalScalarLength, canonicalStringLength } from "./canonical.ts";
import { PyFloat, isMapping, sortedStrings } from "./pyutil.ts";

export const MEMOIZED_STRING_MIN_CHARS = 1024;

export type Measured = [number, Set<string>];

/** Keep only entries used by the current or immediately previous projection. */
export class GenerationalMemo<K, V> {
  private current = new Map<K, V>();
  private previous = new Map<K, V>();

  rotate(): void {
    this.previous = this.current;
    this.current = new Map();
  }

  get(key: K): V | undefined {
    if (this.current.has(key)) return this.current.get(key);
    if (this.previous.has(key)) {
      const value = this.previous.get(key)!;
      this.previous.delete(key);
      this.current.set(key, value);
      return value;
    }
    return undefined;
  }

  put(key: K, value: V): void {
    this.current.set(key, value);
  }

  discard(key: K): void {
    this.current.delete(key);
    this.previous.delete(key);
  }
}

/** Measure canonical length and source-event identities in one walk. */
export class CanonicalSizer {
  private lengths = new GenerationalMemo<string, number>();
  private elements = new WeakMap<object, [number, Set<string>]>();

  rotate(): void {
    this.lengths.rotate();
    this.elements = new WeakMap();
  }

  private stringLength(value: string): number {
    let cached = this.lengths.get(value);
    if (cached === undefined) {
      cached = canonicalStringLength(value);
      this.lengths.put(value, cached);
    }
    return cached;
  }

  /** ``(len(canonical_bytes(value)), recursive _source_event_ids)``. */
  measure(value: unknown): Measured {
    if (Array.isArray(value) && value.length && value.every((item) => isMapping(item))) {
      return this.measureMessageList(value as object[]);
    }
    return this.measureValue(value);
  }

  private measureMessageList(value: object[]): Measured {
    let total = 1 + value.length;
    const found = new Set<string>();
    for (const item of value) {
      let cached = this.elements.get(item);
      if (cached === undefined) {
        cached = this.measureValue(item);
        this.elements.set(item, cached);
      }
      total += cached[0];
      for (const id of cached[1]) found.add(id);
    }
    return [total, found];
  }

  private measureValue(value: unknown): Measured {
    // Counted in one walk, without a scrubbed copy or its serialization:
    // a memoized data URL contributes its own canonical length.
    const found = new Set<string>();
    const length = (item: unknown): number => {
      if (typeof item === "string") {
        return item.length >= MEMOIZED_STRING_MIN_CHARS && item.startsWith("data:")
          ? this.stringLength(item) : canonicalStringLength(item);
      }
      if (item === null || item === undefined || typeof item !== "object" || item instanceof PyFloat) {
        return canonicalScalarLength(item);
      }
      if (Array.isArray(item)) {
        if (!item.length) return 2;
        let total = 1 + item.length;
        for (const child of item) total += length(child);
        return total;
      }
      const raw = (item as Record<string, any>)._source_event_ids;
      if (Array.isArray(raw) && Object.prototype.hasOwnProperty.call(item, "_source_event_ids")) {
        for (const child of raw) if (typeof child === "string") found.add(child);
      }
      let total = 1;
      let members = 0;
      for (const key of Object.keys(item)) {
        const child = (item as Record<string, unknown>)[key];
        if (child === undefined) continue;
        total += canonicalStringLength(key) + 1 + length(child);
        members += 1;
      }
      return members ? total + members : 2;
    };
    return [length(value), found];
  }

  size(value: unknown): number {
    return this.measure(value)[0];
  }
}

let ACTIVE: CanonicalSizer | null = null;

/** The projection-scoped sizer, or a fresh one outside a projection. */
export function activeSizer(): CanonicalSizer {
  return ACTIVE ?? new CanonicalSizer();
}

export function jsonBytes(value: unknown): number {
  return activeSizer().size(value);
}

export function estimatedTokensFromBytes(byteCount: number): number {
  return Math.max(1, Math.floor((byteCount + 2) / 3));
}

/** Bind ``sizer`` for one projection and prune stale memo entries. */
export async function withProjectionSizer<T>(sizer: CanonicalSizer, body: () => Promise<T>): Promise<T> {
  sizer.rotate();
  const previous = ACTIVE;
  ACTIVE = sizer;
  try {
    return await body();
  } finally {
    ACTIVE = previous;
  }
}

export interface PassFields {
  retained_event_ids?: string[];
  externalized_event_ids?: string[];
  replaced_event_ids?: string[];
  artifact_refs?: unknown[];
  [key: string]: unknown;
}

export function measuredPassRecord(
  name: string, before: Measured, after: Measured, fields: PassFields = {},
): Record<string, unknown> {
  const [beforeBytes, beforeIds] = before;
  const [afterBytes, afterIds] = after;
  const rest: Record<string, unknown> = { ...fields };
  const take = (key: string, fallback: () => unknown): unknown => {
    if (Object.prototype.hasOwnProperty.call(rest, key)) {
      const value = rest[key];
      delete rest[key];
      return value;
    }
    return fallback();
  };
  return {
    name,
    pre_bytes: beforeBytes,
    post_bytes: afterBytes,
    pre_estimated_tokens: estimatedTokensFromBytes(beforeBytes),
    post_estimated_tokens: estimatedTokensFromBytes(afterBytes),
    retained_event_ids: take("retained_event_ids", () => sortedStrings(afterIds)),
    externalized_event_ids: take("externalized_event_ids", () => []),
    replaced_event_ids: take("replaced_event_ids",
      () => sortedStrings([...beforeIds].filter((id) => !afterIds.has(id)))),
    artifact_refs: take("artifact_refs", () => []),
    ...rest,
  };
}

export function passRecord(name: string, before: unknown, after: unknown, fields: PassFields = {}): Record<string, unknown> {
  const sizer = activeSizer();
  return measuredPassRecord(name, sizer.measure(before), sizer.measure(after), fields);
}
