// Generic retention hints (see ora/agent/pi/hints.py for the contract).
//
// Python decides which members carry no information, which conventions a
// payload states once and which facts are static within a scope; this module
// only executes those decisions on lossless JSON trees. It knows no robot
// vocabulary: every key name it writes comes from the hint itself.

import { clonePy, dumpsSorted, PyFloat } from "../pyjson.ts";

export type Hint =
  | { op: "omit"; path: Array<string | number>; rule?: string }
  | {
      op: "hoist_shared"; at: Array<string | number>; into: string;
      collections: Array<{ collection: string; placeholder: string; fields: string[][] }>;
    }
  | {
      op: "round_static_facts"; at: Array<string | number>; scope: number; listing: "unchanged" | "absent";
      identity_keys: string[]; listing_key: string; scope_field: string; slot_key: string;
      slot_scope_field: string; slot_sequence_field: string; sequence_member: string; rule: string;
      slot_context_source: string; slot_flag: string;
    };

export class HintError extends Error {}

const MISSING = Symbol("missing");
const OMIT = Symbol("omit");
type Json = any;

function isObject(value: unknown): value is Record<string, Json> {
  return typeof value === "object" && value !== null && !Array.isArray(value) && !(value instanceof PyFloat);
}

function getPath(value: Json, path: Array<string | number>): Json | typeof MISSING {
  let current = value;
  for (const step of path) {
    if (isObject(current) && Object.prototype.hasOwnProperty.call(current, step)) current = current[step as string];
    else if (Array.isArray(current) && typeof step === "number" && step >= 0 && step < current.length) current = current[step];
    else return MISSING;
  }
  return current;
}

// -- omit -----------------------------------------------------------------------

function omit(value: Json, path: Array<string | number>): void {
  const parent = getPath(value, path.slice(0, -1));
  const leaf = path[path.length - 1];
  if (isObject(parent) && Object.prototype.hasOwnProperty.call(parent, leaf)) delete parent[leaf as string];
}

// -- hoist_shared (delivery/dedup._hoist) -------------------------------------------

function fieldGet(entry: Json, path: string[]): Json | typeof MISSING {
  let current = entry;
  for (const step of path) {
    if (!isObject(current) || !Object.prototype.hasOwnProperty.call(current, step)) return MISSING;
    current = current[step];
  }
  return current;
}

function fieldPop(entry: Record<string, Json>, path: string[]): void {
  const parents: Record<string, Json>[] = [entry];
  for (const step of path.slice(0, -1)) parents.push(parents[parents.length - 1][step]);
  delete parents[parents.length - 1][path[path.length - 1]];
  // A parent emptied by the move holds nothing the conventions do not state.
  for (let i = path.length - 2; i >= 0; i -= 1) {
    const parent = parents[i];
    const step = path[i];
    if (Object.keys(parent[step]).length) break;
    delete parent[step];
  }
}

function fieldPut(target: Record<string, Json>, path: string[], value: Json): void {
  let current = target;
  for (const step of path.slice(0, -1)) {
    if (!Object.prototype.hasOwnProperty.call(current, step)) current[step] = {};
    current = current[step];
  }
  current[path[path.length - 1]] = value;
}

function hoist(container: Record<string, Json>, hint: Extract<Hint, { op: "hoist_shared" }>): void {
  const conventions: Record<string, Json> = {};
  for (const { collection, placeholder, fields } of hint.collections) {
    const entries = container[collection];
    if (!isObject(entries) || Object.keys(entries).length < 2) continue;
    const shared: Record<string, Json> = {};
    for (const path of fields) {
      const values = Object.values(entries).map((entry) => fieldGet(entry, path));
      const first = values[0];
      if (typeof first === "string" && values.every((value) => value === first)) {
        fieldPut(shared, path, first);
        for (const entry of Object.values(entries)) fieldPop(entry as Record<string, Json>, path);
      }
    }
    if (Object.keys(shared).length) conventions[`${collection}.${placeholder}`] = shared;
  }
  if (Object.keys(conventions).length) container[hint.into] = conventions;
}

// -- round_static_facts (round_static_facts.capture/reduce/slot_message) ------------

const SEGMENT = /^[A-Za-z0-9_]+$/;

function dumps(value: Json): string {
  return dumpsSorted(value);
}

function utf8Length(text: string): number {
  return Buffer.byteLength(text, "utf8");
}

function worthNaming(path: string, value: Json): boolean {
  const key = path.slice(path.lastIndexOf(".") + 1);
  const saved = utf8Length(dumps(key)) + 2 + utf8Length(dumps(value)) + 2;
  return saved > utf8Length(dumps(path)) + 2;
}

function reduceUnchanged(value: Json, base: Json | typeof MISSING, path: string, unchanged: string[]): Json {
  if (base !== MISSING && dumps(value) === dumps(base) && worthNaming(path, value)) {
    unchanged.push(path);
    return OMIT;
  }
  if (isObject(value) && isObject(base)) {
    const result: Record<string, Json> = {};
    for (const [key, child] of Object.entries(value)) {
      const reduced = SEGMENT.test(key)
        ? reduceUnchanged(child, Object.prototype.hasOwnProperty.call(base, key) ? base[key] : MISSING, `${path}.${key}`, unchanged)
        : clonePy(child);
      if (reduced !== OMIT) result[key] = reduced;
    }
    return result;
  }
  return clonePy(value);
}

function identifierKeys(...mappings: Record<string, Json>[]): boolean {
  return mappings.every((mapping) => Object.keys(mapping).every((key) => SEGMENT.test(key)));
}

function reduceAbsent(value: Json, base: Json | typeof MISSING, path: string, absent: string[]): Json {
  if (base !== MISSING && dumps(value) === dumps(base)) return OMIT;
  if (isObject(value) && isObject(base) && identifierKeys(value, base)) {
    const result: Record<string, Json> = {};
    for (const [key, child] of Object.entries(value)) {
      const reduced = reduceAbsent(child, Object.prototype.hasOwnProperty.call(base, key) ? base[key] : MISSING, `${path}.${key}`, absent);
      if (reduced !== OMIT) result[key] = reduced;
    }
    for (const key of Object.keys(base)) if (!Object.prototype.hasOwnProperty.call(value, key)) absent.push(`${path}.${key}`);
    return Object.keys(result).length ? result : OMIT;
  }
  return clonePy(value);
}

function pythonSorted(values: string[]): string[] {
  // Python sorts str by code point; these paths are ASCII identifiers.
  return values.slice().sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

function reduceFacts(current: Record<string, Json>, fields: Record<string, Json>,
                     hint: Extract<Hint, { op: "round_static_facts" }>): Record<string, Json> {
  const identity = new Set(hint.identity_keys);
  const result: Record<string, Json> = {};
  if (hint.listing === "absent") {
    const absent: string[] = [];
    for (const [key, value] of Object.entries(current)) {
      if (identity.has(key) || !SEGMENT.test(key)) {
        result[key] = clonePy(value);
        continue;
      }
      const reduced = reduceAbsent(value, Object.prototype.hasOwnProperty.call(fields, key) ? fields[key] : MISSING, key, absent);
      if (reduced !== OMIT) result[key] = reduced;
    }
    for (const key of Object.keys(fields)) {
      if (!Object.prototype.hasOwnProperty.call(current, key) && !identity.has(key) && SEGMENT.test(key)) absent.push(key);
    }
    const listing: Record<string, Json> = { [hint.scope_field]: hint.scope };
    if (absent.length) listing.absent = pythonSorted(absent);
    result[hint.listing_key] = listing;
    return result;
  }
  const unchanged: string[] = [];
  for (const [key, value] of Object.entries(current)) {
    if (identity.has(key) || !SEGMENT.test(key)) {
      result[key] = clonePy(value);
      continue;
    }
    const reduced = reduceUnchanged(value, Object.prototype.hasOwnProperty.call(fields, key) ? fields[key] : MISSING, key, unchanged);
    if (reduced !== OMIT) result[key] = reduced;
  }
  result[hint.listing_key] = { [hint.scope_field]: hint.scope, unchanged: pythonSorted(unchanged) };
  return result;
}

export class StaticFactsState {
  scope: unknown = null;
  fields: Record<string, Json> | null = null;

  toJSON(): unknown {
    return { scope: this.scope, fields: this.fields };
  }
}

export interface HintResult {
  delivered: Json;
  slot: Record<string, Json> | null;
}

/** Apply hints in order to a copy of `value`; a new static-facts scope yields its slot message. */
export function applyHints(value: Json, hints: Hint[], state: StaticFactsState): HintResult {
  const delivered = clonePy(value);
  let slot: Record<string, Json> | null = null;
  for (const hint of hints) {
    if (hint.op === "omit") {
      omit(delivered, hint.path);
    } else if (hint.op === "hoist_shared") {
      const target = getPath(delivered, hint.at);
      if (isObject(target)) hoist(target, hint);
    } else if (hint.op === "round_static_facts") {
      const parent = getPath(delivered, hint.at.slice(0, -1));
      const leaf = hint.at[hint.at.length - 1] as string;
      if (!isObject(parent) || !isObject(parent[leaf])) throw new HintError("static facts hint target is not an object");
      const current = parent[leaf] as Record<string, Json>;
      if (state.scope !== hint.scope || state.fields === null) {
        state.scope = hint.scope;
        const identity = new Set(hint.identity_keys);
        state.fields = {};
        for (const [key, child] of Object.entries(current)) if (!identity.has(key)) state.fields[key] = clonePy(child);
        const body = {
          [hint.slot_key]: {
            [hint.slot_scope_field]: hint.scope,
            [hint.slot_sequence_field]: current[hint.sequence_member],
            rule: hint.rule,
            [leaf]: clonePy(state.fields),
          },
        };
        slot = {
          role: "user",
          content: [{ type: "text", text: dumps(body) }],
          _protected_context: true,
          _context_source: hint.slot_context_source,
          [hint.slot_flag]: { round: hint.scope },
        };
      }
      parent[leaf] = reduceFacts(current, state.fields, hint);
    } else {
      throw new HintError(`unknown hint op ${(hint as any).op}`);
    }
  }
  return { delivered, slot };
}
