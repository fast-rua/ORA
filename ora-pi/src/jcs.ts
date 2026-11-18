// RFC 8785 JSON Canonicalization Scheme (JCS).
//
// Used for record hashes that both languages must reproduce (agent-event v2).
// Differences from ORA's Python `canonical_bytes` are listed in its Python
// counterpart, tests/support/agent/jcs.py.

import { PyFloat, PyJsonError } from "./pyjson.ts";

export class JcsError extends PyJsonError {}

function serializeString(value: string): string {
  if (!value.isWellFormed()) throw new JcsError("JCS forbids lone surrogates");
  // ECMAScript JSON.stringify string escaping is the JCS string form.
  return JSON.stringify(value);
}

function serializeNumber(value: number): string {
  if (!Number.isFinite(value)) throw new JcsError("JCS forbids NaN and Infinity");
  // ECMAScript Number::toString; -0 serializes as 0.
  return Object.is(value, -0) ? "0" : String(value);
}

function compareUtf16(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

export function jcsString(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "string") return serializeString(value);
  if (typeof value === "number") return serializeNumber(value);
  if (value instanceof PyFloat) return serializeNumber(value.value);
  if (typeof value === "bigint") {
    // I-JSON: integers outside the IEEE-754 safe range are not interoperable.
    const asNumber = Number(value);
    if (!Number.isSafeInteger(asNumber) || BigInt(asNumber) !== value) {
      throw new JcsError(`JCS cannot represent integer ${value} exactly; send it as a string`);
    }
    return serializeNumber(asNumber);
  }
  if (Array.isArray(value)) return "[" + value.map((item) => jcsString(item)).join(",") + "]";
  if (typeof value === "object") {
    const keys = Object.keys(value as object)
      .filter((key) => (value as any)[key] !== undefined)
      .sort(compareUtf16);
    return "{" + keys.map((key) => serializeString(key) + ":" + jcsString((value as any)[key])).join(",") + "}";
  }
  throw new JcsError(`cannot canonicalize ${typeof value}`);
}

export function jcsBytes(value: unknown): Buffer {
  return Buffer.from(jcsString(value), "utf8");
}
