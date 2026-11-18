// Fast `json.dumps(..., ensure_ascii=False)` for the hot Context path.
//
// Identical output to `pyjson.dumpsPy` with `ensureAscii: false` and no
// indent (checked byte for byte by the Context differential fixtures), but
// strings that need no escaping -- every base64 image URL -- are emitted in
// one piece instead of character by character, and keys sort with `<` unless a
// surrogate makes UTF-16 order differ from code-point order.

import { createHash } from "node:crypto";

import { PyFloat, PyJsonError, compareCodePoints, pyFloatRepr } from "../pyjson.ts";

const NEEDS_ESCAPE = /[\x00-\x1f"\\]/;
const ESCAPE_ALL = /[\x00-\x1f"\\]/g;
const SURROGATE = /[\ud800-\udfff]/;
const ESCAPES: Record<string, string> = {
  '"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f",
};

// Long strings (image data URLs) recur in every projection of a session; V8
// caches a string's hash, so remembering that one needs no escaping is cheap.
const CLEAN_LONG = new Set<string>();
let cleanLongChars = 0;
const CLEAN_LONG_LIMIT = 512 * 1024 * 1024;

function encodeString(value: string): string {
  if (value.length >= 4096) {
    if (CLEAN_LONG.has(value)) return '"' + value + '"';
    if (!NEEDS_ESCAPE.test(value)) {
      if (cleanLongChars + value.length > CLEAN_LONG_LIMIT) {
        CLEAN_LONG.clear();
        cleanLongChars = 0;
      }
      CLEAN_LONG.add(value);
      cleanLongChars += value.length;
      return '"' + value + '"';
    }
  } else if (!NEEDS_ESCAPE.test(value)) {
    return '"' + value + '"';
  }
  return '"' + value.replace(ESCAPE_ALL, (char) => ESCAPES[char]
    ?? "\\u" + char.charCodeAt(0).toString(16).padStart(4, "0")) + '"';
}

function compareKeys(left: string, right: string): number {
  if (SURROGATE.test(left) || SURROGATE.test(right)) return compareCodePoints(left, right);
  return left < right ? -1 : left > right ? 1 : 0;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value) && !(value instanceof PyFloat);
}

export function fastDumps(
  value: unknown, options: { sortKeys: boolean; separators: [string, string]; allowNan: boolean },
): string {
  return dumpParts(value, options).join("");
}

function dumpParts(
  value: unknown, options: { sortKeys: boolean; separators: [string, string]; allowNan: boolean },
): string[] {
  const [itemSep, keySep] = options.separators;
  const parts: string[] = [];
  const number = (x: number): string => {
    if (!Number.isFinite(x)) {
      if (!options.allowNan) throw new PyJsonError(`Out of range float values are not JSON compliant: ${x}`);
      return pyFloatRepr(x);
    }
    if (Number.isInteger(x) && !Object.is(x, -0)) {
      return Number.isSafeInteger(x) ? x.toString() : BigInt(x).toString();
    }
    return pyFloatRepr(x);
  };
  const encode = (item: unknown): void => {
    if (item === null || item === undefined) parts.push("null");
    else if (item === true) parts.push("true");
    else if (item === false) parts.push("false");
    else if (typeof item === "string") parts.push(encodeString(item));
    else if (typeof item === "number") parts.push(number(item));
    else if (typeof item === "bigint") parts.push(item.toString());
    else if (item instanceof PyFloat) {
      if (!Number.isFinite(item.value) && !options.allowNan) {
        throw new PyJsonError("Out of range float values are not JSON compliant");
      }
      parts.push(pyFloatRepr(item.value));
    } else if (Array.isArray(item)) {
      if (!item.length) {
        parts.push("[]");
        return;
      }
      parts.push("[");
      for (let index = 0; index < item.length; index += 1) {
        if (index) parts.push(itemSep);
        encode(item[index]);
      }
      parts.push("]");
    } else if (isPlainObject(item)) {
      let keys = Object.keys(item).filter((key) => item[key] !== undefined);
      if (!keys.length) {
        parts.push("{}");
        return;
      }
      if (options.sortKeys) keys = keys.sort(compareKeys);
      parts.push("{");
      keys.forEach((key, index) => {
        if (index) parts.push(itemSep);
        parts.push(encodeString(key), keySep);
        encode(item[key]);
      });
      parts.push("}");
    } else {
      throw new PyJsonError(`Object of type ${typeof item} is not JSON serializable`);
    }
  };
  encode(value);
  return parts;
}

/** ORA's `canonical_bytes`: compact, sorted, raw UTF-8, no NaN. */
export function canonicalBytes(value: unknown): Buffer {
  const text = fastDumps(value, { sortKeys: true, separators: [",", ":"], allowNan: false });
  if (!text.isWellFormed()) throw new PyJsonError("'utf-8' codec can't encode lone surrogates");
  return Buffer.from(text, "utf8");
}

/** ``(sha256(canonical_bytes(value)), len(canonical_bytes(value)))`` without one giant buffer. */
export function canonicalDigest(value: unknown): { sha256: string; bytes: number } {
  const parts = dumpParts(value, { sortKeys: true, separators: [",", ":"], allowNan: false });
  const hash = createHash("sha256");
  let bytes = 0;
  let pending = "";
  const flush = () => {
    if (!pending) return;
    if (!pending.isWellFormed()) throw new PyJsonError("'utf-8' codec can't encode lone surrogates");
    hash.update(pending, "utf8");
    bytes += Buffer.byteLength(pending, "utf8");
    pending = "";
  };
  for (const part of parts) {
    // A surrogate pair is never split: every part is a whole JSON token.
    if (part.length >= 65536) {
      flush();
      pending = part;
      flush();
    } else {
      pending += part;
      if (pending.length >= 65536) flush();
    }
  }
  flush();
  return { sha256: hash.digest("hex"), bytes };
}

// Printable ASCII but quote and backslash: one byte each, never escaped.
const PLAIN = /^[\x20\x21\x23-\x5b\x5d-\x7e]*$/;

/** UTF-8 bytes of ``encodeString(value)``, counted without building it. */
export function canonicalStringLength(value: string): number {
  if (PLAIN.test(value)) return value.length + 2;
  let bytes = 2;
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code === 0x22 || code === 0x5c) bytes += 2;
    else if (code < 0x20) bytes += code === 0x0a || code === 0x0d || code === 0x09 || code === 0x08 || code === 0x0c ? 2 : 6;
    else if (code < 0x80) bytes += 1;
    else if (code < 0x800) bytes += 2;
    else if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(i + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw new PyJsonError("'utf-8' codec can't encode lone surrogates");
      bytes += 4;
      i += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new PyJsonError("'utf-8' codec can't encode lone surrogates");
    } else bytes += 3;
  }
  return bytes;
}

/** ``len(canonical_bytes(value))`` of a non-container (strings excluded). */
export function canonicalScalarLength(item: unknown): number {
  if (item === null || item === undefined || item === true) return 4;
  if (item === false) return 5;
  if (typeof item === "number") {
    if (!Number.isFinite(item)) throw new PyJsonError(`Out of range float values are not JSON compliant: ${item}`);
    if (Number.isInteger(item) && !Object.is(item, -0)) {
      return Number.isSafeInteger(item) ? item.toString().length : BigInt(item).toString().length;
    }
    return pyFloatRepr(item).length;
  }
  if (typeof item === "bigint") return item.toString().length;
  if (item instanceof PyFloat) {
    if (!Number.isFinite(item.value)) throw new PyJsonError("Out of range float values are not JSON compliant");
    return pyFloatRepr(item.value).length;
  }
  throw new PyJsonError(`Object of type ${typeof item} is not JSON serializable`);
}

/**
 * ``len(canonical_bytes(value))`` without serializing: the byte count of
 * ``fastDumps`` with compact separators, by the same rules (undefined members
 * dropped, NaN and lone surrogates refused), so projections that only need a
 * size never build the request text or hash it.
 */
export function canonicalByteLength(value: unknown): number {
  if (typeof value === "string") return canonicalStringLength(value);
  if (Array.isArray(value)) {
    if (!value.length) return 2;
    let total = 1 + value.length;
    for (const item of value) total += canonicalByteLength(item);
    return total;
  }
  if (isPlainObject(value)) {
    let total = 1;
    let members = 0;
    for (const key of Object.keys(value)) {
      const item = value[key];
      if (item === undefined) continue;
      total += canonicalStringLength(key) + 1 + canonicalByteLength(item);
      members += 1;
    }
    return members ? total + members : 2;
  }
  return canonicalScalarLength(value);
}
