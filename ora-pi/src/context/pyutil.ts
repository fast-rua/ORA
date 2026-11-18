// Python semantics the Context port needs beyond `pyjson.ts`.
//
// The Native Context pipeline is a byte-for-byte port of the Python one, so a
// few Python behaviours have to be reproduced exactly: `json.loads` (NaN,
// Infinity, int/float distinction), `isinstance` on decoded JSON, `sorted` on
// str (code points), `bytes[:n].decode("utf-8", errors="ignore")`, `str`
// slicing by code points, `round(x, n)` and the iteration order of a CPython
// `set` of small ints (which the L2 duplicate pass depends on).

import { createHash } from "node:crypto";
import { PyFloat, compareCodePoints, dumpsPy, pyEqual, clonePy } from "../pyjson.ts";
import { canonicalByteLength, canonicalBytes, canonicalDigest, fastDumps } from "./canonical.ts";

export { PyFloat, compareCodePoints, dumpsPy, pyEqual, clonePy };

/** ORA's ``canonical_bytes`` (fast path; byte-identical to pyjson's). */
export const canonicalBytesPy = canonicalBytes;

// -- Errors ----------------------------------------------------------------------

/** ``ora.agent.contracts.PolicySessionError``. */
export class PolicySessionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PolicySessionError";
  }
}

/** ``projection.ContextCompactionRejected`` (a PolicySessionError). */
export class ContextCompactionRejected extends PolicySessionError {
  constructor(message: string) {
    super(message);
    this.name = "ContextCompactionRejected";
  }
}

/** Python ``ValueError`` raised by the artifact pointer/envelope validators. */
export class PyValueError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ValueError";
  }
}

/** ``str(exc)`` as Python would print it for the errors this port raises. */
export function errorText(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

// -- isinstance on decoded JSON ------------------------------------------------------

export function isMapping(value: unknown): value is Record<string, any> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && !(value instanceof PyFloat) && !Buffer.isBuffer(value);
}

/** ``isinstance(v, int) and not isinstance(v, bool)``. */
export function isInt(value: unknown): value is number | bigint {
  return (typeof value === "number" && Number.isInteger(value) && !Object.is(value, -0))
    || typeof value === "bigint";
}

/** ``isinstance(v, int)`` (Python bool is an int). */
export function isIntOrBool(value: unknown): boolean {
  return isInt(value) || typeof value === "boolean";
}

/** ``isinstance(v, float)``. */
export function isFloat(value: unknown): boolean {
  return value instanceof PyFloat
    || (typeof value === "number" && (!Number.isInteger(value) || Object.is(value, -0)));
}

/** ``isinstance(v, (int, float)) and not isinstance(v, bool)``. */
export function isNumber(value: unknown): boolean {
  return isInt(value) || isFloat(value);
}

/** The numeric value of an int/float JSON value. */
export function num(value: unknown): number {
  if (value instanceof PyFloat) return value.value;
  if (typeof value === "bigint") return Number(value);
  if (typeof value === "boolean") return value ? 1 : 0;
  return value as number;
}

/** A Python float result: integral values keep their float spelling. */
export function pyFloat(value: number): number | PyFloat {
  return Number.isInteger(value) || Object.is(value, -0) ? new PyFloat(value) : value;
}

/** Python ``int(x)`` for a finite float/int. */
export function pyInt(value: number): number {
  return Math.trunc(value);
}

// -- Dict helpers ---------------------------------------------------------------------

/** ``d[key] = value`` that also works for the key ``__proto__``. */
export function setItem(target: Record<string, any>, key: string, value: unknown): void {
  if (key === "__proto__") {
    Object.defineProperty(target, key, {
      value, enumerable: true, writable: true, configurable: true,
    });
  } else {
    target[key] = value;
  }
}

export function hasKey(target: Record<string, any>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(target, key);
}

/** ``mapping.get(key)`` (None for a missing key). */
export function get(target: unknown, key: string): any {
  if (!isMapping(target)) return undefined;
  return hasKey(target, key) ? target[key] : undefined;
}

/** Python ``a == b`` on decoded JSON values (None == missing). */
export function eq(left: unknown, right: unknown): boolean {
  const l = left === undefined ? null : left;
  const r = right === undefined ? null : right;
  return pyEqual(l, r);
}

/** Python ``value is None`` (a missing key reads as None). */
export function isNone(value: unknown): boolean {
  return value === null || value === undefined;
}

/** ``None`` for a missing value, as JSON stores it. */
export function orNull<T>(value: T | undefined): T | null {
  return value === undefined ? null : value;
}

// -- Strings and bytes -------------------------------------------------------------------

export function sha256Hex(data: Buffer | string): string {
  return createHash("sha256").update(typeof data === "string" ? Buffer.from(data, "utf8") : data)
    .digest("hex");
}

export function canonicalSha256(value: unknown): string {
  return canonicalDigest(value).sha256;
}

/** ``len(canonical_bytes(value))``. */
export function canonicalLength(value: unknown): number {
  return canonicalByteLength(value);
}

/** ``len(text.encode("utf-8"))``; refuses lone surrogates like Python. */
export function utf8Len(text: string): number {
  if (!text.isWellFormed()) {
    throw new PolicySessionError("'utf-8' codec can't encode lone surrogates");
  }
  return Buffer.byteLength(text, "utf8");
}

export function utf8(text: string): Buffer {
  if (!text.isWellFormed()) {
    throw new PolicySessionError("'utf-8' codec can't encode lone surrogates");
  }
  return Buffer.from(text, "utf8");
}

/** ``text.encode()[:n].decode("utf-8", errors="ignore")`` for well-formed text. */
export function utf8PrefixIgnore(text: string, n: number): string {
  const data = utf8(text);
  if (n >= data.length) return text;
  let end = Math.max(0, n);
  // A continuation byte at ``end`` means the character holding it was cut.
  while (end > 0 && (data[end] & 0xc0) === 0x80) end -= 1;
  return data.subarray(0, end).toString("utf8");
}

/** Number of code points (Python ``len(str)``). */
export function cpLen(text: string): number {
  let count = 0;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      const next = text.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) i += 1;
    }
    count += 1;
  }
  return count;
}

/** ``text[:n]`` with Python code-point indexing. */
export function cpPrefix(text: string, n: number): string {
  let count = 0;
  let i = 0;
  while (i < text.length && count < n) {
    const code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      const next = text.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) i += 1;
    }
    i += 1;
    count += 1;
  }
  return text.slice(0, i);
}

/** Python ``str.isspace`` characters (``str.lstrip()`` strips these). */
const PY_WHITESPACE = "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0     "
  + "           　";

export function pyLstrip(text: string): string {
  let i = 0;
  while (i < text.length && PY_WHITESPACE.includes(text[i])) i += 1;
  return text.slice(i);
}

/** Python ``str.strip()``. */
export function pyStrip(text: string): string {
  let end = text.length;
  while (end > 0 && PY_WHITESPACE.includes(text[end - 1])) end -= 1;
  return pyLstrip(text.slice(0, end));
}

/** The Python ``\s`` class for str patterns, for regex sources. */
export const PY_REGEX_WHITESPACE = "[\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const PY_WORD = "[\\p{L}\\p{N}_]";
/** Python's Unicode ``\b`` for str patterns, for regex sources. */
export const PY_REGEX_WORD_BOUNDARY = `(?:(?<!${PY_WORD})(?=${PY_WORD})|(?<=${PY_WORD})(?!${PY_WORD}))`;

/** Translate a Python ``re`` source using only ``\b``/``\s`` beyond JS syntax. */
export function pythonRegex(source: string, flags: string): RegExp {
  let out = "";
  let inClass = false;
  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    if (char === "\\" && i + 1 < source.length) {
      const next = source[i + 1];
      if (!inClass && next === "b") out += PY_REGEX_WORD_BOUNDARY;
      else if (!inClass && next === "s") out += PY_REGEX_WHITESPACE;
      else out += char + next;
      i += 1;
      continue;
    }
    if (char === "[") inClass = true;
    else if (char === "]") inClass = false;
    // Python's ``$`` (no MULTILINE) also matches before a final newline.
    if (char === "$" && !inClass) {
      out += "(?=\\n?$)";
      continue;
    }
    out += char;
  }
  return new RegExp(out, flags.includes("u") ? flags : flags + "u");
}

/** ``sorted(strings)`` (code-point order). */
export function sortedStrings(values: Iterable<string>): string[] {
  return Array.from(values).sort(compareCodePoints);
}

/** ``sorted(ints)``. */
export function sortedNumbers(values: Iterable<number>): number[] {
  return Array.from(values).sort((a, b) => a - b);
}

/** Python ``round(x, ndigits)`` for a finite float: correctly rounded, ties to even. */
export function pyRound(x: number, ndigits: number): number {
  if (!Number.isFinite(x)) return x;
  const fixed = x.toFixed(ndigits); // exact decimal expansion, ties away from zero
  const candidate = Number(fixed);
  // Detect an exact tie (x is exactly halfway between two ndigit decimals).
  const scale = 10n ** BigInt(ndigits);
  const [mant, exp] = exactRational(x);
  // x * scale * 2 as a rational: mant * 2^exp * scale * 2
  let numer = mant * scale * 2n;
  let denom = 1n;
  if (exp >= 0) numer *= 2n ** BigInt(exp);
  else denom = 2n ** BigInt(-exp);
  if (numer % denom !== 0n) return candidate;
  const doubled = numer / denom; // 2 * x * scale, integral
  if (doubled % 2n === 0n) return candidate; // x * scale is an integer: no tie
  // Tie between k and k + 1 (k = floor(x * scale)): choose the even one.
  const k = (doubled - 1n) / 2n; // exact: doubled is odd
  const even = k % 2n === 0n ? k : k + 1n;
  return Number(`${even}e-${ndigits}`);
}

function exactRational(x: number): [bigint, number] {
  const buffer = new DataView(new ArrayBuffer(8));
  buffer.setFloat64(0, x);
  const bits = buffer.getBigUint64(0);
  const sign = bits >> 63n ? -1n : 1n;
  const exponent = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  if (exponent === 0) return [sign * fraction, -1074];
  return [sign * (fraction | (1n << 52n)), exponent - 1075];
}

// -- json.loads -----------------------------------------------------------------------

export class PyJsonDecodeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "JSONDecodeError";
  }
}

const INT_LEXEME = /^-?(?:0|[1-9][0-9]*)$/;
const NUMBER_LEXEME = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/;

function numberFromLexeme(lexeme: string): number | bigint | PyFloat {
  if (INT_LEXEME.test(lexeme)) {
    const value = Number(lexeme);
    if (Number.isSafeInteger(value)) return value === 0 ? 0 : value;
    return BigInt(lexeme);
  }
  const value = Number(lexeme); // 1e400 -> Infinity, as Python's float()
  if (Number.isFinite(value) && (Number.isInteger(value) || Object.is(value, -0))) {
    return new PyFloat(value);
  }
  return value;
}

/**
 * Python ``json.loads`` on str: keeps int/float, accepts NaN/Infinity/-Infinity,
 * keeps the last duplicate key at its first position.
 */
export function loadsPy(text: string): any {
  let index = 0;
  const length = text.length;
  function fail(message: string): never {
    throw new PyJsonDecodeError(`${message}: char ${index}`);
  }
  function skip(): void {
    while (index < length) {
      const code = text.charCodeAt(index);
      if (code === 0x20 || code === 0x0a || code === 0x0d || code === 0x09) index += 1;
      else break;
    }
  }
  function parseString(): string {
    let end = index + 1;
    let simple = true;
    while (end < length) {
      const code = text.charCodeAt(end);
      if (code === 0x22) break;
      if (code === 0x5c) {
        simple = false;
        end += 2;
        continue;
      }
      if (code < 0x20) fail("Invalid control character");
      end += 1;
    }
    if (end >= length) fail("Unterminated string");
    const raw = text.slice(index, end + 1);
    index = end + 1;
    if (simple) return raw.slice(1, -1);
    try {
      return JSON.parse(raw) as string;
    } catch {
      return fail("Invalid \\escape");
    }
  }
  function parseValue(depth: number): any {
    if (depth > 900) fail("maximum recursion depth exceeded");
    skip();
    if (index >= length) fail("Expecting value");
    const char = text[index];
    if (char === "{") {
      index += 1;
      const result: Record<string, any> = {};
      skip();
      if (text[index] === "}") {
        index += 1;
        return result;
      }
      for (;;) {
        skip();
        if (text[index] !== '"') fail("Expecting property name enclosed in double quotes");
        const key = parseString();
        skip();
        if (text[index] !== ":") fail("Expecting ':' delimiter");
        index += 1;
        setItem(result, key, parseValue(depth + 1));
        skip();
        if (text[index] === ",") {
          index += 1;
          continue;
        }
        if (text[index] === "}") {
          index += 1;
          return result;
        }
        fail("Expecting ',' delimiter");
      }
    }
    if (char === "[") {
      index += 1;
      const result: any[] = [];
      skip();
      if (text[index] === "]") {
        index += 1;
        return result;
      }
      for (;;) {
        result.push(parseValue(depth + 1));
        skip();
        if (text[index] === ",") {
          index += 1;
          continue;
        }
        if (text[index] === "]") {
          index += 1;
          return result;
        }
        fail("Expecting ',' delimiter");
      }
    }
    if (char === '"') return parseString();
    if (text.startsWith("true", index)) {
      index += 4;
      return true;
    }
    if (text.startsWith("false", index)) {
      index += 5;
      return false;
    }
    if (text.startsWith("null", index)) {
      index += 4;
      return null;
    }
    if (text.startsWith("NaN", index)) {
      index += 3;
      return NaN;
    }
    if (text.startsWith("Infinity", index)) {
      index += 8;
      return Infinity;
    }
    if (text.startsWith("-Infinity", index)) {
      index += 9;
      return -Infinity;
    }
    const match = NUMBER_LEXEME.exec(text.slice(index, index + 512));
    if (!match) fail("Expecting value");
    index += match[0].length;
    return numberFromLexeme(match[0]);
  }
  const value = parseValue(0);
  skip();
  if (index !== length) fail("Extra data");
  return value;
}

/** ``json.loads`` returning ``undefined`` instead of raising JSONDecodeError. */
export function tryLoads(text: string): { ok: true; value: any } | { ok: false } {
  try {
    return { ok: true, value: loadsPy(text) };
  } catch (error) {
    if (error instanceof PyJsonDecodeError) return { ok: false };
    throw error;
  }
}

// -- json.dumps spellings used by the Context code ---------------------------------------

/** ``json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))``. */
export function dumpsCompact(value: unknown): string {
  return fastDumps(value, { sortKeys: true, separators: [",", ":"], allowNan: true });
}

/** ``json.dumps(v, ensure_ascii=False, sort_keys=True)``. */
export function dumpsSortedDefault(value: unknown): string {
  return fastDumps(value, { sortKeys: true, separators: [", ", ": "], allowNan: true });
}

/** ``json.dumps(v, ensure_ascii=False)`` (insertion order, default separators). */
export function dumpsUnsorted(value: unknown): string {
  return fastDumps(value, { sortKeys: false, separators: [", ", ": "], allowNan: true });
}

// -- A CPython set of small non-negative ints --------------------------------------------

const PYSET_MINSIZE = 8;
const LINEAR_PROBES = 9;
const PERTURB_SHIFT = 5n;
const DUMMY = -1;
const EMPTY = -2;

/**
 * Iteration order of a CPython 3.11 ``set`` holding non-negative ints, whose
 * hash is the int itself. ``_externalize_tool_results`` takes the first
 * occurrence of each duplicate digest in set iteration order, which is not
 * ascending once an index exceeds the table size (``{7, 8, 9}`` iterates
 * ``8, 9, 7``), so the L2 selection depends on it.
 */
export class CPythonIntSet {
  private table: number[] = new Array(PYSET_MINSIZE).fill(EMPTY);
  private mask = PYSET_MINSIZE - 1;
  private fill = 0;
  used = 0;

  has(value: number): boolean {
    return this.find(value) >= 0;
  }

  private find(value: number): number {
    let perturb = BigInt(value);
    let i = value & this.mask;
    for (;;) {
      let probes = i + LINEAR_PROBES <= this.mask ? LINEAR_PROBES : 0;
      let j = i;
      do {
        const entry = this.table[j];
        if (entry === EMPTY) return -1;
        if (entry === value) return j;
        j += 1;
      } while (probes-- > 0);
      perturb >>= PERTURB_SHIFT;
      i = Number((BigInt(i) * 5n + 1n + perturb) & BigInt(this.mask));
    }
  }

  add(value: number): void {
    let perturb = BigInt(value);
    let i = value & this.mask;
    let freeslot = -1;
    for (;;) {
      let probes = i + LINEAR_PROBES <= this.mask ? LINEAR_PROBES : 0;
      let j = i;
      do {
        const entry = this.table[j];
        if (entry === EMPTY) {
          if (freeslot >= 0) {
            this.used += 1;
            this.table[freeslot] = value;
            return;
          }
          this.fill += 1;
          this.used += 1;
          this.table[j] = value;
          if (this.fill * 5 < this.mask * 3) return;
          this.resize(this.used > 50000 ? this.used * 2 : this.used * 4);
          return;
        }
        if (entry === value) return;
        if (entry === DUMMY && freeslot < 0) freeslot = j;
        j += 1;
      } while (probes-- > 0);
      perturb >>= PERTURB_SHIFT;
      i = Number((BigInt(i) * 5n + 1n + perturb) & BigInt(this.mask));
    }
  }

  discard(value: number): void {
    const slot = this.find(value);
    if (slot < 0) return;
    this.table[slot] = DUMMY;
    this.used -= 1;
  }

  /** ``set.difference_update(iterable)`` (discard each, then drop dummies). */
  differenceUpdate(values: Iterable<number>): void {
    for (const value of values) this.discard(value);
    if (this.fill - this.used <= Math.floor(this.mask / 4)) return;
    this.resize(this.used > 50000 ? this.used * 2 : this.used * 4);
  }

  update(values: Iterable<number>): void {
    for (const value of values) this.add(value);
  }

  private resize(minused: number): void {
    let size = PYSET_MINSIZE;
    while (size <= minused) size *= 2;
    const old = this.table;
    this.table = new Array(size).fill(EMPTY);
    this.mask = size - 1;
    this.fill = this.used;
    for (const entry of old) {
      if (entry >= 0) this.insertClean(entry);
    }
  }

  private insertClean(value: number): void {
    let perturb = BigInt(value);
    let i = value & this.mask;
    for (;;) {
      if (this.table[i] === EMPTY) {
        this.table[i] = value;
        return;
      }
      if (i + LINEAR_PROBES <= this.mask) {
        for (let j = 1; j <= LINEAR_PROBES; j += 1) {
          if (this.table[i + j] === EMPTY) {
            this.table[i + j] = value;
            return;
          }
        }
      }
      perturb >>= PERTURB_SHIFT;
      i = Number((BigInt(i) * 5n + 1n + perturb) & BigInt(this.mask));
    }
  }

  /** Iteration order. */
  values(): number[] {
    return this.table.filter((entry) => entry >= 0);
  }

  sorted(): number[] {
    return this.values().sort((a, b) => a - b);
  }

  [Symbol.iterator](): Iterator<number> {
    return this.values()[Symbol.iterator]();
  }
}

/** Python ``range(start, end)``. */
export function* range(start: number, end: number): Generator<number> {
  for (let i = start; i < end; i += 1) yield i;
}

// -- repr ----------------------------------------------------------------------------

/** Python ``repr`` of a decoded JSON value (str, int, float, bool, None, list, dict). */
export function pyRepr(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (value === true) return "True";
  if (value === false) return "False";
  if (typeof value === "string") return reprString(value);
  if (typeof value === "bigint") return value.toString();
  if (value instanceof PyFloat) return dumpsPy(value);
  if (typeof value === "number") return dumpsPy(value);
  if (Array.isArray(value)) return "[" + value.map(pyRepr).join(", ") + "]";
  if (isMapping(value)) {
    return "{" + Object.keys(value).map((key) => `${reprString(key)}: ${pyRepr(value[key])}`)
      .join(", ") + "}";
  }
  return String(value);
}

function isPrintable(code: number): boolean {
  // Approximates str.isprintable for the characters a refusal can carry.
  if (code < 0x20 || code === 0x7f) return false;
  if (code >= 0x80 && code < 0xa0) return false;
  if (code === 0xad) return false;
  if (code >= 0xd800 && code <= 0xdfff) return false;
  if (code === 0x2028 || code === 0x2029) return false;
  if (code >= 0x200b && code <= 0x200f) return false;
  if (code >= 0x2060 && code <= 0x2064) return false;
  if (code === 0xfeff) return false;
  return true;
}

function reprString(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = quote;
  for (const char of text) {
    const code = char.codePointAt(0)!;
    if (char === quote || char === "\\") out += "\\" + char;
    else if (char === "\n") out += "\\n";
    else if (char === "\r") out += "\\r";
    else if (char === "\t") out += "\\t";
    else if (!isPrintable(code)) {
      if (code <= 0xff) out += "\\x" + code.toString(16).padStart(2, "0");
      else if (code <= 0xffff) out += "\\u" + code.toString(16).padStart(4, "0");
      else out += "\\U" + code.toString(16).padStart(8, "0");
    } else out += char;
  }
  return out + quote;
}

/** CPython's UTF-8 decoder error, or ``null`` when ``data`` decodes. */
export function utf8DecodeError(data: Buffer): { reason: string; start: number } | null {
  const n = data.length;
  const cont = (byte: number) => (byte & 0xc0) === 0x80;
  let i = 0;
  while (i < n) {
    const ch = data[i];
    if (ch < 0x80) {
      i += 1;
      continue;
    }
    if (ch < 0xc2) return { reason: "invalid start byte", start: i };
    if (ch < 0xe0) {
      if (n - i < 2) return { reason: "unexpected end of data", start: i };
      if (!cont(data[i + 1])) return { reason: "invalid continuation byte", start: i };
      i += 2;
      continue;
    }
    if (ch < 0xf0) {
      const bad2 = (ch2: number) => !cont(ch2) || (ch2 < 0xa0 ? ch === 0xe0 : ch === 0xed);
      if (n - i < 3) {
        if (n - i >= 2 && bad2(data[i + 1])) return { reason: "invalid continuation byte", start: i };
        return { reason: "unexpected end of data", start: i };
      }
      if (bad2(data[i + 1]) || !cont(data[i + 2])) {
        return { reason: "invalid continuation byte", start: i };
      }
      i += 3;
      continue;
    }
    if (ch < 0xf5) {
      const bad2 = (ch2: number) => !cont(ch2) || (ch2 < 0x90 ? ch === 0xf0 : ch === 0xf4);
      if (n - i < 4) {
        if (n - i >= 2 && bad2(data[i + 1])) return { reason: "invalid continuation byte", start: i };
        if (n - i >= 3 && !cont(data[i + 2])) return { reason: "invalid continuation byte", start: i };
        return { reason: "unexpected end of data", start: i };
      }
      if (bad2(data[i + 1]) || !cont(data[i + 2]) || !cont(data[i + 3])) {
        return { reason: "invalid continuation byte", start: i };
      }
      i += 4;
      continue;
    }
    return { reason: "invalid start byte", start: i };
  }
  return null;
}
