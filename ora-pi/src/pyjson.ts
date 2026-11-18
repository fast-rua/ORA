// Python-compatible JSON for values that cross the Python/TypeScript boundary.
//
// JSON.parse loses two things Python's json module keeps: whether a number was
// written as a float (`1.0` vs `1`) and integers beyond 2**53. Both change
// model-visible bytes and hashes when TypeScript re-serializes a payload that
// Python produced. `parsePy` keeps them (PyFloat for integral floats, bigint
// for unsafe integers) and `dumpsPy` writes exactly what Python's
// `json.dumps(value, ensure_ascii=..., sort_keys=..., separators=..., indent=...)`
// writes for the same data. `canonicalBytesPy` is ORA's `canonical_bytes`.

/** A float whose value is integral (Python writes `1.0`, `-0.0`, `1e+16`). */
export class PyFloat {
  readonly value: number;
  constructor(value: number) {
    this.value = value;
  }
  toJSON(): number {
    return this.value;
  }
}

export type PyValue =
  | null
  | boolean
  | number
  | bigint
  | string
  | PyFloat
  | PyValue[]
  | { [key: string]: PyValue };

export class PyJsonError extends Error {}

// -- Parsing -------------------------------------------------------------------

const INT_LEXEME = /^-?(?:0|[1-9][0-9]*)$/;

function numberFromLexeme(lexeme: string): number | bigint | PyFloat {
  if (INT_LEXEME.test(lexeme)) {
    const value = Number(lexeme);
    // Python reads "-0" as the int 0 (it has no negative integer zero).
    if (Number.isSafeInteger(value)) return value === 0 ? 0 : value;
    return BigInt(lexeme);
  }
  const value = Number(lexeme);
  if (!Number.isFinite(value)) {
    throw new PyJsonError(`non-finite JSON number ${lexeme}`);
  }
  if (Number.isInteger(value) || Object.is(value, -0)) return new PyFloat(value);
  return value;
}

/** Parse JSON text keeping Python's int/float distinction. */
export function parsePy(text: string): any {
  let index = 0;
  const length = text.length;

  function fail(message: string): never {
    throw new PyJsonError(`${message} at offset ${index}`);
  }
  function skip(): void {
    while (index < length) {
      const code = text.charCodeAt(index);
      if (code === 0x20 || code === 0x0a || code === 0x0d || code === 0x09) index += 1;
      else break;
    }
  }
  function parseString(): string {
    // index points at the opening quote
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
      if (code < 0x20) fail("control character in string");
      end += 1;
    }
    if (end >= length) fail("unterminated string");
    const raw = text.slice(index, end + 1);
    index = end + 1;
    return simple ? raw.slice(1, -1) : (JSON.parse(raw) as string);
  }
  function parseValue(): any {
    skip();
    if (index >= length) fail("unexpected end of JSON");
    const char = text[index];
    if (char === "{") {
      index += 1;
      const result: { [key: string]: any } = {};
      skip();
      if (text[index] === "}") {
        index += 1;
        return result;
      }
      for (;;) {
        skip();
        if (text[index] !== '"') fail("expected object key");
        const key = parseString();
        skip();
        if (text[index] !== ":") fail("expected ':'");
        index += 1;
        // Python keeps the last duplicate key; so does this.
        Object.defineProperty(result, key, {
          value: parseValue(), enumerable: true, writable: true, configurable: true,
        });
        skip();
        if (text[index] === ",") {
          index += 1;
          continue;
        }
        if (text[index] === "}") {
          index += 1;
          return result;
        }
        fail("expected ',' or '}'");
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
        result.push(parseValue());
        skip();
        if (text[index] === ",") {
          index += 1;
          continue;
        }
        if (text[index] === "]") {
          index += 1;
          return result;
        }
        fail("expected ',' or ']'");
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
    const match = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/.exec(text.slice(index, index + 400));
    if (!match) fail("invalid JSON value");
    index += match[0].length;
    return numberFromLexeme(match[0]);
  }

  const value = parseValue();
  skip();
  if (index !== length) fail("trailing data after JSON value");
  return value;
}

// -- Python float repr -------------------------------------------------------------

/** `repr(float(x))` in CPython (shortest round-trip digits, Python layout). */
export function pyFloatRepr(x: number): string {
  if (Number.isNaN(x)) return "NaN";
  if (x === Infinity) return "Infinity";
  if (x === -Infinity) return "-Infinity";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const sign = x < 0 ? "-" : "";
  // toExponential() without a digit count yields the shortest round-trip
  // digits, the same digits CPython's repr chooses.
  const exponential = Math.abs(x).toExponential();
  const [mantissa, exponentText] = exponential.split("e");
  const digits = mantissa.replace(".", "");
  const decpt = Number(exponentText) + 1;
  if (decpt > -4 && decpt <= 16) {
    if (decpt <= 0) return `${sign}0.${"0".repeat(-decpt)}${digits}`;
    if (decpt >= digits.length) return `${sign}${digits}${"0".repeat(decpt - digits.length)}.0`;
    return `${sign}${digits.slice(0, decpt)}.${digits.slice(decpt)}`;
  }
  const exponent = decpt - 1;
  const head = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
  const magnitude = Math.abs(exponent).toString().padStart(2, "0");
  return `${sign}${head}e${exponent < 0 ? "-" : "+"}${magnitude}`;
}

// -- Python json.dumps ------------------------------------------------------------

export interface DumpsOptions {
  sortKeys?: boolean;
  separators?: [string, string];
  ensureAscii?: boolean;
  allowNan?: boolean;
  indent?: number | null;
}

const ESCAPES: Record<string, string> = {
  '"': '\\"',
  "\\": "\\\\",
  "\n": "\\n",
  "\r": "\\r",
  "\t": "\\t",
  "\b": "\\b",
  "\f": "\\f",
};

function hex4(code: number): string {
  return code.toString(16).padStart(4, "0");
}

/** Python's `json.encoder.py_encode_basestring(_ascii)`. */
export function pyEncodeString(value: string, ensureAscii: boolean): string {
  // For well-formed text, ECMAScript JSON.stringify escapes exactly what
  // Python's ensure_ascii=False encoder escapes (quote, backslash, C0 controls
  // with the same short forms and lowercase \u00xx), natively and fast.
  if (!ensureAscii && value.isWellFormed()) return JSON.stringify(value);
  let out = '"';
  for (let i = 0; i < value.length; i += 1) {
    const char = value[i];
    const code = value.charCodeAt(i);
    const escape = ESCAPES[char];
    if (escape !== undefined) {
      out += escape;
    } else if (code < 0x20) {
      out += `\\u${hex4(code)}`;
    } else if (ensureAscii && code > 0x7e) {
      out += `\\u${hex4(code)}`;
    } else {
      out += char;
    }
  }
  return out + '"';
}

/** Python sorts str keys by code point; JavaScript's default sort uses UTF-16 units. */
export function compareCodePoints(left: string, right: string): number {
  // Compared in place: UTF-16 order and code-point order differ only inside the
  // code point where the strings first differ, so only that one is decoded.
  const n = Math.min(left.length, right.length);
  let i = 0;
  while (i < n && left.charCodeAt(i) === right.charCodeAt(i)) i += 1;
  if (i === n) return left.length - right.length;
  // The first difference may be the low half of a pair whose (equal) high half precedes it.
  const high = (code: number) => code >= 0xd800 && code <= 0xdbff;
  const low = (code: number) => code >= 0xdc00 && code <= 0xdfff;
  if (i > 0 && high(left.charCodeAt(i - 1)) && (low(left.charCodeAt(i)) || low(right.charCodeAt(i)))) i -= 1;
  return left.codePointAt(i)! - right.codePointAt(i)!;
}

function isPlainObject(value: unknown): value is { [key: string]: unknown } {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && !(value instanceof PyFloat);
}

export function dumpsPy(value: unknown, options: DumpsOptions = {}): string {
  const parts: string[] = [];
  writePy(value, options, (text) => {
    parts.push(text);
  });
  return parts.join("");
}

/**
 * `dumpsPy`, handed to `write` piece by piece in order: one pass, no
 * intermediate string per nesting level, so a large document can go straight
 * to a file.
 */
export function writePy(value: unknown, options: DumpsOptions, write: (text: string) => void): void {
  const sortKeys = options.sortKeys ?? false;
  const ensureAscii = options.ensureAscii ?? true;
  const allowNan = options.allowNan ?? true;
  const indent = options.indent ?? null;
  const separators = options.separators ?? (indent !== null ? [",", ": "] : [", ", ": "]);
  const [itemSep, keySep] = separators;

  function number(x: number): string {
    if (!Number.isFinite(x)) {
      if (!allowNan) throw new PyJsonError(`Out of range float values are not JSON compliant: ${x}`);
      return pyFloatRepr(x);
    }
    if (Number.isInteger(x) && !Object.is(x, -0)) {
      return Number.isSafeInteger(x) ? x.toString() : BigInt(x).toString();
    }
    return pyFloatRepr(x);
  }

  function encode(item: unknown, level: number): void {
    if (item === null || item === undefined) return write("null");
    if (item === true) return write("true");
    if (item === false) return write("false");
    if (typeof item === "string") return write(pyEncodeString(item, ensureAscii));
    if (typeof item === "number") return write(number(item));
    if (typeof item === "bigint") return write(item.toString());
    if (item instanceof PyFloat) {
      if (!Number.isFinite(item.value) && !allowNan) {
        throw new PyJsonError("Out of range float values are not JSON compliant");
      }
      return write(pyFloatRepr(item.value));
    }
    const newline = indent !== null ? "\n" + " ".repeat(indent * (level + 1)) : "";
    const closing = indent !== null ? "\n" + " ".repeat(indent * level) : "";
    if (Array.isArray(item)) {
      if (item.length === 0) return write("[]");
      write("[" + newline);
      for (let index = 0; index < item.length; index += 1) {
        if (index) write(itemSep + newline);
        encode(item[index], level + 1);
      }
      return write(closing + "]");
    }
    if (isPlainObject(item)) {
      let keys = Object.keys(item).filter((key) => item[key] !== undefined);
      if (keys.length === 0) return write("{}");
      if (sortKeys) keys = keys.slice().sort(compareCodePoints);
      write("{" + newline);
      for (let index = 0; index < keys.length; index += 1) {
        if (index) write(itemSep + newline);
        write(pyEncodeString(keys[index], ensureAscii) + keySep);
        encode(item[keys[index]], level + 1);
      }
      return write(closing + "}");
    }
    throw new PyJsonError(`Object of type ${typeof item} is not JSON serializable`);
  }

  encode(value, 0);
}

/** UTF-8 bytes of a string; refuses lone surrogates as Python's encoder does. */
export function utf8Strict(text: string): Buffer {
  if (!text.isWellFormed()) {
    throw new PyJsonError("'utf-8' codec can't encode lone surrogates");
  }
  return Buffer.from(text, "utf8");
}

/** ORA's `canonical_bytes`: compact, sorted, raw UTF-8, no NaN. */
export function canonicalBytesPy(value: unknown): Buffer {
  return utf8Strict(
    dumpsPy(value, { sortKeys: true, separators: [",", ":"], ensureAscii: false, allowNan: false }),
  );
}

/** `json.dumps(value, ensure_ascii=False, sort_keys=True)` (default separators). */
export function dumpsSorted(value: unknown): string {
  return dumpsPy(value, { sortKeys: true, ensureAscii: false });
}

/** `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`. */
export function dumpsCompactSorted(value: unknown): string {
  return dumpsPy(value, { sortKeys: true, separators: [",", ":"], ensureAscii: false });
}

/** Deep clone that keeps PyFloat and bigint identities. */
export function clonePy<T>(value: T): T {
  if (value === null || typeof value !== "object") return value;
  if (value instanceof PyFloat) return new PyFloat(value.value) as T;
  if (Array.isArray(value)) return value.map((item) => clonePy(item)) as T;
  const result: { [key: string]: unknown } = {};
  for (const key of Object.keys(value)) {
    Object.defineProperty(result, key, {
      value: clonePy((value as any)[key]), enumerable: true, writable: true, configurable: true,
    });
  }
  return result as T;
}

/** Python `==` on decoded JSON values (1 == 1.0, True != 1 is NOT Python; see note). */
export function pyEqual(left: unknown, right: unknown): boolean {
  // Python: True == 1 and 1 == 1.0. JSON decoding rarely mixes them; ORA code
  // compares canonical bytes when exactness matters.
  const l = left instanceof PyFloat ? left.value : left;
  const r = right instanceof PyFloat ? right.value : right;
  if (typeof l === "bigint" || typeof r === "bigint") {
    return typeof l === typeof r ? l === r : Number(l) === Number(r);
  }
  if (l === null || r === null || typeof l !== "object" || typeof r !== "object") {
    if (typeof l === "boolean" || typeof r === "boolean") {
      const ln = typeof l === "boolean" ? Number(l) : l;
      const rn = typeof r === "boolean" ? Number(r) : r;
      return ln === rn;
    }
    return l === r;
  }
  if (Array.isArray(l) !== Array.isArray(r)) return false;
  if (Array.isArray(l)) {
    const rr = r as unknown[];
    return l.length === rr.length && l.every((item, i) => pyEqual(item, rr[i]));
  }
  const lk = Object.keys(l as object);
  const rk = Object.keys(r as object);
  if (lk.length !== rk.length) return false;
  return lk.every((key) => Object.prototype.hasOwnProperty.call(r, key)
    && pyEqual((l as any)[key], (r as any)[key]));
}
