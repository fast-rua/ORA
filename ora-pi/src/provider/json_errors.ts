// CPython 3.11 `json.loads` failure messages and positions.
//
// Provider attempt records carry `str(exc)` of the JSONDecodeError a malformed
// response raised (and, for "Expecting value", the StopIteration it chains
// through `__context__`). This validator walks text exactly as CPython's C
// scanner does, so the TypeScript record names the same message, line, column
// and code-point offset. It returns null when Python would accept the text.

import { PyError } from "./py.ts";

class Failure {
  msg: string;
  pos: number;
  stopIteration: boolean;
  constructor(msg: string, pos: number, stopIteration = false) {
    this.msg = msg;
    this.pos = pos;
    this.stopIteration = stopIteration;
  }
}

const WHITESPACE = new Set([" ", "\t", "\n", "\r"]);
const SIMPLE_ESCAPES = '"\\/bfnrt';

function hexValue(char: string | undefined): number {
  if (char === undefined || !/^[0-9a-fA-F]$/.test(char)) return -1;
  return parseInt(char, 16);
}

function isDigit(char: string | undefined): boolean {
  return char !== undefined && char >= "0" && char <= "9" && char.length === 1;
}

function scan(s: string[]): void {
  const len = s.length;
  const endIdx = len - 1;
  const skip = (start: number): number => {
    let i = start;
    while (i < len && WHITESPACE.has(s[i])) i += 1;
    return i;
  };
  const scanString = (start: number): number => {
    const begin = start - 1;
    let end = start;
    for (;;) {
      let next = end;
      for (; next < len; next += 1) {
        const d = s[next];
        if (d === '"' || d === "\\") break;
        if (d.codePointAt(0)! <= 0x1f) throw new Failure("Invalid control character at", next);
      }
      if (next >= len) throw new Failure("Unterminated string starting at", begin);
      if (s[next] === '"') return next + 1;
      next += 1; // skip the backslash
      if (next === len) throw new Failure("Unterminated string starting at", begin);
      const c = s[next];
      if (c !== "u") {
        end = next + 1;
        if (!SIMPLE_ESCAPES.includes(c)) throw new Failure("Invalid \\escape", end - 2);
        continue;
      }
      next += 1;
      end = next + 4;
      if (end >= len) throw new Failure("Invalid \\uXXXX escape", next - 1);
      let value = 0;
      for (; next < end; next += 1) {
        const digit = hexValue(s[next]);
        if (digit < 0) throw new Failure("Invalid \\uXXXX escape", end - 5);
        value = value * 16 + digit;
      }
      if (value >= 0xd800 && value <= 0xdbff && end + 6 < len) {
        const backslash = s[next++] === "\\";
        if (backslash && s[next++] === "u") {
          end += 6;
          let low = 0;
          for (; next < end; next += 1) {
            const digit = hexValue(s[next]);
            if (digit < 0) throw new Failure("Invalid \\uXXXX escape", end - 5);
            low = low * 16 + digit;
          }
          if (!(low >= 0xdc00 && low <= 0xdfff)) end -= 6;
        }
      }
    }
  };
  const matchNumber = (start: number): number => {
    let idx = start;
    if (s[idx] === "-") {
      idx += 1;
      if (idx > endIdx) throw new Failure("Expecting value", start, true);
    }
    const c = s[idx];
    if (c >= "1" && c <= "9" && c.length === 1) {
      idx += 1;
      while (idx <= endIdx && isDigit(s[idx])) idx += 1;
    } else if (c === "0") {
      idx += 1;
    } else {
      throw new Failure("Expecting value", start, true);
    }
    if (idx < endIdx && s[idx] === "." && isDigit(s[idx + 1])) {
      idx += 2;
      while (idx <= endIdx && isDigit(s[idx])) idx += 1;
    }
    if (idx < endIdx && (s[idx] === "e" || s[idx] === "E")) {
      const eStart = idx;
      idx += 1;
      if (idx < endIdx && (s[idx] === "-" || s[idx] === "+")) idx += 1;
      while (idx <= endIdx && isDigit(s[idx])) idx += 1;
      if (!isDigit(s[idx - 1])) idx = eStart;
    }
    return idx;
  };
  const word = (idx: number, text: string): boolean => {
    if (idx + text.length - 1 >= len) return false;
    for (let i = 1; i < text.length; i += 1) if (s[idx + i] !== text[i]) return false;
    return true;
  };
  const scanOnce = (idx: number): number => {
    if (idx >= len) throw new Failure("Expecting value", idx, true);
    const c = s[idx];
    if (c === '"') return scanString(idx + 1);
    if (c === "{") return parseObject(idx + 1);
    if (c === "[") return parseArray(idx + 1);
    // C checks idx + n < length before comparing the remaining letters.
    if (c === "n" && word(idx, "null")) return idx + 4;
    if (c === "t" && word(idx, "true")) return idx + 4;
    if (c === "f" && word(idx, "false")) return idx + 5;
    if (c === "N" && word(idx, "NaN")) return idx + 3;
    if (c === "I" && word(idx, "Infinity")) return idx + 8;
    if (c === "-" && word(idx, "-Infinity")) return idx + 9;
    return matchNumber(idx);
  };
  const parseObject = (start: number): number => {
    let idx = skip(start);
    if (idx > endIdx || s[idx] !== "}") {
      for (;;) {
        if (idx > endIdx || s[idx] !== '"') {
          throw new Failure("Expecting property name enclosed in double quotes", idx);
        }
        idx = skip(scanString(idx + 1));
        if (idx > endIdx || s[idx] !== ":") throw new Failure("Expecting ':' delimiter", idx);
        idx = skip(scanOnce(skip(idx + 1)));
        if (idx <= endIdx && s[idx] === "}") break;
        if (idx > endIdx || s[idx] !== ",") throw new Failure("Expecting ',' delimiter", idx);
        idx = skip(idx + 1);
      }
    }
    return idx + 1;
  };
  const parseArray = (start: number): number => {
    let idx = skip(start);
    if (idx > endIdx || s[idx] !== "]") {
      for (;;) {
        idx = skip(scanOnce(idx));
        if (idx <= endIdx && s[idx] === "]") break;
        if (idx > endIdx || s[idx] !== ",") throw new Failure("Expecting ',' delimiter", idx);
        idx = skip(idx + 1);
      }
    }
    return idx + 1;
  };
  const end = skip(scanOnce(skip(0)));
  if (end !== len) throw new Failure("Extra data", end);
}

function format(points: string[], failure: Failure): string {
  let line = 1;
  let lastNewline = -1;
  for (let i = 0; i < failure.pos && i < points.length; i += 1) {
    if (points[i] === "\n") {
      line += 1;
      lastNewline = i;
    }
  }
  return `${failure.msg}: line ${line} column ${failure.pos - lastNewline} (char ${failure.pos})`;
}

/**
 * The JSONDecodeError `json.loads(text)` raises for `text` (a str), or null
 * when CPython accepts it. "Expecting value" errors carry Python's implicit
 * `__context__` StopIteration, as `_failure_evidence` walks into it.
 */
export function pythonJsonDecodeError(text: string, { fromBytes = false } = {}): PyError | null {
  const points = Array.from(text);
  let failure: Failure | null = null;
  if (!fromBytes && points[0] === "﻿") {
    failure = new Failure("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0);
  } else {
    try {
      scan(points);
    } catch (error) {
      if (!(error instanceof Failure)) throw error;
      failure = error;
    }
  }
  if (failure === null) return null;
  const error = new PyError("JSONDecodeError", format(points, failure));
  if (failure.stopIteration) error.context = new PyError("StopIteration", String(failure.pos));
  return error;
}
