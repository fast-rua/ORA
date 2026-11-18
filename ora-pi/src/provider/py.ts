// Python-compatible helpers and exception types for the provider port.
//
// Evidence records name Python exception classes (`error_chain[].type`,
// `provider_failure.exception_type`) and copy Python attribute semantics
// (`getattr(exc, "retryable", False)`), so the TypeScript errors carry the
// Python class name and the same optional attributes.

import { createHash } from "node:crypto";
import { PyFloat } from "../pyjson.ts";

export function sha256Hex(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** `x` written as a Python float: integral values become PyFloat. */
export function pyFloat(x: number): number | PyFloat {
  return Number.isInteger(x) || Object.is(x, -0) ? new PyFloat(x) : x;
}

/** A Python int or float operand (`min(1, 20.0)` keeps the int 1). */
export interface PyNum {
  value: number;
  isFloat: boolean;
}

export function pyNumValue(n: PyNum): number | PyFloat {
  return n.isFloat ? pyFloat(n.value) : n.value;
}

/** Python `min(a, b)`: the first operand unless the second is smaller. */
export function pyMin(a: PyNum, b: PyNum): PyNum {
  return b.value < a.value ? b : a;
}

/** `round(x, 6)` for finite doubles below 1e21. */
export function pyRound6(x: number): number {
  return Number(x.toFixed(6));
}

// Python str.strip() whitespace (str.isspace code points).
const PY_SPACE = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a,
  0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

/** Python `str.strip()` with no argument. */
export function pyStrip(text: string): string {
  let start = 0;
  let end = text.length;
  while (start < end && PY_SPACE.has(text.charCodeAt(start))) start += 1;
  while (end > start && PY_SPACE.has(text.charCodeAt(end - 1))) end -= 1;
  return text.slice(start, end);
}

/** Python `text[-n:]` on code points. */
export function pyTail(text: string, n: number): string {
  if (text.length <= n) return text;
  const points = Array.from(text);
  return points.length <= n ? text : points.slice(points.length - n).join("");
}

// Python str.splitlines() boundaries.
const PY_NEWLINES = "\n\r\x0b\x0c\x1c\x1d\x1e\x85  ";

/** Python `str.splitlines()` (no keepends). */
export function pySplitlines(text: string): string[] {
  const lines: string[] = [];
  let start = 0;
  let i = 0;
  while (i < text.length) {
    const char = text[i];
    if (PY_NEWLINES.includes(char)) {
      lines.push(text.slice(start, i));
      i += char === "\r" && text[i + 1] === "\n" ? 2 : 1;
      start = i;
    } else {
      i += 1;
    }
  }
  if (start < text.length) lines.push(text.slice(start));
  return lines;
}

/** httpx `LineDecoder`: splitlines semantics over incremental text. */
export class LineDecoder {
  buffer: string[] = [];
  trailingCr = false;

  decode(input: string): string[] {
    let text = input;
    if (this.trailingCr) {
      text = "\r" + text;
      this.trailingCr = false;
    }
    if (text.endsWith("\r")) {
      this.trailingCr = true;
      text = text.slice(0, -1);
    }
    if (!text) return [];
    const trailingNewline = PY_NEWLINES.includes(text[text.length - 1]);
    let lines = pySplitlines(text);
    if (lines.length === 1 && !trailingNewline) {
      this.buffer.push(lines[0]);
      return [];
    }
    if (this.buffer.length) {
      lines = [this.buffer.join("") + lines[0], ...lines.slice(1)];
      this.buffer = [];
    }
    if (!trailingNewline) this.buffer = [lines.pop()!];
    return lines;
  }

  flush(): string[] {
    if (!this.buffer.length && !this.trailingCr) return [];
    const lines = [this.buffer.join("")];
    this.buffer = [];
    this.trailingCr = false;
    return lines;
  }
}

// -- Exceptions --------------------------------------------------------------------

/** An error that names the Python exception class it stands for. */
export class PyError extends Error {
  pyType: string;
  /** Python `__context__` (the cause chain reads `__cause__ or __context__`). */
  context?: unknown;
  constructor(pyType: string, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.pyType = pyType;
    this.name = pyType;
  }
}

export interface ProviderFailure {
  exception_type: string;
  cause_type: string | null;
  attempts: number;
  response_started: boolean;
}

/** `ora.agent.contracts.PolicySessionError` with the provider's attributes. */
export class PolicySessionError extends PyError {
  // Optional attributes: `undefined` means Python's getattr default applies.
  retryable?: boolean;
  httpStatus?: number | null;
  retryDelayS?: number | null;
  responseStarted?: boolean;
  partialEvents?: any[];
  upstreamStreamBreak?: boolean;
  credentialEcho?: boolean;
  streamBreaks?: number;
  /** first_token_deadline_v1: the attempt limit that fired and its length. */
  attemptTimeout?: "first_token" | "stall";
  attemptTimeoutS?: number;
  firstTokenTimeouts?: number;
  /** first_token_deadline_v1: failures before content so far, the window they are retried in, its age. */
  noContentFailures?: number;
  noContentWindowS?: number | null;
  noContentElapsedS?: number;
  remoteExceptionType?: string | null;
  remoteErrorChain?: any[];
  providerFailure?: ProviderFailure;
  constructor(message: string, options?: { cause?: unknown; pyType?: string }) {
    super(options?.pyType ?? "PolicySessionError", message, options);
  }
}

export class ProviderRecoveryDeadline extends PolicySessionError {
  constructor(events: any[] = []) {
    super("native provider recovery deadline exhausted; no reply was accepted", {
      pyType: "ProviderRecoveryDeadline",
    });
    this.retryable = false;
    this.partialEvents = [...events];
    this.responseStarted = events.length > 0;
  }
}

/** `recovery.ATTEMPT_TIMEOUT_MESSAGES`. */
export const ATTEMPT_TIMEOUT_MESSAGES = {
  first_token: "native provider attempt timed out before its first event",
  stall: "native provider stream stalled after response content",
} as const;

/** `recovery.ProviderAttemptTimeout`: one first_token_deadline_v1 attempt exceeded a limit. */
export class ProviderAttemptTimeout extends PolicySessionError {
  constructor(limit: "first_token" | "stall", limitS: number, events: any[] = []) {
    super(ATTEMPT_TIMEOUT_MESSAGES[limit], { pyType: "ProviderAttemptTimeout" });
    this.retryable = true;
    this.attemptTimeout = limit;
    this.attemptTimeoutS = limitS;
    this.partialEvents = [...events];
    this.responseStarted = events.length > 0;
  }
}

export class ProviderClosed extends PolicySessionError {
  constructor() {
    super("native provider is closed; no request was sent", { pyType: "ProviderClosed" });
    this.retryable = false;
  }
}

/** `attempts._StreamInterrupted`. */
export class StreamInterrupted extends PolicySessionError {
  constructor(events: any[], transient: boolean, cause?: unknown) {
    super("native provider stream failed before message_stop", {
      pyType: "_StreamInterrupted", cause,
    });
    this.partialEvents = [...events];
    this.retryable = transient;
    this.responseStarted = this.partialEvents.length > 0;
  }
}

/** Python `EOFError`, which `transient_transport` admits. */
export class PyEOFError extends PyError {
  constructor(message: string) {
    super("EOFError", message);
  }
}

/** An httpx `TransportError` subclass (`ConnectError`, `ReadTimeout`, ...). */
export class TransportError extends PyError {
  code?: string;
  constructor(pyType: string, message: string, options?: { cause?: unknown; code?: string }) {
    super(pyType, message, options);
    this.code = options?.code;
  }
}

/** A Python TLS failure (`ssl.SSLError` family). */
export class TlsError extends PyError {
  reason: string;
  constructor(pyType: string, message: string, reason: string, options?: { cause?: unknown }) {
    super(pyType, message, options);
    this.reason = reason;
  }
}

/** The next frame of Python's `__cause__ or __context__` walk. */
export function pyNext(error: unknown): unknown {
  if (error === null || typeof error !== "object") return undefined;
  const cause = (error as any).cause;
  if (cause !== undefined && cause !== null) return cause;
  const context = (error as any).context;
  return context === null ? undefined : context;
}

/** The Python class name an error stands for. */
export function pyTypeName(error: unknown): string {
  if (error instanceof PyError) return error.pyType;
  if (error instanceof Error) return error.name || "Exception";
  return "Exception";
}

export function pyMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

/** Python attribute read with the same defaults as `getattr(exc, name, default)`. */
export function attr<T>(error: unknown, name: keyof PolicySessionError, fallback: T): any {
  if (error === null || typeof error !== "object") return fallback;
  const value = (error as any)[name];
  return value === undefined ? fallback : value;
}
