// Port of ora/agent/providers/recovery.py: same-credential recovery of one
// logical model call. `same_credential_deadline_v1` bounds the whole call by one
// incident deadline, including active I/O and backoff; `first_token_deadline_v1`
// (the maintained default) has no total limit, bounds each attempt's first
// event by the recovery window and later SSE lines by STALL_S, and retries
// either timeout (a stall as a stream break, first-token timeouts on a budget
// of their own of the same size); a failure before the first event, for any
// reason, is retried when its attempt started before the no-content window (the
// recovery window) had passed since the first such failure, and content closes
// the window. The run's deadline bounds both.

import {
  PolicySessionError, ProviderRecoveryDeadline, PyError, TlsError, TransportError, attr, pyMin,
  pyNext, pyStrip, type PyNum,
} from "./py.ts";
import { STREAM_BREAK_RETRIES, isStreamBreak } from "./stream_failures.ts";

export const DEADLINE_RECOVERY_POLICY = "same_credential_deadline_v1";
export const FIRST_TOKEN_RECOVERY_POLICY = "first_token_deadline_v1";
/** The maintained recovery. */
export const RECOVERY_POLICY = FIRST_TOKEN_RECOVERY_POLICY;
export const RECOVERY_POLICIES: ReadonlySet<string> = new Set([DEADLINE_RECOVERY_POLICY, FIRST_TOKEN_RECOVERY_POLICY]);
export const LEGACY_RECOVERY_POLICY = "legacy_v1";
export const RECOVERY_WINDOW_S = 180.0;
/** first_token_deadline_v1: the longest gap between SSE lines after the first event. */
export const STALL_S = 180.0;
const RETRYABLE_TLS_REASONS = new Set(["SSLV3_ALERT_BAD_RECORD_MAC"]);
// TLS flow states that never carry a certificate verdict.
const NON_TRUST_TLS_STATES = new Set(["SSLWantReadError", "SSLWantWriteError", "SSLZeroReturnError"]);

/** Monotonic and wall clocks in seconds, plus an injectable backoff sleep. */
export interface Clock {
  monotonic(): number;
  time(): number;
  sleep(seconds: number, signal?: AbortSignal): Promise<void>;
}

export const SYSTEM_CLOCK: Clock = {
  monotonic: () => performance.now() / 1000,
  time: () => (performance.timeOrigin + performance.now()) / 1000,
  sleep: (seconds, signal) => new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, Math.max(0, seconds * 1000));
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal!.reason);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  }),
};

// -- Retry-After ---------------------------------------------------------------------

const PY_FLOAT = /^[+-]?(?:(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?|inf(?:inity)?|nan)$/i;

/** Python `float(text)`, or null where Python raises ValueError. */
export function pyParseFloat(text: string): number | null {
  const value = pyStrip(text);
  if (!PY_FLOAT.test(value)) return null;
  const lower = value.toLowerCase();
  const sign = lower.startsWith("-") ? -1 : 1;
  const body = lower.replace(/^[+-]/, "");
  if (body.startsWith("inf")) return sign * Infinity;
  if (body === "nan") return NaN;
  return Number(value.replace(/_/g, ""));
}

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const ZONES: Record<string, number> = { UT: 0, UTC: 0, GMT: 0, Z: 0, AST: -400, ADT: -300, EST: -500, EDT: -400, CST: -600, CDT: -500, MST: -700, MDT: -600, PST: -800, PDT: -700 };
const RFC2822 = /^(?:[A-Za-z]+,?\s*)?(\d{1,2})\s+([A-Za-z]{3})[A-Za-z]*\s+(\d{2,4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s+([A-Za-z]+|[+-]\d{4}))?$/;

/** `email.utils.parsedate_to_datetime(text).timestamp()` for RFC 2822 dates, else null. */
export function parseHttpDate(text: string): number | null {
  const match = RFC2822.exec(pyStrip(text));
  if (!match) return null;
  const [, day, monthName, yearText, hour, minute, second, zone] = match;
  const month = MONTHS.indexOf(monthName.toLowerCase());
  if (month < 0) return null;
  let year = Number(yearText);
  if (yearText.length <= 2) year += year > 68 ? 1900 : 2000;
  const fields = [year, month, Number(day), Number(hour), Number(minute), Number(second ?? 0)] as const;
  // datetime() refuses impossible fields (Feb 30, hour 24, second 60).
  const probe = new Date(Date.UTC(...fields));
  if (probe.getUTCFullYear() !== year || probe.getUTCMonth() !== month || probe.getUTCDate() !== fields[2]
      || probe.getUTCHours() !== fields[3] || probe.getUTCMinutes() !== fields[4] || probe.getUTCSeconds() !== fields[5]) {
    return null;
  }
  let offsetMinutes: number | null = null;
  if (zone !== undefined) {
    if (/^[+-]\d{4}$/.test(zone)) {
      const sign = zone[0] === "-" ? -1 : 1;
      const hhmm = Number(zone.slice(1));
      // "-0000" means "no zone information" (a naive datetime).
      offsetMinutes = zone === "-0000" ? null : sign * (Math.floor(hhmm / 100) * 60 + (hhmm % 100));
    } else if (zone.toUpperCase() in ZONES) {
      const hhmm = ZONES[zone.toUpperCase()];
      offsetMinutes = Math.sign(hhmm) * (Math.floor(Math.abs(hhmm) / 100) * 60);
    } else {
      offsetMinutes = null;
    }
  }
  if (offsetMinutes === null) {
    // A naive datetime: Python's .timestamp() reads it as local time.
    return new Date(...fields).getTime() / 1000;
  }
  return (Date.UTC(...fields) - offsetMinutes * 60_000) / 1000;
}

export function retryAfterSeconds(value: string | null | undefined, clock: Clock = SYSTEM_CLOCK): number | null {
  if (value === null || value === undefined) return null;
  let seconds = pyParseFloat(value);
  if (seconds === null) {
    const timestamp = parseHttpDate(value);
    if (timestamp === null) return null;
    seconds = timestamp - clock.time();
  }
  return Number.isFinite(seconds) ? Math.max(0.0, seconds) : null;
}

// -- Transport classification -------------------------------------------------------

/** A transport fault worth one more same-request attempt; bad TLS trust fails closed. */
export function transientTransport(exc: unknown): boolean {
  let retryableSsl = false;
  let transport = false;
  const seen = new Set<unknown>();
  let current: unknown = exc;
  while (current !== undefined && current !== null && !seen.has(current)) {
    seen.add(current);
    if (current instanceof TlsError && !NON_TRUST_TLS_STATES.has(current.pyType)) {
      const reason = (current.reason || "").toUpperCase();
      const message = current.message.toUpperCase();
      const alert = [...RETRYABLE_TLS_REASONS].some(
        (candidate) => reason === candidate || message.includes(`[SSL: ${candidate}]`),
      );
      if (current.pyType === "SSLEOFError" || alert) retryableSsl = true;
      else return false;
    }
    if (current instanceof TransportError || (current instanceof PyError && current.pyType === "EOFError")) {
      transport = true;
    }
    current = pyNext(current);
  }
  return retryableSsl || transport;
}

// -- The incident window ------------------------------------------------------------

export interface RecoveryBinding {
  recoveryPolicy: string;
  recoveryWindowS: number | null;
  /** The run budget's deadline on the clock's monotonic scale (seconds), if any. */
  requestDeadlineMonotonic: number | null;
  preflightProbe: boolean;
  maxRetries: number;
  /** first_token_deadline_v1's stall limit; STALL_S unless a test shortens it. */
  stallS?: number | null;
}

export interface AttemptLimits {
  policy: string;
  firstTokenS: number | null;
  stallS: number | null;
  /** first_token_deadline_v1: how long failures before content are retried; firstTokenS when absent. */
  noContentWindowS?: number | null;
}

/**
 * One logical request's incident: its deadline, attempt limits and counted retries.
 * Under first_token_deadline_v1 `deadline` is the run's (Infinity without one) and
 * `noContentWindowS` bounds how long failures before content are retried, from `noContentSince`.
 */
export class RecoveryWindow {
  deadline: number;
  retriesEnabled: boolean;
  streamBreaks = 0;
  firstTokenTimeouts = 0;
  noContentFailures = 0;
  policy: string;
  firstTokenS: number | null;
  stallS: number | null;
  noContentWindowS: number | null;
  /** When the open no-content window opened (clock.monotonic()); null while none is open. */
  noContentSince: number | null = null;
  clock: Clock;

  constructor(deadline: number, retriesEnabled: boolean, clock: Clock = SYSTEM_CLOCK,
    limits: AttemptLimits = { policy: DEADLINE_RECOVERY_POLICY, firstTokenS: null, stallS: null }) {
    this.deadline = deadline;
    this.retriesEnabled = retriesEnabled;
    this.clock = clock;
    this.policy = limits.policy;
    this.firstTokenS = limits.firstTokenS;
    this.stallS = limits.stallS;
    this.noContentWindowS = limits.noContentWindowS
      ?? (limits.policy === FIRST_TOKEN_RECOVERY_POLICY ? limits.firstTokenS : null);
  }

  static forProvider(provider: RecoveryBinding, clock: Clock = SYSTEM_CLOCK): RecoveryWindow | null {
    if (provider.recoveryPolicy === LEGACY_RECOVERY_POLICY) return null;
    if (!RECOVERY_POLICIES.has(provider.recoveryPolicy)) {
      throw new PolicySessionError("unsupported native provider recovery policy");
    }
    const window = provider.recoveryWindowS || RECOVERY_WINDOW_S;
    let deadline = clock.monotonic() + window;
    const remainingRun = provider.requestDeadlineMonotonic;
    if (remainingRun !== null && remainingRun !== undefined) {
      if (typeof remainingRun !== "number" || !Number.isFinite(remainingRun)) {
        throw new PolicySessionError("provider run deadline is invalid");
      }
      deadline = Math.min(deadline, remainingRun);
    }
    const enabled = !provider.preflightProbe && provider.maxRetries > 0;
    if (provider.recoveryPolicy === FIRST_TOKEN_RECOVERY_POLICY) {
      // No incident deadline: only the run's own deadline bounds the call.
      const run = remainingRun === null || remainingRun === undefined ? Infinity : remainingRun;
      return new RecoveryWindow(run, enabled, clock, {
        policy: FIRST_TOKEN_RECOVERY_POLICY, firstTokenS: window, stallS: provider.stallS ?? STALL_S,
        noContentWindowS: window,
      });
    }
    return new RecoveryWindow(deadline, enabled, clock);
  }

  remaining(): number {
    return Math.max(0.0, this.deadline - this.clock.monotonic());
  }

  delay(exc: unknown, attempt: number): PyNum | null {
    const status = attr(exc, "httpStatus", null);
    if (!this.retriesEnabled || exc instanceof ProviderRecoveryDeadline
        || status === 401 || status === 402 || status === 403
        || !(attr(exc, "retryable", false) || transientTransport(exc))) {
      return null;
    }
    const hint = attr(exc, "retryDelayS", null);
    const exponent = 2 ** Math.min(attempt - 1, 5);
    const delay: PyNum = typeof hint === "number" && Number.isFinite(hint) && hint > 0
      ? { value: hint, isFloat: true }
      : pyMin({ value: exponent, isFloat: false }, { value: 20.0, isFloat: true });
    // Do not start a known-too-early request when Retry-After exceeds the budget.
    return this.remaining() > 0 ? pyMin(delay, { value: this.remaining(), isFloat: true }) : null;
  }

  /**
   * `delay` for a transport-phase failure, bounding stream breaks; `started` is the
   * failed attempt's start (clock.monotonic()). Under first_token_deadline_v1 a
   * failure before content opens the no-content window when none is open (when it
   * is seen) and is retried only when its attempt started less than
   * `noContentWindowS` after the window opened (when seen, a first-token timeout
   * inside an open window, its limit equal to the window, would always end the
   * call); a break after content closes it.
   */
  transportDelay(exc: unknown, attempt: number, started: number): PyNum | null {
    let delay = this.delay(exc, attempt);
    if (isStreamBreak(exc)) {
      this.streamBreaks += 1;
      if (this.streamBreaks > STREAM_BREAK_RETRIES) delay = null;
    }
    if (this.policy === FIRST_TOKEN_RECOVERY_POLICY) {
      // A stall followed content, so it is a stream break; first-token timeouts count apart.
      if (attr(exc, "attemptTimeout", null) === "first_token") {
        this.firstTokenTimeouts += 1;
        if (this.firstTokenTimeouts > STREAM_BREAK_RETRIES) delay = null;
      }
      let elapsed: number | null = null;
      if (isStreamBreak(exc)) {
        this.noContentSince = null; // content arrived: a later failure opens a new window
      } else {
        if (this.noContentSince === null) this.noContentSince = this.clock.monotonic();
        // The window's age when the failed attempt started; 0.0 if it started before the window opened.
        elapsed = Math.max(0.0, started - this.noContentSince);
        this.noContentFailures += 1;
        if (this.noContentWindowS !== null && elapsed >= this.noContentWindowS) delay = null;
      }
      if (exc !== null && typeof exc === "object") {
        if (elapsed !== null) (exc as any).noContentElapsedS = elapsed;
        (exc as any).firstTokenTimeouts = this.firstTokenTimeouts;
        (exc as any).noContentFailures = this.noContentFailures;
        (exc as any).noContentWindowS = this.noContentWindowS;
      }
    }
    if (exc !== null && typeof exc === "object") (exc as any).streamBreaks = this.streamBreaks;
    return delay;
  }

  async pause(delay: PyNum, signal?: AbortSignal): Promise<void> {
    await this.clock.sleep(Math.min(delay.value, this.remaining()), signal);
    if (this.remaining() <= 0) throw new ProviderRecoveryDeadline();
  }
}
