// Port of ora/agent/providers/attempts.py: transport-attempt diagnostics and
// the `ora.provider-attempt-failure.v1` record (handed to a callback; the
// evidence writer persists it, as the retired Python Native core's
// ora/agent/native/provider_calls.py did).

import { dumpsPy, parsePy } from "../pyjson.ts";
import {
  PolicySessionError, attr, pyFloat, pyMessage, pyNext, pyNumValue, pyTypeName, type ProviderFailure,
  type PyNum,
} from "./py.ts";
import { STREAM_BREAK_RETRIES } from "./stream_failures.ts";

export const ATTEMPT_FAILURE_SCHEMA = "ora.provider-attempt-failure.v1";
export const CREDENTIAL_ECHO_MESSAGE = "native provider echoed credential material; response was discarded";

function escapedKey(key: string): string {
  return dumpsPy(key, { ensureAscii: false }).slice(1, -1);
}

/** True when raw provider content contains a credential, before redaction. */
export function echoesCredential(events: any[], keys: readonly string[]): boolean {
  const serialized = dumpsPy([...events], { ensureAscii: false });
  return keys.some((key) => Boolean(key) && serialized.includes(escapedKey(key)));
}

/** A broken attempt echoed a credential in the content it streamed. */
export function partialCredentialEcho(exc: unknown, keys: readonly string[]): boolean {
  if (attr(exc, "credentialEcho", false) === true) return true;
  const partial = attr(exc, "partialEvents", null);
  const echoed = echoesCredential(Array.isArray(partial) ? partial : [], keys);
  if (echoed && exc !== null && typeof exc === "object") (exc as any).credentialEcho = true;
  return echoed;
}

/** Fail closed exactly as an echoing complete response does. */
export function credentialEchoFailure(cause?: unknown): PolicySessionError {
  const failure = new PolicySessionError(CREDENTIAL_ECHO_MESSAGE, { cause });
  failure.retryable = false;
  failure.credentialEcho = true;
  failure.responseStarted = true;
  return failure;
}

/** `transport_failures._record_provider_failure`: bounded metadata on the raised exception. */
export function recordProviderFailure(exc: unknown, attempts: number): void {
  if (exc === null || typeof exc !== "object") return;
  const cause = (exc as any).cause;
  const failure: ProviderFailure = {
    exception_type: pyTypeName(exc),
    cause_type: cause !== undefined && cause !== null ? pyTypeName(cause) : null,
    attempts,
    response_started: Boolean(attr(exc, "responseStarted", false)),
  };
  (exc as any).providerFailure = failure;
}

export function echoFailure(cause: unknown, attempts: number): PolicySessionError {
  const failure = credentialEchoFailure(cause);
  recordProviderFailure(failure, attempts);
  return failure;
}

/** `_failure_evidence`: the bounded error chain and partial events, redacted. */
export function failureEvidence(
  exc: unknown, keys: readonly string[], events: any[],
): { error_chain: any[]; partial_events: any[]; credentials_redacted: boolean } {
  const chain: Array<{ type: string; message: string }> = [];
  const seen = new Set<unknown>();
  let current: unknown = exc;
  while (current !== undefined && current !== null && !seen.has(current) && chain.length < 8) {
    seen.add(current);
    chain.push({ type: pyTypeName(current), message: pyMessage(current) });
    const remote = attr(current, "remoteErrorChain", []);
    if (Array.isArray(remote)) {
      for (const entry of remote) {
        if (chain.length < 8 && typeof entry === "object" && entry !== null && !Array.isArray(entry)
            && typeof entry.type === "string" && typeof entry.message === "string") {
          chain.push({ type: entry.type, message: entry.message });
        }
      }
    }
    current = pyNext(current);
  }
  const original = dumpsPy({ error_chain: chain, partial_events: [...events] }, { ensureAscii: false });
  let redacted = original;
  const unique = [...new Set(keys)].sort((a, b) => b.length - a.length);
  for (const key of unique) {
    if (key) redacted = redacted.split(escapedKey(key)).join("[REDACTED]");
  }
  return { ...parsePy(redacted), credentials_redacted: redacted !== original };
}

export interface AttemptRecordContext {
  recoveryPolicy: string;
  keys: readonly string[];
  requestSha256: string;
  modelCallId: string;
  attempt: number;
  credentialIndex: number;
  nextIndex: number | null;
  startedUnix: number;
  startedMonotonic: number;
  nowUnix: number;
  nowMonotonic: number;
  retryDelay: PyNum | 0;
  events: any[];
  phase: "transport" | "parse";
}

/** `record_attempt_failure`: the exact `ora.provider-attempt-failure.v1` record. */
export function attemptFailureRecord(exc: unknown, ctx: AttemptRecordContext): { [key: string]: any } {
  const streamBreaks = attr(exc, "streamBreaks", null);
  const attemptTimeout = attr(exc, "attemptTimeout", null);
  const firstTokenTimeouts = attr(exc, "firstTokenTimeouts", null);
  const noContentFailures = attr(exc, "noContentFailures", null);
  const noContentWindow = attr(exc, "noContentWindowS", null);
  const noContentElapsed = attr(exc, "noContentElapsedS", null);
  return {
    schema: ATTEMPT_FAILURE_SCHEMA,
    model_call_id: ctx.modelCallId,
    attempt: ctx.attempt,
    request_sha256: ctx.requestSha256,
    credential_index: ctx.credentialIndex,
    next_credential_index: ctx.nextIndex,
    http_status: attr(exc, "httpStatus", null),
    phase: ctx.phase,
    started_unix_s: pyFloat(ctx.startedUnix),
    finished_unix_s: pyFloat(ctx.nowUnix),
    duration_s: pyFloat(Math.max(0.0, ctx.nowMonotonic - ctx.startedMonotonic)),
    retry_scheduled: ctx.nextIndex !== null,
    retry_delay_s: ctx.retryDelay === 0 ? 0 : pyNumValue(ctx.retryDelay),
    partial_event_count: ctx.events.length,
    tool_execution_started: false,
    ...failureEvidence(exc, ctx.keys, ctx.events),
    ...(ctx.recoveryPolicy !== "legacy_v1"
      ? { recovery_policy: ctx.recoveryPolicy, stream_break_retry_limit: STREAM_BREAK_RETRIES } : {}),
    ...(typeof streamBreaks === "number" && Number.isInteger(streamBreaks) ? { stream_breaks: streamBreaks } : {}),
    // first_token_deadline_v1: the attempt limit that fired, if one did, and its length.
    ...(attemptTimeout === "first_token" || attemptTimeout === "stall"
      ? { attempt_timeout: attemptTimeout, attempt_timeout_s: pyFloat(attr(exc, "attemptTimeoutS", 0)) } : {}),
    // first_token_deadline_v1: first-token timeouts so far in this logical call.
    ...(typeof firstTokenTimeouts === "number" && Number.isInteger(firstTokenTimeouts)
      ? { first_token_timeouts: firstTokenTimeouts } : {}),
    // first_token_deadline_v1: attempts so far that failed before their first event, and
    // the no-content window: an attempt that starts inside it is retried.
    ...(typeof noContentFailures === "number" && Number.isInteger(noContentFailures)
      ? { no_content_failures: noContentFailures,
        no_content_window_s: typeof noContentWindow === "number" ? pyFloat(noContentWindow) : null } : {}),
    // first_token_deadline_v1, a failure before content: the window's age when its attempt started.
    ...(typeof noContentElapsed === "number" ? { no_content_elapsed_s: pyFloat(noContentElapsed) } : {}),
    ...(attr(exc, "upstreamStreamBreak", false) === true ? { upstream_stream_break: true } : {}),
    ...(attr(exc, "credentialEcho", false) === true ? { credential_echo: true, credentials_redacted: true } : {}),
  };
}
