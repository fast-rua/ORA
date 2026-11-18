// One Chat Completions HTTP attempt over fetch.
//
// `boundedRequest` replaces recovery.py's `bounded_request` (parent) +
// `recovery_worker` + `_bounded_request_in_process` (worker subprocess): the
// attempt runs in-process under an AbortSignal bounded by the incident
// deadline, and a failed attempt is re-shaped exactly as the Python parent
// reconstructs the worker's result frame (a PolicySessionError carrying the
// worker's verdict, redacted message, partial events and remote error chain).
// `requestOnce` is `OpenAICompatibleProvider._request_once` (legacy_v1).

import { dumpsPy, parsePy } from "../pyjson.ts";
import { echoesCredential, failureEvidence } from "./attempts.ts";
import type { OraFetchInit } from "./http.ts";
import { pythonJsonDecodeError } from "./json_errors.ts";
import {
  LineDecoder, PolicySessionError, ProviderAttemptTimeout, ProviderRecoveryDeadline, PyEOFError, PyError,
  StreamInterrupted, TlsError, TransportError, attr, pyMessage, pyStrip, pyTail, pyTypeName,
} from "./py.ts";
import {
  type Clock, FIRST_TOKEN_RECOVERY_POLICY, RecoveryWindow, retryAfterSeconds, transientTransport,
} from "./recovery.ts";
import { httpStatusFailure, streamFailureAfterContent } from "./stream_failures.ts";

export interface TransportBinding {
  baseUrl: string;
  apiKey: string;
  timeoutSeconds: number;
  sessionAffinityId: string | null;
  fetchImpl: typeof fetch;
  clock: Clock;
  /** Anthropic Messages: its URL and headers, and message_stop/error as the terminal event. */
  anthropic?: { url: string; headers: () => Record<string, string> };
}

export interface AttemptResult {
  events: any[];
  first: number | null;
}

/** The request headers ORA controls (fetch adds its own user-agent/accept). */
export function requestHeaders(binding: { apiKey: string; sessionAffinityId: string | null }): Record<string, string> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (binding.sessionAffinityId !== null) headers["x-session-id"] = binding.sessionAffinityId;
  headers.authorization = "Bearer " + binding.apiKey;
  return headers;
}

function isMapping(value: unknown): value is { [key: string]: any } {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && Object.getPrototypeOf(value) === Object.prototype;
}

// -- fetch error classification ---------------------------------------------------

const TLS_TRUST_CODES = new Set([
  "DEPTH_ZERO_SELF_SIGNED_CERT", "SELF_SIGNED_CERT_IN_CHAIN", "UNABLE_TO_VERIFY_LEAF_SIGNATURE",
  "UNABLE_TO_GET_ISSUER_CERT", "UNABLE_TO_GET_ISSUER_CERT_LOCALLY", "CERT_HAS_EXPIRED", "CERT_NOT_YET_VALID",
  "CERT_REVOKED", "CERT_UNTRUSTED", "CERT_REJECTED", "ERR_TLS_CERT_ALTNAME_INVALID", "HOSTNAME_MISMATCH",
  "CERT_SIGNATURE_FAILURE", "INVALID_CA", "UNABLE_TO_DECRYPT_CERT_SIGNATURE",
]);

function rootCause(error: unknown): any {
  let current: any = error;
  for (let depth = 0; depth < 8 && current && typeof current === "object" && current.cause; depth += 1) {
    current = current.cause;
  }
  return current;
}

/**
 * httpx re-raises httpcore's exception (`from exc`), which wraps h11's for a
 * protocol failure in the body: the Python cause chain repeats the message.
 */
function nested(pyType: string, message: string, depth: number, code: string): TransportError {
  let error: TransportError | undefined;
  for (let i = 0; i < depth; i += 1) {
    error = new TransportError(pyType, message, error === undefined ? { code } : { cause: error, code });
  }
  return error!;
}

/**
 * Map a fetch/undici failure onto the httpx exception the Python worker sees.
 * `stage` is "connect" (before response headers) or "body" (while streaming).
 */
export function pythonTransportError(
  error: unknown, stage: "connect" | "body", flavor: "sync" | "async" = "async",
): PyError {
  if (error instanceof PyError) return error;
  const root = rootCause(error);
  const code: string = typeof root?.code === "string" ? root.code : "";
  const text = pyMessage(root);
  if (TLS_TRUST_CODES.has(code) || /certificate/i.test(text)) {
    const tls = new TlsError("SSLCertVerificationError",
      `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: ${text}`, "CERTIFICATE_VERIFY_FAILED");
    return new TransportError("ConnectError", tls.message, { cause: tls, code });
  }
  if (/BAD_RECORD_MAC/i.test(code) || /bad record mac/i.test(text)) {
    const tls = new TlsError("SSLError", `[SSL: SSLV3_ALERT_BAD_RECORD_MAC] ${text}`, "SSLV3_ALERT_BAD_RECORD_MAC");
    return new TransportError(stage === "connect" ? "ConnectError" : "ReadError", tls.message, { cause: tls, code });
  }
  if (code === "UND_ERR_CONNECT_TIMEOUT") return timeoutError("ConnectTimeout", flavor, code);
  if (code === "UND_ERR_HEADERS_TIMEOUT" || code === "UND_ERR_BODY_TIMEOUT") return timeoutError("ReadTimeout", flavor, code);
  if (stage === "connect") {
    const errno = SOCKET_ERRNO[code];
    if (errno !== undefined && flavor === "sync") {
      // httpx.ConnectError <- httpcore.ConnectError <- the socket's OSError.
      const os = new PyError(errno[0], errno[1]);
      return new TransportError("ConnectError", errno[1], {
        cause: new TransportError("ConnectError", errno[1], { cause: os, code }), code,
      });
    }
    if (errno !== undefined) {
      // anyio (the async worker) wraps the per-address failures in one OSError.
      const address = typeof root?.address === "string" && /^\d+\.\d+\.\d+\.\d+$/.test(root.address)
        ? root.address : null;
      if (code !== "ECONNREFUSED" || address === null) {
        return nested("ConnectError", "All connection attempts failed", 2, code);
      }
      const refused = new PyError("ConnectionRefusedError", `[Errno 111] Connect call failed ('${address}', ${root.port})`);
      const os = new PyError("OSError", "All connection attempts failed", { cause: refused });
      return new TransportError("ConnectError", "All connection attempts failed", {
        cause: new TransportError("ConnectError", "All connection attempts failed", { cause: os, code }), code,
      });
    }
    if (code === "UND_ERR_SOCKET" || code === "ECONNRESET" || code === "EPIPE") {
      return nested("RemoteProtocolError", "Server disconnected without sending a response.", 2, code);
    }
    return new TransportError("ConnectError", text || "fetch failed", { code });
  }
  if (code === "UND_ERR_SOCKET" || code === "ECONNRESET" || /terminated/.test(pyMessage(error))) {
    return nested("RemoteProtocolError",
      "peer closed connection without sending complete message body (incomplete chunked read)", 3, code);
  }
  return new TransportError("ReadError", text, { code });
}

/**
 * An httpx timeout as Python raises it: httpx's exception from httpcore's,
 * from the socket's TimeoutError ("timed out") in the sync client, or from
 * anyio's TimeoutError and the CancelledError of its `fail_after` scope in
 * the async worker (whose message names the scope's id(), a memory address).
 */
function timeoutError(pyType: "ConnectTimeout" | "ReadTimeout", flavor: "sync" | "async", code = ""): TransportError {
  const message = flavor === "sync" ? "timed out" : "";
  const root = flavor === "sync"
    ? new PyError("TimeoutError", message)
    : new PyError("TimeoutError", "", {
      cause: new PyError("CancelledError", `Cancelled via cancel scope ${cancelScopeId()}; reason: deadline exceeded`),
    });
  return new TransportError(pyType, message, { cause: new TransportError(pyType, message, { cause: root, code }), code });
}

function cancelScopeId(): string {
  return (0x7f0000000000 + Math.floor(Math.random() * 0xffffffffff) * 16).toString(16);
}

// Python's view of a failed connect(): (exception class, str(exc)) on Linux.
const SOCKET_ERRNO: Record<string, [string, string]> = {
  ECONNREFUSED: ["ConnectionRefusedError", "[Errno 111] Connection refused"],
  EHOSTUNREACH: ["OSError", "[Errno 113] No route to host"],
  ENETUNREACH: ["OSError", "[Errno 101] Network is unreachable"],
  EADDRNOTAVAIL: ["OSError", "[Errno 99] Cannot assign requested address"],
  ENOTFOUND: ["gaierror", "[Errno -2] Name or service not known"],
  EAI_AGAIN: ["gaierror", "[Errno -3] Temporary failure in name resolution"],
};

// -- body reading ---------------------------------------------------------------------

type AbortKind = "deadline" | "caller" | "read_timeout" | "body_timeout" | "first_token" | "stall";

/** The attempt is over by its own end: the deadline, the caller, or an attempt limit. */
function ended(control: AttemptControl): boolean {
  return control.kind === "deadline" || control.kind === "caller" || control.kind === "first_token"
    || control.kind === "stall";
}

export class AttemptControl {
  controller = new AbortController();
  /** httpx.Client (legacy_v1) or the async worker's httpx.AsyncClient. */
  flavor: "sync" | "async" = "async";
  kind: AbortKind | null = null;
  private timers: ReturnType<typeof setTimeout>[] = [];
  private inactivity: ReturnType<typeof setTimeout> | null = null;
  private inactivityMs = 0;
  private progressTimer: ReturnType<typeof setTimeout> | null = null;
  private stallS: number | null = null;
  private eventSeen = false;
  private callerSignal?: AbortSignal;
  private onCaller = () => this.abort("caller");

  constructor(callerSignal?: AbortSignal) {
    this.callerSignal = callerSignal;
    if (callerSignal?.aborted) this.abort("caller");
    callerSignal?.addEventListener("abort", this.onCaller, { once: true });
  }
  abort(kind: AbortKind): void {
    if (this.kind === null) this.kind = kind;
    this.controller.abort(new PyError("CancelledError", kind));
  }
  after(seconds: number, kind: AbortKind): void {
    this.timers.push(setTimeout(() => this.abort(kind), Math.max(0, seconds * 1000)));
  }
  /**
   * httpx's per-operation read timeout. The headers wait belongs to the
   * dispatcher (http.ts, timed from the written request as httpx times it);
   * the first `touch()`, once the headers are in, arms it for the body, where
   * every received chunk restarts it.
   */
  readTimeout(seconds: number): void {
    this.readTimeoutS = seconds;
    this.inactivityMs = Math.max(0, seconds * 1000);
  }
  readTimeoutS: number | null = null;
  touch(): void {
    if (this.inactivity !== null) clearTimeout(this.inactivity);
    if (this.inactivityMs > 0) this.inactivity = setTimeout(() => this.abort("read_timeout"), this.inactivityMs);
  }
  /**
   * first_token_deadline_v1 (`_bounded_request_in_process.watched`): the first
   * event within `firstTokenS` of the attempt's start, then no gap between SSE
   * lines longer than `stallS`.
   */
  watchProgress(firstTokenS: number, stallS: number): void {
    this.stallS = stallS;
    this.progressTimer = setTimeout(() => this.abort("first_token"), Math.max(0, firstTokenS * 1000));
  }
  /** An SSE line arrived; after the first event it restarts the stall limit. */
  line(): void {
    if (this.eventSeen) this.restartStall();
  }
  /** An SSE event was parsed; the first one ends the first-token limit. */
  event(): void {
    if (this.stallS === null || this.eventSeen) return;
    this.eventSeen = true;
    this.restartStall();
  }
  private restartStall(): void {
    if (this.stallS === null) return;
    if (this.progressTimer !== null) clearTimeout(this.progressTimer);
    this.progressTimer = setTimeout(() => this.abort("stall"), Math.max(0, this.stallS * 1000));
  }
  dispose(): void {
    for (const timer of this.timers) clearTimeout(timer);
    if (this.inactivity !== null) clearTimeout(this.inactivity);
    if (this.progressTimer !== null) clearTimeout(this.progressTimer);
    this.callerSignal?.removeEventListener("abort", this.onCaller);
  }
}

export function abortedError(control: AttemptControl, error: unknown, stage: "connect" | "body"): PyError {
  if (control.kind === "read_timeout") return timeoutError("ReadTimeout", control.flavor);
  if (control.kind !== null) return new PyError("CancelledError", control.kind);
  return pythonTransportError(error, stage, control.flavor);
}

export async function readAll(response: Response, control: AttemptControl): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  const reader = response.body?.getReader();
  if (!reader) return new Uint8Array();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      control.touch();
      chunks.push(value);
    }
  } catch (error) {
    throw abortedError(control, error, "body");
  }
  return Buffer.concat(chunks);
}

export function decodeReplace(bytes: Uint8Array): string {
  return new TextDecoder("utf-8", { fatal: false, ignoreBOM: true }).decode(bytes);
}

/** `json.loads(text)` failures as CPython words them (see json_errors.ts). */
export function loadsText(text: string, fromBytes: boolean): any {
  try {
    return parsePy(text);
  } catch (error) {
    // null: CPython accepts what parsePy refuses (NaN, Infinity, 1e400).
    throw pythonJsonDecodeError(text, { fromBytes })
      ?? new PyError("ValueError", `JSON accepted by Python but not representable here: ${pyMessage(error)}`);
  }
}

/** `json.loads(bytes)`: utf-8 (a BOM is dropped, as utf-8-sig detection does). */
export function loadsBytes(bytes: Uint8Array): any {
  return loadsText(new TextDecoder("utf-8", { fatal: false }).decode(bytes), true);
}

export async function* lines(response: Response, control: AttemptControl): AsyncGenerator<string> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: false, ignoreBOM: true });
  const splitter = new LineDecoder();
  let finished = false;
  try {
    for (;;) {
      let chunk: ReadableStreamReadResult<Uint8Array>;
      try {
        chunk = await reader.read();
      } catch (error) {
        throw abortedError(control, error, "body");
      }
      if (chunk.done) break;
      control.touch();
      const text = decoder.decode(chunk.value, { stream: true });
      if (text) for (const line of splitter.decode(text)) yield line;
    }
    const tail = decoder.decode();
    if (tail) for (const line of splitter.decode(tail)) yield line;
    for (const line of splitter.flush()) yield line;
    finished = true;
  } finally {
    if (!finished) reader.cancel().catch(() => undefined);
  }
}

export async function post(binding: TransportBinding, bytes: Uint8Array, control: AttemptControl): Promise<Response> {
  try {
    const url = binding.anthropic ? binding.anthropic.url : `${binding.baseUrl}/chat/completions`;
    const headers = binding.anthropic ? binding.anthropic.headers() : requestHeaders(binding);
    const init: OraFetchInit = {
      method: "POST", headers, body: bytes as unknown as BodyInit, redirect: "manual",
      signal: control.controller.signal, oraFreshConnection: control.flavor === "async",
      ...(control.readTimeoutS !== null ? { oraReadTimeoutS: control.readTimeoutS } : {}),
    };
    return await binding.fetchImpl(url, init);
  } catch (error) {
    throw abortedError(control, error, "connect");
  }
}

// -- deadline-policy attempt (same_credential_deadline_v1, first_token_deadline_v1) --

/** `_bounded_request_in_process.read()`: one attempt as the worker runs it. */
async function workerRead(
  binding: TransportBinding, bytes: Uint8Array, streaming: boolean, window: RecoveryWindow,
  control: AttemptControl, events: any[],
): Promise<AttemptResult> {
  const timeout = Math.min(binding.timeoutSeconds, window.remaining());
  if (timeout <= 0) throw new ProviderRecoveryDeadline();
  control.readTimeout(timeout);
  const response = await post(binding, bytes, control);
  control.touch();
  if (response.status >= 400) {
    let detail = "";
    // Authentication rejection is terminal at the headers.
    if (![401, 402, 403].includes(response.status)) {
      const bodyControl = control;
      const timer = setTimeout(() => bodyControl.abort("body_timeout"), Math.min(2.0, window.remaining()) * 1000);
      try {
        detail = pyTail(decodeReplace(await readAll(response, control)), 4000);
      } catch (error) {
        if (ended(control)) throw error;
        if (!(error instanceof TransportError) && control.kind !== "body_timeout") throw error;
        detail = "error body unavailable";
      } finally {
        clearTimeout(timer);
      }
    } else {
      response.body?.cancel().catch(() => undefined);
    }
    detail = detail.split(binding.apiKey).join("[REDACTED]");
    const exc = httpStatusFailure(response.status, detail, binding.baseUrl);
    const hint = retryAfterSeconds(response.headers.get("retry-after"), binding.clock);
    exc.retryDelayS = hint !== null ? hint : response.status === 429 ? 20.0 : null;
    throw exc;
  }
  if (!streaming) {
    const value = loadsBytes(await readAll(response, control));
    if (!isMapping(value)) throw new PolicySessionError("native provider returned non-object JSON");
    return { events: [value], first: binding.clock.monotonic() };
  }
  let announced: string | null = null;
  let terminal = false;
  let first: number | null = null;
  try {
    for await (const line of lines(response, control)) {
      control.line();
      if (!line || line.startsWith(":")) continue;
      if (line.startsWith("event:")) {
        if (announced !== null || !pyStrip(line.slice(6))) {
          throw new PolicySessionError("native provider SSE event lacks its data payload");
        }
        announced = pyStrip(line.slice(6));
        continue;
      }
      if (!line.startsWith("data:")) throw new PolicySessionError("native provider returned malformed SSE framing");
      const data = pyStrip(line.slice(5));
      if (data === "[DONE]") {
        terminal = !binding.anthropic;
        break;
      }
      let parsed: any;
      try {
        parsed = loadsText(data, false);
      } catch (error) {
        throw new PolicySessionError("native provider returned malformed SSE JSON", { cause: error });
      }
      const type = isMapping(parsed) && Object.prototype.hasOwnProperty.call(parsed, "type") ? parsed.type : null;
      if (!isMapping(parsed) || (announced !== null && type !== announced)) {
        throw new PolicySessionError("native provider SSE event does not match its payload");
      }
      announced = null;
      if (first === null) first = binding.clock.monotonic();
      events.push(parsed);
      control.event();
      if (binding.anthropic && (type === "message_stop" || type === "error")) {
        terminal = true;
        break;
      }
    }
    if (announced !== null || !terminal) {
      throw new PyEOFError("native provider stream ended before protocol completion");
    }
  } catch (error) {
    if (events.length && !ended(control)) {
      throw new StreamInterrupted(events, transientTransport(error), error);
    }
    throw error;
  }
  return { events, first };
}

/** The Python parent's reconstruction of a failed worker result frame. */
function parentFailure(exc: unknown, apiKey: string): PolicySessionError {
  const rawPartial = attr(exc, "partialEvents", []);
  const partial = Array.isArray(rawPartial) ? rawPartial : [];
  const echo = echoesCredential(partial, [apiKey]);
  const evidence = failureEvidence(exc, [apiKey], partial);
  const failure = new PolicySessionError(pyMessage(exc).split(apiKey).join("[REDACTED]"));
  failure.retryable = !echo && Boolean(attr(exc, "retryable", false) || transientTransport(exc));
  failure.httpStatus = attr(exc, "httpStatus", null);
  failure.retryDelayS = attr(exc, "retryDelayS", null);
  failure.responseStarted = Boolean(attr(exc, "responseStarted", false));
  failure.upstreamStreamBreak = attr(exc, "upstreamStreamBreak", false) === true;
  const limit = attr(exc, "attemptTimeout", null);
  if (limit === "first_token" || limit === "stall") {
    failure.attemptTimeout = limit;
    failure.attemptTimeoutS = attr(exc, "attemptTimeoutS", null);
  }
  failure.partialEvents = evidence.partial_events;
  failure.credentialEcho = echo;
  failure.remoteExceptionType = pyTypeName(exc);
  failure.remoteErrorChain = evidence.error_chain;
  return failure;
}

/** A caller abort (TypeScript-only: Python cannot cancel an in-flight worker). */
export class ProviderAborted extends PolicySessionError {
  constructor(events: any[] = []) {
    super("native provider request was aborted by the caller", { pyType: "ProviderAborted" });
    this.retryable = false;
    this.partialEvents = [...events];
    this.responseStarted = events.length > 0;
  }
}

/**
 * `recovery.bounded_request`: one attempt bounded by the incident deadline,
 * or under first_token_deadline_v1 by its first-token and stall limits and
 * the run's deadline, if any.
 */
export async function boundedRequest(
  binding: TransportBinding, bytes: Uint8Array, streaming: boolean, window: RecoveryWindow,
  signal?: AbortSignal,
): Promise<AttemptResult> {
  if (window.remaining() <= 0) throw new ProviderRecoveryDeadline();
  const control = new AttemptControl(signal);
  // A timer past 2^31-1 ms fires at once; an infinite deadline needs none.
  if (Number.isFinite(window.remaining())) control.after(window.remaining(), "deadline");
  if (window.policy === FIRST_TOKEN_RECOVERY_POLICY) control.watchProgress(window.firstTokenS!, window.stallS!);
  const events: any[] = [];
  let result: AttemptResult;
  try {
    result = await workerRead(binding, bytes, streaming, window, control, events);
  } catch (exc) {
    if (control.kind === "deadline" || window.remaining() <= 0) throw new ProviderRecoveryDeadline(events);
    if (control.kind === "caller") throw new ProviderAborted(events);
    if (control.kind === "first_token" || control.kind === "stall") {
      const limit = control.kind === "first_token" ? window.firstTokenS! : window.stallS!;
      throw parentFailure(new ProviderAttemptTimeout(control.kind, limit, events), binding.apiKey);
    }
    throw parentFailure(exc, binding.apiKey);
  } finally {
    control.dispose();
  }
  if (window.remaining() <= 0) throw new ProviderRecoveryDeadline(result.events);
  window.noContentSince = null; // content arrived: a later failure opens a new window
  return result;
}

// -- legacy_v1 attempt (`_request_once`) -------------------------------------------

export async function requestOnce(
  binding: TransportBinding, bytes: Uint8Array, streaming: boolean, signal?: AbortSignal,
): Promise<AttemptResult> {
  const control = new AttemptControl(signal);
  control.flavor = "sync";
  const events: any[] = [];
  let first: number | null = null;
  try {
    control.readTimeout(binding.timeoutSeconds);
    const response = await post(binding, bytes, control);
    control.touch();
    if (response.status >= 400) {
      let detail = pyTail(decodeReplace(await readAll(response, control)), 4000);
      detail = detail.split(binding.apiKey).join("[REDACTED]");
      throw httpStatusFailure(response.status, detail, binding.baseUrl);
    }
    if (!streaming) {
      const value = loadsBytes(await readAll(response, control));
      if (!isMapping(value)) throw new PolicySessionError("native provider returned non-object JSON");
      return { events: [value], first: 0.0 };
    }
    try {
      let announced: string | null = null;
      let finishReceived = false;
      for await (const line of lines(response, control)) {
        if (!line) continue;
        if (line.startsWith(":")) continue;
        if (line.startsWith("event:")) {
          if (announced !== null) throw new PolicySessionError("native provider SSE event lacks its data payload");
          announced = pyStrip(line.slice(6));
          if (!announced) throw new PolicySessionError("native provider SSE event name is empty");
          continue;
        }
        if (!line.startsWith("data:")) throw new PolicySessionError("native provider returned malformed SSE framing");
        const data = pyStrip(line.slice(5));
        if (data === "[DONE]") {
          announced = null;
          break;
        }
        let parsed: any;
        try {
          parsed = loadsText(data, false);
        } catch (error) {
          throw new PolicySessionError("native provider returned malformed SSE JSON", { cause: error });
        }
        if (!isMapping(parsed)) throw new PolicySessionError("native provider SSE item must be an object");
        const type = Object.prototype.hasOwnProperty.call(parsed, "type") ? parsed.type : null;
        if (announced !== null && type !== announced) {
          throw new PolicySessionError("native provider SSE event name does not match its payload");
        }
        announced = null;
        if (first === null) first = binding.clock.monotonic();
        events.push(parsed);
        const choices = parsed.choices;
        if (Array.isArray(choices)) {
          finishReceived ||= choices.some((choice) => isMapping(choice) && typeof choice.finish_reason === "string");
        }
      }
      if (announced !== null) {
        throw new PolicySessionError("native provider SSE ended before an announced event payload");
      }
      if (!finishReceived) throw new PyEOFError("native provider stream ended before completion status");
    } catch (error) {
      if (events.length && control.kind !== "caller") {
        const failure = streamFailureAfterContent(error);
        failure.partialEvents = events;
        if (dumpsPy(events, { ensureAscii: false }).includes(binding.apiKey)) failure.retryable = false;
        throw failure;
      }
      throw error;
    }
  } catch (error) {
    if (control.kind === "caller") throw new ProviderAborted(events);
    throw error;
  } finally {
    control.dispose();
  }
  return { events, first };
}
