// Port of OpenAICompatibleProvider (ora/agent/providers/openai_compatible.py): the exact
// Chat Completions wire projection, response accumulation and the two
// recovery policies (`legacy_v1`, `same_credential_deadline_v1`, `first_token_deadline_v1`).

import { canonicalBytesPy, clonePy, dumpsPy, parsePy, PyFloat, utf8Strict } from "../pyjson.ts";
import {
  attemptFailureRecord, echoFailure, partialCredentialEcho, recordProviderFailure, CREDENTIAL_ECHO_MESSAGE,
} from "./attempts.ts";
import {
  PROVIDER_CONTINUATION_MESSAGE_KEY, ProviderContinuationState,
} from "./continuation.ts";
import {
  PolicySessionError, ProviderClosed, ProviderRecoveryDeadline, PyError, attr, pyFloat, pyRound6, sha256Hex, type PyNum,
} from "./py.ts";
import {
  type Clock, LEGACY_RECOVERY_POLICY, RECOVERY_POLICIES, RecoveryWindow, SYSTEM_CLOCK, transientTransport,
} from "./recovery.ts";
import { providerFetch } from "./http.ts";
import { urlHostname } from "./stream_failures.ts";
import { openaiToolNames, projectOpenaiToolNames, responseToolName } from "./tool_names.ts";
import { ProviderAborted, boundedRequest, requestHeaders, requestOnce, type TransportBinding } from "./transport.ts";
import {
  REASONING_FIELDS, TOOL_STRICTNESS_MODES, checkpointRequestOptions, normalizeUsage, projectWireBody, validateRequestOptions,
} from "./wire_options.ts";

type Json = { [key: string]: any };

export const OPENAI_PROTOCOL = "openai_chat_completions";
export const OPENAI_CONTINUATION_KIND = "openai_chat_reasoning_content_v1";
const FORBIDDEN_FIELDS = ["input", "instructions", "system", "thinking", "output_config", "input_schema"];

function isMapping(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && Object.getPrototypeOf(value) === Object.prototype;
}

function has(value: Json, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

// -- Wire projection -------------------------------------------------------------------

export interface WireOptions {
  outputTokenParameter?: string;
  toolStrictness?: string | null;
  reasoningField?: string;
  /** Profile-owned sampling / template / routing fields (openai_wire_options). */
  requestOptions?: Json | null;
}

/** `OpenAICompatibleProvider.prepare_request`: the exact Chat Completions wire body. */
export function prepareRequest(request: Json, options: WireOptions = {}): Json {
  const outputTokenParameter = options.outputTokenParameter ?? "max_tokens";
  const forbidden = FORBIDDEN_FIELDS.filter((key) => has(request, key)).sort();
  if (forbidden.length) {
    throw new PolicySessionError(
      "OpenAI canonical request contains Anthropic/Responses fields: " + forbidden.join(", "),
    );
  }
  const messages = request.messages;
  const model = request.model;
  if (!Array.isArray(messages) || typeof model !== "string" || !model) {
    throw new PolicySessionError("OpenAI request lacks messages or model");
  }
  const body = clonePy(request);
  const replayed: Json[] = [];
  for (const rawMessage of messages) {
    if (!isMapping(rawMessage)) throw new PolicySessionError("OpenAI request message must be an object");
    const message = clonePy(rawMessage);
    const rawState = has(message, PROVIDER_CONTINUATION_MESSAGE_KEY) ? message[PROVIDER_CONTINUATION_MESSAGE_KEY] : null;
    delete message[PROVIDER_CONTINUATION_MESSAGE_KEY];
    if (rawState === null) {
      if (has(message, "reasoning_content") || has(message, "reasoning")) {
        throw new PolicySessionError("unsigned OpenAI reasoning_content cannot enter replay history");
      }
      replayed.push(message);
      continue;
    }
    if (message.role !== "assistant" || !isMapping(rawState)) {
      throw new PolicySessionError("OpenAI continuation is not bound to an assistant");
    }
    const state = ProviderContinuationState.fromMapping(rawState);
    state.validateBinding({ message: rawMessage, providerProtocol: OPENAI_PROTOCOL, model });
    if (state.nativeKind !== OPENAI_CONTINUATION_KIND) {
      throw new PolicySessionError("unsupported OpenAI continuation native kind");
    }
    const payloadKeys = Object.keys(state.nativePayload);
    if (payloadKeys.length !== 1 || payloadKeys[0] !== "reasoning_content") {
      throw new PolicySessionError("OpenAI continuation payload is not canonical");
    }
    // An explicitly returned empty string is a valid signed payload.
    const reasoning = state.nativePayload.reasoning_content;
    if (typeof reasoning !== "string") throw new PolicySessionError("OpenAI continuation lacks reasoning_content");
    message.reasoning_content = reasoning;
    replayed.push(message);
  }
  body.messages = replayed;
  projectOpenaiToolNames(body, openaiToolNames(request));
  try {
    projectWireBody(body, {
      toolStrictness: options.toolStrictness ?? null, reasoningField: options.reasoningField,
      requestOptions: options.requestOptions ?? null,
    });
  } catch (error) {
    if (error instanceof PyError && error.pyType === "ValueError") throw new PolicySessionError(error.message, { cause: error });
    throw error;
  }
  if (outputTokenParameter !== "max_tokens" && outputTokenParameter !== "max_completion_tokens") {
    throw new PolicySessionError("unsupported OpenAI output token parameter");
  }
  if (outputTokenParameter === "max_completion_tokens" && has(body, "max_tokens")) {
    const value = body.max_tokens;
    delete body.max_tokens;
    body.max_completion_tokens = value;
  }
  return body;
}

// -- Response accumulation --------------------------------------------------------------

function codePointLength(text: string): number {
  let count = 0;
  for (const _ of text) count += 1;
  return count;
}

/** `_reject_truncated_tools`: reject the whole capped tool turn, keep its wire as evidence. */
export function rejectTruncatedTools(message: Json, providerStopReason: string): [Json, Json] {
  const calls: Json[] = message.tool_calls;
  const rejected = calls.map((call, index) => {
    const fn = has(call, "function") ? call.function : {};
    const fragment = has(fn, "arguments") ? fn.arguments : "";
    return {
      index, tool_call_id: has(call, "id") ? call.id : null, tool_name: has(fn, "name") ? fn.name : null,
      fragment_chars: codePointLength(fragment), fragment_sha256: sha256Hex(utf8Strict(fragment)),
      reason: "provider_output_limit",
    };
  });
  return [{ role: "assistant", content: has(message, "content") ? message.content : "" }, {
    code: "truncated_tool_call", provider_stop_reason: providerStopReason,
    rejected_tool_blocks: rejected, discarded_tool_block_count: calls.length,
    policy_outcome: "continue_same_observation", world_state_changed: false,
  }];
}

/** Why a complete call's arguments cannot be dispatched (as `dispatch.decodeArguments` decides), or null. */
function invalidArgumentsReason(call: Json): string | null {
  const fn = has(call, "function") ? call.function : {};
  if (!has(fn, "arguments")) return null;
  const raw = fn.arguments;
  if (typeof raw !== "string") return isMapping(raw) && !(raw instanceof PyFloat) ? null : "not_an_object";
  let value: unknown;
  try {
    value = parsePy(raw);
  } catch {
    return "invalid_json";
  }
  return isMapping(value) && !(value instanceof PyFloat) ? null : "not_an_object";
}

/** `_reject_invalid_tool_arguments`: a complete reply whose tool arguments do not decode to an object is a
 * recoverable no-world-change response, as on the Anthropic path (`invalid_tool_input_json`): keep its raw events,
 * discard every tool call from the canonical conversation, and let the policy retry on the same observation. Never
 * guess or repair the arguments. Replayed, such a call also made strict servers (SGLang) refuse every later
 * request ("function.arguments must be valid JSON", HTTP 400). */
export function rejectInvalidToolArguments(message: Json, providerStopReason: string | null): [Json, Json] | null {
  const calls: Json[] = message.tool_calls ?? [];
  const rejected: Json[] = [];
  for (const [index, call] of calls.entries()) {
    const reason = invalidArgumentsReason(call);
    if (reason === null) continue;
    const fn = has(call, "function") ? call.function : {};
    const fragment = typeof fn.arguments === "string" ? fn.arguments : dumpsPy(fn.arguments ?? null);
    rejected.push({
      index, tool_call_id: has(call, "id") ? call.id : null, tool_name: has(fn, "name") ? fn.name : null,
      fragment_chars: codePointLength(fragment), fragment_sha256: sha256Hex(utf8Strict(fragment)), reason,
    });
  }
  if (!rejected.length) return null;
  return [{ role: "assistant", content: has(message, "content") ? message.content : "" }, {
    code: "invalid_tool_input_json", provider_stop_reason: providerStopReason,
    rejected_tool_blocks: rejected, discarded_tool_block_count: calls.length,
    policy_outcome: "continue_same_observation", world_state_changed: false,
  }];
}

export interface Accumulated {
  message: Json;
  usage: Json | null;
  finishReason: string;
  outputRejection: Json | null;
}

/** Python `isinstance(index, int)` (bool included) as a dict key. */
function toolIndex(value: unknown, size: number): bigint {
  if (typeof value === "boolean") return value ? 1n : 0n;
  if (typeof value === "bigint") return value;
  if (typeof value === "number" && Number.isInteger(value)) return BigInt(value);
  return BigInt(size);
}

function formatIndex(index: bigint): string {
  return index < 0n ? index.toString() : index.toString().padStart(2, "0");
}

/**
 * The parse phase of `OpenAICompatibleProvider.complete()` (openai_compatible.py
 * ~326-419): merge raw Chat Completions events into one canonical assistant
 * message. Throws PolicySessionError exactly where Python does.
 */
export function accumulateChatCompletion(rawEvents: Json[], options: {
  streaming: boolean; reasoningField: string; apiKey: string | null; modelCallId: string;
  toolNamesByAlias: Map<string, string>; model: string;
}): Accumulated {
  const content: string[] = [];
  const reasoning: string[] = [];
  const calls = new Map<bigint, Json>();
  let usage: Json | null = null;
  let finishReason: string | null = null;
  let finishSeen = false;
  for (const event of rawEvents) {
    const normalized = normalizeUsage(has(event, "usage") ? event.usage : null);
    if (normalized !== null) usage = normalized;
    const choices = has(event, "choices") ? event.choices : null;
    if (!Array.isArray(choices) || !choices.length) continue;
    if (choices.length !== 1 || !isMapping(choices[0])) {
      throw new PolicySessionError("native provider must return exactly one response choice");
    }
    if (finishSeen) throw new PolicySessionError("native provider returned response content after completion");
    const choice = choices[0];
    const finish = has(choice, "finish_reason") ? choice.finish_reason : null;
    if (typeof finish === "string") {
      finishReason = finish;
      finishSeen = true;
    }
    const deltaKey = options.streaming ? "delta" : "message";
    const delta = has(choice, deltaKey) ? choice[deltaKey] : null;
    if (!isMapping(delta)) continue;
    if (typeof delta.content === "string" && has(delta, "content")) content.push(delta.content);
    if (has(delta, options.reasoningField) && typeof delta[options.reasoningField] === "string") {
      reasoning.push(delta[options.reasoningField]);
    }
    const toolCalls = has(delta, "tool_calls") ? delta.tool_calls : null;
    if (Array.isArray(toolCalls)) {
      for (const raw of toolCalls) {
        if (!isMapping(raw)) continue;
        const index = toolIndex(has(raw, "index") ? raw.index : calls.size, calls.size);
        let current = calls.get(index);
        if (current === undefined) {
          current = { id: null, type: "function", function: { name: "", arguments: "" } };
          calls.set(index, current);
        }
        if (has(raw, "id") && typeof raw.id === "string") current.id = raw.id;
        const fn = has(raw, "function") ? raw.function : null;
        if (isMapping(fn)) {
          if (has(fn, "name") && typeof fn.name === "string") current.function.name += fn.name;
          if (has(fn, "arguments") && typeof fn.arguments === "string") current.function.arguments += fn.arguments;
        }
      }
    }
  }
  if (options.apiKey && dumpsPy(rawEvents, { ensureAscii: false }).includes(options.apiKey)) {
    throw new PolicySessionError(CREDENTIAL_ECHO_MESSAGE);
  }
  const normalizedCalls: Json[] = [];
  for (const [index, call] of [...calls.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))) {
    if (!call.id) call.id = `${options.modelCallId}-tool-${formatIndex(index)}`;
    call.function.name = responseToolName(call.function.name, options.toolNamesByAlias);
    normalizedCalls.push(call);
  }
  if (finishReason === null) {
    const incomplete = new PolicySessionError(
      "native provider response has no completion status; request was not retried",
    );
    incomplete.retryable = false;
    throw incomplete;
  }
  let message: Json = { role: "assistant", content: content.join("") };
  if (reasoning.length) message.reasoning_content = reasoning.join("");
  if (normalizedCalls.length) message.tool_calls = normalizedCalls;
  let outputRejection: Json | null = null;
  const invalidArguments = finishReason === "length" ? null : rejectInvalidToolArguments(message, finishReason);
  if (finishReason === "length" && normalizedCalls.length) {
    [message, outputRejection] = rejectTruncatedTools(message, "length");
    finishReason = "invalid_tool_input";
  } else if (invalidArguments !== null) {
    [message, outputRejection] = invalidArguments;
    finishReason = "invalid_tool_input";
  } else if (reasoning.length) {
    const continuation = ProviderContinuationState.create({
      providerProtocol: OPENAI_PROTOCOL, model: options.model, nativeKind: OPENAI_CONTINUATION_KIND,
      nativePayload: { reasoning_content: reasoning.join("") }, canonicalMessage: message,
    });
    message[PROVIDER_CONTINUATION_MESSAGE_KEY] = continuation.toDict();
  }
  return { message, usage, finishReason, outputRejection };
}

/** The dict `complete()` returns, keyed exactly as in Python. */
export function completionResult(acc: Accumulated, fields: {
  rawEvents: Json[]; modelCallId: string; startedUnix: number; finishedUnix: number;
  ttft: number | null; attempts: number;
}): Json {
  return {
    message: acc.message,
    usage: acc.usage,
    finish_reason: acc.finishReason,
    ...(acc.outputRejection !== null ? { output_rejection: acc.outputRejection } : {}),
    raw_events: fields.rawEvents,
    model_call_id: fields.modelCallId,
    started_unix_s: pyFloat(fields.startedUnix),
    finished_unix_s: pyFloat(fields.finishedUnix),
    ttft_s: fields.ttft === null ? null : pyFloat(fields.ttft),
    attempts: fields.attempts,
  };
}

// -- The provider ------------------------------------------------------------------------

export interface OpenAICompatibleOptions {
  baseUrl: string;
  apiKey: string;
  timeoutSeconds?: number;
  maxRetries?: number;
  preflightProbe?: boolean;
  recoveryPolicy?: string;
  recoveryWindowS?: number | null;
  /** first_token_deadline_v1's stall limit (STALL_S); only tests shorten it. */
  stallS?: number | null;
  outputTokenParameter?: string;
  sessionAffinityId?: string | null;
  toolStrictness?: string | null;
  reasoningField?: string;
  requestOptions?: Json | null;
  fetchImpl?: typeof fetch;
  clock?: Clock;
}

export interface CompleteOptions {
  modelCallId: string;
  /** The run budget's deadline in the clock's milliseconds (default clock: performance.now()). */
  deadlineMonotonicMs?: number | null;
  signal?: AbortSignal;
  /** Receives each `ora.provider-attempt-failure.v1` record before backoff or the raise. */
  onAttemptFailure?: (record: Json) => void;
  /** Receives the exact request bytes of every attempt. */
  onRequestBytes?: (bytes: Buffer, sha256: string) => void;
}

const AFFINITY_ID = /^[A-Za-z0-9:._-]{1,200}$/;

function pyRstripSlash(text: string): string {
  return text.replace(/\/+$/, "");
}

export class OpenAICompatibleProvider {
  readonly protocol = OPENAI_PROTOCOL;
  readonly baseUrl: string;
  readonly apiKey: string;
  readonly timeoutSeconds: number;
  readonly maxRetries: number;
  readonly preflightProbe: boolean;
  readonly recoveryPolicy: string;
  readonly recoveryWindowS: number | null;
  readonly stallS: number | null;
  readonly outputTokenParameter: string;
  readonly sessionAffinityId: string | null;
  readonly toolStrictness: string | null;
  readonly reasoningField: string;
  readonly requestOptions: Json | null;
  readonly fetchImpl: typeof fetch;
  readonly clock: Clock;
  closed = false;

  /**
   * `checkpoint_request_options(model=...)`: what the rollout compactor adds to its
   * tool-free checkpoint calls (an instance method, as the compactor reads it from
   * the provider it holds). Ordinary policy requests never consult it.
   */
  checkpointRequestOptions(options: { model: string }): Json {
    return checkpointRequestOptions(options.model);
  }

  constructor(options: OpenAICompatibleOptions) {
    this.baseUrl = pyRstripSlash(options.baseUrl ?? "");
    this.apiKey = options.apiKey ?? "";
    this.timeoutSeconds = options.timeoutSeconds ?? 900.0;
    this.maxRetries = options.maxRetries ?? 2;
    this.preflightProbe = options.preflightProbe ?? false;
    this.recoveryPolicy = options.recoveryPolicy ?? LEGACY_RECOVERY_POLICY;
    this.recoveryWindowS = options.recoveryWindowS ?? null;
    this.stallS = options.stallS ?? null;
    this.outputTokenParameter = options.outputTokenParameter ?? "max_tokens";
    this.sessionAffinityId = options.sessionAffinityId ?? null;
    this.toolStrictness = options.toolStrictness ?? null;
    this.reasoningField = options.reasoningField ?? "reasoning_content";
    this.fetchImpl = options.fetchImpl ?? providerFetch(this.timeoutSeconds);
    this.clock = options.clock ?? SYSTEM_CLOCK;
    if (!this.baseUrl || !this.apiKey) {
      throw new PolicySessionError("native provider base URL and API key are required");
    }
    if (this.outputTokenParameter !== "max_tokens" && this.outputTokenParameter !== "max_completion_tokens") {
      throw new PolicySessionError("unsupported OpenAI output token parameter");
    }
    if ((this.toolStrictness !== null && !TOOL_STRICTNESS_MODES.has(this.toolStrictness))
        || !REASONING_FIELDS.has(this.reasoningField)) {
      throw new PolicySessionError("unsupported OpenAI tool_strictness or reasoning_field");
    }
    try {
      this.requestOptions = validateRequestOptions(options.requestOptions ?? null);
    } catch (error) {
      if (error instanceof PyError && error.pyType === "ValueError") {
        throw new PolicySessionError(`unsupported OpenAI request_options: ${error.message}`, { cause: error });
      }
      throw error;
    }
    if (this.sessionAffinityId !== null) {
      if (!AFFINITY_ID.test(this.sessionAffinityId)) {
        throw new PolicySessionError("invalid provider session-affinity identity");
      }
      let scheme = "";
      try {
        scheme = new URL(this.baseUrl).protocol.replace(/:$/, "").toLowerCase();
      } catch {
        scheme = (/^([A-Za-z][A-Za-z0-9+.-]*):/.exec(this.baseUrl)?.[1] ?? "").toLowerCase();
      }
      const hostname = urlHostname(this.baseUrl);
      if (scheme !== "https" && (scheme !== "http" || !["127.0.0.1", "localhost", "::1"].includes(hostname ?? ""))) {
        throw new PolicySessionError(
          "provider session affinity is admitted only for a loopback gateway or an HTTPS endpoint",
        );
      }
    }
  }

  /** The wire body with this provider's own options (Python's profile_bound attribute). */
  prepareRequest(request: Json): Json {
    return prepareRequest(request, {
      outputTokenParameter: this.outputTokenParameter, toolStrictness: this.toolStrictness,
      reasoningField: this.reasoningField, requestOptions: this.requestOptions,
    });
  }

  /** Headers ORA sets on the wire (fetch adds transport headers of its own). */
  requestHeaders(): Record<string, string> {
    return requestHeaders(this);
  }

  private binding(): TransportBinding {
    return {
      baseUrl: this.baseUrl, apiKey: this.apiKey, timeoutSeconds: this.timeoutSeconds,
      sessionAffinityId: this.sessionAffinityId, fetchImpl: this.fetchImpl, clock: this.clock,
    };
  }

  async complete(request: Json, options: CompleteOptions): Promise<Json> {
    const clock = this.clock;
    const { modelCallId, signal, onAttemptFailure, onRequestBytes } = options;
    const startedUnix = clock.time();
    const startedMono = clock.monotonic();
    const deadline = options.deadlineMonotonicMs;
    const window = RecoveryWindow.forProvider({
      recoveryPolicy: this.recoveryPolicy, recoveryWindowS: this.recoveryWindowS,
      requestDeadlineMonotonic: deadline === null || deadline === undefined ? null
        : typeof deadline === "number" ? deadline / 1000 : (deadline as any),
      preflightProbe: this.preflightProbe, maxRetries: this.maxRetries, stallS: this.stallS,
    }, clock);
    const body = this.prepareRequest(request); // the recorded wire request, exactly
    const bytes = canonicalBytesPy(body);
    const requestSha256 = sha256Hex(bytes);
    const toolNamesByAlias = openaiToolNames(request);
    const streaming = body.stream !== false;
    const record = (exc: unknown, fields: {
      attempt: number; nextIndex: number | null; started: number; startedMono: number;
      retryDelay: PyNum | 0; events: Json[]; phase: "transport" | "parse";
    }) => {
      if (!onAttemptFailure) return;
      onAttemptFailure(attemptFailureRecord(exc, {
        recoveryPolicy: this.recoveryPolicy, keys: [this.apiKey], requestSha256, modelCallId,
        attempt: fields.attempt, credentialIndex: 0, nextIndex: fields.nextIndex,
        startedUnix: fields.started, startedMonotonic: fields.startedMono,
        nowUnix: clock.time(), nowMonotonic: clock.monotonic(), retryDelay: fields.retryDelay,
        events: fields.events, phase: fields.phase,
      }));
    };
    let rawEvents: Json[] = [];
    let firstEventAt: number | null = null;
    let attempts = 0;
    let attemptStarted = 0;
    let attemptMono = 0;
    for (;;) {
      if (this.closed) {
        const closed = new ProviderClosed();
        recordProviderFailure(closed, attempts);
        throw closed;
      }
      attempts += 1;
      attemptStarted = clock.time();
      attemptMono = clock.monotonic();
      try {
        onRequestBytes?.(bytes, requestSha256);
        const result = window !== null
          ? await boundedRequest(this.binding(), bytes, streaming, window, signal)
          : await requestOnce(this.binding(), bytes, streaming, signal);
        rawEvents = result.events;
        firstEventAt = result.first;
        break;
      } catch (exc) {
        recordProviderFailure(exc, attempts);
        const echoed = partialCredentialEcho(exc, [this.apiKey]);
        const aborted = exc instanceof ProviderAborted || Boolean(signal?.aborted);
        const partialRaw = attr(exc, "partialEvents", []);
        const partial: Json[] = Array.isArray(partialRaw) ? partialRaw : [];
        if (window !== null) {
          let delay = window.transportDelay(exc, attempts, attemptMono);
          if (this.closed || echoed || aborted) delay = null;
          record(exc, {
            attempt: attempts, nextIndex: delay !== null ? 0 : null, started: attemptStarted,
            startedMono: attemptMono, retryDelay: delay ?? 0, events: partial, phase: "transport",
          });
          if (echoed) throw echoFailure(exc, attempts);
          if (delay === null) throw exc;
          try {
            await window.pause(delay, signal);
          } catch (pauseError) {
            if (pauseError instanceof ProviderRecoveryDeadline) {
              recordProviderFailure(pauseError, attempts);
              throw pauseError;
            }
            const abortedPause = new ProviderAborted();
            recordProviderFailure(abortedPause, attempts);
            throw abortedPause;
          }
          continue;
        }
        let retryable = Boolean(attr(exc, "retryable", false));
        if (!retryable) retryable = transientTransport(exc);
        const willRetry = !this.closed && !aborted && !this.preflightProbe && retryable
          && attempts < this.maxRetries + 1 && !echoed;
        const delay: PyNum | 0 = willRetry ? { value: Math.min(2 ** (attempts - 1), 4), isFloat: false } : 0;
        record(exc, {
          attempt: attempts, nextIndex: willRetry ? 0 : null, started: attemptStarted, startedMono: attemptMono,
          retryDelay: delay, events: partial, phase: "transport",
        });
        if (echoed) throw echoFailure(exc, attempts);
        if (!willRetry) throw exc;
        try {
          await clock.sleep((delay as PyNum).value, signal);
        } catch {
          const abortedSleep = new ProviderAborted();
          recordProviderFailure(abortedSleep, attempts);
          throw abortedSleep;
        }
      }
    }

    let accumulated: Accumulated;
    try {
      accumulated = accumulateChatCompletion(rawEvents, {
        streaming, reasoningField: this.reasoningField, apiKey: this.apiKey, modelCallId,
        toolNamesByAlias, model: String(body.model ?? ""),
      });
    } catch (exc) {
      if (exc instanceof PolicySessionError) {
        exc.responseStarted = rawEvents.length > 0;
        recordProviderFailure(exc, attempts);
        if (window !== null) {
          record(exc, {
            attempt: attempts, nextIndex: null, started: attemptStarted, startedMono: attemptMono,
            retryDelay: 0, events: rawEvents, phase: "parse",
          });
        }
      }
      throw exc;
    }
    if (window !== null && window.remaining() <= 0) {
      const error = new ProviderRecoveryDeadline(rawEvents);
      recordProviderFailure(error, attempts);
      throw error;
    }
    return completionResult(accumulated, {
      rawEvents, modelCallId, startedUnix, finishedUnix: clock.time(),
      ttft: firstEventAt === null ? null : pyRound6(firstEventAt - startedMono), attempts,
    });
  }

  close(): void {
    this.closed = true;
  }
}

/** `build_provider` for the openai_compatible protocol. */
export function buildOpenAICompatibleProvider(options: OpenAICompatibleOptions & { authScheme?: string | null }): OpenAICompatibleProvider {
  const recoveryPolicy = options.recoveryPolicy ?? LEGACY_RECOVERY_POLICY;
  if (recoveryPolicy !== LEGACY_RECOVERY_POLICY && !RECOVERY_POLICIES.has(recoveryPolicy)) {
    throw new PolicySessionError("unsupported native provider recovery policy");
  }
  if (options.recoveryWindowS !== null && options.recoveryWindowS !== undefined && recoveryPolicy === LEGACY_RECOVERY_POLICY) {
    throw new PolicySessionError("recovery_window_s requires the same-credential deadline recovery policy");
  }
  if (options.authScheme !== null && options.authScheme !== undefined && options.authScheme !== "bearer") {
    throw new PolicySessionError("OpenAI-compatible providers require bearer auth");
  }
  return new OpenAICompatibleProvider({ ...options, recoveryPolicy });
}
