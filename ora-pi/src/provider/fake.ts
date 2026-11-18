// FakeProvider (port of providers/fake.py FakeProvider) and ReplayProvider, which
// returns recorded raw Chat Completions events through the real parser.

import { PyFloat, canonicalBytesPy, clonePy } from "../pyjson.ts";
import { recordProviderFailure } from "./attempts.ts";
import { OPENAI_PROTOCOL, accumulateChatCompletion, completionResult, prepareRequest, type WireOptions } from "./openai.ts";
import { PolicySessionError, sha256Hex } from "./py.ts";
import { type Clock, SYSTEM_CLOCK } from "./recovery.ts";
import { openaiToolNames } from "./tool_names.ts";

type Json = { [key: string]: any };

function isMapping(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    && Object.getPrototypeOf(value) === Object.prototype;
}

/** Return scripted assistant messages without network access. */
export class FakeProvider {
  readonly protocol = "fake";
  responses: Json[];
  requests: Json[] = [];
  index = 0;
  closed = false;
  clock: Clock;

  constructor(responses: Json[], clock: Clock = SYSTEM_CLOCK) {
    this.responses = responses;
    this.clock = clock;
  }

  async complete(request: Json, options: { modelCallId: string }): Promise<Json> {
    if (this.closed) throw new PolicySessionError("FakeProvider is closed");
    if (this.index >= this.responses.length) {
      throw new PolicySessionError("FakeProvider response script is exhausted");
    }
    this.requests.push(clonePy({ ...request }));
    const scripted = clonePy({ ...this.responses[this.index] });
    this.index += 1;
    const hasMessage = Object.prototype.hasOwnProperty.call(scripted, "message");
    const message = hasMessage ? scripted.message : scripted;
    if (!isMapping(message)) throw new PolicySessionError("FakeProvider response message must be an object");
    const usage = hasMessage ? (scripted.usage ?? null) : null;
    let finishReason = scripted.finish_reason ?? null;
    if (finishReason === null) {
      finishReason = Array.isArray(message.tool_calls) && message.tool_calls.length ? "tool_calls" : "stop";
    }
    return {
      message: { ...message },
      usage,
      finish_reason: finishReason,
      raw_events: [{ fake: true, index: this.index - 1 }],
      ...(isMapping(scripted.output_rejection) ? { output_rejection: clonePy(scripted.output_rejection) } : {}),
      model_call_id: options.modelCallId,
      started_unix_s: this.clock.time(),
      finished_unix_s: this.clock.time(),
      ttft_s: new PyFloat(0),
      attempts: 1,
    };
  }

  close(): void {
    this.closed = true;
  }
}

/**
 * Replays recorded wire responses (`native_provider_wire_response` artifacts:
 * the parsed SSE events, or `[json]` for a non-streaming reply) through the
 * same wire projection and accumulation as OpenAICompatibleProvider.
 */
export class ReplayProvider {
  readonly protocol = OPENAI_PROTOCOL;
  recordings: Json[][];
  requests: Json[] = [];
  index = 0;
  closed = false;
  options: WireOptions & { apiKey?: string | null; clock?: Clock };

  constructor(recordings: Json[][], options: WireOptions & { apiKey?: string | null; clock?: Clock } = {}) {
    this.recordings = recordings;
    this.options = options;
  }

  prepareRequest(request: Json): Json {
    return prepareRequest(request, this.options);
  }

  async complete(request: Json, options: {
    modelCallId: string; onRequestBytes?: (bytes: Buffer, sha256: string) => void;
  }): Promise<Json> {
    const clock = this.options.clock ?? SYSTEM_CLOCK;
    if (this.closed) throw new PolicySessionError("ReplayProvider is closed");
    if (this.index >= this.recordings.length) {
      throw new PolicySessionError("ReplayProvider recording is exhausted");
    }
    const startedUnix = clock.time();
    const body = this.prepareRequest(request);
    const bytes = canonicalBytesPy(body);
    options.onRequestBytes?.(bytes, sha256Hex(bytes));
    this.requests.push(body);
    const rawEvents = clonePy(this.recordings[this.index]);
    this.index += 1;
    try {
      const accumulated = accumulateChatCompletion(rawEvents, {
        streaming: body.stream !== false, reasoningField: this.options.reasoningField ?? "reasoning_content",
        apiKey: this.options.apiKey ?? null, modelCallId: options.modelCallId,
        toolNamesByAlias: openaiToolNames(request), model: String(body.model ?? ""),
      });
      return completionResult(accumulated, {
        rawEvents, modelCallId: options.modelCallId, startedUnix, finishedUnix: clock.time(), ttft: 0, attempts: 1,
      });
    } catch (exc) {
      if (exc instanceof PolicySessionError) {
        exc.responseStarted = rawEvents.length > 0;
        recordProviderFailure(exc, 1);
      }
      throw exc;
    }
  }

  close(): void {
    this.closed = true;
  }
}
