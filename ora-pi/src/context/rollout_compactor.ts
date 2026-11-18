// `ora/agent/context/lossless/compactor.py`: one fresh, tool-free provider call
// per semantic Context transition (L5 capsule / L6 checkpoint), traced and
// stored as evidence.
//
// The provider call is injected: `complete(request, {modelCallId})` returns
// the Python `ModelProvider.complete` mapping (`message`, `usage`,
// `finish_reason`, `attempts`, `raw_events`, `started_unix_s`,
// `finished_unix_s`, `response_id`, optional `wire_request`).

import type { ArtifactStore } from "../evidence/artifacts.ts";
import { COMPACTOR_SYSTEM_PROMPT } from "./domain_defaults.ts";
import type { SummaryProvider, SummaryResult } from "./native.ts";
import {
  PolicySessionError, canonicalBytesPy, canonicalSha256, clonePy, errorText, get, hasKey, isMapping,
  isNumber, num, pyEqual, pyFloat, pyInt, pyRepr, pyRound, pyStrip,
} from "./pyutil.ts";

export interface CompactorProvider {
  complete(request: Record<string, any>, options: { modelCallId: string }): Promise<Record<string, any>>;
  /** The provider wire projection; absent means the canonical request is the wire. */
  prepareRequest?(request: Record<string, any>): Record<string, any>;
  /** ``checkpoint_request_options(model=...)``: only reasoning_effort/response_format. */
  checkpointRequestOptions?(options: { model: string }): Record<string, any>;
  protocol?: string;
}

/** ``event_sink(event_type, **fields)`` returning the appended event (with ``event_id``). */
export type CompactorEventSink = (
  eventType: string,
  fields: {
    visibility: string; model_call_id: string; causal_parent_id?: string;
    artifact_ref?: Record<string, any>; payload: Record<string, any>;
  },
) => Record<string, any> | Promise<Record<string, any>>;

function withThousands(value: number): string {
  return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/** Bind the model's concise-output target below the hard admission limit. */
export function compactorSystemPrompt(
  checkpointTargetEstimatedTokens: number, basePrompt: string = COMPACTOR_SYSTEM_PROMPT,
): string {
  if (checkpointTargetEstimatedTokens <= 0) {
    throw new PolicySessionError("context checkpoint token budget must be positive");
  }
  // Dense JSON is estimated at one token per three UTF-8 bytes; ten percent
  // headroom keeps wording variance from becoming the length control.
  const targetBytes = pyInt(checkpointTargetEstimatedTokens * 3 * 0.9);
  return basePrompt + `

The complete minified JSON MUST be at most ${withThousands(targetBytes)} UTF-8 bytes. This is
a strict concise-output target below the ${withThousands(checkpointTargetEstimatedTokens)}-token
Harness target. When a previous checkpoint is supplied, rewrite and
consolidate it; do not append another history. Merge duplicates, delete stale or
resolved detail, and prioritize facts that can change the next decisions.`;
}

function assistantText(message: Record<string, any>): string {
  const content = get(message, "content");
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.filter((block) => isMapping(block) && get(block, "type") === "text")
      .map((block) => {
        const text = get(block, "text");
        return text === undefined ? "" : typeof text === "string" ? text : pyRepr(text);
      }).join("");
  }
  throw new PolicySessionError("context compactor response has no text content");
}

export interface RolloutCheckpointCompactorOptions {
  provider: CompactorProvider;
  model: string;
  reasoningEffort: string;
  stream: boolean;
  artifacts: ArtifactStore;
  eventSink: CompactorEventSink;
  checkpointTargetEstimatedTokens?: number;
  maxOutputTokens?: number;
  systemPrompt?: string;
  /** A compactor bound to its own profile (policy.context_compactor): {provider_profile, model, endpoint}. */
  providerBinding?: Record<string, any> | null;
  /** Wall clock in seconds (``time.time``), injectable for replay. */
  clock?: () => number;
}

/** Execute and trace one fresh, tool-free provider call per transition. */
/**
 * The compactor's output ceiling: its signed thinking and the JSON checkpoint
 * share it (the checkpoint target is the summarize stages' parameter).
 */
export const COMPACTOR_MAX_OUTPUT_TOKENS = 8192;

export class RolloutCheckpointCompactor implements SummaryProvider {
  provider: CompactorProvider;
  model: string;
  reasoningEffort: string;
  stream: boolean;
  artifacts: ArtifactStore;
  eventSink: CompactorEventSink;
  checkpointTargetEstimatedTokens: number;
  // Signed thinking and the JSON checkpoint share the generation budget.
  maxOutputTokens: number;
  systemPrompt: string;
  providerBinding: Record<string, any> | null;
  callSequence = 0;
  calls: Record<string, any>[] = [];
  private clock: () => number;

  constructor(options: RolloutCheckpointCompactorOptions) {
    this.provider = options.provider;
    this.model = options.model;
    this.reasoningEffort = options.reasoningEffort;
    this.stream = options.stream;
    this.artifacts = options.artifacts;
    this.eventSink = options.eventSink;
    this.checkpointTargetEstimatedTokens = options.checkpointTargetEstimatedTokens ?? 1200;
    this.maxOutputTokens = options.maxOutputTokens ?? COMPACTOR_MAX_OUTPUT_TOKENS;
    this.systemPrompt = options.systemPrompt ?? COMPACTOR_SYSTEM_PROMPT;
    this.providerBinding = options.providerBinding ?? null;
    this.clock = options.clock ?? (() => Date.now() / 1000);
  }

  /** A compactor event's payload; a bound compactor names its Provider binding. */
  private payload(fields: Record<string, any>): Record<string, any> {
    return { call_kind: "context_compactor", ...fields,
      ...(this.providerBinding ? { provider_binding: clonePy(this.providerBinding) } : {}) };
  }

  async summarize(args: { messages: Record<string, any>[]; sourceRefs: Record<string, any>[] }): Promise<SummaryResult> {
    const callId = `context-compactor-${String(this.callSequence).padStart(6, "0")}`;
    this.callSequence += 1;
    const request: Record<string, any> = {
      model: this.model,
      messages: [
        { role: "system", content: compactorSystemPrompt(this.checkpointTargetEstimatedTokens, this.systemPrompt) },
        ...clonePy(args.messages.map((message) => ({ ...message }))),
      ],
      max_tokens: this.maxOutputTokens,
      stream: this.stream,
    };
    if (this.reasoningEffort !== "provider_default") request.reasoning_effort = this.reasoningEffort;
    const options = this.provider.checkpointRequestOptions
      ? this.provider.checkpointRequestOptions({ model: this.model }) : {};
    if (!isMapping(options) || Object.keys(options).some((key) => key !== "reasoning_effort" && key !== "response_format")) {
      throw new PolicySessionError("unsupported checkpoint request options");
    }
    Object.assign(request, clonePy({ ...options }));
    if (this.stream) request.stream_options = { include_usage: true };
    const canonicalRef = this.artifacts.putBytes(canonicalBytesPy(request), {
      kind: "rollout_compactor_canonical_request", suffix: ".json", mediaType: "application/json",
    });
    const wire = this.provider.prepareRequest ? this.provider.prepareRequest(request) : clonePy(request);
    if (!isMapping(wire)) throw new PolicySessionError("context compactor wire request is invalid");
    const wireRef = this.artifacts.putBytes(canonicalBytesPy(wire), {
      kind: "rollout_compactor_wire_request", suffix: ".json", mediaType: "application/json",
    });
    const claim = await this.eventSink("context_compactor_attempt", {
      visibility: "derived",
      model_call_id: callId,
      artifact_ref: wireRef,
      payload: this.payload({
        model: this.model,
        provider_protocol: String(this.provider.protocol ?? "canonical_unknown"),
        canonical_request_ref: canonicalRef,
        source_set_sha256: canonicalSha256([...args.sourceRefs]),
      }),
    });
    const started = this.clock();
    let responseEventId: string | null = null;
    try {
      const response = await this.provider.complete(request, { modelCallId: callId });
      const returnedWire = get(response, "wire_request");
      if (returnedWire !== undefined && returnedWire !== null
        && !canonicalBytesPy(returnedWire).equals(canonicalBytesPy(wire))) {
        throw new PolicySessionError("context compactor sent a wire request different from evidence");
      }
      const attempts = hasKey(response, "attempts") ? response.attempts : 1;
      const rawRef = this.artifacts.putJson(hasKey(response, "raw_events") ? response.raw_events : [], {
        kind: "rollout_compactor_wire_response",
      });
      const finished = get(response, "finished_unix_s");
      const responseStarted = get(response, "started_unix_s");
      const latency = (isNumber(finished) || typeof finished === "boolean")
        && (isNumber(responseStarted) || typeof responseStarted === "boolean")
        ? num(finished) - num(responseStarted) : this.clock() - started;
      const latencyS = pyFloat(pyRound(latency, 6));
      const event = await this.eventSink("context_compactor_response", {
        causal_parent_id: String(claim.event_id),
        visibility: "derived",
        model_call_id: callId,
        artifact_ref: rawRef,
        payload: this.payload({
          usage: get(response, "usage") ?? null,
          finish_reason: get(response, "finish_reason") ?? null,
          attempts,
          latency_s: latencyS,
          response_id: get(response, "response_id") ?? null,
          validation_pending: true,
        }),
      });
      responseEventId = String(event.event_id);
      if (!pyEqual(attempts, 1)) {
        throw new PolicySessionError("context compactor provider performed more than one physical attempt");
      }
      if (get(response, "finish_reason") !== "stop") {
        throw new PolicySessionError(
          `context compactor response did not complete with stop: ${pyRepr(get(response, "finish_reason") ?? null)}`,
        );
      }
      const message = get(response, "message");
      if (!isMapping(message) || get(message, "role") !== "assistant") {
        throw new PolicySessionError("context compactor response lacks an assistant message");
      }
      const content = pyStrip(assistantText(message));
      if (!content) throw new PolicySessionError("context compactor returned empty content");
      const record = {
        call_id: callId,
        attempt_event_id: claim.event_id,
        response_event_id: event.event_id,
        canonical_request_ref: canonicalRef,
        wire_request_ref: wireRef,
        raw_response_ref: rawRef,
        usage: clonePy(get(response, "usage") ?? null),
        latency_s: latencyS,
        response_id: get(response, "response_id") ?? null,
      };
      this.calls.push(record);
      return {
        content,
        sourceRefs: clonePy([...args.sourceRefs]),
        sourceSetSha256: canonicalSha256([...args.sourceRefs]),
        metadata: record,
      };
    } catch (error) {
      await this.eventSink("context_compactor_failed", {
        causal_parent_id: responseEventId ?? String(claim.event_id),
        visibility: "derived",
        model_call_id: callId,
        payload: this.payload({ error: errorText(error) }),
      });
      if (error instanceof PolicySessionError) throw error;
      throw new PolicySessionError(`context_compactor_failed: ${errorText(error)}`);
    }
  }
}
