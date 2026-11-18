// Ported from `ora/agent/native/request_assembly.py` of the retired Python Native
// core (robot-harness repository): canonical native_api request assembly
// and the cache-stability evidence bound into a lossless projection manifest.
//
// The provider wire projection (`prepare_request`) is injected; without one the
// canonical execution request is the wire request.

import {
  PolicySessionError, canonicalLength, canonicalSha256, clonePy, get, hasKey, isInt, isMapping, num,
  pyEqual, setItem,
} from "./pyutil.ts";
import { PROVIDER_CONTINUATION_MESSAGE_KEY } from "./projection.ts";
import { CAMERA_TAIL_HEADER } from "./stages/camera_delivery/delivery.ts";

export type PrepareRequest = (executionRequest: Record<string, any>) => Record<string, any>;

/** ``strip_provider_continuation``: remove internal continuation state. */
export function stripProviderContinuation(message: Record<string, any>): Record<string, any> {
  const out: Record<string, any> = {};
  for (const key of Object.keys(message)) {
    if (key !== PROVIDER_CONTINUATION_MESSAGE_KEY) setItem(out, key, clonePy(message[key]));
  }
  return out;
}

function prefixRelation(previous: unknown[], current: unknown[]): Record<string, any> {
  let common = 0;
  const count = Math.min(previous.length, current.length);
  for (let index = 0; index < count; index += 1) {
    if (!pyEqual(previous[index], current[index])) break;
    common += 1;
  }
  const status = pyEqual(previous, current) ? "equal"
    : common === previous.length ? "exact_prefix" : "rewritten";
  return {
    status,
    common_prefix_units: common,
    previous_units: previous.length,
    current_units: current.length,
  };
}

/** Name the message that changed, because the request itself is discarded. */
function rewriteDiagnosis(relation: Record<string, any>, previous: unknown[], current: unknown[]): string {
  const index = Math.trunc(num(relation.common_prefix_units || 0));
  const detail = ` at message ${index} of ${relation.previous_units} (now ${relation.current_units})`;
  const digest = (units: unknown[]): string => index >= units.length
    ? "absent" : canonicalSha256(units[index]).slice(0, 16);
  return `${detail}; was ${digest(previous)}, now ${digest(current)}`;
}

function appendOnlyStatus(relation: Record<string, any>): boolean {
  return ["equal", "exact_prefix"].includes(relation.status);
}

/**
 * Context v8's rolling camera tail: the previous request without its last
 * (tail) message is a prefix of the current one, in canonical messages and in
 * wire units (the tail's units start at the last unit carrying its header).
 */
function cameraTailAppend(
  previousMessages: unknown[], currentMessages: unknown[], previousUnits: unknown[], currentUnits: unknown[],
): boolean {
  const last = previousMessages[previousMessages.length - 1];
  const content = isMapping(last) ? get(last, "content") : null;
  const header = Array.isArray(content) && isMapping(content[0]) ? get(content[0], "text") : null;
  if (typeof header !== "string" || !header.startsWith(CAMERA_TAIL_HEADER)) return false;
  if (!appendOnlyStatus(prefixRelation(previousMessages.slice(0, -1), currentMessages))) return false;
  let tailUnit = -1;
  for (let index = previousUnits.length - 1; index >= 0; index -= 1) {
    if (JSON.stringify(previousUnits[index]).includes(CAMERA_TAIL_HEADER)) {
      tailUnit = index;
      break;
    }
  }
  return tailUnit >= 0 && appendOnlyStatus(prefixRelation(previousUnits.slice(0, tailUnit), currentUnits));
}

const ALIAS_DISCLOSURE = /(?:\n\n)?<wire_tool_aliases>\n[\s\S]*?\n<\/wire_tool_aliases>/g;

/**
 * Wire units without the provider's tool-alias disclosure. The OpenAI and
 * Anthropic projections disclose wire aliases for tool names that are not
 * wire-safe in the leading system text; a malformed name the model emits
 * mid-session therefore rewrites the system prefix without any history
 * change (provider/tool_names.ts).
 */
function withoutAliasDisclosure(units: unknown[]): unknown[] {
  return JSON.parse(JSON.stringify(units), (_key, value) => (
    typeof value === "string" && value.includes("<wire_tool_aliases>") ? value.replace(ALIAS_DISCLOSURE, "") : value
  ));
}

/** The canonical history only grew and the wire differs from an append only by the alias disclosure. */
function aliasDisclosureOnly(
  previousMessages: unknown[], currentMessages: unknown[], previousUnits: unknown[], currentUnits: unknown[],
  cameraTail: boolean,
): boolean {
  const before = withoutAliasDisclosure(previousUnits);
  const after = withoutAliasDisclosure(currentUnits);
  if (pyEqual(before, previousUnits) && pyEqual(after, currentUnits)) return false;
  if (appendOnlyStatus(prefixRelation(previousMessages, currentMessages))
    && appendOnlyStatus(prefixRelation(before, after))) return true;
  return cameraTail && cameraTailAppend(previousMessages, currentMessages, before, after);
}

/** Flatten Anthropic blocks so adjacent canonical user messages compare. */
function wireMessageUnits(request: Record<string, any>): unknown[] {
  const messages = get(request, "messages");
  if (!Array.isArray(messages)) throw new PolicySessionError("native provider wire request lacks messages");
  const anthropicBlocks = messages.every((message) => isMapping(message)
    && Object.keys(message).every((key) => key === "role" || key === "content")
    && Array.isArray(get(message, "content")));
  if (!anthropicBlocks) return clonePy(messages);
  const units: unknown[] = [];
  for (const message of messages) {
    for (const block of message.content) {
      units.push({ role: get(message, "role") ?? null, content: clonePy(block) });
    }
  }
  return units;
}

/** Request fields that must remain fixed inside one cache epoch. */
function cacheStableNonMessageFields(request: Record<string, any>): Record<string, any> {
  const out: Record<string, any> = {};
  for (const key of Object.keys(request)) {
    if (key !== "messages" && key !== "tool_choice") setItem(out, key, clonePy(request[key]));
  }
  return out;
}

export interface PreparedNativeRequest {
  canonical: Record<string, any>;
  wire: Record<string, any>;
  execution: Record<string, any>;
  toolChoiceEvidence: Record<string, any>;
}

export interface NativeRequestAssemblerOptions {
  model: string;
  reasoningEffort: string;
  maxOutputTokens: number;
  stream: boolean;
  /** The provider's ``prepare_request`` (wire projection), when it has one. */
  prepareRequest?: PrepareRequest | null;
  /**
   * A context pack's compact view rewrites the delivered history on every
   * request (``context_pack.ts``): the cache-stability evidence is still bound,
   * a rewritten history is recorded as ``request_relation: context_pack_view``
   * instead of refused. Off for every plan without a pack.
   */
  contextPackView?: boolean;
}

export class NativeRequestAssembler {
  model: string;
  reasoningEffort: string;
  maxOutputTokens: number;
  stream: boolean;
  prepareRequest: PrepareRequest | null;
  readonly contextPackView: boolean;
  private previousCanonical: Record<string, any> | null = null;
  private previousWire: Record<string, any> | null = null;
  private previousEpoch: number | null = null;
  private previousProtectedPrefixSha256: string | null = null;
  private previousProviderImageWindowEpoch: number | null = null;
  /** Context v8: the previous request ended with a rolling camera tail. */
  private previousCameraTail = false;

  constructor(options: NativeRequestAssemblerOptions) {
    this.model = options.model;
    this.reasoningEffort = options.reasoningEffort;
    this.maxOutputTokens = options.maxOutputTokens;
    this.stream = options.stream;
    this.prepareRequest = options.prepareRequest ?? null;
    this.contextPackView = options.contextPackView ?? false;
  }

  prepare(projected: Record<string, any>, toolChoice: unknown = "auto"): PreparedNativeRequest {
    const messages = get(projected, "messages");
    const tools = get(projected, "tools");
    if (!Array.isArray(messages) || !Array.isArray(tools)) {
      throw new PolicySessionError("native context projection is incomplete");
    }
    const executionRequest: Record<string, any> = {
      model: this.model,
      messages: clonePy(messages),
      tools: clonePy(tools),
      tool_choice: clonePy(toolChoice),
      max_tokens: this.maxOutputTokens,
      stream: this.stream,
    };
    if (this.reasoningEffort !== "provider_default") executionRequest.reasoning_effort = this.reasoningEffort;
    if (this.stream) executionRequest.stream_options = { include_usage: true };
    const request: Record<string, any> = clonePy(executionRequest);
    request.messages = messages.map((message) => (isMapping(message) ? stripProviderContinuation(message) : clonePy(message)));
    const prepare = this.prepareRequest;
    const hasContinuation = messages.some((message) => isMapping(message)
      && hasKey(message, PROVIDER_CONTINUATION_MESSAGE_KEY));
    if (hasContinuation && prepare === null) {
      throw new PolicySessionError("provider continuation state requires an explicit wire adapter");
    }
    let wire = prepare !== null ? prepare(executionRequest) : clonePy(executionRequest);
    if (!isMapping(wire)) throw new PolicySessionError("native provider wire request must be an object");
    let wireRequest: Record<string, any> = clonePy({ ...wire });
    const toolChoiceEvidence: Record<string, any> = {
      requested: clonePy(toolChoice),
      effective: clonePy(request.tool_choice),
      mode: isMapping(toolChoice) ? "provider_named" : "automatic",
    };
    const wireThinking = get(wireRequest, "thinking");
    const wireToolChoice = get(wireRequest, "tool_choice");
    if (isMapping(toolChoice) && isMapping(wireThinking)
      && ["adaptive", "enabled"].includes(get(wireThinking, "type"))
      && isMapping(wireToolChoice) && ["tool", "any"].includes(get(wireToolChoice, "type"))) {
      // Anthropic Messages rejects named/any tool choice while adaptive
      // thinking is enabled: keep reasoning and make the exact tool a
      // Runtime-validated requirement instead.
      executionRequest.tool_choice = "auto";
      request.tool_choice = "auto";
      wire = prepare !== null ? prepare(executionRequest) : clonePy(executionRequest);
      if (!isMapping(wire)) throw new PolicySessionError("native provider wire request must be an object");
      wireRequest = clonePy({ ...wire });
      Object.assign(toolChoiceEvidence, {
        effective: "auto",
        mode: "runtime_validated_auto",
        reason: "provider_thinking_incompatible_with_named_choice",
      });
    }
    const manifest = get(projected, "manifest");
    if (isMapping(manifest) && get(manifest, "context_retention") === "action_step_lossless_v4") {
      this.bindCacheStabilityEvidence(manifest, request, wireRequest);
    }
    return {
      canonical: request,
      wire: wireRequest,
      execution: clonePy(executionRequest),
      toolChoiceEvidence,
    };
  }

  private bindCacheStabilityEvidence(
    manifest: Record<string, any>, canonical: Record<string, any>, wire: Record<string, any>,
  ): void {
    const epochValue = get(manifest, "cache_epoch");
    const imageWindow = get(manifest, "provider_image_window");
    const imageWindowEpochValue = isMapping(imageWindow) ? get(imageWindow, "epoch") : 0;
    const protectedHash = get(manifest, "protected_prefix_sha256");
    if (!isInt(epochValue) || num(epochValue) < 0) {
      throw new PolicySessionError("cache-stable projection lacks a valid cache epoch");
    }
    if (typeof protectedHash !== "string" || [...protectedHash].length !== 64) {
      throw new PolicySessionError("cache-stable projection lacks protected prefix identity");
    }
    if (!isInt(imageWindowEpochValue) || num(imageWindowEpochValue) < 0) {
      throw new PolicySessionError("cache-stable projection lacks a valid provider image-window epoch");
    }
    const epoch = num(epochValue);
    const imageWindowEpoch = num(imageWindowEpochValue);
    const canonicalHash = canonicalSha256(canonical);
    const wireHash = canonicalSha256(wire);
    const previousHash = this.previousCanonical !== null ? canonicalSha256(this.previousCanonical) : null;
    let requestRelation: string;
    let canonicalRelation: Record<string, any>;
    let wireRelation: Record<string, any>;
    if (this.previousCanonical === null || this.previousWire === null) {
      requestRelation = "initial";
      canonicalRelation = {
        status: "initial", common_prefix_units: 0, previous_units: 0,
        current_units: (get(canonical, "messages") ?? []).length,
      };
      wireRelation = {
        status: "initial", common_prefix_units: 0, previous_units: 0,
        current_units: wireMessageUnits(wire).length,
      };
    } else {
      const previousImageEpoch = this.previousProviderImageWindowEpoch;
      if (previousImageEpoch === null) {
        throw new PolicySessionError("cache-stable projection lost provider image-window state");
      }
      if (imageWindowEpoch !== previousImageEpoch && imageWindowEpoch !== previousImageEpoch + 1) {
        throw new PolicySessionError("cache_prefix_rewritten: provider image-window epoch is not contiguous");
      }
      const declaresTransition = isMapping(imageWindow) && get(imageWindow, "epoch_transition") === true;
      if ((imageWindowEpoch === previousImageEpoch + 1) !== declaresTransition) {
        throw new PolicySessionError("cache_prefix_rewritten: provider image-window transition evidence differs");
      }
      if (protectedHash !== this.previousProtectedPrefixSha256) {
        throw new PolicySessionError("cache_prefix_rewritten: protected Native context changed inside one session");
      }
      if (!pyEqual(cacheStableNonMessageFields(this.previousCanonical), cacheStableNonMessageFields(canonical))) {
        throw new PolicySessionError("cache_prefix_rewritten: canonical request fields changed inside one session");
      }
      if (!pyEqual(cacheStableNonMessageFields(this.previousWire), cacheStableNonMessageFields(wire))) {
        throw new PolicySessionError("cache_prefix_rewritten: provider wire fields changed inside one session");
      }
      const previousMessages = get(this.previousCanonical, "messages");
      const currentMessages = get(canonical, "messages");
      if (!Array.isArray(previousMessages) || !Array.isArray(currentMessages)) {
        throw new PolicySessionError("canonical Native request lacks messages");
      }
      canonicalRelation = prefixRelation(previousMessages, currentMessages);
      const previousUnits = wireMessageUnits(this.previousWire);
      const currentUnits = wireMessageUnits(wire);
      wireRelation = prefixRelation(previousUnits, currentUnits);
      if (epoch === this.previousEpoch) {
        const imageWindowAdvanced = imageWindowEpoch === previousImageEpoch + 1
          && isMapping(imageWindow) && get(imageWindow, "epoch_transition") === true;
        const appendOnly = ["equal", "exact_prefix"].includes(canonicalRelation.status)
          && ["equal", "exact_prefix"].includes(wireRelation.status);
        if (this.contextPackView && !appendOnly && !imageWindowAdvanced) {
          requestRelation = "context_pack_view";
        } else if (!appendOnly && !imageWindowAdvanced && this.previousCameraTail
          && cameraTailAppend(previousMessages, currentMessages, previousUnits, currentUnits)) {
          requestRelation = "camera_tail_append";
        } else if (!appendOnly && !imageWindowAdvanced && aliasDisclosureOnly(
          previousMessages, currentMessages, previousUnits, currentUnits, this.previousCameraTail,
        )) {
          // The history only grew; the system prefix gained a wire alias disclosure.
          requestRelation = "wire_tool_alias_disclosure";
        } else {
          if (!["equal", "exact_prefix"].includes(canonicalRelation.status) && !imageWindowAdvanced) {
            throw new PolicySessionError(
              "cache_prefix_rewritten: canonical history changed without an epoch"
              + rewriteDiagnosis(canonicalRelation, previousMessages, currentMessages),
            );
          }
          if (!["equal", "exact_prefix"].includes(wireRelation.status) && !imageWindowAdvanced) {
            throw new PolicySessionError(
              "cache_prefix_rewritten: provider wire history changed without an epoch"
              + rewriteDiagnosis(wireRelation, previousUnits, currentUnits),
            );
          }
          requestRelation = imageWindowAdvanced ? "provider_image_window_epoch" : "append_only";
        }
      } else if (this.previousEpoch !== null && epoch === this.previousEpoch + 1) {
        const transition = get(manifest, "epoch_transition");
        if (!isMapping(transition) || !pyEqual(get(transition, "to_epoch") ?? null, epoch)) {
          throw new PolicySessionError("cache_prefix_rewritten: cache epoch changed without transition evidence");
        }
        requestRelation = "projection_epoch";
      } else {
        throw new PolicySessionError("cache_prefix_rewritten: cache epoch sequence is not contiguous");
      }
    }
    Object.assign(manifest, {
      previous_request_sha256: previousHash,
      canonical_request_sha256: canonicalHash,
      provider_wire_request_sha256: wireHash,
      canonical_request_utf8_bytes: canonicalLength(canonical),
      provider_wire_request_utf8_bytes: canonicalLength(wire),
      request_relation: requestRelation,
      canonical_prefix_relation: canonicalRelation,
      provider_wire_prefix_relation: wireRelation,
    });
    this.previousCanonical = clonePy(canonical);
    this.previousWire = clonePy(wire);
    this.previousEpoch = epoch;
    this.previousProviderImageWindowEpoch = imageWindowEpoch;
    this.previousProtectedPrefixSha256 = protectedHash;
    const camera = get(manifest, "camera_delivery");
    this.previousCameraTail = isMapping(camera) && num(get(camera, "tail_images") ?? 0) > 0;
  }
}
