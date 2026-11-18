import { executionEvidence } from "../core/execution.ts";
// Ported from `ora/agent/context/projection.py` of the retired Python Native core:
// pure inventory, projection and validation primitives for the Native Context
// (the parts the lossless v4 path and the legacy rewrite_compact_v1 path reach).
// The causal-value filter and the key sets Python still owns are in
// `ora/agent/context/causal_value.py`.

import type { ArtifactStore } from "../evidence/artifacts.ts";
import type { ContextDomain } from "./domain_defaults.ts";
import * as historicalObservation from "./historical_observation.ts";
import { InlineMediaCache } from "./inline_media.ts";
import {
  ARTIFACT_READ_NAMES, CONTEXT_ARTIFACT_REF_SCHEMA, ContextArtifactCatalog, LOSSLESS_CONTEXT_RETENTION,
  RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA, RECOVERABLE_ARTIFACT_TAG, decodeRecoverableArtifactMarker,
  encodeRecoverableArtifactMarker, recoverableArtifactOpen, recoverableArtifactPointer,
  validateRecoverableArtifactEnvelope, validateRecoverableArtifactPointer,
} from "./memory/artifacts.ts";
import {
  CPythonIntSet, ContextCompactionRejected, PolicySessionError, PyValueError, canonicalBytesPy, clonePy,
  dumpsCompact, dumpsPy, dumpsSortedDefault, dumpsUnsorted, eq, get, hasKey, isInt, isMapping, num,
  pyEqual, pyLstrip, range, setItem, sha256Hex, sortedNumbers, sortedStrings, tryLoads, utf8,
  utf8Len, utf8PrefixIgnore,
} from "./pyutil.ts";
import {
  estimatedTokensFromBytes, jsonBytes, passRecord,
} from "./request_size.ts";
import * as roundStaticFacts from "./round_static_facts.ts";
import { isHeadTailPreview, originalContentText, recoverableToolMarker } from "./stages/services/references.ts";

export const PROVIDER_CONTINUATION_MESSAGE_KEY = "_provider_continuation_state";
/** Exchange aging (stages/exchange_aging): a materialized assistant message's stored arguments and continuation state. */
export const ARGUMENT_REFS_KEY = "_context_argument_refs";
export const ORIGINAL_ARGUMENTS_KEY = "_context_original_arguments";
export const CONTINUATION_REF_KEY = "_context_continuation_ref";
/** A delivered assistant message's provider continuation state that the request leaves out. */
export const WITHHELD_CONTINUATION_KEY = "_context_withheld_continuation_state";
/** The original state of a rewritten assistant message whose unsigned state the request binds to the rewrite. */
export const REBOUND_CONTINUATION_KEY = "_context_rebound_continuation_state";

/** ``message`` sent without its provider continuation state; the state stays as private metadata. */
export function withheldContinuation(message: Message): Message {
  const out: Message = { ...message };
  out[WITHHELD_CONTINUATION_KEY] = out[PROVIDER_CONTINUATION_MESSAGE_KEY];
  delete out[PROVIDER_CONTINUATION_MESSAGE_KEY];
  return out;
}

export const CONTEXT_PROFILES = new Set(["native_minimal_v1", "native_causal_context_v1"]);
export const CONTEXT_RETENTIONS = new Set(["rewrite_compact_v1", LOSSLESS_CONTEXT_RETENTION]);
export const SUMMARY_POLICIES = new Set(["disabled", "rollout_checkpoint_v1"]);
export const IMAGE_PRESSURE_ESTIMATE_TOKENS = 4096;
export const ROLLOUT_PACKET_SCHEMA = "ora.rollout-compaction-packet.v1";
export const EPISODE_CAPSULE_SCHEMA = "ora.episode-context-capsule.v1";
export const EMERGENCY_CHECKPOINT_SCHEMA = "ora.emergency-context-checkpoint.v1";
export const CHECKPOINT_FIELDS = [
  "task_progress",
  "confirmed_facts",
  "invalidated_beliefs",
  "failed_attempts",
  "current_constraints",
  "do_not_repeat",
  "re_ground_next",
];

export type Message = Record<string, any>;

export function jsonBytesOf(value: unknown): number {
  return jsonBytes(value);
}

export function estimatedTokens(value: unknown): number {
  return estimatedTokensFromBytes(jsonBytes(value));
}

/** Estimate request pressure without tokenizing inline base64 as prose; `imageTokens` per inline image. */
export function pressureEstimatedTokens(value: unknown, imageTokens: number = IMAGE_PRESSURE_ESTIMATE_TOKENS): number {
  let imageCount = 0;
  const scrub = (item: unknown): unknown => {
    if (isMapping(item)) {
      if (get(item, "type") === "image_url" && hasKey(item, "image_url")) {
        imageCount += 1;
        return { type: "image_url", image_payload: "externalized_for_estimate" };
      }
      const out: Record<string, unknown> = {};
      for (const key of Object.keys(item)) setItem(out, key, scrub(item[key]));
      return out;
    }
    if (Array.isArray(item)) return item.map(scrub);
    return item;
  };
  const scrubbed = scrub(value);
  return estimatedTokens(scrubbed) + imageCount * imageTokens;
}

export function inlineImageCount(value: unknown): number {
  if (isMapping(value)) {
    let total = get(value, "type") === "image_url" ? 1 : 0;
    for (const key of Object.keys(value)) total += inlineImageCount(value[key]);
    return total;
  }
  if (Array.isArray(value)) return value.reduce((sum: number, child) => sum + inlineImageCount(child), 0);
  return 0;
}

export function messageEventIds(messages: Message[], indexes?: Iterable<number>): string[] {
  const selected = indexes === undefined ? range(0, messages.length) : indexes;
  const found = new Set<string>();
  for (const index of selected) {
    const raw = get(messages[index], "_source_event_ids");
    if (Array.isArray(raw)) for (const item of raw) if (typeof item === "string") found.add(item);
  }
  return sortedStrings(found);
}

/** Require every packet message to carry current-rollout evidence identity. */
export function packetSourceEventIds(messages: Message[]): string[] {
  const found = new Set<string>();
  messages.forEach((message, index) => {
    const raw = get(message, "_source_event_ids");
    if (!Array.isArray(raw) || !raw.length) {
      throw new PolicySessionError(`rollout compaction packet message ${index} lacks source events`);
    }
    for (const eventId of raw) {
      if (typeof eventId !== "string" || !/^agent-event-[0-9]{6}$/.test(eventId)) {
        throw new PolicySessionError("rollout compaction packet has an invalid source event id");
      }
      found.add(eventId);
    }
  });
  return sortedStrings(found);
}

/** Strip Harness metadata while retaining opaque provider continuity. */
export function providerMessage(message: Message): Message {
  const out: Message = {};
  for (const key of Object.keys(message)) {
    if ((!key.startsWith("_") || key === PROVIDER_CONTINUATION_MESSAGE_KEY) && key !== "reasoning_content") {
      setItem(out, key, clonePy(message[key]));
    }
  }
  return out;
}

export function excludeReasoning(messages: Message[]): [Message[], number[]] {
  const output = clonePy(messages);
  const changed: number[] = [];
  output.forEach((message, index) => {
    if (hasKey(message, "reasoning_content")) {
      delete message.reasoning_content;
      changed.push(index);
    }
  });
  return [output, changed];
}

// -- Legacy rewrite_compact_v1 passes -------------------------------------------------

export function stripHistoricalImages(
  messages: Message[], currentStep: number, replacementText: string | null, domain: ContextDomain,
): [Message[], number[]] {
  const changed: number[] = [];
  const output: Message[] = [];
  messages.forEach((message, index) => {
    const item = clonePy(message);
    const metadata = get(item, "_observation");
    const step = isMapping(metadata) ? get(metadata, "step") : get(item, "_tool_image_step");
    const content = get(item, "content");
    if (get(item, "role") === "user" && Array.isArray(content) && isInt(step) && !pyEqual(step, currentStep)) {
      const nextContent: unknown[] = [];
      let removed = false;
      for (const block of content) {
        if (isMapping(block) && get(block, "type") === "image_url") {
          removed = true;
          continue;
        }
        nextContent.push(block);
      }
      if (removed) {
        nextContent.push({
          type: "text",
          text: replacementText || domain.legacyHistoricalImageText.replace("{step}", String(step)),
        });
        item.content = nextContent;
        changed.push(index);
      }
    }
    output.push(item);
  });
  return [output, changed];
}

/** Keep the action/tool transcript while removing superseded world snapshots. */
export function compactOldCausalObservations(
  messages: Message[], currentStep: number, domain: ContextDomain,
): [Message[], number[]] {
  const output = clonePy(messages);
  const changed: number[] = [];
  output.forEach((message, index) => {
    const metadata = get(message, "_causal_observation");
    if (!isMapping(metadata) || eq(get(metadata, "step"), currentStep)) return;
    message.content = [{ type: "text", text: domain.legacyCausalObservationText }];
    changed.push(index);
  });
  return [output, changed];
}

export function spillLargeToolResults(
  messages: Message[], options: {
    artifacts: ArtifactStore; thresholdBytes: number; previewBytes: number; domain: ContextDomain;
  },
): [Message[], Record<string, any>[]] {
  const output: Message[] = [];
  const refs: Record<string, any>[] = [];
  for (const message of messages) {
    const item = clonePy(message);
    if (get(item, "role") === "tool" && typeof get(item, "content") === "string") {
      const raw = utf8(item.content);
      if (raw.length > options.thresholdBytes) {
        const existing = get(item, "_artifact_ref");
        const ref = isMapping(existing) ? { ...existing } : options.artifacts.putBytes(raw, {
          kind: "native_tool_result", suffix: ".txt", mediaType: "text/plain",
        });
        const preview = utf8PrefixIgnore(item.content, options.previewBytes);
        item.content = dumpsSortedDefault({
          complete_result_recorded: true,
          preview,
          truncated: true,
          retrieve_with: options.domain.legacySpillRetrieveWith,
        });
        item._artifact_ref = ref;
        refs.push(ref);
      }
    }
    output.push(item);
  }
  return [output, refs];
}

// -- Causal privilege scans --------------------------------------------------------------

function logicalArtifactIdentity(value: Record<string, any>): boolean {
  const reference = get(value, "ref");
  const digest = get(value, "sha256");
  return Boolean(
    typeof reference === "string" && typeof digest === "string"
    && /^artifact:\/\/[A-Za-z0-9][A-Za-z0-9._-]{0,159}\/[0-9a-f]{64}$/.test(reference)
    && reference.endsWith("/" + digest)
    && !hasKey(value, "path") && !hasKey(value, "artifact_path"),
  );
}

function recoverableArtifactEnvelopeIdentity(value: Record<string, any>): boolean {
  if (!eq(get(value, "schema"), RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA)) return false;
  try {
    validateRecoverableArtifactEnvelope(value);
  } catch (error) {
    if (error instanceof PyValueError) return false;
    throw error;
  }
  return true;
}

function startsJsonContainer(value: string): boolean {
  const stripped = pyLstrip(value);
  return stripped.startsWith("{") || stripped.startsWith("[");
}

export function causalProjectionViolations(
  value: unknown, domain: ContextDomain, location = "messages",
): string[] {
  const forbidden = new Set(domain.causalForbiddenKeys);
  const violations: string[] = [];
  const visit = (item: unknown, where: string): void => {
    if (isMapping(item)) {
      const logical = logicalArtifactIdentity(item);
      const envelope = recoverableArtifactEnvelopeIdentity(item);
      const schema = get(item, "schema");
      if (logical && !(schema === undefined || schema === null || eq(schema, CONTEXT_ARTIFACT_REF_SCHEMA))) {
        violations.push(`${where}.schema`);
      }
      if (eq(schema, RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA) && !envelope) violations.push(`${where}.schema`);
      for (const key of Object.keys(item)) {
        if (forbidden.has(key) && !(
          (logical && (key === "schema" || key === "sha256")) || (envelope && key === "schema")
        )) {
          violations.push(`${where}.${key}`);
        }
        visit(item[key], `${where}.${key}`);
      }
    } else if (Array.isArray(item)) {
      item.forEach((child, index) => visit(child, `${where}[${index}]`));
    } else if (typeof item === "string" && startsJsonContainer(item)) {
      const decoded = tryLoads(item);
      if (decoded.ok && (isMapping(decoded.value) || Array.isArray(decoded.value))) {
        visit(decoded.value, `${where}<json>`);
      }
    }
  };
  visit(value, location);
  return violations;
}

/** A script's result as the causal check reads it (trace_privacy.without_script_authored_fields): without what
 * the script wrote itself, the model's computation under names it chose. The harness's fields stay checked. */
function withoutScriptAuthoredFields(message: Message, domain: ContextDomain): Message {
  if (get(message, "role") !== "tool" || !domain.scriptTools.includes(get(message, "name"))) return message;
  const content = get(message, "content");
  if (typeof content !== "string") return message;
  const decoded = tryLoads(content);
  if (!decoded.ok || !isMapping(decoded.value)) return message;
  const kept: Record<string, any> = {};
  for (const key of Object.keys(decoded.value)) {
    if (!domain.scriptAuthoredFields.includes(key)) setItem(kept, key, decoded.value[key]);
  }
  const handedBack = get(kept, "handed_back");
  if (isMapping(handedBack)) {
    const harness: Record<string, any> = {};
    for (const key of Object.keys(handedBack)) {
      if (!domain.scriptHandedBackFields.includes(key)) setItem(harness, key, handedBack[key]);
    }
    setItem(kept, "handed_back", harness);
  }
  return { ...message, content: dumpsCompact(kept) };
}

/** The causal check of a provider request's messages: what the model did not write itself. */
export function causalMessageViolations(messages: Message[], domain: ContextDomain): string[] {
  return causalProjectionViolations(
    messages.filter((message) => get(message, "role") !== "assistant")
      .map((message) => withoutScriptAuthoredFields(message, domain)),
    domain,
  );
}

export function compactorPrivateViolations(value: unknown, domain: ContextDomain, location = "packet"): string[] {
  const privateKeys = new Set(domain.compactorPrivateKeys);
  const violations: string[] = [];
  const visit = (item: unknown, where: string): void => {
    if (isMapping(item)) {
      for (const key of Object.keys(item)) {
        if (privateKeys.has(key)) violations.push(`${where}.${key}`);
        visit(item[key], `${where}.${key}`);
      }
    } else if (Array.isArray(item)) {
      item.forEach((child, index) => visit(child, `${where}[${index}]`));
    } else if (typeof item === "string" && startsJsonContainer(item)) {
      const decoded = tryLoads(item);
      if (decoded.ok && (isMapping(decoded.value) || Array.isArray(decoded.value))) {
        visit(decoded.value, `${where}<json>`);
      }
    }
  };
  visit(value, location);
  return violations;
}

// -- Recoverable payload materialization ---------------------------------------------------

function isRoundIndex(value: unknown): value is number {
  return isInt(value);
}

/** Write payloads and advertise only references worth their wire cost. */
export function materializeRecoverablePayloads(
  messages: Message[], options: {
    catalog: ContextArtifactCatalog; mutationTools: string[]; advertiseToolResultOverBytes: number;
    mediaCache?: InlineMediaCache | null; domain: ContextDomain;
  },
): [Message[], Record<string, any>[], number[]] {
  const media = options.mediaCache ?? new InlineMediaCache();
  media.bind(options.catalog);
  const output: Message[] = clonePy(messages);
  const mutationNames = new Set(options.mutationTools);
  const refs: Record<string, any>[] = [];
  const advertisedIndexes = new Set<number>();
  const groups = observationGroups(output);
  const groupByMessage = new Map<number, number>();
  const roundByMessage = new Map<number, number>();
  groups.forEach(([start, end], actionStep) => {
    const roundValue = get(output[start], "_session_round");
    for (let index = start; index < end; index += 1) {
      groupByMessage.set(index, actionStep);
      if (isRoundIndex(roundValue)) roundByMessage.set(index, num(roundValue));
    }
  });
  output.forEach((item, index) => {
    const actionStep = groupByMessage.get(index) ?? null;
    const roundIndex = roundByMessage.get(index) ?? null;
    const observationRef = historicalObservation.materialize(item, {
      catalog: media, roundIndex, actionStep, domain: options.domain,
    });
    if (observationRef !== null) refs.push(observationRef);
    const slotRef = roundStaticFacts.materialize(item, { catalog: media, roundIndex, actionStep });
    if (slotRef !== null) refs.push(slotRef);
    const content = get(item, "content");
    if (Array.isArray(content)) {
      const existingRefs = get(item, "_context_image_refs");
      if (existingRefs !== undefined && existingRefs !== null && !Array.isArray(existingRefs)) {
        throw new PolicySessionError("inline media Artifact inventory is invalid");
      }
      const imageRefs: Record<string, any>[] = [];
      const nextContent: unknown[] = [];
      let imageIndex = 0;
      for (const block of content) {
        if (!isMapping(block) || get(block, "type") !== "image_url") {
          nextContent.push(clonePy(block));
          continue;
        }
        const reference = media.admitImage(get(block, "image_url"), {
          existingRefs, imageIndex, roundIndex, actionStep,
        });
        const pointer = recoverableArtifactPointer(reference);
        const marker = encodeRecoverableArtifactMarker(pointer);
        const previous = nextContent.length ? nextContent[nextContent.length - 1] : null;
        const previousText = isMapping(previous) && get(previous, "type") === "text"
          ? get(previous, "text") : undefined;
        if (typeof previousText === "string" && previousText.startsWith(`<${RECOVERABLE_ARTIFACT_TAG}>`)) {
          let advertised: Record<string, any>;
          try {
            advertised = decodeRecoverableArtifactMarker(previousText);
          } catch (error) {
            if (error instanceof PyValueError) {
              throw new PolicySessionError("inline media recovery marker is invalid");
            }
            throw error;
          }
          if (!canonicalBytesPy(advertised).equals(canonicalBytesPy(pointer))) {
            throw new PolicySessionError("inline media recovery marker identity drift");
          }
        } else {
          nextContent.push({ type: "text", text: marker });
          advertisedIndexes.add(index);
        }
        nextContent.push(clonePy(block));
        imageRefs.push(reference);
        refs.push(reference);
        imageIndex += 1;
      }
      if (Array.isArray(existingRefs) && existingRefs.length !== imageIndex) {
        throw new PolicySessionError("inline media Artifact inventory drift");
      }
      if (imageRefs.length) {
        item.content = nextContent;
        item._context_image_refs = imageRefs;
      }
    }
    if (get(item, "role") === "tool" && typeof content === "string") {
      let mediaType = "text/plain";
      let suffix = ".txt";
      let parsed: unknown = null;
      const decoded = tryLoads(content);
      if (decoded.ok) {
        parsed = decoded.value;
        mediaType = "application/json";
        suffix = ".json";
      }
      const reference = media.putBytes(utf8(content), {
        kind: "context_tool_result", suffix, mediaType, roundIndex, actionStep,
      });
      item._context_artifact_ref = reference;
      item._context_original_content = content;
      const name = get(item, "name");
      if (
        !(typeof name === "string" && mutationNames.has(name))
        // A read already returns the original artifact identity and requested
        // bytes; advertising this response's wrapper would create another read.
        && !(typeof name === "string" && ARTIFACT_READ_NAMES.has(name))
        && !toolResultHasExactCausalOutcome(item, options.domain)
        && utf8Len(content) > options.advertiseToolResultOverBytes
      ) {
        const pointer = recoverableArtifactPointer(reference);
        if (isMapping(parsed)) {
          let advertised = get(parsed, "recoverable_artifact");
          if (advertised !== undefined && advertised !== null) {
            try {
              advertised = validateRecoverableArtifactPointer(advertised);
            } catch (error) {
              if (error instanceof PyValueError) {
                throw new PolicySessionError("tool-result recovery pointer is invalid");
              }
              throw error;
            }
            if (!canonicalBytesPy(advertised).equals(canonicalBytesPy(pointer))) {
              throw new PolicySessionError("tool-result recovery pointer identity drift");
            }
          } else {
            parsed = { ...clonePy({ ...parsed }), recoverable_artifact: pointer };
          }
          item.content = dumpsCompact(parsed);
        } else {
          item.content = dumpsCompact({
            payload: mediaType === "application/json" ? parsed : content,
            recoverable_artifact: pointer,
          });
        }
        advertisedIndexes.add(index);
      }
      refs.push(reference);
    }
  });
  const unique = new Map<string, Record<string, any>>();
  for (const reference of refs) {
    if (typeof get(reference, "ref") === "string") unique.set(reference.ref, reference);
  }
  return [
    output,
    sortedStrings(unique.keys()).map((key) => clonePy(unique.get(key)!)),
    sortedNumbers(advertisedIndexes),
  ];
}

/** The leading ActionSteps' message indexes, as the CPython set they are built in. */
export function eligibleMessageIndexes(messages: Message[], throughActionSteps: number): CPythonIntSet {
  const groups = observationGroups(messages);
  const eligible = new CPythonIntSet();
  for (const [start, end] of groups.slice(0, Math.max(0, throughActionSteps))) {
    eligible.update(range(start, end));
  }
  return eligible;
}

export function toolResultHasExactCausalOutcome(message: Message, domain: ContextDomain): boolean {
  if (message._execution && executionEvidence(message._execution, message.tool_call_id).native_actions_used > 0) return true;
  const name = get(message, "name");
  if (typeof name === "string" && domain.exactOutcomeTools.includes(name)) return true;
  const content = get(message, "content");
  if (typeof content !== "string") return true;
  const decoded = tryLoads(content);
  if (!decoded.ok) return false;
  return isMapping(decoded.value) && isMapping(get(decoded.value, "error"));
}

export type ToolSelector = "large" | "duplicate" | "remaining";

export function externalizeToolResults(
  messages: Message[], options: {
    throughActionSteps: number; mutationTools: string[]; selector: ToolSelector;
    thresholdBytes: number; previewBytes: number; excludeTools?: Iterable<string>; domain: ContextDomain;
  },
): [Message[], number[], Record<string, any>[]] {
  // Copy-on-write: unchanged messages are shared with the input.
  const output = [...messages];
  const eligible = eligibleMessageIndexes(output, options.throughActionSteps);
  const groups = observationGroups(output);
  if (groups.length && output.slice(groups[groups.length - 1][0])
    .some((message) => get(message, "_full_current_visual") === true)) {
    eligible.differenceUpdate(range(groups[groups.length - 1][0], output.length));
  }
  const mutationNames = new Set(options.mutationTools);
  const excludedNames = new Set(options.excludeTools ?? []);
  const contentCounts = new Map<string, number>();
  const firstContentIndex = new Map<string, number>();
  // CPython set iteration order decides which duplicate counts as first.
  for (const index of eligible.values()) {
    const message = output[index];
    if (get(message, "role") === "tool" && typeof get(message, "content") === "string") {
      const reference = get(message, "_context_artifact_ref");
      const digest = isMapping(reference) ? String(get(reference, "sha256") ?? "None")
        : sha256Hex(utf8(message.content));
      contentCounts.set(digest, (contentCounts.get(digest) ?? 0) + 1);
      if (!firstContentIndex.has(digest)) firstContentIndex.set(digest, index);
    }
  }
  const changed: number[] = [];
  const refs: Record<string, any>[] = [];
  for (const index of eligible.sorted()) {
    let message = output[index];
    const content = get(message, "content");
    const reference = get(message, "_context_artifact_ref");
    const name = get(message, "name");
    if (
      get(message, "role") !== "tool" || typeof content !== "string" || !isMapping(reference)
      || get(message, "_lossless_externalized") === true
      || (typeof name === "string" && (mutationNames.has(name) || excludedNames.has(name)))
      || toolResultHasExactCausalOutcome(message, options.domain)
    ) {
      continue;
    }
    const original = hasKey(message, "_context_original_content")
      ? originalContentText(message) : content;
    const digest = String(get(reference, "sha256") ?? "None");
    const selected = (options.selector === "large" && utf8Len(original) > options.thresholdBytes)
      || (options.selector === "duplicate" && (contentCounts.get(digest) ?? 0) > 1
        && firstContentIndex.get(digest) !== index)
      || options.selector === "remaining";
    if (!selected) continue;
    const marker = recoverableToolMarker(message, reference, options.previewBytes, options.domain);
    if (options.selector !== "large" && utf8Len(marker) >= utf8Len(original)) continue;
    message = { ...message };
    output[index] = message;
    message.content = marker;
    message._lossless_externalized = true;
    changed.push(index);
    refs.push(clonePy({ ...reference }));
  }
  return [output, changed, refs];
}

export function externalizeHistoricalMedia(
  messages: Message[], throughActionSteps: number, toolImageEventIds: Iterable<string> = [],
): [Message[], number[], Record<string, any>[]] {
  const output = [...messages];
  const eligible = new Set(eligibleMessageIndexes(output, throughActionSteps).values());
  const selectedToolImages = new Set(toolImageEventIds);
  output.forEach((message, index) => {
    const ids = get(message, "_source_event_ids");
    if (Array.isArray(ids) && ids.some((id) => typeof id === "string" && selectedToolImages.has(id))) {
      eligible.add(index);
    }
  });
  const changed: number[] = [];
  const refs: Record<string, any>[] = [];
  for (const index of sortedNumbers(eligible)) {
    let message = output[index];
    const content = get(message, "content");
    const imageRefs = get(message, "_context_image_refs");
    if (!Array.isArray(content) || !Array.isArray(imageRefs)) continue;
    let referenceIndex = 0;
    const nextContent: unknown[] = [];
    let removed = false;
    for (const block of content) {
      if (isMapping(block) && get(block, "type") === "image_url") {
        if (referenceIndex >= imageRefs.length || !isMapping(imageRefs[referenceIndex])) {
          throw new PolicySessionError("historical image lacks its recoverable Artifact reference");
        }
        const reference = clonePy({ ...imageRefs[referenceIndex] });
        referenceIndex += 1;
        refs.push(reference);
        const previous = nextContent.length ? nextContent[nextContent.length - 1] : null;
        const previousText = isMapping(previous) && get(previous, "type") === "text"
          ? get(previous, "text") : undefined;
        if (typeof previousText !== "string" || !previousText.startsWith(`<${RECOVERABLE_ARTIFACT_TAG}>`)) {
          throw new PolicySessionError("historical image lacks its recovery pointer");
        }
        let pointer: Record<string, any>;
        try {
          pointer = decodeRecoverableArtifactMarker(previousText);
        } catch (error) {
          if (error instanceof PyValueError) {
            throw new PolicySessionError("historical image recovery pointer is invalid");
          }
          throw error;
        }
        if (!eq(pointer.ref, get(reference, "ref"))) {
          throw new PolicySessionError("historical image recovery pointer identity drift");
        }
        removed = true;
      } else {
        nextContent.push(clonePy(block));
      }
    }
    if (referenceIndex !== imageRefs.length) {
      throw new PolicySessionError("historical image Artifact inventory drift");
    }
    if (removed) {
      message = { ...message };
      output[index] = message;
      message.content = nextContent;
      message._lossless_media_externalized = true;
      changed.push(index);
    }
  }
  return [output, changed, refs];
}

export function inlineMediaBytesFromRefs(messages: Message[]): number {
  let total = 0;
  for (const message of messages) {
    const content = get(message, "content");
    const references = get(message, "_context_image_refs");
    if (!Array.isArray(content) || !Array.isArray(references)) continue;
    const inlineCount = content.filter((block) => isMapping(block) && get(block, "type") === "image_url").length;
    if (inlineCount !== references.length) continue;
    for (const reference of references) {
      if (isMapping(reference)) total += Math.trunc(num(hasKey(reference, "bytes") ? reference.bytes : 0));
    }
  }
  return total;
}

/** The identity of one stored payload, as the invariant view compares it. */
function storedIdentity(reference: Record<string, any>): Record<string, any> {
  return { recoverable_sha256: get(reference, "sha256") ?? null, bytes: get(reference, "bytes") ?? null };
}

/**
 * Exchange aging: an assistant message with stored arguments or a withheld
 * continuation state compares by the stored identities, after checking that
 * what it delivers is the original or the exact reference to it.
 */
function assistantInvariant(message: Message, visible: Message): void {
  const withheld = get(message, WITHHELD_CONTINUATION_KEY);
  const rebound = get(message, REBOUND_CONTINUATION_KEY);
  const continuationRef = get(message, CONTINUATION_REF_KEY);
  if (withheld !== undefined) {
    if (hasKey(message, PROVIDER_CONTINUATION_MESSAGE_KEY) || rebound !== undefined) {
      throw new PolicySessionError("assistant continuation state is both delivered and withheld");
    }
    visible[PROVIDER_CONTINUATION_MESSAGE_KEY] = clonePy(withheld);
  }
  // A state bound to the rewrite compares as the original it was rebound from.
  if (rebound !== undefined) visible[PROVIDER_CONTINUATION_MESSAGE_KEY] = clonePy(rebound);
  if (isMapping(continuationRef) && isMapping(get(visible, PROVIDER_CONTINUATION_MESSAGE_KEY))) {
    if (sha256Hex(canonicalBytesPy(visible[PROVIDER_CONTINUATION_MESSAGE_KEY])) !== get(continuationRef, "sha256")) {
      throw new PolicySessionError("provider continuation Artifact identity drift");
    }
    visible[PROVIDER_CONTINUATION_MESSAGE_KEY] = storedIdentity(continuationRef);
  }
  const refs = get(message, ARGUMENT_REFS_KEY);
  const originals = get(message, ORIGINAL_ARGUMENTS_KEY);
  if (!isMapping(refs) || !isMapping(originals) || !Array.isArray(get(visible, "tool_calls"))) return;
  visible.tool_calls = visible.tool_calls.map((call: Record<string, any>) => {
    const id = get(call, "id");
    if (typeof id !== "string" || !hasKey(refs, id)) return call;
    const reference = refs[id];
    const original = originals[id];
    if (typeof original !== "string" || sha256Hex(utf8(original)) !== get(reference, "sha256")
      || !pyEqual(utf8Len(original), get(reference, "bytes") ?? null)) {
      throw new PolicySessionError("tool-call arguments Artifact identity drift");
    }
    const delivered = get(get(call, "function") ?? {}, "arguments");
    if (delivered !== original) {
      const decoded = typeof delivered === "string" ? tryLoads(delivered) : { ok: false as const };
      const marker = decoded.ok && isMapping(decoded.value) ? decoded.value : null;
      const pointer = marker !== null ? get(marker, "recoverable_artifact") : null;
      if (marker === null || get(marker, "arguments_externalized") !== true || !isMapping(pointer)
        || !eq(get(pointer, "ref"), get(reference, "ref"))
        || !pyEqual(get(marker, "arguments_bytes") ?? null, utf8Len(original))
        || typeof get(marker, "preview") !== "string" || !isHeadTailPreview(original, String(marker.preview))) {
        throw new PolicySessionError("tool-call arguments differ from their recoverable source");
      }
    }
    return { ...call, function: { ...call.function, arguments: storedIdentity(reference) } };
  });
}

/** Normalize recoverable payloads by identity while keeping policy prose exact. */
export function losslessInvariantView(messages: Message[], domain: ContextDomain): Message[] {
  const normalized: Message[] = [];
  for (const message of messages) {
    const visible = providerMessage(message);
    const toolRef = get(message, "_context_artifact_ref");
    const slotContent = roundStaticFacts.invariantContent(message);
    if (get(message, "role") === "assistant" && (hasKey(message, ARGUMENT_REFS_KEY)
      || hasKey(message, CONTINUATION_REF_KEY) || hasKey(message, WITHHELD_CONTINUATION_KEY))) {
      assistantInvariant(message, visible);
    } else if (get(message, "role") === "tool" && isMapping(toolRef)) {
      visible.content = {
        recoverable_sha256: get(toolRef, "sha256") ?? null,
        media_type: get(toolRef, "media_type") ?? null,
        bytes: get(toolRef, "bytes") ?? null,
      };
    } else if (slotContent !== null) {
      visible.content = slotContent;
    } else if (Array.isArray(get(message, "content"))) {
      const content: unknown[] = [];
      const observationBlock = historicalObservation.invariantBlock(message, domain);
      message.content.forEach((original: unknown, blockIndex: number) => {
        let block = original;
        if (blockIndex === 0 && observationBlock !== null) block = observationBlock;
        if (isMapping(block) && get(block, "type") === "text"
          && String(get(block, "text") ?? "").startsWith("[recoverable historical image externalized]")) {
          return;
        }
        if (isMapping(block) && get(block, "type") === "image_url") return;
        content.push(clonePy(block));
      });
      const imageRefs = get(message, "_context_image_refs");
      if (Array.isArray(imageRefs)) {
        for (const reference of imageRefs) {
          if (!isMapping(reference)) continue;
          content.push({
            type: "recoverable_image_identity",
            sha256: get(reference, "sha256") ?? null,
            media_type: get(reference, "media_type") ?? null,
            bytes: get(reference, "bytes") ?? null,
          });
        }
      }
      visible.content = content;
    }
    normalized.push(visible);
  }
  return normalized;
}

export function observationGroups(messages: Message[]): [number, number][] {
  const starts: number[] = [];
  messages.forEach((message, index) => {
    if (get(message, "role") === "user" && isMapping(get(message, "_observation"))) starts.push(index);
  });
  return starts.map((start, offset) => [start, offset + 1 < starts.length ? starts[offset + 1] : messages.length]);
}

/** Validate observation groups as complete or open physical ActionSteps. */
export function actionStepInventory(
  messages: Message[], options: { mutationTools: string[]; requireSessionRound?: boolean; domain: ContextDomain },
): Record<string, any>[] {
  const groups = observationGroups(messages);
  const mutationNames = new Set(options.mutationTools);
  const acceptedStatuses = options.domain.acceptedMutationStatuses;
  const inventory: Record<string, any>[] = [];
  groups.forEach(([start, end], index) => {
    const observation = get(messages[start], "_observation");
    if (!isMapping(observation)) throw new PolicySessionError("ActionStep lacks its input actual observation");
    let sessionRound = get(messages[start], "_session_round");
    if (options.requireSessionRound && (!isInt(sessionRound) || num(sessionRound) < 1)) {
      throw new PolicySessionError("ActionStep lacks its positive Session round");
    }
    if (!isInt(sessionRound)) sessionRound = 1;
    const callsById = new Map<string, [string, unknown]>();
    const namelessCalls = new Set<string>();
    const toolResults = new Map<string, [string | null, unknown]>();
    const executions = new Map<string, any>();
    for (const message of messages.slice(start, end)) {
      if (get(message, "role") === "assistant") {
        const calls = get(message, "tool_calls");
        if (Array.isArray(calls)) {
          for (const call of calls) {
            const fn = isMapping(call) ? get(call, "function") : undefined;
            let name = isMapping(fn) ? get(fn, "name") : undefined;
            const callId = isMapping(call) ? get(call, "id") : undefined;
            if (typeof callId !== "string" || !callId) {
              throw new PolicySessionError("ActionStep tool call lacks a stable provider id");
            }
            if (callsById.has(callId)) {
              throw new PolicySessionError(`ActionStep ${index} reuses tool-call id ${callId}`);
            }
            if (typeof name !== "string" || !name) {
              namelessCalls.add(callId);
              name = "invalid_tool_call";
            }
            callsById.set(callId, [name as string, call]);
          }
        }
      }
      if (get(message, "role") === "tool" && typeof get(message, "tool_call_id") === "string") {
        const callId = String(message.tool_call_id);
        if (toolResults.has(callId)) {
          throw new PolicySessionError(`ActionStep ${index} duplicates tool result ${callId}`);
        }
        let content = get(message, "content");
        if (typeof content === "string") {
          const decoded = tryLoads(content);
          if (decoded.ok) content = decoded.value;
        }
        if (message._execution) executions.set(callId, executionEvidence(message._execution, callId));
        const resultName = get(message, "name");
        toolResults.set(callId, [typeof resultName === "string" ? resultName : null, content]);
      }
    }
    const callIds = new Set(callsById.keys());
    const resultIds = new Set(toolResults.keys());
    if (callIds.size !== resultIds.size || [...callIds].some((id) => !resultIds.has(id))) {
      throw new PolicySessionError(`ActionStep ${index} does not pair every tool call and result`);
    }
    for (const [callId, [name]] of callsById) {
      const [resultName, content] = toolResults.get(callId)!;
      if (namelessCalls.has(callId) && !(
        resultName === "invalid_tool_call" && isMapping(content)
        && get(content, "ok") !== true && get(content, "success") !== true
        && isMapping(get(content, "error"))
        && eq(get(content.error, "code"), "invalid_tool_call")
        && get(content.error, "retryable") === true
      )) {
        throw new PolicySessionError("ActionStep nameless tool call lacks its invalid_tool_call refusal");
      }
      if (resultName !== null && resultName !== name) {
        throw new PolicySessionError(`ActionStep ${index} tool result name differs from its call`);
      }
    }
    const mutationCalls: [string, string][] = [];
    for (const [callId, [name]] of callsById) if (mutationNames.has(name)) mutationCalls.push([name, callId]);
    const acceptedMutations = mutationCalls.filter(([, callId]) => {
      if (executions.has(callId)) return false;
      const result = toolResults.get(callId)![1];
      return isMapping(result) && !isMapping(get(result, "error"))
        && acceptedStatuses.some((status) => eq(get(result, "status"), status));
    });
    for (const [callId, execution] of executions) {
      if (execution.native_actions_used > 0) acceptedMutations.push([callsById.get(callId)![0], callId]);
    }
    const execution = [...executions.values()].filter((item) => item.native_actions_used > 0).at(-1);
    if (acceptedMutations.length > 1) {
      throw new PolicySessionError(`ActionStep ${index} contains more than one robot mutation`);
    }
    const complete = index < groups.length - 1;
    let postActual: unknown = null;
    if (complete) {
      const nextObservation = get(messages[groups[index + 1][0]], "_observation");
      if (!isMapping(nextObservation)) {
        throw new PolicySessionError(`ActionStep ${index} lacks its post-action actual observation`);
      }
      postActual = get(nextObservation, "step") ?? null;
      // After a no-action outcome the Supervisor presents the same observation again.
      const presentedAgain = !acceptedMutations.length
        && eq(postActual, get(observation, "step"))
        && typeof get(observation, "observation_id") === "string"
        && eq(get(nextObservation, "observation_id"), observation.observation_id);
      if (execution && (!eq(postActual, execution.final_observation.step)
        || !eq(get(nextObservation, "observation_id"), execution.final_observation.observation_id))) {
        throw new PolicySessionError(`ActionStep ${index} differs from its final published observation`);
      }
      if (!execution && isInt(get(observation, "step")) && !presentedAgain
        && (!isInt(postActual) || num(postActual) !== Math.trunc(num(observation.step)) + 1)) {
        throw new PolicySessionError(`ActionStep ${index} post-action observation is not the next step`);
      }
      if (acceptedMutations.length && !execution) {
        const receipt = get(nextObservation, "receipt");
        if (!isMapping(receipt) || typeof get(receipt, "event_id") !== "string" || !receipt.event_id
          || typeof get(receipt, "action_id") !== "string" || !receipt.action_id) {
          throw new PolicySessionError(
            `ActionStep ${index} mutation lacks an identity-bound environment receipt`,
          );
        }
      }
    }
    const sourceIds = messageEventIds(messages, range(start, end));
    const receiptOf = (key: string): unknown => {
      if (!(complete && acceptedMutations.length)) return null;
      const next = get(messages[groups[index + 1][0]], "_observation");
      const receipt = isMapping(next) ? get(next, "receipt") : undefined;
      return isMapping(receipt) ? get(receipt, key) ?? null : null;
    };
    inventory.push({
      action_step_index: index,
      session_round: sessionRound,
      observation_step: get(observation, "step") ?? null,
      post_observation_step: postActual,
      complete,
      message_start: start,
      message_end: end,
      message_count: end - start,
      inline_images: inlineImageCount(messages.slice(start, end)),
      mutation: acceptedMutations.length ? acceptedMutations[0][0] : null,
      mutation_tool_call_id: acceptedMutations.length ? acceptedMutations[0][1] : null,
      ...(execution ? { execution, actions: execution.actions,
        receipt_event_ids: execution.actions.map((a: any) => a.receipt_event_id) } : {}),
      receipt_event_id: execution?.actions.at(-1)?.receipt_event_id ?? receiptOf("event_id"),
      receipt_action_id: execution?.actions.at(-1)?.action_id ?? receiptOf("action_id"),
      source_event_ids: sourceIds,
    });
  });
  return inventory;
}

/** Flatten canonical history into one tool-free multimodal packet message. */
export function packetMessageBlocks(
  section: string, messages: Message[], domain: ContextDomain,
): Record<string, any>[] {
  const blocks: Record<string, any>[] = [{ type: "text", text: `<${section}>` }];
  for (const message of messages) {
    const visible = compactorHistoryMessage(message, domain);
    const content = hasKey(visible, "content") ? visible.content : "";
    delete visible.content;
    const descriptor = clonePy(visible);
    blocks.push({ type: "text", text: dumpsCompact(descriptor) });
    if (typeof content === "string") {
      if (content) blocks.push({ type: "text", text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isMapping(block)) continue;
        if (get(block, "type") === "image_url") blocks.push(clonePy({ ...block }));
        else if (get(block, "type") === "text" && typeof get(block, "text") === "string") {
          blocks.push({ type: "text", text: String(block.text) });
        }
      }
    } else {
      blocks.push({ type: "text", text: dumpsSortedDefault(content ?? null) });
    }
  }
  blocks.push({ type: "text", text: `</${section}>` });
  return blocks;
}

/** Return policy-visible history minus fields reserved from the Compactor (ora/agent/context/lossless/compactor_privacy.py). */
export function compactorHistoryMessage(message: Message, domain: ContextDomain): Message {
  const visible = providerMessage(message);
  delete visible[PROVIDER_CONTINUATION_MESSAGE_KEY];
  // The rollout checkpoint must not carry an official outcome: withhold the
  // checker's Boolean while preserving tool adjacency and evidence identity --
  // the checker's own result, and the same Boolean a finish_episode task-check
  // refusal repeats in its constraints. Only these known places are masked;
  // the field (or any reserved key) anywhere else stays a privacy violation.
  const name = get(visible, "name");
  if (get(visible, "role") !== "tool" || (name !== domain.taskCheckTool && name !== domain.finishTool)) {
    return visible;
  }
  const content = get(visible, "content");
  if (typeof content !== "string") return visible;
  const decoded = tryLoads(content);
  if (!decoded.ok || !isMapping(decoded.value)) return visible;
  const field = domain.taskCheckVerdictField;
  if (name === domain.taskCheckTool) {
    if (hasKey(decoded.value, field)) visible.content = dumpsCompact(domain.taskCheckWithheld);
    return visible;
  }
  const error = get(decoded.value, "error");
  const constraints = isMapping(error) ? get(error, "constraints") : undefined;
  if (isMapping(error) && isMapping(constraints) && domain.finishVerdictRefusalCodes.includes(get(error, "code"))
    && hasKey(constraints, field)) {
    const withheld: Record<string, any> = {};
    for (const key of Object.keys(constraints)) if (key !== field) withheld[key] = constraints[key];
    Object.assign(withheld, domain.finishVerdictWithheld);
    visible.content = dumpsCompact({ ...decoded.value, error: { ...error, constraints: withheld } });
  }
  return visible;
}

/** Deterministic replay/audit fragments rejected from a checkpoint. */
export function checkpointForbiddenFragments(value: Record<string, any>, domain: ContextDomain): string[] {
  const serialized = dumpsUnsorted(value);
  const found = new Set<string>();
  for (const match of serialized.matchAll(domain.checkpointForbidden)) found.add(match[0]);
  for (const match of serialized.matchAll(domain.checkpointOutcomeAssertion)) found.add(match[0]);
  return sortedStrings(found);
}

export function validateCheckpoint(
  content: string, domain: ContextDomain, maxEstimatedTokens: number | null = null,
): Record<string, any> {
  const decoded = tryLoads(content);
  if (!decoded.ok) throw new ContextCompactionRejected("context compactor returned invalid JSON");
  let value = decoded.value;
  if (!isMapping(value) || Object.keys(value).length !== CHECKPOINT_FIELDS.length
    || !CHECKPOINT_FIELDS.every((field) => hasKey(value, field))) {
    throw new ContextCompactionRejected("context compactor checkpoint fields do not match the v1 schema");
  }
  const ordered: Record<string, any> = {};
  for (const field of CHECKPOINT_FIELDS) ordered[field] = value[field];
  value = ordered;
  for (const field of CHECKPOINT_FIELDS) {
    const entries = value[field];
    if (!Array.isArray(entries)) {
      throw new ContextCompactionRejected(`context checkpoint ${field} must be an array`);
    }
    if (field === "failed_attempts") {
      for (const entry of entries) {
        const keys = isMapping(entry) ? Object.keys(entry) : [];
        if (!isMapping(entry) || keys.length !== 3
          || !["failure", "evidence", "correction"].every((key) => hasKey(entry, key))
          || !keys.every((key) => typeof entry[key] === "string")) {
          throw new ContextCompactionRejected("context checkpoint failed_attempts entry is invalid");
        }
      }
    } else if (!entries.every((entry: unknown) => typeof entry === "string")) {
      throw new ContextCompactionRejected(`context checkpoint ${field} entry is invalid`);
    }
  }
  if (!CHECKPOINT_FIELDS.some((field) => value[field].length)) {
    throw new ContextCompactionRejected("context compactor returned an empty checkpoint");
  }
  const forbidden = checkpointForbiddenFragments(value, domain);
  if (forbidden.length) {
    throw new ContextCompactionRejected(
      "context checkpoint contains forbidden replay/audit data: " + forbidden.join(", "),
    );
  }
  if (maxEstimatedTokens !== null && estimatedTokens(value) > maxEstimatedTokens) {
    throw new ContextCompactionRejected("context checkpoint exceeds its Harness token budget");
  }
  return clonePy(value);
}

/** ``transcriptPointer: false`` (Context v8, no artifact reader) leaves the exact-transcript Pointer out. */
export function semanticCapsuleSlot(capsule: Record<string, any>, transcriptPointer = true): Message {
  const summary: Record<string, any> = {};
  for (const field of CHECKPOINT_FIELDS) summary[field] = clonePy(capsule.summary[field]);
  const envelope: Record<string, any> = {
    schema: EPISODE_CAPSULE_SCHEMA,
    session_round: capsule.session_round,
    coverage: clonePy(capsule.coverage),
    official_verdict: clonePy(capsule.official_verdict),
    ...(transcriptPointer ? { exact_transcript: clonePy(capsule.exact_transcript) } : {}),
    summary,
  };
  return {
    role: "user",
    content: "<episode_context_capsule>\n" + dumpsCompact(envelope) + "\n</episode_context_capsule>",
    _protected_context: true,
    _context_source: `episode_context_capsule_${pyStrValue(capsule.session_round)}`,
    _episode_context_capsule: true,
  };
}

/** Python ``str(value)`` for an int/str/None identifier. */
function pyStrValue(value: unknown): string {
  if (value === null || value === undefined) return "None";
  if (typeof value === "string") return value;
  if (value === true) return "True";
  if (value === false) return "False";
  return dumpsPy(value);
}

/** Project exact static prefix, semantic slots, then the untouched tail. */
/** The active emergency checkpoint's slot message; ``pointers`` names its exact transcript. */
export function emergencyCheckpointSlot(
  checkpoint: Record<string, any> | null, transcript: Record<string, any> | null, pointers: boolean,
): Message {
  if (checkpoint === null || transcript === null) {
    throw new PolicySessionError("emergency projection lacks its checkpoint");
  }
  const summary: Record<string, any> = {};
  for (const field of CHECKPOINT_FIELDS) summary[field] = checkpoint[field];
  const exactText = dumpsPy({
    schema: EMERGENCY_CHECKPOINT_SCHEMA,
    summary,
    ...(pointers ? { exact_transcript: { ...clonePy({ ...transcript }), open: recoverableArtifactOpen(transcript) } } : {}),
  }, { ensureAscii: false, sortKeys: false, separators: [",", ":"] });
  return {
    role: "user",
    content: "<emergency_context_checkpoint>\n" + exactText + "\n</emergency_context_checkpoint>",
    _protected_context: true,
    _context_source: "emergency_context_checkpoint",
    _emergency_context_checkpoint: true,
  };
}

export function tieredSemanticProjection(
  messages: Message[], options: {
    episodeCheckpointedActionSteps: number; episodeCapsules: Record<string, any>[];
    emergencySummarizedActionSteps: number; emergencyCheckpoint: Record<string, any> | null;
    emergencyTranscriptRef: Record<string, any> | null;
    /** Context v8 shows no artifact Pointer: the slots leave the exact transcript out. */
    transcriptPointers?: boolean;
  },
): Message[] {
  const pointers = options.transcriptPointers ?? true;
  const groups = observationGroups(messages);
  if (!groups.length) return clonePy(messages.map((message) => ({ ...message })));
  const absorbed = Math.max(options.episodeCheckpointedActionSteps, options.emergencySummarizedActionSteps);
  if (absorbed >= groups.length) {
    throw new PolicySessionError("semantic projection must retain an open ActionStep");
  }
  const prefix = messages.slice(0, groups[0][0])
    .filter((message) => get(message, "_protected_context") === true)
    .map((message) => clonePy({ ...message }));
  const slots: Message[] = [];
  const activeEmergency = options.emergencySummarizedActionSteps > options.episodeCheckpointedActionSteps;
  if (activeEmergency) {
    slots.push(emergencyCheckpointSlot(options.emergencyCheckpoint, options.emergencyTranscriptRef, pointers));
  }
  for (const capsule of options.episodeCapsules) {
    const coverage = get(capsule, "coverage");
    const through = isMapping(coverage) ? get(coverage, "through_action_step_index") : undefined;
    if (activeEmergency && (isInt(through) || typeof through === "boolean")
      && num(through) < options.emergencySummarizedActionSteps) {
      continue;
    }
    slots.push(semanticCapsuleSlot(capsule, pointers));
  }
  const tailStart = groups[absorbed][0];
  // native_slots_v4: a summarized span never takes the current round's slot with it.
  const carried = roundStaticFacts.currentRoundSlots(
    messages.slice(groups[0][0], tailStart),
    get(messages[groups[groups.length - 1][0]], "_session_round") ?? null,
  );
  return [...prefix, ...slots, ...carried, ...clonePy(messages.slice(tailStart))];
}
