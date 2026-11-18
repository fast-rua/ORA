// What an aged exchange shows the model (exchange_aging's render; README.md).
//
// An exchange is one assistant message and every message that follows it up to
// the next assistant message (services/segmenter.ts). Every pass here is a pure
// function of the materialized history and two frontiers exchange_aging keeps,
// which move only at an aging or the hard backstop, each a new projection epoch.
// Nothing here may depend on what a later message changes (such as which
// observations are currently the latest): an already-delivered message changes
// only when a frontier moves.
// * `agedExchanges`: every exchange before it is aged. A tool call's arguments
//   longer than `argument_bytes` become a reference (reasoning_replay then
//   sends that assistant message without its continuation state, which was
//   bound to the original), a tool result longer than `result_bytes` becomes a
//   head-and-tail preview and a reference, every image a tool returned (in its
//   own message or with an observation) is left as the recoverable-artifact
//   marker materialization put before it, and so is every camera image of an
//   observation under camera_delivery `exchange_window_v1` (otherwise camera
//   images keep their camera delivery);
// * `agedObservationGroups`: an aged observation's measured body becomes a
//   reference if its observation group is before this frontier (the groups
//   outside the camera tail when the last aging or backstop ran);
// * `keptExchange`: an exchange before the frontier left as it is, the anchor
//   of an exact tail in exchanges (the latest robot action's exchange) when
//   the last aging or backstop ran (`keep_anchor`).
// Originals are Context Artifacts (`storeArguments` and the shared
// materialization), readable through `artifact_read`; the lossless invariant
// view (projection.ts) restores them by identity.

import type { ContextDomain } from "../../domain_defaults.ts";
import * as historicalObservation from "../../historical_observation.ts";
import type { InlineMediaCache } from "../../inline_media.ts";
import { recoverableArtifactPointer } from "../../memory/artifacts.ts";
import {
  ARGUMENT_REFS_KEY, type Message, ORIGINAL_ARGUMENTS_KEY, observationGroups,
} from "../../projection.ts";
import {
  PolicySessionError, PyValueError, clonePy, dumpsCompact, eq, get, hasKey, isInt, isMapping, num, tryLoads, utf8,
  utf8Len,
} from "../../pyutil.ts";
import { cameraImageIndexes } from "../camera_delivery/delivery.ts";
import {
  type HeadTailPreview, RECOVERABLE_ARTIFACT_TAG, decodeRecoverableArtifactMarker, headTailPreview,
  recoverableToolMarker, structuredToolMarker,
} from "../services/references.ts";
import { exchangesOf } from "../services/segmenter.ts";

/** The end (exclusive) of the span the first ``count`` exchanges cover. */
export function spanEnd(from: number, starts: number[], count: number, length: number): number {
  if (count <= 0) return from;
  return count < starts.length ? starts[count] : length;
}

/** ActionStep and Session round of every message, for the references' location. */
export function locations(messages: Message[]): Map<number, [number | null, number]> {
  const located = new Map<number, [number | null, number]>();
  observationGroups(messages).forEach(([start, end], actionStep) => {
    const round = get(messages[start], "_session_round");
    for (let index = start; index < end; index += 1) located.set(index, [isInt(round) ? num(round) : null, actionStep]);
  });
  return located;
}

/**
 * Store every tool call's arguments longer than ``argumentBytes`` as Context
 * Artifacts, recorded on the (private) materialized assistant message. Edits
 * ``messages`` in place and returns the references.
 */
export function storeArguments(
  messages: Message[], options: { media: InlineMediaCache; argumentBytes: number },
): Record<string, any>[] {
  const refs: Record<string, any>[] = [];
  const located = locations(messages);
  messages.forEach((message, index) => {
    if (get(message, "role") !== "assistant") return;
    const [roundIndex, actionStep] = located.get(index) ?? [null, null];
    const calls = get(message, "tool_calls");
    if (!Array.isArray(calls)) return;
    const argumentRefs: Record<string, any> = {};
    const originals: Record<string, string> = {};
    for (const call of calls) {
      const id = isMapping(call) ? get(call, "id") : undefined;
      const fn = isMapping(call) ? get(call, "function") : undefined;
      const text = isMapping(fn) ? get(fn, "arguments") : undefined;
      if (typeof id !== "string" || typeof text !== "string" || utf8Len(text) <= options.argumentBytes) continue;
      const json = tryLoads(text).ok;
      const reference = options.media.putBytes(utf8(text), {
        kind: "context_tool_call_arguments", suffix: json ? ".json" : ".txt",
        mediaType: json ? "application/json" : "text/plain", roundIndex, actionStep,
      });
      argumentRefs[id] = reference;
      originals[id] = text;
      refs.push(reference);
    }
    if (Object.keys(argumentRefs).length) {
      message[ARGUMENT_REFS_KEY] = argumentRefs;
      message[ORIGINAL_ARGUMENTS_KEY] = originals;
    }
  });
  return refs;
}

/** What an aged tool call's arguments become: a JSON object naming the stored original, its head and its tail. */
export function argumentMarker(original: string, reference: Record<string, any>, preview: HeadTailPreview): string {
  return dumpsCompact({
    arguments_externalized: true,
    arguments_bytes: utf8Len(original),
    preview: headTailPreview(original, preview).preview,
    recoverable_artifact: recoverableArtifactPointer(reference),
  });
}

export interface ExternalizedImages {
  message: Message;
  toolRefs: Record<string, any>[];
  cameraRefs: Record<string, any>[];
}

/**
 * ``message`` with every image removed but its camera images (the content
 * indexes ``cameras``) unless ``ageCameras``: the marker materialization put
 * before each removed image stays as its reference (`open: observe`). Null
 * when it removes none.
 */
export function externalizedImages(
  message: Message, cameras: ReadonlySet<number>, ageCameras: boolean,
): ExternalizedImages | null {
  const content = get(message, "content");
  const refs = get(message, "_context_image_refs");
  if (!Array.isArray(content) || !Array.isArray(refs)) return null;
  const kept: unknown[] = [];
  const toolRefs: Record<string, any>[] = [];
  const cameraRefs: Record<string, any>[] = [];
  let ordinal = 0;
  for (const [index, block] of content.entries()) {
    if (!(isMapping(block) && get(block, "type") === "image_url") || (!ageCameras && cameras.has(index))) {
      if (isMapping(block) && get(block, "type") === "image_url") ordinal += 1;
      kept.push(block);
      continue;
    }
    const reference = refs[ordinal];
    ordinal += 1;
    const previous = kept.length ? kept[kept.length - 1] : null;
    const text = isMapping(previous) && get(previous, "type") === "text" ? get(previous, "text") : undefined;
    if (!isMapping(reference) || typeof text !== "string" || !text.startsWith(`<${RECOVERABLE_ARTIFACT_TAG}>`)) {
      throw new PolicySessionError("tool image lacks its recovery pointer");
    }
    let pointer: Record<string, any>;
    try {
      pointer = decodeRecoverableArtifactMarker(text);
    } catch (error) {
      if (error instanceof PyValueError) throw new PolicySessionError("tool image recovery pointer is invalid");
      throw error;
    }
    if (!eq(pointer.ref, get(reference, "ref"))) throw new PolicySessionError("tool image recovery pointer identity drift");
    (cameras.has(index) ? cameraRefs : toolRefs).push(clonePy({ ...reference }));
  }
  if (ordinal !== refs.length) throw new PolicySessionError("tool image Artifact inventory drift");
  if (!toolRefs.length && !cameraRefs.length) return null;
  const out: Message = { ...message, content: kept, _lossless_media_externalized: true };
  if (toolRefs.length) out._lossless_exchange_images = toolRefs.length;
  if (cameraRefs.length) out._lossless_exchange_camera_images = cameraRefs.length;
  return { message: out, toolRefs, cameraRefs };
}

export interface AgingRenderOptions {
  agedExchanges: number;
  /** An exchange before the frontier that stays as it is (the anchor kept at the last aging); null: none. */
  keptExchange?: number | null;
  /** How an aged result keeps its content: its head and tail, or its structured fields. */
  resultPreview?: "head_and_tail" | "structured";
  agedObservationGroups: number;
  argumentBytes: number;
  resultBytes: number;
  preview: HeadTailPreview;
  /** camera_delivery `exchange_window_v1`: an aged exchange's observations lose their camera images too. */
  cameraImages: boolean;
  mutationTools: string[];
  domain: ContextDomain;
}

/** One render: the history and, per pass, the changed messages and the references they show. */
export interface AgingRender {
  messages: Message[];
  arguments: number[];
  results: number[];
  toolImages: number[];
  cameraImages: number[];
  observations: number[];
  argumentRefs: Record<string, any>[];
  resultRefs: Record<string, any>[];
  toolImageRefs: Record<string, any>[];
  cameraImageRefs: Record<string, any>[];
  observationRefs: Record<string, any>[];
}

/** Age the exchanges before the frontiers; copy-on-write, ``messages`` is never edited. */
export function renderAging(messages: Message[], options: AgingRenderOptions): AgingRender {
  const output = [...messages];
  const { from, starts } = exchangesOf(messages);
  const agedEnd = spanEnd(from, starts, options.agedExchanges, messages.length);
  const groupStarts = new Map(observationGroups(messages).map(([start], index) => [start, index]));
  const mutations = new Set(options.mutationTools);
  const render: AgingRender = {
    messages: output, arguments: [], results: [], toolImages: [], cameraImages: [], observations: [],
    argumentRefs: [], resultRefs: [], toolImageRefs: [], cameraImageRefs: [], observationRefs: [],
  };
  const observationIndexes: number[] = [];
  const kept = options.keptExchange ?? null;
  const keptSpan = kept !== null && kept < starts.length ? exchangesOf(messages).spans[kept] : null;
  for (let index = from; index < agedEnd; index += 1) {
    if (keptSpan !== null && keptSpan[0] <= index && index < keptSpan[1]) continue;
    const message = output[index];
    const role = get(message, "role");
    if (role === "assistant") {
      const refs = get(message, ARGUMENT_REFS_KEY);
      const originals = get(message, ORIGINAL_ARGUMENTS_KEY);
      if (isMapping(refs) && isMapping(originals)) {
        output[index] = {
          ...message,
          tool_calls: (message.tool_calls as Message[]).map((call) => {
            const id = get(call, "id");
            if (typeof id !== "string" || !hasKey(refs, id)) return call;
            render.argumentRefs.push(clonePy(refs[id]));
            return { ...call, function: { ...call.function, arguments: argumentMarker(originals[id], refs[id], options.preview) } };
          }),
          _lossless_arguments_externalized: true,
        };
        render.arguments.push(index);
      }
      continue;
    }
    const content = get(message, "content");
    const reference = get(message, "_context_artifact_ref");
    const name = get(message, "name");
    if (role === "tool" && typeof content === "string" && isMapping(reference)
      && get(message, "_lossless_externalized") !== true
      && !(typeof name === "string" && mutations.has(name))) {
      const original = typeof get(message, "_context_original_content") === "string"
        ? String(message._context_original_content) : content;
      if (utf8Len(original) > options.resultBytes) {
        const marker = options.resultPreview === "structured"
          ? structuredToolMarker(message, reference, options.domain)
          : recoverableToolMarker(message, reference, options.preview, options.domain);
        if (utf8Len(marker) < utf8Len(original)) {
          output[index] = { ...message, content: marker, _lossless_externalized: true };
          render.results.push(index);
          render.resultRefs.push(clonePy({ ...reference }));
        }
      }
      continue;
    }
    // Images a tool returned arrive in their own user message, or with the observation an action published
    // (a skill's receipt images); an observation's camera images keep the camera delivery, unless they age
    // with their exchange (exchange_window_v1).
    if (role === "user") {
      const observation = isMapping(get(message, "_observation"));
      const stripped = externalizedImages(
        message, observation ? cameraImageIndexes(message) : new Set(), options.cameraImages,
      );
      if (stripped !== null) {
        output[index] = stripped.message;
        if (stripped.toolRefs.length) {
          render.toolImages.push(index);
          render.toolImageRefs.push(...stripped.toolRefs);
        }
        if (stripped.cameraRefs.length) {
          render.cameraImages.push(index);
          render.cameraImageRefs.push(...stripped.cameraRefs);
        }
      }
      if (!observation) continue;
    }
    const group = groupStarts.get(index);
    if (group !== undefined && group < options.agedObservationGroups) observationIndexes.push(index);
  }
  if (observationIndexes.length) {
    const [observed, changed, refs] = historicalObservation.externalize(output, observationIndexes, options.domain);
    for (const index of changed) output[index] = observed[index];
    render.observations.push(...changed);
    render.observationRefs.push(...refs);
  }
  return render;
}
