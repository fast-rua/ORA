// What the hard backstop shows when it ages by itself (`aging: own`, a scheme
// without exchange stages: management v6; README.md). One frontier, `aged`:
// in every exchange before it (one assistant message and every message after
// it up to the next, services/segmenter.ts), the open ActionStep's and the
// exact ActionStep tail's included,
// * an assistant message is sent without its provider continuation state (its
//   reasoning), which stays on the message as private metadata;
// * a read-only tool result becomes the head preview and Pointer of the
//   scheme's references (as L4 renders one), when that is shorter; results of
//   mutation tools and exact causal outcomes stay;
// * the images a tool returned in their own message are left as the
//   recoverable-artifact markers materialization put before them
//   (`observe(ref=...)`). Observation messages keep their camera images and
//   bodies, so the latest observation is always whole.
// A pure function of the history and the frontier, which moves only when the
// backstop runs (a new projection epoch); the lossless invariant view
// (projection.ts) restores everything it leaves out by identity.

import type { ContextDomain } from "../../domain_defaults.ts";
import {
  type Message, PROVIDER_CONTINUATION_MESSAGE_KEY, toolResultHasExactCausalOutcome, withheldContinuation,
} from "../../projection.ts";
import { clonePy, get, hasKey, isMapping, utf8Len } from "../../pyutil.ts";
import { externalizedImages, spanEnd } from "../exchange_aging/render.ts";
import { originalContentText, recoverableToolMarker } from "../services/references.ts";
import { exchangesOf } from "../services/segmenter.ts";

export interface OwnBackstopRender {
  messages: Message[];
  results: number[];
  toolImages: number[];
  reasoning: number[];
  resultRefs: Record<string, any>[];
  toolImageRefs: Record<string, any>[];
}

/** Age every exchange before ``agedExchanges``; copy-on-write, ``messages`` is never edited. */
export function renderOwnBackstop(messages: Message[], options: {
  agedExchanges: number; previewBytes: number; mutationTools: string[]; domain: ContextDomain;
}): OwnBackstopRender {
  const output = [...messages];
  const render: OwnBackstopRender = {
    messages: output, results: [], toolImages: [], reasoning: [], resultRefs: [], toolImageRefs: [],
  };
  const { from, starts } = exchangesOf(messages);
  const end = spanEnd(from, starts, options.agedExchanges, messages.length);
  const mutations = new Set(options.mutationTools);
  for (let index = from; index < end; index += 1) {
    const message = output[index];
    const role = get(message, "role");
    if (role === "assistant") {
      if (isMapping(get(message, PROVIDER_CONTINUATION_MESSAGE_KEY))) {
        output[index] = withheldContinuation(message);
        render.reasoning.push(index);
      }
      continue;
    }
    if (role === "tool") {
      const content = get(message, "content");
      const reference = get(message, "_context_artifact_ref");
      const name = get(message, "name");
      if (typeof content !== "string" || !isMapping(reference) || get(message, "_lossless_externalized") === true
        || (typeof name === "string" && mutations.has(name))
        || toolResultHasExactCausalOutcome(message, options.domain)) continue;
      const original = hasKey(message, "_context_original_content") ? originalContentText(message) : content;
      const marker = recoverableToolMarker(message, reference, options.previewBytes, options.domain);
      if (utf8Len(marker) >= utf8Len(original)) continue;
      output[index] = { ...message, content: marker, _lossless_externalized: true };
      render.results.push(index);
      render.resultRefs.push(clonePy({ ...reference }));
      continue;
    }
    // Tool-returned images in their own user message; an observation (camera images, a receipt's views) stays.
    if (role !== "user" || isMapping(get(message, "_observation")) || get(message, "_lossless_media_externalized") === true) {
      continue;
    }
    const stripped = externalizedImages(message, new Set(), false);
    if (stripped === null) continue;
    output[index] = stripped.message;
    render.toolImages.push(index);
    render.toolImageRefs.push(...stripped.toolRefs);
  }
  return render;
}
