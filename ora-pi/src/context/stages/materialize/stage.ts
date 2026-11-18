// materialize (phase materialize, always first): the canonical base of a
// request, built from its raw history (README.md).

import { afterTruncatedOutput } from "../../../core/continuations.ts";
import {
  type Message, PROVIDER_CONTINUATION_MESSAGE_KEY, excludeReasoning, materializeRecoverablePayloads, messageEventIds,
  withheldContinuation,
} from "../../projection.ts";
import { get, isMapping, sortedNumbers } from "../../pyutil.ts";
import { passRecord } from "../../request_size.ts";
import { MATERIALIZE } from "../names.ts";
import type { RequestView } from "../request.ts";
import type { Materialized, Stage } from "../stage.ts";

export interface MaterializeParams {
  /** Reasoning of earlier assistant messages: `exclude` (the one option so far). */
  reasoning: "exclude";
  /**
   * `withheld`: a reply the Provider's output limit cut off before any tool call
   * is sent without its provider continuation state (its reasoning), from the
   * first request that carries it; absent: replayed like any other.
   */
  truncated_reply_reasoning?: "withheld";
}

/** What the core tells the model after a reply the output limit cut off before any tool call. */
const TRUNCATED_CONTINUATION = afterTruncatedOutput().content;

/**
 * The assistant messages of ``messages`` that the output limit cut off before
 * any tool call: no tool call, followed by the core's truncation continuation
 * (core/session.ts `endOfReply`, core/continuations.ts `afterTruncatedOutput`),
 * and still carrying a continuation state.
 */
export function truncatedReplyIndexes(messages: Message[]): number[] {
  const found: number[] = [];
  messages.forEach((message, index) => {
    if (get(message, "role") !== "assistant") return;
    const calls = get(message, "tool_calls");
    if (Array.isArray(calls) && calls.length) return;
    if (!isMapping(get(message, PROVIDER_CONTINUATION_MESSAGE_KEY))) return;
    const next = messages[index + 1];
    if (next === undefined || get(next, "role") !== "user" || get(next, "content") !== TRUNCATED_CONTINUATION) return;
    found.push(index);
  });
  return found;
}

type NoState = Record<string, never>;

export class MaterializeStage implements Stage<MaterializeParams, NoState> {
  readonly id = MATERIALIZE;
  readonly phase = "materialize" as const;
  readonly transitions: readonly string[] = [];
  readonly requires: readonly string[] = [];
  readonly excludes: readonly string[] = [];
  readonly params: MaterializeParams;

  constructor(params: MaterializeParams) {
    this.params = params;
  }

  init(): NoState {
    return {};
  }

  stateFields(): Record<string, unknown> {
    return {};
  }

  restoreFields(): NoState {
    return {};
  }

  materialize(view: RequestView): Materialized {
    const { services } = view;
    // Every caller passes a private deep copy and materialization copies again
    // before editing, so ``raw`` is only read here.
    const [materialized, artifactRefs, firstDeliveryAdvertisedIndexes] = materializeRecoverablePayloads(view.raw, {
      catalog: services.catalog, mutationTools: services.mutationTools,
      // Which results advertise their Pointer at first delivery is the scheme's references setting.
      advertiseToolResultOverBytes: services.references.advertise_over_bytes ?? Number.MAX_SAFE_INTEGER,
      mediaCache: services.inlineMedia, domain: services.domain,
    });
    const [messages, excluded] = excludeReasoning(materialized);
    if (this.params.truncated_reply_reasoning !== "withheld") {
      return { messages, artifactRefs, firstDeliveryAdvertisedIndexes, reasoningChanges: excluded };
    }
    // The model sees its truncated reply's text and the continuation that names the cut, not its reasoning;
    // the reply stays whole in the trace (its provider_response event).
    const truncated = truncatedReplyIndexes(messages);
    for (const index of truncated) messages[index] = withheldContinuation(messages[index]);
    const reasoningChanges = sortedNumbers(new Set([...excluded, ...truncated]));
    return { messages, artifactRefs, firstDeliveryAdvertisedIndexes, reasoningChanges, truncatedReplies: truncated };
  }

  /**
   * Its two passes: payload materialization and reasoning exclusion, each
   * measured raw to materialized; with `truncated_reply_reasoning`, the
   * truncated replies it sent without their reasoning.
   */
  manifest(view: RequestView): Record<string, unknown> {
    const truncated = this.params.truncated_reply_reasoning === undefined ? {} : {
      truncated_reply_reasoning: {
        mode: this.params.truncated_reply_reasoning,
        withheld_messages: [...view.truncatedReplies],
        withheld_event_ids: messageEventIds(view.materialized, view.truncatedReplies),
      },
    };
    return {
      ...truncated,
      passes: [
        passRecord("recoverable_payload_materialization", view.raw, view.materialized, {
          artifact_refs: view.materializedRefs,
          changed_messages: view.firstDeliveryAdvertisedIndexes,
        }),
        passRecord("reasoning_exclusion", view.raw, view.materialized, {
          changed_messages: view.reasoningChanges,
          replaced_event_ids: messageEventIds(view.materialized, view.reasoningChanges),
        }),
      ],
    };
  }
}
