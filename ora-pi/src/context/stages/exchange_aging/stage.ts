// exchange_aging (phase age): while the Provider-reported input is at or above
// a watermark, one rewrite ages every exchange older than the exact exchange
// tail into recoverable references, once a batch of exchanges has left that
// tail since the last aging (README.md; what an aged exchange shows is
// render.ts).

import { type Message, messageEventIds } from "../../projection.ts";
import { PolicySessionError, clonePy, num, pyInt } from "../../pyutil.ts";
import { activeSizer, measuredPassRecord } from "../../request_size.ts";
import { CAMERA_EXCHANGE_WINDOW } from "../../management.ts";
import type { CameraDeliveryParams } from "../camera_delivery/stage.ts";
import {
  BACKSTOP_AGED, CAMERA_DELIVERY, EXCHANGE_AGING, EXCHANGES_AGED, MATERIALIZE, REASONING_REPLAY,
} from "../names.ts";
import type { ReasoningNote, ReasoningReplayParams } from "../reasoning_replay/stage.ts";
import type { RequestView } from "../request.ts";
import type { HeadTailPreview } from "../services/references.ts";
import type { Admission, ApplyContext, NoticeDetail, Rendered, Stage, Transition } from "../stage.ts";
import { type AgingRender, renderAging, storeArguments } from "./render.ts";
import {
  EXCHANGE_CAMERA_PASS, type ExchangeAgingState, type ReasoningReplayState, exchangeChanges, exchangeCount,
  exchangeRewrite, observationGroupsOutsideCameraTail,
} from "./rewrite.ts";

export const EXCHANGE_AGING_RECORD_SCHEMA = "ora.context-exchange-aging.v1";

export interface ExchangeAgingParams {
  /** The absolute watermark in same-epoch Provider input tokens, capped by the ratio. */
  trigger_tokens: number;
  /** The watermark as a share of the hard input limit. */
  trigger_ratio: number;
  /** Latest exchanges an aging keeps exact, and the batch: an aging waits until this many have left that tail. */
  exact_tail_exchanges: number;
  /** An aged tool call's arguments longer than this become a reference; null: arguments never age. */
  argument_bytes: number | null;
  /** An aged tool result longer than this becomes a preview and a reference. */
  result_bytes: number;
  /** The preview an aged argument or result keeps: its head and its tail. */
  preview: HeadTailPreview;
  /** What an aged result keeps: its head and tail (absent) or its structured fields (no output text). */
  result_preview?: "head_and_tail" | "structured";
  /** With an exact tail in exchanges: leave the anchor exchange (the latest robot action's) as it is. */
  keep_anchor?: boolean;
}

interface AgingAdmission extends Admission {
  target: number;
  watermark: number;
}

/** What one render recorded: its input, the render and whether it ages camera images. */
interface AgingNote {
  input: Message[];
  render: AgingRender;
  cameraImages: boolean;
}

export class ExchangeAgingStage implements Stage<ExchangeAgingParams, ExchangeAgingState, AgingAdmission> {
  readonly id = EXCHANGE_AGING;
  readonly phase = "age" as const;
  readonly transitions: readonly string[] = ["exchange_aging"];
  readonly requires: readonly string[] = [MATERIALIZE, REASONING_REPLAY];
  readonly excludes: readonly string[] = [];
  readonly params: ExchangeAgingParams;

  constructor(params: ExchangeAgingParams) {
    this.params = params;
  }

  init(): ExchangeAgingState {
    return { aged: 0, observationGroups: 0, agingCount: 0, keptExchange: null };
  }

  stateFields(state: ExchangeAgingState): Record<string, unknown> {
    return {
      exchangeAgedCount: state.aged,
      exchangeAgedObservationGroups: state.observationGroups,
      exchangeAgingCount: state.agingCount,
      ...(this.params.keep_anchor ? { exchangeKeptExchange: state.keptExchange } : {}),
    };
  }

  restoreFields(fields: Record<string, any>): ExchangeAgingState {
    return {
      aged: fields.exchangeAgedCount, observationGroups: fields.exchangeAgedObservationGroups,
      agingCount: fields.exchangeAgingCount, keptExchange: fields.exchangeKeptExchange ?? null,
    };
  }

  /** The anchor an aging to ``target`` leaves as it is (`keep_anchor`): the exact tail's, when before ``target``. */
  private keptAt(view: RequestView, target: number): number | null {
    const anchor = view.exchangeTail?.anchor ?? null;
    return this.params.keep_anchor && anchor !== null && anchor < target ? anchor : null;
  }

  check(view: RequestView): void {
    if (view.unit !== "exchange") throw new PolicySessionError("exchange_aging needs the segments unit exchange");
  }

  /** Long tool-call arguments, stored so an aging can leave them out. */
  store(view: RequestView, _state: ExchangeAgingState, messages: Message[]): Record<string, any>[] {
    if (this.params.argument_bytes === null) return [];  // arguments never age: nothing to store
    return storeArguments(messages, { media: view.services.inlineMedia, argumentBytes: this.params.argument_bytes });
  }

  /** Whether camera images age with their exchange: camera_delivery `exchange_window_v1`. */
  private cameraImages(view: RequestView): boolean {
    return view.paramsOf<CameraDeliveryParams>(CAMERA_DELIVERY)?.mode === CAMERA_EXCHANGE_WINDOW;
  }

  render(view: RequestView, state: ExchangeAgingState, history: Message[]): Rendered {
    const cameraImages = this.cameraImages(view);
    const key = `${state.aged}:${state.observationGroups}:${state.keptExchange}`;
    const render = view.memoize(this.id, history, key, () => renderAging(history, {
      agedExchanges: state.aged, agedObservationGroups: state.observationGroups, keptExchange: state.keptExchange,
      resultPreview: this.params.result_preview ?? "head_and_tail",
      argumentBytes: this.params.argument_bytes ?? Number.MAX_SAFE_INTEGER, resultBytes: this.params.result_bytes,
      preview: this.params.preview,
      cameraImages, mutationTools: view.services.mutationTools, domain: view.services.domain,
    }));
    const note: AgingNote = { input: history, render, cameraImages };
    return { history: render.messages, note: note as unknown as Record<string, any> };
  }

  /** The watermark: the absolute tokens capped by the window ratio. */
  watermark(hardInputTokens: number): number {
    return Math.min(pyInt(num(this.params.trigger_tokens)), pyInt(hardInputTokens * num(this.params.trigger_ratio)));
  }

  admit(view: RequestView, state: ExchangeAgingState): AgingAdmission {
    const watermark = this.watermark(view.limits.hardInputTokens);
    const anchor = view.admissionInputTokens;
    const batch = this.params.exact_tail_exchanges;
    const target = Math.max(0, exchangeCount(view) - batch);
    // Exchanges that have left the exact tail since the last aging (or the backstop) moved the frontier.
    const pending = Math.max(0, target - state.aged);
    let reason: string;
    if (anchor === null) {
      reason = view.usage.lastProviderInputTokens !== null ? "projection_epoch_unanchored" : "provider_usage_unavailable";
    } else if (anchor < watermark) reason = "below_watermark";
    else if (pending < batch) reason = "awaiting_exchange_batch";
    else reason = "new_exchanges_since_last_aging";
    const admitted = reason === "new_exchanges_since_last_aging" && view.allowTransition;
    return {
      admitted, target, watermark,
      record: {
        authority: "provider_reported_input_tokens",
        basis: "projection_entry_provider_usage",
        status: admitted ? "admitted" : "not_admitted",
        provider_input_tokens: anchor,
        provider_projection_epoch: view.usage.lastProviderProjectionEpoch,
        provider_wire_request_sha256: view.usage.lastProviderWireRequestSha256,
        watermark_tokens: watermark,
        new_exchanges_since_last_aging: pending,
        batch_exchanges: batch,
        reason,
      },
    };
  }

  async apply(
    view: RequestView, state: ExchangeAgingState, admission: AgingAdmission, context: ApplyContext,
  ): Promise<Transition | null> {
    const transition = exchangeRewrite(view, context, "exchange_aging", {
      reasons: ["provider_input_watermark", "new_exchanges_since_last_aging"],
      admission_provider_input_tokens: view.admissionInputTokens,
      admission_provider_projection_epoch: view.usage.lastProviderProjectionEpoch,
      admission_provider_wire_request_sha256: view.usage.lastProviderWireRequestSha256,
      watermark_tokens: admission.watermark,
      batch_exchanges: this.params.exact_tail_exchanges,
    }, () => {
      state.aged = Math.max(state.aged, admission.target);
      if (this.params.keep_anchor) state.keptExchange = this.keptAt(view, state.aged);
      // An aging fixes which observation groups are outside the camera tail: the only point where a body may age.
      state.observationGroups = Math.max(state.observationGroups, observationGroupsOutsideCameraTail(view));
      context.notify(REASONING_REPLAY, EXCHANGES_AGED, { aged: admission.target, exchanges: exchangeCount(view) });
    });
    if (transition !== null) state.agingCount += 1;
    return transition;
  }

  /** The hard backstop ages all but its exact tail, observation bodies outside the camera tail included. */
  notice(state: ExchangeAgingState, notice: string, _from: string, detail: NoticeDetail): void {
    if (notice !== BACKSTOP_AGED) throw new PolicySessionError(`exchange_aging takes no notice ${notice}`);
    state.aged = Math.max(state.aged, detail.target);
    state.observationGroups = Math.max(state.observationGroups, detail.observation_groups);
    if (this.params.keep_anchor) {
      state.keptExchange = detail.anchor >= 0 && detail.anchor < state.aged ? detail.anchor : null;
    }
  }

  /**
   * Its passes and the aging record. The exchange passes (with
   * reasoning_replay's) share one measurement, as the reference did: the first
   * from the history before this stage to the history after reasoning_replay,
   * the rest unchanged at the latter, so the argument pass carries the whole
   * reduction (to be split per pass with the evidence format, K5a).
   */
  manifest(view: RequestView, state: ExchangeAgingState, admission: AgingAdmission | null): Record<string, unknown> {
    const note = view.note(this.id) as unknown as AgingNote;
    const reasoningNote = view.note(REASONING_REPLAY) as unknown as ReasoningNote;
    const output: Message[] = reasoningNote.output;
    const sizer = activeSizer();
    const before = sizer.measure(note.input);
    const after = sizer.measure(output);
    const { render } = note;
    const sections: [string, number[], Record<string, any>[]][] = [
      ["exchange_argument_reference", render.arguments, render.argumentRefs],
      ["exchange_tool_result_reference", render.results, render.resultRefs],
      ["exchange_tool_image_reference", render.toolImages, render.toolImageRefs],
      ...(note.cameraImages
        ? [[EXCHANGE_CAMERA_PASS, render.cameraImages, render.cameraImageRefs] as [string, number[], Record<string, any>[]]]
        : []),
      ["exchange_observation_reference", render.observations, render.observationRefs],
    ];
    const passes = sections.map(([name, changed, refs], index) => measuredPassRecord(
      name, index === 0 ? before : after, after, {
        changed_messages: changed, artifact_refs: refs, externalized_event_ids: messageEventIds(output, changed),
      }));
    const reasoning = view.paramsOf<ReasoningReplayParams>(REASONING_REPLAY)!;
    const withheld = view.stateOf<ReasoningReplayState>(REASONING_REPLAY)!.withheld;
    const externalized = (key: string): number => view.working.reduce((sum, message) => sum + num(message[key] ?? 0), 0);
    return {
      passes: clonePy(passes),
      resolved_exchange_aging_trigger_tokens: this.watermark(view.limits.hardInputTokens),
      exchange_aging: clonePy({
        schema: EXCHANGE_AGING_RECORD_SCHEMA,
        exchanges: exchangeCount(view),
        exact_tail_exchanges: this.params.exact_tail_exchanges,
        aged_exchanges: state.aged,
        aged_observation_groups: state.observationGroups,
        reasoning_replay: reasoning.mode,
        reasoning_withheld_exchanges: withheld,
        aging_count: state.agingCount,
        ...(this.params.keep_anchor ? { kept_exchange: state.keptExchange } : {}),
        // Recorded with an exact tail in exchanges, so earlier schemes' manifests keep their bytes.
        ...(view.exactTailUnit === "exchange" ? {
          reasoning_unbound_messages: reasoningNote.unbound.length,
          reasoning_rebound_messages: reasoningNote.rebound.length,
        } : {}),
        admission: admission === null ? null : admission.record,
        delivered: exchangeChanges(view.notes),
        delivered_tool_images_externalized: externalized("_lossless_exchange_images"),
        ...(note.cameraImages ? { delivered_camera_images_externalized: externalized("_lossless_exchange_camera_images") } : {}),
      }),
    };
  }
}
