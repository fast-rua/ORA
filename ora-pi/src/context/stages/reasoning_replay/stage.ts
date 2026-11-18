// reasoning_replay (phase age, after exchange_aging): which assistant messages
// are sent with their provider continuation state, the reasoning the Provider
// replays (README.md).
//
// * `all`: every assistant message an aging leaves unchanged keeps it;
// * `unaged_exchanges`: an aging also withholds it from every exchange it ages,
//   in that rewrite, so nothing is withheld below the aging watermark;
// * `recent_exchanges`: only the latest `tail_exchanges` to 2 x `tail_exchanges`
//   - 1 exchanges keep it; older ones lose it in strides of the window.
// Whatever the mode, an assistant message whose arguments an aging replaced is
// no longer the message its state is bound to: under `binding: strict` (or
// absent) it is sent without it (counted as unbound when the mode would have
// kept it); under `rebind_unsigned` an unsigned state (OpenAI-compatible
// reasoning_content) is bound to the rewritten message and kept (counted as
// rebound), a signed one is withheld as under strict. The hard backstop
// withholds it from every exchange it ages. An exchange an aging leaves as it is (the
// anchor of an exact tail in exchanges, exchange_aging `keep_anchor`) keeps it.
// A withheld state is stored as a Context Artifact and stays on the message as
// private metadata.

import { ProviderContinuationState } from "../../../provider/continuation.ts";
import {
  CONTINUATION_REF_KEY, type Message, PROVIDER_CONTINUATION_MESSAGE_KEY, REBOUND_CONTINUATION_KEY,
  messageEventIds, withheldContinuation,
} from "../../projection.ts";
import { REASONING_REPLAY_RECENT, REASONING_REPLAY_UNAGED } from "../../management.ts";
import { PolicySessionError, canonicalBytesPy, clonePy, get, isMapping } from "../../pyutil.ts";
import { activeSizer, measuredPassRecord } from "../../request_size.ts";
import { locations, spanEnd } from "../exchange_aging/render.ts";
import {
  type ExchangeAgingState, type ReasoningReplayState, exchangeCount, exchangeRewrite,
} from "../exchange_aging/rewrite.ts";
import { BACKSTOP_AGED, EXCHANGE_AGING, EXCHANGES_AGED, MATERIALIZE, REASONING_REPLAY } from "../names.ts";
import type { RequestView } from "../request.ts";
import { exchangesOf } from "../services/segmenter.ts";
import type { Admission, ApplyContext, NoticeDetail, Rendered, Stage, Transition } from "../stage.ts";

export interface ReasoningReplayParams {
  /** `all`, `unaged_exchanges` or `recent_exchanges` (management.ts REASONING_REPLAY_MODES). */
  mode: string;
  /** `recent_exchanges`: the window; the other modes: null. */
  tail_exchanges: number | null;
  /** A rewritten assistant message: `strict` (absent) withholds its state, `rebind_unsigned` rebinds an unsigned one. */
  binding?: "strict" | "rebind_unsigned";
}

interface WindowAdmission extends Admission {
  target: number;
}

/**
 * What one render recorded: the messages it withheld, those of them withheld
 * only because an aging rewrote their arguments (unbound), the rewritten ones
 * whose unsigned state it bound to the rewrite (rebound), the references the
 * withheld ones show, its output.
 */
export interface ReasoningNote {
  withheld: number[];
  unbound: number[];
  rebound: number[];
  refs: Record<string, any>[];
  output: Message[];
}

/** ``message`` (rewritten) with its unsigned state bound to it; the original stays as private metadata. Null when signed. */
function reboundContinuation(message: Message): Message | null {
  const original = message[PROVIDER_CONTINUATION_MESSAGE_KEY];
  const rebound = ProviderContinuationState.fromMapping(original).reboundTo(message);
  if (rebound === null) return null;
  return { ...message, [PROVIDER_CONTINUATION_MESSAGE_KEY]: rebound.toDict(), [REBOUND_CONTINUATION_KEY]: original };
}

/**
 * Withhold the continuation state of every assistant message before the
 * reasoning frontier but in the kept exchange, and of every one whose
 * arguments an aging replaced unless ``rebind`` binds its unsigned state to
 * the rewrite.
 */
function renderReasoning(
  messages: Message[], withheldExchanges: number, agedExchanges: number, keptExchange: number | null,
  rebind: boolean,
): ReasoningNote {
  const output = [...messages];
  const { from, starts, spans } = exchangesOf(messages);
  const agedEnd = spanEnd(from, starts, agedExchanges, messages.length);
  const reasoningEnd = spanEnd(from, starts, withheldExchanges, messages.length);
  const kept = keptExchange !== null && keptExchange < spans.length ? spans[keptExchange] : null;
  const note: ReasoningNote = { withheld: [], unbound: [], rebound: [], refs: [], output };
  for (let index = from; index < Math.max(agedEnd, reasoningEnd); index += 1) {
    const message = output[index];
    if (get(message, "role") !== "assistant") continue;
    const windowed = index < reasoningEnd && !(kept !== null && kept[0] <= index && index < kept[1]);
    // A rewritten message is no longer the one its continuation state is bound to.
    const rewritten = message._lossless_arguments_externalized === true;
    if (!(windowed || rewritten) || !isMapping(get(message, PROVIDER_CONTINUATION_MESSAGE_KEY))) continue;
    const rebound = !windowed && rebind ? reboundContinuation(message) : null;
    if (rebound !== null) {
      output[index] = rebound;
      note.rebound.push(index);
      continue;
    }
    output[index] = withheldContinuation(message);
    note.withheld.push(index);
    if (!windowed) note.unbound.push(index);
    if (isMapping(get(message, CONTINUATION_REF_KEY))) note.refs.push(clonePy(message[CONTINUATION_REF_KEY]));
  }
  return note;
}

export class ReasoningReplayStage implements Stage<ReasoningReplayParams, ReasoningReplayState, WindowAdmission> {
  readonly id = REASONING_REPLAY;
  readonly phase = "age" as const;
  readonly transitions: readonly string[] = ["reasoning_replay_window"];
  readonly requires: readonly string[] = [MATERIALIZE, EXCHANGE_AGING];
  readonly excludes: readonly string[] = [];
  /** It reads the aging's admission: an aging moves the window itself. */
  readonly after: readonly string[] = [EXCHANGE_AGING];
  readonly params: ReasoningReplayParams;

  constructor(params: ReasoningReplayParams) {
    this.params = params;
  }

  init(): ReasoningReplayState {
    return { withheld: 0 };
  }

  stateFields(state: ReasoningReplayState): Record<string, unknown> {
    return { reasoningWithheldExchangeCount: state.withheld };
  }

  restoreFields(fields: Record<string, any>): ReasoningReplayState {
    return { withheld: fields.reasoningWithheldExchangeCount };
  }

  /** Every provider continuation state, stored so a request can leave it out. */
  store(view: RequestView, _state: ReasoningReplayState, messages: Message[]): Record<string, any>[] {
    const refs: Record<string, any>[] = [];
    const located = locations(messages);
    messages.forEach((message, index) => {
      if (get(message, "role") !== "assistant") return;
      const state = get(message, PROVIDER_CONTINUATION_MESSAGE_KEY);
      if (!isMapping(state)) return;
      const [roundIndex, actionStep] = located.get(index) ?? [null, null];
      const reference = view.services.inlineMedia.putBytes(canonicalBytesPy(state), {
        kind: "context_provider_continuation", suffix: ".json", mediaType: "application/json",
        roundIndex, actionStep,
      });
      message[CONTINUATION_REF_KEY] = reference;
      refs.push(reference);
    });
    return refs;
  }

  render(view: RequestView, state: ReasoningReplayState, history: Message[]): Rendered {
    const { aged, keptExchange } = view.stateOf<ExchangeAgingState>(EXCHANGE_AGING)!;
    const rebind = this.params.binding === "rebind_unsigned";
    const note = view.memoize(this.id, history, `${state.withheld}:${aged}:${keptExchange}`,
      () => renderReasoning(history, state.withheld, aged, keptExchange, rebind));
    return { history: note.output, note: note as unknown as Record<string, any> };
  }

  /** `recent_exchanges`: once 2N exchanges carry their reasoning, withhold it back to the latest N (no aging now). */
  admit(view: RequestView, state: ReasoningReplayState): WindowAdmission | null {
    if (view.admission(EXCHANGE_AGING)?.admitted) return null;
    const window = this.params.mode === REASONING_REPLAY_RECENT ? this.params.tail_exchanges : null;
    if (window === null || !view.allowTransition) return null;
    if (exchangeCount(view) - state.withheld < 2 * window) return null;
    return { admitted: true, record: {}, target: exchangeCount(view) - window };
  }

  async apply(
    view: RequestView, state: ReasoningReplayState, admission: WindowAdmission, context: ApplyContext,
  ): Promise<Transition | null> {
    return exchangeRewrite(view, context, "reasoning_replay_window", {
      reasons: ["reasoning_tail_stride"],
      reasoning_tail_exchanges: this.params.tail_exchanges,
    }, () => {
      state.withheld = Math.max(state.withheld, admission.target);
    });
  }

  /** An aging moves the frontier by mode (`all`: not at all); the backstop to its own target. */
  notice(state: ReasoningReplayState, notice: string, _from: string, detail: NoticeDetail): void {
    let frontier: number;
    if (notice === EXCHANGES_AGED) {
      frontier = this.params.mode === REASONING_REPLAY_UNAGED ? detail.aged
        : this.params.mode === REASONING_REPLAY_RECENT ? detail.exchanges - this.params.tail_exchanges! : 0;
    } else if (notice === BACKSTOP_AGED) {
      frontier = detail.target;
    } else {
      throw new PolicySessionError(`reasoning_replay takes no notice ${notice}`);
    }
    state.withheld = Math.max(state.withheld, frontier);
  }

  /** Its pass, measured with exchange_aging's (one measurement for the exchange passes). */
  manifest(view: RequestView): Record<string, unknown> {
    const note = view.note(this.id) as unknown as ReasoningNote;
    const after = activeSizer().measure(note.output);
    return {
      passes: [clonePy(measuredPassRecord("exchange_reasoning_withheld", after, after, {
        changed_messages: note.withheld, artifact_refs: note.refs,
        externalized_event_ids: messageEventIds(note.output, note.withheld),
      }))],
    };
  }
}
