// The segmenter service: the units a request's history ages and is summarized
// by. Built once per request, right after materialization, so no stage depends
// on another to have set it.
//
// * ActionSteps (unit `action_step`): an observation and everything up to the
//   next one (`observationGroups`); the inventory records each one's Session
//   round and whether it completed. The summarize stages, the exact causal tail
//   and the lossless invariants always use them.
// * Exchanges (unit `exchange`): one assistant message and everything after it
//   up to the next assistant message (its tool results, the images it got back,
//   the observations that arrived); in code mode, one script written and run.
//   Exchange k starts at the k-th assistant message after the static prefix;
//   the messages between the prefix and the first assistant message (the
//   opening observation) age with exchange 0.
//
// A scheme's `segments.unit` names the unit its age stages count in.
//
// The exact tail is what no summary absorbs. In ActionSteps (the default,
// `segments.exact_tail_unit: action_step`) it is the latest `exact_tail`
// completed ActionSteps and the open one. In exchanges
// (`exact_tail_unit: exchange`) it is the latest `exact_tail` exchanges plus
// the anchor exchange, however old: the exchange of the latest robot action
// (the script or call that ran it, its tool result with the receipt, and the
// observation after it), found as the exchange where the current observation
// arrived (exchange 0 before any action). In code mode an ActionStep can hold
// dozens of exchanges, so an ActionStep tail would keep them all.

import type { ContextDomain } from "../../domain_defaults.ts";
import { type Message, actionStepInventory, observationGroups } from "../../projection.ts";
import { PolicySessionError, get } from "../../pyutil.ts";

export type SegmentUnit = "action_step" | "exchange";
export const SEGMENT_UNITS: readonly SegmentUnit[] = ["action_step", "exchange"];

/** The exact tail in exchanges (`exact_tail_unit: exchange`). */
export interface ExchangeTail {
  /** Exchanges the tail keeps (`segments.exact_tail`). */
  count: number;
  /** The first exchange of the latest ``count`` (every exchange from it on is exact). */
  from: number;
  /** The anchor exchange (the latest robot action's), exact however old; null without an exchange. */
  anchor: number | null;
  /** The exact message ranges [start, end): the anchor's when it is before ``from``, then the tail's. */
  ranges: [number, number][];
}

export interface Exchanges {
  /** Where the history after the static prefix starts (the first observation). */
  from: number;
  /** The message index of each exchange's assistant message. */
  starts: number[];
  /** Each exchange's message span [start, end); exchange 0 starts at ``from``. */
  spans: [number, number][];
}

export interface Segmentation {
  unit: SegmentUnit;
  /** ActionSteps: observation groups, their inventory, how many completed, the open one's round. */
  inventory: Record<string, any>[];
  groups: [number, number][];
  completedCount: number;
  currentRound: unknown;
  /**
   * ActionSteps kept exact, and the first of them. With an exchange tail: the
   * ActionStep that holds its first exact message, and the completed ones from it.
   */
  exactTail: number;
  exactTailFrom: number;
  /** The unit the exact tail counts in, and with exchanges, which messages it keeps. */
  exactTailUnit: SegmentUnit;
  exchangeTail: ExchangeTail | null;
  exchanges: Exchanges;
  /** The spans of the scheme's unit. */
  spans: [number, number][];
}

/** The exchanges of ``messages``. */
export function exchangesOf(messages: Message[], groups: [number, number][] = observationGroups(messages)): Exchanges {
  const from = groups.length ? groups[0][0] : messages.length;
  const starts: number[] = [];
  for (let index = from; index < messages.length; index += 1) {
    if (get(messages[index], "role") === "assistant") starts.push(index);
  }
  const spans = starts.map((start, k): [number, number] => [
    k === 0 ? from : start, k + 1 < starts.length ? starts[k + 1] : messages.length,
  ]);
  return { from, starts, spans };
}

/** The exchange whose span holds message ``index`` (null when no span does). */
export function exchangeAt(exchanges: Exchanges, index: number): number | null {
  for (let k = exchanges.spans.length - 1; k >= 0; k -= 1) {
    const [start, end] = exchanges.spans[k];
    if (start <= index && index < end) return k;
  }
  return null;
}

/** The exact tail of ``count`` exchanges and the anchor exchange (see the header). */
export function exchangeTail(
  materialized: Message[], groups: [number, number][], exchanges: Exchanges, count: number,
): ExchangeTail {
  const total = exchanges.starts.length;
  const from = Math.max(0, total - count);
  const tailStart = from < total ? exchanges.spans[from][0] : materialized.length;
  const anchor = total ? exchangeAt(exchanges, groups[groups.length - 1][0]) : null;
  const ranges: [number, number][] = [];
  if (anchor !== null && anchor < from) ranges.push([...exchanges.spans[anchor]]);
  ranges.push([total ? tailStart : exchanges.from, materialized.length]);
  return { count, from, anchor, ranges };
}

export function segment(materialized: Message[], options: {
  unit: SegmentUnit; exactTail: number; exactTailUnit?: SegmentUnit; mutationTools: string[]; domain: ContextDomain;
}): Segmentation {
  const inventory = actionStepInventory(materialized, {
    mutationTools: options.mutationTools, requireSessionRound: true, domain: options.domain,
  });
  const groups = observationGroups(materialized);
  if (!groups.length) throw new PolicySessionError("lossless ActionStep Context requires an observation");
  const completedCount = groups.length - 1;
  const exchanges = exchangesOf(materialized, groups);
  const exactTailUnit = options.exactTailUnit ?? "action_step";
  let exactTailFrom = Math.max(0, completedCount - options.exactTail);
  let exchangeExactTail: ExchangeTail | null = null;
  if (exactTailUnit === "exchange") {
    exchangeExactTail = exchangeTail(materialized, groups, exchanges, options.exactTail);
    // ActionStep consumers (L5, the lossless invariants) see the ActionStep holding the first exact message.
    const first = exchangeExactTail.ranges[0][0];
    exactTailFrom = 0;
    groups.forEach(([start], index) => {
      if (start <= first) exactTailFrom = index;
    });
    exactTailFrom = Math.min(exactTailFrom, completedCount);
  }
  return {
    unit: options.unit,
    inventory,
    groups,
    completedCount,
    currentRound: inventory[inventory.length - 1].session_round,
    exactTail: exactTailUnit === "exchange" ? completedCount - exactTailFrom : options.exactTail,
    exactTailFrom,
    exactTailUnit,
    exchangeTail: exchangeExactTail,
    exchanges,
    spans: options.unit === "exchange" ? exchanges.spans : groups,
  };
}
