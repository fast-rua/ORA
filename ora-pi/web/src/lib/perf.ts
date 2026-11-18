// Free play's speed (components/PerfPanel.tsx): how fast the model is and where the time between
// two robot actions goes, so the operator sees how much faster the loop must get.
//
// Every time here is on the agent process's clock (ms since the epoch), as its timings arrive:
//   model calls     the core times each one (ora-pi/src/play/perf.ts, `ora_model_call`, and the
//                   record so far as the run's `perf`): request sent, first content, end, output tokens
//   observations    when the core received each (the run's `perf`), else the observation message's
//                   own timestamp (Pi stamps it when the core adds it)
//   actions         the Supervisor's action_started / action_finished, stamped when the core received
//                   them (the play-state record carries no clock: server/view.ts `at`)
//   approvals       a real cell's motion questions (the gateway's /ora/cell `answers`): when each was
//                   asked and answered and by whom; their clock is the cell's computer, the same one
// A replay rebuilds calls from the agent trace (server/replay.ts): they have no first-token time, so
// their speed counts the whole call.
//
// One loop is an act: its observation, the model's thinking (every model call and tool call until the
// action starts), waiting for the operator's approval, planning and motion (the action's time without
// that wait) and the next observation. Each is a segment of the loop's bar.

import { duration, t } from "./i18n.ts";

type Json = any;

export interface ModelCall {
  id: string;
  step: number | null;
  startedMs: number;
  firstTokenMs: number | null;
  endedMs: number;
  outputTokens: number;
  inputTokens: number;
}

export interface ActionTime {
  step: number;
  actionId: string | null;
  startedMs: number | null;
  finishedMs: number | null;
  durationS: number | null;
  /** The Supervisor's measure of the motion's wait for the operator (action_finished.approval_wait_s). */
  approvalWaitS?: number | null;
}

/** A real cell's motion question, answered (web_cell.py timed_answers). */
export interface TimedAnswer {
  id: string;
  by: string;
  approved: boolean;
  answeredMs: number;
  /** When it was asked; null when the gateway never saw it waiting. */
  createdMs: number | null;
}

export interface PerfLog {
  calls: ModelCall[];
  observations: Map<number, number>;
  actions: Map<number, ActionTime>;
}

export const emptyPerf: PerfLog = { calls: [], observations: new Map(), actions: new Map() };

const KEEP = 200;
/** The window the summary reads: the last calls and the last loops. */
export const WINDOW = 10;
/** An answer belongs to the action it was given in (with this much slack for the two processes' moments). */
const SLACK_MS = 1000;
export const DEFAULT_TARGET_S = 3;
const TARGET_KEY = "ora.web.loopTarget";

const finite = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);

function callOf(record: Json): ModelCall | null {
  const startedMs = finite(record?.startedMs);
  const endedMs = finite(record?.endedMs);
  if (typeof record?.id !== "string" || startedMs === null || endedMs === null) return null;
  return { id: record.id, step: Number.isInteger(record.step) ? record.step : null, startedMs,
    firstTokenMs: finite(record.firstTokenMs), endedMs, outputTokens: Math.max(0, finite(record.outputTokens) ?? 0),
    inputTokens: Math.max(0, finite(record.inputTokens) ?? 0) };
}

/** A model call ended (the `model_call` event). */
export function withCall(log: PerfLog, record: Json): PerfLog {
  const call = callOf(record);
  if (call === null) return log;
  return { ...log, calls: [...log.calls.filter((item) => item.id !== call.id), call].slice(-KEEP) };
}

/** The Supervisor's action_started / action_finished, at the moment it arrived. */
export function withPlay(log: PerfLog, play: Json, at: number): PerfLog {
  const step = Number(play?.action?.observation_step);
  if (!["action_started", "action_finished"].includes(play?.state) || !Number.isInteger(step)) return log;
  const previous = log.actions.get(step) ?? { step, actionId: null, startedMs: null, finishedMs: null, durationS: null };
  const actionId = typeof play.action.action_id === "string" ? play.action.action_id : previous.actionId;
  // The Supervisor's own wall clock when it gives one, else when the state arrived.
  const started = finite(play.action.started_unix_s);
  const finishedAt = finite(play.action.finished_unix_s);
  const actions = new Map(log.actions);
  actions.set(step, play.state === "action_started" ? { ...previous, actionId, startedMs: started === null ? at : started * 1000 }
    : { ...previous, actionId, startedMs: previous.startedMs ?? (started === null ? null : started * 1000),
      finishedMs: finishedAt === null ? at : finishedAt * 1000, durationS: finite(Number(play.action.duration_s)),
      approvalWaitS: play.action.approval_wait_s === null ? null : finite(play.action.approval_wait_s) });
  return { ...log, actions };
}

/** The core's record so far (the run's `perf`: ora_get_run). */
export function withRunPerf(log: PerfLog, perf: Json): PerfLog {
  if (!perf || typeof perf !== "object") return log;
  let next = log;
  for (const call of Array.isArray(perf.calls) ? perf.calls : []) next = withCall(next, call);
  const observations = new Map(next.observations);
  for (const item of Array.isArray(perf.observations) ? perf.observations : []) {
    if (Array.isArray(item) && Number.isInteger(item[0]) && finite(item[1]) !== null) observations.set(item[0], item[1]);
  }
  const actions = new Map(next.actions);
  for (const item of Array.isArray(perf.actions) ? perf.actions : []) {
    if (!Number.isInteger(item?.step)) continue;
    const previous = actions.get(item.step);
    actions.set(item.step, { step: item.step, actionId: item.actionId ?? previous?.actionId ?? null,
      startedMs: finite(item.startedMs) ?? previous?.startedMs ?? null, finishedMs: finite(item.finishedMs) ?? previous?.finishedMs ?? null,
      durationS: finite(item.durationS) ?? previous?.durationS ?? null,
      approvalWaitS: finite(item.approvalWaitS) ?? previous?.approvalWaitS ?? null });
  }
  return { ...next, observations, actions };
}

/** Observations by their messages' timestamps, where the core's own moment is not known. */
export function withObservations(log: PerfLog, messages: Json[]): PerfLog {
  let observations: Map<number, number> | null = null;
  for (const message of messages) {
    if (message?.role !== "custom" || message.customType !== "ora.play_observation") continue;
    const step = message.details?.ora_observation_step;
    const at = finite(message.timestamp);
    if (!Number.isInteger(step) || at === null || (observations ?? log.observations).has(step)) continue;
    observations ??= new Map(log.observations);
    observations.set(step, at);
  }
  return observations === null ? log : { ...log, observations };
}

/** The gateway's answered motion questions (seconds) on the page's ms. */
export function timedAnswers(list: Json): TimedAnswer[] {
  return (Array.isArray(list) ? list : []).flatMap((item: Json) => {
    const answered = finite(item?.answered_unix_s);
    if (typeof item?.id !== "string" || answered === null) return [];
    const created = finite(item.created_unix_s);
    return [{ id: item.id, by: String(item.by ?? ""), approved: item.approved === true, answeredMs: answered * 1000,
      createdMs: created === null ? null : created * 1000 }];
  });
}

/** Who answered: the operator, or the cell itself (safe mode, auto-approve). */
export function answeredBy(by: string): "person" | "safe" | "auto" {
  if (by.startsWith("safe mode (")) return "safe";
  if (by.startsWith("auto-approve (")) return "auto";
  return "person";
}

export function median(values: number[]): number | null {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

// -- One loop -------------------------------------------------------------------------------

export const SEGMENTS = ["model", "tools", "approval", "motion", "observe"] as const;
export type SegmentKind = typeof SEGMENTS[number];

export interface Loop {
  step: number;
  observedMs: number;
  startedMs: number | null;
  finishedMs: number | null;
  nextMs: number | null;
  calls: number;
  /** Generation time (first token to end) of its calls; their whole time when a call has no first token. */
  generationMs: number;
  /** Ms by segment; null while it has not happened (or, for approval, when it is not known). */
  segments: Record<SegmentKind, number | null>;
  /** Who answered its motion question(s), and whether every wait is known. */
  approvalBy: string | null;
  /** Its last answer approved the motion (null: no answer). */
  approved: boolean | null;
  approvalKnown: boolean;
  totalMs: number | null;
  complete: boolean;
}

function generation(call: ModelCall): number {
  return Math.max(0, call.endedMs - (call.firstTokenMs ?? call.startedMs));
}

/**
 * Every act that started an action, oldest first. `answers` null: the run has no approval record (not a
 * real cell, or a replay): the action's time is all planning and motion.
 */
export function buildLoops(log: PerfLog, answers: TimedAnswer[] | null): Loop[] {
  const observed = [...log.observations].sort((a, b) => a[0] - b[0]);
  const loops: Loop[] = [];
  for (const action of [...log.actions.values()].sort((a, b) => a.step - b.step)) {
    const start = observed.find(([step]) => step === action.step);
    if (!start || action.startedMs === null) continue;
    const observedMs = start[1];
    const thinkingEnd = action.startedMs;
    const calls = log.calls.filter((call) => call.step === action.step && call.startedMs >= observedMs - SLACK_MS
      && call.startedMs <= thinkingEnd);
    const model = Math.min(Math.max(0, thinkingEnd - observedMs),
      calls.reduce((sum, call) => sum + Math.max(0, call.endedMs - call.startedMs), 0));
    const tools = Math.max(0, thinkingEnd - observedMs - model);
    const finished = action.finishedMs;
    const span = finished === null ? null : Math.max(0, finished - thinkingEnd);
    let approval: number | null = null;
    let approvalBy: string | null = null;
    let approved: boolean | null = null;
    let approvalKnown = answers !== null;
    const measured = action.approvalWaitS ?? null;
    if (measured !== null) {
      // The Supervisor measured it from the motion's receipt: exact, page open or not, live or replayed.
      approval = span === null ? measured * 1000 : Math.min(measured * 1000, span);
      approvalKnown = true;
    }
    if (answers !== null) {
      const until = (finished ?? Number.POSITIVE_INFINITY) + SLACK_MS;
      const mine = answers.filter((answer) => answer.answeredMs >= thinkingEnd - SLACK_MS && answer.answeredMs <= until);
      approvalBy = mine.length ? mine[mine.length - 1].by : null;
      approved = mine.length ? mine[mine.length - 1].approved : null;
      if (measured === null) {
        approvalKnown = mine.every((answer) => answer.createdMs !== null);
        // An action still running with no answer yet may still be waiting: not known until it ends.
        if (finished === null && !mine.length) approvalKnown = false;
        approval = approvalKnown ? mine.reduce((sum, answer) => sum + Math.max(0, answer.answeredMs - answer.createdMs!), 0) : null;
        if (approval !== null && span !== null) approval = Math.min(approval, span);
      }
    }
    const next = finished === null ? undefined : observed.find(([step, at]) => step > action.step && at >= finished - SLACK_MS);
    const nextMs = next ? next[1] : null;
    loops.push({
      step: action.step, observedMs, startedMs: thinkingEnd, finishedMs: finished, nextMs, calls: calls.length,
      generationMs: calls.reduce((sum, call) => sum + generation(call), 0),
      segments: { model, tools, approval, motion: span === null ? null : span - (approval ?? 0),
        observe: nextMs === null || finished === null ? null : Math.max(0, nextMs - finished) },
      approvalBy, approved, approvalKnown, totalMs: nextMs === null ? null : nextMs - observedMs, complete: nextMs !== null,
    });
  }
  return loops;
}

// -- The summary --------------------------------------------------------------------------------

export interface CallStats {
  /** Output tokens per second of generation, these calls together (null without timed tokens). */
  tps: number | null;
  ttftMs: number | null;
  durationMs: number | null;
  outputTokens: number | null;
}

/** A call without a first-token time (a replay's) counts its whole time as generation. */
function stats(calls: ModelCall[], pick: "one" | "median"): CallStats {
  let tokens = 0;
  let seconds = 0;
  for (const call of calls) {
    const gen = generation(call);
    if (call.outputTokens <= 0 || gen <= 0) continue;
    tokens += call.outputTokens;
    seconds += gen / 1000;
  }
  const of = (values: number[]) => (pick === "one" ? values[0] ?? null : median(values));
  return {
    tps: seconds > 0 ? tokens / seconds : null,
    ttftMs: of(calls.flatMap((call) => (call.firstTokenMs === null ? [] : [call.firstTokenMs - call.startedMs]))),
    durationMs: of(calls.map((call) => call.endedMs - call.startedMs)),
    outputTokens: of(calls.map((call) => call.outputTokens)),
  };
}

export interface Advice {
  kind: "meets" | "over";
  targetMs: number;
  loopMs: number;
  overMs: number;
  dominant: SegmentKind;
  /** The dominant segment's part of the loop (its median over the medians' sum). */
  share: number;
  segmentMs: number;
  /** What the dominant segment would have to be, the rest unchanged (null: cutting it alone cannot). */
  segmentNeedMs: number | null;
  /** The model's speed now and what it would have to be, the rest unchanged (to: null when tokens alone cannot). */
  tps: { from: number; to: number | null; restMs: number } | null;
}

export interface PerfSummary {
  last: ModelCall | null;
  lastStats: CallStats;
  window: CallStats;
  /** The calls' first tokens are known: tok/s counts generation only (else the whole call: a replay). */
  timedTokens: boolean;
  callsPerAct: number | null;
  latest: Loop | null;
  /** The last complete loops' median per segment and in total. */
  medians: { segments: Record<SegmentKind, number | null>; totalMs: number | null; generationMs: number | null; count: number };
  advice: Advice | null;
}

export function summarize(log: PerfLog, answers: TimedAnswer[] | null, targetS: number): PerfSummary {
  const calls = log.calls.slice(-WINDOW);
  const last = log.calls.at(-1) ?? null;
  const loops = buildLoops(log, answers);
  const complete = loops.filter((loop) => loop.complete).slice(-WINDOW);
  const segments = Object.fromEntries(SEGMENTS.map((kind) => [kind,
    median(complete.flatMap((loop) => (loop.segments[kind] === null ? [] : [loop.segments[kind]!])))])) as Record<SegmentKind, number | null>;
  const medians = { segments, totalMs: median(complete.map((loop) => loop.totalMs!)),
    generationMs: median(complete.map((loop) => loop.generationMs)), count: complete.length };
  const window = stats(calls, "median");
  return {
    last, lastStats: stats(last ? [last] : [], "one"), window,
    timedTokens: calls.some((call) => call.firstTokenMs !== null),
    callsPerAct: median(complete.map((loop) => loop.calls)), latest: loops.at(-1) ?? null, medians,
    advice: advise(medians, window.tps, targetS * 1000),
  };
}

/**
 * How far the median loop is from the target, the segment that dominates it and what it would need,
 * the other segments unchanged; and the model's speed it would take, generation time scaling with
 * 1/TPS (the calls' first-token waits unchanged).
 */
export function advise(medians: PerfSummary["medians"], tps: number | null, targetMs: number): Advice | null {
  const loopMs = medians.totalMs;
  if (loopMs === null || !(targetMs > 0)) return null;
  const known = SEGMENTS.flatMap((kind) => (medians.segments[kind] === null ? [] : [[kind, medians.segments[kind]!] as const]));
  const sum = known.reduce((total, [, value]) => total + value, 0);
  if (!known.length || sum <= 0) return null;
  const [dominant, segmentMs] = known.reduce((best, item) => (item[1] > best[1] ? item : best));
  const overMs = loopMs - targetMs;
  const generationMs = medians.generationMs ?? 0;
  const speed = tps !== null && tps > 0 && generationMs > 0
    ? { from: tps, to: overMs <= 0 ? tps : generationMs > overMs ? tps * generationMs / (generationMs - overMs) : null,
      restMs: Math.max(0, loopMs - generationMs) }
    : null;
  return {
    kind: overMs <= 0 ? "meets" : "over", targetMs, loopMs, overMs, dominant, share: segmentMs / sum, segmentMs,
    segmentNeedMs: overMs <= 0 ? segmentMs : segmentMs > overMs ? segmentMs - overMs : null, tps: speed,
  };
}

// -- Words ---------------------------------------------------------------------------------------

export function segmentName(kind: SegmentKind): string {
  return t(`perf.segment.${kind}`);
}

export function ms(value: number | null): string {
  return value === null ? "—" : duration(value / 1000);
}

export function tokensPerSecond(value: number | null): string {
  return value === null ? "—" : String(Math.round(value));
}

const capital = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

/** The advice in words: the target, the segment that dominates, what it needs, and the model's speed. */
export function adviceLines(advice: Advice | null): string[] {
  if (advice === null) return [t("perf.advice.none")];
  const target = duration(advice.targetMs / 1000);
  const share = `${Math.round(advice.share * 100)}%`;
  const name = segmentName(advice.dominant);
  if (advice.kind === "meets") {
    return [t("perf.advice.meets", { loop: ms(advice.loopMs), target, spare: ms(-advice.overMs) })];
  }
  const lines = [t("perf.advice.over", { loop: ms(advice.loopMs), target, over: ms(advice.overMs) })];
  const tps = advice.tps;
  if (advice.dominant === "model" && tps !== null) {
    lines.push(tps.to !== null
      ? t("perf.advice.modelTps", { share, from: tokensPerSecond(tps.from), to: tokensPerSecond(tps.to), target })
      : t("perf.advice.modelCannot", { share, target, rest: ms(tps.restMs) }));
  } else {
    lines.push(capital(advice.segmentNeedMs !== null
      ? t("perf.advice.segment", { name, share, from: ms(advice.segmentMs), to: ms(advice.segmentNeedMs), target })
      : t("perf.advice.segmentCannot", { name, share, target, rest: ms(advice.loopMs - advice.segmentMs) })));
    if (tps !== null) {
      lines.push(tps.to !== null
        ? t("perf.advice.tps", { from: tokensPerSecond(tps.from), to: tokensPerSecond(tps.to), target })
        : t("perf.advice.tpsCannot", { target, rest: ms(tps.restMs) }));
    }
  }
  lines.push(t("perf.advice.assumption"));
  return lines;
}

/** The page's target loop time, as this browser last set it (localStorage, when it allows). */
export function storedTarget(): number {
  try {
    const value = Number(globalThis.localStorage?.getItem(TARGET_KEY));
    return Number.isFinite(value) && value > 0 ? value : DEFAULT_TARGET_S;
  } catch {
    return DEFAULT_TARGET_S;
  }
}

export function rememberTarget(value: number): void {
  try {
    globalThis.localStorage?.setItem(TARGET_KEY, String(value));
  } catch {
    // a browser that keeps nothing: the target lasts this page
  }
}
