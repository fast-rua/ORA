// Free play's speed, for the operator only (display only: none of it reaches the model's context
// or the agent trace). Each model call is timed here on the core's clock: when Pi sent the request
// (`before_provider_request`), when its first content arrived (the first text, thinking or
// tool-call event of the reply) and when the reply ended (`message_end`), with the reply's output
// tokens. With the observations the core receives and the Supervisor's action_started and
// action_finished announcements (stamped on arrival here: the Supervisor's play-state records carry
// no clock), the terminal's footer says the model's speed and the loop between robot actions:
//
//   45 tok/s · TTFT 0.8 s · loop 6.2 s (target 3 s)
//
// and every Pi RPC client hears each call (`ora_model_call`) and reads the record so far in
// `ora_get_run` (`perf`): `ora web`'s page breaks each loop down (ora-pi/web/src/lib/perf.ts).

type Json = any;

/** One model call, timed on the core's clock (ms since the epoch). */
export interface ModelCallTiming {
  id: string;
  /** The observation step its act answers (null outside an act). */
  step: number | null;
  startedMs: number;
  /** The reply's first content (text, thinking or a tool call); null when none came. */
  firstTokenMs: number | null;
  endedMs: number;
  outputTokens: number;
  inputTokens: number;
  cachedInputTokens: number;
  stopReason: string | null;
}

/** A robot action as the Supervisor announced it, stamped when the announcement arrived here. */
export interface ActionTiming {
  step: number;
  actionId: string | null;
  startedMs: number | null;
  finishedMs: number | null;
  /** The Supervisor's own measure of the execution (monotonic), when it finished. */
  durationS: number | null;
  /** How long the motion waited for the operator (its receipt's timestamps.approval), when known. */
  approvalWaitS: number | null;
}

export interface PerfSnapshot {
  calls: ModelCallTiming[];
  /** [observation step, when the core received it]. */
  observations: Array<[number, number]>;
  actions: ActionTiming[];
}

/** The loop time the footer compares with (ORA_LOOP_TARGET_S, default 3 s). */
export function loopTargetS(env: NodeJS.ProcessEnv = process.env): number {
  const value = Number(env.ORA_LOOP_TARGET_S);
  return Number.isFinite(value) && value > 0 ? value : 3;
}

const KEEP = 200;
/** The footer's window: the last calls and loops it summarizes. */
export const WINDOW = 10;
const CONTENT = /^(text|thinking|toolcall)_(start|delta|end)$/;

export function median(values: number[]): number | null {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Output tokens per second of generation (first token to end), over these calls together. */
export function tokensPerSecond(calls: ModelCallTiming[]): number | null {
  let tokens = 0;
  let seconds = 0;
  for (const call of calls) {
    if (call.firstTokenMs === null || call.outputTokens <= 0) continue;
    const generation = (call.endedMs - call.firstTokenMs) / 1000;
    if (generation <= 0) continue;
    tokens += call.outputTokens;
    seconds += generation;
  }
  return seconds > 0 ? tokens / seconds : null;
}

/** Each finished action's loop: its observation to the next observation (ms), oldest first. */
export function loops(snapshot: PerfSnapshot): Array<{ step: number; ms: number }> {
  const observed = [...snapshot.observations].sort((a, b) => a[0] - b[0]);
  const found: Array<{ step: number; ms: number }> = [];
  for (const action of [...snapshot.actions].sort((a, b) => a.step - b.step)) {
    const start = observed.find(([step]) => step === action.step);
    if (!start || action.finishedMs === null) continue;
    const next = observed.find(([step, at]) => step > action.step && at >= action.finishedMs!);
    if (next) found.push({ step: action.step, ms: next[1] - start[1] });
  }
  return found;
}

function seconds(ms: number): string {
  const s = ms / 1000;
  return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
}

/** The footer's line: the model's speed and the loop, or null before the first timed call. */
export function perfLine(snapshot: PerfSnapshot, targetS: number): string | null {
  const calls = snapshot.calls.slice(-WINDOW);
  if (!calls.length) return null;
  const tps = tokensPerSecond(calls);
  const ttft = median(calls.flatMap((call) => (call.firstTokenMs === null ? [] : [call.firstTokenMs - call.startedMs])));
  const loop = median(loops(snapshot).slice(-WINDOW).map((item) => item.ms));
  return [
    tps === null ? "— tok/s" : `${Math.round(tps)} tok/s`,
    ttft === null ? null : `TTFT ${seconds(ttft)}`,
    `loop ${loop === null ? "—" : seconds(loop)} (target ${targetS} s)`,
  ].filter(Boolean).join(" · ");
}

/** The record of one free-play session's timings (play/driver.ts owns it). */
export class PlayPerf {
  readonly calls: ModelCallTiming[] = [];
  readonly observations = new Map<number, number>();
  readonly actions = new Map<number, ActionTiming>();
  /** Each finished call, as it finishes (the RPC clients' `ora_model_call`, the terminal's footer). */
  onCall: ((call: ModelCallTiming) => void) | null = null;
  private open: { id: string; step: number | null; startedMs: number; firstTokenMs: number | null } | null = null;
  private readonly clock: () => number;

  constructor(clock: () => number = Date.now) {
    this.clock = clock;
  }

  /** Pi is sending a model call's request. */
  requested(id: string, step: number | null): void {
    this.open = { id, step, startedMs: this.clock(), firstTokenMs: null };
  }

  /** A streamed event of the reply (Pi's assistantMessageEvent): its first content is the first token. */
  streamed(update: Json): void {
    if (this.open !== null && this.open.firstTokenMs === null && CONTENT.test(String(update?.type ?? ""))) {
      this.open.firstTokenMs = this.clock();
    }
  }

  /** The reply ended (Pi's assistant message, with its usage). */
  replied(message: Json): ModelCallTiming | null {
    const open = this.open;
    this.open = null;
    if (open === null) return null;
    const usage = message?.usage ?? {};
    const count = (value: unknown) => Math.max(0, Number(value) || 0);
    const call: ModelCallTiming = {
      ...open, endedMs: this.clock(),
      outputTokens: count(usage.output), inputTokens: count(usage.input) + count(usage.cacheRead),
      cachedInputTokens: count(usage.cacheRead), stopReason: message?.stopReason ? String(message.stopReason) : null,
    };
    this.calls.push(call);
    if (this.calls.length > KEEP) this.calls.splice(0, this.calls.length - KEEP);
    this.onCall?.(call);
    return call;
  }

  /** The Supervisor handed the core an observation (an act begins). */
  observed(step: number): void {
    this.observations.set(step, this.clock());
    if (this.observations.size > KEEP) this.observations.delete(this.observations.keys().next().value!);
  }

  /** A play-state announcement (episode/play.py): the robot's actions; returns when it arrived. */
  announced(record: Json): number {
    const now = this.clock();
    const action = record?.action;
    const step = Number(action?.observation_step);
    if (!["action_started", "action_finished"].includes(record?.state) || !Number.isInteger(step)) return now;
    const previous = this.actions.get(step)
      ?? { step, actionId: null, startedMs: null, finishedMs: null, durationS: null, approvalWaitS: null };
    const duration = Number(action.duration_s);
    const wait = action.approval_wait_s === null ? NaN : Number(action.approval_wait_s);
    // The Supervisor's own wall clock when it gives one (same host), else when the announcement arrived.
    const at = (seconds: unknown) => (typeof seconds === "number" && Number.isFinite(seconds) ? seconds * 1000 : now);
    this.actions.set(step, record.state === "action_started"
      ? { ...previous, actionId: action.action_id ?? previous.actionId, startedMs: at(action.started_unix_s) }
      : { ...previous, actionId: action.action_id ?? previous.actionId,
        startedMs: previous.startedMs ?? (typeof action.started_unix_s === "number" ? at(action.started_unix_s) : null),
        finishedMs: at(action.finished_unix_s), durationS: Number.isFinite(duration) ? duration : null,
        approvalWaitS: Number.isFinite(wait) ? wait : null });
    if (this.actions.size > KEEP) this.actions.delete(this.actions.keys().next().value!);
    return now;
  }

  snapshot(): PerfSnapshot {
    return { calls: [...this.calls], observations: [...this.observations], actions: [...this.actions.values()] };
  }

  line(targetS: number = loopTargetS()): string | null {
    return perfLine(this.snapshot(), targetS);
  }
}
