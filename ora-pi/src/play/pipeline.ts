// Free play's pipelined next step (plan play.pipeline_predict; off by default).
//
// On a real cell a motion takes seconds, and the model waits for it: the loop is think, move, look, think.
// With this switch the ORA model starts thinking while the robot moves. When a reply's world-changing call
// is submitted, a prediction request goes out at once: the context of the call that made it, that reply,
// the call's result (submitted), and one note asking for the next call on the assumption that the motion
// succeeds. The prediction never runs by itself. At the next real call, its reply is used only when
// deterministic rules pass on the fresh observation's receipt (`receiptVerdict`): the motion reached its
// target, or a closing gripper stalled on something at least MIN_GRASP_APERTURE_MM wide; and nothing but
// that observation arrived meanwhile (an operator's message discards it). Then the reply is replayed as
// the response to the real request (the request is recorded, marked as a replay); the Supervisor admits
// and checks its call as any other. Otherwise the prediction is discarded and the real request is sent.
//
// The prediction shares the real request's prefix up to the call's result, so it also warms the
// server's prefix cache for that request. Each prediction is recorded (`pipeline_prediction`: its
// request as sent, its reply, and whether it was used and why not).

import {
  createAssistantMessageEventStream, normalizeContext,
  type AssistantMessage, type AssistantMessageEventStream, type Message,
} from "@earendil-works/pi-ai";

type Json = any;
type StreamFn = (model: Json, context: Json, options?: Json) => AssistantMessageEventStream;

/** The note that stands in for the observation the prediction cannot see yet. */
export const PREDICT_NOTE = [
  "[ORA pipeline] The world-changing call you just made is executing on the robot now; its result and a",
  "fresh observation are not here yet. Assume it completes as planned (the motion reaches its target; a",
  "closing gripper stops on the object). Reply now with your next step as you would after that observation.",
  "It will be used only if the real receipt confirms the motion succeeded; otherwise you will be asked",
  "again with the fresh observation.",
].join(" ");

/** A closing gripper that stops at least this wide holds something (narrower: an empty or slipping grasp). */
export const MIN_GRASP_APERTURE_MM = 5;

/** The verdict on the receipt a fresh observation reports (its `last_action_effect`). */
export function receiptVerdict(effect: Json): { pass: boolean; reason: string } {
  if (effect === null || typeof effect !== "object") return { pass: false, reason: "no_receipt" };
  const status = effect.status;
  if (status === "reached" && !effect.failure_code) return { pass: true, reason: "reached" };
  if (effect.action === "set_gripper" && effect.failure_code === "gripper_stalled_before_target") {
    const arms = Object.values(effect.detail?.arms ?? {}) as Json[];
    const closing = arms.map((arm) => arm?.gripper_outcome).filter((outcome) => outcome && outcome.goal === "close");
    if (closing.length && closing.every((outcome) => outcome.state === "stalled_before_target"
        && Number(outcome.measured_aperture_mm) >= MIN_GRASP_APERTURE_MM)) {
      return { pass: true, reason: "grasp_stalled_on_object" };
    }
    return { pass: false, reason: "grasp_empty_or_open" };
  }
  return { pass: false, reason: `${status ?? "unknown"}${effect.failure_code ? `:${effect.failure_code}` : ""}` };
}

/** The receipt in an observation message as the model reads it (its first text block is the JSON projection). */
export function observationEffect(message: Json): Json | null {
  const content = Array.isArray(message?.content) ? message.content : [{ type: "text", text: message?.content }];
  for (const block of content) {
    if (block?.type !== "text" || typeof block.text !== "string" || !block.text.trimStart().startsWith("{")) continue;
    try {
      const parsed = JSON.parse(block.text);
      if (parsed && typeof parsed === "object" && "last_action_effect" in parsed) return parsed.last_action_effect;
    } catch {
      continue;
    }
  }
  return null;
}

interface RealCall { model: Json; context: Json; options: Json }

interface Prediction {
  id: string;
  /** The prediction's context: the real messages it extends (length) and the reply and result it adds. */
  baseLength: number;
  toolCallId: string;
  reply: Promise<AssistantMessage | null>;
  abort: AbortController;
  startedMs: number;
  endedMs: number | null;
}

export interface PipelineRecorder {
  /** Record one evidence event (`pipeline_prediction`), with its artifact. */
  record(fields: Json, artifact: Json): void;
  now(): number;
}

/** The ORA model's request path with predictions (the provider's `streamSimple`). */
export class PipelinePredictor {
  private last: RealCall | null = null;
  private reply: AssistantMessage | null = null;
  private prediction: Prediction | null = null;
  private count = 0;
  /** Records still waiting for their prediction's reply (``discard``); ``close`` waits for them. */
  private readonly recording = new Set<Promise<void>>();
  readonly stats = { started: 0, used: 0, discarded: 0 };

  private readonly base: StreamFn;
  private readonly recorder: PipelineRecorder;

  constructor(base: StreamFn, recorder: PipelineRecorder) {
    this.base = base;
    this.recorder = recorder;
  }

  /** The provider's streamSimple: a verified prediction is replayed, any other request is sent. */
  stream(model: Json, context: Json, options?: Json): AssistantMessageEventStream {
    const prediction = this.prediction;
    this.prediction = null;
    if (prediction !== null) {
      const replay = this.replay(prediction, model, context, options);
      if (replay !== null) return replay;
    }
    this.last = { model, context, options };
    this.reply = null;
    return this.base(model, context, options);
  }

  /** The reply a real call ended with (Pi's message_end for the ORA model). */
  noteReply(message: AssistantMessage): void {
    this.reply = message;
  }

  /** A world-changing call of the last reply was submitted: predict the next step while it runs. */
  start(toolCallId: string, toolName: string, content: Json[]): void {
    this.cancel("superseded");
    const last = this.last;
    const reply = this.reply;
    if (last === null || reply === null) return;
    const calls = reply.content.filter((block: Json) => block.type === "toolCall");
    if (!calls.length || (calls[calls.length - 1] as Json).id !== toolCallId) return;  // a later call would run first
    const result: Json = { role: "toolResult", toolCallId, toolName, content, isError: false, timestamp: Date.now() };
    const note: Json = { role: "user", content: [{ type: "text", text: PREDICT_NOTE }], timestamp: Date.now() };
    const messages: Message[] = [...last.context.messages, reply, result, note];
    const abort = new AbortController();
    const id = `pipeline-prediction-${String(++this.count).padStart(6, "0")}`;
    let payload: Json = null;
    const { onPayload: _payload, onResponse: _response, onProviderStreamEvent: _events, signal: _signal, ...options }
      = last.options ?? {};
    const prediction: Prediction = {
      id, baseLength: last.context.messages.length, toolCallId, abort, startedMs: this.recorder.now(), endedMs: null,
      reply: Promise.resolve(null),
    };
    prediction.reply = (async () => {
      try {
        const message = await this.base(last.model, normalizeContext({ messages } as Json), {
          ...options, signal: abort.signal,
          onPayload: (body: Json) => { payload = body; return undefined; },
        }).result();
        prediction.endedMs = this.recorder.now();
        return message.stopReason === "error" || message.stopReason === "aborted" ? null : message;
      } catch {
        prediction.endedMs = this.recorder.now();
        return null;
      }
    })();
    this.prediction = prediction;
    this.stats.started += 1;
    void prediction.reply.then((message) => {
      (prediction as Json).payload = payload;
      (prediction as Json).message = message;
    });
  }

  /** The session ends: drop the prediction no call will use and wait until every prediction is recorded, so none
   * is lost or reaches the agent trace after it closed (main.ts calls this before the core closes). */
  async close(reason = "session_closed"): Promise<void> {
    this.cancel(reason);
    while (this.recording.size) await Promise.allSettled([...this.recording]);
  }

  /** Drop a prediction no real call will use (the act ended, a stop, a newer submission). */
  cancel(reason: string): void {
    const prediction = this.prediction;
    if (prediction === null) return;
    this.prediction = null;
    prediction.abort.abort();
    this.discard(prediction, reason, null);
  }

  private replay(prediction: Prediction, model: Json, context: Json, options?: Json): AssistantMessageEventStream | null {
    const messages: Json[] = context.messages;
    const added = messages.slice(prediction.baseLength);
    // The real request extends the prediction's base by exactly the reply, its result and one observation.
    const [reply, result, observation, ...rest] = added;
    const shaped = reply?.role === "assistant" && result?.role === "toolResult"
      && result.toolCallId === prediction.toolCallId && observation?.role === "user" && rest.length === 0;
    if (!shaped) {
      prediction.abort.abort();
      this.discard(prediction, "context_changed", null);
      return null;
    }
    const verdict = receiptVerdict(observationEffect(observation));
    if (!verdict.pass) {
      prediction.abort.abort();
      this.discard(prediction, verdict.reason, null);
      return null;
    }
    const stream = createAssistantMessageEventStream();
    void (async () => {
      const message = await prediction.reply;
      if (message === null || !message.content.some((block: Json) => block.type === "toolCall")) {
        this.discard(prediction, message === null ? "prediction_failed" : "prediction_has_no_call", message);
        // Fall back to the real request, streamed through.
        this.last = { model, context, options };
        this.reply = null;
        const real = this.base(model, context, options);
        for await (const event of real) stream.push(event as Json);
        stream.end(await real.result());
        return;
      }
      this.last = { model, context, options };
      this.reply = null;
      this.used(prediction, verdict.reason, message);
      // The request as Pi would have sent it is not sent; the trace records it as a replay of the prediction.
      options?.onPayload?.({ ora_pipeline_replay: { prediction_id: prediction.id, verdict: verdict.reason } }, model);
      for (const event of replayEvents(message)) stream.push(event as Json);
      stream.end(message);
    })();
    return stream;
  }

  private used(prediction: Prediction, reason: string, message: AssistantMessage): void {
    this.stats.used += 1;
    this.recorder.record({
      prediction_id: prediction.id, outcome: "used", reason, tool_call_id: prediction.toolCallId,
      started_ms: prediction.startedMs, ended_ms: prediction.endedMs, replayed_ms: this.recorder.now(),
    }, { request: (prediction as Json).payload ?? null, reply: message });
  }

  private discard(prediction: Prediction, reason: string, message: AssistantMessage | null): void {
    this.stats.discarded += 1;
    const recorded = prediction.reply.then((reply) => {
      this.recorder.record({
        prediction_id: prediction.id, outcome: "discarded", reason, tool_call_id: prediction.toolCallId,
        started_ms: prediction.startedMs, ended_ms: prediction.endedMs,
      }, { request: (prediction as Json).payload ?? null, reply: message ?? reply });
    });
    this.recording.add(recorded);
    void recorded.finally(() => this.recording.delete(recorded));
  }
}

/** The events a finished reply streams as (start, each block's start and end, done), as a provider emits them. */
export function replayEvents(message: AssistantMessage): Json[] {
  const partial: Json = { ...message, content: [] };
  const events: Json[] = [{ type: "start", partial: { ...partial } }];
  message.content.forEach((block: Json, index: number) => {
    if (block.type === "text") {
      partial.content = [...partial.content, { type: "text", text: "" }];
      events.push({ type: "text_start", contentIndex: index, partial: { ...partial } });
      events.push({ type: "text_delta", contentIndex: index, delta: block.text, partial: { ...partial } });
      partial.content = [...partial.content.slice(0, -1), block];
      events.push({ type: "text_end", contentIndex: index, content: block.text, partial: { ...partial } });
    } else if (block.type === "thinking") {
      partial.content = [...partial.content, { type: "thinking", thinking: "" }];
      events.push({ type: "thinking_start", contentIndex: index, partial: { ...partial } });
      events.push({ type: "thinking_delta", contentIndex: index, delta: block.thinking, partial: { ...partial } });
      partial.content = [...partial.content.slice(0, -1), block];
      events.push({ type: "thinking_end", contentIndex: index, content: block.thinking, partial: { ...partial } });
    } else if (block.type === "toolCall") {
      partial.content = [...partial.content, { ...block, arguments: {} }];
      events.push({ type: "toolcall_start", contentIndex: index, partial: { ...partial } });
      partial.content = [...partial.content.slice(0, -1), block];
      events.push({ type: "toolcall_end", contentIndex: index, toolCall: block, partial: { ...partial } });
    }
  });
  events.push({ type: "done", reason: message.stopReason, message });
  return events;
}
