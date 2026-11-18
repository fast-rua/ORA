// The run's conversation as the page draws it, and its index by observation step.
//
// Pi's messages as the run's agent process serves them (ora-pi/src/rpc/images.ts):
//   custom ora.observation / ora.play_observation  an observation; details.ora {step, images}
//   assistant                                     text, thinking and tool calls
//   toolResult                                    a tool call's result; details.ora {step, toolCallId, images}
//   user                                          free play's instruction (an episode's operator
//                                                 messages live in ORA's context, not here)
//   custom ora.play_message                       free play: a message the operator sent while an
//                                                 instruction ran, joined to it; details
//                                                 {ora_message_id, ora_instruction, ora_observation_step}
//   custom ora.continuation                       ORA asking the model to go on (free play: after a
//                                                 reply the output limit cut off, details.ora_reason
//                                                 "output_truncated")
// Free play adds what the agent process knows of the operator's messages
// (run.playMessages: queued ones are shown as queued until the agent reads them)
// and of the instructions that ended (run.playEndings: a marker where each ended).
// An image is inline (an ImageContent block) for recent steps, else listed in
// details.ora.images with a URL the gateway serves.

import { overlayFor, type Overlay } from "./overlays.ts";

type Json = any;

export const OBSERVATIONS = new Set(["ora.observation", "ora.play_observation"]);
export const PLAY_MESSAGE = "ora.play_message";

export interface StepImage {
  camera: string | null;
  src: string;
}

export interface ToolView {
  id: string;
  name: string;
  args: Json;
  step: number | null;
  status: "pending" | "running" | "done" | "error";
  /** The result's text, and its JSON when it is JSON. */
  text: string;
  result: Json;
  images: StepImage[];
  overlay: Overlay | null;
}

export interface ObservationView {
  key: string;
  step: number | null;
  images: StepImage[];
  /** What the model read besides the images (the camera captions left out). */
  text: string;
  play: boolean;
}

export type Item =
  | { kind: "observation"; key: string; observation: ObservationView; time: number | null }
  | { kind: "user"; key: string; text: string; images: StepImage[]; time: number | null; instruction: number | null }
  | { kind: "joined"; key: string; id: string | null; text: string; images: StepImage[]; time: number | null;
    instruction: number | null; step: number | null; state: "queued" | "delivered" | "next" }
  | { kind: "assistant"; key: string; text: string; thinking: string[]; tools: ToolView[]; error: string | null;
    streaming: boolean; step: number | null; time: number | null; answers: string | null;
    /** How long the model call took (Pi's durationMs), the "Thought for N s" of its thinking. */
    durationMs: number | null;
    /** The reply reached the model's output limit (stopReason "length"), with its output tokens when known. */
    truncated: { outputTokens: number | null } | null }
  | { kind: "note"; key: string; title: string; text: string; time: number | null }
  /** A model call skipped at its boundary right after a joined message: superseded, not a failure. */
  | { kind: "skipped"; key: string; time: number | null }
  | { kind: "sent"; key: string; text: string; ok: boolean; error: string | null; time: number }
  | { kind: "ended"; key: string; n: number; ended: string; step: number | null; seconds: number | null;
    decisions: number | null; time: number };

export interface StepView {
  step: number;
  observation: ObservationView | null;
  tools: ToolView[];
  /** The conversation item the step starts at (its observation, else its first tool call). */
  key: string;
  /** When its observation reached the agent, and its last tool result (ms, the agent process's clock). */
  startedAt: number | null;
  lastResultAt: number | null;
  /** Seconds from the observation to the agent's last tool result, and from there to the next observation. */
  think: number | null;
  move: number | null;
}

/** Free play's operator messages and ended instructions, as the agent process knows them (ora_get_run). */
export interface PlayContext {
  messages: Json[];
  endings: Json[];
  /** Measured durations of the robot's actions by the step that ran them (action_finished). */
  moves?: Map<number, number>;
  /** Messages this page sent that the run has not admitted yet, and whether an instruction runs. */
  pending?: string[];
  running?: boolean;
}

export interface Conversation {
  items: Item[];
  steps: Map<number, StepView>;
  order: number[];
  tools: Map<string, ToolView>;
}

function blocks(message: Json): Json[] {
  const content = message?.content;
  if (typeof content === "string") return [{ type: "text", text: content }];
  return Array.isArray(content) ? content : [];
}

export function textOf(message: Json): string {
  return blocks(message).filter((block) => block?.type === "text").map((block) => String(block.text ?? "")).join("\n");
}

/** A caption names a camera as `KIND · sequence N · camera · ...` (ORA's observation layout). */
export function cameraOf(caption: string | null): string | null {
  if (!caption) return null;
  const parts = caption.split(" · ").map((part) => part.trim());
  return parts.length >= 3 && /^sequence \d+$/.test(parts[1]) ? parts[2] : caption.length <= 40 ? caption : null;
}

function isCaption(text: string): boolean {
  return text.length <= 200 && !text.trimStart().startsWith("{") && !text.includes("\n");
}

/** What the abort signal says when a call is stopped before or while it is sent (Node, browsers, Pi). */
const ABORTED = /^(this operation was aborted|the operation was aborted|request was aborted|operation aborted)\.?$/i;

/**
 * A call stopped at its boundary rather than failed: Pi marks it `aborted`, or, when the stop came
 * before the request was sent, `error` with the abort signal's own message.
 */
export function stoppedAtBoundary(message: Json): boolean {
  if (message?.stopReason === "aborted") return true;
  return message?.stopReason === "error" && ABORTED.test(String(message?.errorMessage ?? "").trim());
}

function dataUrl(block: Json): string {
  return `data:${block.mimeType ?? "image/png"};base64,${block.data}`;
}

/** The images of a message, by camera: inline blocks, and the ones listed by URL only. */
export function imagesOf(message: Json): StepImage[] {
  const inline: Array<{ block: Json; caption: string | null }> = [];
  let caption: string | null = null;
  for (const block of blocks(message)) {
    if (block?.type === "text") caption = isCaption(String(block.text ?? "")) ? String(block.text).trim() : null;
    else if (block?.type === "image" && typeof block.data === "string") {
      inline.push({ block, caption });
      caption = null;
    }
  }
  const listed: Json[] = Array.isArray(message?.details?.ora?.images) ? message.details.ora.images : [];
  if (!listed.length) return inline.map(({ block, caption: text }) => ({ camera: cameraOf(text), src: dataUrl(block) }));
  // Completed by the agent process: inline ones are the last image blocks, in the listed order.
  const tail = inline.slice(inline.length - listed.filter((item) => item.inline).length);
  let next = 0;
  const images: StepImage[] = [];
  for (const item of listed) {
    const src = item.inline && tail[next] ? dataUrl(tail[next++].block) : item.url;
    if (typeof src === "string" && src) images.push({ camera: item.camera ?? null, src });
  }
  return images;
}

function stepOf(message: Json): number | null {
  const ora = Number(message?.details?.ora?.step);
  if (Number.isFinite(ora) && message?.details?.ora?.step !== null) return ora;
  const own = Number(message?.details?.ora_observation_step);
  return Number.isFinite(own) && message?.details?.ora_observation_step !== null ? own : null;
}

export function messageKey(message: Json, index: number): string {
  const time = typeof message?.timestamp === "number" ? message.timestamp : null;
  const id = message?.toolCallId ?? message?.customType ?? "";
  return time === null ? `${message?.role ?? "message"}:${index}` : `${message?.role}:${time}:${id}`;
}

function observationText(message: Json): string {
  return blocks(message).filter((block) => block?.type === "text").map((block) => String(block.text ?? ""))
    .filter((text) => !isCaption(text) || text.length > 120).join("\n").trim();
}

function parseResult(text: string): Json {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return null;
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

/** Live progress of tool calls (server/view.ts `tool` events), by call id. */
export type LiveTools = Map<string, { status: string; name?: string; args?: Json; partial?: Json }>;

/** Free play's and the page's own messages, in order, with the run's steps indexed. */
export function buildConversation(messages: Json[], streaming: Json | null = null, live: LiveTools = new Map(),
  sent: Json[] = [], play: PlayContext | null = null): Conversation {
  const results = new Map<string, Json>();
  for (const message of messages) {
    if (message?.role === "toolResult") results.set(String(message.toolCallId ?? ""), message);
  }
  const items: Item[] = [];
  const steps = new Map<number, StepView>();
  const tools = new Map<string, ToolView>();
  let current: number | null = null;
  const stepView = (step: number, key: string): StepView => {
    let view = steps.get(step);
    if (!view) {
      view = { step, observation: null, tools: [], key, startedAt: null, lastResultAt: null, think: null, move: null };
      steps.set(step, view);
    }
    return view;
  };
  // Free play: an instruction's number by its text, in the order the Supervisor opened them.
  const opened = (play?.messages ?? []).filter((item: Json) => item?.joined === false && Number.isInteger(item.instruction));
  const numberOf = (text: string): number | null => {
    const at = opened.findIndex((item: Json) => String(item.text).trim() === text.trim());
    if (at < 0) return null;
    const [found] = opened.splice(at, 1);
    return found.instruction;
  };
  let unanswered: string | null = null;
  const toolView = (call: Json, step: number | null): ToolView => {
    const id = String(call.id ?? "");
    const result = results.get(id) ?? null;
    const resultStep = result ? stepOf(result) : null;
    const progress = live.get(id);
    // A running call's partial result (a code-mode command's robot calls so far) until its result arrives.
    const partial = !result && progress?.status === "running" && progress.partial ? progress.partial : null;
    const text = result ? textOf(result) : partial ? textOf(partial) : "";
    const parsed = parseResult(text);
    const status: ToolView["status"] = result
      ? (result.isError ? "error" : "done")
      : progress?.status === "running" ? "running" : progress?.status === "error" ? "error"
        : progress?.status === "done" ? "done" : "pending";
    const args = call.arguments ?? call.input ?? {};
    const name = String(call.name ?? progress?.name ?? "tool");
    return {
      id, name, args, step: resultStep ?? step, status, text, result: parsed,
      images: result ? imagesOf(result) : partial ? imagesOf(partial) : [], overlay: overlayFor(name, args, parsed, status),
    };
  };
  const assistant = (message: Json, key: string, streamingNow: boolean) => {
    const content = blocks(message);
    const calls = content.filter((block) => block?.type === "toolCall" && block.id);
    const viewTools = calls.map((call) => toolView(call, current));
    for (const tool of viewTools) {
      tools.set(tool.id, tool);
      if (tool.step !== null) stepView(tool.step, key).tools.push(tool);
    }
    const text = content.filter((block) => block?.type === "text").map((block) => String(block.text ?? "")).join("\n").trim();
    const thinking = content.filter((block) => block?.type === "thinking" && block.thinking)
      .map((block) => String(block.thinking));
    const time = typeof message?.timestamp === "number" ? message.timestamp : null;
    // A call stopped at its boundary before it said anything (free play: the act's action was already
    // submitted, or Esc) is no error: nothing was asked of the model, and nothing went wrong. Right after
    // a message that joined the instruction it is that message's doing (Pi went on to answer it, and the
    // robot's action runs first): shown as superseded. Otherwise it is not shown.
    if (stoppedAtBoundary(message) && !text && !thinking.length && !viewTools.length) {
      if (unanswered !== null) items.push({ kind: "skipped", key, time });
      return;
    }
    const error = message?.stopReason === "error" || message?.stopReason === "aborted"
      ? String(message.errorMessage ?? message.stopReason) : null;
    const output = Number(message?.usage?.output);
    const truncated = message?.stopReason === "length"
      ? { outputTokens: Number.isFinite(output) && output > 0 ? output : null } : null;
    if (text || thinking.length || viewTools.length || error || truncated || streamingNow) {
      // The agent's first words after a message that joined the instruction are its answer.
      const answers = text && unanswered !== null ? unanswered : null;
      if (answers !== null) unanswered = null;
      const took = typeof message?.durationMs === "number" ? message.durationMs : NaN;
      items.push({ kind: "assistant", key, text, thinking, tools: viewTools, error, streaming: streamingNow, step: current,
        time, answers, truncated, durationMs: Number.isFinite(took) && took >= 0 ? took : null });
    }
  };

  messages.forEach((message, index) => {
    const key = messageKey(message, index);
    const time = typeof message?.timestamp === "number" ? message.timestamp : null;
    if (message?.role === "custom" && OBSERVATIONS.has(message.customType)) {
      const step = stepOf(message);
      if (step !== null) current = step;
      const observation: ObservationView = { key, step, images: imagesOf(message), text: observationText(message),
        play: message.customType === "ora.play_observation" };
      if (step !== null) {
        const view = stepView(step, key);
        view.observation = observation;
        view.key = key;
        view.startedAt = time;
      }
      items.push({ kind: "observation", key, observation, time });
    } else if (message?.role === "assistant") {
      assistant(message, key, false);
    } else if (message?.role === "toolResult") {
      const resultStep = stepOf(message) ?? current;
      if (resultStep !== null && time !== null) {
        const view = stepView(resultStep, key);
        view.lastResultAt = Math.max(view.lastResultAt ?? 0, time);
      }
      // Drawn on its call's card; one without a call in view stands alone.
      if (!tools.has(String(message.toolCallId ?? ""))) {
        assistant({ role: "assistant", content: [{ type: "toolCall", id: message.toolCallId, name: message.toolName,
          arguments: {} }] }, key, false);
      }
    } else if (message?.role === "user") {
      const text = textOf(message);
      unanswered = null;
      items.push({ kind: "user", key, text, images: imagesOf(message), time, instruction: play ? numberOf(text) : null });
    } else if (message?.role === "custom" && message.customType === PLAY_MESSAGE) {
      const details = message.details ?? {};
      const id = typeof details.ora_message_id === "string" ? details.ora_message_id : null;
      items.push({ kind: "joined", key, id, text: textOf(message), images: imagesOf(message), time,
        instruction: Number.isInteger(details.ora_instruction) ? details.ora_instruction : null,
        step: Number.isInteger(details.ora_observation_step) ? details.ora_observation_step : null, state: "delivered" });
      unanswered = key;
    } else if (message?.role === "custom") {
      const text = textOf(message);
      if (message.display !== false && text) {
        items.push({ kind: "note", key, title: String(message.customType ?? "note"), text, time });
      }
    }
  });
  if (streaming) assistant(streaming, "streaming", true);
  if (play) placePlay(items, play);
  // What pages sent that the conversation does not show (an episode's operator messages, or not yet delivered).
  const shown = new Set(items.filter((item) => item.kind === "user" || item.kind === "joined")
    .map((item) => (item as Json).text.trim()));
  for (const message of play?.messages ?? []) shown.add(String(message?.text ?? "").trim());
  for (const item of sent) {
    if (item.as !== "prompt" && item.as !== "steer" && item.as !== "follow_up") continue;
    if (String(item.text).trim().startsWith("/")) continue;
    if (shown.has(String(item.text).trim())) continue;
    const at = items.findIndex((other) => other.kind === "user" && other.time !== null && other.time > item.at);
    const entry: Item = { kind: "sent", key: `sent:${item.at}`, text: String(item.text), ok: item.ok !== false,
      error: item.error ?? null, time: item.at };
    if (at < 0) items.push(entry);
    else items.splice(at, 0, entry);
  }
  const order = [...steps.keys()].sort((a, b) => a - b);
  order.forEach((step, index) => {
    const view = steps.get(step)!;
    const next = steps.get(order[index + 1]);
    const measured = play?.moves?.get(step) ?? null;
    if (view.startedAt !== null && view.lastResultAt !== null && view.lastResultAt >= view.startedAt) {
      view.think = (view.lastResultAt - view.startedAt) / 1000;
      if (next?.startedAt != null && next.startedAt >= view.lastResultAt) view.move = (next.startedAt - view.lastResultAt) / 1000;
    }
    if (measured !== null) view.move = measured;
  });
  return { items, steps, order, tools };
}

/**
 * Free play: the messages held for the next instruction (a Queue send: the Supervisor's
 * `message_queued_next`), and this page's own not admitted yet; shown above the prompt, oldest first,
 * until the instruction they open starts.
 */
export function queuedNext(messages: Json[] | undefined, pendingNext: string[] | undefined): Array<{ key: string; text: string }> {
  return [
    ...(messages ?? []).filter((message) => message?.state === "queued" && message.next === true)
      .map((message) => ({ key: String(message.id), text: String(message.text ?? "") })),
    ...(pendingNext ?? []).map((text, index) => ({ key: `pending:${index}:${text}`, text })),
  ];
}

/** Free play: messages not read yet at the end (queued), and a marker where each instruction ended. */
function placePlay(items: Item[], play: PlayContext): void {
  for (const ending of play.endings ?? []) {
    if (typeof ending?.at !== "number") continue;
    const marker: Item = {
      kind: "ended", key: `ended:${ending.instruction_id ?? ending.index}`, n: Number(ending.index) + 1,
      ended: String(ending.ended ?? ""), step: Number.isInteger(ending.observation_steps?.end) ? ending.observation_steps.end : null,
      seconds: Number.isFinite(Number(ending.wall_duration_s)) ? Number(ending.wall_duration_s) : null,
      decisions: Number.isInteger(ending.decisions) ? ending.decisions : null, time: ending.at,
    };
    // After the last item the agent process timed before it heard the instruction end (one clock).
    let at = items.length;
    for (let index = items.length - 1; index >= 0; index--) {
      const time = (items[index] as Json).time;
      if (typeof time === "number" && time > ending.at) at = index;
      else if (typeof time === "number") break;
    }
    items.splice(at, 0, marker);
  }
  const shown = new Set(items.flatMap((item) => (item.kind === "joined" && item.id ? [item.id] : [])));
  for (const message of play.messages ?? []) {
    // One held for the next instruction is shown above the prompt (queuedNext) until that one starts.
    if (message?.state !== "queued" || message.next === true || shown.has(message.id)) continue;
    items.push({ kind: "joined", key: `queued:${message.id}`, id: message.id, text: String(message.text ?? ""), images: [],
      time: null, instruction: Number.isInteger(message.instruction) ? message.instruction : null, step: null,
      state: message.joined === true ? "queued" : "next" });
  }
  (play.pending ?? []).forEach((text, index) => {
    items.push({ kind: "joined", key: `pending:${index}:${text}`, id: null, text, images: [], time: null,
      instruction: null, step: null, state: play.running ? "queued" : "next" });
  });
}

/** The camera a tool looked through, when it names one (vision_ground's camera, mark_point's view). */
export function toolCamera(tool: ToolView): string | null {
  return tool.overlay?.camera ?? (typeof tool.args?.camera === "string" ? tool.args.camera
    : typeof tool.args?.view === "string" ? tool.args.view : null);
}

/** The image a step shows for a camera: that camera's, else the step's first. */
export function stepImage(step: StepView | undefined, camera: string | null): StepImage | null {
  const images = step?.observation?.images ?? [];
  return images.find((image) => image.camera === camera) ?? images[0] ?? null;
}

/** A camera's name as a chip shows it: a long caption-like name by its camera part (`head_camera · grasp baseline`). */
export function cameraLabel(name: string): string {
  if (name.length <= 24) return name;
  const parts = name.split(" · ").map((part) => part.trim()).filter(Boolean);
  const camera = parts.find((part) => /^[A-Za-z0-9_-]*(camera|view|wrist)[A-Za-z0-9_-]*$/i.test(part));
  if (camera) return parts[0] === camera ? camera : `${camera} · ${parts[0].toLowerCase()}`;
  return `${name.slice(0, 23)}…`;
}
