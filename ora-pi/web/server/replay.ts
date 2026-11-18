// A finished run, replayed from its evidence: what `ora web` showed while the run was live,
// rebuilt as the page's own events (server/view.ts), each at the moment it happened, for
// `ora replay`. Read-only: nothing here writes, and no Pi RPC is attached.
//
//   native/pi-session/*.jsonl   Pi's session: instructions, observations, the agent's replies with
//                               their thinking, tool calls and results, joined messages, the model
//                               and reasoning effort; each entry stamped when it was appended. Its
//                               images are listed by URL (/ora/artifacts/<sha256>.png) when the run
//                               stored them (native/agent-artifacts), as `ora web` lists older ones
//   events.jsonl                the Supervisor's record: instructions started and finished, operator
//                               messages (typed and delivered), the robot's actions and receipts,
//                               answers, controls, scenes, operator views and the run's end
//   agent-events.jsonl          the observation step each action was proposed at (action_proposed),
//                               and each model call's request and response (their unix_s and the
//                               response's output tokens: the page's speed panel, lib/perf.ts)
//   native/execution-capture/   the original execution's frames with their capture times, simulated
//                               times and sizes (every frame and camera, at whatever rate it was taken)
//   native/post-action-settle.jsonl  the settling window after each action (its physics steps): the
//                               frames inside it are marked as settling
//   native/operator-view/       the environment's operator view of each observation step
//
// What the evidence does not hold is said in `notes`, never made up: a reply appears whole when it
// was finished (its tokens were not timed), and the page's own notices and dialogs are not replayed.
// `publish` leaves out the run ID and shortens absolute paths in what the page shows.

import { createHash } from "node:crypto";
import { createReadStream, existsSync, readdirSync, readFileSync } from "node:fs";
import { basename, join } from "node:path";
import { createInterface } from "node:readline";

type Json = any;

export const REPLAY_SCHEMA = "ora.web-replay.v1";

/** One page event (server/view.ts vocabulary) at the moment it happened (ms since the epoch). */
export interface ReplayEvent {
  at: number;
  event: Json;
}

export interface CaptureFrame {
  at: number;
  index: number;
  phase: string;
  simTimeS: number;
  /** The physics step it was taken at (since the round's reset), when recorded. */
  simStep: number | null;
  /** Taken in the settling window after its action (native/post-action-settle.jsonl). */
  settle: boolean;
  actionId: string | null;
  /** The observation step its action was proposed at. */
  step: number | null;
  /** Camera -> the frame's URL (/ora/capture/<file>). */
  images: Record<string, string>;
}

/** The settling window after an action: its physics steps, their simulated time, and how it ended. */
export interface Settle {
  actionId: string;
  stepFrom: number;
  steps: number;
  simFrom: number | null;
  simTo: number | null;
  ended: string | null;
}

export interface ViewFrame {
  at: number;
  step: number;
  url: string;
}

export interface ReplayScript {
  schema: typeof REPLAY_SCHEMA;
  runId: string | null;
  mode: "play" | "episode";
  /** The first and last moment the replay shows (ms since the epoch, the run's own clock). */
  start: number;
  end: number;
  events: ReplayEvent[];
  /** `sizes`: each camera's largest frame [width, height] (a sharper operator camera is preferred). */
  capture: { cameras: string[]; fps: number | null; sizes: Record<string, [number, number]>; frames: CaptureFrame[];
    settles: Settle[] } | null;
  operatorView: { camera: string; width: number; height: number; frames: ViewFrame[] } | null;
  /** The robot's actions as they executed (submitted, receipt). */
  actions: Array<{ id: string; step: number | null; start: number; end: number | null }>;
  /** Free play's instructions, for the replay's scrubber. */
  instructions: Array<{ index: number; id: string; start: number; end: number | null; ended: string | null }>;
  /** What the evidence does not hold (shown with the replay, never filled in). */
  notes: string[];
  /** The run-relative files read. */
  sources: string[];
  publish: boolean;
}

export interface ReplayOptions {
  /** Leave out the run ID and shorten absolute paths in what the page shows. */
  publish?: boolean;
}

const OBSERVATIONS = new Set(["ora.observation", "ora.play_observation"]);
const FRAME_FILE = /^frames\/(\d{8}-camera-\d+\.png)$/;
const VIEW_FILE = /^native\/operator-view\/(\d{6,9}\.png)$/;
/** A replay starts on its first picture and holds its last moment a little. */
const TAIL_MS = 3000;

// -- Reading the evidence --------------------------------------------------------------

function readJson(path: string): Json {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
}

/** A JSONL file's records (a torn or foreign line is skipped). */
export function readJsonl(path: string): Json[] {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch {
    return [];
  }
  const records: Json[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      records.push(JSON.parse(line));
    } catch {
      // skipped
    }
  }
  return records;
}

/** A large JSONL file (Pi's session holds every image inline), one record at a time. */
async function eachJsonl(path: string, onRecord: (record: Json) => void): Promise<void> {
  const lines = createInterface({ input: createReadStream(path, { encoding: "utf8" }), crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    let record: Json;
    try {
      record = JSON.parse(line);
    } catch {
      continue;
    }
    onRecord(record);
  }
}

function ms(value: Json): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

/** Unix seconds as whole milliseconds. */
function seconds(value: Json): number | null {
  return typeof value === "number" && Number.isFinite(value) ? Math.round(value * 1000) : null;
}

// -- Images: listed by URL, as `ora web` lists a stored one ------------------------------

/** A caption names a camera as `KIND · sequence N · camera · ...` (ora-pi/src/tui/render.ts cameraOf). */
function cameraOf(caption: string | null): string | null {
  if (!caption) return null;
  const parts = caption.split(" · ").map((part) => part.trim());
  return parts.length >= 3 && /^sequence \d+$/.test(parts[1]) ? parts[2] : caption;
}

class StoredImages {
  private readonly names: Set<string>;
  unstored = 0;

  constructor(runRoot: string) {
    let names: string[] = [];
    try {
      names = readdirSync(join(runRoot, "native", "agent-artifacts"));
    } catch {
      names = [];
    }
    this.names = new Set(names);
  }

  /** The stored file of these image bytes (rpc/images.ts names them by their sha256), or null. */
  find(data: string): string | null {
    const digest = createHash("sha256").update(Buffer.from(data, "base64")).digest("hex");
    return [".png", ".img"].map((suffix) => `${digest}${suffix}`).find((name) => this.names.has(name)) ?? null;
  }
}

// -- Pi's session as the page saw it ------------------------------------------------------

/** Pi's createCustomMessage (pi-coding-agent core/messages.js). */
function customMessage(entry: Json): Json {
  return { role: "custom", customType: entry.customType, content: entry.content ?? [], display: entry.display,
    details: entry.details, timestamp: ms(entry.timestamp) ?? 0 };
}

function observationStep(message: Json): number | null {
  if (message?.role !== "custom" || !OBSERVATIONS.has(message.customType)) return null;
  const step = Number(message.details?.ora_observation_step);
  return Number.isInteger(step) ? step : null;
}

/** An item of the merged record, before it becomes the page's events. */
interface Raw {
  at: number;
  rank: number;
  kind: "entry" | "call_start" | "tool_start" | "event" | "queued" | "model_call";
  data: Json;
}

// -- Publishing ---------------------------------------------------------------------------

const ABSOLUTE_PATH = /(?<![\w.~:/-])\/(?:[\w.@+-]+\/)+[\w.@+-]*/g;

/** Shorten absolute paths (`…/name`) and the run's ID in a text the page shows. */
export function redactText(text: string, runId: string | null): string {
  let value = text.replace(ABSOLUTE_PATH, (path) => `…/${path.split("/").filter(Boolean).pop() ?? ""}`);
  if (runId) value = value.split(runId).join("<run>");
  return value;
}

function redactValue(value: Json, runId: string | null): Json {
  if (typeof value === "string") return redactText(value, runId);
  if (Array.isArray(value)) return value.map((item) => redactValue(item, runId));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, redactValue(item, runId)]));
  }
  return value;
}

/** A message as the page shows it when published: texts, thinking and arguments redacted, image URLs kept. */
function redactMessage(message: Json, runId: string | null): Json {
  if (!message || typeof message !== "object") return message;
  const content = typeof message.content === "string" ? redactText(message.content, runId)
    : Array.isArray(message.content) ? message.content.map((block: Json) => {
      if (block?.type === "text") return { ...block, text: redactText(String(block.text ?? ""), runId) };
      if (block?.type === "thinking") return { ...block, thinking: redactText(String(block.thinking ?? ""), runId) };
      if (block?.type === "toolCall") return { ...block, arguments: redactValue(block.arguments, runId) };
      return block;
    }) : message.content;
  const details = message.details && typeof message.details === "object"
    ? Object.fromEntries(Object.entries(message.details).map(([key, item]) => [key, key === "ora" ? item : redactValue(item, runId)]))
    : message.details;
  const { errorMessage } = message;
  return { ...message, content, details,
    ...(typeof errorMessage === "string" ? { errorMessage: redactText(errorMessage, runId) } : {}) };
}

function redactEvent(event: Json, runId: string | null): Json {
  switch (event?.type) {
    case "message":
      return { ...event, message: redactMessage(event.message, runId) };
    case "messages":
      return { ...event, messages: event.messages.map((message: Json) => redactMessage(message, runId)) };
    case "tool":
      return { ...event, tool: { ...event.tool, args: redactValue(event.tool.args, runId) } };
    case "play":
    case "play_message":
    case "sent":
      return redactValue(event, runId);
    case "snapshot":
      return { ...event, run: { ...event.run, runId: null } };
    default:
      return event;
  }
}

// -- The replay ---------------------------------------------------------------------------

function sessionFiles(runRoot: string): string[] {
  const directory = join(runRoot, "native", "pi-session");
  try {
    return readdirSync(directory).filter((name) => name.endsWith(".jsonl")).sort().map((name) => join(directory, name));
  } catch {
    return [];
  }
}

function runMode(runRoot: string): "play" | "episode" {
  const plan = readJson(join(runRoot, "plan.json"));
  return plan?.episode_mode === "play" ? "play" : "episode";
}

/** The replay of the finished run at `runRoot`. */
export async function buildReplay(runRoot: string, options: ReplayOptions = {}): Promise<ReplayScript> {
  const runId = basename(runRoot);
  const mode = runMode(runRoot);
  const notes: string[] = [];
  const sources: string[] = [];
  const images = new StoredImages(runRoot);

  // The Supervisor's record and the agent's.
  const events = readJsonl(join(runRoot, "events.jsonl"));
  if (events.length) sources.push("events.jsonl");
  const actionSteps = new Map<string, number>();
  // Each model call as the page's model_call event (lib/perf.ts): no first-token time is recorded.
  const requested = new Map<string, Json>();
  const modelCalls: Json[] = [];
  for (const record of readJsonl(join(runRoot, "agent-events.jsonl"))) {
    const callId = typeof record?.model_call_id === "string" ? record.model_call_id : null;
    if (record?.type === "provider_request" && callId !== null) requested.set(callId, record);
    if (record?.type === "provider_response" && callId !== null && requested.has(callId)) {
      const startedMs = seconds(requested.get(callId).unix_s);
      const endedMs = seconds(record.unix_s);
      const usage = record.payload?.usage ?? {};
      if (startedMs !== null && endedMs !== null) {
        modelCalls.push({ id: callId, step: Number.isInteger(record.observation_step) ? record.observation_step : null,
          startedMs, firstTokenMs: null, endedMs, outputTokens: Number(usage.output_tokens) || 0,
          inputTokens: Number(usage.input_tokens) || 0 });
      }
      requested.delete(callId);
    }
    if (record?.type !== "action_proposed") continue;
    const id = record.payload?.action_id;
    if (typeof id === "string" && Number.isInteger(record.observation_step)) actionSteps.set(id, record.observation_step);
  }
  if (actionSteps.size || modelCalls.length) sources.push("agent-events.jsonl");
  const stepOf = (actionId: string | null): number | null => (actionId === null ? null
    : actionSteps.get(actionId) ?? actionSteps.get(actionId.split(":")[0]) ?? null);

  const raws: Raw[] = [];
  let order = 0;
  const add = (at: number | null, kind: Raw["kind"], data: Json) => {
    if (at === null) return;
    raws.push({ at, rank: order++, kind, data });
  };

  // Pi's session: its entries, and when each model call and tool call began.
  const files = sessionFiles(runRoot);
  if (!files.length) notes.push("The run has no Pi session (native/pi-session): the conversation cannot be shown.");
  const calls = new Map<string, Json>();
  let lastAssistantAt = 0;
  const storedOnly = (content: Json): Json => (Array.isArray(content) && content.some((block: Json) => block?.type === "image")
    ? content.map((block: Json) => (block?.type === "image"
      ? { type: "image", mimeType: block.mimeType, stored: typeof block.data === "string" ? images.find(block.data) : null }
      : block))
    : content);
  for (const file of files) {
    sources.push(`native/pi-session/${basename(file)}`);
    await eachJsonl(file, (entry) => {
      const at = ms(entry?.timestamp);
      if (entry?.type === "message" && entry.message && typeof entry.message === "object") {
        const message = entry.message;
        if (message.role === "assistant") {
          const started = ms(message.timestamp);
          if (started !== null && at !== null && started <= at) add(started, "call_start", {});
          for (const block of Array.isArray(message.content) ? message.content : []) {
            if (block?.type === "toolCall" && block.id) calls.set(String(block.id), block);
          }
          lastAssistantAt = at ?? lastAssistantAt;
        } else if (message.role === "toolResult" && at !== null) {
          const took = Number(message.durationMs);
          const begun = Number.isFinite(took) && took > 0 ? Math.max(lastAssistantAt, at - took) : at;
          const call = calls.get(String(message.toolCallId ?? ""));
          add(begun, "tool_start", { id: String(message.toolCallId ?? ""), name: message.toolName ?? call?.name ?? "tool",
            args: call?.arguments ?? {} });
        }
      }
      // An image is kept as the name of its stored file: the bytes do not stay in memory.
      if (entry?.type === "message" && entry.message && typeof entry.message === "object") {
        entry = { ...entry, message: { ...entry.message, content: storedOnly(entry.message.content) } };
      } else if (entry?.type === "custom_message") {
        entry = { ...entry, content: storedOnly(entry.content) };
      }
      add(at, "entry", entry);
    });
  }

  for (const call of modelCalls) add(call.endedMs, "model_call", call);

  // The Supervisor's record, and messages typed while an instruction ran (queued until delivered).
  const operatorMessages = new Map<string, Json>();
  const instructionIndex = new Map<string, number>();
  for (const record of events) {
    if (record?.type === "operator_message" && record.payload?.id) operatorMessages.set(String(record.payload.id), record.payload);
    if (record?.type === "play_instruction_started" && record.record?.instruction_id) {
      instructionIndex.set(String(record.record.instruction_id), Number(record.record.index));
    }
  }
  for (const record of events) {
    const at = seconds(record?.time);
    add(at, "event", record);
    if (record?.type === "operator_message" && record.payload?.joined === true) {
      const typed = Number(record.payload.submitted_unix_ns);
      add(Number.isFinite(typed) && typed > 0 ? Math.min(Math.round(typed / 1e6), at ?? Infinity) : null, "queued", record.payload);
    }
  }

  raws.sort((a, b) => a.at - b.at || a.rank - b.rank);

  // -- The page's events, in order -------------------------------------------------------
  const out: ReplayEvent[] = [];
  const emit = (at: number, event: Json) => out.push({ at, event });
  const play = mode === "play";
  let scene = 1;
  let scenes = 1;
  let started = 0;
  let running: string | null = null;
  const playBase = () => ({ schema: "ora.play-state.v1", scene, scenes, instructions: started, running });
  const playEvent = (at: number, state: string, fields: Json = {}) => {
    const record = { ...playBase(), state, ...fields };
    emit(at, { type: "play", play: record, at });
  };
  const playMessages = new Map<string, Json>();
  const playMessage = (at: number, id: string, change: Json) => {
    const previous = playMessages.get(id);
    const next = { id, text: "", state: "queued", joined: null, instruction: null, step: null, at, deliveredAt: null,
      ...previous, ...change };
    if (previous?.state === "delivered") next.state = "delivered";
    playMessages.set(id, next);
    emit(at, { type: "play_message", message: next });
  };
  const activity = (at: number, kind: string, extra: Json = {}) => emit(at, { type: "activity",
    activity: { kind, elapsedMs: 0, ...extra } });

  // The session's messages as Pi's context: a linear append is one message; a branch, a compaction
  // or a new session gives the whole list again (Pi's buildSessionContext, as `get_messages`).
  const byId = new Map<string, Json>();
  const completed = new Map<string, Json>();
  let leaf: string | null = null;
  let current: number | null = null;
  const pageMessage = (entry: Json): Json | null => {
    if (entry.id && completed.has(entry.id)) return completed.get(entry.id);
    let message: Json = null;
    if (entry.type === "message") message = entry.message;
    else if (entry.type === "custom_message") message = customMessage(entry);
    else if (entry.type === "compaction") {
      message = { role: "compactionSummary", summary: entry.summary, tokensBefore: entry.tokensBefore, timestamp: ms(entry.timestamp) };
    } else if (entry.type === "branch_summary" && entry.summary) {
      message = { role: "branchSummary", summary: entry.summary, fromId: entry.fromId, timestamp: ms(entry.timestamp) };
    }
    if (message === null) return null;
    const step = observationStep(message);
    if (step !== null) current = step;
    if (message.role === "custom" && OBSERVATIONS.has(message.customType)) message = completeStored(message, step, null);
    else if (message.role === "toolResult") message = completeStored(message, current, String(message.toolCallId ?? ""));
    if (entry.id) completed.set(entry.id, message);
    return message;
  };
  const contextOf = (id: string | null): Json[] => {
    const path: Json[] = [];
    for (let entry = id ? byId.get(id) : undefined; entry; entry = entry.parentId ? byId.get(entry.parentId) : undefined) {
      path.push(entry);
    }
    path.reverse();
    let from = 0;
    const compaction = path.map((entry) => entry.type).lastIndexOf("compaction");
    let kept: Json[] = path;
    if (compaction >= 0) {
      const first = path.findIndex((entry) => entry.id === path[compaction].firstKeptEntryId);
      from = first >= 0 && first < compaction ? first : compaction;
      kept = [path[compaction], ...path.slice(from, compaction).filter((entry) => !(entry.type === "message"
        && entry.message?.role === "system")), ...path.slice(compaction + 1)];
    }
    return kept.flatMap((entry) => {
      const message = pageMessage(entry);
      return message === null ? [] : [message];
    });
  };

  // Images stored by the run: listed by URL (the bytes were left out while reading).
  function completeStored(message: Json, step: number | null, toolCallId: string | null): Json {
    const content = Array.isArray(message.content) ? message.content : typeof message.content === "string"
      ? [{ type: "text", text: message.content }] : [];
    const listed: Json[] = [];
    let caption: string | null = null;
    for (const block of content) {
      if (block?.type === "text") {
        const text = String(block.text ?? "");
        caption = text.length <= 200 && !text.trimStart().startsWith("{") ? text.split("\n", 1)[0].trim() || null : null;
      } else if (block?.type === "image") {
        const name = typeof block.stored === "string" ? block.stored : null;
        if (name === null) images.unstored += 1;
        listed.push({ camera: cameraOf(caption), artifact: name === null ? null : `native/agent-artifacts/${name}`,
          mimeType: String(block.mimeType ?? "image/png"), inline: false, url: name === null ? null : `/ora/artifacts/${name}` });
        caption = null;
      }
    }
    const ora = { step, ...(toolCallId === null ? {} : { toolCallId }), camera: listed[0]?.camera ?? null,
      artifact: listed[0]?.artifact ?? null, images: listed };
    return { ...message, content: content.filter((block: Json) => block?.type !== "image"),
      details: { ...(message.details ?? {}), ora } };
  }

  const state: Json = { isStreaming: false };
  let first: number | null = null;
  const seen = (at: number) => {
    if (first === null || at < first) first = at;
  };
  let last = 0;
  const actions = new Map<string, { id: string; step: number | null; start: number; end: number | null }>();
  const instructions: ReplayScript["instructions"] = [];
  let termination: string | null = null;

  for (const raw of raws) {
    const at = raw.at;
    last = Math.max(last, at);
    if (raw.kind === "call_start") {
      activity(at, "thinking");
      if (!play) emit(at, { type: "state", state: { ...state, isStreaming: (state.isStreaming = true) } });
      continue;
    }
    if (raw.kind === "model_call") {
      emit(at, { type: "model_call", call: raw.data });
      continue;
    }
    if (raw.kind === "tool_start") {
      emit(at, { type: "tool", tool: { id: raw.data.id, name: raw.data.name, args: raw.data.args, status: "running", isError: false } });
      activity(at, "tool", { tool: String(raw.data.name) });
      continue;
    }
    if (raw.kind === "queued") {
      const payload = raw.data;
      const n = instructionIndex.has(String(payload.instruction_id)) ? instructionIndex.get(String(payload.instruction_id))! + 1 : null;
      if (!play) continue;
      playEvent(at, "message_queued", { message: { id: payload.id, text: String(payload.message ?? ""), instruction: n } });
      playMessage(at, String(payload.id), { text: String(payload.message ?? ""), joined: true, instruction: n });
      continue;
    }
    if (raw.kind === "entry") {
      const entry = raw.data;
      if (entry?.type === "session") {
        if (byId.size) emit(at, { type: "messages", messages: [] });
        byId.clear();
        leaf = null;
        continue;
      }
      // Every entry is a node of the session's tree: one whose parent is not the last is a branch.
      const branched = Boolean(entry?.id) && (entry.parentId ?? null) !== leaf;
      if (entry?.id) {
        byId.set(entry.id, entry);
        leaf = entry.id;
      }
      // The conversation along the new branch, as Pi's get_messages gives it then.
      if (branched || entry?.type === "compaction" || entry?.type === "branch_summary") {
        emit(at, { type: "messages", messages: contextOf(entry.id) });
      }
      if (entry?.type === "model_change") {
        state.model = { provider: entry.provider, id: entry.modelId };
        emit(at, { type: "state", state: { ...state } });
        continue;
      }
      if (entry?.type === "thinking_level_change") {
        state.thinkingLevel = entry.thinkingLevel;
        emit(at, { type: "state", state: { ...state } });
        continue;
      }
      if (entry?.type === "custom") {
        if (entry.customType === "ora.scene" && play) {
          scene = Number(entry.data?.scene) || scene;
          scenes = Number(entry.data?.scenes) || scenes;
          playEvent(at, "scene", { cause: entry.data?.cause ?? null });
          seen(at);
        }
        continue;
      }
      if (!entry?.id || branched || entry.type === "compaction" || entry.type === "branch_summary") continue;
      const message = pageMessage(entry);
      if (message === null || message.role === "system") continue;
      emit(at, { type: "message", message });
      seen(at);
      if (message.role === "toolResult") {
        emit(at, { type: "tool", tool: { id: String(message.toolCallId ?? ""), name: message.toolName,
          args: calls.get(String(message.toolCallId ?? ""))?.arguments ?? {}, status: message.isError ? "error" : "done",
          isError: Boolean(message.isError) } });
      }
      if (message.role === "assistant" && !play) {
        emit(at, { type: "state", state: { ...state, isStreaming: (state.isStreaming = false) } });
      }
      continue;
    }
    // The Supervisor's record.
    const record = raw.data;
    switch (record?.type) {
      case "play_instruction_started": {
        const index = Number(record.record?.index);
        const id = String(record.record?.instruction_id ?? "");
        started = Number.isInteger(index) ? index + 1 : started + 1;
        running = id;
        const text = String(operatorMessages.get(id)?.message ?? "");
        instructions.push({ index: started - 1, id, start: at, end: null, ended: null });
        if (play) {
          playEvent(at, "instruction", { instruction_id: id, message: { id, text, instruction: started } });
          playMessage(at, id, { text, joined: false, instruction: started });
        }
        activity(at, "thinking");
        seen(at);
        break;
      }
      case "operator_message": {
        const payload = record.payload ?? {};
        const id = String(payload.id ?? "");
        const n = instructionIndex.has(String(payload.instruction_id)) ? instructionIndex.get(String(payload.instruction_id))! + 1 : null;
        if (!play) {
          // An episode's operator messages live in ORA's context: the page showed them as sent.
          emit(at, { type: "sent", item: { text: String(payload.message ?? ""), as: "steer", at, ok: true, disposition: null, error: null } });
          break;
        }
        const step = Number.isInteger(payload.observation_step) ? payload.observation_step : null;
        playEvent(at, "message_delivered", { message: { id, text: String(payload.message ?? ""), instruction: n,
          joined: Boolean(payload.joined), observation_step: step } });
        playMessage(at, id, { text: String(payload.message ?? ""), joined: Boolean(payload.joined), instruction: n,
          state: "delivered", step, deliveredAt: at });
        break;
      }
      case "action_submitted": {
        const id = String(record.action?.action_id ?? "");
        const step = stepOf(id);
        actions.set(id, { id, step, start: at, end: null });
        if (play) playEvent(at, "action_started", { action: { action_id: id, observation_step: step } });
        activity(at, "moving", { step });
        break;
      }
      case "motion_receipt": {
        const id = String(record.record?.receipt?.action_id ?? "");
        const action = actions.get(id);
        if (action) action.end = at;
        const step = stepOf(id);
        if (play) {
          playEvent(at, "action_finished", { action: { action_id: id, observation_step: step,
            duration_s: Number(record.duration_s) } });
        }
        activity(at, "observing");
        break;
      }
      case "play_instruction_finished": {
        const instruction = record.record ?? {};
        running = null;
        const item = instructions.find((candidate) => candidate.id === instruction.instruction_id && candidate.end === null);
        if (item) {
          item.end = at;
          item.ended = String(instruction.ended ?? "");
        }
        if (play) {
          playEvent(at, "instruction_finished", { instruction });
          playEvent(at, "awaiting_instruction");
        }
        activity(at, "idle");
        break;
      }
      case "play_answer":
        if (play) playEvent(at, "answer", { answer: record.record });
        break;
      case "play_control":
        if (play) playEvent(at, "control", { control: record.record });
        break;
      case "play_scene_reset":
        scene = Number(record.record?.scene_index ?? scene - 1) + 1;
        break;
      case "play_scene_ended":
        if (play) playEvent(at, "scene_ended", { reason: record.reason ?? null });
        break;
      case "run_finished": {
        termination = String(record.result?.termination ?? readJson(join(runRoot, "status.json"))?.termination ?? "")
          || null;
        if (play) playEvent(at, "ended", { termination });
        activity(at, "idle");
        break;
      }
      default:
        break;
    }
  }

  // The robot's recorded motion and the operator view.
  const manifest = readJson(join(runRoot, "native", "execution-capture", "manifest.json"));
  const frames: CaptureFrame[] = [];
  const sizes: Record<string, [number, number]> = {};
  for (const record of readJsonl(join(runRoot, "native", "execution-capture", "timeline.jsonl"))) {
    const at = seconds(record?.captured_unix_s);
    const files: Record<string, string> = {};
    for (const [camera, image] of Object.entries((record?.images ?? {}) as Record<string, Json>)) {
      const match = FRAME_FILE.exec(String(image?.path ?? ""));
      if (!match) continue;
      files[camera] = `/ora/capture/${match[1]}`;
      const [height, width] = Array.isArray(image?.shape) ? image.shape.map(Number) : [NaN, NaN];
      if (Number.isFinite(width) && Number.isFinite(height) && width * height > (sizes[camera]?.[0] ?? 0) * (sizes[camera]?.[1] ?? 0)) {
        sizes[camera] = [width, height];
      }
    }
    if (at === null || !Object.keys(files).length) continue;
    const actionId = typeof record.action_id === "string" ? record.action_id : null;
    frames.push({ at, index: Number(record.frame_index), phase: String(record.phase ?? ""), simTimeS: Number(record.sim_time_s),
      simStep: Number.isInteger(record.sim_step) ? record.sim_step : null, settle: false, actionId, step: stepOf(actionId),
      images: files });
  }
  frames.sort((a, b) => a.at - b.at || a.index - b.index);
  if (frames.length) sources.push("native/execution-capture/timeline.jsonl");
  // The settling window after each action, by physics step: the frames inside it are the scene settling.
  const stepped = frames.find((frame) => frame.simStep !== null && frame.simStep > 0);
  const dt = stepped ? stepped.simTimeS / stepped.simStep! : null;
  const settles: Settle[] = [];
  for (const record of readJsonl(join(runRoot, "native", "post-action-settle.jsonl"))) {
    const stepFrom = Number(record?.physics_step_start);
    const steps = Number(record?.physics_steps);
    if (typeof record?.action_id !== "string" || !Number.isInteger(stepFrom) || !Number.isInteger(steps)) continue;
    settles.push({ actionId: record.action_id, stepFrom, steps, simFrom: dt === null ? null : stepFrom * dt,
      simTo: dt === null ? null : (stepFrom + steps) * dt, ended: typeof record.ended === "string" ? record.ended : null });
    for (const frame of frames) {
      if (steps > 0 && frame.actionId?.split(":")[0] === record.action_id && frame.simStep !== null
        && frame.simStep > stepFrom && frame.simStep <= stepFrom + steps) frame.settle = true;
    }
  }
  if (settles.length) sources.push("native/post-action-settle.jsonl");
  const fps = Number(manifest?.config?.fps);
  const listedCameras = Array.isArray(manifest?.camera_panels) ? manifest.camera_panels.map(String) : null;
  const allCameras = [...new Set(frames.flatMap((frame) => Object.keys(frame.images)))];
  const capture = frames.length ? { cameras: [...new Set([...(listedCameras ?? []), ...allCameras])],
    fps: Number.isFinite(fps) && fps > 0 ? fps : null, sizes, frames, settles } : null;
  if (!capture) notes.push("No execution capture: the cameras show the operator view and what the model saw, not the robot moving.");

  const views: ViewFrame[] = [];
  let viewCamera: Json = null;
  for (const record of events) {
    if (record?.type !== "operator_view_frame" || !Number.isInteger(record.observation_step)) continue;
    const match = VIEW_FILE.exec(String(record.path ?? ""));
    const at = seconds(record.time);
    if (!match || at === null) continue;
    views.push({ at, step: record.observation_step, url: `/ora/operator-view/${match[1]}` });
    viewCamera ??= { camera: String(record.camera ?? "operator view"), width: Number(record.width), height: Number(record.height) };
  }
  views.sort((a, b) => a.at - b.at);
  const operatorView = views.length ? { ...viewCamera, frames: views } : null;
  if (views.length) sources.push("native/operator-view/");
  // The scene's first picture shows before anything else does.
  for (const time of [views[0]?.at, frames[0]?.at]) {
    if (time !== undefined) first = first === null ? time : Math.min(first, time);
  }

  if (images.unstored) notes.push(`${images.unstored} image(s) of the conversation were not stored by the run and are not shown.`);
  if (!play) notes.push("An episode's observation images are in ORA's own history, not in Pi's session: observations show no images.");
  notes.push("A reply appears whole when the model finished it: the evidence does not time its tokens as they streamed.");

  const start = first ?? (raws[0]?.at ?? 0);
  const ended = events.find((record) => record?.type === "run_finished");
  const end = Math.max(seconds(ended?.time) ?? last, frames.at(-1)?.at ?? 0, views.at(-1)?.at ?? 0) + TAIL_MS;

  const snapshot = {
    type: "snapshot",
    run: { runId, mode, role: "observer", clients: 0, refused: {}, play: null, playMessages: [], playEndings: [], replay: true },
    state: {}, messages: [], streaming: null, tools: [], queue: { steering: [], followUp: [] }, dialogs: [], statuses: {},
    widgets: {}, sent: [], closed: null, activity: { kind: "idle", elapsedMs: 0 },
  };
  let ordered: ReplayEvent[] = [{ at: Math.min(start, raws[0]?.at ?? start), event: snapshot }, ...out];
  const publish = options.publish === true;
  if (publish) ordered = ordered.map((item) => ({ at: item.at, event: redactEvent(item.event, runId) }));
  return {
    schema: REPLAY_SCHEMA, runId: publish ? null : runId, mode, start, end, events: ordered, capture, operatorView,
    actions: [...actions.values()], instructions, notes, sources, publish,
  };
}

/** Whether `runRoot` looks like a run this module can replay (Pi's session or the Supervisor's record). */
export function replayable(runRoot: string): boolean {
  return sessionFiles(runRoot).length > 0 || existsSync(join(runRoot, "events.jsonl"));
}
