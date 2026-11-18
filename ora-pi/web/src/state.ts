// The page's copy of its run (server/view.ts events, applied in order). A live page applies
// them as they arrive, on its own clock; a replay (lib/replay.ts) applies the recorded ones
// up to its clock, each at the moment it happened (`now`).

import { t } from "./lib/i18n.ts";
import { emptyPerf, withCall, withObservations, withPlay, withRunPerf, type PerfLog } from "./lib/perf.ts";
import { activityFrom, type Activity } from "./lib/play.ts";

type Json = any;

export interface Notice {
  id: string;
  text: string;
  tone: "info" | "warning" | "error";
  at: number;
}

export interface PageState {
  connected: boolean;
  loaded: boolean;
  run: Json;
  state: Json;
  messages: Json[];
  streaming: Json | null;
  tools: Map<string, Json>;
  queue: { steering: string[]; followUp: string[] };
  dialogs: Json[];
  statuses: Record<string, string>;
  widgets: Record<string, string[]>;
  sent: Json[];
  notices: Notice[];
  closed: string | null;
  /** The last instruction free play finished (its `instruction_finished` record). */
  lastInstruction: Json | null;
  draft: { text: string; at: number } | null;
  /** What the agent does now (server/view.ts activity), on this page's clock. */
  activity: Activity | null;
  /** The robot's measured action durations by the step that ran them (free play's action_finished). */
  moves: Map<number, number>;
  /** Bumped when a new motion or operator view may exist (the camera panel asks for frames again). */
  framesVersion: number;
  /** Free play: messages this page sent that the run has not admitted yet (shown queued at once). */
  pending: string[];
  /** Free play: messages this page queued for the next instruction that the run has not admitted yet. */
  pendingNext: string[];
  /** Free play's timings: model calls, observations, actions (lib/perf.ts), on the agent process's clock. */
  perf: PerfLog;
}

export const initialState: PageState = {
  connected: false, loaded: false, run: null, state: {}, messages: [], streaming: null, tools: new Map(),
  queue: { steering: [], followUp: [] }, dialogs: [], statuses: {}, widgets: {}, sent: [], notices: [], closed: null,
  lastInstruction: null, draft: null, activity: null, moves: new Map(), framesVersion: 0, pending: [], pendingNext: [],
  perf: emptyPerf,
};

export type Action =
  /** `next`: queued for the next instruction (Queue), not joining the running one. */
  | { type: "pending"; text: string; next?: boolean }
  | { type: "unpend"; text: string; next?: boolean }
  | { type: "connected"; connected: boolean }
  /** `now`: when it happened (a replay's recorded moment); a live page's own clock otherwise. */
  | { type: "event"; event: Json; now?: number }
  | { type: "notice"; text: string; tone?: Notice["tone"] }
  | { type: "dismiss"; id: string };

let noticeSeq = 0;

function notice(state: PageState, text: string, tone: Notice["tone"] = "info", now = Date.now()): PageState {
  const item = { id: `n${++noticeSeq}`, text, tone, at: now };
  return { ...state, notices: [...state.notices.slice(-5), item] };
}

function identity(message: Json): string {
  return `${message?.role}:${message?.timestamp ?? ""}:${message?.toolCallId ?? message?.customType ?? ""}`;
}

function playNotice(state: PageState, play: Json, at: number, now: number): PageState {
  switch (play?.state) {
    case "instruction_finished": {
      // A marker where it ended (messages.ts placePlay), timed by the server (the agent process's host).
      const run = state.run ? { ...state.run, playEndings: [...(state.run.playEndings ?? []),
        { ...(play.instruction ?? {}), at }].slice(-200) } : state.run;
      return { ...state, run, lastInstruction: play.instruction ?? null };
    }
    case "action_finished": {
      const step = Number(play.action?.observation_step);
      const seconds = Number(play.action?.duration_s);
      const moves = new Map(state.moves);
      if (Number.isInteger(step) && Number.isFinite(seconds)) moves.set(step, seconds);
      return { ...state, moves, framesVersion: state.framesVersion + 1 };
    }
    case "control": {
      if (play.control?.status !== "rejected") return state;
      const control = String(play.control.command ?? "");
      const name = ["stop_instruction", "reset", "quit"].includes(control) ? t(`control.${control}` as never) : control;
      return notice(state, t("notice.controlRefused", { control: name,
        reason: String(play.control.reason ?? "refused").replace(/_/g, " ") }), "warning", now);
    }
    case "scene":
      return notice({ ...state, framesVersion: state.framesVersion + 1 },
        t("notice.sceneLoaded", { scene: play.scene, scenes: play.scenes }), "info", now);
    default:
      return state;
  }
}

/** The list without its first `text` (the same list when it has none). */
function without(list: string[], text: string): string[] {
  const at = list.indexOf(text);
  return at < 0 ? list : list.filter((_, index) => index !== at);
}

export function reduce(state: PageState, action: Action): PageState {
  switch (action.type) {
    case "connected":
      return { ...state, connected: action.connected };
    case "pending":
      return action.next ? { ...state, pendingNext: [...state.pendingNext, action.text] }
        : { ...state, pending: [...state.pending, action.text] };
    case "unpend":
      return action.next ? { ...state, pendingNext: without(state.pendingNext, action.text) }
        : { ...state, pending: without(state.pending, action.text) };
    case "notice":
      return notice(state, action.text, action.tone);
    case "dismiss":
      return { ...state, notices: state.notices.filter((item) => item.id !== action.id) };
    case "event":
      break;
  }
  const event = action.event;
  const now = action.now ?? Date.now();
  switch (event?.type) {
    case "snapshot": {
      const tools = new Map<string, Json>((event.tools ?? []).map((tool: Json) => [tool.id, tool]));
      return {
        ...state, loaded: true, run: event.run, state: event.state ?? {}, messages: event.messages ?? [],
        streaming: event.streaming ?? null, tools, queue: event.queue ?? { steering: [], followUp: [] },
        dialogs: event.dialogs ?? [], statuses: event.statuses ?? {}, widgets: event.widgets ?? {}, sent: event.sent ?? [],
        closed: event.closed ?? null, activity: activityFrom(event.activity, now), framesVersion: state.framesVersion + 1,
        perf: withObservations(withRunPerf(state.perf, event.run?.perf), event.messages ?? []),
      };
    }
    case "run":
      return { ...state, run: event.run, perf: withRunPerf(state.perf, event.run?.perf) };
    case "play": {
      const at = typeof event.at === "number" ? event.at : now;
      const next = { ...state, run: state.run ? { ...state.run, play: event.play } : state.run,
        perf: withPlay(state.perf, event.play, at) };
      return playNotice(next, event.play, at, now);
    }
    case "model_call":
      // A model call's timing (ora-pi/src/play/perf.ts, through server/view.ts).
      return { ...state, perf: withCall(state.perf, event.call) };
    case "play_message": {
      if (!state.run || typeof event.message?.id !== "string") return state;
      const list: Json[] = [...(state.run.playMessages ?? [])];
      const at = list.findIndex((item) => item.id === event.message.id);
      if (at < 0) list.push(event.message);
      else list[at] = event.message;
      // The run admitted a message this page sent: it is shown from the run's own record now.
      const text = String(event.message.text ?? "");
      const queued = event.message.next === true || !state.pending.includes(text);
      return { ...state, run: { ...state.run, playMessages: list },
        ...(queued ? { pendingNext: without(state.pendingNext, text) } : { pending: without(state.pending, text) }) };
    }
    case "cell_command": {
      // A real cell's command outcome, the terminal's or a page's (server/view.ts; components/CellPanel.tsx).
      if (!state.run || typeof event.entry?.id !== "string") return state;
      const list: Json[] = (state.run.cellCommands ?? []).filter((item: Json) => item.id !== event.entry.id);
      return { ...state, run: { ...state.run, cellCommands: [...list, event.entry].slice(-50) } };
    }
    case "activity":
      return { ...state, activity: activityFrom(event.activity, now) };
    case "state":
      return { ...state, state: event.state ?? {} };
    case "messages":
      return { ...state, messages: event.messages ?? [], perf: withObservations(state.perf, event.messages ?? []) };
    case "message": {
      const key = identity(event.message);
      const at = state.messages.findIndex((message) => identity(message) === key);
      const messages = at < 0 ? [...state.messages, event.message]
        : state.messages.map((message, index) => (index === at ? event.message : message));
      // A new observation may come with a new operator view.
      const observation = event.message?.role === "custom" && /observation$/.test(String(event.message.customType ?? ""));
      return { ...state, messages, framesVersion: state.framesVersion + (observation ? 1 : 0),
        perf: observation ? withObservations(state.perf, [event.message]) : state.perf };
    }
    case "streaming":
      return { ...state, streaming: event.message ?? null };
    case "tool": {
      const tools = new Map(state.tools);
      tools.set(event.tool.id, event.tool);
      return { ...state, tools };
    }
    case "queue":
      return { ...state, queue: event.queue };
    case "sent":
      return { ...state, sent: [...state.sent, event.item] };
    case "ui": {
      const request = event.request ?? {};
      switch (request.method) {
        case "select":
        case "confirm":
        case "input":
        case "editor":
          return { ...state, dialogs: [...state.dialogs.filter((item) => item.id !== request.id), request] };
        case "notify": {
          const tone = request.notifyType === "error" ? "error" : request.notifyType === "warning" ? "warning" : "info";
          return notice(state, String(request.message ?? ""), tone);
        }
        case "setStatus": {
          const statuses = { ...state.statuses };
          if (request.statusText) statuses[request.statusKey] = String(request.statusText);
          else delete statuses[request.statusKey];
          return { ...state, statuses };
        }
        case "setWidget": {
          const widgets = { ...state.widgets };
          if (Array.isArray(request.widgetLines)) widgets[request.widgetKey] = request.widgetLines.map(String);
          else delete widgets[request.widgetKey];
          return { ...state, widgets };
        }
        case "set_editor_text":
          return { ...state, draft: { text: String(request.text ?? ""), at: Date.now() } };
        default:
          return state;
      }
    }
    case "ui_done":
      return { ...state, dialogs: state.dialogs.filter((item) => item.id !== event.id) };
    case "closed":
      return { ...state, closed: String(event.reason ?? "the run's agent process closed"), dialogs: [],
        state: { ...state.state, isStreaming: false }, activity: null };
    default:
      return state;
  }
}
