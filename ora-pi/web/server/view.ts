// What every open page of one run sees: the run, its conversation, the reply
// being written, running tools, open dialogs and extension notices, built from
// the run's Pi RPC events and fanned out to the pages (server-sent events).
//
// A page opens with a snapshot and then receives changes:
//   snapshot  {run, state, messages, streaming, tools, queue, dialogs, statuses, widgets, sent, closed, activity}
//   run       ORA's view of the run (ora_get_run): id, mode, refused commands, free play's state,
//             the operator's messages (playMessages) and the instructions that ended (playEndings)
//   play      free play's state, as the Supervisor announced it (ora_play_state), `at` when the
//             agent process received it (its atMs; the record carries no clock)
//   model_call  a model call's timing, display only (ora_model_call: ora-pi/src/play/perf.ts;
//             the run's `perf` holds the record so far, kept current here)
//   play_message  an operator message queued or delivered (ora_play_message)
//   cell_command  a real cell's command outcome, the terminal's or the page's (ora_cell_command;
//             the run's latest are its cellCommands)
//   activity  what the agent does now: thinking, writing, a tool, the robot moving, observing,
//             compacting or idle, with the milliseconds it has lasted (elapsedMs: the page
//             counts on from its own clock, so the server's clock never matters)
//   state     Pi's session state (get_state)
//   messages  the whole conversation again (after a reply, a compaction, a new or forked session)
//   message   one finished message (message_end), images completed by the agent process
//   streaming the assistant reply being written, or null
//   tool      a tool call's progress {id, name, args, status, isError}
//   queue     steers and follow-ups waiting
//   ui        an extension request: a dialog (select, confirm, input, editor) or a notice
//   ui_done   a dialog was answered (by any page) or withdrawn
//   sent      what pages sent, as they sent it (an episode's operator messages are not
//             in Pi's session: ORA delivers them in its own context)
//   closed    the run's agent process went away
//
// Pi 1.3.0's RPC events carry an assistant reply as deltas without the
// partial message; the reply is assembled here.

import { RunRpc } from "./rpc.ts";

type Json = any;

export type Listener = (event: Json) => void;

const DIALOGS = new Set(["select", "confirm", "input", "editor"]);

/** A message's identity across get_messages and message_end (both are copies). */
export function identity(message: Json): string {
  return `${message?.role}:${message?.timestamp ?? ""}:${message?.toolCallId ?? message?.customType ?? ""}`;
}
const STREAM_EVERY_MS = 120;

/** Apply one Pi assistantMessageEvent (as toJsonEvent sends it) to the reply being written. */
export function applyAssistantEvent(message: Json | null, update: Json): Json | null {
  if (!update || typeof update !== "object") return message;
  if (update.type === "done") return update.message ?? message;
  if (update.type === "error") return update.error ?? message;
  const base = message ?? { role: "assistant", content: [] };
  const content = Array.isArray(base.content) ? [...base.content] : [];
  const index = Number(update.contentIndex);
  const at = Number.isInteger(index) && index >= 0 ? index : content.length;
  const block = content[at] ?? null;
  switch (update.type) {
    case "text_start":
      content[at] = { type: "text", text: "" };
      break;
    case "text_delta":
      content[at] = { type: "text", text: `${block?.text ?? ""}${update.delta ?? ""}` };
      break;
    case "text_end":
      content[at] = { type: "text", text: String(update.content ?? block?.text ?? "") };
      break;
    case "thinking_start":
      content[at] = { type: "thinking", thinking: "" };
      break;
    case "thinking_delta":
      content[at] = { type: "thinking", thinking: `${block?.thinking ?? ""}${update.delta ?? ""}` };
      break;
    case "thinking_end":
      content[at] = { type: "thinking", thinking: String(update.content ?? block?.thinking ?? "") };
      break;
    case "toolcall_start":
      content[at] = { type: "toolCall", id: update.id ?? "", name: update.toolName ?? "", arguments: {}, partialArguments: "" };
      break;
    case "toolcall_delta":
      content[at] = { ...(block ?? { type: "toolCall", arguments: {} }),
        partialArguments: `${block?.partialArguments ?? ""}${update.delta ?? ""}` };
      break;
    case "toolcall_end":
      content[at] = update.toolCall ?? block;
      break;
    default:
      return base;
  }
  return { ...base, content };
}

const PERF_KEPT = 200;

/** The run's `perf` (ora-pi/src/play/perf.ts snapshot) with an action's start or end, as it arrived. */
export function perfAction(perf: Json, play: Json, at: number): Json {
  const step = Number(play?.action?.observation_step);
  if (!["action_started", "action_finished"].includes(play?.state) || !Number.isInteger(step)) return perf;
  const base = perf ?? { calls: [], observations: [], actions: [] };
  const actions: Json[] = base.actions ?? [];
  const previous = actions.find((item) => item.step === step)
    ?? { step, actionId: null, startedMs: null, finishedMs: null, durationS: null };
  const duration = Number(play.action.duration_s);
  const next = play.state === "action_started"
    ? { ...previous, actionId: play.action.action_id ?? previous.actionId, startedMs: at }
    : { ...previous, actionId: play.action.action_id ?? previous.actionId, finishedMs: at,
      durationS: Number.isFinite(duration) ? duration : null };
  return { ...base, actions: [...actions.filter((item) => item.step !== step), next].slice(-PERF_KEPT) };
}

/** What the agent is doing now, as the page shows it live ("thinking… 4 s", "moving… 2 s"). */
export interface Activity {
  kind: "thinking" | "writing" | "tool" | "moving" | "observing" | "compacting" | "idle";
  since: number;
  tool?: string;
  step?: number | null;
}

export class RunView {
  run: Json = null;
  activity: Activity = { kind: "idle", since: Date.now() };
  state: Json = {};
  messages: Json[] = [];
  streaming: Json | null = null;
  readonly tools = new Map<string, Json>();
  queue: Json = { steering: [], followUp: [] };
  readonly dialogs = new Map<string, Json>();
  readonly statuses = new Map<string, string>();
  readonly widgets = new Map<string, string[]>();
  readonly sent: Json[] = [];
  closed: string | null = null;
  private readonly listeners = new Set<Listener>();
  private streamTimer: NodeJS.Timeout | null = null;
  private refreshing: Promise<void> | null = null;
  /** Messages that arrived while a get_messages was on its way (it may not hold them yet). */
  private arrived: Json[] | null = null;
  readonly rpc: RunRpc;

  constructor(rpc: RunRpc) {
    this.rpc = rpc;
    rpc.on("event", (event: Json) => this.event(event));
    rpc.on("closed", (reason: string) => {
      this.closed = reason;
      for (const id of this.dialogs.keys()) this.broadcast({ type: "ui_done", id });
      this.dialogs.clear();
      this.broadcast({ type: "closed", reason });
    });
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  broadcast(event: Json): void {
    for (const listener of this.listeners) listener(event);
  }

  snapshot(): Json {
    return {
      type: "snapshot", run: this.run, state: this.state, messages: this.messages, streaming: this.streaming,
      tools: [...this.tools.values()], queue: this.queue, dialogs: [...this.dialogs.values()],
      statuses: Object.fromEntries(this.statuses), widgets: Object.fromEntries(this.widgets), sent: this.sent,
      closed: this.closed, activity: this.activityView(),
    };
  }

  activityView(): Json {
    return { ...this.activity, elapsedMs: Math.max(0, Date.now() - this.activity.since) };
  }

  private setActivity(kind: Activity["kind"], extra: Partial<Activity> = {}, keepSince = false): void {
    const since = keepSince ? this.activity.since : Date.now();
    this.activity = { kind, since, ...extra };
    this.broadcast({ type: "activity", activity: this.activityView() });
  }

  private play(): boolean {
    return this.run?.mode === "play";
  }

  /** Read the run, its state and its conversation again (once at a time; a second call waits for the first). */
  refresh(parts: Array<"run" | "state" | "messages"> = ["run", "state", "messages"]): Promise<void> {
    const previous = this.refreshing ?? Promise.resolve();
    const next = previous.then(async () => {
      if (this.closed) return;
      if (parts.includes("messages")) this.arrived = [];
      const [run, state, messages] = await Promise.all([
        parts.includes("run") ? this.rpc.request("ora_get_run") : null,
        parts.includes("state") ? this.rpc.request("get_state") : null,
        parts.includes("messages") ? this.rpc.request("get_messages", {}, 120_000) : null,
      ]);
      if (run?.success) {
        this.run = run.data;
        this.broadcast({ type: "run", run: this.run });
      }
      if (state?.success) {
        this.state = state.data ?? {};
        this.broadcast({ type: "state", state: this.state });
      }
      if (messages?.success) {
        const listed: Json[] = Array.isArray(messages.data?.messages) ? messages.data.messages : [];
        const known = new Set(listed.map(identity));
        this.messages = [...listed, ...(this.arrived ?? []).filter((message) => !known.has(identity(message)))];
        this.broadcast({ type: "messages", messages: this.messages });
      }
    }).catch(() => undefined).finally(() => {
      this.arrived = null;
    });
    this.refreshing = next.finally(() => {
      if (this.refreshing === next) this.refreshing = null;
    });
    return next;
  }

  private flushStreaming(): void {
    if (this.streamTimer !== null) clearTimeout(this.streamTimer);
    this.streamTimer = null;
    this.broadcast({ type: "streaming", message: this.streaming });
  }

  private streamSoon(): void {
    if (this.streamTimer === null) this.streamTimer = setTimeout(() => this.flushStreaming(), STREAM_EVERY_MS);
  }

  private setState(change: Json): void {
    this.state = { ...this.state, ...change };
    this.broadcast({ type: "state", state: this.state });
  }

  event(event: Json): void {
    switch (event?.type) {
      case "agent_start":
        this.setState({ isStreaming: true });
        break;
      case "turn_start":
        // A new model call; still the same thought when the instruction just started it.
        this.setActivity("thinking", {}, this.activity.kind === "thinking");
        break;
      case "agent_end":
        this.streaming = null;
        this.flushStreaming();
        this.tools.clear();
        this.setState({ isStreaming: false });
        // Free play: the Supervisor executes a submitted action next (action_started), or the
        // instruction ends; an episode's robot acts until its next observation.
        this.setActivity(this.play() ? "observing" : "moving");
        // Completed messages again: earlier steps' images become URLs (rpc/images.ts).
        void this.refresh(["state", "messages"]);
        break;
      case "message_start":
        if (event.message?.role === "assistant") {
          this.streaming = event.message;
          this.flushStreaming();
          this.setActivity("writing", {}, this.activity.kind === "thinking");
        }
        break;
      case "message_update":
        this.streaming = applyAssistantEvent(this.streaming, event.assistantMessageEvent);
        this.streamSoon();
        break;
      case "message_end": {
        const message = event.message;
        if (message?.role === "assistant") {
          this.streaming = null;
          this.flushStreaming();
        }
        if (message && typeof message === "object") {
          this.messages = [...this.messages, message];
          this.arrived?.push(message);
          this.broadcast({ type: "message", message });
        }
        break;
      }
      case "tool_execution_start":
      case "tool_execution_update":
      case "tool_execution_end": {
        const id = String(event.toolCallId ?? "");
        const previous = this.tools.get(id) ?? {};
        const tool = {
          ...previous, id, name: event.toolName ?? previous.name, args: event.args ?? previous.args,
          status: event.type === "tool_execution_end" ? (event.isError ? "error" : "done") : "running",
          isError: Boolean(event.isError),
          // What a running call has shown so far (a code-mode command's robot calls); its result replaces it.
          partial: event.type === "tool_execution_update" ? event.partialResult ?? previous.partial
            : event.type === "tool_execution_end" ? undefined : previous.partial,
        };
        this.tools.set(id, tool);
        this.broadcast({ type: "tool", tool });
        if (event.type === "tool_execution_start") this.setActivity("tool", { tool: String(tool.name ?? "tool") });
        break;
      }
      case "queue_update":
        this.queue = { steering: event.steering ?? [], followUp: event.followUp ?? [] };
        this.broadcast({ type: "queue", queue: this.queue });
        break;
      case "compaction_start":
        this.setState({ isCompacting: true });
        this.setActivity("compacting");
        break;
      case "compaction_end":
        this.setState({ isCompacting: false });
        this.setActivity(this.state.isStreaming ? "thinking" : "idle");
        void this.refresh(["state", "messages"]);
        break;
      case "thinking_level_changed":
        this.setState({ thinkingLevel: event.level });
        break;
      case "session_info_changed":
        this.setState({ sessionName: event.name });
        break;
      case "ora_play_state": {
        const play = event.state ?? {};
        const at = typeof event.atMs === "number" && Number.isFinite(event.atMs) ? event.atMs : Date.now();
        if (this.run) {
          const endings = play.state === "instruction_finished" && play.instruction
            ? [...(this.run.playEndings ?? []), { ...play.instruction, at }].slice(-200)
            : this.run.playEndings;
          this.run = { ...this.run, play, playEndings: endings, perf: perfAction(this.run.perf, play, at) };
        }
        this.broadcast({ type: "play", play, at });
        if (play.state === "action_started") this.setActivity("moving", { step: play.action?.observation_step ?? null });
        else if (play.state === "action_finished") this.setActivity("observing");
        else if (["instruction_finished", "awaiting_instruction", "scene", "ended"].includes(play.state)) this.setActivity("idle");
        else if (play.state === "instruction") this.setActivity("thinking", {}, this.activity.kind === "thinking");
        break;
      }
      case "ora_model_call": {
        const call = event.call;
        if (!call || typeof call.id !== "string") break;
        if (this.run) {
          const perf = this.run.perf ?? { calls: [], observations: [], actions: [] };
          const calls = [...(perf.calls ?? []).filter((item: Json) => item.id !== call.id), call].slice(-PERF_KEPT);
          this.run = { ...this.run, perf: { ...perf, calls } };
        }
        this.broadcast({ type: "model_call", call });
        break;
      }
      case "ora_play_message": {
        const message = event.message;
        if (!message || typeof message.id !== "string") break;
        if (this.run) {
          const list: Json[] = [...(this.run.playMessages ?? [])];
          const at = list.findIndex((item) => item.id === message.id);
          if (at < 0) list.push(message);
          else list[at] = message;
          this.run = { ...this.run, playMessages: list };
        }
        this.broadcast({ type: "play_message", message });
        break;
      }
      case "ora_cell_command": {
        const entry = event.entry;
        if (!entry || typeof entry.id !== "string") break;
        if (this.run) {
          const list: Json[] = (this.run.cellCommands ?? []).filter((item: Json) => item.id !== entry.id);
          this.run = { ...this.run, cellCommands: [...list, entry].slice(-50) };
        }
        this.broadcast({ type: "cell_command", entry });
        break;
      }
      case "extension_ui_request":
        this.uiRequest(event);
        break;
      case "extension_error":
        this.broadcast({ type: "ui", request: { method: "notify", notifyType: "error",
          message: `extension error: ${event.error ?? "unknown"}`, id: `error-${Date.now()}` } });
        break;
      case "error":
        this.broadcast({ type: "ui", request: { method: "notify", notifyType: "error", message: String(event.message ?? ""),
          id: `error-${Date.now()}` } });
        break;
      default:
        break;
    }
  }

  private uiRequest(request: Json): void {
    const method = String(request.method ?? "");
    if (DIALOGS.has(method)) {
      this.dialogs.set(String(request.id), request);
      if (typeof request.timeout === "number" && request.timeout > 0) {
        // The agent process resolves a timed-out dialog itself; pages close it then too.
        setTimeout(() => this.dialogDone(String(request.id)), request.timeout).unref?.();
      }
    } else if (method === "setStatus") {
      if (request.statusText) this.statuses.set(String(request.statusKey), String(request.statusText));
      else this.statuses.delete(String(request.statusKey));
    } else if (method === "setWidget") {
      if (Array.isArray(request.widgetLines)) this.widgets.set(String(request.widgetKey), request.widgetLines.map(String));
      else this.widgets.delete(String(request.widgetKey));
    }
    this.broadcast({ type: "ui", request });
  }

  dialogDone(id: string): void {
    if (this.dialogs.delete(id)) this.broadcast({ type: "ui_done", id });
  }

  /** A page answered a dialog: Pi's extension_ui_response, then every page closes it. */
  answer(response: Json): void {
    const id = String(response.id);
    if (!this.dialogs.has(id)) throw new Error("this dialog was already answered or withdrawn");
    this.rpc.send({ type: "extension_ui_response", ...response, id });
    this.dialogDone(id);
  }

  /** Record what a page sent and what the run answered. */
  recordSent(item: Json): void {
    this.sent.push(item);
    if (this.sent.length > 200) this.sent.splice(0, this.sent.length - 200);
    this.broadcast({ type: "sent", item });
  }
}
