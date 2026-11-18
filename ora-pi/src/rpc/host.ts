// Pi's RPC protocol for one run, served by the agent core on a unix socket.
//
// Pi's own `runRpcMode` owns the process's stdin and stdout, which in ORA
// belong to the Supervisor's channel; this is a port of its dispatcher
// (dist/modes/rpc/rpc-mode.js, Pi 1.3.0) onto a socket, so several standard
// Pi RPC clients can attach to a live run: `ora-rpc-attach` (rpc/attach.ts)
// stands in for `pi --mode rpc` and relays a client's stdio here.
//
//   endpoint   <run>/native/pi-rpc.json: {schema, socket, pid, protocol, pi_version}
//   socket     0600, outside the run directory (sealed evidence holds no socket)
//   framing    LF-delimited JSON (rpc/jsonl.ts)
//   clients    one controller, the rest observe. ora-rpc-attach opens with
//              {"type":"ora_attach","role":"controller"|"observer"} (no response):
//              a new controller takes over, as a new `ora attach` does. A client
//              without it controls when no one does. When the controller leaves, the
//              oldest observer takes over; `ora_take_control` claims control.
//   commands   Pi's, minus those an episode refuses (rpc/policy.ts); observers read only
//   events     every session event to every client (Pi's toJsonEvent), with ORA's
//              camera images on observation and tool-result messages (rpc/images.ts)
//   UI         notifications to all, dialogs to the controller (rpc/ui.ts)
//   ORA        `ora_get_run` (the run, its mode, the commands it refuses, free play's
//              state and the operator's messages), `ora_get_frames` (the recorded motion
//              of a step and the operator view: rpc/frames.ts), `ora_play_control` (free
//              play: stop the running instruction, the next scene, quit), `ora_play_queue`
//              (free play: {message}, a queued instruction: held until the running instruction
//              ends, then the next one; a Pi follow-up is one only while the agent streams), the
//              `ora_play_state` event (the Supervisor's announcements, display only, with
//              `atMs`, when the core received it), the `ora_model_call` event (a model call's
//              timing, display only: play/perf.ts; `ora_get_run` lists them so far as `perf`)
//              and the `ora_play_message` event (an operator message queued or delivered:
//              play/messages.ts); on a real cell `ora_cell_note` (a cell command `ora web`'s
//              page ran, from its gateway: tui/cell_commands.ts note) and the
//              `ora_cell_command` event (every cell command's outcome, the terminal's and the
//              page's; `ora_get_run` lists the latest as cellCommands); standard clients ignore them
//
// Prompts and steers pass Pi's input hook as typed input does in the TUI: in
// an episode ORA turns them into operator messages the Supervisor admits. An
// episode has one kind of operator message, delivered before the next model
// call, so a follow-up (what Pi clients queue while the agent works) is
// delivered as a steer; free play keeps Pi's follow-ups.

import { createHash } from "node:crypto";
import { renameSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type Server, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { AgentSessionRuntime } from "@earendil-works/pi-coding-agent";

import type { PlayMessage } from "../play/messages.ts";
import { RunFrames } from "./frames.ts";
import { ImageCompletion, type ImageSource } from "./images.ts";
import { attachJsonlLineReader, serializeJsonLine } from "./jsonl.ts";
import { refusal, refusals, type RunMode } from "./policy.ts";
import { RpcUi } from "./ui.ts";

type Json = any;

export const RPC_ENDPOINT_SCHEMA = "ora.pi-rpc-endpoint.v1";
/** Free play's operator controls a client may send (`ora_play_control`), as the terminal names them. */
export const PLAY_CONTROLS: Record<string, string> = { stop_instruction: "Esc", reset: "/reset", quit: "/quit" };
export const PI_RPC_VERSION = "1.3.0";
/** Longest unix socket path bound in place (macOS 104 bytes with the NUL, Linux 108). */
const MAX_SOCKET_PATH_BYTES = 103;

export interface RpcHostOptions {
  runtime: AgentSessionRuntime;
  runRoot: string;
  runId: string;
  mode: () => RunMode;
  /** ORA's images for clients; null serves Pi's messages as they are. */
  images: ImageSource | null;
  /**
   * Bind this host's uiContext to the session. False when the attached Pi
   * TUI binds it: its InteractiveMode builds its uiContext through
   * `composeUi`, so RPC clients and the terminal share one composite.
   */
  bindUi: boolean;
  inlineSteps?: number;
  /** Free play's state as the Supervisor last announced it (episode_play.py), or null. */
  play?: () => Json | null;
  /** Free play's operator controls, as the terminal sends them (core.submitOperatorMessage). */
  control?: (type: string, message: string) => Promise<Json>;
  /** Free play's queued instruction (play/runtime.ts submitPlayMessage, queued), as the terminal's Alt+Enter. */
  queue?: (message: string) => Promise<Json>;
  /** Free play's operator messages, queued and delivered, and the instructions that ended (play/messages.ts). */
  playMessages?: () => PlayMessage[];
  playEndings?: () => Json[];
  /** A real cell's command the page ran (tui/cell_commands.ts note); throws on a run without cell commands. */
  cellNote?: (note: Json) => Json;
  /** A real cell's latest command outcomes, the terminal's and the page's. */
  cellCommands?: () => Json[];
  /** Free play's timings so far (play/perf.ts: model calls, observations, actions), or null. */
  perf?: () => Json | null;
  /** Where the socket goes (tests); default: the temporary directory. */
  socketDir?: string;
}

/** Pi's toJsonEvent: assistant deltas without the accumulated partial message. */
export function toJsonEvent(event: Json): Json {
  if (event?.type !== "message_update") return event;
  const delta = event.assistantMessageEvent;
  let assistantMessageEvent = delta;
  if (delta?.type === "toolcall_start") {
    const call = delta.partial?.content?.[delta.contentIndex];
    const { partial: _partial, ...rest } = delta;
    assistantMessageEvent = { ...rest, id: call?.id, toolName: call?.name };
  } else if (delta && "partial" in delta) {
    const { partial: _partial, ...rest } = delta;
    assistantMessageEvent = rest;
  }
  return { type: "message_update", usage: event.message?.usage, assistantMessageEvent };
}

class Client {
  closed = false;
  readonly id: number;
  readonly socket: Socket;

  constructor(id: number, socket: Socket) {
    this.id = id;
    this.socket = socket;
  }

  send(message: Json): void {
    if (this.closed || this.socket.destroyed) return;
    this.socket.write(serializeJsonLine(message));
  }
}

export class RpcHost {
  readonly socketPath: string;
  private readonly clients = new Map<number, Client>();
  private controllerId: number | null = null;
  private nextId = 1;
  private server: Server | null = null;
  private unsubscribe: (() => void) | null = null;
  private readonly ui: RpcUi;
  private readonly images: ImageCompletion | null;
  private ended = false;
  private readonly options: RpcHostOptions;
  private readonly frames: RunFrames;

  private constructor(options: RpcHostOptions) {
    this.options = options;
    const uid = typeof process.getuid === "function" ? process.getuid() : 0;
    const digest = createHash("sha256").update(options.runRoot).digest("hex").slice(0, 12);
    const name = `ora-pi-rpc-${uid}-${digest}-${process.pid}.sock`;
    const preferred = join(options.socketDir ?? tmpdir(), name);
    // A deep TMPDIR leaves no room for the name: /tmp (the socket itself is 0600; the
    // file-permission sandbox does not cover listening, but would refuse a new directory).
    this.socketPath = Buffer.byteLength(preferred) <= MAX_SOCKET_PATH_BYTES ? preferred : join("/tmp", name);
    this.ui = new RpcUi(() => this.clients.values(), () => this.controller());
    this.images = options.images ? new ImageCompletion(options.images, options.inlineSteps) : null;
    this.frames = new RunFrames(options.runRoot);
  }

  static async start(options: RpcHostOptions): Promise<RpcHost> {
    return RpcHost.create(options).open();
  }

  /** A host not yet serving: the TUI host can compose its uiContext with it first. */
  static create(options: RpcHostOptions): RpcHost {
    return new RpcHost(options);
  }

  async open(): Promise<RpcHost> {
    await this.bind();
    await this.listen();
    return this;
  }

  /** The composite uiContext over the TUI's own (tui/host.ts). */
  composeUi(base: Json): Json {
    return this.ui.context(base);
  }

  get session(): Json {
    return this.options.runtime.session;
  }

  private controller(): Client | null {
    return this.controllerId === null ? null : this.clients.get(this.controllerId) ?? null;
  }

  private async bind(): Promise<void> {
    if (this.options.bindUi) {
      await this.session.bindExtensions({
        uiContext: this.ui.context(),
        mode: "rpc",
        commandContextActions: {
          waitForIdle: () => this.session.waitForIdle(),
          newSession: async () => ({ cancelled: true }),
          fork: async () => ({ cancelled: true }),
          navigateTree: async () => ({ cancelled: true }),
          switchSession: async () => ({ cancelled: true }),
          reload: async () => undefined,
        },
        shutdownHandler: () => undefined,
        onError: (error: Json) => this.broadcast({
          type: "extension_error", extensionPath: error.extensionPath, event: error.event, error: error.error,
        }),
      });
    }
    this.unsubscribe = this.session.subscribe((event: Json) => {
      const json = toJsonEvent(event);
      this.broadcast(this.images ? this.images.event(json) : json);
    });
  }

  private async listen(): Promise<void> {
    const server = createServer((socket) => this.connection(socket));
    const umask = process.umask(0o077);
    try {
      await new Promise<void>((resolve, reject) => {
        server.once("error", reject);
        server.listen(this.socketPath, () => resolve());
      });
    } finally {
      process.umask(umask);
    }
    server.on("error", (error) => process.stderr.write(`ora-pi rpc: ${error.stack}\n`));
    this.server = server;
    const pointer = join(this.options.runRoot, "native", "pi-rpc.json");
    writeFileSync(`${pointer}.tmp`, JSON.stringify({
      schema: RPC_ENDPOINT_SCHEMA, socket: this.socketPath, pid: process.pid, protocol: "pi-rpc",
      pi_version: PI_RPC_VERSION, run_id: this.options.runId, started_unix_s: Date.now() / 1000,
    }, null, 2) + "\n");
    renameSync(`${pointer}.tmp`, pointer);
  }

  private connection(socket: Socket): void {
    if (this.ended) {
      socket.destroy();
      return;
    }
    const client = new Client(this.nextId++, socket);
    this.clients.set(client.id, client);
    if (this.controllerId === null) this.controllerId = client.id;
    const detachReader = attachJsonlLineReader(socket, (line) => void this.line(client, line));
    const gone = () => {
      if (client.closed) return;
      client.closed = true;
      detachReader();
      this.clients.delete(client.id);
      this.ui.detach(client.id);
      if (this.controllerId === client.id) {
        // The oldest observer takes over, so a dialog always has someone to answer it.
        const next = [...this.clients.keys()].sort((a, b) => a - b)[0];
        this.controllerId = next ?? null;
      }
    };
    socket.on("close", gone);
    socket.on("end", () => socket.end());
    socket.on("error", gone);
  }

  private broadcast(message: Json): void {
    for (const client of this.clients.values()) client.send(message);
  }

  private async line(client: Client, line: string): Promise<void> {
    if (!line.trim()) return;
    let command: Json;
    try {
      command = JSON.parse(line);
    } catch (error) {
      client.send({ type: "response", command: "parse", success: false,
        error: `Failed to parse command: ${(error as Error).message}` });
      return;
    }
    if (command?.type === "extension_ui_response") {
      this.ui.respond(client.id, command);
      return;
    }
    if (command?.type === "ora_attach") {
      if (command.role === "controller") {
        if (this.controllerId !== null && this.controllerId !== client.id) this.ui.detach(this.controllerId);
        this.controllerId = client.id;
      } else if (command.role === "observer" && this.controllerId === client.id) {
        const next = [...this.clients.keys()].filter((key) => key !== client.id).sort((a, b) => a - b)[0];
        this.controllerId = next ?? null;
      }
      return;
    }
    try {
      const response = await this.handle(client, command);
      if (response) client.send(response);
    } catch (error) {
      client.send({ id: command?.id, type: "response", command: command?.type, success: false,
        error: error instanceof Error ? error.message : String(error) });
    }
  }

  /** One command (rpc-mode.js's handleCommand, with ORA's refusals). */
  private async handle(client: Client, command: Json): Promise<Json | undefined> {
    const id = command?.id;
    const type = String(command?.type ?? "");
    const ok = (data?: Json) => (data === undefined ? { id, type: "response", command: type, success: true }
      : { id, type: "response", command: type, success: true, data });
    const fail = (message: string) => ({ id, type: "response", command: type, success: false, error: message });
    const refused = refusal(type, this.options.mode(), this.controllerId === client.id);
    if (refused !== null) return fail(refused);
    const session = this.session;
    const runtime: Json = this.options.runtime;
    switch (type) {
      case "prompt": {
        // The authoritative response follows the prompt's preflight, as in Pi.
        // While it runs, a dialog it raises (an extension command's confirm) asks this client.
        let answered = false;
        this.ui.driver = client.id;
        void session.prompt(command.message, {
          images: command.images, streamingBehavior: command.streamingBehavior, source: "rpc",
          preflightResult: (disposition: Json) => {
            answered = true;
            client.send(ok({ disposition }));
          },
        }).catch((error: Error) => {
          if (!answered) client.send(fail(error.message));
        }).finally(() => {
          if (this.ui.driver === client.id) this.ui.driver = null;
        });
        return undefined;
      }
      case "steer":
        return ok({ disposition: await session.steer(command.message, command.images, { source: "rpc" }) });
      case "follow_up": {
        const queue = this.options.mode() === "play" ? session.followUp : session.steer;
        return ok({ disposition: await queue.call(session, command.message, command.images, { source: "rpc" }) });
      }
      case "abort":
        await session.abort();
        return ok();
      case "clear_queue":
        return ok(session.clearQueue());
      case "new_session": {
        const result = await runtime.newSession(command.parentSession ? { parentSession: command.parentSession } : undefined);
        if (!result.cancelled) await this.rebind();
        return ok(result);
      }
      case "get_state":
        return ok({
          model: session.model, thinkingLevel: session.thinkingLevel, isStreaming: session.isStreaming,
          isCompacting: session.isCompacting, steeringMode: session.steeringMode, followUpMode: session.followUpMode,
          sessionFile: session.sessionFile, sessionId: session.sessionId, sessionName: session.sessionName,
          autoCompactionEnabled: session.autoCompactionEnabled, messageCount: session.messages.length,
          pendingMessageCount: session.pendingMessageCount,
        });
      case "set_model": {
        const model = session.modelRuntime.getAvailableSnapshot()
          .find((item: Json) => item.provider === command.provider && item.id === command.modelId);
        if (!model) return fail(`Model not found: ${command.provider}/${command.modelId}`);
        await session.setModel(model);
        return ok(model);
      }
      case "cycle_model":
        return ok((await session.cycleModel()) ?? null);
      case "get_available_models": {
        // An episode offers the one model its RunPlan binds.
        const models = this.options.mode() === "play" ? session.modelRuntime.getAvailableSnapshot()
          : [session.model].filter(Boolean);
        return ok({ models });
      }
      case "set_thinking_level":
        session.setThinkingLevel(command.level);
        return ok();
      case "cycle_thinking_level": {
        const level = session.cycleThinkingLevel();
        return ok(level ? { level } : null);
      }
      case "get_available_thinking_levels":
        return ok({ levels: this.options.mode() === "play" ? session.getAvailableThinkingLevels() : [session.thinkingLevel] });
      case "set_steering_mode":
        session.setSteeringMode(command.mode);
        return ok();
      case "set_follow_up_mode":
        session.setFollowUpMode(command.mode);
        return ok();
      case "compact":
        return ok(await session.compact(command.customInstructions));
      case "set_auto_compaction":
        session.setAutoCompactionEnabled(command.enabled);
        return ok();
      case "set_auto_retry":
        session.setAutoRetryEnabled(command.enabled);
        return ok();
      case "abort_retry":
        session.abortRetry();
        return ok();
      case "bash": {
        const hook = await session.extensionRunner.emitUserBash({
          type: "user_bash", command: command.command, excludeFromContext: command.excludeFromContext ?? false,
          cwd: session.sessionManager.getCwd(),
        });
        if (hook?.result) {
          session.recordBashResult(command.command, hook.result, { excludeFromContext: command.excludeFromContext });
          return ok(hook.result);
        }
        return ok(await session.executeBash(command.command, undefined, {
          excludeFromContext: command.excludeFromContext, id, operations: hook?.operations,
        }));
      }
      case "abort_bash":
        session.abortBash();
        return ok();
      case "get_session_stats":
        return ok(session.getSessionStats());
      case "export_html":
        return ok({ path: await session.exportToHtml(command.outputPath) });
      case "switch_session": {
        const result = await runtime.switchSession(command.sessionPath);
        if (!result.cancelled) await this.rebind();
        return ok(result);
      }
      case "fork": {
        const result = await runtime.fork(command.entryId);
        if (!result.cancelled) await this.rebind();
        return ok({ text: result.selectedText, cancelled: result.cancelled });
      }
      case "clone": {
        const leafId = session.sessionManager.getLeafId();
        if (!leafId) return fail("Cannot clone session: no current entry selected");
        const result = await runtime.fork(leafId, { position: "at" });
        if (!result.cancelled) await this.rebind();
        return ok({ cancelled: result.cancelled });
      }
      case "get_fork_messages":
        return ok({ messages: session.getUserMessagesForForking() });
      case "get_entries": {
        let entries = session.sessionManager.getEntries();
        if (command.since !== undefined) {
          const index = entries.findIndex((entry: Json) => entry.id === command.since);
          if (index === -1) return fail(`Entry not found: ${command.since}`);
          entries = entries.slice(index + 1);
        }
        return ok({ entries, leafId: session.sessionManager.getLeafId() });
      }
      case "get_tree":
        return ok({ tree: session.sessionManager.getTree(), leafId: session.sessionManager.getLeafId() });
      case "get_last_assistant_text":
        return ok({ text: session.getLastAssistantText() });
      case "set_session_name": {
        const name = String(command.name ?? "").trim();
        if (!name) return fail("Session name cannot be empty");
        session.setSessionName(name);
        return ok();
      }
      case "get_messages":
        return ok({ messages: this.images ? this.images.completeAll(session.messages) : session.messages });
      case "get_commands": {
        const commands: Json[] = [];
        for (const item of session.extensionRunner.getRegisteredCommands()) {
          commands.push({ name: item.invocationName, description: item.description, source: "extension",
            sourceInfo: item.sourceInfo });
        }
        for (const template of session.promptTemplates) {
          commands.push({ name: template.name, description: template.description, source: "prompt",
            sourceInfo: template.sourceInfo });
        }
        for (const skill of session.resourceLoader.getSkills().skills) {
          commands.push({ name: `skill:${skill.name}`, description: skill.description, source: "skill",
            sourceInfo: skill.sourceInfo });
        }
        return ok({ commands });
      }
      case "ora_get_role":
        return ok({ role: this.controllerId === client.id ? "controller" : "observer", clients: this.clients.size });
      case "ora_take_control":
        this.controllerId = client.id;
        return ok({ role: "controller" });
      case "ora_get_run": {
        const mode = this.options.mode();
        return ok({
          runId: this.options.runId, mode, role: this.controllerId === client.id ? "controller" : "observer",
          clients: this.clients.size, refused: refusals(mode), play: this.options.play?.() ?? null,
          playMessages: this.options.playMessages?.() ?? [], playEndings: this.options.playEndings?.() ?? [],
          cellCommands: this.options.cellCommands?.() ?? [], perf: this.options.perf?.() ?? null,
        });
      }
      case "ora_get_frames": {
        // The run's recorded motion and operator view (rpc/frames.ts): a summary, and with `step` that step's.
        const summary = this.frames.summary();
        if (command.step === undefined || command.step === null) return ok(summary);
        const step = Number(command.step);
        if (!Number.isInteger(step) || step < 0) return fail("ora_get_frames: step must be a non-negative integer");
        return ok({ ...summary, step: { step, motion: this.frames.motion(step), operatorView: this.frames.operatorView(step) } });
      }
      case "ora_cell_note": {
        if (!this.options.cellNote) return fail("ora_cell_note: this run has no real-cell commands");
        return ok({ entry: this.options.cellNote(command.note) });
      }
      case "ora_play_control": {
        const control = String(command.control ?? "");
        if (!(control in PLAY_CONTROLS)) {
          return fail(`ora_play_control: control must be one of ${Object.keys(PLAY_CONTROLS).join(", ")}`);
        }
        if (!this.options.control) return fail("ora_play_control: this run takes no free-play controls");
        // Recorded as the terminal's own control (an operator message of this type).
        return ok({ record: await this.options.control(control, PLAY_CONTROLS[control]) });
      }
      case "ora_play_queue": {
        const message = typeof command.message === "string" ? command.message.trim() : "";
        if (!message) return fail("ora_play_queue: message is required");
        if (!this.options.queue) return fail("ora_play_queue: this run takes no free-play messages");
        return ok({ record: await this.options.queue(message) });
      }
      default:
        return fail(`Unknown command: ${type}`);
    }
  }

  /** After a session switch (free play): subscribe to and bind the new session. */
  private async rebind(): Promise<void> {
    this.unsubscribe?.();
    await this.bind();
  }

  /** The Supervisor announced where free play is (display only): every client hears it. */
  announcePlay(record: Json, atMs: number = Date.now()): void {
    // `atMs`: when it arrived here (the record carries no clock; play/perf.ts).
    this.broadcast({ type: "ora_play_state", state: record, atMs });
  }

  /** A model call ended, with its timing (play/perf.ts; display only): every client hears it. */
  announceModelCall(call: Json): void {
    this.broadcast({ type: "ora_model_call", call });
  }

  /** A real cell's command outcome (display only): every client hears it. */
  announceCell(entry: Json): void {
    this.broadcast({ type: "ora_cell_command", entry });
  }

  /** An operator message was queued or delivered (display only): every client hears it. */
  announceMessage(message: PlayMessage): void {
    this.broadcast({ type: "ora_play_message", message });
  }

  get attachedClients(): number {
    return this.clients.size;
  }

  /** The run is over: dialogs cancel, clients are disconnected, the socket goes. */
  async end(): Promise<void> {
    if (this.ended) return;
    this.ended = true;
    this.unsubscribe?.();
    this.ui.cancelAll();
    for (const client of this.clients.values()) client.socket.end();
    const server = this.server;
    this.server = null;
    if (server) {
      await Promise.race([
        new Promise<void>((resolve) => server.close(() => resolve())),
        new Promise<void>((resolve) => setTimeout(resolve, 2000).unref()),
      ]);
      for (const client of this.clients.values()) client.socket.destroy();
    }
    // close() unlinks the socket; the sandbox may refuse a second removal, which changes nothing.
    try {
      rmSync(this.socketPath, { force: true });
    } catch {
      // not ours to remove
    }
  }
}
