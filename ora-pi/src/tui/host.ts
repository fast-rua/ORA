// The core-side host of Pi's InteractiveMode for one run.
//
// The headless agent core (spawned by the Supervisor) creates Pi's
// InteractiveMode on a RemoteTerminal and serves it on a unix socket outside
// the run directory, so sealed evidence never holds a socket. One client is
// attached at a time; a new `ora attach` takes over from the previous one.
// Detaching stops Pi's renderer and the run goes on; re-attaching restarts it
// and redraws in full. Quitting Pi (Ctrl+D, Ctrl+C twice, /quit) and
// suspending it (Ctrl+Z) detach instead of ending the agent process.
//
// The endpoint is announced in <run>/native/pi-tui.json for `ora attach`.

import { createHash } from "node:crypto";
import { renameSync, writeFileSync } from "node:fs";
import { createServer, type Server, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { InteractiveMode, type AgentSessionRuntime } from "@earendil-works/pi-coding-agent";
import { resetCapabilitiesCache, setCapabilities, setKittyProtocolActive } from "@earendil-works/pi-tui";

import type { OraTui, OraTuiHost } from "./extension.ts";
import { LineChannel, parseHello, RemoteTerminal, type Hello, type Message } from "./remote_terminal.ts";
import { wantsOraTheme } from "./theme.ts";

type Json = any;

export const ENDPOINT_SCHEMA = "ora.pi-tui-endpoint.v1";

/** InteractiveMode members the host relies on; checked so a Pi upgrade fails at startup. */
const MODE_MEMBERS = ["init", "run", "shutdown", "handleCtrlZ", "showManagedToolStatus", "createExtensionUIContext"];

/**
 * Route editor submissions through ORA first: Pi matches its built-in slash
 * commands (and `!` shell lines) before any extension hook sees the text.
 * Call after `init()`, which installs Pi's own submit handler.
 */
export function guardEditor(mode: Json, tui: OraTui): void {
  const editor = mode.defaultEditor;
  if (typeof editor?.onSubmit !== "function") {
    throw new Error("Pi's InteractiveMode no longer exposes its editor; the ORA TUI needs it");
  }
  if (wantsOraTheme()) markPrompt(editor, () => `${tui.promptMark()} `);
  const submit = editor.onSubmit;
  tui.resubmit = (text: string) => submit(text);
  editor.onSubmit = async (text: string) => {
    if (tui.intercept(text)) {
      editor.setText("");
      return;
    }
    return submit(text);
  };
}

/**
 * The prompt after "❯" between its two rules (the teaser's): Pi's editor drawn three columns narrower, its
 * first line after the mark, the others and its completion list indented to match, its rules full width.
 * The cursor follows (pi-tui places it at the marker in the line) and a click lands where it points.
 */
export function markPrompt(editor: Json, mark: () => string): void {
  const render = editor.render?.bind(editor);
  const mouse = editor.handleMouse?.bind(editor);
  if (typeof render !== "function" || editor.oraPromptMarked) return;
  editor.oraPromptMarked = true;
  editor.render = (width: number) => {
    if (width < 16) return render(width);
    const lines: string[] = render(width - 3);
    const count = Number(editor.renderedVisibleLineCount);
    if (!Number.isInteger(count) || count < 1 || lines.length < count + 2) return render(width);
    const rule = (line: string) => `${typeof editor.borderColor === "function" ? editor.borderColor("───") : "───"}${line}`;
    return lines.map((line, index) => (index === 0 || index === count + 1 ? rule(line)
      : index === 1 ? ` ${mark()}${line}` : `   ${line}`));
  };
  if (typeof mouse === "function") editor.handleMouse = (event: Json) => mouse({ ...event, x: Math.max(0, Number(event?.x) - 3) });
}

export class TuiHost implements OraTuiHost {
  readonly terminal = new RemoteTerminal();
  readonly socketPath: string;
  private readonly runtime: AgentSessionRuntime;
  private readonly runRoot: string;
  private readonly tui: OraTui;
  private mode: Json = null;
  private server: Server | null = null;
  private channel: LineChannel | null = null;
  private exports = new Map<number, { resolve: (path: string) => void; reject: (error: Error) => void }>();
  private nextExport = 1;

  constructor(runtime: AgentSessionRuntime, runRoot: string, tui: OraTui) {
    this.runtime = runtime;
    this.runRoot = runRoot;
    this.tui = tui;
    tui.host = this;
    const uid = typeof process.getuid === "function" ? process.getuid() : 0;
    const digest = createHash("sha256").update(runRoot).digest("hex").slice(0, 12);
    // Short (unix socket paths stop at ~104 bytes), private, unique per agent process. A free-play
    // agent process inherits the run's own TMPDIR, which can be too deep for a socket: then /tmp.
    const name = `ora-pi-${uid}-${digest}-${process.pid}.sock`;
    const preferred = join(tmpdir(), name);
    this.socketPath = Buffer.byteLength(preferred) <= 100 ? preferred : join("/tmp", name);
  }

  /** The RPC host's composite over the terminal's uiContext (rpc/host.ts), or none. */
  private composeUi: ((base: Json) => Json) | null = null;

  static async start(runtime: AgentSessionRuntime, runRoot: string, tui: OraTui,
                     options: { composeUi?: (base: Json) => Json } = {}): Promise<TuiHost> {
    const host = new TuiHost(runtime, runRoot, tui);
    host.composeUi = options.composeUi ?? null;
    await host.createMode();
    await host.listen();
    return host;
  }

  get attached(): boolean {
    return this.channel !== null;
  }

  private async createMode(): Promise<void> {
    // ORA draws its own header; Pi's startup list of loaded resources is noise here.
    this.runtime.session.settingsManager.setQuietStartup(true);
    const mode: Json = new InteractiveMode(this.runtime, { terminal: this.terminal, verbose: false });
    for (const name of MODE_MEMBERS) {
      if (typeof mode[name] !== "function") throw new Error(`Pi's InteractiveMode no longer has ${name}(); the ORA TUI host needs it`);
    }
    const shutdown = mode.shutdown.bind(mode);
    // A terminal signal still ends the process; every interactive quit only detaches.
    mode.shutdown = async (options?: Json) => (options?.fromSignal ? shutdown(options) : this.detach("detached; the run continues"));
    mode.handleCtrlZ = () => this.detach("detached; the run continues");
    // fd and ripgrep serve Pi's file completion and grep tool, which an ORA run does not have.
    mode.showManagedToolStatus = () => undefined;
    if (this.composeUi !== null) {
      // Every uiContext the terminal hands Pi (the session's binding, a rebinding after a new
      // free-play session, shortcuts) also reaches the RPC clients: notifications to all, a
      // dialog to the client whose prompt raised it, else to this terminal.
      const own = mode.createExtensionUIContext.bind(mode);
      const compose = this.composeUi;
      mode.createExtensionUIContext = () => compose(own());
    }
    await mode.init();
    if (!mode.ui || typeof mode.ui.start !== "function") {
      throw new Error("Pi's InteractiveMode no longer exposes its renderer; the ORA TUI host needs it");
    }
    guardEditor(mode, this.tui);
    this.mode = mode;
    void mode.run().catch((error: Error) => process.stderr.write(`ora-pi tui: ${error?.stack ?? error}\n`));
    if (!this.terminal.attached) mode.ui.stop({ preserveScreen: true });
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
    server.on("error", (error) => process.stderr.write(`ora-pi tui: ${error.stack}\n`));
    this.server = server;
    const pointer = join(this.runRoot, "native", "pi-tui.json");
    writeFileSync(`${pointer}.tmp`, JSON.stringify({
      schema: ENDPOINT_SCHEMA, socket: this.socketPath, pid: process.pid, started_unix_s: Date.now() / 1000,
    }, null, 2) + "\n");
    renameSync(`${pointer}.tmp`, pointer);
  }

  private connection(socket: Socket): void {
    let accepted = false;
    const channel: LineChannel = new LineChannel(socket, (message: Message) => {
      if (!accepted) {
        let hello: Hello;
        try {
          hello = parseHello(message);
        } catch (error) {
          channel.finish({ type: "end", reason: (error as Error).message });
          return;
        }
        accepted = true;
        this.accept(channel, hello);
        return;
      }
      if (channel !== this.channel) return;
      if (message.type === "export_result") {
        const pending = this.exports.get(Number(message.id));
        this.exports.delete(Number(message.id));
        if (typeof message.path === "string") pending?.resolve(message.path);
        else pending?.reject(new Error(String(message.error ?? "export failed")));
        return;
      }
      if (message.type === "world") {
        this.tui.setWorld(message.state ?? null);
        return;
      }
      if (message.type === "kitty") setKittyProtocolActive(message.active === true);
      this.terminal.receive(message);
    }, () => {
      if (channel === this.channel) this.release();
    });
  }

  private accept(channel: LineChannel, hello: Hello): void {
    const previous = this.channel;
    if (previous !== null) {
      this.release();
      previous.finish({ type: "detach", reason: "another terminal attached to this run" });
    }
    resetCapabilitiesCache();
    if (hello.capabilities) setCapabilities(hello.capabilities);
    setKittyProtocolActive(hello.kitty);
    this.tui.applyTheme();                    // ORA's colors for this terminal (truecolor or 256 colors)
    this.terminal.attach(channel, hello);
    this.channel = channel;
    this.tui.setWorld(hello.world); // each terminal brings its own world view, or none
    const ui = this.mode.ui;
    ui.invalidate?.();
    ui.start();
    ui.requestRender(true);
  }

  /** The client went away or is being replaced: stop drawing. */
  private release(): void {
    this.channel = null;
    this.terminal.detach();
    this.mode?.ui.stop({ preserveScreen: true });
    for (const pending of this.exports.values()) pending.reject(new Error("the terminal detached"));
    this.exports.clear();
  }

  detach(reason: string): void {
    const channel = this.channel;
    if (channel === null) return;
    this.mode.ui.stop({ preserveScreen: true }); // leaves the alt screen through the still-attached terminal
    this.channel = null;
    this.terminal.detach();
    channel.finish({ type: "detach", reason });
  }

  exportSession(path: string | null, sessionFile: string): Promise<string> {
    const channel = this.channel;
    if (channel === null) return Promise.reject(new Error("no terminal is attached"));
    const id = this.nextExport++;
    return new Promise<string>((resolve, reject) => {
      this.exports.set(id, { resolve, reject });
      channel.send({ type: "export", id, session_file: sessionFile, path, run_id: this.tui.source.runId });
    });
  }

  /** The run is over: tell the attached terminal and stop serving. */
  async end(reason: string): Promise<void> {
    const channel = this.channel;
    if (channel !== null) {
      this.mode.ui.stop({ preserveScreen: true });
      this.channel = null;
      this.terminal.detach();
      channel.finish({ type: "end", reason });
    }
    this.tui.close();
    const server = this.server;
    this.server = null;
    // close() removes the socket file; it waits for the client to hang up, so bound the wait.
    if (server) {
      await Promise.race([
        new Promise<void>((resolve) => server.close(() => resolve())),
        new Promise<void>((resolve) => setTimeout(resolve, 2000).unref()),
      ]);
    }
  }
}
