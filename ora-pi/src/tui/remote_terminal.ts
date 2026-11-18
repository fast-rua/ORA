// Pi's terminal UI, drawn on a terminal in another process.
//
// The agent core keeps running headless under the Supervisor; Pi's
// InteractiveMode lives in it and draws through a RemoteTerminal. `ora attach`
// runs pi-tui's own ProcessTerminal on the operator's tty (raw mode, keyboard
// protocol negotiation, bracketed paste, resize) and relays: key sequences and
// sizes in, screen writes out. Detaching drops the relay; the run goes on.
//
// Wire: one JSON object per line over a unix socket.
//   client -> core  hello {columns, rows, kitty, capabilities, cwd, world}
//                   input {data} · resize {columns, rows} · kitty {active}
//                   export_result {id, path, error}
//                   world {state}: the browser world view `ora attach` serves
//                   beside this terminal (null without one); optional, so a
//                   core that predates it ignores it
//   core -> client  write {data} · title {title} · progress {active}
//                   program_status {status} · export {id, ...}
//                   detach {reason} (exit 0) · end {reason} (run over, exit 3)

import type { Socket } from "node:net";
import { StringDecoder } from "node:string_decoder";

import type { ProgramStatus, Terminal, TerminalCapabilities } from "@earendil-works/pi-tui";

export const PROTOCOL_VERSION = 1;

export type Message = Record<string, any> & { type: string };

/** Newline-delimited JSON over a stream socket. */
export class LineChannel {
  private buffer = "";
  private readonly decoder = new StringDecoder("utf8");
  private closed = false;
  readonly socket: Socket;

  constructor(socket: Socket, onMessage: (message: Message) => void, onClose: () => void) {
    this.socket = socket;
    socket.on("data", (chunk: Buffer) => {
      this.buffer += this.decoder.write(chunk);
      let newline: number;
      while ((newline = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, newline);
        this.buffer = this.buffer.slice(newline + 1);
        if (!line.trim()) continue;
        let message: Message;
        try {
          message = JSON.parse(line);
        } catch {
          continue;
        }
        onMessage(message);
      }
    });
    const closed = () => {
      if (this.closed) return;
      this.closed = true;
      onClose();
    };
    socket.on("close", closed);
    socket.on("end", closed);
    socket.on("error", closed);
  }

  get isClosed(): boolean {
    return this.closed;
  }

  send(message: Message): void {
    if (this.closed || this.socket.destroyed) return;
    this.socket.write(JSON.stringify(message) + "\n");
  }

  /** Send, then close once the bytes are out. */
  finish(message: Message): void {
    if (this.closed || this.socket.destroyed) return;
    this.socket.end(JSON.stringify(message) + "\n");
  }
}

export interface Hello {
  columns: number;
  rows: number;
  kitty: boolean;
  capabilities: TerminalCapabilities | null;
  cwd: string | null;
  /** The attaching terminal's world view state (see `worldView` in status.ts), or null. */
  world: Record<string, unknown> | null;
}

export function parseHello(message: Message): Hello {
  if (message.type !== "hello" || message.version !== PROTOCOL_VERSION) {
    throw new Error(`unsupported attach protocol (expected hello v${PROTOCOL_VERSION})`);
  }
  const size = (value: unknown, fallback: number) =>
    typeof value === "number" && Number.isInteger(value) && value > 0 ? value : fallback;
  return {
    columns: size(message.columns, 80),
    rows: size(message.rows, 24),
    kitty: message.kitty === true,
    capabilities: message.capabilities && typeof message.capabilities === "object" ? message.capabilities : null,
    cwd: typeof message.cwd === "string" ? message.cwd : null,
    world: message.world && typeof message.world === "object" && !Array.isArray(message.world) ? message.world : null,
  };
}

/** The core-side Terminal: writes go to the attached client, keys come from it. */
export class RemoteTerminal implements Terminal {
  private channel: LineChannel | null = null;
  private inputHandler: ((data: string) => void) | null = null;
  private resizeHandler: (() => void) | null = null;
  private width = 80;
  private height = 24;
  private kitty = false;

  get attached(): boolean {
    return this.channel !== null && !this.channel.isClosed;
  }

  attach(channel: LineChannel, hello: Hello): void {
    this.channel = channel;
    this.width = hello.columns;
    this.height = hello.rows;
    this.kitty = hello.kitty;
  }

  detach(): void {
    this.channel = null;
    this.kitty = false;
  }

  receive(message: Message): void {
    if (message.type === "input" && typeof message.data === "string") {
      this.inputHandler?.(message.data);
    } else if (message.type === "resize") {
      if (Number.isInteger(message.columns) && message.columns > 0) this.width = message.columns;
      if (Number.isInteger(message.rows) && message.rows > 0) this.height = message.rows;
      this.resizeHandler?.();
    } else if (message.type === "kitty") {
      this.kitty = message.active === true;
    }
  }

  start(onInput: (data: string) => void, onResize: () => void): void {
    this.inputHandler = onInput;
    this.resizeHandler = onResize;
  }

  stop(): void {
    this.inputHandler = null;
    this.resizeHandler = null;
  }

  async drainInput(): Promise<void> {
    // The client's own ProcessTerminal drains its tty when it stops.
  }

  write(data: string): void {
    this.channel?.send({ type: "write", data });
  }

  get columns(): number {
    return this.width;
  }

  get rows(): number {
    return this.height;
  }

  get kittyProtocolActive(): boolean {
    return this.kitty;
  }

  moveBy(lines: number): void {
    if (lines > 0) this.write(`\x1b[${lines}B`);
    else if (lines < 0) this.write(`\x1b[${-lines}A`);
  }

  hideCursor(): void {
    this.write("\x1b[?25l");
  }

  showCursor(): void {
    this.write("\x1b[?25h");
  }

  clearLine(): void {
    this.write("\x1b[K");
  }

  clearFromCursor(): void {
    this.write("\x1b[J");
  }

  clearScreen(): void {
    this.write("\x1b[2J\x1b[H");
  }

  setTitle(title: string): void {
    this.channel?.send({ type: "title", title });
  }

  setProgress(active: boolean): void {
    this.channel?.send({ type: "progress", active });
  }

  setProgramStatus(status: ProgramStatus): void {
    this.channel?.send({ type: "program_status", status });
  }
}
