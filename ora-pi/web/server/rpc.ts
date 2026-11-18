// The run's Pi RPC, as one standard Pi RPC client.
//
// `ora-rpc-attach` (ora-pi/src/rpc/attach.ts) stands in for `pi --mode rpc`:
// it relays this process's JSONL to the live run's agent process, refuses a
// run that is not live and never starts a Pi of its own. Commands get
// responses by id; everything else the agent process sends is an event.
//
// Seeded from Watercol/pi-web-ui's server/src/pi-rpc-client.ts (MIT, see
// ../LICENSE): its spawn and request bookkeeping, without its cwd switching,
// git and session listing.

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";

type Json = any;

export interface RpcOptions {
  /** The Node.js that runs the attach stand-in. */
  node: string;
  /** ora-pi/src/rpc/attach.ts */
  attach: string;
  /** The live run's directory. */
  run: string;
  timeoutMs?: number;
}

interface Pending {
  resolve(response: Json): void;
  reject(error: Error): void;
  timer: NodeJS.Timeout;
}

/** LF-delimited JSON lines (Pi's RPC framing; a CR before the LF is tolerated). */
export class JsonLines {
  private buffer = "";

  push(chunk: string): Json[] {
    this.buffer += chunk;
    const values: Json[] = [];
    for (;;) {
      const end = this.buffer.indexOf("\n");
      if (end < 0) return values;
      let line = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 1);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      if (!line.trim()) continue;
      try {
        values.push(JSON.parse(line));
      } catch {
        values.push({ type: "error", message: `not JSON from the run's agent process: ${line.slice(0, 200)}` });
      }
    }
  }
}

/** Events: "event" (a Pi RPC event), "closed" (reason: the relay ended). */
export class RunRpc extends EventEmitter {
  private child: ChildProcessWithoutNullStreams | null = null;
  private readonly pending = new Map<string, Pending>();
  private readonly lines = new JsonLines();
  private counter = 0;
  private stderr = "";
  closed = false;
  closeReason: string | null = null;
  private readonly options: RpcOptions;

  constructor(options: RpcOptions) {
    super();
    this.options = options;
  }

  start(): void {
    if (this.child !== null || this.closed) return;
    const child = spawn(this.options.node, [this.options.attach, "--mode", "rpc", "--ora-run", this.options.run], {
      stdio: ["pipe", "pipe", "pipe"],
      // The web page controls the run as the terminal does (a new controller takes over, rpc/host.ts).
      env: { ...process.env, ORA_RPC_ROLE: "controller", ORA_NODE: this.options.node },
    });
    this.child = child;
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      for (const value of this.lines.push(chunk)) this.receive(value);
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      this.stderr = (this.stderr + chunk).slice(-4000);
    });
    child.stdin.on("error", () => undefined);
    child.on("error", (error) => this.close(`the attach stand-in did not start: ${error.message}`));
    child.on("exit", (code, signal) => {
      const said = this.stderr.trim().split("\n").filter(Boolean).pop();
      this.close(said || `the run's agent process closed the connection (${signal ?? `exit ${code}`})`);
    });
  }

  private close(reason: string): void {
    if (this.closed) return;
    this.closed = true;
    this.closeReason = reason;
    this.child = null;
    for (const [id, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(new Error(reason));
      this.pending.delete(id);
    }
    this.emit("closed", reason);
  }

  private receive(value: Json): void {
    if (value?.type === "response" && typeof value.id === "string" && this.pending.has(value.id)) {
      const pending = this.pending.get(value.id)!;
      this.pending.delete(value.id);
      clearTimeout(pending.timer);
      pending.resolve(value);
      return;
    }
    this.emit("event", value);
  }

  /** One command; resolves with Pi's response (success true or false), rejects when the run is gone. */
  request(type: string, payload: Record<string, Json> = {}, timeoutMs = this.options.timeoutMs ?? 30_000): Promise<Json> {
    const child = this.child;
    if (child === null || this.closed) return Promise.reject(new Error(this.closeReason ?? "not connected to the run"));
    const id = `web-${++this.counter}`;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`the run did not answer ${type} in ${Math.round(timeoutMs / 1000)} s`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      child.stdin.write(`${JSON.stringify({ ...payload, id, type })}\n`);
    });
  }

  /** A line that gets no response (an extension_ui_response). */
  send(message: Json): void {
    if (this.child === null || this.closed) throw new Error(this.closeReason ?? "not connected to the run");
    this.child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  stop(): void {
    this.child?.kill();
  }
}
