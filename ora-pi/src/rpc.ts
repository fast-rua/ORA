// JSON-lines RPC between the Python Supervisor and the TypeScript agent core.
//
// Both peers may send requests: Python drives turns (`act`, runtime events,
// `close`) and TypeScript calls back into Python for tools and observations.
// Lines are parsed with parsePy so Python's numbers survive the crossing, and
// written with dumpsPy. Ids are prefixed by side ("t" here, "p" in Python).

import type { Duplex } from "node:stream";
import { StringDecoder } from "node:string_decoder";

import { dumpsPy, parsePy } from "./pyjson.ts";

export class RpcError extends Error {
  readonly code: string;
  readonly data: unknown;
  constructor(message: string, code: string = "rpc_error", data: unknown = null) {
    super(message);
    this.code = code;
    this.data = data;
  }
}

type Handler = (params: any) => unknown | Promise<unknown>;

interface Pending {
  resolve: (value: any) => void;
  reject: (error: Error) => void;
  method: string;
}

export class RpcPeer {
  private nextId = 1;
  private pending = new Map<string, Pending>();
  private handlers = new Map<string, Handler>();
  private buffer = "";
  private decoder = new StringDecoder("utf8");
  private closedError: Error | null = null;
  private closeListeners: Array<(error: Error | null) => void> = [];

  private readonly stream: Duplex;

  constructor(stream: Duplex) {
    this.stream = stream;
    stream.on("data", (chunk: Buffer) => this.receive(this.decoder.write(chunk)));
    stream.on("end", () => this.shutdown(null));
    stream.on("close", () => this.shutdown(null));
    stream.on("error", (error: Error) => this.shutdown(error));
  }

  on(method: string, handler: Handler): void {
    this.handlers.set(method, handler);
  }

  onClose(listener: (error: Error | null) => void): void {
    this.closeListeners.push(listener);
  }

  get closed(): boolean {
    return this.closedError !== null;
  }

  request<T = any>(method: string, params: unknown = {}): Promise<T> {
    if (this.closedError) return Promise.reject(this.closedError);
    const id = `t${this.nextId++}`;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, method });
      this.write({ id, method, params });
    });
  }

  notify(method: string, params: unknown = {}): void {
    if (!this.closedError) this.write({ method, params });
  }

  private write(message: Record<string, unknown>): void {
    this.stream.write(dumpsPy(message, { ensureAscii: false }) + "\n");
  }

  private receive(text: string): void {
    this.buffer += text;
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      if (line.trim()) this.dispatch(line);
      newline = this.buffer.indexOf("\n");
    }
  }

  private dispatch(line: string): void {
    let message: any;
    try {
      message = parsePy(line);
    } catch (error) {
      this.shutdown(new RpcError(`malformed RPC line: ${(error as Error).message}`));
      return;
    }
    if (typeof message.method === "string") {
      void this.serve(message);
      return;
    }
    const pending = this.pending.get(String(message.id));
    if (!pending) return;
    this.pending.delete(String(message.id));
    if (message.error) {
      const error = message.error;
      pending.reject(new RpcError(String(error.message ?? "remote error"), String(error.code ?? "remote_error"), error.data ?? null));
    } else {
      pending.resolve(message.result);
    }
  }

  private async serve(message: any): Promise<void> {
    const handler = this.handlers.get(message.method);
    const id = message.id;
    try {
      if (!handler) throw new RpcError(`unknown method ${message.method}`, "method_not_found");
      const result = await handler(message.params ?? {});
      if (id !== undefined && id !== null) this.write({ id, result: result === undefined ? null : result });
    } catch (error) {
      const err = error as RpcError;
      if (id !== undefined && id !== null) {
        this.write({ id, error: { message: err.message ?? String(error), code: err.code ?? "handler_error", data: err.data ?? (err as Error).stack ?? null } });
      } else {
        process.stderr.write(`ora-pi: notification ${message.method} failed: ${err.stack ?? err}\n`);
      }
    }
  }

  private shutdown(error: Error | null): void {
    if (this.closedError) return;
    this.closedError = error ?? new RpcError("RPC peer closed", "closed");
    for (const pending of this.pending.values()) pending.reject(this.closedError);
    this.pending.clear();
    for (const listener of this.closeListeners) listener(error);
  }
}
