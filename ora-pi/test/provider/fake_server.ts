// A scripted local Chat Completions server for the recovery differential.
// Path: /<scenario>/<client>/v1/chat/completions; each (scenario, client)
// has its own attempt counter so the Python and TypeScript clients see the
// same scripted sequence. Never contacts anything but loopback.

import { createHash } from "node:crypto";
import { type IncomingMessage, type ServerResponse, createServer } from "node:http";
import type { AddressInfo } from "node:net";

export type Step =
  | { status: number; headers?: Record<string, string>; body?: string }
  | { sse: string[]; end: "close" | "destroy" | "hang"; gapMs?: number }
  | { json: string }
  | { drop: true }
  | { silent: true };

export interface SeenRequest {
  headers: Record<string, string | undefined>;
  sha256: string;
  bytes: number;
}

export class FakeServer {
  scripts: Map<string, Step[]>;
  counters = new Map<string, number>();
  seen = new Map<string, SeenRequest[]>();
  port = 0;
  private server = createServer((req, res) => this.handle(req, res));

  constructor(scripts: Record<string, Step[]>) {
    this.scripts = new Map(Object.entries(scripts));
  }

  async start(): Promise<string> {
    await new Promise<void>((resolve) => this.server.listen(0, "127.0.0.1", resolve));
    this.port = (this.server.address() as AddressInfo).port;
    return `http://127.0.0.1:${this.port}`;
  }

  async stop(): Promise<void> {
    this.server.closeAllConnections();
    await new Promise<void>((resolve) => this.server.close(() => resolve()));
  }

  private handle(req: IncomingMessage, res: ServerResponse): void {
    const [, scenario, client] = (req.url ?? "").split("/");
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      const body = Buffer.concat(chunks);
      const key = `${scenario}/${client}`;
      const list = this.seen.get(key) ?? [];
      list.push({
        headers: {
          "content-type": req.headers["content-type"], authorization: req.headers.authorization,
          "x-session-id": req.headers["x-session-id"] as string | undefined,
          ...(req.headers["x-api-key"] !== undefined ? { "x-api-key": req.headers["x-api-key"] as string } : {}),
          ...(req.headers["anthropic-version"] !== undefined
            ? { "anthropic-version": req.headers["anthropic-version"] as string } : {}),
        },
        sha256: createHash("sha256").update(body).digest("hex"), bytes: body.length,
      });
      this.seen.set(key, list);
      const steps = this.scripts.get(scenario);
      if (!steps) {
        res.writeHead(404).end();
        return;
      }
      const n = this.counters.get(key) ?? 0;
      this.counters.set(key, n + 1);
      this.respond(steps[Math.min(n, steps.length - 1)], res);
    });
  }

  private respond(step: Step, res: ServerResponse): void {
    if ("drop" in step) {
      res.socket?.destroy();
      return;
    }
    // "silent": accept the request and never send the response headers.
    if ("silent" in step) return;
    if ("json" in step) {
      const body = Buffer.from(step.json, "utf8");
      res.writeHead(200, { "content-type": "application/json", "content-length": String(body.length) }).end(body);
      return;
    }
    if ("status" in step) {
      const body = Buffer.from(step.body ?? "", "utf8");
      res.writeHead(step.status, { "content-type": "application/json", ...(step.headers ?? {}),
        "content-length": String(body.length) }).end(body);
      return;
    }
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
    res.flushHeaders();
    const finish = () => {
      if (step.end === "close") {
        res.end();
      } else if (step.end === "destroy") {
        setTimeout(() => res.socket?.destroy(), 100);
      }
      // "hang": leave the stream open until the client goes away.
    };
    if (!step.gapMs) {
      for (const chunk of step.sse) res.write(chunk);
      finish();
      return;
    }
    // A slow but steady stream: one chunk every gapMs.
    let index = 0;
    const next = () => {
      if (res.destroyed) return;
      if (index >= step.sse.length) return finish();
      res.write(step.sse[index++]);
      setTimeout(next, step.gapMs);
    };
    next();
  }
}
