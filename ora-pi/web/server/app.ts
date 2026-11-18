// The web server of one live run: ORA's page (dist/), its server-sent events
// and the few commands a page sends, each mapped to one Pi RPC command.
//
//   GET  /api/snapshot        everything a page shows (also the first event of /api/events)
//   GET  /api/events          server-sent events (server/view.ts)
//   POST /api/send            {text, as?: prompt|steer|follow_up|queue}: a message for the run;
//                             `queue` (free play): the next instruction, held by the Supervisor
//                             until the running one ends (ora_play_queue)
//   POST /api/dialog          {id, value | confirmed | cancelled}: Pi's extension_ui_response
//   POST /api/play            {control: stop_instruction|reset|quit}: free play's controls
//   GET  /api/models          POST /api/model {provider, modelId}
//   GET  /api/thinking-levels POST /api/thinking {level}
//   POST /api/compact {instructions?}   POST /api/name {name}   POST /api/new-session
//   GET  /api/fork-messages   POST /api/fork {entryId}   GET /api/stats
//   GET  /api/frames?step=N   the run's recorded motion and operator view (ora_get_frames: the
//                             summary, and step N's frames by URL; the gateway serves the PNGs)
//   POST /api/cell-note       {name, verb, ok, text, command, ...}: a real-cell command the
//                             gateway ran for the page (ora/app/web_cell.py), for the run's
//                             agent process (ora_cell_note). The gateway's own: no page reaches it
//                             (ora/app/web_gateway.py ALLOWED); the page's cell routes are the
//                             gateway's (/ora/cell)
//
// A finished run's replay (`ora replay`, createReplayApp) serves the same page and one route:
//   GET  /api/replay          the run rebuilt from its evidence (server/replay.ts); no command
//                             reaches anything, and every other /api route is refused (410)
//
// Which of these a run accepts is the agent process's decision (an episode
// refuses the model, reasoning, compaction, session and rename commands:
// ora-pi/src/rpc/policy.ts); a refusal comes back as 409 with its reason.
// There are no file, directory, working-directory, shell or update routes,
// and nothing here reads or writes the run's files. The `ora web` gateway in
// front adds the launch token, the cookie and the Host and Origin checks.
//
// Seeded from Watercol/pi-web-ui's server/src/http.ts (MIT, see ../LICENSE).

import { readFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { extname, join, normalize, sep } from "node:path";

import type { ReplayScript } from "./replay.ts";
import type { RunView } from "./view.ts";

type Json = any;

const MAX_BODY = 256 * 1024;
const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml", ".png": "image/png", ".woff2": "font/woff2", ".json": "application/json",
};
/** The page loads only its own scripts and styles; images may be its own, data: URLs (inline steps) or blobs (live frames). */
export const CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
  + "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
const SENDS = new Set(["prompt", "steer", "follow_up", "queue"]);
const CONTROLS = new Set(["stop_instruction", "reset", "quit"]);

class HttpError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function json(res: ServerResponse, status: number, body: Json): void {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
  res.end(JSON.stringify(body));
}

async function body(req: IncomingMessage): Promise<Json> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY) throw new HttpError(413, "request too large");
    chunks.push(chunk as Buffer);
  }
  if (!chunks.length) return {};
  try {
    const value = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("not an object");
    return value;
  } catch {
    throw new HttpError(400, "the request body must be a JSON object");
  }
}

function text(value: Json, field: string, max = 20_000): string {
  if (typeof value !== "string" || !value.trim()) throw new HttpError(400, `${field} is required`);
  if (value.length > max) throw new HttpError(413, `${field} is longer than ${max} characters`);
  return value;
}

export interface AppOptions {
  view: RunView;
  /** The built page (ora-pi/web/dist). */
  staticDir: string;
}

export function createApp(options: AppOptions): Server {
  const { view, staticDir } = options;
  const rpc = view.rpc;

  /** One Pi RPC command; its refusal (success false) becomes 409 with the agent process's reason. */
  async function command(type: string, payload: Record<string, Json> = {}, timeoutMs?: number): Promise<Json> {
    if (view.closed) throw new HttpError(410, `the run is no longer live: ${view.closed}`);
    const response = await rpc.request(type, payload, timeoutMs);
    if (!response.success) throw new HttpError(409, String(response.error ?? `${type} was refused`));
    return response.data ?? {};
  }

  async function api(req: IncomingMessage, res: ServerResponse, path: string): Promise<void> {
    const route = `${req.method} ${path}`;
    switch (route) {
      case "GET /api/snapshot":
        return json(res, 200, view.snapshot());
      case "GET /api/events":
        return events(req, res);
      case "POST /api/send": {
        const input = await body(req);
        const message = text(input.text, "text");
        const as = input.as === undefined ? (view.state.isStreaming ? "steer" : "prompt") : String(input.as);
        if (!SENDS.has(as)) throw new HttpError(400, "as must be prompt, steer, follow_up or queue");
        if (message.trimStart().startsWith("!")) {
          throw new HttpError(403, "shell commands are not available in the web view");
        }
        const sent = { text: message, as, at: Date.now(), ok: true as boolean, disposition: null as Json, error: null as Json };
        try {
          // A command's prompt (/stop) answers once its dialog is answered: allow the operator time.
          const data = as === "queue" ? await command("ora_play_queue", { message })
            : await command(as, { message }, as === "prompt" ? 15 * 60_000 : undefined);
          sent.disposition = data?.disposition ?? null;
          return json(res, 200, { ok: true, as, data });
        } catch (error) {
          sent.ok = false;
          sent.error = (error as Error).message;
          throw error;
        } finally {
          view.recordSent(sent);
        }
      }
      case "POST /api/dialog": {
        const input = await body(req);
        const id = text(input.id, "id", 200);
        const response: Json = { id };
        if (input.cancelled === true) response.cancelled = true;
        else if (typeof input.confirmed === "boolean") response.confirmed = input.confirmed;
        else if (typeof input.value === "string") response.value = input.value;
        else throw new HttpError(400, "answer with value, confirmed or cancelled");
        try {
          view.answer(response);
        } catch (error) {
          throw new HttpError(409, (error as Error).message);
        }
        return json(res, 200, { ok: true });
      }
      case "POST /api/play": {
        const control = String((await body(req)).control ?? "");
        if (!CONTROLS.has(control)) throw new HttpError(400, "control must be stop_instruction, reset or quit");
        return json(res, 200, await command("ora_play_control", { control }));
      }
      case "GET /api/models":
        return json(res, 200, await command("get_available_models"));
      case "POST /api/model": {
        const input = await body(req);
        const data = await command("set_model", { provider: text(input.provider, "provider", 200),
          modelId: text(input.modelId, "modelId", 200) });
        void view.refresh(["state"]);
        return json(res, 200, data);
      }
      case "GET /api/thinking-levels":
        return json(res, 200, await command("get_available_thinking_levels"));
      case "POST /api/thinking": {
        const data = await command("set_thinking_level", { level: text((await body(req)).level, "level", 40) });
        void view.refresh(["state"]);
        return json(res, 200, data);
      }
      case "POST /api/compact": {
        const input = await body(req);
        const instructions = typeof input.instructions === "string" && input.instructions.trim()
          ? input.instructions.slice(0, 4000) : undefined;
        return json(res, 200, await command("compact", instructions ? { customInstructions: instructions } : {}));
      }
      case "POST /api/name": {
        const data = await command("set_session_name", { name: text((await body(req)).name, "name", 200) });
        void view.refresh(["state"]);
        return json(res, 200, data);
      }
      case "POST /api/new-session": {
        const data = await command("new_session");
        await view.refresh();
        return json(res, 200, data);
      }
      case "GET /api/fork-messages":
        return json(res, 200, await command("get_fork_messages"));
      case "POST /api/fork": {
        const data = await command("fork", { entryId: text((await body(req)).entryId, "entryId", 200) });
        await view.refresh();
        return json(res, 200, data);
      }
      case "GET /api/stats":
        return json(res, 200, await command("get_session_stats"));
      case "POST /api/cell-note":
        return json(res, 200, await command("ora_cell_note", { note: await body(req) }));
      case "GET /api/frames": {
        const raw = new URL(req.url ?? "/", "http://localhost").searchParams.get("step");
        const step = raw === null || raw === "" ? null : Number(raw);
        if (step !== null && (!Number.isInteger(step) || step < 0)) throw new HttpError(400, "step must be a non-negative integer");
        return json(res, 200, await command("ora_get_frames", step === null ? {} : { step }));
      }
      default:
        throw new HttpError(404, `${route} is not an ORA web route`);
    }
  }

  function events(req: IncomingMessage, res: ServerResponse): void {
    res.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-store, no-transform",
      Connection: "keep-alive", "X-Accel-Buffering": "no",
    });
    const write = (event: Json) => {
      if (!res.writable || res.destroyed) return;
      res.write(`data: ${JSON.stringify(event)}\n\n`);
    };
    write(view.snapshot());
    const unsubscribe = view.subscribe(write);
    // A comment line now and then keeps proxies and the SSH tunnel from closing an idle stream.
    const keepAlive = setInterval(() => res.writable && res.write(": keep-alive\n\n"), 15_000);
    const done = () => {
      clearInterval(keepAlive);
      unsubscribe();
    };
    req.on("close", done);
    res.on("error", done);
  }

  return serve(staticDir, api);
}

/** The built page's own files (index, scripts, styles), never anything beside them. */
async function page(res: ServerResponse, staticDir: string, path: string): Promise<void> {
  const relative = path === "/" ? "index.html" : decodeURIComponent(path).replace(/^\/+/, "");
  const file = normalize(join(staticDir, relative));
  if (!file.startsWith(normalize(staticDir) + sep) || relative.split("/").some((part) => part.startsWith("."))) {
    throw new HttpError(404, "not found");
  }
  let data: Buffer;
  try {
    data = await readFile(file);
  } catch {
    throw new HttpError(404, "not found");
  }
  res.writeHead(200, {
    "Content-Type": TYPES[extname(file)] ?? "application/octet-stream", "Cache-Control": "no-store",
    "Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
  });
  res.end(data);
}

function serve(staticDir: string, api: (req: IncomingMessage, res: ServerResponse, path: string) => Promise<void> | void): Server {
  return createServer((req, res) => {
    const path = new URL(req.url ?? "/", "http://localhost").pathname;
    const handle = path.startsWith("/api/") ? Promise.resolve().then(() => api(req, res, path))
      : req.method === "GET" || req.method === "HEAD" ? page(res, staticDir, path)
        : Promise.reject(new HttpError(405, "method not allowed"));
    handle.catch((error: Error) => {
      if (res.headersSent) {
        res.end();
        return;
      }
      json(res, error instanceof HttpError ? error.status : 502, { error: error.message });
    });
  });
}

export interface ReplayAppOptions {
  /** The finished run, rebuilt from its evidence (server/replay.ts). */
  script: ReplayScript;
  /** The built page (ora-pi/web/dist). */
  staticDir: string;
}

/** The page and its replay: read-only, attached to nothing; a command route answers 410. */
export function createReplayApp(options: ReplayAppOptions): Server {
  const body = JSON.stringify(options.script);
  return serve(options.staticDir, (req, res, path) => {
    if (req.method === "GET" && path === "/api/replay") {
      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
      res.end(body);
      return;
    }
    throw new HttpError(410, "a recorded run takes no commands: this page replays its evidence");
  });
}
