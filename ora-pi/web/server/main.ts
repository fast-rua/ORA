// `ora web`'s server for one live run (started by ora/app/web.py):
//
//   node ora-pi/web/server/main.ts
//     ORA_WEB_RUN      the live run's directory
//     ORA_WEB_SOCKET   the unix socket to serve on (made 0600), or ORA_WEB_PORT for 127.0.0.1 (development)
//     ORA_WEB_STATIC   the built page (default ora-pi/web/dist)
//     ORA_WEB_ATTACH   the attach stand-in (default ora-pi/src/rpc/attach.ts)
//     ORA_WEB_REPLAY   1: replay the finished run from its evidence instead (`ora replay`):
//                      no Pi RPC, the page's one route is GET /api/replay (server/replay.ts)
//     ORA_WEB_PUBLISH  1 (replay): leave out the run ID and shorten absolute paths
//
// It attaches to the run's Pi RPC as one controlling client and serves the
// page and its routes (server/app.ts) until the run's agent process goes
// away and a signal ends it; a replay serves until a signal ends it.

import { chmodSync, existsSync, rmSync } from "node:fs";
import type { Server } from "node:http";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { createApp, createReplayApp } from "./app.ts";
import { buildReplay } from "./replay.ts";
import { RunRpc } from "./rpc.ts";
import { RunView } from "./view.ts";

const here = dirname(fileURLToPath(import.meta.url));

function required(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

async function main(): Promise<void> {
  const run = resolve(required("ORA_WEB_RUN"));
  const socket = process.env.ORA_WEB_SOCKET ?? null;
  const port = socket === null ? Number(required("ORA_WEB_PORT")) : null;
  const staticDir = resolve(process.env.ORA_WEB_STATIC ?? join(here, "..", "dist"));
  if (!existsSync(join(staticDir, "index.html"))) throw new Error(`no built page in ${staticDir}; run \`ora web setup\``);
  const replay = process.env.ORA_WEB_REPLAY === "1";
  let rpc: RunRpc | null = null;
  let server: Server;
  if (replay) {
    const script = await buildReplay(run, { publish: process.env.ORA_WEB_PUBLISH === "1" });
    server = createReplayApp({ script, staticDir });
  } else {
    rpc = new RunRpc({
      node: process.execPath, attach: resolve(process.env.ORA_WEB_ATTACH ?? join(here, "..", "..", "src", "rpc", "attach.ts")),
      run,
    });
    const view = new RunView(rpc);
    rpc.on("closed", (reason: string) => process.stderr.write(`ora web: ${reason}\n`));
    rpc.start();
    await view.refresh();
    server = createApp({ view, staticDir });
  }
  const umask = process.umask(0o077);
  try {
    await new Promise<void>((resolveListen, reject) => {
      server.once("error", reject);
      if (socket !== null) {
        if (existsSync(socket)) rmSync(socket);
        server.listen(socket, () => resolveListen());
      } else {
        server.listen(port!, "127.0.0.1", () => resolveListen());
      }
    });
  } finally {
    process.umask(umask);
  }
  if (socket !== null) chmodSync(socket, 0o600);
  process.stderr.write(`ora web: ${replay ? "replaying" : "serving"} ${run} on ${socket ?? `127.0.0.1:${port}`}\n`);
  const stop = () => {
    rpc?.stop();
    server.close();
    if (socket !== null) rmSync(socket, { force: true });
    process.exit(0);
  };
  process.on("SIGTERM", stop);
  process.on("SIGINT", stop);
  process.on("SIGHUP", stop);
}

main().catch((error) => {
  process.stderr.write(`ora web: ${error?.message ?? error}\n`);
  process.exit(2);
});
