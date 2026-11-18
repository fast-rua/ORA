// `ora-rpc-attach`: a stand-in for `pi --mode rpc` that attaches to a live ORA run.
//
//   node ora-pi/src/rpc/attach.ts --mode rpc [--session FILE] [Pi's other options]
//   ora-pi/bin/ora-rpc-attach  (the same, as an executable)
//
// Pi RPC clients start `pi --mode rpc` (Pi's RpcClient: `node <cliPath> --mode
// rpc ...`); pointed here, they talk to the run's agent core, which serves
// Pi's protocol on a unix socket (rpc/host.ts). The run is the one named by
// --ora-run DIR or ORA_RPC_RUN, else the run whose session file --session
// names (<run>/native/pi-session/*.jsonl, symlinks resolved). Stdin goes to the
// socket, the socket's lines to stdout. Pi's other options (--provider,
// --model, --session-dir, ...) are ignored: the run binds them.
//
// The client controls the run (a new controller takes over, as a new `ora
// attach` does) unless --ora-role observer or ORA_RPC_ROLE=observer: then it
// reads and is never asked a dialog.
//
// A client that opened the session by another path (a link to the run's
// session file) keys its state by the path it gave: `get_state` reports that
// --session path, not the file it resolves to.
//
// This never starts Pi. A run that is not live (finished, lost, or another
// host's) is an error, exit 2, so no client can open a run's session file
// with a Pi of its own and write to it. Exit 0 when the run's agent process
// closes the connection, 1 on a connection error.

import { existsSync, readFileSync, realpathSync } from "node:fs";
import { connect } from "node:net";
import { basename, dirname, join, resolve } from "node:path";

import { attachJsonlLineReader } from "./jsonl.ts";

const ENDPOINT = join("native", "pi-rpc.json");

function option(name: string): string | null {
  const index = process.argv.indexOf(name);
  if (index >= 0 && index + 1 < process.argv.length) return process.argv[index + 1];
  const inline = process.argv.find((value) => value.startsWith(`${name}=`));
  return inline ? inline.slice(name.length + 1) : null;
}

function fail(message: string, code = 2): never {
  process.stderr.write(`ora-rpc-attach: ${message}\n`);
  process.exit(code);
}

/** The run directory: --ora-run, ORA_RPC_RUN, or the run that owns the --session file. */
export function runOf(argv: { run: string | null; session: string | null }): string {
  if (argv.run) return resolve(argv.run);
  if (argv.session) {
    let file: string;
    try {
      file = realpathSync(argv.session);
    } catch {
      fail(`no such session file: ${argv.session}`);
    }
    if (basename(dirname(file)) !== "pi-session" || basename(dirname(dirname(file))) !== "native") {
      fail(`${file} is not an ORA run's Pi session (<run>/native/pi-session/*.jsonl); this never starts Pi`);
    }
    return dirname(dirname(dirname(file)));
  }
  fail("name the run: --ora-run DIR, ORA_RPC_RUN, or --session <run>/native/pi-session/FILE");
}

function alive(pid: unknown): boolean {
  try {
    process.kill(Number(pid), 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** The socket of the run's live agent process, or a refusal. */
export function liveSocket(run: string): string {
  const pointer = join(run, ENDPOINT);
  if (!existsSync(pointer)) {
    fail(`${basename(run)} serves no Pi RPC (no ${ENDPOINT}): not a live Pi-core run on this host; `
      + "read a finished run with `ora sessions` or `ora attach`. This never starts Pi.");
  }
  let record: { socket?: string; pid?: number };
  try {
    record = JSON.parse(readFileSync(pointer, "utf8"));
  } catch {
    fail(`${pointer} is unreadable`);
  }
  if (typeof record.socket !== "string" || !alive(record.pid) || !existsSync(record.socket)) {
    fail(`${basename(run)} is not live: its agent process is gone (the run finished, or runs on another host). `
      + "This never starts Pi.");
  }
  return record.socket;
}

/** A get_state response names the session by the client's own path. */
export function sessionAlias(line: string, alias: string): string {
  if (!line.includes('"get_state"') || !line.includes('"sessionFile"')) return line;
  try {
    const message = JSON.parse(line);
    if (message?.type !== "response" || message.command !== "get_state" || !message.data?.sessionFile) return line;
    return JSON.stringify({ ...message, data: { ...message.data, sessionFile: alias } });
  } catch {
    return line;
  }
}

function main(): void {
  const mode = option("--mode");
  if (mode !== "rpc") fail("only `--mode rpc` is served (ORA runs are driven by their Supervisor)");
  const run = runOf({ run: option("--ora-run") ?? process.env.ORA_RPC_RUN ?? null, session: option("--session") });
  const socket = connect(liveSocket(run));
  socket.on("error", (error: NodeJS.ErrnoException) => fail(`cannot reach the run's agent process (${error.code ?? error.message})`, 1));
  const role = (option("--ora-role") ?? process.env.ORA_RPC_ROLE ?? "controller") === "observer" ? "observer" : "controller";
  const alias = option("--session");
  socket.on("connect", () => {
    socket.write(`${JSON.stringify({ type: "ora_attach", role })}\n`);
    process.stdin.pipe(socket);
    if (alias === null) {
      socket.pipe(process.stdout);
      return;
    }
    attachJsonlLineReader(socket, (line) => process.stdout.write(`${sessionAlias(line, resolve(alias))}\n`));
  });
  socket.on("close", () => process.exit(0));
  process.stdin.on("end", () => socket.end());
  for (const signal of ["SIGTERM", "SIGHUP", "SIGINT"] as const) process.on(signal, () => {
    socket.destroy();
    process.exit(0);
  });
}

if (process.argv[1] && realpathSync(process.argv[1]) === realpathSync(new URL(import.meta.url).pathname)) main();
