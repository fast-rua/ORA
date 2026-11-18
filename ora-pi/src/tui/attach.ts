// `ora attach`: this terminal shows a running ORA agent core's Pi TUI.
//
//   node ora-pi/src/tui/attach.ts SOCKET
//
// pi-tui's ProcessTerminal owns the tty here (raw mode, keyboard protocol,
// bracketed paste, resize); key sequences go to the core and its screen writes
// come back (see remote_terminal.ts). Exit codes: 0 detached, 3 the run ended,
// 4 the agent process went away, 2 nothing to attach to.
//
// With a world view (`ora attach` serves the run's cameras on a loopback
// browser page), ORA_WORLD_FD names a pipe of its state, one JSON line per
// change; each state is relayed so the TUI shows the page's address.

import { connect, Socket } from "node:net";

import { detectCapabilities, matchesKey, ProcessTerminal } from "@earendil-works/pi-tui";

import { exportSession } from "./export.ts";
import { LineChannel, PROTOCOL_VERSION, type Message } from "./remote_terminal.ts";

const socketPath = process.argv[2];
if (!socketPath) {
  process.stderr.write("usage: attach.ts SOCKET\n");
  process.exit(2);
}

const terminal = new ProcessTerminal();
let started = false;
let finished = false;
let code = 4;
let farewell = "the agent process closed the connection";
let kittySent = false;

const socket = connect(socketPath);
socket.on("error", (error: NodeJS.ErrnoException) => {
  if (!started) {
    process.stderr.write(`ora attach: cannot reach the run's terminal endpoint (${error.code ?? error.message})\n`);
    process.exit(2);
  }
});

const channel = new LineChannel(socket, receive, () => void finish());

let world: Record<string, unknown> | null = null;
const worldFd = Number(process.env.ORA_WORLD_FD ?? "");
if (Number.isInteger(worldFd) && worldFd > 2) {
  const pipe = new Socket({ fd: worldFd, readable: true, writable: false });
  pipe.unref();
  new LineChannel(pipe, (message: Message) => {
    world = message;
    if (started) channel.send({ type: "world", state: world });
  }, () => undefined);
}

function syncKitty(): void {
  if (terminal.kittyProtocolActive !== kittySent) {
    kittySent = terminal.kittyProtocolActive;
    channel.send({ type: "kitty", active: kittySent });
  }
}

function leaveNow(data: string): boolean {
  try {
    return data === "\x1c" || matchesKey(data, "ctrl+\\" as any);
  } catch {
    return data === "\x1c";
  }
}

socket.on("connect", () => {
  started = true;
  terminal.start((data: string) => {
    if (leaveNow(data)) {
      code = 0;
      farewell = "detached; the run continues";
      socket.end();
      return;
    }
    syncKitty();
    channel.send({ type: "input", data });
  }, () => {
    channel.send({ type: "resize", columns: terminal.columns, rows: terminal.rows });
  });
  channel.send({
    type: "hello", version: PROTOCOL_VERSION, columns: terminal.columns, rows: terminal.rows,
    kitty: terminal.kittyProtocolActive, capabilities: detectCapabilities(), cwd: process.cwd(), world,
  });
});

function receive(message: Message): void {
  switch (message.type) {
    case "write":
      terminal.write(String(message.data));
      break;
    case "title":
      terminal.setTitle(String(message.title));
      break;
    case "progress":
      terminal.setProgress(message.active === true);
      break;
    case "program_status":
      terminal.setProgramStatus(message.status);
      break;
    case "export": {
      const output = typeof message.path === "string" && message.path ? message.path : `${message.run_id}.html`;
      exportSession(String(message.session_file), output).then(
        (path) => channel.send({ type: "export_result", id: message.id, path }),
        (error) => channel.send({ type: "export_result", id: message.id, error: String(error?.message ?? error) }),
      );
      break;
    }
    case "detach":
      code = 0;
      farewell = String(message.reason ?? "detached");
      break;
    case "end":
      code = 3;
      farewell = String(message.reason ?? "the run ended");
      break;
    default:
      break;
  }
}

async function finish(): Promise<void> {
  if (finished) return;
  finished = true;
  if (started) {
    await terminal.drainInput(300);
    terminal.stop();
  }
  process.stdout.write(`\r\n${farewell}\r\n`);
  process.exit(code);
}

for (const signal of ["SIGTERM", "SIGHUP"] as const) {
  process.on(signal, () => {
    code = 0;
    farewell = "detached; the run continues";
    socket.destroy();
    void finish();
  });
}
