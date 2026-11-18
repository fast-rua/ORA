// Entry point of ORA's TypeScript agent core.
//
//   node ora-pi/src/main.ts --rpc-fd N        headless, spawned by the Supervisor; with
//                                             ORA_PI_TUI=1 it also serves Pi's TUI to `ora attach`,
//                                             and Pi's RPC protocol to Pi clients (rpc/host.ts;
//                                             ORA_PI_RPC=1 serves RPC alone)
//   node ora-pi/src/main.ts --attach SOCKET   Pi TUI in this terminal as the core of a run
//                                             waiting on SOCKET (ORA_PI_ATTACH_SOCKET, spike)
//
// The Python Supervisor drives turns over the RPC channel; see core/session.ts
// for the ORA turn, core/pi_bridge.ts for the Pi integration and tui/ for the
// operator's terminal.

import { existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs";
import { connect, Socket } from "node:net";
import { join } from "node:path";

import { RpcPeer } from "./rpc.ts";
import { OraCore, type ActParams, type InitParams } from "./core/session.ts";
import { createDeps } from "./core/deps.ts";
import { createPiRuntime, observationDisplay, persistedOraEntries, runTurn, UNKNOWN_TOOL } from "./core/pi_bridge.ts";
import { parsePy } from "./pyjson.ts";
import { InteractiveMode } from "@earendil-works/pi-coding-agent";
import { report, timed } from "./core/profile.ts";
import { OraTui, type OraTuiSource } from "./tui/extension.ts";
import { guardEditor, TuiHost } from "./tui/host.ts";
import { RpcHost } from "./rpc/host.ts";
import { sandboxedExistsSync } from "./tui/sandbox.ts";
import { readRealCell } from "./play/cell.ts";
import { PlayDriver, wireName } from "./play/driver.ts";
import { createPlayRuntime, scrubCredentials, submitPlayMessage } from "./play/runtime.ts";

sandboxedExistsSync();

type Json = any;

/** What the operator's TUI reads from the live core (`driver`: a free-play run's). */
function tuiSource(core: OraCore, params: InitParams, session: () => Json, driver: PlayDriver | null = null): OraTuiSource {
  const policy = params.policy;
  const toolNames = params.tools.definitions.map((item: Json) => String(item.function?.name));
  return {
    freePlay: Boolean(params.play),
    runRoot: params.run_root,
    runId: params.run_id,
    model: String(policy.model),
    profile: typeof policy.provider_profile === "string" ? policy.provider_profile : null,
    protocol: String(params.provider.kind ?? "openai_compatible"),
    mutationTools: params.tools.mutation_tools.map(String),
    // Free play advertises the robot tools under wire names (play/driver.ts wireName).
    toolNames: [...new Set([...toolNames, ...toolNames.map(wireName)]), UNKNOWN_TOOL],
    history: () => core.history,
    pendingToolImages: () => (core.turn as Json)?.batch?.toolImages ?? [],
    live: () => !core.closed && !core.interrupted,
    inTurn: () => core.turn !== null && (core.turn as Json).outcome === null,
    stop: () => core.operatorAbort(),
    interruptAtBoundary: () => core.interrupt("operator_interrupt"),
    notifySupervisor: () => core.rpc.notify("operator.stop", { reason: "operator_interrupt" }),
    sessionFile: () => session()?.sessionManager.getSessionFile?.() ?? null,
    play: () => core.playState,
    playInstruction: (id: string) => core.playInstructions.get(id) ?? null,
    control: (type: string, message: string) => core.submitOperatorMessage(message, type),
    // A real cell's commands (tui/cell_commands.ts): only a free-play plan on a real rig has them.
    realCell: driver === null ? null : readRealCell(params.run_root),
    perfLine: driver === null ? undefined : () => driver.perf.line(),
    queueNext: driver === null ? undefined : () => driver.queueNext(),
    queuedNext: driver === null ? undefined : () => driver.messages.queuedNext().length,
    record: driver === null ? undefined : (type: string, payload: Json, artifact?: Json) => {
      if (core.closed) throw new Error("the run's agent trace is closed");
      driver.record(type, payload, artifact);
    },
  };
}

/** The line `ora attach` prints when the run's agent session closes. */
function endReason(tui: OraTui | null): string {
  const terminated = tui?.status.refresh().terminated;
  if (!terminated) return "the run's agent session closed";
  return `the episode ended: ${terminated.state ?? "?"} (${terminated.termination ?? "?"}); `
    + "ORA is evaluating and sealing the run";
}

function argument(name: string): string | null {
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1] : null;
}

async function main(): Promise<void> {
  const fd = argument("--rpc-fd");
  const attach = argument("--attach");
  let stream: Socket;
  if (fd !== null) {
    stream = new Socket({ fd: Number(fd), readable: true, writable: true });
  } else if (attach !== null) {
    stream = connect(attach);
  } else {
    throw new Error("usage: main.ts --rpc-fd N | --attach SOCKET");
  }
  const rpc = new RpcPeer(stream);
  let core: OraCore | null = null;
  let session: Json = null;
  let runtime: Json = null;
  let exiting = false;
  let tui: OraTui | null = null;
  let host: TuiHost | null = null;
  let rpcHost: RpcHost | null = null;
  // A free-play run (plan episode_mode "play") runs on Pi's own agent: play/driver.ts.
  let driver: PlayDriver | null = null;
  const serveTui = fd !== null && process.env.ORA_PI_TUI === "1";
  const serveRpc = fd !== null && (serveTui || process.env.ORA_PI_RPC === "1");

  rpc.onClose(() => {
    const code = exiting ? 0 : 3;
    if (!exiting) process.stderr.write("ora-pi: the Supervisor closed the RPC channel\n");
    void rpcHost?.end();
    if (host === null) process.exit(code);
    void host.end("the Supervisor closed the run's agent session").finally(() => process.exit(code));
  });

  rpc.on("init", async (params: InitParams & { resume?: boolean }) => timed("rpc.init", async () => {
    const resume = Boolean(params.resume);
    core = new OraCore(params, rpc, { resume });
    core.deps = await createDeps(core, params);
    const native = join(params.run_root, "native");
    const agentDir = join(native, "pi-agent");
    const sessionDir = join(native, "pi-session");
    mkdirSync(agentDir, { recursive: true });
    // Pi's RPC protocol for Pi clients (rpc/host.ts), once the session exists. With the TUI the
    // terminal builds the session's uiContext through the RPC host's composite (TuiHost composeUi).
    const prepareRpc = () => {
      if (!serveRpc) return;
      const live = core!;
      rpcHost = RpcHost.create({
        runtime, runRoot: params.run_root, runId: params.run_id, mode: () => (params.play ? "play" : "episode"),
        images: { runRoot: params.run_root, history: () => live.history,
          pendingToolImages: () => (live.turn as Json)?.batch?.toolImages ?? [] },
        bindUi: !serveTui,
        play: () => live.playState,
        control: (type, message) => live.submitOperatorMessage(message, type),
        queue: driver === null ? undefined : (message) => submitPlayMessage(live, driver!, message, [], { queued: true }),
        playMessages: () => driver?.messages.list() ?? [],
        playEndings: () => driver?.messages.endings ?? [],
        // A real cell's commands (tui/cell_commands.ts): the page's, told by `ora web`'s gateway.
        cellNote: (note: Json) => {
          if (!tui?.cell) throw new Error("ora_cell_note: this run drives no real cell");
          return tui.cell.note(note);
        },
        cellCommands: () => tui?.cell?.outcomes ?? [],
        perf: () => driver?.perf.snapshot() ?? null,
      });
      const host = rpcHost;
      if (tui?.cell) tui.cell.onOutcome = (entry) => host.announceCell(entry);
    };
    const composeUi = () => (rpcHost === null ? {} : { composeUi: (base: Json) => rpcHost!.composeUi(base) });
    const startRpc = async () => {
      await rpcHost?.open();
    };
    if (params.play) {
      if (resume) throw new Error("a free-play run does not resume a lost agent process");
      if (attach !== null) throw new Error("a free-play run serves its terminal UI from the agent process");
      // Credentials leave this process's environment before Pi (and its shell) starts.
      const { apiKey } = scrubCredentials();
      const play = new PlayDriver(core, params.play);
      driver = play;
      core.operatorDelivery = (payload) => play.deliver(payload);
      play.messages.onChange = (message) => {
        tui?.playMessage(message);
        rpcHost?.announceMessage(message);
      };
      play.perf.onCall = (call) => {
        tui?.refreshPerf();
        rpcHost?.announceModelCall(call);
      };
      if (serveTui) tui = new OraTui(tuiSource(core, params, () => play.session, play), { live: true });
      // The trace starts here: Pi's runtime may already record (the session's model set to ORA's).
      core.start();
      runtime = await createPlayRuntime(core, play, {
        sessionDir, sessionFile: null, extensions: tui === null ? [] : [tui.extension()], apiKey,
        // On a real cell the model gets the robot tools only (play/runtime.ts robotToolsOnly); in code mode,
        // ORA's coding tools and the robot library (codeMode).
        robotToolsOnly: readRealCell(params.run_root) !== null,
        codeMode: params.play.code_mode === true,
      });
      session = runtime.session;
      prepareRpc();
      if (serveTui) host = await TuiHost.start(runtime, params.run_root, tui!, composeUi());
      await startRpc();
      return { ok: true, pi_session_file: session.sessionManager.getSessionFile?.() ?? null };
    }
    const previous = resume && existsSync(sessionDir)
      ? readdirSync(sessionDir).filter((name) => name.endsWith(".jsonl")).sort().map((name) => join(sessionDir, name)).pop() ?? null
      : null;
    if (serveTui || attach !== null) tui = new OraTui(tuiSource(core, params, () => session), { live: true });
    runtime = await createPiRuntime(core, {
      agentDir, sessionDir, sessionFile: previous, extensions: tui === null ? [] : [tui.extension()],
      authPath: params.credentials?.auth_path ?? null,
    });
    session = runtime.session;
    if (resume) {
      const { history, states } = persistedOraEntries(session, params.run_root);
      const events = readFileSync(join(params.run_root, "agent-events.jsonl"), "utf8").split("\n")
        .filter((line) => line.trim()).map((line) => parsePy(line));
      core.restore(history, states, events);
    } else {
      core.start();
    }
    if (attach !== null) {
      // The operator's terminal: Pi's own TUI renders the run and takes steering.
      const mode: Json = new InteractiveMode(runtime, { verbose: false });
      void (async () => {
        await mode.init();
        guardEditor(mode, tui!);
        await mode.run();
      })().catch((error: Error) => process.stderr.write(`ora-pi tui: ${error.stack}\n`));
    } else if (serveTui) {
      prepareRpc();
      host = await TuiHost.start(runtime, params.run_root, tui!, composeUi());
    } else {
      prepareRpc();
    }
    await startRpc();
    return { ok: true, pi_session_file: session.sessionManager.getSessionFile?.() ?? null };
  }));

  rpc.on("act", async (params: ActParams & { resume?: boolean }) => timed("rpc.act", async () => {
    if (core === null) throw new Error("act before init");
    tui?.observeTurn(params.step, params.remaining_budget);
    if (driver !== null) return driver.act(params);
    const resumed = params.resume ? core.resumeTurn(params) : null;
    const turn = resumed ?? core.openTurn(params);
    if (turn.outcome === null) {
      const display = resumed
        ? `ORA resumed this turn after an agent-process restart (observation step ${params.step}).`
        : observationDisplay(turn.observation);
      await runTurn(session, core, display, { ora_observation_step: params.step, ora_resumed: Boolean(resumed) });
    }
    return core.finishTurn();
  }));

  rpc.on("preserve_observation", async (params: ActParams) => {
    core!.preserveObservation(params);
    return null;
  });

  rpc.on("record", async (params: { type: string; fields: Json }) => {
    const event = core!.record(params.type, params.fields ?? {});
    driver?.noteRecord(params.type, params.fields ?? {});  // free play: a running command's calls, live
    return event;
  });
  // The synchronous execution ledger closes only after its referenced Pi evidence is durable.
  rpc.on("execution_evidence_barrier", async () => { await core!.durability.barrier(); return null; });
  rpc.on("official_verdict", async (params: Json) => core!.officialVerdict(params));
  rpc.on("operator_message", async (params: Json) => core!.operatorMessage(params.payload));
  // A free-play Supervisor waiting for an instruction calls in so the core's own requests (an instruction
  // typed in the TUI) are served between turns.
  rpc.on("ping", async () => null);
  // A free-play Supervisor announces where the session is (display only: ora/orchestrator/episode/play.py).
  rpc.on("play.state", async (params: Json) => {
    if (core === null) return null;
    core.playState = params;
    driver?.messages.apply(params);  // a message's state (play/messages.ts): onChange tells the clients
    // The record carries no clock: an action's start and end are stamped on arrival (play/perf.ts).
    const atMs = driver?.perf.announced(params) ?? Date.now();
    tui?.playState(params);
    rpcHost?.announcePlay(params, atMs);
    return null;
  });

  rpc.on("interrupt", async (params: Json) => {
    if (core === null) return null;
    core.interrupt(String(params.reason ?? "operator_interrupt"));
    try {
      await session?.abort();
    } catch {
      // the run may already be idle
    }
    return null;
  });

  rpc.on("close", async () => {
    exiting = true;
    report();
    try {
      // A prediction still running is cancelled and recorded before the agent trace closes (play/pipeline.ts).
      await driver?.predictor?.close().catch(() => undefined);
      await core?.close();
    } finally {
      await rpcHost?.end().catch(() => undefined);
      if (host !== null) await host.end(endReason(tui)).catch(() => undefined);
      try {
        (runtime?.session ?? session)?.dispose();
      } catch {
        // disposing an idle session cannot fail in a way that matters here
      }
      // The Supervisor closes the channel after this response; exit then (or soon).
      setTimeout(() => process.exit(0), 5000).unref();
    }
    return null;
  });
}

main().catch((error) => {
  process.stderr.write(`ora-pi: ${error?.stack ?? error}\n`);
  process.exit(2);
});
