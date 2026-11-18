// Free play's real-cell commands (/home, /hold, /float, /release, /estop, /diagnose): which runs
// have them, the `ora cell` command line, the y/N prompt, the answer, the transcript entry and the
// agent-trace record. `ora cell` itself never runs here: a fake process answers instead.

import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  CELL_COMMANDS, CELL_COMMAND_NAMES, cellArguments, HALT_COMMAND, haltArguments, cellOutcome, commandLine, PROJECT_ROOT, readRealCell, realCellOf, spawnCell,
  type CellExec, type CellProcessResult, type RealCell,
} from "../src/play/cell.ts";
import { CELL_ENTRY, renderCellEntry } from "../src/tui/cell_commands.ts";
import { OraTui, type OraTuiSource } from "../src/tui/extension.ts";
import { setLanguage, t } from "../src/tui/i18n.ts";

setLanguage("en");

type Json = any;

const CELL: RealCell = { kind: "yam", rigMode: "on", cellConfig: "/cells/bench-a.yaml" };
/** Text in, the same text out (rpc/ui.ts PLAIN_THEME). */
const PLAIN: Json = new Proxy({}, { get: () => (...args: Json[]) => String(args[args.length - 1] ?? "") });

function scratch(label: string): string {
  return mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), `ora-pi-cell-${label}-`));
}

function source(runRoot: string, overrides: Partial<OraTuiSource> = {}): OraTuiSource {
  return {
    runRoot, runId: "run-1", model: "m", profile: "p", protocol: "openai_compatible",
    mutationTools: ["robot.submit_action"], toolNames: ["robot.submit_action"],
    history: () => [], pendingToolImages: () => [], live: () => true, inTurn: () => false,
    stop: () => undefined, interruptAtBoundary: () => undefined, notifySupervisor: () => undefined,
    sessionFile: () => null, play: () => ({ state: "awaiting_instruction", scene: 1, scenes: 1, instructions: 0, running: null }),
    playInstruction: () => null, freePlay: true,
    control: async () => assert.fail("a cell command never goes through the play inbox"),
    ...overrides,
  };
}

/** Pi's extension API as far as the TUI's extension uses it. */
function fakePi() {
  const commands = new Map<string, Json>();
  const renderers = new Map<string, Json>();
  const entries: Array<[string, Json]> = [];
  const pi: Json = {
    on: () => undefined, registerMessageRenderer: () => undefined, registerToolRenderer: () => undefined,
    registerEntryRenderer: (type: string, renderer: Json) => renderers.set(type, renderer),
    registerCommand: (name: string, options: Json) => commands.set(name, options),
    appendEntry: (type: string, data: Json) => entries.push([type, data]),
  };
  return { pi, commands, renderers, entries };
}

/** `ora cell` stand-in: each call waits until the test answers it. */
function pendingExec() {
  const calls: Array<{ file: string; args: string[]; cwd: string; answer(result: CellProcessResult): void }> = [];
  const exec: CellExec = (file, args, cwd) => new Promise((answer) => calls.push({ file, args, cwd, answer }));
  return { exec, calls };
}

/** A command context whose terminal answers the y/N prompt with `key`. */
function answering(key: string, shown: string[] = []): Json {
  return {
    ui: {
      custom: (factory: Json) => new Promise((done) => {
        const component = factory(null, PLAIN, null, done);
        shown.push(component.render(100).join("\n"));
        component.handleInput(key);
      }),
    },
  };
}

function live(exec: CellExec, overrides: Partial<OraTuiSource> = {}) {
  const records: Array<[string, Json, Json]> = [];
  const tui = new OraTui(source(scratch("live"), {
    realCell: CELL, record: (type, payload, artifact) => records.push([type, payload, artifact]), ...overrides,
  }), { live: true, cellExec: exec });
  const notices: Array<[string, string]> = [];
  (tui as Json).ui = { notify: (message: string, level: string) => notices.push([level, message]) };
  const fake = fakePi();
  tui.extension()(fake.pi);
  return { tui, notices, records, ...fake };
}

async function waitFor(predicate: () => boolean, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("timed out");
    await new Promise((done) => setTimeout(done, 5));
  }
}

test("only a free-play plan on a real rig (yam, rig_mode read or on) has a real cell", () => {
  assert.deepEqual(realCellOf({ runtime: { kind: "yam", rig_mode: "read" } }), { kind: "yam", rigMode: "read", cellConfig: null });
  assert.deepEqual(realCellOf({ runtime: { kind: "yam", rig_mode: "on", cell_config: "/cells/a.yaml" } }),
    { kind: "yam", rigMode: "on", cellConfig: "/cells/a.yaml" });
  for (const plan of [{ runtime: { kind: "yam", rig_mode: "sim" } }, { runtime: { kind: "yam" } },
    { runtime: { kind: "robotwin", rig_mode: "on" } }, { runtime: null }, {}, null]) {
    assert.equal(realCellOf(plan), null, JSON.stringify(plan));
  }
  const root = scratch("plan");
  assert.equal(readRealCell(root), null, "a run without a plan has no cell");
  writeFileSync(join(root, "plan.json"), JSON.stringify({ episode_mode: "play", runtime: { kind: "yam", rig_mode: "read" } }));
  assert.deepEqual(readRealCell(root), { kind: "yam", rigMode: "read", cellConfig: null });
  writeFileSync(join(root, "plan.json"), "{");
  assert.throws(() => readRealCell(root), SyntaxError, "a plan that cannot be read is an error, not a simulator");
});

test("the command line: --json always, --cell from the plan, --yes only after a y", () => {
  assert.deepEqual(cellArguments("home", CELL, false), ["cell", "home", "--json", "--cell", "/cells/bench-a.yaml"]);
  assert.deepEqual(cellArguments("estop", { ...CELL, cellConfig: null }, true), ["cell", "estop", "--json", "--yes"]);
  assert.equal(commandLine("/repo/scripts/ora", ["cell", "home", "--cell", "/my cells/a.yaml"]),
    "/repo/scripts/ora cell home --cell '/my cells/a.yaml'");
  assert.deepEqual(CELL_COMMANDS.filter((command) => command.confirm).map((command) => command.name),
    ["float", "release", "estop"]);
});

test("ora cell's answer: ok and its result, or the reason it failed", () => {
  assert.deepEqual(cellOutcome({ code: 0, stdout: '{"ok": true, "result": "both arms at rest"}\n', stderr: "ora: no .venv\n" }),
    { ok: true, text: "both arms at rest" });
  assert.deepEqual(cellOutcome({ code: 0, stdout: '{"ok": true, "result": {"holder": null, "faults": []}}', stderr: "" }),
    { ok: true, text: '{\n  "holder": null,\n  "faults": []\n}' });
  assert.deepEqual(cellOutcome({ code: 1, stdout: '{"ok": false, "error": "the cell is held by pid 4242"}', stderr: "" }),
    { ok: false, text: "the cell is held by pid 4242" });
  assert.deepEqual(cellOutcome({ code: 2, stdout: "", stderr: "usage: ora cell ...\nora: error: invalid choice\n" }),
    { ok: false, text: "exit code 2: usage: ora cell ..." });
  assert.equal(cellOutcome({ code: 1, stdout: '{"ok": true, "result": "x"}', stderr: "" }).ok, false,
    "an ok answer with a failing exit code is not a success");
  assert.deepEqual(cellOutcome({ code: null, stdout: "", stderr: "spawn /x ENOENT" }), { ok: false, text: "spawn /x ENOENT" });
});

test("the process: stdout and stderr kept, no stdin, a missing script answered, never thrown", async () => {
  const dir = scratch("spawn");
  const script = join(dir, "ora");
  writeFileSync(script, [
    "#!/bin/sh",
    "read line && echo 'stdin was readable' >&2",
    "echo 'ora: using python3 from PATH' >&2",
    "printf '{\"ok\": true, \"result\": \"%s\"}\\n' \"$*\"",
  ].join("\n"));
  chmodSync(script, 0o755);
  const result = await spawnCell(script, ["cell", "hold", "--json"], dir);
  assert.equal(result.code, 0);
  assert.equal(result.stderr, "ora: using python3 from PATH\n");
  assert.deepEqual(cellOutcome(result), { ok: true, text: "cell hold --json" });
  const missing = await spawnCell(join(dir, "absent"), [], dir);
  assert.equal(missing.code, null);
  assert.match(missing.stderr, /ENOENT/);
});

test("a real cell's commands are registered, listed in /help and run ora cell at once", async () => {
  const { exec, calls } = pendingExec();
  const { tui, notices, records, commands, renderers, entries } = live(exec);
  for (const command of CELL_COMMANDS) {
    assert.equal(commands.get(command.name)?.description, command.description, command.name);
  }
  assert.ok(renderers.has(CELL_ENTRY));
  assert.equal(tui.intercept("/home"), false, "Pi runs the registered command; it never reaches the input hook");
  assert.match(tui.helpRows().join("\n"), /\/estop {9}disable every motor; the arms fall \(asks y\/N\)/);

  // The handler returns while ora cell still runs: the next line (/estop) is read at once.
  await commands.get("home").handler("", answering("n"));
  assert.equal(calls.length, 1);
  assert.equal(calls[0].file, join(PROJECT_ROOT, "scripts", "ora"));
  assert.deepEqual(calls[0].args, ["cell", "home", "--json", "--cell", "/cells/bench-a.yaml"], "/home asks nothing");
  assert.equal(calls[0].cwd, PROJECT_ROOT);
  assert.deepEqual(notices.at(-1), ["info", "/home: running `ora cell home`…"]);
  assert.equal(entries.length, 0, "no outcome before ora cell answers");
  const line = `${join(PROJECT_ROOT, "scripts", "ora")} cell home --json --cell /cells/bench-a.yaml`;
  assert.deepEqual(records, [["operator_shell_command", { command: line, cwd: PROJECT_ROOT, exclude_from_context: true },
    undefined]]);

  await commands.get("estop").handler("", answering("y"));
  assert.equal(calls.length, 2, "/estop starts while /home is still running");
  assert.deepEqual(calls[1].args, ["cell", "estop", "--json", "--cell", "/cells/bench-a.yaml", "--yes"]);

  calls[1].answer({ code: 0, stdout: '{"ok": true, "result": "DISABLE sent to 14 motors"}\n', stderr: "" });
  calls[0].answer({ code: 1, stdout: '{"ok": false, "error": "the arms did not reach rest: joint 3 lags"}\n', stderr: "" });
  await waitFor(() => entries.length === 2);
  assert.deepEqual(entries.map(([type, data]) => [type, data.name, data.ok, data.text]), [
    [CELL_ENTRY, "estop", true, "DISABLE sent to 14 motors"],
    [CELL_ENTRY, "home", false, "the arms did not reach rest: joint 3 lags"],
  ]);
  assert.deepEqual(records.at(-1), ["operator_shell_result", { command: line, exit_code: 1 },
    { output: '{"ok": false, "error": "the arms did not reach rest: joint 3 lags"}\n' }]);

  const drawn = (data: Json) => renderCellEntry({ data }, {}, PLAIN).render(100).map((row: string) => row.trimEnd());
  assert.deepEqual(drawn({ ...entries[1][1], seconds: 0.4 }).filter(Boolean),
    [" ✗ /home · ora cell home · failed after 0s", "   the arms did not reach rest: joint 3 lags"]);
  assert.match(drawn({ ...entries[0][1], seconds: 41.2 }).join("\n"), /✓ \/estop · ora cell estop · done in 41s/);
});

test("/halt ends the running motion where the arms are at once, never asked; Esc halts it first, or refuses a planned one", async () => {
  const { exec, calls } = pendingExec();
  const controls: string[] = [];
  const { tui, commands, entries, records } = live(exec, {
    play: () => ({ state: "instruction", scene: 1, scenes: 1, instructions: 1, running: { index: 0 } }),
    control: async (type) => { controls.push(type); return { id: "c-1" }; },
  });
  assert.ok(CELL_COMMAND_NAMES.has("halt"));
  assert.equal(commands.get("halt")?.description, HALT_COMMAND.description, "listed in Pi's autocomplete and /help");
  assert.match(tui.helpRows().join("\n"), /\/halt {10}end the running motion where the arms are/);
  await commands.get("halt").handler("", { ui: {} });
  assert.deepEqual(calls[0].args, ["cell", "stop-motion", "--json", "--cell", "/cells/bench-a.yaml", "--by", "play terminal"]);
  assert.deepEqual(haltArguments({ ...CELL, cellConfig: null }), ["cell", "stop-motion", "--json", "--by", "play terminal"]);

  // Esc while the model thinks: the same stop, read at the approval of the motion its reply proposes (refused
  // there), then the instruction's stop; the reply proposed a motion (action_started), so nothing is withdrawn.
  const ok = { code: 0, stdout: JSON.stringify({ ok: true, result: {} }), stderr: "" };
  calls[0].answer(ok);
  tui.playState({ state: "instruction", instructions: 1 });
  assert.deepEqual(tui.onKey("\x1b"), { consume: true });
  await waitFor(() => controls.length === 1);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[1].args, haltArguments(CELL));
  calls[1].answer(ok);
  tui.playState({ state: "action_started", running: { index: 0 } });
  assert.deepEqual(tui.onKey("\x1b"), { consume: true }, "while the robot moves the motion ends first");
  await waitFor(() => controls.length === 2);
  assert.equal(calls.length, 3);
  assert.deepEqual(calls[2].args, haltArguments(CELL));
  calls[2].answer(ok);
  tui.playState({ state: "instruction_finished", instruction: { index: 0, ended: "operator_stopped" } });
  await new Promise((done) => setTimeout(done, 20));
  assert.equal(calls.length, 3, "a motion read the stop: nothing to withdraw");
  assert.deepEqual(controls, ["stop_instruction", "stop_instruction"]);

  // Esc while the model thinks, and the reply was words: the instruction ends with no motion started, and the
  // stop is withdrawn (after its own command), quietly: it would refuse the next motion.
  tui.playState({ state: "instruction", instructions: 2, running: { index: 1 } });
  assert.deepEqual(tui.onKey("\x1b"), { consume: true });
  await waitFor(() => calls.length === 4);
  tui.playState({ state: "instruction_finished", instruction: { index: 1, ended: "reported" } });
  await new Promise((done) => setTimeout(done, 20));
  assert.equal(calls.length, 4, "the withdraw waits for the stop's own command");
  calls[3].answer(ok);
  await waitFor(() => calls.length === 5);
  assert.deepEqual(calls[4].args, ["cell", "stop-motion", "--withdraw", "--json", "--cell", "/cells/bench-a.yaml"]);
  const shown = entries.length;
  calls[4].answer({ code: 0, stdout: JSON.stringify({ ok: true, result: { withdrawn: true } }), stderr: "" });
  await waitFor(() => records.length >= 10);
  assert.equal(entries.length, shown, "a withdrawn stop is in the trace only");
  assert.match(String(records.at(-2)?.[1]?.command), /stop-motion --withdraw/);

  // The page's /halt is this terminal's command too.
  assert.equal(tui.cell!.note({ name: "halt", verb: "stop-motion", ok: true, text: "{}",
    command: "scripts/ora cell stop-motion --json --by 'web page'" }).name, "halt");
});

test("/float, /release and /estop run only after y; anything else cancels", async () => {
  const { exec, calls } = pendingExec();
  const { tui, notices, commands } = live(exec);
  const estop = CELL_COMMANDS.find((command) => command.name === "estop")!;
  const shown: string[] = [];
  for (const key of ["n", "N", "\r", "\x1b", "yes", " "]) {
    assert.equal(await tui.cell!.start(estop, answering(key, shown)), null, JSON.stringify(key));
  }
  assert.match(shown[0], /Disable every motor\? \[y\/N\][\s\S]*whoever holds the cell; the arms fall[\s\S]*y: yes · any other key: no/);
  await commands.get("release").handler("", answering("n"));
  assert.deepEqual(notices.at(-1), ["info", "/release cancelled; nothing was run."]);
  await commands.get("float").handler("", { ui: {} });
  assert.equal(calls.length, 0, "without a terminal to ask, nothing runs");
  await commands.get("float").handler("", answering("\x1b[121u"));  // `y` under the kitty keyboard protocol
  assert.deepEqual(calls.map((call) => call.args.at(-1)), ["--yes"]);
});

test("a run that is not on a real cell answers the commands and never sends them to the model", () => {
  for (const overrides of [{ realCell: null }, { realCell: undefined }]) {
    const tui = new OraTui(source(scratch("sim"), overrides), { live: true });
    const notices: Array<[string, string]> = [];
    (tui as Json).ui = { notify: (message: string, level: string) => notices.push([level, message]) };
    const { pi, commands } = fakePi();
    tui.extension()(pi);
    assert.equal(tui.cell, null);
    for (const command of CELL_COMMANDS) assert.equal(commands.has(command.name), false, command.name);
    assert.equal(tui.intercept("/estop"), true, "handled here: Pi would make it an instruction");
    assert.match(notices.at(-1)![1], /\/estop applies to real robot cells only \(rig_mode read or on\)/);
    assert.doesNotMatch(tui.helpRows().join("\n"), /\/estop/);
  }
  // A benchmark run and a finished run's view have no cell commands, whatever the source says.
  assert.equal(new OraTui(source(scratch("episode"), { freePlay: false, realCell: CELL }), { live: true }).cell, null);
  assert.equal(new OraTui(source(scratch("view"), { realCell: CELL }), { live: false }).cell, null);
});

test("a trace that cannot take the record is said; the command still runs", async () => {
  const { exec, calls } = pendingExec();
  const { notices, commands, entries } = live(exec, {
    record: () => {
      throw new Error("the run's agent trace is closed");
    },
  });
  await commands.get("diagnose").handler("", {});
  assert.equal(calls.length, 1);
  assert.deepEqual(notices.at(-1), ["warning", "/diagnose is not in the run's trace: the run's agent trace is closed"]);
  calls[0].answer({ code: 0, stdout: '{"ok": true, "result": {"holder": "ora play (pid 7)"}}', stderr: "" });
  await waitFor(() => entries.length === 1);
  assert.equal(entries[0][1].ok, true);
});

test("every confirmation and notice has its English and Chinese words", () => {
  const keys = [...CELL_COMMANDS.filter((command) => command.confirm).flatMap((command) =>
    [`cell.confirm.${command.name}.title`, `cell.confirm.${command.name}.message`]),
  "cell.running", "cell.cancelled", "cell.realOnly", "cell.failed", "cell.notRecorded", "cell.yesNo",
  "entry.cellDone", "entry.cellFailed"];
  for (const lang of ["en", "zh"] as const) {
    setLanguage(lang);
    try {
      for (const key of keys) assert.notEqual(t(key), key, `${lang} ${key}`);
    } finally {
      setLanguage("en");
    }
  }
});

test("a command the web page ran is shown and recorded as this terminal's own, marked as the page's", () => {
  const { tui, records, entries } = live(pendingExec().exec);
  const heard: Json[] = [];
  tui.cell!.onOutcome = (entry) => heard.push(entry);
  const line = `${join(PROJECT_ROOT, "scripts", "ora")} cell estop --json --cell /cells/bench-a.yaml --yes`;
  const entry = tui.cell!.note({ id: "page-1", name: "estop", verb: "estop", ok: true, text: "DISABLE sent to 14 motors",
    seconds: 0.8, via: "web page", command: line, cwd: PROJECT_ROOT, exit_code: 0, output: '{"ok": true}' });
  assert.deepEqual(entries, [[CELL_ENTRY, { name: "estop", verb: "estop", ok: true, text: "DISABLE sent to 14 motors",
    seconds: 0.8, via: "web page" }]]);
  assert.deepEqual(records, [
    ["operator_shell_command", { command: line, cwd: PROJECT_ROOT, exclude_from_context: true, via: "web page" }, undefined],
    ["operator_shell_result", { command: line, exit_code: 0, via: "web page" }, { output: '{"ok": true}' }],
  ]);
  assert.deepEqual(heard, [entry]);
  assert.deepEqual(tui.cell!.outcomes, [entry]);
  assert.equal(entry.id, "page-1");
  const drawn = renderCellEntry({ data: entries[0][1] }, {}, PLAIN).render(100).join("\n");
  assert.match(drawn, /✓ \/estop · ora cell estop · done in 0s · from the web page/);
  assert.throws(() => tui.cell!.note({ name: "rm", verb: "rm", ok: true, command: "rm -rf /" }), /not a cell command/);
  assert.throws(() => tui.cell!.note({ name: "home", verb: "home", ok: "yes", command: line }), /ok and command/);
  assert.equal(records.length, 2, "a refused note records nothing");
});

test("this terminal's own outcomes are told to every page too", async () => {
  const { exec, calls } = pendingExec();
  const { tui, commands } = live(exec);
  const heard: Json[] = [];
  tui.cell!.onOutcome = (entry) => heard.push(entry);
  await commands.get("home").handler("", answering("n"));
  calls[0].answer({ code: 0, stdout: '{"ok": true, "result": "at rest"}', stderr: "" });
  await waitFor(() => heard.length === 1);
  assert.deepEqual({ ...heard[0], id: "-", at: 0, seconds: 0 }, { id: "-", name: "home", verb: "home", ok: true,
    text: "at rest", seconds: 0, via: "play terminal", at: 0 });
});
