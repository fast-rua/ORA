// /approve and /decline in free play's terminal on a real cell: which questions wait (the approval
// broker's rule, read from its files), the y/N that shows the question with its id, the
// `ora cell approve|decline --id` command line, and the notice when a new question appears.
// `ora cell` never runs here: a fake process answers instead.

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  answerArguments, APPROVAL_COMMANDS, approvalsDirectory, autoApproveArguments, CELL_COMMAND_NAMES, cellModeState,
  MODE_COMMAND, modeArguments, nextMode, pendingApprovals, SAFE_LIMITS, selfAnswers,
  YOLO_COMMAND, type CellExec, type CellProcessResult, type RealCell,
} from "../src/play/cell.ts";
import { CELL_ENTRY, CellCommands, modeText, questionText } from "../src/tui/cell_commands.ts";
import { OraTui, type OraTuiSource } from "../src/tui/extension.ts";
import { setLanguage, t } from "../src/tui/i18n.ts";
import { spawnPython } from "./provider/harness.ts";

setLanguage("en");

type Json = any;

const CELL: RealCell = { kind: "yam", rigMode: "on", cellConfig: "/cells/bench-a.yaml" };
const PLAIN: Json = new Proxy({}, { get: () => (...args: Json[]) => String(args[args.length - 1] ?? "") });
const APPROVE = APPROVAL_COMMANDS.find((command) => command.name === "approve")!;
const DECLINE = APPROVAL_COMMANDS.find((command) => command.name === "decline")!;

function scratch(label: string): string {
  return mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), `ora-pi-approval-${label}-`));
}

/** A question as the broker writes it, from this (live) process unless told otherwise. */
function post(folder: string, id: string, fields: Json = {}): Json {
  const now = Date.now() / 1000;
  const request = {
    schema: "ora.cell-approval-request.v1", id, kind: "motion", title: "左臂动作等待批准",
    text: "左臂移动到智能体选的位置，约 1.5 秒。\n批准 = 执行这个动作；拒绝 = 不动。\n计划：{\"arms\": [\"left\"]}",
    details: { arms: ["left"] }, created_unix_s: now, expires_unix_s: now + 120, timeout_s: 120,
    pid: process.pid, asker: "ORA worker yam rig_mode on", ...fields,
  };
  writeFileSync(join(folder, `${id}.request.json`), JSON.stringify(request));
  return request;
}

function pendingExec() {
  const calls: Array<{ file: string; args: string[]; answer(result: CellProcessResult): void }> = [];
  const exec: CellExec = (file, args) => new Promise((answer) => calls.push({ file, args, answer }));
  return { exec, calls };
}

/** A command context whose terminal answers the y/N prompt with `key`, keeping what it showed. */
function answering(key: string, shown: string[] = []): Json {
  return {
    ui: {
      custom: (factory: Json) => new Promise((done) => {
        const component = factory(null, PLAIN, null, done);
        shown.push(component.render(200).join("\n"));
        component.handleInput(key);
      }),
    },
  };
}

function commands(folder: string, exec: CellExec, cell: RealCell = CELL) {
  const notices: Array<[string, string]> = [];
  const entries: Array<[string, Json]> = [];
  const commands = new CellCommands(cell, {
    notify: (message, level) => notices.push([level, message]),
    appendEntry: (type, data) => entries.push([type, data]),
  }, { exec, approvals: folder });
  return { commands, notices, entries };
}

async function waitFor(predicate: () => boolean, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("timed out");
    await new Promise((done) => setTimeout(done, 5));
  }
}

test("the broker's folder: $ORA_CELL_APPROVALS, else beside the cell lock", () => {
  assert.equal(approvalsDirectory({}), "/tmp/ora-cell-approvals");
  assert.equal(approvalsDirectory({ ORA_CELL_LOCK: "/run/cells/a.lock" }), "/run/cells/ora-cell-approvals");
  assert.equal(approvalsDirectory({ ORA_CELL_LOCK: "/run/a.lock", ORA_CELL_APPROVALS: "/srv/q" }), "/srv/q");
});

test("waiting questions: unanswered, unexpired, asked by a live process, oldest first", () => {
  // Two levels down, so the traversing id below lands inside the scratch directory.
  const folder = join(scratch("pending"), "broker", "approvals");
  mkdirSync(folder, { recursive: true });
  const now = Date.now() / 1000;
  post(folder, "0000000b", { created_unix_s: now - 1 });
  post(folder, "0000000a", { created_unix_s: now - 5 });
  post(folder, "0000000c");
  writeFileSync(join(folder, "0000000c.answer.json"), JSON.stringify({ id: "0000000c", approved: true }));
  post(folder, "0000000d", { expires_unix_s: now - 1 });
  post(folder, "0000000e", { pid: 999999 });
  writeFileSync(join(folder, "0000000f.request.json"), "{\"id\": \"0000000f\", \"expires_unix_s\"");
  writeFileSync(join(folder, ".0000000a.request.json.1.ab.tmp"), "{}");
  post(folder, "../../etc");

  const alive = (pid: number) => pid === process.pid;
  assert.deepEqual(pendingApprovals(folder, now, alive).map((request) => request.id), ["0000000a", "0000000b"]);
  assert.deepEqual(pendingApprovals(join(folder, "absent"), now, alive), []);
});

test("the command line names the id; only a decline carries a reason", () => {
  assert.deepEqual(answerArguments("approve", "0a1b2c3d", CELL, "ignored"),
    ["cell", "approve", "--json", "--cell", "/cells/bench-a.yaml", "--id", "0a1b2c3d", "--by", "play terminal"]);
  assert.deepEqual(answerArguments("decline", "0a1b2c3d", { ...CELL, cellConfig: null }, "杯子挡着"),
    ["cell", "decline", "--json", "--id", "0a1b2c3d", "--by", "play terminal", "--reason", "杯子挡着"]);
  assert.ok(CELL_COMMAND_NAMES.has("approve") && CELL_COMMAND_NAMES.has("decline"));
  assert.deepEqual(answerArguments("approve", "0a1b2c3d", { ...CELL, cellConfig: null }, null, "11x8"),
    ["cell", "approve", "--json", "--id", "0a1b2c3d", "--by", "play terminal", "--value", "11x8"]);
  assert.deepEqual(answerArguments("decline", "0a1b2c3d", { ...CELL, cellConfig: null }, null, "11x8"),
    ["cell", "decline", "--json", "--id", "0a1b2c3d", "--by", "play terminal"]);
});

test("/approve <value> answers a value question; without one, or a value for another question, nothing runs", async () => {
  const folder = scratch("value");
  const { exec, calls } = pendingExec();
  const { commands: cell, notices } = commands(folder, exec);
  post(folder, "0a1b2c3d", { kind: "value", title: "Checkerboard square size", text: "Measure one square.",
    details: { unit: "mm", example: "20" } });
  assert.equal(await cell.answer(APPROVE, "  ", answering("y")), null);
  assert.deepEqual(notices.at(-1), ["warning",
    "/approve: this question asks for a value (in mm, e.g. /approve 20): /approve <value>. Nothing was run."]);
  const shown: string[] = [];
  await cell.answer(APPROVE, " 20.5 ", answering("y", shown));
  assert.match(shown[0], /Measure one square\.[\s\S]*Approve with \/approve <value> \(in mm, e\.g\. \/approve 20\); decline = give up\.[\s\S]*Value sent with the approval: 20\.5/);
  assert.deepEqual(calls.map((call) => call.args), [answerArguments("approve", "0a1b2c3d", CELL, null, "20.5")]);

  const other = scratch("novalue");
  const second = commands(other, exec);
  post(other, "0a1b2c3e", { kind: "task" });
  assert.equal(await second.commands.answer(APPROVE, "20", answering("y")), null);
  assert.deepEqual(second.notices.at(-1), ["warning",
    "/approve: this question (task) takes no value; send /approve alone. Nothing was run."]);
  assert.equal(calls.length, 1);
});

test("/approve shows the waiting question and answers that id after y", async () => {
  const folder = scratch("approve");
  const { exec, calls } = pendingExec();
  const { commands: cell, notices, entries } = commands(folder, exec);

  assert.equal(await cell.answer(APPROVE, "", answering("y")), null);
  assert.deepEqual(notices.at(-1), ["info", "/approve: nothing is waiting for an answer."]);

  const request = post(folder, "0a1b2c3d");
  const shown: string[] = [];
  assert.equal(await cell.answer(APPROVE, "", answering("n", shown)), null);
  assert.deepEqual(notices.at(-1), ["info", "/approve cancelled; nothing was run."]);
  assert.equal(calls.length, 0);
  assert.match(shown[0], /Approve this\? \[y\/N\]/);
  assert.match(shown[0], /左臂动作等待批准 · id 0a1b2c3d · 1[12]\d s left \(no answer counts as a decline\)/);
  assert.match(shown[0], /左臂移动到智能体选的位置，约 1\.5 秒。/);

  const run = await cell.answer(APPROVE, "", answering("y"));
  assert.ok(run !== null);
  assert.deepEqual(calls[0].args, answerArguments("approve", request.id, CELL));
  calls[0].answer({ code: 0, stdout: '{"ok": true, "result": "approved 0a1b2c3d: 左臂动作等待批准"}\n', stderr: "" });
  assert.deepEqual(await run!.done, { ok: true, text: "approved 0a1b2c3d: 左臂动作等待批准" });
  assert.deepEqual(entries.map(([type, data]) => [type, data.name, data.ok]), [[CELL_ENTRY, "approve", true]]);
});

test("/decline [reason] sends the reason with that id", async () => {
  const folder = scratch("decline");
  const { exec, calls } = pendingExec();
  const { commands: cell } = commands(folder, exec);
  post(folder, "0a1b2c3d");
  const shown: string[] = [];
  await cell.answer(DECLINE, "  the cup is in the way ", answering("y", shown));
  assert.match(shown[0], /Decline this\? \[y\/N\][\s\S]*Reason sent with the decline: the cup is in the way/);
  assert.deepEqual(calls[0].args, answerArguments("decline", "0a1b2c3d", CELL, "the cup is in the way"));
});

test("an open `on` run says once when a new question waits", () => {
  const folder = scratch("watch");
  const { exec, calls } = pendingExec();
  const { commands: cell, notices } = commands(folder, exec);
  cell.announce();
  assert.equal(notices.length, 0);
  post(folder, "0a1b2c3d");
  cell.announce();
  cell.announce();
  assert.deepEqual(notices, [["warning",
    "The robot asks: 左臂动作等待批准 (0a1b2c3d) · answer with /approve or /decline"]]);
  post(folder, "0a1b2c3e", { title: "手臂没能回到起始位", kind: "release" });
  cell.announce();
  assert.equal(notices.length, 2);
  post(folder, "0a1b2c3f", { title: "Place the checkerboard", kind: "task" });
  cell.announce();
  assert.equal(notices.at(-1)![1], "The robot asks: Place the checkerboard (0a1b2c3f) · answer with /approve or /decline"
    + " · Approve = done; decline = end the captures.");
  assert.equal(calls.length, 0, "watching never answers");

  setLanguage("zh");
  try {
    assert.equal(t("cell.approval.waiting", { title: "左臂动作等待批准", id: "0a1b2c3d" }),
      "左臂动作等待批准（0a1b2c3d）· 输入 /approve 或 /decline 答复");
  } finally {
    setLanguage("en");
  }

  const read = commands(folder, exec, { ...CELL, rigMode: "read" }).commands;
  read.watchApprovals(10);
  assert.equal((read as Json).watcher, null, "a read-only rig asks nothing");
  cell.watchApprovals(10);
  assert.notEqual((cell as Json).watcher, null);
  cell.stopWatching();
  assert.equal((cell as Json).watcher, null);
});

test("the question shown: title, id, seconds left, its words and the reason", () => {
  const request = { id: "0a1b2c3d", kind: "release", title: "手臂没能回到起始位", text: "批准 = 就地释放\n\n拒绝 = 继续保持",
    details: {}, created_unix_s: 0, expires_unix_s: 100, pid: 1, asker: "" };
  assert.equal(questionText(request, null, 40),
    "手臂没能回到起始位 · id 0a1b2c3d · 60 s left (no answer counts as a decline)\n批准 = 就地释放\n拒绝 = 继续保持");
  assert.match(questionText(request, "not yet", 140), /· 0 s left[\s\S]*Reason sent with the decline: not yet$/);
  const write = { ...request, kind: "write", text: "Changes: wrist camera\nHeld-out error: 3.1 mm\nConcern: none" };
  assert.equal(questionText(write, null, 40).split("\n").slice(1).join("\n"),
    "Changes: wrist camera\nHeld-out error: 3.1 mm\nConcern: none\nApprove = save; decline = keep the old file.");
  assert.match(questionText({ ...request, kind: "task" }, null, 40), /\nApprove = done; decline = end the captures\.$/);
});

test("a real cell registers /approve and /decline and lists them in /help; elsewhere they are answered", async () => {
  const folder = scratch("tui");
  const previous = process.env.ORA_CELL_APPROVALS;
  process.env.ORA_CELL_APPROVALS = folder;
  try {
    const { exec, calls } = pendingExec();
    const source: OraTuiSource = {
      runRoot: scratch("run"), runId: "run-1", model: "m", profile: "p", protocol: "openai_compatible",
      mutationTools: ["robot.submit_action"], toolNames: ["robot.submit_action"],
      history: () => [], pendingToolImages: () => [], live: () => true, inTurn: () => false,
      stop: () => undefined, interruptAtBoundary: () => undefined, notifySupervisor: () => undefined,
      sessionFile: () => null,
      play: () => ({ state: "awaiting_instruction", scene: 1, scenes: 1, instructions: 0, running: null }),
      playInstruction: () => null, freePlay: true, realCell: CELL,
      control: async () => assert.fail("an answer never goes through the play inbox"),
    };
    const tui = new OraTui(source, { live: true, cellExec: exec });
    const registered = new Map<string, Json>();
    tui.extension()({
      on: () => undefined, registerMessageRenderer: () => undefined, registerToolRenderer: () => undefined,
      registerEntryRenderer: () => undefined, appendEntry: () => undefined,
      registerCommand: (name: string, options: Json) => registered.set(name, options),
    } as Json);
    for (const command of APPROVAL_COMMANDS) assert.equal(registered.get(command.name)?.description, command.description);
    assert.match(tui.helpRows().join("\n"),
      /\/approve \[val\] answer the robot's waiting question[\s\S]*\/approve <value>[\s\S]*\/decline \[why\] answer it: no/);

    post(folder, "0a1b2c3d");
    await registered.get("decline").handler("not now", answering("y"));
    await waitFor(() => calls.length === 1);
    assert.deepEqual(calls[0].args, answerArguments("decline", "0a1b2c3d", CELL, "not now"));

    const sim = new OraTui({ ...source, realCell: null }, { live: true });
    const notices: string[] = [];
    (sim as Json).ui = { notify: (message: string) => notices.push(message) };
    assert.equal(sim.intercept("/approve"), true);
    assert.match(notices.at(-1)!, /\/approve applies to real robot cells only/);
  } finally {
    if (previous === undefined) delete process.env.ORA_CELL_APPROVALS;
    else process.env.ORA_CELL_APPROVALS = previous;
  }
});

test("every approval word has its English and Chinese text", () => {
  const keys = [...APPROVAL_COMMANDS.flatMap((command) =>
    [`cell.confirm.${command.name}.title`, `cell.confirm.${command.name}.message`]),
  "cell.approval.head", "cell.approval.reason", "cell.approval.none", "cell.approval.waiting", "cell.approval.value",
  "cell.approval.hint.task", "cell.approval.hint.value", "cell.approval.hint.write", "cell.approval.unitExample",
  "cell.approval.example", "cell.approval.unit", "cell.approval.needsValue", "cell.approval.noValue"];
  for (const lang of ["en", "zh"] as const) {
    setLanguage(lang);
    try {
      for (const key of keys) assert.notEqual(t(key), key, `${lang} ${key}`);
    } finally {
      setLanguage("en");
    }
  }
});

test("a question the Python broker posts is read here and answered by its id", async () => {
  const folder = scratch("python");
  const env = { ORA_CELL_APPROVALS: folder };
  const asker = spawnPython(["-c", "import json; from ora.cell.operator import approval; print(json.dumps(approval.ask("
    + "'release', '手臂没能回到起始位', '批准 = 就地释放', {'q': [0.1, float('inf')]}, timeout_s=60.0)))"],
  { env, stdio: ["ignore", "pipe", "inherit"] });
  let out = "";
  asker.stdout!.on("data", (chunk: Buffer) => { out += chunk.toString("utf8"); });
  const exited = new Promise((done) => asker.on("close", done));
  try {
    await waitFor(() => pendingApprovals(folder).length === 1, 60_000);
    const [request] = pendingApprovals(folder);
    assert.deepEqual([request.kind, request.title, request.details], ["release", "手臂没能回到起始位", { q: [0.1, null] }]);
    const answer = spawnPython(["-c", "import sys; from ora.cell.operator import approval; "
      + "approval.answer(sys.argv[1], False, 'not supported yet', by='play terminal')", request.id],
    { env, stdio: "inherit" });
    await new Promise((done) => answer.on("close", done));
    await exited;
    assert.deepEqual(JSON.parse(out), [false, "not supported yet"]);
    assert.deepEqual(pendingApprovals(folder), []);
  } finally {
    asker.kill();
  }
});

test("the terminal says once when the worker that held the arms has ended", () => {
  const folder = scratch("worker");
  const lock = join(folder, "ora-cell.lock");
  let living = true;
  const notices: Array<[string, string]> = [];
  const cell = new CellCommands(CELL, { notify: (message, level) => notices.push([level, message]), appendEntry: () => {} },
    { exec: pendingExec().exec, approvals: folder, lock, alive: (pid) => pid === 4242 ? living : true });
  writeFileSync(lock, "ORA worker yam rig_mode on [homes on SIGUSR1] (pid 4242)");
  cell.announce();
  assert.deepEqual(notices, []);
  living = false;
  writeFileSync(lock, "");
  cell.announce();
  cell.announce();
  assert.equal(notices.length, 1);
  assert.equal(notices[0][0], "error");
  assert.equal(notices[0][1], t("cell.worker.gone", { pid: 4242 }));
});

const YOLO_ON = { on: true, kinds: ["motion"], by: "play terminal", since_unix_s: 1, pid: 4242,
  run: "ORA worker yam rig_mode on (pid 4242)", stale: null };
const YOLO_OFF = { on: false, kinds: [], by: null, since_unix_s: null, pid: null, run: null, stale: null };

test("/yolo: the command line of each state, and a name only a real cell answers", () => {
  assert.deepEqual(autoApproveArguments("on", CELL),
    ["cell", "auto-approve", "on", "--json", "--cell", "/cells/bench-a.yaml", "--by", "play terminal"]);
  assert.deepEqual(autoApproveArguments("status", { ...CELL, cellConfig: null }),
    ["cell", "auto-approve", "status", "--json", "--by", "play terminal"]);
  assert.equal(YOLO_COMMAND.verb, "auto-approve");
  assert.ok(CELL_COMMAND_NAMES.has("yolo"));
});

test("/yolo on asks y/N once, then switches; /yolo off and /yolo do not ask", async () => {
  const folder = scratch("yolo");
  const { exec, calls } = pendingExec();
  const { commands: cell, notices, entries } = commands(folder, exec);

  const shown: string[] = [];
  assert.equal(await cell.yolo("on", answering("n", shown)), null);
  assert.deepEqual(notices.at(-1), ["info", "/yolo cancelled; nothing was run."]);
  assert.equal(calls.length, 0);
  assert.match(shown[0], /Turn on auto-approve \(YOLO\)\? \[y\/N\]/);
  assert.match(shown[0], /without asking you[\s\S]*Enabling the arms and releasing them are still asked/);

  const on = await cell.yolo(" ON ", answering("y"));
  assert.deepEqual(calls[0].args, autoApproveArguments("on", CELL));
  calls[0].answer({ code: 0, stdout: JSON.stringify({ ok: true, result: YOLO_ON }) + "\n", stderr: "" });
  const outcome = await on!.done;
  assert.equal(outcome.ok, true);
  assert.match(outcome.text, /^auto-approve is on: motions run without asking \(pid 4242, set by play terminal\)/);
  assert.deepEqual(entries.map(([type, data]) => [type, data.name, data.verb, data.text]),
    [[CELL_ENTRY, "yolo", "auto-approve", outcome.text]]);

  const never = (_tui: Json) => assert.fail("only /yolo on asks");
  const off = await cell.yolo("off", { ui: { custom: never } });
  assert.deepEqual(calls[1].args, autoApproveArguments("off", CELL));
  calls[1].answer({ code: 0, stdout: JSON.stringify({ ok: true, result: YOLO_OFF }), stderr: "" });
  assert.match((await off!.done).text, /^auto-approve is off: every motion waits for \/approve/);

  const status = await cell.yolo("", { ui: { custom: never } });
  assert.deepEqual(calls[2].args, autoApproveArguments("status", CELL));
  calls[2].answer({ code: 1, stdout: JSON.stringify({ ok: false, error: "no play worker drives the arms" }), stderr: "" });
  assert.deepEqual(await status!.done, { ok: false, text: "no play worker drives the arms" });

  assert.equal(await cell.yolo("maybe", answering("y")), null);
  assert.deepEqual(notices.at(-1), ["info", "/yolo on · /yolo off · /yolo alone says whether it is on."]);
  assert.equal(calls.length, 3);
});

test("a real cell registers /yolo and lists it in /help; every /yolo word has English and Chinese", () => {
  const source: OraTuiSource = {
    runRoot: scratch("yolo-run"), runId: "run-1", model: "m", profile: "p", protocol: "openai_compatible",
    mutationTools: ["robot.submit_action"], toolNames: ["robot.submit_action"],
    history: () => [], pendingToolImages: () => [], live: () => true, inTurn: () => false,
    stop: () => undefined, interruptAtBoundary: () => undefined, notifySupervisor: () => undefined,
    sessionFile: () => null,
    play: () => ({ state: "awaiting_instruction", scene: 1, scenes: 1, instructions: 0, running: null }),
    playInstruction: () => null, freePlay: true, realCell: CELL,
    control: async () => assert.fail("never through the play inbox"),
  };
  const tui = new OraTui(source, { live: true, cellExec: pendingExec().exec });
  const registered = new Map<string, Json>();
  tui.extension()({
    on: () => undefined, registerMessageRenderer: () => undefined, registerToolRenderer: () => undefined,
    registerEntryRenderer: () => undefined, appendEntry: () => undefined,
    registerCommand: (name: string, options: Json) => registered.set(name, options),
  } as Json);
  assert.equal(registered.get("yolo")?.description, YOLO_COMMAND.description);
  assert.match(tui.helpRows().join("\n"), /\/yolo \[on\|off\] motions run without asking/);

  for (const lang of ["en", "zh"] as const) {
    setLanguage(lang);
    try {
      for (const key of ["cell.confirm.yolo.title", "cell.confirm.yolo.message", "cell.yolo.on", "cell.yolo.off",
        "cell.yolo.usage"]) assert.notEqual(t(key), key, `${lang} ${key}`);
    } finally {
      setLanguage("en");
    }
  }
});

test("the terminal says each alert about the held arms once, a critical one as an error", () => {
  const folder = scratch("alerts");
  const file = join(folder, "ora-cell-alerts.json");
  const notices: Array<[string, string]> = [];
  const cell = new CellCommands(CELL, { notify: (message, level) => notices.push([level, message]), appendEntry: () => {} },
    { exec: pendingExec().exec, approvals: folder, lock: join(folder, "ora-cell.lock"), alerts: file, alive: () => true });
  const warm = { code: "motor_warm", level: "warning", arm: "right", motor: "夹爪", text: "右臂夹爪电机 52 °C，在升温。" };
  const stopped = { code: "arms_not_controlled", level: "critical", arm: "right", text: "右臂的控制循环停了" };
  writeFileSync(file, JSON.stringify({ pid: 4242, alerts: [warm] }));
  cell.announce();
  cell.announce();
  writeFileSync(file, JSON.stringify({ pid: 4242, alerts: [warm, stopped] }));
  cell.announce();
  assert.deepEqual(notices, [["warning", warm.text], ["error", stopped.text]]);
});

// -- The permission mode (/mode, the footer, Shift+Tab) ------------------------------------------

const MODE_SAFE = { mode: "safe", by: "play terminal", since_unix_s: 1, pid: null, run: null, stale: null,
  safe_limits: { ...SAFE_LIMITS } };
const MODE_AUTO = { mode: "auto", by: "play terminal", since_unix_s: 2, pid: 4242, run: "ORA worker yam rig_mode on (pid 4242)",
  stale: null, safe_limits: { ...SAFE_LIMITS }, falls_back_to: "safe" };

function modeFile(folder: string, mode: string, extra: Json = {}): void {
  writeFileSync(join(folder, "mode.json"), JSON.stringify({ schema: "ora.cell-permission-mode.v1", mode, by: "cell board",
    since_unix_s: 5, safe_limits: { ...SAFE_LIMITS }, ...extra }));
}

function autoFile(folder: string, pid: number, kinds = ["motion"]): void {
  writeFileSync(join(folder, "auto-approve.json"), JSON.stringify({ schema: "ora.cell-auto-approve.v1", kinds,
    by: "web page", since_unix_s: 7, pid, run: `ORA worker yam rig_mode on (pid ${pid})` }));
}

test("the mode is read from the broker's files by approval.py's rule", () => {
  const folder = scratch("mode-state");
  const alive = (pid: number) => pid === 4242;
  assert.deepEqual(cellModeState(join(folder, "absent"), alive), { mode: "ask", by: null, since_unix_s: null, pid: null,
    run: null, stale: null, safe_limits: { ...SAFE_LIMITS } });
  modeFile(folder, "safe", { safe_limits: { max_duration_s: 4, min_table_distance_m: 0.01, max_joint_travel_rad: 9 } });
  const safe = cellModeState(folder, alive);
  assert.equal(safe.mode, "safe");
  assert.equal(safe.by, "cell board");
  // A stored limit may only be stricter than the built-in one.
  assert.deepEqual(safe.safe_limits, { ...SAFE_LIMITS, max_duration_s: 4 });
  modeFile(folder, "auto");                                        // never stored: read as ask
  assert.equal(cellModeState(folder, alive).mode, "ask");
  modeFile(folder, "safe");
  autoFile(folder, 4242);
  assert.deepEqual(cellModeState(folder, alive), { mode: "auto", by: "web page", since_unix_s: 7, pid: 4242,
    run: "ORA worker yam rig_mode on (pid 4242)", stale: null, safe_limits: { ...SAFE_LIMITS }, falls_back_to: "safe" });
  autoFile(folder, 5151);                                          // its worker ended: back to the cell's mode
  const ended = cellModeState(folder, alive);
  assert.equal(ended.mode, "safe");
  assert.match(ended.stale!, /the worker it was set for \(pid 5151\) has ended/);
  assert.equal(nextMode("ask", "on"), "safe");
  assert.equal(nextMode("safe", "on"), "auto");
  assert.equal(nextMode("safe", "read"), "ask", "auto needs a worker that drives the arms");
  assert.equal(nextMode("auto", "on"), "ask");
});

test("the mode and a self-answered motion the Python broker writes are read here as Python reads them", async () => {
  const folder = scratch("mode-python");
  const child = spawnPython(["-c", "import json; from ora.cell.operator import approval; "
    + "state = approval.switch_mode('safe', by='cell board'); "
    + "ok = approval.ask('motion', '左臂动作', '下降', {'duration_s': 2.5, 'max_joint_travel_rad': {'left': 0.3}, "
    + "'min_table_distance_m': 0.08, 'min_inter_arm_distance_m': 0.4}, timeout_s=5.0); "
    + "print(json.dumps({'state': approval.mode_state(), 'ok': ok}))"],
  { env: { ORA_CELL_APPROVALS: folder }, stdio: ["ignore", "pipe", "inherit"] });
  let out = "";
  child.stdout!.on("data", (chunk: Buffer) => { out += chunk.toString("utf8"); });
  await new Promise((done) => child.on("close", done));
  const { state, ok } = JSON.parse(out);
  assert.deepEqual(ok, [true, null]);
  assert.deepEqual(cellModeState(folder), state);
  assert.deepEqual(selfAnswers(folder).map((item) => [item.kind, item.title, item.by]),
    [["motion", "左臂动作", "safe mode (2.5 s, 0.30 rad, 8 cm above the table, 40 cm between the arms)"]]);
});

test("/mode: the command line, the y/N only for auto, the mode in plain words with safe's limits", async () => {
  assert.deepEqual(modeArguments("safe", CELL),
    ["cell", "mode", "safe", "--json", "--cell", "/cells/bench-a.yaml", "--by", "play terminal"]);
  assert.ok(CELL_COMMAND_NAMES.has("mode"));
  const folder = scratch("mode");
  const { exec, calls } = pendingExec();
  const { commands: cell, notices, entries } = commands(folder, exec);
  const never = (_tui: Json) => assert.fail("only /mode auto asks");

  const safe = await cell.mode(" Safe ", { ui: { custom: never } });
  assert.deepEqual(calls[0].args, modeArguments("safe", CELL));
  calls[0].answer({ code: 0, stdout: JSON.stringify({ ok: true, result: MODE_SAFE }), stderr: "" });
  const text = (await safe!.done).text;
  assert.equal(text, "mode safe: a small motion runs without asking (at most 10 s and 0.6 rad per joint, at least 3 cm "
    + "above the table and 15 cm between the arms); any other waits for /approve (set by play terminal); enabling and "
    + "releasing the arms are always asked");
  assert.deepEqual(entries.map(([type, data]) => [type, data.name, data.verb]), [[CELL_ENTRY, "mode", "mode"]]);

  const shown: string[] = [];
  assert.equal(await cell.mode("auto", answering("n", shown)), null);
  assert.match(shown[0], /Switch to auto: every motion runs without asking\? \[y\/N\]/);
  assert.deepEqual(notices.at(-1), ["info", "/mode cancelled; nothing was run."]);
  const auto = await cell.mode("auto", answering("y"));
  assert.deepEqual(calls[1].args, modeArguments("auto", CELL));
  calls[1].answer({ code: 1, stdout: JSON.stringify({ ok: false, error: "no play worker drives the arms (rig_mode on) now" }),
    stderr: "" });
  assert.deepEqual(await auto!.done, { ok: false, text: "no play worker drives the arms (rig_mode on) now" });

  modeFile(folder, "safe");
  assert.equal(await cell.mode("", { ui: { custom: never } }), null, "the mode now: read from the files, nothing runs");
  assert.match(notices.at(-1)![1], /^mode safe: a small motion runs without asking/);
  assert.equal(await cell.mode("yolo", answering("y")), null);
  assert.match(notices.at(-1)![1], /^\/mode ask · \/mode safe · \/mode auto/);
  assert.equal(calls.length, 2);
  assert.match(modeText(JSON.stringify(MODE_AUTO)),
    /^mode auto: every motion runs without asking \(pid 4242, set by play terminal\); back to safe when that worker ends/);
  assert.equal(modeText("not json"), "not json");
});

test("a motion that answered itself is said once, with the numbers safe mode checked", () => {
  const folder = scratch("self");
  const notices: Array<[string, string]> = [];
  const cell = new CellCommands(CELL, { notify: (message, level) => notices.push([level, message]), appendEntry: () => {} },
    { exec: pendingExec().exec, approvals: folder, lock: join(folder, "ora-cell.lock"), alerts: join(folder, "a.json"),
      alive: () => true });
  const answer = (id: string, by: string, extra: Json = {}) => writeFileSync(join(folder, `${id}.answer.json`), JSON.stringify({
    schema: "ora.cell-approval-answer.v1", id, approved: true, reason: null, by, answered_unix_s: Date.now() / 1000 + 1,
    kind: "motion", title: "左臂动作：下降 4 cm", ...extra }));
  const by = "safe mode (2.5 s, 0.30 rad, 8 cm above the table, 40 cm between the arms)";
  answer("0000000a", by);
  answer("0000000b", "play terminal");                             // a person's answer: said by its own command
  answer("0000000c", "auto-approve (web page)", { answered_unix_s: 1 });   // before this terminal opened
  cell.announce();
  cell.announce();
  assert.deepEqual(notices, [["info", `Ran without asking: 左臂动作：下降 4 cm · ${by}`]]);
  assert.deepEqual(selfAnswers(folder).map((item) => item.id), ["0000000c", "0000000a"]);
});

test("the footer shows the mode, auto apart; Shift+Tab cycles it on a real cell; the page's /mode is noted", async () => {
  const folder = scratch("mode-footer");
  const previous = process.env.ORA_CELL_APPROVALS;
  process.env.ORA_CELL_APPROVALS = folder;
  try {
    const source: OraTuiSource = {
      runRoot: scratch("mode-run"), runId: "run-1", model: "m", profile: "p", protocol: "openai_compatible",
      mutationTools: ["robot.submit_action"], toolNames: ["robot.submit_action"],
      history: () => [], pendingToolImages: () => [], live: () => true, inTurn: () => false,
      stop: () => undefined, interruptAtBoundary: () => undefined, notifySupervisor: () => undefined,
      sessionFile: () => null,
      play: () => ({ state: "awaiting_instruction", scene: 1, scenes: 1, instructions: 0, running: null }),
      playInstruction: () => null, freePlay: true, realCell: CELL,
      control: async () => assert.fail("never through the play inbox"),
    };
    const { exec, calls } = pendingExec();
    const tui = new OraTui(source, { live: true, cellExec: exec });
    const registered = new Map<string, Json>();
    tui.extension()({
      on: () => undefined, registerMessageRenderer: () => undefined, registerToolRenderer: () => undefined,
      registerEntryRenderer: () => undefined, appendEntry: () => undefined,
      registerCommand: (name: string, options: Json) => registered.set(name, options),
    } as Json);
    assert.equal(registered.get("mode")?.description, MODE_COMMAND.description);
    assert.deepEqual(registered.get("mode").getArgumentCompletions("s").map((item: Json) => item.value), ["safe", "status"]);
    assert.match(tui.helpRows().join("\n"), /\/mode \[m\][\s\S]*Shift\+Tab      the next mode/);
    assert.equal(tui.modeLine()?.text, "▸▸ ask · every motion asked (shift+tab)");

    assert.deepEqual(tui.onKey("\x1b[Z"), { consume: true }, "Shift+Tab is taken before Pi's thinking cycle");
    await waitFor(() => calls.length === 1);
    assert.deepEqual(calls[0].args, modeArguments("safe", CELL));
    modeFile(folder, "safe");
    calls[0].answer({ code: 0, stdout: JSON.stringify({ ok: true, result: MODE_SAFE }), stderr: "" });
    await waitFor(() => tui.cell!.modeState(true).mode === "safe");
    assert.equal(tui.modeLine()?.text, "▸▸ safe · small motions run (shift+tab)");

    const entry = tui.cell!.note({ name: "mode", verb: "mode", ok: true, text: JSON.stringify(MODE_AUTO),
      command: "scripts/ora cell mode auto --json --by 'web page'" });
    assert.match(entry.text, /^mode auto: every motion runs without asking/);

    const plain = new OraTui({ ...source, realCell: null }, { live: true, cellExec: exec });
    assert.equal(plain.modeLine(), null);
    assert.equal(plain.onKey("\x1b[Z"), undefined, "elsewhere Shift+Tab stays Pi's");
  } finally {
    if (previous === undefined) delete process.env.ORA_CELL_APPROVALS;
    else process.env.ORA_CELL_APPROVALS = previous;
  }
});

test("every /mode word has English and Chinese", () => {
  for (const lang of ["en", "zh"] as const) {
    setLanguage(lang);
    try {
      for (const key of ["cell.confirm.mode.title", "cell.confirm.mode.message", "cell.mode.label.ask", "cell.mode.label.safe",
        "cell.mode.label.auto", "cell.mode.cycle", "cell.mode.state.ask", "cell.mode.state.safe", "cell.mode.state.auto",
        "cell.mode.limits", "cell.mode.setBy", "cell.mode.always", "cell.mode.usage", "cell.mode.busy", "cell.selfAnswered"]) {
        assert.notEqual(t(key), key, `${lang} ${key}`);
      }
    } finally {
      setLanguage("en");
    }
  }
});
