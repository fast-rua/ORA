// The operator's verbs for a real robot cell, from free play's terminal.
//
// `/home`, `/hold`, `/float`, `/release`, `/estop` and `/diagnose` act on the
// arms at once, also while the robot executes a motion, so they do not go
// through the play inbox (the Supervisor reads it only between model
// decisions). The terminal runs the same command the cell board's buttons and
// the shell use, `scripts/ora cell <verb> --json` (ora/cell/operator/actions.py
// implements each verb once), and never hands it to the model.
//
// They exist only when the run drives a real cell: the plan's runtime is a
// real-cell backend (today `yam`) on rig_mode `read` or `on`. In `sim` they
// would act on hardware the run does not use.
//
// `/approve [value]` and `/decline [reason]` answer the question the robot's
// worker waits on (enable the arms, a motion, releasing arms that could not get
// home) or a guided calibration asks (`task`: done by hand; `value`: a measured
// value, `/approve <value>`; `write`: save the result). The asker posts it
// through the cell's approval broker (ora/cell/operator/approval.py) as
// `<id>.request.json`; the terminal reads those files itself
// (`pendingApprovals`, the broker's own rule) and answers with `ora cell
// approve|decline --id <id>` (`--value <v>` for a value question), so an answer
// always names the question the operator saw. There is no `/calibrate`: this
// run's worker holds the cell lock calibration needs; it starts from the cell
// board or `ora cell calibrate`.
//
// `/yolo on|off` is the broker's auto-approve switch (`ora cell auto-approve`):
// on, the worker driving the arms answers its own motion questions (each still
// recorded, `by: "auto-approve (play terminal)"`); enabling and releasing the
// arms are still asked. `/yolo` alone says whether it is on.
//
// `/mode ask|safe|auto` is the cell's permission mode (`ora cell mode`): `ask`
// (every motion waits for /approve), `safe` (a motion whose plan stays inside
// the broker's safe limits runs without asking, any other waits) or `auto`
// (every motion of the worker driving the arms, as /yolo on). `ask` and `safe`
// belong to the cell and stay across runs; `auto` belongs to the running worker
// and falls back to the cell's mode when it ends. The terminal reads the mode
// from the broker's own files (`cellModeState`, approval.py's `mode_state`
// rule), with no process per look, for its footer and Shift+Tab.

import { spawn } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

type Json = any;

/** Runtime kinds whose rig can be a real cell (ora/environment/benchmarks/yam.py RIG_MODES). */
const REAL_CELL_KINDS = new Set(["yam"]);
/** Rig modes on the real hardware: `read` (cameras and encoders) and `on` (the arms move). */
const REAL_RIG_MODES = new Set(["read", "on"]);

/** The project checkout this core runs from: `scripts/ora` is its wrapper of the `ora` command. */
export const PROJECT_ROOT = resolve(import.meta.dirname, "../../..");

export interface CellCommand {
  /** The slash command (without the slash). */
  name: string;
  /** The `ora cell` verb it runs. */
  verb: string;
  /** Asked y/N in the terminal first; `--yes` is passed only after a `y`. */
  confirm: boolean;
  /** Pi's autocomplete line. */
  description: string;
}

export const CELL_COMMANDS: readonly CellCommand[] = [
  { name: "home", verb: "home", confirm: false, description: "Real cell: drive the arms to rest now (ora cell home)" },
  { name: "hold", verb: "hold", confirm: false, description: "Real cell: hold the arms where they are now (ora cell hold)" },
  { name: "float", verb: "float", confirm: true,
    description: "Real cell: float the arms for guiding by hand, after a y/N (ora cell float)" },
  { name: "release", verb: "release", confirm: true,
    description: "Real cell: let the arms go limp, after a y/N (ora cell release)" },
  { name: "estop", verb: "estop", confirm: true,
    description: "Real cell: disable every motor now, after a y/N; the arms fall (ora cell estop)" },
  { name: "diagnose", verb: "diagnose", confirm: false,
    description: "Real cell: who holds the cell, buses, faults, distance from rest (ora cell diagnose)" },
];

/** Answers to the worker's question; each shows the question and asks y/N for that id first. */
export const APPROVAL_COMMANDS: readonly CellCommand[] = [
  { name: "approve", verb: "approve", confirm: true,
    description: "Real cell: approve the robot's waiting question (enable, motion, release, task, write; /approve <value> for a value), after a y/N" },
  { name: "decline", verb: "decline", confirm: true,
    description: "Real cell: decline the robot's waiting question, after a y/N (/decline [reason])" },
];

/** The auto-approve switch; only `/yolo on` asks y/N. */
export const YOLO_COMMAND: CellCommand = {
  name: "yolo", verb: "auto-approve", confirm: true,
  description: "Real cell: /yolo on: motions run without asking, after a y/N; /yolo off; /yolo: whether it is on (ora cell auto-approve)",
};

/** The permission mode; only `auto` asks y/N. */
export const MODE_COMMAND: CellCommand = {
  name: "mode", verb: "mode", confirm: true,
  description: "Real cell: /mode ask|safe|auto: every motion asked, small ones run, or none asked (auto after a y/N); /mode: the mode now; Shift+Tab cycles (ora cell mode)",
};

/**
 * Stop here: the motion the worker executes ends where the arms are and they stay held there; the agent
 * gets a fresh observation (`ora cell stop-motion`, ora/cell/operator/approval.py). Never asked: it holds,
 * it never drops. Between motions it does nothing.
 */
export const HALT_COMMAND: CellCommand = {
  name: "halt", verb: "stop-motion", confirm: false,
  description: "Real cell: end the running motion where the arms are and hold there; the agent gets a fresh observation (ora cell stop-motion)",
};

export const CELL_COMMAND_NAMES: ReadonlySet<string> = new Set(
  [...CELL_COMMANDS, ...APPROVAL_COMMANDS, YOLO_COMMAND, MODE_COMMAND, HALT_COMMAND].map((command) => command.name));

/** The real cell a free-play run drives. */
export interface RealCell {
  kind: string;
  rigMode: string;
  /** The plan's `runtime.cell_config`, passed as `--cell`; null when the plan names none. */
  cellConfig: string | null;
}

/** The real cell of a plan, or null (a simulator, or a real-cell backend on rig_mode `sim`). */
export function realCellOf(plan: Json): RealCell | null {
  const runtime = plan?.runtime;
  if (runtime === null || typeof runtime !== "object") return null;
  const kind = String(runtime.kind ?? "");
  const rigMode = String(runtime.rig_mode ?? "sim");
  if (!REAL_CELL_KINDS.has(kind) || !REAL_RIG_MODES.has(rigMode)) return null;
  const config = runtime.cell_config;
  return { kind, rigMode, cellConfig: typeof config === "string" && config.trim() ? config : null };
}

/** The real cell of the run at `runRoot` (its plan.json), or null; a run without a plan has none. */
export function readRealCell(runRoot: string): RealCell | null {
  let text: string;
  try {
    text = readFileSync(join(runRoot, "plan.json"), "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code === "ENOENT") return null;
    throw error;
  }
  return realCellOf(JSON.parse(text));
}

/** A question the worker waits on, as the broker writes it (`ora.cell-approval-request.v1`). */
export interface ApprovalRequest {
  id: string;
  kind: string;
  title: string;
  text: string;
  details: Json;
  created_unix_s: number;
  expires_unix_s: number;
  pid: number;
  asker: string;
}

/** The broker's folder: `$ORA_CELL_APPROVALS`, else `ora-cell-approvals` beside the cell lock. */
export function approvalsDirectory(env: NodeJS.ProcessEnv = process.env): string {
  const named = env.ORA_CELL_APPROVALS;
  if (named) return named === "~" || named.startsWith("~/") ? join(homedir(), named.slice(1)) : named;
  return join(dirname(env.ORA_CELL_LOCK ?? "/tmp/ora-cell.lock"), "ora-cell-approvals");
}

/** The cell lock: `$ORA_CELL_LOCK`, else `/tmp/ora-cell.lock`. */
export function cellLockPath(env: NodeJS.ProcessEnv = process.env): string {
  return env.ORA_CELL_LOCK ?? "/tmp/ora-cell.lock";
}

/** The held arms' alerts (ora/cell/alerts.py): `$ORA_CELL_ALERTS`, else beside the cell lock. */
export function alertsPath(env: NodeJS.ProcessEnv = process.env): string {
  return env.ORA_CELL_ALERTS ?? join(dirname(cellLockPath(env)), "ora-cell-alerts.json");
}

export interface CellAlert {
  code: string;
  level: "warning" | "critical";
  text: string;
  arm?: string;
  motor?: string;
}

/** The alerts the program holding the arms publishes now; a dead publisher's only as far as it says the arms are uncontrolled. Never throws. */
export function cellAlerts(path: string = alertsPath(), alive: (pid: number) => boolean = (pid) => {
  try { process.kill(pid, 0); return true; } catch (error) { return (error as NodeJS.ErrnoException)?.code === "EPERM"; }
}): CellAlert[] {
  let body: Json;
  try {
    body = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return [];
  }
  const live = Number.isInteger(body?.pid) && alive(body.pid);
  return (Array.isArray(body?.alerts) ? body.alerts : [])
    .filter((a: Json) => a && (a.level === "warning" || a.level === "critical") && typeof a.text === "string")
    .filter((a: Json) => live || a.code === "arms_not_controlled");
}

const WORKER_HOLDER = /ORA (?:YAM )?worker .*\(pid (\d+)\)/i;

/** The pid of the play worker that holds the cell, from the lock's label; null for any other holder. Never throws. */
export function cellWorkerPid(lock: string = cellLockPath()): number | null {
  try {
    const found = WORKER_HOLDER.exec(readFileSync(lock, "utf8"));
    return found ? Number(found[1]) : null;
  } catch {
    return null;
  }
}

/** Whether a process lives (another user's process does). */
export type Alive = (pid: number) => boolean;

export const processAlive: Alive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException)?.code === "EPERM";
  }
};

const REQUEST_ID = /^[0-9a-f]{8}$/;

/**
 * The questions waiting for an answer, oldest first, by the broker's rule: no answer yet, not
 * expired, and the asker lives (a dead worker's question is stale). Reads only; never throws.
 */
export function pendingApprovals(folder: string = approvalsDirectory(), now: number = Date.now() / 1000,
                                 alive: Alive = processAlive): ApprovalRequest[] {
  let names: string[];
  try {
    names = readdirSync(folder);
  } catch {
    return [];
  }
  const waiting: ApprovalRequest[] = [];
  for (const name of names) {
    if (name.startsWith(".") || !name.endsWith(".request.json")) continue;
    let request: Json;
    try {
      request = JSON.parse(readFileSync(join(folder, name), "utf8"));
    } catch {
      continue;  // being replaced, or not the broker's
    }
    const id = request?.id;
    if (typeof id !== "string" || !REQUEST_ID.test(id)) continue;
    if (typeof request.expires_unix_s !== "number" || now >= request.expires_unix_s) continue;
    if (existsSync(join(folder, `${id}.answer.json`))) continue;
    if (!Number.isInteger(request.pid) || request.pid <= 0 || !alive(request.pid)) continue;
    waiting.push(request as ApprovalRequest);
  }
  return waiting.sort((a, b) => (a.created_unix_s ?? 0) - (b.created_unix_s ?? 0));
}

/** `ora cell approve|decline` for one question, by its id; a value question's approval carries `--value`. */
export function answerArguments(verb: string, id: string, cell: RealCell, reason: string | null = null,
                                value: string | null = null): string[] {
  return ["cell", verb, "--json", ...(cell.cellConfig === null ? [] : ["--cell", cell.cellConfig]),
    "--id", id, "--by", "play terminal", ...(verb === "decline" && reason ? ["--reason", reason] : []),
    ...(verb === "approve" && value ? ["--value", value] : [])];
}

/** `ora cell auto-approve on|off|status`, said to come from this terminal. */
export function autoApproveArguments(state: "on" | "off" | "status", cell: RealCell): string[] {
  return ["cell", "auto-approve", state, "--json", ...(cell.cellConfig === null ? [] : ["--cell", cell.cellConfig]),
    "--by", "play terminal"];
}

/** `ora cell stop-motion`, said to come from this terminal. */
export function haltArguments(cell: RealCell): string[] {
  return ["cell", "stop-motion", "--json", ...(cell.cellConfig === null ? [] : ["--cell", cell.cellConfig]),
    "--by", "play terminal"];
}

/** `ora cell stop-motion --withdraw`: take back a stop nobody has read yet. */
export function withdrawStopArguments(cell: RealCell): string[] {
  return ["cell", "stop-motion", "--withdraw", "--json", ...(cell.cellConfig === null ? [] : ["--cell", cell.cellConfig])];
}

/** How long the worker honours a stop (ora/cell/operator/approval.py STOP_REQUEST_S). */
export const STOP_REQUEST_MS = 10_000;

/**
 * A stop sent while the agent plans (no motion running): Esc, or a page's Stop & send. The worker reads it
 * at the approval of the motion the reply proposes and refuses that motion before anything moves. A reply
 * that proposes none leaves it unread, and it would refuse the agent's next motion instead: once the
 * operator's message reaches the agent (its `message_delivered`), or the stopped instruction ends, with no
 * motion started in between, the front end withdraws it (`ora cell stop-motion --withdraw`). A motion that
 * started read it (refused, or ended where the arms were); after STOP_REQUEST_MS the worker ignores it.
 */
export class PlanStop {
  private pending: { text: string | null; at: number } | null = null;
  private readonly now: () => number;

  constructor(now: () => number = Date.now) {
    this.now = now;
  }

  /** The stop went out; `text` is the message sent with it (Stop & send), null for Esc alone. */
  sent(text: string | null = null): void {
    this.pending = { text: text === null ? null : text.trim(), at: this.now() };
  }

  get waiting(): boolean {
    return this.pending !== null && this.now() - this.pending.at <= STOP_REQUEST_MS;
  }

  /** A free-play state (ora.play-state.v1): true when the stop is to be withdrawn now. */
  settle(record: Json): boolean {
    const pending = this.pending;
    if (pending === null) return false;
    if (this.now() - pending.at > STOP_REQUEST_MS) {
      this.pending = null;                  // the worker ignores it now
      return false;
    }
    const state = record?.state;
    if (state === "action_started") {
      this.pending = null;                  // that motion read it
      return false;
    }
    const delivered = state === "message_delivered" && pending.text !== null
      && String(record.message?.text ?? "").trim() === pending.text;
    if (delivered || ["instruction_finished", "scene", "scene_ended", "ended"].includes(state)) {
      this.pending = null;
      return true;
    }
    return false;
  }
}

/** The permission modes, in the order Shift+Tab cycles them. */
export const PERMISSION_MODES = ["ask", "safe", "auto"] as const;
export type PermissionMode = typeof PERMISSION_MODES[number];

/** `ora cell mode ask|safe|auto|status`, said to come from this terminal. */
export function modeArguments(state: PermissionMode | "status", cell: RealCell): string[] {
  return ["cell", "mode", state, "--json", ...(cell.cellConfig === null ? [] : ["--cell", cell.cellConfig]),
    "--by", "play terminal"];
}

/** A motion answers itself in `safe` mode only inside all of these (approval.py SAFE_LIMITS). */
export const SAFE_LIMITS = { max_duration_s: 10.0, max_joint_travel_rad: 0.6, min_table_distance_m: 0.03,
  min_inter_arm_distance_m: 0.15 } as const;
export type SafeLimits = { -readonly [key in keyof typeof SAFE_LIMITS]: number };

/** The permission mode as `ora cell mode status --json` says it (approval.py `mode_state`). */
export interface ModeState {
  mode: PermissionMode;
  by: string | null;
  since_unix_s: number | null;
  pid: number | null;
  run: string | null;
  stale: string | null;
  safe_limits: SafeLimits;
  /** In `auto`: the cell's own mode, back once the worker ends. */
  falls_back_to?: "ask" | "safe";
}

function readJson(path: string): Json | null {
  try {
    const value = JSON.parse(readFileSync(path, "utf8"));
    return value !== null && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

/** The stored limits; a stored one may only be stricter than the built-in one (approval.py `_limits`). */
function limitsOf(stored: Json): SafeLimits {
  const limits: SafeLimits = { ...SAFE_LIMITS };
  if (stored === null || typeof stored !== "object") return limits;
  for (const key of Object.keys(SAFE_LIMITS) as Array<keyof SafeLimits>) {
    const value = stored[key];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    limits[key] = key.startsWith("max_") ? Math.min(limits[key], value) : Math.max(limits[key], value);
  }
  return limits;
}

/**
 * The cell's permission mode from the broker's files, by approval.py's `mode_state` rule: `mode.json`
 * (`ask` or `safe`, else `ask`), and `auto` while `auto-approve.json` names a motion switch whose worker
 * lives. Reads only; never throws.
 */
export function cellModeState(folder: string = approvalsDirectory(), alive: Alive = processAlive): ModeState {
  const flag = readJson(join(folder, "mode.json"));
  const chosen = flag?.schema === "ora.cell-permission-mode.v1" && (flag.mode === "ask" || flag.mode === "safe") ? flag : null;
  const state: ModeState = { mode: chosen?.mode ?? "ask", by: chosen?.by ?? null, since_unix_s: chosen?.since_unix_s ?? null,
    pid: null, run: null, stale: null, safe_limits: limitsOf(chosen?.safe_limits) };
  const auto = readJson(join(folder, "auto-approve.json"));
  if (auto?.schema !== "ora.cell-auto-approve.v1") return state;
  const kinds = Array.isArray(auto.kinds) && auto.kinds.includes("motion");
  const pid = auto.pid;
  const living = Number.isInteger(pid) && pid > 0 && alive(pid);
  if (kinds && living) {
    return { ...state, mode: "auto", by: auto.by ?? null, since_unix_s: auto.since_unix_s ?? null, pid, run: auto.run ?? null,
      falls_back_to: state.mode as "ask" | "safe" };
  }
  if (auto.pid !== null && auto.pid !== undefined) {
    state.stale = "auto-approve was set for a worker that has ended: "
      + (kinds ? `the worker it was set for (pid ${pid}) has ended` : "it names no question that can be auto-approved");
  }
  return state;
}

/** The mode Shift+Tab goes to next: ask → safe → auto → ask; `auto` only where a worker drives the arms (`on`). */
export function nextMode(mode: PermissionMode, rigMode: string): PermissionMode {
  if (mode === "ask") return "safe";
  if (mode === "safe") return rigMode === "on" ? "auto" : "ask";
  return "ask";
}

/** A motion that answered itself (safe mode or auto-approve), as its answer file records it. */
export interface SelfAnswer {
  id: string;
  kind: string;
  title: string;
  by: string;
  answered_unix_s: number;
}

/**
 * The questions that answered themselves since `since` (seconds), oldest first: approved answers with
 * the question's `kind` and `title` and `by: "safe mode (...)"` or `"auto-approve (...)"`. Reads only;
 * never throws.
 */
export function selfAnswers(folder: string = approvalsDirectory(), since = 0): SelfAnswer[] {
  let names: string[];
  try {
    names = readdirSync(folder);
  } catch {
    return [];
  }
  const found: SelfAnswer[] = [];
  for (const name of names) {
    if (name.startsWith(".") || !name.endsWith(".answer.json")) continue;
    const answer = readJson(join(folder, name));
    if (answer?.approved !== true || typeof answer.by !== "string" || typeof answer.kind !== "string") continue;
    if (!/^(safe mode|auto-approve) \(/.test(answer.by)) continue;
    if (typeof answer.answered_unix_s !== "number" || answer.answered_unix_s < since) continue;
    if (typeof answer.id !== "string" || !REQUEST_ID.test(answer.id)) continue;
    found.push({ id: answer.id, kind: answer.kind, title: String(answer.title ?? ""), by: answer.by,
      answered_unix_s: answer.answered_unix_s });
  }
  return found.sort((a, b) => a.answered_unix_s - b.answered_unix_s);
}

/** `ora cell` arguments: `--yes` only after the terminal's own confirmation. */
export function cellArguments(verb: string, cell: RealCell, confirmed: boolean): string[] {
  return ["cell", verb, "--json", ...(cell.cellConfig === null ? [] : ["--cell", cell.cellConfig]),
    ...(confirmed ? ["--yes"] : [])];
}

/** A command line as the operator would type it (for the trace and the terminal). */
export function commandLine(file: string, args: string[]): string {
  const quote = (word: string) => (/^[\w@%+=:,./-]+$/.test(word) ? word : `'${word.replace(/'/g, "'\\''")}'`);
  return [file, ...args].map(quote).join(" ");
}

export interface CellProcessResult {
  /** The exit code; null when a signal ended the process or it could not start. */
  code: number | null;
  stdout: string;
  stderr: string;
}

/** Runs `file args` in `cwd`; resolves when it exits (it never rejects). */
export type CellExec = (file: string, args: string[], cwd: string) => Promise<CellProcessResult>;

/** The real process: no stdin (it can never wait on the operator's terminal), stdout and stderr kept. */
export const spawnCell: CellExec = (file, args, cwd) => new Promise((done) => {
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  let child: ReturnType<typeof spawn>;
  try {
    child = spawn(file, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
  } catch (error) {
    done({ code: null, stdout: "", stderr: (error as Error).message });
    return;
  }
  child.stdout?.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr?.on("data", (chunk: Buffer) => stderr.push(chunk));
  let settled = false;
  const finish = (code: number | null, extra = "") => {
    if (settled) return;
    settled = true;
    done({ code, stdout: Buffer.concat(stdout).toString("utf8"),
      stderr: Buffer.concat(stderr).toString("utf8") + extra });
  };
  child.on("error", (error: Error) => finish(null, error.message));
  child.on("close", (code: number | null, signal: string | null) => finish(code, signal ? `ended by ${signal}` : ""));
});

/** What a cell command came to, as the terminal says it. */
export interface CellOutcome {
  ok: boolean;
  /** The verb's result (ok) or the reason it failed; may span lines (diagnose). */
  text: string;
}

function firstLine(text: string): string {
  return (text.split("\n").find((line) => line.trim()) ?? "").trim();
}

/** The JSON object `ora cell --json` prints: the whole of stdout, else its last line. */
function answerOf(stdout: string): Json | null {
  const candidates = [stdout.trim(), stdout.trim().split("\n").at(-1) ?? ""];
  for (const candidate of candidates) {
    try {
      const value = JSON.parse(candidate);
      if (value !== null && typeof value === "object" && !Array.isArray(value)) return value;
    } catch {
      // not the answer; try the next candidate
    }
  }
  return null;
}

function resultText(result: Json): string {
  if (result === undefined || result === null) return "";
  return typeof result === "string" ? result : JSON.stringify(result, null, 2);
}

/** `{"ok": true, "result": ...}` with exit code 0 is a success; anything else says why it failed. */
export function cellOutcome(result: CellProcessResult): CellOutcome {
  const answer = answerOf(result.stdout);
  if (answer?.ok === true && result.code === 0) return { ok: true, text: resultText(answer.result) };
  if (answer?.ok === false) return { ok: false, text: String(answer.error ?? "the cell refused, without a reason") };
  const why = firstLine(result.stderr) || firstLine(result.stdout) || "no answer";
  return { ok: false, text: result.code === null ? why : `exit code ${result.code}: ${why}` };
}
