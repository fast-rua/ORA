// Free play's real-cell commands in Pi's terminal (play/cell.ts says which
// commands, when a run has them, and how `ora cell` answers).
//
// A command that lets the arms move or fall asks y/N first, and only `y`
// proceeds (with `--yes`, so `ora cell` does not ask again). Then it starts
// `scripts/ora cell <verb> --json` and returns at once: Pi awaits a command's
// handler before it reads the next line, and /estop must be typeable while
// /home is still running. The terminal says "running…", then the result or the
// error as a display-only entry in the conversation (`ora.cell_command`; Pi
// never sends custom entries to the model). The run's agent trace records the
// command line and its output as the operator's shell command
// (`operator_shell_command` / `operator_shell_result`, excluded from the
// model's context, as a `!!` line is).
//
// /approve [value] and /decline [reason] answer the question the robot's
// worker waits on: the terminal shows that question (title, plain words, id,
// seconds left, what approving and declining mean for a `task`, `value` or
// `write` question, the reason or value sent), and a `y` runs `ora cell
// approve|decline --id <id>` for exactly that id. A `value` question is
// approved with `/approve <value>` (`--value`); /approve without one, or a
// value for any other question, is refused here and runs nothing. Approving is
// the confirmation itself; nothing asks again. While an `on` run is open the
// terminal also watches the broker's folder (every 0.5 s) and says once when a
// new question appears.
//
// /yolo on|off flips the broker's auto-approve switch (`ora cell auto-approve`);
// only `on` asks y/N. The entry says the switch's state in plain words.
//
// /mode ask|safe|auto sets the permission mode (`ora cell mode`); only `auto`
// asks y/N, and the entry says the mode in plain words (`safe` with its
// limits). /mode alone says the mode from the broker's files, as the footer
// shows it, with no process. Shift+Tab (tui/extension.ts) cycles ask → safe →
// auto → ask through the same path. The terminal also says once each motion
// that answered itself (safe mode or auto-approve), with the plan's numbers.
//
// /halt ends the motion the worker executes now where the arms are and keeps
// them held there (`ora cell stop-motion`; never asked: it holds, it never
// drops); the agent gets a fresh observation. Esc while the robot moves calls
// it before it stops the instruction (tui/extension.ts onKey). Esc while the
// agent plans sends the same stop (`stopPlan`): the worker refuses the motion
// that reply proposes at its approval; when the reply proposes none, the stop
// is taken back (`ora cell stop-motion --withdraw`, play/cell.ts PlanStop) once
// the instruction ends with no motion started, so it never refuses a later one.
//
// `ora web`'s page runs the same commands from its own buttons and typed
// commands (ora/app/web_cell.py, in the gateway) and hands each outcome here
// (`note`, the run's Pi RPC `ora_cell_note`): the conversation shows it as
// this terminal's own, marked *web page*, and the agent trace records its
// command line and output the same way. Every outcome, this terminal's and the
// page's, goes to the run's RPC clients too (`onOutcome`: the `ora_cell_command`
// event), so every open page lists them.

import type { ExtensionAPI, Theme } from "@earendil-works/pi-coding-agent";
import { DynamicBorder } from "@earendil-works/pi-coding-agent";
import { Container, decodeKittyPrintable, Text, type Component } from "@earendil-works/pi-tui";
import { randomUUID } from "node:crypto";
import { join } from "node:path";

import {
  answerArguments, APPROVAL_COMMANDS, approvalsDirectory, autoApproveArguments, CELL_COMMANDS, alertsPath, cellAlerts, cellLockPath, cellModeState, cellWorkerPid, cellArguments, cellOutcome, commandLine,
  HALT_COMMAND, haltArguments, MODE_COMMAND, PlanStop, withdrawStopArguments, modeArguments, nextMode, pendingApprovals, PERMISSION_MODES, PROJECT_ROOT, processAlive, selfAnswers, spawnCell, YOLO_COMMAND,
  type Alive, type ApprovalRequest, type CellCommand, type CellExec, type CellOutcome, type ModeState, type PermissionMode, type RealCell,
} from "../play/cell.ts";
import { elapsed, t } from "./i18n.ts";

type Json = any;

/** The conversation entry that shows a cell command's outcome. */
export const CELL_ENTRY = "ora.cell_command";

/** What a cell command needs from the terminal and the run. */
export interface CellCommandHost {
  /** A notice in the terminal (and to the run's Pi RPC clients). */
  notify(message: string, level: "info" | "warning" | "error"): void;
  /** A display-only entry in the conversation. */
  appendEntry(customType: string, data: Json): void;
  /** The run's agent trace (play/driver.ts record); throws once the trace is closed. */
  record?(type: string, payload: Json, artifact?: Json): void;
}

export interface CellCommandOptions {
  /** How `ora cell` runs (tests replace the process). */
  exec?: CellExec;
  /** The checkout whose `scripts/ora` runs, and its working directory. */
  root?: string;
  /** The approval broker's folder (default: approvalsDirectory()). */
  approvals?: string;
  /** The cell lock (default: cellLockPath()). */
  lock?: string;
  /** The held arms' alerts (default: alertsPath()). */
  alerts?: string;
  /** Whether an asker lives (tests replace it). */
  alive?: Alive;
}

/** How often an open `on` run looks for a new question. */
export const APPROVAL_POLL_MS = 500;
/** How long the footer's mode is kept before the broker's files are read again. */
export const MODE_READ_MS = 1000;

/** A started command: `done` settles once `ora cell` has answered (it never rejects). */
export interface CellRun {
  done: Promise<CellOutcome>;
}

/** Where a cell command was given: this terminal, or `ora web`'s page. */
export type CellVia = "play terminal" | "web page";

/** One cell command's outcome, as the conversation and every page show it. */
export interface CellOutcomeEntry {
  id: string;
  name: string;
  verb: string;
  ok: boolean;
  text: string;
  seconds: number;
  via: CellVia;
  at: number;
}

/** How many outcomes the run keeps for a page that opens later. */
export const CELL_OUTCOMES_KEPT = 50;
const NOTE_TEXT_MAX = 20_000;

/** The y/N prompt: `y` (or `Y`) answers yes; any other key, Enter and Esc included, answers no. */
function yesNoPrompt(theme: Theme, title: string, message: string,
                     done: (yes: boolean) => void): Component & { handleInput(data: string): void } {
  const container = new Container();
  const border = new DynamicBorder((text: string) => theme.fg("warning", text));
  container.addChild(border);
  container.addChild(new Text(theme.bold(title), 1, 0));
  container.addChild(new Text(message, 1, 1));
  container.addChild(new Text(theme.fg("dim", t("cell.yesNo")), 1, 0));
  container.addChild(border);
  return {
    render: (width: number) => container.render(width),
    invalidate: () => container.invalidate(),
    handleInput: (data: string) => {
      const key = decodeKittyPrintable(data) ?? data;
      done(key === "y" || key === "Y");
    },
  };
}

/** Asks in the terminal; no terminal to ask (no `custom` dialog) is a no. */
async function confirmed(ctx: Json, command: CellCommand, message?: string): Promise<boolean> {
  const title = t(`cell.confirm.${command.name}.title`);
  const text = message ?? t(`cell.confirm.${command.name}.message`);
  const answer = await ctx?.ui?.custom?.((_tui: Json, theme: Theme, _keys: Json, done: (yes: boolean) => void) =>
    yesNoPrompt(theme, title, text, done), { overlay: true });
  return answer === true;
}

/** A value question's unit and example (its `details`), as a hint says them: " (in mm, e.g. /approve 20)"; else "". */
export function valueExample(request: ApprovalRequest): string {
  const text = (value: Json) => (typeof value === "string" || typeof value === "number") && String(value).trim()
    ? String(value).trim() : null;
  const unit = text(request.details?.unit);
  const example = text(request.details?.example);
  if (unit && example) return t("cell.approval.unitExample", { unit, example });
  if (example) return t("cell.approval.example", { example });
  return unit ? t("cell.approval.unit", { unit }) : "";
}

/** What approving and declining mean for a `task`, `value` or `write` question; null for the others. */
export function kindHint(request: ApprovalRequest): string | null {
  if (request.kind === "value") return t("cell.approval.hint.value", { example: valueExample(request) });
  return request.kind === "task" || request.kind === "write" ? t(`cell.approval.hint.${request.kind}`) : null;
}

/**
 * The question as the y/N prompt shows it: its title, id and time left, its words, what approving and
 * declining mean for its kind, the reason or the value sent.
 */
export function questionText(request: ApprovalRequest, reason: string | null, now: number = Date.now() / 1000,
                             value: string | null = null): string {
  const left = Math.max(0, Math.round(request.expires_unix_s - now));
  const lines = [t("cell.approval.head", { title: request.title, id: request.id, s: left }),
    ...String(request.text ?? "").split("\n").filter((line) => line.trim())];
  const hint = kindHint(request);
  if (hint) lines.push(hint);
  if (reason) lines.push(t("cell.approval.reason", { reason }));
  if (value) lines.push(t("cell.approval.value", { value }));
  return lines.join("\n");
}

/** The switch's state (`ora cell auto-approve --json`'s result) in plain words; other text as it is. */
export function yoloText(text: string): string {
  let state: Json;
  try {
    state = JSON.parse(text);
  } catch {
    return text;
  }
  if (state === null || typeof state !== "object" || typeof state.on !== "boolean") return text;
  if (state.on) return t("cell.yolo.on", { pid: state.pid, by: state.by });
  return t("cell.yolo.off") + (state.stale ? ` (${state.stale})` : "");
}

/** The permission mode's limits for `safe`, in plain words. */
export function safeLimitsText(state: Pick<ModeState, "safe_limits">): string {
  const limits = state.safe_limits;
  return t("cell.mode.limits", { s: limits.max_duration_s, rad: limits.max_joint_travel_rad,
    table: Math.round(limits.min_table_distance_m * 1000) / 10, apart: Math.round(limits.min_inter_arm_distance_m * 1000) / 10 });
}

/** The mode in plain words: what runs without asking, `safe`'s limits, `auto`'s worker and fallback. */
export function modeStateText(state: ModeState): string {
  const head = state.mode === "safe" ? t("cell.mode.state.safe", { limits: safeLimitsText(state) })
    : state.mode === "auto" ? t("cell.mode.state.auto", { pid: state.pid, by: state.by, back: state.falls_back_to ?? "ask" })
      : t("cell.mode.state.ask");
  const by = state.mode !== "auto" && state.by ? t("cell.mode.setBy", { by: state.by }) : "";
  return head + by + t("cell.mode.always") + (state.stale ? ` (${state.stale})` : "");
}

/** `ora cell mode --json`'s state in plain words; other text as it is. */
export function modeText(text: string): string {
  let state: Json;
  try {
    state = JSON.parse(text);
  } catch {
    return text;
  }
  if (state === null || typeof state !== "object" || !PERMISSION_MODES.includes(state.mode)) return text;
  return modeStateText({ ...state, safe_limits: { ...state.safe_limits } });
}

/** The footer's words for a mode: the mode and what it means, short. */
export function modeLabel(mode: PermissionMode): string {
  return t(`cell.mode.label.${mode}`);
}

/** The outcome entry: ✓ or ✗, the command and its time, then the result or the reason. */
export function renderCellEntry(entry: Json, _options: Json, theme: Theme): Component {
  const data = entry?.data ?? {};
  const ok = data.ok === true;
  const seconds = Number(data.seconds);
  const head = t(ok ? "entry.cellDone" : "entry.cellFailed", {
    name: data.name, verb: data.verb, s: Number.isFinite(seconds) ? elapsed(seconds * 1000) : "?",
  });
  const from = data.via === "web page" ? ` ${t("entry.cellFromPage")}` : "";
  const lines = [`${ok ? theme.fg("success", "✓") : theme.fg("error", "✗")} ${theme.fg(ok ? "muted" : "error", head + from)}`];
  const text = String(data.text ?? "");
  if (text) lines.push(...text.split("\n").map((line) => theme.fg(ok ? "muted" : "warning", `  ${line}`)));
  return new Text(lines.join("\n"), 1, 0);
}

export class CellCommands {
  readonly cell: RealCell;
  private readonly host: CellCommandHost;
  private readonly exec: CellExec;
  private readonly root: string;
  private readonly approvals: string;
  private readonly lock: string;
  private readonly alertsFile: string;
  /** Alerts already said, by code|arm|motor|level; one that clears and returns is said again. */
  private said = new Set<string>();
  private readonly alive: Alive;
  /** The run's worker once seen holding the cell; null after its end was reported. */
  private worker: number | null = null;
  private watcher: NodeJS.Timeout | null = null;
  /** Questions already announced in this terminal. */
  private readonly announced = new Set<string>();
  /** The latest outcomes, this terminal's and the page's (oldest first). */
  readonly outcomes: CellOutcomeEntry[] = [];
  /** Every outcome as it is known (the run's RPC clients: the `ora_cell_command` event). */
  onOutcome: ((entry: CellOutcomeEntry) => void) | null = null;
  /** The mode as last read from the broker's files, and when (ms). */
  private modeRead: { state: ModeState; at: number } | null = null;
  /** The mode the footer last drew. */
  private modeShown: string | null = null;
  /** A /mode or Shift+Tab switch is running. */
  private switching = false;
  /** Self-answered motions are said from this time on (seconds), each once. */
  private readonly selfSince: number;
  private readonly selfSaid = new Set<string>();

  constructor(cell: RealCell, host: CellCommandHost, options: CellCommandOptions = {}) {
    this.cell = cell;
    this.host = host;
    this.exec = options.exec ?? spawnCell;
    this.root = options.root ?? PROJECT_ROOT;
    this.approvals = options.approvals ?? approvalsDirectory();
    this.lock = options.lock ?? cellLockPath();
    this.alertsFile = options.alerts ?? alertsPath();
    this.alive = options.alive ?? processAlive;
    this.selfSince = Date.now() / 1000;
  }

  /** Pi's slash commands, in its help and autocomplete. */
  register(pi: ExtensionAPI): void {
    for (const command of CELL_COMMANDS) {
      pi.registerCommand(command.name, {
        description: command.description,
        handler: async (_args: string, ctx: Json) => {
          await this.start(command, ctx);  // returns once started; the outcome follows in the conversation
        },
      });
    }
    for (const command of APPROVAL_COMMANDS) {
      pi.registerCommand(command.name, {
        description: command.description,
        handler: async (args: string, ctx: Json) => {
          await this.answer(command, args, ctx);
        },
      });
    }
    pi.registerCommand(YOLO_COMMAND.name, {
      description: YOLO_COMMAND.description,
      handler: async (args: string, ctx: Json) => {
        await this.yolo(args, ctx);
      },
    });
    pi.registerCommand(HALT_COMMAND.name, {
      description: HALT_COMMAND.description,
      handler: async (_args: string, _ctx: Json) => {
        this.halt();
      },
    });
    pi.registerCommand(MODE_COMMAND.name, {
      description: MODE_COMMAND.description,
      getArgumentCompletions: (prefix: string) => [...PERMISSION_MODES, "status"].filter((mode) => mode.startsWith(prefix.trim()))
        .map((mode) => ({ value: mode, label: mode })),
      handler: async (args: string, ctx: Json) => {
        await this.mode(args, ctx);
      },
    });
  }

  /** The permission mode from the broker's files, read again once `MODE_READ_MS` has passed (or `fresh`). */
  modeState(fresh = false): ModeState {
    const now = Date.now();
    if (fresh || this.modeRead === null || now - this.modeRead.at >= MODE_READ_MS) {
      this.modeRead = { state: cellModeState(this.approvals, this.alive), at: now };
    }
    return this.modeRead.state;
  }

  /** Whether the mode changed since the footer last drew it (the footer then draws again). */
  modeChanged(): boolean {
    const state = this.modeState();
    const key = `${state.mode}|${state.by}|${state.pid}`;
    if (key === this.modeShown) return false;
    this.modeShown = key;
    return true;
  }

  /**
   * /mode ask|safe|auto, or /mode alone for the mode now (from the broker's files: nothing runs); only
   * `auto` asks y/N. Null when nothing was run.
   */
  async mode(args: string, ctx: Json): Promise<CellRun | null> {
    const state = args.trim().toLowerCase() || "status";
    if (state === "status") {
      this.host.notify(modeStateText(this.modeState(true)), "info");
      return null;
    }
    if (!(PERMISSION_MODES as readonly string[]).includes(state)) {
      this.host.notify(t("cell.mode.usage"), "info");
      return null;
    }
    if (state === "auto" && !await confirmed(ctx, MODE_COMMAND)) {
      this.host.notify(t("cell.cancelled", { name: MODE_COMMAND.name }), "info");
      return null;
    }
    const done = this.execute(MODE_COMMAND, modeArguments(state as PermissionMode, this.cell), modeText).then((outcome) => {
      this.modeRead = null;              // the footer reads the files again
      return outcome;
    });
    return { done };
  }

  /** Shift+Tab: the next mode (ask → safe → auto → ask; `auto` only on an `on` run), as /mode sets it. */
  async cycleMode(ctx: Json): Promise<CellRun | null> {
    if (this.switching) {
      this.host.notify(t("cell.mode.busy"), "info");
      return null;
    }
    this.switching = true;
    let run: CellRun | null = null;
    try {
      run = await this.mode(nextMode(this.modeState(true).mode, this.cell.rigMode), ctx);
    } finally {
      if (run === null) this.switching = false;
    }
    if (run !== null) void run.done.finally(() => { this.switching = false; });
    return run;
  }

  /** /yolo on|off, or /yolo for its state; only `on` asks y/N. Null when nothing was run. */
  async yolo(args: string, ctx: Json): Promise<CellRun | null> {
    const state = args.trim().toLowerCase() || "status";
    if (state !== "on" && state !== "off" && state !== "status") {
      this.host.notify(t("cell.yolo.usage"), "info");
      return null;
    }
    if (state === "on" && !await confirmed(ctx, YOLO_COMMAND)) {
      this.host.notify(t("cell.cancelled", { name: YOLO_COMMAND.name }), "info");
      return null;
    }
    return { done: this.execute(YOLO_COMMAND, autoApproveArguments(state, this.cell), yoloText) };
  }

  /** /halt (and Esc while the robot moves): the running motion ends where the arms are; never asked. */
  halt(): CellRun {
    return { done: this.execute(HALT_COMMAND, haltArguments(this.cell)) };
  }

  /** A stop sent while the agent plans, until a motion reads it or it is withdrawn (play/cell.ts PlanStop). */
  readonly planStop = new PlanStop();
  private planStopRun: Promise<CellOutcome> | null = null;

  /**
   * Esc while the agent plans (no motion running): the same stop as /halt, which the worker reads at the
   * approval of the motion this reply proposes and refuses it there; `text` is a message sent with it.
   */
  stopPlan(text: string | null = null): CellRun {
    this.planStop.sent(text);
    const run = this.halt();
    this.planStopRun = run.done;
    return run;
  }

  /** Free play's state (tui/extension.ts playState): a stop no motion read is withdrawn; null when none is. */
  playState(record: Json): CellRun | null {
    if (!this.planStop.settle(record)) return null;
    // After the stop's own command: one still starting would write the stop again.
    const done = (this.planStopRun ?? Promise.resolve(null)).then(() =>
      this.execute(HALT_COMMAND, withdrawStopArguments(this.cell), (text) => text, { quiet: true }));
    this.planStopRun = null;
    return { done };
  }

  /** Asks first when the command needs it, then starts it; null when the operator did not answer `y`. */
  async start(command: CellCommand, ctx: Json): Promise<CellRun | null> {
    const yes = command.confirm && await confirmed(ctx, command);
    if (command.confirm && !yes) {
      this.host.notify(t("cell.cancelled", { name: command.name }), "info");
      return null;
    }
    return { done: this.execute(command, cellArguments(command.verb, this.cell, yes)) };
  }

  /** The questions the worker waits on now, oldest first. */
  pending(): ApprovalRequest[] {
    return pendingApprovals(this.approvals, Date.now() / 1000, this.alive);
  }

  /**
   * /approve [value] or /decline [reason]: the oldest waiting question, shown with its id; only `y`
   * answers it, by that id. A value question's approval needs a value and no other question takes
   * one: either is said, and nothing runs. Null when nothing ran.
   */
  async answer(command: CellCommand, args: string, ctx: Json): Promise<CellRun | null> {
    const [request] = this.pending();
    if (request === undefined) {
      this.host.notify(t("cell.approval.none", { name: command.name }), "info");
      return null;
    }
    const rest = args.trim() || null;
    const reason = command.verb === "decline" ? rest : null;
    const value = command.verb === "approve" ? rest : null;
    if (command.verb === "approve" && request.kind === "value" && value === null) {
      this.host.notify(t("cell.approval.needsValue", { name: command.name, example: valueExample(request) }), "warning");
      return null;
    }
    if (value !== null && request.kind !== "value") {
      this.host.notify(t("cell.approval.noValue", { name: command.name, kind: request.kind }), "warning");
      return null;
    }
    if (!await confirmed(ctx, command, questionText(request, reason, Date.now() / 1000, value))) {
      this.host.notify(t("cell.cancelled", { name: command.name }), "info");
      return null;
    }
    this.announced.add(request.id);
    return { done: this.execute(command, answerArguments(command.verb, request.id, this.cell, reason, value)) };
  }

  /** While an `on` run is open: say once when a new question waits. */
  watchApprovals(intervalMs: number = APPROVAL_POLL_MS): void {
    if (this.watcher !== null || this.cell.rigMode !== "on") return;
    this.watcher = setInterval(() => this.announce(), intervalMs);
    this.watcher.unref();
    this.announce();
  }

  stopWatching(): void {
    if (this.watcher !== null) clearInterval(this.watcher);
    this.watcher = null;
  }

  /** One look at the broker's folder: a notice for each question not announced yet; and an error
   *  once when the worker that held the arms is gone (nothing controls them any more). */
  announce(): void {
    const now = new Set<string>();
    for (const alert of cellAlerts(this.alertsFile, this.alive)) {
      const key = `${alert.code}|${alert.arm ?? ""}|${alert.motor ?? ""}|${alert.level}`;
      now.add(key);
      if (!this.said.has(key)) this.host.notify(alert.text, alert.level === "critical" ? "error" : "warning");
    }
    this.said = now;
    const holding = cellWorkerPid(this.lock);
    if (holding !== null && this.alive(holding)) this.worker = holding;
    else if (this.worker !== null && !this.alive(this.worker)) {
      this.host.notify(t("cell.worker.gone", { pid: this.worker }), "error");
      this.worker = null;
    }
    for (const answer of selfAnswers(this.approvals, this.selfSince)) {
      if (this.selfSaid.has(answer.id)) continue;
      this.selfSaid.add(answer.id);
      this.host.notify(t("cell.selfAnswered", { title: answer.title, by: answer.by }), "info");
    }
    for (const request of this.pending()) {
      if (this.announced.has(request.id)) continue;
      this.announced.add(request.id);
      const hint = kindHint(request);
      this.host.notify(t("cell.approval.waiting", { title: request.title, id: request.id }) + (hint ? ` · ${hint}` : ""),
        "warning");
    }
  }

  /** `quiet`: recorded in the trace only (a withdrawn stop), said only when it fails. */
  private async execute(command: CellCommand, args: string[], say: (text: string) => string = (text) => text,
    options: { quiet?: boolean } = {}): Promise<CellOutcome> {
    const started = Date.now();
    try {
      const script = join(this.root, "scripts", "ora");
      const line = commandLine(script, args);
      if (!options.quiet) this.host.notify(t("cell.running", { name: command.name, verb: command.verb }), "info");
      this.record(command.name, "operator_shell_command", { command: line, cwd: this.root, exclude_from_context: true });
      const result = await this.exec(script, args, this.root);
      const answered = cellOutcome(result);
      const outcome = answered.ok ? { ok: true, text: say(answered.text) } : answered;
      this.record(command.name, "operator_shell_result", { command: line, exit_code: result.code },
        { output: result.stdout + result.stderr });
      if (options.quiet) {
        if (!outcome.ok) this.host.notify(t("cell.failed", { name: command.name, reason: outcome.text }), "warning");
        return outcome;
      }
      this.show({ id: randomUUID(), name: command.name, verb: command.verb, ok: outcome.ok, text: outcome.text,
        seconds: Math.round((Date.now() - started) / 100) / 10, via: "play terminal", at: Date.now() });
      return outcome;
    } catch (error) {
      // Nothing may reject unobserved here: the handler did not wait for this promise.
      const reason = (error as Error)?.message ?? String(error);
      this.host.notify(t("cell.failed", { name: command.name, reason }), "error");
      return { ok: false, text: reason };
    }
  }

  /**
   * A command `ora web`'s page ran (ora/app/web_cell.py), told by its gateway: shown in the conversation as
   * this terminal's own, recorded in the agent trace as they are, and told to every page. Throws on a note
   * that is not one.
   */
  note(note: Json): CellOutcomeEntry {
    const known = new Set([...CELL_COMMANDS, ...APPROVAL_COMMANDS, YOLO_COMMAND, MODE_COMMAND, HALT_COMMAND]
      .map((item) => `${item.name}:${item.verb}`));
    const name = String(note?.name ?? "");
    const verb = String(note?.verb ?? "");
    if (!known.has(`${name}:${verb}`)) throw new Error(`ora_cell_note: /${name} (ora cell ${verb}) is not a cell command`);
    if (typeof note.ok !== "boolean" || typeof note.command !== "string" || !note.command.trim()) {
      throw new Error("ora_cell_note: ok and command are required");
    }
    const text = (value: Json) => String(value ?? "").slice(0, NOTE_TEXT_MAX);
    const seconds = Number(note.seconds);
    const entry: CellOutcomeEntry = {
      id: typeof note.id === "string" && note.id ? note.id.slice(0, 64) : randomUUID(), name, verb, ok: note.ok,
      text: !note.ok ? text(note.text) : name === YOLO_COMMAND.name ? yoloText(text(note.text))
        : name === MODE_COMMAND.name ? modeText(text(note.text)) : text(note.text), seconds: Number.isFinite(seconds) && seconds >= 0 ? seconds : 0, via: "web page", at: Date.now(),
    };
    const code = Number.isInteger(note.exit_code) ? note.exit_code : null;
    this.record(name, "operator_shell_command", { command: note.command, cwd: text(note.cwd) || this.root,
      exclude_from_context: true, via: "web page" });
    this.record(name, "operator_shell_result", { command: note.command, exit_code: code, via: "web page" },
      { output: text(note.output) });
    if (name === MODE_COMMAND.name || name === YOLO_COMMAND.name) this.modeRead = null;
    // A withdrawn stop (the page's PlanStop) is in the trace only, as this terminal's own is.
    if (note.withdraw !== true) this.show(entry);
    return entry;
  }

  /** An outcome in the conversation, in the run's list and to every page. */
  private show(entry: CellOutcomeEntry): void {
    this.host.appendEntry(CELL_ENTRY, { name: entry.name, verb: entry.verb, ok: entry.ok, text: entry.text,
      seconds: entry.seconds, ...(entry.via === "web page" ? { via: entry.via } : {}) });
    this.outcomes.push(entry);
    if (this.outcomes.length > CELL_OUTCOMES_KEPT) this.outcomes.splice(0, this.outcomes.length - CELL_OUTCOMES_KEPT);
    try {
      this.onOutcome?.(entry);
    } catch {
      // a client that went away misses it; the run's list still has it
    }
  }

  /** Into the run's agent trace; a trace that cannot take it is said, and the command still runs. */
  private record(name: string, type: string, payload: Json, artifact?: Json): void {
    if (this.host.record === undefined) return;
    try {
      this.host.record(type, payload, artifact);
    } catch (error) {
      this.host.notify(t("cell.notRecorded", { name, reason: (error as Error)?.message ?? String(error) }), "warning");
    }
  }
}
