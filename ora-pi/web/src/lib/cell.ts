// A real robot cell's controls on the page, as free play's terminal has them (ora-pi/src/play/cell.ts,
// ora-pi/src/tui/cell_commands.ts): the question the robot's worker or a guided calibration waits on
// (/approve [value], /decline [reason]; a `value` question is approved with its value, and no other
// question takes one), the auto-approve switch (/yolo on|off), the permission mode (/mode ask|safe|auto,
// in the footer: every motion asked, small ones run, or none asked), the stop of the motion in progress
// (/halt: it ends where the arms are, held; sent while the agent plans, the motion its reply proposes is
// refused before it moves, and the stop is withdrawn when no motion came: PlanStop), the operator's verbs (/home, /hold, /float, /release, /estop,
// /diagnose), the held arms' alerts and the end of the worker that held them.
// There is no /calibrate: this run's worker holds the cell lock calibration needs (the cell board or
// `ora cell calibrate` starts it).
//
// The gateway answers them (ora/app/web_cell.py, /ora/cell): the state from the cell's own files,
// the actions as the terminal's own `ora cell` command lines. The page asks first exactly where the
// terminal asks y/N, and says so in the request (`confirmed`). A run that drives no real cell has
// none of it: typed, the commands are answered as the terminal answers them.

import { useEffect, useState } from "react";

import { call } from "./api.ts";
import { t } from "./i18n.ts";

type Json = any;

/** One of the terminal's cell commands, as the page offers it. */
export interface PageCellCommand {
  /** The slash command (without the slash), as typed in the terminal. */
  name: string;
  /** The `ora cell` verb it runs. */
  verb: string;
  /** Asked first (the terminal's y/N; a dialog here). */
  confirm: boolean;
  kind: "verb" | "answer" | "switch" | "mode" | "halt";
}

/** The operator's verbs (play/cell.ts CELL_COMMANDS). */
export const CELL_VERBS: readonly PageCellCommand[] = [
  { name: "home", verb: "home", confirm: false, kind: "verb" },
  { name: "hold", verb: "hold", confirm: false, kind: "verb" },
  { name: "float", verb: "float", confirm: true, kind: "verb" },
  { name: "release", verb: "release", confirm: true, kind: "verb" },
  { name: "estop", verb: "estop", confirm: true, kind: "verb" },
  { name: "diagnose", verb: "diagnose", confirm: false, kind: "verb" },
];

/** Answers to the worker's question (play/cell.ts APPROVAL_COMMANDS); both are asked first. */
export const CELL_ANSWERS: readonly PageCellCommand[] = [
  { name: "approve", verb: "approve", confirm: true, kind: "answer" },
  { name: "decline", verb: "decline", confirm: true, kind: "answer" },
];

/** The auto-approve switch (play/cell.ts YOLO_COMMAND); only turning it on is asked. */
export const CELL_SWITCH: PageCellCommand = { name: "yolo", verb: "auto-approve", confirm: true, kind: "switch" };

/** The permission mode (play/cell.ts MODE_COMMAND); only `auto` is asked. */
export const CELL_MODE: PageCellCommand = { name: "mode", verb: "mode", confirm: true, kind: "mode" };

/**
 * Stop here (play/cell.ts HALT_COMMAND): the motion in progress ends where the arms are, held; the agent gets a
 * fresh observation. Never asked: it holds, it never drops.
 */
export const CELL_HALT: PageCellCommand = { name: "halt", verb: "stop-motion", confirm: false, kind: "halt" };

export const PAGE_CELL_COMMANDS: readonly PageCellCommand[] = [...CELL_VERBS, ...CELL_ANSWERS, CELL_SWITCH, CELL_MODE, CELL_HALT];

/** The permission modes, in the order the terminal's Shift+Tab cycles them (play/cell.ts PERMISSION_MODES). */
export const PERMISSION_MODES = ["ask", "safe", "auto"] as const;
export type PermissionMode = typeof PERMISSION_MODES[number];

/** The permission mode (`ora cell mode --json`, ora/cell/operator/approval.py `mode_state`). */
export interface ModeState {
  mode: PermissionMode;
  by: string | null;
  since_unix_s?: number | null;
  pid?: number | null;
  run?: string | null;
  stale?: string | null;
  safe_limits: { max_duration_s: number; max_joint_travel_rad: number; min_table_distance_m: number; min_inter_arm_distance_m: number };
  falls_back_to?: "ask" | "safe";
}

/** A motion that answered itself (safe mode or auto-approve), as its answer records it. */
export interface SelfAnswer {
  id: string;
  kind: string;
  title: string;
  by: string;
  answered_unix_s: number;
}

export interface RealCell {
  kind: string;
  rig_mode: string;
  cell_config: string | null;
}

/** A question the worker waits on (ora.cell-approval-request.v1). */
export interface CellQuestion {
  id: string;
  kind: string;
  title: string;
  text: string;
  details?: Json;
  created_unix_s: number;
  expires_unix_s: number;
  pid: number;
  asker?: string;
}

export interface CellAlert {
  code: string;
  level: "warning" | "critical";
  text: string;
  arm?: string;
  motor?: string;
}

/** GET /ora/cell. */
export interface CellState {
  cell: RealCell | null;
  now?: number;
  pending?: CellQuestion[];
  auto_approve?: Json;
  /** The permission mode (an older gateway has none). */
  mode?: ModeState;
  /** The latest motions that answered themselves, oldest first. */
  self_answered?: SelfAnswer[];
  /** The motion questions answered lately, with when each was asked and answered (lib/perf.ts timedAnswers). */
  answers?: Json[];
  alerts?: CellAlert[];
  alerts_stale?: boolean;
  worker?: { pid: number | null; alive: boolean };
  worker_ended?: { pid: number; seen_unix_s: number } | null;
}

/** One command's outcome: the page's own (the gateway's answer) or the run's (the cell_command event). */
export interface CellOutcome {
  id: string;
  name: string;
  verb: string;
  ok: boolean;
  text: string;
  seconds: number;
  via: string;
  at?: number;
}

/** What a typed command asks for, on a run that drives a real cell. */
export type CellIntent =
  | { kind: "verb"; command: PageCellCommand }
  | { kind: "answer"; verb: "approve" | "decline"; reason: string | null; value: string | null }
  | { kind: "switch"; state: "on" | "off" | "status" }
  | { kind: "mode"; mode: PermissionMode | "status" }
  | { kind: "halt" }
  | { kind: "usage"; name: string };

const NAMES = new Map(PAGE_CELL_COMMANDS.map((command) => [command.name, command]));

/** Whether `name` is one of the terminal's cell commands. */
export function isCellCommand(name: string): boolean {
  return NAMES.has(name);
}

/** A typed `/name args` of the cell's commands, or null for any other text. */
export function cellIntent(text: string): CellIntent | null {
  const match = /^\/(\S+)\s*([\s\S]*)$/.exec(text.trim());
  const command = match ? NAMES.get(match[1]) : undefined;
  if (!match || command === undefined) return null;
  const args = match[2].trim();
  if (command.kind === "verb") return { kind: "verb", command };
  if (command.kind === "halt") return { kind: "halt" };
  if (command.kind === "answer") {
    return { kind: "answer", verb: command.verb as "approve" | "decline", reason: command.verb === "decline" ? args || null : null,
      value: command.verb === "approve" ? args || null : null };
  }
  const state = args.toLowerCase() || "status";
  if (command.kind === "mode") {
    return state === "status" || (PERMISSION_MODES as readonly string[]).includes(state)
      ? { kind: "mode", mode: state as PermissionMode | "status" } : { kind: "usage", name: command.name };
  }
  return state === "on" || state === "off" || state === "status" ? { kind: "switch", state } : { kind: "usage", name: command.name };
}

/** The mode the terminal's Shift+Tab goes to next: ask → safe → auto → ask; `auto` only on an `on` run. */
export function nextMode(mode: PermissionMode, rigMode: string): PermissionMode {
  if (mode === "ask") return "safe";
  if (mode === "safe") return rigMode === "on" ? "auto" : "ask";
  return "ask";
}

/** `safe`'s limits in plain words. */
export function safeLimitsText(state: Pick<ModeState, "safe_limits">): string {
  const limits = state.safe_limits;
  return t("cell.mode.limits", { s: limits.max_duration_s, rad: limits.max_joint_travel_rad,
    table: Math.round(limits.min_table_distance_m * 1000) / 10, apart: Math.round(limits.min_inter_arm_distance_m * 1000) / 10 });
}

/** The mode in plain words, as the terminal says it (tui/cell_commands.ts modeStateText). */
export function modeStateText(state: ModeState): string {
  const head = state.mode === "safe" ? t("cell.mode.state.safe", { limits: safeLimitsText(state) })
    : state.mode === "auto" ? t("cell.mode.state.auto", { pid: state.pid ?? "?", by: state.by ?? "?", back: state.falls_back_to ?? "ask" })
      : t("cell.mode.state.ask");
  const by = state.mode !== "auto" && state.by ? t("cell.mode.setBy", { by: state.by }) : "";
  return head + by + t("cell.mode.always") + (state.stale ? ` (${state.stale})` : "");
}

/** `ora cell mode --json`'s state, or null for other text. */
export function modeStateOf(text: string): ModeState | null {
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && (PERMISSION_MODES as readonly string[]).includes(value.mode)
      && value.safe_limits && typeof value.safe_limits === "object" ? value : null;
  } catch {
    return null;
  }
}

/** The oldest question still open at `now` (seconds), as the terminal answers /approve and /decline. */
export function oldestQuestion(state: CellState | null, now: number): CellQuestion | null {
  const open = (state?.pending ?? []).filter((question) => question.expires_unix_s > now);
  return open.sort((a, b) => a.created_unix_s - b.created_unix_s)[0] ?? null;
}

export function secondsLeft(question: CellQuestion, now: number): number {
  return Math.max(0, Math.round(question.expires_unix_s - now));
}

/** Longest value the broker keeps (ora/cell/operator/approval.py VALUE_MAX_CHARS). */
export const VALUE_MAX_CHARS = 200;

function detail(question: CellQuestion, name: string): string | null {
  const value = question.details?.[name];
  return (typeof value === "string" || typeof value === "number") && String(value).trim() ? String(value).trim() : null;
}

/** A value question's unit and example (its `details`), as a hint says them: " (in mm, e.g. 20)"; else "". */
export function valueExample(question: CellQuestion): string {
  const unit = detail(question, "unit");
  const example = detail(question, "example");
  if (unit && example) return t("cell.unitExample", { unit, example });
  if (example) return t("cell.example", { example });
  return unit ? t("cell.unit", { unit }) : "";
}

/** The value box's placeholder: "Value in mm, e.g. 20". */
export function valuePlaceholder(question: CellQuestion): string {
  const unit = detail(question, "unit");
  const example = detail(question, "example");
  return (unit ? t("cell.valueIn", { unit }) : t("cell.value")) + (example ? t("cell.valueExample", { example }) : "");
}

/** What approving and declining mean for a `task`, `value` or `write` question; null for the others. */
export function kindHint(question: CellQuestion): string | null {
  if (question.kind === "value") return t("cell.hint.value", { example: valueExample(question) });
  if (question.kind === "task") return t("cell.hint.task");
  return question.kind === "write" ? t("cell.hint.write") : null;
}

/**
 * Why an answer is refused before anything is sent, as the terminal refuses it: a value question's
 * approval without a value, or a value for any other question. Null when it may be sent.
 */
export function answerRefusal(question: CellQuestion, verb: "approve" | "decline", value: string | null): string | null {
  if (verb !== "approve") return null;
  const given = value?.trim() || null;
  if (question.kind === "value" && given === null) return t("cell.needsValue", { name: verb, example: valueExample(question) });
  if (question.kind !== "value" && given !== null) return t("cell.noValue", { name: verb, kind: question.kind });
  return null;
}

/** The switch's state (`ora cell auto-approve --json`'s), or null for other text. */
export function switchState(text: string): { on: boolean; pid?: number; by?: string; stale?: string | null } | null {
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && typeof value.on === "boolean" ? value : null;
  } catch {
    return null;
  }
}

/** GET /ora/cell; an answer without a cell (an older gateway) is a run that drives none. */
export async function cellState(): Promise<CellState> {
  const answer = await call("/ora/cell");
  return answer && typeof answer === "object" && answer.cell && typeof answer.cell === "object"
    ? answer as CellState : { cell: null };
}

export function runVerb(command: PageCellCommand, confirmed: boolean): Promise<CellOutcome> {
  return call(`/ora/cell/verb/${command.verb}`, { confirmed });
}

export function answerQuestion(id: string, verb: "approve" | "decline", reason: string | null,
                               value: string | null = null): Promise<CellOutcome> {
  return call(`/ora/cell/approval/${encodeURIComponent(id)}/${verb}`, { confirmed: true, ...(reason ? { reason } : {}),
    ...(verb === "approve" && value ? { value } : {}) });
}

export function switchAutoApprove(on: boolean): Promise<CellOutcome> {
  return call(`/ora/cell/auto-approve/${on ? "on" : "off"}`, { confirmed: on });
}

/** /halt: the motion in progress ends where the arms are (`ora cell stop-motion`); nothing to confirm. */
export function stopMotion(): Promise<CellOutcome> {
  return call("/ora/cell/stop-motion", {});
}

/** Take back a stop nobody has read yet (`ora cell stop-motion --withdraw`): the agent answered in words. */
export function withdrawStop(): Promise<CellOutcome> {
  return call("/ora/cell/stop-motion", { withdraw: true });
}

/** How long the worker honours a stop (ora/cell/operator/approval.py STOP_REQUEST_S). */
export const STOP_REQUEST_MS = 10_000;

/**
 * A stop sent while the agent plans (no motion running), as the terminal keeps it (ora-pi/src/play/cell.ts
 * PlanStop): the worker refuses the motion that reply proposes at its approval. When the reply proposes
 * none, it is withdrawn once the message sent with it reaches the agent (`message_delivered`), or the
 * stopped instruction ends, with no motion started in between.
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

  /** A free-play state (ora.play-state.v1): true when the stop is to be withdrawn now. */
  settle(record: Json): boolean {
    const pending = this.pending;
    if (pending === null) return false;
    if (this.now() - pending.at > STOP_REQUEST_MS) {
      this.pending = null;
      return false;
    }
    const state = record?.state;
    if (state === "action_started") {
      this.pending = null;
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

export function switchMode(mode: PermissionMode): Promise<CellOutcome> {
  return call(`/ora/cell/mode/${mode}`, { confirmed: mode === "auto" });
}

/** The page's and the run's outcomes, once each, oldest first. */
export function mergeOutcomes(...lists: Array<readonly CellOutcome[] | undefined>): CellOutcome[] {
  const seen = new Map<string, CellOutcome>();
  for (const list of lists) for (const item of list ?? []) if (item && typeof item.id === "string") seen.set(item.id, item);
  return [...seen.values()].sort((a, b) => (a.at ?? 0) - (b.at ?? 0));
}

export const CELL_POLL_MS = 1000;

/**
 * The cell's state, read again every second while the run drives a real cell; null until the first
 * answer, `{cell: null}` for a run that drives none (it is then not read again).
 */
export function useCellState(enabled: boolean, interval = CELL_POLL_MS): [CellState | null, () => void] {
  const [state, setState] = useState<CellState | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!enabled) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      let next: CellState | null = null;
      try {
        next = await cellState();
      } catch {
        next = null;            // the gateway is away for a moment: keep what is shown, try again
      }
      if (stopped) return;
      if (next !== null) setState(next);
      if (next?.cell === null) return;
      timer = setTimeout(() => void poll(), interval);
    };
    void poll();
    return () => {
      stopped = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [enabled, interval, tick]);
  return [state, () => setTick((value) => value + 1)];
}
