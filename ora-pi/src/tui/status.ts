// What `/status` and the footer show, read from the agent-side evidence only.
//
// The agent process may read <run>/agent-events.jsonl and native/ (the Node
// permission sandbox); everything here comes from there or from the turn the
// Supervisor handed to the core (step, remaining budget). No world truth,
// plan internals or verdict-in-progress reach the TUI this way.

import { closeSync, existsSync, openSync, readFileSync, readSync, statSync } from "node:fs";
import { join } from "node:path";

type Json = any;

/** Incremental reader of a JSON-lines file that only grows. */
export class JsonLinesTail {
  private offset = 0;
  private partial = "";
  readonly path: string;

  constructor(path: string) {
    this.path = path;
  }

  read(): Json[] {
    if (!existsSync(this.path)) return [];
    const size = statSync(this.path).size;
    if (size < this.offset) {
      this.offset = 0;
      this.partial = "";
    }
    if (size === this.offset) return [];
    const length = size - this.offset;
    const buffer = Buffer.alloc(length);
    const fd = openSync(this.path, "r");
    try {
      readSync(fd, buffer, 0, length, this.offset);
    } finally {
      closeSync(fd);
    }
    this.offset = size;
    const text = this.partial + buffer.toString("utf8");
    const lines = text.split("\n");
    this.partial = lines.pop() ?? "";
    const events: Json[] = [];
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        events.push(JSON.parse(line));
      } catch {
        // a torn line: the writer appends whole lines, so skip it
      }
    }
    return events;
  }
}

export interface ActionSummary {
  step: number | null;
  name: string;
  arguments: Json;
  ok: boolean | null;
  actionId: string | null;
  receipt: string | null;
  refusal: string | null;
}

export interface TurnSnapshot {
  step: number;
  remainingBudget: Record<string, Json>;
  receivedAt: number;
}

/** Bookkeeping fields a one-line summary leaves out (clock readings, schema names, ids shown elsewhere). */
const QUIET_KEYS = /(^schema$|_unix_s$|^action_id$|^receipt_id$)/;

/** Scalars of a JSON object as `key=value`, for one-line summaries. */
export function scalars(value: Json, limit = 6): string {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "";
  const parts: string[] = [];
  for (const [key, item] of Object.entries(value)) {
    if (QUIET_KEYS.test(key)) continue;
    if (item === null || ["string", "number", "boolean"].includes(typeof item)) {
      const text = typeof item === "string" && item.length > 40 ? `${item.slice(0, 37)}...` : String(item);
      parts.push(`${key}=${text}`);
    }
    if (parts.length >= limit) break;
  }
  return parts.join(" ");
}

export function compactJson(value: Json, width = 120): string {
  let text: string;
  try {
    text = JSON.stringify(value);
  } catch {
    text = String(value);
  }
  if (text === undefined) return "";
  return text.length > width ? `${text.slice(0, width - 3)}...` : text;
}

/** Run progress from agent-events.jsonl, folded incrementally. */
export class RunStatus {
  private readonly tail: JsonLinesTail;
  private readonly runRoot: string;
  private readonly mutationTools: Set<string>;
  modelCalls = 0;
  inputTokens = 0;
  outputTokens = 0;
  decisions = 0;
  operatorMessages = 0;
  observationStep: number | null = null;
  lastAction: ActionSummary | null = null;
  lastTool: { name: string; ok: boolean | null; step: number | null } | null = null;
  terminated: Json = null;
  interrupted: string | null = null;
  officialSuccess: Json = undefined;
  turn: TurnSnapshot | null = null;
  private pending = new Map<string, ActionSummary>();
  /** Every proposed call, for an accepted action whose tool is not listed as a mutation. */
  private proposed = new Map<string, { name: string; arguments: Json; step: number | null }>();

  constructor(runRoot: string, mutationTools: Iterable<string>) {
    this.runRoot = runRoot;
    this.tail = new JsonLinesTail(join(runRoot, "agent-events.jsonl"));
    this.mutationTools = new Set(mutationTools);
  }

  /** The Supervisor opened a policy turn (the act RPC). */
  observeTurn(step: number, remainingBudget: Record<string, Json> | null | undefined): void {
    this.turn = { step, remainingBudget: { ...(remainingBudget ?? {}) }, receivedAt: Date.now() };
  }

  refresh(): this {
    for (const event of this.tail.read()) this.fold(event);
    return this;
  }

  private artifact(ref: Json): Json {
    if (!ref || typeof ref.path !== "string") return null;
    try {
      return JSON.parse(readFileSync(join(this.runRoot, ref.path), "utf8"));
    } catch {
      return null;
    }
  }

  private fold(event: Json): void {
    const payload = event.payload ?? {};
    // A code-mode script's library request (robot.query, robot.explain), recorded as a nested call: no tool call.
    if (payload.origin?.kind === "code_mode" && typeof payload.origin?.op === "string") return;
    switch (event.type) {
      case "provider_request":
        this.modelCalls += 1;
        break;
      case "provider_response": {
        const usage = payload.usage ?? {};
        this.inputTokens += Number(usage.input_tokens ?? 0) || 0;
        this.outputTokens += Number(usage.output_tokens ?? 0) || 0;
        break;
      }
      case "observation_received":
        this.observationStep = event.observation_step ?? this.observationStep;
        break;
      case "tool_call_proposed": {
        const name = String(payload.name ?? "");
        this.proposed.set(String(event.tool_call_id), {
          name, arguments: payload.arguments ?? null, step: event.observation_step ?? null,
        });
        if (this.mutationTools.has(name)) {
          this.pending.set(String(event.tool_call_id), {
            step: event.observation_step ?? null, name, arguments: payload.arguments ?? null,
            ok: null, actionId: null, receipt: null, refusal: null,
          });
        }
        break;
      }
      case "tool_result": {
        const name = String(payload.name ?? "");
        const ok = typeof payload.ok === "boolean" ? payload.ok : null;
        this.lastTool = { name, ok, step: event.observation_step ?? null };
        const action = this.pending.get(String(event.tool_call_id));
        if (action) {
          action.ok = ok;
          if (ok === false) {
            const result = this.artifact(event.artifact_ref);
            const error = result?.error;
            action.refusal = error && typeof error === "object"
              ? [error.code, error.message].filter(Boolean).join(": ") || compactJson(error, 100)
              : compactJson(result, 100);
          }
          this.lastAction = action;
          this.pending.delete(String(event.tool_call_id));
        }
        break;
      }
      case "action_proposed": {
        if (this.lastAction === null || this.lastAction.actionId !== null) {
          // The mutation tool list was not known (a finished run's view): the action names its call.
          const call = this.proposed.get(String(event.tool_call_id));
          if (call) this.lastAction = { ...call, ok: true, actionId: null, receipt: null, refusal: null };
        }
        if (this.lastAction && this.lastAction.actionId === null) this.lastAction.actionId = String(payload.action_id ?? "");
        break;
      }
      case "environment_receipt": {
        this.decisions += 1;
        const receipt = this.artifact(event.artifact_ref);
        if (this.lastAction && (this.lastAction.actionId === null || this.lastAction.actionId === payload.action_id)) {
          this.lastAction.receipt = scalars(receipt) || compactJson(receipt, 100);
        }
        break;
      }
      case "action_discarded": {
        // A round budget ended the round before the Supervisor executed the proposal.
        if (this.lastAction && this.lastAction.actionId === payload.action_id) {
          this.lastAction.receipt = `not executed · ${String(payload.reason ?? "round ended")}`;
        }
        break;
      }
      case "operator_message":
        this.operatorMessages += 1;
        break;
      case "session_interrupted":
        this.interrupted = String(payload.reason ?? "interrupted");
        break;
      case "official_verdict":
        this.officialSuccess = payload.official_success;
        break;
      case "run_terminated":
        this.terminated = payload;
        break;
      default:
        break;
    }
  }

  /** Remaining budget now: wall time counts down from when the turn opened. */
  remaining(): Array<[string, string]> {
    const turn = this.turn;
    if (turn === null) return [];
    const rows: Array<[string, string]> = [];
    const order = ["robot_decisions", "model_calls", "wall_time_s"];
    const keys = [...order.filter((key) => key in turn.remainingBudget),
      ...Object.keys(turn.remainingBudget).filter((key) => !order.includes(key)).sort()];
    for (const key of keys) {
      let value = turn.remainingBudget[key];
      if (value && typeof value === "object" && "value" in value) value = value.value; // PyFloat
      if (key === "wall_time_s" && typeof value === "number") {
        const left = Math.max(0, value - (Date.now() - turn.receivedAt) / 1000);
        rows.push(["wall time", formatSeconds(left)]);
      } else if (value === null || ["string", "number", "boolean"].includes(typeof value)) {
        rows.push([key.replace(/_/g, " "), String(value)]);
      }
    }
    return rows;
  }
}

export function formatSeconds(seconds: number): string {
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0 ? `${h}h${String(m).padStart(2, "0")}m` : `${m}m${String(s).padStart(2, "0")}s`;
}

/** The browser world view `ora attach` serves beside the terminal (ora/app/world.py, TerminalWorldFeed). */
export interface WorldView {
  url: string | null;
  phase: string;
  connected: boolean;
  visible: boolean;
  frameIndex: number | null;
  message: string;
  browserMessage: string;
}

/** Terminal-safe text: no control sequences from the relayed state, bounded length. */
function plain(value: unknown, limit: number): string {
  return typeof value === "string" ? value.replace(/[\x00-\x1f\x7f-\x9f]/g, "").slice(0, limit) : "";
}

/** A relayed world view state, checked; null when the terminal brought none. */
export function worldView(value: Json): WorldView | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const url = plain(value.url, 200);
  return {
    url: /^https?:\/\/[^\s]+$/.test(url) ? url : null,
    phase: plain(value.phase, 32) || "starting",
    connected: value.connected === true,
    visible: value.visible === true,
    frameIndex: Number.isInteger(value.frame_index) ? value.frame_index : null,
    message: plain(value.message, 160),
    browserMessage: plain(value.browser_message, 160),
  };
}

/** What the browser page shows now, in a few words. */
export function worldSummary(view: WorldView): string {
  if (view.phase === "unavailable" || view.phase === "error") return view.message || "unavailable";
  if (view.connected && view.visible && view.frameIndex !== null) return `browser shows frame ${view.frameIndex}`;
  if (view.connected) return "browser connected";
  if (view.browserMessage) return view.browserMessage;
  if (view.phase === "starting") return "starting";
  if (view.frameIndex === null) return "open it in a browser; waiting for the first camera frame";
  return "open it in a browser";
}
