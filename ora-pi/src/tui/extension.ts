// ORA's operator surface inside Pi's terminal UI.
//
// A Pi extension that only draws and routes operator input; it never changes
// what the model sees (the ORA bridge's `context` hook owns every request).
//   /stop     operator stop (the Supervisor's `ora stop` path), after a confirm
//   /status   step, model calls, remaining budget, last action and its receipt
//   /help     keys and commands
//   /detach   leave the run running (also Ctrl+D, Ctrl+Z, /quit)
// In a free-play session (`ora play`, Pi's own agent: ora-pi/src/play/) typed
// text is the next instruction, or, while one runs, a message that joins it
// (drawn "you · joins instruction N · delivered at step S" once delivered), and
// Alt+Enter queues it instead: the Supervisor holds it until the running
// instruction ends, then it opens the next one (a notice, and "N queued" in the
// footer, while it waits); Esc stops the running instruction (marked where it
// ended; on a real cell the motion in progress ends where the arms are, and a
// motion the agent is still planning is refused before it moves), /reset loads another
// scene, /view shows the operator view and /quit ends the session (ORA then
// evaluates and seals it); the footer says what the agent does now with its
// seconds, and under it the model's speed and the loop between robot actions
// (play/perf.ts: "45 tok/s · TTFT 0.8 s · loop 6.2 s (target 3 s)"); Pi's own commands, shell (`!`), model selection and conversation
// branches work as in Pi, and /share and /bug ask before uploading anything.
// A free-play run on a real cell adds /home, /hold, /float, /release, /estop
// and /diagnose (tui/cell_commands.ts): `ora cell <verb>` at once, outside the
// play inbox; elsewhere they are answered, never sent to the model. It also
// adds /approve and /decline [reason], which answer the question the robot's
// worker waits on, and (rig_mode `on`) a notice when a new question appears.
// There the footer starts with the cell's permission mode (ask, safe, auto: the
// broker's files, read at most once a second), /mode switches it and Shift+Tab
// cycles it. Shift+Tab is Pi's thinking-level cycle, which Pi does not let an
// extension's shortcut take (a reserved key), so ORA takes it in its terminal
// input hook before Pi's editor sees it, only on a real cell and never while a
// dialog is open; /thinking still sets the reasoning effort there.
// ORA's words here are English or Chinese (tui/i18n.ts: ORA_UI_LANG).
// Pi built-ins that would rewrite a live episode, switch models or credentials,
// reload extensions or upload the session are refused with the reason;
// /model shows the run's bound model and /export writes where `ora attach` runs.
// Observation and tool images come from ORA's canonical history (the same
// bytes the model received) and render with pi-tui's Image, which falls back
// to the artifact path, type and size on terminals without image support.
//
// It is drawn as the project teaser's terminal (tui/theme.ts; `ora web`'s page
// draws the same): ORA's theme for Pi, a header with the pixel-arm mark, "ORA
// vX", model · effort and environment/task; the model's words after "●" and
// the operator's after "❯" (display-only Markdown transforms: the model's
// context is untouched); a robot tool call as "● name(args)" with its result
// under a "⎿" corner (tui/render.ts callText, toolResultLines); free play's
// status line above the prompt (a teal spinner, "Thinking… (12s · esc to
// interrupt)"); the prompt after "❯" between two rules (tui/host.ts); and the
// footer "▸▸ mode · scene · instruction · … · context" with "● effort · model"
// on the right.

import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import type { ExtensionAPI, Theme } from "@earendil-works/pi-coding-agent";
import { DynamicBorder } from "@earendil-works/pi-coding-agent";
import {
  Box, Container, Image, matchesKey, Spacer, Text, truncateToWidth, visibleWidth, wrapTextWithAnsi,
  type Component,
} from "@earendil-works/pi-tui";

import { CELL_COMMAND_NAMES, PROJECT_ROOT, type CellExec, type RealCell } from "../play/cell.ts";
import type { PlayMessage } from "../play/messages.ts";
import { RunFrames } from "../rpc/frames.ts";
import { CELL_ENTRY, CellCommands, modeLabel, renderCellEntry } from "./cell_commands.ts";

/** The conversation entry of a real cell's setup checklist (display only). */
export const SETUP_ENTRY = "ora.cell_setup";
const SETUP_MARKS: Record<string, [string, string]> = {
  done: ["✓", "success"], running: ["✻", "accent"], failed: ["✗", "error"], stale: ["!", "warning"], missing: ["○", "dim"],
};

/** Which title a setup view takes: where it is now. */
export function setupTitle(setup: Json): string {
  if (setup?.phase === "done") return setup.skipped ? "skipped" : "done";
  return ["running", "failed"].includes(setup?.phase) ? setup.phase : "status";
}

/** The checklist, a line per step (its mark, title and why or detail), then how it ended. */
export function setupLines(setup: Json, theme: Theme): string[] {
  const lines = (Array.isArray(setup?.steps) ? setup.steps : []).map((step: Json) => {
    const [mark, color] = SETUP_MARKS[step?.state] ?? ["·", "dim"];
    const note = step?.why || step?.detail;
    return `${theme.fg(color as any, mark)} ${step?.title ?? step?.id}${note ? theme.fg("dim", `  ${note}`) : ""}`;
  });
  const end = setup?.error ?? setup?.result?.why;
  if (setup?.phase === "failed" && end) lines.push(theme.fg("error", String(end)));
  return lines;
}
import { budgetItem, elapsed, endingText, helpCell, helpPlay, t } from "./i18n.ts";
import {
  argumentLine, callText, cameraOf, commandLines, imageCaptions, isScriptResult, observationLines, RESULT_MAX,
  resultSummary, scriptResultLines, toolResultLines, truncate, type ScriptLine,
} from "./render.ts";
import { logoLines, oraTheme, spinnerFrame, wantsOraTheme } from "./theme.ts";
import { formatSeconds, RunStatus, scalars, worldSummary, worldView, type WorldView } from "./status.ts";

type Json = any;

/** What the TUI reads from the run. The live core and the read-only viewer both provide it. */
export interface OraTuiSource {
  runRoot: string;
  runId: string;
  model: string;
  profile: string | null;
  protocol: string;
  mutationTools: string[];
  toolNames: string[];
  /** Canonical ORA history, images as data URLs. */
  history(): Json[];
  /** Tool-image messages of the reply in progress (not yet in history). */
  pendingToolImages(): Json[];
  /** The run still accepts operator input. */
  live(): boolean;
  /** The core is inside a policy turn (not waiting on the world). */
  inTurn(): boolean;
  /** Operator stop now: interrupt the agent and tell the Supervisor (Esc does the same). */
  stop(): void;
  /** End the current turn at its next boundary, without cutting the model call in flight. */
  interruptAtBoundary(): void;
  /** Tell the Supervisor the operator stopped the run (`ora stop`). */
  notifySupervisor(): void;
  sessionFile(): string | null;
  /** A free-play session's state as the Supervisor announced it (episode_play.py), else null. */
  play(): Json | null;
  /** This run is free play on Pi's own agent (ora-pi/src/play/): Pi's commands are open. */
  freePlay?: boolean;
  /** The text of an instruction typed in this terminal, by its command id. */
  playInstruction(id: string): string | null;
  /** A free-play control for the Supervisor's inbox: stop_instruction, reset or quit. */
  control(type: string, message: string): Promise<Json>;
  /** The real cell a free-play run drives (play/cell.ts), or null/absent: no cell commands. */
  realCell?: RealCell | null;
  /** An operator event into a free-play run's agent trace (play/driver.ts record). */
  record?(type: string, payload: Json, artifact?: Json): void;
  /** Free play's speed in one line (play/perf.ts: tok/s, TTFT, the loop between actions), or null. */
  perfLine?(): string | null;
  /** Free play: the next message submitted is queued (Alt+Enter; play/driver.ts queueNext). */
  queueNext?(): void;
  /** Free play: the messages the Supervisor holds for the next instruction. */
  queuedNext?(): number;
}

/** The attached terminal, as the TUI host offers it. */
export interface OraTuiHost {
  detach(reason: string): void;
  exportSession(path: string | null, sessionFile: string): Promise<string>;
}

export interface OraImage {
  data: string;
  mimeType: string;
  /** The artifact path (pi-tui's text fallback names it). */
  label: string;
  /** What the image shows, from the text before it in the message (a camera, a tool's caption). */
  caption: string | null;
}

const LIVE_REFUSALS: Record<string, string> = {
  new: "a live episode's history cannot be replaced",
  resume: "a live episode's history cannot be replaced",
  fork: "a live episode's history cannot be rewound",
  clone: "a live episode's history cannot be copied into another session",
  tree: "a live episode's history cannot be rewound",
  import: "a live episode's history cannot be replaced",
  compact: "ORA manages the model's context with its own policy",
  thinking: "reasoning effort is bound by the RunPlan",
  "scoped-models": "the model is bound by the RunPlan",
  llama: "the model is bound by the RunPlan",
  reload: "the run's extensions are fixed while it runs",
  share: "run evidence is not uploaded from the terminal",
  bug: "run evidence is not uploaded from the terminal",
  trust: "ORA loads no project resources",
  debug: "the agent process writes only its own evidence",
};

/** Built-ins a finished run's read-only view refuses (nothing may be appended). */
const VIEW_REFUSALS: Record<string, string> = {
  ...LIVE_REFUSALS,
  new: "this view is read-only",
  resume: "this view is read-only; open another run with `ora sessions`",
  fork: "this view is read-only",
  clone: "this view is read-only",
  tree: "this view is read-only",
  import: "this view is read-only",
  name: "this view is read-only",
};

/** Pi built-ins left as they are, and ORA's own commands. */
const PASSED_COMMANDS = new Set([
  "settings", "copy", "name", "session", "changelog", "hotkeys", "stop", "status", "help", "detach", "reset", "view",
  // Any run: credentials go to the user-level ORA store (core/credentials.ts), never the run.
  "login", "logout",
]);

/** Why the Supervisor refused a free-play control, as the operator reads it (episode_play.py). */
const CONTROL_REFUSALS: Record<string, string> = {
  no_instruction_running: "notice.noInstruction",
  no_scene_left: "notice.noSceneLeft",
  scene_ended_reset_required: "notice.sceneEndedResetRequired",
};

/** A Python float that crossed as a PyFloat (pyjson.ts), or a plain number. */
function num(value: Json): number {
  return Number(value?.value ?? value);
}

/** What the agent is doing now, for the footer's live status (display only). */
interface Activity {
  kind: "thinking" | "writing" | "tool" | "moving" | "observing";
  since: number;
  tool?: string;
}

function parseDataUrl(url: string): { mimeType: string; data: string } | null {
  const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
  return match ? { mimeType: match[1], data: match[2] } : null;
}

function imagesOf(message: Json, runRoot: string): OraImage[] {
  if (!Array.isArray(message?.content)) return [];
  const captions = imageCaptions(message);
  const images: OraImage[] = [];
  let index = 0;
  for (const block of message.content) {
    if (block?.type !== "image_url") continue;
    const caption = captions[index++] ?? null;
    const parsed = parseDataUrl(String(block.image_url?.url ?? ""));
    if (parsed === null) continue;
    const digest = createHash("sha256").update(Buffer.from(parsed.data, "base64")).digest("hex");
    const stored = [".png", ".img"].map((suffix) => `native/agent-artifacts/${digest}${suffix}`)
      .find((path) => existsSync(join(runRoot, path)));
    images.push({ ...parsed, label: stored ?? `image ${digest.slice(0, 12)}`, caption });
  }
  return images;
}

function prettyJson(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

/** ORA's version (the project's pyproject.toml), or null. */
let oraVersion: string | null | undefined;
export function versionOf(root: string = PROJECT_ROOT): string | null {
  if (oraVersion !== undefined && root === PROJECT_ROOT) return oraVersion;
  let version: string | null = null;
  try {
    version = /^version\s*=\s*"([^"]+)"/m.exec(readFileSync(join(root, "pyproject.toml"), "utf8"))?.[1] ?? null;
  } catch {
    version = null;
  }
  if (root === PROJECT_ROOT) oraVersion = version;
  return version;
}

/** The run's environment and task, as the header names them (`yam/operator_task`), from its plan. */
export function whereOf(runRoot: string): { place: string | null; effort: string | null } {
  try {
    const plan = JSON.parse(readFileSync(join(runRoot, "plan.json"), "utf8"));
    const kind = typeof plan?.runtime?.kind === "string" ? plan.runtime.kind : null;
    const rig = typeof plan?.runtime?.rig_mode === "string" && plan.runtime.rig_mode !== "sim" ? plan.runtime.rig_mode : null;
    const task = typeof plan?.task === "string" ? plan.task : null;
    const effort = typeof plan?.policy?.reasoning_effort === "string" ? plan.policy.reasoning_effort : null;
    const place = kind ? `${kind}${task ? `/${task}` : ""}` : task;
    return { place: place && rig ? `${place} · ${t("header.realCell", { mode: rig })}` : place, effort };
  } catch {
    return { place: null, effort: null };
  }
}

/** A token count as the footer says it: 950, 37.2k, 1M. */
export function tokenCount(count: number): string {
  if (count >= 1_000_000) return `${Number((count / 1_000_000).toFixed(1))}M`;
  return count >= 1000 ? `${(count / 1000).toFixed(1)}k` : String(count);
}

/** A component that draws lines for a width. */
function linesComponent(build: (width: number) => string[]): Component {
  return { render: (width: number) => build(width), invalidate() {} };
}

/** `first` before the first wrapped line of `text` and `rest` before the others (both of `indent` columns). */
export function hanging(text: string, width: number, first: string, rest: string, indent: number): string[] {
  const wrapped = text.split("\n").flatMap((line) => wrapTextWithAnsi(line, Math.max(8, width - indent)));
  return wrapped.map((line, index) => truncateToWidth(`${index === 0 ? first : rest}${line}`, width));
}

/**
 * The model's words after "●", the operator's after "❯" (display only: Pi's Markdown before it draws it;
 * what the model reads is unchanged). A block that is not a plain paragraph keeps its Markdown.
 */
export function transcriptMarkdown(markdown: string, messageType: string): string {
  const mark = messageType === "assistant" ? "● " : messageType === "user" ? "❯ " : null;
  if (mark === null || !markdown.trim()) return markdown;
  const start = markdown.length - markdown.trimStart().length;
  const head = markdown.slice(start);
  if (/^(#|[-*+] |\d+[.)] |```|~~~|>|\||<|●|❯)/.test(head)) return markdown;
  return `${markdown.slice(0, start)}${mark}${head}`;
}

/** A dismissible framed panel (Enter, Esc or q closes it). */
function panel(theme: Theme, title: string, rows: string[], done: () => void): Component & { handleInput(data: string): void } {
  const container = new Container();
  const border = new DynamicBorder((text: string) => theme.fg("accent", text));
  container.addChild(border);
  container.addChild(new Text(theme.bold(title), 1, 0));
  container.addChild(new Text(rows.join("\n"), 1, 1));
  container.addChild(new Text(theme.fg("dim", "Enter or Esc closes"), 1, 0));
  container.addChild(border);
  return {
    render: (width: number) => container.render(width),
    invalidate: () => container.invalidate(),
    handleInput: (data: string) => {
      if (matchesKey(data, "enter") || matchesKey(data, "escape") || data === "q") done();
    },
  };
}

export class OraTui {
  readonly source: OraTuiSource;
  readonly status: RunStatus;
  /** False for the read-only view of a finished run. */
  readonly live: boolean;
  host: OraTuiHost | null = null;
  private ui: Json = null;
  private context: Json = null;
  private readonly observationImages = new Map<number, OraImage[]>();
  private readonly toolImageCache = new Map<string, OraImage[]>();
  private readonly waitingForImages = new Map<string, () => void>();
  private ticker: NodeJS.Timeout | null = null;
  /** Redraws the ORA footer (bound while Pi's TUI shows it). */
  private requestRender: (() => void) | null = null;
  /** /stop arrived while the model or a tool batch was working. */
  private stopPending = false;
  /** The attached terminal's browser world view (`ora attach` serves it), or null. */
  world: WorldView | null = null;
  private pi: ExtensionAPI | null = null;
  private tui: Json = null;
  /** What the agent is doing now (Pi's events and the Supervisor's), or null between instructions. */
  activity: Activity | null = null;
  /** The run's operator view and recorded motion (rpc/frames.ts). */
  readonly frames: RunFrames;
  /** A live free-play run on a real cell: its /home, /hold, … (tui/cell_commands.ts); else null. */
  readonly cell: CellCommands | null;

  constructor(source: OraTuiSource, options: { live: boolean; cellExec?: CellExec }) {
    this.source = source;
    this.live = options.live;
    this.status = new RunStatus(source.runRoot, source.mutationTools);
    this.frames = new RunFrames(source.runRoot);
    const realCell = options.live && source.freePlay ? source.realCell ?? null : null;
    this.cell = realCell === null ? null : new CellCommands(realCell, {
      notify: (message, level) => this.notify(message, level),
      appendEntry: (customType, data) => this.pi?.appendEntry(customType, data),
      record: source.record?.bind(source),
    }, { exec: options.cellExec });
  }

  // -- Images -------------------------------------------------------------------------

  /** The canonical observation message of a step (the last one, if the step was preserved twice). */
  observationMessage(step: number): Json | null {
    const history = this.source.history();
    for (let index = history.length - 1; index >= 0; index--) {
      if (history[index]?._observation?.step === step) return history[index];
    }
    return null;
  }

  /** "› you: …" for each operator message delivered after observation `step` (before the next one). */
  operatorLines(step: number): string[] {
    const history = this.source.history();
    let start = -1;
    for (let index = history.length - 1; index >= 0; index--) {
      if (history[index]?._observation?.step === step) {
        start = index;
        break;
      }
    }
    if (start < 0) return [];
    const lines: string[] = [];
    for (const message of history.slice(start + 1)) {
      if (message?._observation) break;
      const source = String(message?._context_source ?? "");
      if (!source.startsWith("operator_message_") || typeof message.content !== "string") continue;
      const match = /<operator_message>\s*(\{[\s\S]*\})\s*<\/operator_message>/.exec(message.content);
      try {
        lines.push(`› you: ${JSON.parse(match?.[1] ?? "{}").message}`);
      } catch {
        // not an operator message ORA wrote
      }
    }
    return lines;
  }

  imagesForObservation(step: number): OraImage[] {
    const cached = this.observationImages.get(step);
    if (cached) return cached;
    const images = imagesOf(this.observationMessage(step), this.source.runRoot);
    if (images.length) this.observationImages.set(step, images);
    return images;
  }

  imagesForTool(toolCallId: string): OraImage[] {
    const cached = this.toolImageCache.get(toolCallId);
    if (cached) return cached;
    const candidates = [...this.source.pendingToolImages(), ...this.source.history()];
    const message = candidates.find((item) => item?._tool_image_call_id === toolCallId);
    const images = message ? imagesOf(message, this.source.runRoot) : [];
    if (images.length) this.toolImageCache.set(toolCallId, images);
    return images;
  }

  /** An image file of the run (a run-relative path under native/), or null. */
  private runImage(relative: Json): OraImage | null {
    if (typeof relative !== "string" || !relative.startsWith("native/") || relative.includes("..")) return null;
    const path = join(this.source.runRoot, relative);
    if (!existsSync(path)) return null;
    return { data: readFileSync(path).toString("base64"), mimeType: path.endsWith(".png") ? "image/png" : "image/jpeg",
      label: relative, caption: null };
  }

  /** Each image under its caption (dim), drawn by pi-tui or named by its artifact path. */
  private addImages(container: { addChild(component: Component): void }, images: OraImage[], theme: Theme, camera = false): void {
    for (const image of images) {
      container.addChild(new Spacer(1));
      const caption = camera ? cameraOf(image.caption) : image.caption;
      if (caption) container.addChild(new Text(theme.fg("dim", truncate(caption, 100)), 0, 0));
      container.addChild(new Image(
        image.data, image.mimeType, { fallbackColor: (text: string) => theme.fg("muted", text) },
        { maxWidthCells: 48, filename: image.label },
      ));
    }
  }

  // -- Status ---------------------------------------------------------------------------

  phase(now = Date.now()): string {
    if (!this.live) return t("phase.finished");
    if (this.status.terminated) return t("phase.ended", { state: this.status.terminated.state ?? "landing" });
    if (this.status.interrupted || !this.source.live()) return t("phase.stopping");
    const play = this.source.play();
    if (play?.state === "setup") {
      const running = (play.setup?.steps ?? []).find((step: Json) => step?.state === "running");
      return t("phase.setup", { step: running ? `: ${running.title ?? running.id}` : "" });
    }
    if (play && !play.running) return play.state === "ended" ? t("phase.ending") : t("phase.waiting");
    const activity = this.activity;
    if (activity !== null) {
      const s = elapsed(now - activity.since);
      switch (activity.kind) {
        case "thinking": return t("phase.thinking", { s });
        case "writing": return t("phase.writing", { s });
        case "tool": return t("phase.tool", { tool: activity.tool ?? "tool", s });
        case "moving": return t("phase.moving", { s });
        case "observing": return t("phase.observing");
      }
    }
    if (this.context && !this.context.isIdle()) return t("phase.modelWorking");
    if (this.source.inTurn()) return t("phase.agentTurn");
    return t("phase.robotActing");
  }

  /** Track what the agent is doing from Pi's events (the footer's thinking / writing / tool status). */
  private onAgentEvent(type: string, event: Json): void {
    const now = Date.now();
    switch (type) {
      case "turn_start":
        this.activity = { kind: "thinking", since: now };
        break;
      case "message_start":
        if (event?.message?.role === "assistant" && this.activity?.kind !== "writing") {
          this.activity = { kind: "writing", since: this.activity?.kind === "thinking" ? this.activity.since : now };
        }
        break;
      case "tool_execution_start":
        this.activity = { kind: "tool", since: now, tool: String(event?.toolName ?? "tool") };
        break;
      case "agent_end":
        // Free play: the Supervisor executes a submitted action next (action_started), or the
        // instruction ends. An episode's footer says the robot acts until the next observation.
        this.activity = this.source.play() ? { kind: "observing", since: now } : null;
        break;
    }
  }

  statusRows(): string[] {
    const status = this.status.refresh();
    const rows: string[] = [];
    const row = (label: string, value: string) => rows.push(`${label.padEnd(16)}${value}`);
    row("run", this.source.runId);
    row("model", `${this.source.model} · ${this.source.protocol}${this.source.profile ? ` · ${this.source.profile}` : ""}`);
    row("phase", this.phase());
    row("observation", status.observationStep === null ? "—" : `step ${status.observationStep}`);
    row("model calls", String(status.modelCalls));
    row("robot decisions", String(status.decisions));
    if (status.inputTokens || status.outputTokens) row("tokens", `${status.inputTokens} in · ${status.outputTokens} out`);
    const remaining = status.remaining();
    row("budget left", remaining.length ? remaining.map(([key, value]) => `${value} ${key}`).join(" · ") : "—");
    row("operator msgs", String(status.operatorMessages));
    const action = status.lastAction;
    if (action) {
      const verdict = action.ok === false ? `refused${action.refusal ? `: ${action.refusal}` : ""}`
        : action.ok === true ? "accepted" : "pending";
      row("last action", `${action.name}  ${argumentLine(action.arguments ?? {}, 70)}`);
      row("", `step ${action.step ?? "?"} · ${verdict}${action.actionId ? ` · ${action.actionId}` : ""}`);
      row("receipt", action.receipt ?? (action.ok === true ? "waiting for the environment" : "—"));
    } else {
      row("last action", "none yet");
    }
    if (status.lastTool) row("last tool", `${status.lastTool.name} · ${status.lastTool.ok === false ? "failed" : "ok"}`);
    if (status.terminated) row("terminated", scalars(status.terminated));
    const world = this.worldLine();
    if (this.live) row("world view", world ?? "none in this terminal (--world-view off, or not a terminal)");
    return rows;
  }

  statusLine(options: { phase?: boolean } = {}): string {
    const status = this.status.refresh();
    const play = this.source.play();
    const parts = [
      ...(play ? [t("footer.scene", { scene: play.scene, scenes: play.scenes }),
        t("footer.instruction", { n: play.instructions })] : []),
      status.observationStep !== null ? t("footer.step", { n: status.observationStep })
        : play ? t("footer.sceneLoaded") : t("footer.firstObservation"),
      t("footer.calls", { n: status.modelCalls }), t("footer.decisions", { n: status.decisions }),
      ...(options.phase === false ? [] : [this.phase()]),
    ];
    const queued = this.source.queuedNext?.() ?? 0;
    if (queued > 0) parts.push(t("footer.queued", { n: queued }));
    const remaining = status.remaining();
    if (remaining.length) parts.push(t("footer.budget", { items: remaining.map(([key, value]) => budgetItem(key, value)).join(", ") }));
    return parts.join(" · ");
  }

  /** The footer's mode, first on a real cell's footer: `▸▸ ask · every motion asked (shift+tab)`; null elsewhere. */
  modeLine(): { mode: string; text: string } | null {
    if (this.cell === null) return null;
    const mode = this.cell.modeState().mode;
    return { mode, text: `▸▸ ${modeLabel(mode)} (${t("cell.mode.cycle")})` };
  }

  private refreshFooter(): void {
    this.requestRender?.();
  }

  /** The reasoning effort now (Pi's thinking level), else the plan's. */
  effort(): string | null {
    let level: string | null = null;
    try {
      level = (this.pi as Json)?.getThinkingLevel?.() ?? null;
    } catch {
      level = null;
    }
    return level ?? whereOf(this.source.runRoot).effort;
  }

  /** The context the model holds now (Pi's estimate) of its window, or null before it is known. */
  contextLine(): string | null {
    let usage: Json = null;
    try {
      usage = this.context?.getContextUsage?.() ?? null;
    } catch {
      usage = null;
    }
    // A finished run's view has no window (its recorded model's is a placeholder).
    if (!usage || typeof usage.tokens !== "number" || !(usage.contextWindow >= 1000)) return null;
    return t("footer.context", { used: tokenCount(usage.tokens), window: tokenCount(usage.contextWindow) });
  }

  /** The header: the mark, then "ORA vX", model · effort, where the run is, the keys, the world view. */
  headerLines(theme: Theme, width: number): string[] {
    const version = versionOf();
    const where = whereOf(this.source.runRoot);
    const effort = this.effort();
    const play = this.live && this.source.play() !== null;
    const keys = play ? t("header.play")
      : this.live ? t("header.episode") : t("header.finished");
    const world = this.worldLine();
    const text = [
      `${theme.bold(theme.fg("toolTitle", "ORA"))}${version ? theme.fg("dim", ` v${version}`) : ""}`,
      theme.fg("dim", [this.source.model, effort ? t("header.effort", { effort }) : null,
        play || this.source.freePlay ? t("header.freePlay") : null].filter(Boolean).join(" · ")),
      theme.fg("dim", [where.place, this.source.runId].filter(Boolean).join(" · ")),
      theme.fg("borderMuted", keys),
      ...(world === null ? [] : [`${theme.fg("dim", "world view")} ${theme.fg("accent", world)}`]),
    ];
    const mark = logoLines();
    const markWidth = mark[0].length;
    if (width < markWidth + 30) return ["", ...text.map((line) => truncateToWidth(line, width)), ""];
    const rows = Math.max(mark.length, text.length);
    const lines: string[] = [""];
    for (let index = 0; index < rows; index++) {
      const art = index < mark.length ? theme.fg("accent", mark[index]) : " ".repeat(markWidth);
      lines.push(truncateToWidth(` ${art}  ${text[index] ?? ""}`, width));
    }
    lines.push("");
    return lines;
  }

  /** Free play's status line above the prompt: what the agent does now, with its seconds; null when idle. */
  statusRow(theme: Theme, now = Date.now()): string | null {
    if (!this.live) return null;
    const play = this.source.play();
    if (!play) return null;
    const activity = this.activity;
    if (activity === null || !play.running) {
      return play.state === "setup" || play.running ? null : `${theme.fg("borderMuted", "·")} ${theme.fg("dim", this.phase(now))}`;
    }
    const verb = activity.kind === "tool" ? t("status.tool", { tool: activity.tool ?? "tool" }) : t(`status.${activity.kind}`);
    const seconds = activity.kind === "observing" ? null : elapsed(now - activity.since);
    const hint = [seconds, t("status.hint")].filter(Boolean).join(" · ");
    return `${theme.fg("accent", spinnerFrame(now))} ${theme.fg("text", verb)} ${theme.fg("dim", `(${hint})`)}`;
  }

  /** The footer: "▸▸ mode · where · counts · context" on the left, "● effort · model" on the right; the speed under it. */
  footerLines(theme: Theme, width: number): string[] {
    const mode = this.modeLine();
    const marker = theme.fg("accent", "▸▸ ");
    const label = (text: string) => (mode === null ? theme.fg("text", text) : mode.mode === "auto"
      ? theme.bold(theme.fg("error", text)) : theme.fg(mode.mode === "safe" ? "warning" : "text", text));
    const named = mode === null ? (this.source.play() ? t("footer.freePlay") : this.live ? t("footer.episode") : t("footer.finished"))
      : mode.text.slice(3);
    let shown = `${marker}${label(named)}`;
    // Free play says what the agent does in its status line; an episode says it here. What does not fit
    // goes first: the budget, the counts, the step (/status has them all).
    const status = this.status.refresh();
    const play = this.source.play();
    const remaining = status.remaining();
    const queued = this.source.queuedNext?.() ?? 0;
    const context = this.contextLine();
    const parts: Array<[number, string]> = [
      ...(play ? [[9, t("footer.scene", { scene: play.scene, scenes: play.scenes })],
        [9, t("footer.instruction", { n: play.instructions })]] as Array<[number, string]> : []),
      [3, status.observationStep !== null ? t("footer.step", { n: status.observationStep })
        : play ? t("footer.sceneLoaded") : t("footer.firstObservation")],
      ...(queued > 0 ? [[8, t("footer.queued", { n: queued })]] as Array<[number, string]> : []),
      ...(this.live && !play ? [[7, this.phase()]] as Array<[number, string]> : []),
      ...(context ? [[6, context]] as Array<[number, string]> : []),
      [2, t("footer.calls", { n: status.modelCalls })], [2, t("footer.decisions", { n: status.decisions })],
      ...(remaining.length ? [[1, t("footer.budget", { items: remaining.map(([key, value]) => budgetItem(key, value)).join(", ") })]] as
        Array<[number, string]> : []),
    ];
    const effort = this.effort();
    const right = `${theme.fg("borderMuted", "●")}${theme.fg("dim", ` ${[effort, this.source.model].filter(Boolean).join(" · ")}`)} `;
    const room = width - visibleWidth(right) - 3;
    let shownParts = parts;
    const leftOf = (list: Array<[number, string]>) => ` ${shown}${list.length ? theme.fg("dim", ` · ${list.map(([, text]) => text).join(" · ")}`) : ""}`;
    for (const rank of [1, 2, 3, 0, 6, 7, 8]) {
      if (visibleWidth(leftOf(shownParts)) <= room) break;
      // Rank 0: the mode's key hint ("(shift+tab)") goes before the context does.
      if (rank === 0) shown = `${marker}${label(named.replace(/\s*\([^)]*\)$/, ""))}`;
      else shownParts = shownParts.filter(([level]) => level !== rank);
    }
    const left = leftOf(shownParts);
    const gap = width - visibleWidth(left) - visibleWidth(right);
    // Free play's speed under it (play/perf.ts), once a model call was timed.
    const perf = this.perfLine();
    return [truncateToWidth(gap >= 2 ? left + " ".repeat(gap) + right : left, width),
      ...(perf === null ? [] : [truncateToWidth(` ${theme.fg("dim", perf)}`, width)])];
  }

  /** The prompt's "❯", in the theme now (tui/host.ts markPrompt). */
  promptMark(): string {
    try {
      return this.ui?.theme?.fg?.("toolTitle", "❯") ?? "❯";
    } catch {
      return "❯";
    }
  }

  /** ORA's theme for the attached terminal's colors (again on every attach: tui/host.ts), unless ORA_TUI_THEME=pi. */
  applyTheme(): void {
    if (!wantsOraTheme() || typeof this.ui?.setTheme !== "function") return;
    try {
      this.ui.setTheme(oraTheme());
    } catch {
      // Pi's own theme stays
    }
  }

  /** The footer's second line in free play: the model's speed and the loop (play/perf.ts), or null. */
  perfLine(): string | null {
    return this.source.perfLine?.() ?? null;
  }

  /** A model call ended (play/perf.ts): the footer's speed line changed. */
  refreshPerf(): void {
    this.requestRender?.();
  }

  /** The attached terminal's world view changed (relayed by host.ts). */
  setWorld(state: Json): void {
    this.world = worldView(state);
    this.requestRender?.();
  }

  /** `world view  URL · what the page shows`, or null without a world view. */
  worldLine(): string | null {
    const world = this.world;
    if (world === null) return null;
    return `${world.url ?? "(starting)"} · ${worldSummary(world)}`;
  }

  // -- Operator input -------------------------------------------------------------------

  private notify(message: string, level: "info" | "warning" | "error" = "info"): void {
    this.ui?.notify?.(message, level);
  }

  /** A notice of free play's state for this terminal alone: a page hears the state itself (ora_play_state). */
  private notifyHere(message: string, level: "info" | "warning" | "error" = "info"): void {
    const ui = this.ui;
    if (typeof ui?.notifyLocal === "function") ui.notifyLocal(message, level);
    else ui?.notify?.(message, level);
  }

  helpRows(): string[] {
    const rows = this.live && this.source.play() ? [
      ...helpPlay(),
      ...(this.cell === null ? [] : ["", ...helpCell()]),
      "",
      "Re-attach with `ora attach " + this.source.runId + "`.",
    ] : this.live ? [
      "Enter          steer: the Supervisor delivers it before the next model call",
      "Esc            interrupt the model call now; this stops the run (operator stop)",
      "/stop          stop the run, after a confirmation: a model call in flight is",
      "               recorded first, so the run stays valid; also while the robot acts",
      "/status        step, model calls, remaining budget, last action and receipt",
      "/export [p]    write this session as HTML (.jsonl for the raw Pi session)",
      "               where `ora attach` runs",
      "/model         the model and provider profile this run is bound to",
      "/detach        leave; the run continues (Ctrl+D, Ctrl+Z and /quit too)",
      "Ctrl+O         expand observations and tool results",
      "Ctrl+\\         leave at once, from the attach client",
      "",
      "Re-attach with `ora attach " + this.source.runId + "`.",
      ...(this.world?.url ? [`The robot's cameras: ${this.world.url} (a browser on this computer,`,
        "or through `ora-remote` / an SSH-forwarded port from yours)."] : []),
    ] : [
      "This run has finished; its session is shown read-only.",
      "/export [p]    write this session as HTML (.jsonl for the raw Pi session)",
      "/status        the run's final counts",
      "Ctrl+O         expand observations and tool results",
      "Ctrl+D, /quit  close the view",
    ];
    if (this.live && this.source.freePlay) {
      rows.push("", "Pi's own commands work here: /model (Ctrl+P), /thinking, /login, /compact, /tree, /fork,",
        "/new, /resume, /clone, /import (the conversation moves; the robot and the scene stay as they",
        "are), /reload, /trust, /debug, /share and /bug (after a confirmation), ! shell, @file.");
      return rows;
    }
    const refused = Object.keys(this.live ? LIVE_REFUSALS : VIEW_REFUSALS).map((name) => `/${name}`).join(" ");
    rows.push("", "Not available here:", ...wrapTextWithAnsi(refused, 74));
    return rows;
  }

  /** Editor submissions Pi would handle as built-ins; true when ORA handled the text. */
  intercept(text: string): boolean {
    const trimmed = text.trim();
    if (this.live && this.source.freePlay) return this.interceptPlay(trimmed);
    if (trimmed.startsWith("!")) {
      this.notify("Shell commands are not available in an ORA run.", "warning");
      return true;
    }
    if (!trimmed.startsWith("/")) {
      if (!this.live && trimmed) {
        this.notify("This run has finished; its session is read-only.", "warning");
        return true;
      }
      return false;
    }
    const name = trimmed.slice(1).split(/\s+/, 1)[0];
    const argument = trimmed.slice(1 + name.length).trim();
    if (name === "model") {
      this.notify(`${this.source.model} via ${this.source.protocol}`
        + `${this.source.profile ? ` (profile ${this.source.profile})` : ""}; bound by the RunPlan. `
        + "Choose another for a new run with `ora interactive --provider PROFILE`.");
      return true;
    }
    if (name === "export") {
      void this.export(argument || null);
      return true;
    }
    if (name === "quit" && this.live && this.source.play()) {
      void this.sendControl("quit", "/quit", t("notice.quitting"));
      return true;
    }
    if ((name === "quit" || name === "detach") && this.host !== null) {
      this.host.detach(this.live ? "detached; the run continues" : "closed");
      return true;
    }
    if (name === "quit") return false; // a terminal that is the agent process itself quits
    const reason = (this.live ? LIVE_REFUSALS : VIEW_REFUSALS)[name];
    if (reason !== undefined) {
      this.notify(`/${name} is not available here: ${reason}.`, "warning");
      return true;
    }
    if (!PASSED_COMMANDS.has(name)) {
      // Pi would hand an unknown command to the input hook, which steers the model with it.
      this.notify(`Unknown command /${name}; /help lists ORA's commands.`, "warning");
      return true;
    }
    return false;
  }

  /** Re-run an editor submission through Pi's own handler (set by host.guardEditor). */
  resubmit: ((text: string) => Promise<void> | void) | null = null;

  /** Free play: Pi's commands pass; ORA's session controls and uploads stop here. */
  private interceptPlay(trimmed: string): boolean {
    if (!trimmed.startsWith("/")) return false;  // instructions, `!` shell, @file: Pi and the input hook
    const name = trimmed.slice(1).split(/\s+/, 1)[0];
    const argument = trimmed.slice(1 + name.length).trim();
    if (name === "quit") {
      void this.sendControl("quit", "/quit", t("notice.quitting"));
      return true;
    }
    if (name === "detach" && this.host !== null) {
      this.host.detach("detached; the session waits");
      return true;
    }
    if (name === "export") {
      void this.export(argument || null);
      return true;
    }
    if (name === "share" || name === "bug") {
      void this.confirmUpload(name, trimmed);
      return true;
    }
    if (CELL_COMMAND_NAMES.has(name) && this.cell === null) {
      // Unregistered, Pi would hand it to the input hook: an instruction to the model.
      this.notify(t("cell.realOnly", { name }), "warning");
      return true;
    }
    return false;
  }

  private async confirmUpload(name: string, text: string): Promise<void> {
    const what = name === "share"
      ? "This uploads the whole conversation (instructions, the agent's replies, tool calls and results, "
        + "camera images and shell output) to a GitHub gist, through your `gh` login."
      : "This sends a bug report with a summary of this session (and, if you choose, the session file) "
        + "to Pi's maintainers.";
    const confirmed = await this.ui?.confirm?.(`Upload with /${name}?`, `${what} Nothing is uploaded unless you confirm.`);
    if (!confirmed) {
      this.notify(`/${name} cancelled; nothing was uploaded.`);
      return;
    }
    await this.resubmit?.(text);
  }

  // -- Free play ------------------------------------------------------------------------

  private async sendControl(type: string, message: string, notice: string): Promise<void> {
    try {
      await this.source.control(type, message);
      this.notifyHere(notice, "info");
    } catch (error) {
      this.notifyHere(t("notice.sendRefused", { control: message, reason: (error as Error).message }), "error");
    }
  }

  /**
   * Esc while an instruction runs stops that instruction (consumed before Pi's own abort); on a real cell
   * a motion in progress ends at once where the arms are (/halt), and one the agent still plans is refused
   * at its approval (the same stop, withdrawn if the reply proposes none: cell_commands.ts stopPlan).
   * Alt+Enter while one runs queues the message being typed (Pi submits it: play/runtime.ts). On a real
   * cell, Shift+Tab switches to the next permission mode (consumed before Pi's thinking-level cycle).
   */
  onKey(data: string): { consume: boolean } | undefined {
    if (this.live && this.cell !== null && matchesKey(data, "shift+tab") && !this.tui?.hasOverlay?.()) {
      void this.cell.cycleMode({ ui: this.ui }).then((run) => run?.done.then(() => this.refreshFooter()));
      return { consume: true };
    }
    const play = this.source.play();
    if (!this.live || !play || !this.source.live() || !play.running || this.tui?.hasOverlay?.()) return undefined;
    if (matchesKey(data, "alt+enter")) {
      // Pi's own Alt+Enter submits the text next (a follow-up while its agent streams, else as Enter).
      const text = String(this.ui?.getEditorText?.() ?? "").trim();
      if (text && !text.startsWith("/") && !text.startsWith("!")) this.source.queueNext?.();
      return undefined;
    }
    if (!matchesKey(data, "escape")) return undefined;
    // On a real cell the motion in progress ends where the arms are first (/halt): the Supervisor reads the
    // stop only once the motion it executes has ended.
    let notice = t("notice.stopping");
    if (this.cell !== null && this.activity?.kind === "moving") this.cell.halt();
    else if (this.cell !== null) {
      this.cell.stopPlan(null);
      notice = t("notice.stoppingPlanned");
    }
    void this.sendControl("stop_instruction", "Esc", notice);
    return { consume: true };
  }

  /** The Supervisor announced where the free-play session is (display only). */
  playState(record: Json): void {
    const where = { scene: record?.scene, scenes: record?.scenes };
    this.cell?.playState(record);
    switch (record?.state) {
      case "scene":
        this.activity = null;
        this.notifyHere(record.cause === "environment_ended" ? t("notice.sceneEndedLoaded", where) : t("notice.sceneLoaded", where));
        if (typeof record.image === "string" || typeof record.operator_view === "string") {
          // A display-only entry (not sent to the model): it receives the scene with the first
          // instruction's observation, and never the operator view.
          this.pi?.appendEntry("ora.scene", { scene: record.scene, scenes: record.scenes, image: record.image ?? null,
            operator_view: record.operator_view ?? null });
        }
        break;
      case "instruction":
        // Pi shows the instruction as the user's message when the Supervisor delivers it.
        this.activity = { kind: "thinking", since: Date.now() };
        this.notifyHere(t("notice.instructionStarting", { n: record.instructions }));
        break;
      case "action_started":
        this.activity = { kind: "moving", since: Date.now() };
        break;
      case "message_queued_next":
        this.notifyHere(t("notice.queuedNext", { n: record.instructions, count: this.source.queuedNext?.() ?? 1 }));
        break;
      case "action_finished":
        this.activity = { kind: "observing", since: Date.now() };
        break;
      case "instruction_finished": {
        this.activity = null;
        const instruction = record.instruction ?? {};
        const ended = String(instruction.ended ?? "");
        const n = Number(instruction.index) + 1;
        this.notifyHere(t("notice.instructionFinished", { n, ending: endingText(ended) }), ended === "reported" ? "info" : "warning");
        // A marker in the conversation (display only): where and how the instruction ended.
        this.pi?.appendEntry("ora.instruction_end", { n, ended, step: instruction.observation_steps?.end ?? null,
          decisions: instruction.decisions ?? null,
          seconds: Number.isFinite(num(instruction.wall_duration_s)) ? num(instruction.wall_duration_s) : null });
        break;
      }
      case "control":
        if (record.control?.status === "rejected") {
          const reason = String(record.control.reason ?? "");
          this.notifyHere(t("notice.controlRefused", { control: record.control.command,
            reason: CONTROL_REFUSALS[reason] ? t(CONTROL_REFUSALS[reason]) : reason }), "warning");
        }
        break;
      case "ended":
        this.activity = null;
        this.notifyHere(t("notice.sessionEnding", { reason: record.termination }), "info");
        break;
      case "setup":
        this.cellSetup(record.setup ?? {});
        break;
    }
    this.refreshFooter();
  }

  /**
   * A real cell's setup before its worker starts (ora/orchestrator/episode/cell_setup.py): the checklist as an
   * entry at its start and its end, each step's start and end and each progress line as a notice. Its questions
   * are the broker's, answered with /approve and /decline (cell_commands.ts).
   */
  private cellSetup(setup: Json): void {
    const event = setup.event ?? {};
    const step = (setup.steps ?? []).find((item: Json) => item?.id === event.step);
    const title = String(step?.title ?? event.step ?? "");
    if (event.status !== undefined) {
      this.pi?.appendEntry(SETUP_ENTRY, setup);
      if (setup.next) this.notifyHere(t("setup.answer"), "info");
    } else if (event.step !== undefined && ["running", "done", "failed"].includes(event.state)) {
      this.notifyHere(t(`setup.step.${event.state}`, { title, why: event.why ? `: ${event.why}` : "" }),
        event.state === "failed" ? "error" : "info");
    } else if (typeof event.progress === "string") {
      this.notifyHere(event.progress, "info");
    }
    if (event.phase === "done" || event.phase === "failed") this.pi?.appendEntry(SETUP_ENTRY, setup);
  }

  /** An operator message was queued or delivered (play/messages.ts); the chat shows it once delivered. */
  playMessage(_message: PlayMessage): void {
    this.refreshFooter();
  }

  private async export(path: string | null): Promise<void> {
    const sessionFile = this.source.sessionFile();
    if (this.host === null || sessionFile === null) {
      this.notify("Nothing to export yet.", "warning");
      return;
    }
    if (path !== null && !/\.(html|jsonl)$/.test(path)) {
      this.notify("Export to a .html or .jsonl path.", "warning");
      return;
    }
    try {
      const written = await this.host.exportSession(path, sessionFile);
      this.notify(`Session exported to ${written}`);
    } catch (error) {
      this.notify(`Export failed: ${(error as Error).message}`, "error");
    }
  }

  // -- The Pi extension -----------------------------------------------------------------

  extension() {
    return (pi: ExtensionAPI) => {
      const tools = new Set(this.source.toolNames);
      this.pi = pi;

      pi.on("session_start", async (_event: Json, ctx: Json) => {
        this.context = ctx;
        if (ctx.mode !== "tui") return;
        this.ui = ctx.ui;
        this.applyTheme();
        ctx.ui.setHeader((_tui: Json, theme: Theme) => ({
          render: (width: number) => this.headerLines(theme, width),
          invalidate() {},
        }));
        if (this.live) ctx.ui.onTerminalInput?.((data: string) => this.onKey(data));
        if (this.live && this.source.freePlay) {
          // Free play's status line (statusRow) says what the agent does, also while the robot moves.
          ctx.ui.setWorkingVisible?.(false);
          ctx.ui.setWidget?.("ora.status", (_tui: Json, theme: Theme) => linesComponent((width) => {
            const row = this.statusRow(theme);
            return row === null ? [] : [truncateToWidth(` ${row}`, width)];
          }), { placement: "aboveEditor" });
        } else {
          ctx.ui.setWorkingIndicator?.({ frames: ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"]
            .map((frame) => ctx.ui.theme?.fg?.("accent", frame) ?? frame), intervalMs: 120 });
        }
        ctx.ui.setFooter((tui: Json, theme: Theme) => {
          this.tui = tui;
          this.requestRender = () => tui.requestRender();
          return {
            render: (width: number) => this.footerLines(theme, width),
            invalidate() {},
            dispose: () => {
              this.requestRender = null;
            },
          };
        });
        // The robot's questions (enable, each motion, release): said once each as they appear.
        if (this.live) this.cell?.watchApprovals();
        if (this.live && this.ticker === null) {
          // The live status counts seconds (thinking, moving) with its spinner, and the wall-time budget
          // counts down.
          let ticks = 0;
          this.ticker = setInterval(() => {
            ticks += 1;
            if (this.activity !== null && this.source.freePlay) {
              this.requestRender?.();
              return;
            }
            if (ticks % 8 !== 0) return;
            const mode = this.cell?.modeChanged() ?? false;
            if (mode || this.activity !== null || Date.now() % 5000 < 1000) this.requestRender?.();
          }, 125);
          this.ticker.unref();
        }
      });

      pi.on("agent_end", async () => {
        // The turn interrupted by /stop has ended at a boundary: now the Supervisor stops the run.
        if (!this.stopPending) return;
        this.stopPending = false;
        this.source.notifySupervisor();
      });

      for (const event of ["turn_start", "message_start", "tool_execution_start"] as const) {
        pi.on(event as Json, async (payload: Json) => {
          this.onAgentEvent(event, payload);
          this.refreshFooter();
        });
      }
      for (const event of ["turn_end", "tool_execution_end", "agent_end", "message_end"] as const) {
        pi.on(event as Json, async (payload: Json) => {
          if (event === "agent_end") this.onAgentEvent(event, payload);
          if (event === "turn_end") {
            // Tool images reach the history at the end of the reply.
            for (const [callId, invalidate] of this.waitingForImages) {
              if (this.imagesForTool(callId).length) {
                this.waitingForImages.delete(callId);
                invalidate();
              }
            }
          }
          this.refreshFooter();
        });
      }

      pi.registerMessageRenderer("ora.observation", (message: Json, options: Json, theme: Theme) => {
        const details = message.details ?? {};
        const step = Number(details.ora_observation_step);
        const box = new Box(1, 1, (text: string) => theme.bg("customMessageBg", text));
        const title = details.ora_resumed
          ? `observation ${step} · resumed after an agent-process restart`
          : `observation ${step}`;
        box.addChild(new Text(theme.fg("customMessageLabel", theme.bold(title))
          + (options.expanded ? "" : theme.fg("dim", "  Ctrl+O raw")), 0, 0));
        const display = typeof message.content === "string" ? message.content
          : (message.content ?? []).filter((block: Json) => block.type === "text").map((block: Json) => block.text).join("\n");
        const canonical = Number.isFinite(step) ? this.observationMessage(step) : null;
        if (options.expanded) {
          const raw = (canonical?.content ?? []).filter((block: Json) => block?.type === "text")
            .map((block: Json) => prettyJson(String(block.text))).join("\n") || prettyJson(display);
          box.addChild(new Spacer(1));
          box.addChild(new Text(raw, 0, 0));
        } else {
          const lines = observationLines(canonical, display);
          if (lines.length) box.addChild(new Text(theme.fg("muted", lines.join("\n")), 0, 0));
        }
        this.addImages(box, Number.isFinite(step) ? this.imagesForObservation(step) : [], theme, true);
        // The operator's messages the Supervisor delivered during this observation's turn, once delivered.
        box.addChild({
          render: (width: number) => {
            const lines = Number.isFinite(step) ? this.operatorLines(step) : [];
            if (!lines.length) return [];
            return [...new Spacer(1).render(width), ...new Text(lines.map((line) => theme.fg("accent", line)).join("\n"), 0, 0).render(width)];
          },
          invalidate() {},
        });
        return box;
      });

      // Free play: the observation is Pi's own message, images included (play/driver.ts).
      pi.registerMessageRenderer("ora.play_observation", (message: Json, options: Json, theme: Theme) => {
        const details = message.details ?? {};
        const blocks: Json[] = Array.isArray(message.content) ? message.content : [];
        const canonical = { content: blocks.map((block: Json) => (block?.type === "image"
          ? { type: "image_url", image_url: { url: `data:${block.mimeType};base64,${block.data}` } } : block)) };
        const display = blocks.filter((block: Json) => block?.type === "text").map((block: Json) => block.text).join("\n");
        const box = new Box(1, 1, (text: string) => theme.bg("customMessageBg", text));
        box.addChild(new Text(theme.fg("customMessageLabel", theme.bold(`observation ${details.ora_observation_step}`))
          + (options.expanded ? "" : theme.fg("dim", "  Ctrl+O raw")), 0, 0));
        if (options.expanded) {
          box.addChild(new Spacer(1));
          box.addChild(new Text(prettyJson(display.split("\n", 1)[0] ?? display), 0, 0));
        } else {
          const lines = observationLines(canonical, display);
          if (lines.length) box.addChild(new Text(theme.fg("muted", lines.join("\n")), 0, 0));
        }
        this.addImages(box, imagesOf(canonical, this.source.runRoot), theme, true);
        return box;
      });

      if (this.live && this.source.freePlay) {
        // The conversation can move; the world cannot.
        const moved = () => this.notify(t("notice.moved"), "info");
        pi.on("session_tree" as Json, async () => moved());
        pi.on("session_start" as Json, async (event: Json) => {
          if (["new", "resume", "fork"].includes(event?.reason)) moved();
        });
      }

      pi.registerEntryRenderer(SETUP_ENTRY, (entry: Json, _options: Json, theme: Theme) => {
        const data = entry.data ?? {};
        const box = new Box(1, 1, (text: string) => theme.bg("customMessageBg", text));
        box.addChild(new Text(theme.fg("customMessageLabel", theme.bold(t(`setup.title.${setupTitle(data)}`))), 0, 0));
        box.addChild(new Text(setupLines(data, theme).join("\n"), 0, 0));
        return box;
      });

      pi.registerEntryRenderer("ora.scene", (entry: Json, _options: Json, theme: Theme) => {
        const data = entry.data ?? {};
        const box = new Box(1, 1, (text: string) => theme.bg("customMessageBg", text));
        box.addChild(new Text(theme.fg("customMessageLabel", theme.bold(t("entry.scene", { scene: data.scene, scenes: data.scenes })))
          + theme.fg("dim", `  ${typeof data.operator_view === "string" ? t("header.operatorView") : t("entry.sceneCamera")}`), 0, 0));
        // The operator view first when the environment has one (yours alone), else the main camera.
        this.addImages(box, [data.operator_view, data.operator_view ? null : data.image]
          .map((relative) => this.runImage(relative)).filter((image): image is OraImage => image !== null), theme);
        return box;
      });

      pi.registerEntryRenderer("ora.operator_view", (entry: Json, _options: Json, theme: Theme) => {
        const data = entry.data ?? {};
        const box = new Box(1, 1, (text: string) => theme.bg("customMessageBg", text));
        box.addChild(new Text(theme.fg("customMessageLabel", theme.bold(t("entry.operatorView",
          { camera: data.camera, step: data.step }))), 0, 0));
        const image = this.runImage(data.path);
        if (image) this.addImages(box, [image], theme);
        return box;
      });

      // A message the operator sent while an instruction ran (play/driver.ts): it joined that instruction.
      pi.registerMessageRenderer("ora.play_message", (message: Json, _options: Json, theme: Theme) => {
        const details = message.details ?? {};
        const text = typeof message.content === "string" ? message.content
          : (message.content ?? []).filter((block: Json) => block?.type === "text").map((block: Json) => block.text).join("\n");
        const who = Number.isInteger(details.ora_instruction) ? t("entry.joined", { n: details.ora_instruction })
          : t("entry.joinedNoNumber");
        const read = Number.isInteger(details.ora_observation_step) ? ` · ${t("entry.readAt", { step: details.ora_observation_step })}` : "";
        const box = new Box(1, 0, (line: string) => theme.bg("userMessageBg", line));
        box.addChild(new Text(theme.fg("accent", `› ${who}`) + theme.fg("dim", read), 0, 0));
        box.addChild(new Text(text, 0, 0));
        return box;
      });

      // Where and how an instruction ended (display only).
      pi.registerEntryRenderer("ora.instruction_end", (entry: Json, _options: Json, theme: Theme) => {
        const data = entry.data ?? {};
        const ended = String(data.ended ?? "");
        const line = t("entry.instructionEnd", { n: data.n, ending: endingText(ended), step: data.step ?? "?" });
        const mark = ended === "reported" ? theme.fg("success", "✓") : ended === "operator_stopped" ? theme.fg("error", "■")
          : theme.fg("warning", "■");
        return new Text(`${mark} ${theme.fg(ended === "reported" ? "muted" : "warning", line)}`, 1, 0);
      });

      // A real cell's command and its outcome (display only; a finished run's view draws them too).
      pi.registerEntryRenderer(CELL_ENTRY, renderCellEntry);
      this.cell?.register(pi);


      pi.registerMessageRenderer("ora.continuation", (message: Json, _options: Json, theme: Theme) => {
        const text = typeof message.content === "string" ? message.content : "";
        return new Text(theme.fg("muted", `↻ ${text.split("\n")[0]}`), 1, 0);
      });

      // The model's words after "●", the operator's after "❯" (display only).
      if (wantsOraTheme()) {
        (pi as Json).registerMarkdownTransformer?.((markdown: string, context: Json) =>
          transcriptMarkdown(markdown, String(context?.messageType ?? "")));
      }

      // A robot tool call as the teaser draws it: "● name(args)", its result under a "⎿" corner.
      pi.registerToolRenderer((toolName: string, next: () => Json) => {
        if (!tools.has(toolName)) return next();
        return {
          renderShell: "self",
          renderCall: (args: Json, theme: Theme, context: Json) => {
            const pad = " ".repeat(Number(context?.outputPad ?? 1));
            const title = theme.bold(theme.fg("toolTitle", toolName));
            // White once its result is in, gray while it runs, red when it failed.
            const dot = context?.isError ? theme.fg("error", "●")
              : context?.isPartial === false ? theme.fg("toolTitle", "●") : theme.fg("dim", "●");
            // A code-mode command (ORA's bash): its lines as written, the heredoc's program included.
            const command = toolName === "bash" ? commandLines(args ?? {}, Boolean(context?.expanded)) : null;
            if (command !== null) {
              return linesComponent((width) => [
                ...hanging(title, width, `${pad}${dot} `, `${pad}  `, pad.length + 2),
                ...command.map((line) => truncateToWidth(
                  `${pad}  ${theme.fg("borderMuted", "│ ")}${theme.fg("muted", line)}`, width)),
              ]);
            }
            if (context?.expanded) {
              return linesComponent((width) => [
                ...hanging(title, width, `${pad}${dot} `, `${pad}  `, pad.length + 2),
                ...JSON.stringify(args ?? {}, null, 2).split("\n").map((line) => truncateToWidth(`${pad}  ${theme.fg("muted", line)}`, width)),
              ]);
            }
            const call = callText(toolName, args ?? {});
            return linesComponent((width) => [
              ...hanging(`${title}${theme.fg("muted", call.args)}`, width, `${pad}${dot} `, `${pad}  `, pad.length + 2),
              ...call.more.flatMap((line) => hanging(theme.fg("muted", line), width, `${pad}    `, `${pad}    `, pad.length + 4)),
            ]);
          },
          renderResult: (result: Json, options: Json, theme: Theme, context: Json) => {
            const container = new Container();
            const pad = " ".repeat(Number(context?.outputPad ?? 1));
            const text = (result.content ?? []).filter((block: Json) => block.type === "text")
              .map((block: Json) => block.text).join("\n");
            let lines: string[] = [];
            let script: Json = null;
            if (toolName === "bash" && text) {
              try {
                script = JSON.parse(text);
              } catch {
                script = null;
              }
            }
            if (isScriptResult(script)) {
              // A code-mode command: how it ended, each robot call of its program (a dry run as a plan), its
              // error and the end of what it printed; Ctrl+O shows every call and the whole output.
              const color = (line: ScriptLine) => line.failed ? theme.fg("error", line.text)
                : line.kind === "call" ? (line.dryRun ? theme.fg("accent", line.text) : theme.fg("toolOutput", line.text))
                : line.kind === "output" ? theme.fg("dim", `│ ${line.text}`)
                : line.kind === "more" ? theme.fg("dim", line.text) : theme.fg("toolOutput", line.text);
              lines = scriptResultLines(script, Boolean(options.expanded)).map(color);
            } else if (options.expanded) {
              lines = prettyJson(text).split("\n").map((line) => theme.fg("toolOutput", line));
            } else if (text) {
              let parsed: Json = null;
              try {
                parsed = JSON.parse(text);
              } catch {
                parsed = null;
              }
              const known = toolResultLines(toolName, parsed);
              if (known.length) {
                lines = (known.length > RESULT_MAX ? [...known.slice(0, RESULT_MAX), `… +${known.length - RESULT_MAX} more`] : known)
                  .map((line) => theme.fg("toolOutput", line));
              } else {
                const summary = resultSummary(text);
                lines = [summary.status === "refused" ? theme.fg("error", `✗ ${summary.line}`) : theme.fg("toolOutput", summary.line)];
              }
            }
            if (lines.length) {
              const corner = theme.fg("borderMuted", "⎿  ");
              container.addChild(linesComponent((width) => lines.flatMap((line, index) =>
                hanging(line, width, `${pad}  ${index === 0 ? corner : "   "}`, `${pad}     `, pad.length + 5))));
            }
            const own = (result.content ?? []).filter((block: Json) => block?.type === "image");
            const images = own.length
              ? imagesOf({ content: own.map((block: Json) => ({ type: "image_url",
                image_url: { url: `data:${block.mimeType};base64,${block.data}` } })) }, this.source.runRoot)
              : this.imagesForTool(context.toolCallId);
            if (!images.length && !options.isPartial && !this.source.freePlay) {
              this.waitingForImages.set(context.toolCallId, context.invalidate);
            }
            this.addImages(container, images, theme);
            return container;
          },
        };
      });

      pi.registerCommand("stop", {
        description: "Stop this run now (operator stop; the episode ends as operator_interrupt)",
        handler: async (_args: string, ctx: Json) => {
          if (!this.live || !this.source.live()) {
            ctx.ui.notify("This run is no longer active.", "info");
            return;
          }
          const confirmed = await ctx.ui.confirm(
            "Stop this run?",
            "ORA ends the episode now, with the operator as termination authority. "
              + "The run is still evaluated and sealed (incomplete).",
          );
          if (!confirmed || !this.source.live()) return;
          if (ctx.isIdle()) {
            this.source.stop();
            ctx.ui.notify("Stop requested; the Supervisor is ending the episode.", "warning");
          } else {
            // Cutting a request short leaves it without a recorded response; let it land first.
            this.stopPending = true;
            this.source.interruptAtBoundary();
            ctx.ui.notify("Stopping once the current model reply is recorded; no further call or tool runs.", "warning");
          }
          this.refreshFooter();
        },
      });

      pi.registerCommand("view", {
        description: "Free play: show the operator view (the environment's camera for you, never the agent's)",
        handler: async () => {
          const view = this.frames.operatorView();
          if (view === null) {
            this.notify(t("notice.noOperatorView"), "info");
            return;
          }
          this.pi?.appendEntry("ora.operator_view", { camera: view.camera, step: view.step,
            path: `native/operator-view/${String(view.step).padStart(6, "0")}.png` });
        },
      });

      pi.registerCommand("reset", {
        description: "Free play: load another scene drawn for this session",
        handler: async () => {
          if (!this.live || !this.source.play() || !this.source.live()) {
            this.notify("/reset loads another scene in a live free-play session (`ora play`).", "warning");
            return;
          }
          await this.sendControl("reset", "/reset", t("notice.resetting"));
        },
      });

      pi.registerCommand("status", {
        description: "Step, model calls, remaining budget, last action and its receipt",
        handler: async (_args: string, ctx: Json) => {
          if (ctx.mode !== "tui") return;
          await ctx.ui.custom((_tui: Json, theme: Theme, _keys: Json, done: (value: undefined) => void) =>
            panel(theme, "ORA run status", this.statusRows(), () => done(undefined)), { overlay: true });
        },
      });

      pi.registerCommand("help", {
        description: "ORA keys and commands",
        handler: async (_args: string, ctx: Json) => {
          if (ctx.mode !== "tui") return;
          await ctx.ui.custom((_tui: Json, theme: Theme, _keys: Json, done: (value: undefined) => void) =>
            panel(theme, this.live ? "ORA interactive" : "ORA run (read-only)", this.helpRows(), () => done(undefined)),
          { overlay: true });
        },
      });

      pi.registerCommand("detach", {
        description: this.live ? "Leave this terminal; the run continues (ora attach to return)" : "Close this view",
        handler: async () => {
          this.host?.detach(this.live ? "detached; the run continues" : "closed");
        },
      });
    };
  }

  /** The Supervisor opened a policy turn. */
  observeTurn(step: number, remainingBudget: Json): void {
    this.status.observeTurn(step, remainingBudget);
    this.refreshFooter();
  }

  close(): void {
    if (this.ticker !== null) clearInterval(this.ticker);
    this.ticker = null;
    this.cell?.stopWatching();
  }
}

export { formatSeconds };
