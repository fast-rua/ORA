// What the operator reads in the TUI: one-line summaries of observations, tool
// calls and tool results. Display only: these functions read ORA's canonical
// messages and the tool-result text and never change them; the raw JSON stays
// one Ctrl+O away. They know common field names, not environments, and fall
// back to the first line of the text when a value has an unknown shape.

type Json = any;

const NOISY_KEY = /(sha256|_unix_s$|^schema$|^bound_observation|^policy_state$|^forbidden_modalities)/;
const HEADLINE_KEYS = [
  "success", "task_success", "outcome", "status", "directive", "action", "camera", "arm", "reason",
];
const TEXT_KEYS = ["content", "text", "markdown", "message", "summary"];

export function truncate(text: string, width: number): string {
  return text.length > width ? `${text.slice(0, Math.max(0, width - 1)).trimEnd()}…` : text;
}

function number(value: number): string {
  if (Number.isInteger(value)) return String(value);
  return String(Number(value.toFixed(3)));
}

/** A value as it reads in one line: numbers rounded, short vectors inline, the rest counted. */
export function brief(value: Json, width = 40): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "number") return number(value);
  if (typeof value === "boolean") return String(value);
  if (typeof value === "string") {
    const line = value.split("\n").find((part) => part.trim()) ?? "";
    const short = truncate(line.trim(), width);
    return /\s/.test(short) ? `"${short}"` : short;
  }
  if (typeof value === "object" && "value" in value && typeof value.value === "number" && Object.keys(value).length === 1) {
    return number(value.value); // PyFloat
  }
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    if (value.length <= 8 && value.every((item) => typeof item === "number")) {
      return `[${value.map(number).join(", ")}]`;
    }
    if (value.length <= 4 && value.every((item) => typeof item === "string" && item.length <= 24)) {
      return value.join(", ");
    }
    return `${value.length} items`;
  }
  return "{…}";
}

function isSmall(value: Json): boolean {
  return value !== null && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length <= 4
    && Object.values(value).every((item) => item === null || typeof item !== "object"
      || (Array.isArray(item) && item.length <= 8 && item.every((entry) => typeof entry === "number")));
}

/** Nested objects as `outer.inner=value` leaves (three levels, ten leaves); false when larger. */
function leaves(value: Json, prefix: string, depth: number, out: string[]): boolean {
  const pyFloat = value && typeof value === "object" && Object.keys(value).length === 1 && "value" in value;
  if (value && typeof value === "object" && !Array.isArray(value) && !pyFloat && depth < 3) {
    for (const [key, item] of Object.entries(value)) {
      if (!leaves(item, `${prefix}.${key}`, depth + 1, out)) return false;
    }
    return true;
  }
  out.push(`${prefix}=${brief(value)}`);
  return out.length <= 10;
}

/** A tool call's arguments as `key=value` pairs; small nested objects read as `outer.inner=value`. */
export function argumentLine(args: Json, width = 120): string {
  if (!args || typeof args !== "object" || Array.isArray(args)) return truncate(brief(args), width);
  const parts: string[] = [];
  for (const [key, value] of Object.entries(args)) {
    const nested: string[] = [];
    parts.push(...(leaves(value, key, 1, nested) ? nested : [`${key}=${brief(value)}`]));
  }
  return truncate(parts.join("  "), width);
}

export interface ResultSummary {
  status: "ok" | "refused" | "plain";
  line: string;
}

function firstLine(text: string): string {
  return (text.split("\n").find((line) => line.trim()) ?? "").trim();
}

function reasonOf(error: Json): string {
  if (typeof error === "string") return error;
  if (error && typeof error === "object") {
    const parts = [error.code, error.message ?? error.reason ?? error.detail].filter((part) => typeof part === "string" && part);
    if (parts.length) return parts.join(": ");
  }
  return brief(error, 80);
}

/** A tool result as accepted/refused and why, or its salient fields. */
export function resultSummary(text: string, width = 140): ResultSummary {
  let value: Json;
  try {
    value = JSON.parse(text);
  } catch {
    return { status: "plain", line: truncate(firstLine(text), width) };
  }
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return { status: "plain", line: truncate(brief(value, width), width) };
  }
  if (value.ok === false || ("error" in value && value.error !== null && value.error !== undefined)) {
    return { status: "refused", line: truncate(`refused · ${reasonOf(value.error ?? value.reason ?? "not ok")}`, width) };
  }
  if (value.accepted === false) {
    return { status: "refused", line: truncate(`not accepted · ${reasonOf(value.reason ?? value.refusal ?? "")}`, width) };
  }
  const parts: string[] = [];
  if (value.accepted === true) parts.push(`accepted${typeof value.action_id === "string" ? ` · ${value.action_id}` : ""}`);
  const used = new Set(["accepted", "action_id", "ok"]);
  for (const key of HEADLINE_KEYS) {
    if (key in value && !used.has(key) && (value[key] === null || typeof value[key] !== "object")) {
      parts.push(`${key}=${brief(value[key])}`);
      used.add(key);
    }
  }
  const textKey = TEXT_KEYS.find((key) => typeof value[key] === "string" && !used.has(key));
  if (textKey && parts.length === 0) {
    parts.push(brief(value[textKey], width - 2).replace(/^"|"$/g, ""));
    used.add(textKey);
  }
  for (const [key, item] of Object.entries(value)) {
    if (used.has(key) || NOISY_KEY.test(key) || parts.length >= 6) continue;
    if (Array.isArray(item) && item.length && item.every((entry) => entry && typeof entry === "object")) {
      parts.push(`${key} ×${item.length}`);
    } else if (item === null || typeof item !== "object" || isSmall(item)) {
      if (typeof item === "string" && item.length > 60) continue;
      parts.push(`${key}=${brief(item)}`);
    }
  }
  return { status: "ok", line: truncate(parts.join(" · ") || "done", width) };
}

/** The short text block before each image of a message (its caption), or null. */
export function imageCaptions(message: Json): Array<string | null> {
  const captions: Array<string | null> = [];
  let previous: string | null = null;
  for (const block of Array.isArray(message?.content) ? message.content : []) {
    if (block?.type === "image_url" || block?.type === "ora_image_ref") {
      captions.push(previous);
      previous = null;
    } else if (block?.type === "text") {
      const text = String(block.text ?? "");
      previous = text.length <= 200 && !text.trimStart().startsWith("{") ? firstLine(text) : null;
    }
  }
  return captions;
}

/** The camera a caption names (`KIND · sequence N · camera · ...`), else the caption itself. */
export function cameraOf(caption: string | null): string | null {
  if (!caption) return null;
  const parts = caption.split(" · ").map((part) => part.trim());
  return parts.length >= 3 && /^sequence \d+$/.test(parts[1]) ? parts[2] : caption;
}

function budgetLine(budget: Json): string | null {
  if (!budget || typeof budget !== "object") return null;
  const parts: string[] = [];
  const value = (item: Json) => (item && typeof item === "object" && "value" in item ? item.value : item);
  if (budget.robot_decisions !== undefined) parts.push(`${brief(value(budget.robot_decisions))} robot decisions`);
  if (budget.model_calls !== undefined) parts.push(`${brief(value(budget.model_calls))} model calls`);
  const wall = value(budget.wall_time_s);
  if (typeof wall === "number") {
    const minutes = Math.floor(wall / 60);
    parts.push(minutes >= 60 ? `${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(2, "0")}m`
      : `${minutes}m${String(Math.round(wall % 60)).padStart(2, "0")}s`);
  }
  return parts.length ? `budget left: ${parts.join(" · ")}` : null;
}

/**
 * A summary's comma-separated parts without a part that repeats the one before it. The model-visible
 * `action_summary` can say "executed, executed" (its outcome and its motion clause both read
 * "executed"); the display says it once and the canonical text is unchanged (Ctrl+O shows it).
 */
export function collapseRepeats(summary: string): string {
  const parts = summary.split(", ");
  return parts.filter((part, index) => index === 0 || part !== parts[index - 1]).join(", ");
}

function effectLine(observation: Json): string | null {
  const effect = observation?.last_action_effect;
  if (effect && typeof effect === "object") {
    const what = typeof effect.action === "string" ? effect.action : "last action";
    const how = typeof effect.action_summary === "string" ? collapseRepeats(effect.action_summary)
      : resultSummary(JSON.stringify(effect), 100).line;
    return `after ${what}: ${how}`;
  }
  const previous = observation?.current_observation?.previous_execution;
  if (previous && typeof previous === "object") {
    const what = previous.action?.action ?? "the last action";
    const parts = [previous.consumed === false ? "not executed" : "executed"];
    if (typeof previous.pose_reached === "boolean") parts.push(previous.pose_reached ? "pose reached" : "pose not reached");
    if (typeof previous.state_changed === "boolean" && !previous.state_changed) parts.push("state unchanged");
    return `after ${what}: ${parts.join(", ")}`;
  }
  return null;
}

function endEffectorLine(observation: Json): string | null {
  const effectors = observation?.current_observation?.proprioception?.end_effectors;
  if (!effectors || typeof effectors !== "object") return null;
  const parts = Object.entries(effectors).map(([arm, state]: [string, Json]) => {
    const pose = Array.isArray(state?.pose_xyz_xyzw) ? state.pose_xyz_xyzw.slice(0, 3) : null;
    const gripper = state?.gripper !== undefined ? ` gripper ${brief(state.gripper)}` : "";
    return `${arm} ${pose ? brief(pose) : "?"}${gripper}`;
  });
  return parts.length ? parts.join(" · ") : null;
}

/** The observation's summary: cameras and images, the last action's effect, the arms, the budget. */
export function observationLines(message: Json | null, display: string): string[] {
  const textBlock = Array.isArray(message?.content)
    ? message.content.find((block: Json) => block?.type === "text")?.text
    : typeof message?.content === "string" ? message.content : null;
  let observation: Json = null;
  for (const candidate of [textBlock, display.split("\n", 1)[0]]) {
    if (typeof candidate !== "string") continue;
    try {
      observation = JSON.parse(candidate);
      break;
    } catch {
      // the display text is cut at 4000 characters; the canonical block is whole
    }
  }
  if (observation === null || typeof observation !== "object") return display ? [truncate(firstLine(display), 140)] : [];
  const lines: string[] = [];
  const captions = message ? imageCaptions(message) : [];
  const cameras = captions.map(cameraOf).filter((camera): camera is string => camera !== null);
  const main = observation.current_observation?.main_camera;
  const named = cameras.length ? [...new Set(cameras)] : typeof main === "string" ? [main] : [];
  const images = captions.length;
  if (named.length || images) {
    lines.push([named.length ? named.join(", ") : null, images ? `${images} image${images === 1 ? "" : "s"}` : null]
      .filter(Boolean).join(" · "));
  }
  for (const line of [effectLine(observation), endEffectorLine(observation),
    budgetLine(observation.remaining_budget ?? observation.current_observation?.remaining_budget)]) {
    if (line) lines.push(truncate(line, 160));
  }
  return lines;
}

// -- The teaser's transcript (look/shots/term.py call_lines/result_lines; the page's
// ora-pi/web/src/lib/terminal.ts draws the same): a call as `name(key=value, …)` with the
// arguments the model gave in a fixed order, its result as a few short lines under it.

/** The arguments a call names first, in this order; the rest follow as the model gave them. */
const CALL_ORDER = ["text", "camera", "view", "points_uv", "point_ids", "step", "target_ref", "candidate_id", "arm", "action",
  "ticks", "interaction", "position_mode", "position_frame", "position_m", "translation_m", "approach_world", "jaw_line_world",
  "orientation_mode", "orientation_xyzw", "max_candidates"];
const STRING_MAX = 70;
const ARGS_MAX = 300;
const MORE_MAX = 8;
export const RESULT_MAX = 4;

function clip(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

/** One value as a call line shows it: short floats, quoted strings elided past 70 characters. */
export function callValue(item: Json): string {
  if (item && typeof item === "object" && Object.keys(item).length === 1 && typeof item.value === "number") {
    item = item.value;                                                       // PyFloat
  }
  if (typeof item === "number") {
    if (Number.isInteger(item)) return String(item);
    return Math.abs(item) < 10 ? item.toFixed(4).replace(/0+$/, "").replace(/\.$/, "") : item.toFixed(1);
  }
  if (typeof item === "string") return `"${item.length <= STRING_MAX ? item : `${item.slice(0, STRING_MAX - 2)}…`}"`;
  if (Array.isArray(item)) return `[${item.map(callValue).join(", ")}]`;
  if (item && typeof item === "object") return `{${Object.entries(item).map(([key, inner]) => `${key}: ${callValue(inner)}`).join(", ")}}`;
  return item === undefined ? "" : JSON.stringify(item);
}

function kwargs(args: Json): string {
  const keys = [...CALL_ORDER.filter((key) => key in args), ...Object.keys(args).filter((key) => !CALL_ORDER.includes(key))];
  return keys.map((key) => `${key}=${callValue(args[key])}`).join(", ");
}

function v3(xs: Json): string {
  return Array.isArray(xs) ? `(${xs.slice(0, 3).map((x: Json) => Number(x?.value ?? x).toFixed(3)).join(", ")})` : "";
}

/** A call as the model made it: its arguments in parentheses, and a compute's bindings and expressions below. */
export function callText(name: string, args: Json): { args: string; more: string[] } {
  const given = args && typeof args === "object" && !Array.isArray(args) ? args : {};
  if (name === "compute") {
    const more = [
      ...(Array.isArray(given.bindings) ? given.bindings : []).map((binding: Json) =>
        `${binding?.name} = ${callValue(binding?.value)}${binding?.from ? `   # ${binding.from}` : ""}`),
      ...String(given.expressions ?? "").replace(/\\n/g, "\n").split("\n").filter((line) => line.trim()),
    ].map((line) => clip(line, 160));
    return { args: `(outputs=${callValue(given.outputs ?? [])})`,
      more: more.length > MORE_MAX ? [...more.slice(0, MORE_MAX - 1), `… +${more.length - MORE_MAX + 1}`] : more };
  }
  if (given.targets && typeof given.targets === "object" && !Array.isArray(given.targets)) {
    // Per-arm targets: move_pose(left: interaction=…; right: …), set_gripper(left="close").
    const parts = Object.entries(given.targets).map(([arm, target]) => (target && typeof target === "object" && !Array.isArray(target)
      ? `${arm}: ${kwargs(target)}` : `${arm}=${callValue(target)}`));
    const rest = Object.fromEntries(Object.entries(given).filter(([key]) => key !== "targets"));
    return { args: clip(`(${parts.join("; ")}${Object.keys(rest).length ? `, ${kwargs(rest)}` : ""})`, ARGS_MAX), more: [] };
  }
  return { args: clip(`(${kwargs(given)})`, ARGS_MAX), more: [] };
}

/** What a known tool's result says, a line per candidate or mark; empty for a shape it does not know. */
export function toolResultLines(name: string, result: Json): string[] {
  const out: string[] = [];
  const list = (key: string): Json[] => (Array.isArray(result?.[key]) ? result[key] : []);
  const n = (value: Json) => Number(value?.value ?? value ?? 0);
  if (name === "vision_ground" || name === "vision_segment") {
    for (const candidate of list("candidates").slice(0, 4)) {
      const where = candidate?.localization ?? {};
      const height = where.height_above_support_m;
      out.push(`${candidate?.candidate_id ?? "?"}  score ${n(candidate?.backend_score).toFixed(2)}`
        + (where.center_xyz_m ? `  centre ${v3(where.center_xyz_m)} m` : "")
        + (height !== undefined && height !== null ? `  height ${(1000 * n(height)).toFixed(0)} mm` : ""));
    }
  } else if (name === "grasp_plan") {
    list("candidates").slice(0, 3).forEach((candidate: Json, index: number) => {
      out.push(`G${index}  score ${n(candidate?.backend_score).toFixed(2)}`
        + (candidate?.pregrasp_pose?.position_m ? `  pregrasp ${v3(candidate.pregrasp_pose.position_m)} m` : "")
        + `  opening ${(1000 * n(candidate?.gripper_opening_m)).toFixed(0)} mm  `
        + (candidate?.ik_reachability?.reachable ? "reachable" : "unreachable"));
    });
  } else if (name === "mark_point") {
    for (const mark of list("marks").slice(0, 6)) {
      out.push(`${mark?.point_id ?? "?"}  ${mark?.xyz_m ? `${v3(mark.xyz_m)} m` : mark?.error_code ?? mark?.status ?? ""}`);
    }
  } else if (name === "compute") {
    for (const output of list("outputs")) out.push(clip(`${output?.name} = ${callValue(output?.value)}`, 160));
  } else if (name === "grasp_execute" && Array.isArray(result?.committed_sequence)) {
    out.push(`committed: ${result.committed_sequence.join(" → ")}`);
  } else if (name === "observe" && Array.isArray(result?.views)) {
    out.push(`views ${result.views.join(", ")}`);
  } else if (name === "geometry_query" && result?.region_result?.depth_m) {
    const depth = result.region_result.depth_m;
    out.push(`depth ${n(depth.min).toFixed(3)}–${n(depth.max).toFixed(3)} m, median ${n(depth.median).toFixed(3)} m`
      + (result.camera ? `  ${result.camera}` : ""));
  }
  return out;
}

// -- Code mode: a `bash` command and what its programs did ------------------------------

/** Lines of a code-mode command a collapsed call shows; Ctrl+O shows all of it. */
export const COMMAND_MAX = 12;
/** Script calls a collapsed result lists; the rest are counted. */
export const SCRIPT_CALLS_MAX = 8;
/** Lines of a command's output a collapsed result keeps (its end: what the program printed last). */
export const OUTPUT_TAIL = 4;

/**
 * A code-mode `bash` call (plan policy.code_mode): its command line by line, so the heredoc's program reads as
 * written; null for arguments of another shape. `expanded` (Ctrl+O) keeps every line.
 */
export function commandLines(args: Json, expanded = false): string[] | null {
  const command = args && typeof args === "object" ? args.command : undefined;
  if (typeof command !== "string") return null;
  const lines = command.replace(/\s+$/, "").split("\n");
  const timeout = typeof args.timeout === "number" ? [`# timeout ${args.timeout} s`] : [];
  if (expanded || lines.length <= COMMAND_MAX) return [...lines, ...timeout];
  return [...lines.slice(0, COMMAND_MAX - 1), `… +${lines.length - COMMAND_MAX + 1} lines (Ctrl+O)`, ...timeout];
}

/** Whether a tool result is a code-mode command's (ora/agent/pi/code_mode.py `_code_mode_bash`). */
export function isScriptResult(result: Json): boolean {
  return Boolean(result && typeof result === "object" && typeof result.status === "string"
    && Array.isArray(result.calls) && "output" in result);
}

export interface ScriptLine {
  /** "head": the command's ending; "call": one robot call; "error"; "output": a line it printed. */
  kind: "head" | "call" | "error" | "output" | "more";
  text: string;
  /** The line reads as a failure (red). */
  failed?: boolean;
  /** A dry run's plan (nothing moved). */
  dryRun?: boolean;
}

/** One robot call of a script as its row reads: `#002 move_pose ✓ action 1 · summary` (a dry run: `◇ plan`). */
export function scriptCallLine(entry: Json): ScriptLine {
  const seq = `#${String(entry?.call ?? "?").padStart(3, "0")}`;
  const tool = String(entry?.tool ?? "?");
  const dryRun = entry?.dry_run === true;
  const ok = entry?.ok !== false;
  const mark = entry?.running === true ? (dryRun ? "◇ plan …" : "…") : dryRun ? (ok ? "◇ plan" : "◇ plan ✗") : ok ? "✓" : "✗";
  const parts = [
    ...(entry?.reused ? ["reused"] : []),
    ...(typeof entry?.action === "number" ? [`action ${entry.action}`] : []),
    ...(typeof entry?.error === "string" ? [entry.error] : []),
    ...(typeof entry?.images === "number" && entry.images > 0 ? [`${entry.images} image${entry.images > 1 ? "s" : ""}`] : []),
    ...(typeof entry?.summary === "string" && entry.summary ? [firstLine(entry.summary)] : []),
  ];
  return { kind: "call", text: truncate(`${seq} ${tool} ${mark}${parts.length ? `  ${parts.join(" · ")}` : ""}`, 160),
    failed: !ok, dryRun };
}

/**
 * A code-mode command's result as the operator reads it: how it ended (status, exit code, calls, actions, time),
 * each robot call its program made (in order, a dry run marked as a plan), the error that ended it, and the end
 * of what it printed. Collapsed it keeps `SCRIPT_CALLS_MAX` calls and the last `OUTPUT_TAIL` output lines;
 * `expanded` (Ctrl+O) keeps all.
 */
export function scriptResultLines(result: Json, expanded = false): ScriptLine[] {
  const calls: Json[] = Array.isArray(result?.calls) ? result.calls : [];
  const status = String(result?.status ?? "?");
  const failed = result?.ok === false;
  const actions = Number(result?.actions_executed ?? 0);
  const head = [
    status,
    ...(result?.exit_code !== null && result?.exit_code !== undefined && result.exit_code !== 0 ? [`exit ${result.exit_code}`] : []),
    `${calls.length} call${calls.length === 1 ? "" : "s"}`,
    ...(actions ? [`${actions} action${actions === 1 ? "" : "s"}`] : []),
    ...(calls.some((entry) => entry?.dry_run === true) ? [`${calls.filter((entry) => entry?.dry_run === true).length} planned`] : []),
    ...(typeof result?.wall_s === "number" ? [`${number(result.wall_s)} s`] : []),
  ].join(" · ");
  const lines: ScriptLine[] = [{ kind: "head", text: head, failed }];
  const shown = expanded || calls.length <= SCRIPT_CALLS_MAX ? calls : calls.slice(0, SCRIPT_CALLS_MAX - 1);
  lines.push(...shown.map(scriptCallLine));
  if (shown.length < calls.length) lines.push({ kind: "more", text: `… +${calls.length - shown.length} calls` });
  const detail = result?.error_detail;
  if (detail && typeof detail === "object") {
    const what = [detail.type, detail.code].filter((part) => typeof part === "string" && part).join(" ");
    const message = typeof detail.message === "string" ? detail.message : "";
    lines.push({ kind: "error", text: truncate(`${what}${what && message ? ": " : ""}${message}`, expanded ? 2000 : 200),
      failed: true });
  } else if (result?.error && typeof result.error === "object") {
    lines.push({ kind: "error", text: truncate(reasonOf(result.error), 200), failed: true });
  }
  if (result?.handed_back && typeof result.handed_back === "object") {
    lines.push({ kind: "head", text: truncate(`handed back · ${String(result.handed_back.reason ?? "")}`, 160) });
  }
  const printed = String(result?.output ?? "").replace(/\s+$/, "").split("\n").filter((line, index, all) =>
    line.trim() || (index > 0 && index < all.length - 1));
  const tail = expanded ? printed : printed.slice(-OUTPUT_TAIL);
  const above = printed.length - tail.length;
  if (above > 0) lines.push({ kind: "more", text: `… ${above} line${above === 1 ? "" : "s"} above` });
  lines.push(...tail.map((line) => ({ kind: "output" as const, text: expanded ? line : truncate(line, 160) })));
  return lines;
}
