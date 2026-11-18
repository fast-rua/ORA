// The conversation as ORA's terminal draws it: a tool call as `name(key=value, …)` with long
// values elided (the arguments the model gave, in a fixed order), its result as a few short lines
// under it, token counts as 37.2k; and what the page's terminal footer and instruction header say
// (the context the model holds, the instruction the robot works on).

import { t } from "./i18n.ts";
import type { Conversation, ToolView } from "./messages.ts";

type Json = any;

/** The arguments the terminal names first, in this order; the rest follow as the model gave them. */
const ORDER = ["text", "camera", "view", "points_uv", "point_ids", "step", "target_ref", "candidate_id", "arm", "action", "ticks",
  "interaction", "position_mode", "position_frame", "position_m", "translation_m", "approach_world", "jaw_line_world",
  "orientation_mode", "orientation_xyzw", "max_candidates"];
const STRING_MAX = 70;
const ARGS_MAX = 300;
const MORE_MAX = 8;
const RESULT_MAX = 4;

function clip(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

/** One value as the call line shows it: short floats, quoted strings elided past 70 characters. */
export function value(item: Json): string {
  if (typeof item === "number") {
    if (Number.isInteger(item)) return String(item);
    return Math.abs(item) < 10 ? item.toFixed(4).replace(/0+$/, "").replace(/\.$/, "") : item.toFixed(1);
  }
  if (typeof item === "string") return `"${item.length <= STRING_MAX ? item : `${item.slice(0, STRING_MAX - 2)}…`}"`;
  if (Array.isArray(item)) return `[${item.map(value).join(", ")}]`;
  if (item && typeof item === "object") return `{${Object.entries(item).map(([key, inner]) => `${key}: ${value(inner)}`).join(", ")}}`;
  return item === undefined ? "" : JSON.stringify(item);
}

function kwargs(args: Json): string {
  const keys = [...ORDER.filter((key) => key in args), ...Object.keys(args).filter((key) => !ORDER.includes(key))];
  return keys.map((key) => `${key}=${value(args[key])}`).join(", ");
}

function v3(xs: Json): string {
  return Array.isArray(xs) ? `(${xs.slice(0, 3).map((x: number) => Number(x).toFixed(3)).join(", ")})` : "";
}

/** A call as the model made it: its arguments in parentheses, and a compute's bindings and expressions below. */
export function callText(name: string, args: Json): { args: string; more: string[] } {
  const given = args && typeof args === "object" && !Array.isArray(args) ? args : {};
  if (name === "compute") {
    const more = [
      ...(Array.isArray(given.bindings) ? given.bindings : []).map((binding: Json) =>
        `${binding?.name} = ${value(binding?.value)}${binding?.from ? `   # ${binding.from}` : ""}`),
      ...String(given.expressions ?? "").replace(/\\n/g, "\n").split("\n").filter((line) => line.trim()),
    ].map((line) => clip(line, 160));
    return { args: `(outputs=${value(given.outputs ?? [])})`,
      more: more.length > MORE_MAX ? [...more.slice(0, MORE_MAX - 1), `… +${more.length - MORE_MAX + 1}`] : more };
  }
  if (given.targets && typeof given.targets === "object" && !Array.isArray(given.targets)) {
    // Per-arm targets: move_pose(left: interaction=…; right: …), set_gripper(left="close").
    const parts = Object.entries(given.targets).map(([arm, target]) => (target && typeof target === "object" && !Array.isArray(target)
      ? `${arm}: ${kwargs(target)}` : `${arm}=${value(target)}`));
    const rest = Object.fromEntries(Object.entries(given).filter(([key]) => key !== "targets"));
    return { args: clip(`(${parts.join("; ")}${Object.keys(rest).length ? `, ${kwargs(rest)}` : ""})`, ARGS_MAX), more: [] };
  }
  return { args: clip(`(${kwargs(given)})`, ARGS_MAX), more: [] };
}

function resultOf(name: string, result: Json): string[] {
  const out: string[] = [];
  const list = (key: string): Json[] => (Array.isArray(result?.[key]) ? result[key] : []);
  if (name === "vision_ground" || name === "vision_segment") {
    for (const candidate of list("candidates").slice(0, 4)) {
      const where = candidate?.localization ?? {};
      const height = where.height_above_support_m;
      out.push(`${candidate?.candidate_id ?? "?"}  score ${Number(candidate?.backend_score ?? 0).toFixed(2)}`
        + (where.center_xyz_m ? `  centre ${v3(where.center_xyz_m)} m` : "")
        + (typeof height === "number" ? `  height ${(1000 * height).toFixed(0)} mm` : ""));
    }
  } else if (name === "grasp_plan") {
    list("candidates").slice(0, 3).forEach((candidate: Json, index: number) => {
      out.push(`G${index}  score ${Number(candidate?.backend_score ?? 0).toFixed(2)}`
        + (candidate?.pregrasp_pose?.position_m ? `  pregrasp ${v3(candidate.pregrasp_pose.position_m)} m` : "")
        + `  opening ${(1000 * Number(candidate?.gripper_opening_m ?? 0)).toFixed(0)} mm  `
        + (candidate?.ik_reachability?.reachable ? "reachable" : "unreachable"));
    });
  } else if (name === "mark_point") {
    for (const mark of list("marks").slice(0, 6)) {
      out.push(`${mark?.point_id ?? "?"}  ${mark?.xyz_m ? `${v3(mark.xyz_m)} m` : mark?.error_code ?? mark?.status ?? ""}`);
    }
  } else if (name === "compute") {
    for (const output of list("outputs")) out.push(clip(`${output?.name} = ${value(output?.value)}`, 160));
  } else if (name === "grasp_execute" && Array.isArray(result?.committed_sequence)) {
    out.push(`committed: ${result.committed_sequence.join(" → ")}`);
  } else if (name === "observe" && Array.isArray(result?.views)) {
    out.push(`views ${result.views.join(", ")}`);
  } else if (name === "geometry_query" && result?.region_result?.depth_m) {
    const depth = result.region_result.depth_m;
    out.push(`depth ${Number(depth.min).toFixed(3)}–${Number(depth.max).toFixed(3)} m, median ${Number(depth.median).toFixed(3)} m`
      + (result.camera ? `  ${result.camera}` : ""));
  }
  if (!out.length) {
    const error = result?.error_message ?? result?.error?.message ?? result?.error;
    if (typeof error === "string" && error) out.push(clip(error.split("\n")[0], 160));
    else if (typeof result?.status === "string") {
      // A robot action: submitted (and how far each arm goes, when the pose was compiled).
      const targets = result?.compiled_pose?.targets;
      const moves = targets && typeof targets === "object" ? Object.entries(targets).flatMap(([arm, target]: [string, Json]) => (
        typeof target?.translation_distance_m === "number" ? [`${arm} ${(1000 * target.translation_distance_m).toFixed(0)} mm`] : [])) : [];
      out.push([result.status.replace(/_/g, " "), moves.join(", ")].filter(Boolean).join(" · "));
    }
  }
  return out;
}

/** What a call's result says, as a few lines (more as "… +N more"), with the images it gave the model. */
export function resultLines(tool: ToolView): string[] {
  let lines = tool.result && typeof tool.result === "object" ? resultOf(tool.name, tool.result) : [];
  if (!lines.length && tool.overlay) lines = [tool.overlay.caption];
  if (!lines.length && tool.status === "error" && tool.text) lines = [clip(tool.text.trim().split("\n")[0], 160)];
  if (lines.length > RESULT_MAX) lines = [...lines.slice(0, RESULT_MAX), t("chat.moreLines", { n: lines.length - RESULT_MAX })];
  if (tool.images.length) lines.push(tool.images.length === 1 ? t("chat.oneImageToModel") : t("chat.imagesToModel", { n: tool.images.length }));
  return lines;
}

/** A token count as the footer says it: 950, 37.2k, 1M. */
export function tokens(count: number): string {
  if (count >= 1_000_000) return `${Number((count / 1_000_000).toFixed(1))}M`;
  return count >= 1000 ? `${(count / 1000).toFixed(1)}k` : String(count);
}

/** The context the model held at its last call (its usage), or null before any. */
export function contextTokens(messages: Json[]): number | null {
  for (let index = messages.length - 1; index >= 0; index--) {
    const usage = messages[index]?.role === "assistant" ? messages[index].usage : null;
    if (!usage) continue;
    const total = Number(usage.totalTokens);
    if (Number.isFinite(total) && total > 0) return total;
    const sum = ["input", "output", "cacheRead", "cacheWrite"].reduce((acc, key) => acc + (Number(usage[key]) || 0), 0);
    if (sum > 0) return sum;
  }
  return null;
}

/** The instruction the robot works on: free play's latest, else an episode's task (its observations' task_instruction). */
export function instructionOf(conversation: Conversation): string | null {
  for (let index = conversation.items.length - 1; index >= 0; index--) {
    const item = conversation.items[index];
    if (item.kind === "user" && item.text.trim()) return item.text.trim();
  }
  for (let index = conversation.items.length - 1; index >= 0; index--) {
    const item = conversation.items[index];
    if (item.kind !== "observation") continue;
    const found = /"task_instruction"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(item.observation.text);
    if (found) {
      try {
        return JSON.parse(`"${found[1]}"`);
      } catch {
        return found[1];
      }
    }
  }
  return null;
}


// -- Code mode: a `bash` command and what its programs did (the terminal's tui/render.ts) -----

/** Lines of a command the call shows before it is opened. */
export const COMMAND_MAX = 12;
/** Script calls listed before the result is opened. */
export const SCRIPT_CALLS_MAX = 8;
/** Output lines kept before the result is opened: the end, what the program printed last. */
export const OUTPUT_TAIL = 4;

/** A code-mode command's lines (ORA's bash), all of them when `open`; null for arguments of another shape. */
export function commandLines(args: Json, open = false): string[] | null {
  const command = args && typeof args === "object" ? args.command : undefined;
  if (typeof command !== "string") return null;
  const lines = command.replace(/\s+$/, "").split("\n");
  if (open || lines.length <= COMMAND_MAX) return lines;
  return [...lines.slice(0, COMMAND_MAX - 1), t("script.moreLines", { n: lines.length - COMMAND_MAX + 1 })];
}

/** Whether a result is a code-mode command's (ora/agent/pi/code_mode.py `_code_mode_bash`). */
export function isScriptResult(result: Json): boolean {
  return Boolean(result && typeof result === "object" && typeof result.status === "string"
    && Array.isArray(result.calls) && "output" in result);
}

export interface ScriptLine {
  kind: "head" | "call" | "error" | "more";
  text: string;
  failed?: boolean;
  dryRun?: boolean;
}

/** One robot call of a script: `#002 move_pose ✓ action 1 · summary`; a dry run is a plan. */
export function scriptCallLine(entry: Json): ScriptLine {
  const seq = `#${String(entry?.call ?? "?").padStart(3, "0")}`;
  const dryRun = entry?.dry_run === true;
  const ok = entry?.ok !== false;
  const mark = entry?.running === true ? (dryRun ? `◇ ${t("script.plan")} …` : "…")
    : dryRun ? `◇ ${t("script.plan")}${ok ? "" : " ✗"}` : ok ? "✓" : "✗";
  const parts = [
    ...(typeof entry?.action === "number" ? [`action ${entry.action}`] : []),
    ...(typeof entry?.error === "string" ? [entry.error] : []),
    ...(typeof entry?.images === "number" && entry.images > 0
      ? [entry.images === 1 ? t("chat.oneImageToModel") : t("chat.imagesToModel", { n: entry.images })] : []),
    ...(typeof entry?.summary === "string" && entry.summary ? [entry.summary.split("\n")[0]] : []),
  ];
  return { kind: "call", text: clip(`${seq} ${String(entry?.tool ?? "?")} ${mark}${parts.length ? `  ${parts.join(" · ")}` : ""}`, 200),
    failed: !ok, dryRun };
}

/** A code-mode command's result: how it ended, each robot call of its program, the error that ended it. */
export function scriptLines(result: Json, open = false): ScriptLine[] {
  const calls: Json[] = Array.isArray(result?.calls) ? result.calls : [];
  const actions = Number(result?.actions_executed ?? 0);
  const planned = calls.filter((entry) => entry?.dry_run === true).length;
  const head = [
    String(result?.status ?? "?"),
    ...(result?.exit_code !== null && result?.exit_code !== undefined && result.exit_code !== 0 ? [`exit ${result.exit_code}`] : []),
    calls.length === 1 ? t("script.oneCall") : t("script.calls", { n: calls.length }),
    ...(actions ? [actions === 1 ? t("script.oneAction") : t("script.actions", { n: actions })] : []),
    ...(planned ? [t("script.planned", { n: planned })] : []),
    ...(typeof result?.wall_s === "number" ? [`${Number(result.wall_s.toFixed(2))} s`] : []),
  ].join(" · ");
  const lines: ScriptLine[] = [{ kind: "head", text: head, failed: result?.ok === false }];
  const shown = open || calls.length <= SCRIPT_CALLS_MAX ? calls : calls.slice(0, SCRIPT_CALLS_MAX - 1);
  lines.push(...shown.map(scriptCallLine));
  if (shown.length < calls.length) lines.push({ kind: "more", text: t("script.moreCalls", { n: calls.length - shown.length }) });
  const detail = result?.error_detail;
  if (detail && typeof detail === "object") {
    const what = [detail.type, detail.code].filter((part) => typeof part === "string" && part).join(" ");
    const message = typeof detail.message === "string" ? detail.message : "";
    lines.push({ kind: "error", text: `${what}${what && message ? ": " : ""}${message}`, failed: true });
  }
  if (result?.handed_back && typeof result.handed_back === "object") {
    lines.push({ kind: "head", text: t("script.handedBack", { reason: String(result.handed_back.reason ?? "") }) });
  }
  return lines;
}

/** What a command printed: its last `OUTPUT_TAIL` lines (all when `open`), and how many lines came before. */
export function outputTail(result: Json, open = false): { lines: string[]; above: number } {
  const printed = String(result?.output ?? "").replace(/\s+$/, "");
  const lines = printed ? printed.split("\n") : [];
  const kept = open ? lines : lines.slice(-OUTPUT_TAIL);
  return { lines: kept, above: lines.length - kept.length };
}
