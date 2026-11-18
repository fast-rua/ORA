// What the composer sends for what is typed, as the terminal decides it
// (ora-pi/src/tui/extension.ts intercept): a message goes to the run as Pi's
// prompt, steer or follow-up, or in free play queued for the next instruction
// (the Supervisor holds it until the running one ends); free play's /reset and /quit are its controls;
// /stop asks to end the run; a real cell's commands (/approve, /decline, /yolo,
// /home, ...: lib/cell.ts) act on the cell, and on a run that drives none are
// answered as the terminal answers them; a shell line or another command is not sent.

import { cellIntent, type CellIntent } from "./cell.ts";
import { t } from "./i18n.ts";

type Json = any;

export type Outgoing =
  | { kind: "message"; text: string; as: "prompt" | "steer" | "follow_up" | "queue" }
  | { kind: "control"; control: "stop_instruction" | "reset" | "quit" }
  | { kind: "cell"; intent: CellIntent }
  | { kind: "refused"; reason: string };

/** Commands the page passes to the run as typed (Pi runs ORA's registered command). */
const PASSED = new Set(["stop"]);

export function outgoing(raw: string, options: { play: boolean; streaming: boolean; followUp?: boolean; cell?: boolean }):
Outgoing | null {
  const text = raw.trim();
  if (!text) return null;
  if (text.startsWith("!")) return { kind: "refused", reason: t("composer.shell") };
  if (text.startsWith("/")) {
    const name = text.slice(1).split(/\s+/, 1)[0];
    const intent = cellIntent(text);
    if (intent !== null) {
      return options.cell ? { kind: "cell", intent } : { kind: "refused", reason: t("cell.realOnly", { name }) };
    }
    if (options.play && name === "reset") return { kind: "control", control: "reset" };
    if (options.play && name === "quit") return { kind: "control", control: "quit" };
    if (!PASSED.has(name)) {
      return { kind: "refused", reason: options.play ? t("composer.notHerePlay", { name }) : t("composer.notHere", { name }) };
    }
    return { kind: "message", text, as: "prompt" };
  }
  if (!options.streaming) return { kind: "message", text, as: "prompt" };
  return { kind: "message", text, as: options.followUp ? "follow_up" : "steer" };
}

/** Whether the run takes `command` (ora_get_run's refusals: the agent process would refuse it). */
export function allows(run: Json, command: string): boolean {
  return Boolean(run) && !(command in (run.refused ?? {}));
}
