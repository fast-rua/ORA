// Where free play is, from the Supervisor's own announcements (ora.play-state.v1,
// ora/orchestrator/episode/play.py), never from whether the model is busy: an
// instruction runs until the Supervisor says it ended, and why. What the agent does
// meanwhile (thinking, writing, a tool, the robot moving) is the web server's
// activity (server/view.ts), counted on here from the page's own clock.

import { duration, hasKey, seconds, t } from "./i18n.ts";

type Json = any;

export type PlayPhase = "awaiting_instruction" | "running" | "ended";

export function playPhase(play: Json): PlayPhase | null {
  if (!play || typeof play !== "object") return null;
  if (play.state === "ended") return "ended";
  return play.running ? "running" : "awaiting_instruction";
}

/** How an instruction ended (episode_play.py `ended`), as the terminal says it. */
export function endingText(ended: string): string {
  const key = `ending.${ended}`;
  return hasKey(key) ? t(key) : t("ending.other", { reason: ended.replace(/_/g, " ") });
}

export type EndingKind = "reported" | "stopped" | "budget" | "ended";

export function endingKind(ended: string): EndingKind {
  if (ended === "reported") return "reported";
  if (/budget/.test(ended)) return "budget";
  if (ended.startsWith("operator_") || ended === "superseded") return "stopped";
  return "ended";
}

export interface PlayStatus {
  phase: PlayPhase | null;
  label: string;
  detail: string | null;
}

/** The phase line of the header, and the last instruction's ending while none runs. */
export function playStatus(play: Json, last: Json | null): PlayStatus {
  const phase = playPhase(play);
  const where = play?.scenes ? t("status.where", { scene: play.scene, scenes: play.scenes }) : null;
  if (phase === null) return { phase, label: t("status.freePlay"), detail: null };
  if (phase === "ended") return { phase, label: t("status.sessionEnded"), detail: play.termination ? String(play.termination) : where };
  if (phase === "running") return { phase, label: t("status.running", { n: play.instructions }), detail: where };
  const ended = last?.ended ? t("status.lastEnded", { n: Number(last.index) + 1, ending: endingText(String(last.ended)) }) : null;
  return { phase, label: t("status.waiting"), detail: ended ?? where };
}

/** What the agent does now, from the web server (server/view.ts), with `since` on the page's clock. */
export interface Activity {
  kind: "thinking" | "writing" | "tool" | "moving" | "observing" | "compacting" | "idle";
  since: number;
  tool?: string;
  step?: number | null;
}

/** The server's activity record (with its elapsedMs) on this page's clock. */
export function activityFrom(record: Json, now = Date.now()): Activity | null {
  if (!record || typeof record.kind !== "string") return null;
  return { kind: record.kind, since: now - Math.max(0, Number(record.elapsedMs) || 0), tool: record.tool,
    step: record.step ?? null };
}

/** "Thinking… 4 s", "Robot moving… 2 s"; null when idle. */
export function activityText(activity: Activity | null, now = Date.now()): string | null {
  if (activity === null || activity.kind === "idle") return null;
  const s = seconds(now - activity.since);
  switch (activity.kind) {
    case "thinking": return t("activity.thinking", { s });
    case "writing": return t("activity.writing", { s });
    case "tool": return t("activity.tool", { tool: activity.tool ?? "tool", s });
    case "moving": return t("activity.moving", { s });
    case "observing": return t("activity.observing");
    case "compacting": return t("activity.compacting", { s });
  }
}

export { duration };
