// ORA's look in Pi's terminal: the project teaser's terminal pane (and `ora web`'s page,
// ora-pi/web/src/styles.css) as a Pi theme, its pixel-arm mark and its spinner.
//
// Grays carry everything; the teal accent (#4ecdc4) is for marks only (the logo, the
// spinner, the footer's ▸▸, the prompt's ❯) and green (#7fd3a0) for verdicts only.
// The theme is a Pi Theme instance the extension sets (ui.setTheme), never written to
// the user's Pi settings; it is built for the attached terminal's color mode, so a
// 256-color terminal gets the nearest palette colors. ORA_TUI_THEME=pi keeps Pi's own.
// A terminal program cannot paint the terminal's background: the teaser's #0b0c0f
// pane is the user's terminal background.

import { Theme } from "@earendil-works/pi-coding-agent";
import { getTerminalColorMode } from "@earendil-works/pi-tui";

/** The teaser's palette (look/shots/tui.py; styles.css on the page). */
export const ORA_PALETTE = {
  bg: "#0b0c0f", pane: "#111216", text: "#d6d8dd", dim: "#848992", faint: "#565a62", white: "#f6f7f9",
  args: "#a2a7b0", result: "#9297a0", accent: "#4ecdc4", ok: "#7fd3a0", userBg: "#222329", rule: "#3a3c43",
  error: "#e5786d", warning: "#d9b86c",
} as const;

const P = ORA_PALETTE;

/** Every Pi theme token (Pi 1.3.0 theme-tokens), in ORA's palette. */
export const ORA_THEME_COLORS: Record<string, string> = {
  accent: P.accent, border: P.rule, borderAccent: P.accent, borderMuted: P.faint,
  success: P.ok, error: P.error, warning: P.warning,
  muted: P.args, dim: P.dim, text: P.text, thinkingText: P.dim,
  selectedBg: P.userBg, scrollbarTrack: P.rule, scrollbarThumb: P.faint, searchMatchBg: "#2b2d33", searchMatchText: P.white,
  userMessageBg: P.userBg, userMessageText: P.white,
  customMessageBg: P.pane, customMessageText: P.dim, customMessageLabel: P.text,
  // No boxes around tool calls: a call reads as a line, its result under it (the teaser's transcript).
  toolPendingBg: "", toolSuccessBg: "", toolErrorBg: "#2a1c1e",
  toolTitle: P.white, toolOutput: P.result,
  mdHeading: P.white, mdLink: P.args, mdLinkUrl: P.dim, mdCode: P.args, mdCodeBlock: P.result, mdCodeBlockBorder: P.rule,
  mdQuote: P.dim, mdQuoteBorder: P.rule, mdHr: P.rule, mdListBullet: P.dim,
  toolDiffAdded: P.ok, toolDiffRemoved: P.error, toolDiffContext: P.dim,
  syntaxComment: P.faint, syntaxKeyword: P.args, syntaxFunction: P.white, syntaxVariable: P.text, syntaxString: "#b9c4bd",
  syntaxNumber: P.args, syntaxType: P.text, syntaxOperator: P.dim, syntaxPunctuation: P.dim,
  // The prompt sits between two plain rules whatever the reasoning effort (the footer says the effort).
  thinkingOff: P.rule, thinkingMinimal: P.rule, thinkingLow: P.rule, thinkingMedium: P.rule, thinkingHigh: P.rule,
  thinkingXhigh: P.rule, thinkingMax: P.rule,
  bashMode: P.accent,
};

/** ORA's theme for the terminal's color mode now (truecolor, else the 256-color palette). */
export function oraTheme(mode = getTerminalColorMode()): Theme {
  return new Theme(ORA_THEME_COLORS as any, {} as any, mode, { name: "ora", appearance: "dark" });
}

/** Whether ORA draws in its own theme: yes unless ORA_TUI_THEME=pi (the user's Pi theme). */
export function wantsOraTheme(env: NodeJS.ProcessEnv = process.env): boolean {
  return String(env.ORA_TUI_THEME ?? "").trim().toLowerCase() !== "pi";
}

/** ORA's mark: a pixel robot arm (base, column, shoulder joint, arm, two-finger gripper). */
export const LOGO = [
  "..####.......",
  ".##..##......",
  ".##..#######.",
  "..####....##.",
  "...##.....##.",
  "...##....####",
  "...##....#..#",
  "..####...#..#",
  ".######......",
];

/** The mark in half blocks: two pixel rows per terminal line, each pixel one column (square on screen). */
export function logoLines(): string[] {
  const lines: string[] = [];
  for (let row = 0; row < LOGO.length; row += 2) {
    const top = LOGO[row];
    const bottom = LOGO[row + 1] ?? "".padEnd(top.length, ".");
    lines.push([...top].map((pixel, column) => {
      const up = pixel === "#";
      const down = bottom[column] === "#";
      return up && down ? "█" : up ? "▀" : down ? "▄" : " ";
    }).join("").padEnd(top.length));
  }
  return lines;
}

/** The status line's spinner (the teaser's), one frame per tick. */
export const SPINNER = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"];

export function spinnerFrame(now = Date.now(), intervalMs = 120): string {
  return SPINNER[Math.floor(now / intervalMs) % SPINNER.length];
}
