// The composer's slash-command suggestions, as Pi's own terminal lists them: typing "/" at the start
// of the box lists every command this page takes (lib/composer.ts `outgoing`), each with one line of
// what it does, narrowed as the name is typed; after the name, a command's fixed words (/mode ask|safe|
// auto, /yolo on|off). Only what the page would act on is listed: free play's /reset and /quit, /stop
// (any run), and a real cell's commands (lib/cell.ts) only on a run that drives one. A command the page
// answers "not here" is never offered.

import { PERMISSION_MODES } from "./cell.ts";
import { t, type Key } from "./i18n.ts";

export interface SlashItem {
  /** What completing it puts in the box: "/mode " (an argument follows), "/home", "/mode safe". */
  insert: string;
  /** The command or the word as listed: "/mode", "safe". */
  label: string;
  /** The command's arguments, as its hint shows them ("[ask|safe|auto]"), or null. */
  args: string | null;
  description: string;
}

export interface SlashOptions {
  /** A free-play run. */
  play: boolean;
  /** The run drives a real cell (its state has been read and names one). */
  cell: boolean;
}

interface Spec {
  name: string;
  /** The arguments' i18n key (`slash.args.*`), when it takes any. */
  args?: Key;
  /** Its fixed words, each with its line. */
  words?: readonly string[];
}

/** Free play's own commands, then /stop (lib/composer.ts). */
const PLAY: readonly Spec[] = [{ name: "reset" }, { name: "quit" }];
const ANY: readonly Spec[] = [{ name: "stop" }];
/**
 * A real cell's commands, in the order an operator reaches for them: the stop of the motion in progress,
 * the answers, the mode and the switch, then the verbs (the ones that let the arms fall last).
 */
const CELL: readonly Spec[] = [
  { name: "halt" },
  { name: "approve", args: "slash.args.approve" },
  { name: "decline", args: "slash.args.decline" },
  { name: "mode", args: "slash.args.mode", words: PERMISSION_MODES },
  { name: "yolo", args: "slash.args.yolo", words: ["on", "off"] },
  { name: "home" }, { name: "hold" }, { name: "diagnose" }, { name: "float" }, { name: "release" }, { name: "estop" },
];

/** The commands this page takes now, in the order they are listed. */
export function slashCommands(options: SlashOptions): readonly Spec[] {
  return [...(options.play ? PLAY : []), ...ANY, ...(options.play && options.cell ? CELL : [])];
}

function item(spec: Spec): SlashItem {
  return { insert: `/${spec.name}${spec.args ? " " : ""}`, label: `/${spec.name}`, args: spec.args ? t(spec.args) : null,
    description: t(`slash.${spec.name}` as Key) };
}

/**
 * What the list shows for the box's `text`: while the name is typed ("/", "/ho"), the commands whose name
 * starts with it, then those that contain it; after a command with fixed words ("/mode s"), its words.
 * Empty for any other text (a message, a second line, a command's free argument).
 */
export function slashSuggestions(text: string, options: SlashOptions): SlashItem[] {
  if (!text.startsWith("/") || text.includes("\n")) return [];
  const specs = slashCommands(options);
  const name = /^\/(\S*)$/.exec(text);
  if (name) {
    const typed = name[1].toLowerCase();
    const starts = specs.filter((spec) => spec.name.startsWith(typed));
    const contains = specs.filter((spec) => !spec.name.startsWith(typed) && spec.name.includes(typed));
    return [...starts, ...contains].map(item);
  }
  const word = /^\/(\S+) +(\S*)$/.exec(text);
  const spec = word ? specs.find((candidate) => candidate.name === word[1].toLowerCase()) : undefined;
  if (!word || !spec?.words) return [];
  const typed = word[2].toLowerCase();
  return spec.words.filter((value) => value.startsWith(typed)).map((value) => ({
    insert: `/${spec.name} ${value}`, label: value, args: null, description: t(`slash.${spec.name}.${value}` as Key),
  }));
}

/** Whether `text` already is the item (Enter then runs it as typed, instead of completing it). */
export function isComplete(text: string, suggestion: SlashItem): boolean {
  return text.trim().toLowerCase() === suggestion.insert.trim().toLowerCase();
}

/** The real-cell names the list offers: exactly the page's cell commands (lib/cell.ts; test/composer.test.tsx). */
export const CELL_SLASH_NAMES: readonly string[] = CELL.map((spec) => spec.name);
