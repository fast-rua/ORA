// A pi-tui choice list for ORA's launch questions (provider profile, run to attach).
//
//   node ora-pi/src/tui/pick.ts CHOICES.json OUT
//
// CHOICES.json: {"title": str, "hint": str?, "items": [{"value", "label", "description"?}]}.
// Writes the chosen value to OUT and exits 0; Esc or Ctrl+C exits 1.

import { readFileSync, writeFileSync } from "node:fs";

import { ProcessTerminal, SelectList, Text, TuiMainScreen } from "@earendil-works/pi-tui";

const [choicesPath, outPath] = process.argv.slice(2);
if (!choicesPath || !outPath) {
  process.stderr.write("usage: pick.ts CHOICES.json OUT\n");
  process.exit(2);
}
const choices = JSON.parse(readFileSync(choicesPath, "utf8"));
const items = (choices.items ?? []).map((item: any) => ({
  value: String(item.value), label: String(item.label ?? item.value),
  ...(item.description ? { description: String(item.description) } : {}),
}));
if (!items.length) {
  process.stderr.write("nothing to choose from\n");
  process.exit(1);
}

const bold = (text: string) => `\x1b[1m${text}\x1b[22m`;
const dim = (text: string) => `\x1b[2m${text}\x1b[22m`;
const accent = (text: string) => `\x1b[36m${text}\x1b[39m`;

const ui = new TuiMainScreen(new ProcessTerminal());
// Profile ids are long: give the names their width (up to 48 columns) before the descriptions.
const nameWidth = Math.min(48, Math.max(...items.map((item: any) => item.label.length)) + 2);
const list = new SelectList(items, Math.min(items.length, 12), {
  selectedPrefix: accent, selectedText: (text) => bold(accent(text)), description: dim,
  scrollInfo: dim, noMatch: dim,
}, { minPrimaryColumnWidth: nameWidth, maxPrimaryColumnWidth: nameWidth });
ui.addChild(new Text(bold(String(choices.title ?? "Choose")), 1, 0));
ui.addChild(list);
ui.addChild(new Text(dim(String(choices.hint ?? "↑/↓ choose · Enter select · Esc cancel")), 1, 0));

let done = false;
function finish(value: string | null): void {
  if (done) return;
  done = true;
  ui.stop();
  if (value !== null) writeFileSync(outPath, value);
  process.exit(value === null ? 1 : 0);
}
list.onSelect = (item) => finish(item.value);
list.onCancel = () => finish(null);
ui.addInputListener((data: string) => (data === "\x03" ? (finish(null), { consume: true }) : undefined));
ui.setFocus(list);
ui.start();
