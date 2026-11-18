// Export a run's Pi session: HTML through Pi's own exporter, or the raw JSONL.
//
//   node ora-pi/src/tui/export.ts SESSION_FILE OUTPUT(.html|.jsonl)
//
// Used by `ora sessions export` and by `/export` in the TUI (written by the
// attach client, outside the agent process's sandbox and the run directory).

import { copyFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** Pi's standalone session exporter (not re-exported from the package index). */
async function piExporter(): Promise<(input: string, options: { outputPath: string }) => Promise<string>> {
  const index = import.meta.resolve("@earendil-works/pi-coding-agent");
  const module = await import(new URL("./core/export-html/index.js", index).href);
  if (typeof module.exportFromFile !== "function") throw new Error("Pi's HTML exporter moved; update ora-pi/src/tui/export.ts");
  return module.exportFromFile;
}

export async function exportSession(sessionFile: string, output: string): Promise<string> {
  if (!existsSync(sessionFile)) throw new Error(`no Pi session at ${sessionFile}`);
  const target = resolve(output);
  if (target.endsWith(".jsonl")) {
    copyFileSync(sessionFile, target);
    return target;
  }
  if (!target.endsWith(".html")) throw new Error("export to a .html or .jsonl path");
  const exportFromFile = await piExporter();
  return resolve(await exportFromFile(sessionFile, { outputPath: target }));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [sessionFile, output] = process.argv.slice(2);
  if (!sessionFile || !output) {
    process.stderr.write("usage: export.ts SESSION_FILE OUTPUT.html|OUTPUT.jsonl\n");
    process.exit(2);
  }
  exportSession(sessionFile, output).then((path) => {
    process.stdout.write(`${path}\n`);
  }, (error) => {
    process.stderr.write(`ora-pi export: ${error?.message ?? error}\n`);
    process.exit(1);
  });
}
