// Pi's TUI under Node's permission model.
//
// Pi's interactive mode looks for a git checkout (footer branch) and for
// project skills (.agents/skills, project trust) in every ancestor of its
// working directory with fs.existsSync. Under `node --permission`, existsSync
// on a path outside the allowed roots throws ERR_ACCESS_DENIED instead of
// returning false, so the TUI died unless every ancestor's `.git` and
// `.agents/skills` were whitelisted. Here a path the agent process may not read
// simply does not exist for it: the sandbox stays exactly as narrow, and a
// probe outside it fails closed.

import fs from "node:fs";
import { syncBuiltinESMExports } from "node:module";

let installed = false;

export function sandboxedExistsSync(): void {
  if (installed || !(process as any).permission) return;
  installed = true;
  const original = fs.existsSync;
  fs.existsSync = ((path: fs.PathLike) => {
    try {
      return original(path);
    } catch (error) {
      if ((error as NodeJS.ErrnoException)?.code === "ERR_ACCESS_DENIED") return false;
      throw error;
    }
  }) as typeof fs.existsSync;
  // ESM named imports (`import { existsSync } from "fs"`) follow the builtin's exports.
  syncBuiltinESMExports();
}
