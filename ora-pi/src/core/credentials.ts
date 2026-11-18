// /login and /logout in any run: provider credentials in a user-level ORA store.
//
// A benchmark run's own model and key are bound by its RunPlan; the operator may
// still log in to Pi's providers (for later free play). Credentials are read
// once into memory, so a model call never touches the file, and every change is
// written back best-effort (a failed write is reported, never fatal) to the
// user-level store outside the run, in Pi's auth.json format. Free play uses
// Pi's own file store on the same path (play/runtime.ts).

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { InMemoryCredentialStore } from "@earendil-works/pi-ai";

type Json = any;

export class UserCredentialStore extends InMemoryCredentialStore {
  readonly path: string;

  constructor(path: string) {
    super();
    this.path = path;
    try {
      const stored = JSON.parse(readFileSync(path, "utf8"));
      for (const [providerId, credential] of Object.entries(stored ?? {})) {
        (this as Json).credentials.set(providerId, credential);
      }
    } catch {
      // no store yet, or not readable here: start empty
    }
  }

  override async modify(providerId: string, fn: Json, options?: Json): Promise<Json> {
    const result = await super.modify(providerId, fn, options);
    this.persist();
    return result;
  }

  override async delete(providerId: string, options?: Json): Promise<void> {
    await super.delete(providerId, options);
    this.persist();
  }

  private persist(): void {
    try {
      mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
      const entries = Object.fromEntries((this as Json).credentials as Map<string, Json>);
      writeFileSync(this.path, JSON.stringify(entries, null, 2), { encoding: "utf8", mode: 0o600 });
    } catch (error) {
      process.stderr.write(`ora-pi: the credential store ${this.path} was not written: ${(error as Error).message}\n`);
    }
  }
}
