// Shared plumbing for the differential tests: run a Python helper (the
// provider scripts in tests/support/agent_core/provider/, or a module) and read its JSON
// output with parsePy.
//
// The Python is ORA_TEST_PYTHON, else the active virtual environment's (`uv run
// --locked node --test ...` activates the project environment), else
// `uv run --locked python`. Scratch files go under ORA_TEST_SCRATCH, else the
// system temporary directory.

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { canonicalBytesPy, dumpsPy, parsePy } from "../../src/pyjson.ts";

export const REPO = resolve(import.meta.dirname, "../../..");
export const HELPERS = join(REPO, "tests/support/agent_core/provider");

function pythonCommand(): string[] {
  if (process.env.ORA_TEST_PYTHON) return [process.env.ORA_TEST_PYTHON];
  const venv = process.env.VIRTUAL_ENV;
  if (venv && existsSync(join(venv, "bin", "python"))) return [join(venv, "bin", "python")];
  return ["uv", "run", "--locked", "python"];
}

/** The command (program and leading arguments) the tests run Python with. */
export const PYTHON: readonly string[] = pythonCommand();

/** Spawn Python with `args` from the repository root. */
export function spawnPython(args: string[], options: Parameters<typeof spawn>[2] = {}) {
  return spawn(PYTHON[0], [...PYTHON.slice(1), ...args], {
    cwd: REPO,
    ...options,
    env: { ...process.env, OPENBLAS_NUM_THREADS: "1", OMP_NUM_THREADS: "1", PYTHONDONTWRITEBYTECODE: "1",
      ...(options.env ?? {}) },
  });
}

export function scratchDir(label: string): string {
  const root = process.env.ORA_TEST_SCRATCH ?? tmpdir();
  mkdirSync(root, { recursive: true });
  return mkdtempSync(join(root, `ora-agent-core-${label}-`));
}

/** Where sealed runs are extracted: ORA_TEST_SEALED_CACHE (kept between runs) or a fresh scratch directory. */
export function sealedCache(): string {
  return process.env.ORA_TEST_SEALED_CACHE ?? scratchDir("sealed");
}

/** Run a provider helper script without blocking the event loop (fake servers live in this process). */
export function runPython(script: string, args: string[], timeoutMs = 600_000): Promise<string> {
  return runPythonArgs([join(HELPERS, script), ...args], timeoutMs);
}

/** Run Python with `args` (a script path or `-m module`) and resolve its stdout. */
export function runPythonArgs(args: string[], timeoutMs = 600_000): Promise<string> {
  const script = args[0] === "-m" ? args[1] : args[0];
  return new Promise((resolvePromise, reject) => {
    const child = spawnPython(args);
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    const timer = setTimeout(() => child.kill("SIGKILL"), timeoutMs);
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolvePromise(stdout);
      else reject(new Error(`${script} exited ${code}\n${stderr.slice(-4000)}`));
    });
  });
}

export function readPy(path: string): any {
  return parsePy(readFileSync(path, "utf8"));
}

export function writePy(path: string, value: unknown): void {
  writeFileSync(path, dumpsPy(value, { ensureAscii: false }));
}

export function canonicalText(value: unknown): string {
  return canonicalBytesPy(value).toString("utf8");
}

const TIMING = ["started_unix_s", "finished_unix_s", "duration_s", "ttft_s", "no_content_elapsed_s"];

export function normalizeTiming<T extends { [key: string]: any }>(value: T): T {
  const out: any = { ...value };
  for (const key of TIMING) if (key in out && out[key] !== null) out[key] = "<t>";
  return out;
}

/** The Python-side `describe_error` shape for a TypeScript error. */
export function describeError(error: any): { [key: string]: any } {
  const cause = error?.cause;
  return {
    type: error?.pyType ?? error?.name ?? "Error",
    message: error?.message ?? String(error),
    retryable: error?.retryable ?? null,
    http_status: error?.httpStatus ?? null,
    response_started: error?.responseStarted ?? null,
    credential_echo: error?.credentialEcho ?? null,
    provider_failure: error?.providerFailure ?? null,
    cause_type: cause ? (cause.pyType ?? cause.name ?? "Error") : null,
  };
}
