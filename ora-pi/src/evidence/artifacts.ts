// Content-addressed agent artifacts: `ora/agent/evidence/canonical.py` ArtifactStore.
//
// `native/agent-artifacts/<sha256><suffix>`, published by writing a private
// temporary file and hard-linking it to its content name, so a reader never
// sees a partial artifact and two writers of the same bytes converge. The ref
// shape `{kind, path, sha256, bytes, media_type}` is the Python one.

import { createHash, randomBytes } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

import { canonicalBytesPy } from "../pyjson.ts";

/** Deferred fsync: `submit` after a publish, `barrier` before relying on it. */
export interface Durability {
  submit(filePath: string): void;
  barrier(): void | Promise<void>;
}

/** Skips fsync entirely (tests, scratch runs). */
export const NO_DURABILITY: Durability = {
  submit() {},
  barrier() {},
};

export interface StorageRef {
  kind: string;
  path: string;
  sha256: string;
  bytes: number;
  media_type: string;
  [key: string]: unknown;
}

export class TrajectoryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TrajectoryError";
  }
}

function sha256Bytes(data: Buffer): string {
  return createHash("sha256").update(data).digest("hex");
}

/** `tempfile.mkstemp(prefix=..., suffix=".tmp", dir=...)` with mode 0600. */
export function makeTemp(directory: string, prefix: string): { fd: number; name: string } {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const name = path.join(directory, `${prefix}${randomBytes(6).toString("base64url")}.tmp`);
    try {
      return { fd: fs.openSync(name, "wx", 0o600), name };
    } catch (error: any) {
      if (error?.code !== "EEXIST") throw error;
    }
  }
  throw new TrajectoryError("no usable temporary file name");
}

export function unlinkMissingOk(name: string): void {
  try {
    fs.unlinkSync(name);
  } catch (error: any) {
    if (error?.code !== "ENOENT") throw error;
  }
}

function resolveRoot(runRoot: string): string {
  const absolute = path.resolve(runRoot);
  try {
    return fs.realpathSync(absolute);
  } catch {
    return absolute;
  }
}

export class ArtifactStore {
  runRoot: string;
  root: string;
  /** `null` fsyncs before linking (the Python default). */
  durability: Durability | null;

  constructor(runRoot: string, options: { durability?: Durability | null } = {}) {
    this.durability = options.durability ?? null;
    fs.mkdirSync(path.join(path.resolve(runRoot), "native/agent-artifacts"), { recursive: true });
    this.runRoot = resolveRoot(runRoot);
    this.root = path.join(this.runRoot, "native/agent-artifacts");
  }

  putBytes(
    data: Buffer,
    options: { kind: string; suffix?: string; mediaType?: string },
  ): StorageRef {
    const suffix = options.suffix ?? ".bin";
    const mediaType = options.mediaType ?? "application/octet-stream";
    const digest = sha256Bytes(data);
    const safeSuffix = suffix.startsWith(".") ? suffix : `.${suffix}`;
    const target = path.join(this.root, `${digest}${safeSuffix}`);
    if (fs.existsSync(target)) {
      if (!fs.readFileSync(target).equals(data)) {
        throw new TrajectoryError(`artifact hash collision: ${digest}`);
      }
    } else {
      const { fd, name } = makeTemp(this.root, `.${path.basename(target)}.`);
      let linked = false;
      try {
        try {
          let offset = 0;
          while (offset < data.length) offset += fs.writeSync(fd, data, offset);
          if (this.durability === null) fs.fsyncSync(fd);
        } finally {
          fs.closeSync(fd);
        }
        try {
          fs.linkSync(name, target);
          linked = true;
        } catch (error: any) {
          if (error?.code !== "EEXIST") throw error;
          if (!fs.readFileSync(target).equals(data)) {
            throw new TrajectoryError(`artifact hash collision: ${digest}`);
          }
        }
      } finally {
        unlinkMissingOk(name);
      }
      if (linked && this.durability !== null) this.durability.submit(target);
    }
    return {
      kind: options.kind,
      path: path.relative(this.runRoot, target).split(path.sep).join("/"),
      sha256: digest,
      bytes: data.length,
      media_type: mediaType,
    };
  }

  putJson(value: unknown, options: { kind: string }): StorageRef {
    return this.putBytes(canonicalBytesPy(value), {
      kind: options.kind, suffix: ".json", mediaType: "application/json",
    });
  }

  reference(filePath: string, options: { kind: string; mediaType: string }): StorageRef {
    let resolved: string;
    try {
      resolved = fs.realpathSync(path.resolve(filePath));
    } catch {
      throw new TrajectoryError(`artifact reference escapes run: ${filePath}`);
    }
    const relative = path.relative(this.runRoot, resolved);
    if (!fs.statSync(resolved).isFile() || relative.startsWith("..") || path.isAbsolute(relative)) {
      throw new TrajectoryError(`artifact reference escapes run: ${filePath}`);
    }
    const data = fs.readFileSync(resolved);
    return {
      kind: options.kind,
      path: relative.split(path.sep).join("/"),
      sha256: sha256Bytes(data),
      bytes: data.length,
      media_type: options.mediaType,
    };
  }
}
