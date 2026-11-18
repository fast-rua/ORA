// `ora/agent/context/lossless/inline_media.py`: PolicySession cache for recoverable
// payload admission during projection.
//
// The first admission of a payload runs the complete catalog path (decode,
// content hash, store collision check, index registration or
// resolve-and-compare). Later projections reuse that verified result while the
// backing file keeps the same (inode, size, mtime_ns); any change drops the
// entry and repeats the complete admission, which then fails closed on drift.

import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

import type { ContextArtifactCatalog } from "./memory/artifacts.ts";
import { PolicySessionError, clonePy, get, isMapping } from "./pyutil.ts";
import { GenerationalMemo } from "./request_size.ts";

const STRICT_BASE64 = /^[A-Za-z0-9+/]*={0,2}$/;

/** ``base64.b64decode(encoded, validate=True)`` (CPython 3.11 strict mode). */
export function strictBase64Decode(encoded: string): Buffer {
  if (!STRICT_BASE64.test(encoded)) throw new Error("Only base64 data is allowed");
  if (encoded.startsWith("=")) throw new Error("Leading padding not allowed");
  if (encoded.length % 4 !== 0) {
    throw new Error(encoded.includes("=") ? "Excess data after padding" : "Incorrect padding");
  }
  return Buffer.from(encoded, "base64");
}

/** ``(media_type, bytes)`` of an inline image data URL, or null for anything else. */
export function decodeImageDataUrl(value: unknown): [string, Buffer] | null {
  if (typeof value !== "string" || !value.startsWith("data:image/")) return null;
  const comma = value.indexOf(",");
  const header = comma >= 0 ? value.slice(0, comma) : value;
  if (comma < 0 || !header.includes(";base64")) {
    throw new PolicySessionError("inline image is not a base64 data URL");
  }
  const encoded = value.slice(comma + 1);
  const mediaType = header.slice(5).split(";", 1)[0];
  try {
    return [mediaType, strictBase64Decode(encoded)];
  } catch {
    throw new PolicySessionError("inline image base64 is invalid");
  }
}

interface InlineImage {
  mediaType: string;
  sha256: string;
  byteCount: number;
}

type FileIdentity = string;

function fileIdentity(filePath: string): FileIdentity | null {
  try {
    const status = fs.lstatSync(filePath, { bigint: true });
    return `${status.ino}:${status.size}:${status.mtimeNs}`;
  } catch {
    return null;
  }
}

/** Verified admissions bound to one ContextArtifactCatalog instance. */
export class InlineMediaCache {
  private catalog: ContextArtifactCatalog | null = null;
  private images = new GenerationalMemo<string, InlineImage>();
  // Keyed by (ref, data URL) through nested maps, so a large URL is never re-serialized.
  private resolved = new GenerationalMemo<string, Map<string, [Record<string, any>, string, FileIdentity]>>();
  private stored = new Map<string, [Record<string, any>, string, FileIdentity]>();

  bind(catalog: ContextArtifactCatalog): void {
    if (this.catalog !== catalog) {
      this.images = new GenerationalMemo();
      this.resolved = new GenerationalMemo();
      this.stored = new Map();
      this.catalog = catalog;
    }
    this.images.rotate();
    this.resolved.rotate();
  }

  private image(url: unknown): [InlineImage, Buffer | null] {
    const cached = typeof url === "string" ? this.images.get(url) : undefined;
    if (cached !== undefined) return [cached, null];
    const decoded = decodeImageDataUrl(url);
    if (decoded === null) {
      throw new PolicySessionError("lossless Context supports only inline image data URLs");
    }
    const [mediaType, data] = decoded;
    const image = {
      mediaType, sha256: createHash("sha256").update(data).digest("hex"), byteCount: data.length,
    };
    if (typeof url === "string") this.images.put(url, image);
    return [image, data];
  }

  putBytes(data: Buffer, options: {
    kind: string; suffix: string; mediaType: string; roundIndex: number | null; actionStep: number | null;
  }): Record<string, any> {
    const digest = createHash("sha256").update(data).digest("hex");
    return this.put(digest, data.length, () => data, options);
  }

  private put(digest: string, byteCount: number, data: () => Buffer, options: {
    kind: string; suffix: string; mediaType: string; roundIndex: number | null; actionStep: number | null;
  }): Record<string, any> {
    const catalog = this.catalog!;
    const key = JSON.stringify([
      digest, byteCount, options.kind, options.suffix, options.mediaType, options.roundIndex,
      options.actionStep,
    ]);
    const cached = this.stored.get(key);
    if (cached !== undefined && fileIdentity(cached[1]) === cached[2]) return clonePy(cached[0]);
    const reference = catalog.putBytes(data(), options);
    const safeSuffix = options.suffix.startsWith(".") ? options.suffix : `.${options.suffix}`;
    const filePath = path.join(catalog.artifacts.root, `${digest}${safeSuffix}`);
    const identity = fileIdentity(filePath);
    if (identity !== null) this.stored.set(key, [clonePy(reference), filePath, identity]);
    return reference;
  }

  /** Return the public ref exactly as the uncached admission path would. */
  admitImage(image: unknown, options: {
    existingRefs: unknown; imageIndex: number; roundIndex: number | null; actionStep: number | null;
  }): Record<string, any> {
    const catalog = this.catalog!;
    const url = isMapping(image) ? get(image, "url") : undefined;
    const [identity, data] = this.image(url);
    const payload = (): Buffer => {
      if (data !== null) return data;
      return decodeImageDataUrl(url)![1];
    };
    const existingRefs = options.existingRefs;
    if (!Array.isArray(existingRefs) || options.imageIndex >= existingRefs.length) {
      const suffix = identity.mediaType === "image/png" ? ".png" : ".img";
      return this.put(identity.sha256, identity.byteCount, payload, {
        kind: "context_image", suffix, mediaType: identity.mediaType,
        roundIndex: options.roundIndex, actionStep: options.actionStep,
      });
    }
    const existing = existingRefs[options.imageIndex];
    if (!isMapping(existing) || typeof get(existing, "ref") !== "string") {
      throw new PolicySessionError("inline media Artifact reference is invalid");
    }
    const refKey = String(existing.ref);
    const urlKey = typeof url === "string" ? url : JSON.stringify(url ?? null);
    const byUrl = this.resolved.get(refKey);
    const cached = byUrl?.get(urlKey);
    let entry: Record<string, any>;
    if (cached !== undefined && fileIdentity(cached[1]) === cached[2]) {
      entry = cached[0];
    } else {
      byUrl?.delete(urlKey);
      let stored: Buffer;
      try {
        [entry, stored] = catalog.resolve(String(existing.ref));
      } catch {
        throw new PolicySessionError("inline media Artifact reference cannot be resolved");
      }
      if (!stored.equals(payload())) {
        throw new PolicySessionError("inline media differs from its Artifact reference");
      }
      const storage = get(entry, "storage_ref");
      const filePath = path.join(catalog.runRoot, String(get(storage, "path")));
      const identityKey = fileIdentity(filePath);
      if (identityKey !== null) {
        const map = this.resolved.get(refKey) ?? new Map();
        map.set(urlKey, [clonePy(entry), filePath, identityKey]);
        this.resolved.put(refKey, map);
      }
    }
    if (get(entry, "media_type") !== identity.mediaType || get(existing, "sha256") !== get(entry, "sha256")) {
      throw new PolicySessionError("inline media differs from its Artifact reference");
    }
    return catalog.publicRef(entry, options.roundIndex, options.actionStep);
  }
}
