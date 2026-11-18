// ORA's images on the messages an RPC client reads.
//
// The model's observation and tool images live in ORA's canonical history
// (the bytes the model received, as data URLs); Pi's session holds the
// observation as a text custom message (`ora.observation`) and each tool
// result as text. For a client, a toolResult and an `ora.observation` message
// gain Pi's standard ImageContent blocks, and `details.ora` names where they
// belong: {step, toolCallId, camera, artifact, images: [{camera, artifact,
// mimeType, url, inline}]}. Free play keeps its images in Pi's own messages
// (`ora.play_observation`, robot tool results): they are listed the same way.
// `get_messages` inlines the images of the latest `inlineSteps` observation
// steps only; earlier ones are listed in
// `details.ora.images` with a URL (`/ora/artifacts/<file>`, served by
// `ora web`) and the run-relative artifact path, so a long multi-camera
// episode does not become a response of tens of megabytes.
//
// The session's own message objects are never changed: every message handed
// out is a copy.

import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { join } from "node:path";

import { cameraOf, imageCaptions } from "../tui/render.ts";

type Json = any;

export interface ImageSource {
  runRoot: string;
  /** Canonical ORA history: observation messages carry `_observation.step`, tool images `_tool_image_call_id`. */
  history(): Json[];
  /** Tool-image messages of the reply in progress (not yet in the history). */
  pendingToolImages(): Json[];
}

export interface OraImage {
  data: string;
  mimeType: string;
  camera: string | null;
  /** Run-relative path of the stored bytes (`native/agent-artifacts/<sha256>.png`), when stored. */
  artifact: string | null;
}

export const OBSERVATION = "ora.observation";
/** Observation messages: an episode's (images in ORA's history) and free play's (images in the message). */
export const OBSERVATIONS = new Set([OBSERVATION, "ora.play_observation"]);
export const DEFAULT_INLINE_STEPS = 3;

function parseDataUrl(url: string): { mimeType: string; data: string } | null {
  const match = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  return match ? { mimeType: match[1], data: match[2] } : null;
}

function observationStep(message: Json): number | null {
  if (message?.role !== "custom" || !OBSERVATIONS.has(message.customType)) return null;
  const own = Number(message.details?.ora_observation_step);
  return Number.isFinite(own) ? own : null;
}

export class ImageCompletion {
  private readonly cache = new WeakMap<object, OraImage[]>();
  /** Tool call -> the observation step its turn answered (from the live event stream). */
  private readonly callSteps = new Map<string, number>();
  private liveStep: number | null = null;
  private readonly source: ImageSource;
  readonly inlineSteps: number;

  constructor(source: ImageSource, inlineSteps = DEFAULT_INLINE_STEPS) {
    this.source = source;
    this.inlineSteps = inlineSteps;
  }

  private stored(data: string): string | null {
    const digest = createHash("sha256").update(Buffer.from(data, "base64")).digest("hex");
    return [".png", ".img"].map((suffix) => `native/agent-artifacts/${digest}${suffix}`)
      .find((path) => existsSync(join(this.source.runRoot, path))) ?? null;
  }

  /** The images of an ORA history message (`image_url` data URLs under their captions). */
  private imagesOf(message: Json): OraImage[] {
    if (!message || typeof message !== "object" || !Array.isArray(message.content)) return [];
    const cached = this.cache.get(message);
    if (cached) return cached;
    const captions = imageCaptions(message);
    const images: OraImage[] = [];
    let index = 0;
    for (const block of message.content) {
      if (block?.type !== "image_url") continue;
      const caption = captions[index++] ?? null;
      const parsed = parseDataUrl(String(block.image_url?.url ?? ""));
      if (parsed === null) continue;
      images.push({ ...parsed, camera: cameraOf(caption), artifact: this.stored(parsed.data) });
    }
    this.cache.set(message, images);
    return images;
  }

  /** The images a Pi message carries itself (free play), each under the caption before it. */
  private ownImages(message: Json): OraImage[] {
    if (!message || typeof message !== "object" || !Array.isArray(message.content)) return [];
    const cached = this.cache.get(message);
    if (cached) return cached;
    const images: OraImage[] = [];
    let caption: string | null = null;
    for (const block of message.content) {
      if (block?.type === "text") {
        const text = String(block.text ?? "");
        caption = text.length <= 200 && !text.trimStart().startsWith("{") ? text.split("\n", 1)[0].trim() || null : null;
      } else if (block?.type === "image" && typeof block.data === "string") {
        images.push({ data: block.data, mimeType: String(block.mimeType ?? "image/png"), camera: cameraOf(caption),
          artifact: this.stored(block.data) });
        caption = null;
      }
    }
    this.cache.set(message, images);
    return images;
  }

  observationImages(step: number): OraImage[] {
    const history = this.source.history();
    for (let index = history.length - 1; index >= 0; index--) {
      if (history[index]?._observation?.step === step) return this.imagesOf(history[index]);
    }
    return [];
  }

  toolImages(toolCallId: string): OraImage[] {
    for (const message of [...this.source.pendingToolImages(), ...this.source.history()]) {
      if (message?._tool_image_call_id === toolCallId) return this.imagesOf(message);
    }
    return [];
  }

  /** A copy of `message` with its images; `inline` false lists them by URL only. */
  complete(message: Json, step: number | null, inline: boolean): Json {
    if (!message || typeof message !== "object") return message;
    let images: OraImage[] = [];
    let toolCallId: string | null = null;
    if (message.role === "custom" && OBSERVATIONS.has(message.customType)) {
      step = observationStep(message) ?? step;
      images = step === null || message.customType !== OBSERVATION ? [] : this.observationImages(step);
    } else if (message.role === "toolResult") {
      toolCallId = String(message.toolCallId ?? "");
      step = this.callSteps.get(toolCallId) ?? step;
      images = this.toolImages(toolCallId);
    } else {
      return message;
    }
    // Free play: the message carries its images; listed like ORA's, and only inline while recent.
    const own = images.length === 0 ? this.ownImages(message) : [];
    if (own.length) images = own;
    const listed = images.map((image) => ({
      camera: image.camera, artifact: image.artifact, mimeType: image.mimeType, inline,
      url: image.artifact ? `/ora/artifacts/${image.artifact.split("/").pop()}` : null,
    }));
    const text = typeof message.content === "string"
      ? [{ type: "text", text: message.content }]
      : Array.isArray(message.content)
        ? message.content.filter((block: Json) => !(own.length && block?.type === "image")) : [];
    const content = inline
      ? [...text, ...images.map((image) => ({ type: "image", data: image.data, mimeType: image.mimeType }))]
      : text;
    const ora = {
      step, ...(toolCallId === null ? {} : { toolCallId }),
      camera: listed[0]?.camera ?? null, artifact: listed[0]?.artifact ?? null, images: listed,
    };
    return { ...message, content, details: { ...(message.details ?? {}), ora } };
  }

  /** Messages in order, completed; images inline for the latest `inlineSteps` observation steps. */
  completeAll(messages: Json[]): Json[] {
    const steps: Array<number | null> = [];
    let step: number | null = null;
    for (const message of messages) {
      step = observationStep(message) ?? step;
      steps.push(message?.role === "toolResult" ? (this.callSteps.get(String(message.toolCallId)) ?? step) : step);
    }
    const known = steps.filter((value): value is number => value !== null);
    const newest = known.length ? Math.max(...known) : null;
    return messages.map((message, index) => {
      const at = steps[index];
      const inline = newest === null || at === null || at > newest - this.inlineSteps;
      return this.complete(message, at, inline);
    });
  }

  /** An event as clients receive it: ORA's images on the messages it carries (the live step inline). */
  event(event: Json): Json {
    switch (event?.type) {
      case "message_start":
      case "message_end": {
        const message = event.message;
        this.liveStep = observationStep(message) ?? this.liveStep;
        return { ...event, message: this.complete(message, this.liveStep, true) };
      }
      case "tool_execution_start":
        if (this.liveStep !== null) this.callSteps.set(String(event.toolCallId), this.liveStep);
        return event;
      case "tool_execution_end": {
        const result = event.result;
        if (!result || typeof result !== "object") return event;
        const completed = this.complete({ role: "toolResult", toolCallId: event.toolCallId, ...result },
          this.liveStep, true);
        const { role: _role, toolCallId: _id, ...rest } = completed;
        return { ...event, result: rest };
      }
      case "turn_end":
        return {
          ...event, message: this.complete(event.message, this.liveStep, true),
          toolResults: Array.isArray(event.toolResults)
            ? event.toolResults.map((message: Json) => this.complete(message, this.liveStep, true)) : event.toolResults,
        };
      case "agent_end":
        return Array.isArray(event.messages) ? { ...event, messages: this.completeAll(event.messages) } : event;
      default:
        return event;
    }
  }
}
