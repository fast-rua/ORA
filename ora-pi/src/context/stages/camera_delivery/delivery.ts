// What an append-only (Context management v8, v9) projection shows the model;
// the camera_delivery stage (stage.ts) renders with it.
//
// v8 never ages history into recoverable-artifact Pointers, so none is shown:
// the image markers materialization records stay internal (v9 shows the marker
// of an image an exchange aging removed). Fixed camera images follow
// `camera_images`:
// * `inline_until_compaction_v1` -- they stay in their observation message
//   until a rollout compaction absorbs its ActionStep;
// * `exchange_window_v1` (with exchange_aging) -- the same here; the exchange
//   aging removes them from the observation message when the exchange it
//   arrived in ages, leaving each image's marker as its reference;
// * `rolling_tail_v1` -- every observation message keeps one stable text
//   reference instead of its camera images, and one message at the end of the
//   request carries the camera images of the latest `camera_tail_observations`
//   observations. Only that last message changes from one request to the next,
//   so the history before it stays a Provider-cacheable prefix.
//
// Tool-returned images are untouched here: they stay where the tool delivered
// them until a compaction absorbs their ActionStep (or, with exchange_aging,
// their exchange ages). The functions are pure; the projection's internal
// history (and every lossless invariant over it) keeps its images and markers.

import { CAMERA_EXCHANGE_WINDOW, CAMERA_ROLLING_TAIL } from "../../management.ts";
import { type Message, observationGroups } from "../../projection.ts";
import { clonePy, get, isMapping } from "../../pyutil.ts";
import { isRecoverableMarker as isMarker, withoutRecoverableMarkers as withoutMarkers } from "../services/references.ts";

export const CAMERA_DELIVERY_SCHEMA = "ora.camera-delivery.v1";
/** The tail message's first text starts with this; the request assembler finds the tail by it. */
export const CAMERA_TAIL_HEADER = "CAMERA TAIL · ";
const LABEL = /^(.+) · sequence (\d+) · (.+) · latest at this message$/;

export interface CameraDeliveryOptions {
  mode: string;
  /** Observations whose camera images the tail carries (rolling mode). */
  tailObservations: number | null;
  /** Whether the advertised observe tool takes `step` (it then names the retrieval). */
  observeStep: boolean;
}

/** One camera image of an observation message: its label and image block indexes. */
interface CameraImage {
  camera: string;
  label: string;
  sequence: number;
  labelIndex: number;
  imageIndex: number;
  /** Position among the message's image blocks (its `_context_image_refs` index). */
  ordinal: number;
}

/** The labelled fixed camera images of one observation message, in order. */
function cameraImages(message: Message): CameraImage[] {
  const content = get(message, "content");
  const cameras = get(get(message, "_causal_observation") ?? {}, "cameras");
  if (!isMapping(get(message, "_observation")) || !Array.isArray(content) || !Array.isArray(cameras)) return [];
  const images: CameraImage[] = [];
  let ordinal = 0;
  let label: { index: number; match: RegExpMatchArray } | null = null;
  content.forEach((block, index) => {
    if (!isMapping(block)) return;
    if (get(block, "type") === "image_url") {
      if (label !== null && cameras.includes(label.match[3])) {
        images.push({
          camera: label.match[3], label: label.match[1], sequence: Number(label.match[2]),
          labelIndex: label.index, imageIndex: index, ordinal,
        });
      }
      ordinal += 1;
      label = null;
      return;
    }
    if (isMarker(block)) return;
    const match = get(block, "type") === "text" && typeof get(block, "text") === "string"
      ? String(block.text).match(LABEL) : null;
    label = match ? { index, match } : null;
  });
  return images;
}

/** The content indexes of the fixed camera images of an observation message (exchange_aging ages them by mode). */
export function cameraImageIndexes(message: Message): Set<number> {
  return new Set(cameraImages(message).map((image) => image.imageIndex));
}

function historyReference(step: unknown, images: CameraImage[], options: CameraDeliveryOptions): string {
  const cameras = images.map((image) => image.camera).join(", ");
  const retrieval = options.observeStep ? `; afterwards observe with step=${String(step)} shows them again` : "";
  return `Camera images of observation step ${String(step)} (${cameras}) are not repeated here: `
    + `while it is one of the latest ${options.tailObservations} observations they are attached once at `
    + `the end of the request${retrieval}.`;
}

/** The observation message with its camera images replaced by one reference. */
function referencedObservation(message: Message, images: CameraImage[], options: CameraDeliveryOptions): Message {
  const removed = new Set<number>();
  for (const image of images) {
    removed.add(image.labelIndex);
    removed.add(image.imageIndex);
  }
  const content: unknown[] = [];
  let referenced = false;
  (message.content as unknown[]).forEach((block, index) => {
    if (removed.has(index)) {
      if (!referenced) {
        content.push({ type: "text", text: historyReference(get(message._observation, "step"), images, options) });
        referenced = true;
      }
      return;
    }
    content.push(block);
  });
  const out: Message = { ...message, content };
  const refs = get(message, "_context_image_refs");
  if (Array.isArray(refs)) {
    const kept = new Set(images.map((image) => image.ordinal));
    out._context_image_refs = refs.filter((_ref, ordinal) => !kept.has(ordinal));
    if (!out._context_image_refs.length) delete out._context_image_refs;
  }
  return out;
}

/** The history as v8 shows it, without the tail (also the compactor's input). */
export function deliveredHistory(messages: Message[], options: CameraDeliveryOptions): Message[] {
  return messages.map((message) => {
    const clean = withoutMarkers(message) ?? message;
    if (options.mode !== CAMERA_ROLLING_TAIL) return clean;
    const images = cameraImages(clean);
    return images.length ? referencedObservation(clean, images, options) : clean;
  });
}

export interface CameraTail {
  /** The tail message, or none (inline mode, or no camera image to carry). */
  messages: Message[];
  /** The `ora.camera-delivery.v1` record of this delivery. */
  record: Record<string, any>;
}

/**
 * The delivery record and, in rolling mode, the tail message built from the
 * latest observations of ``materialized`` (a compaction does not stop the
 * tail). The tail goes after the whole delivered history.
 */
export function cameraTail(materialized: Message[], options: CameraDeliveryOptions): CameraTail {
  const record: Record<string, any> = {
    schema: CAMERA_DELIVERY_SCHEMA,
    camera_images: options.mode,
    tail_observations: options.tailObservations,
    retrieval: options.mode === CAMERA_ROLLING_TAIL && options.observeStep ? "observe.step"
      : options.mode === CAMERA_EXCHANGE_WINDOW ? "observe.ref" : null,
  };
  if (options.mode !== CAMERA_ROLLING_TAIL) return { messages: [], record };
  const groups = observationGroups(materialized);
  const references: Record<string, any>[] = [];
  groups.forEach(([start], actionStep) => {
    const count = cameraImages(materialized[start]).length;
    if (count) references.push({ action_step_index: actionStep, camera_images: count });
  });
  const tailGroups = groups.slice(Math.max(0, groups.length - Number(options.tailObservations)));
  const content: Record<string, any>[] = [];
  const refs: Record<string, any>[] = [];
  const tail: Record<string, any>[] = [];
  tailGroups.forEach(([start], offset) => {
    const message = materialized[start];
    const step = get(message._observation, "step");
    const images = cameraImages(message);
    const current = offset === tailGroups.length - 1;
    const imageRefs = get(message, "_context_image_refs");
    for (const image of images) {
      const label = `${image.label} · sequence ${image.sequence} · ${image.camera} · observation step ${String(step)}`
        + (current ? " · current" : "");
      content.push({ type: "text", text: label });
      content.push(clonePy(message.content[image.imageIndex]));
      if (Array.isArray(imageRefs) && isMapping(imageRefs[image.ordinal])) refs.push(clonePy(imageRefs[image.ordinal]));
    }
    tail.push({
      action_step_index: groups.length - tailGroups.length + offset,
      observation_step: step ?? null,
      cameras: images.map((image) => image.camera),
    });
  });
  const tailImages = tail.reduce((total, item) => total + item.cameras.length, 0);
  record.tail = tail;
  record.tail_images = tailImages;
  record.history_references = references;
  if (!tailImages) return { messages: [], record };
  const header = `${CAMERA_TAIL_HEADER}latest ${tail.length} observation${tail.length === 1 ? "" : "s"}, oldest `
    + "first; the last is the current observation. Their automatic camera images are shown only here.";
  const message: Message = {
    role: "user",
    content: [{ type: "text", text: header }, ...content],
    _camera_tail: true,
  };
  if (refs.length === tailImages) message._context_image_refs = refs;
  return { messages: [message], record };
}
