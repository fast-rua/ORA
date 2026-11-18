// A recorded run on the page (`ora replay`): the run rebuilt from its evidence
// (server/replay.ts, GET /api/replay) and a clock over its recorded time. The page at a
// moment is every recorded event up to it, applied in order by the live page's own reducer
// (state.ts), so the replay draws exactly what the live page drew from the same events; the
// cameras at a moment are the capture frame and operator view taken last before it.
//
// Two rates (buildTimeline): the waiting and thinking between the robot's actions play at
// `speed` (16× by default), and each action, from its start to its end with its settling
// window (the frames carry the action's id; native/post-action-settle.jsonl marks which settle),
// plays its execution capture by the frames' simulated time at `motion` speed (1× by default:
// real simulated time, never shorter than the action at `speed`), so a cup swept off the table
// is seen flying. Only captured frames are shown, each the last taken by that simulated time and
// held until the next: nothing is interpolated, and a frame repeating an earlier moment (an
// action's end and the next one's start share one) adds no time. `motion: "same"` keeps one
// speed throughout. The conversation follows the recorded clock in both, at the motion's pace
// during a motion.

import type { CaptureFrame, ReplayScript, ViewFrame } from "../../server/replay.ts";
import { initialState, reduce, type PageState } from "../state.ts";
import type { FramesSummary, StepFrames } from "./frames.ts";
import type { LiveView } from "./live.ts";

export type { ReplayScript };

export const SPEEDS = [1, 2, 4, 8, 16, 32, 64];
export const DEFAULT_SPEED = 16;
/** How fast an action's motion plays against its simulated time; "same": at `speed`, like the rest. */
export type MotionSpeed = number | "same";
export const MOTION_SPEEDS: MotionSpeed[] = [0.25, 0.5, 1, 2, 4, "same"];
export const DEFAULT_MOTION: MotionSpeed = 1;

export interface ReplayOptions {
  speed: number;
  /** The robot's motion against its simulated time, or "same" (one speed throughout). */
  motion: MotionSpeed;
  /** Seconds into the replay to start at. */
  from: number;
  /** Start playing at once. */
  autoplay: boolean;
  /** A video is being made: the clock is driven from outside (window.__oraReplay), no controls. */
  exporting: boolean;
  /** No run ID or path on the page. */
  publish: boolean;
}

export function motionSpeed(value: string | null): MotionSpeed {
  if (value === "same") return "same";
  const speed = Number(value);
  return value !== null && value !== "" && Number.isFinite(speed) && speed > 0 ? speed : DEFAULT_MOTION;
}

/** The replay a link asks for: `?replay=1&speed=16&motion=1|same&from=30&play=0&export=1&publish=1`. */
export function replayOptions(search: string): ReplayOptions | null {
  const params = new URLSearchParams(search);
  if (!params.has("replay")) return null;
  const speed = Number(params.get("speed"));
  const from = Number(params.get("from"));
  return {
    speed: Number.isFinite(speed) && speed > 0 ? speed : DEFAULT_SPEED,
    motion: motionSpeed(params.get("motion")),
    from: Number.isFinite(from) && from > 0 ? from : 0,
    autoplay: params.get("play") !== "0",
    exporting: params.get("export") === "1",
    publish: params.get("publish") === "1",
  };
}

/** The page at a moment: the recorded events up to it, applied in order (forward from the last moment). */
export class ReplayState {
  private readonly script: ReplayScript;
  private index = 0;
  private state: PageState = initialState;
  private time = -Infinity;

  constructor(script: ReplayScript) {
    this.script = script;
  }

  /** How many recorded events the page at the last moment holds. */
  get applied(): number {
    return this.index;
  }

  at(time: number): PageState {
    if (time < this.time) {
      this.index = 0;
      this.state = initialState;
    }
    const events = this.script.events;
    while (this.index < events.length && events[this.index].at <= time) {
      const { at, event } = events[this.index++];
      this.state = reduce(this.state, { type: "event", event, now: at });
    }
    this.time = time;
    return this.state;
  }
}

/** How many of `list` (in time order) happened at or before `time`. */
export function countUpTo(list: Array<{ at: number }>, time: number): number {
  let low = 0;
  let high = list.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (list[middle].at <= time) low = middle + 1;
    else high = middle;
  }
  return low;
}

/** The frames known `captured` and `viewed` frames in: what `ora_get_frames` answered live then. */
export function framesSummary(script: ReplayScript, captured: number, viewed: number): FramesSummary {
  const capture = script.capture;
  const view = script.operatorView;
  let motion: FramesSummary["motion"] = null;
  if (capture && captured > 0) {
    const steps: Record<string, number> = {};
    for (const frame of capture.frames.slice(0, captured)) {
      if (frame.step !== null) steps[String(frame.step)] = (steps[String(frame.step)] ?? 0) + 1;
    }
    motion = { cameras: capture.cameras, fps: capture.fps, steps };
  }
  let operatorView: FramesSummary["operatorView"] = null;
  if (view && viewed > 0) {
    const frames = view.frames.slice(0, viewed);
    const steps = [...new Set(frames.map((frame) => frame.step))].sort((a, b) => a - b);
    const newest = frames[frames.length - 1];
    operatorView = { camera: view.camera, width: view.width, height: view.height, steps,
      latest: { step: newest.step, camera: view.camera, width: view.width, height: view.height, url: newest.url } };
  }
  return { motion, operatorView };
}

/** One step's recorded motion and operator view, as far as they were recorded by `time`. */
export function stepFrames(script: ReplayScript, summary: FramesSummary, step: number, time: number): StepFrames {
  const capture = script.capture;
  // One frame per simulated moment: a frame repeating the one before's moment replaces it (no frame held twice).
  const frames: CaptureFrame[] = [];
  for (const frame of capture ? capture.frames.filter((item) => item.step === step && item.at <= time) : []) {
    if (frames.length && frame.simTimeS === frames[frames.length - 1].simTimeS) frames[frames.length - 1] = frame;
    else frames.push(frame);
  }
  const view: ViewFrame | undefined = script.operatorView?.frames.find((frame) => frame.step === step && frame.at <= time);
  return {
    summary,
    motion: frames.length ? {
      step, actionIds: [...new Set(frames.map((frame) => frame.actionId ?? ""))].filter(Boolean),
      cameras: capture!.cameras, fps: capture!.fps,
      frames: frames.map((frame) => ({ index: frame.index, phase: frame.phase, simTimeS: frame.simTimeS, images: frame.images })),
    } : null,
    operatorView: view ? { step, camera: script.operatorView!.camera, width: script.operatorView!.width,
      height: script.operatorView!.height, url: view.url } : null,
  };
}

// -- The cameras at a moment ---------------------------------------------------------------

const pixels = (size: [number, number] | undefined) => (size ? size[0] * size[1] : 0);

interface CameraIndex {
  /** Each camera's frames (indices into capture.frames, in time order). */
  frames: Map<string, number[]>;
  /** The capture's operator cameras (the operator view's camera, or one named for the operator), sharpest first. */
  operator: string[];
}

const indexes = new WeakMap<ReplayScript, CameraIndex>();

function cameraIndex(script: ReplayScript): CameraIndex {
  const known = indexes.get(script);
  if (known) return known;
  const frames = new Map<string, number[]>();
  (script.capture?.frames ?? []).forEach((frame, index) => {
    for (const camera of Object.keys(frame.images)) {
      if (!frames.has(camera)) frames.set(camera, []);
      frames.get(camera)!.push(index);
    }
  });
  const view = script.operatorView?.camera ?? null;
  const sizes = script.capture?.sizes ?? {};
  const operator = [...frames.keys()].filter((name) => name === view || /operator/i.test(name))
    .sort((a, b) => pixels(sizes[b]) - pixels(sizes[a]));
  const index = { frames, operator };
  indexes.set(script, index);
  return index;
}

/**
 * The shape of the run's camera box (width / height) as its evidence says it: its sharpest operator
 * camera's frames in the execution capture, else its operator-view stills, else its first captured
 * camera's (a run with no operator camera); null when the evidence has no camera sizes.
 */
export function knownCameraAspect(script: ReplayScript): number | null {
  const sizes = script.capture?.sizes ?? {};
  for (const name of cameraIndex(script).operator) {
    const size = sizes[name];
    if (size && size[0] > 0 && size[1] > 0) return size[0] / size[1];
  }
  const view = script.operatorView;
  if (view && view.width > 0 && view.height > 0) return view.width / view.height;
  const first = Object.values(sizes).find((size) => size && size[0] > 0 && size[1] > 0);
  return first ? first[0] / first[1] : null;
}

/** The camera box's shape, fixed for the whole replay: as its evidence says it, else 4:3. */
export function cameraAspect(script: ReplayScript): number {
  return knownCameraAspect(script) ?? 4 / 3;
}

/** The last of `list` (ascending frame indices) below `count`, or null. */
function lastBelow(list: number[], count: number): number | null {
  let low = 0;
  let high = list.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (list[middle] < count) low = middle + 1;
    else high = middle;
  }
  return low > 0 ? list[low - 1] : null;
}

/**
 * The world when `captured` capture frames and `viewed` operator views had been taken, as the
 * live world view showed it: each camera's newest frame (cameras taken at different rates each
 * keep their own newest), the operator's camera through its sharpest capture camera unless a
 * blurrier one is more than a second newer, and the operator view when it was taken later and
 * is as sharp.
 */
export function worldAt(script: ReplayScript, captured: number, viewed: number, moving: boolean,
  settling = false): LiveView | null {
  const capture = script.capture;
  const frames = capture?.frames ?? [];
  const index = cameraIndex(script);
  const sizes = capture?.sizes ?? {};
  const view = viewed > 0 ? script.operatorView!.frames[viewed - 1] : null;
  const cameras: Array<{ name: string; url: string }> = [];
  for (const [name, list] of index.frames) {
    if (index.operator.includes(name)) continue;
    const at = lastBelow(list, captured);
    if (at !== null) cameras.push({ name, url: frames[at].images[name] });
  }
  let operator: { url: string; at: number; pixels: number } | null = null;
  for (const name of index.operator) {
    const at = lastBelow(index.frames.get(name)!, captured);
    if (at === null) continue;
    const candidate = { url: frames[at].images[name], at: frames[at].at, pixels: pixels(sizes[name]) };
    if (operator === null || candidate.at > operator.at + 1000) operator = candidate;
  }
  const viewPixels = script.operatorView ? script.operatorView.width * script.operatorView.height : 0;
  if (view !== null && (operator === null || (view.at > operator.at && viewPixels >= operator.pixels))) {
    operator = { url: view.url, at: view.at, pixels: viewPixels };
  }
  const name = script.operatorView?.camera ?? index.operator[0] ?? null;
  if (operator !== null && name !== null) cameras.unshift({ name, url: operator.url });
  if (!cameras.length) return null;
  const frame = captured > 0 && frames.length ? frames[Math.min(captured, frames.length) - 1] : null;
  return {
    phase: settling ? "settling" : moving ? "executing" : "holding", message: "", connected: true, error: null, cameras,
    frame: frame ? { round_index: 0, sim_time_s: frame.simTimeS, capture_index: frame.index, step: frame.step } : null,
  };
}

// -- Two rates: waiting at `speed`, the robot's motion by its simulated time ---------------

/** One stretch of a replay, by its recorded time and its playback time (ms from the replay's start). */
export interface Segment {
  kind: "wait" | "motion";
  from: number;
  to: number;
  playFrom: number;
  playTo: number;
  /** A motion: its capture frames [first, last) and their simulated time (s), its action and step. */
  frames?: [number, number];
  simFrom?: number;
  simTo?: number;
  action?: string | null;
  step?: number | null;
  /** Its settling window's simulated time (s) and how it ended, when the action had one. */
  settleFrom?: number | null;
  settleTo?: number | null;
  settleEnded?: string | null;
}

export interface Timeline {
  segments: Segment[];
  /** The whole replay's playback time (ms): the video's length. */
  duration: number;
  speed: number;
  motion: MotionSpeed;
}

const baseAction = (id: string | null) => (id === null ? null : id.split(":")[0]);

/**
 * Each action's motion in the execution capture: its frames (they carry the action's id through
 * its settling window), their simulated time, its settling window, and its recorded time from
 * submission to receipt.
 */
export function motionGroups(script: ReplayScript): Array<{ from: number; to: number; first: number; last: number;
  simFrom: number; simTo: number; action: string; step: number | null; settleFrom: number | null; settleTo: number | null;
  settleEnded: string | null }> {
  const frames = script.capture?.frames ?? [];
  const actions = new Map(script.actions.map((action) => [action.id, action]));
  const settles = new Map((script.capture?.settles ?? []).filter((item) => item.steps > 0).map((item) => [item.actionId, item]));
  const groups = [];
  let index = 0;
  while (index < frames.length) {
    const action = baseAction(frames[index].actionId);
    if (action === null) {
      index += 1;
      continue;
    }
    let next = index + 1;
    while (next < frames.length && (baseAction(frames[next].actionId) === action
      || (frames[next].actionId === null && /settl/i.test(frames[next].phase)))) next += 1;
    const sims = frames.slice(index, next).map((frame) => frame.simTimeS).filter(Number.isFinite);
    const recorded = actions.get(action);
    const settle = settles.get(action) ?? null;
    groups.push({
      from: Math.min(recorded?.start ?? Infinity, frames[index].at), to: Math.max(recorded?.end ?? -Infinity, frames[next - 1].at),
      first: index, last: next, simFrom: sims.length ? Math.min(...sims) : 0, simTo: sims.length ? Math.max(...sims) : 0,
      action, step: frames[index].step, settleFrom: settle?.simFrom ?? null, settleTo: settle?.simTo ?? null,
      settleEnded: settle?.ended ?? null,
    });
    index = next;
  }
  return groups;
}

/** The replay's stretches at `speed` and `motion` speed, back to back over its recorded time. */
export function buildTimeline(script: ReplayScript, speed: number, motion: MotionSpeed): Timeline {
  const segments: Segment[] = [];
  let wall = script.start;
  let play = 0;
  const wait = (to: number) => {
    if (to <= wall) return;
    const length = (to - wall) / speed;
    segments.push({ kind: "wait", from: wall, to, playFrom: play, playTo: play + length });
    play += length;
    wall = to;
  };
  if (motion !== "same") {
    for (const group of motionGroups(script)) {
      const from = Math.max(group.from, wall);
      const to = Math.min(group.to, script.end);
      if (to <= from) continue;
      wait(from);
      // Its simulated time at the motion's speed, never shorter than its recorded time at `speed`.
      const length = Math.max(((group.simTo - group.simFrom) * 1000) / motion, (to - from) / speed);
      segments.push({ kind: "motion", from, to, playFrom: play, playTo: play + length, frames: [group.first, group.last],
        simFrom: group.simFrom, simTo: group.simTo, action: group.action, step: group.step, settleFrom: group.settleFrom,
        settleTo: group.settleTo, settleEnded: group.settleEnded });
      play += length;
      wall = to;
    }
  }
  wait(script.end);
  return { segments, duration: play, speed, motion };
}

/** What a playback moment shows: its recorded time, its stretch, and in a motion the captured frames up to it. */
export interface Moment {
  wall: number;
  segment: Segment | null;
  /** In a motion: how many capture frames are shown taken (the last is the one on screen). */
  captured: number | null;
  /** In a motion: the simulated time (s) the playback is at. */
  sim: number | null;
  /** In a motion: the frame on screen was taken in the action's settling window. */
  settling: boolean;
}

function segmentAt(timeline: Timeline, position: number): Segment | null {
  const segments = timeline.segments;
  let low = 0;
  let high = segments.length - 1;
  if (high < 0) return null;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (segments[middle].playFrom <= position) low = middle;
    else high = middle - 1;
  }
  return segments[low];
}

export function momentAt(script: ReplayScript, timeline: Timeline, position: number): Moment {
  const at = Math.max(0, Math.min(timeline.duration, position));
  const segment = segmentAt(timeline, at);
  if (segment === null) return { wall: script.start, segment: null, captured: null, sim: null, settling: false };
  const length = segment.playTo - segment.playFrom;
  const wall = segment.from + (length > 0 ? (at - segment.playFrom) / length : 1) * (segment.to - segment.from);
  if (segment.kind !== "motion" || timeline.motion === "same") return { wall, segment, captured: null, sim: null, settling: false };
  const sim = Math.min(segment.simTo!, segment.simFrom! + ((at - segment.playFrom) * timeline.motion) / 1000);
  // The last frame taken at or before that simulated time, held until the next: nothing in between is made up.
  const frames = script.capture!.frames;
  let shown = segment.frames![0];
  for (let index = segment.frames![0]; index < segment.frames![1]; index++) {
    if (frames[index].simTimeS <= sim + 1e-9) shown = index;
    else break;
  }
  return { wall, segment, captured: shown + 1, sim, settling: frames[shown].settle === true };
}

/** The playback moment of a recorded time (a motion's recorded time maps across its playback). */
export function positionOf(timeline: Timeline, wall: number): number {
  const segments = timeline.segments;
  if (!segments.length || wall <= segments[0].from) return 0;
  for (const segment of segments) {
    if (wall <= segment.to) {
      const span = segment.to - segment.from;
      return segment.playFrom + (span > 0 ? (wall - segment.from) / span : 0) * (segment.playTo - segment.playFrom);
    }
  }
  return timeline.duration;
}

/** m:ss (or h:mm:ss) of a replay's elapsed time. */
export function clockText(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = String(total % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${rest}` : `${minutes}:${rest}`;
}
