// The run's operator-only pictures for a client: the robot's recorded motion and the operator view.
//
// Read from the run as the Supervisor and the environment wrote them, never copied:
//   motion         native/execution-capture/ (manifest.json, timeline.jsonl, frames/*.png):
//                  the original execution's frames, each naming the action it belongs to;
//                  an action belongs to the observation step it was proposed at
//                  (agent-events.jsonl `action_proposed`: observation_step, payload.action_id)
//   operator view  events.jsonl `operator_view_frame` (native/operator-view/<step>.png),
//                  the environment's picture for the operator (episode_play.py)
// Images are named by the URLs `ora web` serves them at (ora/app/web_gateway.py):
// /ora/capture/<frame>.png and /ora/operator-view/<step>.png. Files are read
// incrementally (appended lines only); a file the process may not read is
// simply absent. Nothing here is the model's: no request carries these.

import { closeSync, existsSync, fstatSync, openSync, readFileSync, readSync } from "node:fs";
import { join } from "node:path";
import { StringDecoder } from "node:string_decoder";

type Json = any;

const READ_CHUNK = 1 << 20;
const FRAME_FILE = /^frames\/(\d{8}-camera-\d+\.png)$/;
const VIEW_FILE = /^native\/operator-view\/(\d{6,9}\.png)$/;

/** Complete appended lines of a JSONL file, from where the last read stopped. */
export class JsonlTail {
  private offset = 0;
  private partial = "";
  private decoder = new StringDecoder("utf8");
  private identity: string | null = null;
  readonly path: string;

  constructor(path: string) {
    this.path = path;
  }

  read(): Json[] {
    let fd: number;
    try {
      fd = openSync(this.path, "r");
    } catch {
      return [];
    }
    const lines: Json[] = [];
    try {
      const stat = fstatSync(fd);
      const identity = `${stat.dev}:${stat.ino}`;
      if (this.identity !== identity || stat.size < this.offset) {
        // A new or replaced file starts over.
        this.identity = identity;
        this.offset = 0;
        this.partial = "";
        this.decoder = new StringDecoder("utf8");
      }
      const buffer = Buffer.alloc(READ_CHUNK);
      while (this.offset < stat.size) {
        const read = readSync(fd, buffer, 0, Math.min(READ_CHUNK, stat.size - this.offset), this.offset);
        if (read <= 0) break;
        this.offset += read;
        this.partial += this.decoder.write(buffer.subarray(0, read));
        let newline: number;
        while ((newline = this.partial.indexOf("\n")) >= 0) {
          const line = this.partial.slice(0, newline);
          this.partial = this.partial.slice(newline + 1);
          if (!line.trim()) continue;
          try {
            lines.push(JSON.parse(line));
          } catch {
            // a torn or foreign line: skipped
          }
        }
      }
    } finally {
      closeSync(fd);
    }
    return lines;
  }
}

export interface MotionFrame {
  index: number;
  phase: string;
  simTimeS: number;
  actionId: string | null;
  /** Camera -> the frame file name under native/execution-capture/frames/. */
  files: Record<string, string>;
}

export interface OperatorViewFrame {
  step: number;
  camera: string;
  width: number;
  height: number;
  sha256: string;
  url: string;
}

export interface FramesSummary {
  motion: { cameras: string[]; fps: number | null; steps: Record<string, number> } | null;
  operatorView: { camera: string; width: number; height: number; steps: number[]; latest: OperatorViewFrame } | null;
}

export interface StepMotion {
  step: number;
  actionIds: string[];
  cameras: string[];
  fps: number | null;
  frames: Array<{ index: number; phase: string; simTimeS: number; images: Record<string, string> }>;
}

export function captureUrl(file: string): string {
  return `/ora/capture/${file}`;
}

export class RunFrames {
  readonly runRoot: string;
  private readonly timeline: JsonlTail;
  private readonly agentEvents: JsonlTail;
  private readonly events: JsonlTail;
  private readonly frames: MotionFrame[] = [];
  private readonly actionSteps = new Map<string, number>();
  private readonly views = new Map<number, OperatorViewFrame>();
  private manifest: Json = null;

  constructor(runRoot: string) {
    this.runRoot = runRoot;
    this.timeline = new JsonlTail(join(runRoot, "native", "execution-capture", "timeline.jsonl"));
    this.agentEvents = new JsonlTail(join(runRoot, "agent-events.jsonl"));
    this.events = new JsonlTail(join(runRoot, "events.jsonl"));
  }

  refresh(): void {
    for (const event of this.agentEvents.read()) {
      if (event?.type !== "action_proposed") continue;
      const id = event.payload?.action_id;
      if (typeof id === "string" && Number.isInteger(event.observation_step)) this.actionSteps.set(id, event.observation_step);
    }
    for (const event of this.events.read()) {
      if (event?.type !== "operator_view_frame" || !Number.isInteger(event.observation_step)) continue;
      const match = VIEW_FILE.exec(String(event.path ?? ""));
      if (!match) continue;
      this.views.set(event.observation_step, {
        step: event.observation_step, camera: String(event.camera ?? "operator view"), width: Number(event.width),
        height: Number(event.height), sha256: String(event.sha256 ?? ""), url: `/ora/operator-view/${match[1]}`,
      });
    }
    const manifestPath = join(this.runRoot, "native", "execution-capture", "manifest.json");
    if (this.manifest === null && existsSync(manifestPath)) {
      try {
        this.manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
      } catch {
        this.manifest = null;
      }
    }
    for (const record of this.timeline.read()) {
      const images = record?.images;
      if (!images || typeof images !== "object") continue;
      const files: Record<string, string> = {};
      for (const [camera, image] of Object.entries(images as Record<string, Json>)) {
        const match = FRAME_FILE.exec(String(image?.path ?? ""));
        if (match) files[camera] = match[1];
      }
      if (!Object.keys(files).length) continue;
      this.frames.push({
        index: Number(record.frame_index), phase: String(record.phase ?? ""), simTimeS: Number(record.sim_time_s),
        actionId: typeof record.action_id === "string" ? record.action_id : null, files,
      });
    }
  }

  /** The observation step an action (or a stage of one, `<id>:<stage>`) was proposed at. */
  private stepOf(actionId: string | null): number | null {
    if (actionId === null) return null;
    return this.actionSteps.get(actionId) ?? this.actionSteps.get(actionId.split(":")[0]) ?? null;
  }

  private cameras(): string[] {
    const listed = this.manifest?.camera_panels;
    if (Array.isArray(listed) && listed.length) return listed.map(String);
    return this.frames.length ? Object.keys(this.frames[0].files) : [];
  }

  private fps(): number | null {
    const fps = Number(this.manifest?.config?.fps);
    return Number.isFinite(fps) && fps > 0 ? fps : null;
  }

  summary(): FramesSummary {
    this.refresh();
    const steps: Record<string, number> = {};
    for (const frame of this.frames) {
      const step = this.stepOf(frame.actionId);
      if (step !== null) steps[String(step)] = (steps[String(step)] ?? 0) + 1;
    }
    const ordered = [...this.views.keys()].sort((a, b) => a - b);
    const latest = ordered.length ? this.views.get(ordered[ordered.length - 1])! : null;
    return {
      motion: this.frames.length ? { cameras: this.cameras(), fps: this.fps(), steps } : null,
      operatorView: latest === null ? null : {
        camera: latest.camera, width: latest.width, height: latest.height, steps: ordered, latest,
      },
    };
  }

  /** The frames of the action(s) proposed at observation `step`, in capture order. */
  motion(step: number): StepMotion {
    this.refresh();
    const frames = this.frames.filter((frame) => this.stepOf(frame.actionId) === step);
    return {
      step, actionIds: [...new Set(frames.map((frame) => frame.actionId!))], cameras: this.cameras(), fps: this.fps(),
      frames: frames.map((frame) => ({
        index: frame.index, phase: frame.phase, simTimeS: frame.simTimeS,
        images: Object.fromEntries(Object.entries(frame.files).map(([camera, file]) => [camera, captureUrl(file)])),
      })),
    };
  }

  /** The operator view of observation `step` (or the latest one). */
  operatorView(step: number | null = null): OperatorViewFrame | null {
    this.refresh();
    if (step !== null) return this.views.get(step) ?? null;
    const ordered = [...this.views.keys()].sort((a, b) => a - b);
    return ordered.length ? this.views.get(ordered[ordered.length - 1])! : null;
  }
}
