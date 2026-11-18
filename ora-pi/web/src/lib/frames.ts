// The run's recorded motion and operator view (server: GET /api/frames, the agent process's
// ora_get_frames, ora-pi/src/rpc/frames.ts). Read-only: the PNGs are the run's own files,
// served by the `ora web` gateway at /ora/capture/... and /ora/operator-view/....

import { useEffect, useRef, useState } from "react";

import { call } from "./api.ts";

type Json = any;

export interface OperatorViewFrame {
  step: number;
  camera: string;
  width: number;
  height: number;
  url: string;
}

export interface FramesSummary {
  motion: { cameras: string[]; fps: number | null; steps: Record<string, number> } | null;
  operatorView: { camera: string; width: number; height: number; steps: number[]; latest: OperatorViewFrame } | null;
}

export interface MotionFrame {
  index: number;
  phase: string;
  simTimeS: number;
  images: Record<string, string>;
}

export interface StepMotion {
  step: number;
  actionIds: string[];
  cameras: string[];
  fps: number | null;
  frames: MotionFrame[];
}

export interface StepFrames {
  summary: FramesSummary;
  motion: StepMotion | null;
  operatorView: OperatorViewFrame | null;
}

export const NO_FRAMES: FramesSummary = { motion: null, operatorView: null };

function summaryOf(data: Json): FramesSummary {
  return { motion: data?.motion ?? null, operatorView: data?.operatorView ?? null };
}

/** The summary, asked again whenever `version` changes (a new observation, a finished action). */
export function useFramesSummary(version: number, enabled: boolean): FramesSummary {
  const [summary, setSummary] = useState<FramesSummary>(NO_FRAMES);
  useEffect(() => {
    if (!enabled) return;
    let stopped = false;
    // A short delay lets the Supervisor finish writing what the event announced.
    const timer = setTimeout(() => {
      call("/api/frames").then((data) => {
        if (!stopped) setSummary(summaryOf(data));
      }).catch(() => undefined);
    }, 150);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [version, enabled]);
  return summary;
}

/** One step's motion and operator view; polled every `pollMs` while `poll` (the robot moving now). */
export function useStepFrames(step: number | null, version: number, poll: boolean, pollMs = 400): StepFrames | null {
  const [frames, setFrames] = useState<StepFrames | null>(null);
  const asked = useRef<number | null>(null);
  useEffect(() => {
    if (step === null) {
      setFrames(null);
      return;
    }
    if (asked.current !== step) setFrames(null);
    asked.current = step;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const load = () => {
      call(`/api/frames?step=${step}`).then((data) => {
        if (stopped) return;
        setFrames({ summary: summaryOf(data), motion: data?.step?.motion ?? null, operatorView: data?.step?.operatorView ?? null });
      }).catch(() => undefined).finally(() => {
        if (!stopped && poll) timer = setTimeout(load, pollMs);
      });
    };
    load();
    return () => {
      stopped = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [step, version, poll, pollMs]);
  return frames;
}

/** The operator view's shape (width / height), or null before the run has one. */
export function operatorAspect(summary: FramesSummary): number | null {
  const view = summary.operatorView;
  return view && view.width > 0 && view.height > 0 ? view.width / view.height : null;
}

/** The steps that ran a recorded action, in order. */
export function motionSteps(summary: FramesSummary): number[] {
  return Object.keys(summary.motion?.steps ?? {}).map(Number).filter(Number.isInteger).sort((a, b) => a - b);
}

/** The camera a motion is shown through: `preferred` when the capture has it, else the first that has frames. */
export function motionCamera(motion: StepMotion | null, preferred: (string | null)[]): string | null {
  if (!motion?.frames.length) return null;
  const has = (camera: string | null) => camera !== null && motion.frames.some((frame) => camera in frame.images);
  return preferred.find(has) ?? Object.keys(motion.frames[0].images)[0] ?? null;
}

/** The operator view at `step`: that step's, else the latest before it. */
export function operatorViewAt(summary: FramesSummary, step: number | null): OperatorViewFrame | null {
  const view = summary.operatorView;
  if (view === null) return null;
  if (step === null) return view.latest;
  const at = [...view.steps].reverse().find((item) => item <= step);
  if (at === undefined) return null;
  return { ...view.latest, step: at, url: `/ora/operator-view/${String(at).padStart(6, "0")}.png` };
}
