// The replay on a real simulation-video capture (fixtures/simvideo-run.ts): every frame with its
// simulated time, the operator camera at 1280x720 preferred, the settling window after each
// action marked from the run's post-action settle records, and an action's end and the next
// one's start (one simulated moment) counted once.

import { describe, expect, test } from "vitest";

import { buildReplay } from "../server/replay.ts";
import { buildTimeline, cameraAspect, framesSummary, momentAt, stepFrames, worldAt, type ReplayScript } from "../src/lib/replay.ts";
import { replayRun } from "./fixtures/replay-run.ts";
import { simvideoRun } from "./fixtures/simvideo-run.ts";

describe("the camera box's shape", () => {
  test("the run's sharpest operator camera's, else its operator views', else 4:3", async () => {
    expect(cameraAspect(await buildReplay(simvideoRun()))).toBeCloseTo(16 / 9);
    const old = await buildReplay(replayRun());
    expect(cameraAspect(old)).toBeCloseTo(4 / 3);
    const viewsOnly = structuredClone(old) as ReplayScript;
    viewsOnly.capture = null;
    viewsOnly.operatorView = { ...viewsOnly.operatorView!, width: 1280, height: 720 };
    expect(cameraAspect(viewsOnly)).toBeCloseTo(16 / 9);
    viewsOnly.operatorView = null;
    expect(cameraAspect(viewsOnly)).toBeCloseTo(4 / 3);
  });
});

describe("a simulation-video capture", () => {
  test("its frames, cameras' sizes and settling windows as the capture recorded them", async () => {
    const script = await buildReplay(simvideoRun());
    const capture = script.capture!;
    expect(capture.fps).toBe(30);
    expect(capture.sizes).toEqual({ head: [640, 480], left_wrist: [640, 480], operator_camera: [1280, 720], right_wrist: [640, 480] });
    expect(capture.frames).toHaveLength(39);
    const sweep = capture.frames.filter((frame) => frame.actionId === "simvideo-sweep");
    const settling = sweep.filter((frame) => frame.settle);
    // 130 settle steps from physics step 190: the frames taken after it, through the action's end.
    expect(settling.map((frame) => frame.simStep)).toEqual([194, 203, 212, 221, 230, 239, 248, 257, 266, 275, 284, 293, 302, 311,
      320, 320]);
    expect(sweep.filter((frame) => !frame.settle).every((frame) => frame.simStep! <= 190)).toBe(true);
    expect(capture.frames.filter((frame) => frame.actionId === "simvideo-start").some((frame) => frame.settle)).toBe(false);
    const record = capture.settles.find((item) => item.actionId === "simvideo-sweep")!;
    expect([record.stepFrom, record.steps, record.ended]).toEqual([190, 130, "at_rest"]);
    expect(record.simFrom).toBeCloseTo(0.76);
    expect(record.simTo).toBeCloseTo(1.28);
    expect(script.sources).toContain("native/post-action-settle.jsonl");
    expect(script.start).toBe(capture.frames[0].at);
  });

  test("each action plays its simulated time once: the moment an action ends and the next starts is not counted twice", async () => {
    const script = await buildReplay(simvideoRun());
    const timeline = buildTimeline(script, 16, 1);
    const motions = timeline.segments.filter((segment) => segment.kind === "motion");
    expect(motions.map((segment) => [segment.action, segment.simFrom, segment.simTo])).toEqual([
      ["simvideo-start", 0.44, 0.56], ["simvideo-sweep", 0.56, 1.28], ["simvideo-retreat", 1.28, 1.6]]);
    const lengths = motions.map((segment) => segment.playTo - segment.playFrom);
    expect(lengths.map(Math.round)).toEqual([120, 720, 320]);
    expect(lengths.reduce((sum, value) => sum + value, 0)).toBeCloseTo((1.6 - 0.44) * 1000);
    expect(motions[1]).toMatchObject({ settleTo: expect.closeTo(1.28), settleEnded: "at_rest" });
    expect(motions[1].settleFrom).toBeCloseTo(0.76);
  });

  test("the frame on screen is the one taken by then, the sharp operator camera's, marked settling in the window", async () => {
    const script = await buildReplay(simvideoRun());
    const timeline = buildTimeline(script, 16, 1);
    const sweep = timeline.segments.find((segment) => segment.action === "simvideo-sweep")!;
    const at = (sim: number) => momentAt(script, timeline, sweep.playFrom + (sim - 0.56) * 1000);
    const frames = script.capture!.frames;
    const moving = at(0.7);
    expect(frames[moving.captured! - 1].simTimeS).toBeCloseTo(0.668);
    expect(moving.settling).toBe(false);
    const settling = at(0.9);
    expect(frames[settling.captured! - 1].simStep).toBe(221);
    expect(settling.settling).toBe(true);
    const world = worldAt(script, settling.captured!, 0, true, settling.settling)!;
    expect(world.phase).toBe("settling");
    expect(world.cameras[0]).toEqual({ name: "operator_camera", url: "/ora/capture/00000029-camera-3.png" });
    expect(world.cameras.map((camera) => camera.name)).toEqual(["operator_camera", "head", "left_wrist", "right_wrist"]);
    // The retreat starts on the moment the sweep ended: its first frame, no time added before it.
    const retreat = timeline.segments.find((segment) => segment.action === "simvideo-retreat")!;
    const first = momentAt(script, timeline, retreat.playFrom);
    expect([frames[first.captured! - 1].phase, frames[first.captured! - 1].simTimeS]).toEqual(["action_start", 1.28]);
  });

  test("a step's motion holds one frame per simulated moment", async () => {
    const script = structuredClone(await buildReplay(simvideoRun())) as ReplayScript;
    for (const frame of script.capture!.frames) if (frame.actionId === "simvideo-sweep") frame.step = 7;
    const summary = framesSummary(script, script.capture!.frames.length, 0);
    const motion = stepFrames(script, summary, 7, Infinity).motion!;
    const times = motion.frames.map((frame) => frame.simTimeS);
    expect(times).toHaveLength(script.capture!.frames.filter((frame) => frame.actionId === "simvideo-sweep").length - 1);
    expect(times.every((time, index) => index === 0 || time > times[index - 1])).toBe(true);
    expect(motion.frames.at(-1)!.phase).toBe("action_end");
  });
});
