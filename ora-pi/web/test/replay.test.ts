// A finished run replayed from its evidence (server/replay.ts) and the page's clock over it
// (src/lib/replay.ts): every message at the moment it was appended, its images by URL only,
// what the agent was doing (from the recorded call, tool and action times), free play's
// instructions, joined messages and endings, the capture frames and operator views of each
// moment; a published replay holds neither the run's ID nor its paths; the replay server
// serves the page and its replay and refuses every command.

import { readFileSync } from "node:fs";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";
import { appendFileSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, test } from "vitest";

import { createReplayApp } from "../server/app.ts";
import { buildReplay, redactText } from "../server/replay.ts";
import { buildTimeline, countUpTo, framesSummary, momentAt, motionGroups, positionOf, ReplayState, replayOptions, stepFrames,
  worldAt, type ReplayScript } from "../src/lib/replay.ts";
import { at, MARKED, RUN_ID, replayRun, STORED, T0, UNSTORED } from "./fixtures/replay-run.ts";
import { createHash } from "node:crypto";

const sha = (data: string) => createHash("sha256").update(Buffer.from(data, "base64")).digest("hex");
const servers: Server[] = [];

afterEach(async () => {
  for (const server of servers.splice(0)) await new Promise((done) => server.close(done));
});

describe("a finished run, rebuilt from its evidence", () => {
  test("every message at the moment it was appended, its images listed by URL and never inline", async () => {
    const root = replayRun();
    const script = await buildReplay(root);
    expect(script.schema).toBe("ora.web-replay.v1");
    expect(script.runId).toBe(RUN_ID);
    expect(script.mode).toBe("play");
    expect(script.events[0].event.type).toBe("snapshot");
    expect(script.events.map((item) => item.at)).toEqual([...script.events.map((item) => item.at)].sort((a, b) => a - b));
    const messages = script.events.filter((item) => item.event.type === "message");
    expect(messages.map((item) => [item.at, item.event.message.role])).toEqual([
      [at(10), "custom"], [at(10.1), "user"], [at(14), "assistant"], [at(15), "toolResult"], [at(18), "assistant"],
      [at(18.2), "toolResult"], [at(23), "custom"], [at(23.1), "custom"], [at(26), "assistant"]]);
    const observation = messages[0].event.message;
    expect(observation.details.ora.images).toEqual([
      { camera: "head", artifact: `native/agent-artifacts/${sha(STORED)}.png`, mimeType: "image/png", inline: false,
        url: `/ora/artifacts/${sha(STORED)}.png` },
      { camera: "left_wrist", artifact: null, mimeType: "image/png", inline: false, url: null },
    ]);
    expect(messages[3].event.message.details.ora).toMatchObject({ step: 0, toolCallId: "call-1",
      images: [{ url: `/ora/artifacts/${sha(MARKED)}.png` }] });
    const json = JSON.stringify(script);
    for (const data of [STORED, UNSTORED, MARKED]) expect(json).not.toContain(data);
    expect(script.notes.join(" ")).toMatch(/1 image\(s\) of the conversation were not stored/);
    expect(script.notes.join(" ")).toMatch(/does not time its tokens/);
    expect(script.start).toBe(at(4));
    expect(script.end).toBe(at(31) + 3000);
  });

  test("what the agent was doing: thinking from each call's start, a tool from its start, the robot moving, idle", async () => {
    const script = await buildReplay(replayRun());
    const activity = script.events.filter((item) => item.event.type === "activity")
      .map((item) => [item.at - T0, item.event.activity.kind, item.event.activity.tool ?? item.event.activity.step ?? null]);
    expect(activity).toEqual([
      [10_000, "thinking", null], [11_000, "thinking", null], [14_500, "tool", "mark_point"], [16_000, "thinking", null],
      [18_150, "tool", "set_gripper"], [19_000, "moving", 0], [21_000, "observing", null], [24_000, "thinking", null],
      [27_000, "idle", null], [31_000, "idle", null]]);
    const tools = script.events.filter((item) => item.event.type === "tool").map((item) => [item.at - T0, item.event.tool.status]);
    expect(tools).toEqual([[14_500, "running"], [15_000, "done"], [18_150, "running"], [18_200, "done"]]);
  });

  test("free play: the instruction, a message typed while it ran (queued, then delivered), its end and the session's", async () => {
    const script = await buildReplay(replayRun());
    const play = script.events.filter((item) => item.event.type === "play").map((item) => [item.at - T0, item.event.play.state]);
    expect(play).toEqual([[5_000, "scene"], [10_000, "instruction"], [10_050, "message_delivered"], [19_000, "action_started"],
      [20_000, "message_queued"], [21_000, "action_finished"], [23_050, "message_delivered"], [27_000, "instruction_finished"],
      [27_000, "awaiting_instruction"], [30_000, "control"], [31_000, "ended"]]);
    const joined = script.events.filter((item) => item.event.type === "play_message" && item.event.message.id === "h002")
      .map((item) => [item.at - T0, item.event.message.state, item.event.message.instruction, item.event.message.step]);
    expect(joined).toEqual([[20_000, "queued", 1, null], [23_050, "delivered", 1, 1]]);
    expect(script.instructions).toEqual([{ index: 0, id: "h001", start: at(10), end: at(27), ended: "reported" }]);
    expect(script.actions).toEqual([{ id: "action-a", step: 0, start: at(19), end: at(21) }]);
  });

  test("the page at a moment: the recorded events up to it, applied by the live page's reducer", async () => {
    const script = await buildReplay(replayRun());
    const replay = new ReplayState(script);
    const early = replay.at(at(12));
    expect(early.loaded).toBe(true);
    expect(early.messages.map((message) => message.role)).toEqual(["custom", "user"]);
    expect(early.state).toMatchObject({ model: { provider: "ora", id: "deepseek-flash" }, thinkingLevel: "high" });
    expect(early.run.play).toMatchObject({ state: "message_delivered", running: "h001", instructions: 1 });
    expect(early.activity).toEqual({ kind: "thinking", since: at(11), tool: undefined, step: null });
    const late = replay.at(at(28));
    expect(late.messages).toHaveLength(9);
    expect(late.lastInstruction).toMatchObject({ ended: "reported" });
    expect(late.run.playEndings).toEqual([expect.objectContaining({ instruction_id: "h001", at: at(27) })]);
    expect(late.run.playMessages.map((message: { id: string; state: string }) => [message.id, message.state]))
      .toEqual([["h001", "delivered"], ["h002", "delivered"]]);
    // Back in time: the page is the page of that moment again.
    expect(replay.at(at(12)).messages).toHaveLength(2);
    expect(replay.applied).toBeLessThan(script.events.length);
  });

  test("the cameras at a moment: the capture frame taken last, the operator view when taken later, a step's motion so far", async () => {
    const script = await buildReplay(replayRun());
    expect(script.capture).toMatchObject({ cameras: ["head", "operator_camera"], fps: 10 });
    expect(script.capture!.frames.map((frame) => [frame.step, frame.images.head])).toEqual([
      [null, "/ora/capture/00000000-camera-0.png"], [0, "/ora/capture/00000001-camera-0.png"],
      [0, "/ora/capture/00000002-camera-0.png"], [0, "/ora/capture/00000003-camera-0.png"]]);
    const counts = (time: number) => [countUpTo(script.capture!.frames, time), countUpTo(script.operatorView!.frames, time)] as const;
    const [moving, seen] = counts(at(20.5));
    const world = worldAt(script, moving, seen, true)!;
    expect(world.phase).toBe("executing");
    expect(world.cameras).toEqual([{ name: "operator_camera", url: "/ora/capture/00000002-camera-3.png" },
      { name: "head", url: "/ora/capture/00000002-camera-0.png" }]);
    const [after, viewed] = counts(at(22.6));
    expect(worldAt(script, after, viewed, false)!.cameras.find((item) => item.name === "operator_camera")!.url)
      .toBe("/ora/operator-view/000001.png");
    const summary = framesSummary(script, moving, seen);
    expect(summary.motion!.steps).toEqual({ 0: 2 });
    expect(summary.operatorView!.latest.url).toBe("/ora/operator-view/000000.png");
    expect(stepFrames(script, summary, 0, at(20.5)).motion!.frames).toHaveLength(2);
    expect(worldAt(script, 0, 0, false)).toBeNull();
  });

  test("a link chooses the replay: its speed (16× by default), start, export and publishing", () => {
    expect(replayOptions("?x=1")).toBeNull();
    expect(replayOptions("?replay=1")).toEqual({ speed: 16, motion: 1, from: 0, autoplay: true, exporting: false, publish: false });
    expect(replayOptions("?replay=1&speed=32&motion=same&from=12&play=0&export=1&publish=1"))
      .toEqual({ speed: 32, motion: "same", from: 12, autoplay: false, exporting: true, publish: true });
    expect(replayOptions("?replay=1&motion=0.5")!.motion).toBe(0.5);
    expect(replayOptions("?replay=1&motion=fast")!.motion).toBe(1);
  });
});

describe("two rates: the waiting at its speed, the robot's motion by its simulated time", () => {
  test("an action's motion lasts its simulated time at the motion speed; the rest plays at the speed", async () => {
    const script = await buildReplay(replayRun());
    const timeline = buildTimeline(script, 16, 1);
    expect(timeline.segments.map((segment) => [segment.kind, segment.from - T0, segment.to - T0, segment.playTo - segment.playFrom]))
      .toEqual([["wait", 4000, 19_000, 937.5], ["motion", 19_000, 21_000, 200], ["wait", 21_000, 34_000, 812.5]]);
    expect(timeline.segments[1]).toMatchObject({ frames: [1, 4], simFrom: 0.1, simTo: 0.3, action: "action-a", step: 0 });
    expect(timeline.duration).toBe(1950);
    // One speed throughout keeps the old pace; a fast motion speed never plays shorter than the action at the speed.
    expect(buildTimeline(script, 16, "same").segments.map((segment) => segment.kind)).toEqual(["wait"]);
    expect(buildTimeline(script, 16, "same").duration).toBe(30_000 / 16);
    expect(buildTimeline(script, 16, 4).segments[1].playTo - buildTimeline(script, 16, 4).segments[1].playFrom).toBe(125);
  });

  test("in a motion the frame on screen is the last one taken by then, held: nothing in between is made up", async () => {
    const script = await buildReplay(replayRun());
    const timeline = buildTimeline(script, 16, 1);
    const at_ = (ms: number) => momentAt(script, timeline, 937.5 + ms);
    expect([at_(50).captured, at_(50).wall - T0]).toEqual([2, 19_500]);
    expect(at_(50).sim).toBeCloseTo(0.15);
    expect(at_(150).captured).toBe(3);
    expect(at_(199).captured).toBe(3);
    expect(at_(199).sim).toBeCloseTo(0.299);
    expect(momentAt(script, timeline, 1500)).toMatchObject({ captured: null, sim: null });
    expect(positionOf(timeline, at(20))).toBe(1037.5);
    expect(momentAt(script, timeline, positionOf(timeline, at(20))).wall).toBe(at(20));
    expect(positionOf(timeline, at(30))).toBeCloseTo(937.5 + 200 + 9000 / 16);
  });

  test("settling frames after an action belong to its motion; the scene's reset and close do not", async () => {
    const script = structuredClone(await buildReplay(replayRun())) as ReplayScript;
    const frames = script.capture!.frames;
    frames.splice(4, 0, { ...frames[3], index: 4, at: at(21.3), phase: "settle", actionId: null, simTimeS: 0.5 },
      { ...frames[3], index: 5, at: at(21.6), phase: "settle", actionId: null, simTimeS: 0.7 });
    frames.push({ ...frames[0], index: 6, at: at(31), phase: "close", simTimeS: 0.7 });
    expect(motionGroups(script).map((group) => [group.first, group.last, group.simFrom, group.simTo, group.to - T0]))
      .toEqual([[1, 6, 0.1, 0.7, 21_600]]);
  });
});

describe("cameras taken at their own rates and sizes", () => {
  test("each camera keeps its newest frame; a sharper operator camera is preferred, the usual one stands in", async () => {
    const script = structuredClone(await buildReplay(replayRun())) as ReplayScript;
    const capture = script.capture!;
    capture.sizes = { head: [640, 480], operator_camera: [640, 480], operator_camera_hd: [1280, 720] };
    // Two frames of the sharp operator camera alone, between the robot cameras' frames.
    capture.frames.splice(3, 0, { ...capture.frames[2], index: 30, at: at(20.3), simTimeS: 0.23,
      images: { operator_camera_hd: "/ora/capture/00000030-camera-4.png" } });
    const named = (world: ReturnType<typeof worldAt>) => Object.fromEntries(world!.cameras.map((item) => [item.name, item.url]));
    expect(named(worldAt(script, 3, 1, true))).toEqual({ operator_camera: "/ora/capture/00000002-camera-3.png",
      head: "/ora/capture/00000002-camera-0.png" });
    expect(named(worldAt(script, 4, 1, true))).toEqual({ operator_camera: "/ora/capture/00000030-camera-4.png",
      head: "/ora/capture/00000002-camera-0.png" });
    // The sharp camera stays the operator's while the usual one's newer frame is within a second of it.
    expect(named(worldAt(script, 5, 1, true)).operator_camera).toBe("/ora/capture/00000030-camera-4.png");
    // A 640x480 operator view taken later than the sharp frame does not replace it; without sharp frames it would.
    expect(named(worldAt(script, 5, 2, false)).operator_camera).toBe("/ora/capture/00000030-camera-4.png");
    const plain = await buildReplay(replayRun());
    expect(named(worldAt(plain, 4, 2, false)).operator_camera).toBe("/ora/operator-view/000001.png");
  });
});

describe("a session that branched", () => {
  test("gives the conversation along the new branch, as Pi's get_messages does then", async () => {
    const root = replayRun();
    const session = join(root, "native", "pi-session", "2026-10-20T00-00-00-000Z_session.jsonl");
    appendFileSync(session, `${JSON.stringify({ type: "message", id: "e14", parentId: "e5", timestamp: new Date(at(28)).toISOString(),
      message: { role: "user", timestamp: at(28), content: [{ type: "text", text: "Start over from the first step." }] } })}\n`);
    const script = await buildReplay(root);
    const branch = script.events.find((item) => item.event.type === "messages")!;
    expect(branch.at).toBe(at(28));
    expect(branch.event.messages.map((message: { role: string }) => message.role)).toEqual(["custom", "user", "user"]);
    expect(branch.event.messages[0].details.ora.images[0].url).toMatch(/^\/ora\/artifacts\//);
    expect(new ReplayState(script).at(at(29)).messages).toHaveLength(3);
  });
});

describe("a published replay", () => {
  test("holds neither the run's ID nor its paths; image URLs stay", async () => {
    const root = replayRun();
    const script = await buildReplay(root, { publish: true });
    expect(script.runId).toBeNull();
    expect(script.publish).toBe(true);
    const json = JSON.stringify(script);
    expect(json).not.toContain(RUN_ID);
    expect(json).not.toContain(root);
    const thinking = script.events.find((item) => item.event.type === "message" && item.event.message.role === "assistant")!
      .event.message.content[0].thinking;
    expect(thinking).toBe("The blocks are by the car. My workspace is …/inputs.");
    expect(json).toContain(`/ora/artifacts/${sha(STORED)}.png`);
    expect(redactText("see /srv/ora/runs/x/native/a.png and http://127.0.0.1:8080/api/x; /quit", null))
      .toBe("see …/a.png and http://127.0.0.1:8080/api/x; /quit");
  });
});

describe("the replay server", () => {
  test("serves the page and its replay; every command is refused, nothing is attached", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ora-replay-app-"));
    const staticDir = join(dir, "dist");
    mkdirSync(staticDir);
    writeFileSync(join(staticDir, "index.html"), "<!doctype html><title>ORA</title>");
    const script = await buildReplay(replayRun());
    const server = createReplayApp({ script, staticDir });
    servers.push(server);
    await new Promise<void>((done) => server.listen(0, "127.0.0.1", () => done()));
    const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    expect(await (await fetch(`${base}/`)).text()).toContain("<title>ORA</title>");
    const replay = await (await fetch(`${base}/api/replay`)).json();
    expect(replay.events).toHaveLength(script.events.length);
    for (const [method, path] of [["POST", "/api/send"], ["GET", "/api/events"], ["GET", "/api/snapshot"], ["POST", "/api/play"],
      ["GET", "/api/frames"]]) {
      const response = await fetch(`${base}${path}`, method === "POST" ? { method, body: "{}" } : {});
      expect(response.status).toBe(410);
      expect((await response.json()).error).toMatch(/recorded run takes no commands/);
    }
    expect(readFileSync(join(staticDir, "index.html"), "utf8")).toContain("ORA");
  });
});
