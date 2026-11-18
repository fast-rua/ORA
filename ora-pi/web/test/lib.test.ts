// The page's data: the conversation and its steps, the perception overlays and
// their exact coordinates, free play's phases, the model's text as safe
// Markdown, and what the composer sends for what is typed.

import { describe, expect, test } from "vitest";

import { allows, outgoing } from "../src/lib/composer.ts";
import { renderMarkdown } from "../src/lib/markdown.ts";
import { buildConversation, cameraLabel, cameraOf, imagesOf, stepImage, toolCamera } from "../src/lib/messages.ts";
import { containRect, layout, MARK_COLOURS, overlayFor, toolKind } from "../src/lib/overlays.ts";
import { motionCamera, motionSteps, operatorViewAt } from "../src/lib/frames.ts";
import { browserLang, lang, seconds, setLang, t } from "../src/lib/i18n.ts";
import { activityFrom, activityText, endingKind, endingText, playPhase, playStatus } from "../src/lib/play.ts";

const png = (n: number) => ({ type: "image", mimeType: "image/png", data: `AAA${n}` });

// An episode, as the agent process completes it (rpc/images.ts): images inline for recent steps, else by URL.
const episode = [
  { role: "custom", customType: "ora.observation", content: [{ type: "text", text: "observation 3" }, png(1), png(2)], display: true,
    timestamp: 10, details: { ora_observation_step: 3, ora: { step: 3, camera: "agentview", images: [
      { camera: "agentview", inline: true, url: "/ora/artifacts/a.png" }, { camera: "wrist", inline: true, url: "/ora/artifacts/b.png" }] } } },
  { role: "assistant", timestamp: 11, content: [{ type: "text", text: "Let me find the basket." },
    { type: "toolCall", id: "call-g", name: "vision_ground", arguments: { camera: "agentview", text: "woven basket" } }] },
  { role: "toolResult", toolCallId: "call-g", toolName: "vision_ground", timestamp: 12,
    content: [{ type: "text", text: JSON.stringify({ camera: "agentview", overlay: { width: 256, height: 256, image_index: 0 },
      candidates: [{ candidate_id: "T0", rank: 0, bbox_xyxy: [180, 105, 240, 161], backend_score: 0.73 },
        { candidate_id: "T1", rank: 1, bbox_xyxy: [180, 117, 239, 161], backend_score: 0.41 }] }) }],
    details: { ora: { step: 3, toolCallId: "call-g", images: [{ camera: null, inline: false, url: "/ora/artifacts/o.png" }] } } },
  { role: "custom", customType: "ora.observation", content: "observation 4", display: true, timestamp: 13,
    details: { ora_observation_step: 4, ora: { step: 4, images: [{ camera: "agentview", inline: false, url: "/ora/artifacts/c.png" }] } } },
  { role: "assistant", timestamp: 14, content: [{ type: "toolCall", id: "call-m", name: "perception.mark_point",
    arguments: { point_id: "handle", view: "agentview", x: 100, y: 50 } }] },
];

// Free play, as Pi keeps it (images in the message under their captions), before completion.
const play = [
  { role: "custom", customType: "ora.play_observation", display: true, timestamp: 1, details: { ora_observation_step: 0 },
    content: [{ type: "text", text: "{\"current_actual\": {}}" }, { type: "text", text: "INITIAL ACTUAL · sequence 0 · agentview · latest at this message" },
      png(5), { type: "text", text: "INITIAL ACTUAL · sequence 0 · wrist · latest at this message" }, png(6)] },
  { role: "user", timestamp: 2, content: [{ type: "text", text: "put the basket on the plate" }] },
];

describe("the conversation and its steps", () => {
  test("an episode: observations, tool cards with their step and images, the steps indexed", () => {
    const conversation = buildConversation(episode);
    expect(conversation.items.map((item) => item.kind)).toEqual(["observation", "assistant", "observation", "assistant"]);
    expect(conversation.order).toEqual([3, 4]);
    const step = conversation.steps.get(3)!;
    expect(step.observation!.images.map((image) => [image.camera, image.src])).toEqual([
      ["agentview", "data:image/png;base64,AAA1"], ["wrist", "data:image/png;base64,AAA2"]]);
    expect(conversation.steps.get(4)!.observation!.images).toEqual([{ camera: "agentview", src: "/ora/artifacts/c.png" }]);
    const ground = conversation.tools.get("call-g")!;
    expect(ground).toMatchObject({ step: 3, status: "done", name: "vision_ground" });
    expect(ground.images).toEqual([{ camera: null, src: "/ora/artifacts/o.png" }]);
    expect(ground.overlay!.boxes.map((box) => box.label)).toEqual(["T0", "T1"]);
    expect(step.tools.map((tool) => tool.id)).toEqual(["call-g"]);
    const mark = conversation.tools.get("call-m")!;
    expect(mark).toMatchObject({ step: 4, status: "pending" });
    expect(toolCamera(mark)).toBe("agentview");
    expect(stepImage(conversation.steps.get(4), "agentview")!.src).toBe("/ora/artifacts/c.png");
  });

  test("free play: cameras from the captions, the instruction as the user's", () => {
    const conversation = buildConversation(play);
    const observation = conversation.steps.get(0)!.observation!;
    expect(observation.play).toBe(true);
    expect(observation.images.map((image) => image.camera)).toEqual(["agentview", "wrist"]);
    expect(observation.text).toContain("current_actual");
    expect(conversation.items[1]).toMatchObject({ kind: "user", text: "put the basket on the plate" });
  });

  test("the reply being written and running tools; what pages sent that the conversation lacks", () => {
    const streaming = { role: "assistant", content: [{ type: "toolCall", id: "call-x", name: "move_to", arguments: {} }] };
    const sent = [{ text: "put the basket on the plate", as: "prompt", at: 1, ok: true },
      { text: "keep it level", as: "steer", at: 3, ok: true }, { text: "/stop", as: "prompt", at: 4, ok: true }];
    const conversation = buildConversation(play, streaming, new Map([["call-x", { status: "running" }]]), sent);
    const last = conversation.items[conversation.items.length - 2];
    expect(last).toMatchObject({ kind: "assistant", streaming: true });
    expect((last as any).tools[0].status).toBe("running");
    const pending = conversation.items.filter((item) => item.kind === "sent");
    expect(pending.map((item) => (item as any).text)).toEqual(["keep it level"]);
  });

  test("caption to camera; an image listed by URL only", () => {
    expect(cameraOf("ACTUAL · sequence 12 · wrist · latest at this message")).toBe("wrist");
    expect(cameraOf("agentview")).toBe("agentview");
    expect(cameraLabel("front_camera")).toBe("front_camera");
    expect(cameraLabel("GRASP BASELINE · after close, before lift · head_camera · historical intermediate frame"))
      .toBe("head_camera · grasp baseline");
    expect(imagesOf({ content: "x", details: { ora: { images: [{ camera: "front", inline: false, url: "/ora/artifacts/f.png" }] } } }))
      .toEqual([{ camera: "front", src: "/ora/artifacts/f.png" }]);
  });
});

describe("perception overlays", () => {
  test("which tools draw", () => {
    expect(toolKind("vision_ground")).toBe("grounding");
    expect(toolKind("perception.vision_segment")).toBe("grounding");
    expect(toolKind("perception_mark_point")).toBe("mark");
    expect(toolKind("move_to")).toBeNull();
    expect(overlayFor("vision_ground", {}, { candidates: [{ bbox_xyxy: [0, 0, 1, 1] }] }, "error")).toBeNull();
  });

  test("grounding boxes in the grounded image's frame; marks at their pixels, batched in mark_point's colours", () => {
    const boxes = overlayFor("vision_ground", { camera: "agentview", text: "basket" },
      { camera: "agentview", candidate_previews: [{ source_dimensions: [320, 240] }],
        candidates: [{ rank: 0, bbox_xyxy: [10, 20, 30, 40], backend_score: 0.5 }] })!;
    expect(boxes).toMatchObject({ kind: "grounding", camera: "agentview", frame: { width: 320, height: 240 } });
    expect(boxes.boxes[0]).toMatchObject({ label: "T0", x0: 10, y0: 20, x1: 30, y1: 40, score: 0.5 });
    const batch = overlayFor("mark_point", { view: "wrist", point_ids: ["a", "b"], points_uv: [[5, 6], [7, 8]] },
      { marks: [{ status: "solved" }, { status: "failed" }] })!;
    expect(batch.camera).toBe("wrist");
    expect(batch.frame).toBeNull();
    expect(batch.points.map((point) => [point.label, point.x, point.y, point.color, point.status])).toEqual([
      ["a", 5, 6, MARK_COLOURS[0], "solved"], ["b", 7, 8, MARK_COLOURS[1], "failed"]]);
    expect(overlayFor("mark_point", { point_id: "p", view: "candidate_agentview", x: 1, y: 2 }, null)!.camera).toBe("agentview");
  });

  test("original coordinates land exactly on the drawn image (object-fit: contain)", () => {
    expect(containRect({ width: 256, height: 256 }, { width: 512, height: 384 })).toEqual({ x: 64, y: 0, width: 384, height: 384 });
    expect(containRect({ width: 640, height: 480 }, { width: 320, height: 400 })).toEqual({ x: 0, y: 80, width: 320, height: 240 });
    const ground = overlayFor("vision_ground", {}, { camera: "agentview", overlay: { width: 256, height: 256 },
      candidates: [{ candidate_id: "T0", bbox_xyxy: [180, 105, 240, 161] }] })!;
    // The same frame as the image, drawn 1.5x in a 512x384 stage.
    expect(layout(ground, { width: 256, height: 256 }, { width: 512, height: 384 }).boxes[0])
      .toMatchObject({ left: 64 + 180 * 1.5, top: 105 * 1.5, right: 64 + 240 * 1.5, bottom: 161 * 1.5 });
    // The boxes' frame is 256 px; the image shown is a 512 px version of the same view.
    expect(layout(ground, { width: 512, height: 512 }, { width: 512, height: 384 }).boxes[0])
      .toMatchObject({ left: 64 + 180 * 1.5, top: 105 * 1.5 });
    const mark = overlayFor("mark_point", { point_id: "p", view: "agentview", x: 100, y: 50 }, { status: "solved" })!;
    // A point at its pixel's centre.
    expect(layout(mark, { width: 256, height: 256 }, { width: 512, height: 384 }).points[0])
      .toMatchObject({ cx: 64 + 100.5 * 1.5, cy: 50.5 * 1.5 });
  });
});

describe("free play's phase", () => {
  test("from the Supervisor's announcements, never from whether the model is busy", () => {
    expect(playPhase(null)).toBeNull();
    expect(playPhase({ state: "awaiting_instruction", running: null })).toBe("awaiting_instruction");
    expect(playPhase({ state: "instruction", running: "op-2", instructions: 2 })).toBe("running");
    expect(playPhase({ state: "instruction_finished", running: null })).toBe("awaiting_instruction");
    expect(playPhase({ state: "ended", running: null })).toBe("ended");
    const running = playStatus({ state: "control", running: "op-2", instructions: 2, scene: 1, scenes: 2 }, null);
    expect(running.label).toBe("Instruction 2 running");
    expect(JSON.stringify(running)).not.toMatch(/complete/i);
    const waiting = playStatus({ state: "instruction_finished", running: null, instructions: 2 },
      { index: 1, ended: "instruction_model_call_budget" });
    expect(waiting.label).toBe("Waiting for your instruction");
    expect(waiting.detail).toBe("Instruction 2 stopped: its model calls ran out");
  });

  test("how an instruction ended: reported, stopped, budget", () => {
    expect(endingKind("reported")).toBe("reported");
    expect(endingKind("operator_stopped")).toBe("stopped");
    expect(endingKind("superseded")).toBe("stopped");
    expect(endingKind("instruction_wall_time_budget")).toBe("budget");
    expect(endingKind("session_model_call_budget")).toBe("budget");
    expect(endingKind("scene_ended")).toBe("ended");
  });
});

describe("the model's text", () => {
  test("Markdown, never HTML; links only to http(s) and mailto; no image loads", () => {
    const html = renderMarkdown("**bold** <script>alert(1)</script> <img src=x onerror=alert(1)>\n\n"
      + "[ok](https://example.com) [bad](javascript:alert(1)) ![pic](https://example.com/x.png)");
    expect(html).toContain("<strong>bold</strong>");
    expect(html).not.toMatch(/<script|<img|onerror=alert\(1\)>|javascript:/);
    expect(html).toContain("&lt;script&gt;");
    expect(html).toContain('<a href="https://example.com/" target="_blank" rel="noopener noreferrer nofollow">ok</a>');
    expect(html).toContain("bad");
  });
});

describe("what the composer sends", () => {
  test("prompt, steer or follow-up; free play's controls; no shell, no unknown command", () => {
    expect(outgoing("go", { play: false, streaming: false })).toEqual({ kind: "message", text: "go", as: "prompt" });
    expect(outgoing("slower", { play: false, streaming: true })).toEqual({ kind: "message", text: "slower", as: "steer" });
    expect(outgoing("then stop", { play: true, streaming: true, followUp: true })).toMatchObject({ as: "follow_up" });
    expect(outgoing("/reset", { play: true, streaming: false })).toEqual({ kind: "control", control: "reset" });
    expect(outgoing("/quit", { play: true, streaming: true })).toEqual({ kind: "control", control: "quit" });
    expect(outgoing("/reset", { play: false, streaming: false })).toMatchObject({ kind: "refused" });
    expect(outgoing("/stop", { play: false, streaming: true })).toEqual({ kind: "message", text: "/stop", as: "prompt" });
    expect(outgoing("/model x", { play: true, streaming: false })).toMatchObject({ kind: "refused" });
    expect(outgoing("!ls", { play: true, streaming: false })).toMatchObject({ kind: "refused" });
    expect(outgoing("   ", { play: true, streaming: false })).toBeNull();
    expect(allows({ refused: { compact: "no" } }, "compact")).toBe(false);
    expect(allows({ refused: { compact: "no" } }, "set_model")).toBe(true);
    expect(allows(null, "set_model")).toBe(false);
  });
});

// Free play with a message that joined the running instruction, as the agent process keeps it.
const joinedPlay = [
  { role: "custom", customType: "ora.play_observation", display: true, timestamp: 1000, details: { ora_observation_step: 3 },
    content: [{ type: "text", text: "{}" }] },
  { role: "user", timestamp: 1001, content: [{ type: "text", text: "put the cup on the plate" }] },
  { role: "assistant", timestamp: 1002, content: [{ type: "text", text: "Moving." },
    { type: "toolCall", id: "call-1", name: "robot_submit_action", arguments: {} }] },
  { role: "toolResult", toolCallId: "call-1", toolName: "robot_submit_action", timestamp: 5000, content: [{ type: "text", text: "{}" }],
    details: { ora: { step: 3, images: [] } } },
  { role: "custom", customType: "ora.play_observation", display: true, timestamp: 7500, details: { ora_observation_step: 4 },
    content: [{ type: "text", text: "{}" }] },
  { role: "custom", customType: "ora.play_message", display: true, timestamp: 7501, content: "how far along are you?",
    details: { ora_message_id: "m1", ora_instruction: 1, ora_observation_step: 4, ora_joined: true } },
  { role: "assistant", timestamp: 7600, content: [{ type: "text", text: "Halfway; I keep going." },
    { type: "toolCall", id: "call-2", name: "robot_submit_action", arguments: {} }] },
  { role: "assistant", timestamp: 9000, content: [{ type: "text", text: "Done." }] },
  { role: "custom", customType: "ora.play_observation", display: true, timestamp: 20000, details: { ora_observation_step: 5 },
    content: [{ type: "text", text: "{}" }] },
  { role: "user", timestamp: 20001, content: [{ type: "text", text: "now the bowl" }] },
];

describe("free play: messages joined to an instruction, answers, where instructions ended", () => {
  const context = {
    messages: [
      { id: "i1", text: "put the cup on the plate", state: "delivered", joined: false, instruction: 1, step: 3 },
      { id: "m1", text: "how far along are you?", state: "delivered", joined: true, instruction: 1, step: 4 },
      { id: "i2", text: "now the bowl", state: "delivered", joined: false, instruction: 2, step: 5 },
      { id: "m2", text: "use the left arm?", state: "queued", joined: true, instruction: 2, step: null },
    ],
    endings: [{ index: 0, instruction_id: "i1", ended: "operator_stopped", observation_steps: { start: 3, end: 5 },
      wall_duration_s: 18.4, decisions: 2, at: 15000 }],
    moves: new Map([[4, 1.25]]),
  };

  test("the instruction is numbered, the message joins it with its step, the next words are its answer", () => {
    const conversation = buildConversation(joinedPlay, null, new Map(), [], context);
    const kinds = conversation.items.map((item) => item.kind);
    expect(kinds).toEqual(["observation", "user", "assistant", "observation", "joined", "assistant", "assistant", "ended",
      "observation", "user", "joined"]);
    expect(conversation.items[1]).toMatchObject({ kind: "user", instruction: 1 });
    expect(conversation.items[4]).toMatchObject({ kind: "joined", id: "m1", instruction: 1, step: 4, state: "delivered" });
    expect(conversation.items[5]).toMatchObject({ kind: "assistant", answers: conversation.items[4].key });
    expect(conversation.items[6]).toMatchObject({ kind: "assistant", answers: null }, );
    expect(conversation.items[7]).toMatchObject({ kind: "ended", n: 1, ended: "operator_stopped", step: 5, decisions: 2 });
    expect(conversation.items[9]).toMatchObject({ kind: "user", instruction: 2 });
    expect(conversation.items[10]).toMatchObject({ kind: "joined", id: "m2", state: "queued", instruction: 2 });
  });

  test("each step's thinking and moving time, the measured move first", () => {
    const conversation = buildConversation(joinedPlay, null, new Map(), [], context);
    expect(conversation.steps.get(3)).toMatchObject({ think: 4, move: 2.5 });
    expect(conversation.steps.get(4)!.move).toBe(1.25);
  });

  test("a message sent from this page is not shown twice once the agent process knows it", () => {
    const sent = [{ text: "use the left arm?", as: "steer", at: 1, ok: true }];
    const conversation = buildConversation(joinedPlay, null, new Map(), sent, context);
    expect(conversation.items.filter((item) => item.kind === "sent")).toEqual([]);
  });
});

describe("the page's language and live status", () => {
  test("English or Chinese, from the browser unless chosen", () => {
    expect(browserLang(["zh-CN", "en"])).toBe("zh");
    expect(browserLang(["en-GB"])).toBe("en");
    const before = lang();
    setLang("zh", false);
    try {
      expect(t("status.waiting")).toBe("等待你的指令");
      expect(endingText("operator_stopped")).toBe("已停止（Esc）");
      expect(playStatus({ state: "instruction", running: "i2", instructions: 2, scene: 1, scenes: 3 }, null))
        .toMatchObject({ label: "第 2 条指令执行中", detail: "场景 1/3" });
      expect(seconds(65_000)).toBe("1 分 05 秒");
    } finally {
      setLang(before, false);
    }
    expect(t("chat.joins", { n: 2 })).toBe("Joins instruction 2");
  });

  test("what the agent does now, counted on the page's clock", () => {
    const activity = activityFrom({ kind: "thinking", elapsedMs: 4200 }, 10_000)!;
    expect(activity.since).toBe(5800);
    expect(activityText(activity, 10_000)).toBe("Thinking… 4 s");
    expect(activityText({ kind: "moving", since: 0 }, 2500)).toBe("Robot moving… 2 s");
    expect(activityText({ kind: "tool", since: 0, tool: "vision_ground" }, 1000)).toBe("Running vision_ground… 1 s");
    expect(activityText({ kind: "idle", since: 0 }, 1000)).toBeNull();
    expect(activityFrom(null)).toBeNull();
  });
});

describe("recorded motion and the operator view", () => {
  const summary = { motion: { cameras: ["head_camera"], fps: 10, steps: { 4: 3, 2: 5 } },
    operatorView: { camera: "observer_camera", width: 64, height: 48, steps: [0, 2, 4],
      latest: { step: 4, camera: "observer_camera", width: 64, height: 48, url: "/ora/operator-view/000004.png" } } };

  test("the steps with motion; the view at a step or the latest before it", () => {
    expect(motionSteps(summary)).toEqual([2, 4]);
    expect(operatorViewAt(summary, 3)!.url).toBe("/ora/operator-view/000002.png");
    expect(operatorViewAt(summary, null)!.url).toBe("/ora/operator-view/000004.png");
    expect(operatorViewAt({ motion: null, operatorView: null }, 3)).toBeNull();
    const motion = { step: 4, actionIds: ["a"], cameras: ["head_camera", "observer_camera"], fps: 10,
      frames: [{ index: 0, phase: "control", simTimeS: 0, images: { head_camera: "/ora/capture/00000000-camera-0.png" } }] };
    expect(motionCamera(motion, ["observer_camera", "head_camera"])).toBe("head_camera");
    expect(motionCamera(null, ["x"])).toBeNull();
  });
});

describe("a message this page sends while the robot moves", () => {
  test("shows queued at once, until the run's own record of it arrives", async () => {
    const { initialState, reduce } = await import("../src/state.ts");
    let state = reduce(initialState, { type: "event", event: { type: "snapshot", run: { mode: "play", playMessages: [] },
      messages: [], state: {} } });
    state = reduce(state, { type: "pending", text: "are you done?" });
    const conversation = buildConversation([], null, new Map(), [], { messages: [], endings: [], pending: state.pending, running: true });
    expect(conversation.items).toMatchObject([{ kind: "joined", id: null, text: "are you done?", state: "queued" }]);
    state = reduce(state, { type: "event", event: { type: "play_message", message: { id: "m9", text: "are you done?",
      state: "queued", joined: true, instruction: 1 } } });
    expect(state.pending).toEqual([]);
    expect(state.run.playMessages).toHaveLength(1);
    state = reduce(reduce(state, { type: "pending", text: "x" }), { type: "unpend", text: "x" });
    expect(state.pending).toEqual([]);
  });
});

describe("a model call stopped at its boundary", () => {
  test("an empty aborted reply is not shown; an error still is", () => {
    const conversation = buildConversation([
      { role: "assistant", timestamp: 1, content: [], stopReason: "aborted", errorMessage: "Request was aborted" },
      { role: "assistant", timestamp: 2, content: [], stopReason: "error", errorMessage: "429 Too Many Requests" },
    ]);
    expect(conversation.items.map((item) => (item as any).error)).toEqual(["429 Too Many Requests"]);
  });

  test("stopped before it was sent (Pi says error, with the abort signal's words): superseded after a joined message, never red", () => {
    // Free play: the act's action is submitted, a message joins meanwhile, Pi goes on to answer it and the
    // boundary stops that call before its request; the next act reads the message with the new observation.
    const skipped = { role: "assistant", timestamp: 3, content: [], stopReason: "error", errorMessage: "This operation was aborted" };
    const conversation = buildConversation([
      ...play,
      { role: "custom", customType: "ora.play_message", display: true, timestamp: 2, content: "which toy is your favourite?",
        details: { ora_message_id: "m1", ora_instruction: 1, ora_observation_step: 6 } },
      skipped,
      { role: "assistant", timestamp: 4, content: [{ type: "text", text: "The little owl. Back to the blocks." }] },
    ]);
    expect(conversation.items.map((item) => item.kind)).toEqual(["observation", "user", "joined", "skipped", "assistant"]);
    expect(conversation.items.some((item) => (item as any).error)).toBe(false);
    expect(conversation.items[4]).toMatchObject({ kind: "assistant", answers: conversation.items[2].key });
    // Without a joined message before it (Esc, a spent budget) it is not shown at all.
    expect(buildConversation([...play, skipped]).items.map((item) => item.kind)).toEqual(["observation", "user"]);
  });
});

describe("a reply cut off at the model's output limit", () => {
  test("marked as cut off with its output tokens, and ORA's continuation is a note, not the agent's error", () => {
    const conversation = buildConversation([
      ...play,
      { role: "assistant", timestamp: 3, stopReason: "length", usage: { output: 16384 },
        content: [{ type: "thinking", thinking: "Let me weigh pushing the box over..." }] },
      { role: "custom", customType: "ora.continuation", display: true, timestamp: 4,
        content: "Your previous response reached the Provider output limit before it was complete.",
        details: { ora_continuation: true, ora_reason: "output_truncated" } },
    ]);
    expect(conversation.items.map((item) => item.kind)).toEqual(["observation", "user", "assistant", "note"]);
    expect(conversation.items[2]).toMatchObject({ kind: "assistant", text: "", error: null, truncated: { outputTokens: 16384 } });
    expect(conversation.items[3]).toMatchObject({ kind: "note", title: "ora.continuation" });
    // Cut off with nothing at all (not even reasoning) is still shown, as cut off.
    expect(buildConversation([{ role: "assistant", timestamp: 1, stopReason: "length", content: [] }]).items)
      .toMatchObject([{ kind: "assistant", truncated: { outputTokens: null } }]);
  });

  test("its instruction's ending reads as an ending, not a report", () => {
    expect(endingKind("output_truncated")).toBe("ended");
    expect(endingText("output_truncated")).toBe("ended: the model's reply was cut off at its output limit");
  });
});

describe("free play's queued instructions", () => {
  test("held by the run above the prompt until the instruction they open starts, never in the conversation", async () => {
    const { initialState, reduce } = await import("../src/state.ts");
    const { queuedNext } = await import("../src/lib/messages.ts");
    let page = { ...initialState, run: { mode: "play", playMessages: [] as any[] } };
    page = reduce(page, { type: "pending", text: "then stack them", next: true });
    page = reduce(page, { type: "pending", text: "slower" });
    expect(queuedNext(page.run.playMessages, page.pendingNext)).toEqual([{ key: "pending:0:then stack them", text: "then stack them" }]);
    // The run admitted it (the Supervisor's message_queued_next): shown from its record now.
    const held = { id: "q1", text: "then stack them", state: "queued", joined: false, next: true, instruction: 2 };
    page = reduce(page, { type: "event", event: { type: "play_message", message: held } });
    expect([page.pendingNext, page.pending]).toEqual([[], ["slower"]]);
    expect(queuedNext(page.run.playMessages, page.pendingNext)).toEqual([{ key: "q1", text: "then stack them" }]);
    const context = { messages: page.run.playMessages, endings: [], pending: [], running: true };
    expect(buildConversation([], null, new Map(), [], context).items).toEqual([]);
    // Its instruction starts: it leaves the chips and is shown where it opened that instruction.
    page = reduce(page, { type: "event", event: { type: "play_message", message: { ...held, next: false, instruction: 3 } } });
    expect(queuedNext(page.run.playMessages, page.pendingNext)).toEqual([]);
    expect(buildConversation([], null, new Map(), [], { ...context, messages: page.run.playMessages }).items
      .map((item: any) => [item.kind, item.state])).toEqual([["joined", "next"]]);
  });

  test("a stop sent while the agent plans is withdrawn only when no motion read it", async () => {
    const { PlanStop } = await import("../src/lib/cell.ts");
    let now = 0;
    const stop = new PlanStop(() => now);
    expect(stop.settle({ state: "instruction_finished" })).toBe(false);
    stop.sent(null);
    expect(stop.settle({ state: "message_delivered", message: { text: "x" } })).toBe(false);
    expect(stop.settle({ state: "instruction_finished" })).toBe(true);
    stop.sent(" not that cup ");
    expect(stop.settle({ state: "action_started" })).toBe(false);
    expect(stop.settle({ state: "message_delivered", message: { text: "not that cup" } })).toBe(false);
    stop.sent("not that cup");
    expect(stop.settle({ state: "message_delivered", message: { text: "not that cup" } })).toBe(true);
    stop.sent(null);
    now = 10_001;
    expect(stop.settle({ state: "instruction_finished" })).toBe(false);
  });
});
