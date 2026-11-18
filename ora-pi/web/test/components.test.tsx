// @vitest-environment jsdom
// The page's components in a DOM: a tool card selects its step and the camera
// panel draws that call's overlay on the step's image; an episode shows none of
// the session tools free play has; free play's phase comes from the Supervisor;
// dialogs answer in Pi's format; nothing on the page is Pi's mark. The compact
// conversation folds each thought and tool call to a line; the camera picture sits
// in one fixed box whatever its source; a replay is the same page at its clock.

import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { App, ReplayApp } from "../src/App.tsx";
import { CameraPanel } from "../src/components/CameraPanel.tsx";
import { Composer } from "../src/components/Composer.tsx";
import { ConversationView } from "../src/components/Conversation.tsx";
import { Dialog } from "../src/components/Dialogs.tsx";
import { MotionPlayer } from "../src/components/MotionPlayer.tsx";
import type { Display } from "../src/lib/display.ts";
import { lang, setLang } from "../src/lib/i18n.ts";
import { buildConversation } from "../src/lib/messages.ts";
import { buildReplay } from "../server/replay.ts";
import { at, replayRun } from "./fixtures/replay-run.ts";
import { simvideoRun } from "./fixtures/simvideo-run.ts";

const png = { type: "image", mimeType: "image/png", data: "AAAA" };
const messages = [
  { role: "custom", customType: "ora.observation", content: [{ type: "text", text: "observation 3" }, png], display: true,
    timestamp: 10, details: { ora_observation_step: 3, ora: { step: 3, images: [{ camera: "agentview", inline: true, url: null }] } } },
  { role: "assistant", timestamp: 11, content: [{ type: "toolCall", id: "call-g", name: "vision_ground",
    arguments: { camera: "agentview", text: "basket" } }] },
  { role: "toolResult", toolCallId: "call-g", toolName: "vision_ground", timestamp: 12, details: { ora: { step: 3, images: [] } },
    content: [{ type: "text", text: JSON.stringify({ camera: "agentview", overlay: { width: 256, height: 256 },
      candidates: [{ candidate_id: "T0", bbox_xyxy: [1, 2, 30, 40], backend_score: 0.9 }] }) }] },
];

class FakeEventSource {
  static last: FakeEventSource | null = null;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((message: { data: string }) => void) | null = null;
  constructor() {
    FakeEventSource.last = this;
  }
  emit(event: unknown) {
    act(() => this.onmessage?.({ data: JSON.stringify(event) }));
  }
  close() {}
}

beforeEach(() => {
  vi.stubGlobal("EventSource", FakeEventSource);
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200 })));
  HTMLCanvasElement.prototype.getContext = (() => null) as never;
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function snapshot(run: unknown, extra: Record<string, unknown> = {}) {
  return { type: "snapshot", run, state: { isStreaming: false }, messages, streaming: null, tools: [],
    queue: { steering: [], followUp: [] }, dialogs: [], statuses: {}, widgets: {}, sent: [], closed: null, ...extra };
}

describe("tool cards and the camera panel", () => {
  test("a tool card shows its step's image and selects the step and call", () => {
    const conversation = buildConversation(messages);
    const onTool = vi.fn();
    render(<ConversationView conversation={conversation} focus={{ step: null, tool: null }} play={false} scrollTo={null}
      onTool={onTool} onStep={() => undefined} empty="" display={{ view: "detailed", thinking: false }} />);
    const card = document.querySelector("[data-tool='vision_ground']")!;
    expect(card.getAttribute("data-step")).toBe("3");
    expect(card.querySelector("img")!.getAttribute("src")).toBe("data:image/png;base64,AAAA");
    expect(card.querySelector(".stage")!.getAttribute("data-overlay")).toBe("grounding");
    fireEvent.click(screen.getByText("vision_ground"));
    expect(onTool.mock.calls[0][0]).toMatchObject({ id: "call-g", step: 3 });
  });

  test("the panel shows the selected step through the call's camera, its boxes over it", () => {
    const conversation = buildConversation(messages);
    render(<CameraPanel conversation={conversation} focus={{ mode: "step", step: 3, tool: "call-g" }} camera={null}
      onCamera={() => undefined} onFocus={() => undefined} replaying={false} onReplay={() => undefined} liveWorld={false} />);
    expect(screen.getByTestId("panel-caption").textContent).toBe("Step 3 · agentview · vision_ground · 1 candidate “basket”");
    expect(document.querySelector(".stage.main")!.getAttribute("data-overlay")).toBe("grounding");
    expect(screen.getByRole("option", { name: /Step 3/ }).getAttribute("aria-selected")).toBe("true");
  });
});

describe("the page", () => {
  test("an episode: no session tools, its phase, and nothing of Pi's mark", () => {
    render(<App liveWorld={false} />);
    FakeEventSource.last!.emit(snapshot({ runId: "run-1", mode: "episode", refused: { compact: "x", set_model: "x",
      set_thinking_level: "x", set_session_name: "x", new_session: "x", fork: "x" }, play: null }));
    expect(screen.getByText("episode")).toBeTruthy();
    expect(screen.getByTestId("phase").textContent).toContain("Robot acting");
    expect(screen.queryByTitle(/Session: model/)).toBeNull();
    expect(screen.getByRole("button", { name: /Stop run/ })).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/π|pi-web/);
    expect([...document.querySelectorAll("img")].map((image) => image.getAttribute("src"))).not.toContain(expect.stringMatching(/pi/));
  });

  test("free play: its session tools, and its phase from the Supervisor while the model is idle", () => {
    render(<App liveWorld={false} />);
    const source = FakeEventSource.last!;
    source.emit(snapshot({ runId: "run-2", mode: "play", refused: { export_html: "x" },
      play: { state: "instruction", running: "op-1", instructions: 1, scene: 1, scenes: 2 } }));
    expect(screen.getByText("free play")).toBeTruthy();
    expect(screen.getByTitle(/Session: model/)).toBeTruthy();
    expect(screen.getByTestId("phase").textContent).toContain("Instruction 1 running");
    expect(screen.getByRole("button", { name: /Stop instruction/ }).hasAttribute("disabled")).toBe(false);
    source.emit({ type: "play", play: { state: "instruction_finished", running: null, instructions: 1, scene: 1, scenes: 2,
      instruction: { index: 0, ended: "reported" } } });
    expect(screen.getByTestId("phase").textContent).toBe("Waiting for your instructionInstruction 1 done; the agent reported");
    expect(screen.getByRole("button", { name: /Stop instruction/ }).hasAttribute("disabled")).toBe(true);
  });

  test("a dialog from the run is answered in Pi's format and closes", async () => {
    render(<App liveWorld={false} />);
    const source = FakeEventSource.last!;
    source.emit(snapshot({ runId: "run-1", mode: "episode", refused: {}, play: null }));
    source.emit({ type: "ui", request: { type: "extension_ui_request", id: "d-1", method: "confirm", title: "Stop this run?",
      message: "ORA ends the episode now" } });
    expect(screen.getByRole("dialog", { name: "Stop this run?" })).toBeTruthy();
    fireEvent.click(screen.getByText("Yes"));
    const [path, init] = (fetch as unknown as { mock: { calls: [string, RequestInit][] } }).mock.calls.at(-1)!;
    expect(path).toBe("/api/dialog");
    expect(JSON.parse(String(init.body))).toEqual({ id: "d-1", confirmed: true });
    source.emit({ type: "ui_done", id: "d-1" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});

describe("dialogs and the composer", () => {
  test("select answers {value}; input answers the text; Escape cancels", () => {
    const onAnswer = vi.fn();
    const { unmount } = render(<Dialog request={{ id: "s", method: "select", title: "Pick", options: ["a", "b"] }} onAnswer={onAnswer} />);
    fireEvent.click(screen.getByText("b"));
    expect(onAnswer).toHaveBeenLastCalledWith({ value: "b" });
    unmount();
    render(<Dialog request={{ id: "i", method: "input", title: "Name" }} onAnswer={onAnswer} />);
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "left gripper" } });
    fireEvent.click(screen.getByText("OK"));
    expect(onAnswer).toHaveBeenLastCalledWith({ value: "left gripper" });
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(onAnswer).toHaveBeenLastCalledWith({ cancelled: true });
  });

  test("Enter steers while the agent works; Alt+Enter queues a follow-up; free play's Esc stops the instruction", async () => {
    const onSend = vi.fn(async () => true);
    const onControl = vi.fn();
    const { rerender } = render(<Composer play={false} phase={null} streaming draft={null} closed={false} onSend={onSend}
      onControl={onControl} onStopRun={() => undefined} onRefused={() => undefined} />);
    const box = screen.getByRole("textbox");
    fireEvent.change(box, { target: { value: "slower" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(onSend).toHaveBeenLastCalledWith({ kind: "message", text: "slower", as: "steer" });
    fireEvent.change(box, { target: { value: "then stop" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter", altKey: true }));
    expect(onSend).toHaveBeenLastCalledWith({ kind: "message", text: "then stop", as: "follow_up" });
    rerender(<Composer play phase="running" streaming={false} draft={null} closed={false} onSend={onSend}
      onControl={onControl} onStopRun={() => undefined} onRefused={() => undefined} />);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onControl).toHaveBeenLastCalledWith("stop_instruction");
  });
});

describe("the conversation follows its end", () => {
  // jsdom lays nothing out: the list's geometry is a box whose content height the test sets, and its
  // scroll position clamps the way a browser's does.
  function geometry(element: HTMLElement, height: number) {
    const box = { content: height, top: 0, view: 400 };
    Object.defineProperty(element, "clientHeight", { configurable: true, get: () => box.view });
    Object.defineProperty(element, "scrollHeight", { configurable: true, get: () => box.content });
    Object.defineProperty(element, "scrollTop", {
      configurable: true, get: () => box.top,
      set: (value: number) => { box.top = Math.max(0, Math.min(value, box.content - box.view)); },
    });
    return box;
  }
  const steps = (n: number) => buildConversation(Array.from({ length: n }, (_, index) => (
    { role: "assistant", timestamp: 100 + index, content: [{ type: "text", text: `reply ${index}` }] })));
  const view = (n: number) => (
    <ConversationView conversation={steps(n)} focus={{ step: null, tool: null }} play scrollTo={null}
      onTool={() => undefined} onStep={() => undefined} empty="" />);

  test("a big growth that lands before the scroll event does not leave the end", () => {
    const { rerender } = render(view(2));
    const list = document.querySelector(".conversation") as HTMLElement;
    const box = geometry(list, 1000);
    rerender(view(3));
    expect(box.top).toBe(600);
    // The race: the page jumped to the end, then content grew by far more than the margin before the
    // browser delivered that scroll's event (a long report, a tall image, a whole instruction at once).
    box.content = 1800;
    fireEvent.scroll(list);
    rerender(view(4));
    expect(box.top).toBe(1400);
  });

  test("only the operator's own scrolling leaves the end, and scrolling back to it follows again", () => {
    const { rerender } = render(view(2));
    const list = document.querySelector(".conversation") as HTMLElement;
    const box = geometry(list, 1000);
    rerender(view(3));
    fireEvent.wheel(list, { deltaY: -300 });
    box.top = 300;
    fireEvent.scroll(list);
    box.content = 1600;
    rerender(view(4));
    expect(box.top).toBe(300);
    fireEvent.wheel(list, { deltaY: 900 });
    box.top = 1200;
    fireEvent.scroll(list);
    box.content = 2000;
    rerender(view(5));
    expect(box.top).toBe(1600);
  });
});

describe("free play's conversation and live status", () => {
  const playMessages = [
    { role: "custom", customType: "ora.play_observation", display: true, timestamp: 1000, details: { ora_observation_step: 3 },
      content: [{ type: "text", text: "{}" }] },
    { role: "user", timestamp: 1001, content: [{ type: "text", text: "put the cup on the plate" }] },
    { role: "custom", customType: "ora.play_message", display: true, timestamp: 2000, content: "how far along are you?",
      details: { ora_message_id: "m1", ora_instruction: 1, ora_observation_step: 4 } },
    { role: "assistant", timestamp: 2100, content: [{ type: "text", text: "Halfway there; I keep going." }] },
  ];
  const run = { runId: "run-3", mode: "play", refused: {}, play: { state: "action_started", running: "i1", instructions: 1,
    scene: 1, scenes: 2, action: { action_id: "a", observation_step: 4 } },
  playMessages: [{ id: "i1", text: "put the cup on the plate", state: "delivered", joined: false, instruction: 1 },
    { id: "m2", text: "and then the bowl?", state: "queued", joined: true, instruction: 1 }],
  playEndings: [] };

  test("a joined message, its state, the answer, the robot moving with its seconds, and a stop marked where it happened", () => {
    render(<App liveWorld={false} />);
    const source = FakeEventSource.last!;
    source.emit(snapshot(run, { messages: playMessages, activity: { kind: "moving", elapsedMs: 2300, step: 4 } }));
    expect(screen.getByText("Instruction 1")).toBeTruthy();
    expect(screen.getAllByText("Joins instruction 1")).toHaveLength(2);
    expect(document.querySelector("[data-joined='delivered']")!.textContent).toContain("delivered at step 4");
    expect(document.querySelector("[data-joined='queued']")!.textContent).toContain("queued · the agent reads it before its next step");
    expect(document.querySelector(".bubble.answer")!.textContent).toContain("Answer");
    expect(screen.getByTestId("doing").textContent).toBe("Robot moving… 2 s");
    expect(screen.getByTestId("activity").textContent).toContain("a message you send now joins this instruction");
    source.emit({ type: "play", at: 5000, play: { state: "instruction_finished", running: null, instructions: 1, scene: 1, scenes: 2,
      instruction: { index: 0, instruction_id: "i1", ended: "operator_stopped", observation_steps: { start: 3, end: 6 },
        wall_duration_s: 12.2, decisions: 3 } } });
    expect(document.querySelector("[data-ended='operator_stopped']")!.textContent)
      .toContain("Instruction 1 stopped (Esc) · at step 6");
    expect(screen.queryByTestId("doing")).toBeNull();
    expect(screen.getByTestId("phase").textContent).toContain("Waiting for your instruction");
  });

  test("a reply cut off at the output limit says so; a call a joined message superseded is neutral, not red", () => {
    const conversation = buildConversation([
      ...playMessages.slice(0, 3),
      { role: "assistant", timestamp: 2050, content: [], stopReason: "error", errorMessage: "This operation was aborted" },
      { role: "assistant", timestamp: 2100, stopReason: "length", usage: { output: 16384 },
        content: [{ type: "thinking", thinking: "Weighing how to tip the box back..." }] },
      { role: "custom", customType: "ora.continuation", display: true, timestamp: 2200,
        content: "Your previous response reached the Provider output limit before it was complete.",
        details: { ora_continuation: true, ora_reason: "output_truncated" } },
    ]);
    render(<ConversationView conversation={conversation} focus={{ step: null, tool: null }} play scrollTo={null}
      onTool={() => undefined} onStep={() => undefined} empty="" />);
    expect(document.querySelector(".error-line")).toBeNull();
    expect(document.body.textContent).not.toContain("This operation was aborted");
    expect(document.querySelector("[data-skipped='superseded']")!.textContent)
      .toBe("Superseded by your message: the robot's action runs first, and the agent reads your message at its next step");
    expect(document.querySelector("[data-truncated='output_limit']")!.textContent)
      .toBe("Cut off: the reply reached the model's output limit (16384 tokens)");
    expect(document.querySelector("[data-note='ora.continuation']")!.textContent).toContain("ORA · continue");
  });

  test("the language switch turns the page Chinese, and it stays the browser's choice", () => {
    const before = lang();
    try {
      render(<App liveWorld={false} />);
      FakeEventSource.last!.emit(snapshot(run, { messages: playMessages, activity: { kind: "thinking", elapsedMs: 4000 } }));
      expect(screen.getByTestId("doing").textContent).toBe("Thinking… 4 s");
      fireEvent.click(screen.getByRole("button", { name: "中文" }));
      expect(screen.getByTestId("doing").textContent).toBe("思考中… 4 秒");
      expect(screen.getAllByText("并入第 1 条指令")).toHaveLength(2);
      expect(screen.getByRole("button", { name: /停止指令/ })).toBeTruthy();
      expect(screen.getByTestId("phase").textContent).toContain("第 1 条指令执行中");
      expect(localStorage.getItem("ora.web.lang")).toBe("zh");
    } finally {
      setLang(before);
      localStorage.removeItem("ora.web.lang");
    }
  });
});

describe("a live run's camera box", () => {
  test("takes the shape of the run's first operator view and keeps it", async () => {
    const summary = (width: number, height: number) => ({ motion: null, operatorView: { camera: "operator_camera", width, height,
      steps: [0], latest: { step: 0, camera: "operator_camera", width, height, url: "/ora/operator-view/000000.png" } } });
    let calls = 0;
    vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify(String(url).startsWith("/api/frames")
      ? (calls++ === 0 ? summary(1280, 720) : summary(640, 480)) : { ok: true }), { status: 200 })));
    render(<App liveWorld={false} />);
    expect(document.querySelector(".split")!.className).toBe("split");
    FakeEventSource.last!.emit(snapshot({ runId: "run-5", mode: "play", refused: {}, play: null }));
    await act(async () => {
      await new Promise((done) => setTimeout(done, 250));
    });
    expect(document.querySelector(".split")!.className).toBe("split wide");
    FakeEventSource.last!.emit({ type: "message", message: { role: "custom", customType: "ora.play_observation", timestamp: 99,
      content: [{ type: "text", text: "{}" }], details: { ora_observation_step: 1 } } });
    await act(async () => {
      await new Promise((done) => setTimeout(done, 250));
    });
    expect(calls).toBeGreaterThan(1);
    expect((document.querySelector(".camera-panel") as HTMLElement).style.getPropertyValue("--camera-aspect")).toBe("1.7778");
  });
});

describe("the operator view and the robot's motion", () => {
  const summary = { motion: { cameras: ["head_camera"], fps: 10, steps: { 3: 3 } },
    operatorView: { camera: "observer_camera", width: 64, height: 48, steps: [3],
      latest: { step: 3, camera: "observer_camera", width: 64, height: 48, url: "/ora/operator-view/000003.png" } } };
  const motion = { step: 3, actionIds: ["a"], cameras: ["head_camera"], fps: 10, frames: [0, 1, 2].map((index) => ({
    index, phase: "control", simTimeS: index / 10, images: { head_camera: `/ora/capture/0000000${index}-camera-0.png` } })) };

  test("the operator view is the main picture, the robot's camera a picture to switch to", () => {
    const conversation = buildConversation(messages);
    const onCamera = vi.fn();
    render(<CameraPanel conversation={conversation} focus={{ mode: "live" }} camera={null} summary={{ ...summary, motion: null }}
      onCamera={onCamera} onFocus={() => undefined} replaying={false} onReplay={() => undefined} liveWorld={false} />);
    expect(document.querySelector(".stage.main img")!.getAttribute("src")).toBe("/ora/operator-view/000003.png");
    expect(screen.getByTestId("panel-caption").textContent).toBe("Operator view · observer_camera · step 3");
    expect(document.querySelector(".camera-thumb img")!.getAttribute("src")).toBe("data:image/png;base64,AAAA");
    fireEvent.click(document.querySelector(".camera-thumb")!);
    expect(onCamera).toHaveBeenLastCalledWith("agentview");
    expect(screen.getByRole("button", { name: /Operator view/ }).className).toContain("on");
  });

  test("a step's motion plays, can be paused and scrubbed", async () => {
    vi.useFakeTimers();
    try {
      vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify(String(url).includes("step=3")
        ? { ...summary, step: { step: 3, motion, operatorView: null } } : summary), { status: 200 })));
      const conversation = buildConversation(messages);
      render(<CameraPanel conversation={conversation} focus={{ mode: "step", step: 3, tool: null }} camera="head_camera"
        summary={summary} onCamera={() => undefined} onFocus={() => undefined} replaying={false} onReplay={() => undefined}
        liveWorld={false} />);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50);
      });
      const player = screen.getByTestId("motion");
      expect(player.getAttribute("data-frames")).toBe("3");
      expect(player.getAttribute("data-frame")).toBe("0");
      await act(async () => {
        await vi.advanceTimersByTimeAsync(250);
      });
      expect(player.getAttribute("data-frame")).toBe("2");
      expect(screen.getByTestId("panel-caption").textContent).toContain("Step 3 · head_camera · the robot's motion");
      fireEvent.change(screen.getByRole("slider"), { target: { value: "1" } });
      expect(player.getAttribute("data-frame")).toBe("1");
      expect(screen.getByRole("button", { name: /Motion/ }).getAttribute("aria-pressed")).toBe("true");
      fireEvent.click(screen.getByRole("button", { name: "What the model saw" }));
      expect(screen.queryByTestId("motion")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  test("a step's motion leads over the still operator view, which becomes the small picture", async () => {
    vi.useFakeTimers();
    try {
      vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify(String(url).includes("step=3")
        ? { ...summary, step: { step: 3, motion, operatorView: summary.operatorView.latest } } : summary), { status: 200 })));
      render(<CameraPanel conversation={buildConversation(messages)} focus={{ mode: "step", step: 3, tool: null }} camera={null}
        summary={summary} onCamera={() => undefined} onFocus={() => undefined} replaying={false} onReplay={() => undefined}
        liveWorld={false} />);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50);
      });
      expect(screen.getByTestId("motion").getAttribute("data-frames")).toBe("3");
      expect(document.querySelector(".camera-thumb img")!.getAttribute("src")).toBe("/ora/operator-view/000003.png");
    } finally {
      vi.useRealTimers();
    }
  });

  test("a motion still being recorded is followed as its frames arrive", () => {
    vi.useFakeTimers();
    try {
      const frames = [0, 1].map((index) => ({ src: `/f${index}.png`, simTimeS: index / 10 }));
      const { rerender } = render(<MotionPlayer frames={frames} fps={10} follow autoPlay playKey="live:4" caption={() => ""} />);
      act(() => vi.advanceTimersByTime(300));
      expect(screen.getByTestId("motion").getAttribute("data-frame")).toBe("1");
      const more = [...frames, ...[2, 3].map((index) => ({ src: `/f${index}.png`, simTimeS: index / 10 }))];
      rerender(<MotionPlayer frames={more} fps={10} follow autoPlay playKey="live:4" caption={() => ""} />);
      act(() => vi.advanceTimersByTime(250));
      expect(screen.getByTestId("motion").getAttribute("data-frame")).toBe("3");
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("the compact conversation", () => {
  const thoughtful = [
    ...messages.slice(0, 1),
    { role: "assistant", timestamp: 11, durationMs: 12_400, content: [
      { type: "thinking", thinking: "The basket is left of the plate." },
      { type: "text", text: "Looking for the basket." },
      { type: "toolCall", id: "call-g", name: "vision_ground", arguments: { camera: "agentview", text: "basket" } }] },
    messages[2],
  ];
  const draw = (display: Display = { view: "compact", thinking: false }) => (
    <ConversationView conversation={buildConversation(thoughtful)} focus={{ step: null, tool: null }} play={false} scrollTo={null}
      onTool={() => undefined} onStep={() => undefined} empty="" display={display} />);

  test("a thought is one line that opens; a tool call is one line (its name and arguments) that opens to the whole call", () => {
    render(draw());
    const thought = document.querySelector(".thinking")!;
    expect(thought.getAttribute("data-thinking")).toBe("closed");
    expect(thought.textContent).toBe("Thought for 12 s");
    expect(document.body.textContent).not.toContain("The basket is left of the plate.");
    fireEvent.click(screen.getByRole("button", { name: /Thought for 12 s/ }));
    expect(thought.textContent).toContain("The basket is left of the plate.");
    const line = document.querySelector(".tool-line[data-tool='vision_ground']")!;
    expect(line.querySelector(".tool-args")!.textContent).toBe('(text="basket", camera="agentview")');
    expect(line.querySelector("img")).toBeNull();
    expect(line.querySelector(".tool-result")).toBeNull();
    // The observation is one line too: its image folded away.
    expect(document.querySelector(".observation-line")!.textContent).toContain("1 image");
    expect(document.querySelectorAll(".conversation img")).toHaveLength(0);
    fireEvent.click(line.querySelector(".tool-head")!);
    expect(line.querySelector("img")!.getAttribute("src")).toBe("data:image/png;base64,AAAA");
    expect(line.querySelector(".stage")!.getAttribute("data-overlay")).toBe("grounding");
    expect([...line.querySelectorAll(".tool-result")].map((item) => item.textContent)).toEqual([
      JSON.stringify({ camera: "agentview", text: "basket" }, null, 2), expect.stringContaining("\"backend_score\": 0.9")]);
  });

  test("expand all thinking opens every thought; detailed draws cards with their images", () => {
    const { rerender } = render(draw({ view: "compact", thinking: true }));
    expect(document.querySelector(".thinking")!.getAttribute("data-thinking")).toBe("open");
    expect(document.querySelector(".thinking-text")!.textContent).toBe("The basket is left of the plate.");
    rerender(draw({ view: "detailed", thinking: false }));
    expect(document.querySelector(".thinking")!.getAttribute("data-thinking")).toBe("closed");
    expect(document.querySelector(".tool-card[data-tool='vision_ground'] img")).toBeTruthy();
    expect(document.querySelector(".observation-images img")).toBeTruthy();
  });

  test("the page switches compact and detailed above the conversation, and remembers the choice", () => {
    render(<App liveWorld={false} />);
    FakeEventSource.last!.emit(snapshot({ runId: "run-1", mode: "episode", refused: {}, play: null }, { messages: thoughtful }));
    expect(document.querySelector(".conversation")!.className).toContain("view-compact");
    fireEvent.click(screen.getByRole("button", { name: "Detailed" }));
    expect(document.querySelector(".conversation")!.className).toContain("view-detailed");
    expect(JSON.parse(localStorage.getItem("ora.web.display")!)).toEqual({ view: "detailed", thinking: false });
    fireEvent.click(screen.getByRole("checkbox", { name: "Expand all thinking" }));
    expect(document.querySelector(".thinking")!.getAttribute("data-thinking")).toBe("open");
    localStorage.removeItem("ora.web.display");
  });
});

describe("the camera box", () => {
  const css = readFileSync(resolve(import.meta.dirname, "..", "src", "styles.css"), "utf8");
  const rule = (selector: string) => {
    const at = css.indexOf(`${selector} {`);
    return at < 0 ? "" : css.slice(at, css.indexOf("}", at));
  };

  test("is one box of one shape; the bars around it keep their heights; overlays lie over it", () => {
    expect(rule(".camera-box")).toMatch(/aspect-ratio: var\(--camera-aspect, 4 \/ 3\)/);
    expect(rule(".camera-box")).toMatch(/width: min\(100cqw, calc\(100cqh \* var\(--camera-aspect, 4 \/ 3\)\)\)/);
    expect(rule(".split.wide")).toMatch(/grid-template-columns: minmax\(380px, 1.1fr\) minmax\(440px, 1fr\)/);
    expect(rule(".panel-stage")).toMatch(/container-type: size/);
    expect(rule(".stage img")).toMatch(/object-fit: contain/);
    for (const selector of [".panel-bar", ".panel-caption", ".timeline", ".header", ".replay-bar"]) {
      expect(rule(selector)).toMatch(/(^|[ ;])height: \d+px/);
    }
    for (const selector of [".motion-bar", ".tool-strip"]) expect(rule(selector)).toMatch(/position: absolute/);
  });

  test("keeps its place whatever it shows: the operator view, a step's image, a returned image, a motion", async () => {
    vi.useFakeTimers();
    try {
      const summary = { motion: { cameras: ["head_camera"], fps: 10, steps: { 3: 2 } },
        operatorView: { camera: "observer_camera", width: 1280, height: 720, steps: [3],
          latest: { step: 3, camera: "observer_camera", width: 1280, height: 720, url: "/ora/operator-view/000003.png" } } };
      const motion = { step: 3, actionIds: ["a"], cameras: ["head_camera"], fps: 10, frames: [0, 1].map((index) => ({
        index, phase: "control", simTimeS: index / 10, images: { head_camera: `/ora/capture/0000000${index}-camera-0.png` } })) };
      vi.stubGlobal("fetch", vi.fn(async (url: string) => new Response(JSON.stringify(String(url).includes("step=3")
        ? { ...summary, step: { step: 3, motion, operatorView: null } } : summary), { status: 200 })));
      const withImages = [...messages.slice(0, 2), { ...messages[2], content: [...messages[2].content,
        { type: "image", mimeType: "image/png", data: "BBBB" }] }];
      const conversation = buildConversation(withImages);
      const panel = (focus: Parameters<typeof CameraPanel>[0]["focus"], camera: string | null = null) => (
        <CameraPanel conversation={conversation} focus={focus} camera={camera} summary={summary} onCamera={() => undefined}
          onFocus={() => undefined} replaying={false} onReplay={() => undefined} liveWorld={false} />);
      const { rerender } = render(panel({ mode: "live" }));
      const box = screen.getByTestId("camera-box");
      const shows = () => [box.getAttribute("data-shows"), box.querySelector(".stage.main img")?.getAttribute("src")];
      expect(shows()).toEqual(["operator", "/ora/operator-view/000003.png"]);
      rerender(panel({ mode: "step", step: 3, tool: "call-g" }));
      expect(shows()).toEqual(["image", "data:image/png;base64,AAAA"]);
      fireEvent.click(document.querySelector(".strip-thumb")!);
      expect(shows()).toEqual(["image", "data:image/png;base64,BBBB"]);
      expect(box.querySelector(".tool-strip")).toBeTruthy();
      rerender(panel({ mode: "step", step: 3, tool: null }, "head_camera"));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(50);
      });
      expect(shows()).toEqual(["motion", "/ora/capture/00000000-camera-0.png"]);
      expect(box.querySelector(".motion-bar")).toBeTruthy();
      // The same box all along: only what is inside it changed.
      expect(screen.getByTestId("camera-box")).toBe(box);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("a recorded run", () => {
  async function open(search = "?replay=1&play=0") {
    const script = await buildReplay(replayRun());
    vi.stubGlobal("fetch", vi.fn(async (url: string) => (String(url) === "/api/replay"
      ? new Response(JSON.stringify(script), { status: 200 })
      : new Response(JSON.stringify({ error: "a recorded run takes no commands" }), { status: 410 }))));
    const { replayOptions } = await import("../src/lib/replay.ts");
    render(<ReplayApp options={replayOptions(search)!} />);
    await act(async () => {
      await new Promise((done) => setTimeout(done, 0));
    });
    return script;
  }
  const seek = async (time: number) => {
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => setTimeout(() => callback(0), 0));
    await act(async () => {
      await window.__oraReplay!.seek(time);
    });
  };

  test("is the page at its clock: the conversation so far, what the agent was doing then, the world then; nothing to send", async () => {
    const script = await open();
    expect(window.__oraReplay!.info()).toMatchObject({ start: script.start, end: script.end, mode: "play" });
    expect(screen.getByTestId("replay-label").textContent).toBe("waiting · 16×");
    expect(document.querySelector("textarea")).toBeNull();
    expect(screen.queryByRole("button", { name: /Send|Stop instruction|End session/ })).toBeNull();
    expect(screen.queryByTitle(/Session: model/)).toBeNull();
    await seek(at(12.5));
    expect(within(document.querySelector(".conversation") as HTMLElement).getByText("Line up the blocks, please.")).toBeTruthy();
    expect(document.querySelector(".instruction-text")!.textContent).toBe("Line up the blocks, please.");
    expect(screen.queryByText("Measuring the blocks first.")).toBeNull();
    expect(screen.getByTestId("doing").textContent).toBe("Thinking… 1 s");
    expect(screen.getByTestId("activity").textContent).toBe("Thinking… 1 s");
    expect(screen.getByTestId("phase").textContent).toContain("Instruction 1 running");
    expect(screen.getByTestId("replay-label").textContent).toBe("thinking · 16×");
    await seek(at(20.5));
    expect(screen.getByTestId("replay-label").textContent).toBe("robot motion · real time");
    expect(screen.getByText("Measuring the blocks first.")).toBeTruthy();
    expect(screen.getByTestId("doing").textContent).toBe("Robot moving… 1 s");
    const box = screen.getByTestId("camera-box");
    expect(box.querySelector(".stage.main img")!.getAttribute("src")).toBe("/ora/capture/00000002-camera-3.png");
    expect(document.querySelector(".camera-thumb img")!.getAttribute("src")).toBe("/ora/capture/00000002-camera-0.png");
    expect(screen.getByTestId("panel-caption").textContent).toBe("Recorded · Operator view · sim 0.20 s · the robot moving");
    expect(screen.getByRole("tab", { name: /Now/ }).getAttribute("aria-selected")).toBe("true");
    await seek(at(28));
    expect(document.querySelector("[data-ended='reported']")!.textContent).toContain("Instruction 1 done; the agent reported");
    expect(document.querySelector("[data-joined='delivered']")!.textContent).toContain("delivered at step 1");
    expect(document.querySelector(".bubble.answer")!.textContent).toContain("The owl.");
    expect(box.querySelector(".stage.main img")!.getAttribute("src")).toBe("/ora/operator-view/000001.png");
    expect(screen.getByTestId("replay-label").textContent).toBe("waiting · 16×");
    await seek(at(12.5));
    expect(screen.queryByText("Measuring the blocks first.")).toBeNull();
    const calls = (fetch as unknown as { mock: { calls: [string][] } }).mock.calls.map(([url]) => url);
    expect(calls).toEqual(["/api/replay"]);
  });

  test("a simulation-video capture: its sharp operator frames by simulated time, the settling window said", async () => {
    const script = await buildReplay(simvideoRun());
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(script), { status: 200 })));
    const { replayOptions } = await import("../src/lib/replay.ts");
    render(<ReplayApp options={replayOptions("?replay=1&play=0")!} />);
    await act(async () => {
      await new Promise((done) => setTimeout(done, 0));
    });
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => setTimeout(() => callback(0), 0));
    const sweep = window.__oraReplay!.info().segments.find((segment) => segment.action === "simvideo-sweep")!;
    await act(async () => {
      await window.__oraReplay!.seekPlayback(sweep.playFrom + 100);
    });
    const box = screen.getByTestId("camera-box");
    expect(screen.getByTestId("replay-label").textContent).toBe("robot motion · real time");
    // This capture has no operator-view stills (no Supervisor events): its operator camera is shown by its own name.
    expect(box.querySelector(".stage.main img")!.getAttribute("src")).toBe("/ora/capture/00000022-camera-3.png");
    expect(screen.getByTestId("panel-caption").textContent).toBe("Recorded · operator_camera · sim 0.63 s · the robot moving");
    await act(async () => {
      await window.__oraReplay!.seekPlayback(sweep.playFrom + 340);
    });
    expect(box.querySelector(".stage.main img")!.getAttribute("src")).toBe("/ora/capture/00000029-camera-3.png");
    expect(screen.getByTestId("panel-caption").textContent).toBe("Recorded · operator_camera · sim 0.88 s · the scene settling after the action");
  });

  test("the camera box takes the run's operator camera's shape, 16:9 for a 1280x720 one, 4:3 for the older 640x480", async () => {
    const wide = await buildReplay(simvideoRun());
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(wide), { status: 200 })));
    const { replayOptions } = await import("../src/lib/replay.ts");
    render(<ReplayApp options={replayOptions("?replay=1&play=0")!} />);
    await act(async () => {
      await new Promise((done) => setTimeout(done, 0));
    });
    expect(document.querySelector(".split")!.className).toBe("split wide");
    expect((document.querySelector(".camera-panel") as HTMLElement).style.getPropertyValue("--camera-aspect")).toBe("1.7778");
    expect(window.__oraReplay!.info().aspect).toBeCloseTo(16 / 9);
    cleanup();
    await open("?replay=1&play=0");
    expect(document.querySelector(".split")!.className).toBe("split");
    expect((document.querySelector(".camera-panel") as HTMLElement).style.getPropertyValue("--camera-aspect")).toBe("1.3333");
  });

  test("seeks; switches the motion's speed or one speed throughout; a video's page has no controls and no language switch", async () => {
    const script = await open("?replay=1&play=0&speed=4");
    expect(screen.getByTestId("replay-label").textContent).toBe("waiting · 4×");
    expect(window.__oraReplay!.info()).toMatchObject({ speed: 4, motion: 1, duration: 15_000 / 4 + Math.max(200, 2000 / 4) + 13_000 / 4 });
    fireEvent.change(screen.getByRole("combobox", { name: "Robot motion speed" }), { target: { value: "same" } });
    expect(screen.getByTestId("replay-label").textContent).toBe("recorded · 4× speed");
    expect(window.__oraReplay!.info()).toMatchObject({ motion: "same", duration: 30_000 / 4 });
    fireEvent.change(screen.getByRole("slider", { name: "Replay position" }), { target: { value: String(at(15) - script.start) } });
    expect(screen.getByText("Measuring the blocks first.")).toBeTruthy();
    expect(screen.getByTestId("replay-time").textContent).toBe(`0:${String(Math.floor((at(15) - script.start) / 1000)).padStart(2, "0")} / 0:${Math.floor((script.end - script.start) / 1000)}`);
    cleanup();
    await open("?replay=1&export=1&publish=1");
    expect(screen.queryByTestId("replay-bar")).toBeNull();
    expect(screen.queryByRole("button", { name: "中文" })).toBeNull();
    expect(screen.queryByRole("group", { name: "How the conversation is drawn" })).toBeNull();
    expect(document.querySelector(".app")!.className).toContain("exporting");
    expect(document.querySelector(".run-id")).toBeNull();
  });
});
