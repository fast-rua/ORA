// @vitest-environment jsdom
// A real robot cell on the page (components/CellPanel.tsx in App): the robot's question as a card bound to
// its id, answered only after a confirmation that only its Yes button gives; Home at once, E-stop after a
// confirmation; the auto-approve switch; the held arms' alerts and the end of the worker that held them;
// the outcomes of the terminal's commands and the page's; typed commands; and none of it on a simulator.

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { App } from "../src/App.tsx";
import { setLang } from "../src/lib/i18n.ts";

type Json = any;

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

const NOW = Date.now() / 1000;
const QUESTION = { id: "0a1b2c3d", kind: "motion", title: "左臂动作：下降 4 cm", text: "左臂下降到桌面上方 8 cm。\n速度：慢。",
  details: { arms: ["left"] }, created_unix_s: NOW - 5, expires_unix_s: NOW + 120, pid: 4242, asker: "ORA YAM worker" };

function cellState(extra: Json = {}): Json {
  return { cell: { kind: "yam", rig_mode: "on", cell_config: "/cells/bench.yaml" }, now: NOW, pending: [QUESTION],
    auto_approve: { on: false, stale: null }, alerts: [], alerts_stale: false, worker: { pid: 4242, alive: true },
    worker_ended: null, ...extra };
}

let posts: Array<{ url: string; body: Json }> = [];

function stubFetch(state: Json) {
  posts = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (init?.method === "POST") {
      const body = JSON.parse(String(init.body ?? "{}"));
      posts.push({ url: String(url), body });
      const verb = String(url).split("/").pop();
      return new Response(JSON.stringify({ id: `out-${posts.length}`, name: verb, verb, ok: true, text: "done", seconds: 0.4,
        via: "web page" }), { status: 200 });
    }
    if (String(url) === "/ora/cell") return new Response(JSON.stringify(state), { status: 200 });
    return new Response(JSON.stringify({}), { status: 200 });
  }));
}

async function open(state: Json, mode = "play") {
  stubFetch(state);
  render(<App liveWorld={false} />);
  FakeEventSource.last!.emit({ type: "snapshot", run: { runId: "run-cell", mode, refused: {}, play: null, cellCommands: [] },
    state: { isStreaming: false }, messages: [], streaming: null, tools: [], queue: { steering: [], followUp: [] },
    dialogs: [], statuses: {}, widgets: {}, sent: [], closed: null });
  await act(async () => {
    await new Promise((done) => setTimeout(done, 20));
  });
}

beforeEach(() => {
  setLang("en", false);
  vi.stubGlobal("EventSource", FakeEventSource);
  HTMLCanvasElement.prototype.getContext = (() => null) as never;
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("a real cell's controls", () => {
  test("the robot's question is a card bound to its id; only Yes answers it, Enter says no", async () => {
    await open(cellState());
    const card = await screen.findByTestId("cell-question");
    expect(card.getAttribute("data-id")).toBe("0a1b2c3d");
    expect(card.textContent).toContain("左臂动作：下降 4 cm");
    expect(card.textContent).toContain("速度：慢。");
    expect(card.textContent).toMatch(/id 0a1b2c3d · 1[12]\d s left/);
    fireEvent.click(screen.getByRole("button", { name: "Approve…" }));
    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toContain("Approve this?");
    expect(dialog.textContent).toContain("左臂下降到桌面上方 8 cm。");
    fireEvent.keyDown(dialog, { key: "Enter" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(posts).toEqual([]);
    fireEvent.click(screen.getByRole("button", { name: "Approve…" }));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Yes" })));
    expect(posts).toEqual([{ url: "/ora/cell/approval/0a1b2c3d/approve", body: { confirmed: true } }]);
  });

  test("a decline carries the reason typed on the card", async () => {
    await open(cellState());
    fireEvent.change(await screen.findByRole("textbox", { name: /Reason for a decline/ }), { target: { value: "too close" } });
    fireEvent.click(screen.getByRole("button", { name: "Decline…" }));
    expect(screen.getByRole("dialog").textContent).toContain("Reason sent with the decline: too close");
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Yes" })));
    expect(posts).toEqual([{ url: "/ora/cell/approval/0a1b2c3d/decline", body: { confirmed: true, reason: "too close" } }]);
  });

  test("Home runs at once; E-stop, Float and Release only after their confirmation", async () => {
    await open(cellState({ pending: [] }));
    const bar = await screen.findByTestId("cell-bar");
    expect(bar.textContent).toContain("Real cell · yam · on");
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Home" })));
    expect(posts).toEqual([{ url: "/ora/cell/verb/home", body: { confirmed: false } }]);
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /E-stop/ }));
    expect(screen.getByRole("dialog").textContent).toContain("Disable every motor?");
    fireEvent.click(screen.getByRole("button", { name: "No" }));
    expect(posts).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: /E-stop/ }));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Yes" })));
    expect(posts[1]).toEqual({ url: "/ora/cell/verb/estop", body: { confirmed: true } });
    await waitFor(() => expect(screen.getAllByTestId("cell-outcome").length).toBe(2));
    expect(screen.getAllByTestId("cell-outcome")[0].textContent).toContain("/home · ora cell home · done in 0.4 s · web page");
  });

  test("the auto-approve switch: turning it on is confirmed first, turning it off is not", async () => {
    await open(cellState({ pending: [] }));
    expect((await screen.findByTestId("cell-auto")).textContent).toContain("Auto-approve is off");
    fireEvent.click(screen.getByRole("button", { name: "Turn on auto-approve…" }));
    expect(screen.getByRole("dialog").textContent).toContain("Turn on auto-approve (YOLO)?");
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Yes" })));
    expect(posts).toEqual([{ url: "/ora/cell/auto-approve/on", body: { confirmed: true } }]);
  });

  test("the held arms' alerts and the end of the worker that held them are banners", async () => {
    await open(cellState({ pending: [], worker_ended: { pid: 4242, seen_unix_s: NOW },
      alerts: [{ code: "motor_hot", level: "critical", text: "左臂肘电机 61°C：已停止" },
        { code: "gripper_push", level: "warning", text: "右夹爪持续用力" }] }));
    const banners = await screen.findByTestId("cell-banners");
    expect(banners.textContent).toContain("(pid 4242) has ended: nothing controls the arms now");
    expect(banners.querySelector(".tone-error")?.textContent).toContain("has ended");
    expect([...banners.querySelectorAll(".cell-banner")].map((item) => item.className)).toEqual([
      "cell-banner tone-error", "cell-banner tone-error", "cell-banner tone-warning"]);
  });

  test("the terminal's commands are listed beside the page's", async () => {
    await open(cellState({ pending: [] }));
    await screen.findByTestId("cell-bar");
    FakeEventSource.last!.emit({ type: "cell_command", entry: { id: "t-1", name: "diagnose", verb: "diagnose", ok: true,
      text: "{\n  \"holder\": null\n}", seconds: 1.2, via: "play terminal", at: 1 } });
    const item = screen.getByTestId("cell-outcome");
    expect(item.textContent).toContain("/diagnose · ora cell diagnose · done in 1.2 s · terminal");
    expect(item.querySelector("pre")?.textContent).toContain("\"holder\": null");
  });

  test("typed: /approve answers the oldest question after its confirmation; /yolo says the switch", async () => {
    await open(cellState());
    await screen.findByTestId("cell-question");
    const box = screen.getByRole("textbox", { name: "Instruction" });
    fireEvent.change(box, { target: { value: "/approve" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(screen.getByRole("dialog").textContent).toContain("id 0a1b2c3d");
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Yes" })));
    expect(posts).toEqual([{ url: "/ora/cell/approval/0a1b2c3d/approve", body: { confirmed: true } }]);
    fireEvent.change(box, { target: { value: "/yolo" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(document.querySelector(".notices")!.textContent).toContain("Auto-approve is off");
  });
});

describe("stopping a motion where the arms are (/halt)", () => {
  test("Esc while the robot moves halts it before the instruction stops; typed /halt and Stop here do it at once", async () => {
    await open(cellState({ pending: [] }));
    const running = { state: "instruction", running: "op-1", instructions: 1, scene: 1, scenes: 1 };
    FakeEventSource.last!.emit({ type: "snapshot", run: { runId: "run-cell", mode: "play", refused: {}, play: running, cellCommands: [] },
      state: { isStreaming: false }, messages: [], streaming: null, tools: [], queue: { steering: [], followUp: [] },
      dialogs: [], statuses: {}, widgets: {}, sent: [], closed: null, activity: { kind: "moving", elapsedMs: 100 } });
    await screen.findByTestId("cell-bar");
    await act(async () => fireEvent.keyDown(window, { key: "Escape" }));
    expect(posts).toEqual([{ url: "/ora/cell/stop-motion", body: {} },
      { url: "/api/play", body: { control: "stop_instruction" } }]);
    // The motion read it: an instruction that then ends takes nothing back.
    FakeEventSource.last!.emit({ type: "play", play: { ...running, state: "instruction_finished", running: null,
      instruction: { index: 0, ended: "operator_stopped" } } });
    await act(async () => { await new Promise((done) => setTimeout(done, 10)); });
    expect(posts).toHaveLength(2);

    FakeEventSource.last!.emit({ type: "play", play: { ...running, state: "awaiting_instruction", running: null } });
    const box = screen.getByRole("textbox", { name: "Instruction" });
    fireEvent.change(box, { target: { value: "/halt" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(posts.at(-1)).toEqual({ url: "/ora/cell/stop-motion", body: {} });
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Stop here" })));
    expect(posts.at(-1)).toEqual({ url: "/ora/cell/stop-motion", body: {} });
    expect(posts).toHaveLength(4);
  });

  test("Esc or Stop & send while the agent plans: the motion its reply proposes is refused, a stop no motion read is withdrawn",
    async () => {
      await open(cellState({ pending: [] }));
      const running = { state: "instruction", running: "op-1", instructions: 1, scene: 1, scenes: 1 };
      FakeEventSource.last!.emit({ type: "snapshot", run: { runId: "run-cell", mode: "play", refused: {}, play: running,
        cellCommands: [], playMessages: [] },
      state: { isStreaming: true }, messages: [], streaming: null, tools: [], queue: { steering: [], followUp: [] },
      dialogs: [], statuses: {}, widgets: {}, sent: [], closed: null, activity: { kind: "thinking", elapsedMs: 100 } });
      await screen.findByTestId("cell-bar");
      const settle = async () => act(async () => { await new Promise((done) => setTimeout(done, 10)); });

      // Esc while the model thinks: the stop goes out with the instruction's stop; the reply proposed a motion
      // (action_started: the worker read the stop and refused it), so nothing is withdrawn.
      await act(async () => fireEvent.keyDown(window, { key: "Escape" }));
      expect(posts).toEqual([{ url: "/ora/cell/stop-motion", body: {} },
        { url: "/api/play", body: { control: "stop_instruction" } }]);
      FakeEventSource.last!.emit({ type: "play", play: { ...running, state: "action_started", action: { action_id: "a1" } } });
      FakeEventSource.last!.emit({ type: "play", play: { ...running, state: "instruction_finished", running: null,
        instruction: { index: 0, ended: "operator_stopped" } } });
      await settle();
      expect(posts).toHaveLength(2);

      // Stop & send while the model thinks: the stop, then the steer; the agent answered in words (its message
      // was delivered, no motion came): the stop is withdrawn.
      const second = { ...running, running: "op-2", instructions: 2 };
      FakeEventSource.last!.emit({ type: "play", play: second });
      FakeEventSource.last!.emit({ type: "activity", activity: { kind: "thinking", elapsedMs: 100 } });
      fireEvent.click(screen.getByRole("radio", { name: "Stop & send" }));
      const box = screen.getByRole("textbox", { name: "Instruction" });
      fireEvent.change(box, { target: { value: "not that cup" } });
      await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
      expect(posts.slice(2)).toEqual([{ url: "/ora/cell/stop-motion", body: {} },
        { url: "/api/send", body: { text: "not that cup", as: "steer" } }]);
      FakeEventSource.last!.emit({ type: "play", play: { ...second, state: "message_delivered",
        message: { id: "m1", text: "someone else's", joined: true, observation_step: 3 } } });
      await settle();
      expect(posts).toHaveLength(4);
      FakeEventSource.last!.emit({ type: "play", play: { ...second, state: "message_delivered",
        message: { id: "m2", text: "not that cup", joined: true, observation_step: 3 } } });
      await settle();
      expect(posts.at(-1)).toEqual({ url: "/ora/cell/stop-motion", body: { withdraw: true } });
      expect(posts).toHaveLength(5);
    });
});

const LIMITS = { max_duration_s: 10, max_joint_travel_rad: 0.6, min_table_distance_m: 0.03, min_inter_arm_distance_m: 0.15 };

describe("the permission mode", () => {
  test("the footer shows the mode and switches it from its menu; auto only after its confirmation", async () => {
    await open(cellState({ pending: [], mode: { mode: "ask", by: null, stale: null, safe_limits: LIMITS } }));
    const button = await screen.findByTestId("term-mode");
    expect(button.textContent).toBe("▸▸ ask · every motion asked");
    expect(screen.getByTestId("term-footer").textContent).not.toContain("robot mode on");
    fireEvent.click(button);
    const menu = screen.getByRole("menu", { name: "Permission mode" });
    expect(menu.textContent).toContain("at most 10 s and 0.6 rad per joint, at least 3 cm above the table and 15 cm between the arms");
    await act(async () => fireEvent.click(screen.getByRole("menuitemradio", { name: /safe · small motions run/ })));
    expect(posts).toEqual([{ url: "/ora/cell/mode/safe", body: { confirmed: false } }]);
    expect(screen.queryByRole("menu")).toBeNull();

    fireEvent.click(screen.getByTestId("term-mode"));
    fireEvent.click(screen.getByRole("menuitemradio", { name: /auto · every motion runs/ }));
    expect(screen.getByRole("dialog").textContent).toContain("Switch to auto: every motion runs without asking?");
    fireEvent.click(screen.getByRole("button", { name: "No" }));
    expect(posts).toHaveLength(1);
    fireEvent.click(screen.getByTestId("term-mode"));
    fireEvent.click(screen.getByRole("menuitemradio", { name: /auto · every motion runs/ }));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Yes" })));
    expect(posts[1]).toEqual({ url: "/ora/cell/mode/auto", body: { confirmed: true } });
  });

  test("auto is shown apart; on a read-only rig it cannot be chosen; an older gateway keeps the old label", async () => {
    await open(cellState({ pending: [], mode: { mode: "auto", by: "web page", pid: 4242, falls_back_to: "ask", stale: null,
      safe_limits: LIMITS } }));
    const button = await screen.findByTestId("term-mode");
    expect(button.className).toContain("mode-auto");
    expect(button.getAttribute("title")).toContain("mode auto: every motion runs without asking (pid 4242, set by web page)");
    cleanup();
    await open(cellState({ pending: [], cell: { kind: "yam", rig_mode: "read", cell_config: null },
      mode: { mode: "safe", by: "cell board", stale: null, safe_limits: LIMITS } }));
    fireEvent.click(await screen.findByTestId("term-mode"));
    expect((screen.getByRole("menuitemradio", { name: /auto/ }) as HTMLButtonElement).disabled).toBe(true);
    cleanup();
    await open(cellState({ pending: [] }));
    await screen.findByTestId("cell-bar");
    expect(screen.queryByTestId("term-mode")).toBeNull();
    expect(screen.getByTestId("term-footer").textContent).toContain("robot mode on");
  });

  test("typed: /mode says the mode, /mode safe switches it, /mode auto is confirmed; outcomes and self-answered motions are worded", async () => {
    const by = "safe mode (2.5 s, 0.30 rad, 8 cm above the table, 40 cm between the arms)";
    await open(cellState({ pending: [], mode: { mode: "safe", by: "cell board", stale: null, safe_limits: LIMITS },
      self_answered: [{ id: "0000000a", kind: "motion", title: "左臂动作：下降 4 cm", by, answered_unix_s: NOW }] }));
    expect((await screen.findByTestId("cell-self-answered")).textContent).toContain(`ran without asking: 左臂动作：下降 4 cm · ${by}`);
    const box = screen.getByRole("textbox", { name: "Instruction" });
    fireEvent.change(box, { target: { value: "/mode" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(document.querySelector(".notices")!.textContent).toContain("mode safe: a small motion runs without asking");
    fireEvent.change(box, { target: { value: "/mode ask" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(posts).toEqual([{ url: "/ora/cell/mode/ask", body: { confirmed: false } }]);
    fireEvent.change(box, { target: { value: "/mode auto" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(screen.getByRole("dialog").textContent).toContain("Switch to auto");
    fireEvent.click(screen.getByRole("button", { name: "No" }));
    fireEvent.change(box, { target: { value: "/mode maybe" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(document.querySelector(".notices")!.textContent).toContain("/mode ask · /mode safe · /mode auto");
    expect(posts).toHaveLength(1);
    FakeEventSource.last!.emit({ type: "cell_command", entry: { id: "t-2", name: "mode", verb: "mode", ok: true,
      text: JSON.stringify({ mode: "ask", by: "play terminal", safe_limits: LIMITS }), seconds: 0.3, via: "play terminal", at: 9 } });
    await waitFor(() => expect(screen.getAllByTestId("cell-outcome").map((item) => item.textContent).join("\n"))
      .toContain("mode ask: every motion waits for your approval (set by play terminal)"));
  });
});

describe("a guided calibration's questions", () => {
  const ask = (kind: string, extra: Json = {}) => ({ ...QUESTION, kind, ...extra });

  test("a value question: Approve waits for its value, which is sent with the approval", async () => {
    await open(cellState({ pending: [ask("value", { title: "Square size", details: { unit: "mm", example: "20" } })] }));
    const card = await screen.findByTestId("cell-question");
    expect(card.textContent).toContain("A value");
    const approve = screen.getByRole("button", { name: "Approve…" }) as HTMLButtonElement;
    expect(approve.disabled).toBe(true);
    const box = screen.getByRole("textbox", { name: "Value" }) as HTMLInputElement;
    expect(box.placeholder).toBe("Value in mm, e.g. 20");
    fireEvent.change(box, { target: { value: " 20.5 " } });
    expect(approve.disabled).toBe(false);
    fireEvent.click(approve);
    expect(screen.getByRole("dialog").textContent).toContain("Value sent with the approval: 20.5");
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Yes" })));
    expect(posts).toEqual([{ url: "/ora/cell/approval/0a1b2c3d/approve", body: { confirmed: true, value: "20.5" } }]);
  });

  test("typed: /approve on a value question needs the value; a value for another question is refused", async () => {
    await open(cellState({ pending: [ask("value", { details: { example: "11x8" } })] }));
    await screen.findByTestId("cell-question");
    const box = screen.getByRole("textbox", { name: "Instruction" });
    fireEvent.change(box, { target: { value: "/approve" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.querySelector(".notices")!.textContent).toContain("this question asks for a value (e.g. 11x8)");
    fireEvent.change(box, { target: { value: "/approve 11x8" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    await act(async () => fireEvent.click(screen.getByRole("button", { name: "Yes" })));
    expect(posts).toEqual([{ url: "/ora/cell/approval/0a1b2c3d/approve", body: { confirmed: true, value: "11x8" } }]);
    cleanup();

    await open(cellState({ pending: [ask("task")] }));
    await screen.findByTestId("cell-question");
    const again = screen.getByRole("textbox", { name: "Instruction" });
    fireEvent.change(again, { target: { value: "/approve 20" } });
    await act(async () => fireEvent.keyDown(again, { key: "Enter" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.querySelector(".notices")!.textContent).toContain("this question (task) takes no value");
    expect(posts).toEqual([]);
  });

  test("a task's and a save's buttons say what they mean; a save's words are shown whole", async () => {
    await open(cellState({ pending: [ask("task", { title: "Place the board" })] }));
    await screen.findByTestId("cell-question");
    expect(screen.getByRole("button", { name: "Done…" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "End captures…" }));
    expect(screen.getByRole("dialog").textContent).toContain("Approve = done; decline = end the captures.");
    cleanup();

    const text = "Changes: wrist camera pose\nHeld-out error: 3.1 mm\nConcern: none";
    await open(cellState({ pending: [ask("write", { title: "Save the calibration?", text })] }));
    const card = await screen.findByTestId("cell-question");
    expect(card.querySelector(".cell-question-text")!.textContent).toBe(text);
    expect(card.textContent).toContain("Save the result");
    fireEvent.click(screen.getByRole("button", { name: "Save…" }));
    expect(screen.getByRole("dialog").textContent).toContain("Approve = save; decline = keep the old file.");
    fireEvent.click(screen.getByRole("button", { name: "No" }));
    expect(screen.getByRole("button", { name: "Don't save…" })).toBeTruthy();
    setLang("zh", false);
    await act(async () => undefined);
    expect(screen.getByRole("button", { name: "保存…" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "不保存…" })).toBeTruthy();
    setLang("en", false);
    expect(posts).toEqual([]);
  });
});

describe("a simulated run", () => {
  test("has none of it, and a typed cell command is answered as the terminal answers it", async () => {
    await open({ cell: null });
    expect(screen.queryByTestId("cell-bar")).toBeNull();
    expect(screen.queryByTestId("cell-question")).toBeNull();
    const box = screen.getByRole("textbox", { name: "Instruction" });
    fireEvent.change(box, { target: { value: "/estop" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(document.querySelector(".notices")!.textContent).toContain("/estop applies to real robot cells only");
    expect(posts).toEqual([]);
  });
});
