// @vitest-environment jsdom
// The composer as Pi's terminal has it: "/" lists the commands this page takes (each with its line, narrowed as
// the name is typed, ↑/↓, Tab, Enter, Esc, a click), and only those; while a free-play instruction runs, the send
// modes say what Enter does with a message: steer it in, queue it as the next instruction, or stop and send.

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test, vi } from "vitest";

import { Composer } from "../src/components/Composer.tsx";
import { PAGE_CELL_COMMANDS } from "../src/lib/cell.ts";
import { outgoing } from "../src/lib/composer.ts";
import { lang, setLang, t, type Key } from "../src/lib/i18n.ts";
import { CELL_SLASH_NAMES, slashCommands, slashSuggestions } from "../src/lib/slash.ts";

setLang("en", false);
afterEach(cleanup);

const names = (text: string, options = { play: true, cell: false }) => slashSuggestions(text, options).map((item) => item.label);

describe("the slash list", () => {
  test("lists what the page takes: free play's controls and /stop, a real cell's commands only on a real cell", () => {
    expect(names("/")).toEqual(["/reset", "/quit", "/stop"]);
    expect(names("/", { play: false, cell: false })).toEqual(["/stop"]);
    expect(names("/", { play: false, cell: true })).toEqual(["/stop"]);
    expect(names("/", { play: true, cell: true })).toEqual(["/reset", "/quit", "/stop", "/halt", "/approve", "/decline", "/mode",
      "/yolo", "/home", "/hold", "/diagnose", "/float", "/release", "/estop"]);
    expect([...CELL_SLASH_NAMES].sort()).toEqual(PAGE_CELL_COMMANDS.map((command) => command.name).sort());
    // Everything listed is acted on (never answered "not here").
    for (const spec of slashCommands({ play: true, cell: true })) {
      expect(outgoing(`/${spec.name}`, { play: true, streaming: false, cell: true })?.kind).not.toBe("refused");
    }
  });

  test("narrowed as the name is typed, then a command's own words; other text lists nothing", () => {
    const cell = { play: true, cell: true };
    expect(names("/ho", cell)).toEqual(["/home", "/hold"]);
    expect(names("/st", cell)).toEqual(["/stop", "/estop"]);
    expect(names("/mode ", cell)).toEqual(["ask", "safe", "auto"]);
    expect(slashSuggestions("/mode s", cell).map((item) => item.insert)).toEqual(["/mode safe"]);
    expect(names("/yolo o", cell)).toEqual(["on", "off"]);
    expect(names("/approve ", cell)).toEqual([]);
    expect(names("/nope", cell)).toEqual([]);
    expect(names("pick up /home", cell)).toEqual([]);
    expect(names("/home\nnext", cell)).toEqual([]);
    const approve = slashSuggestions("/appr", cell)[0];
    expect(approve).toMatchObject({ insert: "/approve ", label: "/approve", args: "[value]" });
    expect(approve.description).toMatch(/approve the robot's waiting question/);
  });

  test("every line has its English and Chinese words", () => {
    const keys: string[] = [];
    for (const item of slashSuggestions("/", { play: true, cell: true })) keys.push(item.description);
    const previous = lang();
    try {
      for (const language of ["en", "zh"] as const) {
        setLang(language, false);
        for (const spec of slashCommands({ play: true, cell: true })) {
          const key = `slash.${spec.name}` as Key;
          expect(t(key)).not.toBe(key);
          for (const word of spec.words ?? []) expect(t(`slash.${spec.name}.${word}` as Key)).not.toBe(`slash.${spec.name}.${word}`);
          if (spec.args) expect(t(spec.args)).not.toBe(spec.args);
        }
      }
    } finally {
      setLang(previous, false);
    }
    expect(keys).toHaveLength(14);
  });
});

function composer(props: Partial<Parameters<typeof Composer>[0]> = {}) {
  const handlers = { onSend: vi.fn(async () => true), onControl: vi.fn(), onCell: vi.fn(), onStop: vi.fn(), onNotice: vi.fn(),
    onRefused: vi.fn() };
  const all = { play: true, phase: "running" as const, streaming: true, closed: false, draft: null, onStopRun: () => undefined,
    cell: true, realCell: true, ...handlers, ...props };
  const view = render(<Composer {...all} />);
  return { ...handlers, box: screen.getByRole("textbox") as HTMLTextAreaElement,
    rerender: (next: Partial<Parameters<typeof Composer>[0]>) => view.rerender(<Composer {...all} {...next} />) };
}

const options = () => screen.queryAllByRole("option").map((option) => option.querySelector(".slash-name")?.textContent);
const chosen = () => screen.getByRole("option", { selected: true }).querySelector(".slash-name")?.textContent;

describe("the composer's slash list", () => {
  test("/ opens it; ↑/↓ move, Tab completes, Enter completes a partial name and runs a full one", async () => {
    const { box, onControl, onCell } = composer({ phase: "awaiting_instruction" });
    expect(screen.queryByTestId("slash")).toBeNull();
    fireEvent.change(box, { target: { value: "/" } });
    expect(options()[0]).toBe("/reset");
    expect(options()).toContain("/halt");
    expect(chosen()).toBe("/reset");
    fireEvent.keyDown(box, { key: "ArrowUp" });
    expect(chosen()).toBe("/estop");
    fireEvent.keyDown(box, { key: "ArrowDown" });
    fireEvent.keyDown(box, { key: "ArrowDown" });
    expect(chosen()).toBe("/quit");

    fireEvent.change(box, { target: { value: "/mo" } });
    fireEvent.keyDown(box, { key: "Tab" });
    expect(box.value).toBe("/mode ");
    expect(options()).toEqual(["ask", "safe", "auto"]);
    fireEvent.keyDown(box, { key: "ArrowDown" });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(box.value).toBe("/mode safe");
    expect(onCell).not.toHaveBeenCalled();
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(onCell).toHaveBeenLastCalledWith({ kind: "mode", mode: "safe" });

    fireEvent.change(box, { target: { value: "/res" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(box.value).toBe("/reset");
    expect(onControl).not.toHaveBeenCalled();
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(onControl).toHaveBeenLastCalledWith("reset");
  });

  test("a click completes; Esc closes the list and leaves the running instruction alone", () => {
    const { box, onControl } = composer();
    fireEvent.change(box, { target: { value: "/h" } });
    expect(options()).toEqual(["/halt", "/home", "/hold"]);
    fireEvent.click(screen.getAllByRole("option")[2]);
    expect(box.value).toBe("/hold");
    fireEvent.change(box, { target: { value: "/" } });
    fireEvent.keyDown(box, { key: "Escape" });
    expect(screen.queryByTestId("slash")).toBeNull();
    expect(onControl).not.toHaveBeenCalled();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onControl).toHaveBeenLastCalledWith("stop_instruction");
    fireEvent.change(box, { target: { value: "/q" } });
    expect(options()).toEqual(["/quit"]);
  });

  test("a run that drives no real cell never lists a cell's commands", () => {
    const { box } = composer({ realCell: false, cell: false, play: false, phase: null });
    fireEvent.change(box, { target: { value: "/" } });
    expect(options()).toEqual(["/stop"]);
  });
});

describe("the send modes while an instruction runs", () => {
  test("Steer by default; Queue sends it for the run to hold until the instruction ends; Alt+Enter queues", async () => {
    const { box, onSend, onNotice, rerender } = composer({ realCell: false });
    expect(screen.getByRole("radio", { name: "Steer" }).getAttribute("aria-checked")).toBe("true");
    fireEvent.change(box, { target: { value: "slower" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(onSend).toHaveBeenLastCalledWith({ kind: "message", text: "slower", as: "steer" });

    fireEvent.click(screen.getByRole("radio", { name: "Queue" }));
    expect(screen.getByRole("button", { name: /Queue$/ })).toBeTruthy();
    fireEvent.change(box, { target: { value: "then the blue one" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(onSend).toHaveBeenLastCalledWith({ kind: "message", text: "then the blue one", as: "queue" });
    fireEvent.change(box, { target: { value: "then stack them" } });
    fireEvent.click(screen.getByRole("radio", { name: "Steer" }));
    await act(async () => fireEvent.keyDown(box, { key: "Enter", altKey: true }));
    expect(onSend).toHaveBeenLastCalledWith({ kind: "message", text: "then stack them", as: "queue" });
    expect(onSend).toHaveBeenCalledTimes(3);
    expect(onNotice).toHaveBeenCalledTimes(2);
    expect(box.value).toBe("");

    // What the run holds is shown above the box, until the instruction it opens starts; nothing is held here.
    await act(async () => rerender({ queued: [{ key: "q1", text: "then the blue one" }, { key: "q2", text: "then stack them" }] }));
    expect(screen.getAllByTestId("queued").map((row) => row.querySelector(".queued-text")?.textContent))
      .toEqual(["then the blue one", "then stack them"]);
    await act(async () => rerender({ phase: "awaiting_instruction", streaming: false, queued: [] }));
    expect(onSend).toHaveBeenCalledTimes(3);
    expect(screen.queryAllByTestId("queued")).toHaveLength(0);
  });

  test("a refused queued message comes back into the box", async () => {
    const { box, onSend } = composer({ realCell: false });
    onSend.mockImplementationOnce(async () => false);
    fireEvent.change(box, { target: { value: "first" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter", altKey: true }));
    expect(box.value).toBe("first");
  });

  test("Stop & send on a real cell stops the robot (a motion moving or being planned) and steers the message in", async () => {
    const { box, onSend, onStop, onControl } = composer();
    fireEvent.click(screen.getByRole("radio", { name: "Stop & send" }));
    expect(screen.getByRole("radio", { name: "Stop & send" }).getAttribute("title")).toMatch(/ends at once where the arms are.*still planning is refused/);
    fireEvent.change(box, { target: { value: "not that cup, the red one" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(onStop).toHaveBeenCalledWith("not that cup, the red one");
    expect(onSend).toHaveBeenLastCalledWith({ kind: "message", text: "not that cup, the red one", as: "steer" });
    expect(onControl).not.toHaveBeenCalled();
  });

  test("Stop & send without a real cell queues the message and stops the instruction: it opens the next one", async () => {
    const { box, onSend, onStop, onControl, rerender } = composer({ realCell: false, cell: false });
    fireEvent.click(screen.getByRole("radio", { name: "Stop & send" }));
    fireEvent.change(box, { target: { value: "put it back" } });
    await act(async () => fireEvent.keyDown(box, { key: "Enter" }));
    expect(onSend).toHaveBeenLastCalledWith({ kind: "message", text: "put it back", as: "queue" });
    expect(onControl).toHaveBeenLastCalledWith("stop_instruction");
    expect(onStop).not.toHaveBeenCalled();
    // The next instruction starts with Steer again.
    await act(async () => rerender({ phase: "awaiting_instruction", streaming: false }));
    await act(async () => rerender({ phase: "running", streaming: true }));
    expect(screen.getByRole("radio", { name: "Steer" }).getAttribute("aria-checked")).toBe("true");
  });

  test("with no instruction running they collapse to Send", () => {
    composer({ phase: "awaiting_instruction", streaming: false });
    expect(screen.queryByTestId("send-modes")).toBeNull();
    expect(screen.getByRole("button", { name: /^Send$/ })).toBeTruthy();
  });
});
