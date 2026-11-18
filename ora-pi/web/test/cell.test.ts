// A real robot cell's commands on the page (src/lib/cell.ts): the same commands as free play's terminal
// (ora-pi/src/play/cell.ts), asked first where the terminal asks y/N; typed, they act on the cell on a
// real-cell run and are answered as the terminal answers them on any other.

import { describe, expect, test } from "vitest";

import { APPROVAL_COMMANDS, CELL_COMMANDS, HALT_COMMAND, MODE_COMMAND, nextMode as terminalNextMode, PERMISSION_MODES as TERMINAL_MODES,
  YOLO_COMMAND } from "../../src/play/cell.ts";
import { answerRefusal, CELL_ANSWERS, CELL_HALT, CELL_MODE, CELL_SWITCH, CELL_VERBS, cellIntent, mergeOutcomes, modeStateOf, modeStateText,
  nextMode, oldestQuestion, PAGE_CELL_COMMANDS, PERMISSION_MODES, type CellState, type ModeState } from "../src/lib/cell.ts";
import { outgoing } from "../src/lib/composer.ts";
import { setLang } from "../src/lib/i18n.ts";

setLang("en", false);

const shape = (commands: readonly { name: string; verb: string; confirm: boolean }[]) =>
  commands.map(({ name, verb, confirm }) => ({ name, verb, confirm }));

describe("parity with the terminal", () => {
  test("every cell command of the terminal has the page's counterpart, asked first exactly where the terminal asks", () => {
    // A command added to play/cell.ts fails here until the page (lib/cell.ts, CellPanel, App) and the gateway
    // (ora/app/web_cell.py, guarded by tests/ora/app/test_web_cell.py) offer it too.
    expect(shape(CELL_VERBS)).toEqual(shape(CELL_COMMANDS));
    expect(shape(CELL_ANSWERS)).toEqual(shape(APPROVAL_COMMANDS));
    expect(shape([CELL_SWITCH])).toEqual(shape([YOLO_COMMAND]));
    expect(shape([CELL_MODE])).toEqual(shape([MODE_COMMAND]));
    expect(shape([CELL_HALT])).toEqual(shape([HALT_COMMAND]));
    expect(PAGE_CELL_COMMANDS.map((command) => command.name).sort())
      .toEqual([...CELL_COMMANDS, ...APPROVAL_COMMANDS, YOLO_COMMAND, MODE_COMMAND, HALT_COMMAND].map((command) => command.name).sort());
  });

  test("the permission mode is cycled as the terminal cycles it and said in plain words, safe with its limits", () => {
    expect([...PERMISSION_MODES]).toEqual([...TERMINAL_MODES]);
    for (const mode of PERMISSION_MODES) for (const rig of ["on", "read"]) expect(nextMode(mode, rig)).toBe(terminalNextMode(mode, rig));
    const limits = { max_duration_s: 10, max_joint_travel_rad: 0.6, min_table_distance_m: 0.03, min_inter_arm_distance_m: 0.15 };
    const states: ModeState[] = [{ mode: "ask", by: null, safe_limits: limits }, { mode: "safe", by: "cell board", safe_limits: limits },
      { mode: "auto", by: "web page", pid: 4242, falls_back_to: "safe", safe_limits: limits, stale: null }];
    setLang("zh", false);
    try {
      expect(modeStateText(states[0])).toBe("模式 ask（每个动作都问）：每个动作都等你批准；给手臂通电和释放手臂总会询问");
    } finally {
      setLang("en", false);
    }
    expect(modeStateText(states[1])).toBe("mode safe: a small motion runs without asking (at most 10 s and 0.6 rad per joint, "
      + "at least 3 cm above the table and 15 cm between the arms); any other waits for your approval (set by cell board); "
      + "enabling and releasing the arms are always asked");
    expect(modeStateOf(JSON.stringify(states[2]))?.mode).toBe("auto");
    expect(modeStateOf("{\"mode\": \"maybe\"}")).toBeNull();
  });
});

describe("typed commands", () => {
  test("a typed command is read as the terminal reads it", () => {
    expect(cellIntent("/home")).toMatchObject({ kind: "verb", command: { verb: "home", confirm: false } });
    expect(cellIntent("/estop")).toMatchObject({ kind: "verb", command: { verb: "estop", confirm: true } });
    expect(cellIntent("/decline  too close to the cup "))
      .toEqual({ kind: "answer", verb: "decline", reason: "too close to the cup", value: null });
    expect(cellIntent("/decline")).toEqual({ kind: "answer", verb: "decline", reason: null, value: null });
    expect(cellIntent("/approve  20 ")).toEqual({ kind: "answer", verb: "approve", reason: null, value: "20" });
    expect(cellIntent("/approve")).toEqual({ kind: "answer", verb: "approve", reason: null, value: null });
    expect(cellIntent("/yolo")).toEqual({ kind: "switch", state: "status" });
    expect(cellIntent("/yolo ON")).toEqual({ kind: "switch", state: "on" });
    expect(cellIntent("/yolo maybe")).toEqual({ kind: "usage", name: "yolo" });
    expect(cellIntent("/mode")).toEqual({ kind: "mode", mode: "status" });
    expect(cellIntent("/mode SAFE")).toEqual({ kind: "mode", mode: "safe" });
    expect(cellIntent("/mode yolo")).toEqual({ kind: "usage", name: "mode" });
    expect(cellIntent("/halt")).toEqual({ kind: "halt" });
    expect(cellIntent("/stop")).toBeNull();
    expect(cellIntent("home the arms")).toBeNull();
  });

  test("on a real cell they act on it; on any other run they are refused as the terminal refuses them", () => {
    expect(outgoing("/release", { play: true, streaming: false, cell: true }))
      .toMatchObject({ kind: "cell", intent: { kind: "verb", command: { name: "release" } } });
    const refused = outgoing("/approve", { play: true, streaming: false, cell: false });
    expect(refused).toEqual({ kind: "refused", reason: expect.stringContaining("/approve applies to real robot cells only") });
    expect(outgoing("/estop", { play: false, streaming: false })).toMatchObject({ kind: "refused" });
    expect(outgoing("/reset", { play: true, streaming: false, cell: true })).toEqual({ kind: "control", control: "reset" });
    expect(outgoing("pick up the cup", { play: true, streaming: false, cell: true }))
      .toEqual({ kind: "message", text: "pick up the cup", as: "prompt" });
  });

  test("/approve and /decline answer the oldest question still open", () => {
    const question = (id: string, created: number, expires: number) => ({ id, kind: "motion", title: id, text: "",
      created_unix_s: created, expires_unix_s: expires, pid: 1 });
    const state: CellState = { cell: { kind: "yam", rig_mode: "on", cell_config: null },
      pending: [question("bbbbbbbb", 20, 200), question("aaaaaaaa", 10, 50), question("cccccccc", 5, 300)] };
    expect(oldestQuestion(state, 100)?.id).toBe("cccccccc");
    expect(oldestQuestion(state, 400)).toBeNull();
    expect(oldestQuestion(null, 0)).toBeNull();
  });

  test("a value question's approval needs its value and no other question takes one, as in the terminal", () => {
    const question = (kind: string, details: object = {}) => ({ id: "0a1b2c3d", kind, title: "", text: "", details,
      created_unix_s: 0, expires_unix_s: 100, pid: 1 });
    const value = question("value", { unit: "mm", example: "20" });
    expect(answerRefusal(value, "approve", " ")).toBe(
      "/approve: this question asks for a value (in mm, e.g. 20): /approve <value>. Nothing was run.");
    expect(answerRefusal(value, "approve", "20")).toBeNull();
    expect(answerRefusal(value, "decline", null)).toBeNull();
    expect(answerRefusal(question("task"), "approve", "20"))
      .toBe("/approve: this question (task) takes no value; send /approve alone. Nothing was run.");
    expect(answerRefusal(question("motion"), "approve", null)).toBeNull();
  });

  test("the page's outcomes and the run's are listed once each, in order", () => {
    const item = (id: string, at: number) => ({ id, name: "home", verb: "home", ok: true, text: "", seconds: 1, via: "web page", at });
    expect(mergeOutcomes([item("a", 2), item("b", 5)], [item("b", 5), item("c", 3)]).map((entry) => entry.id))
      .toEqual(["a", "c", "b"]);
  });
});
