// @vitest-environment jsdom
// Free play's speed (src/lib/perf.ts, components/PerfPanel.tsx): the model's tok/s and time to the first
// token, each loop between robot actions broken into model thinking, tools, the approval wait, planning
// and motion and the next observation, and what the loop needs to meet the target; the page's reducer
// folds the run's timings in from its events, and a replay rebuilds the calls from the agent trace.

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import { PerfPanel } from "../src/components/PerfPanel.tsx";
import { setLang } from "../src/lib/i18n.ts";
import { adviceLines, advise, answeredBy, buildLoops, emptyPerf, median, storedTarget, summarize, timedAnswers, withCall,
  withObservations, withPlay, withRunPerf, type PerfLog } from "../src/lib/perf.ts";
import { initialState, reduce } from "../src/state.ts";
import { buildReplay } from "../server/replay.ts";
import { perfAction } from "../server/view.ts";
import { at, replayRun } from "./fixtures/replay-run.ts";

const B = 1_800_000_000_000;
const s = (seconds: number) => B + Math.round(seconds * 1000);

/** One loop: observed at 0 s, one call (0.2 → first token 0.6 → 5.0 s, 198 tokens: 45 tok/s), the action
 * 5.2 → 6.0 s with a person's approval 5.3 → 5.5 s, the next observation at 6.2 s. */
function oneLoop(offset = 0): PerfLog {
  let log = withRunPerf(emptyPerf, { observations: [[0, s(offset)], [1, s(offset + 6.2)]] });
  log = withCall(log, { id: `c${offset}`, step: 0, startedMs: s(offset + 0.2), firstTokenMs: s(offset + 0.6),
    endedMs: s(offset + 5), outputTokens: 198, inputTokens: 9000 });
  log = withPlay(log, { state: "action_started", action: { action_id: "a", observation_step: 0 } }, s(offset + 5.2));
  return withPlay(log, { state: "action_finished", action: { action_id: "a", observation_step: 0, duration_s: 0.8 } }, s(offset + 6));
}

const person = timedAnswers([{ id: "0a1b2c3d", kind: "motion", by: "play terminal", approved: true,
  created_unix_s: s(5.3) / 1000, answered_unix_s: s(5.5) / 1000 }]);

afterEach(() => {
  cleanup();
  setLang("en", false);
  try {
    localStorage.clear();
  } catch {
    // nothing kept
  }
});

describe("one loop", () => {
  test("is broken down from its observation to the next one", () => {
    const [loop] = buildLoops(oneLoop(), person);
    expect(loop.step).toBe(0);
    expect(loop.calls).toBe(1);
    expect(loop.segments).toEqual({ model: 4800, tools: 400, approval: 200, motion: 600, observe: 200 });
    expect(loop.totalMs).toBe(6200);
    expect(loop.generationMs).toBe(4400);
    expect([loop.approvalBy, loop.approved, loop.approvalKnown]).toEqual(["play terminal", true, true]);
  });

  test("without an approval record the action's time is all planning and motion; an unseen question is not known", () => {
    expect(buildLoops(oneLoop(), null)[0].segments).toMatchObject({ approval: null, motion: 800 });
    expect(buildLoops(oneLoop(), [])[0].segments).toMatchObject({ approval: 0, motion: 800 });
    const unseen = buildLoops(oneLoop(), [{ ...person[0], createdMs: null }])[0];
    expect([unseen.approvalKnown, unseen.segments.approval, unseen.segments.motion]).toEqual([false, null, 800]);
    const safe = timedAnswers([{ id: "0e0e0e0e", by: "safe mode (2.5 s)", approved: true, created_unix_s: s(5.4) / 1000,
      answered_unix_s: s(5.4) / 1000 }]);
    expect(buildLoops(oneLoop(), safe)[0].segments.approval).toBe(0);
    expect(answeredBy("safe mode (2.5 s)")).toBe("safe");
    expect(answeredBy("auto-approve (on)")).toBe("auto");
    expect(answeredBy("web page")).toBe("person");
  });

  test("an action still running is a loop in progress", () => {
    let log = withRunPerf(emptyPerf, { observations: [[0, s(0)]] });
    log = withPlay(log, { state: "action_started", action: { observation_step: 0 } }, s(3));
    const [loop] = buildLoops(log, null);
    expect([loop.complete, loop.totalMs, loop.segments.motion, loop.segments.tools]).toEqual([false, null, null, 3000]);
    // On a real cell with no answer yet, the motion may still wait for its approval: not known until it ends.
    expect(buildLoops(log, [])[0].segments.approval).toBeNull();
  });
});

describe("the summary and the advice", () => {
  test("says the model's speed, the median loop and the TPS the target needs", () => {
    const summary = summarize(oneLoop(), person, 4);
    expect(summary.window.tps).toBeCloseTo(45);
    expect(summary.window.ttftMs).toBe(400);
    expect(summary.callsPerAct).toBe(1);
    expect(summary.medians.totalMs).toBe(6200);
    const advice = summary.advice!;
    expect([advice.kind, advice.dominant, Math.round(advice.share * 100)]).toEqual(["over", "model", 77]);
    expect(advice.tps!.to).toBeCloseTo(90);
    expect(adviceLines(advice)).toEqual([
      "The median loop 6.2 s is 2.2 s over the 4.0 s target.",
      "Model thinking is 77% of the loop; TPS must go from 45 to 90 tok/s to meet 4.0 s.",
      "Assumes the other segments stay as they are and generation time scales with 1/TPS (first-token waits unchanged).",
    ]);
  });

  test("a target faster tokens cannot reach, another dominant segment, a target met", () => {
    const cannot = summarize(oneLoop(), person, 1).advice!;
    expect(cannot.tps!.to).toBeNull();
    expect(adviceLines(cannot)[1]).toBe("Model thinking is 77% of the loop, but faster tokens alone cannot meet 1.0 s: "
      + "without any generation the loop is still 1.8 s.");
    const waiting = advise({ segments: { model: 1000, tools: 0, approval: 4000, motion: 500, observe: 500 }, totalMs: 6000,
      generationMs: 800, count: 3 }, 50, 3000)!;
    expect(adviceLines(waiting).slice(1, 3)).toEqual([
      "Waiting for approval is 67% of the loop; it must drop from 4.0 s to 1.0 s to meet 3.0 s.",
      "Faster tokens alone cannot meet 3.0 s: without any generation the loop is still 5.2 s.",
    ]);
    expect(adviceLines(summarize(oneLoop(), person, 8).advice)).toEqual(["The median loop 6.2 s meets the 8.0 s target (1.8 s to spare)."]);
    expect(adviceLines(null)).toEqual(["No complete loop yet: an action and the observation after it."]);
    expect(median([5, 1])).toBe(3);
  });
});

describe("the page's record", () => {
  test("the reducer folds in the run's perf, each call, each action and each observation message", () => {
    let page = reduce(initialState, { type: "event", event: { type: "snapshot", run: { mode: "play", perf: {
      calls: [{ id: "c0", step: 0, startedMs: s(0.2), firstTokenMs: s(0.6), endedMs: s(5), outputTokens: 198, inputTokens: 1 }],
      observations: [], actions: [] } }, messages: [
      { role: "custom", customType: "ora.play_observation", details: { ora_observation_step: 0 }, timestamp: s(0) }] } });
    page = reduce(page, { type: "event", event: { type: "play", play: { state: "action_started",
      action: { action_id: "a", observation_step: 0 } }, at: s(5.2) } });
    page = reduce(page, { type: "event", event: { type: "play", play: { state: "action_finished",
      action: { action_id: "a", observation_step: 0, duration_s: 0.8 } }, at: s(6) } });
    page = reduce(page, { type: "event", event: { type: "model_call", call: { id: "c1", step: 1, startedMs: s(6.5),
      firstTokenMs: null, endedMs: s(7), outputTokens: 3, inputTokens: 1 } } });
    page = reduce(page, { type: "event", event: { type: "message", message: { role: "custom", customType: "ora.play_observation",
      details: { ora_observation_step: 1 }, timestamp: s(6.2) } } });
    expect(page.perf.calls.map((call) => call.id)).toEqual(["c0", "c1"]);
    expect(buildLoops(page.perf, null)[0].totalMs).toBe(6200);
    expect(withObservations(page.perf, [{ role: "custom", customType: "ora.play_observation",
      details: { ora_observation_step: 1 }, timestamp: s(99) }]).observations.get(1)).toBe(s(6.2));
  });

  test("the web server keeps the run's perf current with each action", () => {
    const started = perfAction(null, { state: "action_started", action: { action_id: "a", observation_step: 3 } }, 100);
    const finished = perfAction(started, { state: "action_finished", action: { action_id: "a", observation_step: 3,
      duration_s: 1.5 } }, 1700);
    expect(finished.actions).toEqual([{ step: 3, actionId: "a", startedMs: 100, finishedMs: 1700, durationS: 1.5 }]);
    expect(perfAction(finished, { state: "instruction" }, 5)).toBe(finished);
  });

  test("a replay rebuilds each model call from the agent trace, without a first-token time", async () => {
    const script = await buildReplay(replayRun());
    const calls = script.events.filter((item) => item.event.type === "model_call");
    expect(calls.map((item) => [item.at, item.event.call.id, item.event.call.outputTokens, item.event.call.firstTokenMs])).toEqual([
      [at(14), "play-call-000000", 150, null], [at(18), "play-call-000001", 50, null]]);
    let page = initialState;
    for (const item of script.events) page = reduce(page, { type: "event", event: item.event, now: item.at });
    const [loop] = buildLoops(page.perf, null);
    // Observed at 10 s, the action submitted at 19 s and its receipt at 21 s, the next observation at 23 s.
    expect([loop.segments.model, loop.segments.tools, loop.segments.motion, loop.segments.observe, loop.totalMs])
      .toEqual([5000, 4000, 2000, 2000, 13000]);
    expect(summarize(page.perf, null, 3).timedTokens).toBe(false);
  });
});

describe("the panel", () => {
  test("one line at the foot; opened, the calls, both bars, the advice; the target is the operator's", () => {
    render(<PerfPanel perf={oneLoop()} answers={person} />);
    expect(screen.getByTestId("perf-strip").textContent).toContain("45 tok/s · TTFT 0.4 s · loop 6.2 s (target 3.0 s)");
    expect(screen.queryByTestId("perf-drawer")).toBeNull();
    fireEvent.click(screen.getByTestId("perf-strip"));
    expect(screen.getByTestId("perf-latest").querySelectorAll("[data-segment]")).toHaveLength(5);
    expect(screen.getByTestId("perf-median")).toBeTruthy();
    expect(screen.getByText("approved by play terminal")).toBeTruthy();
    expect(screen.getByTestId("perf-advice").textContent).toContain("TPS must go from 45 to");
    const input = screen.getByTestId("perf-target") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "4" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(screen.getByTestId("perf-advice").textContent).toContain("TPS must go from 45 to 90 tok/s to meet 4.0 s.");
    expect(storedTarget()).toBe(4);
  });

  test("in Chinese, and before any call", () => {
    setLang("zh", false);
    render(<PerfPanel perf={emptyPerf} answers={null} replay />);
    expect(screen.getByTestId("perf-strip").textContent).toContain("还没有计时的模型调用");
    fireEvent.click(screen.getByTestId("perf-strip"));
    expect(screen.getByText("还没有动作。")).toBeTruthy();
    expect(screen.getByTestId("perf-drawer").textContent).toContain("回放");
  });
});

test("the Supervisor's approval wait is exact without any answer the page saw", () => {
  let log = withRunPerf(emptyPerf, { observations: [[0, s(0)], [1, s(6.2)]] });
  log = withCall(log, { id: "c", step: 0, startedMs: s(0.2), firstTokenMs: s(0.6), endedMs: s(5), outputTokens: 198,
    inputTokens: 9000 });
  log = withPlay(log, { state: "action_started", action: { action_id: "a", observation_step: 0,
    started_unix_s: s(5.2) / 1000 } }, s(9));
  log = withPlay(log, { state: "action_finished", action: { action_id: "a", observation_step: 0, duration_s: 0.8,
    started_unix_s: s(5.2) / 1000, finished_unix_s: s(6) / 1000, approval_wait_s: 0.3 } }, s(9));
  const [loop] = buildLoops(log, null);
  expect(loop.startedMs).toBe(s(5.2));
  expect(loop.finishedMs).toBe(s(6));
  expect(loop.approvalKnown).toBe(true);
  expect(loop.segments.approval).toBe(300);
  expect(loop.segments.motion).toBe(500);
});
