// Free play's speed for the operator (src/play/perf.ts): each model call timed on the core's clock
// (request, first content, end, output tokens), actions stamped on arrival, and the footer's line.

import assert from "node:assert/strict";
import { test } from "node:test";

import { loopTargetS, loops, median, perfLine, PlayPerf, tokensPerSecond } from "../src/play/perf.ts";

function clocked() {
  let now = 1_000_000;
  return { at: (ms: number) => { now = ms; }, clock: () => now };
}

test("a call is timed from its request to its first content and its end, with its output tokens", () => {
  const time = clocked();
  const perf = new PlayPerf(time.clock);
  const heard: unknown[] = [];
  perf.onCall = (call) => heard.push(call);
  time.at(1000);
  perf.requested("play-call-000001", 4);
  time.at(1500);
  perf.streamed({ type: "start" });                 // headers in: not yet a token
  time.at(1800);
  perf.streamed({ type: "thinking_start", contentIndex: 0 });
  time.at(2000);
  perf.streamed({ type: "text_delta", contentIndex: 1, delta: "x" });
  time.at(3800);
  const call = perf.replied({ role: "assistant", stopReason: "toolUse", usage: { input: 900, cacheRead: 100, output: 90 } });
  assert.deepEqual(call, { id: "play-call-000001", step: 4, startedMs: 1000, firstTokenMs: 1800, endedMs: 3800,
    outputTokens: 90, inputTokens: 1000, cachedInputTokens: 100, stopReason: "toolUse" });
  assert.deepEqual(heard, [call]);
  assert.equal(tokensPerSecond([call!]), 45);
  assert.equal(perf.replied({ role: "assistant" }), null, "no request open: nothing to time");
});

test("the loop runs from an observation to the next one after the action finished", () => {
  const time = clocked();
  const perf = new PlayPerf(time.clock);
  time.at(0);
  perf.observed(0);
  time.at(100);
  perf.requested("c0", 0);
  time.at(600);
  perf.streamed({ type: "toolcall_start" });
  time.at(1100);
  perf.replied({ usage: { output: 50 } });
  time.at(1500);
  assert.equal(perf.announced({ state: "action_started", action: { action_id: "a0", observation_step: 0 } }), 1500);
  time.at(4500);
  perf.announced({ state: "action_finished", action: { action_id: "a0", observation_step: 0, duration_s: 2.9 } });
  time.at(5200);
  perf.observed(1);
  const snapshot = perf.snapshot();
  assert.deepEqual(snapshot.actions, [{ step: 0, actionId: "a0", startedMs: 1500, finishedMs: 4500, durationS: 2.9,
    approvalWaitS: null }]);
  assert.deepEqual(loops(snapshot), [{ step: 0, ms: 5200 }]);
  assert.equal(perfLine(snapshot, 3), "100 tok/s · TTFT 0.5 s · loop 5.2 s (target 3 s)");
  perf.announced({ state: "instruction", instructions: 1 });
  assert.equal(perf.snapshot().actions.length, 1, "other announcements change nothing");
});

test("before any loop the footer still says the model's speed; before any call it says nothing", () => {
  assert.equal(perfLine({ calls: [], observations: [], actions: [] }, 3), null);
  const call = { id: "c", step: 0, startedMs: 0, firstTokenMs: null, endedMs: 900, outputTokens: 0, inputTokens: 0,
    cachedInputTokens: 0, stopReason: "error" };
  assert.equal(perfLine({ calls: [call], observations: [[0, 0]], actions: [] }, 2.5), "— tok/s · loop — (target 2.5 s)");
});

test("median and the loop target", () => {
  assert.equal(median([]), null);
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(loopTargetS({}), 3);
  assert.equal(loopTargetS({ ORA_LOOP_TARGET_S: "1.5" }), 1.5);
  assert.equal(loopTargetS({ ORA_LOOP_TARGET_S: "fast" }), 3);
});

test("the Supervisor's own clock and approval wait win over the arrival time", () => {
  const perf = new PlayPerf(() => 9_999_000);
  perf.announced({ state: "action_started", action: { action_id: "a1", observation_step: 3, started_unix_s: 100.5 } });
  perf.announced({ state: "action_finished", action: { action_id: "a1", observation_step: 3, duration_s: 2,
    started_unix_s: 100.5, finished_unix_s: 102.5, approval_wait_s: 1.25 } });
  assert.deepEqual(perf.snapshot().actions, [{ step: 3, actionId: "a1", startedMs: 100_500, finishedMs: 102_500,
    durationS: 2, approvalWaitS: 1.25 }]);
});
