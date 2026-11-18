// Free play's truncated replies (src/play/driver.ts `afterTurn`): a reply the provider's output limit cut
// off before any tool call is no report. As an episode does (core/session.ts `endOfReplyInner` with
// `afterTruncatedOutput`), the model goes on with the same observation after ORA's continuation; when the
// act's budget leaves no call for that, the act ends as `output_truncated`, never `policy_no_action`.

import assert from "node:assert/strict";
import { test } from "node:test";

import { PLAY_TRUNCATED_CONTINUATION, PlayDriver } from "../src/play/driver.ts";

type Json = any;

function fakeCore() {
  const events: Json[] = [];
  const core: Json = {
    config: { tools: { definitions: [], typed_effects: false, mutation_tools: [] } },
    interrupted: false, events,
    rpc: { request: async () => null },
    append: (type: string, fields: Json) => {
      const event = { type, event_id: `event-${events.length}`, ...fields };
      events.push(event);
      return event;
    },
    flush: () => undefined,
  };
  return core;
}

function driverWith(core: Json, fields: Json = {}): Json {
  const driver: Json = new PlayDriver(core, { workspace: "/tmp", auth_path: "/tmp/auth.json", agent_dir: null });
  driver.turn = {
    step: 9, observationEvent: { event_id: "event-observation" }, calls: 3, callLimit: 80, deadlineMs: null,
    outcome: null, submitted: null, modelCallId: "play-call-000045", lastStopReason: "length", lastError: null,
    ...fields,
  };
  return driver;
}

const cutOff = (content: Json[] = [{ type: "thinking", thinking: "x".repeat(50) }]): Json =>
  ({ role: "assistant", stopReason: "length", content });

test("cut off in its reasoning: recorded, and the model continues on the same observation", () => {
  const core = fakeCore();
  const driver = driverWith(core);
  const result = driver.afterTurn(cutOff());
  assert.equal(result.continue, true);
  assert.equal(result.entries.length, 1);
  assert.deepEqual(result.entries[0], {
    type: "custom_message", customType: "ora.continuation", display: true, content: PLAY_TRUNCATED_CONTINUATION,
    details: { ora_continuation: true, ora_reason: "output_truncated", ora_event_id: "event-0" },
  });
  assert.deepEqual(core.events, [{
    type: "provider_output_truncated", event_id: "event-0", visibility: "derived", observation_step: 9,
    model_call_id: "play-call-000045",
    payload: { tool_call_count: 0, text_chars: 0, play: true, policy_outcome: "continue_same_observation" },
  }]);
  assert.equal(driver.turn.outcome, null, "the act goes on: the next call's boundary checks the inbox and budget");
});

test("a joined message the cut-off call read first is answered by the call that continues it", () => {
  const driver = driverWith(fakeCore());
  driver.firstRead.set("m1", 45);
  driver.firstRead.set("m0", 40);
  driver.callSequence = 46;
  driver.afterTurn(cutOff());
  assert.deepEqual([...driver.firstRead], [["m1", 46], ["m0", 40]]);
  driver.turn.modelCallId = "play-call-000046";
  driver.turn.lastStopReason = "stop";
  assert.deepEqual(driver.finish().answers, ["m1"]);
});

test("cut off with words but no tool call is continued too, as an episode does", () => {
  const core = fakeCore();
  const driver = driverWith(core);
  assert.equal(driver.afterTurn(cutOff([{ type: "text", text: "Done — here is where things st" }])).continue, true);
  assert.equal(core.events[0].payload.text_chars, 30);
});

test("with the act's model calls spent the act ends as output_truncated, not as the agent's report", () => {
  const core = fakeCore();
  const driver = driverWith(core, { calls: 80, callLimit: 80 });
  assert.equal(driver.afterTurn(cutOff()), undefined);
  assert.equal(core.events[0].payload.policy_outcome, "output_truncated");
  assert.deepEqual(driver.finish(), { submitted: false, termination: "output_truncated", attempts: 80 });
});

test("with the act's time spent the act ends as output_truncated", () => {
  const core = fakeCore();
  const driver = driverWith(core, { deadlineMs: performance.now() - 1 });
  assert.equal(driver.afterTurn(cutOff()), undefined);
  assert.equal(driver.finish().termination, "output_truncated");
});

test("a run that ended on a cut-off reply without a continuation is never the report", () => {
  const driver = driverWith(fakeCore());
  assert.deepEqual(driver.finish(), { submitted: false, termination: "output_truncated", attempts: 3 });
});

test("not a truncation: a finished reply, a cut-off tool call (Pi fails it and goes on), a stopped act", () => {
  const core = fakeCore();
  assert.equal(driverWith(core).afterTurn({ role: "assistant", stopReason: "stop", content: [] }), undefined);
  assert.equal(driverWith(core).afterTurn(cutOff([{ type: "toolCall", id: "c", name: "move_pose", arguments: {} }])),
    undefined);
  assert.equal(driverWith(core, { outcome: { submitted: false, termination: "operator_interrupt", attempts: 3 } })
    .afterTurn(cutOff()), undefined);
  core.interrupted = true;
  assert.equal(driverWith(core).afterTurn(cutOff()), undefined);
  assert.equal(core.events.length, 0);
});
