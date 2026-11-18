// Free play's act boundary (src/play/driver.ts): once the act's one world-changing action is submitted,
// the act ends before any further model call. An operator message steered into the run while that
// action was being proposed makes Pi go on past the terminating tool batch; without this stop the run
// kept calling the model and every further action was refused as a second one for the observation.

import assert from "node:assert/strict";
import { test } from "node:test";

import { PlayDriver } from "../src/play/driver.ts";

type Json = any;

function fakeCore() {
  const polls: Json[] = [];
  const core: Json = {
    config: { tools: { definitions: [], typed_effects: false, mutation_tools: [] } },
    interrupted: false,
    polls,
    rpc: { request: async (method: string, params: Json) => { polls.push({ method, params }); return null; } },
    append: (type: string, fields: Json) => ({ type, event_id: `event-${type}`, ...fields }),
    flush: () => undefined,
  };
  return core;
}

function turn(fields: Json = {}): Json {
  return {
    step: 3, observationEvent: { event_id: "event-observation" }, calls: 2, callLimit: 80, deadlineMs: null,
    outcome: null, submitted: null, modelCallId: "play-call-000002", lastStopReason: null, lastError: null,
    ...fields,
  };
}

test("a submitted action ends the act at the next boundary, without polling, and finish hands it over", async () => {
  const core = fakeCore();
  const driver = new PlayDriver(core, { workspace: "/tmp", auth_path: "/tmp/auth.json", agent_dir: null });
  driver.turn = turn({ submitted: { action_event_id: "event-action", tool_name: "robot.submit_action" } });
  assert.equal(await driver.boundary(), "action_submitted");
  assert.equal(core.polls.length, 0, "no inbox poll after the act's action: the next act polls");
  assert.equal(driver.turn.outcome, null, "the stop is not an outcome: the submitted action is");
  const outcome = (driver as Json).finish();
  assert.deepEqual(outcome, { submitted: true, attempts: 2, action_event_id: "event-action",
    tool_name: "robot.submit_action" });
});

test("an interrupt after the action was submitted still hands the action over (Esc lets a proposed motion run)",
  async () => {
    const core = fakeCore();
    core.interrupted = true;
    const driver = new PlayDriver(core, { workspace: "/tmp", auth_path: "/tmp/auth.json", agent_dir: null });
    driver.turn = turn({ submitted: { action_event_id: "event-action", tool_name: "robot.submit_action" } });
    assert.equal(await driver.boundary(), "action_submitted");
    assert.equal((driver as Json).finish().submitted, true);
  });

test("before any action the boundary still polls the operator's inbox and goes on", async () => {
  const core = fakeCore();
  const driver = new PlayDriver(core, { workspace: "/tmp", auth_path: "/tmp/auth.json", agent_dir: null });
  driver.turn = turn();
  assert.equal(await driver.boundary(), null);
  assert.equal(core.polls.length, 1);
  assert.equal(core.polls[0].method, "operator.poll");
});
