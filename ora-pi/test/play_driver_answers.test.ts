// Free play's answers (src/play/driver.ts): a text-only reply answers the joined operator messages its
// model call was the first to read. Pi steers a message in after the current tool batch, so the call that
// reads it is known only from the context Pi builds for each call (noteContext, from the `context` hook).

import assert from "node:assert/strict";
import { test } from "node:test";

import { PLAY_MESSAGE, PlayDriver } from "../src/play/driver.ts";

type Json = any;

function driverAt(callSequence: number): Json {
  const core: Json = {
    config: { tools: { definitions: [], typed_effects: false, mutation_tools: [] } },
    interrupted: false, rpc: { request: async () => null }, append: () => ({}), flush: () => undefined,
  };
  const driver: Json = new PlayDriver(core, { workspace: "/tmp", auth_path: "/tmp/auth.json", agent_dir: null });
  driver.callSequence = callSequence;
  return driver;
}

function joined(id: string): Json {
  return { role: "custom", customType: PLAY_MESSAGE, display: true, content: "how is it going?", timestamp: 1,
    details: { ora_message_id: id, ora_joined: true } };
}

function textOnlyTurn(lastCall: number): Json {
  return { step: 4, observationEvent: {}, calls: 2, callLimit: 80, deadlineMs: null, outcome: null, submitted: null,
    modelCallId: `play-call-${String(lastCall).padStart(6, "0")}`, lastStopReason: "stop", lastError: null };
}

test("a text-only reply answers the joined messages its call read first, and only those", () => {
  const driver = driverAt(7);
  driver.noteContext([{ role: "user", content: "raise the arm" }, joined("m1")]);
  driver.callSequence = 8;
  driver.noteContext([joined("m1"), joined("m2")]);
  driver.turn = textOnlyTurn(8);
  assert.deepEqual(driver.finish(), { submitted: false, termination: "policy_no_action", attempts: 2,
    answers: ["m2"] }, "m1 was read by call 7, which answered it (with a tool call or as its own reply)");
  driver.turn = textOnlyTurn(9);
  assert.deepEqual(driver.finish(), { submitted: false, termination: "policy_no_action", attempts: 2 },
    "a later text-only reply with no newly read message is the report");
});
