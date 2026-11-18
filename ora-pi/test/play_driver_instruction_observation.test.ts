// Free play's instruction start (src/play/driver.ts `act`): the Supervisor publishes a fresh observation
// before a new instruction's first model call (episode/play.py `_observe_for_instruction`), so the act that
// delivers the instruction sends that observation first and the instruction after it. When the fresh look
// failed, the instruction's delivery carries an `observation_note`: the model reads it with the observation.
// Labelled tool images: an observe result's images each follow their own label (`image_labels`).

import assert from "node:assert/strict";
import { test } from "node:test";

import { PlayDriver } from "../src/play/driver.ts";

type Json = any;

function harness(delivery: Json[]) {
  const sent: Json[] = [];
  const core: Json = {
    config: { tools: { definitions: [], typed_effects: false, mutation_tools: [] } },
    interrupted: false, playState: { instructions: 1 },
    resolveDraft: (draft: Json) => ({ message: draft.message }),
    append: (type: string, fields: Json) => ({ type, event_id: `event-${type}`, ...fields }),
    flush: () => undefined,
  };
  const driver: Json = new PlayDriver(core, { workspace: "/tmp", auth_path: "/tmp/auth.json", agent_dir: null });
  // The first boundary's inbox poll is where Python delivers the instruction (operator_message -> deliver).
  core.rpc = { request: async (method: string) => {
    if (method === "operator.poll") for (const payload of delivery.splice(0)) driver.deliver(payload);
    return null;
  } };
  driver.session = {
    isStreaming: false,
    sendCustomMessage: async (message: Json, options: Json) => { sent.push({ kind: "custom", message, options }); },
    sendUserMessage: async (content: Json, options: Json) => { sent.push({ kind: "user", content, options }); },
    waitForIdle: async () => undefined,
  };
  return { driver, sent };
}

function params(step: number): Json {
  return {
    step, static: { system_text: "planner", messages: [] }, remaining_budget: { model_calls: 80 },
    draft: { message: { role: "user", content: [{ type: "text", text: `INITIAL ACTUAL · sequence ${step} · overhead` }],
      _observation: { observation_id: `obs-${step}`, state_ref: null } } },
  };
}

const instruction = (fields: Json = {}): Json =>
  ({ id: "i1", type: "instruction", message: "put the cup on the plate", joined: false, instruction_id: "i1",
    observation_step: 5, ...fields });

test("a new instruction's act sends the fresh observation, then the instruction, in one run", async () => {
  const { driver, sent } = harness([instruction()]);
  await driver.act(params(5));
  assert.equal(sent.length, 2);
  assert.equal(sent[0].message.customType, "ora.play_observation");
  assert.deepEqual(sent[0].message.content, [{ type: "text", text: "INITIAL ACTUAL · sequence 5 · overhead" }]);
  assert.equal(sent[0].message.details.ora_observation_step, 5);
  assert.deepEqual(sent[0].options, { triggerTurn: false }, "the instruction starts the run");
  assert.equal(sent[1].kind, "user");
  assert.equal(sent[1].content, "put the cup on the plate");
});

test("a failed fresh look: the note follows the observation it describes, before the instruction", async () => {
  const note = "ORA could not take a fresh observation at the start of this instruction (RuntimeError: x). "
    + "The observation with it is the latest one, step 4, captured 3 s ago; the scene may have changed since.";
  const { driver, sent } = harness([instruction({ observation_step: 4, observation_note: note })]);
  await driver.act(params(4));
  assert.deepEqual(sent[0].message.content, [
    { type: "text", text: "INITIAL ACTUAL · sequence 4 · overhead" }, { type: "text", text: note }]);
  assert.equal(sent[1].content, "put the cup on the plate", "the operator's words stay their own");
});

test("an observe result's images each follow their label; without labels they follow the text alone", () => {
  const image = (letter: string) => ({ type: "image_url", image_url: { url: `data:image/png;base64,${letter}` } });
  const labelled = PlayDriver.toolImageContent([image("A"), image("B")],
    ["observe image 1 of 2: overhead camera.", "observe image 2 of 2: front camera."]);
  assert.deepEqual(labelled, [
    { type: "text", text: "observe image 1 of 2: overhead camera." }, { type: "image", mimeType: "image/png", data: "A" },
    { type: "text", text: "observe image 2 of 2: front camera." }, { type: "image", mimeType: "image/png", data: "B" }]);
  assert.deepEqual(PlayDriver.toolImageContent([image("A")], null), [{ type: "image", mimeType: "image/png", data: "A" }]);
  assert.deepEqual(PlayDriver.toolImageContent([image("A")], ["one", "two"]),
    [{ type: "image", mimeType: "image/png", data: "A" }], "labels that do not match the images one to one are dropped");
});
