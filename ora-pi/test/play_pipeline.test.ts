// Free play's pipelined next step (src/play/pipeline.ts): a prediction goes out when a world-changing call
// is submitted, and its reply answers the next real request only when the fresh observation's receipt passes
// the deterministic rules and nothing but that observation arrived; otherwise the real request is sent.

import assert from "node:assert/strict";
import { test } from "node:test";

import { createAssistantMessageEventStream, normalizeContext } from "@earendil-works/pi-ai";

import { observationEffect, PipelinePredictor, PREDICT_NOTE, receiptVerdict, replayEvents } from "../src/play/pipeline.ts";

type Json = any;

const reply = (id: string, call: string | null, text = "ok"): Json => ({
  role: "assistant", api: "openai-completions", provider: "ora", model: "m", stopReason: call ? "toolUse" : "stop",
  usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
  content: [{ type: "text", text }, ...(call ? [{ type: "toolCall", id: call, name: "move_pose", arguments: { targets: {} } }] : [])],
  timestamp: 0, responseId: id,
});

/** A base stream that answers each request with the next scripted reply and keeps every context it got. */
function fakeBase(replies: Json[]) {
  const seen: Json[] = [];
  const base = (_model: Json, context: Json, options?: Json) => {
    seen.push({ context, options });
    options?.onPayload?.({ messages: context.messages.length }, _model);
    const stream = createAssistantMessageEventStream();
    const message = replies.shift();
    queueMicrotask(() => {
      for (const event of replayEvents(message)) stream.push(event);
      stream.end(message);
    });
    return stream;
  };
  return { base, seen };
}

const effect = (fields: Json) => ({ type: "text", text: JSON.stringify({ last_action_effect: fields }) });
const observation = (fields: Json): Json => ({ role: "user", content: [effect(fields), { type: "text", text: "views" }], timestamp: 0 });
const start = [{ role: "user", content: [{ type: "text", text: "pick it up" }], timestamp: 0 }];

function recorder() {
  const events: Json[] = [];
  return { events, rec: { now: () => 1000, record: (fields: Json, artifact: Json) => events.push({ fields, artifact }) } };
}

async function drain(stream: Json): Promise<Json> {
  for await (const _event of stream) { /* consume */ }
  return stream.result();
}

test("the receipt rules: reached, a grasp that stalled on something, nothing else", () => {
  assert.deepEqual(receiptVerdict({ action: "move_pose", status: "reached", failure_code: null }), { pass: true, reason: "reached" });
  const grasp = (mm: number) => ({ action: "set_gripper", status: "not_reached", failure_code: "gripper_stalled_before_target",
    detail: { arms: { left: { gripper_outcome: { goal: "close", state: "stalled_before_target", measured_aperture_mm: mm } },
      right: { gripper_outcome: null } } } });
  assert.equal(receiptVerdict(grasp(35.4)).pass, true);
  assert.deepEqual(receiptVerdict(grasp(0.4)), { pass: false, reason: "grasp_empty_or_open" });
  assert.deepEqual(receiptVerdict({ action: "move_pose", status: "not_reached", failure_code: "execution_fault" }),
    { pass: false, reason: "not_reached:execution_fault" });
  assert.equal(receiptVerdict({ action: "move_pose", status: "plan_rejected", failure_code: "table_clearance" }).pass, false);
  assert.equal(receiptVerdict(null).reason, "no_receipt");
  assert.deepEqual(observationEffect(observation({ status: "reached" })), { status: "reached" });
  assert.equal(observationEffect({ role: "user", content: "no json here" }), null);
});

test("a confirmed motion's prediction answers the next request, which is never sent", async () => {
  const { base, seen } = fakeBase([reply("r1", "call-1"), reply("p1", "call-2", "predicted")]);
  const { events, rec } = recorder();
  const predictor = new PipelinePredictor(base as Json, rec);
  const first = await drain(predictor.stream({ id: "m" }, normalizeContext({ messages: start } as Json), {}));
  predictor.noteReply(first);
  const result = [{ type: "text", text: "{\"status\": \"submitted\"}" }];
  predictor.start("call-1", "move_pose", result);
  // The prediction: the real context, the reply, its result and the note.
  assert.equal(seen.length, 2);
  const predicted = seen[1].context.messages;
  assert.deepEqual(predicted.map((m: Json) => m.role), ["user", "assistant", "toolResult", "user"]);
  assert.equal(predicted[3].content[0].text, PREDICT_NOTE);
  // The next real request: the same messages with the fresh observation in place of the note.
  const payloads: Json[] = [];
  const next = normalizeContext({ messages: [...start, first, { role: "toolResult", toolCallId: "call-1", toolName: "move_pose",
    content: result, isError: false, timestamp: 0 }, observation({ action: "move_pose", status: "reached" })] } as Json);
  const answer = await drain(predictor.stream({ id: "m" }, next, { onPayload: (payload: Json) => payloads.push(payload) }));
  assert.equal(seen.length, 2, "the real request was not sent");
  assert.equal(answer.content[0].text, "predicted");
  assert.equal(payloads[0].ora_pipeline_replay.verdict, "reached");
  assert.deepEqual(predictor.stats, { started: 1, used: 1, discarded: 0 });
  assert.equal(events.length, 1);
  assert.equal(events[0].fields.outcome, "used");
  assert.ok(events[0].artifact.request, "the prediction's request is recorded");
});

for (const [label, last, reason] of [
  ["a motion that faulted", observation({ action: "move_pose", status: "not_reached", failure_code: "execution_fault" }), "not_reached:execution_fault"],
  ["an operator message that arrived", null, "context_changed"],
] as Json[]) {
  test(`${label} discards the prediction and the real request is sent`, async () => {
    const { base, seen } = fakeBase([reply("r1", "call-1"), reply("p1", "call-2", "predicted"), reply("r2", "call-3", "real")]);
    const { events, rec } = recorder();
    const predictor = new PipelinePredictor(base as Json, rec);
    const first = await drain(predictor.stream({ id: "m" }, normalizeContext({ messages: start } as Json), {}));
    predictor.noteReply(first);
    predictor.start("call-1", "move_pose", [{ type: "text", text: "submitted" }]);
    const added = last ?? observation({ action: "move_pose", status: "reached" });
    const extra = last === null ? [{ role: "user", content: [{ type: "text", text: "stop, use the left arm" }], timestamp: 0 }] : [];
    const next = normalizeContext({ messages: [...start, first, { role: "toolResult", toolCallId: "call-1", toolName: "move_pose",
      content: [{ type: "text", text: "submitted" }], isError: false, timestamp: 0 }, added, ...extra] } as Json);
    const answer = await drain(predictor.stream({ id: "m" }, next, {}));
    assert.equal(answer.content[0].text, "real");
    assert.equal(seen.length, 3);
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(events[0].fields.outcome, "discarded");
    assert.equal(events[0].fields.reason, reason);
  });
}

test("no prediction when the submitted call is not the reply's last call", async () => {
  const { base, seen } = fakeBase([reply("r1", "call-1")]);
  const predictor = new PipelinePredictor(base as Json, recorder().rec);
  const first = await drain(predictor.stream({ id: "m" }, normalizeContext({ messages: start } as Json), {}));
  predictor.noteReply(first);
  predictor.start("call-0", "move_pose", []);
  assert.equal(seen.length, 1);
  assert.equal(predictor.stats.started, 0);
});

test("a prediction still running when the session ends is cancelled and recorded before close returns", async () => {
  let aborted = false;
  let finish: (message: Json) => void = () => undefined;
  const base = (_model: Json, context: Json, options?: Json) => {
    const stream = createAssistantMessageEventStream();
    if (context.messages.length === 1) {                       // the real call answers at once
      const message = reply("r1", "call-1");
      queueMicrotask(() => { for (const event of replayEvents(message)) stream.push(event); stream.end(message); });
    } else {                                                   // the prediction answers only when told to
      options?.signal?.addEventListener("abort", () => { aborted = true; finish({ ...reply("p1", null), stopReason: "aborted" }); });
      finish = (message: Json) => stream.end(message);
    }
    return stream;
  };
  const { events, rec } = recorder();
  const predictor = new PipelinePredictor(base as Json, rec);
  const first = await drain(predictor.stream({ id: "m" }, normalizeContext({ messages: start } as Json), {}));
  predictor.noteReply(first);
  predictor.start("call-1", "move_pose", [{ type: "text", text: "submitted" }]);
  assert.equal(events.length, 0);
  await predictor.close();
  assert.ok(aborted, "the prediction's request is aborted");
  assert.equal(events.length, 1, "recorded before close returned");
  assert.deepEqual([events[0].fields.outcome, events[0].fields.reason], ["discarded", "session_closed"]);
  await predictor.close();                                     // a second close changes nothing
  assert.equal(events.length, 1);
});
