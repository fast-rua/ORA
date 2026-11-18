// Free play's image window (src/play/context_images.ts): a long session's requests stay bounded while the
// latest observations and tool results keep their images. Sent in full, a real session's request reached
// 50.8 MB and the provider refused it (HTTP 413).

import assert from "node:assert/strict";
import { test } from "node:test";

import { boundContextImages, contextImageWindow, DEFAULT_CONTEXT_IMAGES, OBSERVATION_MESSAGE }
  from "../src/play/context_images.ts";

type Json = any;
const CAMERAS = ["head", "left_wrist", "right_wrist"];
const PNG = "A".repeat(600_000); // one camera image as base64, about what a 640 x 480 PNG takes

function observation(step: number): Json {
  return {
    role: "custom", customType: OBSERVATION_MESSAGE, display: true, timestamp: step,
    details: { ora_observation_step: step },
    content: [{ type: "text", text: `{"observation_step": ${step}}` },
      ...CAMERAS.flatMap((camera) => [
        { type: "text", text: `ACTION RESULT ACTUAL · sequence ${step} · ${camera} · latest at this message` },
        { type: "image", mimeType: "image/png", data: `${step}-${camera}-${PNG}` }])],
  };
}

function toolResult(step: number): Json {
  return { role: "toolResult", toolCallId: `call-${step}`, toolName: "observe", isError: false, timestamp: step,
    content: [{ type: "text", text: "{}" }, { type: "image", mimeType: "image/png", data: `tool-${step}-${PNG}` }] };
}

/** One scripted session's conversation after `acts` acts: each act an observation, a call, a tool result. */
function session(acts: number): Json[] {
  const messages: Json[] = [{ role: "user", content: [{ type: "text", text: "put the mug into the box" },
    { type: "image", mimeType: "image/png", data: "operator-attached" }], timestamp: 0 }];
  for (let step = 0; step < acts; step += 1) {
    messages.push(observation(step));
    messages.push({ role: "assistant", content: [{ type: "text", text: `act ${step}` }], timestamp: step });
    messages.push(toolResult(step));
  }
  return messages;
}

function imageCount(messages: Json[]): number {
  return messages.flatMap((message) => (Array.isArray(message.content) ? message.content : []))
    .filter((block: Json) => block.type === "image").length;
}

function imageBytes(messages: Json[]): number {
  return messages.flatMap((message) => (Array.isArray(message.content) ? message.content : []))
    .filter((block: Json) => block.type === "image").reduce((sum: number, block: Json) => sum + block.data.length, 0);
}

test("a long session's request carries a bounded amount of images, the latest ones intact", () => {
  const sizes = [5, 20, 60].map((acts) => {
    const messages = session(acts);
    const { messages: sent, omitted } = boundContextImages(messages, DEFAULT_CONTEXT_IMAGES);
    assert.equal(omitted, (acts - 3) * 3 + (acts - 3), `${acts} acts`);
    // The latest three observations and tool results are sent as they are, in place.
    for (let step = acts - 3; step < acts; step += 1) {
      assert.deepEqual(sent[1 + step * 3], messages[1 + step * 3]);
      assert.deepEqual(sent[3 + step * 3], messages[3 + step * 3]);
    }
    // The operator's own attached image is always sent.
    assert.equal(sent[0], messages[0]);
    return { images: imageCount(sent), bytes: imageBytes(sent) };
  });
  assert.deepEqual(sizes.map((size) => size.images), [3 * 3 + 3 + 1, 3 * 3 + 3 + 1, 3 * 3 + 3 + 1],
    "the images of a request do not grow with the session");
  for (const size of sizes) assert.ok(size.bytes <= (3 * 3 + 3) * (PNG.length + 32) + 64);
});

test("an older image becomes one line naming its observation step and camera, or its tool", () => {
  const messages = session(5);
  const { messages: sent } = boundContextImages(messages, { observations: 1, tool_results: 1 });
  const old = sent[1];
  assert.notEqual(old, messages[1], "a new message: the session's own is never changed");
  assert.equal(messages[1].content.filter((block: Json) => block.type === "image").length, 3);
  const lines = old.content.filter((block: Json) => block.type === "text").map((block: Json) => block.text);
  assert.ok(lines.includes("[earlier image not resent: observation step 0, head; "
    + "the latest observations carry the current view]"));
  assert.ok(lines.some((text: string) => text.includes("observation step 0, right_wrist")));
  assert.equal(old.content.filter((block: Json) => block.type === "image").length, 0);
  assert.deepEqual(sent[3].content[1], { type: "text", text: "[earlier image not resent: observe result image 1]" });
  assert.equal(sent[13], messages[13], "the latest observation keeps its images");
});

test("a short session sends everything as it is, and the window comes from the plan or the default", () => {
  const messages = session(2);
  const bounded = boundContextImages(messages);
  assert.equal(bounded.messages, messages);
  assert.equal(bounded.omitted, 0);
  assert.deepEqual(contextImageWindow({ observations: 5, tool_results: 2 }), { observations: 5, tool_results: 2 });
  assert.deepEqual(contextImageWindow(null), DEFAULT_CONTEXT_IMAGES);
  assert.deepEqual(contextImageWindow({ observations: 0, tool_results: 2 }), DEFAULT_CONTEXT_IMAGES);
});

// -- Batch pruning (window.prune_at): images leave where an instruction begins, never inside one ------

/** `instructions` instructions of `acts` acts each: a fresh observation, the operator's message, then the acts. */
function instructed(instructions: number, acts: number): Json[] {
  const messages: Json[] = [];
  let step = 0;
  for (let index = 0; index < instructions; index += 1) {
    messages.push(observation(step));
    messages.push({ role: "user", content: [{ type: "text", text: `instruction ${index}` }], timestamp: step });
    for (let act = 0; act < acts; act += 1) {
      step += 1;
      messages.push({ role: "assistant", content: [{ type: "text", text: `act ${step}` }], timestamp: step });
      messages.push(toolResult(step));
      messages.push(observation(step));
    }
    step += 1;
  }
  return messages;
}

const BATCH = { observations: 3, tool_results: 3, prune_at: 20 };
const carriesImages = (message: Json) => Array.isArray(message.content)
  && message.content.some((block: Json) => block.type === "image");
/** What a request sends of each message, cheaply: its role and how many images it still carries. */
const shape = (messages: Json[]) => messages.map((message) => `${message.role}:${Array.isArray(message.content)
  ? message.content.filter((block: Json) => block.type === "image").length : 0}`);

test("with prune_at, nothing sent earlier in an instruction changes while it runs", () => {
  // Three instructions of eight acts: 27 observations and 24 tool results by the end.
  const whole = instructed(3, 8);
  const opens = whole.flatMap((message, position) => (message.role === "user" ? [position] : []));
  let previous: string[] = [];
  let rewrites = 0;
  for (let length = 1; length <= whole.length; length += 1) {
    const sent = shape(boundContextImages(whole.slice(0, length), BATCH).messages);
    if (previous.some((message, position) => message !== sent[position])) {
      rewrites += 1;
      assert.ok(opens.includes(length - 1), `the prefix changed at message ${length - 1}, inside an instruction`);
    }
    previous = sent;
  }
  // The third instruction begins with 19 observations and 16 tool results: under 20, nothing leaves.
  // A sliding window of three would have rewritten the prefix at nearly every act.
  assert.equal(rewrites, 0);
  assert.equal(boundContextImages(whole, BATCH).omitted, 0);
});

test("with prune_at, an instruction that begins past the threshold drops all but the latest three at once", () => {
  const whole = instructed(4, 8);     // the fourth begins with 28 observations (its own fresh one too), 24 tool results
  const fourth = whole.flatMap((message, position) => (message.role === "user" ? [position] : []))[3];
  const before = boundContextImages(whole.slice(0, fourth), BATCH);
  assert.equal(before.omitted, 0, "until the instruction's message arrives everything is sent");
  const { messages: sent, omitted } = boundContextImages(whole.slice(0, fourth + 1), BATCH);
  assert.equal(omitted, (28 - 3) * CAMERAS.length + (24 - 3));
  const kept = sent.filter(carriesImages);
  assert.equal(kept.filter((message) => message.role === "custom").length, 3);
  assert.equal(kept.filter((message) => message.role === "toolResult").length, 3);
  assert.equal(sent[fourth - 1], whole[fourth - 1], "the instruction's own fresh observation keeps its images");
  // Through the rest of that instruction the pruned prefix stays exactly as it was sent then.
  const later = boundContextImages(whole, BATCH).messages;
  assert.deepEqual(later.slice(0, fourth + 1), sent);
  assert.equal(later.filter(carriesImages).filter((message) => message.role === "custom").length, 3 + 8);
});

test("with prune_at, one instruction that never ends is pruned in batches at the hard limit", () => {
  const whole = instructed(1, 120);                     // 121 observations in one instruction
  let previous: string[] = [];
  let rewrites = 0;
  for (let length = 1; length <= whole.length; length += 1) {
    const sent = shape(boundContextImages(whole.slice(0, length), BATCH).messages);
    if (previous.some((message, position) => message !== sent[position])) rewrites += 1;
    previous = sent;
    assert.ok(sent.filter((message) => message === `custom:${CAMERAS.length}`).length <= 50, "never more than the hard limit");
  }
  // 121 observations and 120 tool results, each kind pruned from 51 down to 3: twice each, not 230 times.
  assert.equal(rewrites, 4);
});

test("the window reads prune_at from the plan, and a plan without it slides as before", () => {
  assert.deepEqual(contextImageWindow({ observations: 3, tool_results: 3, prune_at: 20 }), BATCH);
  assert.deepEqual(contextImageWindow({ observations: 3, tool_results: 3 }), DEFAULT_CONTEXT_IMAGES);
  assert.equal(contextImageWindow({ observations: 5, tool_results: 3, prune_at: 4 }).prune_at, undefined,
    "a threshold under the window is not a batch window");
  const whole = instructed(4, 8);
  assert.equal(boundContextImages(whole, contextImageWindow({ observations: 3, tool_results: 3 })).omitted,
    (36 - 3) * CAMERAS.length + (32 - 3), "the sliding window still bounds a plan sealed before prune_at");
});
