// The trusted terminal boundary's refusal (`mutationProtocolError`): a world mutation on a causal-slot turn after
// the boundary is refused `episode_terminal`; a tool with its own wording (code mode's script tools, given by Python)
// keeps the code and replaces only the message and constraints, and every other tool keeps the core's words.
import assert from "node:assert/strict";
import { test } from "node:test";
import { mutationProtocolError, toolErrorDict } from "../src/core/dispatch.ts";

const codes = { argumentRefusalCodes: new Set<string>(), infrastructureCodes: new Set<string>() };
const wording = { message: "no script runs any more; call the tools left directly: check_task, finish_episode",
  constraints: { allowed: ["check_task", "finish_episode"] } };

test("a mutation after the terminal boundary keeps the core's refusal without a wording", () => {
  const refused = mutationProtocolError("native_slots_v4", "mutation", true);
  assert.deepEqual(toolErrorDict(refused!, codes), {
    code: "episode_terminal", message: "world mutation is unavailable after the trusted terminal boundary",
    retryable: false, field: null, received: null, constraints: {}, infrastructure: false,
  });
});

test("a tool's own wording replaces the message and constraints, never the code", () => {
  const refused = toolErrorDict(mutationProtocolError("native_slots_v3", "mutation", true, wording)!, codes);
  assert.equal(refused.code, "episode_terminal");
  assert.equal(refused.retryable, false);
  assert.equal(refused.message, wording.message);
  assert.deepEqual(refused.constraints, wording.constraints);
});

test("nothing is refused before the boundary, for a read or outside the causal slots", () => {
  assert.equal(mutationProtocolError("native_slots_v4", "mutation", false, wording), null);
  assert.equal(mutationProtocolError("native_slots_v4", "read_only", true, wording), null);
  assert.equal(mutationProtocolError("native_slots_v2", "mutation", true, wording), null);
  assert.equal(mutationProtocolError(null, "mutation", true), null);
});
