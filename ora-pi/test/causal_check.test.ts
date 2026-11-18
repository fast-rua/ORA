// The causal check of a provider request (causalMessageViolations): a script's result without what the script
// wrote itself, whose names are the model's. The rest of that result, and every other message, stays checked.

import assert from "node:assert/strict";
import { test } from "node:test";

import { DEFAULT_CONTEXT_DOMAIN, resolveContextDomain } from "../src/context/domain_defaults.ts";
import { causalMessageViolations } from "../src/context/projection.ts";

const CODE_MODE = resolveContextDomain({ scriptTools: ["bash", "run_python"] });
const tool = (name: string, result: unknown) => ({
  role: "tool", name, tool_call_id: `call-${name}`, content: JSON.stringify(result),
});
const POINT = [0.1, 0.2, 0.3];

test("what a script printed, returned or handed back passes under any name", () => {
  for (const name of ["bash", "run_python"]) {
    assert.deepEqual(causalMessageViolations([
      tool(name, { ok: true, output: JSON.stringify({ red: { cands: [{ contact: POINT, path: "a", hash: "b" }] } }) }),
      tool(name, { ok: true, stdout: JSON.stringify({ contact: POINT }), stderr: "", value: { schema: "mine" } }),
      tool(name, { ok: true, handed_back: { reason: "done", data: { out: JSON.stringify({ lift: { contact: null } }) } } }),
    ], CODE_MODE), []);
  }
});

test("the rest of a script's result is checked", () => {
  assert.deepEqual(causalMessageViolations([
    tool("bash", { ok: true, output: "done", calls: [{ tool: "observe", receipt: { contact: POINT } }] }),
    tool("bash", { ok: true, handed_back: { reason: "done", data: {}, object_gt: POINT } }),
    tool("bash", { ok: true, script_error: { message: JSON.stringify({ success_predicate: "x" }) } }),
  ], CODE_MODE), [
    "messages[0].content<json>.calls[0].receipt.contact",
    "messages[1].content<json>.handed_back.object_gt",
    "messages[2].content<json>.script_error.message<json>.success_predicate",
  ]);
});

test("another tool's result is checked under the names a script's result leaves out", () => {
  assert.deepEqual(causalMessageViolations([
    tool("observe", { ok: true, value: { contact: POINT }, stdout: JSON.stringify({ task_source: "x" }) }),
  ], CODE_MODE), [
    "messages[0].content<json>.value.contact",
    "messages[0].content<json>.stdout<json>.task_source",
  ]);
});

test("tool-call mode has no script, whatever a tool is called", () => {
  assert.deepEqual(causalMessageViolations([tool("bash", { value: { contact: POINT } })], DEFAULT_CONTEXT_DOMAIN), [
    "messages[0].content<json>.value.contact",
  ]);
});

test("a message that is no script's result is checked as it is", () => {
  assert.deepEqual(causalMessageViolations([
    { role: "user", name: "bash", content: JSON.stringify({ value: { contact: POINT } }) },
    { role: "tool", name: "bash", tool_call_id: "c", content: "contact: not json" },
  ], CODE_MODE), ["messages[0].content<json>.value.contact"]);
});

test("the model's own turns are left out and do not move the indexes of the rest", () => {
  assert.deepEqual(causalMessageViolations([
    { role: "assistant", content: JSON.stringify({ contact: POINT }) },
    tool("observe", { contact: POINT }),
  ], CODE_MODE), ["messages[0].content<json>.contact"]);
});
