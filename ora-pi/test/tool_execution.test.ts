import assert from "node:assert/strict";
import { test } from "node:test";
import { executionEvidence } from "../src/core/execution.ts";
import { actionStepInventory, toolResultHasExactCausalOutcome } from "../src/context/projection.ts";

function execution() {
  const obs = (step: number) => ({ step, observation_id: `obs-${step}` });
  return { schema: "ora.tool-execution.v1", tool_call_id: "parent", model_call_id: "model",
    native_action_budget_requested: 20, native_action_budget_admitted: 20,
    native_actions_used: 5, native_actions_remaining: 15,
    initial_observation: obs(0), final_observation: obs(2),
    actions: [2, 3].map((used, i) => ({ action_id: `action-${i}`,
      native_action_budget_admitted: 20 - i * 2, native_actions_used: used,
      before_observation: obs(i), after_observation: obs(i + 1),
      receipt_event_id: `receipt-${i}`, receipt_ref: { sha256: "a".repeat(64) } })) };
}
const domain: any = { exactOutcomeTools: [], acceptedMutationStatuses: ["submitted"] };
function messages(result: object = { answer: "computed after moving" }) {
  return [
    { role: "user", _observation: { step: 0, observation_id: "obs-0" } },
    { role: "assistant", tool_calls: [{ id: "parent", function: { name: "custom", arguments: "{}" } }] },
    { role: "tool", name: "custom", tool_call_id: "parent", content: JSON.stringify(result), _execution: execution() },
    { role: "user", _observation: { step: 2, observation_id: "obs-2", receipt: null } },
  ];
}
test("two actions cross multiple publications with no visible next-observation receipt", () => {
  const inventory = actionStepInventory(messages(), { mutationTools: [], domain });
  assert.equal(inventory[0].mutation, "custom");
  assert.deepEqual(inventory[0].receipt_event_ids, ["receipt-0", "receipt-1"]);
  assert.equal(inventory[0].post_observation_step, 2);
});
test("execution followed by tool failure is still a world-changing parent", () => {
  const inventory = actionStepInventory(messages({ error: { code: "tool_error" } }), { mutationTools: [], domain });
  assert.equal(inventory[0].mutation, "custom");
});
test("custom structured result is protected by actual spend across serialization", () => {
  const message = JSON.parse(JSON.stringify(messages()[2]));
  assert.equal(toolResultHasExactCausalOutcome(message, domain), true);
  assert.deepEqual(executionEvidence(message._execution), execution());
});
test("bad receipt chains and final observations are rejected", () => {
  const e = execution(); e.actions[1].receipt_event_id = e.actions[0].receipt_event_id;
  assert.throws(() => executionEvidence(e), /receipt/);
  const history = messages(); history[3]._observation!.observation_id = "wrong";
  assert.throws(() => actionStepInventory(history, { mutationTools: [], domain }), /final published/);
});
function refusedThenExecuted(): any {
  // A code-mode script's parent: the Supervisor refused its first action, the next one on the same publication
  // carries the same identity and executes.
  const obs = (step: number) => ({ step, observation_id: `obs-${step}` });
  const e: any = execution(); const [first] = e.actions;
  e.actions = [
    { ...first, native_actions_used: 0, after_observation: obs(0), refusal: "stale_or_mismatched_recovery_target" },
    { ...first, receipt_event_id: "receipt-1", native_actions_used: 3, refusal: null }];
  e.native_actions_used = 3; e.native_actions_remaining = 17; e.final_observation = obs(1);
  return e;
}
test("a refused action leaves its identity to the next action on the same publication", () => {
  const e = refusedThenExecuted();
  assert.deepEqual(executionEvidence(e), e);
  const executedTwice = refusedThenExecuted(); executedTwice.actions[0].refusal = null;
  assert.throws(() => executionEvidence(executedTwice), /chain/);
  const spent = refusedThenExecuted(); spent.actions[0].native_actions_used = 1;
  spent.native_actions_used = 4; spent.native_actions_remaining = 16;
  assert.throws(() => executionEvidence(spent), /chain/);
});
test("all refused zero-spend calls stay on the same publication", () => {
  const history = messages(); const e = history[2]._execution!;
  e.actions = []; e.native_actions_used = 0; e.native_actions_remaining = 20;
  e.final_observation = e.initial_observation;
  history[3]._observation = { step: 0, observation_id: "obs-0" };
  const inventory = actionStepInventory(history, { mutationTools: ["custom"], domain });
  assert.equal(inventory[0].mutation, null);
});

import { OraCore } from "../src/core/session.ts";
import { providerMessage } from "../src/context/projection.ts";

function coreWithExecutionEvents(e: any) {
  const core: any = Object.create(OraCore.prototype);
  core.executionEvents = new Map(); core.replayLedger = new Map();
  for (const a of e.actions) {
    const proposed = { type: "action_proposed", event_id: `proposed-${a.action_id}`,
      model_call_id: "model", tool_call_id: e.tool_call_id,
      payload: { action_id: a.action_id, observation_id: a.before_observation.observation_id } };
    core.executionEvents.set(proposed.event_id, proposed);
    core.replayLedger.set(`action_proposed|model|${e.tool_call_id}||${a.action_id}|`, proposed);
    core.executionEvents.set(`published-${a.action_id}`, { type: "fresh_observation_published",
      observation_step: a.after_observation.step, payload: { observation_id: a.after_observation.observation_id,
        after_receipt: { event_id: a.receipt_event_id } } });
    core.executionEvents.set(a.receipt_event_id, { type: "environment_receipt", event_id: a.receipt_event_id,
      causal_parent_id: proposed.event_id, artifact_ref: a.receipt_ref, payload: { action_id: a.action_id } });
  }
  return core;
}
test("restarted core consumes distinct trusted action records without repeating actions", () => {
  const e = execution(); const core = coreWithExecutionEvents(e);
  assert.deepEqual(core.acceptExecution(e, "parent"), e);
  assert.equal(core.replayLedger.size, 0);
  core.closeReplay();
  const afterPersistence = JSON.parse(JSON.stringify(messages()[2]));
  assert.deepEqual(core.acceptExecution(afterPersistence._execution, "parent"), e);
  assert.equal("_execution" in providerMessage(afterPersistence), false);
});
test("a receipt hash or parent substitution cannot enter trusted execution history", () => {
  const e = execution(); const core = coreWithExecutionEvents(e);
  e.actions[0].receipt_ref = { sha256: "b".repeat(64) };
  assert.throws(() => core.acceptExecution(e, "parent"), /trusted action evidence/);
});

for (const scenario of ["two_actions", "spent_then_error", "all_refused", "world_unknown", "interrupted"]) {
  test(`executeTool settles ${scenario} from trusted RPC evidence`, async () => {
    const e = execution();
    if (scenario === "all_refused") {
      e.actions = []; e.native_actions_used = 0; e.native_actions_remaining = 20;
      e.final_observation = e.initial_observation;
    }
    const core: any = coreWithExecutionEvents(e);
    core.runRoot = "/tmp/ora-unused-execution-test";
    core.interrupted = false;
    core.config = { policy_session_id: "session", domain: { dotted_tool_names: {} },
      tools: { delivered: [], null_as_omitted: false } };
    const turn: any = { outcome: null, fatal: null, attempts: 1, callLimit: 5, step: 0,
      observationEvent: { event_id: "obs-event" }, reply: { modelCallId: "model" },
      batch: { byCallId: new Map([["parent", 0]]), parsed: [{ name: "custom", args: {},
        effect: "mutation", proposal: { event_id: "proposal" } }],
        admission: { typedEffects: true }, done: 0 } };
    core.requireTurn = () => turn;
    core.recordToolResult = (_turn: any, _batch: any, _index: number, _name: string, result: any) => {
      assert.deepEqual(result.execution, e);
      return JSON.stringify(result.visible);
    };
    let invokes = 0;
    core.rpc = { request: async () => {
      invokes++;
      return { execution: e, raw_ok: scenario !== "spent_then_error",
        visible: { calculation: [1, 2] }, directive: scenario === "interrupted" ? "tool_execution_interrupted" : scenario === "world_unknown" ? "world_state_unknown"
          : scenario === "all_refused" ? null : "actions_executed" };
    } };
    const result = await core.executeTool("parent");
    assert.equal(invokes, 1);
    assert.equal(result.terminate, scenario !== "all_refused");
    if (scenario === "all_refused") assert.equal(turn.outcome, null);
    else if (scenario === "interrupted") assert.equal(turn.outcome.termination, "tool_execution_interrupted");
    else if (scenario === "world_unknown") assert.equal(turn.outcome.termination, "world_state_unknown");
    else {
      assert.equal(turn.outcome.actions_executed, true);
      assert.equal(turn.outcome.execution.native_actions_used, 5);
    }
  });
}

import { externalizeToolResults } from "../src/context/projection.ts";
for (const [layer, selector] of [["L1", "large"], ["L2", "duplicate"], ["L4", "remaining"]] as const) {
  test(`${layer} retains a custom action parent's full structured result`, () => {
    const history: any[] = messages({ calculation: "exact".repeat(1000) });
    history[2]._context_artifact_ref = { sha256: "f".repeat(64) };
    history.splice(3, 0, JSON.parse(JSON.stringify(history[2])));
    const [projected, changed] = externalizeToolResults(history, {
      throughActionSteps: 1, mutationTools: [], selector, thresholdBytes: 1, previewBytes: 1, domain,
    });
    assert.deepEqual(changed, []);
    assert.equal(projected[2].content, history[2].content);
    assert.equal(projected[3].content, history[3].content);
    assert.deepEqual(projected[2]._execution, execution());
  });
}
// L3 externalizes media only; these structured text results contain no media.

function zeroSpendResume(withCompletion = true) {
  const e = execution();
  e.native_actions_used = 0; e.native_actions_remaining = 20;
  e.final_observation = e.initial_observation;
  e.actions = e.actions.map((a) => ({ ...a, native_actions_used: 0,
    before_observation: e.initial_observation, after_observation: e.initial_observation }));
  const core = coreWithExecutionEvents(e);
  core.replayLedger = null;
  const observation = { type: "observation_received", event_id: "obs", observation_step: 0 };
  const evidence = [...core.executionEvents.values()].filter((event: any) =>
    ["action_proposed", "environment_receipt"].includes(event.type));
  core.resumeEvents = [observation,
    { type: "provider_request", event_id: "request-1", model_call_id: "model" },
    { type: "provider_response", event_id: "response-1", model_call_id: "model", payload: { accepted: true } },
    { type: "tool_call_proposed", event_id: "tool-1", model_call_id: "model", tool_call_id: "parent" },
    ...evidence,
    { type: "tool_result", event_id: "result-1", model_call_id: "model", tool_call_id: "parent",
      payload: withCompletion ? { execution: e } : {} },
    { type: "provider_request", event_id: "request-2", model_call_id: "model-2" },
    { type: "provider_response", event_id: "response-2", model_call_id: "model-2", payload: { accepted: true } },
  ];
  core.lastTurnOutcome = null;
  core.history = [{ role: "user", _observation: { step: 0 }, _source_event_ids: ["obs"] },
    { role: "assistant", _source_event_ids: ["response-1"] }];
  core.config = { policy: {} };
  core.append = () => ({ event_id: "resumed" });
  return { core, e };
}
test("resume accepts earlier zero-spend primitives with trusted closed receipts", () => {
  const { core } = zeroSpendResume();
  const turn = core.resumeTurn({ step: 0, remaining_budget: {}, terminal_lifecycle: false });
  assert.equal(turn.replay.modelCallId, "model-2");
  assert.equal(turn.outcome, null);
  assert.equal(core.replayLedger.size, 0); // the earlier reply is never re-run
});
test("resume still rejects an earlier proposal without trusted zero-spend completion", () => {
  const { core } = zeroSpendResume(false);
  assert.throws(() => core.resumeTurn({ step: 0, remaining_budget: {}, terminal_lifecycle: false }), /outside the reply/);
});
test("an earlier positive-spend parent cannot masquerade as a zero-spend continuation", () => {
  const { core, e } = zeroSpendResume();
  e.actions[0].native_actions_used = 1; e.native_actions_used = 1; e.native_actions_remaining = 19;
  assert.throws(() => core.resumeTurn({ step: 0, remaining_budget: {}, terminal_lifecycle: false }), /outside the reply/);
});
test("zero-spend finish and read-only results do not bind the next round's reset", () => {
  for (const name of ["finish_episode", "compute", "observe"]) {
    const e = execution();
    e.actions = []; e.native_actions_used = 0; e.native_actions_remaining = 20;
    e.final_observation = e.initial_observation;
    const history: any[] = [
      { role: "user", _observation: e.initial_observation, _session_round: 1 },
      { role: "assistant", tool_calls: [{ id: "parent", function: { name, arguments: "{}" } }] },
      { role: "tool", name, tool_call_id: "parent", content: "{}", _execution: e },
      { role: "user", _observation: { step: 1, observation_id: "round-2-reset" }, _session_round: 2 },
    ];
    const withExecution = actionStepInventory(history, { mutationTools: [], domain });
    delete history[2]._execution;
    assert.deepEqual(withExecution, actionStepInventory(history, { mutationTools: [], domain }));
  }
});
test("trusted execution cannot change its model identity while reusing a provider tool id", () => {
  const e = execution(); const core = coreWithExecutionEvents(e);
  assert.throws(() => core.acceptExecution(e, "parent", "other-model"), /identity/);
});
