// Free play in ORA's code mode (plan play.code_mode, src/play/driver.ts): a command whose program acted ends
// the act with the Supervisor's ledger, a later call of that reply is refused, a delivered result is settled
// before the next model call, and the calls a running command's program makes reach the terminal and the
// page as the command's partial result.

import assert from "node:assert/strict";
import { test } from "node:test";

import { PlayDriver } from "../src/play/driver.ts";

type Json = any;

const BASH = { type: "function", function: { name: "bash", description: "run a command", parameters: {} } };

function execution(): Json {
  const before = { observation_id: "obs-3", step: 3 };
  const after = { observation_id: "obs-4", step: 4 };
  return {
    schema: "ora.tool-execution.v1", model_call_id: "play-call-000002", tool_call_id: "call-1",
    native_action_budget_requested: 20, native_action_budget_admitted: 20, native_actions_used: 1,
    native_actions_remaining: 19, initial_observation: before, final_observation: after,
    actions: [{ action_id: "action-1", receipt_event_id: "event-receipt", receipt_ref: { sha256: "a".repeat(64) },
      native_actions_used: 1, native_action_budget_admitted: 20, before_observation: before, after_observation: after,
      refusal: null }],
  };
}

function fakeCore(invoke: (params: Json, driver: PlayDriver) => Promise<Json> | Json) {
  const requests: Json[] = [];
  const appended: Json[] = [];
  const accepted: Json[] = [];
  let driver: PlayDriver | null = null;
  const core: Json = {
    config: { tools: { definitions: [BASH], typed_effects: true, mutation_tools: [] }, policy_session_id: "ps" },
    interrupted: false,
    rpc: {
      request: async (method: string, params: Json) => {
        requests.push({ method, params });
        if (method === "tools.classify") return [{ effect: "read_only" }];
        if (method === "tools.invoke") return invoke(params, driver!);
        return null;
      },
    },
    append: (type: string, fields: Json) => {
      appended.push({ type, ...fields });
      return { type, event_id: `event-${appended.length}`, ...fields };
    },
    acceptExecution: (value: Json) => {
      accepted.push(value);
      return value;
    },
    artifacts: { putJson: () => ({ path: "native/agent-artifacts/x.json", sha256: "b".repeat(64) }) },
    dataUrl: () => "data:image/png;base64,AAAA",
    flush: () => undefined,
  };
  const make = () => {
    driver = new PlayDriver(core, { workspace: "/tmp", auth_path: "/tmp/auth.json", agent_dir: null, code_mode: true });
    driver.turn = {
      step: 3, observationEvent: { event_id: "event-observation" }, calls: 3, callLimit: 80, deadlineMs: null,
      outcome: null, submitted: null, modelCallId: "play-call-000002", lastStopReason: null, lastError: null,
    } as Json;
    return driver;
  };
  return { core, requests, appended, accepted, make };
}

function projected(fields: Json = {}): Json {
  return {
    canonical: { ok: true, status: "completed", calls: [], output: "" }, visible: { ok: true, status: "completed",
      calls: [], output: "" }, images: [], raw_ok: true, directive: null, execution_delivery_guard: true, ...fields,
  };
}

test("a command whose program acted ends the act with the Supervisor's ledger, and settles before the next call",
  async () => {
    const fake = fakeCore(() => projected({ directive: "actions_executed", execution: execution() }));
    const driver = fake.make();
    const result = await driver.robotTool("bash", { command: "python - <<'EOF'\nEOF" }, "call-1");
    assert.equal(result.terminate, true);
    assert.equal(fake.accepted.length, 1, "the ledger is checked against the recorded receipts");
    const outcome = driver.turn!.outcome as Json;
    assert.equal(outcome.actions_executed, true);
    assert.equal(outcome.execution.native_actions_used, 1);
    assert.equal(outcome.attempts, 3);
    const recorded = fake.appended.find((event) => event.type === "tool_result");
    assert.equal(recorded.payload.execution.native_actions_used, 1, "the result records the ledger, as an episode's");
    // A later call of the same reply is refused: the next act brings the new observation first.
    const later = await driver.robotTool("bash", { command: "ls" }, "call-2");
    assert.match(later.content[0].text, /turn_ended/);
    assert.equal(fake.requests.filter((item) => item.method === "tools.invoke").length, 1);
    // The ended act stops at its boundary; the delivered result is settled first only when the act goes on.
    assert.equal(await driver.boundary(), "stopped");
    driver.turn!.outcome = null;
    await driver.boundary();
    const settled = fake.requests.find((item) => item.method === "tools.settled");
    assert.deepEqual(settled.params.calls, [{ model_call_id: "play-call-000002", tool_call_id: "call-1" }]);
  });

test("an execution that left the world unknown ends the act as that", async () => {
  const fake = fakeCore(() => projected({ directive: "world_state_unknown" }));
  const driver = fake.make();
  const result = await driver.robotTool("bash", { command: "true" }, "call-1");
  assert.equal(result.terminate, true);
  assert.equal((driver.turn!.outcome as Json).termination, "world_state_unknown");
});

test("a running command's robot calls reach the terminal as its partial result", async () => {
  const partials: Json[] = [];
  const fake = fakeCore((params, driver) => {
    const origin = { kind: "code_mode", parent_tool_call_id: "call-1", seq: 1 };
    driver.noteRecord("tool_call_proposed", { payload: { name: "move_pose", origin } });
    driver.noteRecord("tool_call_proposed", { payload: { name: "elsewhere", origin: { ...origin, parent_tool_call_id: "x" } } });
    driver.noteRecord("tool_result", { payload: { name: "move_pose", ok: true, origin } });
    driver.noteRecord("tool_image", { artifact_ref: { path: "native/agent-artifacts/p.png" },
      payload: { image_index: 0, origin: { ...origin, dry_run: true } } });
    return projected();
  });
  const driver = fake.make();
  await driver.robotTool("bash", { command: "python plan.py" }, "call-1", (partial) => partials.push(partial));
  assert.equal(partials.length, 3, "a call, its result and its image; another command's call is not this one's");
  const first = JSON.parse(partials[0].content[0].text);
  assert.deepEqual(first, { status: "running", calls: [{ call: 1, tool: "move_pose", ok: true, running: true }], output: "" });
  const last = JSON.parse(partials[2].content[0].text);
  assert.deepEqual(last.calls, [{ call: 1, tool: "move_pose", ok: true, images: 1 }]);
  assert.equal(partials[2].content[1].type, "image", "the picture the call returned (a dry run's plan)");
  // Once the command returned, nothing more is shown as its partial result.
  driver.noteRecord("tool_call_proposed", { payload: { name: "late", origin: { kind: "code_mode", parent_tool_call_id: "call-1", seq: 2 } } });
  assert.equal(partials.length, 3);
});
