/** Supervisor-produced execution evidence. Never decoded from visible tool JSON. */
export function executionEvidence(value: any, callId?: string, modelCallId?: string): any {
  if (!value || value.schema !== "ora.tool-execution.v1"
    || typeof value.model_call_id !== "string" || !value.model_call_id
    || (modelCallId && value.model_call_id !== modelCallId)
    || typeof value.tool_call_id !== "string" || (callId && value.tool_call_id !== callId)) {
    throw new Error("invalid trusted tool execution identity");
  }
  const integer = (n: any) => Number.isSafeInteger(n) && n >= 0;
  for (const key of ["native_action_budget_requested", "native_action_budget_admitted",
    "native_actions_used", "native_actions_remaining"]) {
    if (!integer(value[key])) throw new Error(`invalid execution ${key}`);
  }
  if (value.native_action_budget_admitted > value.native_action_budget_requested
    || value.native_actions_remaining + value.native_actions_used !== value.native_action_budget_admitted
    || !Array.isArray(value.actions)) throw new Error("execution budget balance differs");
  const identity = (o: any) => o && typeof o.observation_id === "string" && integer(o.step);
  if (!identity(value.initial_observation) || !identity(value.final_observation)) {
    throw new Error("execution lacks publication identities");
  }
  let prior = value.initial_observation;
  let used = 0;
  // A refused action ran nothing and published nothing, so the next action on the same observation carries
  // its identity (a code-mode script's next call); an executed identity never repeats (tool_execution_log.py).
  const executed = new Set(), receipts = new Set();
  for (const a of value.actions) {
    const refused = a?.refusal !== null && a?.refusal !== undefined;
    if (!a || typeof a.action_id !== "string" || executed.has(a.action_id)
      || (refused && (a.native_actions_used !== 0 || a.after_observation?.step !== prior.step
        || a.after_observation?.observation_id !== prior.observation_id))
      || typeof a.receipt_event_id !== "string" || !a.receipt_event_id || receipts.has(a.receipt_event_id)
      || !/^[0-9a-f]{64}$/.test(a.receipt_ref?.sha256 ?? "")
      || !integer(a.native_actions_used) || !integer(a.native_action_budget_admitted)
      || a.native_actions_used > a.native_action_budget_admitted
      || !identity(a.before_observation) || !identity(a.after_observation)
      || a.before_observation.step !== prior.step || a.before_observation.observation_id !== prior.observation_id
      || a.after_observation.step < prior.step) throw new Error("invalid action receipt/publication chain");
    if (!refused) executed.add(a.action_id);
    receipts.add(a.receipt_event_id);
    used += a.native_actions_used;
    prior = a.after_observation;
  }
  if (used !== value.native_actions_used || prior.step !== value.final_observation.step
    || prior.observation_id !== value.final_observation.observation_id) {
    throw new Error("execution final publication or consumption differs");
  }
  return value;
}
