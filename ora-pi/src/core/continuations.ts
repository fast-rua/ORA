// Harness messages that keep the policy on the same observation
// (verbatim texts of ora/agent/native/continuations.py, the retired Python
// Native core's, now only in the robot-harness repository).

export interface Continuation {
  eventType: string;
  payload: Record<string, unknown>;
  content: string;
}

export function afterRejectedOutput(outputRejection: Record<string, any>): Continuation {
  return {
    eventType: "provider_output_rejected",
    payload: {
      code: outputRejection.code ?? "invalid_provider_output",
      provider_stop_reason: outputRejection.provider_stop_reason ?? null,
      policy_outcome: "continue_same_observation",
      world_state_changed: false,
    },
    content: outputRejection.code === "truncated_tool_call"
      ? "Your previous response reached the Provider output limit while proposing tools. The entire tool batch "
        + "was discarded. No tool ran and the world did not change. Retry one concise valid tool call for this "
        + "same observation."
      : "Your previous response ended with incomplete or invalid tool-input JSON. No tool ran and the world did "
        + "not change. Retry one concise valid tool call for this same observation.",
  };
}

export function afterTruncatedOutput(): Continuation {
  return {
    eventType: "provider_output_truncated",
    payload: { tool_call_count: 0, policy_outcome: "continue_same_observation" },
    content: "Your previous response reached the Provider output limit before proposing an action. Continue "
      + "working on this same observation and use the available tools.",
  };
}

export function afterNoAction(mutationTools: string[]): Continuation {
  const mutations = mutationTools.join(", ");
  return {
    eventType: "provider_no_action_continuation",
    payload: { tool_call_count: 0, policy_outcome: "continue_same_observation" },
    content: "No action was proposed for this observation. Continue; use read-only tools if needed, then "
      + `submit one action with ${mutations}, or use an advertised lifecycle tool when its contract applies.`,
  };
}
