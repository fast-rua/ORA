// Which Pi RPC commands an attached client may send, decided in the agent process.
//
// A standard Pi client offers a model picker, a thinking-level switch,
// compaction, a shell, new sessions and forks. In an ORA episode the RunPlan
// binds the model and reasoning effort, ORA's context policy owns compaction,
// the world cannot be rewound or replaced, and the session file is evidence;
// so those commands are refused here, whatever the client shows. Free play
// (`ora play`, Pi's own agent) opens Pi's commands, as its terminal does.
// Observers read; only the controlling client changes anything.

/** Commands that only read the session. */
export const READ_ONLY = new Set([
  "get_state", "get_messages", "get_entries", "get_tree", "get_session_stats", "get_commands",
  "get_available_models", "get_available_thinking_levels", "get_fork_messages", "get_last_assistant_text",
  "ora_get_role", "ora_take_control", "ora_get_run", "ora_get_frames",
]);

/** Refused in an episode, with the reason the client is told. */
export const EPISODE_REFUSALS: Record<string, string> = {
  new_session: "a live episode's history cannot be replaced",
  switch_session: "a live episode's history cannot be replaced",
  fork: "a live episode's history cannot be rewound",
  clone: "a live episode's history cannot be copied into another session",
  set_model: "the model is bound by the RunPlan",
  cycle_model: "the model is bound by the RunPlan",
  set_thinking_level: "reasoning effort is bound by the RunPlan",
  cycle_thinking_level: "reasoning effort is bound by the RunPlan",
  compact: "ORA manages the model's context with its own policy",
  set_auto_compaction: "ORA manages the model's context with its own policy",
  set_auto_retry: "provider retries are bound by the RunPlan",
  bash: "shell commands are not available in an ORA episode",
  abort_bash: "shell commands are not available in an ORA episode",
  set_session_name: "the run's session is evidence and is not renamed",
  export_html: "export a run where you read it: `ora sessions export RUN`",
  ora_play_control: "free-play controls (stop the instruction, next scene, quit) belong to `ora play`",
  ora_play_queue: "a queued instruction belongs to `ora play`; an episode's message is a steer",
};

/** Refused in free play too (they would write files where the agent process runs). */
export const PLAY_REFUSALS: Record<string, string> = {
  export_html: "export a run where you read it: `ora sessions export RUN`",
};

export type RunMode = "episode" | "play";

/** The commands this mode refuses, with their reasons (a client hides them: `ora_get_run`). */
export function refusals(mode: RunMode): Record<string, string> {
  return { ...(mode === "play" ? PLAY_REFUSALS : EPISODE_REFUSALS) };
}

/** Null when `command` may run for this client, else the refusal. */
export function refusal(command: string, mode: RunMode, controller: boolean): string | null {
  const table = mode === "play" ? PLAY_REFUSALS : EPISODE_REFUSALS;
  if (command in table) return `${command} is not available here: ${table[command]}`;
  if (!controller && !READ_ONLY.has(command)) {
    return `${command}: this client observes the run; the controlling client sends commands `
      + "(ora_take_control makes this one the controller)";
  }
  return null;
}
