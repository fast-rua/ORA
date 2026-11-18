// A stand-in for ora-rpc-attach in the web server's tests: answers like the run's
// agent process (rpc/host.ts) and records every line it receives in FAKE_ATTACH_LOG.
import { appendFileSync } from "node:fs";

const log = process.env.FAKE_ATTACH_LOG;
const mode = process.env.FAKE_ATTACH_MODE ?? "episode";
let buffer = "";
let streaming = false;
const send = (message) => process.stdout.write(`${JSON.stringify(message)}\n`);
const ok = (command, data) => send({ id: command.id, type: "response", command: command.type, success: true, data });
const fail = (command, error) => send({ id: command.id, type: "response", command: command.type, success: false, error });
const messages = [
  { role: "custom", customType: "ora.observation", content: "observation 0", display: true, timestamp: 1,
    details: { ora_observation_step: 0, ora: { step: 0, images: [] } } },
];
const refused = mode === "play" ? { export_html: "export where you read it" }
  : { compact: "ORA manages the model's context with its own policy", set_model: "the model is bound by the RunPlan" };
let pendingStop = null;

function handle(command) {
  switch (command.type) {
    case "ora_attach": return;
    case "ora_get_run": return ok(command, { runId: "run-web", mode, role: "controller", clients: 1, refused,
      play: mode === "play" ? { state: "awaiting_instruction", scene: 1, scenes: 1, instructions: 0, running: null } : null });
    case "get_state": return ok(command, { isStreaming: streaming, model: { provider: "ora", id: "scripted" }, thinkingLevel: "off" });
    case "get_messages": return ok(command, { messages });
    case "prompt":
      if (command.message === "/stop") {
        pendingStop = command;
        return send({ type: "extension_ui_request", id: "dialog-1", method: "confirm", title: "Stop this run?", message: "really" });
      }
      return ok(command, { disposition: "queued" });
    case "steer":
    case "follow_up":
      return ok(command, { disposition: "queued" });
    case "compact":
      return command.type in refused ? fail(command, `compact is not available here: ${refused.compact}`) : ok(command, {});
    case "ora_play_control":
      return mode === "play" ? ok(command, { record: { id: "op-1", type: command.control } })
        : fail(command, "ora_play_control is not available here: free play only");
    case "ora_play_queue":
      return mode === "play" ? ok(command, { record: { id: "op-2", type: "queued_instruction" } })
        : fail(command, "ora_play_queue is not available here: free play only");
    case "ora_cell_note": {
      // As a real cell's run does (rpc/host.ts, tui/cell_commands.ts note): the entry, told to every client.
      if (mode !== "play") return fail(command, "ora_cell_note: this run has no real-cell commands");
      const entry = { ...command.note, via: "web page", at: 1 };
      send({ type: "ora_cell_command", entry });
      return ok(command, { entry });
    }
    case "start_streaming":
      streaming = true;
      send({ type: "agent_start" });
      send({ type: "message_start", message: { role: "assistant", content: [] } });
      send({ type: "message_update", assistantMessageEvent: { type: "text_start", contentIndex: 0 } });
      send({ type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "hel" } });
      send({ type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "lo" } });
      return ok(command, {});
    default:
      return fail(command, `Unknown command: ${command.type}`);
  }
}

process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buffer += chunk;
  let end;
  while ((end = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, end);
    buffer = buffer.slice(end + 1);
    if (!line.trim()) continue;
    if (log) appendFileSync(log, `${line}\n`);
    const command = JSON.parse(line);
    if (command.type === "extension_ui_response") {
      if (pendingStop) ok(pendingStop, { disposition: "handled" });
      pendingStop = null;
      continue;
    }
    handle(command);
  }
});
