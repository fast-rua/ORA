// A small finished free-play run on disk, as the Supervisor and the agent core write one:
// Pi's session (an observation with a stored and an unstored image, an instruction, a reply
// with its thinking and a mark_point call whose result returned an image, a joined message,
// its answer), the Supervisor's events (the instruction, the messages typed and delivered, one
// action and its receipt, operator views, the end), the agent's model calls and action_proposed, and an
// execution capture of the action. Times are seconds after T0 (the run's own clock).

import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export const T0 = 1_792_000_000_000;
export const RUN_ID = "ora-play-play-0123456789ab--20261020T000000Z";
export const STORED = Buffer.from("stored head image").toString("base64");
export const UNSTORED = Buffer.from("never stored wrist image").toString("base64");
export const MARKED = Buffer.from("mark_point panel").toString("base64");

const sha = (data: string) => createHash("sha256").update(Buffer.from(data, "base64")).digest("hex");
const iso = (s: number) => new Date(T0 + Math.round(s * 1000)).toISOString();
const unix = (s: number) => (T0 + Math.round(s * 1000)) / 1000;
const lines = (records: unknown[]) => `${records.map((record) => JSON.stringify(record)).join("\n")}\n`;

export function replayRun(): string {
  const root = join(mkdtempSync(join(tmpdir(), "ora-replay-test-")), RUN_ID);
  const native = join(root, "native");
  for (const directory of ["pi-session", "agent-artifacts", "execution-capture/frames", "operator-view"]) {
    mkdirSync(join(native, directory), { recursive: true });
  }
  writeFileSync(join(root, "plan.json"), JSON.stringify({ episode_mode: "play", task: "toys" }));
  writeFileSync(join(native, "agent-artifacts", `${sha(STORED)}.png`), Buffer.from(STORED, "base64"));
  writeFileSync(join(native, "agent-artifacts", `${sha(MARKED)}.png`), Buffer.from(MARKED, "base64"));
  const workspace = join(root, "sandbox", "workspace");

  writeFileSync(join(native, "pi-session", "2026-10-20T00-00-00-000Z_session.jsonl"), lines([
    { type: "session", version: 3, id: "s", timestamp: iso(0), cwd: workspace },
    { type: "model_change", id: "e1", parentId: null, timestamp: iso(0), provider: "ora", modelId: "deepseek-flash" },
    { type: "thinking_level_change", id: "e2", parentId: "e1", timestamp: iso(0), thinkingLevel: "high" },
    { type: "custom", customType: "ora.scene", data: { scene: 1, scenes: 1 }, id: "e3", parentId: "e2", timestamp: iso(5) },
    { type: "custom_message", customType: "ora.play_observation", display: true, id: "e4", parentId: "e3", timestamp: iso(10),
      details: { ora_observation_step: 0 }, content: [
        { type: "text", text: "{\"current_actual\": {\"observation_step\": 0}}" },
        { type: "text", text: "INITIAL ACTUAL · sequence 0 · head · latest at this message" },
        { type: "image", mimeType: "image/png", data: STORED },
        { type: "text", text: "INITIAL ACTUAL · sequence 0 · left_wrist · latest at this message" },
        { type: "image", mimeType: "image/png", data: UNSTORED },
      ] },
    { type: "message", id: "e5", parentId: "e4", timestamp: iso(10.1), message: { role: "user", timestamp: at(10.1),
      content: [{ type: "text", text: "Line up the blocks, please." }] } },
    { type: "message", id: "e6", parentId: "e5", timestamp: iso(14), message: { role: "assistant", timestamp: T0 + 11_000,
      durationMs: 3000, stopReason: "toolUse", content: [
        { type: "thinking", thinking: `The blocks are by the car. My workspace is ${workspace}/inputs.` },
        { type: "text", text: "Measuring the blocks first." },
        { type: "toolCall", id: "call-1", name: "mark_point", arguments: { view: "head", points_uv: [[1, 2]] } },
      ] } },
    { type: "message", id: "e7", parentId: "e6", timestamp: iso(15), message: { role: "toolResult", toolCallId: "call-1",
      toolName: "mark_point", timestamp: T0 + 15_000, durationMs: 500, isError: false, content: [
        { type: "text", text: `{"marks": [], "saved": "${root}/native/marks.json"}` },
        { type: "image", mimeType: "image/png", data: MARKED },
      ] } },
    { type: "message", id: "e8", parentId: "e7", timestamp: iso(18), message: { role: "assistant", timestamp: T0 + 16_000,
      durationMs: 2000, stopReason: "toolUse", content: [
        { type: "toolCall", id: "call-2", name: "set_gripper", arguments: { targets: { left: "open" } } },
      ] } },
    { type: "message", id: "e9", parentId: "e8", timestamp: iso(18.2), message: { role: "toolResult", toolCallId: "call-2",
      toolName: "set_gripper", timestamp: at(18.2), durationMs: 50, isError: false,
      content: [{ type: "text", text: "{\"status\": \"submitted\"}" }] } },
    { type: "custom_message", customType: "ora.play_observation", display: true, id: "e10", parentId: "e9", timestamp: iso(23),
      details: { ora_observation_step: 1 }, content: [{ type: "text", text: "{\"current_actual\": {\"observation_step\": 1}}" }] },
    { type: "custom_message", customType: "ora.play_message", display: true, id: "e11", parentId: "e10", timestamp: iso(23.1),
      content: "Which toy do you like?", details: { ora_message_id: "h002", ora_instruction_id: "h001", ora_instruction: 1,
        ora_observation_step: 1, ora_joined: true } },
    { type: "message", id: "e12", parentId: "e11", timestamp: iso(26), message: { role: "assistant", timestamp: T0 + 24_000,
      durationMs: 2000, stopReason: "stop", content: [{ type: "text", text: "The owl. The blocks are in a row now." }] } },
    { type: "custom", customType: "ora.instruction_end", data: { n: 1, ended: "reported" }, id: "e13", parentId: "e12", timestamp: iso(27) },
  ]));

  writeFileSync(join(root, "events.jsonl"), lines([
    { type: "run_started", run_id: RUN_ID, time: unix(0) },
    { type: "operator_view_frame", observation_step: 0, path: "native/operator-view/000000.png", camera: "operator_camera",
      width: 640, height: 480, sha256: "x", time: unix(6) },
    { type: "play_instruction_started", record: { index: 0, instruction_id: "h001", observation_step: 0, scene_index: 0 },
      time: unix(10) },
    { type: "operator_message", payload: { id: "h001", instruction_id: "h001", joined: false, message: "Line up the blocks, please.",
      observation_step: 0, submitted_unix_ns: (T0 + 9_900) * 1e6 }, time: unix(10.05) },
    { type: "action_submitted", action: { action: "set_gripper", action_id: "action-a" }, time: unix(19) },
    { type: "motion_receipt", duration_s: 2, record: { receipt: { action_id: "action-a" } }, time: unix(21) },
    { type: "operator_view_frame", observation_step: 1, path: "native/operator-view/000001.png", camera: "operator_camera",
      width: 640, height: 480, sha256: "y", time: unix(22.5) },
    { type: "operator_message", payload: { id: "h002", instruction_id: "h001", joined: true, message: "Which toy do you like?",
      observation_step: 1, submitted_unix_ns: (T0 + 20_000) * 1e6 }, time: unix(23.05) },
    { type: "play_instruction_finished", record: { index: 0, instruction_id: "h001", ended: "reported", decisions: 1,
      observation_steps: { start: 0, end: 1 }, wall_duration_s: 17 }, time: unix(27) },
    { type: "play_control", record: { command: "quit", status: "applied" }, time: unix(30) },
    { type: "run_finished", result: { termination: "operator_quit" }, time: unix(31) },
  ]));
  writeFileSync(join(root, "agent-events.jsonl"), lines([
    { type: "provider_request", observation_step: 0, model_call_id: "play-call-000000", payload: { play: true }, unix_s: unix(11) },
    { type: "provider_response", observation_step: 0, model_call_id: "play-call-000000",
      payload: { play: true, usage: { input_tokens: 9000, output_tokens: 150 } }, unix_s: unix(14) },
    { type: "provider_request", observation_step: 0, model_call_id: "play-call-000001", payload: { play: true }, unix_s: unix(16) },
    { type: "provider_response", observation_step: 0, model_call_id: "play-call-000001",
      payload: { play: true, usage: { input_tokens: 9400, output_tokens: 50 } }, unix_s: unix(18) },
    { type: "action_proposed", observation_step: 0, payload: { action_id: "action-a" }, unix_s: unix(18.1) },
  ]));
  const frame = (index: number, at: number, phase: string, action: string | null) => ({
    frame_index: index, captured_unix_s: unix(at), phase, sim_time_s: index / 10, action_id: action,
    images: { head: { path: `frames/0000000${index}-camera-0.png` }, operator_camera: { path: `frames/0000000${index}-camera-3.png` } },
  });
  writeFileSync(join(native, "execution-capture", "timeline.jsonl"), lines([
    frame(0, 4, "reset", null), frame(1, 19.1, "action_start", "action-a"), frame(2, 20, "control", "action-a"),
    frame(3, 20.9, "action_end", "action-a"),
  ]));
  writeFileSync(join(native, "execution-capture", "manifest.json"), JSON.stringify({
    camera_panels: ["head", "operator_camera"], config: { fps: 10 } }));
  return root;
}

export const at = (s: number) => T0 + Math.round(s * 1000);
