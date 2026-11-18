// The operator TUI's own parts: the remote-terminal relay, the sandboxed
// existsSync, ORA's command interception, run-status folding and the image
// lookup. Pi's InteractiveMode itself is exercised end to end by
// tests/support/agent_core/tui_attach_demo.py.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { connect, createServer, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { test } from "node:test";

import { OraTui, type OraTuiSource } from "../src/tui/extension.ts";
import { elapsed, endingText, setLanguage, t, uiLanguage } from "../src/tui/i18n.ts";
import { PlayMessages } from "../src/play/messages.ts";
import { PlayPerf } from "../src/play/perf.ts";
import { LineChannel, parseHello, PROTOCOL_VERSION, RemoteTerminal, type Message } from "../src/tui/remote_terminal.ts";
import { RunStatus, worldSummary, worldView } from "../src/tui/status.ts";
import { argumentLine, cameraOf, collapseRepeats, imageCaptions, observationLines, resultSummary } from "../src/tui/render.ts";

const ORA_PI = resolve(import.meta.dirname, "..");
setLanguage("en");

function scratch(label: string): string {
  return mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), `ora-pi-tui-${label}-`));
}

test("the remote terminal relays keys in and screen writes out", async () => {
  const path = join(scratch("relay"), "t.sock");
  const terminal = new RemoteTerminal();
  const keys: string[] = [];
  let resized = 0;
  terminal.start((data) => keys.push(data), () => resized++);
  const server = createServer((socket: Socket) => {
    let channel: LineChannel;
    channel = new LineChannel(socket, (message: Message) => {
      if (message.type === "hello") terminal.attach(channel, parseHello(message));
      else terminal.receive(message);
    }, () => terminal.detach());
  });
  await new Promise<void>((done) => server.listen(path, done));
  const received: Message[] = [];
  const client = connect(path);
  const channel = new LineChannel(client, (message) => received.push(message), () => undefined);
  await new Promise<void>((done) => client.on("connect", () => done()));
  channel.send({ type: "hello", version: PROTOCOL_VERSION, columns: 132, rows: 40, kitty: true, capabilities: null });
  channel.send({ type: "input", data: "\x1b[A" });
  channel.send({ type: "resize", columns: 100, rows: 30 });
  await waitFor(() => keys.length === 1 && resized === 1);
  assert.deepEqual([terminal.columns, terminal.rows, terminal.kittyProtocolActive], [100, 30, true]);
  terminal.write("frame");
  terminal.moveBy(-2);
  terminal.setTitle("ORA");
  await waitFor(() => received.length === 3);
  assert.deepEqual(received, [
    { type: "write", data: "frame" }, { type: "write", data: "\x1b[2A" }, { type: "title", title: "ORA" },
  ]);
  assert.throws(() => parseHello({ type: "hello", version: PROTOCOL_VERSION + 1 }), /unsupported attach protocol/);
  client.destroy();
  await waitFor(() => !terminal.attached);
  terminal.write("dropped while detached");
  server.close();
});

test("under the permission model a path outside the sandbox does not exist", () => {
  const probe = [
    `import { existsSync } from "node:fs";`,
    `import { sandboxedExistsSync } from ${JSON.stringify(join(ORA_PI, "src/tui/sandbox.ts"))};`,
    `let before; try { existsSync("/.git"); before = "no throw"; } catch (error) { before = error.code; }`,
    `sandboxedExistsSync();`,
    `console.log(JSON.stringify([before, existsSync("/.git"), existsSync("/.agents/skills"), existsSync(${JSON.stringify(join(ORA_PI, "package.json"))})]));`,
  ].join("\n");
  const result = spawnSync(process.execPath, [
    "--permission", `--allow-fs-read=${ORA_PI}`, "--input-type=module", "-e", probe,
  ], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), ["ERR_ACCESS_DENIED", false, false, true]);
});

function source(runRoot: string, overrides: Partial<OraTuiSource> = {}): OraTuiSource {
  return {
    runRoot, runId: "run-1", model: "m", profile: "p", protocol: "openai_compatible",
    mutationTools: ["robot.submit_action"], toolNames: ["robot.submit_action"],
    history: () => [], pendingToolImages: () => [], live: () => true, inTurn: () => false,
    stop: () => undefined, interruptAtBoundary: () => undefined, notifySupervisor: () => undefined,
    sessionFile: () => null, play: () => null, playInstruction: () => null, control: async () => null,
    ...overrides,
  };
}

function withNotices(tui: OraTui): Array<[string, string]> {
  const notices: Array<[string, string]> = [];
  (tui as any).ui = { notify: (message: string, level: string) => notices.push([level, message]) };
  return notices;
}

test("Pi built-ins that would rewrite or leave a live run are refused", () => {
  const tui = new OraTui(source(scratch("intercept")), { live: true });
  const notices = withNotices(tui);
  for (const text of ["/tree", "/compact keep this", "/share", "!ls", "/frobnicate"]) {
    assert.equal(tui.intercept(text), true, text);
  }
  assert.match(notices[0][1], /\/tree is not available here: a live episode's history cannot be rewound/);
  assert.match(notices[3][1], /Shell commands are not available/);
  assert.match(notices[4][1], /Unknown command \/frobnicate/);
  for (const text of ["/status", "/stop", "/help", "/session", "/login", "/logout", "steer the robot left"]) {
    assert.equal(tui.intercept(text), false, text);
  }
  assert.equal(tui.intercept("/model"), true);
  assert.match(notices.at(-1)![1], /m via openai_compatible \(profile p\); bound by the RunPlan/);
  assert.equal(tui.intercept("/quit"), false, "without a host the terminal is the agent process: Pi quits");
  const detached: string[] = [];
  tui.host = { detach: (reason) => detached.push(reason), exportSession: async () => "x" };
  assert.equal(tui.intercept("/quit"), true);
  assert.deepEqual(detached, ["detached; the run continues"]);

  const view = new OraTui(source(scratch("view")), { live: false });
  withNotices(view);
  assert.equal(view.intercept("move on"), true, "a finished run's view is read-only");
  assert.equal(view.intercept("/name x"), true);
});

test("run status folds the agent events: calls, decisions, last action and its receipt", () => {
  const root = scratch("status");
  mkdirSync(join(root, "native", "agent-artifacts"), { recursive: true });
  writeFileSync(join(root, "native/agent-artifacts/r.json"), JSON.stringify({ consumed: true, pose_reached: true, action: {} }));
  writeFileSync(join(root, "native/agent-artifacts/e.json"), JSON.stringify({ ok: false, error: { code: "out_of_bounds", message: "z" } }));
  const events = [
    { type: "observation_received", observation_step: 0 },
    { type: "provider_request" }, { type: "provider_response", payload: { usage: { input_tokens: 10, output_tokens: 2 } } },
    { type: "tool_call_proposed", tool_call_id: "a", observation_step: 0, payload: { name: "robot.submit_action", arguments: { x: 1 } } },
    { type: "tool_result", tool_call_id: "a", observation_step: 0, payload: { name: "robot.submit_action", ok: false },
      artifact_ref: { path: "native/agent-artifacts/e.json" } },
    { type: "provider_request" },
    { type: "tool_call_proposed", tool_call_id: "b", observation_step: 0, payload: { name: "robot.submit_action", arguments: { x: 2 } } },
    { type: "tool_result", tool_call_id: "b", observation_step: 0, payload: { name: "robot.submit_action", ok: true } },
    { type: "action_proposed", tool_call_id: "b", payload: { action_id: "act-1" } },
    { type: "environment_receipt", payload: { action_id: "act-1" }, artifact_ref: { path: "native/agent-artifacts/r.json" } },
    { type: "operator_message", payload: {} },
  ];
  writeFileSync(join(root, "agent-events.jsonl"), events.slice(0, 5).map((e) => JSON.stringify(e)).join("\n") + "\n");
  const status = new RunStatus(root, ["robot.submit_action"]).refresh();
  assert.equal(status.lastAction?.refusal, "out_of_bounds: z");
  writeFileSync(join(root, "agent-events.jsonl"), events.map((e) => JSON.stringify(e)).join("\n") + "\n");
  status.refresh();
  assert.deepEqual([status.modelCalls, status.decisions, status.operatorMessages, status.inputTokens], [2, 1, 1, 10]);
  assert.deepEqual(status.lastAction, {
    step: 0, name: "robot.submit_action", arguments: { x: 2 }, ok: true, actionId: "act-1",
    receipt: "consumed=true pose_reached=true", refusal: null,
  });
  status.observeTurn(1, { robot_decisions: 2, model_calls: 7, wall_time_s: 125.5 });
  // A finished run's view does not know the mutation tools: the action names its call.
  const view = new RunStatus(root, []).refresh();
  assert.deepEqual([view.lastAction?.name, view.lastAction?.actionId, view.lastAction?.receipt],
    ["robot.submit_action", "act-1", "consumed=true pose_reached=true"]);
  assert.deepEqual(status.remaining().map(([key]) => key), ["robot decisions", "model calls", "wall time"]);
  // A proposal a round budget discarded is never executed.
  const discarded = [...events.slice(0, 9), { type: "action_discarded", payload: { action_id: "act-1", reason: "model_call_budget" } }];
  writeFileSync(join(root, "agent-events.jsonl"), discarded.map((e) => JSON.stringify(e)).join("\n") + "\n");
  const ended = new RunStatus(root, ["robot.submit_action"]).refresh();
  assert.deepEqual([ended.decisions, ended.lastAction?.actionId, ended.lastAction?.receipt],
    [0, "act-1", "not executed · model_call_budget"]);
});

test("observation and tool images come from ORA's history, labelled by their artifact", () => {
  const root = scratch("images");
  mkdirSync(join(root, "native", "agent-artifacts"), { recursive: true });
  const png = Buffer.from("89504e470d0a1a0a0000000d4948445200000004000000030806000000", "hex");
  const digest = createHash("sha256").update(png).digest("hex");
  writeFileSync(join(root, `native/agent-artifacts/${digest}.png`), png);
  const url = `data:image/png;base64,${png.toString("base64")}`;
  const history = [
    { role: "user", _observation: { step: 3 }, content: [{ type: "text", text: "obs" }, { type: "image_url", image_url: { url } }] },
    { role: "user", _tool_image_call_id: "call-9", content: [{ type: "image_url", image_url: { url } }] },
  ];
  const tui = new OraTui(source(root, { history: () => history }), { live: true });
  assert.deepEqual(tui.imagesForObservation(3).map((image) => [image.mimeType, image.label]),
    [["image/png", `native/agent-artifacts/${digest}.png`]]);
  assert.equal(tui.imagesForObservation(4).length, 0);
  assert.equal(tui.imagesForTool("call-9").length, 1);
});

async function waitFor(predicate: () => boolean, timeoutMs = 5000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error("timed out");
    await new Promise((done) => setTimeout(done, 10));
  }
}

test("tool calls read as their key arguments, results as accepted or refused and why", () => {
  assert.equal(argumentLine({ arm: "left", pose: { position_m: [0, 0, 1.02], orientation_xyzw: [0, 0, 0, 1] } }),
    "arm=left  pose.position_m=[0, 0, 1.02]  pose.orientation_xyzw=[0, 0, 0, 1]");
  assert.equal(argumentLine({ targets: { right: { position_m: [0.006, -0.096, 1.02] } } }),
    "targets.right.position_m=[0.006, -0.096, 1.02]");
  assert.equal(argumentLine({ targets: { right: 90 } }), "targets.right=90");
  assert.equal(argumentLine({ check: {} }), "");
  assert.equal(argumentLine({ bindings: [{ name: "q" }, { name: "c" }], rationale: "Top-down grasp at bottle middle, right arm" }),
    'bindings=2 items  rationale="Top-down grasp at bottle middle, right…"');
  assert.deepEqual(resultSummary('{"accepted": true, "action_id": "action-1", "directive": "action_submitted"}'),
    { status: "ok", line: "accepted · action-1 · directive=action_submitted" });
  assert.deepEqual(resultSummary('{"ok": false, "error": {"code": "out_of_bounds", "message": "z beyond the workspace"}}'),
    { status: "refused", line: "refused · out_of_bounds: z beyond the workspace" });
  assert.deepEqual(resultSummary('{"content": "## Actions\\n\\n`robot.submit_action` accepts..."}'),
    { status: "ok", line: "## Actions" });
  assert.equal(resultSummary('{"camera": "head_camera", "candidate_previews": [{"rank": 0}, {"rank": 1}], "target_ref_count": 2}').line,
    "camera=head_camera · candidate_previews ×2 · target_ref_count=2");
  assert.deepEqual(resultSummary("not json\nsecond line"), { status: "plain", line: "not json" });
});

test("an observation reads as cameras, the last action's effect, the arms and the budget", () => {
  const causal = {
    content: [
      { type: "text", text: JSON.stringify({
        comparison: { available: true }, current_actual: { actual_sequence: 2 },
        last_action_effect: { action: "move_to", action_summary: "right arm 179 mm from target, planner failed, not moved" },
        remaining_budget: { model_calls: 197, robot_decisions: 78, wall_time_s: 21480.5 },
      }) },
      { type: "text", text: "ACTION RESULT ACTUAL · sequence 2 · front_camera · latest at this message" },
      { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
      { type: "text", text: "ACTION RESULT ACTUAL · sequence 2 · head_camera · latest at this message" },
      { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
    ],
  };
  assert.deepEqual(observationLines(causal, "cut at 4000 characters {"), [
    "front_camera, head_camera · 2 images",
    "after move_to: right arm 179 mm from target, planner failed, not moved",
    "budget left: 78 robot decisions · 197 model calls · 5h58m",
  ]);
  assert.deepEqual(imageCaptions(causal).map(cameraOf), ["front_camera", "head_camera"]);
  const synthetic = {
    content: [{ type: "text", text: JSON.stringify({ current_observation: {
      main_camera: "head_camera", previous_execution: { action: { action: "target_state" }, consumed: true, pose_reached: true },
      proprioception: { end_effectors: { left: { gripper: 1.0, pose_xyz_xyzw: [0, 0, 1.02, 0, 0, 0, 1] } } },
      remaining_budget: { model_calls: 50, robot_decisions: 5, wall_time_s: 3590 },
    } }) }, { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }],
  };
  assert.deepEqual(observationLines(synthetic, ""), [
    "head_camera · 1 image", "after target_state: executed, pose reached", "left [0, 0, 1.02] gripper 1",
    "budget left: 5 robot decisions · 50 model calls · 59m50s",
  ]);
  assert.deepEqual(observationLines(null, "ORA resumed this turn after an agent-process restart"),
    ["ORA resumed this turn after an agent-process restart"]);
});

test("the attaching terminal's world view reaches the header and /status, as plain text", () => {
  const hello = parseHello({ type: "hello", version: PROTOCOL_VERSION, columns: 80, rows: 24, kitty: false,
    capabilities: null, world: { url: "http://127.0.0.1:43211/live", phase: "waiting", frame_index: null } });
  assert.equal(hello.world?.url, "http://127.0.0.1:43211/live");
  assert.equal(parseHello({ type: "hello", version: PROTOCOL_VERSION }).world, null, "an older client brings none");

  const tui = new OraTui(source(scratch("world")), { live: true });
  assert.equal(tui.worldLine(), null);
  assert.ok(tui.statusRows().some((row) => /^world view\s+none in this terminal/.test(row)));
  tui.setWorld(hello.world);
  assert.equal(tui.worldLine(), "http://127.0.0.1:43211/live · open it in a browser; waiting for the first camera frame");
  tui.setWorld({ url: "http://127.0.0.1:43211/live", phase: "holding", connected: true, visible: true, frame_index: 7 });
  assert.ok(tui.statusRows().includes(`${"world view".padEnd(16)}http://127.0.0.1:43211/live · browser shows frame 7`));
  tui.setWorld(null);
  assert.equal(tui.worldLine(), null, "a terminal without a world view clears the previous one");

  // The relayed state is drawn on the operator's terminal: no escape sequences, no other schemes.
  const hostile = worldView({ url: "javascript:alert(1)", phase: "unavailable",
    message: "World view unavailable: \x1b]52;c;cGF3bmVk\x07OSError: in use" });
  assert.equal(hostile?.url, null);
  assert.equal(worldSummary(hostile!), "World view unavailable: ]52;c;cGF3bmVkOSError: in use");
  assert.equal(worldSummary(worldView({ url: "http://127.0.0.1:1/live", phase: "waiting",
    browser_message: "SSH session: forward this port and open the URL on your computer." })!),
    "SSH session: forward this port and open the URL on your computer.");
});

test("free play: Pi's commands pass, Esc stops only the running instruction, uploads ask first", async () => {
  const sent: Array<[string, string]> = [];
  let play: any = { state: "awaiting_instruction", scene: 1, scenes: 3, instructions: 0, running: null };
  const tui = new OraTui(source(scratch("play"), {
    freePlay: true, play: () => play,
    control: async (type, message) => { sent.push([type, message]); return { id: "c" }; },
  }), { live: true });
  const notices = withNotices(tui);
  const entries: any[] = [];
  (tui as any).pi = { appendEntry: (type: string, data: any) => entries.push([type, data]) };
  assert.equal(tui.phase(), "waiting for your instruction");
  assert.equal(tui.onKey("\x1b"), undefined, "Esc with nothing running is Pi's own key");
  for (const text of ["pick up the red block", "!ls", "/tree", "/model", "/compact", "/login", "/new", "/thinking",
    "/reload", "/skill:look", "@notes.md what is this"]) {
    assert.equal(tui.intercept(text), false, `${text} goes to Pi (instructions to the input hook)`);
  }

  play = { ...play, state: "instruction", instructions: 1, running: "pi-1", instruction_id: "pi-1" };
  tui.playState(play);
  assert.match(notices.at(-1)![1], /Instruction 1 is starting/);
  assert.notEqual(tui.phase(), "waiting for your instruction");
  (tui as any).tui = { hasOverlay: () => true };
  assert.equal(tui.onKey("\x1b"), undefined, "Esc closes an open panel first");
  (tui as any).tui = { hasOverlay: () => false };
  assert.deepEqual(tui.onKey("\x1b"), { consume: true }, "Esc is taken before Pi's abort, which would stop the run");
  assert.equal(tui.onKey("x"), undefined);

  tui.playState({ ...play, state: "instruction_finished", running: null,
    instruction: { index: 0, ended: "operator_stopped", observation_steps: { start: 0, end: 4 } } });
  assert.match(notices.at(-1)![1], /Instruction 1 stopped \(Esc\)\. Type the next one\./);
  assert.deepEqual(entries.at(-1), ["ora.instruction_end", { n: 1, ended: "operator_stopped", step: 4, decisions: null,
    seconds: null }], "the stop is marked in the conversation, with the step it stopped at");
  tui.playState({ ...play, state: "control", running: null,
    control: { command: "reset", status: "rejected", reason: "no_scene_left" } });
  assert.match(notices.at(-1)![1], /reset: every scene drawn for this session has been used/);
  tui.playState({ ...play, state: "scene", scene: 2, cause: "operator", running: null,
    image: "native/play-scenes/scene-0001.png" });
  assert.match(notices.at(-1)![1], /Scene 2 of 3 is loaded/);
  assert.deepEqual(entries.at(-1), ["ora.scene", { scene: 2, scenes: 3, image: "native/play-scenes/scene-0001.png",
    operator_view: null }], "a display-only entry: the model gets the scene with the next instruction's observation");

  const resubmitted: string[] = [];
  tui.resubmit = (text) => { resubmitted.push(text); };
  let answer = false;
  (tui as any).ui.confirm = async (title: string, message: string) => {
    assert.match(`${title} ${message}`, /Upload with \/share\?.*whole conversation.*gist/s);
    return answer;
  };
  assert.equal(tui.intercept("/share"), true);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(resubmitted, [], "nothing is uploaded without a confirmation");
  answer = true;
  assert.equal(tui.intercept("/share"), true);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(resubmitted, ["/share"]);

  tui.host = { detach: () => assert.fail("/quit ends a free-play session; it does not detach"), exportSession: async () => "x" };
  assert.equal(tui.intercept("/quit"), true);
  assert.equal(tui.intercept("/reset"), false, "/reset is ORA's command");
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(sent, [["stop_instruction", "Esc"], ["quit", "/quit"]]);
  const help = tui.helpRows().join("\n");
  assert.match(help, /Esc {12}stop the running instruction/);
  assert.match(help, /a message joins it/);
  assert.doesNotMatch(help, /replaces/);
  assert.match(help, /the robot and the scene stay as/);
  assert.doesNotMatch(help, /Not available here/);
});

test("a steer shows as the operator's message once the Supervisor delivered it", () => {
  const history: any[] = [
    { role: "user", content: [], _observation: { step: 2 } },
    { role: "assistant", content: "thinking" },
    { role: "user", _context_source: "operator_message_s1",
      content: "<operator_message>\n{\"id\":\"s1\",\"instruction\":\"...\",\"message\":\"keep the gripper level\"}\n</operator_message>" },
    { role: "user", content: [], _observation: { step: 3 } },
  ];
  const tui = new OraTui(source(scratch("echo"), { history: () => history }), { live: true });
  assert.deepEqual(tui.operatorLines(2), ["› you: keep the gripper level"]);
  assert.deepEqual(tui.operatorLines(3), []);
  assert.match(tui.statusLine(), /budget left|waiting for the first observation/);
});

test("a summary that repeats a part says it once on screen", () => {
  assert.equal(collapseRepeats("right arm 155 mm / 0.3° from target, executed, executed, round ended"),
    "right arm 155 mm / 0.3° from target, executed, round ended");
  assert.equal(collapseRepeats("executed, not moved, executed"), "executed, not moved, executed");
  const lines = observationLines(null, JSON.stringify({ last_action_effect: {
    action: "move_pose", action_summary: "right arm 155 mm / 0.3° from target, executed, executed, round ended",
  } }));
  assert.deepEqual(lines, ["after move_pose: right arm 155 mm / 0.3° from target, executed, round ended"]);
});

test("free play's footer says what the agent does now, with the seconds it has taken", () => {
  let play: any = { state: "instruction", scene: 1, scenes: 2, instructions: 1, running: "pi-1" };
  const tui = new OraTui(source(scratch("live"), { freePlay: true, play: () => play }), { live: true });
  withNotices(tui);
  (tui as any).pi = { appendEntry: () => undefined };
  (tui as any).onAgentEvent("turn_start", {});
  const start = (tui as any).activity.since;
  assert.equal(tui.phase(start + 4200), "thinking… 4s");
  (tui as any).onAgentEvent("message_start", { message: { role: "assistant" } });
  assert.equal(tui.phase(start + 6000), "writing… 6s", "a reply being written counts from the call's start");
  (tui as any).onAgentEvent("tool_execution_start", { toolName: "vision_ground" });
  assert.match(tui.phase(), /^running vision_ground… \ds$/);
  tui.playState({ ...play, state: "action_started", action: { action_id: "a1", observation_step: 3 } });
  assert.equal(tui.phase((tui as any).activity.since + 2000), "moving… 2s");
  tui.playState({ ...play, state: "action_finished", action: { action_id: "a1", observation_step: 3, duration_s: 1.5 } });
  assert.equal(tui.phase(), "observing…");
  assert.match(tui.statusLine(), /scene 1\/2 · instruction 1 · .*observing…/);
  play = { ...play, state: "instruction_finished", running: null, instruction: { index: 0, ended: "reported" } };
  tui.playState(play);
  assert.equal(tui.phase(), "waiting for your instruction");
  assert.equal(elapsed(65_000), "1m 05s");
});

test("free play's footer has a second line with the model's speed and the loop (play/perf.ts)", () => {
  const perf = new PlayPerf(() => 0);
  const tui = new OraTui(source(scratch("perf"), { freePlay: true, perfLine: () => perf.line(3) }), { live: true });
  assert.equal(tui.perfLine(), null, "nothing before the first timed call");
  let now = 0;
  const timed = new PlayPerf(() => now);
  timed.observed(0);
  timed.requested("c", 0);
  now = 800;
  timed.streamed({ type: "text_delta" });
  now = 3800;
  timed.replied({ usage: { output: 135 } });
  const live = new OraTui(source(scratch("perf2"), { freePlay: true, perfLine: () => timed.line(3) }), { live: true });
  assert.equal(live.perfLine(), "45 tok/s · TTFT 0.8 s · loop — (target 3 s)");
  assert.equal(new OraTui(source(scratch("perf3")), { live: true }).perfLine(), null, "an episode has none");
});

test("a message joins the running instruction: queued, then read at a step, its state for every client", () => {
  const log = new PlayMessages();
  const changes: any[] = [];
  log.onChange = (message) => changes.push({ ...message });
  log.submitted("m1", "how far along are you?", 2);
  assert.deepEqual(log.queued().map((item) => [item.id, item.joined, item.instruction]), [["m1", true, 2]]);
  log.apply({ state: "message_queued", message: { id: "m1", text: "how far along are you?", instruction: 2 } });
  log.apply({ state: "message_delivered", message: { id: "m1", text: "how far along are you?", instruction: 2,
    joined: true, observation_step: 7 } });
  assert.deepEqual(log.get("m1"), { ...log.get("m1"), state: "delivered", joined: true, instruction: 2, step: 7 });
  assert.equal(log.queued().length, 0);
  // A message typed between instructions opens the next one.
  log.submitted("i3", "put the cup on the plate", null);
  log.apply({ state: "instruction", message: { id: "i3", text: "put the cup on the plate", instruction: 3 } });
  assert.deepEqual([log.get("i3")!.joined, log.get("i3")!.instruction], [false, 3]);
  assert.equal(log.apply({ state: "scene" }), null);
  assert.deepEqual(changes.map((item) => [item.id, item.state]), [
    ["m1", "queued"], ["m1", "queued"], ["m1", "delivered"], ["i3", "queued"], ["i3", "queued"]]);
});

test("a queued message waits for the running instruction to end: a notice, the footer's count, then the next one", async () => {
  const { submitPlayMessage } = await import("../src/play/runtime.ts");
  const { PlayDriver } = await import("../src/play/driver.ts");
  const log = new PlayMessages();
  const submitted: Array<[string, string]> = [];
  const core: any = { playState: { state: "instruction", running: "pi-2", instructions: 2, scene: 1, scenes: 1 },
    submitOperatorMessage: async (message: string, type: string) => { submitted.push([type, message]); return { id: `c${submitted.length}` }; } };
  const driver: any = { attachments: new Map(), messages: log };
  const said: string[] = [];
  await submitPlayMessage(core, driver, " then stack them ", [], { queued: true, notify: (text) => said.push(text) });
  await submitPlayMessage(core, driver, "slower", [], { notify: (text) => said.push(text) });
  assert.deepEqual(submitted, [["queued_instruction", "then stack them"], ["instruction", "slower"]]);
  assert.match(said[0], /Queued: the next instruction, once instruction 2 ends/);
  assert.match(said[1], /joins instruction 2/);
  assert.deepEqual(log.queuedNext().map((item) => [item.id, item.joined, item.next]), [["c1", false, true]]);
  // The Supervisor's announcement says the same; the instruction it opens takes it out of the queue.
  log.apply({ state: "message_queued_next", message: { id: "c1", text: "then stack them", instruction: 2 } });
  const tui = new OraTui(source(scratch("queued"), { freePlay: true, play: () => core.playState,
    queuedNext: () => log.queuedNext().length }), { live: true });
  const notices = withNotices(tui);
  (tui as any).pi = { appendEntry: () => undefined };
  tui.playState({ ...core.playState, state: "message_queued_next", message: { id: "c1", text: "then stack them", instruction: 2 } });
  assert.match(notices.at(-1)![1], /Held for the next instruction \(1 queued\): it starts once instruction 2 ends/);
  assert.match(tui.statusLine(), / · 1 queued$/);
  log.apply({ state: "instruction", message: { id: "c1", text: "then stack them", instruction: 3 } });
  assert.deepEqual([log.queuedNext().length, log.get("c1")!.instruction, log.get("c1")!.joined], [0, 3, false]);
  assert.doesNotMatch(tui.statusLine(), /queued/);

  // Alt+Enter while an instruction runs: the message Pi submits next is queued (taken once, within a second).
  let asked = 0;
  const keys = new OraTui(source(scratch("alt"), { freePlay: true, play: () => core.playState, queueNext: () => { asked += 1; } }),
    { live: true });
  let editor = "";
  (keys as any).ui = { notify: () => undefined, getEditorText: () => editor };
  assert.equal(keys.onKey("\x1b\r"), undefined, "Pi's own Alt+Enter submits it");
  assert.equal(asked, 0, "nothing typed, nothing queued");
  editor = "/halt";
  keys.onKey("\x1b\r");
  assert.equal(asked, 0, "a command is never queued");
  editor = "then the blue one";
  keys.onKey("\x1b\r");
  assert.equal(asked, 1);
  const play = Object.create(PlayDriver.prototype);
  play.queueNext(1000);
  assert.equal(play.takeQueued(1500), true);
  assert.equal(play.takeQueued(1600), false, "taken once");
  play.queueNext(1000);
  assert.equal(play.takeQueued(2500), false, "a stale Alt+Enter queues nothing");
});

test("free play's terminal words in Chinese: footer, endings, help", () => {
  assert.equal(uiLanguage({ ORA_UI_LANG: "zh" }), "zh");
  assert.equal(uiLanguage({ LANG: "zh_CN.UTF-8" }), "zh");
  assert.equal(uiLanguage({ ORA_UI_LANG: "en", LANG: "zh_CN.UTF-8" }), "en");
  assert.equal(uiLanguage({ LANG: "C.UTF-8" }), "en");
  setLanguage("zh");
  try {
    const play: any = { state: "instruction_finished", scene: 1, scenes: 2, instructions: 2, running: null };
    const tui = new OraTui(source(scratch("zh"), { freePlay: true, play: () => play }), { live: true });
    const notices = withNotices(tui);
    (tui as any).pi = { appendEntry: () => undefined };
    tui.playState({ ...play, instruction: { index: 1, ended: "operator_stopped" } });
    assert.equal(notices.at(-1)![1], "第 2 条指令已停止（Esc）。请输入下一条。");
    assert.match(tui.statusLine(), /^场景 1\/2 · 第 2 条指令 · .* · 等待你的指令/);
    assert.equal(endingText("reported"), "已完成，智能体已汇报");
    assert.equal(t("play.messageQueued", { n: 2 }), "已排队：并入第 2 条指令，智能体会在下一步之前读到。按 Esc 停止这条指令。");
    assert.match(tui.helpRows().join("\n"), /指令执行中发送的消息会并入当前指令/);
    assert.equal(elapsed(4200), "4 秒");
    tui.observeTurn(3, { robot_decisions: 397, model_calls: 1590 });
    assert.match(tui.statusLine(), /剩余预算 397 次机器人动作, 1590 次模型调用$/);
  } finally {
    setLanguage("en");
  }
});

test("a real cell's setup reads as a checklist with each step's mark and why", async () => {
  const { setupLines, setupTitle } = await import("../src/tui/extension.ts");
  const theme = { fg: (color: string, text: string) => `<${color}>${text}`, bold: (text: string) => text } as any;
  const setup = { phase: "failed", error: "the cell's setup stopped at table: 没有看到桌面",
    steps: [{ id: "devices", title: "设备", state: "done" }, { id: "table", title: "桌面高度", state: "failed", why: "没有看到桌面" },
      { id: "verify", title: "验证", state: "missing", detail: "还没做" }] };
  assert.equal(setupTitle(setup), "failed");
  assert.equal(setupTitle({ phase: "done", skipped: true }), "skipped");
  assert.deepEqual(setupLines(setup, theme), ["<success>✓ 设备", "<error>✗ 桌面高度<dim>  没有看到桌面", "<dim>○ 验证<dim>  还没做",
    "<error>the cell's setup stopped at table: 没有看到桌面"]);
});

test("the play terminal is drawn as the teaser: header, calls and results, status line, prompt, footer", async () => {
  const { oraTheme, wantsOraTheme, logoLines, ORA_PALETTE } = await import("../src/tui/theme.ts");
  const { transcriptMarkdown } = await import("../src/tui/extension.ts");
  const { markPrompt } = await import("../src/tui/host.ts");
  const { callText, toolResultLines } = await import("../src/tui/render.ts");
  const { Editor } = await import("@earendil-works/pi-tui");
  const plain = (text: string) => text.replace(/\x1b\[[0-9;]*m/g, "").replace(/\x1b_pi:[^\x07]*\x07/g, "");
  const theme = oraTheme("truecolor");
  // Truecolor where the terminal has it, the nearest 256-color palette entry where it does not.
  assert.equal(theme.fg("accent", "x"), "\x1b[38;2;78;205;196mx\x1b[39m");
  assert.equal(oraTheme("256color").fg("accent", "x"), "\x1b[38;5;80mx\x1b[39m");
  assert.equal(theme.bg("toolPendingBg", "x"), "\x1b[49mx\x1b[49m", "no box around a tool call");
  assert.equal(ORA_PALETTE.accent, "#4ecdc4");
  assert.equal(wantsOraTheme({ ORA_TUI_THEME: "pi" }), false);
  assert.equal(wantsOraTheme({}), true);
  assert.deepEqual(logoLines(), [" ▄█▀▀█▄      ", " ▀█▄▄█▀▀▀▀██ ", "   ██    ▄██▄", "  ▄██▄   █  █", " ▀▀▀▀▀▀      "]);

  const root = scratch("teaser");
  writeFileSync(join(root, "plan.json"), JSON.stringify({ task: "operator_task", runtime: { kind: "yam", rig_mode: "sim" },
    policy: { reasoning_effort: "low" } }));
  let play: any = { state: "instruction", scene: 1, scenes: 2, instructions: 3, running: "op-3" };
  const tui = new OraTui(source(root, { freePlay: true, model: "deepseek-flash", runId: "run-7", play: () => play,
    queuedNext: () => 1, perfLine: () => "45 tok/s · TTFT 0.8 s" }), { live: true });
  (tui as any).ui = { theme, notify() {} };
  (tui as any).pi = { getThinkingLevel: () => "low" };
  (tui as any).context = { getContextUsage: () => ({ tokens: 175_700, contextWindow: 1_000_000 }) };
  const header = tui.headerLines(theme, 120).map(plain);
  assert.match(header[1], /^ {2}▄█▀▀█▄ {8}ORA v\d/);
  assert.match(header[2], /deepseek-flash · low effort · free play$/);
  assert.match(header[3], /yam\/operator_task · run-7$/);
  assert.match(header[4], /Alt\+Enter queues · Esc stops it/);

  (tui as any).activity = { kind: "moving", since: 1_000 };
  assert.equal(plain(tui.statusRow(theme, 13_400)!).slice(1), " Moving on the robot… (12s · esc to interrupt)");
  assert.match(tui.statusRow(theme, 13_400)!, /^\x1b\[38;2;78;205;196m/, "a teal spinner");
  const footer = plain(tui.footerLines(theme, 120)[0]);
  assert.match(footer, /^ ▸▸ free play · scene 1\/2 · instruction 3 · scene loaded · 1 queued · context 175\.7k \/ 1M +● low · deepseek-flash $/);
  assert.equal(footer.length, 120, "the counts and the budget make room (/status has them)");
  assert.match(plain(tui.footerLines(theme, 200)[0]), /context 175\.7k \/ 1M · 0 calls · 0 decisions +●/);
  assert.equal(plain(tui.footerLines(theme, 120)[1]), " 45 tok/s · TTFT 0.8 s");
  play = { ...play, state: "instruction_finished", running: null };
  (tui as any).activity = null;
  assert.equal(plain(tui.statusRow(theme)!), "· waiting for your instruction");

  // The model's words after ●, the operator's after ❯ (display only); other Markdown as it is.
  assert.equal(transcriptMarkdown("Blue localized.", "assistant"), "● Blue localized.");
  assert.equal(transcriptMarkdown("stack the blocks", "user"), "❯ stack the blocks");
  assert.equal(transcriptMarkdown("# Plan\n- a", "assistant"), "# Plan\n- a");
  assert.equal(transcriptMarkdown("thinking", "assistant-thinking"), "thinking");

  // A call as the model made it, its result as a few lines.
  assert.deepEqual(callText("vision_segment", { max_candidates: 1, points_uv: [[357, 247]], camera: "head" }),
    { args: '(camera="head", points_uv=[[357, 247]], max_candidates=1)', more: [] });
  assert.equal(callText("set_gripper", { targets: { left: "close", right: "close" } }).args, '(left="close"; right="close")');
  assert.deepEqual(toolResultLines("grasp_execute", { committed_sequence: ["open", "pregrasp", "close"] }),
    ["committed: open → pregrasp → close"]);
  const renderers: any[] = [];
  tui.extension()({ on() {}, registerMessageRenderer() {}, registerEntryRenderer() {}, registerCommand() {},
    registerMarkdownTransformer() {}, registerToolRenderer: (resolver: any) => renderers.push(resolver) } as any);
  const draw = renderers[0]("robot.submit_action", () => undefined);
  assert.equal(draw.renderShell, "self");
  const call = draw.renderCall({ arm: "right", target_ref: "target-9" }, theme, { outputPad: 1, isPartial: false });
  assert.deepEqual(call.render(80).map(plain), [' ● robot.submit_action(target_ref="target-9", arm="right")']);
  assert.match(call.render(80)[0], theme.bold("x") === "x" ? /\x1b\[38;2;246;247;249mrobot\.submit_action/
    : /\x1b\[1m\x1b\[38;2;246;247;249mrobot\.submit_action/, "the name (bold) white");
  const result = draw.renderResult({ content: [{ type: "text", text: JSON.stringify({ ok: false, error: { code: "no_target" } }) }] },
    { expanded: false }, theme, { outputPad: 1, toolCallId: "t1" });
  assert.deepEqual(result.render(80).map(plain), ["   ⎿  ✗ refused · no_target"]);

  // The prompt after ❯ between two full-width rules.
  const editor: any = new Editor({ requestRender() {}, terminal: { rows: 20, columns: 40 } } as any,
    { borderColor: (text: string) => text, selectList: {} } as any);
  editor.setText("put the cup down");
  markPrompt(editor, () => "❯ ");
  const box = editor.render(40).map(plain);
  assert.deepEqual([box[0], box[1].trimEnd(), box[2]], ["─".repeat(40), " ❯ put the cup down", "─".repeat(40)]);
  tui.close();
});

test("a code-mode command reads as its lines, and its result as its calls, its error and the end of its output", async () => {
  const { commandLines, isScriptResult, scriptResultLines, COMMAND_MAX, OUTPUT_TAIL } = await import("../src/tui/render.ts");
  const program = "python - <<'EOF'\nfrom robot import tools\nr = tools.move_pose(arm='left', dry_run=True)\nEOF";
  assert.deepEqual(commandLines({ command: program }), program.split("\n"));
  assert.deepEqual(commandLines({ command: "ls", timeout: 30 }), ["ls", "# timeout 30 s"]);
  const long = Array.from({ length: 30 }, (_, index) => `print(${index})`).join("\n");
  const collapsed = commandLines({ command: long })!;
  assert.equal(collapsed.length, COMMAND_MAX);
  assert.equal(collapsed[COMMAND_MAX - 1], `… +${30 - COMMAND_MAX + 1} lines (Ctrl+O)`);
  assert.equal(commandLines({ command: long }, true)!.length, 30);
  assert.equal(commandLines({ path: "a.py" }), null, "read, write and edit keep their own call line");

  const result = {
    ok: false, status: "stopped", exit_code: null, actions_executed: 1, wall_s: 4.25, output: "a\nb\nc\nd\nwaiting\n",
    error_detail: { type: "Stopped", code: "operator_instruction_stop", message: "the operator stopped the command" },
    calls: [
      { call: 1, tool: "observe", ok: true, images: 1 },
      { call: 2, tool: "move_pose", ok: true, dry_run: true, summary: "plan: 0.42 m, 3.1 s\nmore" },
      { call: 3, tool: "move_pose", ok: true, action: 1 },
      { call: 4, tool: "close_gripper", ok: false, error: "operator_stop_pending" },
    ],
  };
  assert.equal(isScriptResult(result), true);
  assert.equal(isScriptResult({ ok: true, output: "x" }), false);
  const lines = scriptResultLines(result);
  assert.deepEqual(lines.map((line) => line.text), [
    "stopped · 4 calls · 1 action · 1 planned · 4.25 s",
    "#001 observe ✓  1 image",
    "#002 move_pose ◇ plan  plan: 0.42 m, 3.1 s",
    "#003 move_pose ✓  action 1",
    "#004 close_gripper ✗  operator_stop_pending",
    "Stopped operator_instruction_stop: the operator stopped the command",
    "… 1 line above",
    ...["b", "c", "d", "waiting"],
  ]);
  assert.equal(lines.filter((line) => line.kind === "output").length, OUTPUT_TAIL);
  assert.deepEqual(lines.map((line) => Boolean(line.failed)), [true, false, false, false, true, true, false, false, false, false, false]);
  assert.equal(lines[2].dryRun, true);
  assert.equal(scriptResultLines({ ...result, calls: Array.from({ length: 12 }, (_, index) => ({ call: index + 1, tool: "observe", ok: true })) })
    .filter((line) => line.kind === "call").length, 7);
  assert.equal(scriptResultLines(result, true).filter((line) => line.kind === "output").length, 5);
  assert.equal(scriptResultLines({ ok: true, status: "completed", exit_code: 0, calls: [], output: "", wall_s: 0.1 })[0].text,
    "completed · 0 calls · 0.1 s");
});
