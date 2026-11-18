// Pi's RPC protocol served by ORA (src/rpc/): Pi's own RpcClient, started on
// `ora-rpc-attach` (src/rpc/attach.ts) as its `pi`, against an RpcHost on a
// Pi session with a scripted provider. Covers get_state, prompt, abort, an
// extension dialog (controller only) and its notification (everyone), the
// episode's refusals and observer rules, the image completion, ORA's own
// commands (the run and its mode, free play's controls, state and messages,
// the recorded motion and the operator view), and the attach stand-in's
// refusal of a run that is not live.

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { after, before, test } from "node:test";

import { createAssistantMessageEventStream, InMemoryCredentialStore } from "@earendil-works/pi-ai";
import {
  createAgentSessionFromServices, createAgentSessionRuntime, createAgentSessionServices, ModelRuntime,
  RpcClient, SessionManager, SettingsManager, type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

import { sessionAlias } from "../src/rpc/attach.ts";
import { RunFrames } from "../src/rpc/frames.ts";
import { RpcHost } from "../src/rpc/host.ts";
import { ImageCompletion } from "../src/rpc/images.ts";
import { refusal } from "../src/rpc/policy.ts";
import { RpcUi } from "../src/rpc/ui.ts";

type Json = any;

const ATTACH = resolve(import.meta.dirname, "../src/rpc/attach.ts");
const scratch = (label: string) => mkdtempSync(join(process.env.ORA_PI_SCRATCH ?? tmpdir(), `ora-rpc-${label}-`));

/** A reply after `delay` ms (aborted when the signal fires), and a /ask command that asks the operator. */
function scripted(delay: () => number) {
  return (pi: ExtensionAPI) => {
    pi.registerProvider("ora", {
      name: "ORA (test)", baseUrl: "http://ora.invalid", apiKey: "none", api: "ora-native",
      models: [{ id: "scripted", name: "scripted", reasoning: false, input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 512 }],
      streamSimple: (model: Json, _context: Json, options?: Json) => {
        const stream = createAssistantMessageEventStream();
        const base = { role: "assistant", api: model.api, provider: model.provider, model: model.id,
          usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, timestamp: Date.now() };
        const timer = setTimeout(() => {
          const message = { ...base, content: [{ type: "text", text: "pong" }], stopReason: "stop" };
          stream.push({ type: "start", partial: message } as Json);
          stream.push({ type: "done", reason: "stop", message } as Json);
          stream.end(message as Json);
        }, delay());
        options?.signal?.addEventListener("abort", () => {
          clearTimeout(timer);
          const message = { ...base, content: [], stopReason: "aborted", errorMessage: "aborted" };
          stream.push({ type: "start", partial: message } as Json);
          stream.push({ type: "error", reason: "aborted", error: message } as Json);
          stream.end(message as Json);
        }, { once: true });
        return stream;
      },
    } as Json);
    pi.registerCommand("ask", {
      description: "Ask the operator",
      handler: async (_args: string, ctx: Json) => {
        const yes = await ctx.ui.confirm("Proceed?", "The test asks the controlling client.");
        ctx.ui.notify(yes ? "answered yes" : "answered no", "info");
      },
    });
  };
}

async function runtimeFor(root: string, delay: () => number): Promise<Json> {
  const agentDir = join(root, "native", "pi-agent");
  mkdirSync(agentDir, { recursive: true });
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } } as Json,
    { projectTrusted: false } as Json);
  const runtime = await createAgentSessionRuntime((async ({ sessionManager, sessionStartEvent }: Json) => {
    const modelRuntime = await ModelRuntime.create({
      credentials: new InMemoryCredentialStore(), authPath: join(agentDir, "unused-auth.json"),
      modelsPath: null, modelsStorePath: join(agentDir, "models-store.json"), refreshOnCreate: false,
    } as Json);
    const services = await createAgentSessionServices({
      cwd: agentDir, agentDir, settingsManager, modelRuntime,
      resourceLoaderOptions: { extensionFactories: [scripted(delay)], noSkills: true, noPromptTemplates: true,
        noThemes: true, noContextFiles: true, systemPrompt: "" },
    } as Json);
    const result = await createAgentSessionFromServices({ services, sessionManager, sessionStartEvent, noTools: "builtin" } as Json);
    return { ...result, services, diagnostics: services.diagnostics };
  }) as Json, { cwd: agentDir, agentDir, sessionManager: SessionManager.inMemory(agentDir) });
  const model = runtime.session.modelRuntime.getModel("ora", "scripted");
  await runtime.session.setModel(model as Json);
  return runtime;
}

let root: string;
let host: RpcHost;
let delay = 50;
const clients: RpcClient[] = [];

async function client(run = root, role = "controller"): Promise<RpcClient> {
  const rpc = new RpcClient({ cliPath: ATTACH, args: ["--ora-run", run, "--ora-role", role] });
  await rpc.start();
  clients.push(rpc);
  return rpc;
}

function respond(rpc: RpcClient, message: Json): void {
  (rpc as Json).process.stdin.write(`${JSON.stringify(message)}\n`);
}

function nextEvent(rpc: RpcClient, predicate: (event: Json) => boolean, timeout = 10_000): Promise<Json> {
  return new Promise((resolveEvent, reject) => {
    const timer = setTimeout(() => reject(new Error("timed out waiting for an event")), timeout);
    const stop = rpc.onEvent((event) => {
      if (!predicate(event)) return;
      clearTimeout(timer);
      stop();
      resolveEvent(event);
    });
  });
}

before(async () => {
  root = scratch("run");
  const runtime = await runtimeFor(root, () => delay);
  host = await RpcHost.start({ runtime, runRoot: root, runId: "run-rpc", mode: () => "episode", images: null,
    bindUi: true, socketDir: scratch("socket") });
});

after(async () => {
  for (const rpc of clients) await rpc.stop().catch(() => undefined);
  await host.end();
});

test("Pi's RpcClient reads the state and prompts through ora-rpc-attach", async () => {
  const rpc = await client();
  const state = await rpc.getState();
  assert.equal(state.model?.id, "scripted");
  assert.equal(state.isStreaming, false);
  const events = await rpc.promptAndWait("ping");
  assert.ok(events.some((event: Json) => event.type === "agent_start"));
  assert.equal(await rpc.getLastAssistantText(), "pong");
  const messages = await rpc.getMessages();
  assert.deepEqual(messages.map((message: Json) => message.role).slice(-2), ["user", "assistant"]);
  await rpc.stop();
});

test("abort ends a reply in flight", async () => {
  delay = 5_000;
  const rpc = await client();
  const settled = rpc.waitForIdle(10_000);
  const started = nextEvent(rpc, (event) => event.type === "message_start" && event.message?.role === "user");
  await rpc.prompt("wait for me");
  await started;
  await rpc.abort();
  await settled;
  const last = (await rpc.getMessages()).at(-1);
  assert.equal(last.stopReason, "aborted");
  delay = 50;
  await rpc.stop();
});

test("a dialog goes to the controller only; its notification to everyone; observers only read", async () => {
  const controller = await client();
  const observer = await client(root, "observer");
  assert.equal((await (controller as Json).send({ type: "ora_get_role" })).data.role, "controller");
  assert.equal((await (observer as Json).send({ type: "ora_get_role" })).data.role, "observer");
  const seenByObserver: Json[] = [];
  observer.onEvent((event) => {
    if (event.type === "extension_ui_request") seenByObserver.push(event);
  });
  const dialog = nextEvent(controller, (event) => event.type === "extension_ui_request" && event.method === "confirm");
  const notified = nextEvent(observer, (event) => event.type === "extension_ui_request" && event.method === "notify");
  // An extension command's prompt is answered once the command is done (as in Pi), so after the dialog.
  const prompted = controller.prompt("/ask");
  const request = await dialog;
  assert.equal(request.title, "Proceed?");
  respond(observer, { type: "extension_ui_response", id: request.id, confirmed: false }); // not the observer's to answer
  respond(controller, { type: "extension_ui_response", id: request.id, confirmed: true });
  assert.equal((await notified).message, "answered yes");
  await prompted;
  assert.ok(!seenByObserver.some((event) => event.method === "confirm"), "the observer never sees the dialog");

  await assert.rejects(observer.prompt("from the observer"), /observes the run/);
  const abort = await (observer as Json).send({ type: "abort" }); // RpcClient.abort() does not look at success
  assert.equal(abort.success, false);
  assert.match(abort.error, /observes the run/);
  assert.equal((await observer.getState()).model?.id, "scripted", "observers read");
  await controller.stop();
  // The controller left: the oldest observer controls now.
  await new Promise((done) => setTimeout(done, 200));
  assert.equal((await (observer as Json).send({ type: "ora_get_role" })).data.role, "controller");
  await observer.stop();
});

test("an episode refuses model, reasoning, compaction, shell, session and export changes", async () => {
  const rpc = await client();
  // The host answers success: false (some RpcClient helpers do not look, so the raw response is checked).
  for (const [command, pattern] of [
    [{ type: "set_model", provider: "ora", modelId: "scripted" }, /set_model is not available here: the model is bound by the RunPlan/],
    [{ type: "cycle_model" }, /the model is bound by the RunPlan/],
    [{ type: "set_thinking_level", level: "high" }, /reasoning effort is bound by the RunPlan/],
    [{ type: "compact" }, /ORA manages the model's context/],
    [{ type: "set_auto_compaction", enabled: true }, /ORA manages the model's context/],
    [{ type: "bash", command: "ls" }, /shell commands are not available/],
    [{ type: "new_session" }, /history cannot be replaced/],
    [{ type: "switch_session", sessionPath: "/tmp/other.jsonl" }, /history cannot be replaced/],
    [{ type: "fork", entryId: "x" }, /cannot be rewound/],
    [{ type: "clone" }, /cannot be copied/],
    [{ type: "set_session_name", name: "renamed" }, /session is evidence/],
    [{ type: "export_html", outputPath: "/tmp/x.html" }, /ora sessions export/],
  ] as Array<[Json, RegExp]>) {
    const response = await (rpc as Json).send(command);
    assert.equal(response.success, false, command.type);
    assert.match(response.error, pattern);
  }
  assert.deepEqual((await rpc.getAvailableModels()).map((model: Json) => model.id), ["scripted"]);
  assert.equal((await rpc.getState()).thinkingLevel, (await rpc.getAvailableThinkingLevels())[0]);
  await rpc.stop();
  assert.equal(refusal("bash", "play", true), null, "free play opens Pi's commands");
  assert.match(refusal("export_html", "play", true) ?? "", /ora sessions export/);
});

test("ora_get_run names the run, its mode and what it refuses; free play takes its controls and hears its state", async () => {
  const rpc = await client();
  const run = await (rpc as Json).send({ type: "ora_get_run" });
  assert.equal(run.success, true);
  assert.equal(run.data.runId, "run-rpc");
  assert.equal(run.data.mode, "episode");
  assert.equal(run.data.role, "controller");
  assert.equal(run.data.play, null);
  assert.match(run.data.refused.compact, /ORA manages the model's context/);
  assert.match(run.data.refused.set_model, /bound by the RunPlan/);
  const control = await (rpc as Json).send({ type: "ora_play_control", control: "stop_instruction" });
  assert.equal(control.success, false);
  assert.match(control.error, /belong to `ora play`/);
  const queue = await (rpc as Json).send({ type: "ora_play_queue", message: "then stack them" });
  assert.match(queue.error, /a queued instruction belongs to `ora play`/);
  await rpc.stop();

  const playRoot = scratch("play");
  const sent: Array<[string, string]> = [];
  let state: Json = { schema: "ora.play-state.v1", state: "awaiting_instruction", scene: 1, scenes: 2, instructions: 0, running: null };
  writeFrames(playRoot);
  const messages: Json[] = [{ id: "m1", text: "how far along?", state: "queued", joined: true, instruction: 1, step: null,
    at: 1, deliveredAt: null }];
  const play = await RpcHost.start({ runtime: await runtimeFor(playRoot, () => delay), runRoot: playRoot, runId: "run-play",
    mode: () => "play", images: null, bindUi: true, socketDir: scratch("play-socket"), play: () => state,
    control: async (type, message) => {
      sent.push([type, message]);
      return { id: "op-1", type, received_monotonic_ns: 2n ** 60n };  // Python's big integers arrive as bigint
    }, playMessages: () => messages,
    queue: async (message) => { sent.push(["queued_instruction", message]); return { id: "op-q", type: "queued_instruction" }; } });
  try {
    const player = await client(playRoot);
    const info = (await (player as Json).send({ type: "ora_get_run" })).data;
    assert.equal(info.mode, "play");
    assert.deepEqual(Object.keys(info.refused), ["export_html"], "free play opens Pi's commands");
    assert.equal(info.play.state, "awaiting_instruction");
    const stopped = await (player as Json).send({ type: "ora_play_control", control: "stop_instruction" });
    assert.equal(stopped.success, true);
    assert.deepEqual(stopped.data.record, { id: "op-1", type: "stop_instruction", received_monotonic_ns: 2 ** 60 });
    assert.deepEqual(sent, [["stop_instruction", "Esc"]], "recorded as the terminal's own control");
    const queued = await (player as Json).send({ type: "ora_play_queue", message: " then stack them " });
    assert.deepEqual([queued.success, queued.data.record.id], [true, "op-q"]);
    assert.deepEqual(sent.at(-1), ["queued_instruction", "then stack them"], "held by the Supervisor, as Alt+Enter");
    assert.match((await (player as Json).send({ type: "ora_play_queue", message: " " })).error, /message is required/);
    const wrong = await (player as Json).send({ type: "ora_play_control", control: "rm -rf" });
    assert.match(wrong.error, /control must be one of stop_instruction, reset, quit/);
    const heard = nextEvent(player, (event) => event.type === "ora_play_state");
    state = { ...state, state: "instruction", instructions: 1, running: "op-2" };
    play.announcePlay(state);
    assert.equal((await heard).state.running, "op-2");
    assert.deepEqual(info.playMessages, messages, "the operator's messages, queued or delivered");
    const message = nextEvent(player, (event) => event.type === "ora_play_message");
    play.announceMessage({ ...messages[0], state: "delivered", step: 4 });
    assert.equal((await message).message.step, 4);
    const observer = await client(playRoot, "observer");
    const frames = await (observer as Json).send({ type: "ora_get_frames", step: 3 });
    assert.equal(frames.success, true, "an observer reads the frames too");
    assert.deepEqual(frames.data.motion.steps, { 3: 3, 4: 2 });
    assert.deepEqual(frames.data.step.motion.frames.map((frame: Json) => frame.images.head_camera),
      ["/ora/capture/00000001-camera-0.png", "/ora/capture/00000002-camera-0.png", "/ora/capture/00000003-camera-0.png"]);
    assert.equal(frames.data.step.operatorView.url, "/ora/operator-view/000003.png");
    assert.equal((await (observer as Json).send({ type: "ora_get_frames", step: -1 })).success, false);
    await observer.stop();
    await player.stop();
  } finally {
    await play.end();
  }
});

test("a real cell's command the page ran reaches the run's cell commands; every client hears each outcome", async () => {
  const cellRoot = scratch("cell");
  const noted: Json[] = [];
  const outcomes: Json[] = [{ id: "t-1", name: "home", verb: "home", ok: true, text: "at rest", seconds: 2, via: "play terminal", at: 1 }];
  const cell = await RpcHost.start({ runtime: await runtimeFor(cellRoot, () => delay), runRoot: cellRoot, runId: "run-cell",
    mode: () => "play", images: null, bindUi: true, socketDir: scratch("cell-socket"),
    cellNote: (note) => {
      if (note?.name !== "estop") throw new Error("ora_cell_note: /x is not a cell command");
      noted.push(note);
      return { ...note, via: "web page" };
    }, cellCommands: () => outcomes });
  try {
    const page = await client(cellRoot);
    assert.deepEqual((await (page as Json).send({ type: "ora_get_run" })).data.cellCommands, outcomes);
    const answer = await (page as Json).send({ type: "ora_cell_note", note: { name: "estop", verb: "estop", ok: true } });
    assert.equal(answer.success, true);
    assert.equal(answer.data.entry.via, "web page");
    assert.deepEqual(noted, [{ name: "estop", verb: "estop", ok: true }]);
    const refused = await (page as Json).send({ type: "ora_cell_note", note: { name: "x" } });
    assert.match(refused.error, /not a cell command/);
    const observer = await client(cellRoot, "observer");
    const watched = await (observer as Json).send({ type: "ora_cell_note", note: { name: "estop" } });
    assert.match(watched.error, /observes the run/, "only the controlling client tells the run");
    const heard = nextEvent(observer, (event) => event.type === "ora_cell_command");
    cell.announceCell(outcomes[0]);
    assert.deepEqual((await heard).entry, outcomes[0]);
    await observer.stop();
    await page.stop();
  } finally {
    await cell.end();
  }
  const rpc = await client();
  const none = await (rpc as Json).send({ type: "ora_cell_note", note: {} });
  assert.match(none.error, /no real-cell commands/);
  await rpc.stop();
});

test("in an episode a follow-up is delivered as a steer (one kind of operator message)", async () => {
  const session = (host as Json).options.runtime.session;
  const calls: string[] = [];
  const steer = session.steer, followUp = session.followUp;
  session.steer = async () => { calls.push("steer"); return "queued"; };
  session.followUp = async () => { calls.push("followUp"); return "queued"; };
  try {
    const rpc = await client();
    assert.equal(await rpc.followUp("after this, keep the gripper level"), "queued");
    assert.deepEqual(calls, ["steer"]);
    await rpc.stop();
  } finally {
    session.steer = steer;
    session.followUp = followUp;
  }
});

test("ora-rpc-attach refuses a run that is not live, and never starts Pi", async () => {
  const finished = scratch("finished");
  mkdirSync(join(finished, "native"), { recursive: true });
  const refusedWith = async (pattern: RegExp) => {
    const rpc = new RpcClient({ cliPath: ATTACH, args: ["--ora-run", finished] });
    await rpc.start().catch(() => undefined); // the stand-in may exit before or after start() looks
    await assert.rejects(rpc.getState(), pattern);
  };
  await refusedWith(/serves no Pi RPC/);
  writeFileSync(join(finished, "native", "pi-rpc.json"), JSON.stringify({ socket: join(finished, "gone.sock"), pid: 2 ** 22 + 1 }));
  await refusedWith(/is not live/);
});

test("observation and tool-result messages carry ORA's images; get_messages inlines the latest steps", () => {
  const run = scratch("images");
  mkdirSync(join(run, "native", "agent-artifacts"), { recursive: true });
  const png = (n: number) => Buffer.from(`\x89PNG fake ${n}`);
  const stored = (n: number) => {
    const digest = createHash("sha256").update(png(n)).digest("hex");
    writeFileSync(join(run, "native", "agent-artifacts", `${digest}.png`), png(n));
    return `native/agent-artifacts/${digest}.png`;
  };
  const dataUrl = (n: number) => `data:image/png;base64,${png(n).toString("base64")}`;
  const history = [0, 1, 2].map((step) => ({
    role: "user", _observation: { step },
    content: [{ type: "text", text: "camera head_camera" }, { type: "image_url", image_url: { url: dataUrl(step) } }],
  }));
  history.push({ role: "user", _tool_image_call_id: "call-2", content: [{ type: "text", text: "mark_point" },
    { type: "image_url", image_url: { url: dataUrl(9) } }] } as Json);
  const artifacts = [0, 1, 2, 9].map(stored);
  const images = new ImageCompletion({ runRoot: run, history: () => history, pendingToolImages: () => [] }, 1);
  const observation = (step: number) => ({ role: "custom", customType: "ora.observation", content: `observation ${step}`,
    display: true, details: { ora_observation_step: step } });
  const session = [observation(0), observation(1), observation(2),
    { role: "toolResult", toolCallId: "call-2", toolName: "perception.mark_point", content: [{ type: "text", text: "{}" }] }];
  const completed = images.completeAll(session);
  assert.deepEqual(completed.map((message: Json) => message.details.ora.step), [0, 1, 2, 2]);
  assert.deepEqual(completed.map((message: Json) => message.content.filter((block: Json) => block.type === "image").length),
    [0, 0, 1, 1], "only the latest step's images are inline");
  assert.equal(completed[0].details.ora.artifact, artifacts[0]);
  assert.equal(completed[0].details.ora.images[0].url, `/ora/artifacts/${artifacts[0].split("/").pop()}`);
  assert.equal(completed[3].details.ora.toolCallId, "call-2");
  assert.equal(completed[3].details.ora.artifact, artifacts[3]);
  assert.equal(session[0].content, "observation 0", "the session's own messages are not changed");
  // The live stream: an observation's step is the step of the tool calls after it.
  images.event({ type: "message_end", message: observation(5) });
  images.event({ type: "tool_execution_start", toolCallId: "call-7" });
  const end = images.event({ type: "tool_execution_end", toolCallId: "call-7", result: { content: [{ type: "text", text: "ok" }] } });
  assert.equal(end.result.details.ora.step, 5);
});

test("free play's observation and tool images, in the messages themselves, are listed the same way", () => {
  const run = scratch("play-images");
  mkdirSync(join(run, "native", "agent-artifacts"), { recursive: true });
  const png = (n: number) => Buffer.from(`\x89PNG play ${n}`);
  const stored = (n: number) => {
    const digest = createHash("sha256").update(png(n)).digest("hex");
    writeFileSync(join(run, "native", "agent-artifacts", `${digest}.png`), png(n));
    return digest;
  };
  const image = (n: number) => ({ type: "image", mimeType: "image/png", data: png(n).toString("base64") });
  const observation = (step: number) => ({ role: "custom", customType: "ora.play_observation", display: true,
    details: { ora_observation_step: step }, content: [{ type: "text", text: "{\"current_actual\": {}}" },
      { type: "text", text: `ACTUAL · sequence ${step} · agentview · latest at this message` }, image(step * 10),
      { type: "text", text: `ACTUAL · sequence ${step} · wrist · latest at this message` }, image(step * 10 + 1)] });
  const digests = [0, 1, 10, 11, 99].map(stored);
  const images = new ImageCompletion({ runRoot: run, history: () => [], pendingToolImages: () => [] }, 1);
  const session = [observation(0), observation(1), { role: "toolResult", toolCallId: "call-g", toolName: "vision_ground",
    content: [{ type: "text", text: "{\"camera\": \"agentview\"}" }, image(99)] }];
  const completed = images.completeAll(session);
  assert.deepEqual(completed.map((message: Json) => message.details.ora.step), [0, 1, 1]);
  assert.deepEqual(completed[0].details.ora.images.map((item: Json) => [item.camera, item.inline, item.url]),
    [["agentview", false, `/ora/artifacts/${digests[0]}.png`], ["wrist", false, `/ora/artifacts/${digests[1]}.png`]]);
  assert.deepEqual(completed[0].content.map((block: Json) => block.type), ["text", "text", "text"],
    "an earlier step's images are listed by URL, not inline");
  assert.equal(completed[1].content.filter((block: Json) => block.type === "image").length, 2);
  assert.equal(completed[2].details.ora.images[0].url, `/ora/artifacts/${digests[4]}.png`);
  assert.equal(session[0].content.length, 5, "the session's own messages are not changed");
});

test("a client that opened the session by a link keys it by the link", () => {
  const state = JSON.stringify({ id: "req_1", type: "response", command: "get_state", success: true,
    data: { sessionFile: "/runs/r/native/pi-session/s.jsonl", sessionId: "s" } });
  assert.equal(JSON.parse(sessionAlias(state, "/tmp/view/s.jsonl")).data.sessionFile, "/tmp/view/s.jsonl");
  const event = JSON.stringify({ type: "message_end", message: { role: "assistant", content: [] } });
  assert.equal(sessionAlias(event, "/tmp/view/s.jsonl"), event, "everything else passes as it came");
});

test("with the TUI: notifications reach both, a dialog goes to the client that asked, else to the terminal", async () => {
  const sent: Array<[number, Json]> = [];
  const clients = [1, 2].map((id) => ({ id, send: (message: Json) => sent.push([id, message]) }));
  const ui = new RpcUi(() => clients, () => clients[0]);
  const terminal: Json[] = [];
  const base = {
    notify: (message: string) => terminal.push(["notify", message]),
    confirm: async (title: string) => (terminal.push(["confirm", title]), true),
  };
  const composite = ui.context(base);
  composite.notify("steer queued", "info");
  assert.deepEqual(terminal, [["notify", "steer queued"]]);
  assert.deepEqual(sent.map(([id, message]) => [id, message.method]), [[1, "notify"], [2, "notify"]]);
  // Free play's state, said by the terminal for itself: clients hear the state (ora_play_state) instead.
  composite.notifyLocal("Instruction 2 is starting.", "info");
  assert.deepEqual(terminal.at(-1), ["notify", "Instruction 2 is starting."]);
  assert.equal(sent.length, 2, "no client hears the terminal's own notice");

  // Typed in the terminal (no RPC prompt running): the terminal answers.
  assert.equal(await composite.confirm("Stop this run?", "from the terminal"), true);
  assert.deepEqual(terminal.at(-1), ["confirm", "Stop this run?"]);

  // Raised by client 2's prompt: client 2 is asked, nobody else.
  sent.length = 0;
  ui.driver = 2;
  const answer = composite.confirm("Stop this run?", "from the page");
  const [[to, request]] = sent;
  assert.equal(to, 2);
  assert.equal(request.method, "confirm");
  assert.equal(ui.respond(1, { id: request.id, confirmed: true }), false, "client 1 did not ask");
  assert.equal(ui.respond(2, { id: request.id, confirmed: true }), true);
  assert.equal(await answer, true);
  assert.equal(terminal.filter(([kind]) => kind === "confirm").length, 1);

  // No terminal: the controller is asked.
  sent.length = 0;
  ui.driver = null;
  const alone = ui.context(null).confirm("Proceed?", "");
  assert.equal(sent[0][0], 1);
  ui.detach(1);
  assert.equal(await alone, false, "a client that leaves cancels its dialog");
});

/** A run's recorded motion (two actions, one with a stage) and two operator views, as the Supervisor writes them. */
function writeFrames(run: string): void {
  const capture = join(run, "native", "execution-capture");
  mkdirSync(join(capture, "frames"), { recursive: true });
  writeFileSync(join(capture, "manifest.json"), JSON.stringify({ camera_panels: ["head_camera", "front_camera"],
    config: { fps: 10 } }));
  const frame = (index: number, action: string | null, phase: string) => JSON.stringify({
    frame_index: index, capture_index: index, action_id: action, phase, sim_time_s: index / 10,
    images: { head_camera: { path: `frames/${String(index).padStart(8, "0")}-camera-0.png` },
      front_camera: { path: `frames/${String(index).padStart(8, "0")}-camera-1.png` } } });
  writeFileSync(join(capture, "timeline.jsonl"), [frame(0, null, "reset"), frame(1, "a1", "action_start"),
    frame(2, "a1", "control"), frame(3, "a1", "action_end"), frame(4, "a2:00-approach", "control"),
    frame(5, "a2", "action_end")].join("\n") + "\n" + frame(6, "a3", "action_start").slice(0, 20));
  writeFileSync(join(run, "agent-events.jsonl"), [
    { type: "action_proposed", observation_step: 3, payload: { action_id: "a1" } },
    { type: "tool_result", observation_step: 3, payload: { name: "robot.submit_action" } },
    { type: "action_proposed", observation_step: 4, payload: { action_id: "a2" } },
  ].map((event) => JSON.stringify(event)).join("\n") + "\n");
  writeFileSync(join(run, "events.jsonl"), [3, 4].map((step) => JSON.stringify({ type: "operator_view_frame",
    observation_step: step, path: `native/operator-view/${String(step).padStart(6, "0")}.png`, camera: "observer_camera",
    width: 64, height: 48, sha256: "0".repeat(64) })).join("\n") + "\n");
}

test("the recorded motion of a step and the operator view, read as the run writes them", () => {
  const run = scratch("frames");
  const frames = new RunFrames(run);
  assert.deepEqual(frames.summary(), { motion: null, operatorView: null }, "a run without either has none");
  writeFrames(run);
  const summary = frames.summary();
  assert.deepEqual(summary.motion, { cameras: ["head_camera", "front_camera"], fps: 10, steps: { 3: 3, 4: 2 } });
  assert.deepEqual(summary.operatorView!.steps, [3, 4]);
  assert.equal(summary.operatorView!.latest.url, "/ora/operator-view/000004.png");
  const motion = frames.motion(4);
  assert.deepEqual(motion.actionIds, ["a2:00-approach", "a2"], "a stage belongs to its action's step");
  assert.deepEqual(motion.frames.map((item) => item.phase), ["control", "action_end"]);
  assert.deepEqual(frames.motion(9).frames, []);
  // Appended lines (the torn one completed) are read on the next call.
  writeFileSync(join(run, "native", "execution-capture", "timeline.jsonl"), "\n", { flag: "a" });
  writeFileSync(join(run, "agent-events.jsonl"), JSON.stringify({ type: "action_proposed", observation_step: 5,
    payload: { action_id: "a7" } }) + "\n", { flag: "a" });
  assert.equal(frames.summary().motion!.steps["5"], undefined, "a torn line is never read as a frame");
  assert.equal(frames.operatorView(3)!.camera, "observer_camera");
  assert.equal(frames.operatorView(2), null);
});
