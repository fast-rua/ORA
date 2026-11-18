// The run's web server (server/) against a stand-in for ora-rpc-attach: the
// page's routes map to Pi RPC commands as the protocol has them, a dialog is
// answered with Pi's extension_ui_response, refusals keep their reasons, and
// there is no file, directory, working-directory or shell route.

import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { afterEach, describe, expect, test } from "vitest";

import { createApp, CSP } from "../server/app.ts";
import { JsonLines, RunRpc } from "../server/rpc.ts";
import { applyAssistantEvent, RunView } from "../server/view.ts";

const FAKE = resolve(import.meta.dirname, "fixtures", "fake-attach.mjs");
const open: Array<{ server: Server; rpc: RunRpc }> = [];

async function start(mode = "episode") {
  const dir = mkdtempSync(join(tmpdir(), "ora-web-test-"));
  const staticDir = join(dir, "dist");
  mkdirSync(staticDir);
  writeFileSync(join(staticDir, "index.html"), "<!doctype html><title>ORA</title>");
  writeFileSync(join(dir, "secret.txt"), "not for the page");
  const log = join(dir, "lines.jsonl");
  process.env.FAKE_ATTACH_LOG = log;
  process.env.FAKE_ATTACH_MODE = mode;
  const rpc = new RunRpc({ node: process.execPath, attach: FAKE, run: join(dir, "run") });
  const view = new RunView(rpc);
  rpc.start();
  await view.refresh();
  const server = createApp({ view, staticDir });
  await new Promise<void>((done) => server.listen(0, "127.0.0.1", () => done()));
  open.push({ server, rpc });
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const lines = () => readFileSync(log, "utf8").trim().split("\n").map((line) => JSON.parse(line));
  const post = async (path: string, body: unknown) => {
    const response = await fetch(base + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return { status: response.status, body: await response.json() };
  };
  return { base, view, rpc, lines, post };
}

afterEach(async () => {
  for (const { server, rpc } of open.splice(0)) {
    rpc.stop();
    await new Promise((done) => server.close(done));
  }
});

describe("the run's web server", () => {
  test("a page starts from a snapshot of the run: its mode, refusals and conversation", async () => {
    const { base } = await start();
    const snapshot = await (await fetch(`${base}/api/snapshot`)).json();
    expect(snapshot.run.mode).toBe("episode");
    expect(snapshot.run.refused.compact).toMatch(/context/);
    expect(snapshot.messages[0].customType).toBe("ora.observation");
    expect(snapshot.closed).toBeNull();
  });

  test("a message is Pi's prompt when idle, a steer while the agent works, a follow-up when asked", async () => {
    const { post, lines, rpc } = await start();
    expect((await post("/api/send", { text: "keep the gripper level" })).status).toBe(200);
    await rpc.request("start_streaming");
    await new Promise((done) => setTimeout(done, 50));
    expect((await post("/api/send", { text: "slower" })).body.as).toBe("steer");
    expect((await post("/api/send", { text: "then stop", as: "follow_up" })).body.as).toBe("follow_up");
    const sent = lines().filter((line) => ["prompt", "steer", "follow_up"].includes(line.type));
    expect(sent.map((line) => [line.type, line.message])).toEqual([
      ["prompt", "keep the gripper level"], ["steer", "slower"], ["follow_up", "then stop"]]);
  });

  test("a dialog is answered with Pi's extension_ui_response {type, id, confirmed}", async () => {
    const { post, lines, view } = await start();
    const stop = post("/api/send", { text: "/stop", as: "prompt" });
    for (let tries = 0; tries < 50 && !view.dialogs.size; tries++) await new Promise((done) => setTimeout(done, 20));
    expect([...view.dialogs.keys()]).toEqual(["dialog-1"]);
    expect((await post("/api/dialog", { id: "dialog-1", confirmed: true })).status).toBe(200);
    expect((await stop).status).toBe(200);
    const answer = lines().find((line) => line.type === "extension_ui_response");
    expect(answer).toEqual({ type: "extension_ui_response", id: "dialog-1", confirmed: true });
    expect((await post("/api/dialog", { id: "dialog-1", confirmed: true })).status).toBe(409);
    expect((await post("/api/dialog", { id: "dialog-2" })).status).toBe(400);
  });

  test("the agent process's refusals come back with their reasons", async () => {
    const { post } = await start();
    const compact = await post("/api/compact", {});
    expect(compact.status).toBe(409);
    expect(compact.body.error).toMatch(/compact is not available here/);
    const control = await post("/api/play", { control: "stop_instruction" });
    expect(control.status).toBe(409);
    expect((await post("/api/play", { control: "rm -rf" })).status).toBe(400);
  });

  test("free play's controls reach the run", async () => {
    const { post, lines } = await start("play");
    expect((await post("/api/play", { control: "reset" })).body.record).toEqual({ id: "op-1", type: "reset" });
    expect(lines().filter((line) => line.type === "ora_play_control").map((line) => line.control)).toEqual(["reset"]);
    // Queue: the next instruction, held by the Supervisor until the running one ends.
    expect((await post("/api/send", { text: "then stack them", as: "queue" })).body.data.record.type).toBe("queued_instruction");
    expect(lines().filter((line) => line.type === "ora_play_queue").map((line) => line.message)).toEqual(["then stack them"]);
  });

  test("a real cell's command the gateway ran reaches the run as ora_cell_note, and every page hears its outcome", async () => {
    const { post, lines, view } = await start("play");
    const heard: any[] = [];
    view.subscribe((event) => heard.push(event));
    const note = { id: "c-1", name: "home", verb: "home", ok: true, text: "at rest", seconds: 3.2,
      command: "scripts/ora cell home --json", exit_code: 0, output: "{}" };
    const answer = await post("/api/cell-note", note);
    expect(answer.status).toBe(200);
    expect(lines().filter((line) => line.type === "ora_cell_note").map((line) => line.note)).toEqual([note]);
    for (let tries = 0; tries < 50 && !heard.some((event) => event.type === "cell_command"); tries++) {
      await new Promise((done) => setTimeout(done, 20));
    }
    expect(heard.find((event) => event.type === "cell_command").entry).toMatchObject({ id: "c-1", name: "home", via: "web page" });
    expect(view.run.cellCommands.map((item: any) => item.id)).toEqual(["c-1"]);
    expect(view.snapshot().run.cellCommands).toHaveLength(1);
  });

  test("no shell, file, directory, working-directory or update route; the page has a strict CSP", async () => {
    const { base, post } = await start();
    expect((await post("/api/send", { text: "!cat /etc/passwd" })).status).toBe(403);
    for (const path of ["/api/files", "/api/file-content?path=x", "/api/file-raw?path=x", "/api/directories", "/api/update-check"]) {
      expect((await fetch(base + path)).status, path).toBe(404);
    }
    expect((await post("/api/cwd", { cwd: "/" })).status).toBe(404);
    const page = await fetch(`${base}/`);
    expect(page.status).toBe(200);
    expect(page.headers.get("content-security-policy")).toBe(CSP);
    expect(CSP).toContain("script-src 'self'");
    for (const path of ["/..%2fsecret.txt", "/%2e%2e/secret.txt", "/.hidden"]) {
      expect((await fetch(base + path)).status, path).toBe(404);
    }
  });

  test("server-sent events: a snapshot, then the reply as it is written", async () => {
    const { base, rpc } = await start();
    const controller = new AbortController();
    const response = await fetch(`${base}/api/events`, { signal: controller.signal });
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    const events: any[] = [];
    let text = "";
    const until = async (predicate: () => boolean) => {
      while (!predicate()) {
        const { value } = await reader.read();
        text += decoder.decode(value);
        const parts = text.split("\n\n");
        text = parts.pop() ?? "";
        for (const part of parts) if (part.startsWith("data: ")) events.push(JSON.parse(part.slice(6)));
      }
    };
    await until(() => events.length > 0);
    expect(events[0].type).toBe("snapshot");
    await rpc.request("start_streaming");
    await until(() => events.some((event) => event.type === "streaming" && event.message?.content?.[0]?.text === "hello"));
    controller.abort();
  });
});

describe("assembling a reply from Pi 1.3's deltas", () => {
  test("text, thinking and tool calls", () => {
    let message = applyAssistantEvent(null, { type: "thinking_start", contentIndex: 0 });
    message = applyAssistantEvent(message, { type: "thinking_delta", contentIndex: 0, delta: "look " });
    message = applyAssistantEvent(message, { type: "thinking_delta", contentIndex: 0, delta: "first" });
    message = applyAssistantEvent(message, { type: "text_start", contentIndex: 1 });
    message = applyAssistantEvent(message, { type: "text_delta", contentIndex: 1, delta: "Grounding" });
    message = applyAssistantEvent(message, { type: "toolcall_start", contentIndex: 2, id: "call-1", toolName: "vision_ground" });
    message = applyAssistantEvent(message, { type: "toolcall_delta", contentIndex: 2, delta: "{\"camera\":" });
    expect(message.content[0]).toEqual({ type: "thinking", thinking: "look first" });
    expect(message.content[1]).toEqual({ type: "text", text: "Grounding" });
    expect(message.content[2]).toMatchObject({ type: "toolCall", id: "call-1", name: "vision_ground", partialArguments: "{\"camera\":" });
    const call = { type: "toolCall", id: "call-1", name: "vision_ground", arguments: { camera: "agentview", text: "basket" } };
    message = applyAssistantEvent(message, { type: "toolcall_end", contentIndex: 2, toolCall: call });
    expect(message.content[2]).toEqual(call);
    const done = { role: "assistant", content: [{ type: "text", text: "final" }], stopReason: "toolUse" };
    expect(applyAssistantEvent(message, { type: "done", reason: "toolUse", message: done })).toBe(done);
  });

  test("LF-delimited JSON lines, a CR tolerated, split anywhere", () => {
    const lines = new JsonLines();
    expect(lines.push('{"a":1}\r\n{"b"')).toEqual([{ a: 1 }]);
    expect(lines.push(':2}\n')).toEqual([{ b: 2 }]);
  });
});
