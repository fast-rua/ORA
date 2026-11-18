// A finished run's Pi session in Pi's terminal UI, read-only (`ora attach` on a landed run).
//
//   node ora-pi/src/tui/view.ts RUN_ROOT
//
// The session file is loaded into memory and never written back; ORA's
// renderers draw the observations, tool calls and images from the run's
// recorded history (the same bytes the model received). Typed input is
// refused; /export writes here; Ctrl+D closes.

import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";

import { createAssistantMessageEventStream, InMemoryCredentialStore } from "@earendil-works/pi-ai";
import {
  createAgentSessionFromServices, createAgentSessionRuntime, createAgentSessionServices, InteractiveMode,
  ModelRuntime, parseSessionEntries, SessionManager, SettingsManager, type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

import { hydrate } from "../core/persistence.ts";
import { parsePy } from "../pyjson.ts";
import { exportSession } from "./export.ts";
import { OraTui } from "./extension.ts";
import { guardEditor } from "./host.ts";

type Json = any;

function readJson(path: string): Json {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
}

async function main(): Promise<void> {
  const runRoot = resolve(process.argv[2] ?? "");
  const sessions = join(runRoot, "native", "pi-session");
  const file = readdirSync(sessions).filter((name) => name.endsWith(".jsonl")).sort().pop();
  if (!file) throw new Error(`no Pi session under ${sessions}`);
  const sessionFile = join(sessions, file);
  const entries = parseSessionEntries(readFileSync(sessionFile, "utf8"));
  const header = entries.find((entry: Json) => entry.type === "session") as Json;
  const manager = SessionManager.inMemory(header?.cwd ?? runRoot, undefined, entries);

  const history: Json[] = [];
  const toolNames = new Set<string>(["ora_unadvertised_tool"]);
  let modelId = "ora";
  for (const entry of manager.getBranch() as Json[]) {
    if (entry.type === "custom" && entry.customType === "ora.history") {
      history.push(...parsePy(entry.data.pyjson).messages.map((message: Json) => hydrate(message, runRoot)));
    } else if (entry.type === "message" && entry.message?.role === "assistant") {
      for (const block of entry.message.content ?? []) if (block?.type === "toolCall") toolNames.add(String(block.name));
    } else if (entry.type === "model_change" && entry.provider === "ora") {
      modelId = String(entry.modelId);
    }
  }
  const policy = readJson(join(runRoot, "plan.json"))?.policy ?? {};
  const tui = new OraTui({
    runRoot, runId: basename(runRoot), model: String(policy.model ?? modelId),
    profile: typeof policy.provider_profile === "string" ? policy.provider_profile : null,
    protocol: String(policy.provider ?? "?"), mutationTools: [], toolNames: [...toolNames],
    history: () => history, pendingToolImages: () => [], live: () => false, inTurn: () => false,
    stop: () => undefined, interruptAtBoundary: () => undefined, notifySupervisor: () => undefined,
    play: () => null, playInstruction: () => null, control: async () => null,
    sessionFile: () => sessionFile,
  }, { live: false });

  // The recorded model resolves to a provider that never calls anything (input is refused).
  const recordedModel = (pi: ExtensionAPI) => {
    pi.registerProvider("ora", {
      name: "ORA (finished run)", baseUrl: "http://ora.invalid", apiKey: "none", api: "ora-native",
      models: [{
        id: modelId, name: modelId, reasoning: true, input: ["text", "image"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 1, maxTokens: 1,
      }],
      streamSimple: () => {
        const stream = createAssistantMessageEventStream();
        stream.end();
        return stream;
      },
    } as Json);
  };

  const agentDir = mkdtempSync(join(tmpdir(), "ora-pi-view-"));
  const settingsManager = SettingsManager.inMemory(
    { compaction: { enabled: false }, retry: { enabled: false }, quietStartup: true } as Json,
    { projectTrusted: false } as Json,
  );
  const cwd = manager.getCwd();
  const runtime = await createAgentSessionRuntime((async ({ sessionManager, sessionStartEvent }: Json) => {
    const modelRuntime = await ModelRuntime.create({
      credentials: new InMemoryCredentialStore(), authPath: join(agentDir, "unused-auth.json"),
      modelsPath: null, modelsStorePath: join(agentDir, "models-store.json"), refreshOnCreate: false,
    } as Json);
    const services = await createAgentSessionServices({
      cwd, agentDir, settingsManager, modelRuntime,
      resourceLoaderOptions: {
        extensionFactories: [recordedModel, tui.extension()],
        noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true, systemPrompt: "",
      },
    } as Json);
    const result = await createAgentSessionFromServices({ services, sessionManager, sessionStartEvent, noTools: "builtin" } as Json);
    return { ...result, services, diagnostics: services.diagnostics };
  }) as Json, { cwd, agentDir, sessionManager: manager });

  const mode: Json = new InteractiveMode(runtime, { verbose: false });
  tui.host = {
    detach: () => void mode.shutdown(),
    exportSession: (path, file) => exportSession(file, path ?? `${basename(runRoot)}.html`),
  };
  // Pi's tmux keyboard advice concerns typing; this view takes no input.
  if (typeof mode.checkTmuxKeyboardSetup === "function") mode.checkTmuxKeyboardSetup = async () => undefined;
  await mode.init();
  guardEditor(mode, tui);
  await mode.run();
}

main().catch((error) => {
  process.stderr.write(`ora view: ${error?.stack ?? error}\n`);
  process.exit(1);
});
