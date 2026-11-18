/**
 * Generate the "Robots and tasks" pages from the files the runtime itself reads, so the
 * site never restates a robot or task fact by hand:
 *
 *   robots/embodiments/<robot>.md  the public robot manual each default prompt bundle gives
 *                                  the model (ora/policy/prompt_bundles/current_v4/*native_api*
 *                                  and RoboDojo's toolkit_v7/robodojo.yaml),
 *                                  with the environment boundary it is paired with;
 *   robots/tasks/robotwin.md       every RoboTwin task: public instruction
 *                                  (ora/environment/benchmarks/robotwin-instructions-v1.json) and
 *                                  public completion contract (task-contracts/registry-v*.json);
 *   robots/tasks/robodojo.md       every admitted RoboDojo task: native action limit
 *                                  (ora/environment/benchmarks/robodojo.py) and its contract;
 *   robots/tasks/libero.md         the forty source-paired LIBERO / LIBERO-Pro tasks: the official
 *                                  instruction and the four core perturbations' BDDL language
 *                                  (ora/environment/benchmarks/libero_pro_language.py);
 *   robots/tasks/robocasa.md       every admitted RoboCasa365 task: official horizon, base use
 *                                  (ora/environment/benchmarks/robocasa.py) and its contract;
 *   robots/tasks/robolab.md        every admitted RoboLab task: instruction, native action limit
 *                                  (ora/environment/benchmarks/robolab.py) and its contract.
 *
 * Returned routes are listed in navigation.json; tests/content.test.mjs checks both ways.
 */
import { readFile, readdir, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";

const ROBOT_NAMES = {
  "franka-panda": { title: "Franka Panda", environment: "LIBERO / LIBERO-Pro", page: "../../environments/libero.md" },
  "aloha-agilex": { title: "ALOHA-AgileX", environment: "RoboTwin", page: "../../environments/robotwin.md" },
  "arx-x5": { title: "ARX X5", environment: "RoboDojo", page: "../../environments/robodojo.md" },
  "panda-omron": { title: "PandaOmron", environment: "RoboCasa365", page: "../../environments/robocasa.md" },
  "droid-franka-robotiq": {
    title: "DROID Franka with Robotiq",
    environment: "RoboLab",
    page: "../../environments/robolab.md",
  },
};

/** Drop the first H1 and shift the remaining headings down `shift` levels, outside fenced code. */
function demote(markdown, shift = 0) {
  let fence = null;
  let droppedTitle = false;
  return markdown
    .split("\n")
    .flatMap((line) => {
      const marker = line.match(/^\s*(`{3,}|~{3,})/);
      if (marker) {
        fence = fence === null ? marker[1] : marker[1][0] === fence[0] ? null : fence;
        return [line];
      }
      if (fence !== null) return [line];
      if (!droppedTitle && /^# /.test(line)) {
        droppedTitle = true;
        return [];
      }
      return [shift ? line.replace(/^(#{1,5}) /, (match, hashes) => "#".repeat(Math.min(6, hashes.length + shift)) + " ") : line];
    })
    .join("\n")
    .trim();
}

async function latestRegistry(dir) {
  const files = (await readdir(dir)).filter((name) => /^registry-v\d+\.json$/.test(name));
  files.sort((a, b) => Number(a.match(/\d+/)[0]) - Number(b.match(/\d+/)[0]));
  if (!files.length) throw new Error(`${dir}: no task-contract registry`);
  return { name: files.at(-1), data: JSON.parse(await readFile(path.join(dir, files.at(-1)), "utf8")) };
}

/** The contract body: the bullet list after "Public completion contract for this task:". */
function contractBody(markdown) {
  return markdown.replace(/^Public completion contract for this task:\s*/m, "").trim();
}

async function embodimentPages(repo, out) {
  const bundles = path.join(repo, "ora", "policy", "prompt_bundles");
  const current = (await readdir(path.join(bundles, "current_v4")))
    .filter((file) => /native_api.*\.yaml$/.test(file))
    .sort()
    .map((file) => path.join("current_v4", file));
  const routes = [];
  const seen = new Set();  // two bundles may share one robot (yam-station and its perception variant): one page each
  for (const name of [...current, path.join("toolkit_v7", "robodojo.yaml")]) {
    const yaml = await readFile(path.join(bundles, name), "utf8");
    const manual = yaml.match(/policy\/embodiments\/([\w-]+)\/([\w.-]+\.md)/);
    const boundary = yaml.match(/policy\/environments\/([\w-]+)\/([\w.-]+\.md)/);
    if (!manual) continue;
    const [, robot, manualFile] = manual;
    if (seen.has(robot)) continue;
    seen.add(robot);
    const info = ROBOT_NAMES[robot] ?? { title: robot, environment: boundary?.[1] ?? "", page: null };
    const manualPath = `ora/policy/embodiments/${robot}/${manualFile}`;
    const manualText = await readFile(path.join(repo, manualPath), "utf8");
    const parts = [
      `# ${info.title}`,
      "",
      `The robot in ${info.page ? `[${info.environment}](${info.page})` : info.environment}. ` +
        "This page is the robot manual ORA gives the model at the start of every episode, " +
        `generated from \`${manualPath}\`: what you read here is what the policy reads. ` +
        "Values measured during a run (poses, calibration, finger state) arrive with each observation instead; " +
        "see [Policy I/O](../../reference/policy-io.md) and [Frames and units](../frames-and-units.md).",
      "",
      demote(manualText),
    ];
    if (boundary) {
      const boundaryPath = `ora/policy/environments/${boundary[1]}/${boundary[2]}`;
      parts.push(
        "",
        "## What the policy is told about the environment",
        "",
        `The environment boundary paired with this robot, from \`${boundaryPath}\`:`,
        "",
        demote(await readFile(path.join(repo, boundaryPath), "utf8"), 1),
      );
    }
    const target = path.join(out, "robots", "embodiments", `${robot}.md`);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, parts.join("\n") + "\n");
    routes.push(`/robots/embodiments/${robot}`);
  }
  return routes;
}

async function robotwinTasks(repo, out) {
  const contracts = path.join(repo, "ora", "policy", "environments", "robotwin", "task-contracts");
  const registry = await latestRegistry(contracts);
  const instructions = JSON.parse(
    await readFile(path.join(repo, "ora", "environment", "benchmarks", "robotwin-instructions-v1.json"), "utf8"),
  ).tasks;
  const tasks = [...new Set([...Object.keys(instructions), ...Object.keys(registry.data.tasks)])].sort();
  const lines = [
    "# RoboTwin tasks",
    "",
    `${tasks.length} RoboTwin tasks. For each: the public instruction and the public completion contract ORA gives ` +
      "the model, which restates in plain words what RoboTwin's own `check_success` requires. " +
      "Success is decided only by that official check at the end of the round; the contract never replaces it. " +
      "Generated from `ora/environment/benchmarks/robotwin-instructions-v1.json` and " +
      `\`ora/policy/environments/robotwin/task-contracts/${registry.name}\`.`,
    "",
    "| Task | Instruction |",
    "| --- | --- |",
    ...tasks.map((task) => `| [\`${task}\`](#${task}) | ${(instructions[task]?.instruction ?? "—").replace(/\|/g, "\\|")} |`),
  ];
  for (const task of tasks) {
    const entry = registry.data.tasks[task];
    lines.push("", `## ${task} [#${task}]`, "");
    if (instructions[task]) lines.push(`**Instruction.** ${instructions[task].instruction}`, "");
    if (entry?.path) {
      lines.push("**Completion contract.**", "", contractBody(await readFile(path.join(repo, entry.path), "utf8")));
    } else {
      lines.push("No reviewed completion contract; the official check alone defines success.");
    }
  }
  const target = path.join(out, "robots", "tasks", "robotwin.md");
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, lines.join("\n") + "\n");
  return ["/robots/tasks/robotwin"];
}

async function robodojoTasks(repo, out) {
  const contracts = path.join(repo, "ora", "policy", "environments", "robodojo", "task-contracts");
  const registry = await latestRegistry(contracts);
  const source = await readFile(path.join(repo, "ora", "environment", "benchmarks", "robodojo.py"), "utf8");
  const block = source.match(/NATIVE_ACTION_LIMITS\s*=\s*\{([\s\S]*?)\n\}/);
  if (!block) throw new Error("robodojo.py: NATIVE_ACTION_LIMITS not found");
  const limits = Object.fromEntries([...block[1].matchAll(/"([\w]+)"\s*:\s*(\d+)/g)].map(([, task, n]) => [task, Number(n)]));
  const tasks = Object.keys(limits).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
  const lines = [
    "# RoboDojo tasks",
    "",
    `${tasks.length} admitted RoboDojo tasks on the dual-arm ARX X5. For each: RoboDojo's own action limit for the ` +
      "task and, where one has been reviewed, the public completion contract ORA gives the model. " +
      "Success is decided only by RoboDojo's official result; the contract restates it and never replaces it. " +
      "Generated from `ora/environment/benchmarks/robodojo.py` (`NATIVE_ACTION_LIMITS`) and " +
      `\`ora/policy/environments/robodojo/task-contracts/${registry.name}\`.`,
    "",
    "| Task | Native action limit | Contract |",
    "| --- | ---: | --- |",
    ...tasks.map((task) => `| [\`${task}\`](#${task.toLowerCase()}) | ${limits[task]} | ${registry.data.tasks[task] ? "reviewed" : "—"} |`),
  ];
  for (const task of tasks) {
    const entry = registry.data.tasks[task];
    lines.push("", `## ${task} [#${task.toLowerCase()}]`, "", `**Native action limit.** ${limits[task]}`, "");
    if (entry?.layer) {
      lines.push("**Completion contract.**", "", contractBody(await readFile(path.join(repo, entry.layer), "utf8")));
    } else {
      lines.push("No reviewed completion contract; the official result alone defines success.");
    }
  }
  const target = path.join(out, "robots", "tasks", "robodojo.md");
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, lines.join("\n") + "\n");
  return ["/robots/tasks/robodojo"];
}

/** LIBERO's ``Task.language``: libero_pro_language.language_from_filename, mirrored. */
function languageFromFilename(name) {
  let stem = name.replace(/\.bddl$/, "");
  if (/^[A-Z]/.test(stem)) stem = stem.slice(stem.indexOf("SCENE") + (stem.includes("SCENE10") ? 8 : 7));
  return stem.split("_").join(" ");
}

const cell = (text) => String(text).replace(/\|/g, "\\|");

async function liberoTasks(repo, out) {
  const sourcePath = "ora/environment/benchmarks/libero_pro_language.py";
  const source = await readFile(path.join(repo, sourcePath), "utf8");
  const rows = [...source.matchAll(/\("(\w+)",\s*(\d+),\s*"(\w+)",\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\)/g)]
    .map(([, family, id, file, task, swap, lan, object]) => ({ family, id: Number(id), file, task, swap, lan, object }));
  if (rows.length !== 40) throw new Error(`${sourcePath}: expected forty TASK_ROWS, found ${rows.length}`);
  const families = [
    ["spatial", "LIBERO-Spatial", "libero_spatial"],
    ["object", "LIBERO-Object", "libero_object"],
    ["goal", "LIBERO-Goal", "libero_goal"],
    ["libero_10", "LIBERO-10", "libero_10"],
  ];
  const lines = [
    "# LIBERO / LIBERO-Pro tasks",
    "",
    "The forty source-paired tasks: ten in each of LIBERO-Spatial, -Object, -Goal and -10. Each runs in its base " +
      "suite (`libero_<family>`) and in LIBERO-Pro's four core perturbation suites (`libero_<family>_task`, `_swap`, " +
      "`_lan`, `_object`, reported as Task, Pos, Sem and Obj), each task on 50 official initial states. " +
      "A base cell instructs with LIBERO's official instruction (the task's BDDL file name as text); a core cell " +
      "instructs with its own BDDL `(:language ...)` field, and launch preflight compares that field with the BDDL it " +
      "loads. Success is decided only by LIBERO's own termination; ORA gives no completion contract for these tasks. " +
      `Generated from \`${sourcePath}\` (\`TASK_ROWS\`).`,
    "",
    "| Family | Base suite | Tasks |",
    "| --- | --- | ---: |",
    ...families.map(([family, title, suite]) =>
      `| [${title}](#${family.replace("_", "-")}) | \`${suite}\` | ${rows.filter((row) => row.family === family).length} |`),
  ];
  for (const [family, title, suite] of families) {
    const tasks = rows.filter((row) => row.family === family).sort((a, b) => a.id - b.id);
    lines.push(
      "", `## ${title} [#${family.replace("_", "-")}]`, "",
      `Base suite \`${suite}\`; core suites \`${suite}_task\`, \`${suite}_swap\`, \`${suite}_lan\` and \`${suite}_object\`. ` +
        "The position (`_swap`) variant moves objects and keeps the base BDDL wording.",
      "",
      "| Task | Official instruction (base) | Task (`_task`) | Position (`_swap`) | Language (`_lan`) | Object (`_object`) |",
      "| ---: | --- | --- | --- | --- | --- |",
      ...tasks.map((row) =>
        `| ${row.id} | ${cell(languageFromFilename(row.file))} | ${cell(row.task)} | ${cell(row.swap)} | ${cell(row.lan)} | ${cell(row.object)} |`),
    );
  }
  const target = path.join(out, "robots", "tasks", "libero.md");
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, lines.join("\n") + "\n");
  return ["/robots/tasks/libero"];
}

/** The ``TASKS = {...}`` dict of a benchmark module as {task: body text}. */
function taskBlocks(source, file) {
  const block = source.match(/^TASKS[^=]*=\s*\{([\s\S]*?)\n\}/m);
  if (!block) throw new Error(`${file}: TASKS not found`);
  return Object.fromEntries([...block[1].matchAll(/^ {4}"(\w+)":\s*\{([\s\S]*?)\n? {4}\},?$/gm)].map(([, task, body]) => [task, body]));
}

function field(body, name) {
  const match = body.match(new RegExp(`"${name}":\\s*("[^"]*"|\\d+|True|False)`));
  if (!match) return null;
  return match[1].startsWith('"') ? match[1].slice(1, -1) : match[1];
}

async function contractLines(repo, registry, task) {
  const entry = registry.data.tasks[task];
  if (!entry?.path) return ["No reviewed completion contract; the official check alone defines success."];
  return ["**Completion contract.**", "", contractBody(await readFile(path.join(repo, entry.path), "utf8"))];
}

async function robocasaTasks(repo, out) {
  const file = "ora/environment/benchmarks/robocasa.py";
  const tasks = taskBlocks(await readFile(path.join(repo, file), "utf8"), file);
  const registry = await latestRegistry(path.join(repo, "ora", "policy", "environments", "robocasa", "task-contracts"));
  const names = Object.keys(tasks).sort();
  const lines = [
    "# RoboCasa365 tasks",
    "",
    `${names.length} admitted RoboCasa365 tasks of the target split on the PandaOmron mobile manipulator. For each: ` +
      "RoboCasa's evaluation horizon (the native-action limit the policy sees), whether the task needs the mobile base, " +
      "and the public completion contract ORA gives the model. RoboCasa samples the kitchen and the target from the " +
      "episode seed, so the instruction is read from that episode's reset, not listed here. Success is decided only by " +
      "RoboCasa's own `_check_success`; the contract restates it and never replaces it. " +
      `Generated from \`${file}\` (\`TASKS\`) and \`ora/policy/environments/robocasa/task-contracts/${registry.name}\`.`,
    "",
    "| Task | Horizon (control steps) | Mobile base | Contract |",
    "| --- | ---: | --- | --- |",
    ...names.map((task) => `| [\`${task}\`](#${task.toLowerCase()}) | ${field(tasks[task], "horizon_control_steps")} | ` +
      `${field(tasks[task], "mobile_base_required") === "True" ? "required" : "not required"} | ` +
      `${registry.data.tasks[task] ? "reviewed" : "—"} |`),
  ];
  for (const task of names) {
    lines.push("", `## ${task} [#${task.toLowerCase()}]`, "",
      `**Horizon.** ${field(tasks[task], "horizon_control_steps")} control steps at 20 Hz.`, "",
      ...(await contractLines(repo, registry, task)));
  }
  const target = path.join(out, "robots", "tasks", "robocasa.md");
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, lines.join("\n") + "\n");
  return ["/robots/tasks/robocasa"];
}

async function robolabTasks(repo, out) {
  const file = "ora/environment/benchmarks/robolab.py";
  const tasks = taskBlocks(await readFile(path.join(repo, file), "utf8"), file);
  const registry = await latestRegistry(path.join(repo, "ora", "policy", "environments", "robolab", "task-contracts"));
  const names = Object.keys(tasks).sort();
  const lines = [
    "# RoboLab tasks",
    "",
    `${names.length} admitted RoboLab tasks on the DROID Franka with a Robotiq 2F-85 gripper. For each: the upstream ` +
      "default instruction, the native-action limit (the upstream episode length at 15 Hz) and the public completion " +
      "contract ORA gives the model. Success is decided only by RoboLab's own success termination; the contract " +
      "restates it and never replaces it. " +
      `Generated from \`${file}\` (\`TASKS\`) and \`ora/policy/environments/robolab/task-contracts/${registry.name}\`.`,
    "",
    "| Task | Instruction | Native action limit | Contract |",
    "| --- | --- | ---: | --- |",
    ...names.map((task) => `| [\`${task}\`](#${task.toLowerCase()}) | ${cell(field(tasks[task], "instruction"))} | ` +
      `${field(tasks[task], "native_action_limit")} | ` +
      `${registry.data.tasks[task] ? "reviewed" : registry.data.pending_review?.[task] ? "pending witness" : "—"} |`),
  ];
  for (const task of names) {
    lines.push("", `## ${task} [#${task.toLowerCase()}]`, "",
      `**Instruction.** ${field(tasks[task], "instruction")}`, "",
      `**Native action limit.** ${field(tasks[task], "native_action_limit")} ` +
        `(${field(tasks[task], "episode_length_s")} s at 15 Hz).`, "",
      ...(registry.data.pending_review?.[task]
        ? ["A completion contract is generated and waits for its seed-0 witness; it is not delivered yet, and " +
           "launch preflight refuses the task until it is reviewed."]
        : await contractLines(repo, registry, task)));
  }
  const target = path.join(out, "robots", "tasks", "robolab.md");
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, lines.join("\n") + "\n");
  return ["/robots/tasks/robolab"];
}

export async function generateRobotPages(repo, out) {
  return [
    ...(await embodimentPages(repo, out)),
    ...(await liberoTasks(repo, out)),
    ...(await robotwinTasks(repo, out)),
    ...(await robodojoTasks(repo, out)),
    ...(await robocasaTasks(repo, out)),
    ...(await robolabTasks(repo, out)),
  ];
}
