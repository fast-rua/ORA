// The words ORA's own terminal surface uses (notices, footer, header, help, its chat entries),
// in English or Chinese. Pi's own UI keeps its language.
//
// The language is ORA_UI_LANG (`zh` or `en`), else the locale (LC_ALL, LC_MESSAGES, LANG):
// a `zh*` locale reads Chinese. It is read once, when the agent process starts.

export type Lang = "en" | "zh";

export function uiLanguage(env: NodeJS.ProcessEnv = process.env): Lang {
  const chosen = String(env.ORA_UI_LANG ?? "").trim().toLowerCase();
  if (chosen.startsWith("zh")) return "zh";
  if (chosen.startsWith("en")) return "en";
  const locale = env.LC_ALL || env.LC_MESSAGES || env.LANG || "";
  return /^zh/i.test(locale) ? "zh" : "en";
}

let current: Lang = uiLanguage();

export function language(): Lang {
  return current;
}

/** Tests and a terminal that knows better. */
export function setLanguage(lang: Lang): void {
  current = lang;
}

const STRINGS: Record<string, { en: string; zh: string }> = {
  // Free play: what a typed message became.
  "play.instructionQueued": { en: "Instruction sent; it starts at once.", zh: "指令已发送，马上开始。" },
  "play.messageQueued": {
    en: "Queued: joins instruction {n}; the agent reads it before its next step. Esc stops the instruction.",
    zh: "已排队：并入第 {n} 条指令，智能体会在下一步之前读到。按 Esc 停止这条指令。",
  },
  "play.queuedNext": {
    en: "Queued: the next instruction, once instruction {n} ends. Esc stops it.",
    zh: "已排队：第 {n} 条指令结束后作为下一条指令开始。按 Esc 停止当前指令。",
  },
  "play.refused": { en: "Not sent: {reason}", zh: "未发送：{reason}" },
  "play.waitToMove": {
    en: "Wait until the running instruction ends (or press Esc) before moving the conversation.",
    zh: "请等当前指令结束（或按 Esc 停止）后再移动对话。",
  },
  // How an instruction ended (episode_play.py `ended`).
  "ending.reported": { en: "done; the agent reported", zh: "已完成，智能体已汇报" },
  "ending.operator_stopped": { en: "stopped (Esc)", zh: "已停止（Esc）" },
  "ending.operator_reset": { en: "stopped for /reset", zh: "因换场景（/reset）停止" },
  "ending.operator_quit": { en: "stopped for /quit", zh: "因结束会话（/quit）停止" },
  "ending.operator_interrupt": { en: "stopped by the operator", zh: "被操作员停止" },
  "ending.scene_ended": { en: "ended with the scene (the environment ended it)", zh: "随场景结束（环境结束了场景）" },
  "ending.instruction_wall_time_budget": { en: "stopped: its time ran out", zh: "已停止：时间用完" },
  "ending.instruction_robot_decision_budget": { en: "stopped: its robot decisions ran out", zh: "已停止：机器人动作次数用完" },
  "ending.instruction_model_call_budget": { en: "stopped: its model calls ran out", zh: "已停止：模型调用次数用完" },
  "ending.no_progress": { en: "stopped: the agent made no progress", zh: "已停止：智能体没有进展" },
  "ending.provider_error": { en: "ended: the model answered with an error", zh: "结束：模型返回了错误" },
  "ending.output_truncated": {
    en: "ended: the model's reply was cut off at its output limit", zh: "结束：模型的回复达到输出上限被截断",
  },
  "ending.control_fault": { en: "ended: a control fault", zh: "结束：控制故障" },
  "ending.superseded": { en: "replaced by a newer instruction (older sessions)", zh: "被新指令取代（旧版会话）" },
  "ending.other": { en: "ended ({reason})", zh: "已结束（{reason}）" },
  // Notices.
  "notice.sceneLoaded": { en: "Scene {scene} of {scenes} is loaded. Type an instruction.", zh: "场景 {scene}/{scenes} 已载入。请输入指令。" },
  "notice.sceneEndedLoaded": {
    en: "The scene ended by itself; scene {scene} of {scenes} is loaded. Type an instruction.",
    zh: "场景已自行结束；场景 {scene}/{scenes} 已载入。请输入指令。",
  },
  "notice.queuedNext": {
    en: "Held for the next instruction ({count} queued): it starts once instruction {n} ends.",
    zh: "已留作下一条指令（排队 {count} 条）：第 {n} 条指令结束后开始。",
  },
  "notice.stoppingPlanned": {
    en: "Stopping this instruction: a motion the agent is still planning is refused before it moves.",
    zh: "正在停止这条指令：智能体还在规划的动作会在执行前被拒绝。",
  },
  "notice.instructionStarting": { en: "Instruction {n} is starting.", zh: "第 {n} 条指令开始执行。" },
  "notice.instructionFinished": { en: "Instruction {n} {ending}. Type the next one.", zh: "第 {n} 条指令{ending}。请输入下一条。" },
  "notice.sessionEnding": { en: "The session is ending ({reason}); ORA evaluates and seals it.", zh: "会话正在结束（{reason}），ORA 将评估并封存。" },
  "notice.stopping": {
    en: "Stopping this instruction: a model reply in flight is recorded and its motion runs, then nothing more.",
    zh: "正在停止这条指令：进行中的模型回复会被记录、它提出的动作会执行完，之后不再继续。",
  },
  "notice.resetting": { en: "Loading another scene once the running instruction stops.", zh: "当前指令停止后载入下一个场景。" },
  "notice.quitting": { en: "Ending the session; ORA then evaluates and seals it.", zh: "正在结束会话，随后 ORA 会评估并封存。" },
  "notice.controlRefused": { en: "{control}: {reason}.", zh: "{control}：{reason}。" },
  "notice.sendRefused": { en: "{control} refused: {reason}", zh: "{control} 被拒绝：{reason}" },
  "notice.noInstruction": { en: "no instruction is running", zh: "当前没有正在执行的指令" },
  "notice.noSceneLeft": { en: "every scene drawn for this session has been used", zh: "本次会话抽到的场景已全部用完" },
  "notice.sceneEndedResetRequired": { en: "the scene has ended; /reset or /quit", zh: "场景已结束，请 /reset 或 /quit" },
  "notice.moved": { en: "The conversation moved; the robot and the scene stay as they are now.", zh: "对话已移动；机器人和场景保持现在的状态。" },
  "notice.noOperatorView": { en: "This run has no operator view yet.", zh: "这次运行还没有操作员视角画面。" },
  // Footer and phases.
  "footer.scene": { en: "scene {scene}/{scenes}", zh: "场景 {scene}/{scenes}" },
  "footer.instruction": { en: "instruction {n}", zh: "第 {n} 条指令" },
  "footer.step": { en: "step {n}", zh: "第 {n} 步" },
  "footer.sceneLoaded": { en: "scene loaded", zh: "场景已载入" },
  "footer.firstObservation": { en: "waiting for the first observation", zh: "等待第一个观测" },
  "footer.calls": { en: "{n} calls", zh: "{n} 次模型调用" },
  "footer.decisions": { en: "{n} decisions", zh: "{n} 次动作" },
  "footer.context": { en: "context {used} / {window}", zh: "上下文 {used} / {window}" },
  "footer.freePlay": { en: "free play", zh: "自由模式" },
  "footer.episode": { en: "robot episode", zh: "机器人回合" },
  "footer.finished": { en: "finished run · read-only", zh: "已结束的运行 · 只读" },
  "footer.queued": { en: "{n} queued", zh: "排队 {n} 条" },
  "footer.budget": { en: "budget left {items}", zh: "剩余预算 {items}" },
  "phase.finished": { en: "finished", zh: "已结束" },
  "phase.ended": { en: "ended: {state}", zh: "已结束：{state}" },
  "phase.stopping": { en: "stopping", zh: "正在停止" },
  "phase.ending": { en: "ending", zh: "正在结束" },
  "phase.waiting": { en: "waiting for your instruction", zh: "等待你的指令" },
  "phase.setup": { en: "setting up the cell{step}", zh: "正在设置机器人{step}" },
  // A real cell's setup before its worker starts (ora play --setup).
  "setup.title.status": { en: "Cell setup: what is done", zh: "机器人设置：已完成和未完成的步骤" },
  "setup.title.running": { en: "Cell setup: running", zh: "机器人设置：进行中" },
  "setup.title.done": { en: "Cell setup done: starting the arms", zh: "机器人设置完成：正在启动手臂" },
  "setup.title.skipped": { en: "The cell is set up: starting the arms", zh: "机器人已经设置好：正在启动手臂" },
  "setup.title.failed": { en: "Cell setup stopped: no arm was enabled", zh: "机器人设置停止：手臂没有通电" },
  "setup.answer": { en: "Its questions are answered with /approve and /decline.", zh: "它的问题用 /approve 和 /decline 回答。" },
  "setup.step.running": { en: "{title}: running", zh: "{title}：进行中" },
  "setup.step.done": { en: "{title}: done", zh: "{title}：完成" },
  "setup.step.failed": { en: "{title}: failed{why}", zh: "{title}：失败{why}" },
  "phase.thinking": { en: "thinking… {s}", zh: "思考中… {s}" },
  "phase.writing": { en: "writing… {s}", zh: "回复中… {s}" },
  "phase.tool": { en: "running {tool}… {s}", zh: "运行 {tool}… {s}" },
  "phase.moving": { en: "moving… {s}", zh: "机器人移动中… {s}" },
  "phase.observing": { en: "observing…", zh: "观察中…" },
  "phase.modelWorking": { en: "model working", zh: "模型工作中" },
  "phase.agentTurn": { en: "agent turn", zh: "智能体回合" },
  "phase.robotActing": { en: "robot acting / next observation", zh: "机器人执行中 / 等待下一个观测" },
  // Header (free play).
  "header.play": {
    en: "Enter: instruction or message · Alt+Enter queues · Esc stops it · /reset · /quit ends · /help",
    zh: "回车：发送指令或消息 · Alt+回车 排队 · Esc 停止当前指令 · /reset 换场景 · /quit 结束 · /help",
  },
  "header.episode": {
    en: "Enter steers · Esc stops the run · /status · /help · Ctrl+D detaches",
    zh: "回车：插话 · Esc 停止运行 · /status · /help · Ctrl+D 离开",
  },
  "header.finished": {
    en: "finished run, read-only · /export · /help · Ctrl+D closes",
    zh: "已结束的运行，只读 · /export · /help · Ctrl+D 关闭",
  },
  "header.effort": { en: "{effort} effort", zh: "推理强度 {effort}" },
  "header.realCell": { en: "real cell, rig {mode}", zh: "真实机器人，rig {mode}" },
  "header.freePlay": { en: "free play", zh: "自由模式" },
  // The status line above the prompt (free play).
  "status.thinking": { en: "Thinking…", zh: "思考中…" },
  "status.writing": { en: "Writing…", zh: "回复中…" },
  "status.tool": { en: "Running {tool}…", zh: "运行 {tool}…" },
  "status.moving": { en: "Moving on the robot…", zh: "机器人移动中…" },
  "status.observing": { en: "Observing…", zh: "观察中…" },
  "status.hint": { en: "esc to interrupt", zh: "Esc 中断" },
  "header.operatorView": { en: "operator view", zh: "操作员视角" },
  // Chat entries.
  "entry.joined": { en: "you · joins instruction {n}", zh: "你 · 并入第 {n} 条指令" },
  "entry.joinedNoNumber": { en: "you · joins the running instruction", zh: "你 · 并入当前指令" },
  "entry.readAt": { en: "delivered at step {step}", zh: "第 {step} 步送达" },
  "entry.instructionEnd": { en: "instruction {n} {ending} · at step {step}", zh: "第 {n} 条指令{ending} · 第 {step} 步" },
  "entry.scene": { en: "scene {scene} of {scenes}", zh: "场景 {scene}/{scenes}" },
  "entry.sceneCamera": { en: "the main camera now", zh: "当前主相机" },
  "entry.operatorView": { en: "operator view · {camera} · step {step}", zh: "操作员视角 · {camera} · 第 {step} 步" },
  // A real cell's commands (tui/cell_commands.ts).
  "cell.running": { en: "/{name}: running `ora cell {verb}`…", zh: "/{name}：正在运行 `ora cell {verb}`…" },
  "cell.cancelled": { en: "/{name} cancelled; nothing was run.", zh: "/{name} 已取消，没有执行。" },
  "cell.realOnly": {
    en: "/{name} applies to real robot cells only (rig_mode read or on); this run drives none, so nothing was run.",
    zh: "/{name} 只用于真实机器人单元（rig_mode 为 read 或 on）；这次运行没有连接真实单元，未执行。",
  },
  "cell.failed": { en: "/{name} failed: {reason}", zh: "/{name} 失败：{reason}" },
  "cell.notRecorded": { en: "/{name} is not in the run's trace: {reason}", zh: "/{name} 未记入运行轨迹：{reason}" },
  "cell.yesNo": { en: "y: yes · any other key: no", zh: "y：确认 · 其他任意键：取消" },
  "cell.confirm.float.title": { en: "Float the arms? [y/N]", zh: "让双臂浮动？[y/N]" },
  "cell.confirm.float.message": {
    en: "`ora cell float`: gravity compensation. The arms hold no pose; they move where a hand guides them.",
    zh: "`ora cell float`：重力补偿。双臂不再保持位置，用手引导到哪里就停在哪里。",
  },
  "cell.confirm.release.title": { en: "Release the arms? [y/N]", zh: "释放双臂？[y/N]" },
  "cell.confirm.release.message": {
    en: "`ora cell release`: the arms go limp; an arm that rests on nothing falls.",
    zh: "`ora cell release`：双臂失去保持力，没有支撑的手臂会落下。",
  },
  "cell.confirm.estop.title": { en: "Disable every motor? [y/N]", zh: "紧急失能所有电机？[y/N]" },
  "cell.confirm.estop.message": {
    en: "`ora cell estop`: every motor is disabled, whoever holds the cell; the arms fall.",
    zh: "`ora cell estop`：无论谁占用单元，所有电机都被失能，双臂会落下。",
  },
  "cell.confirm.approve.title": { en: "Approve this? [y/N]", zh: "批准这个请求？[y/N]" },
  "cell.confirm.approve.message": {
    en: "`ora cell approve`: the robot's waiting question is answered yes.",
    zh: "`ora cell approve`：答复机器人正在等的请求：批准。",
  },
  "cell.confirm.decline.title": { en: "Decline this? [y/N]", zh: "拒绝这个请求？[y/N]" },
  "cell.confirm.decline.message": {
    en: "`ora cell decline`: the robot's waiting question is answered no.",
    zh: "`ora cell decline`：答复机器人正在等的请求：拒绝。",
  },
  "cell.confirm.yolo.title": { en: "Turn on auto-approve (YOLO)? [y/N]", zh: "打开自动批准（YOLO）？[y/N]" },
  "cell.confirm.yolo.message": {
    en: "`ora cell auto-approve on`: every motion the agent plans runs at once, without asking you, until /yolo off or this run ends. Enabling the arms and releasing them are still asked. Keep a hand on the stop.",
    zh: "`ora cell auto-approve on`：智能体规划的每个动作都直接执行，不再问你，直到 /yolo off 或本次运行结束。给手臂通电和释放手臂仍然会问。手请放在急停上。",
  },
  "cell.yolo.on": {
    en: "auto-approve is on: motions run without asking (pid {pid}, set by {by}); enabling and releasing the arms are still asked · /yolo off",
    zh: "自动批准已打开：动作直接执行，不再询问（pid {pid}，由 {by} 打开）；通电和释放手臂仍会询问 · /yolo off 关闭",
  },
  "cell.yolo.off": { en: "auto-approve is off: every motion waits for /approve", zh: "自动批准已关闭：每个动作都等 /approve" },
  "cell.yolo.usage": {
    en: "/yolo on · /yolo off · /yolo alone says whether it is on.",
    zh: "/yolo on 打开 · /yolo off 关闭 · 单独 /yolo 查看是否打开。",
  },
  "cell.confirm.mode.title": { en: "Switch to auto: every motion runs without asking? [y/N]", zh: "切换到 auto：全部动作自动执行？[y/N]" },
  "cell.confirm.mode.message": {
    en: "`ora cell mode auto`: every motion the agent plans runs at once, without asking you, until you switch back (/mode ask, /mode safe, Shift+Tab) or this run ends. Enabling the arms and releasing them are still asked. Keep a hand on the stop.",
    zh: "`ora cell mode auto`：智能体规划的每个动作都直接执行，不再问你，直到你切回（/mode ask、/mode safe 或 Shift+Tab）或本次运行结束。给手臂通电和释放手臂仍然会问。手请放在急停上。",
  },
  "cell.mode.label.ask": { en: "ask · every motion asked", zh: "ask · 每个动作都问" },
  "cell.mode.label.safe": { en: "safe · small motions run", zh: "safe · 小动作自动" },
  "cell.mode.label.auto": { en: "auto · every motion runs", zh: "auto · 全部自动" },
  "cell.mode.cycle": { en: "shift+tab", zh: "shift+tab 切换" },
  "cell.mode.state.ask": { en: "mode ask: every motion waits for /approve", zh: "模式 ask（每个动作都问）：每个动作都等 /approve" },
  "cell.mode.state.safe": {
    en: "mode safe: a small motion runs without asking ({limits}); any other waits for /approve",
    zh: "模式 safe（小动作自动）：小动作直接执行（{limits}），其他动作等 /approve",
  },
  "cell.mode.state.auto": {
    en: "mode auto: every motion runs without asking (pid {pid}, set by {by}); back to {back} when that worker ends",
    zh: "模式 auto（全部自动）：每个动作都直接执行（pid {pid}，由 {by} 打开）；该进程结束后回到 {back}",
  },
  "cell.mode.limits": {
    en: "at most {s} s and {rad} rad per joint, at least {table} cm above the table and {apart} cm between the arms",
    zh: "最长 {s} 秒、每个关节最多 {rad} rad、离桌面至少 {table} cm、两臂相距至少 {apart} cm",
  },
  "cell.mode.setBy": { en: " (set by {by})", zh: "（由 {by} 设置）" },
  "cell.mode.always": { en: "; enabling and releasing the arms are always asked", zh: "；给手臂通电和释放手臂总会询问" },
  "cell.mode.usage": {
    en: "/mode ask · /mode safe · /mode auto · /mode alone says the mode now; Shift+Tab cycles them.",
    zh: "/mode ask · /mode safe · /mode auto · 单独 /mode 查看当前模式；Shift+Tab 依次切换。",
  },
  "cell.mode.busy": { en: "The mode is being switched; wait for the result.", zh: "正在切换模式，请等结果。" },
  "cell.selfAnswered": { en: "Ran without asking: {title} · {by}", zh: "未询问直接执行：{title} · {by}" },
  "cell.approval.head": {
    en: "{title} · id {id} · {s} s left (no answer counts as a decline)",
    zh: "{title} · 编号 {id} · 剩余 {s} 秒（不答复按拒绝处理）",
  },
  "cell.approval.reason": { en: "Reason sent with the decline: {reason}", zh: "随拒绝发送的原因：{reason}" },
  "cell.approval.value": { en: "Value sent with the approval: {value}", zh: "随批准发送的数值：{value}" },
  "cell.approval.hint.task": {
    en: "Approve = done; decline = end the captures.",
    zh: "批准 = 已完成；拒绝 = 结束采集。",
  },
  "cell.approval.hint.value": {
    en: "Approve with /approve <value>{example}; decline = give up.",
    zh: "用 /approve <数值> 批准{example}；拒绝 = 放弃。",
  },
  "cell.approval.hint.write": {
    en: "Approve = save; decline = keep the old file.",
    zh: "批准 = 保存；拒绝 = 保留原来的文件。",
  },
  "cell.approval.unitExample": { en: " (in {unit}, e.g. /approve {example})", zh: "（单位 {unit}，例如 /approve {example}）" },
  "cell.approval.example": { en: " (e.g. /approve {example})", zh: "（例如 /approve {example}）" },
  "cell.approval.unit": { en: " (in {unit})", zh: "（单位 {unit}）" },
  "cell.approval.needsValue": {
    en: "/{name}: this question asks for a value{example}: /approve <value>. Nothing was run.",
    zh: "/{name}：这个请求需要一个数值{example}：/approve <数值>。未执行。",
  },
  "cell.approval.noValue": {
    en: "/{name}: this question ({kind}) takes no value; send /approve alone. Nothing was run.",
    zh: "/{name}：这个请求（{kind}）不需要数值，单独输入 /approve 即可。未执行。",
  },
  "cell.approval.none": { en: "/{name}: nothing is waiting for an answer.", zh: "/{name}：没有等待答复的请求。" },
  "cell.worker.gone": {
    en: "The robot's control process (pid {pid}) has ended: nothing controls the arms now; they go limp when the motors time out. End this run with /quit and check the cell board.",
    zh: "控制两臂的进程（pid {pid}）已经结束：现在没有程序控制手臂，电机超时后会断电。用 /quit 结束这次运行，再看看板。",
  },
  "cell.approval.waiting": {
    en: "The robot asks: {title} ({id}) · answer with /approve or /decline",
    zh: "{title}（{id}）· 输入 /approve 或 /decline 答复",
  },
  "entry.cellDone": { en: "/{name} · ora cell {verb} · done in {s}", zh: "/{name} · ora cell {verb} · 完成，用时 {s}" },
  "entry.cellFailed": { en: "/{name} · ora cell {verb} · failed after {s}", zh: "/{name} · ora cell {verb} · 失败，用时 {s}" },
  "entry.cellFromPage": { en: "· from the web page", zh: "· 来自网页" },
};

const HELP_PLAY: Record<Lang, string[]> = {
  en: [
    "Free play: no task; the agent carries out one instruction at a time.",
    "Enter          the next instruction; while one runs, a message joins it: the",
    "               agent reads it before its next step, answers a question and",
    "               keeps working, or takes a changed request as the new goal",
    "Alt+Enter      queue it: the next instruction, once the running one ends",
    "Esc            stop the running instruction at its next boundary: a model",
    "               reply in flight is recorded and its motion runs, then nothing more;",
    "               on a real cell a motion in progress ends at once, held (/halt),",
    "               and a motion the agent is still planning is refused before it moves",
    "/reset         load another scene drawn for this session",
    "/quit          end the session; ORA evaluates and seals it",
    "/view          show the operator view (the environment's camera for you)",
    "/stop          stop the run as an operator stop (it stays incomplete)",
    "/status        step, model calls, remaining budget, last action and receipt",
    "/export [p]    write this session as HTML (.jsonl for the raw Pi session)",
    "/detach        leave; the session waits (Ctrl+D, Ctrl+Z too)",
    "Ctrl+O         expand observations and tool results",
  ],
  zh: [
    "自由模式：没有预设任务，智能体一次执行一条指令。",
    "回车           发送下一条指令；指令执行中发送的消息会并入当前指令：",
    "               智能体在下一步之前读到它，回答问题并继续工作，",
    "               或把改变的要求当作新的目标",
    "Alt+回车       排队：当前指令结束后作为下一条指令开始",
    "Esc            在下一个边界停止当前指令：进行中的模型回复会被记录、",
    "               它提出的动作会执行完，之后不再继续；",
    "               真实单元上正在执行的动作会立即就地停下并保持（/halt），",
    "               智能体还在规划的动作会在执行前被拒绝",
    "/reset         载入本次会话抽到的下一个场景",
    "/quit          结束会话；ORA 会评估并封存",
    "/view          显示操作员视角（环境给你看的相机）",
    "/stop          以操作员停止结束运行（运行记为未完成）",
    "/status        步数、模型调用、剩余预算、最近的动作和回执",
    "/export [p]    把会话导出为 HTML（.jsonl 为原始 Pi 会话）",
    "/detach        离开；会话继续等待（也可按 Ctrl+D、Ctrl+Z）",
    "Ctrl+O         展开观测和工具结果",
  ],
};

export function helpPlay(): string[] {
  return HELP_PLAY[current];
}

const HELP_CELL: Record<Lang, string[]> = {
  en: [
    "Real cell: these act on the arms at once, also while the robot moves,",
    "and never reach the agent (each runs `ora cell <verb>`):",
    "/home          drive the arms to rest",
    "/hold          hold the arms where they are",
    "/halt          end the running motion where the arms are and hold there;",
    "               the agent gets a fresh observation (Esc while moving does it too)",
    "/float         float the arms for guiding by hand (asks y/N)",
    "/release       let the arms go limp (asks y/N)",
    "/estop         disable every motor; the arms fall (asks y/N)",
    "/diagnose      who holds the cell, buses, faults, distance from rest",
    "/approve [val] answer the robot's waiting question (enable, motion, release,",
    "               a calibration step): yes, after showing it (asks y/N);",
    "               a question that asks for a value: /approve <value>",
    "/decline [why] answer it: no, with an optional reason (asks y/N)",
    "/yolo [on|off] motions run without asking (on asks y/N); enabling and",
    "               releasing the arms are still asked; /yolo alone: whether it is on",
    "/mode [m]      the permission mode, in the footer: ask (every motion asked),",
    "               safe (small motions run), auto (every motion runs; asks y/N);",
    "               enabling and releasing are always asked; /mode alone: the mode now",
    "Shift+Tab      the next mode: ask → safe → auto → ask (/thinking sets the",
    "               reasoning effort here)",
  ],
  zh: [
    "真实机器人单元：以下命令立即作用于双臂，机器人移动时也一样，",
    "且不会发给智能体（每条运行 `ora cell <动作>`）：",
    "/home          回到起始位",
    "/hold          保持不动",
    "/halt          让正在执行的动作就地停下并保持；智能体会拿到新的观测",
    "               （机器人移动时按 Esc 也会这样做）",
    "/float         浮动，便于用手引导（需确认 y/N）",
    "/release       释放，双臂失去保持力（需确认 y/N）",
    "/estop         紧急失能所有电机，双臂会落下（需确认 y/N）",
    "/diagnose      诊断：谁占用单元、总线、故障、离起始位多远",
    "/approve [值]  答复机器人正在等的请求（通电、动作、释放、标定步骤）：",
    "               批准，先显示请求内容（需确认 y/N）；",
    "               需要数值的请求：/approve <数值>",
    "/decline [原因] 答复：拒绝，可附原因（需确认 y/N）",
    "/yolo [on|off] 自动批准：动作不再询问直接执行（打开需确认 y/N）；",
    "               通电和释放手臂仍会询问；单独 /yolo 查看是否打开",
    "/mode [模式]   批准模式，显示在底栏：ask（每个动作都问）、safe（小动作",
    "               自动）、auto（全部自动，需确认 y/N）；通电和释放总会询问；",
    "               单独 /mode 查看当前模式",
    "Shift+Tab      切换到下一个模式：ask → safe → auto → ask（推理强度用 /thinking）",
  ],
};

/** The help rows of a real cell's commands. */
export function helpCell(): string[] {
  return HELP_CELL[current];
}

/** The text of `key` in the current language, `{name}` replaced from `params`. */
export function t(key: string, params: Record<string, string | number | null | undefined> = {}): string {
  const entry = STRINGS[key];
  const text = entry ? entry[current] : key;
  return text.replace(/\{(\w+)\}/g, (match, name: string) => (params[name] === undefined || params[name] === null
    ? match : String(params[name])));
}

/** How an instruction ended, as the operator reads it. */
export function endingText(ended: string): string {
  return STRINGS[`ending.${ended}`] ? t(`ending.${ended}`) : t("ending.other", { reason: ended.replace(/_/g, " ") });
}

const BUDGET_ZH: Record<string, string> = {
  "robot decisions": "次机器人动作", "model calls": "次模型调用", "wall time": "时间",
};

/** A remaining-budget item as the footer says it: "400 robot decisions", "400 次机器人动作". */
export function budgetItem(key: string, value: string): string {
  return current === "zh" && BUDGET_ZH[key] ? `${value} ${BUDGET_ZH[key]}` : `${value} ${key}`;
}

/** Seconds as the footer shows them: 7s, 1m 05s. */
export function elapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return current === "zh" ? `${seconds} 秒` : `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = String(seconds % 60).padStart(2, "0");
  return current === "zh" ? `${minutes} 分 ${rest} 秒` : `${minutes}m ${rest}s`;
}
