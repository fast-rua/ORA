# Toolkit 历史边界试运行

`ora toolkit replay` 的可信装配层。`selection.py` 读取父工具提案与公开发布边界；
`services.py` 通过现有 readiness report 绑定感知客户端；`replay.py` 创建独立输出、
构建 Replay/LegacyReplay、运行显式给定的 `run(api, args)`，保存结果和错误。
只向下 import agent toolkit、现有感知客户端及 orchestrator 的 spec 输入装配。
recorded 没有 simulator/动作能力；live 只经 backend registry 获取独立诊断 worker，
再经原 action_executor 执行显式允许的一次动作；同步工具（toolkit v4 起）要动作回执才能继续，
所以它的动作在工具内执行，同样要 `--execute-action`，工具状态从空的 `tool.marks`、`tool.held.*`
开始。没有 router 分配或模型提供商调用。

用法和产物见 [toolkit-io](../../../docs/reference/toolkit-io.md)；脚本示例见
[examples](../../agent/toolkit/examples/README.md)。这不是续跑，且脚本仍在原进程中执行；
没有进程隔离。源码中的 run 工具输入、CallContext 和调用顺序不会被换代码的参数覆盖。
`--args` 只替换交给脚本的参数。运行脚本须由使用者显式选择，不能自动执行轨迹里的代码。

新包对应测试在 `tests/ora/orchestrator/toolkit/`。

## M5 交接：2026-10-23

实现 `57b012999d838930621d13d7d0ddb94b9345cb50`，命令总表修正 `557a2d18`。
入口为 `ora toolkit replay RUN --list` 或 `--at K/ID --script PATH`，服务只经显式
`--service-report` 绑定，输出在新目录；`--tool` 明确等待 M4。lib 仍为 0.1.0，
代码/README 哈希未改；原 runtime dispatch、默认记录开关和模型工具面不变。

专项（含 toolkit、只记录 fail-open、事件索引、结构、CLI）197 passed。
首次完整 quick 仅发现手写命令表漏行，修正后该项 4 passed；第二次完整 quick：
**6477 passed，140 skipped，1 个既有 xfailed，1 个既有 overflow warning**。
release tests 88、research 8、catalog 6、goldens 7、no-write 1 均通过。
model-visible 六个 cell 全同，release-scan 和作者检查通过；59 个保护文件及
pyproject.toml / uv.lock 与 69912fbc 原字节相同。此后只追加本交接记录，最终 HEAD
再核对可见内容和发布扫描。日志 `/tmp/ora-toolkit-m5-check-final-2/`、
`/tmp/ora-toolkit-m5-model-visible-final.json`、`/tmp/ora-toolkit-m5-release-scan-final.log`。

真实旧 general_pickup 的只读 --list 得到 9 次调用；没有启动仿真或请求真实服务。
公开 fixture 的 fake-service 示例保存于 `/tmp/ora-toolkit-m5-example/result.json`：
3 个全图假 mask，各 307200 个有效点，world 坐标，comparison=unavailable
（legacy_response_scope_not_equivalent）。这不是实际 scissors 检测结果。
另有 exact 假服务记录验证 equal、仿真 envelope 原字节、同参重复查询按次序消费，
以及新查询 NotRecorded、动作 Divergence、gap 优先拒绝、源文件全树 hash 不变。

已知边界：脚本仍是可信的原进程 Python；旧 run 不补算缺失探测或猜测回执；
不同输入表示/参数/服务配置/游标的感知结果可以重算，但不能声明历史等价；
非默认 CGN 配置可能不匹配当前默认客户端身份。没有真实 toolkit-io 联调结论，
等待独立运行产物。工具内部新线程不继承 recorder ContextVar 的既有边界未改。
本里程碑完成后停下等审，不进入 M4。


## M4 原子工具装配

`--tool` 支持 vision_ground / grasp_plan / grasp_execute / move_pose / move_to / compute /
check_task / finish_episode / observe，必须与选中父调用同名。
`tool_inputs.py` 校验单个边界输入表、注册 CAS 引用，再把同一数据接口交给工具。
有 input-index 的脚本也得到该输入表；不扫描最后状态的产物目录。旧 run 仅支持
保守 vision 重算，缺少 grasp 输入时明确 NotRecorded。源 run 始终只读。
工具缓存不会跳过回放感知请求，准备文件的 hash 也参与动作分歧判定。
正式使用方式和限制见上面的 reference 链接；M5 交接段保留历史检查记录。

## M7 recorded 工作台

M4 段的同名限制已解除：该 run 冻结工具面中的已迁移工具均可选，换工具必须给参数；
check_task 独立开放并标能力覆盖。`boundary.py` 只授予调用前公开观测与冻结配置，
`interface.experiment` 区分严格顺序和替代工具；缺仿真查询给结构化 missing_requests。
`progress.py` 即时写阶段进度与计时，`visible.py` 复用 Pi 投影，分别比较模型文本与
canonical 存档。完整字段和用法见 reference 的 M7 节。
M7.4 增加 live，见下节；同进程脚本权限边界未变。


## M7.4 live 诊断

`live_boundary.py` 固定已发布 state、观测物理时钟、原生控制前缀；
`live_observation.py` / `sensor_metrics.py` 记录新渲染的传感器差异，工具始终使用原发布字节；
`render_history.py` 按原 calibration 的渲染引用选择同步前缀，包含执行中途截帧；
`live_host.py` 只在输出目录组装冻结输入与原 skill driver 所需的兼容产物；
`surface_audit.py` 核对冻结 advertised surface 自身完整性，和当前 runtime 的差异仅告警，
保存逐字段差异与两侧身份，不要求上下文投影与执行定义全等；
`live_api.py` 把 simulator 查询交给已验证 worker，把感知交给显式服务与比较路径。
submit_action 只提议；显式 execute-action 在工具返回后使用原 action_executor。
backend host 管理独立进程/lease/scratch，状态不一致不授予能力，官方判定只经 check_task。
`experiment(motion_stack=...)` 让这次调用跑在另一套运动栈上（`ora replay --incident` 给 RoboDojo 用默认 baseline）：
worker 用 plan runtime 的副本覆盖规划器和时序字段，host 换工具面与授权，录制配置换 `motion` 段；密封 plan 不动。

只复原一次工具调用，不续跑 policy。未知的历史工具 ledger/configuration 不由真值或
最终目录推断；缺少时仍拒绝。M7.4b 只用零容差物理核对放行；传感器差异写统计，
标明“未逐字节复现的传感器”，不阻止工具运行。完整命令、证据和限制见 reference 的 M7.4 节。新增诊断入口独立记录哈希，
原 worker kernel/59 个保护文件保持不变；真实 lane 验收由审阅会话另行运行。

## Phase-one completed snapshots

`snapshot_boundary.py` selects and verifies a complete state publication, and
adds the exact pre-parent tool delta overlay for `--snapshot --at`.
`snapshot_api.py` grants original same-round observation bytes, private tool
writes, fresh external services and output-only episode/context declarations.
`snapshot_replay.py` owns the separate CLI experiment; `--step` implies this
path. `snapshot_config.py` gives a requested tool (another tool, or other
arguments) the configuration the live dispatch would build at that boundary, or
a structured refusal naming the boundaries that record it. It never assembles an online action or worker-planner capability and never
uses later tool-input records. Plain `--at` remains the legacy diagnostic path.
`native_boundary.py` additionally selects completed native tick counts, freezes
exact control/physics prefixes and maps pre-action tool deltas through the
execution ledger. `native_snapshot.py` uses the backend diagnostic adapter to
verify and freshly render that endpoint, releases it, and then assembles the
same action-free snapshot facade. It never imports simulator backend modules. See toolkit-io.md
for command examples, missing-input behavior and the in-process trust boundary.

Snapshot comparisons are audit-only: scripts get no historical comparison or
response blob grants. An in-parent --step publication marks tool.* missing;
--snapshot --at restores settled tool state. Native workers use OUTPUT as their
shared root, check render destinations before launch, and unwind on SIGTERM.
See toolkit-io.md for precise missing fields and legacy-route separation.
