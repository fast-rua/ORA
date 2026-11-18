# 数据接口：runtime / replay

M2 实现进度：协议、语义 envelope、事件日志、严格序列重放、旧 snapshot.v0
导入、现有客户端与控制协议的可信装配适配器。默认工具 dispatch 尚未切换；可选只记录模式已接旧 runtime；
M4 再接新原子工具。本包不 import 旧 tools/environment/orchestrator；可信装配
在更高层 `tools/runtime/toolkit_adapter.py`。这版仍同进程，不是安全沙箱。

- `api.py`：15 个方法，明确参数、默认值与来源；没有 GT API。
  M3 另加 read_blob/save_array 存储能力，只读已授予引用/写派生数组，不占语义 ordinal。
- `types.py`：严格 canonical JSON、不可变 CallContext/Response/PublicData；
  response.payload 保持历史语义 envelope 原字节，比较字段独立。
- `blobs.py`：公开文件 allowlist、显式注册、hash/长度校验、数组 dtype/shape/
  C/F order/byteorder。接口不提供任意路径读取。工具不获得 register capability。
- `protocol.py`：公共协议声明与调用日志的只读兼容性审计；版本不同/缺失只警告，内容完整性仍严格。
- `journal.py`：单 writer、每次调用 begin/end、失败/超时/重试/不完整记录。
- `runtime.py`：可信回调只返回逐字段 PublicData；主动接口先记录 begin 再执行；动作不重试。
  只记录 capture 保留原调用结果/异常，记录失败也执行原调用且只执行一次。
- `gaps.py`：尽力写入记录器诊断，写入失败不影响原工具；replay 拒绝所选/无父身份缺口。
- `projections.py`：motion-probe、align、reach、回执摘要与控制结果公开字段投影。
- `services.py`：现有 SAM3 客户端、CGN 原候选池适配；假服务可注入。
- `comparison.py`：各外部方法的 volatile 清单、数组/候选逐字段差异与默认绝对容差。
- `replay.py`：simulator/控制只有历史游标，没有在线能力；动作不符报分歧。
  外部服务每次执行；同参重复以 ordinal 区分；新服务 query 不推进历史游标。
- `legacy.py`：snapshot.v0 原观测/本体感知引用；旧回执/check 明确绑定才恢复。
  无完整请求身份的旧 probe 不猜测；原工具 manifest 不是裸服务响应。
- `stages.py`：`stage(name, **fields)`，工具记下结果里没有的决定（试过又丢掉的候选、从默认值解出的值）。没有调用方收集（`collecting()`）时是空操作：
  工具调用模式从不收集，结果、账本和日志都不变；code mode（`library: extended`）在每次脚本调用外收集，供 `explain` 渲染，事件不进任何结果。
- `snapshot.py:open_replay`：给 run 根、快照、独立输出根、服务/身份配置；不 import RSI。
  native-action 中途快照不提供接口游标，报未记录。

协议、落盘布局和限制见 [证据格式](../../../../docs/reference/toolkit-io.md)。
测试位于 `tests/ora/agent/toolkit/interface/`；真实旧数据 fixture 位于
`tests/data/toolkit-general-pickup/`，单元测试不联网、不占 GPU、不启动 simulator。

## 装配

在线由 `bind_runtime` 绑定现有运行时、一个父 tool_call_id 及发布观测。
观测 manifest 和相机产物 hash 验证后注册；worker 回调只返回公开投影。
submit 必须显式传现有已经验证的 proposal/skill-sequence 入口，接口不另写绕过
原准入协议的动作通道。旧工具派发维持原提案语义；新工具面以 execute_action
同步执行多段动作，由 Supervisor 的预算和持久账本准入，不暴露 submit_action。
冻结的 v1 提案通道保留 CallScope 的单次提案守卫；同步执行不使用此守卫。

服务 bindings 由装配者提供 backend/service/model/configuration identity，不在接口
层猜端点或读取凭据。CGN 保留 `_canonical_backend_output` 中全部已验证数组，
忽略原 proposer 的排序后编译候选；SAM3 保留既有验证客户端边界（单点连通域
归一化、定位候选范围与 bbox clipping），不声称返回原始 HTTP mask。

replay 的 source BlobStore 只读；evaluation Runtime 仅装配 external 方法，写到
独立根，借用 source 的已注册输入。返回本次重算结果；comparison.py 对去 volatile 的内容给出 equal / within_tolerance /
different，并保留完整 envelope hash 审计。默认容差与逐字段指标见证据格式。
没有历史同范围输出则 unavailable。
source simulator envelope 不重编码，引用文件再按 hash 读取。相同 hash 不代表
两个不同编码范围等价。

## 时刻与缺失

submit 只确认 proposal，不同步执行、不暴露未来回执；接口游标不是物理时钟。
state_after 在本版未独立采样时为 null，不复制 state_before 冒充执行后状态。
check_task 仅使用当时控制 latch 或匹配的历史 check 调用，禁用/缺失不读 final verdict。

旧轨迹没有统一请求键。legacy 观测保留原文件字节，同时明确返回的 envelope 是
重建的，`exact_interface_bytes=false`。缺 request/call 绑定的 IK 工具片段不能冒充
精确底层 probe 结果。v0 的 final/posthoc 证据与 available_at_boundary 分开。

## SDK 核实与后续记录

只读核查 SDK `ee67a1468510da7624a089164402359f2afc72c8`、其 cuRobo
`d17b54ce32cba095c0b000c4c58777075d11de0e`：SDK 的 InverseKinematics 和
MotionPlanner 都来自 cuRobo；ORA 本基线 sdk_candidates 调 SDK 的 solve_pose。
但 ORA motion_adapter.plan_global 只用 IK 包单端点轨迹，保留 non_parity_motion，
没有调用 SDK 的 plan_path。接口不改变 backend 证明范围或现有 lane qualification。

NOTES：F10 包归档/外部工具包身份、F12 更多 backend/中途边界 fixture、F13 文档
口径整理及跨进程隔离/负向安全测试留后续。默认工具和模型可见 schema/prompt 不改。

## M2 验证记录与交接（2026-10-23）

实现提交 `763360d2f641e69bce922b6dda4b1a043da48ab4` 的完整 quick 已通过：
6336 passed、140 skipped、1 xfailed；只有既有极端几何输入 overflow warning。
release tests 88、research 8、catalog 6、goldens 7、no-write 1 均通过；release scan
通过。model-visible 对比基线 69912fbc 的 6 个 cell 全同，0 different/0 unresolved。
59 个保护文件（58 个 WORKER_KERNEL_PATHS + action_executor）原字节全同。

收尾证据校验提交 f39000b0 后，接口专项 36 测试通过。最终代码的 quick 全部通过：
6338 passed、140 skipped、1 xfailed，1 个既有 warning；model-visible 6/6 same，
release scan 通过。该验证之后只更新此交接文字。最终日志 `/tmp/ora-toolkit-m2-check-clean/`、
`/tmp/ora-toolkit-m2-model-visible-release.json`、`/tmp/ora-toolkit-m2-release-scan.log`。

实际 writer 生成的 3 次调用已交叉通过 RSI main faa1c44 的 load_toolkit：同参数
check_task 两次具有相同 semantic hash、不同 invocation id，外部 ground source
分类正确；RSI 的固定日志缓存可保持 snapshot 字节。RSI 的 json_pointer 名称
对齐已通知轨迹会话；跨仓库正式接口约定已同步给审阅方维护。

明确限制：默认旧工具 dispatch 尚未切换（M4）；本版未启动 simulator 或请求真实服务；
旧轨迹没有完整底层请求/调用身份就未记录；中途 native-action 边界没有接口游标；
未知 service/model/config identity 的比较为 unavailable；receipt 只投影公开动作结果
与测得机器人状态，不把任意原始控制日志序列化。后续增加字段须明确投影版本。
M2 推送后等 E0003 审阅，未经通过不开始 M3。

## M2.1 / M2.2 验证与交接（2026-10-23）

代码提交 b52fafe0、bd3acd31：comparison 的内容/完整字节审计两层已分开；
RunSpec 顶层 `record_toolkit_io: true`（默认 false）只包装现有工具的真实 I/O，
不切换旧 dispatch，不把开关传入 worker 环境配置。详细开法、容差、父调用绑定与
失败语义见 [证据格式](../../../../docs/reference/toolkit-io.md)。

最终代码专项加结构检查 80 passed；quick 为 6368 passed、140 skipped、1 个既有
xfailed、1 个既有 overflow warning。release tests 88、research 8、catalog 6、
goldens 7、no-write 1 均通过；release scan 通过；model-visible 六个 cell 全同。
59 个保护文件仍与开发基点原字节相同。此后仅补交接文字。

固定数据覆盖四个外部方法的 equal / volatile-only equal / within_tolerance /
different；mask 像素差/IoU、候选排序/计数、pose 差异；原 dispatch 的 ground、
check_task、report_issue、set_gripper 开关前后结果/动作文件一致；原异常对象、
原 CGN 编译结果保留；重复父提案分开记录；历史观测复制也验证公开路径及 hash。
旧 dispatch 记录的 observation → ground → observation 经 RSI load_toolkit /
toolkit_index 验证，exact_interface_replay=true；观测 replay 原字节相同，假 SAM3
重算内容 equal。未启动真实 simulator、模型服务或资源任务。

日志：`/tmp/ora-toolkit-m21-m22-check-final/`、
`/tmp/ora-toolkit-m21-m22-model-visible-final.json`、
`/tmp/ora-toolkit-m21-m22-release-scan-final.log`。交接后停在 M2.2 等审阅，M3 未开始。

## M2.3 验证与交接（2026-10-23）

代码提交 `90f52099f7a3d8524d5dd1cf5b87ab134bcd8270`：只记录模式 fail-open，
原调用恰好一次，返回原对象/原异常。上下文、begin、输入/输出 store、project、end、
初始化或并发/递归记录失败只写尽力 gap；gap 写失败也不改变工具结果。
M2.3 当时首次故障停止 session 记录；M6 改为可恢复故障仅停止当前父调用：已知
begin 可用 error/recording_gap 闭合，下一父调用重新记录。不确定的 append 或闭合
失败仍冻结 writer、不续接残缺日志；所有后续父调用仍执行。动作文件已提交后投影失败不再变成 Pi raw_ok=false。
Replay 拒绝所选/无父身份 gap 或损坏 gap 文件；不接在线动作。

增量索引与 journal 字节计数避免反复读取/重编码历史。16 MiB、70 调用离线测量
记录额外中位耗时约 2.02 ms/调用，日志只读一遍；详细范围、格式与复现入口见证据格式。
开关仍是 RunSpec 顶层 `record_toolkit_io: true`，默认关闭，旧 dispatch 不变。

专项加结构检查 **131 passed**。最终代码 quick：**6424 passed、140 skipped、
1 个既有 xfailed、1 个既有 overflow warning**；release tests 88、research 8、
catalog 6、goldens 7、no-write 1 全通过。release scan 通过；model-visible 六个
cell 全同；58 个 worker kernel 文件加 action_executor 共 59 个文件与 69912fbc
原字节全同。此后只补本交接文字，不修改代码。

故障测试覆盖原返回/原异常、真实 Pi RPC 投影、submit 后 project/store/end 失败时
动作文件相同、gap 本身写失败、真实递归/并发、初始化失败和后续调用继续执行。
增量索引测试覆盖追加/部分尾行/截断/替换/身份歧义；Replay 测试覆盖 gap 拒绝。
实际 writer 的合成故障输出经独立 RSI coverage 读取：1 个 recorded、2 个有 gap，
1 条无父 ID gap、1 个 incomplete attempt；没有改 RSI 实现或导入其代码到 ORA。

日志：`/tmp/ora-toolkit-m23-check/`、`/tmp/ora-toolkit-m23-model-visible.json`、
`/tmp/ora-toolkit-m23-release-scan.log`、`/tmp/ora-toolkit-m23-benchmark.json`、
`/tmp/ora-toolkit-m23-reader.json`。gap 写入仍可能失败，缺 gap 不能证明完整；
真实服务/仿真验证待运行会话执行，本次未启动。M2.3 推送后停下等审，M3 未开始。

M2.3 审阅确认的边界：工具内部若另开线程，ContextVar 默认 False，线程内调用
不记录，也不留 gap；当前 tool runtime 没有这种线程。M3 只记录限制，不改变行为。

## M3 lib v0 验证与交接（2026-10-23）

实现 `563a365402a54ee1f0194929585610172743cb6b`，约定修正 `f3beefab`。
lib 版本 0.1.0，内容身份 `8144e1d2f4a24757cc03888219d742ce046e3be2ddee041bd7e5600e45a45f97`。
模块、16 个不可变类型、函数/replay 分类与已知范围见 [lib 手册](../lib/README.md)。
本地 helpers 与人审提升流程见 [工具目录约定](../tools/README.md)。默认 dispatch
尚未切换；M3 没改旧 tools、Pi、policy、environment 或 orchestrator。

接口增加 read_blob/save_array 存储能力，lib 不持有文件根、BlobStore 或客户端。
只读已授予的公开引用；派生数组只写输出 CAS；无新语义 ordinal，无 calls/snapshot
schema 变更。主仓库接口约定已同步，读取层不需要为此改变 schema。

专项、结构及打包检查 166 passed；其中新增 lib 测试 30 个。首次完整 quick 找到
三处约定问题，已补测试目录索引、使用相对 toolkit 的 manifest 路径、沿用 canonical
arm_joint_qpos 并提供单位明确的只读属性；不改模型单位映射。修正定向回归 56 passed。
最终完整 quick **6454 passed、140 skipped、1 个既有 xfailed、1 个既有 overflow
warning**。release tests 88、research 8、catalog 6、goldens 7、no-write 1 均通过。
model-visible 六个 cell 全同、release scan 通过；59 个保护文件及 pyproject.toml /
uv.lock 原字节与 69912fbc 相同。此后仅更新本交接文字，最终 HEAD 再核对可见内容/发布扫描。

固定数据覆盖：原始 legacy manifest/proprioception 字节（不同 JSON 编码域分开验证）；
所有公开接口 lib 包装；runtime/replay 的观测→ground→完整 32 候选 CGN 链；全部当前
SAM3 提示及候选；新感知请求可重算，新 IK 请求未记录即失败；派生数组不写源 run；
不可变 bytes-backed 数组/JSON；已知坐标点、深度过滤、PCA 退化及受控点变换。

日志 `/tmp/ora-toolkit-m3-check-final/`、`/tmp/ora-toolkit-m3-model-visible-final.json`、
`/tmp/ora-toolkit-m3-release-scan-final.log`。未请求真实服务、未启动仿真或占卡。
SAM3 未开放路由、完整 embodiment compiler 与目标产物迁移、plan lib 身份绑定及
跨进程隔离仍是明确后续；通用几何转换不等价于旧抓取执行器。M3 推送后停下等审，不进 M4。

## M4 公开工具输入与派生产物

`tool_data.py` 提供 tool_context / tool_artifact / publish_tool_artifact。可信装配
冻结公开 target、grounding/grasp query、参数与 ledger，通过 CAS 引用交给工具；
不传 host/worker/路径权限。校验 identity、成员集合、hash/bytes、边界元数据。
`cache=True` 在 replay 不提供查询缓存；默认读取允许使用当时的候选作执行输入。
动作的可选 preparation BlobRefs 把 ledger/commitment 一起纳入分歧检测。

正式 schema、成员白名单、tool-inputs.jsonl、开关与 CLI 用法见
[toolkit-io](../../../../docs/reference/toolkit-io.md)。
快照协议不变。不存在额外 simulator cursor、任意文件读能力或进程隔离。

## M7 recorded 实验边界

`experiment.RecordedExperiment` 包装严格 Replay/LegacyReplay；同工具/脚本继续顺序消费。
换工具只查询选中父调用首次副作用之前的同语义证据，歧义拒绝，缺失请求用
`missing.py` 生成结构化 NotRecorded。新动作只提议，最多一次；没有在线仿真器能力。
研究侧 check_task 权限覆盖只在该包装器中，原 CallContext 和 runtime 不变。
已发布观测重建由可信 orchestrator 提供；感知仍重新请求，纯算法仍在 lib。
结果、进度、投影及完整限制见 toolkit-io 的 M7 节。

## Architecture v1 classified API

Current tool/lib callers use `assemble.py`'s immutable categories; see
[toolkit-data](../../../../docs/reference/toolkit-data.md) for the nine sources,
permissions and replay availability. The directories `perception`, `state_access`,
`sim_query`, `sim_exec`, `lifecycle`, `privileged`, `artifacts`, `compute`, and
`context` contain those capabilities. Live and replay share implementations
where forwarding semantics are identical; state and planner/action replay have
explicit snapshot implementations. Historical wire adapters above remain for
evidence reads until the phase-four replay command migration. Mutable recording
progress, replay sessions and proposal authority live outside this package.

## Phase-one snapshot transport

`snapshot_services.py` is an external-only transport composed through
`assemble_snapshot`. Simulator values come from verified state publications;
planner always raises SimulatorRequired, actions are refused and episode/context
controls are written only by a supplied replay-output owner. External outputs
are recomputed and compared without consuming a historical simulator cursor.
The completed-step CallContext boundary is `snapshot_step` with a nonnegative
`step`; original pre-parent boundaries remain unchanged. This variant is distinct
from the legacy diagnostic Replay/RecordedExperiment implementation.

统一协议号与重放门禁清单见 [toolkit-protocol](../../../../docs/reference/toolkit-protocol.md)。
