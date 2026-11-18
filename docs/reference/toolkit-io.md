# Toolkit I/O evidence v1

本页定义可选的数据接口记录。默认关闭；旧 run 缺少本目录不影响原评估。实现入口见 [interface README](../../ora/agent/toolkit/interface/README.md)。

## 布局

`native/toolkit-io/calls.jsonl` 是 `ora.toolkit.io-event.v1` 单 writer 事件流；
`native/toolkit-io/blobs/sha256/<digest>` 保存原字节和公开 semantic envelope。
replay 写独立输出根；同一布局的 `comparisons.jsonl` 每行
`ora.toolkit.comparison.v1`，包含 reference（comparison blob）、source_invocation_id
和 evaluation_invocation_id。源 run 不修改。

每次 begin/end 包括 sequence（全文件从 0 起）、invocation_id、identity、attempt、
semantic_request_hash、request、state_before。end 加 status=ok/error/timeout、
response_ref、source、cache_hit、error、state_after。state_after 未独立采样即 null。
未配对 begin 是 incomplete，不当成成功；重复/跳号/重叠/缺 begin 拒绝。
工具可以给调用加 `note`（`interface/call_note.py`），begin 和 end 都带，只作证据：
不进 request，也不进 semantic_request_hash，所以 replay 的匹配不变，环境也看不到它。
目前 `note.route` 记每次关节轨迹移动的路线（`planner.PathPlan.route`；`recover_pose`、
`return_home` 记 `recovery`），没有 note 的调用事件字节不变。
重试同一 invocation，attempt 从 0 起，只对显式配置的外部服务自动重试；动作永不重试。

request = `{identity: <语义身份>, inputs: <带路径的实际引用>}`。
身份包含 schema=ora.toolkit.request.v1、method、method_version=1、parameters（展开
默认值）、inputs（不含 path）、observation（id/hash）、binding、capabilities。
semantic_request_hash 只 hash request.identity。

canonical JSON 为 UTF-8、ensure_ascii=False、sort_keys=True、separators=(",", ":")、
allow_nan=False、无末尾换行。只接受严格有限 JSON；数组顺序保留，int/float/bool
不混同，未知方法参数由显式 Python 签名拒绝。输入身份保留 sha256/bytes/media_type/
array/json_pointer/byte_scope；不包含路径或端点地址。

invocation_id 是 identity 的同一 canonical SHA-256。identity 包含 run_id、round_index、
tool_call_id、interface_ordinal、boundary 和 before/after 逻辑 token。token 为
`{tool_call_id, interface_ordinal, phase}`；不是物理时间。重试不换 identity。

source：segment/segment_points/ground/grasp_candidates 为 external_service；
submit_action/report_issue/finish_episode 为 agent_control；其他为 simulator。

## 公开响应

response_ref 为 BlobRef，指向 `ora.toolkit.response.v1` JSON 原字节：method、status、
data、provenance、error。provenance 有 source、projection（method.v1）、observation、
binding、sources。公开字段投影定义在 projections/services；从不 dump 原 worker 或控制
对象。响应 envelope 的错误只保存公开 code/retryable，不保存异常文本、URL、凭据或堆栈。
记录器自己的故障另存 gaps.jsonl（见下文），不属于模型可见响应。

BlobRef：path、sha256、小写 hex、bytes、media_type；裸数组额外
`array={dtype,shape,order,byteorder}`，禁止 object/structured dtype。NPY 容器与裸数组
编码区分。每次读校验长度和 hash，symlink/parent traversal/根外路径拒绝。

公开路径只允许：toolkit 的 digest blob；policy-observations/<id>/observation.json、
proprioception.json、<camera>/rgb.png、depth-m.npy、calibration.json。路径类别合格之外
还必须由可信装配层依据该边界发布引用注册；不得枚举未来文件。其他 receipt/probe/
policy-state/manifest 由可信装配器投影进 toolkit blob，不整目录开放。

## Replay 与 snapshot.v0

simulator/控制严格按父调用的 ordinal 消费。缺请求报 NotRecorded；动作/控制偏离报
Divergence；第二次动作提案拒绝；绝不在线补调用 worker。submit 返回历史提交确认，
不把后续回执或观测变成调用前可见。官方 check 没记录不使用 final verdict 补齐。

外部服务每次重算。精确匹配当前历史请求后比较并消费；新服务请求可以计算，comparison
unavailable 且不推进游标。内容比对与完整 envelope 审计分开，见下节；原 tool-result、
manifest 或不同投影不是可比裸响应。comparison 与 payload 分开，simulator 的 payload
逐字节不变。

接受 rsi.traj.snapshot.v0，available_at_boundary 与 replay_evidence 分开。
可选 toolkit_io 为 ora.toolkit.snapshot-index.v1：calls 文件引用、snapshot_root（固定
日志副本所在根）、selected_tool_call_id、selected_round_index、next_interface_ordinal=0、
exact_interface_replay=true。原产物按 source run 解析，calls 按 snapshot_root 解析。
日志必须已固定字节、校验 hash；不读取追加中的最新日志替代快照。

before_call 为 K 从 0 起；after_native_action 为 N 从 1 起，但 v1 没有中途接口 cursor，
next_interface_ordinal=null/exact=false，导入接口 replay 报未记录。仅显示 simulator/
controller 数据不等于能够从该物理边界继续原调用。

旧 snapshot 恢复观测与本体感知的原文件；明确标 provenance.reconstruction=legacy_artifact、
exact_interface_bytes=false，不声称存在从未记录的接口 envelope。receipt/check 要有明确
调用/完成绑定。缺 request 的 probe/align/reach 不按相似参数猜。旧服务响应 scope 不同则
只重算、comparison unavailable。没有读取 GT 的能力；check_task 必须显式开启。


## 外部服务的内容比对

四个方法 segment、segment_points、ground、grasp_candidates 各自声明相同的 volatile
JSON pointer：`/data/metadata/request_id`、`/data/metadata/latency_ms`、
`/data/metadata/timings`、`/data/metadata/queue`。只删这些明确路径，绝不递归删同名
字段。后两项为公开投影预留；当前客户端投影未发布的原始字段不会为了比较而导出。
服务/model/configuration identity、随机 seed、候选计数和排序都是内容，不属于 volatile。

comparison 的 status 为 equal / within_tolerance / different / unavailable；前三者 scope
为 semantic_content。equal 表示除 volatile 和存储路径外，字段及数组值相同；
within_tolerance 表示全部差异都在下面的绝对容差内；任何离散变化或超容差即 different。
每次比较仍保留 reference_sha256 / actual_sha256（完整 envelope）、envelope_equal、
reference_content_sha256 / actual_content_sha256（去 volatile 和 BlobRef.path 后的 JSON，
数组保留原 blob hash，因此容差内变化仍有不同 content hash）、volatile_fields。
没有绑定完整服务身份或当前游标没有可比请求则 unavailable。

| 字段 | 默认绝对容差（rtol=0） | 摘要 |
| --- | --- | --- |
| backend_score、model_score、scores | 1e-6 | 最大绝对差 |
| poses_camera 的平移（m）、contact_points_camera（m）、openings_m（m） | 1e-5 | 最大绝对差 |
| poses_camera 的旋转矩阵元素（无量纲） | 1e-5 | 最大绝对差；不是角度容差 |
| mask | 0 不同像素 | different_pixels、pixel_iou（空并集为 1） |
| dtype、shape、存储顺序、计数、rank/source_index/query_index、其余字段 | 精确 | 字段变化/缺失/数量变化 |

`differences` 是逐 JSON pointer 的差异摘要，含 kind、within_tolerance；数值另有
max_absolute_difference、absolute_tolerance、relative_tolerance。大数组从两个 store
校验 hash/长度后读取，绝不把 BlobRef 的路径差异当数值误差。
`candidates` 单独报告数量及排序：SAM3 比较按位的 query_index/source_index/rank；
CGN 报告 pose 行的精确置换，无法证明身份对应时 ordering_changed=null。不会重排候选使结果看起来相同；数值噪声伴随置换时，
仍按位置比较，可能无法证明候选身份的对应关系。

## 开启现有工具的只记录模式

在 RunSpec **顶层**（与 runtime / policy 平级）添加：

```yaml
record_toolkit_io: true
```

缺省或 false 不安装任何包装；RunPlan 仅在 true 时保留此键。它属于 orchestration，
**不放进 runtime 环境配置**，不传给 worker。仅支持 ManipulationToolRuntime 的
native_atomic_causal surface；其他 runtime 开启会在解析时拒绝。程序装配入口为
`build_tool_runtime(..., record_toolkit_io=True)`。

工具 factory 在旧客户端和工具初始化完成后安装 `toolkit/host/recording.py`。
原 invoke dispatch、工具算法、worker callback、动作准入和动作持久化各执行原来的那一次；
返回原 Python 对象，原调用异常原样重抛。不开重试、不补 service 请求、不执行在线 replay。
记录覆盖当前/已发布历史观测选择（含 proprioception 文件引用）、SAM3 三个方法、CGN
propose 的原池、probe/align/reach 回调、submit_action 提交确认、check_task、issue/finish。
CGN 编译/排序结果仍原样给旧工具，接口只投影完整 backend pool。
提交确认不等于执行回执；稍后 worker 执行结果没有在调用前发生，不能在这里补写。

父身份取同 run 的 agent-events.jsonl 真实 tool_call_proposed：call 是全部提案的 0-based
索引，seq/model_call_id/tool_call_id 原样绑定。无上下文的聚合只读批次按提案顺序取首个
未完成的同名调用；显式上下文按 tool_call_id 精确选取。没有可绑定提案即拒绝记录，
不制造父 ID。状态只取此前最后发布观测；state_after 仍 null，不伪造额外采样。
每个外部请求保存实际 RGB/depth/pointcloud 的内容寻址数组引用；历史观测文件复制到 CAS，
当前观测文件仍引用 policy-observations。嵌套 _current 读取属于包住它的公共操作，
不产生重叠 begin/end。旧工具仍保持一次 submit_action 的派发语义。数据接口不再维护独立的次数守卫；
同步 toolkit（v5；分层的 v2 已于 2026-11-01 退役）通过 execute_action 逐次执行，由 Supervisor 的父调用预算与执行账本控制。

此开关是同进程证据仪表，不是工具隔离。记录器 fail-open：上下文、begin、输入 blob、
投影、输出 blob、end 或装配/恢复失败，都不能改变原调用的返回对象或原异常，不能阻止
或重复原调用。特别是 submitted-action.json 已落盘后投影失败，原 action_submitted
仍正常返回。原工具的准入、一次动作语义和拒绝保持原样；capture 不另行执行动作准入。
主动 DataInterface 调用仍严格校验，只有包住旧调用的 capture 使用 fail-open。
记录增加磁盘写入/校验成本，真实 wall-clock/预算可能受影响，
固定输入和时钟的工具结果与动作文件保持一致，不承诺带真实时钟的整个 episode 字节相同。
测试使用假服务与合成观测；真实 general_pickup smoke 由运行会话另行执行。

### 记录缺口：gaps.jsonl

`native/toolkit-io/gaps.jsonl` 是尽力追加的 UTF-8 JSONL。v1 每行是以下**扁平**对象，
不使用嵌套 error 或 identity：

```json
{"schema":"ora.toolkit.gap.v1","run_id":"run-123","tool_call_id":"call-0","method":"submit_action","interface_ordinal":2,"stage":"project","error_type":"ValueError","message":"projection failed","unix_s":1792713600.0}
```

- run_id 为源 run 目录名；tool_call_id 已知时保留真实父 ID，否则 null，不能借用并发调用的 ID。
- method 是接口方法；context/装配阶段尚无接口方法时使用父工具名，初始化失败为 null。
- interface_ordinal 在 capture 内已知时填写，从 0 起；上下文、准备输入等阶段为 null。
- stage 只取 context / begin / project / end / store。store 包含输入/输出 blob I/O；
  project 内的 OSError 也归 store；hash drift 等完整性投影错误归 project。
- error_type 是记录器异常类名，message 是其文本，最多 4096 字符；unix_s 是记录诊断的
  Unix 秒，不是 simulator 时刻。该诊断不进入 tool result 或模型消息，也没有堆栈。

记录故障停止当前父调用的记录，原操作照常执行。context 或输入 store 故障发生在
begin 之前时，不污染 journal；下一个父调用重新尝试记录。已成功写入 begin 后的
project/response store 故障，会尽力追加 `status=error`、`error.code=recording_gap` 的
end 和同码响应 envelope，明确表示记录失败，绝不冒充原操作失败或成功。该 envelope
仅用于日志闭合，不返回给工具/模型；gaps.jsonl 仍以原 stage/error 标记该父调用，
replay 必须拒绝它。成功闭合后，后续父调用正常记录、可独立 replay。

若 begin/end append 结果不确定（短写、flush/fsync 失败等），或 recording_gap 的闭合
也失败，冻结当前 writer，保留原字节，不截断、不重试 append、不补造回执。旧工具
与新工具此后都继续放行调用并写 begin gap，不为某个工具重开 writer 来绕过冻结。
同一个 run runtime 的 Recorder 和迁移工具 Dispatch 共用唯一 Journal，包含相同的
sequence、已写字节数和关闭状态；按既有 RPC 顺序执行父调用。新旧工具交替调用不会
被误判为外部写入。真正的外部追加仍拒绝记录，不刷新游标来掩盖损坏；这不增加多进程
或并发写入支持。这种磁盘/日志损坏不能保证恢复。
初始化失败则保留未包装的 runtime，只有一条父 ID 为 null 的 gap。
并发/递归调用只旁路自己的记录并写 context gap，不阻塞、不拒绝原调用，不借用当前
父身份；外层调用仍可记录。它不赋予原工具额外的并发安全保证。
工具内部自己新建线程时，ContextVar 默认 False，线程内调用既不记录也不留 gap。
当前 tool runtime 没有这种线程；这是已知边界，不据此声称线程内覆盖完整。

lib v0 另经 DataInterface.read_blob(ref) / save_array(value) 读取已授予的公开字节或
保存派生数值数组。它们没有新 simulator/service 调用，不分配 interface_ordinal，
不改变现有 15 方法的请求键与 calls schema。没有 register/path discovery 能力；
read_blob 每次校验授权、长度与 hash。replay 的 save_array 只写 evaluation CAS，
输入的 dtype/shape/order/hash 随后绑定进服务请求。源 run 不修改。

gap 使用单次 append 写入；格式化、创建目录、open/write/close 失败全部吞掉。
磁盘满、短写或进程退出可能使 gap 缺失/不完整，因此没有 gap **不证明记录完整**。
覆盖率需要与 agent-events 中父调用对照；有记录与 gap 并存时优先标 gap。
Replay 打开源 run 时，所选父调用存在 gap、存在无父 ID gap，或 gap 文件损坏/不可读，
均报 NotRecorded（未记录）；其他已知父调用的 gap 不影响当前调用。不会用残缺记录
声称完整接口 replay。只读轨迹展示和覆盖率不执行任何接口。

### 增量索引与离线开销

`toolkit/host/event_index.py` 记住文件身份和字节偏移，只解析新增完整行；
缓存提案身份、完成 ID 和发布事件，不保留无关大 payload。尾行未结束则留待下一次，
截断/替换或身份歧义走 context gap。发布观测仍按父提案之前的 seq 选取。
journal 的已写字节数也递增维护，避免每次 append 重编码全部历史。

2026-10-23 在本机 /tmp、Python 3.11 的离线 fixture：16,788,005 字节已存在日志，
70 次 check_task，假官方结果、真实公开 manifest 读取及 envelope/blob/journal 写入。
安装和构造 fixture 不计时；强制全读一栏只把每次索引重建，用于比较读取策略，
不是对旧 commit 的完整性能复现。一次采样，非服务延迟或集群磁盘性能保证。

| 模式 | 首次 ms | 中位 ms | p95 ms | 均摊 ms | 读日志字节 / 解析行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 开关关闭 | 1.770 | 0.626 | 0.707 | 0.651 | 0 / 0 |
| 增量记录 | 60.817 | 2.647 | 2.835 | 3.487 | 16,788,005 / 72 |
| 每次强制全读 | 54.092 | 63.204 | 70.728 | 64.112 | 1,175,160,350 / 5,040 |

该 fixture 中记录的中位额外耗时约 2.02 ms/调用，包含首次扫描的均摊额外耗时约
2.84 ms/调用。追加 70 条提案的单元测试另验证完整行只读一次，不依赖耗时阈值。
复现（传入新的临时目录）：

```python
from pathlib import Path
from tempfile import mkdtemp
from tests.support.agent.toolkit_recording_benchmark import measure
print(measure(Path(mkdtemp(prefix="toolkit-recording-benchmark-"))))
```

## 阶段事件：`stage()`

工具可以用 `stage(name, **fields)`（`ora/agent/toolkit/interface/stages.py`）记下结果里没有的决定：试过又丢掉的候选、从默认值解出的值、被上限截掉的池。
没有调用方收集时它是空操作（工具调用模式从不收集），所以不改任何工具的结果、账本事件、本页的 begin/end 记录或耗时；字段是纯 JSON 值，一次调用最多 64 个事件，
事件不进工具的结果。code mode（`policy.code_mode.library: extended`）在每次脚本调用外收集它们，和结果、拒绝以及回执一起交给
`ora/agent/toolkit/explain.py`，按工具的 `tools/<name>/explain.json` 渲染成 `robot.explain(r)`（见 [公共接口](public-interfaces.md) 的 code mode 一节）。
事件随该次脚本调用的记录写进 agent trace（嵌套调用 `tool_result` 的 `payload.explain_inputs`，`ora.code-mode-explain-inputs.v1`，见 `ora/agent/code_mode/record.py`），所以 `explain` 在会话重启后也能从记录渲染，不写进上面的 `calls.jsonl`。

现有的事件：`vision_ground`：`query_cache{hit}`；`vision_segment`：`mask_omitted{query_index, reason, backend_score}`；
`geometry_query`：`depth_window{window_px}`；`grasp_plan`：`display_cap{compiled, shown}`、`cut_candidate{candidate_id, backend_rank, reachable}`、`ik_unavailable{failure}`。
模板声明的事件必须和工具源码里的调用一致（测试用 AST 核对）。

## M5：在历史工具调用前换代码试运行

```bash
ora toolkit replay RUN --list --json
ora toolkit replay RUN --at 0 --script ora/agent/toolkit/examples/ground_geometry.py \
  --service-report /path/to/qualified-report.json --output /path/to/new-evaluation --json
ora toolkit replay RUN --at TOOL_CALL_ID --script /path/to/candidate.py \
  --args '{"camera":"head","text":"red cup","max_candidates":3}' --json
```

`RUN` 是源 run 目录。`--at K` 为从 0 起的第 K 次父工具提案**之前**，也可用唯一
`tool_call_id`；数字恰好同时匹配另一个 ID 时拒绝歧义。`--list` 返回
`ora.toolkit.call-list.v1`：`calls[]` 含 `call / tool_call_id / tool / has_toolkit_io /
has_gap`；无父 ID 和未知父 ID 的 gap 数分别为 `unbound_gaps / unknown_parent_gaps`。
记录与 gap 可同时为 true，执行时 gap 优先拒绝；无父 ID 的 gap 保守拒绝所有调用。
没有 gap 不能推断全覆盖。`--list` 不创建输出、不请求服务。

脚本须定义 `run(api, args)`，返回严格 JSON 值（禁止 NaN、numpy 对象等）；`api` 是
所选边界的 Replay 或 LegacyReplay，能使用只读 lib。`args` 默认是父工具的原始
arguments；`--args` 显式替换脚本输入，但不会改 CallContext、历史请求或游标。
`--tool NAME` 支持 vision_ground、grasp_plan、grasp_execute、move_pose、move_to、
compute、check_task、finish_episode、observe；也可选该 run 冻结工具面中的其他已迁移工具。
换工具必须传 `--args JSON`。同工具的不同动作/准备文件仍会报 Divergence；换工具只提议动作。示例源码见
[ground_geometry.py](../../ora/agent/toolkit/examples/ground_geometry.py) 及
[examples README](../../ora/agent/toolkit/examples/README.md)。

有 toolkit-io 的调用使用记录中的 run/round/parent/boundary/state/check_task 能力，
核对 agent-events 的父提案，再从 interface ordinal 0 开始逐项消费。只读地把
calls.jsonl 的固定字节复制到输出 `source-calls.jsonl`，后续源日志追加不改变本次游标。
不会搜索“某处同参的第一条”、跨过未消费的调用，或把后续仿真结果贴到新动作之后。
仿真响应保持记录的 envelope 字节；新仿真查询报 `NotRecorded`；不同动作报
`Divergence`；重复提交仍拒绝；有 gap、未完成接口调用或证据漂移均拒绝。
录下的 CallContext 原样保留。M7 在 replay 装配层单独开放 check_task 并标记能力覆盖，
仍须有该边界的判定记录，不倒灌最终判定；见下面的 M7 规则。

没有选中调用的 toolkit-io 时，使用 `interface/legacy.py`。RUN 若只有显式
`snapshot.json`（snapshot.v0，before_call），就按其唯一边界读取；普通旧 run 则只
选取提案之前最近的 `fresh_observation_published`，注册 manifest 声明的公开文件。
不从调用后的新观测、最终判定、未绑定控制日志猜测数据。能恢复的观测/本体感知保留
原始文件字节；缺失文件、探测和未绑定回执报“未记录”。旧工具输出与裸服务 envelope
范围不同，不能声明 equal；legacy 感知重算后写 `unavailable` comparison，reason 为
`legacy_response_scope_not_equivalent`。已提供的 snapshot 显式绑定旧回执时，仍沿用
legacy importer 的严格路径、哈希和动作 ID 核对。

带 `--tool` 时还要有 `native/toolkit-io/tool-inputs.jsonl`。没有这份记录的旧 run 报“未记录”：
此前 vision_ground、compute、check_task 用旧混合工具面近似重建配置，该工具面已于 2026-11-01 退役，
这条路径随之移除。`ora replay --check` 不受影响。
录制回放的工具状态：RoboDojo 的 run 仍用空的只读 store，读 `tool.marks` 报“未记录”；
工具上下文带 `motion.environment_plans` 的 run（环境规划动作的工具面）从
`state/tool-deltas.jsonl` 取该调用之前的 `tool.*`，写在私有副本上，没有日志报“未记录”
（[toolkit-execution](toolkit-execution.md)）。

感知必须显式传 `--service-report`，格式与 `ora spec robodojo compose --service-report`
相同：内容寻址文件名、`ora.perception-readiness.v1`、24 小时时效、协议版本、通过项及
`ORA_SAM_PORT / ORA_GRASP_PORT`（兼容既有服务环境变量）的端点匹配规则均沿用。
CLI 不启动服务、不分配资源、不读取 provider key。客户端在实际请求时核对 live
identity。SAM3 使用现有默认客户端；CGN 使用默认 RoboDojo grasp_plan 配置
（hover_offset_m=0.04、32 raw candidates、1 forward pass），接口仅公开完整 canonical
候选池。非默认旧配置、不同 report/service identity、不同输入编码/参数或不匹配游标
会给 unavailable，不能伪装成同一服务。不给 report 时观察/纯计算仍能运行，首次请求
感知明确抛 `ServiceUnavailable: service unavailable`。每个重算请求的响应和 comparison
均落盘，比较口径仍是上文的 content 容差与完整 envelope 审计 SHA。

`--output` 必须是尚不存在、与源 run 互不包含的目录；默认是
`<workspace>/toolkit-replay/<run_id>/<UTC时间戳>-<8位随机>/`，toolkit-replay 与 runs/
（及 ground-truth/）平级，不干扰 run 扫描。非 runs/ 布局的独立样本以源目录的父目录
作为 workspace，也不在源样本内写入。
正常退出和脚本异常均打印输出路径（`--json` 的 `output` 字段），执行成功返回 0，
失败返回 2；选择参数/输出路径无效时可能尚未创建结果目录。输出含：

| 路径 | 内容 |
| --- | --- |
| `result.json` | `ora.toolkit.replay-result.v1`；status、source_run、output、selected、arguments、script path/SHA、lib version/SHA、context、mode、result 或 error.type/message、时间；exact 模式另含 source_calls 引用和成功后的 next_interface_ordinal |
| `native/toolkit-io/blobs/sha256/*` | 派生输入、重算响应、比较产物的 evaluation CAS；按需创建 |
| `native/toolkit-io/calls.jsonl` | 本次重算的接口请求/响应事件（没有外部请求时可以不存在） |
| `native/toolkit-io/comparisons.jsonl` | 始终创建；每行仍为 ora.toolkit.comparison.v1，reference 指向 comparison JSON，source_invocation_id 可为 null |
| `source-calls.jsonl` | exact 模式使用的源 calls 固定字节副本；其 hash/bytes 记入 result.json |
| `script.log` | 脚本导入和运行时的 stdout/stderr，避免污染 CLI JSON |

这是同一 Python 进程中的**可信用户脚本执行**，还没有沙箱或进程隔离。CLI 和数据接口
不写源 run、不提供在线动作能力；使用者的任意 Python 本身仍有普通进程权限，须遵守
源数据只读约定。不是物理状态续跑，也不支持 native-action 中途边界的接口 replay。

CPU 验证使用 `tests/data/toolkit-general-pickup` 的原始公开 head RGB/depth/calibration，
按快照已有 SHA 核对后补入；假感知返回三个候选，再完成 mask_to_cloud / principal_geometry。
示例结果含 `geometries[].{points,frame,center_m,extents_m,axes,ambiguous_axes}` 和
`comparison.status="unavailable"`（旧响应范围不足）。合成 exact 记录另验证相同内容
得到 equal、原仿真字节保持、同参多次调用按次序消费、动作分歧、gap 和新查询拒绝；
测试不连接真实服务，不把假服务输出当作原 run 的历史结果。


## M4 / M4b：可切换的九个原子工具

当时 RunSpec 顶层的 `use_toolkit_tools` 在原混合工具面
`robodojo_placement_closed_skills_v2` 上切换九个 handler；该开关和工具面已于 2026-11-01 退役，toolkit-only 工具面是现在的唯一入口。当时新增的
`robodojo_placement_toolkit_v1` 只公开这九个工具，自动启用 toolkit
且拒绝旧工具派发（该工具面已于 2026-10-31 退役，封存的 run 保留其记录的定义）；其它不支持的 surface 显式拒绝。coordinator 保留原
schema、说明、参数门禁、错误投影和单动作提交规则，替换上列九个 handler。
finish 的门禁在新工具中按相同顺序执行，官方判定只经 check_task。工具代码只调用 lib/interface/其他工具，
排序、选择、抬升和提交策略留在工具目录。lib 0.3.0 新增参数化抓取评审渲染、
可见表面几何与深度质量测量。所有渲染输入来自公开 RGBD 和候选数据。

新工具的数据接口调用始终写 toolkit-io；`record_toolkit_io` 仍默认关闭，只决定
其余旧 handler 的记录。两个开关同时开启不会重复记录这些新工具。记录磁盘、
journal、上下文定位失败继续按 gaps.jsonl 放行；派生字节由当前调用内存持有，
记录写失败不会让已经成功的计算或动作提交变成模型错误。兼容工具产物本身的
发布失败仍是工具错误，与记录写入失败分开处理。其它工具、worker 和执行器不变。

```bash
ora toolkit replay RUN --at 0 --tool vision_ground --service-report REPORT --json
ora toolkit replay RUN --at PLAN_CALL_ID --tool grasp_plan --service-report REPORT --json
ora toolkit replay RUN --at EXECUTE_CALL_ID --tool grasp_execute --json
```

`native/toolkit-io/tool-inputs.jsonl` 每个父调用一行，schema 为
`ora.toolkit.tool-inputs.v1`，顶层字段：

| 字段 | 语义 |
| --- | --- |
| `tool_call_id` | 与 agent-events/calls/gaps 相同的父调用 ID |
| `context` | 完整 CallContext，必须与所选 calls 的 context 相同 |
| `configuration` | 公开 surface/算法参数、编译帧与单位、目标/查询引用列表、公开技能 ledger/提交门禁快照；无 worker 对象或真值 |
| `inputs` | `{ "kind:key": {"files": {"member": BlobRef}, "metadata": {...}} }`；不可用的已有产物用 `{"error": <原 ToolError 公开字段>}`，仅查到该产物才拒绝 |

可信装配把调用前的公开 target/query 成员复制到 CAS 后写此表；旧 ledger 的
reconcile 仅在原参数门禁通过后执行，之后冻结公开准备状态。不会从回放时的最终
artifact 目录推断历史输入。各 BlobRef 仍只落在原 CAS/公开观测白名单。表不是
simulator 调用，不占 interface ordinal。缺失/重复 input-index 或边界不匹配不能
用于完整新工具回放。该表可由读取器作为可选附加文件保留；snapshot.v0、calls、
gaps、两种请求键及 native-action 中途不提供接口游标的规则都不变。

接口增加 `tool_context()`、`tool_artifact(*, kind, key, cache=False)`、
`publish_tool_artifact(*, kind, key, files, metadata)`，只收逻辑 ID 和固定成员名：

| kind | ID | 允许成员 |
| --- | --- | --- |
| target | target-24hex | manifest.json、mask.npy、overlay.png |
| grounding_query | grounding-query-24hex | manifest.json、overlay.png、candidate-T0/T1/T2.png |
| grasp_query | grasp-query-24hex | manifest.json、overlay.png、input-pointcloud.npz、review-pointcloud.npz、global-review.png、backend-output.npz |
| skill_preparation | skill-24hex | manifest.json、ledger.json、commitment.json |

manifest 的 schema、identity 哈希、成员集合、成员 hash/bytes 和观测/相机元数据均校验。
兼容 target/query 目录先完整暂存再原子 rename 发布；重复发布校验全部原字节。
`cache=True` 在 replay 的查询种类上总返回 None，迫使感知重新请求；默认查找允许
下一工具读取历史候选作为输入。skill_preparation 只在 evaluation/live CAS 保存；
提交适配器校验动作和准备内容后，按原编码写 ledger/commitment，最后发布动作。
`submit_action(*, action, preparation=None)` 的可选 preparation 为
`{"skill_ledger": BlobRef, "skill_commitment": BlobRef}`，进入 request.inputs 和
语义 hash；无此参数保留旧请求格式。故同一 action JSON 配不同准备内容也报分歧，
回放不能写控制文件或接通在线执行器。

CLI 为有索引的 --tool 和 --script 绑定同一 ToolData，冻结输入表到输出
`source-tool-inputs.json`，result.tool_inputs 含 mode/path/sha256/bytes。
旧 run 可用原 public observation 重算 vision_ground（无服务时报 ServiceUnavailable）；
其历史配置来自明确绑定或注明默认 surface 假设。缺少调用边界 target/query/ledger
输入的旧 grasp_plan/grasp_execute 报 NotRecorded，不扫描未来产物来凑输入。

固定数据测试覆盖左右臂、空/非空 CGN 池、三工具结果与全部 query 图像/NPZ/manifest、
执行 action/ledger/commitment 原字节、目标/缓存错误的 RPC 投影、记录写失败、
两开关组合、三工具 exact replay、感知 comparison 和动作分歧。使用假服务和
公开 fixture，不构成真实感知质量或机器人成功结论。静态 import/I/O 检查是首道
边界检查；工具和可信脚本仍同进程，没有权限或进程隔离。

### M4b：剩余原子工具与边界补充

```bash
ora toolkit replay RUN --at MOVE_CALL --tool move_pose --json
ora toolkit replay RUN --at COMPUTE_CALL --tool compute --json
ora toolkit replay RUN --at OBSERVE_CALL --tool observe --args '{"step": 0, "views": ["head"]}' --json
ora toolkit replay RUN --at CHECK_CALL --tool check_task --json
ora toolkit replay RUN --at FINISH_CALL --tool finish_episode --json
```

move_to 同样支持。
动作保持一次父调用最多一个 proposal。记录打开与否均不改变原工具结果/schema。
本轮未迁移 mark_point、geometry_query、align_plan、gripper/rotation/recovery 等其它入口。

- `evaluate_numeric(*, source, bindings, outputs=None)` 是纯本地计算能力：可信装配
  注入原受限数值执行器，保留 AST 白名单和子进程资源限制。它不属于 simulator、
  external_service 或 agent_control 记录方法，不消费 ordinal。输入与输出均为
  JSON 数值；无任意 Python/file/network 权限。纯 compute 的 input-index 可有
  零个 calls；replay 严格验证 context 对应父 proposal 后创建空游标，仍拒绝 gap。
- `check_task` 响应可附 `data.verdict`：仅显式的 `task_success`、`terminal`（缺失不补），
  用于 finish 的锁定/终局规则。缺文件时 verdict={}，原 success/reason/retryable 保留。
  check_task 工具的模型结果不包含 verdict。此数据只在开启 check_task 能力的方法
  响应中记录；tool_context/input-index 不含官方判定。finish 在无该能力的官方判定
  context-pack 模式不查询它；旧记录没有所需 verdict 元数据则 NotRecorded。
- observe 的 `observation` 响应可附 `camera_errors`（相机名→原公开错误投影）和
  `public_robot_facts_by_view`（相机名→`{"value": <公开 facts>}` 或 `{"error": ...}`）。
  二者在读该 publication 时生成并记录，不改变 manifest 原字节。历史查询的请求
  parameters.observation 指向已发布旧观测；request.identity.observation 仍是父调用
  边界当前观测，不能据此把过去/未来状态混成一个时刻。输入索引 observe.history
  只授予当时已发布且通过公开状态链验证的本轮观测身份。
- 新 artifact kind `observation_overlay`：ID `overlay-24hex`，只允许 manifest.json、
  image.png。key 对原渲染 identity 的 JSON 编码求 SHA 前 24 位，manifest 额外绑定
  观测、相机、图像 SHA。实时适配器只发布既有 overlay 固定位置，重复产物比较
  原字节；replay 只写 evaluation CAS，不增加源 run BlobRef 路径白名单。
- motion 的 advisory IK 失败仅记录公开 `error.reference_failure.exception_type`；
  不序列化 worker 异常或消息。已记录失败仍生成原 unavailable 引用；缺失/变更
  查询、损坏证据、动作分歧不作为 advisory 错误吞掉。

lib 0.4.0 新增 pose、clearance、images、numeric；各模块 docstring 给出单位与 replay
类别。工具仍同进程；原 compute kernel 的受限子进程不代表所有工具已隔离。


### M6：实时观测来源与故障范围

`control/current-observation.json.source_root` 在线运行时指向可变的
`sandbox/workspace/inputs/current`。这是旧工具的输入挂载，不是 toolkit BlobRef 的
公开来源。可信装配 `toolkit/host/publication.py` 统一按控制指针/已发布状态的
**完整 observation_id** 定位 `policy-observations/<observation_id>/observation.json`，
验证 `control.observation_sha256`（历史选择用已发布 state 中的 hash）与 manifest
自身 observation_id；相机文件继续逐个检查 manifest 声明的 sha256/bytes。原始
manifest/proprioception/相机引用只来自此目录，不增加 sandbox allowlist、不用未来
观测替代。record-only、新工具 observation、observe 和 closed-context 都使用此解析。

发布顺序已由代码核实：`episode.observation.publish_observation` 先通过 runtime.observe
在 policy-observations 写出观测（技能路径也先完成 captured observation），再发布
policy state；`context.observation.projector.publish` 从该源复制 sandbox 输入，最后
写 current-observation 指针。fresh_observation_published 和父工具提案在此之后。因此
正常调用时不可变源已经存在，不需要等待异步复制。若源缺失、被改写或为路径别名，
record-only 保留原调用结果并记 gap；新工具拒绝缺失/损坏证据，不退回可变 sandbox，
也不请求模拟器重新观测。closed-context 的计算/finish 仍遵循已有上下文故障放行规则。

CPU 测试包含 sandbox 指针 + 相同不可变发布副本，旧工具只记录与新工具的
vision_ground/grasp_plan 结果相同、begin/end 完整；另含 observe、closed-context、
源缺失/哈希漂移/符号链接拒绝，以及单次 context/project/store 故障后的下一父调用
成功记录并精确 replay。测试相机补全与服务均为显式合成。真实完成 run 的四个
publication、各十个文件引用另做只读核验；没有执行仿真或请求真实服务。

## M7 recorded 工作台：任意工具、判定覆盖、进度和投影

```bash
ora toolkit replay RUN --at 3 --tool compute --args '{"expressions":"x = 6 * 7","outputs":["x"]}' --json
ora toolkit replay RUN --at 3 --tool observe --args '{"views":["head"]}' --json
ora toolkit replay RUN --at 3 --tool check_task --args '{}' --json
```

已迁移的九个工具沿用上面的清单。不同工具必须有显式 `--args`，且出现在源 run 冻结
的 `policy_composition.tool_surface`（或 compiled-policy composition）中，并受该局
`policy.tool_allowlist` 限制；check_task 是唯一例外。没有冻结定义时拒绝猜测工具面。
原工具仍兼容早期 fixture。不会执行轨迹中携带的任意代码。

同名工具和 `--script` 保持顺序消费接口的严格语义；同名工具即使参数改变也不能提交
不同动作。换工具使用 `replay_semantics=alternative_tool`，没有原工具的接口游标
（`next_interface_ordinal=null`），只检索选中父调用首次副作用之前的同一语义请求。
重复请求的响应 hash/status 不同就报 AmbiguousRecord；不取第一条，不读取其他父调用
的探测结果，不把动作之后的判定当作调用前数据。缺少的当前观测可以从该边界已发布的
immutable manifest/files 重建，标 `exact_interface_bytes=false`；原始文件字节不变。
历史 observe 只用边界之前明确发布、hash 绑定的 policy-state。纯公开事实投影可重算。
缺少的 motion/grasp/execution/lifecycle 配置和 point ledger 不从最终目录猜测。

不同工具调用 submit_action 只返回 proposal-only 回执，结果的 `proposed_actions[]`
记录 action、request、semantic_request_hash、executed=false、compared=false；每次仍
最多一个动作。report_issue/finish_episode 的提议在 `proposed_controls[]`，也不修改
源局。感知始终重新请求，只有同一服务身份和语义请求、无歧义的历史响应才用作比较；
没有对应记录也可以重算。没有任何在线仿真器、动作或判定回调。

NotRecorded 的 `missing_requests[]` 每项包含：

```json
{"kind":"interface","method":"reach_check","semantic_request_hash":"<sha256>",
 "request":{"identity":{"method":"reach_check","parameters":{},"inputs":{}},"inputs":{}},
 "reason":"no_matching_request_at_selected_boundary","boundary":{"kind":"before_call"}}
```

上例 identity 省略了原协议的服务/观测/能力字段；实际保存完整请求，hash 仍仅取 canonical
identity，输入 refs 与去 path 的 identity 分开。缺配置/产物用 `kind=tool_input`、
`ora.toolkit.input-request.v1`，method 为 tool_context/tool_artifact；选边界失败用
`kind=boundary`、method=replay_boundary。失败时 `missing_requests_complete=false`：
列出实际走到的缺失请求，不声称已知未执行分支会需要什么。成功时为 true，范围仍只是
已执行路径（脚本可自行捕获 NotRecorded，已观察到的缺失请求仍保留）。

check_task 在 replay 总是获得研究侧调用权限，即使原局策略禁用了能力或没有该工具。
`capability_override={check_task:true, original_policy_available:false, reason:"原局策略不可用"}`
明确标记这项权限覆盖；否则 check_task=false、reason=null。这是 replay 授权信息，不
表示工具已调用它。原 context/check_task 及源日志不改，默认运行时权限不变。仍只能返回
该调用前边界实际记录的官方判定；没有就 NotRecorded，绝不读 final/control 判定补值。

`result.json.mode` 现在是执行模式 `recorded`（live 待 M7.4）；旧的记录来源分类移到
`evidence_mode=recorded|legacy|null`。源 calls、snapshot 和接口日志 schema 没有改变。
输出新增 `progress.jsonl`，每行均即时 flush，形状严格为
`{t:<Unix秒>,stage:<名称>,done:<整数或null>,total:<整数或null>,message:<文字>}`。
阶段依次 resolve、boot_worker、replay_controls、verify_state、run_tool、compare、done。
recorded 的三个仿真阶段明确 skipped，done=total=0，耗时为 0；未走到的失败阶段不伪造
事件。`timing` 含全部七阶段的单调时钟秒数。外部请求及接口内容比较在 run_tool 内同步
进行，该耗时计入 run_tool；compare 计结果收集和下面的模型投影。done 是即时终态事件。
解析失败也写 result/progress（源路径或输出目录本身无效时尚不能写）。

`--tool` 成功时新增 `model_visible`，直接复用实时 Pi 的 `_project`、causal projection
及 `apply_hints`；不修改模型投影函数，不用 replay 自己的单位转换规则。依赖源 run 的
冻结 surface 和 policy.context_composition，缺少时 status=unavailable，不猜默认值。
status=ok 时字段为 text（精确 UTF-8 文本）、sha256、bytes、artifact_ref（输出内原字节
.txt 文件）、stored_projection（输出内 canonical JSON 引用）、images 和 image_text。
输出图片也只写 replay 的 `native/agent-artifacts/`，原 run 完全只读。

同工具还返回 `model_visible.comparison`。原工具结果引用从该父调用唯一的 tool_result
事件取，校验 path、hash、bytes：

- `artifact.byte_equal/content_equal`：本次 stored_projection 与原存档字节/去易变字段
  后的内容比；原存档在 native causal 模式比模型文本丰富。
- `visible.byte_equal/content_equal`：本次文本与从原存档经过 causal 投影得到的文本比；
  **后者不是 provider transcript**，不承诺重建历史压缩或重复错误的会话状态。
- `original_artifact`、`original_visible_sha256`、`original_visible_scope` 保留比较来源。

字节 SHA 不剔除任何字段。内容比较按实际感知结果结构，只删显式 JSON 路径：
vision_ground 的 `/diagnostics/{request_id,latency_ms,timings,queue}`；
grasp_plan 的 `/diagnostics/{request_id,latency_ms,timings,queue}` 及
`/diagnostics/learned/{request_id,latency_ms,timings,queue}`。花括号表示四条路径，
实际展开清单也放在每项比较的 volatile_paths 中。不递归删同名字段，
不改 compute 用户输出，不设数值容差（感知接口容差仍由前文 comparison 负责）。
缺历史结果时 comparison.status=unavailable，不声称相同。脚本结果不假定属于某工具，
不自动做模型投影。当前仍是同进程可信脚本；live 复原与动作执行见下节。

### M7.4：live 诊断复原

调用方先分配 lane，再在相应执行主机使用：

```bash
ora toolkit replay RUN --at 6 --tool move_to --args '{"targets":{"right":{"position_m":[0.2,0.0,0.9]}}}' --live --resource-profile PROFILE --json
ora toolkit replay RUN --at 8 --tool check_task --args '{}' --live --resource-profile PROFILE --json
ora toolkit replay RUN --at 4 --tool grasp_execute --live --resource-profile PROFILE --execute-action --json
```

同步工具（toolkit v4 起）要动作回执才能继续，动作在工具内执行，同样要 `--execute-action`；
没有这个标志时工具停在第一个动作，只留下提案。`--live` 不复原工具状态：`tool.marks` 和
`tool.held.*` 从空值开始，`result.json.execution` 带 `objects_before`，`state.objects`
是动作结束时的私有物体位姿，只给开发者看。`ora replay RUN --incident N` 在这条路径上
找到动作 N 的父调用并汇总结果（[incident replay](configuration-and-evidence.md#incident-replay)），
并让工具跑在默认 baseline 的工具面、规划器和时序上，不跑 plan 钉住的那一套；单独用 `--live` 时
仍是 run 自己的。run 的 `runtime.motion_timing` 由 `live-worker/preflight.json` 交给 worker，
没有这份记录时 worker 退回 `motion_pace` 的时序。

`--service-report` 仍单独指定感知服务；不提供就没有感知能力。`--live` 必须显式给
`--resource-profile`；`--execute-action` 只适用于 live。ORA 不代替调用方申请 router
资源，但诊断进程仍按现有 ORA 协议 claim/release 所选 slot 的 lease 和 scratch。
退出、异常、超时和 SIGTERM 都经过 worker 关闭与释放路径。源 run 始终只读；默认
输出仍在 workspace/toolkit-replay/run_id/UTC-random，不能放入源 run。

实现是独立的 toolkit 诊断入口 `environment.backends.robodojo.toolkit_worker`，只由
backend registry 的 `toolkit_live` hook 启动。它复用原 RobodojoSession、control_replay、
RPC dispatcher 和 action_executor，未改 policy worker、worker kernel 或已认证路径。
`live-worker/launch.json` 分别记录原 worker kernel、当前提交及诊断入口/恢复/host
适配文件哈希。live 要求源局与当前 worker kernel 完全相同，并保留源 revision、layout、
actual_source_fingerprint 检查；不能把已认证 kernel 的结论当作新增诊断入口已真机验收。

恢复次序如下：

1. `live-boundary.json` 固定调用前最后一次明确发布的 state/observation、只读观测
   physics clock、原生回执与 controller JSON/JSONL 前缀。未发布动作中间、含 gap、
   不唯一时钟、漂移引用均拒绝。不用时间戳猜测边界，不读源局最终可变 control 文件。
2. `boot_worker` 启动 Kit；`replay_controls` 只逐个执行记录的原生 controls，done/total
   为 native tick 数（每 tick 10 个 physics steps），不重新规划原动作。回放日志在
   live-worker/environment/restored-prefix/；复制的源前缀在 source-native/。
3. `verification.json` 核对逐 physics step 的机器人测量、施加目标、时钟/计数，以及
   边界 round、physics clock、sim_steps、native count、完整公开 proprioception。
   有单独 GT capture 时还逐值比 rigid object position/orientation；GT 只给 verifier，
   不注册进 BlobStore、lib 或工具。记录声明有 GT 但边界缺失/损坏时拒绝。比较不设容差；
   差异报 Divergence，不能继续跑工具。
4. M7.4b 的放行门槛只看物理核对，保持零容差。重新渲染的传感器只用于统计，
   不作为工具授权条件；`sensor-verification.json`（`ora.toolkit.live-sensors.v2`）
   同时嵌入 `result.sensor_report`。工具始终使用源 run 当时发布的原字节，保留原
   manifest/mask/candidate 身份；新捕获原件单独保存在 `live-worker/observations/`，
   绝不把新像素绑定到旧 observation 身份。捕获失败也只记传感器诊断错误；捕获后
   再核对 round、物理时钟、sim_steps、native_actions、proprioception，变化仍拒绝。
5. 最后才授予工具能力。新的 reach/probe/align 请求走复原 worker；check_task 读该
   worker 的官方终止状态，继续标 capability_override。感知重新请求并保留 comparison。
   原生回执和已授予的公开工具输入保留记录身份；无法从历史恢复的工具私有 ledger、
   candidate/configuration 仍明确 NotRecorded，不能用真值或未来缓存补齐。

工具调用期间 submit_action 只在独立输出中验证并提议，最多一次，不立即执行。
`--execute-action` 在工具成功返回后，通过原 action_executor 执行一次 motion 或
完整 grasp skill sequence；不续跑 policy。`result.execution` 包括 executed、receipt、
state、observation、check_task。执行报错时仍尽力读取执行后状态；执行效果不确定则
`executed=null`，附 error，不能宣称“未执行”。没有动作则 reason=no_action_proposed；
没有开执行开关则 reason=execute_action_not_requested。finish/report_issue 只记诊断
提议，不终止或修改源局。脚本仍可信、同进程；这不是安全沙箱。

live 的 `mode=live`、`replay_semantics=restored_runtime`；`evidence_mode` 仍说明输入
是 recorded 还是 legacy。`verification` 和 `sensor_verification` 是相对输出路径。
已有 progress/timing、模型可见投影和原结果比较字段保持兼容。真实硬件确定性、渲染
是否逐字节匹配和新增诊断入口的完整运行由独立 lane 验收；CPU 假 worker 测试不替代它。

### M7.4b：传感器统计与渲染历史

传感器报告的 `blocking=false`、`gate=physical_only`；`passed` 和
`sensor_byte_equal` 只描述传感器原字节相等，**不代表物理授权结果**。不相等时
`status=not_byte_equal`、`warning="未逐字节复现的传感器"`；捕获失败为 unavailable。
`tool_observation_source=recorded_publication` 和 `fresh_capture_used_for_tools=false`
明确本次工具输入来源。逐路 `files[]` 保存原/新 hash、byte_equal 与下列 metrics：

- RGB：按三个 8-bit 通道的均方误差计算 PSNR；完全相同用 psnr_db=null、
  psnr_infinite=true（合法 JSON）。fraction_pixels_gt_16 是任一通道绝对差 >16 的
  像素比例；附像素数、最大绝对差和 shape mismatch。
- 深度：有限值配对的绝对米差 p50/p90/p95/p99/max；另列非有限值和 finite-mask
  变化数，不用过滤后的分位数掩盖无效像素。
- 标定：按 JSON pointer 给数字/数组字段的最大绝对差、变化字段；原/新
  renderer reference_time（帧计数分子/分母）和 simulation_time_s 原样保留。

复原快进从 `native/render-synchronization.jsonl` 定位原发布观测的 calibration
reference_time，截取其之前的同步序列；同一物理时钟的多次同步都保留。序列包括
执行截帧的 get_obs，不只是 policy 发布观测。新增诊断模块 toolkit_render 在现有
execution_capture_callback 位置逐物理步执行 get_obs（不推进物理），没有改 native_tick、
control_replay 或 worker kernel。最终选中观测对应的同步由 verify_state 阶段捕获完成，
不重复一次。无法唯一定位旧记录时明确 unavailable，不猜补帧数；这也不放松物理检查。
每个复原的 round 在第一个控制之前，若历史在该位置没有同步，就在 round 起点补渲染一次（`round_start_renders`）：reset 初始化腕部相机位姿后的第一次渲染决定相机相对 link6 的安装变换；历史 unavailable（源运行没有 `native/render-synchronization.jsonl`，或发布观测的渲染引用选不出唯一同步）时若第一次渲染发生在已移动的边界，腕部相机会停在 reset 视角并带着错误偏移跟随 link6（2026-10-29 修复）。

`live-worker/environment/render-history-replay.json` 和 sensor_report.render_history_replay
保存调度数、消费数、每次位置/耗时/错误和未到达位置；具体 worker 路径以
sensor_report 的内容及 live 输出目录为准。render_history 保存原始调度证据。
原日志中的 recorded_render_calls 是原同步内的 render 调用数；重放调用原 get_obs
同步逻辑，实际 render 次数仍由该逻辑收敛决定，记录在新 native/render-synchronization.jsonl。
不强制补齐最终计数，也不承诺 RTX 时间累积逐字节确定。中途同步耗时计入
replay_controls；最终捕获/统计计入 verify_state。物理时钟被渲染推进会立即拒绝。

M7.4b 不实现协议版本兼容策略；后续独立里程碑 M7.6 将引入编号 protocol_version，
在公开 run 元数据（包括未开 toolkit-io 的局）与接口日志中写入，版本不匹配只告警，
物理复原不一致仍拒绝。内容 hash 保留作证据；具体字段和支持范围随 M7.6 交审。


### M7.4b 修复：冻结工具定义与诊断 runtime 的差异

源 `plan.json.policy_composition.tool_surface` 是模型实际获知的 advertised surface，
可能已叠加上下文 artifact reader、observe(ref=...)、allowlist、工作空间描述；
诊断 host 的 surface 则是当前实现用于执行的基础工具面。二者在同一源码版本也可能
不同，不再把逐字段不等判为 CorruptRecord，也不将当前定义覆盖到源记录中。

`surface-compatibility.json`（`ora.toolkit.surface-compatibility.v1`）记录 source/current
的角色、profile_id、比较内容 sha256，source 的 advertised_composition_sha256，
以及 `differences[]`（JSON pointer、source/current 值、两侧 present 标记）、affected_tools。
`blocking=false`；有差异时 `result.warnings[]` 添加 `code=tool_surface_mismatch` 和
证据路径，`result.surface_compatibility` 指向此文件。工具继续使用当前代码、已冻结
配置及公开输入；工具许可仍由源工具面与 check_task override 规则决定，没有扩大授权。

源声明的 advertised_composition_sha256 与源内容不符、定义形状损坏、重复工具名仍为
CorruptRecord；其他 BlobRef 完整性与物理核对不变。来源版本/投影变化与存档内容损坏
是两种检查，不用当前 runtime hash 代替源存档的完整性校验。未声明该旧字段的 fixture
不凭空补 hash；比较结果仍保存本次计算的证据 hash。

这里只修复 surface 全等造成的阻断；worker/source 指纹等其余版本兼容策略仍留 M7.6。
用户确认顺序：本修复真卡跑通、合入 → 纯迁移工具面 → W1/W2 → 热会话 → M7.6。

## Phase-one completed-snapshot replay (P1-S4)

These options select a separate implementation from the legacy M7 diagnostic
path. Existing plain `--at`, including `--live --execute-action`, retains its
previous behavior. The snapshot path never executes tool actions or answers a
planner query from historical results.

```bash
# Before a parent call: completed snapshot plus tool-state deltas.
ora toolkit replay RUN --snapshot --at 3 --script trial.py --args '{}' --json

# A completed publication, with no implied parent tool arguments.
ora toolkit replay RUN --step 2 --script trial.py --args '{}' --json
ora toolkit replay RUN --step 2 --tool compute --args '{"expressions":"x = 6 * 7","outputs":["x"]}' --json
ora toolkit replay RUN --step 2 --tool check_task --args '{}' --json

# Explicit fresh perception, using original snapshot pixels.
ora toolkit replay RUN --snapshot --at 0 --script ora/agent/toolkit/examples/ground_geometry.py \
  --args '{"camera":"head","text":"cup","max_candidates":1}' --service-report SERVICE_REPORT --json
```

`--at`, `--step` and `--native-action` are mutually exclusive. `--step` and
`--native-action` imply `--snapshot`. Snapshot options reject legacy `--live`
and `--execute-action`; only native-action restoration accepts/requires an
already allocated `--resource-profile`. Use plain `--at` for legacy diagnostics.

The supplied `api` is the classified RSI facade:

| Capability | Snapshot behavior |
| --- | --- |
| state / privileged | Read verified completed state and original public file bytes. Missing keys/verdicts report NotRecorded; no final-state fallback. |
| state.write | Only tool.* in a private in-memory overlay; never source state or its delta journal. |
| perception | Request explicitly supplied services again; comparisons are separate output evidence. No report means service unavailable. |
| compute / lib | Recompute bounded numeric work and pure geometry. |
| planner | Always SimulatorRequired, even if a similar query exists in the source journal. |
| act | Always refuses; no proposal is executed or queued. |
| episode | Append intent to output/episode-controls.jsonl, executed=false; no live round or run is modified. |
| context | Append attachment declarations to output/context-attachments.jsonl; no live model delivery. |

`--snapshot --at` verifies the selected publication's full state chain and
replays tool-state writes only up to that parent. It uses only that exact
parent's frozen tool-input table, matched by model-call identity, tool-call
identity and observation binding. `--step` has no parent table: state,
privileged, perception, compute and pure lib scripts are usable; tools requiring
unrecorded dynamic configuration or targets fail with explicit missing requests.
Neither selector scans later input tables or final artifact stores to fill gaps.

`--tool` with another tool than the parent's, or with other arguments, gets the
configuration the live dispatch would have built for that call
(`snapshot_config.py`). The parent table's frozen part (`host/runtime/inputs.freeze`)
is the same for every tool at one boundary and is kept; the per-call sections
are rebuilt with the live builders over a read-only host that sees the run as
it was at the boundary (its current observation and the policy-state chain cut
at the parent's publication): `geometry_query` and `vision_segment` from their
`step` argument, `observe` from the boundary state. A `step` published after the
boundary is refused as the live runtime refused it then. The parent's own tool
with its own arguments keeps the recorded configuration. Where the boundary has
no evidence for the request (the parent was `compute`, `observe`,
`report_issue` or `finish_episode`, whose calls freeze no tool configuration; a
`finish_episode` lifecycle section outside a `finish_episode` boundary; no
parent row) the result is `NotRecorded` with a `refusal` object
`{status: "refused", reason, tool, section, usable_boundaries: [{at, tool}]}`
naming the parents whose recorded configuration serves the requested tool.

`mode: snapshot` results carry the selected state hash/boundary, private final
tool-state values, source request context, progress/timing, missing requests and
service comparisons. `capability_override` identifies the RSI-only task-check
access; it reads the recorded boundary verdict and never runs a live check.
The additive CallContext boundary `{kind: "snapshot_step", step: K}` identifies
a completed publication without claiming an original parent invocation.
Output uses the same separate toolkit-replay directory convention as diagnostics.
Scripts run in-process and are trusted local code; this is not process isolation.


### Native-tick snapshot restoration

```bash
# The resource profile is supplied by the caller; this command does not schedule it.
ora toolkit replay RUN --native-action 6 --resource-profile ALLOCATED_PROFILE \
  --script ora/agent/toolkit/examples/ground_geometry.py \
  --args '{"camera":"head","text":"cup","max_candidates":1}' \
  --service-report SERVICE_REPORT --json
```

**N is the number of completed native actions (ticks)**: select the end of the
controller sample whose `native_action` equals N. With a controller counter
starting at 1, `--native-action N` means after the Nth tick. **N=0 means reset,
before any native action**. An action interval `[first,last]` on the recorded
1-based tick labels corresponds to `[first-1,last)` on the completed-action
axis. N is not a parent-tool ordinal or receipt index. If multiple rounds have
the same N, report NotRecorded; this version has no implicit round choice.

`native_boundary.py` validates and truncates the controller trace and its ten
physics lines per tick, preserving original and derived hashes separately in
`native-boundary.json`. It maps action_id through the trusted execution ledger
to its pre-action complete snapshot and unique execute_action begin. Tool deltas
are applied through that interface ordinal, including multiple writes in that
gap, without consuming the I/O cursor. Missing mappings or relevant gaps refuse
reconstruction. No later input table or artifact directory supplies configuration.
Native selection consequently has the same conservative script/configuration
scope as --step; --tool requires explicit --args.

The independent diagnostic worker replays only this prefix, including preceding
recorded render synchronizations. Every available control/physics prefix and
endpoint measurement must match exactly before capture is granted. Captured
object poses are also checked when present. Version/source fingerprint changes
are warnings with both identities; evidence corruption and physical divergence
still refuse. Unlike the legacy live path, **fresh rendered pixels are the tool
observation**. Rendering must not advance physics. The source's old pixels are
not relabeled as a new capture. This path makes no sensor byte-equality claim.

The worker, lease and scratch are released before the supplied script runs.
The script receives the same detached snapshot facade: actions remain forbidden,
planner reports SimulatorRequired, lifecycle/context requests only write output,
and perception requests use the newly rendered images. Official status is read
once from the verified restored simulator and exposed only through the RSI
privileged/task-check capability. It does not use the source run's final verdict.

`native-state.json` records the fresh env values, reconstructed exec state and
private tool overlay, bound to the original complete snapshot hash and action
mapping. It is explicitly **not** an original completed-action publication and
does not append to either source hash chain. The last receipt remains the last
completed one; no current action's future receipt or final command buffer is
imported. Current recovery previous anchors include the current pose action's
pre-action update. Recovery/gripper/wait preserve their original anchors, as
those execution routes do not call _remember_backtrack. Result `replay_semantics=native_tick_fresh_render` distinguishes
this path from `completed_snapshot`; `verification.json`, stage progress/timings,
warning evidence and independent service comparisons stay in the output.


Intermediate state field availability:

| Field | Availability after N > 0 ticks |
| --- | --- |
| env.camera / env.proprioception | Fresh render and measured robot state at the verified tick. |
| exec.command.<arm>.gripper_directive | Exact tick `gripper_feedback.requested_opening_normalized`. |
| exec.command.<arm>.pose_target_xyz_xyzw | **Omitted**: executor bookkeeping is not recorded per tick; neither measured pose nor final-action target is substituted. |
| exec.command.<arm>.joint_target_rad | The joint target the plant retains after the restored ticks (read from the SDK command buffer at every request's end; absent from runs recorded before it). |
| exec.ik_sampler_state | The boundary snapshot's CuRobo IK seed generator state (`ora.robodojo-ik-sampler-state.v1`), restored before any new IK; a run recorded before it restarts the generators from their initial state and every restore result and kinematics answer carries `ik_sampler_state: unrecorded` with a warning. |
| exec.recovery.<arm>.previous | Current pose action pre-update; recovery/gripper/wait preserve their starting anchors. |
| exec.last_receipt | Last completed receipt only; the current action is not fabricated as complete. |
| tool.* | Last complete snapshot plus verified deltas through current execute_action begin. |
| privileged.ground_truth | Exact-clock captured evidence if present and physically verified; otherwise NotRecorded. |

`missing_state_fields` in both `native-state.json` and `result.json` lists each
omitted pose-target field and its reason. The diagnostic worker privately keeps
pre-action pose bookkeeping for capture only; no new action can consume it.
N=0 and ordinary complete snapshots retain their recorded command pose targets.

### Snapshot review boundaries

Snapshot service comparisons can refer to the selected parent's historical
response, which occurred **after** its before-call boundary. They are therefore
output-only audit evidence in `comparisons.jsonl` and its referenced CAS blobs.
The script receives the new service response with `Response.comparison=None`.
Neither historical response/array refs nor comparison blobs are granted to
`api.artifacts.read_blob`. The legacy diagnostic comparison behavior is unchanged.

For `--step`, a complete physical/observation publication can occur inside an
unfinished parent tool. A later zero-spend rollback or verify write can change
`tool.*` before the policy resumes. When the snapshot watermark is inside such
an invocation (or lacks complete tool-state evidence), the snapshot API omits
all `tool.*` keys and reads report NotRecorded. `result.json.missing_state_fields`
lists `tool.*` and its reason; exact published `env.*`, `exec.*` and privilege
values remain available. To read settled policy-facing tool state, select the
next parent with `--snapshot --at K+1`. Native reconstruction uses the raw
publication plus verified deltas through execute_action begin instead; it does
not infer a settled tool state from a step publication.

Before-parent selection rejects an action proposal, receipt or round reset
between the last publication and the selected proposal. It does not combine a
stale publication with a newer physical world. Intermediate native snapshots
keep the pre-action `exec.world_mutation_sequence`: this counter advances when
the real action receipt is committed, not merely when a tick finishes.

Native `--tool` currently admits only `compute` and `check_task` without a
parent's frozen dynamic configuration. Other tools, unknown tools and invalid
explicit arguments are refused before starting a worker; scripts may still use
state, fresh perception and pure lib computations. The output may be any legal
independent directory, including outside the global ORA workspace. Only the
new native path places the worker at `OUTPUT/environment`, making the RPC
shared root `OUTPUT`; the observation destination is checked against that root
before staging resources. Source and output must remain disjoint. SIGTERM uses
normal context unwinding to close the worker, release its lease and reclaim
scratch; the signal handler is restored afterwards. The old live path retains
its original output layout, diagnostic-source list and early-error behavior.

The native action/ledger link compares the submitted interface action after
one `validate_runtime_action` pass against the stored owner action, exactly as
the execution owner does before `log.begin`. This includes quaternion
normalization. The ledger is not normalized a second time and no pose tolerance
is introduced: different positions, orientations, observation bindings or
extra fields still fail the action-identity check. Interface evidence remains
in its original submitted form.


## Protocol declaration (M7.6)

All new runs declare the manually numbered data-interface protocol in public
`toolkit-protocol.json`, even when recording is off. Recording also declares it
in `native/toolkit-io/manifest.json` and every begin/end event. Replay compares
these declarations and reports both sides in `protocol_compatibility`; drift
or missing legacy declarations warn and continue. Evidence integrity and
physical restoration checks are unchanged. See [protocol and gate audit](toolkit-protocol.md)
for schemas, affected paths, decoder limits and version increment rules.
