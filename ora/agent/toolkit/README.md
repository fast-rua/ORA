# Toolkit：数据接口、lib 与工具

## 职责

智能体侧的工具代码：可信数据接口（`interface/`）、可组合的能力与纯计算（`lib/`）、
工具入口（`tools/<name>/tool.py`）、按名字找到工具入口的 `registry.py`，以及运行
这些工具的可信宿主（`host/`）。工具只经数据接口拿外部数据，不持有 worker、host、
文件根或真值能力。

## 放什么，不放什么

- `tools/<name>/`：一个工具一个目录。`tool.py` 定义 `run(api, args)`；
  `description.json` 是模型看到的这个工具的说明、输入 schema、结果 schema（原样交付，
  由 `policy.lock` 钉住）；`README.md` 给开发者看。工具私有 helper 也放这里，
  跨工具的放 `tools/` 下的模块。
- 工具面只管"选哪些工具、怎么组合"：文件式工具面（目前是 `robodojo_placement_toolkit_v5`；v4 是旧的观测回执方式）
  在 `ora/policy/tool_surfaces/<面>/` 下放 `surface.json`（工具顺序、能力、授权等）和
  `prompt.md`（工具面级说明，可空）。改一个工具的说明，所有列出它的工具面都跟着变；
  已封存的局靠计划里的提交和每个文件的哈希保持原样。格式见
  [file-backed toolkit surfaces](../../../docs/reference/toolkit-surface.md)。
- `rsi_interface.py`：RSI workdir 依赖的接口版本号和逐版本的迁移说明（ora-rsi 的 `rsi-workdir update` 读它）；改了工具目录约定、说明文件格式、lib/interface、工具面或 task 文件的规矩，就把版本加一并写迁移说明。
- `host/`：可信宿主：各工具数据接口背后的适配器、I/O 记录、受限 `compute` 求值器
  （`host/compute/`，在独立子进程里求值）、`finish_episode` 和 `check_task` 的规则（`host/finish/`）、
  抓取提议器与编译器（`host/grasp/`）、`grasp_execute` 的分段循环（`host/skills/`），以及每局构建的
  工具运行时框架（`host/runtime/`：派发 `dispatch.py`、发布 `tool_data.py`、输入冻结 `inputs.py`；
  由旧栈 `ora/agent/tools` 搬来，旧包已删除），见 [host/README.md](host/README.md)。
- 不放 worker、感知服务实现、真值读取；进程沙箱只在 `host/`。

## 依赖方向

工具 → lib → interface；工具也可直接调用 interface。`tools/` 下只允许 import 公开
接口、lib 和其他工具（`tests/ora/agent/toolkit/tools/test_boundary.py`）。interface
通过注入回调接 worker，不 import environment 或 orchestrator。`host/` 可以 import
工具、lib、interface 和 `composition`；`tools/`、`lib/`、`interface/` 不 import
`host/`（同一测试文件），结构规则 L6 把 `toolkit/host` 排在 `toolkit` 之上。

## 入口

- 运行时：工具面声明 `toolkit_only_v1` 时（每个已注册的工具面都声明，`dispatch.served`），
  `ora/agent/toolkit/host/runtime/factory.py` 装上
  `host/runtime/dispatch.py`，它经 `registry.py` 按名字调用工具入口；
  同步执行、旧版可见回执等行为按工具面声明的能力分支，不按工具面名字。
- 回放：`ora toolkit replay RUN --at K --tool NAME`，用法见
  [toolkit-io](../../../docs/reference/toolkit-io.md)。
- 工具面怎么选：[file-backed surfaces](../../../docs/reference/toolkit-surface.md)、
  [synchronous execution](../../../docs/reference/toolkit-execution.md)。工具面 v1 到 v4
  和旧混合工具面已退役，现存 v5（文件式，RoboDojo 默认）。

## 文档与进度

- 2026-10-26：19 个工具全部迁到 `tools/<name>/tool.py`；派发改为按目录注册、按能力
  分支；新增文件式工具面 v4（与 v3 模型可见内容逐字节相同）。`artifact_read` 属于
  上下文管理，不在这里，随上下文重构处理。
- 下文是 M1（2026-10-23）的盘点和工作记录，按当时状态保留，不代表现状。

---

# Toolkit 工作记录

2026-10-23，M1，基线 `69912fbc056c54172441248ac58259e715a0f196`。

## 当前交付

- 只新增本 README，附能力清单、108 个工具 Python 文件/618 个顶层函数或类方法的
  签名索引和本记录；没有改运行代码、模型可见资产、worker 或 action_executor。
- 完成十家族盘点，输入/输出/坐标/依赖/副作用/权限/记录状态及新增/不适用项分列。
- SAM3 三种现有调用、CGN 有界全集、probe/align/reach-check 与底层未暴露能力有区别。
- 更新采用 runtime/replay 新口径；仿真读历史，外部感知重新请求并比较。
- M1 后等待审阅；M2 接口约定文件及实现尚未开始。候选工具 I/O 路径尚未冻结。

## 验证

首轮 `check --quick`：6299 passed、140 skipped、1 xfailed、3 failed。
其中一个为本次将非 README Markdown 放入 ora/ 引发的 package-data 检查，已将
全部文档合入 README；另两个是本机缺 ffprobe。测试用临时 PATH 提供 ffmpeg/ffprobe，
不改依赖锁、系统安装或运行 checkout。

最终 `scripts/dev/check --quick` 全部已运行步骤通过：主 Python suite **6302 passed、
140 skipped、1 xfailed**（1 条现有极大浮点投影 overflow warning）；release tests 88、
research tests 8、catalog tests 6、goldens 7、no-write 1 均通过，policy-lock/whitespace/
research-check/catalog-check/release-scan 通过。跳过项包括本机缺少固定机器人资产；
`--quick` 按定义不跑完整 agent-core/TypeScript/web/docs lanes。没有 GPU 或真服务测试。

提交后的验证：`compare-model-visible 69912fbc HEAD` **6 cells、6 same、0 different、
0 unresolved**；`release-ora scan` **SCAN PASSED**。58 个 WORKER_KERNEL_PATHS 与
`ora/orchestrator/episode/action_executor.py` 已逐文件比对基线字节，全同。
作者检查 `check-authors 69912fbc..HEAD --allow 'memset0 <memset0@outlook.com>'` 通过。
本节补记验证结果后，对最终提交再次执行模型可见比较和发布扫描。

清单静态核对：十家族全部108个模块有对应条目，618个函数/方法签名已索引，
默认 manifest 的19个工具逐项对应；表格列数及 whitespace 校验通过。

提交作者按任务要求为 `memset0 <memset0@outlook.com>`；仓库默认作者白名单仅
freemty，本次用检查器的 `--allow` 验证。没有改仓库身份规则；远端默认 CI 的作者
检查仍可能拒绝该身份，这属于协作配置限制，不冒称全 CI 已绿。

轨迹会话已提供 `rsi.traj.snapshot.v0`，参考主仓库提交
`46e1961dbba6872645bb95a7f2472135c0c95cdc` 的 `rsi/README.md`。
M2 待对齐：available_at_boundary 与事后 replay_evidence 分开，协议键缺失为 null。

## 已接受的审阅约束

F01/F06：逐字段公开语义 envelope、数组 blob 与来源、服务身份、失败/超时/retry；
不序列化 worker 或整份 control/plan。F02：语义键与 occurrence 分离，顺序 cursor。
F03：动作不匹配即分歧，replay 不连在线执行。F04：判定按边界，无记录即缺失。
F05/F09：每个实际放出的能力须有 contract test；现有/需新增/不适用明确区分。
F08：参数化算法进 lib，默认任务偏好与编排留工具。
F11：worker kernel 与 action_executor 都保持字节不变。

## 后续范围（不是本轮完成声明）

- F07：同进程只有接口组织边界；OS/进程/文件/网络隔离及其负向测试属于后续。
- F10：外部包不可变归档、依赖/资产闭包、版本 decoder、封存覆盖与损坏/path 测试。
  本轮预留加载入口不等于完成内容寻址包发布或旧 run 任意版本可执行。
- F12：真实短局只覆盖少数路径，且 learned CGN pool 为空；需加入非空/拒绝/截断
  假样本、单点多连通域、batch 候选、重复请求/cache、零 tick 失败、跨 round stale
  引用、错误/timeout、partial blobs；等价验收比较文本/图像顺序/错误/动作与预算。
  GPU 数值一致性与任意时刻在线恢复没有测试，本轮不跑。
- F13：以最新明确决定覆盖早期示意；区分公开配置/实测数据、捕获/发布/读取、
  proposal/receipt、可查看历史/可执行旧工具/可驱动在线三种兼容承诺。

---

# M1 能力清单

阅读日期：2026-10-23。ORA 基线：`yulun/tool-rsi-dev` @
`69912fbc056c54172441248ac58259e715a0f196`；实现分支：`yulun/toolset-refactor`。
旧 SDK 只读参考：`mem/exp15a-dev` @ `eec772d220a2fec3e0840b7437d9ebe94c017893`，
`robot_harness/agent/tools/composed/sdk.py`。本文是源码事实与迁移清单，不是新 API。

## 读法、坐标与证据边界

下表按可组合能力分项；同一项可以覆盖多个只服务这一能力的私有辅助函数。
[全量源码符号索引](#m1-全量源码符号索引) 按文件逐项列出全部函数/方法签名、返回注解和行号，包含
私有函数、所有十个家族及空包，因此不会把“默认工具未用到”误当作“代码不存在”。
下表的路径均相对 `ora/agent/tools/`，括号内为代表入口；索引补全同文件的辅助入口。该包已在 P21 删除：下表是迁移时的清单，其中的源码链接不再指向现存文件。

- **W**：世界系，距离米，位姿 `[x,y,z,qx,qy,qz,qw]`；四元数为 xyzw。
- **C**：指定相机 OpenCV 系，距离米；外参通常 world→camera，反投影须求逆，
  不可把其 3×4/4×4 存储形状当作方向声明。
- **P**：源图左上原点像素 `[u,v]`，框 `[x0,y0,x1,y1]`；显示图与源图须明确区分。
- **J**：关节弧度、夹爪实测量按 embodiment 解释；不把夹爪开合 directive 当米。
- **L**：显式 embodiment controlled-point 局部系。RoboDojo X5 link6、Panda grip-site、
  ALOHA grip-center 等不能互换；approach / jaw 轴从 descriptor 读取。
- **—**：不涉空间；通用数值函数由调用者保证量纲，三角函数区分 rad/deg。
- **纯计算**表示给定显式输入可以离线算；**文件**表示现有实现读写 run 或绑定资产；
  **服务**表示 SAM3/CGN；**回调**表示 coordinator 注入的 worker 能力。
  “只读”指不改变物理世界，不保证不写缓存、plan handle 或诊断文件。

数据接口未来没有读 GT 的方法。mask 是公共 RGB 上的模型预测，不是模拟器真值
segmentation。IK、可见表面 clearance、jaw-stall、lift 证据均不是官方成功判定。

### 清单状态与记录标签

下表每项的“现有”是基线能力存在，**不是新 toolkit API 已实现**；全部新 facade
与 contract test 都需在 M2/M3 新增。只有接口实际放出后才计覆盖，一个 wrapper
名字存在不算验收。测试应覆盖正常、边界和失败；M1 文档不新增镜像实现的测试。

权限标签：**P**=只允许公共观测/派生计算/公开配置；**H**=可信 host 的控制与
证据存取，只能经逐字段投影开放，不把 control/plan/worker 整对象传给工具；
**O**=官方布尔判定例外，受 check_task 权限控制；**A**=动作必须通过 supervisor。
同一项多个标签表示拆分后的边界，不表示工具获准读取其内部文件。

记录标签（均指旧实现，**没有统一工具 I/O occurrence 索引**）：

- **D**：派生计算无独立原始 I/O 记录；部分结果可见于工具结果或产物，不能保证所有中间值。
- **B**：`policy-observations/` 与 `policy-state/` 不可变产物/发布事件可查；仍需确定边界。
- **V**：`openeta-tool-artifacts/` 保存派生产物/manifest；字段覆盖按方法校验，非原始响应全集保证。
- **I**：probe 有 `native/motion-probes/` 诊断/plan；align/reach 回调没有同等完整 I/O 记录。
- **C**：control/内存会更新，只有部分事件可还原，缺历史快照就“未记录”。
- **R**：action/proposal 事件与 `native/receipts/` 存在，一次 skill 对多个 native receipt，需关联。
- **T**：当前 task-check latch 可被最终 verdict 覆盖，不能用作所有历史边界的值。

## 1. compute / shared

| 能力与源码 | 输入 | 输出 | 坐标/单位 | 当前依赖与拟归属 | 状态 / 后端 | 副作用 | 权限 / 旧记录 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `compute/arguments.py`：normalize_block/read_bindings/invoke | source、命名绑定及 origin、surface | 规范化表达式、数值结果与证据/拒绝 | —，沿用输入量纲 | 文件/host + evaluator；工具参数逻辑与接口记录分开 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |
| `compute/evaluator.py`：validate_source/RestrictedComputeRuntime | 受限 AST、绑定、预算 | 输出测量、中间值分类、语法/资源错误 | — | 独立 CPU worker/fork、纯计算；不是模拟器 worker | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |
| `compute/namespace.py`：geom/np/math/Rotation | 向量、矩阵、四元数、数值 | 归一化、角度、旋转、变换、数值运算 | 通用帧；angle_deg=度，Rotation 明确单位 | NumPy/SciPy 纯计算 → lib | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |
| `compute/bindings.py`：_digest | 数值绑定 | 稳定摘要 | — | 哈希纯计算 → interface 证据辅助 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |
| `shared/rgbd_fusion.py`：fuse_rgbd_pointcloud_sources | 多相机 RGB-D/K/T、workspace、voxel/max-depth | W 点云、颜色、来源/融合记录 | C→W，米 | 文件读取 + NumPy；读取经 interface，融合进 lib | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |
| 同文件：materialize_two_view_pointcloud/orthographic_click_constraint | 两视图、投影参数、正交像素 | 点云/渲染产物、两个世界轴约束 | P↔W，米 | 文件、PIL；渲染与几何进 lib | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |
| `shared/artifact_io.py`：_atomic_* / _artifact / _verify_artifact | JSON/array/image/path/hash | 原子文件、引用、完整性验证、base64、事件行 | — | 文件/编码 → interface；不是工具任意路径权限 | 现有；封装需新增 / CPU/公共数据 | 文件原子写/读/校验/事件追加 | H / V/C |
| `shared/saved_steps.py`：round_timeline/route_current | step、当前观测、policy-state | 当前/历史选择及 round 范围拒绝 | — | 已验证 policy state 文件 → interface | 现有；封装需新增 / CPU/公共数据 | 读观测/历史状态并校验 | P/H / B |
| `shared/errors.py`、`artifact_errors.py` | 字段、约束、失败原因 | ToolError 与类型化产物错误 | — | 纯构造/校验 → 共用契约 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |
| `shared/service_refusals.py`、`target_refusals.py` | 服务异常、target、质量和可用 ref | 重试性、字段和测量齐全的拒绝 | 沿用输入 | 文件索引 + 纯投影；不能丢失原始错误身份 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |

compute 的现有词汇包括：`np` 的 array/asarray、dot/cross/outer、sqrt/abs/sign、
sum/mean/median/min/max/argmin/argmax/minimum/maximum、round/clip/where、
concatenate/stack/column_stack/vstack/hstack/transpose、float64/eye/zeros/ones、
反三角/三角/角度转换；`np.linalg` 的 norm/solve/lstsq/inv/det/eigh/svd；
`Rotation` 的 from_quat/from_matrix/from_rotvec/from_euler/identity；`geom` 的
unit/angle_deg/rotate/transform_point；`math` 的显式白名单及纯数值内建。
这不是可 import 任意 Python/NumPy 的权限；M1 不改 grammar 或说明。

## 2. observe / perception

| 能力与源码 | 输入 | 输出 | 坐标/单位 | 当前依赖与拟归属 | 状态 / 后端 | 副作用 | 权限 / 旧记录 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `observe/arguments.py`、`retention_view.py` | step、views、history、surface | 当前/历史路由与参数拒绝 | P / — | surface + saved state；工具层 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |
| `observe/dual_arm.py`：_dual_arm_observe | 相机选择、overlay、host 当前/历史观测 | 双臂状态、带标注图片和来源 | P、W/J | 文件 + host；数据经 interface、投影进 lib | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / B/V/D |
| `observe/mark_point.py`：robust_depth_measurement | depth、像素、窗口/阈值 | 局部深度、支持质量或拒绝 | P→深度米 | NumPy 纯计算 → lib | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |
| 同文件：opencv_camera_point_to_world | C 点与外参 | W 点 | C→W，米 | NumPy 纯计算 → lib | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |
| 同文件：measure/execute | view、point id、像素、当前观测、已有点记录 | 世界标记/验证距离/质量、overlay | P↔C↔W | 文件，点缓存与跨视图累计；工具层管理状态 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / B/V/D |
| `observe/mark_point_batch.py`、`mark_point_schema.py` | 同一 view 最多 8 个 id/像素 | 逐点成功/错误和 attribution、绑定 schema | P/W | 文件 + 测量；不是 8 点联合分割 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / B/V/D |
| `observe/mark_point_overlay.py` | marks、views、颜色、批上下文 | 合并 overlay、图像归属 | P | PIL + 文件；begin/finish 有状态，不能直接当无状态 lib | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / B/V/D |
| `perception/arguments.py` | 像素列表、region、world point | 边界/有限性校验与结构化拒绝 | P/W | 纯校验；region 是 geometry ROI，不是 SAM 框提示 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |
| `perception/backends.py` | RGB-D+单点/多点，或 RGB+文字 | mask/全部服务候选与 metadata | P、深度米 | 三个 Protocol；未来 interface 的服务边界 | 现有；封装需新增 / SAM3/CGN（按依赖列） | 服务请求、产物/缓存 | P / V |
| `perception/fallback_segmenter.py`：DepthPromptSegmenter | RGB-D、positive point | 确定性深度 mask 与 metadata | P，深度米 | CPU fallback/资格测试基线；不是 SAM3 | 现有；封装需新增 / SAM3/CGN（按依赖列） | 服务请求、产物/缓存 | P / V |
| `perception/segmentation_batch.py`：select_masks/materialize_batch | raw masks/scores、queries、总量 cap、RGB-D | 连通域、去重、逐点 ledger、target refs/产物 | P/C/W | 纯选择 + 文件；全量服务响应必须先记录再裁剪 | 现有；封装需新增 / SAM3/CGN（按依赖列） | 服务请求、产物/缓存 | P/H / V |
| `perception/grounding.py`：materialize_grounded_targets | RGB、depth/calibration、text、candidate cap | masks、target manifests、定位、overlay、query manifest | P→C→W | SAM3 + 文件；拆服务 I/O 与后处理 | 现有；封装需新增 / SAM3/CGN（按依赖列） | 服务请求、产物/缓存 | P/H / V |
| 同文件：grounding_public_result | query manifest、产物 | 模型可见结果投影 | P/W | 纯投影 → 工具层，保留原 schema | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |
| `perception/grounding_palette.py` | candidate index | 稳定颜色和展示上限 | RGB | 纯计算；展示上限不是模型能力上限 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |
| `perception/grounding_preview.py` | mask、RGB-D/K/T、preview options | 候选放大图、深度剖面与刻度、绑定记录 | P/C/W | PIL/NumPy + 文件；计算/绘图进 lib | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / B/V/D |
| `perception/mask_localization.py` | mask、depth、相机标定、质量参数 | inliers、中心、范围、footprint、PCA 主轴、target frame | C/W，米 | NumPy 纯计算 → lib；质量失败须保留 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |
| `perception/target_support.py` | 同一视图的 mask 邻域深度/点 | support plane、相对高度及质量/不可用 | W，米 | NumPy 鲁棒拟合 → lib；不是 GT 桌面 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |
| `perception/target_geometry.py` | 绑定 target+RGB-D、workspace | scene/target cloud、世界几何、主轴 | C↔W，米 | 文件验证 + 纯计算；拆到 interface/lib | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / B/V/D |
| `perception/localization.py` | localization/ref、机器人夹爪反馈 | 规范位置、实测 jaw separation、closure evidence | W/L/J，米按 descriptor | 文件与纯派生；不输出官方 holding 真值 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / B/V/D |
| `perception/overhead_view.py` | 目标 footprint、末端姿态/轴 | overhead view pose/标注或 unavailable | W/L，米、xyzw | 纯几何 → lib | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |
| `perception/target_store.py` | observation id+hash、RGB/depth/calibration hash、target ref | 内容寻址 target、索引、来源验证、选择局限 | P/C/W | 文件 → interface；旧缓存不能跨观测混用 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / B/V/D |
| `perception/target_identity.py` | target ref、grounding manifests | 由 query 得到的文字语义或 unnamed | — | 文件查询 → interface；不是任务物体真值 | 现有；封装需新增 / CPU/公共数据 | 只读 grounding manifest | P/H / B/V/D |
| `perception/feedback.py` | surface、感知/抓取结果、图像标签 | 紧凑决策证据 | 沿用输入 | 纯投影 → 工具层 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / B/V/D |

## 3. grasp

| 能力与源码 | 输入 | 输出 | 坐标/单位 | 当前依赖与拟归属 | 状态 / 后端 | 副作用 | 权限 / 旧记录 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `contact_graspnet_client.py`：ContactGraspNetApiProposer.propose | scene/target Nx3、camera_to_world、target ref、workspace、compiler、count | 有界候选、拒绝 ledger、原始对齐数组、seed/服务/timing | C→W，米、4×4 | CGN HTTP + 编译；原始 response 先进入 interface | 现有；封装需新增 / SAM3/CGN（按依赖列） | 服务请求、产物/缓存 | P / V |
| `proposers.py`：GraspProposalBackend/PcaTopDownGraspProposer | target/scene points、bounds、compiler、count | PCA top-down proposals、诊断 | C/W、米 | NumPy 纯计算 → lib | 现有；封装需新增 / 按 embodiment/compiler | 无物理副作用 | P / V/D |
| 同文件：AlohaHybridGraspProposer | RGB-D clouds、learned proposer、geometry | learned+topdown 合并候选与分族诊断 | C/W/L | CGN + 纯计算；服务经 interface | 现有；封装需新增 / SAM3/CGN（按依赖列） | 服务请求、产物/缓存 | P / V |
| `x5_topdown.py` | surface、target footprint、公共 scene cloud | X5 compiler 选择、support z、contact height | W，米 | 纯几何/配置 → lib | 现有；封装需新增 / 按 embodiment/compiler | 无物理副作用 | P / V/D |
| `aloha_gripper.py`、`arx_x5_gripper.py` | 固定源几何、opening、可选 aperture calibration | jaw 模型、可用开度上限、双臂 compiler identity | L，米 | source-bound descriptor；名义开度与实测校准分开 | 现有；封装需新增 / 按 embodiment/compiler | 无物理副作用 | P / V/D |
| `compilation.py`、`compiler.py`、`compiler_registry.py` | source pose/axes/opening、embodiment、hover | xyzw、controlled-point contact/pregrasp、编译候选/拒绝、公开 identity | C/W/L，米 | 纯计算 → lib；不改 policy 绑定 | 现有；封装需新增 / 按 embodiment/compiler | 无物理副作用 | P / V/D |
| `scene_admission.py` | 指定相机、RGB-D validity、融合范围 | 可用相机、完整输入 cloud、拒绝及 profile | C/W，米 | 文件 + 融合；不要只给默认工具选中的相机 | 现有；封装需新增 / 按 embodiment/compiler | 读写文件/状态，见依赖列 | P/H / V/D |
| `query.py`：materialize_grasp_query | 当前 target、backend/config、compiler、候选选择 | 内容寻址 query、input cloud、backend-output、manifest、缓存验证 | C/W/L | 文件 + backend + selector；接口不等于最终 public_result | 现有；封装需新增 / 按 embodiment/compiler | 读写文件/状态，见依赖列 | P/H / V/D |
| `selection.py`：apply_candidate_selection | 全部已编译候选、selector、显示 cap | 排序全池与截断显示 | W/L | 纯选择或注入 selector（可调 reach）；先保存全池 | 现有；封装需新增 / 按 embodiment/compiler | 无物理副作用 | P / V/D |
| `lift_geometry.py` | contact/pregrasp、lift height、workspace | 抬升目标、越界拒绝、liftable/withheld | W +Z，米 | 纯计算 → lib；编排仍在工具层 | 现有；封装需新增 / 按 embodiment/compiler | 无物理副作用 | P / V/D |
| `diagnostics.py`、`results.py` | family metadata / query manifest | 诊断汇总、grasp_plan/grasp_query 公开结果 | 沿用输入 | 纯投影 → 工具层 | 现有；封装需新增 / 按 embodiment/compiler | 无物理副作用 | P / V/D |
| `overlay.py` | 候选 gripper geometry、target RGB/mask/points、相机 | 局部 crop、wireframe、等距图/contact sheet | P/C/W/L | PIL/NumPy；部分读取文件，拆 I/O | 现有；封装需新增 / 按 embodiment/compiler | 读写文件/状态，见依赖列 | P/H / V/D |
| `review.py`、`scene_extent_review.py` | query、source cameras、cloud、bounds | review image/cloud、全 scene extent、产物引用或不可用原因 | P/W，米 | 文件+PIL/NumPy；review 与 backend 输入范围分别记录 | 现有；封装需新增 / 按 embodiment/compiler | 读写文件/状态，见依赖列 | P/H / V/D |

## 4. reach / motion

| 能力与源码 | 输入 | 输出 | 坐标/单位 | 当前依赖与拟归属 | 状态 / 后端 | 副作用 | 权限 / 旧记录 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reach/alignment.py`：frame_local_axes/synthesise_approach_poses | target centroid、side、standoff、roll count、local axes | world approach poses | W/L，米、内部 roll rad、xyzw | 纯计算 → lib | 现有；封装需新增 / 按 backend/surface | 无物理副作用 | P / D |
| 同文件：validate_arguments/build_request/plan | target ref、side、arm、standoff/count、centroid callback | align request、逐候选 reach/error | W，米 | centroid 文件 + align_plan 回调；工具默认 3、允许 1..6 | 现有；封装需新增 / 后端回调（RoboDojo 已有） | 不推进物理；可写探测证据/handle | P/H / I/D |
| `reach/check.py`：check_queries | arm、pose/interaction/from_index、workspace | 按输入顺序的完整 reach rows | W/J，米/rad | align_plan 回调上的 reach-check schema → interface | 现有；封装需新增 / 后端回调（RoboDojo 已有） | 不推进物理；可写探测证据/handle | P / I/D |
| 同文件：select_grasp_candidates/grasp_candidate_selection | compiled pool、jaw rule、backend query | pregrasp→approach、exact/jaw flip、倾角排序与选择摘要 | W/L/J，米/deg/rad | 纯编排 + reach 回调；候选排序逻辑是工具层 | 现有；封装需新增 / 后端回调（RoboDojo 已有） | 不推进物理；可写探测证据/handle | P / I/D |
| `reach/jaw_center.py` | surface/descriptor hash | 每臂 approach axis、jaw center/contact offset | L，米 | hash-bound 资产文件 → interface；纯提取进 lib | 现有；封装需新增 / 按 backend/surface | 读写文件/状态，见依赖列 | P/H / C/D |
| `reach/jaw_symmetry.py` | measured joints、两成员各阶段 reach rows | wrist headroom、成员选择、unknown/summary | J，rad | 纯计算 → lib | 现有；封装需新增 / 按 backend/surface | 无物理副作用 | P / D |
| `reach/move_pose.py` | 原始目标、当前姿态、jaw geometry | planned target、沿线可达位置、jaw choice 与 annotations | W/L/J，米/rad | align_plan 回调 + SLERP/二分等纯计算 | 现有；封装需新增 / 后端回调（RoboDojo 已有） | 不推进物理；可写探测证据/handle | P / I/D |
| `reach/orientation.py` | 原始目标、jaw center、turn axes、角度搜索 | 最小可达转角、目标 pose、失败诊断 | W/L，米/deg | reach 回调 + 纯旋转；不是执行许可 | 现有；封装需新增 / 后端回调（RoboDojo 已有） | 不推进物理；可写探测证据/handle | P / I/D |
| `reach/reference.py`、`probe_feedback.py` | surface、raw backend answer/variants | basis、reach/nearest orientation、IK 候选计数/摘要 | W/J，米/rad | 纯投影；完整诊断不能只留此摘要 | 现有；封装需新增 / 按 backend/surface | 无物理副作用 | P / D |
| `motion/arguments.py`、`refusals.py` | flat/nested target、schema、bounds | 统一 arm target、copyable targets、结构化拒绝 | W/L，米/deg/xyzw | 纯编译/校验 → lib；工具 schema 不改 | 现有；封装需新增 / 按 backend/surface | 无物理副作用 | P / D |
| `motion/world_axes.py` | approach/jaw 世界方向、局部轴、target | xyzw 与 target_axes_world | W/L，单位向量 | 纯几何 → lib | 现有；封装需新增 / 按 backend/surface | 无物理副作用 | P / D |
| `motion/clearance.py`：footprint_at/robot_volume/measure | jaw boxes、opening、held box、pose、DepthViews | footprint、robot occlusion、可见最低 clearance/coverage | L→W，米 | 几何纯计算；view_from_files 的文件读取进 interface | 现有；封装需新增 / 按 backend/surface | 读写文件/状态，见依赖列 | P/H / C/D |
| `motion/clearance_geometry.py`、`clearance_runtime.py` | descriptor、current RGB-D、grasp/gripper记录 | jaw geometry、held extent、clearance annotation | L/W/J，米 | 文件/缓存 + 几何；held extent 是估计，不是 GT | 现有；封装需新增 / 按 backend/surface | 读写文件/状态，见依赖列 | P/H / C/D |
| `motion/gripper_intent.py` | gripper args、可选 target_ref | 拆分/标注 close intent | — | 纯构造；不验证已夹住 | 现有；封装需新增 / 按 backend/surface | 无物理副作用 | P / D |
| `motion/pose_target_annotations.py` | compiled target、reach+clearance | 合并只读 annotation | W/J | host + 只读回调/观测 | 现有；封装需新增 / 后端回调（RoboDojo 已有） | 不推进物理；可写探测证据/handle | P/H / I/D |
| `motion/target_state.py` | target-state proposal、host | 校验后提交结果 | 按契约 | host submission port；非默认 19 工具亦列入 | 现有；封装需新增 / 按 backend/surface | 读写文件/状态，见依赖列 | P/H / C/D |
| `motion/wait.py` | tick count、当前 commands | 单个 motion-v3 wait proposal | tick/按契约 | 当前状态 + submit；不是本进程直接推进物理 | 现有；封装需新增 / 按 action codec/embodiment | 提交单动作；由 supervisor 执行 | P/H/A / R/C |
| `motion/move_base.py` | planar base target、host | 单个 base action proposal/拒绝 | world planar pose，按契约 | submit；其他 embodiment 能力，不在默认 RoboDojo 19 工具中 | 现有；封装需新增 / 按 action codec/embodiment | 提交单动作；由 supervisor 执行 | P/H/A / R/C |

## 5. skills / finish / runtime

| 能力与源码 | 输入 | 输出 | 坐标/单位 | 当前依赖与拟归属 | 状态 / 后端 | 副作用 | 权限 / 旧记录 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `skills/candidates.py` | 当前 observation、query/candidate id | 绑定候选或 withheld/stale 拒绝 | W/L | 文件 → interface；解析/选择在工具层 | 现有；封装需新增 / CPU/公共数据 | 读当前 target/query/withheld 证据 | P/H / V/C |
| `skills/grasp_feasibility.py` | committed geometry、member policy、probe callback | pregrasp/approach 可达成员、采用后的姿态或拒绝 | W/J，米/rad | probe_motion 回调；几何/选择在 lib/工具层 | 现有；封装需新增 / 后端回调（RoboDojo 已有） | 不推进物理；可写探测证据/handle | P / I/D |
| `skills/ledger.py` | skill、stage outcomes、ledger | pending stage、advance/abandon、terminal status | — | 纯状态机 → 工具层，不是 lib 阶段编排 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |
| `skills/driver.py`：begin_skill/_submit | candidate、rationale、观测与技能约束 | commitment、ledger、**一个** skill-sequence action | W/L/J | 文件+submit；不改成逐段多次提交 | 现有；封装需新增 / 按 action codec/embodiment | 写 commitment/ledger 并提交单 sequence | P/H/A / R/C |
| 同文件：drive_skill_stage/build_skill_receipt/reconcile | commitment、阶段、fresh observation、结果 | 一阶段指令/只读测量、汇总 receipt、释放状态 | W/J | host/文件；由既有执行流程驱动，不新增工具动作循环 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |
| `skills/jaw_hold.py` | close/lift gripper outcomes、open directive | held/not-held/unknown 与来源 | J/夹爪量 | 纯派生 → lib；没有证据时 unknown | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |
| `skills/lift_evidence.py` | target 可见表面、基准/抬升观测 | TCP/jaw/表面变化的有限证据 | W/L，米 | 文件 + 几何；compare_surfaces 不宣称 holding | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |
| `skills/receipt.py` | rationale、stage outcomes、image bindings、字节预算 | 有界 receipt 与阶段图像归属 | 沿用输入 | 纯构造 → 工具层；保留归因和预算 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |
| `finish/abort_evidence.py` | 本 round 事件与 receipts | infrastructure fault signals | — | 文件 + 纯分类；不能让工具伪造权威证据 | 现有；封装需新增 / CPU/公共数据 | 只读事件/回执并派生 ledger | P/H / C/D |
| `finish/candidates.py` | offered/tried events、当前 round | 未尝试候选 ledger 与姿态匹配 | W，米/角度 | 文件 + 纯比较；保留 finish 约束 | 现有；封装需新增 / CPU/公共数据 | 只读事件/回执并派生 ledger | P/H / C/D |
| `finish/outcomes.py` | outcome/reason/category、surface、ledger | admission/refusal、持久 finish 字段 | — | host/文件；工具层逻辑 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |
| `finish/task_check.py` | round 的 policy-task-check、finish outcome | latched/unconfirmed 拒绝 | — | supervisor 写的文件；唯一官方判定入口受开关约束 | 现有；封装需新增 / CPU/公共数据 | 只读当前 latch 和校验 finish | H/O / T |
| `runtime/base.py`：_current/_select_observation/_camera | control pointer、step/camera、state_root | 验 hash 的 observation、相机 RGB/depth/calibration paths | P/C/W/J | 文件 → interface；返回路径不等于允许任意路径读取 | 现有；封装需新增 / CPU/公共数据 | 读观测/历史状态并校验 | P/H / B |
| `runtime/base.py`、`protocol.py`、`factory.py` | name/args、tool surface、runtime 配置 | advertised definitions、分派/校验/结果、runtime instance | — | host glue；仍由 coordinator 构造 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |
| `runtime/core.py` | surface、backend、service bindings、roots | mixins、dispatch、round registry/rollback、capability 选择 | — | 文件/服务/回调/配置；不把整个 host 塞给可改工具 | 现有；封装需新增 / SAM3/CGN（按依赖列） | 服务请求、产物/缓存 | P/H / V |
| `runtime/common.py` | q/matrix、policy state 路径 | rotation conversion、verified state | W/L / — | NumPy + evidence 文件 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |
| `runtime/backend_selection.py` | embodiment/arm/surface、result | scene cameras、compiler id、closed schema 投影 | — | 纯选择；最终工具结果保持不变 | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |
| `runtime/observe.py` | views、step、marks/overlays、current grip | RGB/derived views、点云、public facts、历史观测、marks | P/C/W/J | 文件/资产 + observe/shared lib；历史读取不推进物理 | 现有；封装需新增 / CPU/公共数据 | 读写文件/状态，见依赖列 | P/H / C/D |
| `runtime/geometry_query.py` | world_points_m / points / region、camera/step | 投影、反投影、区域深度统计 | P↔C↔W | 当前观测/产物 + NumPy；非 SAM region prompt | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |
| `runtime/perception.py` | point(s)/text、camera、max_candidates、target | segmentation/grounding/geometry 结果 | P/C/W | 服务+文件+后处理；迁移首批 vision_ground | 现有；封装需新增 / SAM3/CGN（按依赖列） | 服务请求、产物/缓存 | P/H / V |
| `runtime/grasp.py` | target/arm/count 或 normalized probe queries | plan manifest/public result；motion probe 完整 reply 与 handle registry | W/L/J | CGN、文件、probe_motion；本轮重点 | 现有；封装需新增 / SAM3/CGN（按依赖列） | 服务请求、产物/缓存 | P/H / V |
| `runtime/move_pose.py` | absolute/delta/axis-angle/world_axes/primitive args | preview、move_pose/move_to/rotate_* 单动作与 reach annotations | W/L，米/度/xyzw | 观测、contracts.pose、reach、submit | 现有；封装需新增 / 按 action codec/embodiment | 提交单动作；由 supervisor 执行 | P/H/A / R/C |
| `runtime/motion_submit.py` | 当前 control、编译 pose、action codec | admission、submission、mutation bookkeeping | W/L/J | 文件：submitted-action.json；最多一份待执行 proposal | 现有；封装需新增 / 按 action codec/embodiment | 提交单动作；由 supervisor 执行 | P/H/A / R/C |
| `runtime/motion.py` | gripper directive/recover mode、arm | set_gripper/recover_pose 单动作 | J/W | submit；官方成功 latch 已确认时拒绝继续动作 | 现有；封装需新增 / 按 action codec/embodiment | 提交单动作；由 supervisor 执行 | P/H/A / R/C |
| `runtime/skills.py` | skill 或 target/side/arm | grasp_execute 单 sequence、align_plan reply | W/L/J | skills driver 或 align_plan callback | 现有；封装需新增 / 后端回调（RoboDojo 已有） | 不推进物理；可写探测证据/handle | P / I/D |
| `runtime/lifecycle.py` | compute args、issue text、空 check args、finish args | compute/result、issue ack、task_success、finish directive/预算 | — | 文件；check_task 读已发布判定，不发新 worker 查询 | 现有；封装需新增 / CPU/公共数据 | 读 latch；finish/issue 写控制文件（按入口） | P/H/O / T/C/D |
| `runtime/mutation_compensation.py` | 调用前 Snapshot、失败后 host | 撤销本调用新写 action/ledger，并重开它关闭的观测指针 | — | 文件+内存；保留 action 原子性 | 现有；封装需新增 / 可信 host | 失败补偿，撤销新写 proposal/ledger | H/A / C |
| `runtime/robot_tools.py` | dump_state/inspect/submit/historical read 参数 | 旧 robot-runtime 工具结果/单动作 | P/W/J | 文件/contracts；非默认 19 工具，但不能漏掉现有能力 | 现有；封装需新增 / 按 action codec/embodiment | 提交单动作；由 supervisor 执行 | P/H/A / R/C |
| 所有 `__init__.py` | 导入属性/包名 | 包边界、部分 lazy facade | — | 非能力本体；完整索引保留，禁止新 eager import | 现有；封装需新增 / CPU/公共数据 | 无物理副作用 | P / D |

## 6. 服务与 worker：比默认工具更宽的部分

源码路径本节相对仓库根。下面把“服务可调用”“模型实现可能支持”“没有暴露”分开。

| 底层入口 | 请求 → 响应；单位/坐标 | 默认工具使用情况与 M2/M3 边界 |
| --- | --- | --- |
| `ora/agent/model_services/clients.py` Sam3ApiSegmenter.segment | RGB H×W×3 uint8 + depth H×W + positive point → 单 mask、score、request/service id、timing；P/米 | 客户端把 mask 规范为包含提示点的 8 连通域；原 mask 在此转换丢失，不能把该输出称 HTTP 原始响应 |
| 同类 segment_points；`ora/agent/model_services/resident.py` infer_batch | 同图 1..8 个独立正点 → 每点最多 3 个 multimask alternatives、source/query index、错误；P | 不是一次联合正负点交互。现有 segmentation_batch 去重/裁剪/连通域规范化；lib 应暴露全部客户端候选和逐点失败 |
| 同类 ground；`ora/agent/model_services/grounding_grasp.py` infer_ground | RGB+1..120 字文字+1..3 count → masks/scores/xyxy；P | 服务已先按 score 截到 3 以内，客户端再剔空 mask/裁框/排序；“完整”只能是此协议返回全集 |
| SAM3 框、负点、mask_input、多点联合提示 | 当前 HTTP 三条路由没有对应请求字段/入口 | 模型可能支持不等于 ORA 已支持；本轮不改服务，不能在 lib 假装可调用。geometry_query.region 不代表框提示 |
| `ora/agent/toolkit/host/grasp/contact_graspnet_client.py` propose；`ora/agent/model_services/resident.py` GraspService.infer | scene/target float32 Nx3（各至少16点）、seed、forward_passes=1、max=1..32 → C poses Nx4×4、score/opening/contact、service id/timing | proposer 派生 seed、编译、按 workspace/aperture 拒绝，再截到工具 cap；`_canonical_backend_output` 可保存服务返还全集。服务自己截断前的模型全集不可取，多个 forward pass 明确拒绝 |
| `ora/environment/backends/robodojo/session.py` probe_motion → `sparse_motion.py` | observation id/hash、1..12 query(id,arm,pose W)、exact/parallel_jaw_equivalents → per-variant IK、selected target、plan_ref、failure | 非公开默认工具，通过 grasp feasibility 使用。v2 planner_feedback 带完整 IK diagnostics 引用；应保留全部 variants，不只 selected。写诊断/plan registry，但检查物理时钟不动 |
| 同 session.align_plan → `ora/environment/controllers/alignment.py` | align schema：target_ref、side、arm（可 both）、1..12 任意 W poses → per-arm/source-index reach/error/iterations | 默认 align 工具只生成 1..6 个同一 side 的 roll poses。lib 可暴露回调已有的任意 pose batch，不能扩大 schema 上限；返回 motion_planner_called=false |
| 同 align_plan → `ora/environment/controllers/reach_check.py` | reach schema：arm、最多16个 pose_xyzw/interaction/from_index → route、IK model、matched admission、joint margin/delta、误差与拒绝 | 已被 grasp/move_pose 用到；raw bridge 比高层工具摘要宽。from_index 依赖较早 query 的解，不可排序 query 数组来做键 |
| `ora/environment/backends/robodojo/motion_adapter.py` | align_plan 的只读桥：solve_batch_reachability、reach_check_batch、kinematics | 不能给 agent 直接 import；SDK 执行器的 plan_global、local target、write/step 已随执行器删除 |
| cuRobo 规划/碰撞能力 | 其他 backend 内部实现，不是 RoboDojo 的通用 RPC | 本基线 RoboDojo 使用 SDK IK；不能把设计举例的 cuRobo 当已开放数据接口能力。新增 worker RPC 超出本轮边界 |

SAM3/CGN 的 readiness、binding、model identity、schema、timing 也属于返回语义。
外部服务 replay 按用户修正重新请求，可能得到不同 request_id/timing；新响应与历史
语义 envelope 比对并报告差异，不宣称新响应是历史字节。仿真数据才按历史字节返回。

### 尚不可通过本轮接口开放的能力

| 项目 | 状态 | 后端/权限/副作用 | 记录状态 |
| --- | --- | --- | --- |
| SAM3 框、负点、联合多点、mask_input | 需新增服务协议，本轮不实现 | 当前 ORA 服务无入口；只可用公共图像 | 无该请求历史记录 |
| CGN 超过32候选、多 forward passes | 需新增服务协议，本轮不实现 | 当前服务拒绝，不可绕过 cap | 无该请求历史记录 |
| 独立 FK/关节限位/任意 seed/路径碰撞 RPC | 需新增 worker API，本轮不实现 | worker 内部函数存在≠agent 可直接用；H | 不能承诺任意查询已有记录 |
| RoboDojo 通用 cuRobo/world collision | 不适用当前 backend | cuRobo 在其他后端；现有 response 要求 world_model=none | 无通用场景碰撞证据 |
| 旧 VLA、ask、工具多动作循环 | 不适用本轮 | 无 VLA；ask/多动作预算待定 | 不新增 |
| 工具进程/文件/网络隔离 | 需新增，后续任务 | 本轮同进程；没有 OS 隔离保证 | 负向隔离测试后续 |

## 7. 数据访问和动作的真实时间语义

1. 当前观测入口是 `control/current-observation.json`，数据在 `policy-observations/`，
   policy state 有历史 step/round 索引。RGB、depth、calibration 各自有来源/哈希。
   “时刻”不能只用路径 `current-*`：那些文件会覆盖，M2 需要绑定调用前的不可变引用。
2. target/grasp query 的源观测与内容哈希严格绑定；point history、known plan refs、
   skill commitment/ledger、submitted mutations、round registry 都是工具有效状态。
   重放一个工具只给 RGB-D 不够；M2 时刻快照须说明每项可重建还是未记录。
3. `check_task` 当前从 `control/policy-task-check.json` 读 supervisor 已发布判定。
   无文件时返回 task_checker_unavailable。它不是按调用触发一次独立 worker 判分。
   旧 run 的最后一份 latch 不能冒充任意早期时刻的判定。
4. submit 写 `control/submitted-action.json` 并返回 proposal/工具结果；执行与
   `native/receipts/` 出现在 supervisor 处理之后。数据接口必须分开提交响应与后续
   receipt 读取；不能同步捏造一个执行回执，不能读取未来观测。
5. `grasp_execute` 提交一个 skill sequence，由既有执行路径跑多个 stage；工具层
   改写仍最多提交一次。sequence 内阶段不变成额外模型工具调用，也不改计数口径。
6. 原始客户端、派生 manifest、最终 public_result 是不同层级的 response。
   M2 必须声明字节等价到底钉在哪个接口方法，不能拿 public_result 代替完整 backend
   response。旧 run 从产物只能重建有证据支持的内容，无法无损复原即明确“未记录”。

## 8. 默认 19 工具的对照

这张表对照的是旧混合工具面 `robodojo_placement_closed_skills_v2` 的 19 个工具（该工具面已于 2026-11-01 退役，封存的 run 保留其记录的定义）。
当前集合看 `ora/policy/tool_surfaces/robodojo_placement_toolkit_v5/surface.json`。

| 工具 | 当前 runtime 入口/家族 | 动作/副作用 |
| --- | --- | --- |
| observe | observe._observe | 读观测、派生图像/缓存 |
| mark_point | observe._mark_point → observe/mark_point | 测量、点记录/overlay |
| geometry_query | geometry_query._geometry | 公共 RGB-D 几何 |
| compute | lifecycle._restricted_compute → compute | 受限 CPU 计算/证据 |
| vision_segment | perception | SAM3、target 产物 |
| vision_ground | perception | SAM3 文字、target 产物 |
| grasp_plan | grasp | CGN+编译/选择、可能 reach 查询、query/review 产物 |
| grasp_execute | skills._closed_skill | 一个技能序列动作 |
| align_plan | skills._align_plan → reach/alignment | 只读 IK callback |
| set_gripper | motion | 单动作 |
| move_to | move_pose | 单动作 |
| move_pose | move_pose | 单动作 |
| rotate_wrist | move_pose | 单动作 |
| rotate_pitch | move_pose | 单动作 |
| recover_pose | motion | 单动作 |
| wait_ticks | motion/wait | 单动作 |
| report_issue | lifecycle._report_issue | 追加 issue 文件 |
| check_task | lifecycle._check_task | 读官方判定 latch |
| finish_episode | lifecycle._finish | 写 finish directive，不执行物理动作 |

## 9. 旧 SDK 查漏与下一步

| 旧 SDK 能力 | ORA 可复用位置 | 本轮边界 |
| --- | --- | --- |
| robot_state、capture、read(tool,args) | base/observe + public state + 工具只读部分 | capture 不可隐式推动在线环境；历史只能按记录 |
| grasp.target、precommit_probe、check.grasp/check_hold | skills/candidates、grasp_feasibility、jaw_hold、lift_evidence | 把证据和推断分别暴露；不引入 GT |
| gripper、move、call(stage) | motion_submit、单 sequence、只读工具组合 | 不迁入会 yield 多动作的 Starlark executor |
| vla | 无现有 ORA 对应 | XR1 不迁回 |
| ask(question,images,values) | 本轮无通用工具内模型调用接口 | 用户未定预算/记录口径，不新增 |
| measure、commit、refuse、complete、abort | ToolError、skills receipt、finish evidence | 旧符号不是新 API；不让工具写可信测量 |
| vec_add/sub/scale/norm/dot/cross、quat_mul/axis_angle/rotate、pose_translate、math | compute namespace、contracts.pose、上述纯几何 | 适合 lib，标注 frame/unit，不依赖隐式 host |

M2 先建立可版本化的请求/响应边界与按 tool-call 的 I/O 索引，M3 再把可用能力全部
列入类型化 lib。M4 保持旧默认分派、逐工具等价测试；M5 使用 replay 数据接口：仿真读轨迹，感知重请求并比对。
本清单没有验证 GPU 回放确定性、真实模型数值一致性或进程隔离。

## 10. M2 必须保持的契约（2026-10-23 修正，尚未实现）

- 两种实现命名 **runtime / replay**。simulator 的 observation/proprioception、
  receipt、IK/reach、官方判定：replay 只读当时轨迹字节；缺失报“未记录”。
- SAM3/CGN 等外部服务：replay 从指定边界的输入 blob 和参数重新请求，可问新问题；
  历史响应仅作 comparison。无可比历史记录须标 unavailable，不冒充一致；服务失败
  不回退成历史成功。单元测试只用假服务，不申请服务或 GPU。
- 两种键：语义请求 hash（方法/协议、展开默认参数、输入 blob、服务/配置身份）；
  invocation id（run、round、父工具 call id、接口调用 ordinal、before/after）。
  replay 按序消费 cursor，同参多次不默选第一条，有歧义明确失败。
- replay 动作仅匹配当前 cursor 的原动作才可关联历史 ack/receipt 并推进；否则报分歧。
  proposal ack 与执行 receipt 分开，不能接在线执行端或把原未来接给改过的动作。
- 接口记录固定编码的公开 semantic envelope 和 provenance，数组 blob 保留 dtype、
  shape、order、字节序。只投影获准字段；失败、超时、retry/cache-hit 也记录 occurrence。
- check_task 只读当前边界证据，不把 final verdict 倒灌；RSI 获授权不代表证据存在。
- 不改 `WORKER_KERNEL_PATHS` 中任何文件，也不改
  `ora/orchestrator/episode/action_executor.py`（lane revalidation 依赖）。
- lib 放参数化算法和全部可公开输入输出；默认排序偏好、候选选择、抬升策略留在工具层。

---

# M1 全量源码符号索引

覆盖 108 个 Python 文件、618 个顶层函数或类方法（包含 __init__.py）。

基线：`69912fbc056c54172441248ac58259e715a0f196`，日期：2026-10-23。
这是基线源码的静态 AST 索引；不是承诺公开这些私有函数。
输入参数和返回注解照录，`未注解` 表示源码没有类型注解，不推断不存在的契约。
能力语义、坐标、后端、副作用、权限与记录状态见 [能力清单](#m1-能力清单)。
包初始化、错误类与协议也保留在索引内；嵌套局部函数跟随其外层函数，不另算 API。

## __init__.py

[源码](../tools/__init__.py)

```text
L25 __getattr__(name: str) -> Any
```

## compute/__init__.py

[源码](host/compute/__init__.py)

无函数/方法定义；包说明、常量或类型声明。

## compute/arguments.py

[源码](host/compute/arguments.py)

```text
L32 normalize_block(source: Any) -> tuple[Any, str | None]
L63 read_bindings(value: Any) -> dict[str, Any]
L137 _check_binding_name(name: Any, *, field: str) -> None
L151 _count_elements(value: Any) -> int
L157 _check_binding_value(value: Any, *, field: str) -> None
L166 _check_binding_origin(origin: Any, *, field: str) -> None
L177 _split_bindings(bindings: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]
L217 handle(arguments: Mapping[str, Any]) -> dict[str, Any]
L253 invoke(host: Any, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## compute/bindings.py

[源码](host/compute/bindings.py)

```text
L7 _digest(value: Any) -> str
```

## compute/evaluator.py

[源码](host/compute/evaluator.py)

类 `_NotNumeric`（L243）。

类 `RestrictedComputeRuntime`（L592）。

```text
L72 _reject(message: str, *, received: Any=None, field: str='expressions', **constraints: Any) -> ToolError
L89 _literal_exponent(node: ast.AST) -> float | None
L101 validate_source(source: str) -> ast.Module
L223 _reachable_attributes() -> frozenset[str]
L246 _NotNumeric.__init__(self, reason: str) -> None
L251 _serialise(name: str, value: Any) -> dict[str, Any]
L320 _assign_targets(node: ast.Assign) -> list[ast.AST]
L332 _consumed_names(tree: ast.Module) -> frozenset[str]
L356 _collect(namespace: Mapping[str, Any], last: Any, bound_keys: Sequence[str], consumed: frozenset[str]=frozenset(), selected: Sequence[str] | None=None) -> dict[str, Any]
L447 _resolve_lazy_numpy_imports() -> None
L459 _child_evaluate(tree: ast.Module, bindings: Mapping[str, Any], outputs: Sequence[str] | None=None) -> dict[str, Any]
L490 _worker_main() -> None
L513 _fork_and_run(request: Mapping[str, Any]) -> dict[str, Any]
L595 RestrictedComputeRuntime.__init__(self) -> None
L599 RestrictedComputeRuntime._ensure_worker(self) -> subprocess.Popen[str]
L623 RestrictedComputeRuntime.evaluate(self, source: str, bindings: Mapping[str, Any] | None=None, *, outputs: Sequence[str] | None=None) -> dict[str, Any]
L678 RestrictedComputeRuntime.close(self) -> None
L689 shared_runtime() -> RestrictedComputeRuntime
L712 handle(arguments: Mapping[str, Any]) -> dict[str, Any]
```

## compute/namespace.py

[源码](host/compute/namespace.py)

类 `_Table`（L24）。

```text
L29 _Table.__init__(self, name: str, entries: Mapping[str, Any]) -> None
L33 _Table.__getattr__(self, attr: str) -> Any
L40 _Table.__setattr__(self, attr: str, value: Any) -> None
L43 _Table.__dir__(self) -> list[str]
L86 _vector(value: Any, *, size: int=3) -> np.ndarray
L93 _unit(value: Any, *, size: int=3) -> np.ndarray
L102 _angle_deg(a: Any, b: Any) -> float
L108 _rotate(q_xyzw: Any, vector: Any) -> np.ndarray
L112 _transform_point(translation: Any, q_xyzw: Any, point: Any) -> np.ndarray
L127 _namespace() -> dict[str, Any]
```

## finish/__init__.py

[源码](host/finish/__init__.py)

无函数/方法定义；包说明、常量或类型声明。

## finish/abort_evidence.py

[源码](host/finish/abort_evidence.py)

```text
L25 _event(line: str) -> Mapping[str, Any] | None
L36 _tool_error_signal(event: Mapping[str, Any]) -> list[dict[str, Any]]
L44 _fault_class(receipt: Mapping[str, Any]) -> str | None
L58 _receipt_signals(record: Any) -> list[dict[str, Any]]
L79 round_signals(lines: Iterable[str]) -> list[dict[str, Any]]
L95 recorded_signals(event_path: Path) -> list[dict[str, Any]]
```

## finish/candidates.py

[源码](host/finish/candidates.py)

```text
L43 methods_for(surface: Any) -> tuple[str, ...]
L50 _round_events(lines: Iterable[str]) -> list[Mapping[str, Any]]
L68 _pose(value: Any) -> tuple[list[float], list[float] | None] | None
L92 _targets(value: Any) -> Iterator[tuple[list[float], list[float] | None]]
L104 _candidate_poses(candidate: Mapping[str, Any]) -> list
L109 _angle_deg(first: Sequence[float], second: Sequence[float]) -> float
L117 _matches(target, poses) -> bool
L129 _offer(entries: list[dict[str, Any]], latest: dict[str, dict[str, Any]], event: Mapping[str, Any], call: int) -> None
L146 _try(entry: dict[str, Any], method: str) -> None
L151 ledger(lines: Iterable[str], *, reason: str, methods: Sequence[str]) -> dict[str, Any]
L188 for_host(host: Any, reason: str) -> dict[str, Any]
```

## finish/outcomes.py

[源码](host/finish/outcomes.py)

```text
L54 declared(surface: Any) -> bool
L58 accepted_outcomes(surface: Any) -> tuple[str, ...] | None
L64 reason_problem(outcome: str, reason: Any) -> str | None
L85 names_evidence(text: str) -> bool
L92 _refuse_reason(outcome: str, reason: Any, problem: str, *, without_check_task: bool=False) -> ToolError
L109 _check_arguments(arguments: Any, allowed: tuple[str, ...]) -> str
L130 _check_category(outcome: str, category: Any) -> None
L139 admit(host: Any, arguments: Mapping[str, Any]) -> None
L155 _admit_abort(host: Any, allowed: tuple[str, ...]) -> None
L176 record_fields(host: Any, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## finish/task_check.py

[源码](host/finish/task_check.py)

```text
L46 task_check(host: Any) -> Mapping[str, Any]
L53 refuse_latched(host: Any, arguments: Any) -> None
L70 refuse_unconfirmed(host: Any, outcome: str) -> None
```

## grasp/__init__.py

[源码](../tools/grasp/__init__.py)

无函数/方法定义；包说明、常量或类型声明。

## grasp/aloha_gripper.py

[源码](host/grasp/aloha_gripper.py)

无函数/方法定义；包说明、常量或类型声明。

## grasp/arx_x5_gripper.py

[源码](../tools/grasp/arx_x5_gripper.py)

```text
L120 nominal_jaw_aperture_m(gripper_command: float) -> float
L134 jaw_aperture_admission(calibration: Mapping[str, Any] | None) -> dict[str, Any]
L189 validate_jaw_aperture_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]
L238 arx_x5_link6_compiler_identities(calibration: Mapping[str, Any] | None=None) -> dict[str, dict[str, Any]]
```

## grasp/compilation.py

[源码](host/grasp/compilation.py)

```text
L6 _rotation_matrix_to_quaternion_xyzw(matrix: Any) -> list[float]
L49 _public_grasp_compiler_identity(compiler_identity: Mapping[str, Any]) -> dict[str, Any]
```

## grasp/compiler.py

[源码](host/grasp/compiler.py)

```text
L57 compiled_grasp_geometry(candidate: Mapping[str, Any]) -> dict[str, Any]
L161 compile_grasp_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]
L204 prepare_compiled_grasp_candidates(candidates: list[dict[str, Any]], *, compile_for_execution: bool, compile_candidate: Callable[[Mapping[str, Any]], dict[str, Any]] | None=None, check_geometry: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
L247 _compiled_grasp_geometry(candidate: Mapping[str, Any], *, compiler_identity: Mapping[str, Any]=PANDA_GRIP_SITE_COMPILER_IDENTITY) -> dict[str, Any]
L370 _compile_grasp_candidate(candidate: Mapping[str, Any], *, compiler_identity: Mapping[str, Any]=PANDA_GRIP_SITE_COMPILER_IDENTITY, public_contact_pose: bool=False) -> dict[str, Any]
L460 _prepare_compiled_grasp_candidates(candidates: list[dict[str, Any]], *, compile_for_execution: bool, compiler_identity: Mapping[str, Any]=PANDA_GRIP_SITE_COMPILER_IDENTITY, public_contact_pose: bool=False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
```

## grasp/compiler_registry.py

[源码](../tools/grasp/compiler_registry.py)

```text
L113 _grasp_hover_offset_m(surface: Any, *, grasp_plan_enabled: bool) -> float
```

## grasp/contact_graspnet_client.py

[源码](../tools/grasp/contact_graspnet_client.py)

类 `ContactGraspNetApiProposer`（L30）。

```text
L45 ContactGraspNetApiProposer._check_settings(self) -> None
L55 ContactGraspNetApiProposer.configuration(self) -> Mapping[str, Any]
L73 ContactGraspNetApiProposer._inside(point: Any, bounds: Mapping[str, list[float]]) -> bool
L79 ContactGraspNetApiProposer.propose(self, *, scene_points_camera_m: Any, target_points_camera_m: Any, camera_to_world: Any, target_ref: str, workspace_bounds_m: Mapping[str, list[float]], max_candidates: int, compiler_identity: Mapping[str, Any] | None=None) -> tuple[list[dict[str, Any]], Mapping[str, Any]]
```

## grasp/diagnostics.py

[源码](host/grasp/diagnostics.py)

```text
L33 _family(diagnostics: Mapping[str, Any]) -> dict[str, Any]
L37 hybrid_grasp_diagnostics(*, learned_diagnostics: Mapping[str, Any], geometric_diagnostics: Mapping[str, Any], learned: Sequence[Any], geometric: Sequence[Any], selected: Sequence[Any]) -> dict[str, Any]
```

## grasp/lift_geometry.py

[源码](host/grasp/lift_geometry.py)

```text
L13 _unusable(message: str, candidate_id: Any, **constraints: Any) -> ToolError
L19 lift_outside_workspace(contact: Mapping[str, Any], height_m: float, bounds: Mapping[str, Any]) -> dict[str, Any] | None
L27 split_liftable(candidates: list[dict[str, Any]], compiled: list[dict[str, Any]], height_m: float, bounds: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
L53 _pose(value: Any, what: str, candidate_id: Any=None) -> dict[str, Any]
L74 _lifted(pose: Mapping[str, Any], height_m: float) -> dict[str, Any]
L89 validate_prepared_targets(poses: Mapping[str, Mapping[str, Any]], bounds: Mapping[str, Any], candidate_id: Any=None) -> None
```

## grasp/overlay.py

[源码](../tools/grasp/overlay.py)

类 `_OverlayTarget`（L24）。

类 `_OverlayFonts`（L34）。

```text
L40 _project_world(point: Any, *, intrinsic: Any, extrinsic: Any) -> tuple[int, int] | None
L53 _tinted_target_source(rgb_path: Path, mask_path: Path) -> tuple[Any, Any, Any]
L83 _overlay_camera_matrices(calibration_path: Path) -> tuple[Any, Any]
L96 _overlay_fonts() -> _OverlayFonts
L108 _overlay_target(source: Any, mask: Any, source_array: Any) -> _OverlayTarget
L117 _square_crop(target: _OverlayTarget, pixels: list[tuple[int, int] | None], *, minimum_side: int, target_scale: float) -> tuple[int, int, int, int]
L151 _gripper_wireframe(geometry: Mapping[str, Any], opening: float) -> dict[str, Any]
L183 _new_contact_sheet(*, observation_id: str, target_ref: str, fonts: _OverlayFonts) -> tuple[Any, Any]
L200 _draw_target_panel(image: Any, draw: Any, target: _OverlayTarget, fonts: _OverlayFonts) -> tuple[int, int, int, int]
L222 _draw_contact_sheet_legend(draw: Any, *, compiler_identity: Mapping[str, Any], execution_tool: str, fonts: _OverlayFonts) -> None
L256 _map_crop(pixel: tuple[int, int] | None, crop: tuple[int, int, int, int], origin: tuple[int, int]) -> tuple[int, int] | None
L265 _isometric_pixels(points: Any, bounds_points: Any, origin: tuple[int, int]) -> Any
L285 _draw_candidate_card(image: Any, draw: Any, *, index: int, compiled_records: list[dict[str, Any]], target: _OverlayTarget, intrinsic: Any, extrinsic: Any, target_world: Any, fonts: _OverlayFonts) -> None
L311 _draw_candidate_camera_view(image: Any, draw: Any, *, record: Mapping[str, Any], card_left: int, target: _OverlayTarget, intrinsic: Any, extrinsic: Any, body_font: Any) -> None
L375 _draw_candidate_isometric_view(draw: Any, *, record: Mapping[str, Any], card_left: int, target_world: Any, small_font: Any) -> None
```

## grasp/proposers.py

[源码](../tools/grasp/proposers.py)

类 `GraspProposalBackend`（L20）。

类 `PcaTopDownGraspProposer`（L35）。

类 `AlohaHybridGraspProposer`（L241）。

```text
L24 GraspProposalBackend.configuration(self) -> Mapping[str, Any]
L26 GraspProposalBackend.propose(self, *, scene_points_camera_m: Any, target_points_camera_m: Any, camera_to_world: Any, target_ref: str, workspace_bounds_m: Mapping[str, list[float]], max_candidates: int, compiler_identity: Mapping[str, Any] | None=None) -> tuple[list[dict[str, Any]], Mapping[str, Any]]
L44 PcaTopDownGraspProposer.configuration(self) -> Mapping[str, Any]
L53 PcaTopDownGraspProposer._canonical_axis(axis: Any) -> Any
L63 PcaTopDownGraspProposer._inside(point: Any, bounds: Mapping[str, list[float]]) -> bool
L71 PcaTopDownGraspProposer.propose(self, *, scene_points_camera_m: Any, target_points_camera_m: Any, camera_to_world: Any, target_ref: str, workspace_bounds_m: Mapping[str, list[float]], max_candidates: int, compiler_identity: Mapping[str, Any] | None=None) -> tuple[list[dict[str, Any]], Mapping[str, Any]]
L256 AlohaHybridGraspProposer.configuration(self) -> Mapping[str, Any]
L271 AlohaHybridGraspProposer.close(self) -> None
L276 AlohaHybridGraspProposer.require_ready(self) -> Mapping[str, Any]
L282 AlohaHybridGraspProposer.propose(self, *, scene_points_camera_m: Any, target_points_camera_m: Any, camera_to_world: Any, target_ref: str, workspace_bounds_m: Mapping[str, list[float]], max_candidates: int, compiler_identity: Mapping[str, Any] | None=None) -> tuple[list[dict[str, Any]], Mapping[str, Any]]
```

## grasp/query.py

[源码](../tools/grasp/query.py)

类 `_GraspQueryPoints`（L184）。

类 `_GraspProposals`（L193）。

```text
L51 materialize_grasp_query(*, artifact_root: Path, observation_id: str, target_ref: str, observation_root: Path, observation: Mapping[str, Any], rgb_path: Path, depth_path: Path, calibration_path: Path, workspace_bounds_m: Mapping[str, list[float]], max_candidates: int, backend: GraspProposalBackend, compile_for_execution: bool=False, scene_cameras: Sequence[str]=GRASP_SCENE_CAMERAS, backend_scene_cameras: Sequence[str] | None=None, compiler_id: str | None=None, execution_tool: str='move_free_space', public_contact_pose: bool=False, review_source_binding: Mapping[str, Any] | None=None, candidate_selection: Mapping[str, Any] | None=None, observed_extent: bool=False, review_policy_visible: bool=False, lift_m: float | None=None) -> tuple[dict[str, Any], Path]
L156 _current_grasp_target(*, artifact_root: Path, observation_id: str, target_ref: str, rgb_path: Path, depth_path: Path, calibration_path: Path) -> tuple[Mapping[str, Any], Path, Mapping[str, Any], dict[str, Any]]
L205 _grasp_backend_record(backend: GraspProposalBackend) -> dict[str, Any]
L213 _registered_grasp_compiler(compiler_id: str | None) -> Mapping[str, Any]
L222 _grasp_query_identity(*, observation_id: str, target_ref: str, identity: Mapping[str, Any], actual_source: Mapping[str, Any], backend_input_profile: dict[str, Any], workspace_bounds_m: Mapping[str, list[float]], max_candidates: int, backend_record: dict[str, Any], compiler: Mapping[str, Any], compile_for_execution: bool, execution_tool: str, target: Mapping[str, Any], public_contact_pose: bool, review_source_binding: Mapping[str, Any] | None, candidate_selection: Mapping[str, Any] | None, observed_extent: bool) -> dict[str, Any]
L266 _verified_grasp_query(query_root: Path, manifest_path: Path, query_identity: Mapping[str, Any]) -> tuple[dict[str, Any], Path]
L288 _grasp_query_points(*, mask_path: Path, depth_path: Path, calibration_path: Path, backend_input: Mapping[str, Any]) -> _GraspQueryPoints
L304 _propose_grasp_candidates(*, backend: GraspProposalBackend, points: _GraspQueryPoints, target_ref: str, workspace_bounds_m: Mapping[str, list[float]], max_candidates: int, candidate_selection: Mapping[str, Any] | None, compiler: Mapping[str, Any], compile_for_execution: bool, public_contact_pose: bool, lift_m: float | None=None) -> _GraspProposals
L350 _write_grasp_input_pointcloud(query_root: Path, points: _GraspQueryPoints, target_world: Any) -> Path
L366 _write_grasp_backend_output(query_root: Path, backend_output: Mapping[str, Any] | None, input_regime: Any) -> dict[str, Any] | None
L391 _grasp_input_pointcloud_record(*, query_root: Path, input_pointcloud_path: Path, identity: Mapping[str, Any], target_source_camera: str, proposer_input: _GraspBackendInput, points: _GraspQueryPoints, target_world: Any) -> dict[str, Any]
L481 _grasp_query_manifest(*, query_ref: str, query_identity: dict[str, Any], backend_record: dict[str, Any], proposals: _GraspProposals, backend_output_record: dict[str, Any] | None, visualization: dict[str, Any], review_visualization: dict[str, Any], review_pointcloud: dict[str, Any] | None, input_pointcloud: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]
```

## grasp/results.py

[源码](../tools/grasp/results.py)

```text
L15 grasp_public_result(manifest: Mapping[str, Any], overlay_path: Path) -> dict[str, Any]
L41 grasp_plan_public_result(manifest: Mapping[str, Any], overlay_path: Path) -> dict[str, Any]
```

## grasp/review.py

[源码](../tools/grasp/review.py)

```text
L50 _render_grasp_overlay(*, rgb_path: Path, depth_path: Path, mask_path: Path, calibration_path: Path, observation_id: str, target_ref: str, candidates: list[Mapping[str, Any]], compiler_identity: Mapping[str, Any]=PANDA_GRIP_SITE_COMPILER_IDENTITY, execution_tool: str='move_free_space') -> tuple[Any, dict[str, Any]]
L106 _grasp_overlay_candidate_records(candidates: list[Mapping[str, Any]], colours: Any, *, compiler_identity: Mapping[str, Any], intrinsic: Any, extrinsic: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]
L143 _render_grasp_global_review(*, camera_overlay: Any, camera_panel_resolution: list[int], scene_world: Any, target_world: Any, scene_source_index: Any, scene_sources: list[Mapping[str, Any]], workspace_bounds_m: Mapping[str, list[float]], observation_id: str, target_ref: str, camera: str, candidates: list[Mapping[str, Any]], compiler_identity: Mapping[str, Any]=PANDA_GRIP_SITE_COMPILER_IDENTITY, backend_input_scope: str=GRASP_BACKEND_SINGLE_VIEW_SCOPE, backend_cameras: Sequence[str] | None=None, backend_points_by_camera: Mapping[str, int] | None=None, backend_excluded_cameras: Sequence[Mapping[str, Any]] | None=None, observed_extent: Mapping[str, Any] | None=None, policy_visible: bool=False) -> tuple[Any, dict[str, Any]]
L411 _write_grasp_overlay(*, query_root: Path, rgb_path: Path, depth_path: Path, mask_path: Path, calibration_path: Path, observation_id: str, target_ref: str, candidates: list[Mapping[str, Any]], compiler: Mapping[str, Any], execution_tool: str) -> tuple[Path, Any, dict[str, Any]]
L433 _write_grasp_global_review(*, query_root: Path, observation_root: Path, observation: Mapping[str, Any], workspace_bounds_m: Mapping[str, list[float]], scene_cameras: Sequence[str], camera_to_world: Any, target_world: Any, overlay: Any, observation_id: str, target_ref: str, target_source_camera: str, candidates: list[Mapping[str, Any]], compiler: Mapping[str, Any], proposer_input: _GraspBackendInput, observed_extent: bool, review_policy_visible: bool) -> tuple[dict[str, Any], dict[str, Any] | None]
L485 _write_grasp_review_pointcloud(query_root: Path, fused: Mapping[str, Any], camera_to_world: Any) -> tuple[Any, Path]
L504 _write_grasp_review_image(*, query_root: Path, observation_root: Path, observation: Mapping[str, Any], workspace_bounds_m: Mapping[str, list[float]], scene_cameras: Sequence[str], target_world: Any, overlay: Any, fused: Mapping[str, Any], review_scene_world: Any, observation_id: str, target_ref: str, target_source_camera: str, candidates: list[Mapping[str, Any]], compiler: Mapping[str, Any], proposer_input: _GraspBackendInput, observed_extent: bool, review_policy_visible: bool) -> dict[str, Any]
L555 _grasp_review_pointcloud_record(fused: Mapping[str, Any], scene_cameras: Sequence[str], review_pointcloud_path: Path, query_root: Path) -> dict[str, Any]
```

## grasp/scene_admission.py

[源码](../tools/grasp/scene_admission.py)

类 `_GraspBackendInput`（L192）。

```text
L41 _camera_rgbd_admission(*, observation_root: Path, observation: Mapping[str, Any], camera_name: str) -> tuple[str | None, dict[str, Any] | None]
L113 _admit_backend_scene_cameras(*, observation_root: Path, observation: Mapping[str, Any], workspace_bounds_m: Mapping[str, list[float]], target_source_camera: str, requested_cameras: Sequence[str], voxel_size_m: float, max_depth_m: float) -> dict[str, Any]
L202 _grasp_backend_input(*, observation_root: Path, observation: Mapping[str, Any], workspace_bounds_m: Mapping[str, list[float]], target_source_camera: str, backend_scene_cameras: Sequence[str] | None) -> _GraspBackendInput
```

## grasp/scene_extent_review.py

[源码](../tools/grasp/scene_extent_review.py)

```text
L57 identity(enabled: bool) -> dict[str, Any]
L63 captions(enabled: bool, *, scene_sources: Sequence[Mapping[str, Any]], backend_cameras: Sequence[str] | None, camera: str) -> tuple[str, str]
L79 panel_title(enabled: bool, title: str) -> str
L89 _signed(value: float, digits: int) -> str
L96 fusion_box(*, observation_root: Path, observation: Mapping[str, Any], camera_names: Sequence[str], max_depth_m: float) -> dict[str, list[float]]
L133 cell(*, observation_root: Path, observation: Mapping[str, Any], camera_names: Sequence[str], max_depth_m: float, workspace_bounds_m: Mapping[str, list[float]], target_world: Any, grip_sites: Sequence[tuple[int, tuple[int, int, int], Any]]) -> dict[str, Any]
L189 _outside_counts(points: Any, workspace: Mapping[str, list[float]]) -> dict[str, Any]
L203 _unavailable_image() -> Any
L213 _height_colours(z: Any, low: float, high: float) -> Any
L223 _tick_values(low: float, high: float, scale: float) -> list[float]
L238 _render(*, points: Any, source_index: Any, bounds: Mapping[str, list[float]], workspace: Mapping[str, list[float]], target: Any, grip_sites: Sequence[tuple[int, tuple[int, int, int], Any]]) -> tuple[Any, dict[str, Any]]
```

## grasp/selection.py

[源码](../tools/grasp/selection.py)

```text
L21 apply_candidate_selection(candidates: list[dict[str, Any]], compiled_candidates: list[dict[str, Any]], selection: Mapping[str, Any] | None, max_candidates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]
```

## grasp/x5_topdown.py

[源码](../tools/grasp/x5_topdown.py)

```text
L85 compiler_id(arm: str) -> str
L89 hybrid_backend_arguments(surface: Any) -> dict[str, Any] | None
L99 support_plane_z(target_world: Any, scene_world: Any) -> float | None
L121 topdown_contact_height(target_world: Any, scene_points_camera_m: Any, camera_to_world: Any, profile: Mapping[str, Any]) -> dict[str, Any]
```

## motion/__init__.py

[源码](../tools/motion/__init__.py)

无函数/方法定义；包说明、常量或类型声明。

## motion/arguments.py

[源码](host/motion_rules/arguments.py)

```text
L69 declares_flat_targets(schema: Mapping[str, Any]) -> bool
L81 _reject(message: str, field: str, **constraints: Any) -> ToolError
L88 _one_choice(target: Mapping[str, Any], *, kind: str, modes: Sequence[str], fields: Mapping[str, Sequence[str]], prefix: str) -> dict[str, Any]
L138 arm_target(target: Any, *, arm: str='target') -> dict[str, Any]
L206 arm_target_fields(schema: Mapping[str, Any], *, position: Mapping[str, Any], orientation: Mapping[str, Any], interaction: str | None=None) -> dict[str, Any]
L242 arm_targets(targets: Mapping[str, Any]) -> dict[str, Any]
L278 surface_move_pose_schema(surface: Any) -> dict[str, Any]
L292 surface_flat_targets(surface: Any) -> bool
L309 project_copyable_approach_targets(result: Mapping[str, Any], *, semantic_interaction: bool=False, allow_grip_site_alias: bool=False, move_pose_schema: Mapping[str, Any] | None=None) -> None
```

## motion/clearance.py

[源码](host/motion_rules/clearance.py)

类 `FingerBounds`（L74）。

类 `DepthView`（L83）。

类 `Footprint`（L91）。

类 `RobotVolume`（L110）。

类 `_Candidates`（L125）。

```text
L97 Footprint.points(self) -> Any
L100 Footprint.lowest(self) -> tuple[list[float], str]
L138 root_face(approach_local: Sequence[float]) -> tuple[int, ...]
L146 box_corners(minimum: Sequence[float], maximum: Sequence[float]) -> Any
L152 rotation(quaternion_xyzw: Sequence[float]) -> Any
L165 footprint_at(pose_xyz_xyzw: Sequence[float], *, finger_qpos: Sequence[float], fingers: Sequence[FingerBounds], held_local_corners: Any=None) -> Footprint
L181 robot_volume(solids: Sequence[tuple[str, Any]], *, controlled_point: Sequence[float], extra_points: Sequence[Any]=(), margin_m: float | None=None, root_indices: Sequence[int]=ROOT_FACE) -> RobotVolume
L196 _hull(points: Any) -> Any
L222 _within(points: Any, hull: Any, reach: float) -> Any
L235 _near(points_xy: Any, hulls: Sequence[Any], reach: float) -> Any
L242 _pixel_window(view: DepthView, box_low: Any, box_high: Any) -> tuple[slice, slice]
L256 _world_points(view: DepthView, window: tuple[slice, slice]) -> Any
L273 _in_robot(points: Any, robot: Sequence[RobotVolume]) -> Any
L287 _collect(views: Sequence[DepthView], footprint: Footprint, robot: Sequence[RobotVolume], hulls: Sequence[Any], radius_m: float) -> _Candidates
L312 _observed_fraction(points: Any, hulls: Sequence[Any], radius_m: float, low_xy: Any, high_xy: Any) -> float
L325 measure(views: Sequence[DepthView], *, observation_id: str, footprint: Footprint, robot: Sequence[RobotVolume]=(), radius_m: float | None=None) -> dict[str, Any]
L360 view_from_files(camera: str, depth_path: Any, calibration: Mapping[str, Any]) -> DepthView
```

## motion/clearance_geometry.py

[源码](host/motion_rules/clearance_geometry.py)

```text
L49 load_jaw_geometry_v2(path: Path, convention: str) -> dict[str, JawGeometry]
```

## motion/clearance_runtime.py

[源码](host/motion_rules/clearance_runtime.py)

类 `JawGeometry`（L62）。

类 `HeldExtent`（L76）。

```text
L82 load_jaw_geometry(path: Path) -> dict[str, JawGeometry]
L111 jaws_hold(effector: Mapping[str, Any], geometry: JawGeometry, observation: Mapping[str, Any]) -> bool
L128 _qpos(effector: Mapping[str, Any], geometry: JawGeometry) -> list[float]
L148 _views(observation: Mapping[str, Any], root: Path) -> tuple[list[oc.DepthView], dict[str, Any]]
L170 annotate_observation(observation: Mapping[str, Any], root: Path, *, arm: str, target_pose: Sequence[float], geometry: Mapping[str, JawGeometry], held: Mapping[str, HeldExtent], views: tuple[list[oc.DepthView], dict[str, Any]] | None=None) -> dict[str, Any]
L207 _unknown(reason: str) -> dict[str, Any]
L211 held_local_corners(localization: Mapping[str, Any], attach_pose: Sequence[float], contact_center_local: Sequence[float]) -> tuple[Any, bool] | None
L235 declared(host: Any) -> bool
L240 _holds(host: Any) -> dict[str, dict[str, Any]]
L248 note_gripper(host: Any, targets: Mapping[str, str], *, target_ref: str | None, observation: Mapping[str, Any]) -> None
L273 note_grasp_execute(host: Any) -> None
L291 after_skill(host: Any, skill: str, result: Any) -> Any
L298 _held_extent(host: Any, arm: str, geometry: JawGeometry) -> HeldExtent | None
L317 _geometry(host: Any) -> dict[str, JawGeometry]
L336 _cached_views(host: Any, observation: Mapping[str, Any], root: Path) -> tuple[list[oc.DepthView], dict]
L346 reference(host: Any, arm: str, target_pose_xyz_xyzw: Sequence[float], *, observation: Mapping[str, Any], root: Path) -> dict[str, Any]
```

## motion/gripper_intent.py

[源码](../tools/motion/gripper_intent.py)

```text
L27 extract(arguments: Mapping[str, Any], *, surface: Any) -> tuple[dict[str, Any], str | None, bool]
L55 annotate(result: dict[str, Any], *, targets: Mapping[str, Any], reference: str | None, published: bool) -> dict[str, Any]
```

## motion/move_base.py

[源码](../tools/motion/move_base.py)

```text
L30 _number(arguments: Mapping[str, Any], key: str) -> float
L37 _base_pose(observation: Mapping[str, Any]) -> tuple[float, float, float]
L45 submit(host: Any, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## motion/pose_target_annotations.py

[源码](../tools/motion/pose_target_annotations.py)

```text
L32 merged(host: Any, arm: str, compiled: Mapping[str, Any], reference: Mapping[str, Any], observation: Mapping[str, Any], root: Path) -> dict[str, Any]
```

## motion/refusals.py

[源码](host/motion_rules/refusals.py)

类 `PrimitiveAngleError`（L113）。

```text
L40 _finite_number(value: Any) -> bool
L45 check_rotation_angle(value: Any, field: str) -> None
L58 check_quaternion(value: Any, field: str) -> None
L74 check_flat_request(request: Mapping[str, Any], prefix: str) -> None
L84 target_field(surface: Any, arm: str, name: str | None=None) -> str | None
L92 primitive_field(surface: Any, arm: str, primitive: str) -> str | None
L104 primitive_refusal(error: Exception, *, primitive: str, field: str | None) -> ToolError
L119 _flat_target(schema: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]
L131 move_pose_examples(surface: Any, examples: Any) -> list[dict[str, Any]]
L153 _admitted(candidate: dict[str, Any], schema: Mapping[str, Any], single_arm: bool) -> dict[str, Any] | None
```

## motion/target_state.py

[源码](../tools/motion/target_state.py)

```text
L10 _submit_target_state(self, arguments: Mapping[str, Any], target: Path) -> dict[str, Any]
```

## motion/wait.py

[源码](../tools/motion/wait.py)

```text
L24 submit(host: Any, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## motion/world_axes.py

[源码](../tools/motion/world_axes.py)

```text
L41 declared(surface: Any) -> bool
L45 _local_axes(surface: Any) -> dict[str, list[float]]
L51 _reject(message: str, field: str, **constraints: Any) -> ToolError
L55 _direction(value: Any, field: str) -> list[float]
L66 _dot(left: Sequence[float], right: Sequence[float]) -> float
L70 _cross(left: Sequence[float], right: Sequence[float]) -> list[float]
L75 _perpendicular_jaw(approach: list[float], jaw: list[float], field: str) -> list[float]
L87 quaternion_from_world_axes(approach_world: Any, jaw_line_world: Any, *, local_axes: Mapping[str, Sequence[float]], arm: str) -> list[float]
L105 resolve_request(surface: Any, request: Mapping[str, Any], *, arm: str) -> Mapping[str, Any]
L117 resolve_targets(surface: Any, targets: Mapping[str, Any]) -> dict[str, Any]
L122 _clean(vector: Sequence[float]) -> list[float]
L126 target_axes_world(orientation_xyzw: Sequence[float], *, local_axes: Mapping[str, Sequence[float]]) -> dict[str, Any]
L136 echo(surface: Any, target_pose: Mapping[str, Any]) -> dict[str, Any]
```

## observe/__init__.py

[源码](../tools/observe/__init__.py)

无函数/方法定义；包说明、常量或类型声明。

## observe/arguments.py

[源码](../tools/observe/arguments.py)

```text
L15 solved_history(points: Mapping[str, Any], history: Any) -> list[tuple[str, list[float], tuple[int, int, int]]]
L36 refuse_historical_arguments(arguments: Mapping[str, Any]) -> None
```

## observe/dual_arm.py

[源码](../tools/observe/dual_arm.py)

```text
L8 _dual_arm_observe(self, arguments: Mapping[str, Any], *, allowed: set[str], backend: str) -> dict[str, Any]
```

## observe/mark_point.py

[源码](../tools/observe/mark_point.py)

```text
L60 _quality(sink: dict[str, Any] | None, *, status: str, method: str, **fields: Any) -> None
L80 robust_depth_measurement(depth: Any, *, source_x: int, source_y: int, calibration: Mapping[str, Any], quality_sink: dict[str, Any] | None=None) -> tuple[float, dict[str, Any]]
L193 opencv_camera_point_to_world(camera_point_xyzw: Any, *, calibration: Mapping[str, Any]) -> list[float]
L248 measure(runtime: Any, *, point_id: str, view: str, x: int, y: int, suppress_images: bool=False, quality_sink: dict[str, Any] | None=None) -> dict[str, Any]
L469 execute(runtime: Any, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## observe/mark_point_batch.py

[源码](../tools/observe/mark_point_batch.py)

```text
L74 _reject(message: str, *, field: str | None=None, received: Any=None, **constraints: Any) -> ToolError
L92 _batch_arguments(runtime: Any, arguments: Mapping[str, Any]) -> tuple[str, list[str], list[tuple[int, int]]]
L176 _entry(*, source_index: int, point_id: str, point_uv: tuple[int, int], quality: Mapping[str, Any]) -> dict[str, Any]
L211 execute_batch(runtime: Any, arguments: Mapping[str, Any]) -> dict[str, Any]
L326 _distinct_views(values: Sequence[Any]) -> list[str]
L343 _observation_step(control: Mapping[str, Any]) -> int | None
L353 _attach_stored_evidence(runtime: Any, marks: Sequence[dict[str, Any]]) -> None
```

## observe/mark_point_overlay.py

[源码](../tools/observe/mark_point_overlay.py)

```text
L39 render_panels(runtime: Any, *, root: Path, observation: Mapping[str, Any], views: Sequence[str], selected: Sequence[tuple[str, list[float], tuple[int, int, int]]]) -> list[tuple[str, Path]]
L69 compose_batch_feedback(runtime: Any, *, observation_id: str, panels: Sequence[tuple[str, Path]], labelled_points: Sequence[tuple[str, list[float], tuple[int, int, int]]]) -> Path
L172 begin_feedback_batch(runtime: Any, name: str, calls: Sequence[Mapping[str, Any]]) -> bool
L200 finish_feedback_batch(runtime: Any, name: str, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]
```

## observe/mark_point_schema.py

[源码](../tools/observe/mark_point_schema.py)

```text
L86 batch_input_schema(view_names: Sequence[str]) -> dict[str, Any]
```

## observe/retention_view.py

[源码](../tools/observe/retention_view.py)

```text
L17 enabled(surface: Any) -> bool
```

## perception/__init__.py

[源码](../tools/perception/__init__.py)

无函数/方法定义；包说明、常量或类型声明。

## perception/arguments.py

[源码](../tools/perception/arguments.py)

```text
L22 _is_pixel(value: Any) -> bool
L26 _outside(field: str, value: Any, width: int, height: int) -> ToolError
L35 batch_point_arguments(points: Any, max_candidates: Any, *, width: int, height: int, max_points: int=8, max_per_point: int=3) -> int
L65 region_box(box: Any, *, width: int, height: int) -> list[int]
L89 query_pixel(point: Any, index: int, *, width: int, height: int) -> list[int]
L103 finite_camera_point(camera_point: Any, raw_point: Any, index: int) -> None
L114 finite_pixel(u: float, v: float) -> bool
```

## perception/backends.py

[源码](../tools/perception/backends.py)

类 `PointSegmentationBackend`（L11）。

类 `BatchPointSegmentationBackend`（L22）。

类 `TextGroundingBackend`（L33）。

```text
L15 PointSegmentationBackend.configuration(self) -> Mapping[str, Any]
L17 PointSegmentationBackend.segment(self, *, rgb_u8: Any, depth_m: Any, point_uv: tuple[int, int]) -> tuple[Any, Mapping[str, Any]]
L26 BatchPointSegmentationBackend.configuration(self) -> Mapping[str, Any]
L28 BatchPointSegmentationBackend.segment_points(self, *, rgb_u8: Any, depth_m: Any, points_uv: list[list[int]]) -> tuple[list[dict[str, Any]], Mapping[str, Any]]
L37 TextGroundingBackend.configuration(self) -> Mapping[str, Any]
L39 TextGroundingBackend.ground(self, *, rgb_u8: Any, text: str, max_candidates: int) -> tuple[list[dict[str, Any]], Mapping[str, Any]]
```

## perception/fallback_segmenter.py

[源码](../tools/perception/fallback_segmenter.py)

类 `DepthPromptSegmenter`（L12）。

```text
L28 DepthPromptSegmenter.configuration(self) -> Mapping[str, Any]
L36 DepthPromptSegmenter.segment(self, *, rgb_u8: Any, depth_m: Any, point_uv: tuple[int, int]) -> tuple[Any, Mapping[str, Any]]
```

## perception/feedback.py

[源码](host/perception/feedback.py)

```text
L15 enabled(surface: Any) -> bool
L19 image_label(index: int, role: str, *, camera: str='', candidate_id: str='', target_ref: str='', step: int | None=None) -> dict[str, Any]
L25 grasp_review(result: dict[str, Any], manifest: Mapping[str, Any], overlay: Path) -> None
L60 enrich(host: Any, name: str, arguments: Mapping[str, Any], result: dict[str, Any]) -> None
```

## perception/grounding.py

[源码](../tools/perception/grounding.py)

```text
L42 _render_mask_overlay(*, rgb_u8: Any, mask: Any, point_uv: tuple[int, int], summary: Mapping[str, Any]) -> Any
L69 _render_grounding_overlay(*, rgb_u8: Any, candidates: list[Mapping[str, Any]]) -> Any
L107 materialize_grounded_targets(*, artifact_root: Path, observation_id: str, camera: str, rgb_path: Path, depth_path: Path, calibration_path: Path, text: str, max_candidates: int, backend: TextGroundingBackend, depth_profile: Mapping[str, Any] | None=None) -> tuple[dict[str, Any], list[Path]]
L333 grounding_public_result(manifest: Mapping[str, Any], overlay_path: Path) -> dict[str, Any]
```

## perception/grounding_palette.py

[源码](host/perception/grounding_palette.py)

```text
L67 grounding_candidate_colour(rank: int) -> tuple[int, int, int]
```

## perception/grounding_preview.py

[源码](host/perception/grounding_preview.py)

```text
L80 options(surface: Any) -> dict[str, Any] | None
L89 identity(profile_options: Mapping[str, Any] | None) -> dict[str, Any]
L95 profile_input(profile_options: Mapping[str, Any] | None, *, depth_m: Any, calibration_path: Path, localization: Mapping[str, Any] | None) -> dict[str, Any] | None
L108 crop_box(*, bbox_xyxy: list[int], image_width: int, image_height: int) -> list[int]
L129 render_candidate_preview(*, rgb_u8: Any, candidate: Mapping[str, Any], text: str, points_uv: Sequence[Sequence[int]]=(), depth_profile: Mapping[str, Any] | None=None) -> tuple[Any, dict[str, Any]]
L223 _world_points(depth: Any, rows: Any, columns: Any, intrinsic: Any, camera_to_world: Any) -> Any
L235 _calibration(path: Path) -> tuple[Any, Any]
L248 _tick_step(span_m: float, most: int) -> float
L257 _ticks(low: float, high: float, step: float) -> list[float]
L270 _strip_unavailable(canvas: Any, record: dict[str, Any], reason: str) -> dict[str, Any]
L282 _draw_depth_profile(canvas: Any, *, mask: Any, crop_xyxy: list[int], colour: tuple[int, int, int], profile: Mapping[str, Any]) -> dict[str, Any]
L329 _profile_geometry(*, mask: Any, depth: Any, crop_xyxy: list[int], profile: Mapping[str, Any]) -> dict[str, Any]
L365 _draw_view(canvas: Any, *, left: int, spec: tuple, geometry: Mapping[str, Any], colour: tuple[int, int, int]) -> dict[str, Any]
```

## perception/localization.py

[源码](host/perception/localization.py)

```text
L47 _finite_vector(value: Any, size: int, name: str) -> list[float]
L61 localize(geometry: Mapping[str, Any]) -> dict[str, Any]
L108 measured_jaw_separation_m(record: Any) -> float | None
L141 closure_evidence(*, gripper_opening_m: Any, localization: Mapping[str, Any] | None, near_closed_m: float=NEAR_CLOSED_M) -> dict[str, Any]
```

## perception/mask_localization.py

[源码](host/perception/mask_localization.py)

```text
L16 _mask_summary(mask: Any, depth_m: Any) -> dict[str, Any]
L48 _mask_depth_inlier_selection(*, mask: Any, depth_m: Any) -> tuple[Any, dict[str, Any]]
L111 _mask_depth_localization(*, mask: Any, depth_m: Any, calibration_path: Path, extended_frame: bool=False) -> dict[str, Any]
L201 _footprint(inlier_mask: Any, depth: Any, intrinsic: Any, camera_to_world: Any, world_points: Any) -> dict[str, Any]
L221 _principal_frame(centered_points: Any, *, extended: bool) -> dict[str, Any]
L295 project_target_frame(result: Any, *, declared: bool, support_declared: bool=False, footprint_declared: bool=False) -> Any
```

## perception/overhead_view.py

[源码](host/perception/overhead_view.py)

```text
L56 empty() -> dict[str, Any]
L62 footprint(world_points: Any, camera_position: Any) -> dict[str, Any]
L94 _matrix(pose_xyz_xyzw: Any) -> Any
L104 view_pose(target_xyz: Any, grip_pose_xyz_xyzw: Any, wrist_camera_to_world: Any, *, height_m: float=VIEW_HEIGHT_M) -> dict[str, Any] | None
L149 annotate(result: Any, *, grip_pose_xyz_xyzw: Any, wrist_camera_to_world: Any) -> Any
```

## perception/segmentation_batch.py

[源码](../tools/perception/segmentation_batch.py)

```text
L49 select_contract(surface: Any) -> Any
L59 _normalize(raw: list[dict[str, Any]], *, points_uv: list[list[int]], shape: tuple[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]
L96 _deduplicate(normalized: list[dict[str, Any]]) -> list[dict[str, Any]]
L114 select_masks(raw: list[dict[str, Any]], *, points_uv: list[list[int]], shape: tuple[int, int], max_candidates: int, max_returned: int=MAX_RETURNED_CANDIDATES, max_previews: int=MAX_PREVIEW_IMAGES) -> tuple[list[dict[str, Any]], dict[str, Any]]
L191 _write(path: Path, value: Any) -> None
L203 _render_overview(image: Any, *, points_uv: list[list[int]], best: list[Mapping[str, Any] | None]) -> Any
L242 _materialize_target(*, artifact_root: Path, source: Mapping[str, Any], backend: Mapping[str, Any], camera: str, observation_id: str, rgb: Any, depth: Any, calibration_path: Path, item: Mapping[str, Any], position: int) -> tuple[dict[str, Any], dict[str, Any]]
L294 materialize_batch(*, artifact_root: Path, observation_id: str, camera: str, rgb_path: Path, depth_path: Path, calibration_path: Path, points_uv: list[list[int]], max_candidates: int, backend: backends.BatchPointSegmentationBackend, depth_profile: Mapping[str, Any] | None=None) -> tuple[dict[str, Any], list[Path], bool]
L407 public_result(manifest: Mapping[str, Any], *, cache_reused: bool) -> dict[str, Any]
```

## perception/target_geometry.py

[源码](../tools/perception/target_geometry.py)

```text
L25 segmented_target_geometry(*, artifact_root: Path, target_ref: str, depth_path: Path, calibration_path: Path, voxel_size_m: float=0.01, use_depth_inliers: bool=False, rgb_path: Path | None=None) -> dict[str, Any]
L134 _scene_and_target_points_camera(*, mask_path: Path, depth_path: Path, calibration_path: Path) -> tuple[Any, Any, Any]
L164 _camera_points_to_world(points: Any, camera_to_world: Any) -> Any
L174 _world_points_to_camera(points: Any, camera_to_world: Any) -> Any
```

## perception/target_identity.py

[源码](host/perception/target_identity.py)

```text
L29 unnamed_target() -> dict[str, Any]
L38 resolve_target_semantics(artifact_root: Any, target_ref: str) -> dict[str, Any]
```

## perception/target_store.py

[源码](host/perception/target_store.py)

```text
L26 _safe_path_component(value: str, *, field_name: str) -> str
L36 _segmentation_source_identity(*, observation_id: str, camera: str, rgb_path: Path, depth_path: Path, calibration_path: Path) -> dict[str, Any]
L70 _segmented_target_root(*, artifact_root: Path, observation_id: str, camera: str, target_ref: str) -> Path
L79 _target_index_path(artifact_root: Path, target_ref: str) -> Path
L83 _materialize_target_index(*, artifact_root: Path, target_root: Path, manifest_path: Path, observation_id: str, camera: str, target_ref: str) -> None
L106 _resolve_segmented_target_root(*, artifact_root: Path, target_ref: str) -> Path
L138 load_segmented_target(*, artifact_root: Path, target_ref: str) -> tuple[dict[str, Any], Path, Path]
L161 validate_segmented_target_source(target: Mapping[str, Any], *, observation_id: str, rgb_path: Path, depth_path: Path, calibration_path: Path) -> None
L179 perception_selection_evidence(*, result: Mapping[str, Any], artifact_root: Path, source_size: tuple[int, int], point_uv: Sequence[int] | None=None) -> dict[str, Any]
```

## perception/target_support.py

[源码](host/perception/target_support.py)

```text
L45 empty() -> dict[str, Any]
L51 _ring(mask: Any) -> Any
L68 _fit_plane(points: Any) -> tuple[Any, Any] | None
L89 estimate(*, mask: Any, depth_m: Any, intrinsic: Any, camera_to_world: Any, object_points: Any) -> dict[str, Any]
L134 projection(surface: Any, principal_frame_capability: str) -> dict[str, bool]
```

## reach/__init__.py

[源码](../tools/reach/__init__.py)

无函数/方法定义；包说明、常量或类型声明。

## reach/alignment.py

[源码](host/reach/alignment.py)

```text
L59 frame_local_axes(convention: str) -> dict[str, list[float]]
L65 _quaternion_from_approach(direction: Sequence[float], roll_rad: float, *, local_axes: Mapping[str, Sequence[float]]) -> list[float]
L122 synthesise_approach_poses(centre_xyz_m: Sequence[float], *, approach_side: str, standoff_m: float, candidate_count: int, local_axes: Mapping[str, Sequence[float]]) -> list[list[float]]
L145 validate_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]
L186 build_request(validated: Mapping[str, Any], *, centre_xyz_m: Sequence[float], local_axes: Mapping[str, Sequence[float]]) -> dict[str, Any]
L208 plan(arguments: Mapping[str, Any], *, bridge: Callable[[Mapping[str, Any]], dict[str, Any]] | None, centroid_of: Callable[[str], Sequence[float]], frame_convention: str) -> dict[str, Any]
```

## reach/check.py

[源码](../tools/reach/check.py)

```text
L43 _number(value: Any) -> float | None
L47 _backend_answer(bridge: Bridge, arm: str, queries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]
L69 _blank(index: int, reason: str) -> dict[str, Any]
L75 check_queries(arm: str, queries: Sequence[Mapping[str, Any]], *, bridge: Bridge | None, workspace_bounds_m: Mapping[str, Any]) -> dict[str, Any]
L148 _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]
L155 _rotate(orientation: Sequence[float], vector: Sequence[float]) -> list[float]
L161 _member(orientation: Sequence[float], member: str, axis: Sequence[float]) -> list[float]
L170 approach_tilt_deg(candidate: Mapping[str, Any], axis: Sequence[float]) -> float
L177 _stage(row: Mapping[str, Any] | None) -> dict[str, Any] | None
L181 _pairs(candidates: Sequence[Mapping[str, Any]], member: str, axis: Sequence[float]) -> list[dict[str, Any]]
L197 _check_member(arm: str, candidates: Sequence[Mapping[str, Any]], member: str, *, bridge: Bridge, workspace_bounds_m: Mapping[str, Any], axis: Sequence[float]) -> tuple[list[tuple[dict, dict]], str | None, bool | None]
L216 _released_verdicts(arm: str, candidates: Sequence[Mapping[str, Any]], *, bridge: Bridge, workspace_bounds_m: Mapping[str, Any], axis: Sequence[float]) -> tuple
L246 _symmetric_verdicts(arm: str, candidates: Sequence[Mapping[str, Any]], *, bridge: Bridge, workspace_bounds_m: Mapping[str, Any], axis: Sequence[float], rule: Mapping[str, Any]) -> tuple
L276 _publish_jaw_flip(candidate: dict[str, Any], axis: Sequence[float]) -> None
L283 select_grasp_candidates(compiled: Sequence[Mapping[str, Any]], *, arm: str, bridge: Bridge | None, workspace_bounds_m: Mapping[str, Any], approach_axis_local: Sequence[float], jaw_symmetry: Mapping[str, Any] | None=None) -> dict[str, Any]
L369 grasp_candidate_selection(host: Any, arm: Any, observation: Any=None) -> dict[str, Any] | None
L401 publish_grasp_selection(host: Any, result: dict[str, Any], manifest: Mapping[str, Any]) -> None
```

## reach/jaw_center.py

[源码](host/reach/jaw_center.py)

类 `JawCenterError`（L44）。

```text
L48 load_descriptor(view_config: Mapping[str, Any], *, project_root: Path=_ROOT) -> dict[str, Any]
L61 grip_centre_offsets(descriptor: Mapping[str, Any], convention: str) -> dict[str, list[float]]
L76 jaw_contact_center_offsets(view_config: Mapping[str, Any], *, project_root: Path=_ROOT) -> dict[str, list[float]]
L111 jaw_center_offsets(view_config: Mapping[str, Any], *, project_root: Path=_ROOT) -> dict[str, list[float]]
L121 approach_and_offset(surface: Any, arm: str) -> tuple[np.ndarray, np.ndarray]
```

## reach/jaw_symmetry.py

[源码](../tools/reach/jaw_symmetry.py)

```text
L75 measured_arm_joints(observation: Any, arm: str) -> list[float] | None
L88 rule_for(surface: Any, arm: str, observation: Any) -> dict[str, Any] | None
L102 _stage_headroom(row: Mapping[str, Any], reach: float | None, bound: float | None) -> dict[str, Any]
L108 member_headroom(pregrasp: Mapping[str, Any], approach: Mapping[str, Any], rule: Mapping[str, Any]) -> dict[str, dict[str, Any]]
L126 _smallest(values: Sequence[Any]) -> float | None
L130 _compare(rows: Mapping[str, Sequence[Mapping[str, Any]]], joint6: Mapping[str, Any], headroom: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]
L148 choose(rows: Mapping[str, Sequence[Mapping[str, Any]]], rule: Mapping[str, Any]) -> dict[str, Any]
L167 unavailable() -> dict[str, Any]
L176 summary() -> dict[str, str]
```

## reach/move_pose.py

[源码](../tools/reach/move_pose.py)

```text
L76 _round(value: float | None) -> float | None
L80 _joint6_margin(row: Mapping[str, Any], measured: Sequence[float] | None, bound: float | None) -> float | None
L88 _jaw_choice(rows: Mapping[str, Mapping[str, Any]], margins: Mapping[str, float | None]) -> tuple[str, str]
L102 _slerp(start: Sequence[float], end: Sequence[float], fraction: float) -> list[float]
L118 _unit(quaternion: Sequence[float]) -> list[float]
L123 _rotation_rad(start: Sequence[float], end: Sequence[float]) -> float
L127 reach_along_request(arm: str, *, current_pose: Sequence[float], target_pose: Sequence[float], interaction: str, target_row: Mapping[str, Any], bridge: Bridge, workspace_bounds_m: Mapping[str, Any]) -> dict[str, Any]
L159 _unsolved_reference(host: Any, arm: str, *, current_pose: Sequence[float], target_pose: Sequence[float], interaction: str, target_row: Mapping[str, Any], bridge: Bridge, v2_answer: Mapping[str, Any] | None=None) -> dict[str, Any]
L176 primitive_reference(host: Any, arm: str, compiled: Mapping[str, Any], *, current_pose: Sequence[float], interaction: str) -> dict[str, Any]
L202 declared(surface: Any) -> bool
L207 plan_target(host: Any, arm: str, *, current_pose: Sequence[float], target_pose: Sequence[float], interaction: str, orientation_mode: str, observation: Mapping[str, Any]) -> tuple[list[float], dict[str, Any]]
L272 apply_reference(host: Any, arm: str, compiled: Mapping[str, Any], *, current_pose: Sequence[float], interaction: str, observation: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
```

## reach/orientation.py

[源码](../tools/reach/orientation.py)

```text
L31 jaw_geometry(surface: Any, arm: str) -> tuple[np.ndarray, np.ndarray]
L41 _axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray
L47 turn_axes(approach_world: np.ndarray, directions: int=TURN_DIRECTIONS) -> list[np.ndarray]
L58 turned_candidates(target_pose: Sequence[float], *, approach_local: np.ndarray, offset_local: np.ndarray, rotation_deg: float, directions: int=TURN_DIRECTIONS) -> tuple[list[dict[str, Any]], np.ndarray]
L74 _preference(row: Mapping[str, Any]) -> tuple[float, float]
L79 _found(angle: float, candidate: Mapping[str, Any], *, reachable: int, center: np.ndarray, queries: int, claim: str=CLAIM) -> dict[str, Any]
L94 nearest_orientation(arm: str, *, target_pose: Sequence[float], interaction: str, approach_local: np.ndarray, offset_local: np.ndarray, bridge: Bridge | None, workspace_bounds_m: Mapping[str, Any], directions: int=TURN_DIRECTIONS, claim: str=CLAIM) -> dict[str, Any]
L122 reference(host: Any, arm: str, *, target_pose: Sequence[float], interaction: str) -> dict[str, Any]
```

## reach/probe_feedback.py

[源码](host/reach/probe_feedback.py)

```text
L17 probe_request_fields(surface: Any) -> dict[str, str]
L26 _ik_candidates(variants: list[Any]) -> dict[str, Any]
L57 summarize_motion_probe(result: Mapping[str, Any], *, ik_candidates: bool=False) -> list[dict[str, Any]]
```

## reach/reference.py

[源码](../tools/reach/reference.py)

```text
L35 move_pose_declared(surface: Any) -> bool
L39 primitive_declared(surface: Any) -> bool
L43 basis(answer: Mapping[str, Any]) -> dict[str, Any]
L52 solved(answer: Mapping[str, Any]) -> dict[str, Any]
L57 unsolved(host: Any, arm: str, reach: Mapping[str, Any], *, answer: Mapping[str, Any], target_pose: Sequence[float], interaction: str) -> dict[str, Any]
L64 nearest(host: Any, arm: str, *, target_pose: Sequence[float], interaction: str) -> dict[str, Any]
```

## runtime/__init__.py

[源码](../tools/runtime/__init__.py)

```text
L24 __getattr__(name: str) -> Any
```

## runtime/backend_selection.py

[源码](host/backend_selection.py)

```text
L5 dual_arm_grasp_scene_cameras(surface: Any, arm: str) -> tuple[str, str]
L13 dual_arm_grasp_compiler_id(surface: Any, arm: str) -> str | None
L27 conform_to_result_schema(schema: Any, result: dict[str, Any]) -> dict[str, Any]
```

## runtime/base.py

[源码](../tools/runtime/base.py)

类 `RobotToolRuntime`（L92）。

```text
L47 manual_sections(path: Path) -> tuple[str, ...]
L61 tool_definitions(allowed_tools: Sequence[str] | None=None, *, manual_path: Path | None=None, surface: ResolvedToolSurface | None=None, profile_id: str='robot_runtime_v2', component_overrides: Mapping[str, Any] | None=None) -> list[dict[str, Any]]
L105 RobotToolRuntime.__post_init__(self) -> None
L139 RobotToolRuntime.definitions(self) -> list[dict[str, Any]]
L144 RobotToolRuntime.per_round_control_files(self) -> tuple[str, ...]
L158 RobotToolRuntime.round_registry_names(self) -> tuple[str, ...]
L163 RobotToolRuntime.residual_round_registries(self) -> list[str]
L170 RobotToolRuntime.round_start_isolation(self) -> dict[str, Any]
L205 RobotToolRuntime.reset_round(self, next_round_index: int) -> None
L224 RobotToolRuntime.classify_call_effect(self, name: str, arguments: Mapping[str, Any]) -> str
L236 RobotToolRuntime.invoke(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]
L308 RobotToolRuntime._current(self) -> tuple[dict[str, Any], Path, dict[str, Any]]
L333 RobotToolRuntime._select_observation(self, step: Any) -> tuple[int | None, dict[str, Any] | None, Path, dict[str, Any]]
L391 RobotToolRuntime._camera(root: Path, observation: Mapping[str, Any], camera: Any) -> tuple[dict[str, Any], Path, Path, Path]
```

## runtime/common.py

[源码](../tools/runtime/common.py)

```text
L26 _quat_matrix(xyzw: list[float]) -> Any
L32 _matrix_quat(matrix: Any) -> list[float]
L36 load_policy_state(*args: Any, **kwargs: Any) -> dict[str, Any]
L50 verify_policy_state(*args: Any, **kwargs: Any) -> dict[str, Any]
```

## runtime/core.py

[源码](../tools/runtime/core.py)

类 `ManipulationToolRuntime`（L155）。

```text
L177 ManipulationToolRuntime.__post_init__(self) -> None
L334 ManipulationToolRuntime.definitions(self) -> list[dict[str, Any]]
L337 ManipulationToolRuntime.per_round_control_files(self) -> tuple[str, ...]
L346 ManipulationToolRuntime.round_registry_names(self) -> tuple[str, ...]
L360 ManipulationToolRuntime.reset_round(self, next_round_index: int) -> None
L384 ManipulationToolRuntime.classify_call_effect(self, name: str, arguments: Mapping[str, Any]) -> str
L394 ManipulationToolRuntime.invoke(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]
L469 ManipulationToolRuntime._validate_public_arguments(self, name: str, arguments: Mapping[str, Any]) -> None
```

## runtime/factory.py

[源码](../tools/runtime/factory.py)

```text
L15 build_tool_runtime(*, control_root: Path, manual_path: Path, event_path: Path, workspace_bounds_m: Mapping[str, list[float]], allowed_tools: Sequence[str] | None, state_root: Path | None, profile_id: str, component_overrides: Mapping[str, Any] | None, prerequisite_binding: Mapping[str, Any] | None=None, controlled_end_effector: str | None=None, motion_probe_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None=None, align_plan_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None=None, public_facts_projector: Callable[..., Mapping[str, Any]] | None=None) -> ToolRuntime
```

## runtime/geometry_query.py

[源码](../tools/runtime/geometry_query.py)

类 `GeometryQueryMixin`（L24）。

```text
L27 GeometryQueryMixin._geometry(self, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## runtime/grasp.py

[源码](../tools/runtime/grasp.py)

类 `_GraspReviewSources`（L53）。

类 `GraspRuntimeMixin`（L97）。

```text
L66 _grasp_request_fields(arguments: Mapping[str, Any], *, current_observation_id: str) -> tuple[str, str, int]
L100 GraspRuntimeMixin._grasp_plan(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L143 GraspRuntimeMixin._require_grasp_argument_names(self, arguments: Mapping[str, Any]) -> None
L162 GraspRuntimeMixin._grasp_target_camera(self, target_ref: str, observation_id: str) -> str
L187 GraspRuntimeMixin._grasp_scene_cameras(self, arm: Any, root: Path, observation: Mapping[str, Any]) -> tuple[str, ...]
L205 GraspRuntimeMixin._grasp_review_sources(self, *, root: Path, observation: Mapping[str, Any], camera: str, scene_cameras: tuple[str, ...], target_ref: str='') -> _GraspReviewSources
L254 GraspRuntimeMixin._usable_grasp_review_source(self, root: Path, observation: Mapping[str, Any], name: str) -> dict[str, Any]
L273 GraspRuntimeMixin._materialize_grasp_manifest(self, *, arm: Any, observation_id: str, target_ref: str, root: Path, observation: Mapping[str, Any], rgb_path: Path, depth_path: Path, calibration_path: Path, max_candidates: int, scene_cameras: tuple[str, ...], review_sources: _GraspReviewSources) -> tuple[dict[str, Any], Path]
L332 GraspRuntimeMixin._record_grasp_materialization(self, *, observation_id: str, target_ref: str, manifest: Mapping[str, Any], overlay_path: Path, review_source_binding: dict[str, Any] | None) -> Any
L379 GraspRuntimeMixin._attach_grasp_review_evidence(self, result: dict[str, Any], manifest: Mapping[str, Any], *, review: Any, camera: str, scene_cameras: tuple[str, ...], review_sources: _GraspReviewSources) -> None
L405 GraspRuntimeMixin._qualify_grasp_plan_candidates(self, result: dict[str, Any], manifest: Mapping[str, Any], arm: Any) -> None
L434 GraspRuntimeMixin._grasp_compiler_id(self, arm: str) -> str
L449 GraspRuntimeMixin._motion_probe(self, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## runtime/lifecycle.py

[源码](../tools/runtime/lifecycle.py)

类 `LifecycleRuntimeMixin`（L39）。

```text
L42 LifecycleRuntimeMixin._restricted_compute(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L49 LifecycleRuntimeMixin._report_issue(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L66 LifecycleRuntimeMixin._check_task(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L76 LifecycleRuntimeMixin._runtime_remaining_budget(self) -> dict[str, Any]
L117 LifecycleRuntimeMixin._finish(self, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## runtime/motion.py

[源码](../tools/runtime/motion.py)

类 `MotionRuntimeMixin`（L20）。

```text
L23 MotionRuntimeMixin._atomic_gripper(self, arguments: Mapping[str, Any], *, target: float) -> dict[str, Any]
L38 MotionRuntimeMixin._atomic_close_gripper(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L43 MotionRuntimeMixin._atomic_open_gripper(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L48 MotionRuntimeMixin._gripper_atomic_v3(self, arguments: Mapping[str, Any], *, directive: str) -> dict[str, Any]
L99 MotionRuntimeMixin._motion_set_gripper(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L102 MotionRuntimeMixin._motion_set_gripper_v2(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L131 MotionRuntimeMixin._motion_recover_pose_v2(self, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## runtime/motion_submit.py

[源码](../tools/runtime/motion_submit.py)

类 `MotionSubmissionRuntimeMixin`（L27）。

```text
L30 MotionSubmissionRuntimeMixin._load_previews(self) -> dict[str, Any]
L33 MotionSubmissionRuntimeMixin._ensure_mutation_available(self, tool: str) -> None
L44 MotionSubmissionRuntimeMixin._atomic_arm(self, observation: Mapping[str, Any]) -> str
L66 MotionSubmissionRuntimeMixin._atomic_pose_arguments(self, arguments: Mapping[str, Any], *, current_pose: list[float], arm: str) -> dict[str, Any]
L134 MotionSubmissionRuntimeMixin._compile_pose_v2(self, arguments: Mapping[str, Any], *, current_pose: list[float], arm: str) -> tuple[dict[str, Any], dict[str, Any]]
L160 MotionSubmissionRuntimeMixin._atomic_compiled_target(self, arguments: Mapping[str, Any], *, current_pose: list[float], arm: str) -> tuple[dict[str, Any], dict[str, Any] | None]
L174 MotionSubmissionRuntimeMixin._atomic_submit(self, *, control: Mapping[str, Any], action_name: str, arguments: Mapping[str, Any], compiled_pose: Mapping[str, Any] | None=None, schema: str='ora-action.v3', public_action_name: str | None=None) -> dict[str, Any]
L221 MotionSubmissionRuntimeMixin._surface_targets(self, arguments: Mapping[str, Any], *, extra: frozenset[str]=frozenset(), scalar_key: str | None=None) -> dict[str, Any]
L260 MotionSubmissionRuntimeMixin._surface_arm_states(self, observation: Mapping[str, Any]) -> dict[str, list[float]]
L282 MotionSubmissionRuntimeMixin._sparse_targets(arguments: Mapping[str, Any], *, extra: frozenset[str]=frozenset()) -> dict[str, Any]
L303 MotionSubmissionRuntimeMixin._public_end_effector_compilation(compiled: Mapping[str, Any], *, delta_frame: str | None=None) -> dict[str, Any]
L320 MotionSubmissionRuntimeMixin._public_sparse_pose_compilation(self, compiled: Mapping[str, Any], *, delta_frame: str | None=None) -> dict[str, Any]
L332 MotionSubmissionRuntimeMixin._validate_target_workspace(self, target_state: Mapping[str, Any], *, current_position_m: Sequence[float] | None=None, field: str | None='xyz_m') -> None
```

## runtime/move_pose.py

[源码](../tools/runtime/move_pose.py)

类 `MovePoseRuntimeMixin`（L70）。

```text
L73 MovePoseRuntimeMixin._atomic_preview_move(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L116 MovePoseRuntimeMixin._atomic_move_pose(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L162 MovePoseRuntimeMixin._motion_move_pose_v2(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L180 MovePoseRuntimeMixin._motion_move_pose_v2_admit(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L272 MovePoseRuntimeMixin._submit_pose_primitive_v2(self, arguments: Mapping[str, Any], *, primitive: str, request_for_target: Callable[[Any], Mapping[str, Any]], flat_scalar_key: str | None=None) -> dict[str, Any]
L343 MovePoseRuntimeMixin._move_to(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L346 MovePoseRuntimeMixin._motion_move_to_primitive_v2(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L367 MovePoseRuntimeMixin._motion_rotate_wrist_v2(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L384 MovePoseRuntimeMixin._motion_rotate_pitch_v2(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L391 MovePoseRuntimeMixin._motion_rotate_primitive_v2(self, arguments: Mapping[str, Any], *, primitive: str, axis: str) -> dict[str, Any]
```

## runtime/mutation_compensation.py

[源码](../tools/runtime/mutation_compensation.py)

类 `Snapshot`（L22）。

```text
L28 snapshot(host: Any, name: str, arguments: Mapping[str, Any]) -> Snapshot
L42 restore(host: Any, taken: Snapshot) -> None
```

## runtime/observe.py

[源码](../tools/runtime/observe.py)

类 `ObserveRuntimeMixin`（L36）。

```text
L39 ObserveRuntimeMixin._rgb_output_resolution(self, source: Path) -> tuple[int, int]
L49 ObserveRuntimeMixin._derived_resolution(self) -> tuple[int, int]
L54 ObserveRuntimeMixin._image_dimensions(path: Path) -> dict[str, int]
L61 ObserveRuntimeMixin._current_grip_state(self) -> tuple[dict[str, Any], Path, dict[str, Any], list[float], float]
L73 ObserveRuntimeMixin._dual_arm_states(observation: Mapping[str, Any]) -> dict[str, list[float]]
L92 ObserveRuntimeMixin._pointcloud_artifact_key(self, observation: Mapping[str, Any]) -> str
L105 ObserveRuntimeMixin._pointcloud(self, root: Path, observation: Mapping[str, Any]) -> dict[str, Any]
L145 ObserveRuntimeMixin._write_overlay(self, *, source: Path, observation_id: str, identity: Mapping[str, Any], points: list[tuple[int, int, str, tuple[int, int, int]]], output_resolution: tuple[int, int] | None=None) -> Path
L194 ObserveRuntimeMixin._annotated_rgb(self, *, root: Path, observation: Mapping[str, Any], camera: str, labelled_world_points: list[tuple[str, list[float], tuple[int, int, int]]], identity_kind: str) -> Path
L247 ObserveRuntimeMixin._annotated_pointcloud(self, *, observation: Mapping[str, Any], pointcloud: Mapping[str, Any], view: str, labelled_world_points: list[tuple[str, list[float], tuple[int, int, int]]], identity_kind: str) -> Path
L285 ObserveRuntimeMixin.begin_read_only_feedback_batch(self, name: str, calls: Sequence[Mapping[str, Any]]) -> bool
L290 ObserveRuntimeMixin.finish_read_only_feedback_batch(self, name: str, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]
L295 ObserveRuntimeMixin._attach_observe_public_facts(self, result: dict[str, Any], root: Path, observation: Mapping[str, Any]) -> dict[str, Any]
L330 ObserveRuntimeMixin._observe(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L388 ObserveRuntimeMixin._retention_observation(self, step: Any) -> 未注解
L402 ObserveRuntimeMixin._retention_observe(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L434 ObserveRuntimeMixin._physical_policy_cameras(self) -> set[str]
L439 ObserveRuntimeMixin._robotwin_policy_cameras(self) -> set[str]
L443 ObserveRuntimeMixin._dual_arm_gripper_target(effector: Mapping[str, Any]) -> str
L455 ObserveRuntimeMixin._dual_arm_observe(self, arguments: Mapping[str, Any], *, allowed: set[str], backend: str) -> dict[str, Any]
L461 ObserveRuntimeMixin._load_points(self) -> dict[str, Any]
L467 ObserveRuntimeMixin._mark_point(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L470 ObserveRuntimeMixin._marked_point_workspace(self, xyz_m: Sequence[float]) -> dict[str, Any]
```

## runtime/perception.py

[源码](../tools/runtime/perception.py)

类 `PerceptionRuntimeMixin`（L49）。

```text
L52 PerceptionRuntimeMixin._vision_segment(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L62 PerceptionRuntimeMixin._overhead_view_poses(self, result: Any) -> Any
L83 PerceptionRuntimeMixin._vision_segment_batch(self, arguments: Mapping[str, Any], contract: Any) -> dict[str, Any]
L127 PerceptionRuntimeMixin._vision_ground(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L235 PerceptionRuntimeMixin._current_target_geometry(self, *, control: Mapping[str, Any], root: Path, observation: Mapping[str, Any], target_ref: str) -> dict[str, Any]
```

## runtime/protocol.py

[源码](../tools/runtime/protocol.py)

类 `ToolRuntime`（L8）。

```text
L11 ToolRuntime.definitions(self) -> Sequence[Mapping[str, Any]]
L12 ToolRuntime.classify_call_effect(self, name: str, arguments: Mapping[str, Any]) -> str
L15 ToolRuntime.invoke(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]
```

## runtime/robot_tools.py

[源码](../tools/runtime/robot_tools.py)

类 `RobotToolsMixin`（L21）。

```text
L25 RobotToolsMixin._dump_state(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L74 RobotToolsMixin._inspect(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L154 RobotToolsMixin._manual(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L177 RobotToolsMixin._submit(self, arguments: Mapping[str, Any]) -> dict[str, Any]
L186 RobotToolsMixin._submit_target_state(self, arguments: Mapping[str, Any], target: Path) -> dict[str, Any]
L192 RobotToolsMixin._observed_arm_position(observation: Mapping[str, Any], arm: str) -> list[float] | None
L207 RobotToolsMixin._persist_action(self, target: Path, control: dict[str, Any], action: Mapping[str, Any]) -> dict[str, Any]
```

## runtime/skills.py

[源码](../tools/runtime/skills.py)

类 `SkillRuntimeMixin`（L20）。

```text
L23 SkillRuntimeMixin._closed_skill(self, skill: str, arguments: Mapping[str, Any]) -> dict[str, Any]
L29 SkillRuntimeMixin.drive_skill_stage(self, *args: Any, **kwargs: Any) -> dict[str, Any]
L33 SkillRuntimeMixin.build_skill_receipt(self, **kwargs: Any) -> dict[str, Any]
L37 SkillRuntimeMixin._align_plan(self, arguments: Mapping[str, Any]) -> dict[str, Any]
```

## shared/__init__.py

[源码](host/shared/__init__.py)

无函数/方法定义；包说明、常量或类型声明。

## shared/artifact_errors.py

[源码](host/shared/artifact_errors.py)

类 `TargetRefMalformed`（L23）。

类 `TargetRefUnknown`（L27）。

类 `ArtifactIntegrityError`（L31）。

类 `ArtifactHashMismatch`（L37）。

类 `TargetInsufficient`（L43）。

无函数/方法定义；包说明、常量或类型声明。

## shared/artifact_io.py

[源码](host/shared/artifact_io.py)

```text
L24 _canonical_hash(value: Mapping[str, Any]) -> str
L31 _hash(path: Path) -> str
L39 _file_hash(path: Path) -> str
L43 _temporary_path(path: Path) -> Path
L47 _atomic_json(path: Path, value: Mapping[str, Any]) -> None
L61 _atomic_npy(path: Path, value: Any) -> None
L74 _atomic_npz(path: Path, **arrays: Any) -> None
L87 _atomic_png(path: Path, image: Any) -> None
L97 _artifact(path: Path, root: Path) -> dict[str, Any]
L105 _verify_artifact(root: Path, reference: Mapping[str, Any]) -> Path
L117 _png_base64(path: Path) -> str
L121 _append(path: Path, value: Mapping[str, Any]) -> None
```

## shared/errors.py

[源码](../error_classes.py)

类 `ToolError`（L60）。

```text
L63 ToolError.__init__(self, message: str, *, code: str='invalid_arguments', retryable: bool=True, field: str | None=None, received: Any=None, constraints: Mapping[str, Any] | None=None, retry_requires_changed_arguments: bool | None=None) -> None
L79 ToolError.as_dict(self) -> dict[str, Any]
L96 schema_refusal(error: Any) -> ToolError
L120 blank_text_refusal(name: str, value: Any, *, message: str) -> ToolError
L134 argument_names(arguments: Mapping[str, Any]) -> list[str]
L140 _refuse(arguments: Mapping[str, Any], message: str, constraints: Mapping[str, Any]) -> ToolError
L149 require_exactly_one_of(arguments: Mapping[str, Any], names: Sequence[str], *, message: str) -> str
L162 require_at_most_one_of(arguments: Mapping[str, Any], names: Sequence[str], *, message: str) -> str | None
L175 require_known_arguments(arguments: Mapping[str, Any], *, allowed: Collection[str], required: Collection[str], message: str) -> None
L190 _finite_vector(value: Any, length: int, label: str) -> list[float]
```

## shared/rgbd_fusion.py

[源码](host/shared/rgbd_fusion.py)

```text
L20 _sha(path: Path) -> str
L24 _atomic_json(path: Path, value: Mapping[str, Any]) -> None
L34 _artifact(path: Path, root: Path) -> dict[str, Any]
L42 fuse_rgbd_pointcloud_sources(*, observation_root: Path, observation: Mapping[str, Any], workspace_bounds_m: Mapping[str, list[float]], camera_names: Sequence[str]=('agentview', 'wrist'), voxel_size_m: float=0.01, max_depth_m: float | None=None) -> dict[str, Any]
L196 materialize_two_view_pointcloud(*, observation_root: Path, observation: Mapping[str, Any], output_root: Path, workspace_bounds_m: Mapping[str, list[float]], voxel_size_m: float=0.01, resolution: int | Sequence[int]=512) -> dict[str, Any]
L283 orthographic_click_constraint(view: Mapping[str, Any], x: int, y: int) -> dict[str, float]
```

## shared/saved_steps.py

[源码](host/shared/saved_steps.py)

```text
L30 current_step(host: Any) -> int | None
L46 route_current(host: Any, step: Any) -> Any
L52 round_timeline(host: Any, control: Mapping[str, Any], *, subject: str='saved-step selection') -> tuple[list[dict[str, Any]], int]
L86 round_step_refusal(step: Any, states: list[Mapping[str, Any]], round_start: int) -> ToolError | None
L102 saved_step_refusal(host: Any, step: Any, error: Exception) -> ToolError
```

## shared/service_refusals.py

[源码](host/shared/service_refusals.py)

```text
L38 refusal(error: ValueError, *, code: str, target_ref: Any=None) -> ToolError
```

## shared/target_refusals.py

[源码](host/shared/target_refusals.py)

类 `GeometryQualityRefusal`（L36）。

```text
L39 GeometryQualityRefusal.__init__(self, geometry: Mapping[str, Any]) -> None
L44 stale_target(*, target_ref: str, identity: Any, current_observation_id: str, field: str='target_ref') -> ToolError
L67 unmeasurable_geometry(*, target_ref: str, geometry: Mapping[str, Any]) -> ToolError
L91 current_target_refs(artifact_root: Path, observation_id: str) -> list[str]
L106 unavailable_target(error: ValueError, *, target_ref: Any, field: str='target_ref', artifact_root: Path | None=None, observation_id: str | None=None) -> ToolError
```

## skills/__init__.py

[源码](host/skills/__init__.py)

无函数/方法定义；包说明、常量或类型声明。

## skills/candidates.py

[源码](host/skills/candidates.py)

```text
L12 _advertised(host: Any, tool: str) -> str
L30 _grasp_candidate(host: Any, candidate_id: str, observation_id: str) -> dict[str, Any]
L96 _withheld(manifest: Mapping[str, Any], candidate_id: str) -> dict[str, Any] | None
```

## skills/driver.py

[源码](host/skills/driver.py)

```text
L109 ledger_path(host: Any) -> Path
L113 load_ledger(host: Any) -> dict[str, Any] | None
L122 store_ledger(host: Any, state: Mapping[str, Any] | None) -> None
L130 load_commitment(host: Any) -> dict[str, Any] | None
L137 store_commitment(host: Any, value: Mapping[str, Any]) -> None
L141 _write_json(path: Path, value: Mapping[str, Any]) -> None
L150 reconcile(host: Any) -> dict[str, Any] | None
L182 _require_capability(host: Any) -> None
L190 begin_skill(host: Any, skill: str, arguments: Mapping[str, Any]) -> dict[str, Any]
L237 _submit(host: Any, control: Mapping[str, Any], state: Mapping[str, Any], *, resume_from: int, commitment: Mapping[str, Any]) -> dict[str, Any]
L294 drive_skill_stage(host: Any, action: Mapping[str, Any], *, stage: str, stage_index: int, observation: Mapping[str, Any], observation_root: Path, completed: Sequence[Mapping[str, Any]], capture_observation: Callable[[], tuple[dict[str, Any], Path]]) -> dict[str, Any]
L393 _commit_geometry(host: Any, *, skill: str, candidate_id: str, observation_id: str, control: Mapping[str, Any], root: Path, observation: Mapping[str, Any], prepared: bool | None=None) -> dict[str, Any]
L450 _confirm_candidate_still_bound(host: Any, state: Mapping[str, Any]) -> None
L462 _target_localization(host: Any, *, control: Mapping[str, Any], root: Path, observation: Mapping[str, Any], target_ref: str) -> dict[str, Any] | None
L475 _measure(host: Any, stage: str, observation: Mapping[str, Any], commitment: Mapping[str, Any], completed: Sequence[Mapping[str, Any]]=()) -> dict[str, Any]
L526 build_skill_receipt(host: Any, *, action: Mapping[str, Any], status: str, stages: Sequence[Mapping[str, Any]], observation: Mapping[str, Any], stop_reason: str) -> dict[str, Any]
```

## skills/grasp_feasibility.py

[源码](host/skills/grasp_feasibility.py)

```text
L41 _pose_vector(value: Any) -> list[float] | None
L59 _contract(message: str) -> ToolError
L63 _same_pose(left: list[float], right: list[float]) -> bool
L79 _compiled_vector(pose: Mapping[str, Any]) -> list[float]
L83 _admitted_members(query: Mapping[str, Any]) -> dict[str, list[float]]
L111 _choose_member(stages: list[str], members: Mapping[str, Mapping[str, list[float]]], compiled: Mapping[str, list[float]], backend_selected: Mapping[str, Any]) -> tuple[str | None, str]
L142 _adopt(host: Any, commitment: dict[str, Any], stages: list[str], members: Mapping[str, Mapping[str, list[float]]], compiled: Mapping[str, list[float]], variant_id: str) -> list[str]
L178 check_pregrasp(host: Any, commitment: dict[str, Any]) -> dict[str, Any]
```

## skills/jaw_hold.py

[源码](host/skills/jaw_hold.py)

```text
L67 _number(value: Any) -> float | None
L73 _stage_outcome(completed: Sequence[Mapping[str, Any]], arm: str, name: str) -> Mapping[str, Any] | None
L85 close_outcome(completed: Sequence[Mapping[str, Any]], arm: str) -> Mapping[str, Any] | None
L90 open_position(completed: Sequence[Mapping[str, Any]], arm: str) -> float
L96 after_lift_opening(record: Any) -> float | None
L105 after_lift_offset(record: Any) -> float | None
L111 hold_evidence(*, outcome: Mapping[str, Any] | None, after_lift: float | None, tolerance: float | None, opened: float | None=OPEN_POSITION_NORMALIZED, after_lift_offset: float | None=None) -> dict[str, Any]
L156 verify_stage_evidence(completed: Sequence[Mapping[str, Any]], arm: str, record: Any) -> dict[str, Any]
```

## skills/ledger.py

[源码](host/skills/ledger.py)

```text
L56 stages_for(skill: str, *, prepared_grasp: bool=False) -> tuple[str, ...]
L64 begin(*, skill: str, candidate_id: str, rationale: str, observation_id: str, observation_sha256: str, prepared_grasp: bool=False) -> dict[str, Any]
L95 validate_state(value: Any) -> dict[str, Any]
L109 pending_stage(state: Mapping[str, Any]) -> str | None
L121 record(state: Mapping[str, Any], *, stage: str, status: str, reason: str, action_id: str | None=None, measurements: Sequence[Mapping[str, Any]]=()) -> tuple[dict[str, Any], str]
L160 abandon(state: Mapping[str, Any], reason: str) -> tuple[dict[str, Any], str]
L171 terminal_status(state: Mapping[str, Any]) -> str
```

## skills/lift_evidence.py

[源码](host/skills/lift_evidence.py)

```text
L35 reference(host: Any, **kwargs: Any) -> dict[str, Any] | None
L50 capture_baseline(observation: Mapping[str, Any], root: Path) -> dict[str, Any]
L61 _effector(observation: Mapping[str, Any], arm: str) -> Mapping[str, Any]
L65 _pose(observation: Mapping[str, Any], arm: str) -> tuple[np.ndarray, np.ndarray]
L75 compare_surfaces(*, points: np.ndarray, before_scene: np.ndarray, after_scene: np.ndarray, before_position: np.ndarray, before_rotation: np.ndarray, after_position: np.ndarray, after_rotation: np.ndarray) -> dict[str, Any]
L120 measure(host: Any, *, commitment: Mapping[str, Any], observation: Mapping[str, Any], observation_root: Path) -> dict[str, Any]
```

## skills/receipt.py

[源码](host/skills/receipt.py)

```text
L86 validate_rationale(value: Any) -> str
L110 stage(name: str, status: str, reason: str, *, action_id: str | None=None, measurements: Sequence[Mapping[str, Any]]=()) -> dict[str, Any]
L135 image_bindings(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]
L164 body_bytes(result: Mapping[str, Any]) -> int
L179 assemble(*, skill: str, status: str, rationale: str, candidate_id: str | None, observation_id: str, observation_sha256: str, stages: Sequence[Mapping[str, Any]], verification: Mapping[str, Any] | None=None, images: Sequence[Mapping[str, Any]]=(), claim_boundary: str, budget_bytes: int=RECEIPT_BUDGET_BYTES) -> dict[str, Any]
```

Architecture v1 phase-one step 2: [categorized data API](../../../docs/reference/toolkit-data.md).
`recording.py` and `replay_session.py` own recording/replay bookkeeping;
`interface/assemble.py` exposes immutable capabilities without those fields.


Synchronous toolkit-v2 uses a Supervisor-owned parent budget and durable action
ledger. Its action capability has no legacy submit_action method or adapter.
Released toolkit-v1 retains the one-proposal guard in call_scope.py, including
the existing rejection text; synchronous execute_action never consults it.
See [execution contract](../../../docs/reference/toolkit-execution.md).
