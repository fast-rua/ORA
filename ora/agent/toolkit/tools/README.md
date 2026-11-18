# 工具层与本地辅助模块约定

28 个工具，每个一个目录：`tool.py` 是入口（`run(api, args)`），`description.json` 是模型
看到的说明和输入、结果 schema（文件式工具面原样交付），`README.md` 给开发者看。
`../registry.py` 按目录名找入口；工具面声明 `toolkit_only_v1`，或是 Synthetic、`ora local` 的原子工具面和
`robot_runtime_v2`（`runtime/toolkit_dispatch.served`）时，运行时经它派发。`robot_runtime_v2` 的点号工具名
（`observation.dump_state` 等）对应下划线目录名（`observation_dump_state` 等）；这些工具和 `preview_move`、
`open_gripper`、`close_gripper` 没有 `description.json`，模型看到的文字仍由各自的分层工具面给出。
`ora toolkit replay RUN --at K --tool NAME` 在记录边界运行同名入口。

`tools/<tool_name>/` 放入口和私有 helpers，`tools/shared/`
放跨工具 helpers。不要求提前合并共用代码，也不创建空 Python 入口或兼容 shim。

RSI 可改工具与 helpers，不改共享 lib 或 interface。外部数据只经接口，不直接持有
worker/client/host、文件根或真值能力。工具可以调用 lib/interface 及其他工具的
只读部分；v1 一次父调用最多一个动作 proposal，v2 起同步执行、按预算可有多个动作（见下文）。候选排序/选择、任务阈值和抓取阶段
留工具层；lib 提供完整数据和参数化算法。仍同进程，无沙箱保证。

提升 helper：提交输入输出与复用证据，去掉任务偏好/隐藏状态，补类型、帧/单位、
replay 类别和失败方式，经人审后由 lib 维护者实现、测试、升级版本/hash，工具再
显式改依赖。RSI 可以提出库修改请求；本版不自动提升、发布或绑定外部包。

可选的 `tools/<name>/explain.json`（`ora.toolkit-explain.v1`）是这个工具的 `explain` 模板：它的各个阶段、每段读哪些结果字段和 `stage()` 事件、每段一行。
渲染器在 `../explain.py`；测试把模板读的字段对到工具的输入/结果 schema、工具源码里 `stage()` 调用的事件，以及封存局记录的结果。
工具里的 `stage(name, **fields)`（`from ...interface.stages import stage`）只在 code mode 收集事件时才记，其余都是空操作。目前有模板的是
vision_ground、vision_segment、geometry_query、grasp_plan 和六个动臂工具（move_pose、move_to、rotate_wrist、rotate_pitch、recover_pose、return_home）。
每个移动机器人的工具都接受 `dry_run`（`interface/dry_run.py`），工具调用和代码模式都一样：这些工具只规划、不执行，结果形状和预览在 `tools/_dry_run.py` 与 `lib/action_preview.py`，环境规划的后端由 `agent/tools/runtime/action_preview.py` 经只读桥询问；模板也渲染这个计划；
被拒的计划带一行 `summary`（代码加拒绝自己的原因；环境没给 message 时，用规划器的第一句和拒绝细节的话，`tools/_refusal_words.py`，与回执 summary 行同一套措辞）；
move_to、rotate_* 经 move_pose 编译，它们的 `compiled` 事件由 `move_pose/tool.py` 记录。
工具配置带 `latency_switches` 时（只有 latency profile 关掉了面向模型的开关的计划才有，`tools/_latency.py`）：`reach_reference` 为 false，位姿工具不为 `reach_along_request` 问 IK，标注写明关闭；`preview_images` 为 false，dry run 保留数字但不画图，真实调用被拒时不附图。没有这个键的配置（所有 benchmark 调用和封存局）两个开关都开。

工具上下文带 `motion.environment_plans` 时由环境规划动作：`move_pose`、`move_to`、`rotate_*`、`set_gripper`、`wait_ticks`、`recover_pose` 只交出一个动作提案（`move_pose/tool.py` 的 `propose`），不执行也不读 `exec.*`；单臂工具面的扁平参数由 `surface_targets` 读取。`move_base`（RoboCasa 的移动底座，`mobile_base_navigation_v1`）只在这种上下文里运行，交出一个世界系目标的 `move_base` 提案。没有这个键的上下文按 RoboDojo 处理，封存局都是这样，所以新分支不给它们增加任何读取。

目标是回到一次工具调用，拿到原输入后试另一份代码，不是恢复并继续仿真。
新 simulator 查询未记录就失败，感知和纯计算可重算。CLI 在 M5。

## Synchronous migration (architecture v1, step 5)

The layered surface these revisions built, `robodojo_placement_toolkit_v2`, was
retired on 2026-11-01. Its tools run on toolkits v5 and v6
(`ora/policy/tool_surfaces/robodojo_placement_toolkit_v5/` and `.../robodojo_placement_toolkit_v6/`,
file-backed), which fix each parent budget offline (`budgets.py`) instead of taking it from the call.

`robodojo_placement_toolkit_v2` revision 3 adds `set_gripper`, synchronously
executing explicit jaw commands. Its holding intention and grasp open/close
intentions live in `tool.held.<arm>`; move clearance reads this state and public
geometry without the old runtime cache. See `docs/reference/toolkit-data.md` for
snapshot timing and zero-spend rollback. v1 retains its original nine-tool
proposal behavior. v2 supports multiple budgeted actions per parent; it does
not use the earlier one-proposal contract. Both remain in-process capabilities.

Revision 4 adds `recover_pose`: it reads state-layer previous/home anchors,
passes them explicitly, and retains the old public strategy mapping. The owner
rejects any stale or invented anchor with zero spend before worker dispatch.

Revision 5 adds `mark_point` and `geometry_query`. Named measurements preserve
cross-call reuse/verification through `tool.marks`, including observe overlays
and reset lifecycle; no point file is read or written by the new tools. Geometry
selects a verified public publication and recomputes pixel/world measurements.

Revision 6 adds read-only `vision_segment`: independent point prompts,
per-point candidate allocation, public target artifacts and old image bytes.
Live verified query reuse is explicit; replay re-requests the service.

Revision 7 completes the released robot cohort with align_plan, rotate_wrist,
rotate_pitch, wait_ticks and report_issue. There are 18 default robot tools;
independent check_task grants add the nineteenth. All execute through the new
interface, with no legacy handler fallback. Context's artifact reader is separate.
