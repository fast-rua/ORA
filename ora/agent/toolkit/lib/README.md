# lib v0：只读能力库与作者手册

版本 `0.4.0`，覆盖九个可切换原子工具的基础能力。
RSI agent 只读（D-12），人审后才修改版本。九个原子工具已通过默认关闭的
use_toolkit_tools 接入 coordinator，也可用 ora toolkit replay 回到调用前换代码。
默认仍用旧 handler，本版不做仿真续跑。

数据访问只经显式传入的 DataInterface；没有 host、文件根、环境变量、服务客户端、
全局观测缓存或隐式随机状态。游标、服务绑定和证据状态由接口拥有。库不提供任意路径、
真值或模型调用方法。算法只用显式输入与 CPU 数值库。维护权限和 import 边界不是
OS 沙箱：本版仍同进程。

## 模块与类型

| 模块 | 类型 / 函数 | 范围 |
| --- | --- | --- |
| values | JsonValue、Array、PoseM、TransformM、PointCloudM、PublicResult | 不可变值和完整公开响应 |
| blobs | read_json、read_array、save_array | 读取已授予 BlobRef；保存派生输入数组 |
| observation | Observation、CameraRefs、CameraPixels、CameraRGBD、RobotState、ArmState；observe、proprioception、load_pixels、load_camera、robot_state | 所有相机声明、公开标定和测得状态 |
| perception | MaskCandidate、Segmentation；segment、segment_points、ground | SAM3 当前三种路由全部客户端输出 |
| grasp | GraspCandidates；candidates、candidate_pose、controlled_pose | 完整 CGN 池与显式受控点转换 |
| geometry | DepthSelection、PrincipalGeometry；mask_to_cloud、depth_inliers、principal_geometry、transform_points、transform_pose、offset_pose | 参数化纯计算，无任务排序/选择/抬升策略 |
| vision | InspectionImage；mask_summary、localize_mask、mask_overview、candidate_inspection、png_bytes、npy_bytes | 完整公开 RGB-D 测量、预览和内存编码；无文件 I/O |
| fusion | FusionSource、FusedCloud；fuse | 显式相机顺序、世界坐标米、体素尺寸/深度范围；完整来源与计数 |
| pointcloud | CloudView、VIEW_AXES；render、annotate | 融合点云 → 俯视、正视、侧视三张正交 RGB 视图，世界坐标点画在视图像素上；不是任何相机的渲染 |
| compilation | compile_contact | 显式 compiler 参数的坐标编译及完整检查（X5/ALOHA，以及 Panda/Robotiq 的 grip-site handoff）；不排序、不做 IK |
| motion | probe_motion、align_plan、reach_check、kinematics | 原版本请求与全部公开诊断 |
| motion_plan/（子包） | paths、timing、kinematics、checks、route、planner、execute、move、recovery、kept | 手臂运动在 lib 里规划、由环境执行一次 `follow_joint_trajectory`：路径（直线、关节空间直线）、速度与加速度上限下的时间参数化、经只读运动学查询的连续 IK、检查（IK 空洞、连续性、预算）、move_pose 的交互路线规则、recover_pose 的恢复规划（沿上一次运动实测路径回退；按筛查顺序的关节插值；home 先抬升）；速度由工具定 |
| control | receipt、check_task、submit_action、report_issue、finish_episode | 原协议和公开回执；编排留工具层 |

| pose | compile_target | 当前 PoseM 与显式请求 → 完整姿态编译，米/xyzw/角度单位明确 |
| clearance | estimate、along_path | 所给公开深度、机器人指爪模型、持物范围 → 目标处或沿路径的可见表面间隙；非碰撞证明 |
| action_preview | Camera；cameras_from、preview、arm_numbers、path_of、drawn、jaw_point | 一次动作的规划段、公开标定和最新图像 → 每臂数字、最小间隙和一张图；不含任务、工具或后端名，不跑物理 |
| images | RenderedView；dimensions、annotate | 原 PNG、标定和世界坐标点 → 同字节 PNG/像素尺寸/渲染身份 |
| numeric | evaluate | 受限数值 kernel；JsonValue 输入/输出，不访问仿真器或感知服务 |

Array 用自有 bytes 保存 dtype/shape/C 或 F 顺序；numpy() 返回不可改写视图。
JsonValue 保存有限 JSON 原字节，json() 返回独立树。类型为 frozen dataclass，
集合为 tuple，不只是给共享可变数组设置 writeable=False。PublicResult 保留完整
envelope 原字节、来源、错误；comparison 分开；data 在 error/timeout 时抛
InterfaceError。类型化视图不裁剪公开字段，也不重编码历史 payload。
不同编码域（manifest 子树与独立 proprioception 文件）只比较字段，不混比字节。

遵循 [frames and units](../../../../docs/robots/frames-and-units.md)：长度米，
joint_positions_rad 弧度，pose 保留父 frame 和 xyzw。PoseM 要求有限近单位四元数，
拒绝坏输入，不默默改记录。TransformM 为 target←source 列向量变换，校验 SE(3)。
camera/<name> 为 OpenCV 右/下/前；K 为像素，extrinsic_cv 为 world→camera，
load_camera 显式求逆。只支持零 skew 针孔。深度为光轴 z 米，不是射线距离。
RobotState 不将原 pose_frame 的 ee_link 自动当成 TCP/world；gripper_normalized
不是米，完整 finger/feedback/command 字段仍在 public 中。

## Replay 分类与失败

| 入口 | 行为 |
| --- | --- |
| observe、proprioception、receipt | 从当前游标记录取；未记录失败，旧 snapshot 只恢复明确公开产物 |
| probe_motion、align_plan、reach_check | 精确请求从记录取；换目标且未记录即 NotRecorded，不补在线 IK |
| check_task | 仅该边界记录，受 capability 开关；不读最终判定 |
| segment、segment_points、ground、candidates | 重新请求服务，历史仅比对；新请求不消费游标，comparison unavailable |
| submit_action、report_issue、finish_episode | 精确匹配历史控制；偏离报 Divergence，绝不在线执行 |
| motion_plan.kinematics | 每次 ik_chain、fk、screen_path 是一次记录的 `kinematics` 调用：精确请求从记录取，换请求即 NotRecorded，不在线求 IK |
| motion_plan.execute.follow | 整个运动是一次 execute_action：与记录的轨迹原语精确匹配，偏离报 Divergence，绝不在线执行；其余为纯计算 |
| read_json、read_array、load_camera | 读取已授予字节；缺失/hash 漂移失败，不新增语义调用 |
| save_array | 接口将派生数组存入 evaluation CAS；不写源 run，不推进游标 |
| robot_state、candidate_pose、controlled_pose、geometry、vision、fusion、compilation 函数 | 纯计算；形状/帧/数值错误失败，不访问仿真 |

每个函数 docstring 给出单位、失败方式、例子和 replay 类别，本手册与源码同版本。
版本化 motion 诊断、receipt 的 per-arm/phase 字段都完整保存在 JsonValue/PublicResult；
没有专用结构体不等于删数据。接口未导出的 controller 内部阶段不能补齐；没有原样
dump worker、HTTP 或真值对象的通道。

## 示例：公开数据到完整候选

```python
from ora.agent.toolkit.lib import observation, perception, geometry, grasp
from ora.agent.toolkit.lib.values import JsonValue

obs = observation.observe(api)  # 可信装配层提供 runtime 或 replay
refs = obs.camera("head")
camera = observation.load_camera(api, refs)
seg = perception.ground(api, rgb=refs.rgb, text="red cup", max_candidates=3)
# 选哪个 mask 是工具策略；这里只示范索引。
target = geometry.mask_to_cloud(camera, seg.candidates[0].mask)
world_target = geometry.transform_points(target, camera.camera_to_world)
shape = geometry.principal_geometry(world_target)
# scene 由调用者以同一 camera frame 的公开深度构造。
pool = grasp.candidates(api, scene=scene, target=target,
    camera_to_world=camera.camera_to_world,
    workspace_bounds_m=JsonValue.of({"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}))
```

CGN 的 1–32 个请求候选、分数、contact/opening 对齐且保持原序，输入显式 float32
C-order、单 forward pass；不套旧 compiler 的裁剪。candidate_pose 是原 grasp
frame，不能当 TCP。controlled_pose 要调用者显式提供 embodiment 绑定的转换，
不能替代旧 compiler 的 aperture admission、reach 或执行资格检查。

SAM3 开放单正点、1–8 个**独立**正点（各点全部 multimask alternatives）、文字
定位（最多 3）。单点保留客户端连通域规范化，metadata 保留逐点失败和服务身份。
框、负点、联合多点、mask_input 没有当前 HTTP 入口，不伪造可用函数；新增服务
协议、worker 规划和碰撞能力另审。depth_inliers 的范围、连通域、最小点数、MAD
系数均参数化，默认对应旧局部定位口径。PCA 返回全部轴/方差/配对 extent，重复
特征值标 ambiguous_axes，不能称为稳定物体方向；可见点云不是完整物体形状。

## 本地辅助、提升与版本

目录约定见 [tools README](../tools/README.md)。RSI 可写工具 helpers，仍只经
lib/interface 拿数据。任务排序/选择、预抓取→接近→闭合→抬起留工具层。
提升流程：提交复用场景及固定输入/完整输出 → 去掉任务常量、隐藏状态和策略 →
声明类型/帧/单位/replay 类别 → 人审 → 维护者加入契约测试、更新版本/hash →
工具显式更新依赖。RSI 可提修改请求，不能自行改共享 lib。这里只落目录和文档，
无自动提升流程、权限沙箱或外部包加载器。

manifest.py 的不可变常量列出版本及本目录 Python/README 文件 SHA-256；CONTENT_SHA256 为
canonical JSON `{version, files}` 的 SHA-256，相对 toolkit 路径（lib/...）、排序键、无末尾换行。
manifest.py 不参与自哈希；测试核对内容，lib 不读取自身文件或隐式探测版本。
回放结果记录精确 version/content_sha256；当前 RunPlan 尚无独立 lib 身份绑定字段。

## v0 范围

已覆盖可切换原子工具所需的公开 RGBD、完整感知候选、目标测量、X5/ALOHA 编译、
跨相机融合和评审图。工具层负责排序、选择与执行策略。固定数据的旧新结果及
产物原字节等价测试已通过；默认 dispatch 保持旧实现，显式开关才选新实现。
未跑真实服务/GPU，假服务不能证明真实模型稳定。Panda 专用编译和其它工具
仍属后续；检查记录与已知边界见 [M4 工作记录](../../../../docs/developer/toolkit-m4-progress.md)。

## 0.2.0 迁移计算

vision 的私有 `_vision_*` 模块保留旧 RGB-D 测量与渲染配方，输入改为显式的
数组/公开标定；公开入口只返回不可变值或 bytes。`localize_mask` 明确使用原有
primary-component/MAD 配方，不作目标选择。fusion 参数化体素/深度/工作空间，
保留原始源顺序和全部来源计数。compile_contact 适用 X5/ALOHA 的通用坐标映射，
0.22.0 起也编译 Panda/Robotiq 的 grip-site handoff；它只做参数化编译，grasp_plan 策略位于工具层。

## Grasp computation additions (0.3.0)

`grasp_review.contact_sheet` and `global_review` accept immutable calibrated
pixels, clouds, compiler identity, ordered candidates and explicit labels. They
return `InspectionImage` (owned RGB bytes plus full JSON metadata). No candidate
selection happens in lib. `npz_bytes` encodes explicit named numeric arrays.
`geometry.visible_surface_geometry` exposes mean-centered PCA and stable voxel
samples. `vision.measured_depth_selection` returns the retained mask and every
measured quality fraction; acceptance thresholds remain with the tool. These
are all pure computations in replay. Private rendering modules perform no
source-file access; PIL handles in-memory images and installed renderer fonts.

M4b 的 pose/images 是纯计算；clearance 只读本次已授予深度再计算，无额外 simulator
查询；numeric 经 evaluate_numeric 重算，保留原 AST 白名单、CPU/内存/墙钟限制。
这四项在 replay 中不消费接口游标。模型任务偏好不放入 lib。

Architecture v1 callers use categorized data methods (`api.state.observation`,
`api.perception.ground`, `api.planner.probe_motion`, `api.artifacts.read_blob`,
`api.compute.evaluate_numeric`, and the action/lifecycle categories). Lib remains
stateless. On an `assemble_snapshot` API, observations come from exported state
keys, perception recomputes, planner methods require a simulator and actions are
refused. Historical evidence transports still support their old exact-cursor
semantics until the phase-four command migration.

`control.execute_action` forwards a primitive and explicit budget, returning only
the semantic receipt and parent balance. Live publication completes before it
returns; recorded replay never connects an online executor.

`clearance.held_extent` is pure computation: explicit localized world bounds,
dispatch-time link6 pose (metres / xyzw), and local jaw contact center produce
immutable local box corners or an unknown extent. It does not infer attachment
or keep a cache. Live and replay use the same calculation.

`images.contact_sheet` composes explicit in-memory PNG panels and labelled
world points/RGB colours into one bounded PNG, with the existing overlay legend.
It is pure computation in live/replay and never creates a source artifact path.

Version 0.9.0 adds `point_images.point_overview` and `point_mask_overlay`
(lossless pixel overlays with explicit caller-selected candidates and colours),
and `vision.npz_bytes` (named immutable arrays to in-memory compressed bytes).
All are pure computation in runtime/replay; no service call or path access.

Version 0.10.0 adds `pose.approach_orientation`: pure alignment of supplied
local approach/closing axes to a world direction with an explicit roll in
radians, returning xyzw. Named sides, standoff and candidate sampling remain
policy in align_plan. Invalid/zero/nonorthogonal vectors fail explicitly.

Version 0.11.0 adds `pacing`: one compiled move_pose action executed along the
straight line from the measured pose at a caller-chosen speed (m/s, deg/s),
one `execute_action(..., 1)` per waypoint on a trapezoidal profile, with a
tracking guard, a hold on stop and a final settle (up to 5 native actions).
Path checks take reach-check rows the tool obtained; speed presets and budgets
belong to the tools (then move_pose_normal, move_pose_careful). Planning is pure
computation; each executed waypoint is a recorded primitive in replay.

Version 0.12.0 adds `pacing.fit_budget` and `pacing.part_of`: when a move needs
more native actions than the tool's budget at its speed, the longest part of
the same straight line the budget covers (ramps and settle included), so a
paced tool moves part of the way and reports it instead of refusing.

Version 0.13.0 replaces the execution of `pacing` (one move_pose primitive per
waypoint, each restarting from the measured joints) with the `motion_plan`
sub-package: a
move is planned in lib and executed by the environment as one
`follow_joint_trajectory` whose targets the environment interpolates every
physics step with a velocity feed-forward. Parts:

- `paths`: the straight Cartesian line (position lerp, orientation slerp),
  the joint-space line, their grids (at most 3 mm / 1.5 degrees, or 0.02 rad,
  apart; fixed by the path, never by the speed), the staged pre-contact point.
- `timing`: `Pace` (Cartesian and joint speed and acceleration caps; no ramps
  for moves under 2 cm) and `knots`: a forward-backward pass over the grid,
  sampled every native action and stretched to whole native actions, so no
  cap is exceeded; `cut` and `concatenate` (a stop where two parts meet).
- `kinematics`: the backend's read-only kinematics query
  (`api.planner.kinematics`, `ora.kinematics-request.v1`): IK chains each
  seeded from the previous solution, FK, lockstep sphere screens. The
  environment answers with the SDK's own CuRobo IK; lib stays pure.
- `checks`: IK holes and branch changes (`paced_path_unreachable`,
  `paced_path_joint_jump`, with the fraction and pose), joint-line bridges and
  their Cartesian and orientation deviation (`turn_deviation_deg`; a bridge that
  spins the wrist is refused as `paced_path_wrist_unwind`), `fit_budget` (the
  longest first part that fits, never cut inside a `timing.Grid.atomic` span),
  `joint6_extremes` (the wrist roll along a planned path).
- `route`: move_pose's interaction route (free_space joint line; precision
  line within 50 mm / 15 degrees; contact_approach line, or staged).
- `planner`: `plan_line`, `plan_joint`, `endpoint_joints`, `expected_poses`.
- `execute`: `trajectory_action`, `follow` (one `execute_action`), `summary`,
  `gripper_action` (0.14.0).
- `move` (0.14.0): `path`, `plan`, `run`, `caps`, `outcome`, `speed_segments`:
  one move end to end with any `Pace`; `peaks` (0.15.2): the top speed and
  turn rate a planned path reaches under every cap, joints included.
- `orientation`: the jaw-symmetry rule (`nearest_jaw_twin`, `jaw_twin`,
  `turn_deg`, `JAW_SYMMETRY_LABELS`): of a requested orientation and its
  half-turn twin about the approach axis, an empty jaw takes the one nearer
  the current wrist, a holding jaw never takes the twin, an exact request is
  kept. The default motion tools apply it where joint trajectories run; a
  tool of its own may apply or deviate from it.

The sub-package is listed in `manifest.py` by its relative paths and follows
the same import and I/O rules as the flat modules (`test_contract.py`).
Speeds, settles and budgets belong to the tools (`tools/move_pose/_smooth.py`,
`tools/budgets.py`).

Version 0.13.1 kept `pacing` as a deprecated facade for tools written against
0.12.

Version 0.19.0 adds `action_preview` (with `_preview_render`), `clearance.along_path` and `motion_plan.kept`: the
preview of a dry run, and the plan it keeps for the real call on the same observation.

Version 0.14.0 removes `pacing`: tools plan and run arm motion with
`motion_plan` (the migration guide for agent-written tools maps every old name).
`kinematics.screen_path` now also screens each moving arm against the SDK
planner's table top, and a planned move is screened with it (self, table and
both arms in lockstep) before it runs (`checks.screened`: `self_collision`,
`table_collision`, `inter_arm_collision`). `execute.trajectory_action` takes
`part_of_move` for a move cut to fit its budget, so the environment reports it
not_reached with the distance left instead of an arrival. `planner.PathPlan`
carries `warnings` (kinematics answers that carried a warning, e.g. a restored
run without recorded IK generator state; `kinematics.warnings`).
`timing.ramp_free` names a constant-speed move; its knots advance evenly
inside every grid segment.

0.14.0 also makes everything move_pose does reachable from a tool of any
author, with its own caps: `move.plan` (path by route, timing under any
`timing.Pace`, an optional slower `Pace` over the last `timing.SLOW_APPROACH_M`
(2 cm) of each arm's path entered without a stop (`timing.slow_final`, a
per-segment `timing.Zone`), budget fit, screen, expected poses) and `move.run`
(one `follow_joint_trajectory`, the receipt and the `motion` summary move_pose
returns, with `speed_segments` and the native actions each spent), and
`move.Planned.screen` is the screen's answer for the planned path (it was clear
or `plan` refused), which a dry run reports as the screening;
`execute.gripper_action` builds a rate-limited jaw command with any rate and
lead. move_pose's speed tiers (fast, normal, careful) and settles are defaults
in `tools/move_pose/_smooth.py`; a tool of its own may replace them.

Version 0.15.0 adds `motion_plan.recovery`, recover_pose's planning as lib
code (the environment's recovery executor stays for the released surfaces):
`retrace_nodes` (the anchor's recorded `measured_path_rad`, the joints
measured at the end of every native action of the move that set it, from
where the arm stands back to the anchor; refused as `not_recorded`,
`record_of_another_move` or `arm_moved_since`), `unwind_target` (the stored
joints' nearest whole-turn equivalent, never winding a pitch joint),
`unwind_orders` / `joint_options` (straight, fold-then-yaw, yaw-then-fold),
`lift_goal` / `plan_lift` (link6 up to 0.10 m, a straight line), `combine`
(the arms' phases on one grid, a stop between phases), `plan` (candidates in
preference order, each timed and screened before anything moves; the first
clear one fitted to the budget; none clear refuses with
`recovery_self_collision_path`, `recovery_inter_arm_collision_path` or
`recovery_table_collision_path`), `run` (one `follow_joint_trajectory` marked
`recovery`, so it keeps the anchor) and `outcome` (the released
`recovery_progress` terms).

Version 0.15.1: `recovery.run` passes the trajectory's `recovery` argument as
given, so it also takes the retrace form (`targets: {arm: retrace}`,
`retrace: {arm: {moves, arm_joint_qpos}}`) recover_pose uses to undo the top
entry of `exec.recovery.<arm>.undo` (the last 5 arm-moving tool calls);
`recovery.path_position` says how far along a recorded path a joint vector
lies. recover_pose plans only the retrace (no joint-interpolation fallback);
return_home plans the home lift and unwind. `timing.parameterize` gives a
segment that starts and ends at rest (a one-segment cut) its
accelerate-then-brake time instead of an infinite one.

Version 0.15.2: `recovery.outcome` follows the trajectory's measured evidence
(arrived when every knot ran and the settle settled; arrived_not_settled;
cut_short; stalled; still_moving_at_window_end only when the budget ended
before the last knot; episode_terminal; unknown when the evidence is missing,
never a guess), and `recovery.measured_end` reads the measured final joints
from the evidence (`final_joint_error_per_joint_rad`), so
`joint_distance_end_rad` is filled without an after-state.

Also in 0.15.2: `timing.Grid.atomic` lists node spans a budget cut may not fall
strictly inside. `planner.plan_line` marks every joint-line bridge, and
`planner.plan_joint` marks the whole line when joint 6 turns by at least
`planner.LARGE_WRIST_TURN_RAD` (90 degrees); `checks.fit_budget` moves a cut
back to the span's start, and refuses with `paced_motion_budget`
(`constraints.unsplit_span`) when the move starts with that span, so a budget
never stops a bridge or a large wrist turn part way. `move.run`'s `motion.joint6`
and `recovery.outcome`'s `joint6` report the wrist roll per arm from the start
joints through every planned knot (`checks.joint6_extremes`: start, end, min,
max, travel and the least margin to `checks.JOINT6_LIMIT_RAD`), since the two
ends alone can hide a swing.

Version 0.16.0 puts arm motion on the shared motion base. `timing` no longer
computes a profile: `timing.timed` asks the environment (`kinematics.pace_path`,
recorded like every kinematics answer) to time each arm's path under a `Pace`, with
the wrist check and the budget fit, and puts the arms on one timeline. `Pace` keeps
its six caps (`caps()`), `Grid` keeps fractions, joints and stops; `parameterize`,
`knots`, `ramp_free`, `cut`, `slow_final`, `Zone`, `checks.fit_budget` and
`recovery.with_cartesian_lengths` are gone, and a budget cut no longer keeps
joint-line bridges whole (large wrist turns stay whole). `kinematics.plan_path` is
the shared planner's collision-free path in the world of the current public depth,
with the other arm in it: `planner.plan_global`, `route_kind="global"`, and with
`global_planner=True` the `free_space` route, the far part of a staged approach and
the way around a straight line that cannot be followed. `recovery.global_option` is
the same path home. `expect_contact=True` in `move.plan` takes `contact_approach`'s
route whatever the label and refuses a contact that would drive a link behind the
fingers into the world (`kinematics.contact_path`, `contact_path_blocked`).

Version 0.17.0 makes the same check cover every straight contact: `move.plan` asks
`kinematics.contact_path` for a `contact_approach` label with or without
`expect_contact`, and for the new `route_kind="straight"`, the whole straight line,
which is refused rather than replaced by another path. A line that fell back to the
planner's path is not checked again. `expect_contact=True` under
`route_kind="interaction"` is now that same whole line, no longer staged when it is
long, so a push stays on its line at every speed. `move.peaks` reports on every route.

Version 0.18.0 takes an arm the planner cannot start from by the joint route. The shared planner keeps the robot within
its joint limits, and a measured start past them (`start_outside_limits`) says nothing about the goal, so
`planner.plan_global` raises `PlannerUnavailable`, as with no planner, and the joint line brings the arm back into range.

Version 0.20.0: `execute.follow` takes `route`, which goes on the toolkit call's evidence
(`interface/call_note.py`, `note.route` in `calls.jsonl`), not in the action; `move.run` and
`recovery.run` pass it.

Version 0.21.0 is perception on the other backends' surfaces. `pointcloud` renders the three orthographic views.
`observation.load_pixels` reads a camera with no pinhole or rigid-pose claim, for `mark_point` and `geometry_query`, which
check the calibration fields they use: a camera with no `extrinsic_cv` still answers a region query. `observation.observe`
leaves a camera's missing file `None` where it raised, and `robot_state` leaves a pose with no `pose_frame` untyped
(`ArmState.pose` is `None`, the numbers stay in `public`); RoboTwin publishes none. `_vision_footprint.annotate` moved
here from `ora/agent/tools/perception/overhead_view.py`.

Version 0.22.0: `compilation.compile_contact` takes the grip-site compilers (the Panda's and the Robotiq TCP's), with the
handoff those embodiments' tools publish: the contact pose always, a 50 mm approach limit, no opening check, their own
check names and the quaternion form of `_pose_math`.

Version 0.23.0 is for Synthetic Native, `ora local` and `robot_runtime_v2`. `pointcloud.click_constraint` is the two world
coordinates one pixel of a derived view shows, which the one-pixel `mark_point` accumulates across views;
`images.inspected` crops and shrinks an image for `vision.inspect` and `observation.dump_state`, and `images.depth_colours`
is the depth colour map `vision.inspect` returns, with the statistics of the valid depths.
Version 0.24.0: the preview is drawn to be read at a glance. Every mark has a dark edge and every label a dark box, kept inside its panel; the jaws are sketched where the hand is now and where it ends (`action_preview.jaws_at`, at the
measured opening, `openings_m`), except on a camera the hand carries; a refused request ends in a red cross and red jaws at what was asked, and its legend repeats the refusal's own reason.
Version 0.25.0: a preview reports the plan only. `action_preview.arm_numbers` and `preview` take no `remaining`, and an
arm's numbers and summary line carry no share of what the episode has left (`share_of_native_actions_left` is gone).
Version 0.26.0: a preview says how high a path rises. `action_preview.arm_numbers` takes the arm's `hand` and adds
`rise_above_ends_m`, how far the path's highest point (up to the first contact; the jaw contact centre, or the controlled
point without jaw geometry) is above the higher of its two ends, and `rise_at_fraction`, where it is; the summary line
states it ("rises 19 mm above its higher end at 52% of the path", or "never rises above its higher end").
Version 0.27.0: a goal the world refused below its support names its heights in the world frame.
`motion_plan.checks.planner_refusal` adds to the refusal's constraints the goal refusal's `in_world` heights
(`frame: world`, `support_z_m` and `fingertip_z_m` or `load_z_m`) with its `depth_m` and `tolerance_m`, where the
planner's answer carries them.
Version 0.28.0: a path's clearance says how far the jaws reach into what the depth shows. Where a finger box (or a held
object's box) holds a seen point beyond the path's ends, `_clearance.along_path` keeps `clearance_m` 0 and adds
`inside_m`, the deepest such point's distance inside the box, with `at_fraction` and `position_m` naming it; the summary
line says "the jaws reach 12 mm into what the depth shows at 30% of the path" and the picture marks it "12 mm inside".
Version 0.29.0: a preview shows what the jaws hold, from the measured state, whatever closed them. `clearance.held` says
an arm holds something when its last command was close and the jaws stopped apart (`_clearance_inputs.jaws_hold`), with
its box: the target named when they closed (`extent: localized`), else the box between the jaws at their measured
opening (`_clearance_inputs.between_jaws_corners`, `extent: between_jaws`; what reaches past them is not measured).
`measured_held` gives a path's clearance that box to sweep and to leave out as robot; `action_preview.preview` adds each
arm's `held_object` (`target_ref`, `extent`, `box_size_m`, `jaw_gap_m`), opens its summary line with it ("holds what
its jaws closed on (86x20x61 mm between the jaws, 20 mm apart; what reaches past them is not measured)") and draws the
box where the hand is and where it ends (`drawn`'s `held_boxes`, the renderer's `held_start` and `held_end`).
Version 0.30.0: `action_preview.preview` takes `picture` (default true). False, for a run whose latency profile renders
no preview pictures (`preview_images` off, `tools/_latency.py`), returns every number, the segments and the summary and
draws nothing: `image` is `PICTURE_OFF` and there is no `png`.
Version 0.31.0: `motion_plan.timing.Pace` takes an optional `joint_jerk_rad_s3`, the joint jerk the environment's timing
certifies a move at (rad/s^3, every joint; `None`, every preset, leaves it the robot's). Only a robot motion profile's
own pace names one (RoboDojo's ARX X5 profile v2 `fast`, 5000). `Pace.caps()`, the timing request, and `Pace.record()`,
the finite-JSON form a kept plan stores, name it only when set, so a pace without one requests and keys what it always did.
Version 0.32.0: `_clearance_inputs` takes the `half_measured_jaw_separation_minus_reference` opening rule: symmetric
jaws whose descriptor states the fingers at a non-zero reference opening (the I2RT YAM linear_4310, fully open), each
finger displaced by half the measured separation less its reference. `observation.load_camera` refuses a camera whose
calibration has no `extrinsic_cv` (a real cell's uncalibrated fixed camera) with a `ValueError` naming it, not a `KeyError`.

Version 0.33.0: principal axes no longer depend on the BLAS kernel. Where a variance repeats (a square or a disc
seen from above), `np.linalg.eigh` may return any basis of that eigenspace, and which one depends on the machine;
`_principal_axes.principal_axes` keeps eigh's axes where the variances differ and, in a repeated eigenspace, takes
the world axes projected onto it in `x, y, z` order and made orthonormal, the repeated variance reported as its mean.
`geometry.principal_geometry`, `geometry.visible_surface_geometry`, the vision measurement's principal frame and the
overhead footprint use it, as do the top-down grasp proposers (`tools/grasp_plan/_topdown.py`,
`host/grasp/proposers.py`; of tied minor axes they keep the first) and the sandbox's `ora_lib.geometry` (a copy).
Points move between frames by `_rigid.apply` and `_rigid.invert` (`geometry.transform_points`, the RGB-D fusion's
lifting, grasp_plan's `world_to_camera`): elementwise multiply-adds and a cofactor inverse, within rounding of the
BLAS forms, so the grasp service's inference seed (the sha256 of the scene points) and the review artifacts' bytes are
the same on every machine.

Version 0.34.0: what the vision measurement, its preview and grasp_plan compute from a camera no longer depends on the
BLAS kernel either. The camera's pose is inverted by `_rigid.invert` (`observation.load_camera`'s `camera_to_world`,
the mask localization, the preview's depth profile, the observed-extent box), not `np.linalg.inv`. Covariances over
points are numpy sums of elementwise products, `_principal_axes.scatter` and `covariance` (the principal frame, the
overhead footprint, the top-down proposer in `tools/grasp_plan/_topdown.py`), not a matrix product or `np.cov`,
whose summation order is the kernel's. `principal_axes` diagonalises a 3 x 3 by Jacobi rotations in Python floats
(eigh's eigenvectors to rounding; LAPACK's Householder reduction of a 3 x 3 rounds by kernel) and keeps eigh for a
2 x 2, which LAPACK solves in closed form; the sandbox copy follows. The support plane under a target is the
least-squares plane solved about its points' mean in numpy sums (`_vision_support._least_squares_plane`), not
`np.linalg.lstsq`. The fixed-pool grasp_plan answer and the grounding beside it are then byte-identical under the
Haswell, Zen, SkylakeX and Sapphire Rapids kernels. A dot product or length of a single vector (`np.dot`,
`np.linalg.norm` without `axis`) still rounds by kernel; where one reaches a hashed value, sum it in numpy instead.
