# vision_ground

`run(api, args)` 是数据接口上的原子工具实现。toolkit-only 工具面都经它执行，CLI 可用 `--tool vision_ground` 在记录边界运行。

读取已发布的公开 RGB-D，重新请求 SAM3，经 lib 测量并绘制图像，再通过受限
`publish_tool_artifact` 发布 target 和 grounding query。文本/候选上限、候选顺序、
表面所声明的字段和输出图像仍由工具决定。测量和图像编码在 lib。

只有 lib、interface、工具内辅助模块、标准计算库的依赖；没有 host、文件根或服务客户端。
live/replay 的 ToolData 必须由可信装配提供。Replay 不读历史 query cache 来跳过服务，
产物只进入 evaluation CAS；没有感知服务时失败。观测缺失则按接口报未记录。

目前固定公开像素及假服务覆盖 0/1/3 个候选：旧新结果、完整 target/query manifest、
PNG 和 NPY 字节一致。旧实现保持原位，默认 dispatch 仍选择它，供固定数据比较。
仍是同进程；静态 import/I/O 检查不是隔离沙箱。

工具配置带 `overhead_view_pose`（工具面声明 `overhead_footprint_localization_v1`）时，
`overhead_view_poses` 给每个 oblique 的足迹测量加 `overhead_view_pose`：位姿由当前观测的抓取点
位姿和腕部标定算出，与候选来自哪一步观测无关，这也是 vision_segment 带 `step` 时用的。没有当前腕部
相机或 `cam2world_cv` 时不加。没有这个配置键（RoboDojo 和已封存的工具配置）就没有这次读取。
