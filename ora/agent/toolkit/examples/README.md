# 可修改的 replay 脚本

这里的 Python 文件是给 RSI agent 改写的实验入口，不属于只读 lib 的版本哈希。
每个文件定义 `run(api, args)`，返回严格 JSON 可序列化的值；显式通过
`ora toolkit replay RUN --at K --script PATH` 运行，不会自动安装进模型工具面。
只通过数据接口和 lib 取数据；这些脚本没有模拟器或在线动作能力。

`ground_geometry.py` 使用调用前的公开观测，在新感知请求的全部候选上计算点云和
世界坐标系下的 PCA。输入为 `camera`、`text`、`max_candidates`，默认取父工具参数。
输出包括 observation identity、每个候选的点数/中位中心/主轴/尺寸（米）、轴退化标记
及独立 comparison。空点云返回空几何，不能当作完整物体形状或抓取决策。

观测从记录取，缺失即失败；ground 重新请求；mask_to_cloud 和 principal_geometry
是纯计算。必须传有效 `--service-report` 才能请求感知。未记录的 IK/新动作不能靠
此入口补算。示例的 decoded RGB 数组输入与只记录开关使用同一 BlobRef 表示；仍须
同一接口游标、参数和服务/配置身份才能得到有效比较。

CLI 装配在 `ora/orchestrator/toolkit/`；完整用法、输出和原进程执行边界见
[toolkit-io](../../../../docs/reference/toolkit-io.md)。
