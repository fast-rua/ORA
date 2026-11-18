# Toolkit legacy replay fixture

2026-10-23 general_pickup 首局的调用 0 前观测，source run id 为
`e0003-first-episode-robodojo-a8b9a603d7ce--20261023T021000Z`。
原 run 的 plan.source.commit 为 `36b17f4e3804fbac4f6c051e21a5c5c5d438aba5`（dirty=false）；这是数据接口改动前的旧格式。
接口开发基线为 `69912fbc056c54172441248ac58259e715a0f196`，不把它冒充样本运行版本。

原始 `observation.json` 和 `proprioception.json` 按原字节复制，sha256 在
`snapshot.json`，未重编码。快照由 RSI `snapshot.v0` 生成后裁剪：绝对路径
改为 run-relative，source_run 改为 `.`，所有没有复制的文件标记 recorded=false。
M5 为端到端 geometry 示例增加 head 的 rgb.png、depth-m.npy、calibration.json；
均从同一原 run 按原字节复制，逐个核对快照已有 sha256 和 bytes，才把对应 recorded
改为 true。总计 1,670,046 字节。两只 wrist 相机文件仍缺失；不复制真值、凭据、
控制日志或服务输出。原始 manifest/proprioception 和 rsi-provenance.json 字节不变。

用于断言发布观测和本体感知的字节保持、缺失接口请求不能猜测恢复。服务测试
另用小型合成数组和假客户端；M5 示例也用假客户端处理真实 head 图像。
不把缺失的真实服务产物伪造成历史响应，legacy comparison 始终为 unavailable。
