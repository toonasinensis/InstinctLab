# BPTT 首轮超时同步与错峰

## 现象与已核实原因

`c0fa1cc` 和 `57587f8` 的 real-H32 日志都存在每 200 次 actor 更新一次的
完成数尖峰。它对应 500 个真实控制步，即默认 10 秒 episode 时限；每轮
交替收集 2/3 步。随机参考动作起始帧没有改变所有环境同时从零开始的 episode
计时。提前失败、动作结束或出界只能逐渐打散其中一部分环境。

这里的 actor 从 PPO `model_2000.pt` 初始化，WM 与独立 critic 才是新建的。
`57587f8` 第一次集中超时所在的 3 个控制步内，每卡平均约 3743/4096 个
环境完成，失败数为零。集中 reset 前后还存在明显跟踪误差变化，并伴随地形
课程升级。这些证据支持采样阶段同步，不能单凭在线曲线认定策略权重震荡。

## 工程处理

- 仅 BPTT 训练增加独立的 `initial_time_out` 截断项，首次期限在 1 到完整时限
  之间按环境均匀采样。它使用独立、按环境 seed 初始化的 RNG。
- 不改 `episode_length_buf`、参考时间、机器人状态或已有历史。任意真实 episode
  结束后，该环境不再触发初始截断，后续仍用完整 500 步时限。
- wrapper 与 runner 在第一次控制步前会重复 reset；这些调用保留首次期限。
- 原 `time_out` 完成条件保持完整时长，因此初始短截断不能计完成或成功升级。
  失败仍正常判定。初始截断沿用终端 value bootstrap 和 replay 跨 episode mask。
- `PGMT/initial_truncations` 单独计数。PLAY 与 PPO 默认不启用；历史同步协议
  可用 `env.terminations.initial_time_out=null` 显式选择。

这改变了采集协议。应单独记录开关，不能将曲线更平滑直接解释为训练效果提高。
新协议仍可能出现由真实失败或特定动作造成的相关 reset，需要保留失败原因诊断。

## 验证与证据入口

CPU 回归入口是 `scripts/pgmt/test_initial_time_out.py`，覆盖首次期限、三次完整
周期、重复 startup reset、提前失败、真实步数、RNG 隔离和训练/播放配置范围。
原生固定策略对照、500 次 actor 更新短训及其验收文件存放在仓库同级的
`InstinctLab-local/reset-phase-20261102/`，开发机对应
`/mnt/data/JIN/PGMT/InstinctLab-local/reset-phase-20261102/`。

原始日志诊断在 `log-diagnosis.json`；运行源码摘要在 `source-manifest.json`。
短训使用单卡、4096 环境，历史训练是双卡，每卡 4096 环境，因此历史日志用于
比较周期现象；固定策略成对实验用于隔离 reset 机制，不声称训练质量严格等价。
