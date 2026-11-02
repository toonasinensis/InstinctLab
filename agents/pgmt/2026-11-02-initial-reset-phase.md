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
初始短 episode 不升级，因此首次完整成功和课程升级也会相应延后。比较前 500
轮跟踪误差时必须同时查看地形等级，不能把更低难度的误差降低归因于策略改善。

## 验证与证据入口

CPU 回归入口是 `scripts/pgmt/test_initial_time_out.py`，覆盖首次期限、三次完整
周期、重复 startup reset、提前失败、真实步数、RNG 隔离和训练/播放配置范围。
开发机与分布式节点都通过了 5 项 CPU 回归。原生固定策略对照使用 seed 42、
128 环境，每组 1500 个控制步；初始噪声观测和归一化权重摘要完全一致。
跨 reset 与 capacity64 覆盖的原生 10 帧噪声历史重建精确，初始截断不计完成、
不升级及终端 bootstrap 检查通过。

| 指标 | 同步首轮 | 首轮错峰 |
| --- | --- | --- |
| 单步完成数峰值 / 128 环境 | 122 | 3 |
| 单步 reset 占比峰值 | 95.3125% | 2.34375% |
| 步 501--1500 身体误差标准差 | 2.11118 mm | 1.01561 mm |
| 同窗口身体误差峰谷差 | 28.8534 mm | 4.86858 mm |
| 同窗口平均地形等级 | 0.867766 | 0.553875 |

该成对实验确认 reset 集中度下降；误差波动同时降低，但地形曝光也发生变化。
它不证明策略能力提高。分布式节点实测驱动 550.54.15、RTX4090、
Python 3.12.13、Torch 2.11.0+cu126，并已通过原生 CUDA/Warp 推进。

原生固定策略对照、500 次 actor 更新短训及其验收文件存放在仓库同级的
`InstinctLab-local/reset-phase-20261102/`，分布式持久挂载对应
`/mnt/data/JIN/PGMT/InstinctLab-local/reset-phase-20261102/`。

原始日志诊断在 `log-diagnosis.json`；运行源码摘要在 `source-manifest.json`。
固定策略结果在 `effect-analysis.json` 和 `paired/`，短训为分布式任务
[26241](https://open.tars-ai.com/pai/distributed-training/detail/26241)。
短训使用双卡、每卡 4096 环境，与历史训练保持相同规模；固定策略成对实验
用于隔离 reset 机制。500 次更新只覆盖早期变化，不能据此声称最终训练质量提高。

该任务最终 Succeeded，前台训练及验收退出码均为 0。两 rank 各 532 条记录
（32 warmup + 500 train），每 rank 的 actor/critic/WM 更新分别为
500/10640/5320，真实采集 1362 步，PPO 更新为 0。保存的 checkpoint 中
392 个张量全部有限，BPTT/WM/terrain 各保存两 rank RNG。该检查不包含
运行中两 rank 参数的逐字节摘要。

按真实采集步附近正负 30 步、同一日志口径比较，完成数峰值为：

| 控制步窗口 | 历史同步训练 | 首轮错峰短训 |
| --- | --- | --- |
| 500 附近 | 1247.666748 | 8.666667 |
| 1000 附近 | 1116.000000 | 8.750000 |

这是每个 learner 行对当轮 2/3 个采集步及两 rank 的均值，不是单步瞬时
完成数或成功率。两处历史集中超时尖峰均被打散。完整短训中此日志指标的
最大值为 11，仍应保留自然采样波动和真实失败的诊断。
最终计数、checkpoint SHA256 和检查范围见 `verification.json`；前台退出
见 `platform-exit.json`；图见 `reset-phase-comparison.png`。

## 尚未解决的早期起伏

本次只确认同步 reset 的周期尖峰被打散，不能声称所有训练波动改善。
历史与错峰短训的整段误差曲线标准差（532 行，包含 warmup）分别为
4.59078 mm、4.96066 mm；只计 500 个 train 行则为 3.91486 mm、4.56679 mm。
这些是随训练时间变化的在线指标，含均值趋势及不同地形曝光，不是固定场景
泛化性能或跨 seed 方差，也不能据此认定修复伤害了训练。

两版都有较宽的早期波峰：历史在 actor 更新 43、控制步 219 达到
58.8685 mm，错峰在更新 39、控制步 209 达到 62.6902 mm。此时地形均为
L0，且早于首次完整 500 步超时。开始 actor 更新后的学习瞬态是一种解释，
但 WM、critic、actor 或其他因素的因果尚未隔离；本次没有修改学习率、
warmup、模型更新次数或奖励去抹平它。

## 5000 次 actor 更新完整实验（验收待完成）

后续完整训练已提交为 TARS
[26274](https://open.tars-ai.com/pai/distributed-training/detail/26274)，唯一 run ID
为 `pgmt-reset-phase-2x4096-5000-25d4fce-20261102-r1`。运行代码复用已验证的
`25d4fce` 不可变快照，从相同 PPO 初始化和新 WM 开始，不从短训 checkpoint
续训。配置保持单节点双 RTX4090、每卡 4096 环境、H32、real critic、seed 42
及 FP32，目标 5000 次 actor 更新。

证据入口为仓库同级 `InstinctLab-local/reset-phase-5000-20261102/`，远端持久
挂载为 `/mnt/data/JIN/PGMT/InstinctLab-local/reset-phase-5000-20261102/`。
`launch-manifest.json` 固定配置及输入摘要；`job-identity.json`、提交回执和
`observe.py` 用于定位及只读跟踪任务。每 100 轮保存 checkpoint，1000/2000/
3000/4000/5000 里程碑检查张量有限性及 RNG。完整计数预期为每 rank
5032 条记录（32 warmup + 5000 train）、actor/critic/WM 分别
5000/100640/50320、collection 12612、PPO 0。

同一前台任务在训练后自动比较历史 `57587f8` 与新模型的最终 5000 轮
checkpoint。三个评估 seed（1001/1002/1003）各覆盖 10 难度 × 10 地形列
× 4 重复，每模型合计 1200 个首 episode。评估禁用初始错峰终止、课程和
观测/动作噪声，保持标准 500 控制步上限，并检查成对初始状态及归一化器。
`training-verification.json`、`training-comparison.json` 与
`quality-comparison.json` 分别记录执行完整性、在线尖峰变化及固定条件质量。
单个训练 seed 的评估区间不能代表训练 seed 方差；脚本通过也不自动表示
质量改善。完整训练及上述评估结束前，此实验不记为验收通过。
