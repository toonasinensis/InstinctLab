# BPTT 采集量与高难度地形学习

## 现象与验收口径

用户已在当前地形和课程上实测：平均 terrain level 稳定在 6 左右可作为所有
地形学会的经验指标。此前 BPTT 5000 次 actor 更新的末段均值约 5.7--5.8，
且用户在可视化中观察到 L9 楼梯失败，因此尚未满足这一标准。
PPO no-contact-force 版本在第 2000 轮最终记录为 6.0872。
最高级通关后随机分配等级的理论均值解释，不替代上述项目经验和实际回放。

当前 BPTT UTD 0.4，每个 learner round 交替采集 2/3 个控制步；500 步完整
episode 约需 200 次更新。PPO 每轮采集 24 步，约需 21 轮。相同迭代数不能
直接代表相同真实环境经历。BPTT 从 PPO actor 初始化，也需检查原有高难度
能力是否在训练中退化；采集量不足和 WM/策略优化偏差尚未因果隔离。

## 三组完整实验

代码提交为 `232730a`，只新增 `agent.bptt.train_collect_steps`，默认 `None`
保留原有调度。参数只作用于正式 actor 训练，初始预填 32 步与 32 个 warmup
round 继续使用 UTD 0.4，三组均在采集 112 步后开始 actor 更新。
旧 checkpoint 缺省此参数时按 `None` 解释；更改采集配置不允许直接 resume。

| 正式训练每轮采集 | actor 更新 | 每 rank 总采集步数 | 每 rank WM/critic 更新 |
| --- | --- | --- | --- |
| 8 | 5000 | 40112 | 50320 / 100640 |
| 16 | 5000 | 80112 | 50320 / 100640 |
| 24 | 5000 | 120112 | 50320 / 100640 |

每组使用单节点双 RTX4090、每卡 4096 环境和 actor batch 4096、H32、real
critic、FP32、seed 42、同一 PPO2000 初始化、fresh WM、首轮独立超时错峰。
每轮 WM 更新 10 次、critic 更新 20 次，奖励、学习率、replay 容量、噪声、
地形和 episode 时长均保持一致。当前 2.5 步对照为任务 26274，使用此前
不可变代码快照，不覆盖其输入。三组独立提交，不以其他组进度作为启动门槛。

三组已于 2026-11-02 唯一提交到 C1 RTX4090-Group1，均为双卡、32 CPU、
205 GiB 内存；启动观察时平台均为 Running，正在运行预检，原生短训及完整
训练验收尚未完成：

| 正式采集步数 | TARS 任务 |
| --- | --- |
| 8 | [26290](https://open.tars-ai.com/pai/distributed-training/detail/26290) |
| 16 | [26292](https://open.tars-ai.com/pai/distributed-training/detail/26292) |
| 24 | [26293](https://open.tars-ai.com/pai/distributed-training/detail/26293) |

三组正式命令已逐项比较，只在采集参数、输出目录和 run name 上存在差异，
记录见外部 `config-consistency.json`。实际资源及运行时检查仍以各组
`preflight.json` 为准。当前聊天的 `pgmt-5000` heartbeat 已扩展为统一跟踪
原对照及这三组，完整训练和固定评估后再作联合验收。

首次实际预检确认三组都是双 RTX4090，Torch/CUDA/Warp 版本相同。平台分配
的节点仍有差异：8/16/24 步组可见显存分别约 44.40/44.52/47.38 GiB，
驱动分别为 570.153.02/550.54.15/570.153.02，均在已验证运行时范围内。
记录见外部 `resource-consistency.json`。训练规模和参数一致，墙钟比较需保留
这些节点差异，不能声称实际硬件及驱动完全相同。

每组先执行同规模 5 次 actor 更新的有界验证，再从相同起点重新正式训练。
验证应有每 rank 37 行、WM 370、critic 740、warmup 末端采集 112 步；总
采集分别为 152/192/232 步。完整训练应有每 rank 5032 行和上表最终计数，
检查两 rank 计数、有限张量、checkpoint 及 RNG。

后续现场检查确认三组原生双卡短验证全部通过：各 rank 37 行，warmup 末端
均为 112 步，正式 5 轮分别固定采集 8/16/24 步，最终计数符合上述预期。
三份 checkpoint 的张量/优化器有限，BPTT/WM/terrain 各保存两 rank RNG。
证据见外部 `smoke-acceptance.json` 及每组 `smoke-verification.json`、
`smoke-checkpoint.json`。各 worker 已自动转入 fresh 5000 轮流程，完整训练
和质量验收仍待完成。

## 后续比较

同时按 actor 更新、真实采集步数和墙钟时间比较 terrain level 曲线，保留
首次达到 6、滚动及末段统计，单次越过 6 不作稳定学会的结论。采集更多也
会降低单位新数据获得的训练量，因此需结合 WM H8/H32 预测误差检查适应情况。

最终模型使用 seeds 1001/1002/1003，每个 seed 对 10 难度 × 10 地形列 × 4
重复进行固定首 episode 评估。评估关闭课程、噪声和首轮短超时，保留标准
500 步上限；与相同条件的 PPO 初始化和已有 BPTT 对照比较，特别报告 L9
楼梯的失败率、完整完成率和跟踪误差。只跑一个训练 seed，不将评估区间当作
训练 seed 方差。

证据目录为仓库同级 `InstinctLab-local/bptt-collect-sweep-20261102/`，远端为
`/mnt/data/JIN/PGMT/InstinctLab-local/bptt-collect-sweep-20261102/`。
`plan.json`、各组启动清单、任务身份及后续验证文件是实际执行入口。
本文当前只记录实验设计，完整验收待训练及评估结束。
