# PGMT 调试经验索引

这里保存项目级开发经验，供后续开发者和代码助手查阅。一条经验一个 Markdown；本文件只维护索引。
这些文件属于当前仓库的工程文档，不是全局 agent 配置或系统记忆。

| 日期 | 经验 | 工程决策 | 解释的验证状态 |
| --- | --- | --- | --- |
| 2026-11-02 | [BPTT 短时域与 critic 依赖](2026-11-02-bptt-horizon-and-critic.md) | 默认 actor H32，保留 real/imagined critic | 表现趋势已观察；critic 因果解释待验证 |
| 2026-11-02 | [BPTT 训练吞吐与重复 FK](2026-11-02-bptt-throughput.md) | 保留 support 递归；原生批量奖励与软终止，拒绝未通过的编译路径 | 原实现瓶颈已实测；加速与正确性见外部验收报告 |
| 2026-11-03 | [BPTT BF16 与编译边界](2026-11-02-bptt-bf16-and-compilation.md) | BF16关闭；闭环/FK保留原生，选择性编译显式启用 | real H32正式5000轮耗时−27.9%；用户反馈趋势相近，整体配对指标改善、局部有退步；单训练seed |

## 记录约定

每条经验包含：现象、适用配置、证据位置、原因假设、尚未排除的因素、工程决策及后续验证方向。
区分用户反馈、在线训练统计、固定场景评测和机制假设。新增证据可修订同一条经验，并说明修订原因。

日志、checkpoint、测速数据及完整实验报告继续保存在仓库同级的 `InstinctLab-local/`，不提交到 Git。
经验文件只沉淀可复用的判断、适用边界和证据入口，不复制原始实验报告。

当前训练协议见 [BPTT_EXPERIMENTS.md](../../source/instinctlab/instinctlab/tasks/shadowing/pgmt/BPTT_EXPERIMENTS.md)。
