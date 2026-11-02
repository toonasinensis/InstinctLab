# BPTT 短时域与 critic 依赖

记录日期：2026-11-02。工程决策已采用；机制解释待验证。

## 现象与适用范围

用户测试反馈短时域表现不如长时域。本轮在线训练统计也显示，在 real 和 imagined 两条 critic
路线中，H16/H32 的跟踪误差和失败占比均优于 H8。当前选择 actor H32 作为工程默认值。

该观察限于 `dance1_subject2`、简化地形及既有课程、19 项非 contact/force 奖励、两卡各 4096
环境、5000 次 actor 更新和单个 seed 的这轮实验。Actor 由 PPO25836 的 `model_2000.pt` 初始化，
2048×3 WM 和独立 MLP critic 从头初始化。WM 监督长度固定 H8，actor 使用 H8/H16/H32。
这不是 stage1-only 实验，也不是“任意任务下 horizon 越长越好”的结论。

各组在线课程最终难度不同。用户测试趋势与在线日志相符，但目前没有统一固定场景协议下的多 seed
统计，不能把在线结果直接解释成严格隔离后的因果结论。

## 证据入口

历史源码快照：`c0fa1cc9feb04f1c71d55713e2842fb283affb18`。

仓库同级证据目录：`InstinctLab-local/bptt-vibed-aligned-20261101/`。

- `completion-audit.json`：六组最终 checkpoint 路径、SHA、有限值和更新计数。
- `completion-metrics-summary.json`：actor 4901–5000 的在线统计，以及 actor 更新 4976 的 WM 验证。
- `final-training-statistics.json`：各组固定/当前 holdout、共同 19 项奖励误差、有效比例及测速显存。
- `active-experiments.json` 和 `manifests/`：实验身份、来源模型和不可变运行配置。

原始日志和模型留在上述目录。需要精确复现历史六组时使用该历史快照和原始 manifest。

## 首要机制假设

忽略软终止时，当前 actor 目标为：

```text
J_H = sum(t=0..H-1, gamma^t * r_t) + gamma^H * V(z_H)
```

窗口内的奖励通过 WM 和解析奖励直接回传；窗口以后的收益由末端 critic 补充。
当前控制步为 20 ms，H8/H16/H32 分别覆盖 160/320/640 ms。短窗口可能还看不到动作对后续
平衡、落脚和跟踪的影响，因此更依赖 critic 给出有用的长期方向。

Actor 更新时 critic 参数冻结，但其输入梯度仍回传给 policy。Critic 的值回归误差小，并不保证
它对状态的梯度能正确指导动作。当前 critic 从头初始化，仅有 32 轮 actor 冻结预热，因此
“critic 尾部价值及梯度尚未学稳，影响短时域策略更新”是首要怀疑。

这里不能只比较 bootstrap 的折扣系数：gamma=0.99 时 H8/H32 的系数约为 0.923/0.725，
都不小。实际奖励梯度和末端价值梯度谁占主导，尚未分别测量。

## 尚未排除的耦合因素

- Actor horizon 同时决定 critic 的训练窗口和 lambda-return 长度。每轮候选状态为 `4096×H`，
  H32 比 H8 多四倍；真实数据还会受到终止 mask 的影响。这轮并非只改变 actor 的 bootstrap 距离。
- 较大 WM 也从头初始化。WM 预测误差、状态分布变化和动力学梯度误差可能共同影响结果。
- 更长 rollout 累积更多模型误差，不能把当前趋势外推到更长时域。
- 课程和 motion 失败采样会随策略表现变化，最终各组实际训练分布不同。

## 当前工程决策

默认 actor H32，结束日常 actor horizon 消融，保留 real/imagined critic 两种监督来源。
继续使用 WM 监督 H8 和 H1/4/8/16/32 验证；它们用于检查预测随时间的误差，不是 actor horizon 消融。

本次清理保留 H32 的采样窗口、解析目标、软终止、critic 架构、优化器及精度。
Actor 与 critic 训练长度解耦属于单独的算法调整，应与代码整理分开验证。

## 后续验证方向

- 在裁剪前分别测量窗口内奖励和末端价值对 actor 的梯度范数、比例及夹角。
- 在独立真实轨迹上检查 critic 对多步回报的误差、偏差及解释方差，区分自举目标拟合与真实回报预测。
- 固定初始状态、课程难度、motion 和训练预算后比较，进一步隔离 critic 数据量与目标长度的影响。

这些诊断尚未实现或执行；它们是检验上述假设的方向。

## 相关原始研究

- [SHAC](https://arxiv.org/abs/2204.07137)：短窗口可微奖励与末端价值估计的组合。
- [Model-Based Value Estimation](https://arxiv.org/abs/1803.00101)：限制模型展开深度以控制模型不确定性。
