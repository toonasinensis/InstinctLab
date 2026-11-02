# BPTT 训练吞吐：先定位执行开销

## 现象与适用范围

H32、每卡 4096 actor 轨迹、WM 2048 × 10 次、critic 20 次更新的 PGMT
训练，比本地 SuperMimic Vibed 的历史测速慢。不能只凭低 GPU 利用率推断
batch 太小，也不能把 `policy_critic_seconds` 全部归因于 critic。

双卡开发机的细分采样显示，critic 拟合只占这部分的一小部分；想象前向与
actor 反向是主要开销。原实现每步为 support、当前奖励和前一状态奖励各做
一次 FK，H32 合计 96 次。重复 FK 还产生了独立的 autograd 子图。

## 工程处理

- support 保留原有递归滤波和梯度，供奖励与终端 value 使用。
- 奖励与软终止不反馈下一步动作，可在完整 rollout 后按 B × H 批量计算。
  当前及前一状态共享一次轨迹 FK，保留加速度奖励的跨步梯度。
- 生产路径保留原生几何矩阵运算和精度设置；随机采样、观测噪声、更新计数、
  checkpoint 和出界拒绝逻辑保持原样。
- 汇总 critic loss 后再传回 CPU，避免为每个 minibatch 的日志值同步。
- 后续进一步让 support 只计算脚位置所需的祖先链，省去无用的 COM 速度。
  G1 的 support 链从40个 URDF link 缩到13个，全身奖励链保留30个祖先 link。
  仍使用同样的原生矩阵/四元数公式。实际 G1 CPU 输出与梯度逐位一致，
  满尺寸 GPU H32 对照的状态/输出/RNG 一致，平均梯度相对差约1.1e-6。
  这证明计算一致性；减少算子本身不保证端到端提速，需测完整训练轮。

这些属于执行优化。不能用缩小 batch、减少 WM/critic 更新、缩短 H、detach
support/状态或更改精度来冒充同配置加速。

## 比较与验收边界

Vibed 的历史配置使用 BF16、MLP actor、较小 critic，并且没有 PGMT 的地形
观测和完整奖励集合。即使 batch/H/更新次数一致，也不是相同计算图。
本次 PGMT 保留 FP32 与原有 TF32 设置。精度优化需单独比较。

尝试过编译几何核，但全尺寸前后对照出现明显状态特征和梯度偏差，因此没有
纳入生产路径。独立实验还显示，TF32 开启时，逐元素乘加不能直接代替原生
矩阵乘法并声称数值等价；这项尝试同样撤回。问题的完整归因尚未完成，不能
仅凭 dtype 或精度开关相同就宣称执行精度一致。

测速需固定硬件、每卡规模和更新次数，区分真实收集、WM、想象前向、actor
反向、真实 critic 数据构建及 critic 拟合。普通轮、验证轮、启动/编译和
profiler 本身的开销分别统计；各 rank 同一轮只计一个全局样本。

原生 TF32 下也不应要求批量与逐项计算逐位相等：实测前向特征完全一致，
批量计算的梯度相对差约 0.09%。保留该生产设置下的诊断记录，同时在独立
IEEE FP32 对照中检查数学重排；正式计时恢复原 TF32 设置。

正确性依靠前后实现的同输入 loss/梯度与噪声 RNG 比较，以及已有验收中的
策略参数有限差分、未来 heightmap 梯度、WM 参数冻结、两 rank 一致、出界
拒绝和 optimizer/RNG reload。有限差分只证明计算图一致性，不证明真实
动力学 Jacobian 准确；吞吐提升也不能代替跟踪效果评测。

## 激活重算与计时归因

每卡4096、H32、FP32、满 Replay1000 下，关闭 actor 激活重算曾在前向阶段
达到约44.5GiB可见显存上限并 OOM，未进入更新；两项重算都关闭没有继续试。
仅关闭 WM 重算能完成68次有界更新，状态/输出/RNG 一致，IEEE 平均梯度
相对差约1.1e-6，Torch 分配峰值由约23.1GiB到28.1GiB。

两个独立窗口的整轮均值曾显示2.887秒到2.277秒，但真实收集段也由0.528秒
到0.270秒。该段不受 WM 重算开关影响，不能把整轮差归因于开关。进一步
固定数据与噪声、同进程交替 ABBA，各20次 actor 前向/反向计时，均值仅从
1.181秒到1.172秒，中位数约快3%；配对 cycle 的时间差区间跨零。
默认仍保留两项重算，不用额外约5GiB激活显存换未经证实的显著吞吐收益。

该交替测试的准备阶段统一填充Replay并拟合10次WM；计时阶段不执行优化器
更新。它只隔离受影响的 actor 计算，不代表完整训练轮或长期学习效果。
证据在 `InstinctLab-local/bptt-amp-compile-20261102/final-optimization-report.json`。

## 后续 5000 轮对照

`57587f8` 的 real-H32 训练完成5000次 actor 更新，两 rank 各5032条记录，
退出与 checkpoint 核验通过。满 Replay1000 的普通轮计算时间均值由4.872秒
到2.756秒，各4460个全局样本；原版为开发机 driver550，加速版为分布式
driver570，因此该吞吐差包含后端因素。

同一推理 runtime、相同初始状态的每版1200个首次 episode 中，加速版身体
误差多2.45mm、关节误差少0.00391rad，总失败率从3.58%到3.00%，但 level8
失败从2/120到9/120。结果有取舍，不能声称完全无损。只有一个训练 seed；
配对 episode 区间不代表训练 seed 方差。训练尾段统计还需同时看课程难度。
完整里程碑、WM holdout 及分地形结果位于
`InstinctLab-local/bptt-speed-quality-20261102/final-training-comparison.json`
和 `quality/comparison.json`。新的 BF16/编译候选未用于这次正式训练。

## 证据入口

基线源码为 `e1e1aa4`，实验与性能报告在仓库同级
`InstinctLab-local/bptt-speed-20261102/`，开发机对应
`/mnt/data/JIN/PGMT/InstinctLab-local/bptt-speed-20261102/`。
其中保留源文件和 checkpoint 摘要、逐 rank 计时、CPU profiler、原始 CUDA
trace、FK 单项前后测试及梯度误差。原始测速表和运行报告不提交到 Git。

Vibed 历史证据在参考仓库
`logs/ddp-unified-20261027-053500/performance.json` 及其逐 run 配置中。
编译行为参考 [PyTorch torch.compile 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile)。
