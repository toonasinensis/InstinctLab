# BPTT 的 BF16 与整段编译

## 适用范围

PGMT 地形任务，actor H32，headless WM 监督 H8；actor4096/GPU、
WM2048/GPU、每轮10次 WM 更新、Replay1000、real/imagined 独立 critic。
本条记录实现约束和验证方法；数值与吞吐结论以外部报告为准。

## 已确认的工程问题

- 网络输出转回 FP32 不能恢复 BF16 已经丢失的尾数。尤其是动作头、
  地形采样位置的 selector 头和 WM delta 头，误差会进入后续递归状态。
  FP32 的几何运算本身不能保证 H32 的梯度方向与 FP32 网络相同。
- PyTorch2.11 的 AOTAutograd/min-cut 在整个 actor 前向被 checkpoint 时，
  曾因跨边界复用的 `split_with_sizes` 元组报错。将输入解包放在 checkpoint
  外，只重算纯网络部分后，FP32 eager 与原实现的独立对照通过。
  这不构成编译数值或训练效果的验收。
- 将原生 FK、噪声、历史、奖励全部展成 H32 的单一大图，冷启动可能远大于
  一轮训练耗时。冷编译、重新编译和稳态时间必须分开记录，不能用仍在编译
  的 GPU 利用率解释正常训练吞吐。
  最新 C1 两卡完整 H32 候选也到达40分钟单 case 上限，未完成编译后的
  前向/反向，BF16 编译阶段因前置失败未启动。未计作加速成果。

## 本轮已观察的候选结果

在相同2×4096、满Replay1000、两种激活重算均开启的窗口中，FP32 eager
普通轮均值2.695秒，全部神经层含输出头使用BF16的 eager 为2.620秒；每种60个
普通轮、4个验证轮，前4轮不计。BF16 的显存均值没有降低。

同一批窗口与同一噪声的 H32 检查中，BF16 候选的跨 rank 平均 actor 梯度
余弦为-0.72866、相对差2.01905，valid mask 和 RNG 一致。此候选未通过数值
验收，保持默认关闭。它虽然完成68次有界更新且 checkpoint/优化器/重载均通过，
这些结果不推翻数值验收失败；后续固定场景结果也不能单独证明长期训练非劣。

固定初始状态的3个推理seed、每版1200个首次episode中，BF16分叉的平均身体
误差少1.40mm、关节误差多0.00167rad，失败率由3.42%到3.92%。失败率差的
配对episode区间跨零。结果有好有坏，不能据此将数值偏差解释为所有指标退化，
也不能证明长期效果保持。两版推理统一使用FP32，比较的是训练后权重的差异。

这里比较的是已训练源模型之后的短程分叉，不能归因于 fresh critic 或 WM
初始化差异。误差在动作、地形 selector、WM delta、critic 等路径中的具体贡献
尚未隔离；不能仅凭当前结果断言某一个输出头是唯一原因。

## 实现约束

最初候选的局部 autocast 包含神经网络主干和输出头。相同批次、生产 TF32 下
5 次 actor 更新的参数更新量余弦，fresh Adam 为0.54560，恢复 Adam moments
为0.89858；相对差分别0.95320和0.45086。保持 moments 能缓和偏差，但不能
由此认定更新等价。

用户随后批准“优先保住精度”的定位验证：`bf16_heads=False` 保持动作、地形
selector、WM delta 和 value 的最终线性层为 FP32，主干仍允许 BF16。
进入输出头的隐藏特征仍经过 BF16，不能假定该修改必然解决梯度偏差。
`bf16_heads=True` 保留原候选的精度语义。BF16 总开关仍默认关闭。
后续同批数据的5步 Adam 定位中，FP32 输出头候选的参数更新量余弦为
0.71997（fresh moments）和0.93501（restored moments），相对差为0.77444
和0.36491。该批次全部 BF16 候选的对应余弦为0.82158和0.95414。
恢复最终输出头没有在这次检查中消除偏差；这不能推出单个头的普遍因果作用，
也不能单凭某一批次判断长期学习质量。没有据此启用 BF16 或启动5000轮。
网络边界回到 FP32。
状态积分、旋转、FK、地形坐标、奖励、软存活、return 和 loss 归约保持原精度；
参数、梯度和 Adam 状态保持 FP32。已有 normalizer 的统计类型与计数器不变。

完整 WM-H8 loss、policy-H32 和 value forward 是独立编译入口；原模块保留
参数与 checkpoint 的所有权。随机数、优化器、通信、计数和错误落盘留在入口外。
每步噪声预生成必须保留原始调用顺序和张量形状，并核对 RNG 状态完全一致。
actor/WM 在想象阶段的激活重算分别控制；不能顺带改变监督训练更新数或 loss。

分层编译可作为复用原生重复计算的候选，但不能用 eager graph break、截断梯度
或替换几何公式来伪装完整 H32 编译。候选必须独立完成数值和满尺寸验证。

`fullgraph=False` 允许编译/eager 区域混合，并不等于截断 BPTT；只要没有
detach/no_grad，autograd 仍连接各区域的反向图。但该开关不会自动将固定 H32
循环拆成32个可复用小图。可选的后续方案是让外层循环留在 Python，明确编译
单步转换，再分别编译 WM-H8、value 和奖励/return。该方案与本轮尚未完成的
整段 H32 编译不同，需要单独记录编译边界及冷启动/重编译/稳态证据。

## 验收方法

1. 用独立的原始实现核对 FP32 状态、输出、梯度和 RNG；另记录生产 TF32 下
   的差异。不要把关闭 TF32 的对照结果说成生产设置下的逐位一致。
2. 同一窗口和噪声比较局部及跨 rank 平均梯度方向、相对误差和范数；比较
   多步 Adam 的参数**更新量**，分别使用 fresh/restored optimizer moments。
   总参数范数接近不能说明更新接近。BF16 不适合作为平滑有限差分的参照。
3. 满 Replay、2×4096 的正式尺寸窗口测速，剔除首轮和重新编译轮；普通轮与
   验证轮、compute 与 wall cycle 分开。128 环境仅验证兼容性。
4. 核对未来地形梯度、invalid-window 保存、两种 critic、rank 一致性、原始
   checkpoint key、精确 reload、参数/梯度/优化器 dtype，以及 PPO0。
5. WM 固定/当前 holdout 看 H8/H32、共同19项奖励和 valid_fraction；固定场景
   看跟踪误差、失败、motion-end、timeout、episode 长度和地形分层。
   有限值、低 loss 或一次梯度余弦都不能单独证明训练效果保持。

## 证据入口与边界

外部记录：`InstinctLab-local/bptt-amp-compile-20261102/`。
短程验证从同一个已训练 checkpoint 分叉；不能据此推断 fresh WM/critic 的
5000轮训练非劣。单个训练 seed 下的配对 episode 区间也不代表训练 seed 方差。
本轮原有正式训练26085和其配对评测使用不可变57587f8快照，未切换到这些候选。

PyTorch2.11 参考：
[backward autocast](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/torch.compiler_backward.html)、
[nested compile region](https://docs.pytorch.org/docs/2.11/generated/torch.compiler.nested_compile_region.html)、
[混合 compiled/eager 反向](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/torch.compiler_faq.html)。
