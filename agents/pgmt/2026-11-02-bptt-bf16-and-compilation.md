# BPTT 的 BF16 与编译边界

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
  C1 任务26190的两卡完整 H32 候选到达40分钟单 case 上限，未完成编译后的
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

原模块保留参数与 checkpoint 的所有权。随机数、优化器、通信、计数和错误落盘
留在编译入口外。每步噪声预生成保留原始调用顺序和张量形状，并核对 RNG 状态
完全一致。不能顺带改变监督训练更新数、loss、重算或优化器设置。

`fullgraph=False` 允许编译/eager 区域混合，并不等于截断 BPTT；只要没有额外
的 detach/no_grad，autograd 仍连接各区域的反向图。但该开关不会自动将固定 H32
循环拆成32个可复用小图。必须明确记录实际编译边界。

## 分段编译的数值定位

`segmented-r16` 将 H32 循环留在 Python，单步转换和批量奖励/return 分别编译。
开发机2×128的完整前向/反向在冷启动约967秒后完成；生产 TF32 下，跨 rank
平均 actor 梯度余弦0.99979153、相对误差0.0263744，未通过既定0.002门槛。
原生拆分代码对原始实现的 IEEE 对照，状态/输出/RNG 一致，平均梯度相对误差
约2.75e-7。因此不能把结构拆分通过或编译完成当作数值验收通过。

`factor-r18` 是1×128的同批数据定位，不能用于报告满尺寸速度或训练效果。
仅编译奖励/return 时状态完全一致，梯度相对误差0.0003742；仅编译单步时
误差0.0146122。关闭矩阵填充、epilogue fusion和图模式替换的单步候选仍有
0.0484663的误差，未修复问题。不能据此认定某一个变换是唯一原因。

据此选择的实现保留原生 H32 闭环转换，只独立编译批量奖励/return、WM-H8 loss 和
value forward；全部32步梯度继续连接。`compiled_regions` 写入 contract 和
checkpoint，resume 检查边界不被默默改变。C1任务26264仍包含编译后的FK和终端value；2×128下平均梯度相对误差
0.0020522（rank0为0.0022261），仍未通过0.002门槛，未进入满尺寸。
后续候选将FK和终端value bootstrap保留原生，只编译解析奖励、软存活与return
的张量计算，再独立核验WM/value。该阶段未完成验收前保持默认关闭，没有启动
新的5000轮；后续已授权的正式验证见下文。
冷编译和稳态耗时分别记录；共享盘上曾出现每 rank 32个编译子进程大量等待
I/O，新验证将编译并发限制为每 rank 4个进程，不改变训练计算或宿主设置。

## 选择性编译的有界实测

C1任务26269在同一2×4096、actor4096/GPU、WM2048/GPU×10、H32/H8、
Replay1000、FP32/原TF32设置下完成68次短程更新与checkpoint精确重载。
原生闭环、FK和终端value保持不变，仅编译解析奖励/软存活/return、WM-H8 loss
和独立critic forward。满尺寸平均梯度相对误差分别约0.000909、0.000333和
4.09e-7；状态、valid mask和噪声RNG一致。5个编译图，区域内部无graph break。

剔除前4轮和重新编译轮，按唯一全局iteration取两rank较慢耗时：普通轮60个，
compute均值2.089秒、median2.065、P95为2.154；wall cycle 59个，均值2.095秒。
验证轮4个，含验证总耗时均值3.187秒。Torch峰值allocated约22.14/22.15GiB，
reserved约25.03/25.17GiB；actor窗口NVML显存均值约30.69/30.46GiB。
这些是有界已训练模型分叉的速度，不能直接替代fresh5000轮或跨后端对比。

C1任务26280进一步固定同一进程、数据、参数、Adam和RNG，每次计时前还原，
按4个ABBA周期、每模式8个样本，比较10次WM+1次actor+20次critic的learner。
原生均值2.067秒、编译1.732秒，耗时减少16.2%；WM部分约0.585到0.284秒。
该计时不含collect、moments估计、采样、验证和日志；全部更新探针均丢弃，最后
核对参数、optimizer、RNG和计数恢复一致。不能将1.732秒说成完整一轮耗时。

完整learner的一轮参数更新量，actor余弦0.998554、相对差0.053865，WM相对差
约0.001006、critic约0.000351。小的WM数值差经过更新与H32后仍可放大；这些
数据不足以宣称训练无损，编译继续默认关闭，BF16仍关闭。高精度除法舍入和关闭
浮点融合的单步定位候选也未消除偏差（1×128梯度相对误差约0.01419），未采用。

完整证据见外部`compiled-regions-report.json`和`checkpoint-verification-r20.json`。
其中actor5068/critic102000/WM51010是源actor5000之后的68轮有界分叉；
`bounded.pt`不是新的正式5000轮结果。初始化contract记录加载旧eager源，实际
编译fork的execution/边界以benchmark报告和保存checkpoint为准，不能混淆两者。

回归验证必须每项梯度使用独立前向和相同噪声。PyTorch2.11的AOT donated buffers
不支持在同一编译图上反复retain_graph；不要为通过测试关闭生产缓冲复用。

C1任务26294的两rank×128 real/imagined回归均通过，每组3次actor更新。
已检查第一步动作到H32末状态及未来heightmap的非零梯度、独立critic、
纯计算图FD、实际越界窗口集体拒绝/保存，以及模型/optimizer/RNG精确恢复。
这仍是功能与计算一致性验证，不是实际动力学Jacobian或5000轮质量证明。

## 正式5000轮与配对评测（2026-11-03）

在上述有界验收后，C1任务26350从同一个PPO25836原model2000初始化，
WM和独立real critic fresh，完成2×4096、H32的5000次actor更新。
直接对照为FP32 eager任务26085；完整env/agent导出配置的差异只有
`bptt_execution`、运行名称和日志目录。仍使用原TF32、19项奖励、PPO0，
BF16关闭，两种激活重算均开启，未修改采样、损失或优化器。

两rank各5032行（32 warmup＋5000 train），actor5000、critic100640、
WM50320、collection12612、sourceWM0、PPO0；worker退出码0，原始键布局、
两rank恢复状态及392个checkpoint张量的CPU有限值检查通过。

满Replay1000、剔0/1轮并按唯一全局iteration统计：普通轮各4460个，
compute均值2.756→1.987秒、median2.701→1.964、P95为3.220→2.768，
耗时减少27.9%。普通wall cycle各4459个，均值2.785→2.006秒；
验证轮各185个，wall cycle均值3.958→3.199秒。编译版两卡NVML显存
均值/最大值分别为29.603/29.603和29.665/29.665GiB，利用率均值
80.74%/78.49%、最大100%；Torch峰值allocated约22.15GiB。
这是两次正式训练的实测；驱动同为570，GPU UUID不同，不当作同进程配对测速。

开发机驱动由旧评测的550变为570后，两版model5000都在当前相同驱动与
不可变57587f8推理代码下重新评测：3个推理seed、每版1200个首次episode，
覆盖10 levels×10 columns×4初始片段。初始状态、normalizer、地形与样本身份
完全匹配；评测关闭观测噪声、动作探索与课程，不执行优化器更新。

| 指标 | 原版26085 | 编译26350 |
| --- | ---: | ---: |
| Body位置误差 | 0.052372m | 0.050654m |
| Joint位置误差 | 0.125636rad | 0.119318rad |
| 失败次数/1200 | 38 | 30 |
| Tile exit次数/1200 | 2 | 5 |
| Timeout次数/1200 | 1080 | 1085 |
| Motion end次数/1200 | 80 | 80 |
| 平均episode步数 | 475.99 | 476.62 |

用户审阅两版TensorBoard后反馈训练趋势没有明显区别，认为本次未见明显性能
损失，并同意提交选择性编译实现。该反馈与整体配对指标改善一并记录；它不是
对所有地形、训练seed或BF16路径无损的证明。L6失败3→6/120、L9失败
9→11/120；在线末1000轮失败率4.306%→4.908%，局部指标仍有退步。
失败率差的配对episode95%区间为−1.917至+0.500个百分点，包含零。
每版只有一个训练seed，不能用episode区间证明统计非劣，也不能将改善归因于
编译本身。固定/当前WM holdout来自各自训练分布，不作为配对模型质量对照。

工程决策：提交已经完成正式验证的选择性编译边界，保留显式
`agent.bptt_execution.compile=True`启用方式；默认、BF16、critic和所有训练
超参数不变。real H32有本次5000轮证据；imagined只有前述短程回归证据。
checkpoint继续严格检查execution与compiled_regions，避免隐式切换执行路径。

正式证据位于外部`InstinctLab-local/bptt-compiled-quality-20261102/`：
`completion-report.md/json`、`actual-config-comparison.json`、
`final-training-comparison.json`及`quality-r2/comparison.json`。
运行产物不提交到Git；model5000的完整路径和SHA保存在上述报告中。

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
有界验证从同一个已训练 checkpoint 分叉，其报告不替代后续任务26350的fresh
WM/critic正式5000轮结果。单个训练seed下的配对episode区间不代表训练seed方差。
原版26085保持不可变57587f8快照；26350使用单独冻结的选择性编译快照，
未修改原版或其他活跃任务。

PyTorch2.11 参考：
[backward autocast](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/torch.compiler_backward.html)、
[nested compile region](https://docs.pytorch.org/docs/2.11/generated/torch.compiler.nested_compile_region.html)、
[混合 compiled/eager 反向](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/torch.compiler_faq.html)。
