# SuperMimic heightscan 验证方法

## 环境与产物

在已配置的 IsaacLab 3 / Newton Python 3.12 环境中，从仓库根目录运行。
`PYTHONPATH` 应包含当前工作树的 `source/instinctlab`，避免导入其他冻结快照。
格式检查也应使用 Python 3.12，以支持复制源码的泛型和 type-alias 语法。

所有短测、报告、events 和缓存写入仓库外的
`<项目产物根>/validation/pgmt/<时间戳>_<用途>/`。
正式训练不复用短测目录，也不将验证 events 加入正式 TensorBoard 总览。

本文只记录可复用的检查方法；具体实验结果、耗时、GPU、源码快照和命令记录保存在
仓库外 `InstinctLab-local/validation/` 或 `InstinctLab-local/runtime/`。

## 组件与生命周期

```bash
python -m unittest discover -s source/instinctlab/test/pgmt -p 'test_supermimic*.py' -v
python scripts/pgmt/test_initial_time_out.py
```

覆盖 replay 环绕、终止 transition、reset 历史、representation 数值/梯度、
soft termination returns、episode 日志、Muon 图内学习率、实际继承模板，以及 WXYZ/XYZW 边界。

历史模型数值对照需要 `PGMT_LEGACY_SOURCE` 显式指向仓库外的旧源码根目录；未设置时
对应 oracle 测试会跳过。当前训练后端不依赖这个历史目录。

## Newton 同输入配对

入口：`source/instinctlab/test/pgmt/validate_supermimic.py`。
运行前设置 `PGMT_LEGACY_SOURCE` 和绝对路径 `PGMT_VALIDATION_DIR`，并将
`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`、`WARP_CACHE_PATH` 指向同一验证目录内的缓存子目录。

```bash
python -u source/instinctlab/test/pgmt/validate_supermimic.py \
  --task Instinct-PGMT-SuperMimic-G1-Newton-v0 --num_envs 64 --seed 42 --viz none \
  --run_name SuperMimicParity --logroot "$PGMT_VALIDATION_DIR" --relative-tolerance 0.002 \
  agent.supermimic.dynamics.hidden_width=128 \
  agent.supermimic.policy.batch_size=32 \
  agent.supermimic.world_model.batch_size=32 \
  agent.supermimic.world_model.num_batches=2 \
  agent.supermimic.world_model.validation_batch=16 \
  agent.supermimic.collection.replay_steps=128
```

使用同参数、同采样和同噪声比较 WM loss/梯度、actor loss/梯度、features、targets、survival、
valid mask 和 reward means；contact scale 覆盖 0、0.5、1，检查 actor 更新时 WM 参数冻结。

同时验证旧 play 字段动作导出、显式旧权重导入、新 checkpoint 重载，及重载后清空 replay、
重置 CUDA Graph 状态并重新捕获。检查报告写入给定的验证目录。

## 满尺寸执行检查

使用正式训练入口和完整默认配置，将输出明确指定到独立验证根目录：

```bash
python -u scripts/instinct_rl/train.py \
  --task Instinct-PGMT-SuperMimic-G1-Newton-v0 --num_envs 4096 --max_iterations 70 \
  --seed 42 --viz none --run_name SuperMimicFullSize --log_dir_name SuperMimicFullSize \
  --logroot "$PGMT_VALIDATION_DIR"
```

核对以下实际状态：

- 32 轮 WM warmup 后 actor 更新持续推进；collect2.5 逐轮交替采集 2/3 步。
- loss、梯度和日志标量有限，WM/actor/critic 三段图捕获成功。
- 学习率调度、EMA、课程和 checkpoint 计数符合有效配置。
- 同 seed 的重整前后 actor、WM、critic、EMA 和 normalizer 初值一致。
- `<run>/tensorboard/` 为唯一 events 目录，实际 TensorBoard run 列表无重复；诊断只在配置轮次记录。

性能报告将冷编译、图捕获、普通轮、周期验证和保存开销分开。
`Performance/iteration_seconds` 截止 logger 写文件之前；另用相邻日志时间戳核对整轮墙钟间隔。
后 50 轮均值用于稳态摘要，并注明其是否覆盖周期验证或 checkpoint 保存。

## 验收边界

以上检查面向 Newton 的 PGMT heightscan 路径，不能替代独立的策略质量、收敛或迁移评估。
复制保留的原版 mjlab CLI、PPO runner 和原版任务需使用对应环境单独验收。
