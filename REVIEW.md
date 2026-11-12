# SuperMimic heightscan 基线与审核入口

代码分支：`xie/supermimic-review`。基线标签：`pgmt-supermimic-baseline-20261112`。
后续变更用 `git diff pgmt-supermimic-baseline-20261112..HEAD` 对照；本轮重整可比较该标签与父提交。

## 本轮结果

以桌面 SuperMimic 的运行时源码为基础，保留其 `runners/`、`envs/`、`learners/`、
representation、replay、SimContext 和 logger 分工。PGMT 子类与原版文件放在同一层，
使用 `pgmt_` 前缀。只有一个 `learners/`，没有另建一套框架。

此前过渡实现的 `Collector`、`Components/setup`、`interfaces`、`core/` 和旧 SM runners
已从这个审核树删除；一阶段 PPO、地形 PPO、heightscan、MuJoCo play 保留。
正式训练仍使用各自冻结的源码，不受此工作树修改影响。

## 建议审核顺序

| 顺序 | 文件 | 核对内容 |
|---|---|---|
| 1 | [结构与调用链](SM_ARCHITECTURE.md) | 责任边界、实际继承、必要的语义适配 |
| 2 | [复制来源和改动说明](source/instinctlab/instinctlab/tasks/shadowing/pgmt/supermimic/README.md) | 原文件、改动钩子、PGMT 子类的对应关系 |
| 3 | [SM runner](source/instinctlab/instinctlab/tasks/shadowing/pgmt/supermimic/runners/supermimic_runner.py) → [PGMT runner](source/instinctlab/instinctlab/tasks/shadowing/pgmt/supermimic/runners/pgmt_runner.py) | 共用真实采集和训练循环，PGMT 只实现任务钩子 |
| 4 | [WorldModelEnv](source/instinctlab/instinctlab/tasks/shadowing/pgmt/supermimic/envs/world_model_env.py) → [PGMT 环境](source/instinctlab/instinctlab/tasks/shadowing/pgmt/supermimic/envs/pgmt_world_model_env.py) | `reset/step(actions)`、状态/参考/地形更新和奖励；不持有 actor/critic |
| 5 | [策略 learner](source/instinctlab/instinctlab/tasks/shadowing/pgmt/supermimic/learners/pgmt_policy_learner.py)；[WM learner](source/instinctlab/instinctlab/tasks/shadowing/pgmt/supermimic/learners/pgmt_wm_learner.py) | 想象、目标、BCE、终止 mask、更新和图捕获 |
| 6 | [配置](source/instinctlab/instinctlab/tasks/shadowing/pgmt/config/g1/pgmt_supermimic_cfg.py)；[类型定义](source/instinctlab/instinctlab/tasks/shadowing/pgmt/supermimic/config.py) | CNN187 AllCombined 默认参数、实验轴和具名常量 |
| 7 | [验证方法](SM_VALIDATION.md) | 组件、Newton 配对、满尺寸短测、checkpoint 和日志检查 |

## 这次没有改变的训练内容

默认保留 CNN187 heightscan AllCombined：35 维速度增量与解析位姿积分、Bryson loss scale、
Muon＋Adam 分组、选择性编译、WM/actor/critic 三段 CUDA Graph、actor EMA、独立 imagined
critic、base19＋contact4＋analytic2，以及原地形课程、数据与奖励统计口径。

187 点扫描为 17×11、10 cm 间距、前移 40 cm；基座正下方高度独立查询。
默认 H32、4096 环境、actor batch4096、WM H8/batch2048/10 次更新、collect2.5、seed42。
环境、奖励、算法、执行和网络分别配置；不加入四块 TF 选点诊断或深度图后端。

## 审核范围和验证边界

复制来源是桌面目录的**本地工作树**，包含其未提交修改，并非只由 Git commit 标识的干净源码。
原版基类增加了必要的工厂和生命周期钩子，runner 循环也做了提取；这不是零修改的上游复制。
精确来源和基类改动见 SuperMimic README / NOTICE。

本轮验收面向 IsaacLab 3 / Newton 的 PGMT heightscan 路径。
复制保留的原版 mjlab CLI、PPO runner 和原版任务配置未作为本项目入口验收。
本版作为后续开发与实验基线；PRO 5000 正式训练使用独立源码快照。
验收数据、性能报告和训练记录存放在仓库外的 `InstinctLab-local/`，不随代码提交。
