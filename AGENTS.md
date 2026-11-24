# Agent Notes

## Project Shape

- Read `agents/CommonAgentInstructions.md` as the coding starting point.
- Read computation resources in `agents/computation/` before any verified compute or test workflow.
- For PGMT development, read `agents/pgmt/DEBUGGING_NOTES.md`; distinguish observed results from unverified explanations.
- If this is a MAC, run all tests and code on the external computation server, not on the local machine.

## External reference files

- Keep downloaded reference repositories, temporary dependency checkouts, and reference archives outside this repository.
- Use the sibling directory `<repository-name>-reference/` (currently `../InstinctLab-reference/`); do not recreate `.reference/` inside the checkout.
- Keep checkpoints, datasets, experiment reports, TensorBoard events, screenshots, and local runtime configuration out of Git. Store generated experiment files in the sibling `<repository-name>-local/` directory.

# 项目协作规范

## 项目路径与同步

| 环境 | 项目根目录 |
|---|---|
| 本地 | `/home/tars/project_2026/InstinctLab` |
| `dsw_4090_1` | `/mnt/data/mgg/project_2026/InstinctLab` |
| `dsw_a100_1` | `/mnt/data/mgg/project_2026/InstinctLab` |

**每次要训练前，必须同步到上述两台远程开发机。**

训练默认使用 RTX4090。只有用户明确指定其他 GPU 时，才切换到 A100、A800 或其他资源；RTX4090 需要排队时仍提交到 RTX4090，不自动更换 GPU 类型。

T3_27 默认使用 `$PROJECT_DIR/train_release_scripts/release_model_T3_27.sh` 从头训练。新建有变更的实验时，以该脚本为基线，按下文规范创建对应实验脚本。

## 本地回放

- 用户未指定其他播放器时，PGMT 的 `play` 默认使用 `$PROJECT_DIR/source/instinctlab/instinctlab/tasks/shadowing/pgmt/msic/play.sh` 对应的 MuJoCo 播放器。T3_27 已有本地适配时，使用对应实验目录下的 `$PROJECT_DIR/../InstinctLab-local/playback/<run>/msic-flat/play.sh`。

## 说明文档规范

- **路径表示**：除本文档外，项目说明文档中的项目根路径统一使用 `$PROJECT_DIR` 代称，不得使用绝对路径。

## 代码书写规范

- 不要编写任何代码注释。
- 所有解释只能写在说明文档中。

## 训练任务模板规范

### 启动命令

适用于所有 `train_job_template*.yaml` 和 `train_job_template*.yml`。

`command` 只能采用以下形式，使用 `bash` 执行一个 `.sh` 训练脚本：

```yaml
command: bash {sh训练脚本}
```

- **模板中禁止**：内联训练命令、`bash -c`、命令串联和额外脚本参数。
- **训练脚本负责**：工作目录切换、环境变量、训练参数及日志处理。

### 训练脚本存放位置

实验任务的 `.sh` 训练脚本按日期统一写入 `$PROJECT_DIR/tmp/train_scripts/{年月日}/`。日期目录采用 8 位 `YYYYMMDD` 格式，对应 `date +%Y%m%d`，并与脚本文件名时间戳的前 8 位一致。实验任务模板的 `command` 必须指向该日期目录中的对应脚本。

### 训练脚本命名

发版模型的训练脚本单独存放在 `$PROJECT_DIR/train_release_scripts/`，使用
`release_model_{用途}.sh` 命名，不受本节日期目录及时间戳命名规则限制。
平坦地形发版模型的训练脚本为 `release_model_flat.sh`。

实验任务的 `.sh` 训练脚本必须遵循以下格式：

```text
{年月日时分秒}_{机器人名称}_{环境数}_{集群名称}_{训练信息}_xpgmt.sh
```

| 字段 | 规则 | 示例 |
|---|---|---|
| 年月日时分秒 | 14 位时间戳 `YYYYMMDDHHmmss`，对应 `date +%Y%m%d%H%M%S` | `20261102121030` |
| 机器人名称 | 仅允许使用下节列出的名称，保持大小写和下划线 | `T3_27` |
| 环境数 | 实际并行环境数量 | `4096` |
| 集群名称 | 与模板的 `clusterName` 一致,但是使用英文 | `group1` |
| 训练信息 | 简述动作、数据集或训练配置 | `taiji_mjwarp` |
| 固定后缀 | 在 `.sh` 扩展名前必须添加 `_xpgmt` | `_xpgmt.sh` |

完整示例：

```yaml
command: bash "$PROJECT_DIR/tmp/train_scripts/20261102/20261102121030_T3_27_4096_集群1_wtf_xpgmt.sh"
```

## 机器人名称规范

训练任务和训练脚本命名中的机器人名称仅允许以下四种，**大小写和下划线必须保持一致**：

- `g1`
- `T3_27`
- `T3_29`
- `T4_27`
- `T4_29`
