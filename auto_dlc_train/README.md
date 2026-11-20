# PAI 训练任务 — YAML 快速创建

> 📌 核心工具：**create_train_job_from_yaml** —— 写一个 YAML 文件，填中文名，一键创建分布式训练任务。
> 工具自动把中文名转成数字 id，无需手动查 id。

本目录使用项目根目录的 uv 工作区、`uv.lock` 和 `.venv/`，无需单独安装依赖。

## 工作原理

工具分两层，核心逻辑完全共用，区别只在入口与鉴权：

- **TarsAI 平台 agent 工具** —— `pai_tool/` 内所有函数都用 `@tool` 注册，供平台大模型调用（agent 直接填参数，工具负责查 id、建任务）。
- **独立命令行** —— `run_example.py` 脱离平台运行，手动构造 `_ctx`（`SimpleNamespace(access_token=token)`），便于调试 / 脚本化。

```
YAML(填中文名) ──> create_train_job_from_yaml ──> list_* 查询工具 ──> POST create-job ──> job_url
                        │                        (Name → 数字 id)
                        └─ resolve_token(): TARS_TOKEN → 缓存 refresh → 账号密码登录
```

- **`pai_tool/registry.py` — `@tool` 装饰器**：纯元数据层，反射函数签名生成 Anthropic `input_schema`（`_build_schema`）。约定：参数名以 `_` 开头的（`_root` / `_ctx` / `_emit_command` / `_delegation` / `_session`）由 agent loop 注入，**不进 schema、模型看不到**；其余关键字参数才是模型可填的入参。`needs_*` 系列帮助函数判断某工具需要注入哪些参数。
- **`pai_tool/agent_tools.py` — 业务实现**：`create_train_job`（L124）POST `/api/v1/pai/train/create-job`；`list_*` 系列查询接口供 Name→id 转换；`_match_item`（L461）模糊匹配中文名；`_cloud_type`（L491）按集群区分**阿里云**（填 `specName`+`gpuCount`）与**火山引擎**（填 `instanceType`）；`_coerce_json`（L30）兼容 LLM 返回的字符串化 JSON；`list_train_job_resources`（L754）并发聚合集群/资源池/规格/项目，开场一张表。
- **`get_token.py` — 鉴权链**：`resolve_token()` 按 `TARS_TOKEN` 环境变量 → `.token_cache.json` 缓存 refresh 换新 → `credentials.yaml` 账号密码登录（密码 SHA-256），三级取 token。
- **`run_example.py` — 独立入口**：`--list-clusters` / `--list-resources` / `--from-yaml`（可配 `--dry-run` 预览）。

> PAI 接口基址 `https://open.tars-ai.com`，请求头 `Authorization: Bearer <access_token>`。

## 快速开始

三步创建训练任务：

1. 编写 YAML 文件（复制下方模板，改几个字段）
2. `--dry-run` 预览，确认参数正确
3. 正式提交，拿到 job_url 去 PAI 页面看任务

```bash
# 在项目根目录安装统一环境
cd "$PROJECT_DIR"
bash scripts/setup_env.sh
source env.sh
cd auto_dlc_train

# ① 预览：只解析 + 名字转 id + 展示将提交的参数，不真正创建
#    （token 自动获取：环境变量 TARS_TOKEN → 缓存 refresh 换新 → credentials.yaml 登录）
uv run --locked python run_example.py --from-yaml ./my_train_job.yaml --dry-run

# ② 正式提交：创建分布式训练任务，返回 job_url
uv run --locked python run_example.py --from-yaml ./my_train_job.yaml
```

> 首次运行没有 token 时，会提示三种配置方式，见下节「token 获取」。

## 训练模板选择（按训练机器）

仓库有 3 个可用资源池模板，按训练脚本的目标机器选（训练机器名称见「训练脚本书写规则」）：

| 训练机器 | 模板文件 | clusterName | specName | resourcePoolName |
|-|-|-|-|-|
| 4090 | `train_job_template_4090_group1.yaml` | 集群1 | NVIDIA-RTX4090 | RTX4090-Group1 |
| a100 | `train_job_template_a100_group2.yaml` | 集群2 | NVIDIA-A100 | A100-Group1 |
| a100 | `train_job_template_a800_group2.yaml` | 集群2 | NVIDIA-A800 | A800-Group1 |

**4090**：一律用 `train_job_template_4090_group1.yaml`。

**a100**：`train_job_template_a100_group2.yaml` 与 `train_job_template_a800_group2.yaml` 二选一：

- 用户明确指定用哪个 → 用指定的那个模板
- 用户没指定 → 选**不用排队**的那个：先 `uv run --locked python run_example.py --list-resources`，看 A100-Group1 / A800-Group1 两个资源池的 `availableCount`；`availableCount > 0`（有可用卡）即无需排队，选它
- 两个资源池都要排队 → 随便选一个模板即可（PAI 会自动排队，不算报错）

> 判断是否排队：`--list-resources` 中对应资源池 `availableCount > 0` = 无需排队；池满则提交后自动排队，不用换池。

## 训练流程（基于模板创建）

训练任务基于**按训练机器选的模板**（见上「训练模板选择」）修改创建，六步：

1. **获得 token**：`export TARS_TOKEN=<平台登录 token>`（获取方式见下「token 获取」一节）
2. **查询训练资源情况**：`uv run --locked python run_example.py --list-resources`，确认所选模板资源池的可用卡数（`availableCount`）；池满了不用管，提交后会自动排队
3. **复制模板**：`cp train_job_template_<机器>_<group>.yaml my_train_job.yaml`（如 4090 → `train_job_template_4090_group1.yaml`）
4. **改 `jobName`**：默认取所给训练脚本名（去 `.sh`）。如脚本 `train_t3_4090_2gpu_stage3_maxdelay1.sh` → `jobName: train_t3_4090_2gpu_stage3_maxdelay1`
5. **改 `command`**：按所给训练脚本改，指向远端脚本路径，模板中形如：
   ```yaml
   command: |
     export PROJECT_DIR=/mnt/data/mgg/project_2026/motion_tracking
     bash $PROJECT_DIR/tmp/train_t3_4090_2gpu_stage3_maxdelay1.sh
   ```
6. **改 `gpuCount`**：N 卡训练就把 `gpuCount` 改为 N（模板默认 1）；其余字段（clusterName / imageName / specName / resourcePoolName / dataSet 等）按需调整

提交前先 `--dry-run` 预览，确认无误再正式创建：

```bash
uv run --locked python run_example.py --from-yaml ./my_train_job.yaml --dry-run
uv run --locked python run_example.py --from-yaml ./my_train_job.yaml
```

**规则**：每次改好的训练配置（job YAML）统一保存到 `tmp/train_job_yaml/` 下归档，文件名带 jobName，如 `tmp/train_job_yaml/0812153012_train_t3_4090_2gpu_stage3_maxdelay1.yaml`，便于追溯复现。

## 代码位置

| 文件 | 说明 |
|-|-|
| `pai_tool/agent_tools.py` | 工具实现（`create_train_job_from_yaml` 第 513 行） |
| `pai_tool/registry.py` | `@tool` 装饰器（纯元数据，仅用于标注 schema，业务无关） |
| `run_example.py` | 命令行入口 |
| `get_token.py` | token 自动获取（环境变量 → 缓存 refresh 换新 → 账号密码登录） |

## token 获取（get_token.py）

工具调用靠平台登录态鉴权（`Authorization: Bearer <access_token>`）。每次运行前按下面优先级自动取 token，无需手动配置：

1. **环境变量 `TARS_TOKEN`**（最高优先）—— 浏览器 DevTools → 任一 `/api` 请求的 `Authorization: Bearer <token>`，复制到 shell：
   ```bash
   export TARS_TOKEN=<你的平台登录 token>
   ```
2. **本地缓存 refresh_token** —— 把 refresh_token 写入 `.token_cache.json`（gitignored），之后自动用它换新 access token（refresh 自动轮换，写回缓存），过期自动重取：
   ```bash
   uv run --locked python get_token.py --refresh-cache <refresh_token>
   ```
3. **账号密码登录**（终极兜底）—— 创建 `credentials.yaml`（gitignored）：
   ```yaml
   data_catalog:
     username: <你的账号>
     password: <你的密码>
   ```
   运行时代码会把密码 SHA-256 哈希后走登录接口，拿到 (access, refresh) 并把 refresh 持久化到缓存，下次优先走第 2 步。

常用辅助命令：

| 命令 | 说明 |
|-|-|
| `uv run --locked python get_token.py` | 打印当前可用的 access token（方便排查） |
| `uv run --locked python get_token.py --refresh-cache <refresh_token>` | 写入 refresh_token 到缓存 |
| `uv run --locked python get_token.py --clear-cache` | 删除 token 缓存文件 |

> 以上三种都不可用时，程序会报错并提示配置方式。token 失效时接口返回 `401`。

## 命令行入口（run_example.py）

| 参数 | 说明 |
|-|-|
| `--list-clusters` | 列出可用集群 |
| `--list-resources` | 聚合查询建训练 job 所需核心选项（集群/资源池/机器规格/项目），开场一张表 |
| `--from-yaml PATH` | 按 YAML 文件创建分布式训练任务 |
| `--dry-run` | 与 `--from-yaml` 搭配：仅解析+转 id+预览，不真正创建（默认正式创建） |

## YAML 模板（直接复制使用）

> 仓库自带 3 个真实可用模板（`train_job_template_4090_group1.yaml` / `train_job_template_a100_group2.yaml` / `train_job_template_a800_group2.yaml`，HEFT 训练任务样例），按训练机器选择（见「训练模板选择」），可直接 `--dry-run` 试跑。

### 阿里云集群（集群 1/2）

> 阿里云用 `specName` + `gpuCount`；**不要填** `instanceType`

```yaml
# 必填
jobName: my-train-job-20250806          # 任务名称（唯一）
clusterName: 集群1                        # 中文名，工具自动转 clusterId
priority: 9                              # 优先级
podCount: 1                              # 节点数
command: |                               # 启动命令
  conda activate your_env && cd /workspace && bash train.sh
imageName: james-gao-shanghai-dsw:v1    # 镜像中文名，工具自动转 imageId

# 阿里云机器规格（必填）
specName: NVIDIA-RTX4090                 # GPU 类型中文名 → specId
gpuCount: 1                              # 单节点卡数
cpuCount: 16                             # 单节点 CPU 数（选填）
errorMonitor: true                       # 自动容错（选填）

# 选填
projectName: AWE                         # 项目中文名 → projectId
resourcePoolName: RTX4090-Group1         # 资源池中文名 → resourcePoolId
maxRunTime: 120                          # 最长运行小时
```

### 火山引擎集群（集群 3/4）

> 火山引擎用 `instanceType`；**不要填** `specName` / `gpuCount`

```yaml
# 必填
jobName: my-train-job-20250806
clusterName: 集群3                        # 火山引擎集群中文名
priority: 9
podCount: 1
command: |
  conda activate your_env && cd /workspace && bash train.sh
imageName: james-gao-shanghai-dsw:v1

# 火山引擎机器规格（必填）
instanceType: ml.xni3c.22xlarge          # 实例规格 id（直接填 id，非中文名）
enableTensorboard: false                 # 选填

# 选填
projectName: AWE
resourcePoolName: A800-Group1
maxRunTime: 120
```

### 带数据集和存储挂载的完整模板

```yaml
jobName: my-full-train-job
clusterName: 集群1
priority: 9
podCount: 1
command: |
  conda activate wlcb && cd /workspace && bash train.sh
imageName: james-gao-shanghai-dsw:v1

# 阿里云规格
specName: NVIDIA-RTX4090
gpuCount: 1

projectName: AWE
resourcePoolName: RTX4090-Group1

# 数据集挂载（选填，可多条）
dataSet:
  - datasetName: VLTA-CPFS-taplab        # 数据集中文名 → datasetLocalId
    mountPath: /mnt/data/                # 挂载路径
    readOnly: true
  - datasetName: another-dataset
    mountPath: /mnt/data2/
    version: v2                          # 选填，数据集版本

# 存储挂载（选填，可多条）
storageMountList:
  - storageName: tars-datasets-wlcb      # 存储中文名 → fileStorageId
    fileSystemPath: /                    # 文件系统路径
    mountPath: /mnt/oss_wlcb/            # 挂载路径
    readOnly: true
```

## 核心原理：Name 后缀字段自动转 id

YAML 中带 **Name** 后缀的字段填中文名，工具自动调查询接口转成数字 id：

| YAML 字段（填名字） | 转换后的字段（数字 id） | 查询接口 |
|-|-|-|
| `clusterName: 集群1` | `clusterId: 1` | `list_clusters` |
| `imageName: xxx:v1` | `imageId: "xxx"` | `list_images` |
| `specName: NVIDIA-RTX4090` | `specId: 123` | `list_gpu_types` |
| `projectName: AWE` | `projectId: 5` | `list_projects` |
| `resourcePoolName: Group1` | `resourcePoolId: 10` | `list_resource_pools` |
| `dataSet[].datasetName: VLTA` | `datasetLocalId: 8` | `list_datasets` |
| `storageMountList[].storageName` | `fileStorageId: 3` | `list_storages` |

## 完整 Tool 清单

### create_train_job_from_yaml（⭐ 推荐）

| 项目 | 内容 |
|-|-|
| 代码位置 | `pai_tool/agent_tools.py` 第 513 行 |
| 注册名 | `create_train_job_from_yaml` |
| 参数 | `yaml_path`（YAML 文件路径）, `dry_run`（true=预览，false=正式创建） |
| 功能 | 读 YAML → Name 字段转 id → 调 create_train_job 创建任务 |

### create_train_job（直接模式）

| 项目 | 内容 |
|-|-|
| 代码位置 | `pai_tool/agent_tools.py` 第 124 行 |
| 注册名 | `create_train_job` |
| 参数 | `clusterId, jobName, specId, imageId, podCount, priority, command` 等 |
| 功能 | 直接调 PAI 接口 POST `/api/v1/pai/train/create-job` |

### 查询类 Tool（按使用顺序）

| Tool 名 | 行号 | 用途 |
|-|-|-|
| `list_train_job_resources` | L754 | 聚合查询：集群 + 资源池 + 规格 + 项目，一次拉全 |
| `list_clusters` | L235 | 集群列表 `[{id, name, company}]` |
| `list_images` | L343 | 镜像列表（clusterId, imageName 可选） |
| `list_gpu_types` | L260 | 阿里云 GPU 类型（clusterId） |
| `list_instance_specs` | L290 | 火山引擎计算规格（resourceId, kind） |
| `list_resource_pools` | L437 | 资源池 + GPU 可用数（clusterId 可选） |
| `list_projects` | L315 | 研发项目列表 |
| `list_datasets` | L374 | 数据集列表（clusterId, datasetName 可选） |
| `list_storages` | L407 | 文件存储列表（clusterId, resourceId 可选） |

## 使用示例

```bash
# 1. 先看有哪些资源可选（集群/资源池/规格/项目一览）
uv run --locked python run_example.py --list-resources

# 2. 生成 YAML 并 dry run（展示将要提交的参数，不真正创建）
uv run --locked python run_example.py --from-yaml ./my_train_job.yaml --dry-run

# 3. 确认无误后正式提交，返回 job_url 和 jobName
uv run --locked python run_example.py --from-yaml ./my_train_job.yaml
```

Python 直调示例（与命令行等价）：

```python
import asyncio
from types import SimpleNamespace
from pai_tool.agent_tools import create_train_job_from_yaml

async def main():
    ctx = SimpleNamespace(access_token="<你的 token>")
    # 预览
    print(await create_train_job_from_yaml(yaml_path="./my_train_job.yaml", dry_run=True, _ctx=ctx))
    # 正式创建
    print(await create_train_job_from_yaml(yaml_path="./my_train_job.yaml", dry_run=False, _ctx=ctx))

asyncio.run(main())
```

## 注意

- **RTX4090-Group1 满了会自动排队**：PAI 平台在资源不足时把任务挂起排队，不算报错，不用换池；正常 `--dry-run` 预览后提交即可，有空卡了自动跑起来。
- **YAML 中填中文名**，工具自动查 id；建议始终先 `--dry-run` 预览，确认后再正式创建。
- **火山引擎集群填 `instanceType`**，不填 `specName` / `gpuCount`。
- 每次调用都需要有效 token；获取方式见「token 获取」一节，token 失效时接口返回 `401`。
- 所有工具返回单行 JSON：`{"status": "ok|error", ...}`。
