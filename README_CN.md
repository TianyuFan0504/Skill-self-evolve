# Skill Self-Evolve：基于 verl 的 ALFWorld agent 训练

[English](README.md) · 简体中文

这是一个基于 **verl 0.9.1** 的研究分支：训练 agent 解决任务，从自己的轨迹中提炼可复用的 skill，并在后续任务中评估这些 skill。在线模型同时接受 solver loss 和延迟到达的 curate loss；独立的 curator 副本**只通过模型参数 EMA 更新**。

技术栈为 **Qwen3.5-4B + 原生 Codex CLI + Uni-Agent Responses Gateway + ALFWorld + verl Ray/FSDP2/TransferQueue**。Codex 负责执行 agent 循环，实际文本生成来自本地 Qwen 模型。

benchmark 协议、reward 公式、延迟归因、off-policy 假设和逐文件改造说明见 [AAAA_Design_Idea_CN.md](AAAA_Design_Idea_CN.md)。本分支保留上游的 [Apache-2.0 许可证](LICENSE)。底层训练基础设施仍可参考 [verl 官方文档](https://verl.readthedocs.io/)。

本中文版本对应英文文档提交 `4c2c8bb5d703f25f128617c7de64aa1d5f5ff100`；其中的实验状态按原文记录时点解释。

## 已实现的行为

- 使用原始 ALFWorld 任务，按场景划分开发集，并保留官方 seen/unseen 测试集；不额外生成 A→B 任务链。
- 持久化、带版本的 SkillStore，以及 rollout 前的宿主侧 skill 注入。
- 将通过验收的 solver batch 中每条逻辑 episode 加入 curator 审阅队列；同一 episode 的多个 token-in/token-out（TITO）片段去重。
- curator 可以提出一个 skill，也可以明确弃权（abstain）。
- 使用冻结评估器，在后续原始任务上执行“加载 skill／不加载 skill”的配对测试。
- 持久保存 curator 上下文、动作 token、掩码和原始 EMA behavior logprob，不依赖短生命周期的 TransferQueue 对象。
- 使用一个在线 optimizer，同时训练 solver GRPO 和延迟 curate replay。
- 联合保存模型、optimizer、EMA、skill 库、replay、ledger 和 dataloader checkpoint。
- 使用四个隔离的 ALFWorld 环境进程、异步持久化 I/O、动作请求标识，以及 solver／curator 的无 fork Native 启动路径。

solver **目前仍通过 `exec_command`／`write_stdin` 和 `alfworld-step` 命令行桥接与环境交互**。仅暴露结构化工具 `alfworld_step(action)` 是拟议的改进，不是这个代码版本已经实现的功能。

## 快速开始：一条命令启动 rollout 和训练

以下命令面向已配置好的 Linux 训练服务器，会执行真实的原生 agent rollout。启动器不会下载模型，也不会安装新的训练环境。

### 前置条件

| 组件 | 已测试的部署配置 |
| --- | --- |
| 代码目录 | `/workspace/verl-0.9.1` |
| 训练 Python | `/opt/conda/bin/python`，Python 3.11.13 |
| 软件栈 | Torch 2.11.0+cu130、vLLM 0.24.0、Transformers 5.10.4、Ray 2.50.1、TransferQueue 0.1.12 |
| Uni-Agent 源码 | `/root/nas_chaohuang/benchmarks/alfworld/uni-agent` |
| ALFWorld Python | `/root/nas_chaohuang/benchmarks/alfworld/env/bin/python` |
| benchmark | `/root/nas_chaohuang/benchmarks/alfworld/data/json_2.1.1` |
| 模型 | `/workspace/models/Qwen3.5-4B-nothink` |
| 原生 agent | Codex CLI 0.167.0 |
| GPU | 八张 H100 80 GB：四张 solver、两张 EMA curator、两张冻结评估器 |
| 持久化输出 | `/root/nas_chaohuang/skill_training/runs` 下的新目录 |

`Qwen3.5-4B-nothink` 是后训练版本 Qwen3.5-4B 的本地非思考模式配置，并非 Qwen3.5-4B-Base。仓库不包含模型权重。已测试配置保留 36,864 token 容量，GPU microbatch 为 1。

首次配置服务器需要准备：

1. benchmark、模型文件和 Python ABI 一致的依赖，见[环境说明](docs/skill_self_evolve/environment.md)。
2. 固定版本的 [Uni-Agent 兼容补丁](examples/skill_self_evolve/patches/UNI_AGENT_COMPATIBILITY.md)。bootstrap 会检查实际 import 来源，防止其他 editable 安装的 verl 悄悄替换 0.9.1 训练源码。
3. [私有 attempt 账号注册表](examples/skill_self_evolve/ACCOUNT_ISOLATION.md)和专用 curator 账号。启动器不会自行创建任意系统用户。
4. 两个 Linux Native 启动辅助库。在启动 worker 前编译：

```bash
cd /workspace/verl-0.9.1
cc -O2 -shared -fPIC -Wall -Wextra -Werror \
  examples/skill_self_evolve/scripts/native_posix_spawn.c \
  -o examples/skill_self_evolve/scripts/native_posix_spawn_helper.so
cc -std=c11 -O2 -shared -fPIC -Wall -Wextra -Werror \
  examples/skill_self_evolve/scripts/curator_posix_spawn.c \
  -o examples/skill_self_evolve/scripts/curator_posix_spawn_helper.so
```

当前实验服务器已经具备这两个辅助库。不要在 worker 运行期间重新编译其加载的库。具体要求见[启动机制说明](examples/skill_self_evolve/ALL_NATIVE_SPAWN.md)。本部署的公共下载和 GitHub 操作通过本机中转；训练服务器上的模型和依赖离线执行。

### 启动完整的联合 rollout 与训练

在仓库根目录执行：

```bash
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment \
  --run-dir /root/nas_chaohuang/skill_training/runs/skill_joint_001
```

请使用新的运行目录。命令会准备数据、检查已有运行环境、启动环境服务，再启动后台 supervisor，并输出运行目录及进程／状态信息。preflight 会检查 GPU 占用；不要在已被运行中实验占用的设备上再启动一份完整任务。

实际选用的入口是 `train_all_native_spawn.py`，组合了并行环境／客户端、两种 Native 启动策略、checkpoint 直接读取、被动诊断，以及固定状态后的最终评估。

### 首个 batch 的 rollout 冒烟测试

以下命令会采集原生 solver 轨迹，执行**一次真实 optimizer 更新**，保存最终 checkpoint，并跳过完整官方测试：

```bash
TOTAL_TRAINING_STEPS=1 \
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment --foreground \
  --run-dir /root/nas_chaohuang/skill_training/runs/skill_smoke_001 \
  -- skill_harness.evaluation.final=false
```

这是 **rollout 加一次训练更新**，不是无梯度推理。它采集 32 个任务组，每题四条 solver rollout。短跑退出前，后台 curator 可能尚未处理完队列；一次更新不能证明延迟 skill credit 已进入训练。完整端到端流程应使用上面的完整运行命令。

### 仅准备或检查，不启动训练

```bash
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment \
  --prepare-only \
  --run-dir /root/nas_chaohuang/skill_training/runs/skill_prepared_001
```

`--prepare-only` 写入数据／配置文件并检查选定的辅助库配置，不执行模型生成。`--preflight-only` 还会启动并检查临时环境服务及 CPU／Native 执行边界，随后停止；它不运行模型 rollout，也不执行 optimizer 更新。

联合启动器没有通用的 `--rollout-only` 参数。历史诊断脚本 `scripts/probe_native_pair.py --execute` 可以在没有 learner 的情况下执行两条冻结模型 rollout，但依赖它指定的 v4 已归档 producer／skill／后续任务输入；它不是新环境的快速启动方式，也不是任意 skill 的通用评估入口。

### 查看状态与恢复

```bash
cat /root/nas_chaohuang/skill_training/runs/skill_joint_001/status.json
tail -n 40 /root/nas_chaohuang/skill_training/runs/skill_joint_001/training.log
```

`status=running` 表示存在受监督进程，不等于一次参数更新已经完成。已确认的 optimizer 进度应查看 `controller/state.json` 和 `metrics.jsonl`。

前一进程停止、相关资源状态核对处理完成后，可以恢复：

```bash
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment \
  --run-dir /root/nas_chaohuang/skill_training/runs/skill_joint_001 --resume
```

保留原有模型、输入、运行配置及相关覆盖参数。恢复时会验证 **COMPLETE 联合 checkpoint**，而不是简单选择最新目录。不可变 tensor 文件直接读取；可写的 store／replay／ledger 状态会获得独立工作副本。详见[启动器行为](examples/skill_self_evolve/RESILIENT_LAUNCHER.md)和[恢复机制](examples/skill_self_evolve/DIRECT_CHECKPOINT_READ.md)。

## 默认实验协议

下表是公开 resilient 联合启动器的默认值，不代表所有历史实验。每轮运行初始化后的实际配置回执才是权威依据。

| 配置项 | 数值 |
| --- | --- |
| Train／dev／官方 seen／官方 unseen | 3,154／399／140／134 道任务 |
| Batch | 32 道原始任务 × 每题 4 条 solver rollout |
| Native solver 并发 | 32 个 session |
| Curator／配对评估并发 | 8／8 |
| Curator 审阅机会 | 通过验收的全部逻辑 solver episode；每条来源提出一个候选或弃权 |
| Credit 预算 | 每次更新最多安排 32 对评估；每个候选需要两个不同的后续任务 |
| 最小 credit 延迟 | 相对于持久记录的 credit window 开启 step，至少延迟两个 step |
| 总 optimizer 更新数 | 64 |
| 开发集评估 | 每八次更新，全量评估 399 道任务 |
| 官方测试 | 最后一次更新后，全量评估 140 道 seen + 134 道 unseen |
| 训练前官方／开发集评估 | 当前优先启动训练的入口中关闭 |
| Checkpoint | 每十次更新保存，并在最后一次更新后保存 |
| Episode 限制 | 50 个环境动作；声明的 Native 时间预算为 900 秒 |
| Token 容量 | 4,096 prompt 分量 + 32,768 response 分量 = 36,864 |
| 学习率 | `1e-6` |
| EMA | `decay=0.95`，每次真实 optimizer 更新后执行一次 |
| 目标函数 | `L_solver_GRPO + 0.1 L_curate + 0.001 L_solver_KL` |

trainer 按 `floor(3154 / 32) = 98` 计算每个 epoch 的 step 数。虽然允许四个 epoch，显式设置的 64 次更新上限仍会在一个名义 epoch 结束前停止训练。失败任务组的补采样可能额外消耗来源 prompt。每题生成四条 rollout，不会将 epoch 的 step 数再除以四。

支持 `TRAIN_BATCH_SIZE`、`TOTAL_TRAINING_STEPS`、`SAVE_FREQUENCY`、`MODEL_PATH` 等已记录的环境变量。调整全局 batch 时，要保持 `PPO_MINI_BATCH_SIZE` 与全局 batch 及单次更新约定一致。resilient 入口显式固定 N=4、Native 并发 32。详见[联合训练配置](examples/skill_self_evolve/config/alfworld_joint.yaml)。

## 保存的轨迹与 reward 证据

| 产物 | 内容 |
| --- | --- |
| `data/split_manifest.json` | 原始任务 ID、场景切分、数据哈希 |
| `episodes/<game>/attempt_<uuid>/` | 初始状态、全部游戏动作／观察、Native 事件、原始退出状态、结果 reward 和加载的 skill |
| `agent_logs/step_<n>/<session>/trajectory.{json,npz}` | 原始 TITO token、response mask 和 behavior logprob |
| `curate/<collection>/` | 精确的 curator 输入／输出、完成或弃权记录 |
| `controller/state.json` | 实际 optimizer 时钟、来源队列、credit window 和后台任务 |
| `skill_store/` | 候选、admission 决策和不可变快照 |
| `credit_probes/<pair>/` | 加载／不加载两侧轨迹、实际输入审计和配对事件 |
| `curate_replay/`、`credit_ledger/` | 原始 curator 动作，以及已结算／已应用的延迟标签 |
| `metrics.jsonl` | 记录的 train／dev 结果和分角色 loss |
| `checkpoints/global_step_<n>/` | 联合 checkpoint 与 COMPLETE 发布证据 |
| `driver_runs/`、`launcher_runs/` | 实际入口、组件哈希和恢复后的有效配置 |
| `evaluation/final/` | 官方测试用例、覆盖率审计及固定 skill 库的评估生命周期 |

恢复后，实际使用的 store／replay／ledger 可能位于 `restored_state/` 下。应跟随初始化后的 driver 回执定位，不能继续读取过时的原始 store。临时 TQ 对象的清理不会删除持久化的 curator replay 记录。

11 月 12 日的一份本地证据导出包含四组选定的 skill 链路和 40 条完整 episode，包括原始 GRPO 同组轨迹及配对评估。四组 utility 为 `+0.5`、`+1`、`-0.5`、`0`，分别在 optimizer update 5、6、5、5 中消费。两个正收益案例还保留了后续普通任务加载该 skill 并获得环境 reward 1 的记录。这些私有原始轨迹及模型权重没有提交到 GitHub；精确案例标识和可支持的结论见设计文档。

训练 reward 不是留出集准确率。只有未训练模型的延后基线评估与训练后最终评估使用匹配的推理条件，才可以讨论整体提升。以下命令比较的是**已经完成的两轮测试**：

```bash
/opt/conda/bin/python examples/skill_self_evolve/scripts/compare_official_runs.py \
  --base-stage /path/to/base_run/evaluation/initial \
  --trained-stage /path/to/trained_run/evaluation/final \
  --output /path/to/new_comparison.json
```

完整运行的 comparer 目前检查 base version 0 与 final version 64。它不会启动任何一侧的评估。详见[支持的比较条件](examples/skill_self_evolve/CROSS_RUN_COMPARISON.md)。

## 验证与局限

公开启动入口和评估集成的可移植 CPU 检查：

```bash
python examples/skill_self_evolve/tests/test_launch_resilient.py -q
python examples/skill_self_evolve/tests/test_parallel_environment_launcher.py -q
python examples/skill_self_evolve/tests/test_train_all_native_spawn.py -q
python examples/skill_self_evolve/tests/test_quiescent_eval.py -q
python examples/skill_self_evolve/tests/test_compare_official_runs.py -q
```

核心 loss／replay／controller 测试位于 [`tests/experimental/skill_harness`](tests/experimental/skill_harness)，部分测试需要训练 Python 环境中的依赖。CPU 检查属于工程证据，不是 benchmark 轨迹，也不能证明准确率提升。

已知局限包括：shell 工具误用与长尾 episode、后台中断任务恢复不完整、检索 query 可能仍是通用说明而非真实目标、缺少语义级 skill 去重，以及候选生成速度超过 credit 处理能力。弃权会被记录，但没有可学习的 utility 标签。带裁剪的重要性权重无法修正所有分布变化。详见[局限与后续实验](AAAA_Design_Idea_CN.md#局限与后续实验)。

目前已观察到真实 optimizer 更新、skill 生成／加载、带符号的延迟 credit，以及 replay 消费。在这份文档记录的代码与实验时点，**尚未证明**匹配条件下的完整测试集提升，也尚未完成 EMA 消融实验。
