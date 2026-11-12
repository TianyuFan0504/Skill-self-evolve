# Skill Self-Evolve: ALFWorld agent training on verl

A research fork of **verl 0.9.1** that trains an agent to solve tasks, extract
reusable skills from its trajectories, and evaluate those skills on later tasks.
The online model receives both the solver loss and a delayed curate loss. A
separate curator replica receives **parameter EMA only**.

The stack is **Qwen3.5-4B + native Codex CLI + Uni-Agent Responses Gateway +
ALFWorld + verl Ray/FSDP2/TransferQueue**. Codex executes the agent loop;
generation comes from the local Qwen model.

Read [AAAA_Design_Idea.md](AAAA_Design_Idea.md) for the benchmark protocol, reward
equations, delayed attribution, off-policy assumptions, and file-level diff map.
This fork retains the upstream [Apache-2.0 license](LICENSE).
[Upstream verl documentation](https://verl.readthedocs.io/) remains applicable
to the underlying training infrastructure.

## Implemented behavior

- Original ALFWorld tasks, a scene-disjoint development holdout, and reserved
  official seen/unseen test sets. No generated A-to-B task chains.
- A persistent, versioned SkillStore and host-side skill injection before rollout.
- Every logical episode in an accepted solver batch queued for curator review,
  deduplicated across token-in/token-out (TITO) segments.
- A curator decision to propose one skill or explicitly abstain.
- Frozen-evaluator, with-skill/without-skill paired tests on later original tasks.
- Durable curator context, action tokens, masks and original EMA behavior
  logprobs, independent of short-lived TransferQueue objects.
- One online optimizer for solver GRPO and delayed curate replay.
- Joint model/optimizer/EMA/library/replay/ledger/dataloader checkpoints.
- Four isolated ALFWorld engine processes, asynchronous durable I/O, action
  request identities, and fork-free Native launches for solver and curator.

The solver **still uses `exec_command` / `write_stdin` and the `alfworld-step`
command-line bridge**. A sole structured `alfworld_step(action)` tool is a
proposed improvement, not an implemented feature of this source cut.

## Quick start: one command to collect rollouts and train

These commands target the provisioned Linux training host and execute real
native-agent rollouts. The launcher does not download models or install a new
training environment.

### Prerequisites

| Component | Tested deployment |
| --- | --- |
| Checkout | `/workspace/verl-0.9.1` |
| Training Python | `/opt/conda/bin/python`, Python 3.11.13 |
| Stack | Torch 2.11.0+cu130, vLLM 0.24.0, Transformers 5.10.4, Ray 2.50.1, TransferQueue 0.1.12 |
| Uni-Agent source | `/root/nas_chaohuang/benchmarks/alfworld/uni-agent` |
| ALFWorld Python | `/root/nas_chaohuang/benchmarks/alfworld/env/bin/python` |
| Benchmark | `/root/nas_chaohuang/benchmarks/alfworld/data/json_2.1.1` |
| Model | `/workspace/models/Qwen3.5-4B-nothink` |
| Native agent | Codex CLI 0.167.0 |
| GPUs | Eight H100 80 GB: four solver, two EMA curator, two frozen evaluator |
| Durable output | A fresh directory under `/root/nas_chaohuang/skill_training/runs` |

`Qwen3.5-4B-nothink` is a local non-thinking configuration of the post-trained
Qwen3.5-4B checkpoint, not Qwen3.5-4B-Base. Weights are not included in this repo.
The tested profile retains 36,864-token capacity and GPU microbatch 1.

First-time host setup must provide the following:

1. Benchmark/model files and same-ABI dependencies; see
   [environment details](docs/skill_self_evolve/environment.md).
2. The pinned [Uni-Agent compatibility patch](examples/skill_self_evolve/patches/UNI_AGENT_COMPATIBILITY.md).
   Bootstrap checks actual imports so another editable verl installation cannot
   silently replace the 0.9.1 training source.
3. The [private attempt-account registry](examples/skill_self_evolve/ACCOUNT_ISOLATION.md)
   and dedicated curator account. The launcher does not provision arbitrary users.
4. Both Linux Native spawn helpers. Build these before starting workers:

```bash
cd /workspace/verl-0.9.1
cc -O2 -shared -fPIC -Wall -Wextra -Werror \
  examples/skill_self_evolve/scripts/native_posix_spawn.c \
  -o examples/skill_self_evolve/scripts/native_posix_spawn_helper.so
cc -std=c11 -O2 -shared -fPIC -Wall -Wextra -Werror \
  examples/skill_self_evolve/scripts/curator_posix_spawn.c \
  -o examples/skill_self_evolve/scripts/curator_posix_spawn_helper.so
```

The configured host already has these helpers. Do not rebuild a helper under
live workers. See [spawn requirements](examples/skill_self_evolve/ALL_NATIVE_SPAWN.md).
Public downloads and GitHub operations use the local relay computer in this
deployment; model/dependency execution on the training host is offline.

### Full joint rollout and training

From the repository root:

```bash
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment \
  --run-dir /root/nas_chaohuang/skill_training/runs/skill_joint_001
```

Use a new run directory. This prepares data, checks the installed runtime,
starts the environment service, and launches a detached supervisor. It prints
the run directory and process/status information. Preflight checks GPU ownership;
do not start a second full job on devices belonging to a live experiment.

The selected entry is `train_all_native_spawn.py`, composing the parallel
environment/client, both Native spawn policies, direct checkpoint reads,
passive diagnostics and sealed final evaluation.

### First-batch rollout smoke test

To collect native solver trajectories, apply **one actual optimizer update**,
save its final checkpoint, and skip the full official test:

```bash
TOTAL_TRAINING_STEPS=1 \
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment --foreground \
  --run-dir /root/nas_chaohuang/skill_training/runs/skill_smoke_001 \
  -- skill_harness.evaluation.final=false
```

This is **rollout plus one training update**, not no-gradient inference. It
collects 32 task groups with four solver rollouts each. A background curator may
not finish before this short run exits; one update cannot establish delayed
skill-credit learning. The full run above is the supported end-to-end recipe.

### Prepare or check without training

```bash
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment \
  --prepare-only \
  --run-dir /root/nas_chaohuang/skill_training/runs/skill_prepared_001
```

`--prepare-only` writes data/configuration assets and checks the selected helper
profile without model generation. `--preflight-only` additionally starts and
checks a temporary environment service and the CPU/Native boundary, then stops;
it does not run a model rollout or an optimizer update.

There is no generic `--rollout-only` flag on the joint launcher. The historical
`scripts/probe_native_pair.py --execute` diagnostic runs two frozen-model
rollouts without a learner, but requires its specifically archived v4
producer/skill/future-task fixture. It is not a fresh-install quick start or an
arbitrary skill evaluator.

### Observe and resume

```bash
cat /root/nas_chaohuang/skill_training/runs/skill_joint_001/status.json
tail -n 40 /root/nas_chaohuang/skill_training/runs/skill_joint_001/training.log
```

`status=running` identifies a supervised process, not a completed update. Use
`controller/state.json` and `metrics.jsonl` for confirmed optimizer progress.

After the previous process has stopped and its resources have been reconciled:

```bash
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment \
  --run-dir /root/nas_chaohuang/skill_training/runs/skill_joint_001 --resume
```

Keep the original model, inputs, profile and relevant overrides. Resume validates
a **COMPLETE joint checkpoint**, not merely the newest directory. Immutable
tensor files are read directly; writable store/replay/ledger state gets an
independent working copy. See [launcher behavior](examples/skill_self_evolve/RESILIENT_LAUNCHER.md)
and [restore details](examples/skill_self_evolve/DIRECT_CHECKPOINT_READ.md).

## Default experiment protocol

These are the public resilient joint launcher's defaults, not every historical
run's settings. The effective post-initialization receipt is authoritative.

| Setting | Value |
| --- | --- |
| Train / dev / official seen / official unseen | 3,154 / 399 / 140 / 134 tasks |
| Batch | 32 original tasks × 4 solver rollouts |
| Native solver concurrency | 32 sessions |
| Curator / paired-evaluation concurrency | 8 / 8 |
| Curator opportunities | All accepted logical solver episodes; one candidate or abstention per source |
| Credit budget | At most 32 pairs scheduled per update; two distinct future tasks per candidate |
| Minimum credit delay | Two steps relative to the persisted credit-window opening step |
| Total optimizer updates | 64 |
| Development evaluation | All 399 tasks every eight updates |
| Official tests | All 140 seen + 134 unseen tasks after the final update |
| Initial official/dev evaluation | Disabled in this training-first entry |
| Checkpoints | Every ten updates and at the final update |
| Episode limits | 50 environment actions; declared 900-second Native budget |
| Token capacity | 4,096 prompt component + 32,768 response component = 36,864 |
| Learning rate | `1e-6` |
| EMA | `decay=0.95`, once per actual optimizer update |
| Objective | `L_solver_GRPO + 0.1 L_curate + 0.001 L_solver_KL` |

The trainer calculates `floor(3154 / 32) = 98` steps per epoch. An explicit
64-update limit stops before one nominal epoch, even though four epochs are
allowed. Failed-group refill can consume additional source prompts. Four
rollouts per prompt do not divide the epoch length by four.

`TRAIN_BATCH_SIZE`, `TOTAL_TRAINING_STEPS`, `SAVE_FREQUENCY`, `MODEL_PATH` and
other documented environment controls are supported. Keep `PPO_MINI_BATCH_SIZE`
consistent with global batch size and the one-update contract. The resilient
entry explicitly fixes N=4 and Native concurrency 32. See
[joint configuration](examples/skill_self_evolve/config/alfworld_joint.yaml).

## Saved trajectories and reward evidence

| Artifact | Contents |
| --- | --- |
| `data/split_manifest.json` | Original task IDs, scene split, dataset hashes |
| `episodes/<game>/attempt_<uuid>/` | Initial state, all game actions/observations, Native events, raw exits, result reward and loaded skills |
| `agent_logs/step_<n>/<session>/trajectory.{json,npz}` | Original TITO tokens, response masks and behavior logprobs |
| `curate/<collection>/` | Exact curator input/output, completion or abstention record |
| `controller/state.json` | Actual optimizer clock, source queue, credit windows and background jobs |
| `skill_store/` | Candidates, admissions and immutable snapshots |
| `credit_probes/<pair>/` | With/without trajectories, actual-input audit and paired event |
| `curate_replay/`, `credit_ledger/` | Original curator action and settled/applied delayed labels |
| `metrics.jsonl` | Recorded train/dev results and per-role losses |
| `checkpoints/global_step_<n>/` | Joint checkpoint and COMPLETE publication evidence |
| `driver_runs/`, `launcher_runs/` | Actual entry, component hashes and effective restored configuration |
| `evaluation/final/` | Official cases, coverage audit and fixed-library lifecycle |

After resume, active store/replay/ledger paths may be under `restored_state/`.
Follow the initialized driver receipt instead of reading an obsolete original
store. Temporary TQ cleanup does not delete durable curator replay records.

A local November 12 evidence export contains four selected skill chains and
40 complete episodes, including original GRPO siblings and paired evaluations.
Utilities are `+0.5`, `+1`, `-0.5` and `0`, consumed at optimizer updates 5, 6,
5 and 5. The two positive cases also have ordinary later skill loading with
environment reward 1. These private raw trajectories and model weights are not
committed to GitHub. Exact identities and claim limits are in the design document.

Training reward is not held-out accuracy. A deferred untouched-model evaluation
and a trained-final evaluation must use matched inference conditions before
claiming overall improvement. This command compares **already completed** tests:

```bash
/opt/conda/bin/python examples/skill_self_evolve/scripts/compare_official_runs.py \
  --base-stage /path/to/base_run/evaluation/initial \
  --trained-stage /path/to/trained_run/evaluation/final \
  --output /path/to/new_comparison.json
```

The full-run comparer currently checks base version 0 against final version 64.
It does not launch either evaluation. See
[supported comparison conditions](examples/skill_self_evolve/CROSS_RUN_COMPARISON.md).

## Validation and limitations

Portable CPU checks for public launch and evaluation integration:

```bash
python examples/skill_self_evolve/tests/test_launch_resilient.py -q
python examples/skill_self_evolve/tests/test_parallel_environment_launcher.py -q
python examples/skill_self_evolve/tests/test_train_all_native_spawn.py -q
python examples/skill_self_evolve/tests/test_quiescent_eval.py -q
python examples/skill_self_evolve/tests/test_compare_official_runs.py -q
```

Core loss/replay/controller tests are in
[`tests/experimental/skill_harness`](tests/experimental/skill_harness); some
require the training Python dependencies. CPU checks are engineering evidence,
not benchmark trajectories or proof of improved accuracy.

Known limitations include shell-tool misuse and long tails, incomplete recovery
of interrupted background jobs, generic rather than goal-specific retrieval
queries, no semantic skill deduplication, and candidate production exceeding
credit throughput. Abstention is recorded but has no learned utility label.
Clipped importance weighting does not correct every distribution shift.
See [limitations and next experiments](AAAA_Design_Idea.md#limitations-and-next-experiments).

Real optimizer updates, skill generation/loading, signed delayed credit and replay
consumption have been observed. A matched full-test improvement or an EMA
ablation has **not** been established at this documentation cut.
