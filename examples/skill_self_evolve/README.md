# ALFWorld Skill Self-Evolution Training

This recipe connects the **native Codex CLI** to Qwen3.5-4B through the official
Uni-Agent Responses Gateway and trains the resulting token trajectories with
the pinned verl 0.9.1 checkout. Codex remains the task-solving harness; it is not
replaced by a separately implemented chat agent.

The initial `solver_baseline` mode trains the solver. The custom entry point
must reject unimplemented modes instead of silently omitting skill, curator or
EMA behavior. A passing baseline is not evidence that the full skill-evolution
system has already been implemented.

## Runtime boundaries

| Component | Required location or role |
|---|---|
| Training source | `/workspace/verl-0.9.1` |
| Training Python | `/opt/conda/bin/python` |
| Uni-Agent source | `/root/nas_chaohuang/benchmarks/alfworld/uni-agent` |
| Additional Python dependencies | Same-ABI `uni-env` site-packages, appended without executing `.pth` files |
| ALFWorld engine Python | `/root/nas_chaohuang/benchmarks/alfworld/env/bin/python` |
| Original benchmark | `/root/nas_chaohuang/benchmarks/alfworld/data/json_2.1.1` |
| Model | `/workspace/models/Qwen3.5-4B-nothink` |
| Artifacts | A new private directory under the benchmark NAS root |

The inference environment contains a second editable **verl 0.10 development
checkout**. The bootstrap fails closed unless the actual training import is
the requested source and version 0.9.1. It appends dependency packages only
after checking their Python minor ABI; it never calls `site.addsitedir`, executes
the inference environment's editable `.pth` files, or installs packages.

All Git operations, downloads and dependency acquisition must use the local
relay computer. On the server, this recipe sets Hugging Face, datasets, uv,
pip and tracking to offline operation. Local Ray, Gateway and ALFWorld HTTP
traffic remains necessary for the experiment.

## Data and evaluation protocol

`prepare_data.py` indexes original solvable games and creates:

- `train.parquet`: original official training games only.
- `dev.parquet`: a deterministic holdout of whole scenes from official train.
- `test_seen.parquet` and `test_unseen.parquet`: unchanged official evaluation
  tasks, reserved for explicit final evaluation.
- `catalog.json` and `split_manifest.json`: stable game identities, content
  hashes, scene split decisions and dataset hashes.

No synthetic task or expert plan is placed in a model prompt. The training
environment service permits only the official `train` split, including its
train-derived dev holdout. Scheduled validation uses dev, never the official
test files. Final test evaluation must start a separately configured evaluation
service and must not feed its failed trajectories back into training.

## Controlled native Codex instructions

The tracked instructions are:

- `prompts/solver_system.md`: native Codex model instructions.
- `prompts/solver_task.txt`: the user task instruction passed to each episode.

The launcher installs a per-run public instruction file and executable wrapper
under `/opt/alfworld-tools/skill_self_evolve/`. Uni-Agent still invokes its
upstream `CodexAgent`. The wrapper adds the native
`-c model_instructions_file=...` override before executing the unmodified Codex
binary. A per-run model catalog declares the same model name, context capacity
and parallel-tool capability. Per-run paths prevent an instruction edit in a
new run from changing an active experiment.

Codex runs under the `alfworld_eval` account and interacts using only
`alfworld-step 'EXACT ACTION'`. Private server capabilities, game files, expert
plans and other episode artifacts are not supplied to that account. The Task
creates a unique attempt directory for every rollout and epoch.

## Start a run

From the server checkout:

```bash
cd /workspace/verl-0.9.1
TOTAL_TRAINING_STEPS=64 \
TRAIN_BATCH_SIZE=16 ROLLOUT_N=4 ROLLOUT_CONCURRENCY=64 \
bash examples/skill_self_evolve/run.sh
```

By default the launcher prepares the original data, starts a private local
ALFWorld service, runs CPU import/configuration checks, and starts a detached
supervisor. It prints the run directory, supervisor PID and status path. The
training subprocess is exactly:

```bash
/opt/conda/bin/python -m verl.experimental.skill_harness.train \
  --config-path=/workspace/verl-0.9.1/examples/skill_self_evolve/config \
  --config-name=alfworld_train
```

Use `--foreground` to wait in the current terminal. Use `--prepare-only` to write
the split/configuration artifacts without starting the service or any model.
Use `--preflight-only` for CPU import/configuration and local service checks;
this starts and then stops a new environment service but never starts training.

Explicit run paths and Hydra overrides are supported:

```bash
bash examples/skill_self_evolve/run.sh \
  --run-dir /root/nas_chaohuang/benchmarks/alfworld/training_runs/experiment_001 \
  -- trainer.total_training_steps=16 trainer.test_freq=4
```

Resume only after confirming the previous training process is no longer alive:

```bash
bash examples/skill_self_evolve/run.sh \
  --run-dir /root/nas_chaohuang/benchmarks/alfworld/training_runs/experiment_001 \
  --resume
```

The launcher checks a recorded training PID against its command and run identity
before submitting a replacement. A transient observation failure is not evidence
that the previous job stopped.

## Important configuration controls

| Setting | Default | Meaning |
|---|---:|---|
| `TRAINING_GPUS` | All visible GPUs | FSDP2 trainer/rollout GPUs on one node |
| `TRAIN_BATCH_SIZE` | 16 | Distinct task prompts per update |
| `ROLLOUT_N` | 4 | Stochastic responses per prompt; must be at least 2 |
| `ROLLOUT_CONCURRENCY` | 64 | Maximum in-flight native Codex sessions |
| `GATEWAY_COUNT` | 8 | Gateway actor count |
| `LEARNING_RATE` | `1e-6` | Online optimizer learning rate |
| `MAX_PROMPT_LENGTH` | 4096 | Initial prompt budget |
| `MAX_RESPONSE_LENGTH` | 32768 | Entire trajectory response capacity, not one API response |
| `GPU_MEMORY_UTILIZATION` | 0.55 | Rollout engine memory fraction |
| `DEV_MAX_SAMPLES` | 64 | Maximum train-derived dev games per scheduled evaluation |
| `DEV_EVAL_FREQUENCY` | 4 | Dev evaluation interval |
| `SAVE_FREQUENCY` | 8 | Full model/optimizer/extra checkpoint interval |
| `MAX_CHECKPOINTS` | 2 | Model checkpoint retention; trajectory/metric files are retained |

`MAX_MODEL_LENGTH` must equal prompt plus response capacity in this recipe.
The launcher keeps the native Codex catalog declaration consistent with that
capacity. Keep `TRAIN_BATCH_SIZE` and `PPO_MINI_BATCH_SIZE` equal when using a
custom trainer that assumes one optimizer update per role RPC. Changes to GPU
count, concurrency or token limits must be recorded in a new run manifest.

## Saved evidence

Each NAS run directory contains:

- `launch_manifest.json`, `preflight.json`, `resolved-config.preflight.yaml`,
  task config, model catalog and prompt hashes.
- `status.json`, `exit-code.txt`, supervisor and training logs.
- `data/`: immutable split/catalog manifests and prepared Parquet files.
- `episodes/`: per-attempt observations, actions, Codex JSONL, stderr, results
  and execution metadata, including partial logs after a timeout.
- `agent_logs/`: Uni-Agent session trajectories with original generated token
  IDs, response masks and behavior log probabilities.
- `rollouts/` and `validation/`: trainer generation/reward records.
- `checkpoints/`: model, optimizer and extra state; skill/EMA joint state is
  owned by the custom trainer when that mode is enabled.
- `metrics.jsonl`: authoritative observations written by the custom trainer.

After training, the supervisor exports recorded metrics when available:

```bash
/opt/conda/bin/python examples/skill_self_evolve/scripts/plot_metrics.py RUN_DIRECTORY
```

This writes `metrics.csv`, `reward_accuracy.png` and `reward_accuracy.svg`.
Training reward and recorded accuracy are plotted separately. Missing metrics
remain missing; they are not replaced by fabricated observations or smoothed
into an improvement. A successful exit only proves execution, not learning.

## Integration references

- [verl extension points](https://verl.readthedocs.io/en/latest/extend_guide.html)
- [Uni-Agent RL training](https://uni-agent.readthedocs.io/en/latest/quickstart/rl-training/)
- [Codex custom providers](https://learn.chatgpt.com/docs/config-file/config-reference)

This recipe uses the official `AgentFrameworkRolloutAdapter`, Task registry,
native CodexAgent, Gateway token-in/token-out protocol, TransferQueue and FSDP2.
It does not replace those components with an inference-only agent loop.
# Initial and final official tests

Set `EVALUATE_OFFICIAL_TESTS=true` for a full training experiment. The same
initialized native Uni-Agent/Codex manager evaluates all 140 official seen and
134 unseen games before optimization and after successful training. Greedy
sampling, model context limits, prompt hashes and complete task identities
must match between stages. Scheduled intermediate validation still uses only
the train-derived development partition. Official tests never generate curate
actions or optimizer updates. Their trajectories, protocols and metrics are
saved under `evaluation/initial` and `evaluation/final`; append-only telemetry
also records the stage explicitly. The default is false for integration smoke
runs, which do not establish improved test performance.

