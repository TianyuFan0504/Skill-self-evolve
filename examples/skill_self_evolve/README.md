# ALFWorld Skill Self-Evolution Training

This recipe connects the **native Codex CLI** to Qwen3.5-4B through the official
Uni-Agent Responses Gateway and trains the resulting token trajectories with
the pinned verl 0.9.1 checkout. Codex remains the task-solving harness; it is not
replaced by a separately implemented chat agent.

Use `run_resilient.sh` for the online solver's combined solver/curate update,
parameter EMA curator and frozen counterfactual evaluator. The original
`run.sh` retains its separate solver-only and deferred-reference entrypoints.
A passing solver-only run does not establish delayed skill-credit behavior.

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

The wrapper enables `skip_host_skill_discovery` and sets
`-c skills.include_instructions=false`. The second setting removes Codex's
automatic Skills catalog and Skill roots from the model input, including the
per-attempt `CODEX_HOME` path. The discovery flag alone does not remove that
block. This does not remove the harness's explicit SkillStore content from the
Task user message. Native tokenization and the declared tools remain unchanged.
A CPU-only request capture on Codex 0.167.0 checked two distinct homes: the
automatic block and home paths disappeared, while all nine tool declarations
were identical. Every capture endpoint response was HTTP 400; no model or
generated response was involved. The evidence manifest is
`/root/nas_chaohuang/skill_training/diagnostics/native_solver_skill_instructions_1794433121134095868/evidence-manifest.json`.

The joint checkpoint fingerprint includes the solver wrapper, launcher,
`solver_system.md` and `solver_task.txt` bytes. Changing these input assets
requires a compatible new source cut; it cannot silently resume an old joint
checkpoint.

Codex runs under an exclusively leased `alfworld_ep2_000..127` identity and
interacts using `alfworld-step 'EXACT ACTION'`. Provision the
[private account registry](ACCOUNT_ISOLATION.md) before launch. Private server
capabilities, game files, expert plans and other episode artifacts are not
supplied to that identity. The Task creates a unique attempt directory for
every rollout and epoch.

## Start a run

Apply the narrowly pinned [Uni-Agent compatibility fix](patches/UNI_AGENT_COMPATIBILITY.md)
offline after stopping old workers. The launcher verifies the dependency and
prints the explicit application command if it is absent.

From the server checkout:

To prioritize actual joint training, skip the complete official initial
evaluation and development pre-validation. Model initialization and the
kernel/Native preflight still run; the first model rollout then uses training
tasks directly:

```bash
cd /workspace/verl-0.9.1
bash examples/skill_self_evolve/run_resilient.sh \
  --run-dir /root/nas_chaohuang/skill_training/runs/my_joint_experiment
```

This command prepares data, checks the existing runtime and starts resilient
joint training with batch32/N4, Native concurrency32 and a64-update default.
It requires no private captured-environment helper. See the
[launcher flags, resume and receipts](RESILIENT_LAUNCHER.md).
This retains quiescent final official evaluation. The immutable base checkpoint remains
available for a later matched base-versus-trained comparison; a complete
274-game initial evaluation is an optional experiment measurement, not a
training prerequisite. Do not claim improvement by comparing a training
batch against test scores or by reusing scores from a different backend.

For the separate solver-only mode:

```bash
cd /workspace/verl-0.9.1
TOTAL_TRAINING_STEPS=64 \
TRAIN_BATCH_SIZE=16 ROLLOUT_N=4 ROLLOUT_CONCURRENCY=64 \
bash examples/skill_self_evolve/run.sh
```

By default the launcher prepares the original data, starts a private local
ALFWorld service, runs CPU import/configuration checks, and starts a detached
supervisor. It prints the run directory, supervisor PID and status path. The
solver-only training subprocess is:

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
| `SAVE_FREQUENCY` | 10 (public joint launcher) | Full model/optimizer/extra checkpoint interval; final step is always saved |
| `MAX_CHECKPOINTS` | 2 | Model checkpoint retention; trajectory/metric files are retained |

`MAX_MODEL_LENGTH` must equal prompt plus response capacity in this recipe.
The launcher keeps the native Codex catalog declaration consistent with that
capacity. Keep `TRAIN_BATCH_SIZE` and `PPO_MINI_BATCH_SIZE` equal when using a
custom trainer that assumes one optimizer update per role RPC. Changes to GPU
count, concurrency or token limits must be recorded in a new run manifest.

## Saved evidence

The [formal v3 initial baseline](../../docs/skill_self_evolve/native_formal_initial_v3.md)
records all 274 official games, independent per-case reconciliation and the
boundary between a complete baseline and demonstrated training improvement.

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

## Joint native skill training

Complete the one-time [private account setup](ACCOUNT_ISOLATION.md) and
[host acceptance](ACCOUNT_ISOLATION_ACCEPTANCE.md) before starting the recipe.

On the eight-H100 host, start the full solver/curate harness with:

```bash
EVALUATE_OFFICIAL_TESTS=true TOTAL_TRAINING_STEPS=64 \
  bash examples/skill_self_evolve/run.sh --joint \
  --run-dir /root/nas_chaohuang/skill_training/runs/alfworld_joint_001
```

The joint recipe reserves four solver/FSDP GPUs, two EMA-curator GPUs and two
frozen-credit GPUs. Use `--joint --preflight-only` to check imports, restricted
native execution, the real GPU budget and data isolation without allocating a
model. `TRAIN_BATCH_SIZE`, `ROLLOUT_CONCURRENCY` and standard Hydra overrides
control throughput. The joint actor makes exactly one optimizer step over its
actual mixed solver/curate batch, then advances EMA from the observed step.

Skill candidates become eligible for delayed credit on later original training
tasks from the same family. The fixed evaluator runs native paired probes;
unobserved windows remain pending. Settled signed credit enters retained curate
actions through replay and positive utility admits skills into the main store.
Curate actions never enter the solver's GRPO groups. Recovery requires the
separate COMPLETE joint manifest, including online optimizer, EMA, skill store,
replay, credit ledger, controller and dataloader state.

The joint recipe samples one curator candidate per source episode. Its signed
counterfactual advantage supplies an external baseline, so curator candidates
do not need a multi-sample GRPO group. This also avoids censoring an otherwise
valid candidate when a sibling has invalid output. Solver `ROLLOUT_N` remains
four. Curator sampling stays at temperature one with original processed
behavior probabilities; malformed JSON and missing identity fields remain
rejected. The system instructions enumerate every required field, and the task
ends with the exact host-assigned identity and output structure. These prompt
and candidate-count changes require a fresh protocol/template version.

The production joint recipe queues every completed logical solver episode,
rather than selecting one source per update. Dedicated background workers run
curation and paired evaluation while solver training continues; the learner
uses only matured labels and remains the sole optimizer owner. Curators may
explicitly abstain without inventing a skill or reward. The default credit
budget is 32 pairs per update and ready replay accepts up to 32 labels.
See [background scheduling](BACKGROUND_SKILL_PIPELINE.md) and
[abstention records](CURATOR_ABSTENTION.md) for version pins and checkpoint cuts.
The supported custom sampler refills complete failed prompt groups, including
materialized siblings. A host-local archive-failure record prevents a partial
group from slipping through the framework's provisional finished status.
See [local archive failures](LOCAL_ARCHIVE_FAILURES.md) for the fault boundary.

Curator inputs require a nonempty observed environment action sequence. A
completed training episode with a consistent private zero-step result and an
absent or empty action trace is recorded as `source_censored` before sampling.
It creates no curate action, reward event or training label, and preserves the
solver outcome and completed online/EMA update. Missing traces for positive-step
episodes, malformed sources and inconsistent private artifacts remain invalid.
Background collection records those source failures locally; it does not accept
their content or stop an already completed solver update. The synchronous
protocol-test path still raises these errors directly.

The curator wrapper disables native goals, web search and request-user-input
registration in addition to execution features. A CPU-only native HTTP capture
on Codex 0.167.0 confirmed that these settings removed the remaining five tool
declarations. This capture used an explicit failing mock endpoint and provides
configuration evidence only; it contains no model tokens or training reward.
The catalog's parallel-tool flag alone did not control the observed HTTP flag,
so the harness preserves the native request value without rewriting it.
The capture inventory is retained at
`/root/nas_chaohuang/skill_training/diagnostics/native_curator_tool_flags_1794430776109031851/evidence-manifest.json`.

The common model configuration enables the official fused Torch output head.
It was checked on a saved 33,322-token native trajectory: finite likelihood,
entropy and masked backward, about 24.24 GiB peak allocation on one H100. The
unfused full-vocabulary entropy path exhausted an 80 GiB GPU on that batch.
These kernel checks are separate from actual distributed training results.

The solver wrapper sets `allow_login_shell=false`; environment observations
therefore exclude the host's login banner. This is a supported native Codex
configuration verified on the installed version. Initial and final comparisons
must use the same wrapper, prompts, generation budgets and full task identities.

