# ALFWorld joint-training recipe

One online model and optimizer train the solver and curator objective. The
curator samples from a parameter-EMA replica, and a frozen evaluator assigns
delayed credit through paired with-skill/without-skill episodes. Read the
[current design](../../docs/skill_self_evolve/design.md) for the data flow.

## Provision the deployment

Use a Linux GPU host with installed verl, Uni-Agent, native Codex, ALFWorld,
TextWorld, Ray, TransferQueue and the selected FSDP2/vLLM dependencies. Prepare
and transfer dependencies and weights from the networked local workstation;
the recipe does not download, install, provision users, or edit Python search
paths. See [deployment requirements](../../docs/skill_self_evolve/environment.md).

Set these values to paths and identities in your deployment, or override the
corresponding YAML fields:

| Variable | Required value |
| --- | --- |
| `TRAINING_PYTHON` | Installed training interpreter; defaults to `python` |
| `MODEL_PATH`, `MODEL_NAME` | Local model weights and the matching native model-catalog entry |
| `SKILL_RUN_DIR` | Durable output directory for this run |
| `SKILL_DATA_DIR` | Prepared `train.parquet`, `dev.parquet`, `test_seen.parquet`, `test_unseen.parquet` |
| `ALFWORLD_DATA_DIR` | Prepared `catalog.json` and `split_manifest.json` |
| `ALFWORLD_BENCHMARK_ROOT` | Existing benchmark root, with its `data` directory |
| `ALFWORLD_TASK_CONFIG` | Uni-Agent task configuration naming `alfworld_skill_training` |
| `ALFWORLD_SERVER_CONFIG` | Configuration published by the provisioned training environment service |
| `ALFWORLD_RUNTIME_CONFIG` | Trusted JSON selecting the native runtime-provider factory and its arguments |
| `ALFWORLD_ENV_PYTHON` | Interpreter with the environment engine dependencies |
| `ALFWORLD_ENV_SERVER` | Backend script implementing the ALFWorld action request-ID protocol |
| `CODEX_NATIVE_BIN` | Actual executable native Codex binary |
| `ALFWORLD_CODEX_PACKAGE_ROOT` | Readable native Codex package root |
| `ALFWORLD_SOLVER_INSTRUCTIONS_FILE` | Solver instructions, normally the deployed copy of `prompts/solver_system.md` |
| `CURATOR_USER` | Provisioned curator account, separate from solver accounts |
| `CURATOR_PUBLIC_ROOT`, `CURATOR_WORK_ROOT` | Public curator assets and isolated curator work directories |

The runtime file declares `account_pool.factory` and `account_pool.kwargs`.
The optional `examples.skill_self_evolve.runtime.linux_accounts:LinuxAccountPool`
provider takes explicit `root`, `roster`, `work_root` and `read_roots` values.
Each roster entry declares the deployed account's name, UID and GID. Provision
these accounts and writable directories before launch. No account names, UID
ranges, host paths or private deployment files are bundled in the recipe.
Initialize the optional Linux provider explicitly with `AccountPool().initialize()`
during deployment setup; the normal launcher does not initialize its registry.
Set its `command_path` to include the deployed `alfworld-step` directory.
If the checkout is private, `launcher_path` can select a public, readable copy
of `runtime/linux_accounts.py` for the leased account's launcher.
The optional provider is part of this recipe checkout; use the documented
editable installation and repository working directory, or install your own
provider package under an importable factory name.

Prepare the data once, using the same run directory as training:

```bash
"$TRAINING_PYTHON" verl/experimental/skill_harness/alfworld/prepare_data.py \
  --benchmark-root "$ALFWORLD_BENCHMARK_ROOT" \
  --output-dir "$SKILL_DATA_DIR" --run-dir "$SKILL_RUN_DIR" \
  --prompt-file examples/skill_self_evolve/prompts/solver_task.txt
```

Set `ALFWORLD_DATA_DIR` to that output directory. Configure the environment
service with the resulting catalog and only the official `train` split.
The generic `scripts/parallel_environment_server.py` backend accepts explicit
benchmark, run, catalog, config-output, process-count and I/O-count arguments.
Deploy `scripts/alfworld_request_client.py` as the agent's `alfworld-step` command
on its allowed tool path. The backend and client share the action request-ID
protocol so an uncertain response can be queried without repeating an action.
They are environment helpers, and do not start training.

The task configuration supplies the actual native binary, readable model
catalog, agent timeout and local sandbox settings. The solver instructions must
be loaded by that configured native agent. The training process and workers
must read the environment and runtime files; native agent processes receive
only the session URL and their configured public assets.
The protected service configuration may declare `native_policy.instructions_file`
and `native_policy.disabled_features`; the instructions path otherwise comes
from `ALFWORLD_SOLVER_INSTRUCTIONS_FILE`. The task passes this policy directly
to the actual native executable while preserving Uni-Agent's prompt transport.

## Run

From the repository root, with the deployment configured:

```bash
bash examples/skill_self_evolve/run.sh
```

The script uses the selected interpreter and forwards Hydra arguments directly
to `verl.experimental.skill_harness.train`, with the single configuration
`config/alfworld_joint.yaml`. For example:

```bash
bash examples/skill_self_evolve/run.sh trainer.total_training_steps=64
```

The default GPU allocation is four solver GPUs, two EMA-curator GPUs and two
frozen-evaluator GPUs. Change `TRAINING_GPUS`, `CURATOR_GPUS`, `CREDIT_GPUS` and
`PHYSICAL_GPU_COUNT`, or their YAML fields, to match the provisioned host.
Development validation is train-derived. The optional final official test uses
the trained policy and final skill library; configure it with
`skill_harness.evaluation.final`.

## Check configuration and inspect results

This separate check resolves the configuration and checks installed modules and
input files on CPU. It does not start an environment, native agent or trainer:

```bash
"$TRAINING_PYTHON" examples/skill_self_evolve/scripts/preflight.py
```

The run directory keeps raw native/Gateway trajectories, environment actions,
solver rewards, skill proposals or abstentions, paired credit outcomes,
`metrics.jsonl`, the resolved configuration and joint checkpoints. Use the
standard verl checkpoint settings for save/resume behavior. All joint state
needed by the learner must come from the same checkpoint.

To export recorded metrics and plots:

```bash
"$TRAINING_PYTHON" examples/skill_self_evolve/scripts/plot_metrics.py "$SKILL_RUN_DIR"
```

The exporter writes a CSV and plots from saved observations without smoothing
or inventing values. A valid configuration or CPU test is not evidence of an
optimizer update or benchmark improvement.
