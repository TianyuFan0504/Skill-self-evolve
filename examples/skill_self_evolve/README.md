# ALFWorld joint-training recipe

One online model and optimizer train the solver and curator objective. The
curator samples from a parameter-EMA replica, and a frozen evaluator assigns
delayed credit through paired with-skill/without-skill episodes. Read the
[current design](../../docs/skill_self_evolve/design.md) for the data flow.

## One-command launch

On a provisioned Linux GPU host, run from the installed repository:

```bash
bash examples/skill_self_evolve/run.sh \
  --deployment /absolute/path/deployment.json \
  --run-dir /absolute/path/new-run
```

`run.sh` calls the single `scripts/launch.py` entry. The launcher publishes the
declared public assets and `alfworld-step` command, initializes a new optional
Linux account registry when explicitly requested, prepares fresh benchmark
parquet files and catalog, starts the parallel training environment, checks its
loopback HTTP health, checks the resolved joint configuration, and runs
`python -m verl.experimental.skill_harness.train` with `config/alfworld_joint.yaml`.
The default remains 64 joint updates with four environment processes and four
I/O workers. Existing run directories are rejected.

For a short real joint run:

```bash
bash examples/skill_self_evolve/run.sh \
  --deployment /absolute/path/deployment.json \
  --run-dir /absolute/path/new-smoke-run --smoke
```

`--smoke` performs six optimizer updates with batch size 16, four rollouts per
prompt, 16 credit probes per step and replay batch size eight. Curator and
frozen-evaluator concurrency follow the deployment settings (recipe default
eight). Delayed credit retains its two-step minimum
and two required exposures. It saves at update six and skips the final official
test. Solver, EMA curator, paired credit and curator replay all remain enabled.

Hydra overrides follow the launch options. The order is deployment overrides,
smoke overrides, then command-line overrides, so the last value takes priority:

```bash
bash examples/skill_self_evolve/run.sh \
  --deployment /absolute/path/deployment.json --run-dir /absolute/path/new-run \
  trainer.total_training_steps=64 trainer.save_freq=8
```

The launcher stays in the foreground and owns its backend and training child
processes. On success, failure or a stop signal, it stops the children it started.
Redirect its output or use your host's job manager when a detached run is needed.
`launch-processes.json` records owned PIDs, Linux process start ticks and exit
codes. It never searches for or terminates another run.

## Deployment JSON

Keep host-specific deployment files outside Git. All paths, installed
interpreters, identities and model locations are explicit. Only the literal
strings `{run_dir}` and `{run_id}` are substituted recursively; `{run_id}` is
the new output directory's basename. No shell expressions are evaluated.

| Field | Meaning |
| --- | --- |
| `environment` | String-valued environment variables listed below, plus recipe settings such as GPU allocation and token limits |
| `runtime.account_pool` | Importable `factory` and explicit provider `kwargs` |
| `runtime.initialize_if_missing` | Optional `true` initializes the bundled Linux provider only when its configured registry root does not exist |
| `runtime.native_policy` | Solver `instructions_file` and optional `disabled_features`, passed to the actual native binary |
| `task_config` | Full Uni-Agent task configuration, written privately for this run; alternatively set `environment.ALFWORLD_TASK_CONFIG` to an existing file |
| `public_assets` | Optional list of public files with absolute `source`, `destination`, and optional octal-string `mode` (default `0644`) |
| `tool_dir`, `tool_python` | Public command directory and agent-readable Python interpreter for the standard-library request client, for example `/usr/bin/python3` |
| `environment_processes`, `io_workers` | Backend process and I/O counts, both defaulting to four |
| `max_steps`, `budget_seconds` | Backend limits, defaulting to 50 and 900 |
| `readiness_timeout_seconds` | Maximum backend startup wait, defaulting to 180 |
| `hydra_overrides` | Optional list of Hydra override strings |

Required values in `environment`:

| Variable | Required value |
| --- | --- |
| `TRAINING_PYTHON` | Absolute interpreter path in the installed training environment |
| `ALFWORLD_ENV_PYTHON` | Absolute interpreter path with ALFWorld, TextWorld and aiohttp |
| `MODEL_PATH`, `MODEL_NAME` | Local model weights and matching native model-catalog entry |
| `ALFWORLD_BENCHMARK_ROOT` | Existing benchmark root containing its `data` directory |
| `CODEX_NATIVE_BIN`, `ALFWORLD_CODEX_PACKAGE_ROOT` | Actual native executable and readable installed package root |
| `ALFWORLD_SOLVER_INSTRUCTIONS_FILE` | Deployed public solver instructions |
| `CURATOR_USER` | Existing curator account, separate from solver accounts |
| `CURATOR_PUBLIC_ROOT`, `CURATOR_WORK_ROOT` | Public curator assets and isolated curator work directories |

`--run-dir` can instead be supplied as `environment.SKILL_RUN_DIR`. The launcher
sets fresh run-local data, runtime, service and generated task paths. The
optional `ALFWORLD_ENV_SERVER` selects the existing request-ID backend protocol;
its default is this recipe's `scripts/parallel_environment_server.py`.

The task configuration selects `alfworld_skill_training`, the actual native
Codex executable, a public model catalog and local sandbox settings. Native
policy is applied directly by the task. `alfworld-step` is only the public
ALFWorld request command, with the profile's explicit `tool_python` interpreter.
The runtime provider's `command_path` must include `tool_dir`; its `read_roots`
must include the public tools, native package, model catalog and instructions.
Publish only declared public assets. Benchmark files, credentials, checkpoints
and service capabilities belong outside those public roots.

The optional provider
`verl.experimental.skill_harness.backends.linux:LinuxAccountPool` takes
explicit `root`, `roster`, `work_root`, `read_roots`, `command_path` and optional
public `launcher_path` arguments. Each roster entry gives an existing account's
name, UID and GID. The launcher does not create users or choose identities.
Existing registries are used as-is: it never resets or reconciles old accounts.
See [deployment requirements](../../docs/skill_self_evolve/environment.md).

Use one installed training environment for data preparation, configuration
checks, the trainer and Ray workers. Dependencies must be installed there;
Python import overlays and mixed environments are rejected. The launcher does
not download weights or data, install packages or edit Python search paths.
The environment-engine and public command interpreters remain explicit.

## Check configuration and inspect results

For a separate configuration-only check, point the documented environment
variables at prepared inputs and run:

```bash
"$TRAINING_PYTHON" examples/skill_self_evolve/scripts/preflight.py
```

This CPU check starts no environment, native agent or trainer. The output
directory keeps the data catalog and split manifest, private configuration,
raw native/Gateway trajectories, environment actions, solver rewards, skill
proposals or abstentions, paired credit outcomes, `metrics.jsonl`, resolved
configuration and joint checkpoints. Standard verl checkpoint settings control
saving; all joint state used by the learner comes from the same checkpoint.

To export recorded metrics and plots:

```bash
"$TRAINING_PYTHON" examples/skill_self_evolve/scripts/plot_metrics.py "$SKILL_RUN_DIR"
```

The exporter writes a CSV and plots from saved observations without smoothing
or inventing values. A configuration check or CPU test alone is not evidence
of an optimizer update or benchmark improvement.
