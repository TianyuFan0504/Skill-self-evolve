# One native paired-input diagnostic

This driver runs one pair after the source cut that removes native Skills
instructions and enables the exact native-input gate. It uses the actual v4
S0 artifact and the original future-B row saved in that run's controller:

- S0 collection: `0bb3c733d0963f8a9dc0535b38e912bb8ccf6e8c18aa49c9a15f592015ab75bc`.
- Producer A: `train_13f5d21048f1b70d237bd27c`, training step 1.
- Future B: `train_6408c0ae5173d1bf3ac38651`, the original step-2 train assignment.

Preparation copies the existing data and SkillStore bytes into a fresh run and
records every file hash. It does not generate tasks. The Task run directory is
the only changed field in the saved B row; the official context pin supplies
the cloned store and the assigned immutable snapshot for each branch. The store
is opened read-only and its bytes are checked again after execution. Original
v4 artifacts are never modified.

Use the same verified Python/Uni-Agent environment as the recipe. Preparation
starts no service, Ray cluster or GPU process:

```bash
/opt/conda/bin/python examples/skill_self_evolve/scripts/probe_native_pair.py \
  --run-dir /root/nas_chaohuang/skill_training/runs/alfworld_native_pair_input_v6_20261112
```

Review `probe-input.json`, then use `--execute` to run the prepared pair. Select
one idle physical GPU; the driver rejects sharing it. Without `--execute`, the
command only prepares and validates input files.

```bash
CUDA_VISIBLE_DEVICES=0 \
/opt/conda/bin/python examples/skill_self_evolve/scripts/probe_native_pair.py \
  --run-dir /root/nas_chaohuang/skill_training/runs/alfworld_native_pair_input_v6_20261112 \
  --execute
```

The script uses `launch.install_codex_runtime` and `launch.start_environment`
for a fresh train-only ALFWorld service. It creates an official standalone
`LLMServerManager` with `worker_group=None`, wraps it in a frozen `PolicyRoute`,
then calls `create_frozen_probe_gateway` and `collect_counterfactual_probe`.
It creates no solver, learner, optimizer, EMA, replay writer or credit ledger.
Both branches use the original frozen checkpoint, greedy sampling and seed 42,
the full existing horizon, one common empty work directory and separate native
homes. The native CodexAgent, Gateway encoding and sampled token IDs are kept.

The private account roster must be initialized and its production preflight
verified first. The outer collector retains one leased identity across both
variants, including source archival and backend/process drain. See
[account isolation](ACCOUNT_ISOLATION.md).

The copied archived configuration preserves the model and rollout geometry.
Before allocating a model, the driver applies the current common recipe's
rollout API mode and backend engine options. In this pinned cut that retains
the asynchronous API with vLLM `async_scheduling=false`. Actual resolved values
are saved in `diagnostic-config.yaml`. Runtime provenance also hashes the
account, Task, archive, launcher and common recipe sources alongside the
existing context, prompt and collector assets.

`result.json` reports `input_control_passed` only when the collector's immutable
`paired-input-audit.json` reports `observed` and proves the actual initial model
inputs differ only by S0. No path normalization is allowed. Raw outcomes remain
visible for audit, but the script creates no utility label, admission or learning
claim. A single pair cannot establish that a skill is beneficial.

An execution intent prevents a second launch or uncertain retry. Failed or timed
out branches retain their original captures; use a new reviewed run rather than
resampling the same diagnostic identity. The driver shuts down its owned Gateway
and private local Ray cluster, stops only the ALFWorld process whose command line
matches this run, saves Ray logs under `ray_logs/`, removes its temporary scratch
directories, and checks that the selected GPU has no compute process. See
`cleanup.json`; an unconfirmed cleanup makes the command fail. Other Ray clusters,
services and GPU processes are never stopped.

CPU boundary tests use temporary fixture files and mocked lifecycle calls:

```bash
uv run --offline --no-project --python 3.10 python -m unittest discover \
  -s examples/skill_self_evolve/tests -p test_probe_native_pair.py -v
```
