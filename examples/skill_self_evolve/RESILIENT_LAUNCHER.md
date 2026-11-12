# Public one-command joint launcher

With the documented original benchmark, model, training Python, compatible
Uni-Agent and existing private account registry installed, start a fresh joint
run without a captured private environment:

```bash
cd /workspace/verl-0.9.1
bash examples/skill_self_evolve/run_resilient.sh \
  --run-dir /root/nas_chaohuang/skill_training/runs/my_joint_experiment
```

The default profile is solver4 + curator2 + frozen2, batch32/N4, Native
concurrency32, 64 optimizer updates, original ALFWorld data, Native900s and full
36,864 capacity. Initial official/development validation is disabled; final
official evaluation remains enabled. Final evaluation reuses the quiescent
checkpoint/library seal. The normal data preparation creates the same
3154/399/140/134 original split under the pinned benchmark/catalog profile.
No additional tasks or synthetic chains are generated.

The default checkpoint interval is10 optimizer updates: steps10,20,30,40,50,
and60, plus the final step64. The final checkpoint is saved even though64 is
not divisible by10. An explicitly provided `SAVE_FREQUENCY` or Hydra
`trainer.save_freq` override remains supported. Existing running trainers have
already resolved their configuration; applying a new interval requires their
next controlled startup rather than changing a file underneath the process.

`--foreground`, `--prepare-only`, `--preflight-only`, `--resume`, `--run-dir`
and `--allow-busy-gpus` have the existing launcher's meanings. `--joint` is
accepted but unnecessary; this new command is joint-only. Preparation and
preflight modes are mutually exclusive. Unknown flags are rejected. Remaining
arguments are ordinary Hydra `key=value` overrides; pretraining evaluation,
the safe runner/sampler, Native concurrency32 and N4 are fixed explicitly.
Use the old `run.sh` for solver-only training or the separately planned deferred
zero-epoch base reference with official initial evaluation.

`--run-dir` controls the artifact destination. An explicit Hydra
`skill_harness.artifact_root` is accepted only when it names that same absolute
resolved directory; redirection and whole-harness replacements are rejected
before preparation writes. The resolved preflight config is checked again
before creating a fresh training manifest. Read-only model/data input paths
remain configurable.

Environment variables retain the existing recipe's model/data/training
controls. For example, set `MODEL_PATH`, `TRAINING_PYTHON`,
`ALFWORLD_BENCHMARK_ROOT`, `TOTAL_TRAINING_STEPS` or `TRAIN_BATCH_SIZE` before
the command. `ROLLOUT_CONCURRENCY` and `ROLLOUT_N`, if provided, must be32 and4.
Retain the same custom environment and Hydra overrides when resuming:

```bash
bash examples/skill_self_evolve/run_resilient.sh \
  --run-dir /root/nas_chaohuang/skill_training/runs/my_joint_experiment --resume
```

Resume keeps the strict COMPLETE checkpoint compatibility checks. A recorded
live training process or owned supervisor is refused before preparation. A
failed run's incidents/resources must be reconciled through the existing
procedures; the launcher does not reset accounts, clear incidents or bypass
the learner barrier.

For threaded Ray workers, install and verify the helper described in
[NATIVE_SPAWN.md](NATIVE_SPAWN.md), then select the fork-free Native boundary:

```bash
bash examples/skill_self_evolve/run_resilient.sh --spawn-safe \
  --run-dir /root/nas_chaohuang/skill_training/runs/my_joint_experiment
```

Keep `--spawn-safe` when resuming that profile. The launcher verifies that the
installed helper can load before preparing files or starting the service. It
never compiles or replaces a helper while a worker may be using it. This mode
selects `train_spawn_safe.py` and its public runner class while preserving the
same sampler, N4, concurrency32 and existing supervision. Launch receipts also
include the composed resilient driver, C/Python adapter and built helper hashes.

The separate `--parallel-environment` profile includes that spawn boundary and
selects four independent TextWorld processes, four durable-I/O workers and the
request-ID-aware action client together:

```bash
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment \
  --run-dir /root/nas_chaohuang/skill_training/runs/my_joint_experiment
```

Sessions remain attached to one process. Each session waits for its ordered
durable writes while other sessions can continue. Retries query the original
request result and retain the same action ID. The launcher refuses an existing
legacy service rather than silently changing a live service. Resume requires
the same profile and the separately reconciled service transition.

For `--parallel-environment --resume`, the launcher automatically selects
`train_direct_checkpoint.py`. Its coordinator reuses one full checkpoint
validation and reads model, optimizer, EMA and dataloader files directly from
the original immutable checkpoint. Only writable skill/replay/ledger state
and small JSON metadata get independent working copies. Fresh parallel starts
keep their existing entry. See [DIRECT_CHECKPOINT_READ.md](DIRECT_CHECKPOINT_READ.md)
and [DIRECT_CHECKPOINT_ENTRY.md](DIRECT_CHECKPOINT_ENTRY.md) for the source
guards and explicitly recorded outer-entry/component provenance. This choice
does not restart an already running experiment.

Training and actor/reference logprob microbatch sizes remain explicit Hydra
overrides. Choose them from the isolated memory, throughput and numerical
checks for the actual model and trajectory lengths; process parallelism alone
does not establish that a larger GPU microbatch fits.

## Shared implementation and receipts

The new Python adapter calls the existing `launch.py --prepare-only` in the
same process, temporarily routing CLI arguments rather than replacing its
functions. Existing preparation archives old attempt metadata before rewriting
it and leaves resolved runtime paths in `os.environ`. Service creation,
preflight and supervision then use the existing helpers with the new resilient
driver command. Detached jobs use the original `launch.py --supervise` mode.
Preflight failure stops only a service started by this invocation.

For a fresh actual training start, the adapter creates the original training
manifest once from the resolved CPU-preflight config and existing manifest
writer. Resume preserves that original manifest. Every invocation also writes
a new `launcher_runs/*.resilient.json` receipt containing the exact command and
launcher/shell/driver/quiescent-driver/namespace hashes. Post-init effective
config and final lifecycle receipts remain the resilient driver's responsibility.
This launcher adds no core source fingerprint change and does not rewrite the
existing `run.sh`, `launch.py` or active training drivers.

CPU tests verify archival/order, same-process path handoff, duplicate refusal,
flag rejection, owned-service cleanup, exact supervisor payload and manifest
preservation with all service/model/process calls stubbed. They are not a
training or GPU acceptance run.
