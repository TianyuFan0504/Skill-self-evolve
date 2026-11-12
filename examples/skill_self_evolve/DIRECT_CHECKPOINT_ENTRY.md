# Entry composition for direct checkpoint reads

`scripts/train_direct_checkpoint.py` composes the direct restore provider with
the existing parallel environment, idempotent action client, Native spawn
boundary and sealed final-evaluation trainer. It introduces no new model,
optimizer, training data, prompt or sampling budget.

The public parallel launcher selects this entry when resuming a joint run:

```bash
bash examples/skill_self_evolve/run_resilient.sh --parallel-environment \
  --run-dir /path/to/existing-run --resume
```

Fresh launches retain their original entry. The existing entry files remain
available for reproducing historical executions. A running process is not
patched or restarted when these new source files are installed; the new
coordinator is selected only at a future startup.

The new trainer inherits the frozen spawn/quiescent trainer and adds only
`DirectReadRestoreMixin`. That mixin delegates the actual frozen core
`_load_checkpoint` method after installing the direct-read coordinator. All
model/optimizer/EMA/data loaders keep the original checkpoint source paths.
See [DIRECT_CHECKPOINT_READ.md](DIRECT_CHECKPOINT_READ.md) for the one-validation
proof and mutable-state isolation contract.

## Explicit source and restore evidence

The old core fingerprint does not include the examples extension. The entry
therefore writes separate records rather than pretending an old outer entry
was executed:

- `driver_runs/*.direct_checkpoint.json` identifies the actual selected entry,
  both new implementation source hashes, unchanged parallel/spawn component
  hashes, the core provider hashes, live environment identity, client adapter
  and effective prelaunch config. Its kind explicitly describes component
  composition and is not a post-init or served-model receipt.
- `driver_runs/*.direct_checkpoint.initialized.json` links the actual source
  identity to the resolved config, source checkpoint specification, unchanged
  core version binding, direct restore receipt and the spawn component's real
  initialized receipt.
- The mutable working root's `DIRECT_READ_RESTORE.json` records the original
  actor/EMA/data paths, one full validation, zero copied EMA files and the
  bounded mutable-state copies. The existing `RESTORE_COMPLETE.json` success
  boundary remains unchanged.

The original training manifest is preserved. Downstream comparison tools must
consume these new explicit records when assessing a run resumed through this
entry; an older parallel-driver observation is not substituted for them.

## CPU composition checks

```bash
python examples/skill_self_evolve/tests/test_train_direct_checkpoint.py -q
```

The entry suite stubs every model, Ray and process operation. It checks that
the Native spawn policy is installed before trainer construction, initialization
records precede sampling, failure evidence and queue/tracking cleanup retain
their previous order, the frozen fit method remains inherited, and the new
initialized record names the actual outer entry and both restore/spawn
component proofs. It does not claim a completed GPU startup or model update.
