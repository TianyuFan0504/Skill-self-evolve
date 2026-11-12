# Quiescent final evaluation entrypoint

`scripts/train_quiescent_eval.py` is an examples-only TaskRunner extension using
the public `run_ppo(config, task_runner_class=...)` contract. The core joint
trainer, losses, optimizer, EMA and checkpoint compatibility code are unchanged.
This driver does not enable initial evaluation or development validation before
training; it rejects both pretraining switches.

At the final configured checkpoint, the wrapper stops background work before
the original save. It seals the actual optimizer clock and immutable main-library
head into that COMPLETE checkpoint. Training proceeds through its normal final
solver-weight publication; the original joint `fit` finally stops background
work and closes EMA/CF gateways. The solver manager remains alive.

Only then does a new validation manager use a **copy of the actual restored
configuration**, with `loading.snapshot_id` explicitly set to the seal. The
trainer's original loading configuration and checkpoint protocol remain
unchanged. The unchanged official evaluator runs both seen and unseen against
one snapshot. The wrapper reopens/closes the dump executor and verifies that the
actor counter and library head do not change during testing. The final trainer
cursor can be 65 while the actual parameter version is 64.

The script accepts the same Hydra configuration arguments as the existing
training module. The operator supplies the captured environment, original run
and dataset bindings, and a verified COMPLETE resume path. It is intended for a
controlled checkpoint-boundary handoff; it does not stop or edit another job,
select checkpoints, migrate fingerprints, or reset training. The original
`training-manifest.json` and `resolved-config.yaml` are never overwritten by this
entrypoint. Existing final artifacts are refused rather than overwritten.

A strict restore of an already COMPLETE final checkpoint also works when no
training iterations remain. It uses that checkpoint's exact snapshot and clock,
quiesces through the original cleanup, performs no new optimizer step or save,
and records `seal_origin=restored_complete_final_checkpoint` before evaluation.

New `driver_runs/*.json` records retain the driver hash and redacted requested
configuration. `*.initialized.json` records the effective configuration **after
restore**, including the actual restored store/replay/ledger locations, resume
checkpoint identity and original training-manifest hash. New
`evaluation_lifecycle/<driver-id>/` records the final COMPLETE seal, fixed head,
actual store root and final evaluation clocks. This external lifecycle is
explicitly recorded; an unchanged core fingerprint does not claim unchanged
overall execution. Offline comparison must resolve these explicit restored
bindings rather than assume the original manifest's store path still applies.
Successful `evaluation/final/lifecycle.json` explicitly links the three hashed
lifecycle receipts and records the published driver source commit/hash. Review
tools must follow those exact references rather than select a latest directory.

When final evaluation is disabled, the wrapper delegates the original fit
without extra evaluation or checkpoint sealing. No separate posttraining replay
or overwrite mode is provided. CPU fixtures check lifecycle ordering, fixed-head
configuration copying, exception cleanup and clock invariants; the actual
OmegaConf factory contract runs in the existing server CPU environment. These
checks are engineering evidence, not benchmark or training-improvement results.

```bash
python examples/skill_self_evolve/tests/test_quiescent_eval.py -q
```
