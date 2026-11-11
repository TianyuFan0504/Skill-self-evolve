# First independently checked native learner update

The native Uni-Agent/Codex/Qwen3.5-4B joint run completed one real online
optimizer update and one EMA update. It then failed during episode metric
unwrapping; that failure is retained and was fixed separately. This run is not
evidence of improved reward/accuracy or successful delayed-curate learning.

The COMPLETE checkpoint is under:

`/root/nas_chaohuang/skill_training/runs/alfworld_joint_smoke_20261112/checkpoints/global_step_1`

An independent guarded CPU inspection read the actual rank-0 model, optimizer
and EMA files, without initializing CUDA, Ray or a process group:

- All 723 materialized Adam states had `step=1`, with finite nonzero moment
  examples.
- The EMA payload had initial step 0, last step 1, one update, decay 0.95, and
  physical world size 4.
- Eight named language-model norm tensors were finite. All 256 inspected local
  elements differed between online and EMA tensors. The maximum difference was
  `1.0132789611816406e-6`; the same-weight arithmetic comparison exceeded a
  conservative four-epsilon scaled rounding bound.
- Full model, optimizer and EMA file SHA256 values matched the COMPLETE
  checkpoint manifest. File-hash differences alone were not used as proof of
  a parameter update.

The full numeric proof, named tensors, shapes, dtypes and file/tensor hashes are
retained at:

`/root/nas_chaohuang/skill_training/runs/alfworld_joint_smoke_20261112/diagnostics/optimizer_ema_rank0_proof_20261112.json`

The first native curate response was rejected by the then-current event/schema
gates. Its raw output and behavior probabilities remain archived. Later code
separates verified startup diagnostics from actual tool actions and retains
strict skill schema checks. Full delayed credit, skill loading, mixed-role
updates and complete same-protocol test improvement still require runtime
verification.
