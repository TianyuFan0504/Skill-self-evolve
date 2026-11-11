# Official fused Torch training-kernel verification

Result: **PASSED** on physical GPU 0. Diagnostic PID 102510 exited, and all eight GPUs reported 0 MiB at release verification. No optimizer step or new agent rollout was run.

## Archived native input

The input is an existing Uni-Agent/native Codex TITO archive from the failed baseline's step 1:

- Archive: /root/nas_chaohuang/skill_training/runs/alfworld_codex_training_smoke_20261112/agent_logs/step_1/session-sample-7-rollout-1-403d7089bd58435e849d121d3e1038a3/trajectory.npz
- Archive SHA-256: 94427ecb08af445c2b6e4d016a498f18bb9dc616e76bdd1d9395b5e45dfaa4c4
- Exact sequence: 33,322 tokens (5,800 prompt + 27,522 response).
- Actual model-authored action tokens: 4,019; original masks and token IDs were preserved without decoding or re-encoding.
- Loss position p+j-1 predicts archived response token j. Prompt/tool context is visible but excluded from the action loss; the final wrapped label position is excluded.

The original baseline failure and its archives were preserved. This diagnostic is evidence about the training kernel and does not turn the failed baseline into a successful RL update.

## Actual path and results

The offline process used /opt/conda/bin/python (Python 3.11.13), core verl 0.9.1 from /workspace/verl-0.9.1, Torch 2.11.0+cu130, Transformers 5.10.4, and NVIDIA H100 80GB HBM3.

The actual HF class was Qwen3_5ForConditionalGeneration with BF16 parameters. The official apply_monkey_patch selected verl.models.transformers.qwen3_5.forward_with_torch_backend with use_remove_padding=false, use_fused_kernels=true, and fused_kernels_backend=torch. Official FusedLinearForPPO used its default 512-token chunks. Gradient checkpointing was enabled with use_reentrant=false.

| Phase | Runtime | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: |
| Full 33,322-token likelihood + entropy forward | 10.24 s | 17.32 GiB | 19.66 GiB |
| Actual authored-mask backward, cumulative forward/backward peak | 48.47 s | 24.24 GiB | 26.94 GiB |

Full likelihood was finite FP32; entropy was finite BF16. All 723 parameter tensors (4,539,265,536 parameters) had finite gradients. Global gradient L2 norm was 7.04664679.

The stress objective was masked NLL minus 0.001 times masked entropy: NLL 0.33819479, entropy 0.36091554, loss 0.33783388. This deliberately exercises both likelihood and entropy derivatives. It is a diagnostic loss, not a GRPO reward or optimizer update. Backward runtime includes first-use TileLang compilation, retained in probe.log.

## Numerical parity

Only a 128-token prefix of the same real archive was compared with an unfused reference, avoiding a repeated full-vocabulary OOM. Forward used the complete patched model. Backward compared fused and unfused heads with identical actual model hidden states and vocabulary weights.

- Maximum likelihood error: 1.90734863281e-06.
- Maximum BF16 entropy error: 0.
- Hidden-gradient relative L2 error: 2.80412707727e-08.
- Vocabulary-weight-gradient relative L2 error: 1.82355827931e-07.
- Both gradient cosine similarities: 1.

## Integration consequence

The no-remove-padding unfused branch in core verl/workers/engine/fsdp/transformer_impl.py materializes full vocabulary logits and calls entropy_from_logits directly. Its fused branch consumes scalar per-token likelihood/entropy instead. The existing official Qwen3.5 Torch implementation supports the verified fused path, including actual backward; no core kernel edit was needed.

For the V1 worker, model configuration is authoritative: TrainingWorker overwrites engine.use_fused_kernels from model_config. The relevant supported configuration is:

    actor_rollout_ref.model.use_fused_kernels=true
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch

The actor and reference workers both receive the model configuration. The fused path requires a uniform likelihood temperature; this experiment's solver and curator temperatures are both 1.0.

## Limits and retained evidence

This was a single-GPU BF16 HF/kernel test with gradient checkpointing, not distributed FSDP2, Adam optimizer-state allocation, a GRPO step, or a native skill experiment. An actual training retry must still verify those integration boundaries. The selected archive was 33,322 tokens; the separate 36,832-token archive was not tested here.

Machine-readable results: report.json. Exact script snapshot: probe_fused_training.py. Runtime output and compilation evidence: probe.log. Input/source/script digests: evidence.json. No downloads, dependency installs, new rollouts, or existing core/trainer/config modifications were performed for this diagnostic.
