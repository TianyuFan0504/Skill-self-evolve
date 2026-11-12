# Isolated microbatch measurements

`scripts/benchmark_microbatch.py` measures the installed Qwen3.5-4B FSDP2
model, fused output head, FlashAttention path and production PPO loss on a
fixed set of 16 distinct archived Native trajectory segments. Tokens and action
masks come directly from the archived NPZ files; model inputs are never
replaced by newly tokenized text. The selected corpus spans the observed
step 4 length distribution, including its longest segment.

The actor and reference engines coexist on each four-GPU group. One
zero-learning-rate, zero-gradient diagnostic optimizer step creates resident
AdamW states without updating model weights. The benchmark does not resume
the experiment's optimizer, create rollouts, assign task rewards or modify a
training checkpoint. Its `training_optimizer_updates` field stays zero.

The actual processor and official position builder derive the text position
planes. A public `nested_tensor_from_jagged(..., jagged_dim=2)` packing adapter
preserves their layout. The adapter calls the official engine's `forward_step`,
PPO loss, backward, global normalization and output aggregation. The active
training profile uses `no_sync=False`; the diagnostic retains this setting.
This deliberately measures the model kernels independently of the
TransferQueue's tensor serialization/chunking path. It is not validation of
that whole training path. An actual resumed update remains necessary before
treating a candidate as proven in the complete harness.

The actor compares likelihood microbatches 1/2/4 and forward/backward
microbatches 1/2. The reference compares likelihood microbatches 1/2/4. Every
variant uses the same tokens, masks, weights and global diagnostic batch.
Training uses explicit fixed +/-0.5 diagnostic advantages solely to probe the
PPO kernel and its gradients; they are not benchmark reward estimates.

## Commands

Run only after confirming the selected GPUs are idle. Use a fresh output
directory for every invocation. The `--` separator before the script path is
required because torchrun otherwise interprets the script's `--run` option.

```bash
cd /workspace/verl-0.9.1
export VERL_SOURCE_ROOT=/workspace/verl-0.9.1
export PYTHONPATH=/workspace/verl-0.9.1
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -- \
  examples/skill_self_evolve/scripts/benchmark_microbatch.py \
  --run /path/to/archived/run --output /path/to/fresh/diagnostic/actor \
  --role actor --repeats 2

CUDA_VISIBLE_DEVICES=4,5,6,7 /opt/conda/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -- \
  examples/skill_self_evolve/scripts/benchmark_microbatch.py \
  --run /path/to/archived/run --output /path/to/fresh/diagnostic/reference \
  --role reference --repeats 2

/opt/conda/bin/python examples/skill_self_evolve/scripts/summarize_microbatch.py \
  /path/to/fresh/diagnostic
```

## Reading results

Each rank records phase, elapsed time, allocated/reserved peaks and numeric
comparisons. The summary uses the slowest rank for each measured repeat;
aggregate throughput divides the fixed global unpadded token count by that
time. It retains both repetitions, their median and the last repeat rather
than hiding first-shape warmup costs. Logprob comparisons cover authored action
tokens. Training comparisons cover all local parameter-gradient shards and
combine their error norms across ranks.

A failed or incomplete rank prevents an overall passing summary. An OOM is a
failed candidate and must not be treated as a usable configuration. Passing
this diagnostic establishes numerical and memory behavior for this fixed
corpus; it does not prove heldout accuracy gains or capacity for every future
batch. vLLM is absent from the diagnostic's resident allocation, so the real
colocated rollout/learner lifecycle needs its own final confirmation.

## Recorded deployment measurement, 2026-11-12

The fixed corpus contains 247,704 unpadded tokens, with segment lengths
3,273–34,215. Results below use the slowest of four ranks and the maximum
memory peak across ranks/repetitions. The first measured full-length repeat
follows a 128-token warmup; it is not a fresh-process cold start.

| Operation | Microbatch/GPU | First repeat, s | Second repeat, s | Peak allocated, GiB | Peak reserved, GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actor logprob | 1 | 2.942 | 2.917 | 18.70 | 21.35 |
| Actor logprob | 2 | 12.633 | 3.447 | 23.33 | 28.62 |
| Actor logprob | 4 | 14.139 | 4.638 | 32.28 | 41.31 |
| Reference logprob | 1 | 3.010 | 2.988 | 18.70 | 21.97 |
| Reference logprob | 2 | 10.997 | 3.474 | 23.33 | 29.24 |
| Reference logprob | 4 | 13.673 | 4.668 | 32.28 | 41.93 |
| PPO forward/backward | 1 | 15.751 | 11.755 | 33.44 | 41.59 |
| PPO forward/backward | 2 | 23.856 | 12.854 | 46.59 | 64.55 |

All candidates fit without OOM. Authored-token logprobs were bit-identical.
The full-gradient relative L2 error for training microbatch 2 was 0.01582, with
cosine 0.999875 and loss difference 0; repeating microbatch 1 itself produced
relative L2 variation 0.01533. All recorded numeric checks passed.

Increasing microbatch size reduced throughput on this variable-length corpus.
Microbatches 2/4 increase dense padded input slots to 1.265/1.855 times the
microbatch 1 count. That is a
measured input-size explanation, not a separate kernel-profiler attribution.
Retain training/actor-logprob/reference-logprob microbatches 1/1/1 for this
profile. Larger batches fitting memory is insufficient evidence of a speedup.

The complete raw rank reports, launch identity and release receipt are under
`/root/nas_chaohuang/skill_training/diagnostics/microbatch_1794488334360764545`.
The combined summary SHA256 is
`0d90a52d58e5fb52120e3181d4fe9fbcc52cf2ae573d6f451c241c730626e13c`.
All eight rank reports completed, both torchrun parents and the rank processes
exited, and the post-run GPU process query was empty. These are isolated kernel measurements,
not evidence of completed training or accuracy improvement.
