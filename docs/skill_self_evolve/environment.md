# Training Environment and Offline Relay

## Current instance

The November 12 training instance exposes eight NVIDIA H100 80GB devices,
104 CPU quota cores, and 1600 GiB of container memory. `/dev/shm` is 800 GiB
and counts toward memory use. Large run artifacts and checkpoints belong on
`/root/nas_chaohuang`, not the 49 GiB overlay root filesystem.

The training interpreter is `/opt/conda/bin/python` (Python 3.11.13), with
PyTorch 2.11.0+cu130, vLLM 0.24.0, Transformers 5.10.4, Ray 2.50.1,
TransferQueue 0.1.12, and the editable core at `/workspace/verl-0.9.1`.

Uni-Agent source is provided by the benchmark checkout. Its inference venv
also exposes a different development version of verl. The runtime bootstrap
appends same-ABI dependency packages without executing editable `.pth` files,
pins actual imports to core 0.9.1, and applies the same setup in Ray workers.

## H100 FlashAttention repair

The previous FlashAttention 2.8.3.post2 extension contained native sm89 code
for L40S and PTX 9.2. H100 fell back to PTX JIT, which driver 580 could not
compile. All eight devices passed ordinary PyTorch bf16 matrix backward,
but the original FlashAttention kernel failed.

The exact cached 2.8.3.post2 source was rebuilt offline with the coherent
CUDA 13.2 toolkit, `FLASH_ATTN_CUDA_ARCHS=90`,
`TORCH_CUDA_ARCH_LIST=9.0`, `MAX_JOBS=16`, and `NVCC_THREADS=2`.
Only the same-version FlashAttention wheel was installed with offline uv and
no dependency resolution. PyTorch and vLLM were not replaced.

Preview and fresh-process post-install validation passed with PTX JIT
disabled: fp16/bf16 dense attention at head dimensions 64, 128 and 256, plus
variable-length grouped-query attention matching Qwen3.5's 16 query heads,
four KV heads and head dimension 256. All tested gradients were finite and
nonzero. Maximum bf16 absolute error against fp32 SDPA was 0.00972.

Installed extension SHA256:

```text
10f9001e45f397d05c51041c513f5d3695b2aad7ec428d225372bce414b14d74
```

Original package backups, exact source/wheel, build logs and repair evidence:

```text
/root/nas_chaohuang/skill_training/environment/flash_attn_sm90_20261112_010608/
```

This establishes H100 kernel compatibility, not successful distributed
ALFWorld training. The native Uni-Agent/Codex training run is a separate gate.

## Network boundary

All public downloads and GitHub operations run on the local workstation.
Source archives, wheels and offline server commit bundles move over SSH.
No GitHub credential is copied to the server. Internal loopback services
and Ray/Gateway traffic remain inside the training instance.
# Colocated solver weight synchronization

The pinned synchronous V1 trainer explicitly uses the `naive` checkpoint
manager for its colocated online solver replicas. Keep the recipe's solver
checkpoint backend consistent with that manager. An `nccl` setting here eagerly
initializes an unused optional transport in actor workers and can fail before
rollout when CuPy is absent. The CPU preflight now validates the selected
backend before launching model workers. Dedicated EMA curator transport uses
its own separately configured checkpoint engine and must not reuse this route.

The first native rollout exposed an intermediate public-runtime directory with
mode 0700 under the host's restrictive umask. Runtime installation now makes
only that public launcher directory traversable, and CPU preflight executes
the actual wrapper as `alfworld_eval` to verify access before any model loads.
It does not relax private benchmark, run or administration directories.

Concurrent first-use GPU kernel compilation briefly filled the 49 GiB root
filesystem. New runs put transient compiler files and Ray scratch in the large
`/dev/shm` filesystem. The supervisor records GPU memory, utilization, power,
host load and available disk every 15 seconds in `resources.jsonl`, and copies
owned Ray logs back to the durable NAS run after training exits. Failed attempts
and the append-only training log remain preserved.

## Frozen Qwen3.5 evaluation

The actual first joint launch showed that vLLM 0.24 rejects batch-invariant
execution for Qwen3.5's `GDN_ATTN` backend. Both auxiliary routes therefore keep
`full_determinism=false`. The frozen evaluator still uses immutable verified
weights, temperature zero, and a declared per-request reference seed. Its
protocol and route attestation record these settings. This does not promise
bitwise invariance across different runtime batches.

## Offline plotting dependencies

The base training interpreter originally lacked plotting support. Only six
missing packages were added from workstation-downloaded Linux cp311 wheels:
Matplotlib 3.11.2, contourpy 1.3.3, cycler 0.12.1, fonttools 4.68.0,
kiwisolver 1.5.1, and pyparsing 3.3.3. Installation used offline/no-index/no-deps
mode. All 371 preexisting package versions and metadata hashes were unchanged,
including Torch, vLLM, Transformers and NumPy.

CPU Agg rendering and the unchanged metric exporter passed. PNG/SVG curves use
the exact recorded observations; the failed baseline contains only one initial
development score and therefore supplies no training-improvement evidence.
Wheel, package and rendering evidence is retained at
`/root/nas_chaohuang/skill_training/env/plotting_1794422702454`.

