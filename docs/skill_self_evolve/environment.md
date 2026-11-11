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
