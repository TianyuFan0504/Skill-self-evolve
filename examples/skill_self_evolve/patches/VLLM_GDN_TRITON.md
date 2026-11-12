# Qwen3.5 GDN prefill backend isolation

The formal v5 run completed its full initial seen partition, then suffered a
real nonterminal backend failure during unseen evaluation. EngineCore 968352,
worker 968407 and HTTP server 966291 on physical GPU 3 stopped returning an
`execute_model` worker response. The pinned vLLM 0.24.0 watchdog raised its
default 300-second RPC timeout at 13:07:15 on 2026-11-12. Later EngineDead,
local HTTP 500 and Native high-demand messages are downstream symptoms.

This is not a 900-second task-budget failure. No optimizer update had occurred.
The affected outcomes and incomplete unseen baseline remain infrastructure
failures, without score-zero relabeling or reuse in a fresh baseline.

## Supported change

The common recipe sets:

```yaml
actor_rollout_ref:
  rollout:
    engine_kwargs:
      vllm:
        async_scheduling: false
        gdn_prefill_backend: triton
```

The installed H100 path selected FlashInfer GDN prefill under `auto`, despite
using FA3 for ordinary attention. The failing worker explicitly logged that
FlashInfer GDN path. Installed FlashInfer is 0.6.12. The installed vLLM
`AsyncEngineArgs` supports `gdn_prefill_backend=triton`, and verl forwards
that engine option without an upstream kernel patch or package replacement.

This change isolates the GDN prefill implementation. Prefix caching, chunked
prefill, asynchronous verl API, eager execution, the 300-second backend
watchdog, 900-second episode budget, model, token capacity and rollout
concurrency are retained. FlashInfer's random sampler was already excluded
by the installed `processed_logprobs` path; greedy evaluation also bypasses
random sampling. Adding a sampler-disable flag would not isolate an active
path in this failure.

## Evidence and limits

Original logs, first exception, failed case and request correlation are under
the v5 run's `diagnostics/independent_backend_failure_1794460183576949836/`.

| Artifact | SHA256 |
| --- | --- |
| `summary.json` | `2dda11351fa58b4c73be749bd9d45c23cdb8922d4ae80dcf7668fecf95f1a2a3` |
| `scheduler-request-correlation.json` | `0ea189f9dcc3d835369d357020e9161f6984aa23ee06402d1304e0e88d3e1ae1` |
| `installed-gdn-backend-and-scheduler.json` | `54f12d4b92c5b5224824792bc19327a72f6c1a033ec059bba08df5ff00270536` |

The failed scheduler step contained seven new requests and eight cached
requests, with 3,969 scheduled tokens. Six new prefill segments had length 528
and one had length 368. Their nonalignment with 128 is an observation, not a
proof of a specific kernel defect. Cgroup OOM counters were zero and no CUDA
OOM line was observed. Host kernel logs were inaccessible.

The official [Qwen recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)
labels Mamba align-mode prefix caching experimental. The upstream
[concurrent GDN hang discussion](https://github.com/vllm-project/vllm/issues/37729)
includes related reports and Blackwell-specific findings. Those reports do
not establish the root cause of this H100 failure. Triton is an isolation
candidate; sustained native rollout and full training must establish runtime
acceptance.

The installed failure dump anonymizes token IDs, retaining lengths only. The
exact failed prompts and complete prefix-cache state therefore cannot be
replayed from that dump. Tests using original retained initial inputs or
prefixes of clean numeric trajectories must be labeled context stress, not
exact crash reproduction, benchmark reward or proof of learning.

A fresh complete initial/final evaluation under the changed backend is
required. The earlier seen score remains historical evidence under its own
configuration.

## Bounded actual GPU validation

`backend_triton_context_stress_20261112` executed on the same physical GPU 3
as the failed v5 worker. Seven retained actual initial inputs and 20 mixed
rounds of 32 derived contexts completed: **647 requests, 8,854 output tokens**,
with aligned finite likelihoods and no request/deadline failure. The contexts
exercise cache priming and growth up to 23,533 tokens; 645 requests reported
cache hits. The actual worker log states that Triton/FLA GDN was selected.
All owned Ray/GPU resources were released.

The raw driver's final log matcher incorrectly required one particular field
name on the same line as the backend value. It wrote `observed=false` and
raised a final matcher exception. Those raw artifacts are preserved. A
separate independent review checked the genuine worker-selection line, actual
adjacent CLI argument/value lines, all original outputs and hashes, and
cleanup. No successful raw driver exit is claimed.

The review under the diagnostic run is
`diagnostics/independent_stress_review_1794461341678889257.json`, SHA256
`c3e9404effb89321b84f7e7562d2f65c9f2982c3bfdbf21a81ca6fbea3531e4a`.
This is bounded backend evidence using 16-token diagnostic outputs, not a
replay of the failed cache state or proof that full Native training is fixed.
