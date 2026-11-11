# Isolating the vLLM backend sampling stall

The `alfworld_joint_full64_v2_20261112` initial development run lost its
GPU-3 engine at 07:34:58 on 2026-11-12. The first retained error is a worker
reply timeout: `step_with_batch_queue -> sample_tokens -> response_mq.dequeue`
in installed vLLM 0.24.0. It is not an observed OOM exception. The failed
engine subsequently caused native episode errors; those results are not a
completed official baseline or a training result.

## Evidence and limits

The installed `multiproc_executor.py` uses
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` for both model execution and sampling.
Its installed default is 300 seconds; the dead worker's environment override
was not captured. The shared-memory stack is where the coordinator waited for
a reply, not proof that shared memory was exhausted.

During the approximately five-minute stall, recorded GPU-3 utilization stayed
at 100%, resident memory stayed at 53,681 MiB, and power was about 116–119 W.
Other replicas continued generating. Host load was 22–36 against a 104-CPU
quota, `/tmp` had about 12 GiB free, and host cgroup OOM counters were zero.
The worker's last JIT warning was at 07:14:42, well before the timeout. Eager
execution had already disabled Torch compilation and CUDA graphs. These facts
do not establish whether a device kernel, stream dependency or asynchronous
output path stalled; the worker's contemporaneous stack is unavailable.

## One supported configuration isolation

The recipe now sets:

```yaml
actor_rollout_ref:
  rollout:
    mode: async
    engine_kwargs:
      vllm:
        async_scheduling: false
```

The official verl server forwards `engine_kwargs.vllm` through its CLI builder
to `AsyncEngineArgs`. The actual pinned parser accepts the explicit negative
boolean flag. In vLLM 0.24.0, TP1/PP1 backend scheduling uses two concurrent
batches when this setting is true and one when false. Disabling it removes
the observed backend batch/output overlap while retaining the asynchronous
HTTP API, Uni-Agent, native Codex, all task/token/action budgets, greedy
evaluation, prefix caching and the original sampling settings.

The 300-second timeout is unchanged. No package, model, cache strategy or
inference backend is replaced. This is an isolation candidate, not a proven
root cause or a claimed stability fix. Greedy dispatch uses Torch `argmax`,
so toggling the separate top-k/top-p sampler alone does not change that branch.

Two CPU tests exercise the real CLI parser and the installed backend queue
property without creating an engine or CUDA/Ray state. A fresh, full native
development/official-evaluation run must establish actual sustained behavior;
CPU checks do not demonstrate rollout stability, skill benefit or learning.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 /opt/conda/bin/python \
  tests/experimental/skill_harness/test_vllm_scheduling_cli.py -v
```

The [pinned vLLM configuration source](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/config/vllm.py)
defines the scheduling switch. The project also has a
[Qwen3.5 concurrent-hang report](https://github.com/vllm-project/vllm/issues/37729),
but its versions and graph/precision conditions differ, so it is supporting
context rather than identification of this failure. Hybrid/Mamba prefix caching
remains a secondary hypothesis; it is deliberately unchanged in this cut.
