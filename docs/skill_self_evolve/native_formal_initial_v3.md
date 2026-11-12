# Complete official initial baseline: formal run v3

The fresh run `alfworld_joint_full64_v3_20261112` completed all 274 official
initial evaluation games through Uni-Agent, native Codex 0.167.0 and the local
Qwen3.5-4B model. Every observed model version was zero. The retained trainer
cursor is one, which is not an optimizer update count.

## Source and predeclared protocol

The executable source cut is
`6a8f0e797ad3db2a66d845ec381b9227497bdfc6`. It includes the supported
[backend scheduling isolation](vllm_async_sampling_stall.md) and the independent
[targeted abort compatibility repair](../../examples/skill_self_evolve/patches/VLLM_TARGETED_ABORT.md).
The asynchronous verl API remains enabled; the pinned vLLM backend uses
`async_scheduling=false`. The existing timeout, prefix cache strategy and
36,864-token capacity are retained.

The private run root is:

```text
/root/nas_chaohuang/skill_training/runs/alfworld_joint_full64_v3_20261112
```

`experiment_plan.json` was written before launch. Original ALFWorld data is
split into 3,154 training games, 399 train-derived development games, 140
official seen games and 134 official unseen games. Catalog digest:
`807ca811954c24bc78ae2b5d92ae0662dab163f5d6aedad0b859d873f11d9b0b`.
No previous model, optimizer, EMA, replay, library or partial baseline was
imported.

The run reserves four solver/FSDP GPUs, two EMA-curator GPUs and two frozen
credit GPUs. Training uses 32 prompts, four responses per prompt and 64
concurrent native agents. The predeclared endpoint is update 64, with all 399
development games and a checkpoint every eight updates. Official final
evaluation uses that endpoint without test-based checkpoint selection.

`trainer.val_before_train=false` skips the redundant development pass before
the training hook. The hook still creates the dedicated routes and performs
the complete official initial evaluation before any optimizer update. Both
official stages use greedy `n=1`; test games cannot update the optimizer,
curator, credit ledger or skill library. Delayed training credit requires two
distinct future training games, a minimum delay of two steps and up to four
paired probes per step.

## Initial results

| Partition | Games | Environment wins | Mean reward | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Seen | 140 | 65 | 0.4642857143 | 46.42857143% |
| Unseen | 134 | 68 | 0.5074626866 | 50.74626866% |
| Pooled | 274 | 133 | 0.4854014599 | 48.54014599% |

Reward is the environment's binary success result, so its mean equals
accuracy. The pooled value weights games equally; it is not the unweighted
mean of the two partition means.

Independent reviews read each original private engine result, compare the
exact game-ID sets against the catalog and validate the numeric trajectories.
Seen has 157 prefix trajectories and unseen has 139, totaling 296; prefixes
are not additional benchmark games. All token IDs, masks and behavior
probability arrays align, with finite nonpositive probabilities on authored
tokens. Both partitions have zero unexpected native errors and zero
unconfirmed backends.

## Interpreting native interruptions

The retained `native_error_cases` counters are 127 seen and 109 unseen. Every
one is a verified host/backend interruption after an absorbing environment
outcome, with private state hashes and exact backend bindings preserved:

| Exit category | Seen | Unseen |
| --- | ---: | ---: |
| Successful terminal state, host stopped CLI | 65 | 68 |
| Failed at the 50-step limit, host stopped CLI | 62 | 41 |
| Native normal exit without success | 13 | 25 |

The original interruptions and exit codes remain in the artifacts. They are
not unexpected engine failures, and all 274 games remain in the denominator.
One normally completed unsuccessful episode took 792.89 seconds; long model
generations remain a measured latency cost under the full token budget.

## Retained evidence

All paths below are relative to the private run root.

| Artifact | SHA256 |
| --- | --- |
| `evaluation/initial/summary.json` | `dd63b47efc15e88c0257e4f80d94865b6ff3e0c061dfda37848688284f42b853` |
| `diagnostics/config_independent_review_1794442163942794829.json` | `c684e637c043d96df4cfa164e4757734e7112e59fd8d32a1ccd7872b598ad8d0` |
| `diagnostics/first_native_archive_review_1794442241590075263.json` | `37a456ff891a5cee5a414e52fdcfe3be2aa699982d70e382f3eef93ec99167e9` |
| `diagnostics/independent_initial_seen140_1794442596443000922.json` | `a85e274a015f88c44def4946068b26e3b3bca3f58b11f31d6fee0b726fd27516` |
| `diagnostics/independent_initial_unseen134_1794443333163248587.json` | `dacec20abf7c9ba62d448ff8efacb2093790668fe6e1174a10db9a1e0442d553` |
| `diagnostics/independent_initial_full274_1794443439774707010.json` | `fa1059b52224119fd04a3d3c1e46baad8bd5360737ee2c7faf146ab76d493fef` |
| `diagnostics/first_optimizer_EMA_curate_independent_1794444459932139966.json` | `7b60b643c22ce83744df4c3a9e76af2b3dc1a79734a1c961483772f097da090c` |

The summary indexes the two full per-case audit sidecars. Original environment
interactions, native events, raw stderr, numeric TITO, masks and behavior
probabilities remain under `episodes/` and `agent_logs/`. The sampled native
archive independently matched all 88 source members, including every file's
content hash; its original HOME, catalog and working directory were removed
after verified archival. `/tmp` returned to approximately 12.31 GiB free at
the end of initial evaluation. Resource samples remain in `resources.jsonl`.
The combined independent review also observed optimizer zero, no curator
collections, pending windows or applied updates, an empty main library and no
persisted replay/ledger objects at the initial cut.

## First actual training update

After initial evaluation, the same driver began its first original training
batch. An actual observation found 64 active attempts for 16 distinct training
games, exactly four per game, confirming restoration of training fanout after
greedy evaluation. The optimizer counter was still zero at that observation.

The first complete 32-prompt/four-response batch subsequently finished its real
online optimizer update. The retained `metrics.jsonl` training row reports:

| Quantity | Actual observation |
| --- | ---: |
| Training episodes | 128 |
| Successful training episodes | 77 |
| Training mean reward / accuracy | 0.6015625 |
| Actor gradient norm | 1.2116317749023438 |
| Learning rate | 0.000001 |
| Solver / joint loss | 0.010764850303530693 |
| Curate loss | 0 |
| Applied optimizer count | 1 |
| Newly retained curate actions / windows | 1 / 1 |
| Curate semantic rejections / censored jobs | 0 / 0 |
| Training infrastructure error rate | 0 |

The controller records one applied update, clears the prepared update and
persists the new curate collection. Its replay candidate count is zero at this
first update, so the nonzero gradient is from the solver branch. The newly
sampled curate action remains pending its two future exposures; a retained
action is not a nonzero delayed curate gradient. The same driver has begun its
second original training batch.

An independent review confirmed the actual curator observed parameter version
one and policy `EMA:1`. Its 4,119-token prompt and 136 action tokens, action
mask entries and processed behavior log probabilities match the original
generated capture exactly. All action probabilities are finite and nonpositive.
The same durable record and ReplayRef remained readable after the next batch
had created 82 real attempts. This is retention evidence, not delayed utility.

## Controlled stop before the second update

The second batch exposed an additional native tool/protocol problem. A solver
created a native subagent, which attempted `view_image` and accessed a bundled
image outside this attempt's own HOME. The target HOME has not been attributed
to a particular other task. This is observed access outside the attempt;
reading another task's benchmark solution is not established.

The gateway recorded 30 request exceptions caused by missing `qwen_vl_utils`
while the world was at approximately steps 22–23. These occurred before the
terminal state, not merely during expected postterminal refusal. The main
trajectory later continued to the 50-step failure state and its final host
stop classification concealed the earlier recovered gateway errors. Both
parent and child numeric trajectories had real action masks and probabilities,
so they could have entered the next update without stronger admission checks.

The verified owned supervisor was stopped while the actual optimizer counter
was still one. No second update was applied. The raw supervisor status remains
`failed`, exit code `-6`; it has not been rewritten as a clean training exit.
The owned environment service was subsequently stopped, and all eight GPU
compute processes were released. Partial second-batch attempts remain partial;
they are not converted to ordinary zero-reward samples. No resumable joint
checkpoint at step one is claimed.

Incident evidence:

- `diagnostics/qwen_vl_utils_nonterminal_incident_1794444795714526521.json`,
  SHA256 `3e97ca361c198d57227263fd2e64201c688cf5eb735a9c9486b52c6bc1e32a8e`.
- `diagnostics/qwen_vl_utils_incident_snapshot_1794444795714526521/snapshot-manifest.json`,
  SHA256 `a4396d4da0a664f337451737b617de80a8da454078ef49ec581a37deb4d111a8`.
- `diagnostics/nonterminal_tool_scope_stop_1794444850679151041.json` records
  the observed counter and the exact supervisor signal.

The next cold run needs a native tool policy matching the ALFWorld text
interface, isolation between attempt files, and rejection of unsupported media
before training fields can discard it. Installing the optional image decoder
alone does not address those protocol issues. A changed tool/isolation policy
requires a fresh initial baseline; this run's scores remain historical evidence
under its original policy.

## Claim boundary

This report establishes a complete initial baseline, verified native trajectory
retention and the first actual online update. The training score of 77/128 is
measured on different games and sampling settings from the official baseline;
it is not an estimate of test improvement. Nonzero delayed curate gradients,
positive skill admission/reuse and final evaluation require their own later
evidence. These observations also do not identify the root cause of the earlier
asynchronous sampling stall.
