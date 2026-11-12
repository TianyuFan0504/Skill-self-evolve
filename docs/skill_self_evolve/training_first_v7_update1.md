# Training-first v7: first actual update

The run `alfworld_joint_full64_v7_20261112` completed its first actual global
optimizer update under source `0a2785dfbfa836d6e21d2039960c270413ce71ce`.
Uni-Agent and native Codex generated original ALFWorld training episodes
against local Qwen3.5-4B. The next training batch has started. This establishes
real training execution, not completed training or held-out improvement.

Per the user's priority, both complete initial official evaluation and initial
development validation are disabled. Final official evaluation remains
enabled. The immutable base checkpoint and fingerprint remain available for a
later matched base-versus-trained comparison under the same inference setup.

## Actual first-update observations

| Quantity | Observation |
| --- | ---: |
| Original task prompts | 32 |
| Responses per prompt | 4 |
| Logical training episodes | 128 |
| Environment wins | 82 |
| Training reward / accuracy | 0.640625 |
| Actor gradient norm | 1.1792855263 |
| Solver / joint loss | 0.0533354972 |
| Curate loss | 0 |
| Mature replay candidates | 0 |
| Applied optimizer count | 1 |
| Observed EMA version | 1 |

Use `train/reward` and `train/accuracy` for the logical episode denominator.
The separate `critic/rewards/mean` value, 0.5857142806, weights emitted prefix
rows differently and is not the 128-episode success rate. The materialized
batch contains 140 real TITO rows plus 116 masked computational padding rows;
padding does not represent additional tasks or episodes.

The four solver ranks' actual optimizer post-hook clocks, controller
acknowledgment, committed post-update record and metric row agree. Private
engine and Task results reconcile with all 128 original training labels.
There were no infrastructure/isolation failures admitted as ordinary reward
zeros. One genuine zero-action 900-second budget failure independently passed
the shared private deadline/backend/UID/archive verifier, retaining its raw
Native timeout fields.

## Curator and delayed replay

The observed curator policy is `ema:1`, with minimum and maximum weight version
one. Its original 5,523 prompt tokens and 113 generated action tokens, action
mask entries and processed behavior likelihoods match persistent replay
exactly, including hashes and immutable references. All behavior probabilities
are finite and nonpositive. The same record remained readable after the next
batch had created 81 real attempts.

This first record is pending its later task exposures. No delayed utility,
positive admission or nonzero delayed curate gradient is inferred from its
existence. Zero curate loss at update one is expected; mature replay requires
the configured delay and independent future tasks.

## Retained evidence and limits

Paths are relative to
`/root/nas_chaohuang/skill_training/runs/alfworld_joint_full64_v7_20261112`.

| Artifact | SHA256 |
| --- | --- |
| `diagnostics/independent_first_actual_update_1794466866994023027.json` | `02ce3c805dc6fb1615a0a5280be2b0744c732268acb6be6d4626869bb6116186` |
| `diagnostics/independent_EMA1_curate_retention_1794467031899875848.json` | `9370a9b0ab493ef45a50d4620b27fe0fed56176a0c0f485f360c7569281259fe` |
| `diagnostics/independent_actual_budget900_1794466601133757886.json` | `c3472d5795ca0c9a24a3899e1f8f1a83ca2b55d244829362fa9de5e490a4be2e` |

Original trajectories, Native events, private results, replay payloads and
controller records remain retained. `metrics.jsonl`, `metrics.csv`,
`reward_accuracy.png` and `reward_accuracy.svg` record the actual observations.
A single plotted training point is not a learning trend.

No complete joint checkpoint at update one is claimed. The running code saves
at the configured update-eight boundaries and at the final update; it exposes
no supported external on-demand complete joint-save request. Calling an actor
weight-save RPC alone would not produce a consistent resumable joint state.

Training-batch success cannot be compared with an earlier run's test score to
claim improvement. Full training and a matched held-out evaluation remain
required for the requested reward/accuracy gain.
