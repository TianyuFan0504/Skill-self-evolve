# Formal v5: official initial evaluation

The fresh run `alfworld_joint_full64_v5_20261112` uses Uni-Agent, native Codex
0.167.0 and local Qwen3.5-4B under executable source
`d8b3de2dc4a60171d1bbb0ea35d26884b1c54131`. The initial seen partition has
completed and passed independent review. The unseen partition is still
running; a complete pooled initial baseline is not yet claimed.

| Initial partition | Games | Wins | Mean reward | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Seen, complete | 140 | 59 | 0.4214285714 | 42.14285714% |
| Unseen, pending | 134 expected | pending | pending | pending |

Reward is binary environment success, so its mean equals accuracy. No
previous run's baseline, model, optimizer, EMA, skills, replay or ledger is
imported. The actual optimizer and observed model weight versions are zero;
the trainer cursor is one and is not an optimizer update count.

## Seen evidence

The original private engine results contain 59 wins across the exact 140
expected game IDs. Task rewards, the official case audit, parent metrics and
score dumps reconcile with those outcomes. The 152 emitted prefix trajectories
are not 152 benchmark games. Their original numeric token arrays, masks and
finite behavior probabilities passed review; no missing case was removed or
censored into a smaller denominator.

The outcome breakdown is 59 successful terminal episodes, 60 failures at the
real 50-action world limit, and 21 normal Native exits without success. The
production report retains 119 Native bookkeeping-error cases. Independent
Native-log inspection located all 114 raw error events after already observed
world terminal outcomes. No unexplained nonterminal Native error or unconfirmed
backend remains in this partition. Original errors and exit codes are retained.

All 140 UID/backend closures, archive receipts, source-manifest hashes and
removal of original temporary roots were checked. The independent full-seen
review additionally hashed a fixed sample of eight archive files. It does not
claim to have decompressed every archive again; the earlier zero/win review
fully read back all 194 members of its two archives. Production archival
already performs complete readback before deleting sources.

Initial testing did not create curate collections or pending windows. The
library remained its empty genesis snapshot, and replay/credit-ledger state
remained empty. This is an initial evaluation result, not training progress or
evidence of improvement.

## Retained artifacts

The following paths are relative to
`/root/nas_chaohuang/skill_training/runs/alfworld_joint_full64_v5_20261112`.

- `evaluation/initial/test_seen/metrics.json`
- `evaluation/initial/test_seen/case-session-audit-d40d1719ce444ef99bc828ae72dcb091.json`
- `evaluation/initial/test_seen/1.jsonl`
- `diagnostics/independent_full_initial_seen140_1794459634404533408.json`

The independent review SHA256 is
`a9cfca205188a2f43973c9bf8ae36f5284212d053beb03cf7f6118f5aaff6ca6`.
Raw Native events, stderr, numeric captures, private world results and archive
receipts remain under `agent_logs/` and `episodes/`.

## Continuing experiment

The predeclared experiment retains 64 actual updates, 32 prompts with four
responses each, four solver GPUs, two EMA GPUs and two frozen-credit GPUs.
Both official partitions use greedy `n=1`, the full 900-second episode budget
and 36,864-token capacity. The complete 274-game initial stage precedes the
first optimizer update. All 399 development games and a checkpoint are
scheduled every eight updates; final official evaluation uses the predeclared
last update without test-based checkpoint selection.

A completed initial stage and subsequent final-stage comparison are still
required before claiming reward or accuracy improvement.
