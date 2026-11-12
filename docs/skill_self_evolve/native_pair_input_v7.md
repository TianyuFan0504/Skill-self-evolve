# Native skill-input and lifecycle acceptance: v7

`alfworld_native_pair_input_v7_20261112` completed the original S0/future-B
diagnostic under source `1cdeab74200c80a14aed14b61ff37ccec064cc4b`. Both
Native processes exited normally, the actual inputs differed only by the
assigned skill, and all owned resources were closed. This establishes input
control and lifecycle behavior, not skill benefit or learning.

The original S0 collection is
`0bb3c733d0963f8a9dc0535b38e912bb8ccf6e8c18aa49c9a15f592015ab75bc`.
The original downstream training game is `train_6408c0ae5173d1bf3ac38651`.
Only the diagnostic run paths changed from the archived assignment; no new
task was generated.

## Actual observations

| Branch | Native seconds | World actions | Native exit | Reward / accuracy |
| --- | ---: | ---: | ---: | ---: |
| Without skill | 545.942 | 34 | 0 | 0 / 0 |
| With skill | 821.170 | 44 | 0 | 0 / 0 |

Both retained the private 900-second deadline and ended before it. Their raw
world `done` values remained false. Neither branch exercised actual deadline
exhaustion, deadline rejection or a winning deadline race. The observed
paired score difference is zero; no skill benefit is established.

Independent checks compare the actual admission-zero inputs, containing 1,969
and 2,157 tokens. Replacing only the exact assigned task-user skill block makes
the decoded Native prefixes identical. The original 547-character skill block
and reference are preserved. No HOME, path or time normalization is used.
Both branches have the same original private world state, leased UID/GID,
lease and empty working directory, with different private HOME directories.

The final independent review passed 34 conditions. All **178 archive members**
were read back with matching content hashes. Both attempts closed,
the pair lease released, all owned temporary sources were removed, and the
driver, model, environment service and scratch were absent. There were no
Native errors, media payloads or isolation incidents.

The four frozen-route numeric trajectories have valid token IDs and masks but
retain `response_logprobs=None`. They are not training-admission or probability
retention evidence. The subsequent
`1845622eb806589ca5dfa424c4435044cfd8b36f` repair requests those probabilities
on frozen routes and passed 30 server CPU regressions. Its probabilities do
not enter Native conversation text or tool arguments. This historical v7 run
is not reclassified as a real execution of that repair.

## Retained evidence

The following paths are relative to
`/root/nas_chaohuang/skill_training/runs/alfworld_native_pair_input_v7_20261112`.

| Artifact | SHA256 |
| --- | --- |
| `result.json` | `71bcf8ed08ec1af7d398198daefd392bd79afacb1a844a8e002d8b9e2a706f11` |
| `cleanup.json` | `783b629eeecb4c3506c000f5c9b0f3a244e8fe39586d7d3de38f194e52a9f84e` |
| `diagnostics/independent_actual_first_pair_inputs_1794455102386763990.json` | `b752b43354363977666b56c7457cede8f43da7fa4bb615ab2c611d08980a9738` |
| `diagnostics/independent_budget_clock_closure_1794455957039805750.json` | `6e12769aafeba97cacaf2a6868d28988c7332a52629684f16a6a3a3f5d67de6c` |
| `diagnostics/independent_final_v7_pair_1794456100462007117.json` | `3319b49a24d2e2030f1d9ca1ec706190f6800b289e63327365138c0e0e741e7f` |

Raw captures, engine results, action journals, Native events and complete
archives remain under the run's `credit_probes/` and `episodes/` directories.
The failed v6 evidence and quarantined UID are unchanged.

## Fresh formal experiment

After binding these audits and confirming all GPUs idle,
`alfworld_joint_full64_v4_20261112` launched under code source `1845622e`.
Its supervisor, learner driver and private environment service started. The
predeclared protocol remains 64 updates, 32 prompts with four responses,
four solver GPUs, two EMA GPUs and two frozen-credit GPUs. It runs a fresh
274-game official initial evaluation at optimizer zero, all 399 development
games and a checkpoint every eight updates, and all 274 official final games
at the predeclared final update. No previous baseline or training state is
imported.

The startup checks SHA256 is
`f68f7509a85094d1593a2c9580539fdf0428e26514b753a0b3c50f86d81c57ff`;
the revised prelaunch plan SHA256 is
`4c0c40c19074dfc56ae2c7173e4d52ddbe7fe0c02cb496be402ecd8ad5776dc2`.
Launching the experiment is not a completed training or improvement result.
