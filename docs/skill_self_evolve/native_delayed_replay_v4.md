# Native delayed-replay diagnostic and a causal-control gap

The `alfworld_joint_smoke_v4_20261112` run completed two actual optimizer/EMA
updates. Two native curator generations passed the strict five-field contract.
The first delayed replay entered the second real mixed update, but its recorded
utility was zero. A subsequent audit found an extra native prompt difference.
This run therefore proves the retention/replay plumbing, not causal skill
benefit, a nonzero curate gradient, or improved official test performance.

## Actual generated action and retention

The first collection was
`0bb3c733d0963f8a9dc0535b38e912bb8ccf6e8c18aa49c9a15f592015ab75bc`.
Native Codex emitted unmodified plain JSON with all five required fields and
no tool calls. Gateway weight tags were exactly EMA1. The durable record kept
5524 prompt tokens, 116 action tokens, 116 action-mask entries and 116 original
finite behavior logprobs. The entire host input was 4703 tokens; native additions
were 821 tokens, below the reserved 4096-token allowance. No source truncation
or output repair occurred.

After actual TQ cleanup and a COMPLETE step-1 checkpoint, the active replay and
checkpoint copy still matched the original capture element for element. For
step 2, reconstruction of the official anchor input exactly matched the real
controller fingerprint:

`7c9a1ec579e8468610b003ee0457299e40e495be9441df87da723bed5dfb062e`

The 5640-token input retained the full context. Only the 116 generated curate
tokens carried this role's action/loss mask; solver masks were zero. The same
record and actual credit lease were later acknowledged at optimizer step 2.
Since its advantage was zero, no nonzero curate gradient is inferred.

## Paired evaluation and discovered confound

The earlier source A and later B were distinct original clean-family training
games. Both native branches used the same frozen model/lifecycle, private initial
environment state and controlled empty cwd. Each used a fresh isolated Codex
home; all four workspace/home archives matched their recorded hashes and were
removed after use. Native exits were clean, backend requests were drained, and
both environment outcomes were failure after 31 actions. The original ledger
therefore recorded zero utility and did not admit the skill to the main library.

The target skill's 633-character block was verified inside actual model prompt
tokens, not merely host metadata. However, after accounting for that block,
the model prompts still differed in a native automatic Skills fragment:

`r0 = /tmp/codex-home-<different UUID>/skills/.system`

Consequently the pair did not strictly isolate the target skill. A post-hoc path
normalization is not used to repair this causal gap. The original zero label,
optimizer acknowledgement and raw inputs remain unchanged. Frozen probes did
not compute behavior probabilities because they were evaluation-only; the
solver/curate behavior probabilities required for training were retained.

The next source cut must suppress automatic native Skills instructions and
verify actual native prompt equality outside the target skill before creating
any UsageEvent. Snapshot versions remain host metadata. Missing or confounded
pairs must not become zero labels.

## Other observed results and retained artifacts

Two actual native compactions were observed: a checkpoint-compaction request
generated 488/288 real model tokens, and the following prompt loaded the exact
summary before continuing environment actions. Original IDs and logprobs were
preserved. The raw HTTP parallel-tool flag was not captured and is not attested.

| Update | Episode reward / accuracy | Gradient norm | Peak actor allocation |
| --- | ---: | ---: | ---: |
| 1 | 0.75000 | 1.1541901 | 23.8860 GiB |
| 2 | 0.59375 | 1.2665881 | 32.6644 GiB |

The eight-game dev check after update 2 scored 0.75. The training batches differ,
there is no initial dev point in this integration run, and no official tests
were performed. None of these values establishes learning improvement.

The NAS run root contains raw trajectories, metrics, plots and the COMPLETE
step-2 checkpoint with manifest:

`sha256:cea3eeb604119e6eae1dcfc5c739d7aa5c5c18d799c4a74c386d5794687d9a72`

The operator stopped the diagnostic cut before further potentially confounded
credit. Its original supervisor status remains `failed`, exit `-6`; the stop
record and partial third-batch artifacts are retained. No old labels or model
states are silently migrated into the corrected protocol.

Primary evidence is under the run's `diagnostics/` and the shared
`skill_training/diagnostics/` directory, including
`v4_first_native_paired_cf_audit_1794432905513845732.json` and
`v4_native_compaction_token_proof_1794432406079779783.json`.
