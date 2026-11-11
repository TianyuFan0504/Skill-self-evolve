# Real native paired-input control

The focused diagnostic `alfworld_native_pair_input_v5_20261112` completed with
exit code 0 using the official frozen PolicyRoute, Uni-Agent Gateway, native
Codex 0.167.0 and the original Qwen3.5-4B checkpoint. It used one GPU, no learner
or optimizer, and created no utility label or skill admission.

The input was the actual archived v4 S0 skill and its original later training
task B, copied with file/tree hashes into a new private run. Neither original
task data nor the archived source SkillStore was modified. The run used the
same full context and environment-action budgets as the training recipe.

## Observed model input

The new collector gate reported `observed`. An independent inspection decoded
the actual sampled first-request prompt IDs and confirmed:

- With-skill prompt: 5411 tokens; without-skill prompt: 5223 tokens.
- The 547-character S0 block appeared only in the with-skill prompt.
- Directly removing that verified block left byte-identical UTF-8 model inputs.
- No automatic native Skills/Skill-roots block or actual Codex-home path was
  present. No HOME/path/time normalization was used to obtain equality.
- Frozen model/lifecycle, private initial world state and common empty cwd
  matched. Separate fresh homes and all four archive hashes were verified.
- Both native exits were 0, with confirmed matching backend drain receipts.

The raw outcomes were both failure: 44 actions without S0 and 33 actions with
S0. This proves input control and loading, not skill benefit, a curate gradient,
or a learning improvement. Frozen evaluation intentionally did not compute
behavior logprobs; the separate solver/curate training captures retain theirs.

## Retained evidence

Run root:

`/root/nas_chaohuang/skill_training/runs/alfworld_native_pair_input_v5_20261112`

The immutable collector input audit has SHA-256:

`13967f5ddedbd3c16bc8ac0cf42316bc52cea889c7d60b9c3d23e7bb609099b3`

The independent token-level audit is
`diagnostics/independent_native_pair_input_1794434890795799856.json`, SHA-256:

`b71f42ac94c4f76940f891f711887d201676be93a341c32543f3f7c079dbb9ee`

`result.json`, raw branch captures, Task/engine traces, context manifests and
Ray logs remain available. `cleanup.json` confirms Gateway/local-Ray shutdown,
the owned environment service stopped, no cleanup errors, and no selected-GPU
compute process. A subsequent host inventory found all eight GPUs released.

## Formal experiment transition

Only after this control check completed did the 64-update joint run start.
Its predeclared plan uses full train/dev data, initial/final official 140+134
tests, two distinct future-game exposures per credit estimate and a minimum
two-step delay. The final checkpoint is fixed in advance as update 64.
Historical integration diagnostics are retained and are not silently resumed
or reused as corrected causal training labels. Actual reward/accuracy gains
remain a requirement to be measured in that formal experiment.
