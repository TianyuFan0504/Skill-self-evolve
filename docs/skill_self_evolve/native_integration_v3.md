# Native integration cut: two real updates

The diagnostic run `alfworld_joint_smoke_v3_20261112` completed two actual
online optimizer updates and two EMA updates using Uni-Agent, native Codex
0.167.0, and Qwen3.5-4B. It does not establish skill learning or improved test
performance. No valid curator collection or delayed-credit window was created.

## Run and measurements

Artifacts are retained under:

`/root/nas_chaohuang/skill_training/runs/alfworld_joint_smoke_v3_20261112`

The configuration used four solver GPUs, two EMA-curator GPUs, two frozen-credit
GPUs, eight original prompts per update, four solver samples per prompt, full
36,864-token capacity, static one-sequence training microbatches, and synchronized
gradient accumulation. No task or context budget was shortened to pass memory
checks.

| Observed update | Unique episodes | Episode reward / accuracy | Gradient norm | Peak actor allocation |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32 | 0.6875 | 1.1598594 | 23.8641 GiB |
| 2 | 32 | 0.6250 | 0.9150884 | 33.0439 GiB |

The second batch produced 37 native prefix trajectories and was padded to 64
rows for the stock batch layout. Padding has zero action loss. Its stock
trajectory-weighted `critic/rewards/mean` was 20/37; the separately deduplicated
`train/reward` and `train/accuracy` were both 20/32. The exported reward plot
uses the episode metric. Different training tasks do not constitute a controlled
before/after comparison.

The eight-game development check after update 2 scored 0.625. This integration
run skipped initial validation and did not run official tests. Its single dev
point is not evidence of improvement. Raw JSONL, CSV, PNG and SVG are retained.

## Independently checked native evidence

`diagnostics/terminal_native_first_batch_20261112.json` records paths and hashes
for all 32 first-batch Task/engine results and numeric Gateway trajectories.
All sampled-action behavior logprobs were finite, nonpositive, correctly masked
and aligned with their original tokens. Engine outcomes and recorded rewards
matched.

There were 29 terminal pre-submission refusals. Environment termination to Task
result took 5.034–7.273 seconds, with median 7.094 seconds. Original CLI exits
and errors remain visible. This proves the pre-submission path; no actual
in-flight abort/raw-tail case was observed, so that runtime path remains unproven.

One first-batch episode failed after 16 environment actions because the Responses
adapter rejected `parallel_tool_calls=false`. Its failure, nonterminal state and
actual numeric trajectory remain archived. Pinned Codex source and timing support
compaction as the likely trigger; the rejected HTTP body was not recorded, so
the precise request type is not proven.

## Curator outcome and source transition

Both sampled curator candidates were rejected by the strict output gate. The
first contained Markdown fences and omitted a required field. Its diagnostic is
`diagnostics/curate_native_first_cohort_20261112.json`: the real prompt included
the complete system and final five-field instructions, fit its token budget,
and retained the exact EMA1 tokens/logprobs. The output was not truncated or
repaired. Invalid cohorts produced no fabricated SkillRef, reward or curate loss.

The COMPLETE step-2 checkpoint binds model, optimizer, EMA, library, replay,
ledger, controller and dataloader state. Its manifest ID is:

`sha256:4cb308859a478603f7b2ac23028bddb8b84e2970971c5c3f518e973afd4a7777`

The operator ended the run after retaining that cut to address the two known
issues in a fresh source/template version. The stop request is retained in
`diagnostics/source_cut_stop_request_1794430494864931092.json`. The supervisor's
original terminal state is `failed`, exit `-6`; it is not rewritten as successful
completion. Partial third-batch artifacts are retained. All owned GPU processes
were confirmed gone before any dependency or runtime source transition.
