# Deferred base versus trained-final comparison

This offline command compares two **already completed** official evaluations.
It never launches evaluation, imports a model runtime, uses a GPU, loads a
checkpoint, changes a skill library or rewrites a raw artifact. It does not
change training's priority or enable a skipped initial evaluation.

Run it after the separately measured untouched-base reference and the trained
final stage both exist. This does not run the deferred baseline now:

```bash
python examples/skill_self_evolve/scripts/compare_official_runs.py \
  --base-stage /path/to/fresh-base-run/evaluation/initial \
  --trained-stage /path/to/trained-run/evaluation/final \
  --output /path/to/new-comparison.json
```

The output must be new. Exit status is zero for a supported complete comparison,
including a complete result showing no improvement; it is two for missing,
invalid, changed-during-read or unsupported evidence. Inspect
`comparison_supported` and `improvement_supported` separately. The base remains
honestly labeled as a deferred reference in its own run; nothing is transplanted
or backdated into the training run's skipped initial directory.

## What is checked

- Exact official schema, isolated evaluation and complete indexed case audits
  for **140 seen and 134 unseen games**. Their exact dump pointers, UID/session/
  attempt identities and all prefix scores must reconcile with original private
  engine outcomes. Prefix rows never increase the case denominator.
- Original catalog, official game files and per-game initial worlds. The two
  official parquet files must agree in full row content and order except for
  explicit Task `metadata.run_dir` bindings. Unused base train/dev preparation
  and training budgets need not match; unknown official protocol fields remain
  unsupported. No arbitrary root replacement is applied inside prompts.
- The actual Task configuration's two 900-second limits, private per-attempt
  budget registrations, 50-action environment capacity, 4,096/32,768/36,864
  token capacities, seed 42, greedy n=1, Triton GDN, scheduler settings and
  configured 4+2+2 topology. Backend evidence has the qualified tiers below.
- Native binary/wrapper/catalog/instruction bytes and recorded runtime package
  versions. Tokenizer bytes and the untouched base HF directory are hashed
  without loading a model; the default expected base fingerprint is the pinned
  Qwen3.5-4B fingerprint `670423a20133031174a5c4769f8b5aa9cd4ce3f541b503d0cd522dde72173a22`.
  Trained weights are expected to differ: their identity uses the retained
  served/controller policy version, not equality with base weight-file hashes.
- Recorded source cuts are compared through local Git's read-only runtime blob
  inventory, with matching dirty-diff and adapter identities. Different commit
  labels caused only by offline scripts/docs can pass if runtime bytes match.
  Missing historical source or changed runtime code remains unsupported.
- Actual sampled numeric Gateway trajectories, confirmed backend/UID/archive
  closure, and original errors. Proven budget failures use the existing shared
  strict verifier, including the zero-action/absent-trace case; failures are not
  manufactured from missing records.
- Base policy tags exactly `[0]`, trained tags exactly `[64]`, corresponding
  controller clocks, and an empty genesis base library. Original Task messages
  are reconstructed from the parquet and initial state. Any trained prompt
  additions must reproduce the recorded versioned skill artifacts and exact
  injection format. No raw prompt or token array is normalized or rewritten.

## Explicit differences allowed

The implementation's `RUN_PATHS` table lists exact run-local data/output/store/
service/task-config fields. Only those paths may bind to each stage's own root.
Run names, derived temporary directories, and the declared source-root/
`PYTHONPATH` prefix may differ through their checked bindings. Native wrapper,
instructions and model-catalog paths may differ only after validating their
referenced bytes. Unknown configuration/protocol changes are rejected.

The base must execute zero optimizer iterations, selected by zero training steps
or zero epochs, with resume disabled. Its unused optimizer/GRPO/curate budgets
need not equal the trained run's budgets. Known training-only sections and fields
are enumerated in the script; unknown inference changes are still rejected.
Source-relative custom sampler paths are validated against the recorded source
root. Generation settings remain matched. The intended treatment is policy
version 0 versus 64 and an empty base library versus trained library snapshots.

## Restored final skill-store binding

A resumed trainer uses `restored_state/step_<n>_<uuid>/skill_store`, while its
original training manifest still correctly records the original preparation.
The comparer never repairs that manifest or guesses a newer store directory.
It reads the explicit `evaluation/final/lifecycle.json` receipt and its three
file-hashed references to `checkpoint_seal`, `evaluation_entry` and
`evaluation_complete` records in one `evaluation_lifecycle/driver_<uuid>`
directory. Missing receipts on restored/external-driver runs are unsupported;
untouched initial baselines do not require a lifecycle receipt. Original fresh
final stages without restore/driver evidence retain their original binding.

The receipt must bind the original manifest hash, published examples driver
Git blob and file hash, unchanged enumerated runtime Git blobs, actual policy
64 and one sealed main snapshot. The referenced `global_step_64` COMPLETE
manifest, every retained checkpoint file hash, four EMA64 acknowledgements and
saved controller clock are checked without loading any checkpoint tensor.
This adds checkpoint-file reading cost to the offline comparison.

Final skill snapshots and artifacts are then read from that exact sealed
restored store. Its head must agree with the COMPLETE checkpoint copy, and all
274 final Task contexts must use the same snapshot. Contradictory paths, heads,
clocks, driver identities or receipt hashes remain unsupported. No `latest`
pointer, directory-name heuristic or arbitrary run-path replacement selects
the store. Original training configuration and raw artifacts remain unchanged.

## Backend evidence tiers

A genuine owned vLLM Worker kernel-selection line is direct log evidence. When
production uses `VLLM_LOGGING_LEVEL=WARN`, its INFO selection line is legitimately
absent. The alternative tier requires a complete owned `vLLMHttpServer` serve
argument list, an exact `--gdn-prefill-backend`/`--gdn_prefill_backend` value of
`triton`, matching recorded configuration/package identity and the reviewed
installed vLLM 0.24.0 selector function. The selector is parsed and hashed; it
is not imported or executed. Its source identity is recorded in the report.

That tier sets `kernel_log_observed=false` and explicitly labels the active
backend as inferred. It does not manufacture a per-call kernel trace. Missing,
incomplete, unowned, ambiguous or mismatched CLI evidence, or an unknown selector
implementation, remains unsupported. Actual raw log files are never changed.

## Scores and limits

`comparisons` reports the two splits; `pooled` weights all 274 original games
equally. Mean reward and accuracy are both `engine_wins / games` for this binary
benchmark. A positive improvement requires more trained pooled wins after every
comparison check passes. Training-batch curves and forged/copied aggregate
metrics cannot establish this result. Auxiliary metrics such as turn counts are
not incorrectly compared to the reward mean.

This establishes an empirical comparison of the final joint system. It does
not separate parameter learning from skill memory, establish statistical
significance, or independently attest every in-memory operator. Weight tags are
the exact aggregate tags retained by official case audits; saved numeric
metadata does not contain independent per-case weight spans. Existing training
and lineage audits remain separate responsibilities. A source/trainer cursor of
1 for base or 65 for final can correctly coexist with actual served policy
versions 0 and 64; cursor values are not optimizer counters.

Use the existing training Python with PyArrow installed. The tool only imports
the pure file-hashing/protocol/store utilities by source path; it performs no
package installation or network access. It also requires the retained source
repositories, HF files, public Native runtime assets, raw case artifacts and
archives to remain readable.

The focused CPU fixture uses explicit JSON-parquet parser-boundary injection;
its case/proof/context/Git/numeric checks are real. These synthetic tests are
engineering evidence, not actual ALFWorld results:

```bash
python examples/skill_self_evolve/tests/test_compare_official_runs.py -q
```
