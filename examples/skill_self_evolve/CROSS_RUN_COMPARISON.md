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

## Safe setup continuation and the actual inference profile

The safe-setup examples driver records `lifecycle_extension` and the same
file-hashed `initialized_receipt_ref` in all three lifecycle records. The
comparer validates the published driver and dedicated namespace Git blobs,
their current file hashes, the exact `exact-pre-create-no-native-v1` policy,
matching runtime/adapter imports, and the preserved original manifest hash.
Its explicitly referenced source COMPLETE checkpoint4 must still be intact and
compatible with the final checkpoint; no `latest` pointer selects that source.

The final inference signature uses the validated
`effective_config_after_trainer_init`, rather than pretending the immutable
original manifest's 64-session profile was the actual 32-session execution.
Only the known safe runner FQN, `ResilientReplayBuffer` source binding, 32-session
limit, explicit checkpoint4 resume/store/replay/ledger paths, and verified core
constructor normalizations are permitted. The latter comprise single-prompt
refill, computed optimizer schedule horizons, and unused disabled reward-model
placement. All other resolved fields, including model/tokenizer/forward budgets,
algorithm and counterfactual budgets, must equal the original recorded values.
The 900-second and 36,864-token limits are not relaxed.

The untouched base uses the original entrypoint with zero optimizer iterations
and explicit 32-session/safe-runner/sampler overrides, so its original fresh
manifest already records its actual inference profile. It does not need a
fabricated initialized receipt or an additional initial-stage pointer. Its
driver/namespace bytes and inference settings must match the final extension.
A legacy 64-session base is unsupported against the actual safe 32-session
final; unknown or missing extension receipts never fall back to stale settings.
Original quiescent and fresh legacy comparisons remain supported when both
sides use that same original profile. This code launches neither reference nor
final evaluation.

## Spawn boundary and parallel environment/client profile

The spawn-safe continuation adds published C/Python boundary sources and the
generated helper identity to its composed lifecycle. The initialized receipt
must identify the installed Native policy and counterfactual loop, the public
`SpawnSafeRunner` constructor, and a loaded helper digest matching the retained
binary. The base's original zero-epoch entry can use that public runner without
running the continuation wrapper, which intentionally rejects initial tests.
Its prospective helper identity is an observation of bytes before launch;
baseline policy/helper installation remains source-linked inference.

The parallel service and request-ID client are additional behavior outside the
legacy core fingerprint. The comparer therefore requires their separate real
observation and immutable Task-profile binding. It verifies the published
client producer, request client and environment server; regenerates the public
Native/client wrapper and exact environment-interpreter adapter; checks the
tool's shared allowed inode; and permits only the documented `agent.codex_bin`
Task change. Solver instructions remain identical. Actual configuration must
bind those generated assets, `SpawnSafeRunner` with32 sessions, four environment
processes/four I/O workers, and the measured1/1/1 microbatch profile.

For the trained stage, `driver_runs/*.parallel_client.json` is a prelaunch
observer. Its immutable profile must agree with the final initialized config
and official protocol's Task-file hash. For the deferred base,
`runtime/parallel_client.expected.json` is produced from actual fresh
preparation, generated assets, current source/helper bytes, resolved config and
live health before the original zero-epoch command. Its kind is explicitly
`prospective_parallel_environment_runtime_identity`; it must never be presented
as a post-init or served-model receipt. Existing spawn-only profiles retain
their older prospective source bundle format.

Both observations must bind the request-ID capability, stable service instance,
source digest and4/4 service profile. Each official case's private budget
registration must also identify a retained **test-only** service configuration
with those source/profile values; the train service observation alone cannot
stand in for the service that actually produced held-out cases. Reports retain
hash references and selected public metadata, never the private administration
token. Base and final source/helper/client/interpreter bytes must match even
though their run paths, service instances and intended policy/library differ.

The deferred launch plan remains gated on the main driver's successful64-update
completion, its COMPLETE checkpoint and finished sealed274-case final stage.
This offline comparer does not execute that plan or start a baseline.

## Direct-checkpoint continuation evidence

A run resumed through `train_direct_checkpoint.py` retains its actual outer
entry identity. The comparer reads `*.direct_checkpoint.json` component
observations and requires a matching `*.direct_checkpoint.initialized.json`
record linked to the lifecycle's original spawn initialized receipt. An older
parallel-entry observation cannot replace that initialized chain.
Matching Direct observations take precedence over retained older parallel
outer-entry observations. The lifecycle-linked initialized record selects the
active published Direct identity; its observations must match the current
train-service configuration hash. Older Direct records tied to another
initialized chain or service are retained as history. Conflicting matching
active observations remain unsupported; no newest-file heuristic selects a
startup.

The entry and restore provider, unchanged parallel/spawn components, and both
checkpoint/trainer core files must match their published Git objects and
retained file hashes. Their complete enumerated runtime source inventory must
remain unchanged. The initialized direct and spawn records must agree on the
effective configuration, preserved original manifest, source checkpoint and
core versions. The recorded prelaunch config must lead to that configuration
through only the existing constructor normalizations and explicit restore
bindings.
The Direct route also permits the user-selected operational save cadence to
change from the preserved manifest's `trainer.save_freq=1` to actual `10`.
The prelaunch config and both linked initialized records must agree on `10`;
other cadence changes remain unsupported. Legacy initialized profiles keep
their exact original cadence check. Save cadence was already excluded from
the common inference signature.

This route uses its explicitly recorded source checkpoint N, rather than
assuming checkpoint4. The supported current profile has matching optimizer
and global clocks, with 1 <= N < 64 for continued training. Source64 is allowed
only for a final-only restore sealed as `restored_complete_final_checkpoint`.
The source COMPLETE publication and files are verified, and its full
world/layout/version/evaluator/protocol compatibility must equal the final
checkpoint. The writable root must name the same source step.

The linked `DIRECT_READ_RESTORE.json` and `RESTORE_COMPLETE.json` must bind that
source and writable root. The original actor, EMA and `data.pt` paths, one full
validation pass, zero copied EMA files, three copied mutable components and
their original byte total are checked. Existing mutable files must not alias
source checkpoint files. The source guard and elapsed time are recorded
operational evidence; the comparer does not claim to have observed the original
startup's disk operations or measured a speedup.

The common inference signature continues to compare the unchanged parallel
producer, request client, environment server, interpreter, spawn helper,
generated assets and generation profile. The deferred zero-epoch base does not
need to execute a restore entry. Direct entry/provider identity is reported
separately under `direct_checkpoint_restore_evidence`; this exception permits
the verified operational composition only, not arbitrary profile migrations.

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
python examples/skill_self_evolve/tests/test_compare_parallel_profile.py -q
python examples/skill_self_evolve/tests/test_compare_direct_checkpoint.py -q
```
