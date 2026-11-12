# Queued curator sources after checkpoint recovery

`scripts/train_curate_generation.py` is an explicit opt-in entry for a future
training process. It composes the AllNative transport, direct checkpoint reads,
existing solver sampler and frozen training/final-evaluation body. The public
launcher and ordinary AllNative entry retain their existing default behavior.
The actual deployed entry is identified by its runtime receipts.

The opt-in entry requires remaining optimizer updates. Restoring a checkpoint
whose optimizer clock has already reached the target is rejected before
namespace publication. Final-evaluation-only recovery for this new policy is
outside this module and needs explicit lifecycle support; no fallback to
another entry is claimed to have been verified.

## Why a new execution namespace is needed

A checkpoint copies mutable SkillStore, replay, ledger and controller state,
but the run's append-only curator journals can contain later work. A source
that was queued with no curator jobs at the checkpoint can therefore encounter
an old `intent.json` or `complete.json` after restoration. The original
collection ID uses producer/group/index, whereas the immutable input identity
also includes the original timestamp, snapshot and behavior. Reconstructing
the request under that old ID correctly fails integrity checks.

Adopting `complete.json` alone is unsafe: its original candidate, replay record,
causal indices, snapshot and physical replay reference may not exist in the
restored mutable roots. This implementation never imports orphan results,
changes their hashes, or calls a new action the original draw.

## Execution and persistence order

1. In the actual learner, the v2 provider sets and observes the multiprocessing
   default as `spawn` before constructing the trainer. It records this
   observation separately from prelaunch source/provider intent.
2. The existing trainer initializes and restores the actual checkpoint. Before
   either loader starts an iterator, the provider applies the transport repair
   described below to the existing train/dev dataset instances.
3. The provider creates and fsyncs a fresh namespace document under
   `controller/generation_epochs/`. It records the process generation, actual
   source checkpoint and provider hash. The initialized entry receipt calls
   this a **prepared** namespace; no controller activation or model generation
   is claimed at this stage.
4. The actual `GenerationController` publishes a separate activation document
   and persists its reference in controller state before the unchanged
   background initializer starts its pump.
5. For a newly claimed source whose `curator_jobs` is empty, the adapter derives
   `candidate_group_id = "cohort_" + SHA256([original_update_id,
   episode_output_dir, namespace])`. It persists the full request, actual EMA
   behavior and namespace before delegating to the unchanged collector.
6. The collector retains its exact full context, action tokens, processed
   behavior logprobs, EMA version, abstention and immutable commit checks. The
   controller's existing credit windows and replay acknowledgments still
   identify a single action in the active restored branch.

The source UID, source episode directory, source step, original update ID and
source identity hash do not change. The operational namespace is outside the
scientific configuration and protocol fingerprint. A same-execution retry uses
the same timestamp, group, context and behavior. A different process restoring
the checkpoint receives another namespace and cannot collide with orphan
journals from the abandoned branch.

Existing nonempty jobs are never renamed. A job can be delegated again only
when it belongs to this execution and its saved behavior matches the actual
curator route; otherwise it receives an explicit local failure requiring
reconciliation. The frozen initializer continues to mark interrupted running
actions as failed without resampling them. Terminal dispositions, including
the128 failed sources observed during the earlier checkpoint10 transition,
remain immutable. Old artifacts remain available for audit, contribute no
duplicate admission or loss, and are not negative/zero utility observations.

## Operational safeguards in v2

The original learner exit cause is unknown. Four forked DataLoader children
were observed holding inherited Ray/GPU descriptors after that learner exited.
Selecting spawn addresses descriptor inheritance; it does not establish why
the learner exited or claim to repair that unproven cause.

The learner changes its own default multiprocessing context before trainer
construction. Configured train/dev worker counts remain4. Installed torchdata
uses this default when its loader has no explicit context override; the
initialized receipt records the default and actual loader attributes, rather
than asserting that every worker has already spawned.

The production default `RLHFDataset.__getstate__` omits its filtered dataframe,
and the Qwen processor contains a dynamically bound method that cannot be
unpickled by spawn. Therefore spawn alone is insufficient. After filtering and
checkpoint state loading, but before any loader iterator exists, v2:

- requires the exact default `RLHFDataset` and verifies plain text prompt
  content with no media payloads;
- sets the existing dataset-instance `serialize_dataset` transport flag so
  spawn carries its original filtered Arrow dataframe;
- removes only the dataset's unused processor reference. The trainer's
  processor, tokenizer, sampler, filtered frame, data config and pending
  DataLoader state remain unchanged.

The provider rejects custom/multimodal datasets, explicit fork overrides and
already-started iterators. It never migrates to workers0. These are transport
operations outside the scientific config and checkpoint protocol fingerprint.
Scheduled dev uses the repaired four-worker loader. The existing official
evaluation loader uses workers0 and needs no multiprocessing serialization.

An async semaphore admits at most2 background probe bodies to the existing
lease/RPC path at a time. The frozen pump and `JointSettings.credit_concurrency`
retain ceiling8, so other logical jobs await the semaphore without a lease,
RPC or blocked event loop. No wait is added to the learner's update path.
Cancellation releases a held permit; unknown observations retain their
original unknown/interrupted disposition and create no zero reward or label.

This limit bounds admitted probe bodies under their confirmed lifecycle. It
does not release retained uncertain leases, prove a global count of orphan
processes, or reset quarantined accounts. Their separate ownership/drain
evidence remains required. Source8/physical2 is recorded explicitly in the
operational evidence instead of silently rewriting protocol settings.

## Invocation and evidence

Prepare the run, environment service, offline model/runtime, account pool and
Native client configuration exactly as for the AllNative entry. In that
prepared launch command, select this driver instead of
`train_all_native_spawn.py`:

```python
# Inside an owned launch integration, before starting its supervisor:
from pathlib import Path

command = list(prepared_supervisor_config["command"])
old_entry = Path(command[1])
assert old_entry.name == "train_all_native_spawn.py"
command[1] = str(old_entry.with_name("train_curate_generation.py"))
prepared_supervisor_config["command"] = command
```

Pass the updated command through the same supervisor setup. Preserve the
training executable, every resolved override, the already prepared environment
mapping and the owned environment-service PID; do not reconstruct private
environment values. The provider receives the resolved config through
`train_curate_generation.entry(config)` and delegates to the shared AllNative
entry with its explicit provider module. Its runner uses the provider's trainer
factory and prepares the namespace after `trainer.init()`.

Perform this selection only for a new supervised attempt after verifying the
previous owner has exited and selecting its complete source checkpoint. Never
edit a running supervisor's command or status file. This module supplies the
driver interface, not a new preparation/supervision tool or public launcher
flag. Automatic launcher selection remains unchanged. The following command
only illustrates the selected driver and Hydra interface; it is **not** a
standalone provisioning or one-command recovery procedure:

```bash
/opt/conda/bin/python \
  /workspace/verl-0.9.1/examples/skill_self_evolve/scripts/train_curate_generation.py \
  --config-path=/workspace/verl-0.9.1/examples/skill_self_evolve/config \
  --config-name=alfworld_joint \
  trainer.val_before_train=false skill_harness.evaluation.initial=false \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path=/path/to/complete/checkpoint \
  trainer.save_freq=10
```

The command requires the same prepared `ALFWORLD_*`, runtime and model
environment as the existing entry; it does not provision a second service or
infer ownership of an existing process. Preserve the normal single-learner
launch guards and supply the experiment's remaining resolved overrides.

Receipts use policy
`parallel-direct-all-native-spawn-curate-generation-v2`, distinct
`*.curate_generation.json` and `*.curate_generation.initialized.json` names,
the actual new entry/provider/hook hashes, the composed AllNative source, and
the unchanged solver transport component. The comparer accepts historical v1
evidence separately. For v2 it additionally checks the observed learner
default before construction, post-init dataset transport observation, and
actual semaphore activation with its configured ceiling. It verifies the
prepared namespace and later controller activation, including the activation
captured in the final checkpoint; it does not relabel these records as the old
AllNative implementation.

## CPU validation and limits

```bash
uv run --offline --no-project --python 3.10 python -m unittest discover \
  -s examples/skill_self_evolve/tests -p 'test_*curate_generation.py' -v
```

Tests exercise fresh identities across restores, stable retries, persistence
before delegation, unchanged source provenance and terminal jobs, rejection of
changed owner/checkpoint/EMA, the actual frozen running-action recovery code,
and the real collector/replay with inert model/native boundaries. The latter
retains an orphan journal unchanged, creates one distinct current-branch
action, preserves its actual fixture tokens/logprobs, and verifies neutral
abstention. Entry tests distinguish prepared from activated evidence. Synthetic
full274-case comparer fixtures check explicit new identities and legacy
compatibility; they are not real benchmark or model results.

`scripts/test_stateful_spawn_restore_linux.py` separately checks the installed
torchdata implementation using four CPU workers and the real checkpoint's
`data.pt`, original local dataset/tokenizer, sampler and collate function. It
compares fork-to-spawn continuation and deliberately inheritable sentinel
descriptors; no model or benchmark-environment request is involved. The test's
own output records exact source versions, file hashes, expected serialization
failures and successful continuation checks. Its worker observations are
limited to that owned CPU proof, not attestation of future training workers.

The checkpoint10 CPU proof on Torch2.11/torchdata0.11 retained four workers:
fork and adapted default-spawn produced the same next32 original task IDs,
raw-prompt hashes and entire loader/sampler state. A saved spawn state restored
into fork matched the following32 tasks. All four fork workers inherited the
test sentinel descriptor and all four spawn workers excluded it; all12 final
proof workers joined with exit0. The checkpoint file hash remained unchanged
and CUDA was never initialized. The dev399 dataset received the same guarded
transport preparation, but its workers were not iterated by that proof. The
proof redirected only its cache directory to an owned temporary location;
original parquet/model-tokenizer inputs and production config were not edited.

This fix does not recover completed orphan actions transactionally, retry an
uncertain native launch, rewrite current failed sources, or establish reward
improvement. It also does not change GRPO, EMA, delayed credit, admission,
optimizer coordination, checkpoints, prompts or sampling.
