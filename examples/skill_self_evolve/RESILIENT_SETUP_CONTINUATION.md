# Narrow pre-Native setup failure continuation

`scripts/train_resilient_setup.py` is an examples-only continuation entrypoint.
It reuses the original strict checkpoint restore, single learner/optimizer,
EMA, raw capture, rewards and quiescent final-evaluation lifecycle. The published
quiescent entrypoint and core runtime files are unchanged; the new extension is
recorded explicitly rather than presented as an identical overall lifecycle.

The observed failure was an HTTP timeout awaiting the first `/admin/create`
response. The server later wrote initial states, so environment worlds may
exist even though the Task never obtained a session ID, UID or Native/backend
binding. This extension never reports those worlds as closed and never turns
this error into reward0 or a completed episode. Existing incidents and orphan
worlds require separate root-owned reconciliation before resuming. No incident,
account, source or old run is automatically deleted or released here.

## Exact local-data boundary

A new public Uni-Agent Task alias inherits the original Task. Its episode
override delegates the original implementation. Only `TimeoutError` at the
exact original function/code frame and first `/admin/create` await can be
classified, with `created`, session ID, UID/account, sandbox, agent-start and
backend bindings still uninitialized. The host run root, attempt directory,
original attempt manifest and rollout UID must agree. All later timeouts,
cancellation, identity errors and unrecognized failures retain original
handling, including the global unsafe-isolation barrier.

`NoNativeSetupDataError` subclasses the existing `LocalArchiveDataError` solely
to reuse its explicit nonsticky data-invalid catch. It does **not** mean an
archive failed or that resources/worlds are closed. Before raising, a private
immutable `NoNativeStartupInvalid` record retains the original attempt and
rollout UID, error location, `training_data_valid=false`, no Native launch and
`world_closure="unknown"`. The record does not contain a reward or invented
tokens/probabilities.

The examples custom sampler extends the existing archive-aware sampler. It
unions startup-invalid UIDs only after all siblings reach terminal status,
rejecting/refilling the entire N4 prompt group even if three siblings have
materialized valid rows. CF/evaluation markers cannot invalidate solver groups.
No partial group is trained and no late environment result is adopted as a
solver outcome.

## Public SDK composition and CF preservation

The configured `agent_runners.task.runner_fqn` delegates to the unchanged SDK
runner. It aliases only a deep copy of the internal Task sample and an
immutable copied Task profile. Original dataset/TQ Task names remain
`alfworld_skill_training`; the registry is never overwritten. Sandbox and
agent timeouts, model binding, metadata, raw prompt and host skill context are
preserved. The original profile and training manifest remain untouched.

The dedicated `skill_harness_examples` namespace points only to this project's
examples script directory. It avoids the installed third-party `examples`
package without replacing that package or changing global import registries.
The configured runner is `skill_harness_examples.train_resilient_setup.safe_run_task`.
The extension receipt also hashes the small namespace shim; both examples
drivers and that shim remain outside the unchanged core source fingerprint.

For CF, `safe_native_probe_branch` aliases copied internal fields/profile and
delegates the original `native_probe_branch`. Its Gateway finalization, exact
numeric capture and cleanup remain original. The outer paired collector still
sees original fields and verifies the same skill-only input change. A controller
subclass injects the collector's public `run_branch` argument before its
background pump starts. Unknown/failed pairs yield no UsageEvent or utility.
Coordinator-owned pair UID lifecycle is left to its existing context manager.

## Resume and final evaluation receipts

Root may invoke this new Hydra entrypoint with the same existing config and
strict COMPLETE checkpoint after stopping/reconciling the old owned run.
This module does not stop, launch or migrate an existing job by itself.
Main Native concurrency becomes32; N4, batch32, 900 seconds, 36,864 capacity,
CF32 and curator/credit concurrency8 remain unchanged. The reason is the
single TextWorld executor's observed initialization queue, rather than a
change in the task budget or reward definition.

The post-init immutable driver receipt records the **actual** config, including
restored store/replay/ledger roots, safe runner/custom sampler and concurrency32,
plus checkpoint and extension identities. The quiescent driver's hashed seal,
evaluation-entry and evaluation-complete records link that initialized receipt
and this extension's bytes/Git identity. Its success pointer still honestly
identifies the unchanged quiescent lifecycle implementation. Offline comparison
must follow these hash-linked effective-config bindings instead of assuming the
preserved original manifest's concurrency64/store path is still effective.
The later deferred baseline must match the actual final inference profile.

CPU contracts cover the exact nonsticky catch, strict rejects, alias/profile
copying, CF delegation, terminal whole-group eviction, pre-pump injection and
receipt linkage. Real-SDK tests independently exercise its actual resolver,
registry and configured runner. These checks are engineering evidence, not a
new training result or proof of heldout improvement.
