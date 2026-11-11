# Native temporary-storage failure and recovery

The first `alfworld_joint_full64_20261112` experiment failed during its initial
development evaluation, before optimization or official tests. Of 256 submitted
dev cases, 166 had numeric trajectories and 90 did not. These partial results
are not a baseline; missing trajectories are not zero-reward observations.

The primary failure was native app-server initialization with `ENOSPC`. Both
bytes and available inodes on the 49 GiB root filesystem were exhausted. The
later `Received an empty list as keys` error was secondary: the framework had a
batch containing only failures and no materializable token rows. TQ correctly
rejected the empty request. No TQ bypass, skipped-case completion or reward
repair was added. The original status, failed attempts and logs remain intact.

## Cause and lifecycle correction

The pinned upstream CodexAgent hardcodes UUID HOME and catalog paths under
`/tmp`, independent of `TMPDIR`. It has no matching cleanup. Ordinary Task
completion previously protected its cwd but did not archive/remove those files;
paired probes cleaned their homes/cwd but not catalogs. Curator workspaces also
accumulated. Large shared memory and NAS capacity did not protect the root mount.

The new lifecycle registers only trusted native-created paths. It archives raw
HOME/catalog/cwd or curator collection state to private NAS artifacts, verifies
the complete member inventory and file/link contents by readback, checks source
identity/stability, and fsyncs the archive and provenance before removal. Failure
preserves originals. Existing paired-workdir semantics and model inputs, IDs,
behavior probabilities, environment rewards and RL losses are unchanged.

Real Codex 0.167 storage-only checks independently verified 98 normal-role and
92 curator archive members and confirmed the original paths were removed. The
HTTP endpoints deliberately returned 400; those checks started no model, GPU or
Ray and provide no benchmark or learning result.

## Historical recovery

An exact 1023-path candidate inventory was derived from original Task stderr
HOME references and catalog instruction hashes. Each path was rechecked for live
process references before archival and removal. Complete archive readback,
source checks and durable receipts preceded every deletion.

1022 paths were recovered into
`/root/nas_chaohuang/skill_training/recovery/owned_codex_tmp_20261112`.
One actively referenced HOME was retained. Another 171 homes lacked sufficient
attribution and were not targeted; `/root/.codex` and unrelated caches were
untouched. `recovery-summary.json` records the counts and approximately 13.2 GB
of available root space after recovery. Initial transient process-read failures
were retained and explicitly retried without relaxing the checks.

## Fresh experiment

The corrected run is `alfworld_joint_full64_v2_20261112`. It starts from the
original model with no old checkpoint, replay or partial baseline migration.
Native `/tmp` startup preflight now checks both actual free bytes and inodes;
the defaults are 10 GiB and 650,000, based on measured 64-session capacity.
Resource telemetry preserves bytes and records inode availability too.

Validation batches now queue up to 512 original tasks while the runner semaphore
still permits at most 64 active sessions. This reduces repeated batch-tail
barriers without shortening tasks, action limits or token budgets. Both official
test stages use the new settings. Full reward/accuracy improvement still needs
to be measured after the 64 real updates; recovery and passing preflight do not
establish that result.
