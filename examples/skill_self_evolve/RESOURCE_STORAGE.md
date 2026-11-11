# Native temporary-storage budget

Preflight checks the filesystem actually mounted at `/tmp`, before package
imports, native startup checks, or the training process. It requires both free
bytes and free inodes. Upstream UniAgent constructs temporary native launch
paths under `/tmp`; setting `TMPDIR=/dev/shm` or having free NAS space does not
make those paths use another filesystem.

| Setting | Default | Basis |
| --- | ---: | --- |
| `MIN_NATIVE_TMP_FREE_BYTES` | `10737418240` (10 GiB) | Approximately 100 MiB per observed maximum native home ×64 active homes, plus 3.75 GiB reserve. |
| `MIN_NATIVE_TMP_FREE_INODES` | `650000` | Approximately 7,900 entries per observed maximum home ×64 =505,600, plus 144,400 reserve. |

Overrides must be positive integer counts: bytes and inodes respectively. The
preflight report records the configured limits, actual availability, whether
`/tmp` shares the root filesystem, and the measurement basis. An insufficient
budget fails startup with the actual values and relevant setting names. It does
not delete or move directories.

The budget is based on **active native concurrency**, not validation batch size.
The current runner has one framework worker with a semaphore limit of 64. A
`VAL_BATCH_SIZE=512` can queue all 399 dev games or all 140/134 official cases while
retaining that limit of 64 active processes. Larger active concurrency or another native
profile requires an explicitly reviewed budget; changing a queue batch alone
does not multiply the home-storage estimate.

Owned per-episode cleanup and dedicated runtime placement bound ongoing use.
This startup check complements that lifecycle; it cannot reserve capacity against
other processes filling the filesystem after launch.

Every 15 seconds, `resources.jsonl` retains the existing `root_available_bytes`
and `scratch_available_bytes` fields and adds:

- `root_available_inodes`
- `tmp_available_bytes`
- `tmp_available_inodes`

These storage values are sampled even if the GPU inventory query fails. Root
and `/tmp` may refer to the same filesystem; their values are observations of
each path and must not be added as separate capacity. The scratch field can
describe `/dev/shm` while `/tmp` is full, so inspect the explicit `/tmp` fields.

A storage failure produces incomplete native evidence. It must not be repaired
by creating zero-reward trajectories, allowing empty TQ keys, skipping official
cases, or accepting a partial evaluation baseline. Preserve raw artifacts,
restore storage headroom and the owned native lifecycle, then use a fresh run.

CPU regression:

```bash
python examples/skill_self_evolve/tests/test_resource_storage.py -q
```
