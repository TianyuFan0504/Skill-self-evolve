# Fresh host account pool, epoch3

The opt-in `skill_harness_examples.train_curate_epoch` entry is policy
`parallel-direct-all-native-spawn-curate-generation-account-epoch-v3`.
It composes the v2 generation namespace, direct checkpoint reader, spawn learner
and guarded text-only DataLoader transport. Its physical counterfactual probe
limit is8, within the unchanged configured `JointSettings` ceiling8. The v2
entry retains its historical physical limit2. Four configured DataLoader
workers, sampling, rewards, loss, model routes and scientific configuration
remain unchanged.

The retired registry `/var/lib/alfworld-skill-harness/account_pool_v2`, all
UID/GID20000–20127, all old leases/quarantines and protected UID20071 remain
fenced. This provider never resets, releases, drains, remaps or initializes
that registry. It reads historical settled local-archive dispositions so failed
training groups remain excluded. The new independent registry is
`/var/lib/alfworld-skill-harness/account_pool_epoch3`, containing128 accounts
`alfworld_ep3_000`…`alfworld_ep3_127`, UID=GID30000…30127. It uses the unchanged
core account isolation policy `alfworld-uid-landlock-v1`.

All files under `verl/`, recipe config and prompts remain unchanged. Pool
selection is operational evidence outside `NativeCuratorRuntime`, the core
source fingerprint and checkpoint protocol identity. No checkpoint version,
source hash or manifest check is relaxed. A new pool does not establish the
cause of a prior learner exit. The separately reviewed periodic traceback
change and its crash reproduction evidence remain independent.

## Operator provisioning, before any model or environment work

Importing these modules does not provision accounts, initialize a registry,
request a model, or start an environment. Run commands from the pinned source
checkout using the existing training interpreter and source `PYTHONPATH`.

1. Print the read-only plan:

   ```sh
   python -m skill_harness_examples.account_pool_epoch plan > /private/path/epoch3-plan.json
   ```

   The API checks existing user names, UIDs, group names and GIDs. Existing
   matches must pass the frozen locked-password/nologin/home/comment and group
   membership checks. An unrelated user's primary GID in the new range also
   fails. Missing identities produce reviewed `groupadd`/`useradd` argument
   arrays, with locked passwords, no home and no login shell. The API executes
   none of these commands. No command modifies an existing user or group.

2. The operator reviews and executes only those missing-account commands. The
   host-owned parent directory and manifest directory must already be private
   mode0700. After provisioning the exact identities, explicitly initialize:

   ```sh
   python -m skill_harness_examples.account_pool_epoch initialize \
     --manifest /var/lib/alfworld-skill-harness/epoch3.manifest.json
   ```

   The root must be completely absent. An exclusive directory creation closes
   the initializer race; the frozen initializer holds its registry lock. Even a
   partially existing root is rejected, rather than recreated. Failures retain
   partial evidence for host review. The operator manifest is immutable,
   mode0600, and reports its absolute path and raw SHA256. Its contents bind the
   exact new root/roster, isolation policy and explicit old-pool fence.

3. Set both hash-bound variables for driver, preflight and workers:

   ```sh
   export ALFWORLD_ACCOUNT_POOL_MANIFEST=/var/lib/alfworld-skill-harness/epoch3.manifest.json
   export ALFWORLD_ACCOUNT_POOL_MANIFEST_SHA256=<reported-raw-file-sha256>
   python -m skill_harness_examples.account_pool_epoch preflight
   ```

   The read-only API checks manifest privacy/hash, exact roster, locked system
   identities, registry roster/policy/slots, quarantines and live processes.
   It does not acquire a lease, write a lock, release an account or create a
   directory. The driver gate requires at least40 available accounts (32 primary
   sessions plus eight paired probe UIDs). The existing explicit native
   preflight retains its stricter64-account gate and uses the selected root
   when either epoch variable is present; a missing counterpart fails closed.
   The general historical preflight also performs its documented Native/runtime
   probes, so use the dedicated read-only command for this provisioning check.

After provisioning, perform a model-free Ray acceptance check before training:
apply the normal bootstrap runtime environment, then
`account_pool_epoch.prepare_runtime_environment(config)` and run
`ray.remote(account_pool_epoch.inspect_worker_bindings)` with that environment.
The acceptance body requires the hook to have run, constructs the actual Task
and probe pool aliases, verifies their common root/roster, and acquires no UID.
It must report `lease_acquired:false` and epoch3. It does not repair a missing
hook. This check is an explicit operator action, not performed by import or by
the provisioning CLI.

## Process selection and evidence

Installed Ray2.50 primary source was inspected at
`ray/_private/runtime_env/setup_hook.py`: lines103–115 dispatch a callable or
string; lines136–143 load a string as a full callable path and invoke it with
no arguments. The selected string is therefore
`skill_harness_examples.account_pool_epoch.setup_worker`, rather than a module
name. The v3 shared entry installs the driver provider before environment
health/preflight work. It applies this worker hook **after** the original
bootstrap merge, which would otherwise replace a caller-supplied hook.

The hook first executes the unchanged bootstrap configuration and then validates
the private provisioned manifest and host registry. It binds only these three
import aliases in that owned process:

- `alfworld.task.AccountPool` → frozen `AccountPool(new_root, roster=new_roster)`;
- `probes.AccountPool` → the same constructor/root/roster;
- `group_sampler.local_archive_failed_groups` → the unchanged reader applied to
  both retired and new roots.

The core `accounts.AccountPool` class, constants and methods are never replaced.
Paired probes retain one lease across both branches; the Task validates that
lease against the same fresh registry. Unknown-artifact filtering rules are
unchanged. The actual learner additionally rejects a lost runtime hook or
missing matching Ray activation before trainer construction.

The examples Native transport had a separate old UID-range guard. Its new
explicit `register_account_pool_epoch` API registers only the verified manifest
roster in the current process. Without an epoch, the historical UID range stays
20000–20127. With epoch variables but no completed registration, native launch
fails; old UIDs are rejected after registration. Argv shape, ready FD, descriptor
closure, UID/GID equality and launch options retain their checks. The existing
`SpawnSafeRunner` is unchanged. Native `isolate-exec` itself already accepts
validated supplied identities and needs no code change.

The environment service and wall-budget code do not instantiate AccountPool or
apply a fixed UID range. Their authenticated registration validates the supplied
positive UID/GID, lease ownership, backend binding and drained receipts. No
provider is inserted into the independently running environment service.

Private immutable observations are published under
`runtime/account_pool_epoch/activations/` with process PID/start ticks and an
explicit role: driver, Ray worker, learner, Task pool construction or probe pool
construction. They bind provider source/hash, operator manifest/hash, root,
roster digest, hook and Native registration. An initialized receipt lists only
observations already present and the directory where later primary-worker
observations appear; it never claims future activation. Pool construction
observations do not attest a model request or a particular episode.

## Untouched-model comparison

Use `skill_harness_examples.train_account_pool_baseline` for the common-pool
zero-epoch baseline. It calls the original entry's public validation,
`save_run_manifest`, `run_ppo` and original `SkillTaskRunner`, with only the
explicit pool preparation/hook/evidence added after the base bootstrap merge.
It rejects nonzero epochs, restore mode/path and `val_only`. The caller retains
the existing common inference preparation: same primary `SpawnSafeRunner`,
concurrency32, parallel client/environment adapter and original loading
protocol. The original mode remains `skill_joint`; modelversion0, optimizer0
and empty genesis are verified by the comparer.

This wrapper preserves the original joint trainer lifecycle, including its
curator metadata/runtime preflight, model-route allocation, EMA synchronization
and gateway initialization. It adds no generation namespace or AllNative curator
extension and performs no new candidate generation from the empty source queue.
It does not claim those original initialization operations disappear at epochs0.
The original official evaluation creates its own worker0 loader for both sides.

The baseline sidecar `runtime/account_pool_epoch.expected.json` identifies both
the actual wrapper and original entry component. The v3 comparer requires the
same operator manifest, provider, roster, hook and primary inference signature
on both sides. It checks actual worker/pool activation and every official
case's recorded account UID/GID/name, rather than accepting selection intent
alone. The trained side additionally requires actual learner/probe activation
and cap8. Historical v1/v2 policies and source64 limits remain supported; their
receipts are not relabeled as epoch3.

## CPU validation

`test_account_pool_epoch.py` uses the real frozen AccountPool against private
temporary registries, with system-account/process lookups replaced by fixtures.
It covers inert import, read-only behavior, hash/privacy/roster rejection,
no-reset initialization, account conflicts, quarantine/live/lease capacity,
paired Task/probe registry agreement, Native roster/FD guards and historical plus
new archive dispositions. Entry fixtures cover hook ordering and original
baseline behavior. The comparer fixtures exercise a complete synthetic274-case
comparison and reject mismatched or merely claimed pool/worker evidence. No
fixture provisions host identities or starts a model/environment.
