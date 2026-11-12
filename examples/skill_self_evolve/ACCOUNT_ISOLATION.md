# Private ALFWorld attempt identities

ALFWorld native solver and counterfactual episodes use a fixed roster of 128
non-login users, `alfworld_ep_000` through `alfworld_ep_127`, each with a private
UID/GID in `19000..19127`. There is no shared `alfworld_eval` fallback. Account
creation is an explicit operator action, never a Task or import side effect.

## Provisioning and preflight

From the pinned source checkout, an operator first inspects the JSON command
plan, executes only its absent-identity commands, and initializes the registry:

```bash
python3 verl/experimental/skill_harness/alfworld/accounts.py provision-plan
python3 verl/experimental/skill_harness/alfworld/accounts.py initialize
python3 verl/experimental/skill_harness/alfworld/accounts.py check
```

`provision-plan` only prints commands. An existing name, UID or GID with different
attributes is an error; the module does not repair or modify it. All identities
must have the exact marker comment, `/nonexistent` home, nologin shell, locked
password and no supplementary memberships. The registry is host-private under
`/root/nas_chaohuang/skill_training/account_pool_v1`.

The normal recipe preflight requires at least 64 clean available identities,
Linux Landlock ABI1 or newer, and the supported Linux x86_64 syscall guard. It
runs the production isolated launcher with native Codex `--version`, without a
model request, then validates its readiness proof and native archive receipt
before releasing that lease. A missing or unsupported boundary fails closed.

## Actual child boundary

The trusted host creates the exact fresh UUID `CODEX_HOME`, cwd and catalog
before starting native code. `setpriv` sets all real/effective/saved UID/GID
values to the leased identity, clears groups, drops effective/permitted/
inheritable/bounding capabilities, and enables `no_new_privs`. The child applies
inherited Landlock rules and a narrow inherited seccomp filter, writes the
readiness proof to a private pipe, closes that descriptor, and only then execs
the original native command.

Writable roots are that attempt's HOME and cwd, with a private `.native-tmp`
inside cwd, plus required `/dev/null`, `/dev/zero` and `/dev/pts`. Trusted system
binaries/libraries, the public game helper, configured native package, exact
instructions/wrapper, and exact catalog are readable. Global `/tmp`, NAS, other
attempt HOMEs, and global `/proc` are not readable roots. Catalogs remain
host-owned, mode `0440`, with the lease's private GID. Their original owner,
group, device and inode must match the archive receipt.

Landlock ABI1 alone does not control every truncation operation. The x86_64
seccomp filter rejects path-based `truncate`, and `open`/`openat` with `O_TRUNC`
when access mode is read-only or ioctl-only (mode 3). Normal writable opens and
`ftruncate` remain available for native SQLite files. `openat2`, io_uring
setup/enter/register, and x32 syscall aliases return `ENOSYS`, so they cannot
bypass the inspected open flags. Native/SQLite fallback compatibility requires
an actual host acceptance run; CPU filter interpretation alone does not prove
it. The boundary does not claim network or metadata/stat isolation.

## Lease and pair lifecycle

An ordinary episode exclusively retains a UID until its backend is drained,
all processes with that UID are gone, every registered source has been archived
and read back, the receipt is durable, and the exact sources have been removed.
Drain signals use pidfds and recheck the UID; PID disappearance or elapsed time
never authorizes reuse.

The outer counterfactual coordinator retains one lease across both variants.
The same controlled pair uses the same UID/GID, username, visible HOME
(`/nonexistent`), private TMP path and cwd. Each branch gets a distinct native
`CODEX_HOME`, and its cwd contents are durably archived and removed before the
next branch recreates the same empty path. The pair group is scoped to the run.
No `UsageEvent` is published until both branches pass the existing only-S input
gate and the pair lease has closed. A failed first branch has an explicit outer
abandonment path; a safely closed branch need not wait for an unstarted second
branch. Partial captures are never silently resampled.

Any boundary establishment, process drain, archive, receipt or release failure
creates a sticky run incident and retains/quarantines the UID. Unconfirmed
UID or backend drain prevents archival/removal entirely; live writers' original
HOME/cwd/catalog sources remain for reconciliation. The pair control lock is
released with a failed marker, without deleting those sources. An incident
directory alone blocks the learner, even if writing its JSON details fails.
A separate sticky per-UID quarantine prevents reuse even after a partial
release-state publication failure. Ordinary agent commands correctly receiving
`EACCES`/`EPERM` are not isolation incidents.

There is no automatic quarantine reset. An operator must reconcile all process,
backend, archive and source evidence before explicitly repairing registry state
or removing a quarantine marker. Starting a new process, waiting for a TTL, or
initializing an existing pool does not reset it.

The public Task entry point resolves the trusted host run root and checks the
Task metadata against it. Host Python/SDK exceptions, including failures before
environment creation and after native execution starts, persist an incident
and remain fatal after adapter filtering. Normal `AgentResult`/`ExecResult`
failures, expected terminal interruptions and cancellation retain their own
semantics. A host exception must not become an ordinary zero-reward sample.

## Validation scope

CPU tests exercise exclusive/fenced leases, both pair closures, first-branch
abandonment, residual processes, damaged archives, undeleted sources, sticky
write failures, partial release persistence, readiness identities, private TMP
reuse, pidfd ownership checks, and the emitted BPF decisions. Capture/workdir
regressions use real temporary files and ordinary CPU subprocesses with the
privilege launcher explicitly mocked. They create no OS accounts and install no
kernel restrictions. Before training, the operator must additionally verify the
real roster, production launcher, cross-attempt path denials, truncation denials,
PTY/helper/SQLite operation, native startup, and a controlled real pair.

See the [recorded host acceptance](ACCOUNT_ISOLATION_ACCEPTANCE.md) for actual
kernel/native/lease evidence and its precise scope.
