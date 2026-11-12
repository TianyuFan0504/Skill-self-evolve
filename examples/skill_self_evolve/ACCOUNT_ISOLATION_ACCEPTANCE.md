# Host acceptance of the attempt filesystem boundary

These checks ran on `temp-server` with the actual installed Codex 0.167.0.
They establish host/kernel and native startup behavior, not an ALFWorld score
or successful reinforcement learning. The tested `accounts.py` SHA256 is
`252250b5860ac3e0a381bb5035142b7db1bc70d667ba82b96fd79fa634045b5e`.

## System identities and kernel behavior

The operator created and validated only the 128 missing, dedicated locked
identities and private groups in the reviewed roster. Existing identities were
not modified. Registry initialization subsequently observed 128 clean free
slots and no active owners.

An actual production `launch_arguments` child ran as UID/GID 19000. Its real,
effective and saved identities were all 19000, supplementary groups were
empty, capabilities were zero and `no_new_privs` was set. The child attested
Landlock ABI1 and the inherited `abi1-truncate-x86_64-v1` seccomp guard.

All 19 real OS checks passed:

- Own files could be read and written; a declared read-only file was readable.
- Reading a world-readable foreign file and global `/proc/1/cmdline`, or
  creating a file in an otherwise writable outside directory, was denied.
- Path-based truncation, legacy read-only truncating open, read-only truncating
  `openat`, and mode-3 truncating open were denied. The read-only fixture's
  original contents remained unchanged.
- `openat2` and all three io_uring entry points returned `ENOSYS`.
- Own writable opens and `ftruncate`, SQLite WAL transactions, PTY creation,
  nested Python standard-library execution and the public helper's loopback
  transport remained usable.

The helper transport used an explicit CPU fixture endpoint, without creating
an ALFWorld game or awarding a benchmark reward. No process remained under
the tested UID, and all owned fixture paths were removed.

## Native startup and storage

The same production launcher started the original Codex binary under UID
19000, using the updated solver wrapper and a host-owned private-GID catalog.
Normal native storage was enabled. A loopback endpoint recorded the original
first POST and explicitly returned `400 CAPTURE_ONLY_NO_MODEL`.

The effective tool names were exactly `exec_command` and `write_stdin`; all
four input messages contained only `input_text`, and parallel calls remained
enabled. Native exited with code 1 as expected for that endpoint. Its stderr
contained only the stdin notice. Native SQLite databases and their WAL/SHM
files were created for state, logs, goals, memories, queue and thread history.
The UID had no remaining processes after exit. No model response or likelihood
was fabricated, and this capture produced no learning update.

A separate actual production preflight exercised acquisition, exact resource
registration, the kernel readiness pipe, native `--version`, complete native
archive/readback/removal, successful lease closure and release. The version
was `codex-cli 0.167.0`; the pool returned to 128 clean free slots.

After the final Task host-exception patch, the server ran all 117 related CPU
regressions successfully: 74 lease/launcher/Task/probe/learner-boundary checks,
20 controller checks, 11 joint-trainer checks and 12 media checks. Raw logs and
the summary are under
`module56_final_cpu_1794449458509638510/`. The tested Task SHA256 is
`302e3943a879fa0fb99fc42af404f189e112f9f05bfb11a8c3ebb9d30129534e`.
These CPU fixtures do not establish model or benchmark improvement.

## Evidence

All paths are relative to
`/root/nas_chaohuang/skill_training/diagnostics/`.

| Artifact | SHA256 |
| --- | --- |
| `uid_roster_provision_1794447160641895946/summary.json` | `7aa38a4b6079f13897814211c73ea410c0e908eb2c539f2f708c85413fe7f15b` |
| `module56_pool_initialized_1794448433882394338.json` | `5163a35a166ef88bf25e81f90b7c0f5a8464f74e0e9cf0242a597996ed48e72e` |
| `module56_real_os_1794448078292676751/result.json` | `9161c213d0daada2688dc23a4fda459e2ab1e61e239b20c7c86e9c7e71002bb5` |
| `module56_native_acceptance_20261112/capture/capture-summary.json` | `69a8d50244a089c224e8be237da5dedfa400ed4d88d0873dd0a20a44aa2c90ad` |
| `module56_native_acceptance_20261112/capture/kernel-ready.json` | `043dd8bfbd90b5a7af2eac6371144558ab982f961a839f43360a55abfe597712` |
| `module56_native_acceptance_20261112/capture/storage-observation.json` | `6e250f62219f81942190fe08c3181179498d97f2d3ed776cf002a5569017a171` |
| `module56_production_preflight_20261112/acceptance.json` | `c4d0f1bf93f9fc37eaecc1de18ff6e05c93bb249390ecd17158c16d236323193` |

The [policy and lifecycle documentation](ACCOUNT_ISOLATION.md) describes
quarantine and failure handling. This boundary does not claim network or
metadata/stat isolation. A real controlled native pair must additionally
establish that both variants use one identity and that the actual model inputs
only differ by the target skill. A new complete initial/final evaluation is
still required to establish training improvement under this policy.

## Conda Python pidfd compatibility

The formal v4 initial evaluation exposed a previously unexercised cleanup path:
the production `/opt/conda/bin/python` 3.11.13 has neither `os.pidfd_open` nor
`signal.pidfd_send_signal`. Its glibc 2.35 also omits the corresponding exported
wrappers, although the Linux 5.15 kernel supports both operations. The previous
version/normal-exit checks had no surviving UID process to signal, and the CPU
signal test supplied mock Python APIs. Those checks therefore did not establish
the availability of the production interpreter's signal path.

`accounts.py` now prefers the Python APIs, then exported libc wrappers. If both
are absent, it uses the verified Linux x86_64 `pidfd_open` (434) and
`pidfd_send_signal` (424) syscalls. These numbers match the installed
`asm/unistd_64.h`. This is the same kernel process-handle mechanism, with no
numeric-PID signal fallback. The opened process's UID is still checked before
signaling, under the existing lease lock. A vanished process remains benign;
permission errors, unsupported syscalls and unverified syscall architectures
remain fatal. Descriptor closure also runs on refusal and error paths.

The local CPU suite passed 22 checks: 10 launcher checks, 11 lease checks and
one production-preflight contract. Regression cases cover the missing Python
and libc wrappers, exported libc wrappers, UID mismatch, `ESRCH`, `EPERM`,
`ENOSYS` and refusal of an unverified architecture. The production base
interpreter also passed nine direct helper checks using one newly owned
UID/GID 65534 child, empty supplementary groups, zero capabilities and
`no_new_privs=1`. Wrong-UID signaling was refused without killing the child.
The actual patched helper delivered `SIGTERM` through its real syscall
compatibility path; the direct child's retained `Popen.wait` result was `-15`.
An unrelated root-owned control child remained alive at measurement, and both
owned children were removed afterward. Signaling the already vanished child
was benign. No Native agent, model, learner or reward was created.

The direct helper evidence is
`/root/nas_chaohuang/skill_training/diagnostics/module64_direct_pidfd_1794457545202366689/acceptance.json`,
SHA256 `ff0306dfd3010f5f23f8116dd73bc9e052d1ac5f542259d7f0915e72d5cc3ba7`.
The tested `accounts.py` SHA256 is
`77caeeb95330261024ef2c3d334a6e5c423b37733099bb1333eaf9e318948c08`.
Server CPU logs and the 22-check summary are retained under
`/root/nas_chaohuang/skill_training/diagnostics/module64_pidfd_acceptance_1794457295/`.

At that acceptance cut, the separate actual `AccountPool.drain` probe was waiting for the
existing Ceph pool lock before acquiring a lease or spawning any child. It
was subsequently superseded and cancelled after a fresh disjoint local
registry passed the [actual pool and Native acceptance](ACCOUNT_ISOLATION.md).
The direct helper check and that later end-to-end observation retain separate
source and evidence identities. Existing failed run results, incidents and quarantined
UIDs are retained; this repair does not relabel or automatically release them.
