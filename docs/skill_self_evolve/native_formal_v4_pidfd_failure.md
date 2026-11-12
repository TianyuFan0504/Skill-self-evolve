# Formal v4: real trajectory evidence and host cleanup failure

`alfworld_joint_full64_v4_20261112` started with code source `1845622e` and
the complete predeclared 64-update protocol. All eight physical GPUs were
owned by the run. The initial official seen evaluation began at optimizer
zero, but host cleanup failed before that initial evaluation completed. The
raw run status is `failed`, exit code 1; no optimizer update or complete
initial-versus-final comparison is claimed.

The actual Conda Python 3.11.13 interpreter lacks both `os.pidfd_open` and
`signal.pidfd_send_signal`. Glibc 2.35 also lacks their exported wrappers.
The original host guard therefore rejected cleanup whenever a surviving
Native UID process required signaling. Kernel 5.15 supports the real pidfd
syscalls. Earlier version/normal-exit checks could pass without exercising
that branch.

One affected episode had genuinely won after 15 world actions, with its
backend already drained. It nevertheless failed UID closure and was not an
admitted baseline result. Run-level isolation incidents prevented training.
The supervisor was explicitly signaled while the verified optimizer counter
was zero. Its model processes and private environment service then stopped.
Failed outcomes, incident records and uncertain leases remain unmodified.

## Evidence retained before stopping

Independent review verified one clean unsuccessful episode and one clean
successful episode. Their actual float32 behavior likelihoods align with
response tokens and masks, are finite, and are nonpositive on authored tokens.

| Clean case | Prompt tokens | Response tokens | Authored tokens | Reward |
| --- | ---: | ---: | ---: | ---: |
| Unsuccessful | 1,915 | 3,198 | 668 | 0 |
| Successful | 1,825 | 3,362 | 696 | 1 |

Both clean cases completed archival and UID/backend closure. These are actual
online trajectory-retention observations, not a training update or an
estimate of test accuracy. The 29-check review is retained at
`diagnostics/independent_first_completed_initial_cases_1794456991920584346.json`
under the v4 run, SHA256
`4fcf66b40480f571f98910ee5ec048b12c828262841e9800ca2e5bab6589fcb9`.
The retained-account inventory is
`diagnostics/read_only_retained_account_inventory_1794457028540090688.json`,
SHA256 `a7a61c1d98ea09228ef66ffbba3fa7a4ff172b38734f2f55356dd49203f3dd0e`.

## Repairs and fresh experiment

`5e099e5f` preserves pidfd-owned signaling using Python APIs, libc wrappers or
the verified x86_64 kernel syscalls. It never substitutes numeric-PID signals.
Actual direct-helper acceptance passed on the production interpreter.

The old NAS registry subsequently blocked administration while waiting for
Ceph distributed-file-lock completion. No granted local holder was visible;
an external holder and an MDS delay could not be distinguished. No old lock
was deleted or replaced. `d8b3de2d` introduced a fresh host-local registry with
128 disjoint UID/GID 20000-series identities. All old 19000-series identities,
leases and quarantine evidence remain retained. The new pool's real drain
and restricted Native preflight passed 23 checks; see
[account isolation](../../examples/skill_self_evolve/ACCOUNT_ISOLATION.md).

The fresh `alfworld_joint_full64_v5_20261112` experiment started under
`d8b3de2dc4a60171d1bbb0ea35d26884b1c54131`. It imports no v4 model, optimizer,
EMA, skill, replay, ledger, baseline or outcome. The full 64-update protocol,
32×N4 training, 4+2+2 GPU allocation, 900-second budget, 36,864-token capacity,
all 399 scheduled development games and complete 274-game initial/final
official evaluations remain predeclared.

Its plan SHA256 is
`58cd16a603fab15e3652123a1cccf68f9bbdf227c4be58ebdf76e83fd2fa9278`;
startup checks SHA256 is
`8d3d86b2e94c122bfc1867c707861052e59e13ea9288b2494bceaa2e2a21d63f`.
Launching v5 does not establish completed training or reward/accuracy gains.
