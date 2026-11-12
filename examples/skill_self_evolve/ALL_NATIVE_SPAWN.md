# Fork-free solver and curator process transport

The captured curator stall occurred before exec in Python's subprocess path.
The earlier solver-only policy recognized `setpriv` / `isolate-exec`, while
`CuratorSandbox.exec_shell` launches `runuser -u USER -- /bin/bash -c SCRIPT`
with a working directory and a new session. That legitimate second shape fell
back to `Popen`. Its exec-error pipe could also be inherited by concurrent
fork-context DataLoader workers, extending a constructor wait after the original
child had exited.

The new examples-only `all_native_spawn.py` handles both shapes:

* Solver/counterfactual launches still use the original four-descriptor helper,
  including ready descriptor3 and the existing isolation/UID attestation.
* Curator launches use the separate `curator_posix_spawn_helper.so`, built from
  `curator_posix_spawn.c`. It duplicates only stdin/stdout/stderr, closes all
  descriptors above2, preserves cwd/new-session/signal behavior, and executes
  the original runuser argv and input environment. No dummy ready descriptor
  or shell-script rewrite is used.

The learner registers its actual `NativeCuratorRuntime` before starting the
controller. Custom non-root accounts are supported; the exact account UID/GID,
host-owned work root, collection directory, private home and eight-key clean
environment must match. Curator-shaped calls with no registered profile fail
before launching. Unrelated commands retain normal routing. The script itself
is passed unchanged; it is not logged, decoded or interpreted by the matcher.

## Parent descriptor lifetime and concurrent forks

The policy installs one public `os.register_at_fork` exclusion lock before the
trainer/DataLoader is constructed. Native stdio creation, the C spawn call and
parent writer closure take this lock. No await occurs inside it; parent writers
close before asynchronous reader attachment. A concurrent Python `os.fork`
therefore cannot retain these transient writers and delay capture EOF.

This protects the observed Python DataLoader fork boundary, not arbitrary
forks inside third-party C libraries. The solver's pre-existing ready pipe is
created by its frozen caller before this transport boundary; it is not claimed
to be covered by the parent-stdio lock. A successful launcher proof is read as
data rather than waiting for ready-pipe EOF.

## Build and verification

Build the new helper on the target Linux host, keeping the loaded original
solver helper unchanged:

```bash
cc -std=c11 -O2 -shared -fPIC -Wall -Wextra -Werror \
  examples/skill_self_evolve/scripts/curator_posix_spawn.c \
  -o examples/skill_self_evolve/scripts/curator_posix_spawn_helper.so
```

Pure matcher/entry tests create no processes, accounts, models or worlds:

```bash
python examples/skill_self_evolve/tests/test_all_native_spawn.py -q
python examples/skill_self_evolve/tests/test_train_all_native_spawn.py -q
```

The separate Linux CPU script tests real stdio/cwd/environment/session behavior,
large outputs, cancellation, pidfd signaling/reaping, errors, concurrent children,
restored signals and a concurrent Python fork while a real C spawn is in flight.
It checks that native capture EOF arrives while the unrelated fork child remains
alive. A four-descriptor regression checks the original solver ready pipe through
the new parent transport; it is not a new kernel-UID-attestation claim.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /opt/conda/bin/python \
  examples/skill_self_evolve/scripts/test_all_native_spawn_linux.py \
  --output /tmp/a-new-private-diagnostic/result.json
```

An explicitly requested `--production-curator-user USER` additionally runs one
benign Python child through the exact public runuser matcher, only if that
existing non-root account is idle. It creates a fresh controlled collection/home,
uses no model/gateway/world or account-pool lease, and never performs UID-wide
termination. Runuser/PAM may add or normalize its ordinary HOME/USER/LOGNAME
fields just as before; the host's clean input environment remains unchanged.

The syscall/transport evidence is CPU-only. It does not establish training
accuracy, full-rollout speedup, or that every possible host fork is excluded.
See [ALL_NATIVE_SPAWN_ENTRY.md](ALL_NATIVE_SPAWN_ENTRY.md) for the truthful
new outer-entry/loop/helper receipts and frozen-component lifecycle scope.
