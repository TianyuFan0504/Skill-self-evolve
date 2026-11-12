# Native launch in threaded Ray workers

`scripts/train_spawn_safe.py` composes the existing resilient setup and
quiescent final-evaluation drivers. It adds a narrowly matched Native process
launch through glibc `posix_spawn`, using public asyncio policy, event-loop and
transport interfaces. The training algorithm, optimizer, checkpoint format,
data sampler and existing core source fingerprint remain unchanged.

Python 3.11's standard subprocess path cannot use its fast spawn path with the
existing combination of a working directory, a new session and a passed ready
descriptor. A worker can consequently block before exec while starting a child
from a process that already has native threads. Because the launch call blocks
the event loop, a timer created after that call cannot stop this failure.

The extension recognizes only the exact existing `setpriv` / `isolate-exec`
command and required subprocess options. It preserves the UID/GID arguments,
working directory, environment, new session, stdout/stderr pipes and kernel
startup attestation. The ready descriptor is relocated to descriptor3; all
other descriptors above3 are closed by a glibc file action. Unrelated commands
retain the standard event-loop behavior. The C helper uses the installed
headers for opaque glibc structures and pidfd syscall constants.

## Build and CPU verification

Build on the target Linux host; the shared library is a generated local
artifact and must not be copied from a different platform:

```bash
cd /workspace/verl-0.9.1
cc -O2 -shared -fPIC -Wall -Wextra -Werror \
  examples/skill_self_evolve/scripts/native_posix_spawn.c \
  -o examples/skill_self_evolve/scripts/native_posix_spawn_helper.so

PYTHONPATH=/workspace/verl-0.9.1:/root/nas_chaohuang/benchmarks/alfworld/uni-agent \
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /opt/conda/bin/python \
  examples/skill_self_evolve/scripts/test_native_posix_spawn_cpu.py
```

The helper filename must end in `_helper.so`; naming it `native_posix_spawn.so`
would shadow the Python adapter during import. The CPU suite verifies actual
SDK runner construction before event-loop creation, descriptor closure,
session and working-directory behavior, streams under backpressure, exit
status, cancellation, concurrent launches and child reaping. It does not run
an ALFWorld episode or allocate a benchmark account.

The separate deployment diagnostic `test_native_posix_spawn_isolation.py`
allocates exactly one existing clean pool UID, exercises the production
`setpriv` / `isolate-exec` ready-pipe proof with a benign Python child, and uses
the normal drain, archive and release lifecycle. It creates no model, backend
session or ALFWorld world. This run-specific fixture also verifies that the
historical UID20071 quarantine stays byte-identical; it requires that protected
fixture to exist and is not a generic fresh-host installation test. A failure
retains only its own lease in quarantine rather than resetting the pool.

## Training integration

After preparing the existing joint recipe and environment, substitute
`scripts/train_spawn_safe.py` for the resilient driver in the same command.
Keep the original config path, config name, overrides and strict COMPLETE
checkpoint resume arguments. The driver installs the policy before creating
the learner's counterfactual loop. The public SDK runner class
`skill_harness_examples.train_spawn_safe.SpawnSafeRunner` installs it before
each episode loop is created. No live worker is patched.

The public launcher exposes this selection as `--spawn-safe`:

```bash
bash examples/skill_self_evolve/run_resilient.sh --spawn-safe \
  --run-dir /root/nas_chaohuang/skill_training/runs/my_joint_experiment
```

Build the helper first, and keep the flag when resuming this profile. See
[RESILIENT_LAUNCHER.md](RESILIENT_LAUNCHER.md) for the shared launcher options.

Each driver invocation writes `driver_runs/*.spawn_safe.json`; after trainer
initialization, `*.spawn_safe.initialized.json` records the effective config,
restored checkpoint, loop/policy classes, composed driver hashes, helper source
and binary hashes, and the original manifest hash. Final evaluation retains
the existing sealed model/library lifecycle. Matched reference evaluation
must use the same runner, helper identity and actual concurrency profile.

The extension does not reconcile old incidents, release quarantined accounts,
assign rewards to failed infrastructure attempts or bypass checkpoint
compatibility checks. Those operations require separate evidence about the
exact affected resources. An invalid pre-exec attempt remains excluded from
training even after its resources have been closed.
