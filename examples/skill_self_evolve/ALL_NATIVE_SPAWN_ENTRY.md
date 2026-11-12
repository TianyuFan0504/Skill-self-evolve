# Solver and curator spawn entry

`scripts/train_all_native_spawn.py` composes the existing direct-read restore,
passive diagnostics, parallel environment/client and frozen training/final-test
components. It installs the new `all_native_spawn` policy **before** constructing
the trainer, so the learner's controller loop can launch both native shapes:

* Solver and counterfactual tasks retain the original setpriv/isolate-exec
  helper, ready descriptor3, UID/kernel attestation and pidfd transport.
* Curator tasks retain their original runuser user switch, argv, script,
  working directory and clean environment. A separately compiled three-stdio
  helper closes every descriptor above2; no artificial ready descriptor is
  added to this shape.

After trainer initialization and before `fit`, the entry registers the actual
`NativeCuratorRuntime` account, UID/GID, work root and package root. Custom
non-root curator accounts are supported. Unknown curator-shaped requests fail
before execution rather than silently returning to the fork path. Unrelated
commands retain the standard event-loop implementation. The pure matcher,
stdio/process tests and owned runuser CPU handshake are documented with the
spawn provider; no model or account-pool lease is needed for them.

The primary SDK episode-worker FQN remains `SpawnSafeRunner`. Those workers run
solver tasks and keep the original solver-only policy. The learner's new loop
handles curator and counterfactual work and is recorded with its actual class
name; no module identity is spoofed.

## Evidence and component scope

Prelaunch records are `driver_runs/*.all_native_spawn.json` with kind
`prelaunch_all_native_spawn_parallel_components`. They identify the actual new
entry, the direct/parallel/spawn source components, both helpers and the
generated client/service assets. The prelaunch driver has no registered learner
curator runtime yet and does not claim a post-init or model-serving result.

`*.all_native_spawn.initialized.json` records the actual learner loop/policy,
registered curator runtime, effective config, original manifest hash, restored
checkpoint and direct-read restore receipt. The entry does **not** invoke or
invent the old solver-only initialized receipt, which rejects the new loop.

The frozen final-fit method remains the code performing checkpoint sealing and
official tests. Its own source identity is a component record; each lifecycle
record's extension and initialized reference identify this actual outer entry.
Offline comparison must explicitly understand this new operational chain; an
older comparer must not be made to accept it by relabeling an old receipt.

This entry changes process transport only. It does not change model weights,
instructions, sampling, rewards, optimizer coordination, checkpoint contents,
Native budgets, or test-library sealing. The original core fingerprint still
describes the unchanged core; the new source/helper identities supplement it.
