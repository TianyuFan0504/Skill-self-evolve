# Request IDs and the parallel environment client

`scripts/train_parallel_client.py` prepares a new, immutable public client
bundle and delegates training to the frozen spawn-safe entrypoint. It changes
only the original Task profile's `agent.codex_bin` field. The solver instruction
bytes, Native arguments, action strings and successful state JSON remain the
same. The installed `/opt/alfworld-tools/alfworld-step` is not overwritten.

The bundle exposes `alfworld-step` through a symlink to its `codex-entrypoint`.
Both names resolve to the same host-owned inode already permitted by the
existing Landlock boundary. The entrypoint prepends that public directory to
`PATH` before forwarding the original Native shell launcher.

Each logical action receives one UUID request ID. The client first requires
the server's `alfworld-step-request-id-v1` capability. After a lost response it
queries the original request, polls a pending result, and resubmits the exact
same ID/action only after an explicit unknown-ID response. Cached errors are
final. A server-instance change stops recovery instead of executing an
ambiguous action again. The tool does not assign rewards or extend the host's
900-second episode budget. Successful requests produce only the original
state JSON; timing diagnostics require `ALFWORLD_REQUEST_TRACE=1`.

The separate `environment-python` executable preserves the original benchmark
interpreter invocation path, including its virtual environment. When the
frozen evaluator invokes its exact `alfworld/server.py` path, this executable
substitutes the reviewed `parallel_environment_server.py` and appends four
environment processes and four I/O workers. All remaining arguments, official
test splits and limits are preserved. It uses `exec`, retaining the owned
service PID. Other scripts and already-parallel invocations pass through
unchanged. This lets training and official evaluation use the same protocol
without copying the evaluator or changing its globals.

The outer entry requires a ready service with four environment processes,
four I/O workers, the request-ID capability and the expected extension policy.
It checks the live health identity against the environment configuration and
the actual extension source bytes. An immutable
`driver_runs/*.parallel_client.json` observer records these facts, the source
and public-asset hashes, the original interpreter identity and the effective
configuration before delegation. Its hash reference is printed once. This is
prelaunch metadata, not a post-initialization or served-model receipt. The
unchanged core fingerprint describes the legacy core component; it does not
claim that the new environment implementation is unchanged. The immutable Task
profile path/hash links this observer to the later initialized configuration.

Use the reviewed launcher's `--parallel-environment` mode to combine this
entry with the four-process service and spawn-safe launcher. The provisioning
function is `prepare_adapter(run, task_profile, instructions,
environment_python=..., server_script=...)`; it returns immutable asset
references and the three environment-variable changes needed by the entry.
An eventual untouched-base reference must use the same service, client,
interpreter adapter and actual inference profile.

Run the CPU fixtures with:

```bash
python examples/skill_self_evolve/scripts/test_alfworld_request_client.py
```

They exercise dropped-response recovery over real loopback HTTP, stable IDs,
cached failures, deadlines, the public wrapper's command resolution, exact
interpreter argument translation and an owned tiny test-only HTTP service.
They do not launch models, ALFWorld worlds or benchmark UID leases. The Native
shell-path check requires Linux's `/usr/bin/bash` and is skipped on macOS.
