# Deployment requirements

## Installed dependencies

Use one installed training environment for the driver and Ray workers, with
verl, Uni-Agent, Torch, FSDP2, vLLM, Ray, TransferQueue, Hydra, PyArrow and the
model/tokenizer dependencies. The environment-engine interpreter needs
ALFWorld, TextWorld and aiohttp. Native Codex must be installed as an actual
executable with readable package assets. The configured Uni-Agent version must
support the native Codex agent, Responses Gateway trajectories, custom Task
registration and the empty effective tool set used by the curator.

Use `uv` to manage environments. Acquire source distributions, compatible
Linux GPU wheels, model weights and benchmark data on the networked local Mac
workstation, then transfer them to the Linux host. Prepare a complete wheelhouse
for the target Python, operating system, GPU stack and architecture. For
example, install transferred dependencies on the host without network access:

```bash
uv venv --python 3.12 "$TRAINING_ENV"
uv pip install --python "$TRAINING_ENV/bin/python" --offline --no-index \
  --find-links "$WHEELHOUSE" -e '.[fsdp,vllm]' "$UNI_AGENT_WHEEL"
```

Supply the environment and wheelhouse paths explicitly. The recipe neither
installs dependencies nor alters another environment's `.pth` files. Set
`TRAINING_PYTHON` to that environment's Python and ensure the same installed
packages are available to each worker. Prepare engine dependencies in the
explicit `ALFWORLD_ENV_PYTHON` environment in the same way.

## Data and service

The one-command [recipe launcher](../../examples/skill_self_evolve/README.md)
reads an explicit host deployment JSON, prepares fresh catalog and parquet files
with `verl/experimental/skill_harness/alfworld/prepare_data.py`, and supervises
both the environment service and trainer. The benchmark root must contain
existing official games under `data`; preparation does not retrieve data or
synthesize tasks. Each launch requires a new output directory. Full official
test partitions are preserved for final evaluation.

The backend uses four engine processes and four I/O workers by default. Its
private service configuration contains a loopback HTTP endpoint and
administration capability. The launcher waits for health to match its owned
backend PID, prepared catalog, process count and I/O count before training.
It records owned PIDs and Linux process start ticks in `launch-processes.json`.
On exit or a stop signal it stops its own children, preserving other runs.

The deployment declares public asset copies and an explicit `tool_dir` and
agent-readable `tool_python`. The launcher publishes the standard-library
`scripts/alfworld_request_client.py` and an `alfworld-step` command there. The
backend and public client share the action request-ID protocol. Only the
session URL and declared public assets are passed to native agents; protected
service capabilities stay in private run-local configuration.

Final evaluation starts a separate test-only backend with the configured
`skill_harness.evaluation.server_script` and optional `server_args`, using
`ALFWORLD_ENV_PYTHON`. The backend preserves the training action limit and
episode budget. Its endpoint is temporary; the existing training service is
not replaced or terminated.

## Native runtime provider

`ALFWORLD_RUNTIME_CONFIG` is a trusted JSON document containing
`account_pool.factory` and `account_pool.kwargs`. The factory is an importable
provider implementing the native lifecycle in `alfworld/accounts.py`. It owns
leases, process isolation, launch arguments, cleanup and capacity. Alternative
deployments can provide another factory through this interface without adding
a training entry point.

The optional factory
`verl.experimental.skill_harness.backends.linux:LinuxAccountPool` requires:

- `root`: absolute private registry directory;
- `roster`: explicit existing non-root account names, UIDs and GIDs;
- `work_root`: absolute native episode workspace root;
- `read_roots`: public executable, package, instruction and tool paths needed by
  the native agent.
- `command_path`: executable search path containing the deployed `alfworld-step`
  command and required system tools.
- `launcher_path`: optional absolute public copy of `verl/experimental/skill_harness/backends/linux.py`
  when leased accounts cannot read the checkout.

The optional provider lives in the installed `verl` package. Deployments may
instead install their own provider package and select its importable factory.

Provision the roster, directory ownership, required `setpriv` support and Linux
Landlock/seccomp facilities before launch. A deployment may explicitly set
`runtime.initialize_if_missing=true` for this provider. The launcher then calls
`AccountPool().initialize()` only when the configured registry root is absent.
Existing registry roots are never initialized, reset or reconciled. Alternatively,
initialize the provider explicitly during deployment setup. The harness does not
create Unix users, select a UID range or infer a host account layout. The
provider reuses only resources whose process and cleanup lifecycle completed.

Configure the curator's separate Unix account using `CURATOR_USER`, and its
readable public-assets and work directories using `CURATOR_PUBLIC_ROOT` and
`CURATOR_WORK_ROOT`. Solver and curator need access to their declared public
assets and isolated work areas. Benchmark files, administration capabilities,
other episodes and training checkpoints stay outside the agent's allowed
native workspace. Configure `CODEX_NATIVE_BIN`, `ALFWORLD_CODEX_PACKAGE_ROOT`
and the Uni-Agent task file with the actual installed locations.
The protected environment service configuration may also declare
`native_policy.instructions_file` and `native_policy.disabled_features`.
The task uses the configured instructions directly, with
`ALFWORLD_SOLVER_INSTRUCTIONS_FILE` as the explicit fallback, and preserves the
actual native binary and Uni-Agent prompt transport.

## Configuration checks

The [recipe](../../examples/skill_self_evolve/README.md) lists required
variables. Its CPU preflight checks dependency availability, resolves Hydra
configuration and verifies input files and the provider declaration. It starts
no agent, engine, model or optimizer. GPU count, memory, process limits and
provider capacity must match the explicit rollout, curator and credit
concurrency settings of the chosen deployment.
