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

Prepare catalog and parquet files using
`verl/experimental/skill_harness/alfworld/prepare_data.py`. The benchmark root
must contain existing official games under `data`; preparation does not
retrieve data or synthesize tasks. Use a fresh dataset output directory and the
same durable run directory that the recipe will use. Preserve the full official
test partitions for final evaluation.

Start the backend as a separately managed deployment process, using your
chosen interpreter and explicit paths:

```bash
"$ALFWORLD_ENV_PYTHON" "$ALFWORLD_ENV_SERVER" \
  --benchmark-root "$ALFWORLD_BENCHMARK_ROOT" --run-root "$SKILL_RUN_DIR" \
  --catalog-path "$ALFWORLD_DATA_DIR/catalog.json" \
  --config-path "$ALFWORLD_SERVER_CONFIG" --allowed-splits train \
  --environment-processes "$ENVIRONMENT_PROCESSES" --io-workers "$ENVIRONMENT_IO_WORKERS"
```

`ALFWORLD_ENV_SERVER` selects the generic
`examples/skill_self_evolve/scripts/parallel_environment_server.py` backend.
Its configuration file contains a loopback endpoint and administration
capability; keep it readable only by the trusted training process. It advertises
the action request-ID protocol used by the public request client. Publish
`scripts/alfworld_request_client.py` as `alfworld-step` on the native agent's
configured tool path. The runtime must provide that path and a compatible Python
interpreter to agent commands.

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
`examples.skill_self_evolve.runtime.linux_accounts:LinuxAccountPool` requires:

- `root`: absolute private registry directory;
- `roster`: explicit existing non-root account names, UIDs and GIDs;
- `work_root`: absolute native episode workspace root;
- `read_roots`: public executable, package, instruction and tool paths needed by
  the native agent.
- `command_path`: executable search path containing the deployed `alfworld-step`
  command and required system tools.
- `launcher_path`: optional absolute public copy of `runtime/linux_accounts.py`
  when leased accounts cannot read the checkout.

The recipe provider is imported from the checkout in the documented editable
deployment. A standalone verl wheel does not include the `examples` namespace;
wheel-only deployments must install a provider package and select its factory.

Provision the roster, directory ownership, required `setpriv` support and Linux
Landlock/seccomp facilities before launch. Initialize this provider explicitly
with `AccountPool().initialize()` during deployment setup. The harness does not
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
