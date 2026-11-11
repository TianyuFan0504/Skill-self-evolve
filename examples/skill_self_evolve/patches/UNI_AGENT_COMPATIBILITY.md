# Pinned Uni-Agent Responses compatibility

The fixed [29-line patch](uni_agent_effective_no_tools.patch) permits
`parallel_tool_calls=false` only when the provider-normalized request has no
effective tools. It includes `additional_tools` and respects `tool_choice=none`.
Requests with effective tools remain rejected. The original payload value is
preserved; sampling, token IDs, logprobs, stop reasons and structured-output
support are unchanged. A catalog flag alone does not prove that tools are absent.

Only these exact external adapter hashes are recognized:

| State | SHA-256 |
| --- | --- |
| Pristine | `a71653b73a19ff4e501871f6eaf458a06bd00f7cc00070093c6997287635ee1f` |
| Patched | `573a7c74221189e377d2b2db631ac3b2fc1802453e642284173fc85ad42138fa` |

The installer contains literal minimal replacements, not a vendored adapter.
Unknown source or backup bytes are rejected. Verification never writes anything.

## Explicit offline application

First stop or reconcile every worker using the checkout. Do not modify the
dependency beneath a live run. Retain the old run/checkpoints as historical
evidence and use fresh interpreter processes after application.

Use the base interpreter and file-path CLI, which imports only the standard
library and performs no network request:

```bash
/opt/conda/bin/python \
  /workspace/verl-0.9.1/verl/experimental/skill_harness/uniagent_compat.py \
  --uni-agent-root /root/nas_chaohuang/benchmarks/alfworld/uni-agent --apply
```

The explicit apply creates a durable pristine `.bak`, preserves adapter
permissions/ownership, and atomically publishes only the approved target bytes.
Repeated application is idempotent. Omitting `--apply` verifies the patched
state and prints provenance; it never installs or edits dependencies.

The launcher only verifies and reports this command when remediation is needed.
Task registration and Ray worker bootstrap require the patched adapter.
Non-task CPU imports may record the pristine/missing adapter so lightweight
tests remain usable. Checkpoint identity independently measures the actual
imported dependency's adapter SHA and includes it in source/version fingerprints;
`register_tasks=False` does not erase that identity. A changed adapter cannot
reuse initialized checkpoint compatibility.

## Offline tests

Installer/runtime tests use temporary checkouts and no GPU, Ray startup,
package installation or dependency downloads. Supply the independently archived
pristine file to include the exact pinned-source cases:

```bash
SKILL_UNIAGENT_PRISTINE_ADAPTER=/path/to/pristine/openai_response.py \
  python -m unittest discover -s tests/experimental/skill_harness \
    -p test_uniagent_compat.py -v
```

The external adapter semantic tests are separate: they validate empty/disabled
tools, `additional_tools`, preserved false payload values, rejection with tools,
and unchanged numeric TITO/streaming behavior. No installer test authorizes
application to a live server.
