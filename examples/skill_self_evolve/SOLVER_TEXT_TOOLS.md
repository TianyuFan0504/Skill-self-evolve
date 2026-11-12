# Native solver text tool boundary

The ALFWorld solver uses the original Codex 0.167.0 CLI with terminal tools to
interact with the text environment through `alfworld-step`. The solver wrapper
explicitly enables `shell_tool`, `unified_exec`, and `unified_exec_tty`, and
disables multi-agent, image, browser, apps, plugins, hooks, skill search,
workspace dependency, sleep, and goal tools. Web search is disabled and
`tools.experimental_request_user_input.enabled=false`. Host skill instructions
remain disabled; skills supplied by this harness still enter through its
versioned host context builder.

The generated native model catalog declares `input_modalities: ["text"]`.
This is a supported `ModelInfo` field in the pinned
[Codex 0.167.0 source](https://github.com/openai/codex/blob/rust-v0.167.0/codex-rs/protocol/src/openai_models.rs#L480).
The enum uses lowercase wire values; omitting the field defaults to accepting
both text and images. The capacity, truncation policy, native model identity,
parallel-call declaration, benchmark data, and training budgets are unchanged.

This boundary prevents unrelated native tools from being advertised. It does
not isolate the filesystem reached through shell commands. Production runs
also require the separate per-attempt Unix-account isolation change. Both
changes must be deployed together in a fresh run; do not reuse the earlier
partially trained run as a clean baseline.

## Verification

Run the portable CPU regression checks:

```bash
python examples/skill_self_evolve/tests/test_solver_native_tools.py -q
```

An independent CPU capture used the actual installed Codex 0.167.0 binary and
the updated wrapper under a nonroot account. A loopback HTTP endpoint recorded
the original first POST and returned `400 CAPTURE_ONLY_NO_MODEL`; no model,
environment episode, response tokens, likelihoods, or training update ran.
The native process exited with code 1 as expected. The actual effective tool
set was exactly `exec_command` and `write_stdin`; all four input messages
contained only `input_text`, and the original `parallel_tool_calls` value was
`true`.

The capture artifacts are under
`/root/nas_chaohuang/skill_training/diagnostics/solver_text_tools_module55_20261112/capture`:

- `capture-summary.json`, SHA256
  `c15b08d626a51be9959c5e4616bf4af72c039bfa8d192e65e7a79ca251e4b702`
- `post-000.body.json`, SHA256
  `9454d80386fe78a83babc7f2e30199bf13c21b4188a5324d0daf961291e4749b`
- `launched.json` records the exact argv and owned process 873665.
- `purpose.json` records native and wrapper hashes; `catalog.json` preserves
  the exact capture catalog. `stdout.jsonl` and `stderr.log` preserve failure
  output without presenting it as successful rollout evidence.

The capture verifies request declarations only. It is not an ALFWorld accuracy
test, evidence of successful multimodal processing, or evidence of improved
training.
