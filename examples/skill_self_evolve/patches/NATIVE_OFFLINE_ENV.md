# Native offline environment propagation

The v6 review found that `AlfworldUserSandbox._exec` rebuilt a clean process
environment without the five package/data offline flags set by bootstrap.
Recorded ordinary pip commands downloaded and installed packages without
explicitly clearing those flags. The original events and audit remain intact:
`native_offline_env_pip_gap_1794464573832709015.json`, SHA256
`bb3982dc48b4f95f331b3a6323d76c7086feb43a25f67653a7ce985a6e51c2a2`.

The Native launcher now sets `PIP_NO_INDEX`, `UV_OFFLINE`, `HF_HUB_OFFLINE`,
`HF_DATASETS_OFFLINE` and `TRANSFORMERS_OFFLINE` to `1` after caller and controlled
environment merges. Bootstrap currently defines these inline rather than as an
exported constant. Local Gateway/world addresses and `NO_PROXY` are unchanged.

All ten Native capture CPU checks passed locally. Actual spawned children
received all five flags even when the caller supplied `0`. A real pip dry-run
against a reachable fake loopback package index made zero index requests and
found no distribution; it installed nothing and contacted no external site.

These are package-client environment controls, not kernel network isolation or
a guarantee against explicitly overridden clients or arbitrary network tools.
No model/data input, reward, timeout or training/evaluation priority changed.
