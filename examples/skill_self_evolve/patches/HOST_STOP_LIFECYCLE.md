# Bound host stops and early first-input evidence

A native CLI deadline previously killed only the CLI, leaving its outstanding
backend request active. Host stops now close admission before sending a CLI
signal. The existing private, capability-bound fence records a `stop_kind`:
`mdp_terminal`, `budget_timeout`, `task_cancel`, or `host_error`. The first fence
is immutable. A host stop has no terminal environment state or state hash and
cannot later be upgraded into an absorbing-MDP proof.

`AlfworldUserSandbox` writes the bound stop before timeout/cancellation signals.
Task cancellation and host Python errors also fence before UID process drain.
The Gateway watcher refuses new submissions and retries abort only for the
bound request's actual backend UUID, with prefix-cache reset disabled. The
original generate future remains owned across frontend cancellation. An abort
acknowledgement alone never establishes drain: the original future must resolve.

Controlled partial outputs retain exact prompt IDs, output IDs, behavior
logprobs, raw stop reason and extra fields in private archive-only tails. They
are not returned through a codec as synthetic EOS or ordinary policy output.
Backend exceptions and unknown raw stops produce an unconfirmed/backend-failed
drain receipt; they cannot become a successful zero-reward outcome. Existing
MDP-terminal flags remain separate from `expected_host_stop` and `stop_kind`.

For the first admitted request, a normal `completed` backend output publishes
`binding.directory/first-observed-input.json` before returning to the Gateway.
The private proof contains the actual Client `prompt_ids`, actual request UUID,
input/output hashes and bound attempt identities. It does not retokenize text.
Submission records, aborted outputs and later requests cannot substitute for
this admission-0 proof. `read_first_observed_input(binding)` returns its path,
file SHA and verified payload, or `None` if observation is unproven. The proof
is evidence only: `archive_only=true`, `trajectory_complete=false`, and
`behavior_gradient=false`. A later Task failure does not remove it.

Task results retain native exit/error/finished fields and expose host-stop,
budget-exhaustion and cancellation flags. `budget_reward_eligible` remains
false in this module. This change does not authorize timeout counterfactual
zeros, change the 900-second budget, fabricate environment `done`, alter data,
or rehabilitate previously quarantined leases. Server-side action deadlines
and a shared budget-outcome verifier are a separate module.

The subsequent [budget protocol](../../../docs/skill_self_evolve/native_budget_protocol.md)
implements that separate verifier. Only its independently proven deadline
failure can set `budget_reward_eligible=true`; the raw Native completion and
error fields remain unchanged. The [real backend cancellation test](../../../docs/skill_self_evolve/native_backend_cancel_contract.md)
records transport acceptance independently of any ALFWorld reward claim.

CPU checks cover the existing MDP races, host-only UUID cancellation, abort ACK
without drain, frontend cancellation, durable actual first-input evidence,
backend errors/unknown stops, immutable stop kinds, and fence-before-CLI-signal
ordering. The local lightweight runner extracts the actual `LLMServerClient`
class from repository source to avoid optional Torch/Ray imports; it does not
replace its implementation. Full dependency-stack CPU and real backend
acceptance are performed separately by the operator.

## Unexpected RPC failure input evidence

The v5 engine watchdog dump anonymized prompt token values, so it could not
support exact failed-input replay. Future unexpected backend RPC exceptions
retain the exact copied client `prompt_ids` and `sampling_params` under
`binding.directory/failed-inputs/<request_nonce>.json`. The private artifact
binds the attempt, Gateway/environment sessions, admission index and assigned
backend UUID/server when available. Separate canonical digests cover the
tokens and sampling parameters; its payload digest is linked into the existing
failure request journal as `failed_input_ref`.

This is explicitly `CLIENT_INPUT_PRE_SUBMIT` evidence. Sampling parameters are
the client's values before backend defaults or clamping, and
`actual_model_observed=false` means observation is unproven. It does not capture
KV-cache state, effective kernel inputs or an exact scheduler replay. The
artifact is archive-only, incomplete, reward-ineligible and excluded from
gradients, with no invented EOS. It never substitutes for the completed
admission-zero `first-observed-input.json` proof.

Normal successful requests add no archival I/O. Controlled MDP and host stops
keep their existing raw-tail path. Unexpected errors are not retried. A failed
input archive or subsequent failure-journal write cannot replace the original
backend exception: secondary errors are attached as exception notes and, when
the journal remains writable, `failed_input_archive_error` diagnostics. An
incomplete write never becomes positive observation or successful drainage.

CPU regressions inject an engine failure and an acquisition RPC failure,
verify exact tokens/parameters and bound digests, retain the original exception
through archive/journal `ENOSPC`, and confirm that successful or controlled-stop
paths do not publish these failure inputs. These checks do not rehabilitate
v5's failed cases or establish that the next backend configuration is fixed.
