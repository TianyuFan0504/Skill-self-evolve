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

CPU checks cover the existing MDP races, host-only UUID cancellation, abort ACK
without drain, frontend cancellation, durable actual first-input evidence,
backend errors/unknown stops, immutable stop kinds, and fence-before-CLI-signal
ordering. The local lightweight runner extracts the actual `LLMServerClient`
class from repository source to avoid optional Torch/Ray imports; it does not
replace its implementation. Full dependency-stack CPU and real backend
acceptance are performed separately by the operator.
