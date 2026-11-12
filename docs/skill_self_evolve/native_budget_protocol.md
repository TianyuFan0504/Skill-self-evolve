# Native episode budget and observed failure

The ALFWorld episode protocol declares a 900-second wall budget. A Native CLI
timeout alone is not an observed task failure: it can leave generation running,
race with a successful environment action, or conceal an infrastructure error.
The Task, counterfactual probes and official case audit therefore share
`alfworld.budget.verify_budget_failure`. Missing or contradictory evidence raises
`BudgetOutcomeUnknown`; it never produces an implicit zero reward.

## Clock and action ordering

Immediately before Native launch, the host registers the exact environment
session, backend attempt binding and UID lease through the private authenticated
`POST /admin/budget/{session}/start` endpoint. Its monotonic registration and
deadline are durable, and retries return the original registration without
extending the clock. The Native process wait uses the remaining time from this
same deadline. The production budget remains 900 seconds.

An action must enter the serialized TextWorld engine before the deadline. An
admitted action finishes under the session lock before an observation or close
can inspect the world. Admission is checked again after waiting for the engine
executor; queueing before the deadline does not authorize late execution.
Handler cancellation waits for the actual engine thread and records an unknown
handler outcome, rather than releasing the lock while the world can change.

The private ordered-state endpoint is read-only before the deadline. Its first
postdeadline view is persisted immutably; subsequent reads return that exact
view even if rejected action attempts append later journal entries. The host
stop reason binds the registration and this view's digest. Budget clocks and
administration credentials are absent from model observations.

If a predeadline action wins or reaches an actual MDP terminal state before the
ordered view, the host retains that world outcome and uses the existing MDP
terminal fence. An unfinished world with a verified deadline instead uses a
typed `budget_timeout` fence. It does not set world `done`, change a win, or
invent Native completion or an EOS token.

## Evidence required before score zero

The shared verifier checks the declared clock and private registration, raw
Native timeout diagnostics, absence of unexplained Native transport errors,
the ordered admission journal and unchanged unsuccessful world state. It also
checks the actual bound stop, every admitted backend RPC and its completed
original future, exact request ownership, archive-only stopped tails and their
unaltered token/probability arrays, UID/backend drain, durable native archive
hashes and removal of all three owned temporary roots. An abort acknowledgment
alone is insufficient.

The Task retains raw Native error, exit and `native_finished` independently.
`TaskResult.finished` describes completion of the host episode lifecycle; it
does not assert that Native ended naturally. A proven budget failure has binary
reward/accuracy zero with an explicit private proof. Both counterfactual probes
and official evaluation independently recheck that proof before accepting it.
An observed zero stays in the official benchmark denominator.

An unknown solver outcome records a run-level incident and prevents a partial
GRPO group from training. An unknown counterfactual outcome can become
`ProbeNotObserved` only after the Task has proved safe UID/backend/archive
closure. It creates no `UsageEvent`, marginal utility or curate advantage.
Failed infrastructure or incomplete cleanup remains fatal/quarantined.

## Changes and validation scope

`alfworld/server.py` owns the registration, ordered state, action journal and
deadline admission. `alfworld/budget.py` owns the common pure verifier.
`alfworld/task.py` registers the clock, requests the typed stop and admits a
reward only after closure. `probes.py` handles verified budget outcomes and safe
censoring; `validation_audit.py` reconciles independently verified budget cases
with the complete official denominator.

CPU regressions exercise synthetic proof rejection, retained raw stopped-tail
tokens, an official-audit forged closure, consumer dispatch/censoring, immutable
observations, executor admission delay and cancellation/terminal races. They are
engineering checks, not benchmark performance results. Existing pre-protocol
failed attempts are not retrospectively relabeled. A fresh real Native pair
and complete training experiment are still required for runtime acceptance and
any claim of reward or accuracy improvement.

The deployed server environment subsequently passed **108 CPU checks with no
skips**: 90 budget, Task, archival, paired-input and failure-capture checks,
plus 18 official-case-audit and host-stop regressions. These include the actual
aiohttp service methods and the installed Uni-Agent Task path, using synthetic
engine/proof fixtures. Logs are retained under
`/root/nas_chaohuang/skill_training/diagnostics/module61_cpu_1794454263866438237`.
The separate [real backend cancellation contract](native_backend_cancel_contract.md)
passed on Qwen3.5-4B; it supplies transport evidence, not real ALFWorld budget
or training evidence.
