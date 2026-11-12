# Native input-control v6: budget cleanup remained unproven

The real diagnostic `alfworld_native_pair_input_v6_20261112` used the archived
v4 S0 and its original future training task B. Its source cut was
`6a5a9e8563ea8138708f0429a22906e797e89b05`. It created no learner or optimizer
and changed no benchmark task, skill content, 900-second case budget or token
capacity.

The first `without_skill` branch ended normally with reward zero after 36 game
actions. Its original numeric capture and complete native archive were retained.
An independent full archive readback checked all 87 source members.

The `with_skill` branch first emitted invalid bare shell commands, then resumed
real `alfworld-step` actions and reached step 34. At 900.078 seconds its actual
native deadline terminated the CLI: the raw native exit was -15 and the
returned timeout exit was -1. It had not reached an absorbing environment state.
The private backend journal still contained one owned request with no abort
attempt and no confirmed RPC completion. The host had stopped its CLI without
fencing/cancelling that outstanding backend request.

The UID/filesystem boundary behaved conservatively: no second-branch source
was archived or removed, the paired control marker was marked failed, and
UID/GID 19001 remained quarantined. The two branches had used the same lease,
identity and controlled cwd, with distinct native HOMEs. Global cleanup later
stopped the owned Ray/environment services and released the selected GPU;
that did not retrospectively establish the missing per-attempt closure proof.
The failed identity was not automatically returned to the pool.

The first branch's reward is an observed task outcome. The second branch's raw
engine result contains zero, but its unknown closure cannot become a verified
counterfactual zero. No paired event, utility label or skill admission was
produced. The second branch's numeric first prompt was not retained before the
Task exception skipped Gateway finalization. Therefore actual native only-S
control was **unobserved**, rather than passed or mismatched.

Independent host-message comparison found exactly one differing 547-character
skill block without path normalization. Both private initial world states had
hash `f54845611dafa8cd4567a0788211abbe5cda5d9c75243155b926347f3bcc9229`.
Those facts cannot substitute for the missing actual native-token comparison.
Missing native weight clocks remain `None`; no observed version was invented.

Evidence under the private run root:

| Artifact | SHA256 |
| --- | --- |
| `diagnostics/pair_lease_independent_review_1794451683139461307.json` | `1b1cf44d6660fbba298a6ba388911d279d3be8262a44a649cf0752a835609f96` |
| `diagnostics/independent_failed_v6_input_audit_1794451846619034283.json` | `cdf441fb32fd464ff00fc4f3dbc3ea1b4293f9767e09520756dc0748770d831a` |

The fixes following this failure add a typed host-stop fence before CLI signals,
targeted cancellation and original-future drain, early first-observed-input
evidence, and bounded failure capture before Gateway abort. Explicit budget
scoring additionally requires private server action-admission/clock evidence
and verified closure. Neither fix rehabilitates this failed run. The prepared
full64 v4 experiment remains unstarted pending new real acceptance evidence.
