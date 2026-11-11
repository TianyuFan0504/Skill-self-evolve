# Read-only training evidence index

Run the standard-library reviewer aid on the machine that can read the run's
absolute artifact paths:

```bash
python examples/skill_self_evolve/scripts/audit_training_run.py /path/to/run \
  --output /path/to/new-audit-report.json
```

The output path must not already exist. The script never launches evaluation,
imports the model/runtime packages, or modifies a raw artifact. It is an evidence
index for human review, not an acceptance oracle. Tests use synthetic evidence;
they do not demonstrate benchmark performance.

## Official evaluation joins

For current runs, each `evaluation/{initial,final}/summary.json` explicitly maps
`case_audits.test_seen` and `case_audits.test_unseen` to their scoped case/session
reports. The index follows those exact paths and the report's exact `dump_path`;
it never selects a latest report or merges unrelated retries. A phase
`metrics.json` can expose partial-stage evidence, but cannot replace a completed
stage summary.

The core validation JSONL legitimately contains no raw Task pointer. Its
`uid = rollout_uid_session_index_prefix_index` joins to `cases[].trajectory_keys`
in the sidecar. The index independently checks:

- Stage, official phase, host cursor, audit ID, UID/session, attempt ID, and the
  full expected set of 140 seen or 134 unseen game identities.
- Raw Task and private engine artifacts, including environment transitions and
  binary engine `won`, against per-prefix score, reward, and accuracy.
- A linked native Gateway trajectory with actual numeric prompt/response arrays
  and a positive sampled-action mask. A sidecar status is insufficient evidence.

Prefix rows remain one game. Missing Task evidence or missing numeric trajectories
remain unobserved; neither becomes a zero. Development dumps in `validation/`
cannot establish an official test. Older dumps with explicit Task pointers remain
readable, but do not gain the stronger scoped sidecar identity checks.

## Native termination and clocks

A proven environment-terminal host/backend stop can retain a nonzero CLI exit,
native error, and no `turn.completed`. The index preserves these original values
and accepts the stop only with matching binding identities, drained/fenced
backend evidence, terminal-state hash, and private engine agreement. Unexpected
native errors remain separate quality issues; an observed engine win is never
silently rewritten as a loss or removed to improve the denominator.

The parent's validation cursor is not an optimizer-update counter. Initial
evaluation can report cursor 1 while the model still has weight version 0. The
report retains observed weight tags separately and never infers updates from a
cursor alone. Failed attempts with repeated training cursors remain ambiguous.

A positive controlled comparison requires complete initial and final 140+134
case evidence and identical recorded datasets/budgets. A rising training reward
on different tasks, a manifest, or passing synthetic tests is not a substitute.
Live files that change during reading require another audit after a durable cut.

```bash
python examples/skill_self_evolve/tests/test_run_audit.py -q
```
