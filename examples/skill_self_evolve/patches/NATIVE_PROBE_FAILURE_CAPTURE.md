# Native probe failure capture

A failed Task must remain a failed counterfactual branch. Previously its
exception skipped Gateway finalization, and the `finally` abort cleared all
committed chains. A second UID-closure exception could also replace the original
Task failure. This change preserves the original exception and bounded evidence
collection. It does not change credit eligibility, task budgets or outcomes.

The pinned public call chain is `GatewayManager.finalize_session` →
`GatewayActor.finalize_session` → `GatewaySession.finalize`. The session takes
its request lock and materializes committed chains. It does not wait for the
backend, and it cannot recover an uncommitted first request. Conversely,
`GatewaySession.abort` clears active and materialized chains. Neither operation
proves GPU drain; the separate owned-request fence/drain protocol remains
responsible for that proof.

On a Task or finalization exception, `native_probe_branch`:

1. Publishes immutable, fsynced `native-failure.json` with the original error,
   traceback and cause chain, before attempting any Gateway cleanup.
2. If finalization has not already been attempted, waits at most 10 seconds for
   public finalization. An already timed-out finalization is not duplicated.
3. Publishes `native-failure-capture.json` containing only exact returned
   Gateway trajectories. Its top-level `finished` is false, `reward` and
   `accuracy` are null, and `archive_only` is true. Individual Gateway rows,
   including their IDs, masks, optional probabilities and original completion
   metadata, are preserved without edits. Empty returned results are explicitly
   `empty`; unavailable, timed-out or unreadable results are `unknown`.
4. If finalization did not return, waits at most 5 seconds for public abort.
   A returned finalization already removes the session and manager route, so it
   is not followed by an invalid second operation on that route. Abort errors
   are separately archived with an unknown backend-stop status.
5. Rethrows the same original exception, including `CancelledError`. Secondary
   capture, I/O, abort and pair-account closure failures become exception notes
   and, when writable, independent failure artifacts. Account closure failure
   remains fatal when no earlier failure exists.

The timeouts bound the caller's await, rather than treating local cancellation
as a remote acknowledgment. A remote operation may still be unresolved after a
timeout; this is recorded as unknown. No failure archive is named
`native-capture.json`, returned as a successful branch, retried as a fresh pair,
or converted to `UsageEvent`. No zero reward, model EOS, response token,
probability or re-encoded prompt is invented. The frozen policy attestation in
a failure archive is explicitly a claim, not evidence that a pending request
was model-observed. A separate first-observed-input artifact may prove a
completed backend output; it does not become a fabricated committed trajectory.

CPU regression tests use the actual pinned `Trajectory` dataclass and unchanged
public manager/actor/session lifecycle method bodies. Only their transport and
already-committed buffers are fixtures; no Ray runtime, HTTP service, native
process, GPU, model or benchmark task is started. The verified offline source
may be supplied with `UNI_AGENT_SOURCE_ROOT`:

```bash
python -m unittest discover -s tests/experimental/skill_harness \
  -p test_probe_failure_capture.py -v
python -m unittest discover -s tests/experimental/skill_harness \
  -p test_probes.py -v
```

These tests establish failure archival and exception preservation only. They
do not establish successful native decoding, backend cancellation or learning.
