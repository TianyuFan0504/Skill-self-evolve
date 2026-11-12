# Local raw-bundle failures

Raw Native HOME/catalog/workspace bundles help debugging and reproducibility.
They are distinct from the solver trajectory, sampled token likelihoods,
curate replay and complete model checkpoints needed by RL. A bundle failure
alone is not evidence that process isolation failed.

After the owned UID and its exact backend binding are confirmed drained,
known storage errors or archive readback failures produce
`LocalArchiveDataError`. The one owned UID stays quarantined and existing
sources remain for reconciliation. A durable marker is written under the
host-local pool's `quarantined-uids/<uid>/local-archive.json`; a separate
`diagnostics/local_archive_data_incidents` report is copied to the run when
storage permits. No global isolation-incident directory is created.

The Task retains the actual private outcome and original Native error, marks
`training_data_valid=false`, and raises the typed data error. It returns no
TaskResult, zero substitute, reward label or fabricated likelihoods. The
training sampler rejects/refills the entire affected prompt group, including
successful siblings, rather than training a partial GRPO group. The host-local
failure marker exists before the Task raises and closes the provisional
finished-status sampling race. Counterfactual and other-run markers do not
invalidate an ordinary solver group.

Unknown backend drain, active UID processes, incorrect ownership/path/source
identity and unknown error categories retain the global fatal isolation path.
Legacy incidents and quarantines are never cleared or relabeled. The strict
budget-outcome verifier is unchanged: unavailable archive evidence is not
silently admitted as a valid budget reward.

This change does not add an optional deletion path or disable raw archival.
CPU tests exercise temporary-file data failure, owned-UID quarantine, group
accounting and fatal-path preservation; they are not training results.
