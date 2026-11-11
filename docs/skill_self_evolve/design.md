# ALFWorld Skill Self-Evolution Training

## Objective and acceptance

Train Qwen3.5-4B through the official Uni-Agent Gateway and the native Codex
agent, using `/workspace/verl-0.9.1` as the training framework. The benchmark
root is `/root/nas_chaohuang/benchmarks/alfworld/data/json_2.1.1`.

Completion requires a real training run and a reproducible comparison showing
increased reward and accuracy under the same evaluation protocol. A successful
import, a synthetic episode, a nonzero gradient, or an improved training reward
alone does not establish this result. Unsuccessful experiments remain part of
the experiment record.

## Required invariants

1. Run the real Uni-Agent CodexAgent, with Qwen3.5-4B as its local policy.
2. Train and curate only from the official train split. Create development
   partitions from train; reserve valid_seen and valid_unseen for testing.
3. Preserve the environment's authoritative `won` result as task reward and
   accuracy. Never replace environment outcomes with model self-assessments.
4. Keep generated model tokens, masks, and actual behavior log probabilities
   from the Gateway. Do not decode and re-encode generated actions for training.
5. Treat every episode attempt as a unique session, even when its game is
   repeated by GRPO or across epochs. Record relative path and content hash;
   numeric catalog indices are not stable benchmark identities.
6. Use one online parameter set and optimizer for solver and curate losses.
   The curator sampling replica receives parameter EMA only.
7. Archive curate actions durably before short-lived TransferQueue data can be
   cleared. Preserve the original EMA behavior probabilities unchanged.
8. Bind every rollout group to an immutable skill snapshot and rendered
   context. Skill retrieval happens in the host context builder, not in the
   dataset loader.
9. Collect delayed credit with a fixed, versioned evaluator and exposure
   protocol. Keep actual live-solver ROI distinct from reference credit.
10. Recover model, optimizer, EMA, skill snapshots, replay, credit ledger, and
    update cursors from the same completed joint checkpoint.
11. Perform all external downloads, GitHub fetches, and pushes on the local
    workstation. Transfer code and artifacts over SSH; never place GitHub
    credentials on the server.
12. Keep prompts, technical documentation, commit descriptions, and artifact
    schemas in English. Preserve complete trajectories and failed-run evidence.

## Extension strategy

Use the documented `agent_loop_manager_class`, custom Task, TaskRunner, trainer
hooks, actor-worker subclass, and public `set_loss_fn` RPC. Preserve the standard
solver GRPO path. Do not replace the working core with Uni-Agent's vendored
development copy of verl.

The native Codex process calls a session-scoped Responses Gateway. The Gateway
collects token-level trajectories and forwards generation to the local policy.
The ALFWorld service owns world state and rewards; the agent only receives the
game interface and its allowed observations.

The online learner computes `L_solver + lambda_curate * L_curate`. The solver
uses task outcome rewards. The curate branch uses settled counterfactual utility
minus declared costs, with role-specific masks and explicit off-policy handling.
The EMA replica never receives backward calls or an independent optimizer.

## Module sequence

| Module | Result | Verification |
| --- | --- | --- |
| Benchmark bridge | Complete catalog, isolated train/dev/test rows, repeatable native Codex tasks | Stable IDs, split isolation, unique attempt directories, engine reward provenance |
| Solver training recipe | Uni-Agent + native Codex + Qwen3.5-4B through core 0.9.1 | Import provenance, real Gateway tokens, one training update, matching-protocol baseline |
| Skill storage and loading | Immutable snapshots, bounded host injection, producer/consumer lineage | Atomic publication, branch isolation, actual injected token accounting |
| EMA policy route | Separate online and EMA replica versions | EMA numerical update, correct export, no accidental online overwrite |
| Delayed credit and replay | Durable actions, idempotent usage events, mature credit | Delayed reward linkage, negative/zero records, restart consistency |
| Role-aware learner | Solver and curate losses update the online model | Per-role gradients, actual parameter changes, genuine behavior correction |
| Training and evaluation | Saved runs and increased reward/accuracy | Fixed protocol, development-only selection, untouched final test comparison |

Commit and push each completed module independently. Each commit records what
changed and what was actually tested. The local workstation relays the server's
offline commits to the configured GitHub repository.

## Reward and causal scope

For a skill S, compare the same downstream task with S and without S, keeping
other skills, evaluator policy, environment state, and evaluation protocol fixed.
Record the difference, sample count, costs, and uncertainty. Deduplicate usage
events before aggregating a credit window. Unobserved or censored skills are not
automatically failures.

Reference credit trains a curator conditional objective on recorded contexts.
Action importance ratios do not correct every change in the live solver,
retriever, task distribution, and library. Clip, rejection, and replay limits
are conservative approximations, not guarantees for arbitrarily old actions.

## Artifacts

Every run writes a manifest, source commit and dirty-state record, data and
protocol hashes, task configuration, model/policy versions, environment actions,
Codex events, Gateway token trajectories, rollout rewards, per-role losses,
resource telemetry, checkpoints, and machine-readable metric series. Curves are
derived from those saved series. No credentials or private runtime tokens belong
in the repository or experiment exports.
