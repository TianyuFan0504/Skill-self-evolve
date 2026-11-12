# Background skill generation and delayed credit

The solver keeps the supported synchronous verl GRPO trainer. Skill generation
and counterfactual evaluation run independently on its dedicated curator and
frozen-evaluator routes, using one persistent event loop on a driver thread.
The learner remains the sole owner of online optimizer steps.

## Update boundary

1. Materialize the unchanged solver batch and its GRPO advantages.
2. Pin eligible future training-task inputs for background paired evaluation.
   This queues work; it does not wait for any pair to finish.
3. Sample only already matured delayed labels. Retain the original curator
   prompt/action tokens, action mask and actual processed behavior logprobs.
4. Compute the current online anchor for this ready replay and perform one
   combined solver/curate optimizer update. Update the parameter EMA once.
5. Acknowledge consumed labels and persist all completed source opportunities.
   Clear the optimizer transaction without awaiting curator generation.
6. Publish the current EMA to the curator route only when no curator generation
   is active. An ongoing generation retains its loaded EMA and original logprob
   provenance; it is never updated halfway through a model action.

An EMA refresh request pauses new curator dispatch until current generations
finish. The short tensor transfer then runs while the learner is outside its
online-model phase; the next learner phase waits only for that transfer. This
prevents a permanent backlog from starving EMA publication.

The background dispatcher progresses while the trainer is waiting for solver
rollouts. Curator and credit concurrency are bounded separately. Each logical
episode gets one source opportunity, deduplicated across TITO segments. Thus
32 prompts with four solver sessions enqueue 128 opportunities rather than
selecting one episode by pathname. The source queue stores unread opportunities;
it never converts an unread source into an observed zero reward.

## Outcomes and delayed attribution

A curator action may generate a candidate or explicitly abstain. Abstention
retains its actual action evidence without inventing a skill or a counterfactual
label. Unusable sources and failed jobs have explicit local dispositions and
coverage counters. Candidate admission still requires observed positive utility.

Each evaluation job pins a later, naturally encountered training task, the
with/without skill snapshots, and the immutable frozen evaluator. Different
pairs can run concurrently. The two branches within one pair remain sequential
to preserve their controlled UID and work-directory conditions.

Only observed paired events settle a credit window. Pending, censored and failed
evaluations do not become zero-utility labels. Ready labels train the original
curate action through historical importance weighting and proximal clipping;
they bypass solver GRPO group centering. The current learner computes the online
anchor immediately before the one combined update.

## Checkpoints and failures

Checkpointing briefly gates background storage commits, rather than draining
all model generations. Core model/data/EMA state and controller queue, SkillStore,
replay and ledger are copied from one consistent cut. Native generation capture
may continue outside the copied store trees; its durable store/replay commit
waits at `before_commit` until the cut is released.

Running jobs captured in a restored checkpoint are explicitly recorded as
interrupted work (failed sources or unobserved pairs), without automatically
drawing a replacement action or making
up reward. Queued work remains queued. Original Native artifacts are retained
for investigation. A bookkeeping/persistence failure in the dispatcher remains
visible to the learner; individual model/source failures have local dispositions.

Raw workspace archives are diagnostic artifacts. They are distinct from model
checkpoints and from the trajectory, reward and behavior-probability records
required by this learning algorithm. Ordinary archive I/O failures must not be
used as evidence of a run-wide isolation breach.

## Measurements

Training metrics include source opportunities, backlog, host reads, model
sampling, generated candidates, abstentions, censored/failed sources, evaluation
queue states, and the curator's loaded weight version. These counters distinguish
available evidence from actually processed evidence. CPU contract tests establish
scheduling and attribution behavior; real model training and matched held-out
evaluation are still required to establish reward or accuracy improvement.
