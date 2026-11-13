# Joint skill self-evolution

## Training path

`examples/skill_self_evolve/run.sh` calls the single `scripts/launch.py` entry.
The launcher reads explicit deployment settings, prepares fresh data and public
assets, supervises the parallel environment service, and invokes the installed
training interpreter with the joint YAML and Hydra overrides through
`verl.experimental.skill_harness.train`. That module supplies one custom
TaskRunner to the official `verl.trainer.main_ppo.run_ppo` entry. The TaskRunner
creates one `SkillJointSyncTrainer`.

The rollout adapter delegates native agent execution to Uni-Agent. Codex runs
the agent loop, and its session-scoped Responses Gateway records the local
model's generated token IDs, action masks and behavior log probabilities.
ALFWorld owns game state and the authoritative `won` reward. Training consumes
the original generated tokens, rather than decoding and re-encoding actions.

## One online learner

The online parameters and optimizer minimize

```text
L_online = L_solver_GRPO + lambda_curate * L_curate_replay
```

Solver rows retain the normal GRPO objective and rollout correction. Curator
rows use their own action mask, settled signed advantage and recorded sampling
probabilities. Both losses update the same online parameters in the joint
learner. The separate curator sampling replica receives parameter EMA updates;
it has no optimizer or backward pass. A separate frozen evaluator supplies a
stable policy for counterfactual skill tests.

## Skill creation and delayed credit

1. Each accepted logical solver episode enters the source queue once, even if
   its trajectory spans several transport segments.
2. The EMA curator sees the recorded episode context and either proposes one
   reusable skill or explicitly abstains. The complete sampled action, mask,
   context and original behavior log probabilities are persisted before
   temporary transport data can disappear.
3. The host retrieves skills from immutable SkillStore snapshots and injects
   bounded guidance into later original ALFWorld tasks. Each rollout group
   records the snapshot and rendered context it actually used.
4. The credit workers compare the same later task with and without the
   candidate skill. They hold the evaluator policy, other skills and task setup
   fixed. A credit window settles only after the configured exposure and delay
   requirements are met; censored observations remain distinguishable from
   observed failures.
5. The settled counterfactual utility, less declared costs, labels the original
   curator action. Delayed replay applies bounded importance correction and
   explicit replay-age limits before the joint learner consumes that action.

Action importance correction addresses the recorded curator sampling policy.
It does not fully correct changes in the live solver, retriever, task
population or library. Clipping and replay limits are practical approximations;
a positive paired estimate is not a claim of improved full benchmark accuracy.

## Data and final evaluation

Only official train tasks enter optimization. Development tasks are held-out
training scenes. Official `valid_seen` and `valid_unseen` games remain reserved
for the final trained-policy evaluation. No generated task chains are needed.
Catalog identities, dataset splits and token/sampling metadata are checked for
consistency because they define the learning and evaluation observations.

Final evaluation runs after a completed trained joint checkpoint. Background
curation and credit generation stop, and evaluation pins the trained policy and
final library snapshot. It uses greedy single-sample validation on each
complete official test split, with no optimizer updates or skill publication.
The environment backend and interpreter are explicit deployment configuration.
A separate test-only service is routed through the existing task interface and
the training service configuration is restored on exit. Raw case outcomes are
reconciled with the recorded validation metrics.

## Runtime and persistence

The runtime-provider interface separates the learning algorithm from the
host's process isolation and cleanup mechanism. A deployment explicitly selects
an importable provider and its settings. The optional Linux account provider
uses a provisioned roster and configured workspace/read roots. The generic
process-based ALFWorld backend and idempotent `alfworld-step` client provide the
environment transport; they are not alternate training entry points.

Joint checkpoints bind the online model, optimizer, EMA clock and weights,
skill snapshots, replay, credit ledger, source queue and loader progress at one
consistent cut. Run outputs also preserve native/Gateway trajectories,
environment outcomes, skill decisions, paired outcomes, losses and
`metrics.jsonl`. Plots derive from those raw series. Deployment-specific host
histories and source-code fingerprint receipts are outside the training path.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `train.py`, `joint_trainer.py`, `joint_worker.py` | Official trainer integration and joint update lifecycle |
| `framework.py`, `group_sampler.py`, `alfworld/task.py`, `alfworld/runner.py` | Skill-aware rollout, complete-group refill and native Task execution |
| `context.py`, `store.py` | Immutable skill snapshots and bounded host retrieval |
| `controller.py`, `source_queue.py`, `curate.py` | Background source review and EMA curator actions |
| `probes.py`, `credit.py` | Frozen paired evaluation and delayed credit windows |
| `replay.py`, `replay_batch.py`, `losses.py` | Durable action replay and joint objective |
| `ema.py`, `ema_worker.py`, `policy_routes.py` | Parameter EMA and separate generation routes |
| `joint_checkpoint.py`, `evaluation.py` | Consistent joint state and final trained-only evaluation |
| `telemetry.py`, `scripts/plot_metrics.py` | Recorded metrics and plots |

See the [recipe](../../examples/skill_self_evolve/README.md) and
[deployment requirements](environment.md) for configuration.
