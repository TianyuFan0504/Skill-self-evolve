# Skill Self-Evolve

Joint ALFWorld solver and skill-curator training on verl. One online model and
optimizer learn from solver GRPO and delayed counterfactual skill credit. The
curator samples from a separate parameter-EMA replica; a frozen policy evaluates
with-skill/without-skill pairs on later original tasks.

The public training entry is:

```bash
bash examples/skill_self_evolve/run.sh
```

Prepare the installed dependencies, benchmark data, agent task configuration,
environment service and explicit runtime provider first. Deployment paths come
from environment variables or Hydra overrides. See the
[recipe and configuration](examples/skill_self_evolve/README.md) and
[deployment requirements](docs/skill_self_evolve/environment.md).

The launch path is `run.sh` → `scripts/launch.py` →
`python -m verl.experimental.skill_harness.train --config-name=alfworld_joint`.
It uses the official `run_ppo` custom TaskRunner extension and one joint trainer.
The native Codex agent runs through Uni-Agent, with the local policy providing
its model generations and the ALFWorld engine providing task rewards.

Training uses official train tasks with a scene-disjoint development holdout.
Official seen/unseen tests evaluate the final trained state. Raw trajectories,
metrics, skill decisions, paired outcomes and joint checkpoints remain in the
configured run directory. Metric plots are derived from the recorded series.

Read the [current design](docs/skill_self_evolve/design.md) for the loss,
delayed-replay protocol and module responsibilities. This research fork retains
the upstream [Apache-2.0 license](LICENSE); the
[upstream documentation](https://verl.readthedocs.io/) covers the underlying
training infrastructure. This source does not claim a completed training run or
an improvement in benchmark performance.
