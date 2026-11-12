#!/usr/bin/env python3
"""Opt-in AllNative entry with explicit queued-source recovery generations."""

from __future__ import annotations

import hashlib
from pathlib import Path

POLICY = "parallel-direct-all-native-spawn-curate-generation-v2"
SUFFIX = "curate_generation"
PRELAUNCH_KIND = "prelaunch_curate_generation_parallel_components"


def extension_identity(policy):
    from skill_harness_examples import curate_generation, learner_runtime, train_all_native_spawn, train_resilient_setup

    identity = train_all_native_spawn.extension_identity(policy)
    identity["policy"] = POLICY
    identity["actual_entry"] = str(Path(__file__).resolve())
    identity["files"]["all_native_component"] = identity["files"]["entry"]
    entry = Path(__file__).resolve()
    identity["files"]["entry"] = {"path": str(entry), "sha256": hashlib.sha256(entry.read_bytes()).hexdigest()}
    identity["curate_generation_extension"] = {
        "policy": curate_generation.POLICY,
        "files": {name: {"path": str(Path(module.__file__).resolve()),
                         "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
                  for name, module in {"provider": curate_generation, "lifecycle_hook": train_resilient_setup}.items()},
        "controller_class": "skill_harness_examples.curate_generation.GenerationController",
    }
    identity["runtime_guards_extension"] = {
        "policy": learner_runtime.POLICY,
        "files": {name: {"path": str(Path(module.__file__).resolve()),
                         "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
                  for name, module in {"learner": learner_runtime, "probe": curate_generation}.items()},
        "learner_start_method": learner_runtime.START_METHOD,
        "credit_probe_concurrency": learner_runtime.PROBE_CONCURRENCY,
    }
    return identity


def trainer_class():
    from skill_harness_examples import curate_generation, train_direct_checkpoint

    class GenerationJointTrainer(curate_generation.GenerationTrainerMixin, train_direct_checkpoint.trainer_class()):
        pass

    return GenerationJointTrainer


def prepare_trainer(trainer):
    from skill_harness_examples.curate_generation import prepare_namespace
    from skill_harness_examples.learner_runtime import prepare_loaders

    trainer._learner_runtime_observation = prepare_loaders(trainer)
    prepare_namespace(trainer)


def prepare_learner(config):
    from skill_harness_examples.learner_runtime import prepare_learner as prepare

    return prepare(config)


def initialized_receipt(trainer, config, imports, policy):
    from skill_harness_examples import train_all_native_spawn

    return train_all_native_spawn.initialized_receipt(
        trainer, config, imports, policy, identity=extension_identity(policy), suffix=SUFFIX,
        kind="initialized_curate_generation_entry", curate_generation_ref=trainer._curate_generation_ref,
        learner_runtime_ref=trainer._learner_runtime_ref,
        learner_runtime_observation=trainer._learner_runtime_observation,
    )


def entry(config):
    from skill_harness_examples import train_all_native_spawn

    return train_all_native_spawn.entry(config, extension_module="skill_harness_examples.train_curate_generation")


if __name__ == "__main__":
    import hydra

    hydra.main(config_path=None, config_name=None, version_base=None)(entry)()
