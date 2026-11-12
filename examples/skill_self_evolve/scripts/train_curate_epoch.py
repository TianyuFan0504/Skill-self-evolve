#!/usr/bin/env python3
"""Generation recovery with an explicit fresh host account epoch and cap8."""

from __future__ import annotations

import hashlib
from pathlib import Path

POLICY = "parallel-direct-all-native-spawn-curate-generation-account-epoch-v3"
SUFFIX = "curate_epoch"
PRELAUNCH_KIND = "prelaunch_curate_epoch_parallel_components"
PHYSICAL_PROBE_CONCURRENCY = 8
_controller_class = None


def extension_identity(policy):
    from skill_harness_examples import account_pool_epoch, train_curate_generation

    identity = train_curate_generation.extension_identity(policy)
    identity["policy"] = POLICY
    entry = Path(__file__).resolve()
    identity["actual_entry"] = str(entry)
    identity["files"]["generation_component"] = identity["files"]["entry"]
    identity["files"]["entry"] = {"path": str(entry), "sha256": hashlib.sha256(entry.read_bytes()).hexdigest()}
    identity["account_pool_epoch_extension"] = account_pool_epoch.identity()
    identity["runtime_guards_extension"]["credit_probe_concurrency"] = PHYSICAL_PROBE_CONCURRENCY
    identity["curate_generation_extension"]["controller_class"] = (
        "skill_harness_examples.train_curate_epoch.EpochGenerationController"
    )
    return identity


class EpochGenerationControllerMixin:
    def _credit_probe_concurrency(self):
        from skill_harness_examples import account_pool_epoch, curate_generation

        observation = curate_generation.read_ref(self.trainer._learner_runtime_ref)
        if observation.get("physical_credit_probe_concurrency") != PHYSICAL_PROBE_CONCURRENCY:
            raise ValueError("The epoch learner and controller physical probe limits differ")
        # The driver already required >=40 clean UIDs. This process must still
        # have the exact active provider; no old-pool fallback is permitted.
        if account_pool_epoch._active != account_pool_epoch.identity():
            raise ValueError("The fresh account epoch is not active in the actual learner")
        return PHYSICAL_PROBE_CONCURRENCY


def controller_class():
    global _controller_class
    if _controller_class is None:
        from skill_harness_examples.curate_generation import controller_class as generation_class

        _controller_class = type("EpochGenerationController", (EpochGenerationControllerMixin, generation_class()),
                                 {"__module__": __name__})
    return _controller_class


def trainer_class():
    from skill_harness_examples import train_direct_checkpoint

    class EpochGenerationJointTrainer(train_direct_checkpoint.trainer_class()):
        def _joint_controller_class(self):
            return controller_class()

    return EpochGenerationJointTrainer


def prepare_driver(config):
    from skill_harness_examples.account_pool_epoch import prepare_driver as prepare

    return prepare(config)


def prepare_runtime_environment(config):
    from skill_harness_examples.account_pool_epoch import prepare_runtime_environment as prepare

    return prepare(config)


def prepare_learner(config):
    from skill_harness_examples import account_pool_epoch, learner_runtime

    account_pool_epoch.require_worker_activation(config)
    account_pool_epoch.install_process(role="learner")
    return learner_runtime.prepare_learner(config, physical_probe_concurrency=PHYSICAL_PROBE_CONCURRENCY)


def prepare_trainer(trainer):
    from skill_harness_examples.train_curate_generation import prepare_trainer as prepare

    return prepare(trainer)


def initialized_receipt(trainer, config, imports, policy):
    from skill_harness_examples import account_pool_epoch, train_all_native_spawn

    return train_all_native_spawn.initialized_receipt(
        trainer, config, imports, policy, identity=extension_identity(policy), suffix=SUFFIX,
        kind="initialized_curate_epoch_entry", curate_generation_ref=trainer._curate_generation_ref,
        learner_runtime_ref=trainer._learner_runtime_ref,
        learner_runtime_observation=trainer._learner_runtime_observation,
        account_pool_epoch_activation_refs=account_pool_epoch.activation_refs(),
        account_pool_epoch_activation_directory=str(account_pool_epoch.activation_directory()),
    )


def entry(config):
    from skill_harness_examples import train_all_native_spawn

    return train_all_native_spawn.entry(config, extension_module="skill_harness_examples.train_curate_epoch")


if __name__ == "__main__":
    import hydra

    hydra.main(config_path=None, config_name=None, version_base=None)(entry)()
