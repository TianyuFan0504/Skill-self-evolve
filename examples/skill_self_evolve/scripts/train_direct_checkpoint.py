#!/usr/bin/env python3
"""Parallel Native joint training with one validation and direct checkpoint reads.

The existing parallel client/server and spawn/final-evaluation components are
used unchanged. Separate receipts identify this actual outer entry and restore
provider; the frozen component receipts retain their original identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

POLICY = "parallel-native-direct-checkpoint-v1"


def extension_identity():
    from skill_harness_examples import train_spawn_safe

    path = Path(__file__).resolve()
    repository = path.parents[3]
    files = {
        "entry": path,
        "restore_provider": path.with_name("direct_checkpoint_restore.py"),
        "parallel_component": path.with_name("train_parallel_client.py"),
        "spawn_component": path.with_name("train_spawn_safe.py"),
        "checkpoint_core": repository / "verl/experimental/skill_harness/joint_checkpoint.py",
        "trainer_core": repository / "verl/experimental/skill_harness/joint_trainer.py",
    }
    return {
        "policy": POLICY,
        "driver_source_commit": train_spawn_safe._source_commit(repository),
        "actual_entry": str(path),
        "files": {
            name: {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
            for name, file in files.items()
        },
        "core_fingerprint_scope": "Unchanged legacy core; direct-restore implementation is identified separately",
    }


def trainer_class():
    from skill_harness_examples.direct_checkpoint_restore import DirectReadRestoreMixin

    from skill_harness_examples import train_spawn_safe

    class DirectReadJointTrainer(DirectReadRestoreMixin, train_spawn_safe.trainer_class()):
        pass

    return DirectReadJointTrainer


def initialized_receipt(trainer, config):
    from omegaconf import OmegaConf

    from skill_harness_examples import train_spawn_safe
    from verl.experimental.skill_harness.train import _redact

    restored = getattr(trainer, "_joint_restored", None)
    checkpoint = getattr(restored, "checkpoint", None)
    root = Path(config.skill_harness.artifact_root)
    restore = getattr(trainer._joint_coordinator, "last_restore_receipt", None)
    restore_ref = None
    if restore is not None:
        path = Path(restore["mutable_state_root"]) / "DIRECT_READ_RESTORE.json"
        restore_ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return train_spawn_safe._base()._publish(
        root / "driver_runs" / (str(time.time_ns()) + ".direct_checkpoint.initialized.json"),
        {
            "schema_version": 1,
            "kind": "initialized_direct_checkpoint_entry",
            **extension_identity(),
            "effective_config_after_trainer_init": _redact(OmegaConf.to_container(config, resolve=True)),
            "original_training_manifest_sha256": hashlib.sha256(
                (root / "training-manifest.json").read_bytes()
            ).hexdigest(),
            "restore_receipt_ref": restore_ref,
            "spawn_component_initialized_ref": trainer._resilient_initialized_receipt,
            "restored_checkpoint": None
            if checkpoint is None
            else {
                "checkpoint_dir": checkpoint.checkpoint_dir,
                "manifest_id": checkpoint.manifest_id,
                "spec": checkpoint.payload["spec"],
            },
            "restore_policy": None if restore is None else restore["policy"],
            "core_version_binding": None if checkpoint is None else checkpoint.payload["spec"]["versions"],
        },
    )


def runner_class():
    import ray

    @ray.remote
    class DirectReadTaskRunner:
        def run(self, config):
            import transfer_queue as tq
            from omegaconf import OmegaConf
            from skill_harness_examples.native_posix_spawn import install_spawn_policy

            from skill_harness_examples import train_spawn_safe
            from verl.experimental.skill_harness.bootstrap import configure_runtime
            from verl.experimental.skill_harness.train import preserve_failed_queue
            from verl.utils.import_utils import load_class_from_fqn
            from verl.utils.logging_utils import configure_verl_logging

            imports = configure_runtime()
            configure_verl_logging()
            if config.trainer.v1.trainer_mode != "sync" or config.skill_harness.mode != "skill_joint":
                raise ValueError("Direct checkpoint entry requires the synchronous joint trainer")
            train_spawn_safe.configure_extensions(config)
            install_spawn_policy()
            config.transfer_queue.enable = True
            OmegaConf.resolve(config)
            trainer, succeeded = None, False
            tq.init(config.transfer_queue)
            try:
                trainer = trainer_class()(config)
                trainer.init()
                trainer._resilient_extension_identity = train_spawn_safe.extension_identity()
                trainer._resilient_initialized_receipt = train_spawn_safe.initialized_receipt(trainer, config, imports)
                initialized_receipt(trainer, config)
                factory = load_class_from_fqn(
                    config.actor_rollout_ref.rollout.agent.agent_loop_manager_class, "AgentLoopManager"
                )
                manager = factory.create(
                    config=config,
                    llm_client=trainer.get_llm_client(),
                    teacher_client=trainer.get_teacher_client(),
                    reward_loop_worker_handles=trainer.get_reward_handles(),
                )
                trainer.fit(manager)
                succeeded = True
            except BaseException as error:
                try:
                    preserve_failed_queue(config, trainer, tq, error)
                except Exception as secondary:
                    error.add_note("Failure capture also failed: " + str(secondary))
                raise
            finally:
                try:
                    tracking = getattr(trainer, "logger", None)
                    if tracking is not None:
                        tracking.finish(exit_code=0 if succeeded else 1)
                finally:
                    tq.close()

    return DirectReadTaskRunner


def entry(config):
    from omegaconf import OmegaConf, open_dict

    from skill_harness_examples import train_parallel_client as parallel
    from skill_harness_examples import train_spawn_safe as spawn
    from verl.experimental.skill_harness.bootstrap import apply_runtime_environment, configure_runtime

    run = Path(config.skill_harness.artifact_root)
    settings_path = Path(os.environ["ALFWORLD_SERVER_CONFIG"])
    settings = json.loads(settings_path.read_bytes())
    health = parallel.environment_observation(settings)
    adapter = parallel.prepare_adapter(
        run, os.environ["ALFWORLD_TASK_CONFIG"], os.environ["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"]
    )
    os.environ.update(adapter["environment_variable_changes"])
    with open_dict(config):
        config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path = (
            adapter["task_profile"]["path"]
        )
    imports = configure_runtime()
    import ray

    from verl.experimental.skill_harness.train import _redact
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    if config.trainer.val_before_train or config.skill_harness.evaluation.initial:
        raise ValueError("Keep pretraining evaluation disabled in this continuation")
    spawn.configure_extensions(config)
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    apply_runtime_environment(config)
    identity = extension_identity()
    spawn._base()._publish(
        run / "driver_runs" / (str(time.time_ns()) + ".direct_checkpoint.json"),
        {
            "schema_version": 1,
            "kind": "prelaunch_direct_checkpoint_parallel_components",
            **identity,
            "observed_at_ns": time.time_ns(),
            "imports": imports,
            "parallel_component_ref": identity["files"]["parallel_component"],
            "spawn_component_identity": spawn.extension_identity(),
            "client_adapter": adapter,
            "environment_health": health,
            "environment_config_ref": {
                "path": str(settings_path),
                "sha256": parallel.digest(settings_path.read_bytes()),
            },
            "config": _redact(OmegaConf.to_container(config, resolve=True)),
            "original_training_manifest_preserved": True,
            "not_a_post_init_or_served_model_receipt": True,
        },
    )
    try:
        run_ppo(config, task_runner_class=runner_class())
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    import hydra

    hydra.main(config_path=None, config_name=None, version_base=None)(entry)()
