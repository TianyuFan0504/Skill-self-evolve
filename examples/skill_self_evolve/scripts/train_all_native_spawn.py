#!/usr/bin/env python3
"""Direct-read training with solver and registered curator spawn transports.

The final-evaluation/trainer body, agent instructions, sampler and model paths
remain supplied by frozen components. New receipts identify this actual entry,
the actual learner loop and both native helpers without claiming the old
solver-only initialized receipt was executed for the learner.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

POLICY = "parallel-direct-all-native-spawn-v1"


def extension_identity(policy):
    from skill_harness_examples import all_native_spawn, train_direct_checkpoint, train_spawn_safe

    path = Path(__file__).resolve()
    sources = {
        "entry": path,
        "direct_component": Path(train_direct_checkpoint.__file__).resolve(),
        "restore_provider": path.with_name("direct_checkpoint_restore.py"),
        "parallel_component": path.with_name("train_parallel_client.py"),
        "spawn_component": Path(train_spawn_safe.__file__).resolve(),
    }
    return {
        "policy": POLICY,
        "actual_entry": str(path),
        "driver_source_commit": train_spawn_safe._source_commit(path.parents[3]),
        "files": {name: {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                  for name, source in sources.items()},
        "all_role_native_spawn": all_native_spawn.identity(policy),
        "solver_spawn_component_identity": train_spawn_safe.extension_identity(),
        "core_fingerprint_scope": "Unchanged core; actual all-role spawn and direct-read providers recorded separately",
    }


def initialized_receipt(trainer, config, imports, policy, *, identity=None,
                        suffix="all_native_spawn", kind="initialized_all_native_spawn_entry",
                        curate_generation_ref=None, learner_runtime_ref=None, learner_runtime_observation=None):
    from omegaconf import OmegaConf

    from skill_harness_examples import all_native_spawn, train_spawn_safe
    from verl.experimental.skill_harness.train import _redact

    if (not isinstance(trainer._joint_loop, all_native_spawn.AllNativeSelectorLoop)
            or asyncio.get_event_loop_policy() is not policy
            or trainer._joint_loop.native_spawner is not policy.spawner
            or trainer._joint_loop.curator_profiles is not policy.curator_profiles
            or not policy.curator_profiles):
        raise RuntimeError("The registered all-role spawn policy must precede learner loop construction")
    restored = getattr(trainer, "_joint_restored", None)
    checkpoint = getattr(restored, "checkpoint", None)
    restore = getattr(trainer._joint_coordinator, "last_restore_receipt", None)
    restore_ref = None
    if restore is not None:
        restore_path = Path(restore["mutable_state_root"]) / "DIRECT_READ_RESTORE.json"
        restore_ref = {"path": str(restore_path), "sha256": hashlib.sha256(restore_path.read_bytes()).hexdigest()}
    root = Path(config.skill_harness.artifact_root)
    return train_spawn_safe._base()._publish(
        root / "driver_runs" / (str(time.time_ns()) + "." + suffix + ".initialized.json"),
        {
            "schema_version": 1,
            "kind": kind,
            **(extension_identity(policy) if identity is None else identity),
            "imports": imports,
            "effective_config_after_trainer_init": _redact(OmegaConf.to_container(config, resolve=True)),
            "original_training_manifest_sha256": hashlib.sha256(
                (root / "training-manifest.json").read_bytes()
            ).hexdigest(),
            "counterfactual_loop_class": type(trainer._joint_loop).__module__ + "."
            + type(trainer._joint_loop).__qualname__,
            "event_loop_policy_class": type(policy).__module__ + "." + type(policy).__qualname__,
            "registered_curator_scope": "Actual NativeCuratorRuntime account, work root, package and clean environment",
            "primary_worker_scope": "Unchanged SpawnSafeRunner installs the solver-only component in episode workers",
            "restore_receipt_ref": restore_ref,
            "restored_checkpoint": None if checkpoint is None else {
                "checkpoint_dir": checkpoint.checkpoint_dir,
                "manifest_id": checkpoint.manifest_id,
                "spec": checkpoint.payload["spec"],
            },
            "restore_policy": None if restore is None else restore["policy"],
            "core_version_binding": None if checkpoint is None else checkpoint.payload["spec"]["versions"],
            "component_receipt_scope": (
                "Frozen final-fit body identity is a component; lifecycle extension uses this receipt"
            ),
            **({"curate_generation_ref": curate_generation_ref} if curate_generation_ref is not None else {}),
            **({"learner_runtime_ref": learner_runtime_ref,
                "learner_runtime_observation": learner_runtime_observation} if learner_runtime_ref is not None else {}),
        },
    )


def runner_class(extension_module=None):
    import ray

    @ray.remote
    class AllNativeTaskRunner:
        def run(self, config):
            import transfer_queue as tq
            from omegaconf import OmegaConf

            from skill_harness_examples import all_native_spawn, train_direct_checkpoint, train_spawn_safe
            from verl.experimental.skill_harness.bootstrap import configure_runtime
            from verl.experimental.skill_harness.train import preserve_failed_queue
            from verl.utils.import_utils import load_class_from_fqn
            from verl.utils.logging_utils import configure_verl_logging

            provider = None
            if extension_module is not None:
                import importlib

                provider = importlib.import_module(extension_module)

            imports = configure_runtime()
            configure_verl_logging()
            if config.trainer.v1.trainer_mode != "sync" or config.skill_harness.mode != "skill_joint":
                raise ValueError("All-role Native spawn requires the synchronous joint trainer")
            train_spawn_safe.configure_extensions(config)
            policy = all_native_spawn.install_policy()
            config.transfer_queue.enable = True
            OmegaConf.resolve(config)
            trainer, succeeded = None, False
            tq.init(config.transfer_queue)
            diagnostics = train_direct_checkpoint.passive_diagnostics(config.skill_harness.artifact_root)
            try:
                learner_runtime_ref = None
                if provider is not None and hasattr(provider, "prepare_learner"):
                    diagnostics.stage("learner_start_method")
                    learner_runtime_ref = provider.prepare_learner(config)
                diagnostics.stage("trainer_construct")
                trainer = (train_direct_checkpoint if provider is None else provider).trainer_class()(config)
                if learner_runtime_ref is not None:
                    trainer._learner_runtime_ref = learner_runtime_ref
                diagnostics.attach(trainer)
                diagnostics.stage("trainer_init")
                trainer.init()
                diagnostics.stage("register_curator_spawn")
                policy.register_curator(trainer._identity().runtime)
                if provider is not None:
                    provider.prepare_trainer(trainer)
                trainer._resilient_extension_identity = (
                    extension_identity(policy) if provider is None else provider.extension_identity(policy)
                )
                trainer._resilient_initialized_receipt = (
                    initialized_receipt if provider is None else provider.initialized_receipt
                )(trainer, config, imports, policy)
                diagnostics.stage("manager_create")
                factory = load_class_from_fqn(
                    config.actor_rollout_ref.rollout.agent.agent_loop_manager_class, "AgentLoopManager"
                )
                manager = factory.create(
                    config=config,
                    llm_client=trainer.get_llm_client(),
                    teacher_client=trainer.get_teacher_client(),
                    reward_loop_worker_handles=trainer.get_reward_handles(),
                )
                diagnostics.stage("trainer_fit")
                trainer.fit(manager)
                succeeded = True
            except BaseException as error:
                diagnostics.stage("failure_cleanup")
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
                    try:
                        diagnostics.stage("queue_close")
                        tq.close()
                    finally:
                        diagnostics.close()

    return AllNativeTaskRunner


def entry(config, *, extension_module=None):
    from omegaconf import OmegaConf, open_dict

    from skill_harness_examples import all_native_spawn
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
    policy = all_native_spawn.install_policy()
    provider = None
    if extension_module is not None:
        import importlib

        provider = importlib.import_module(extension_module)
    identity = extension_identity(policy) if provider is None else provider.extension_identity(policy)
    suffix = "all_native_spawn" if provider is None else provider.SUFFIX
    kind = "prelaunch_all_native_spawn_parallel_components" if provider is None else provider.PRELAUNCH_KIND
    spawn._base()._publish(
        run / "driver_runs" / (str(time.time_ns()) + "." + suffix + ".json"),
        {
            "schema_version": 1,
            "kind": kind,
            **identity,
            "observed_at_ns": time.time_ns(),
            "imports": imports,
            "parallel_component_ref": identity["files"]["parallel_component"],
            "client_adapter": adapter,
            "environment_health": health,
            "environment_config_ref": {
                "path": str(settings_path), "sha256": parallel.digest(settings_path.read_bytes()),
            },
            "config": _redact(OmegaConf.to_container(config, resolve=True)),
            "original_training_manifest_preserved": True,
            "not_a_post_init_or_served_model_receipt": True,
        },
    )
    try:
        run_ppo(config, task_runner_class=runner_class(extension_module))
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    import hydra

    hydra.main(config_path=None, config_name=None, version_base=None)(entry)()
