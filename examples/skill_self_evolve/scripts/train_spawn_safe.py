#!/usr/bin/env python3
"""Examples-only continuation with a public, fork-free Native spawn boundary.

The resilient setup Task, replay sampler, controller, checkpoint and quiescent
final-evaluation implementations are composed unchanged.  The new policy is
installed before the learner's counterfactual loop and each primary SDK loop.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

MODULE = "skill_harness_examples.train_spawn_safe"
POLICY = "native-posix-spawn-boundary-v1"


def _base():
    from skill_harness_examples import train_resilient_setup

    return train_resilient_setup


def _source_commit(repository):
    """Read the current Git identity without spawning inside a Ray worker."""
    git = repository / ".git"
    if git.is_file():
        pointer = git.read_text().strip()
        if not pointer.startswith("gitdir: "):
            raise ValueError("Unknown Git directory pointer")
        git = (repository / pointer[8:]).resolve()
    head = (git / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        reference = head[5:]
        common = git
        if (git / "commondir").exists():
            common = (git / (git / "commondir").read_text().strip()).resolve()
        loose = common / reference
        if loose.is_file():
            head = loose.read_text().strip()
        else:
            packed = common / "packed-refs"
            matches = [line.split()[0] for line in packed.read_text().splitlines()
                       if line and not line.startswith(("#", "^")) and line.split()[1] == reference]
            if len(matches) != 1:
                raise ValueError("Current Git reference is not uniquely resolvable")
            head = matches[0]
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise ValueError("Invalid current Git commit")
    return head


def extension_identity():
    from skill_harness_examples import native_posix_spawn, train_quiescent_eval

    resilient = _base()
    path = Path(__file__).resolve()
    repository = path.parents[3]
    namespace = repository / "skill_harness_examples/__init__.py"
    base_path = Path(resilient.__file__).resolve()
    base = {
        "policy": resilient.POLICY,
        "driver_path": str(base_path),
        "driver_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "namespace_path": str(namespace),
        "namespace_sha256": hashlib.sha256(namespace.read_bytes()).hexdigest(),
        "driver_source_commit": _source_commit(repository),
    }
    quiescent = Path(train_quiescent_eval.__file__).resolve()
    return {
        **base,
        "policy": POLICY,
        "driver_path": str(path),
        "driver_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "composed_extensions": {
            "resilient_setup": base,
            "quiescent_final": {
                "path": str(quiescent),
                "sha256": hashlib.sha256(quiescent.read_bytes()).hexdigest(),
            },
            "native_spawn": native_posix_spawn.spawn_identity(),
        },
    }


class SpawnSafeRunner:
    """The SDK constructs this callable before it creates its asyncio loop."""

    def __init__(self, **runner_kwargs):
        from skill_harness_examples.native_posix_spawn import install_spawn_policy

        install_spawn_policy()
        self.runner_kwargs = runner_kwargs

    async def __call__(self, **episode_kwargs):
        return await _base().safe_run_task(**{**self.runner_kwargs, **episode_kwargs})


def configure_extensions(config):
    from omegaconf import open_dict

    _base().configure_extensions(config)
    with open_dict(config):
        config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn = (
            MODULE + ".SpawnSafeRunner"
        )


def initialized_receipt(trainer, config, imports):
    import asyncio

    from omegaconf import OmegaConf

    from skill_harness_examples import native_posix_spawn
    from verl.experimental.skill_harness.train import _redact

    loop = trainer._joint_loop
    policy = asyncio.get_event_loop_policy()
    if type(loop).__module__ != native_posix_spawn.__name__:
        raise RuntimeError("Counterfactual loop was created before the Native spawn policy")
    if type(policy).__module__ != native_posix_spawn.__name__:
        raise RuntimeError("Native spawn policy was replaced during learner initialization")
    restored = getattr(trainer, "_joint_restored", None)
    checkpoint = getattr(restored, "checkpoint", None)
    root = Path(config.skill_harness.artifact_root)
    return _base()._publish(
        root / "driver_runs" / (str(time.time_ns()) + ".spawn_safe.initialized.json"),
        {
            **trainer._resilient_extension_identity,
            "imports": imports,
            "effective_config_after_trainer_init": _redact(OmegaConf.to_container(config, resolve=True)),
            "original_training_manifest_sha256": hashlib.sha256(
                (root / "training-manifest.json").read_bytes()
            ).hexdigest(),
            "counterfactual_loop_class": type(loop).__module__ + "." + type(loop).__qualname__,
            "event_loop_policy_class": type(policy).__module__ + "." + type(policy).__qualname__,
            "primary_runner_policy_install": "synchronous runner constructor before SDK asyncio.run",
            "restored_checkpoint": None if checkpoint is None else {
                "checkpoint_dir": checkpoint.checkpoint_dir,
                "manifest_id": checkpoint.manifest_id,
                "spec": checkpoint.payload["spec"],
            },
        },
    )


def trainer_class():
    """Preserve the frozen final lifecycle, replacing its Git subprocess only."""
    import uuid

    from skill_harness_examples.train_quiescent_eval import QuiescentFinalMixin, get, write_new

    class SpawnSafeJointTrainer(_base().trainer_class()):
        def fit(self, manager):
            # This is the frozen quiescent fit body with two substitutions:
            # metadata comes from this extension, and super skips the old fit.
            evaluation = self.config.skill_harness.evaluation
            if get(self.config.trainer, "val_before_train", False) or get(evaluation, "initial", False):
                raise ValueError("This entrypoint never runs development/official evaluation before training")
            enabled = get(evaluation, "final", False)
            self._quiescent_final_enabled = enabled
            if not enabled:
                return super(QuiescentFinalMixin, self).fit(manager)
            final = Path(self.config.skill_harness.artifact_root) / "evaluation/final"
            if final.exists() and any(final.iterdir()):
                raise FileExistsError("Preserve existing final artifacts; this entrypoint never overwrites them")
            self._lifecycle_id = "driver_" + uuid.uuid4().hex
            self._driver_identity = extension_identity()
            self._lifecycle_paths = {}
            self._quiescent_seal = None
            restored = getattr(self, "_joint_restored", None)
            checkpoint = getattr(restored, "checkpoint", None)
            if checkpoint is not None and checkpoint.payload["spec"]["optimizer_step"] == self.total_training_steps:
                self._quiescent_seal = {
                    "seal_origin": "restored_complete_final_checkpoint",
                    "optimizer_step": checkpoint.payload["spec"]["optimizer_step"],
                    "checkpoint_cursor": checkpoint.payload["spec"]["global_step"],
                    "checkpoint_dir": checkpoint.checkpoint_dir,
                    "manifest_id": checkpoint.manifest_id,
                    "snapshot_id": checkpoint.payload["snapshots"]["skill_heads"]["main"],
                    "store_root": str(self.config.skill_harness.store_root),
                    "checkpoint_core_versions": checkpoint.payload["spec"].get("versions", {}),
                }
            evaluation["final"] = False
            try:
                result = super(QuiescentFinalMixin, self).fit(manager)
            finally:
                evaluation["final"] = enabled
            seal = self._quiescent_seal
            if seal is None:
                raise RuntimeError("Final evaluation requires the wrapper's final COMPLETE checkpoint seal")
            if (
                self._closed_loop_clock() != seal["optimizer_step"]
                or self._joint_controller.store.head() != seal["snapshot_id"]
            ):
                raise RuntimeError("Policy/library changed after the final checkpoint seal")
            if seal["seal_origin"] == "restored_complete_final_checkpoint":
                self._lifecycle_record("checkpoint_seal.json", seal)
            self.agent_loop_manager = self._pinned_manager(seal["snapshot_id"])
            self._lifecycle_record(
                "evaluation_entry.json",
                {
                    **seal,
                    "evaluation_cursor": self.global_steps,
                    "background_stopped_before_test_service": True,
                    "validation_factory_uses_config_copy": True,
                },
            )
            self._init_dump_executor()
            try:
                self._official_final()
                if (
                    self._closed_loop_clock() != seal["optimizer_step"]
                    or self._joint_controller.store.head() != seal["snapshot_id"]
                ):
                    raise RuntimeError("Final evaluation changed the optimizer clock or library head")
                self._lifecycle_record(
                    "evaluation_complete.json",
                    {
                        **seal,
                        "evaluation_cursor": self.global_steps,
                        "optimizer_updates_during_test": 0,
                    },
                )
                original = Path(self.config.skill_harness.artifact_root) / "training-manifest.json"
                write_new(
                    final / "lifecycle.json",
                    {
                        "schema_version": 1,
                        "kind": "quiescent_final_evaluation",
                        **self._driver_identity,
                        "original_training_manifest_sha256": hashlib.sha256(original.read_bytes()).hexdigest()
                        if original.exists()
                        else None,
                        **seal,
                        "evaluation_cursor": self.global_steps,
                        "checkpoint_seal": self._lifecycle_paths["checkpoint_seal.json"],
                        "evaluation_entry": self._lifecycle_paths["evaluation_entry.json"],
                        "evaluation_complete": self._lifecycle_paths["evaluation_complete.json"],
                    },
                )
            finally:
                self._shutdown_dump_executor()
            return result

    return SpawnSafeJointTrainer


def runner_class():
    import ray

    @ray.remote
    class SpawnSafeTaskRunner:
        def run(self, config):
            import transfer_queue as tq
            from omegaconf import OmegaConf
            from skill_harness_examples.native_posix_spawn import install_spawn_policy

            from verl.experimental.skill_harness.bootstrap import configure_runtime
            from verl.experimental.skill_harness.train import preserve_failed_queue
            from verl.utils.import_utils import load_class_from_fqn
            from verl.utils.logging_utils import configure_verl_logging

            imports = configure_runtime()
            configure_verl_logging()
            if config.trainer.v1.trainer_mode != "sync" or config.skill_harness.mode != "skill_joint":
                raise ValueError("Spawn-safe continuation requires the original synchronous joint trainer")
            configure_extensions(config)
            install_spawn_policy()  # Must precede trainer construction/new_event_loop.
            config.transfer_queue.enable = True
            OmegaConf.resolve(config)
            trainer, succeeded = None, False
            tq.init(config.transfer_queue)
            try:
                trainer = trainer_class()(config)
                trainer.init()
                trainer._resilient_extension_identity = extension_identity()
                trainer._resilient_initialized_receipt = initialized_receipt(trainer, config, imports)
                factory = load_class_from_fqn(
                    config.actor_rollout_ref.rollout.agent.agent_loop_manager_class, "AgentLoopManager"
                )
                manager = factory.create(config=config, llm_client=trainer.get_llm_client(),
                                         teacher_client=trainer.get_teacher_client(),
                                         reward_loop_worker_handles=trainer.get_reward_handles())
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

    return SpawnSafeTaskRunner


def entry(config):
    from omegaconf import OmegaConf

    from verl.experimental.skill_harness.bootstrap import apply_runtime_environment, configure_runtime

    imports = configure_runtime()
    import ray

    from verl.experimental.skill_harness.train import _redact
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    if config.trainer.val_before_train or config.skill_harness.evaluation.initial:
        raise ValueError("Keep pretraining evaluation disabled in this continuation")
    configure_extensions(config)
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    apply_runtime_environment(config)
    _base()._publish(
        Path(config.skill_harness.artifact_root) / "driver_runs" / (str(time.time_ns()) + ".spawn_safe.json"),
        {
            **extension_identity(), "imports": imports,
            "config": _redact(OmegaConf.to_container(config, resolve=True)),
            "original_training_manifest_preserved": True,
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
