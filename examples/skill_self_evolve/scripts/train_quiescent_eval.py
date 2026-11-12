#!/usr/bin/env python3
"""Examples-only training entrypoint with a sealed final evaluation lifecycle.

Core trainer/loss/checkpoint code is unchanged. No runtime imports occur until
the entrypoint or its production factories are invoked.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import subprocess
import time
import uuid
from collections.abc import Mapping
from pathlib import Path


def get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def driver_identity():
    path = Path(__file__).resolve()
    commit = subprocess.check_output(["git", "-C", str(path.parents[3]), "rev-parse", "HEAD"], text=True).strip()
    return {
        "driver_path": str(path),
        "driver_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "driver_source_commit": commit,
    }


def record_initialized_driver(root, effective_config, imports, checkpoint=None):
    """Retain the actual restore bindings without replacing original manifests."""
    root = Path(root)
    original = root / "training-manifest.json"
    payload = {
        **driver_identity(),
        "effective_config_after_trainer_init": effective_config,
        "imports": imports,
        "original_training_manifest_sha256": hashlib.sha256(original.read_bytes()).hexdigest()
        if original.exists()
        else None,
        "restored_checkpoint": (
            {
                "checkpoint_dir": checkpoint.checkpoint_dir,
                "manifest_id": checkpoint.manifest_id,
                "spec": checkpoint.payload["spec"],
            }
            if checkpoint is not None
            else None
        ),
    }
    write_new(root / "driver_runs" / (str(time.time_ns()) + ".initialized.json"), payload)


class QuiescentFinalMixin:
    """Override lifecycle only; super() remains the original joint trainer."""

    def _lifecycle_record(self, name, payload):
        root = Path(self.config.skill_harness.artifact_root)
        path = root / "evaluation_lifecycle" / self._lifecycle_id / name
        original = root / "training-manifest.json"
        write_new(
            path,
            {
                "schema_version": 1,
                **self._driver_identity,
                "original_training_manifest_sha256": hashlib.sha256(original.read_bytes()).hexdigest()
                if original.exists()
                else None,
                "core_fingerprint_unchanged_scope": (
                    "The unchanged core fingerprint excludes this explicitly recorded examples lifecycle extension."
                ),
                **payload,
            },
        )
        self._lifecycle_paths[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def _save_checkpoint(self):
        if not getattr(self, "_quiescent_final_enabled", False) or self.global_steps != self.total_training_steps:
            return super()._save_checkpoint()
        with self._joint_lock:
            controller = self._joint_controller
            self._await(controller.stop_background())
            head, clock = controller.store.head(), self._ema_clock()
            result = super()._save_checkpoint()
            checkpoint = self._last_joint_checkpoint
            if (
                controller.store.head() != head
                or checkpoint.payload["spec"]["optimizer_step"] != clock
                or checkpoint.payload["snapshots"]["skill_heads"]["main"] != head
            ):
                raise RuntimeError("Final COMPLETE checkpoint did not preserve the quiescent library/clock")
            self._quiescent_seal = {
                "seal_origin": "final_checkpoint_saved_after_quiescence",
                "optimizer_step": clock,
                "checkpoint_cursor": self.global_steps,
                "checkpoint_dir": checkpoint.checkpoint_dir,
                "manifest_id": checkpoint.manifest_id,
                "snapshot_id": head,
                "store_root": str(controller.store.root),
                "checkpoint_core_versions": checkpoint.payload["spec"].get("versions", {}),
            }
            self._lifecycle_record("checkpoint_seal.json", self._quiescent_seal)
            return result

    def _closed_loop_clock(self):
        from verl.experimental.skill_harness.joint_trainer import _resolve

        rows = asyncio.run(_resolve(self.actor_rollout_wg.get_ema_metadata()))
        clocks = [row["last_optimizer_step"] for row in rows]
        if len(clocks) != self.actor_rollout_wg.world_size or not clocks or any(c != clocks[0] for c in clocks):
            raise RuntimeError("Actor optimizer clocks disagree during final evaluation")
        return clocks[0]

    def _pinned_manager(self, snapshot_id):
        from omegaconf import open_dict

        from verl.utils.import_utils import load_class_from_fqn

        config = copy.deepcopy(self.config)  # Includes actual restored store/replay/ledger bindings.
        with open_dict(config.skill_harness.loading):
            config.skill_harness.loading.snapshot_id = snapshot_id
        factory = load_class_from_fqn(
            config.actor_rollout_ref.rollout.agent.agent_loop_manager_class, "AgentLoopManager"
        )
        return factory.create(
            config=config,
            llm_client=self.get_llm_client(),
            teacher_client=self.get_teacher_client(),
            reward_loop_worker_handles=self.get_reward_handles(),
        )

    def _official_final(self):
        from verl.experimental.skill_harness.evaluation import evaluate_official_splits

        return evaluate_official_splits(self, phase_prefix="final")

    def fit(self, manager):
        evaluation = self.config.skill_harness.evaluation
        if get(self.config.trainer, "val_before_train", False) or get(evaluation, "initial", False):
            raise ValueError("This entrypoint never runs development/official evaluation before training")
        enabled = get(evaluation, "final", False)
        self._quiescent_final_enabled = enabled
        if not enabled:
            return super().fit(manager)
        final = Path(self.config.skill_harness.artifact_root) / "evaluation/final"
        if final.exists() and any(final.iterdir()):
            raise FileExistsError("Preserve existing final artifacts; this entrypoint never overwrites them")
        self._lifecycle_id = "driver_" + uuid.uuid4().hex
        self._driver_identity = driver_identity()
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
            result = super().fit(manager)
        finally:
            evaluation["final"] = enabled
        # Original JointTrainer.fit finally already stopped background work,
        # closed EMA/CF gateways and its loop. The solver manager is still alive.
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


def trainer_class():
    from verl.experimental.skill_harness.joint_trainer import SkillJointSyncTrainer

    class QuiescentEvalJointTrainer(QuiescentFinalMixin, SkillJointSyncTrainer):
        pass

    return QuiescentEvalJointTrainer


def runner_class():
    import ray

    @ray.remote
    class QuiescentEvalTaskRunner:
        def run(self, config):
            import transfer_queue as tq
            from omegaconf import OmegaConf

            from verl.experimental.skill_harness.bootstrap import configure_runtime
            from verl.experimental.skill_harness.train import _redact, preserve_failed_queue
            from verl.utils.import_utils import load_class_from_fqn
            from verl.utils.logging_utils import configure_verl_logging

            imports = configure_runtime()
            configure_verl_logging()
            if config.trainer.v1.trainer_mode != "sync" or config.skill_harness.mode != "skill_joint":
                raise ValueError("This lifecycle wrapper requires the unchanged synchronous joint trainer")
            config.transfer_queue.enable = True
            OmegaConf.resolve(config)
            trainer, succeeded = None, False
            tq.init(config.transfer_queue)
            try:
                trainer = trainer_class()(config)
                trainer.init()
                restored = getattr(trainer, "_joint_restored", None)
                record_initialized_driver(
                    config.skill_harness.artifact_root,
                    _redact(OmegaConf.to_container(config, resolve=True)),
                    imports,
                    restored.checkpoint if restored is not None else None,
                )
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

    return QuiescentEvalTaskRunner


def entry(config):
    from omegaconf import OmegaConf

    from verl.experimental.skill_harness.bootstrap import apply_runtime_environment, configure_runtime

    provenance = configure_runtime()
    import ray

    from verl.experimental.skill_harness.train import _redact
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    if config.trainer.val_before_train or config.skill_harness.evaluation.initial:
        raise ValueError("Keep both pretraining evaluation switches disabled")
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    apply_runtime_environment(config)
    root = Path(config.skill_harness.artifact_root)
    write_new(
        root / "driver_runs" / (str(time.time_ns()) + ".json"),
        {
            "driver_path": str(Path(__file__).resolve()),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "imports": provenance,
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
